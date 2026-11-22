import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
from difflib import SequenceMatcher


STOP_ATOMS = {
    "true", "false", "null", "none", "ok", "success", "error", "message", 
    "role", "content", "type", "json", "action", "observation", "thought",
    "http", "https", "com", "org", "www",
    # High-frequency trace boilerplate that creates noisy token-overlap edges.
    "updated", "ledger", "answer", "reason", "request", "satisfied",
    "progress", "speaker", "instruction", "question", "step", "task",
}

MAX_DATA_FLOW_LOOKBACK = 8
MAX_DATA_FLOW_IN_EDGES_PER_NODE = 3
MIN_TOKEN_OVERLAP_SCORE = 0.75

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False)
        except Exception:
            text = str(text)
    
    text = re.sub(r'[#*`_>-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def extract_str_from_io(raw_val: Any) -> str:
    if raw_val is None:
        return ""

    if isinstance(raw_val, str):
        s = raw_val.strip()
        if s and s[0] in "{[":
            try:
                parsed = json.loads(s)
                return extract_str_from_io(parsed)
            except Exception:
                return s
        return s

    if isinstance(raw_val, dict):
        parts = []
        for k, v in raw_val.items():
            parts.append(str(k))
            parts.append(extract_str_from_io(v))
        return " ".join(parts)

    if isinstance(raw_val, list):
        return " ".join(extract_str_from_io(x) for x in raw_val)

    return str(raw_val)

def extract_atoms(raw_val: Any) -> List[str]:
    atoms = []

    def walk(x):
        if x is None:
            return
        if isinstance(x, str):
            s = x.strip()
            if len(s) > 1 and s[0] in "{[":
                try:
                    parsed = json.loads(s)
                    walk(parsed)
                    return
                except Exception:
                    pass
            norm_str = normalize_text(s)
            if norm_str:
                atoms.append(norm_str)
        elif isinstance(x, dict):
            for k, v in x.items():
                norm_k = normalize_text(str(k))
                if norm_k and len(norm_k) > 3 and norm_k not in STOP_ATOMS:
                    atoms.append(norm_k)
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for item in x:
                walk(item)
        else:
            norm_str = normalize_text(str(x))
            if norm_str:
                atoms.append(norm_str)

    walk(raw_val)
    return atoms

def is_data_flow_present(src_out_raw: Any, dst_in_raw: Any) -> Tuple[bool, float, str]:
    src_atoms = {a for a in extract_atoms(src_out_raw) if len(a) >= 3 and a.lower() not in STOP_ATOMS}
    dst_atoms = {a for a in extract_atoms(dst_in_raw) if len(a) >= 3 and a.lower() not in STOP_ATOMS}

    if not src_atoms or not dst_atoms:
        return False, 0.0, "none"

    exact = src_atoms & dst_atoms
    if exact:
        return True, 1.0, "exact_atom_match"

    src_text = normalize_text(extract_str_from_io(src_out_raw))
    dst_text = normalize_text(extract_str_from_io(dst_in_raw))
    if not src_text or not dst_text:
        return False, 0.0, "none"

    src_tokens_clean = {t for t in set(src_text.split()) if len(t) > 2 and t.lower() not in STOP_ATOMS}
    dst_tokens_clean = {t for t in set(dst_text.split()) if len(t) > 2 and t.lower() not in STOP_ATOMS}
    
    token_overlap = src_tokens_clean & dst_tokens_clean
    if token_overlap:
        score = len(token_overlap) / max(1, min(len(src_tokens_clean), len(dst_tokens_clean)))
        if score >= 0.5: 
            return True, round(score, 4), "token_overlap"

    if len(src_text) < 500 and len(dst_text) < 500:
        ratio = SequenceMatcher(None, src_text, dst_text).ratio()
        if ratio >= 0.8:
            return True, round(ratio, 4), "fuzzy_ratio"

    return False, 0.0, "none"

def _clean_field(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in {"", "<MISSING>", "None", "null", "NULL"}:
        return ""
    return text


def _classify_role(item: dict) -> str:
    raw_role = _clean_field(item.get("role"))
    raw_name = _clean_field(item.get("name"))

    role_l = raw_role.lower()
    name_l = raw_name.lower()

    tool_action_names = {
        "computer_terminal",
        "computerterminal",
    }

    if name_l in tool_action_names or role_l in tool_action_names:
        return "action"

    if role_l.startswith("orchestrator"):
        return "reasoning"

    if "websurfer" in role_l or "web surfer" in role_l:
        return "reasoning"

    if "filesurfer" in role_l or "file surfer" in role_l:
        return "reasoning"

    if raw_name and ("expert" in name_l or name_l.endswith("_expert")):
        return "reasoning"

    if raw_name:
        return "reasoning"

    if role_l in {"human", "user", "system"}:
        return "container"

    if role_l == "assistant":
        return "reasoning"

    return "reasoning" if raw_role else "container"


def canonicalize_agent_id(role: Any, name: Any) -> str:
    """Return the benchmark-facing agent identity.

    Hand-crafted traces often store the actual agent in `role` and leave `name`
    empty. Evaluation labels use coarse identities such as Orchestrator,
    WebSurfer, FileSurfer, and Assistant, so keep this canonicalized copy on
    every node instead of relying on the raw `name` field.
    """
    raw_role = _clean_field(role)
    raw_name = _clean_field(name)
    role_l = raw_role.lower()
    name_l = raw_name.lower()

    if raw_name and raw_name.lower() not in {"none", "unknown"}:
        return raw_name

    if role_l.startswith("orchestrator"):
        return "Orchestrator"
    if "websurfer" in role_l or "web surfer" in role_l:
        return "WebSurfer"
    if "filesurfer" in role_l or "file surfer" in role_l:
        return "FileSurfer"
    if "computerterminal" in role_l or "computer_terminal" in role_l or "computer terminal" in role_l:
        return "ComputerTerminal"
    if "assistant" in role_l:
        return "Assistant"
    if role_l in {"human", "user"}:
        return "User"

    if raw_role:
        return raw_role
    return "Unknown"


def is_tool_executor(role: Any, name: Any) -> bool:
    raw_role = _clean_field(role).lower().replace(" ", "_")
    raw_name = _clean_field(name).lower().replace(" ", "_")
    return raw_role in {"computer_terminal", "computerterminal"} or raw_name in {
        "computer_terminal",
        "computerterminal",
    }


def is_terminal_node(role: Any, name: Any, content: Any) -> bool:
    raw_role = _clean_field(role).lower()
    raw_name = _clean_field(name).lower()
    text = extract_str_from_io(content).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return (
        raw_role in {"final", "termination", "output"}
        or raw_name in {"final", "termination", "output"}
        or "termination" in raw_role
        or text in {"terminate", "terminated"}
        or text.startswith("terminate")
    )


def convert_history_record(file_path: str, trace_id: Optional[str] = None) -> Dict[str, Any]:
    """Reads a single JSON file containing a 'history' array and converts it to the graph schema."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history", [])
    if not history:
        raise ValueError(f"No 'history' key found in {file_path}")

    final_trace_id = trace_id or data.get("question_ID") or path.stem

    query = data.get("question")
    ground_truth = data.get("ground_truth")
    is_corrected = data.get("is_corrected")
    mistake_agent = data.get("mistake_agent")
    mistake_step = data.get("mistake_step")
    mistake_reason = data.get("mistake_reason")

    nodes: List[Dict[str, Any]] = []
    last_idx = len(history) - 1

    for idx, item in enumerate(history):
        # role = item.get("role", "")
        content = item.get("content", "")
        semantic_role = _classify_role(item)
        is_final = idx == last_idx
        terminal = is_terminal_node(item.get("role", ""), item.get("name", ""), content)
        
        node_id = f"{final_trace_id}_step_{idx}"

        node = {
            "node_id": node_id,
            "trace_id": str(final_trace_id),
            "role": item.get("role", ""),
            "name": item.get("name", ""),
            "agent_id": canonicalize_agent_id(item.get("role", ""), item.get("name", "")),
            "is_tool_executor": is_tool_executor(item.get("role", ""), item.get("name", "")),
            "semantic_role": semantic_role,
            "is_final_node": is_final,
            "is_terminal_node": terminal,
            "metadata": content,
            "step_index": idx
        }

        nodes.append(node)

    edges = _build_edges(nodes)

    return {
        "trace_id": str(final_trace_id),
        "file_name": path.name,
        "query": query,
        "ground_truth": ground_truth,
        "is_corrected": is_corrected,
        "nodes": nodes,
        "edges": edges,
        "raw_labels": {
            "mistake_agent": mistake_agent,
            "mistake_step": mistake_step,
            "mistake_reason": mistake_reason
        },
    }


def _build_edges(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    edges = []

    def _add_edge(src: str, dst: str, kind: str, evidence: Optional[Dict[str, Any]] = None):
        if not src or not dst or src == dst:
            return
        key = (src, dst, kind, json.dumps(evidence or {}, sort_keys=True))
        if key in seen:
            return
        seen.add(key)
        edges.append({
            "src": src,
            "dst": dst,
            "kind": kind,
            "evidence": evidence or {}
        })

    last_reasoning_id = None
    last_action_ids = []
    for pair in zip(nodes[:len(nodes)-1], nodes[1:]):
        node, next_node = pair
        _add_edge(node["node_id"], next_node["node_id"], "sequential_flow", {"rule": "sequential"})
    for node in nodes: 
        nid = node["node_id"]
        role = node.get("semantic_role")

        if role == "reasoning":
            if last_action_ids:
                for act_id in last_action_ids:
                    _add_edge(act_id, nid, "execution_flow", {"rule": "action_to_next_reasoning"})
                last_action_ids = []
            elif last_reasoning_id:
                _add_edge(last_reasoning_id, nid, "execution_flow", {"rule": "sequential_reasoning"})
            
            last_reasoning_id = nid

        elif role == "action":
            if last_reasoning_id:
                _add_edge(last_reasoning_id, nid, "execution_flow", {"rule": "reasoning_to_action"})
            last_action_ids.append(nid)

        elif role == "container":
            if last_reasoning_id:
                _add_edge(nid, last_reasoning_id, "execution_flow", {"rule": "container_to_reasoning"})

    for i, dst_node in enumerate(nodes):
        dst_id = dst_node["node_id"]
        dst_content = dst_node.get("metadata")
        data_candidates: List[Tuple[float, int, str, str]] = []

        for j in range(0, i):
            src_node = nodes[j]
            src_id = src_node["node_id"]
            src_content = src_node.get("metadata")
            
            if i - j > MAX_DATA_FLOW_LOOKBACK:
                continue

            has_flow, score, evidence_type = is_data_flow_present(src_content, dst_content)
            if not has_flow:
                continue

            if evidence_type == "token_overlap" and score < MIN_TOKEN_OVERLAP_SCORE:
                continue
            if evidence_type == "fuzzy_ratio":
                continue

            data_candidates.append((score, -j, src_id, evidence_type))

        data_candidates.sort(reverse=True)
        for score, _neg_j, src_id, evidence_type in data_candidates[:MAX_DATA_FLOW_IN_EDGES_PER_NODE]:
            _add_edge(src_id, dst_id, "data_flow", {"score": score, "evidence_type": evidence_type})

    return edges


def process_json_directory(input_dir: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
    dir_path = Path(input_dir)
    json_files = sorted(dir_path.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    
    graphs = []
    for json_file in json_files:
        try:
            graph = convert_history_record(str(json_file))
            graphs.append(graph)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for graph in graphs:
                f.write(json.dumps(graph, ensure_ascii=False) + "\n")

    return graphs

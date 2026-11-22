import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...common.graph_utils import (
    extract_str_from_io,
    is_data_flow_present as _is_data_flow_present,
    normalize_edge_types,
)
from ...common.io import load_json_records, write_jsonl


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
EDGE_TYPES = ("sequential_flow", "execution_flow", "data_flow")


# History node identity and role helpers
def is_data_flow_present(src_out_raw: Any, dst_in_raw: Any) -> Tuple[bool, float, str]:
    return _is_data_flow_present(
        src_out_raw,
        dst_in_raw,
        stop_atoms=STOP_ATOMS,
        token_overlap_threshold=0.5,
    )

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


# History conversion and edge construction
def _convert_history_data(
    data: Dict[str, Any],
    source_name: str,
    trace_id: Optional[str] = None,
    edge_types: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    path = Path(source_name)
    history = data.get("history", [])
    if not history:
        raise ValueError(f"No 'history' key found in {source_name}")

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
        content = item.get("content", "")
        semantic_role = _classify_role(item)
        is_final = idx == last_idx
        terminal = is_terminal_node(item.get("role", ""), item.get("name", ""), content)
        agent_id = canonicalize_agent_id(item.get("role", ""), item.get("name", ""))
        tool_executor = is_tool_executor(item.get("role", ""), item.get("name", ""))
        
        node_id = f"{final_trace_id}_step_{idx}"

        node = {
            "node_id": node_id,
            "trace_id": str(final_trace_id),
            "role": item.get("role", ""),
            "name": item.get("name", ""),
            "agent_id": agent_id,
            "is_tool_executor": tool_executor,
            "semantic_role": semantic_role,
            "is_final_node": is_final,
            "is_terminal_node": terminal,
            "metadata": content,
            "step_index": idx
        }

        nodes.append(node)

    suppressed_edges: List[Dict[str, Any]] = []
    edges = _build_edges(nodes, edge_types=edge_types, suppressed_edges=suppressed_edges)
    graph_audit = _diagnose_edges(nodes, edges)
    graph_audit["suppressed_edges"] = suppressed_edges

    return {
        "trace_id": str(final_trace_id),
        "file_name": path.name,
        "query": query,
        "ground_truth": ground_truth,
        "is_corrected": is_corrected,
        "nodes": nodes,
        "edges": edges,
        "graph_audit": graph_audit,
        "raw_labels": {
            "mistake_agent": mistake_agent,
            "mistake_step": mistake_step,
            "mistake_reason": mistake_reason
        },
    }


def convert_history_record(
    file_path: str,
    trace_id: Optional[str] = None,
    edge_types: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Reads a single JSON file containing a 'history' array and converts it to the graph schema."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _convert_history_data(data, str(path), trace_id=trace_id, edge_types=edge_types)


def _diagnose_edges(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Report direction, cycles, and final-node ancestor coverage without changing edges."""
    order = {str(node["node_id"]): index for index, node in enumerate(nodes)}
    children: Dict[str, set[str]] = defaultdict(set)
    parents: Dict[str, set[str]] = defaultdict(set)
    indegree = {node_id: 0 for node_id in order}
    reverse_count = 0
    for edge in edges:
        src, dst = str(edge["src"]), str(edge["dst"])
        if order[src] >= order[dst]:
            reverse_count += 1
        if dst not in children[src]:
            children[src].add(dst)
            parents[dst].add(src)
            indegree[dst] += 1

    ready = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    topological_count = 0
    while ready:
        current = ready.popleft()
        topological_count += 1
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)

    reachable: set[str] = set()
    if nodes:
        pending = [str(nodes[-1]["node_id"])]
        while pending:
            current = pending.pop()
            if current in reachable:
                continue
            reachable.add(current)
            pending.extend(parents[current] - reachable)

    return {
        "is_dag": topological_count == len(nodes),
        "reverse_edge_count": reverse_count,
        "reachable_ancestor_count": len(reachable),
        "unreachable_node_ids": [node_id for node_id in order if node_id not in reachable],
    }


def _build_edges(
    nodes: List[Dict[str, Any]],
    edge_types: Optional[Iterable[str]] = None,
    *,
    suppressed_edges: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    enabled_edge_types = normalize_edge_types(edge_types, supported=EDGE_TYPES)
    seen = set()
    edges = []

    def _add_edge(src: str, dst: str, kind: str, evidence: Optional[Dict[str, Any]] = None):
        if kind not in enabled_edge_types or not src or not dst or src == dst:
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
                # A later user/system message cannot cause an earlier reasoning step.
                # The existing forward sequential edge records their temporal order.
                # Record the removed execution claim so graph consumers can audit it.
                if suppressed_edges is not None and "execution_flow" in enabled_edge_types:
                    suppressed_edges.append({
                        "src": nid,
                        "dst": last_reasoning_id,
                        "kind": "execution_flow",
                        "rule": "container_to_reasoning",
                        "reason": "backward_time_direction_and_unsupported_causality",
                    })

    if "data_flow" in enabled_edge_types:
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


# File and directory conversion entrypoints
def process_json_directory(
    input_dir: str,
    output_path: Optional[str] = None,
    edge_types: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    edge_types = tuple(normalize_edge_types(edge_types, supported=EDGE_TYPES))
    dir_path = Path(input_dir)
    json_files = sorted(dir_path.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    
    graphs = []
    for json_file in json_files:
        try:
            graph = convert_history_record(str(json_file), edge_types=edge_types)
            graphs.append(graph)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    if output_path:
        write_jsonl(output_path, graphs)

    return graphs


def process_json_path(
    input_path: str,
    output_path: Optional[str] = None,
    edge_types: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    edge_types = tuple(normalize_edge_types(edge_types, supported=EDGE_TYPES))
    path = Path(input_path)
    if path.is_dir():
        return process_json_directory(str(path), output_path=output_path, edge_types=edge_types)

    if path.suffix.lower() == ".json":
        graphs = [convert_history_record(str(path), edge_types=edge_types)]
    elif path.suffix.lower() == ".jsonl":
        graphs = []
        for index, record in enumerate(load_json_records(str(path))):
            if not isinstance(record, dict):
                continue
            source_name = str(record.get("file_name") or f"{path.stem}_{index}.json")
            graphs.append(_convert_history_data(record, source_name, edge_types=edge_types))
    else:
        raise ValueError(f"Who&When input must be a directory, .json, or .jsonl: {input_path}")

    if output_path:
        write_jsonl(output_path, graphs)
    return graphs

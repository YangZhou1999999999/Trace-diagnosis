import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from ...common.graph_utils import (
    extract_str_from_io,
    is_data_flow_present as _is_data_flow_present,
    normalize_edge_types,
)
from ...common.io import load_json_records, write_jsonl


STOP_ATOMS = {"true", "false", "null", "none", "ok", "success", "error", "message", "role", "content", "type", "json"}
DEFAULT_EDGE_TYPES = ("contains", "execution_flow", "data_flow")


# Span parsing and node construction
def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            repaired = re.sub(r",\s*([}\]])", r"\1", value)
            return json.loads(repaired)
    raise TypeError(f"Unsupported JSON value type: {type(value)!r}")


def _classify_semantic_role(span: Dict[str, Any]) -> str:
    name = (span.get("span_name") or "").strip()
    attrs = span.get("span_attributes") or {}
    logs = span.get("logs") or []
    kind = (attrs.get("openinference.span.kind") or "").upper()

    ignored_setup_names = {
        "create_agent_hierarchy",
        "setup_agent",
        "init_agent",
        "build_graph",
        "create_tools"
    }
    if name in ignored_setup_names:
        return "container"

    if name == "CodeAgent.run":
        return "agent"

    if name in {"main", "process_item", "answer_single_question"} or re.match(r"^Step\s+\d+$", name):
        return "container"

    if name == "LiteLLMModel.__call__" or kind == "LLM":
        return "reasoning"

    has_func_log = any(
        isinstance(l.get("body"), dict) and ("function.arguments" in l["body"] or "function.output" in l["body"])
        for l in logs
    )
    if kind == "TOOL" or has_func_log or "tool" in name.lower():
        return "action"

    return "container"


def _extract_input(span: Dict[str, Any]) -> Any:
    attrs = span.get("span_attributes") or {}
    messages = []
    key_pattern = re.compile(r"^llm\.input_messages\.(\d+)\.message\.content$")
    key_pattern_tool = re.compile( r"^llm\.output_messages\.(\d+)\.message\.tool_calls\.(\d+)"
        r"\.tool_call\.function\.arguments$")
    match_keys = [k for k in attrs.keys() if key_pattern.match(k) or key_pattern_tool.match(k)]
    if match_keys:
        for key in match_keys:
            if key_pattern.match(key):
                idx = int(key_pattern.match(key).group(1))
                messages.append({
                    "role": attrs.get(f"llm.input_messages.{idx}.message.role"),
                    "content": attrs[key]
                })
            elif key_pattern_tool.match(key):
                idx_msg, _idx_call = map(int, key_pattern_tool.match(key).groups())
                messages.append({
                    "role": attrs.get(f"llm.input_messages.{idx_msg}.message.role"),
                    "content": attrs[key]
                })
    return messages or None


def _extract_output(span: Dict[str, Any]) -> Any:
    attrs = span.get("span_attributes") or {}
    logs = span.get("logs") or []
    

    messages = []
    key_pattern = re.compile(r"^llm\.output_messages\.(\d+)\.message\.content$")
    key_pattern_tool = re.compile( r"^llm\.output_messages\.(\d+)\.message\.tool_calls\.(\d+)"
        r"\.tool_call\.function\.output$")
    match_keys = [k for k in attrs.keys() if key_pattern.match(k) or key_pattern_tool.match(k)]
    if match_keys:
        for key in match_keys:
            if key_pattern.match(key):
                idx = int(key_pattern.match(key).group(1))
                messages.append({
                    "role": attrs.get(f"llm.output_messages.{idx}.message.role"),
                    "content": attrs[key]
                })
            elif key_pattern_tool.match(key):
                idx_msg, _idx_call = map(int, key_pattern_tool.match(key).groups())
                messages.append({
                    "role": attrs.get(f"llm.output_messages.{idx_msg}.message.role"),
                    "content": attrs[key]
                })

    tool_out = (
        attrs.get("tool.output")
        or attrs.get("return_value")
        or attrs.get("gcp.vertex_ai.tool.output")
        or attrs.get("traceloop.entity.output")
        or attrs.get("function.output")
    )
    messages.append({
        "role": "tool",
        "content": tool_out
    })


    for log in logs:
        body = log.get("body")
        if isinstance(body, dict) and "function.output" in body:
            messages.append({
                "role": "function",
                "content": body["function.output"]
            })

    return messages or None


def _build_node(span: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    attrs = span.get("span_attributes") or {}
    semantic_role = _classify_semantic_role(span)
    
    node = {
        "node_id": span.get('span_id'),
        "trace_id": trace_id,
        "parent_id": span.get("parent_span_id"),
        "name": span.get("span_name"),
        "semantic_role": semantic_role,
        "timestamp": span.get("timestamp"),
        "duration": span.get("duration"),
        "input": attrs.get("input.value") or None,
        "output": attrs.get("output.value") or None,
        "input_messages": [m for m in (_extract_input(span) or []) if isinstance(m, dict)],
        "output_messages": [m for m in (_extract_output(span) or []) if isinstance(m, dict)],
        "metadata": attrs,
    }
    return node


# Data-flow extraction and edge construction
def is_data_flow_present(src_out_raw: Any, dst_in_raw: Any) -> Tuple[bool, float, str]:
    return _is_data_flow_present(
        src_out_raw,
        dst_in_raw,
        stop_atoms=STOP_ATOMS,
        token_overlap_threshold=0.4,
    )



def _extract_all_input_texts(rec: Dict[str, Any]) -> List[Tuple[int, str]]:
    node = rec.get("node") or {}
    extracted_map = {}

    input_msgs = node.get("input_messages") or []
    for idx, msg in enumerate(input_msgs):
        if isinstance(msg, dict):

            content = msg.get("content") or ""
            msg_str = extract_str_from_io(content).strip()
            if msg_str:
                extracted_map[idx] = msg_str

    if not extracted_map:
        node_input = node.get("input")
        if node_input:
            inp_str = extract_str_from_io(node_input).strip()
            if inp_str:
                extracted_map[0] = inp_str

    results = [(idx, extracted_map[idx]) for idx in sorted(extracted_map.keys())]
    return results

def _extract_all_output_texts(rec: Dict[str, Any]) -> List[str]:
    node = rec.get("node") or {}
    outputs = []


    node_out = node.get("output")
    if node_out:
        out_str = extract_str_from_io(node_out).strip()
        if out_str:
            outputs.append(out_str)


    output_msgs = node.get("output_messages") or []
    for msg in output_msgs:
        if isinstance(msg, dict):
            content = msg.get("content")
            if content:
                c_str = extract_str_from_io(content).strip()
                if c_str:
                    outputs.append(c_str)

    return outputs

def _build_edges(
    records: List[Dict[str, Any]],
    edge_types: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    enabled_edge_types = normalize_edge_types(edge_types, supported=DEFAULT_EDGE_TYPES)
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

    node_map = {r["node"]["node_id"]: r for r in records}

    # =====================================================
    # 1. Contains edges (包含关系)
    # =====================================================
    for r in records:
        node = r["node"]
        parent_id = node.get("parent_id")
        if parent_id in node_map:
            _add_edge(parent_id, node["node_id"], "contains", {"rule": "parent_child"})

    sorted_records = sorted(
        records,
        key=lambda x: x["node"].get("timestamp") or ""
    )

    # =====================================================
    # 2. Execution Flow (控制流)
    # =====================================================
    last_reasoning_id = None
    last_action_ids = []

    for r in sorted_records:
        node = r["node"]
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

    if "data_flow" in enabled_edge_types:
        # =====================================================
        # 3. Data Flow (全方位交叉匹配: Input/Input_Msg <-> Output/Output_Msg)
        # =====================================================
        for i, dst_rec in enumerate(sorted_records):
            dst_node = dst_rec["node"]
            dst_id = dst_node["node_id"]

            # 提取 dst 的所有输入文本项 [(msg_idx, text)]
            dst_input_items = _extract_all_input_texts(dst_rec)
            if not dst_input_items:
                continue

            for src_rec in sorted_records[:i]:
                src_node = src_rec["node"]
                src_id = src_node["node_id"]


                if src_node.get("semantic_role") == "container":
                    continue


                src_output_texts = _extract_all_output_texts(src_rec)
                if not src_output_texts:
                    continue

                best_match = None 

                for in_idx, dst_text in dst_input_items:
                    for src_text in src_output_texts:
                        is_matched, score, rule = is_data_flow_present(
                            src_text, dst_text
                        )

                        if is_matched:
                            if best_match is None or score > best_match["score"]:
                                best_match = {
                                    "score": score,
                                    "idx": in_idx,
                                    "rule": rule,
                                }

                if best_match:
                    _add_edge(
                        src_id,
                        dst_id,
                        "data_flow",
                        {
                            "rule": f"cross_match_{best_match['rule']}",
                            "message_index": best_match["idx"],
                            "match_score": best_match["score"],
                        },
                    )

    return edges


# Span traversal and graph assembly
def _walk_spans(span: Dict[str, Any], trace_id: str) -> Iterable[Dict[str, Any]]:
    node = _build_node(span, trace_id)
    span_name = span.get("span_name", "")
    is_step = bool(re.match(r"^Step\s+\d+$", span_name))
    is_container = node.get("semantic_role") == "container"

    if not is_step and not is_container:
        yield {"node": node}

    for child in span.get("child_spans") or []:
        yield from _walk_spans(child, trace_id)


def _find_agent_root(spans: List[Dict[str, Any]], root_name: str) -> Optional[Dict[str, Any]]:
    for span in spans:
        if span.get("span_name") == root_name:
            return span
        children = span.get("child_spans", [])
        result = _find_agent_root(children, root_name)
        if result:
            return result
    return None


def _build_error_anchors(labels: Dict[str, Any]) -> List[Dict[str, Any]]:
    anchors = []
    for err in labels.get("errors") or []:
        anchors.append({
            "location": err.get("location"),
            "category": err.get("category"),
            "impact": err.get("impact"),
            "description": err.get("description"),
            "evidence": err.get("evidence"),
        })
    return anchors


def _trail_final_anchor_node_id(graph: Dict[str, Any]) -> Optional[str]:
    """Select the diagnosis anchor using the unfiltered graph structure."""
    nodes = list(graph.get("nodes") or [])
    if not nodes:
        return None
    ordered = sorted(
        nodes,
        key=lambda node: (node.get("timestamp") is None, node.get("timestamp") or 0),
    )
    for node in reversed(ordered):
        role = str(node.get("semantic_role") or "").lower()
        name = str(node.get("name") or node.get("span_name") or "").lower()
        if "output" in role or "final" in role or "output" in name or "writer" in name:
            value = node.get("node_id", node.get("span_id", node.get("id")))
            return str(value) if value is not None else None
    node_ids = {
        str(node.get("node_id", node.get("span_id", node.get("id"))))
        for node in ordered
        if node.get("node_id", node.get("span_id", node.get("id"))) is not None
    }
    outgoing = {
        str(edge.get("src")) for edge in (graph.get("edges") or [])
        if isinstance(edge, dict)
        and str(edge.get("src")) in node_ids
        and str(edge.get("dst")) in node_ids
    }
    candidates = [
        node for node in ordered
        if str(node.get("node_id", node.get("span_id", node.get("id")))) not in outgoing
    ]
    chosen = candidates[-1] if candidates else ordered[-1]
    value = chosen.get("node_id", chosen.get("span_id", chosen.get("id")))
    return str(value) if value is not None else None


def convert_record(
    record: Dict[str, Any],
    root_name: str = "answer_single_question",
    edge_types: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    selected_edge_types = normalize_edge_types(edge_types, supported=DEFAULT_EDGE_TYPES)
    ordered_edge_types = [edge_type for edge_type in DEFAULT_EDGE_TYPES if edge_type in selected_edge_types]
    trace = _safe_json_loads(record["trace"])
    labels = _safe_json_loads(record.get("labels")) or {}
    trace_id = trace.get("trace_id")
    spans = trace.get("spans") or []
    agent_root = _find_agent_root(spans, root_name=root_name)
    
    if agent_root is None:
        agent_root = {
        "span_name": "virtual_root",
            "child_spans": spans,
            "input": spans[0].get("input") if spans else None
        }
        # raise ValueError(f"Trace {trace_id} does not contain {root_name} span.")
    
        
    records = list(_walk_spans(agent_root, trace_id))
    nodes = [r["node"] for r in records]
    final_anchor_node_id = None
    if edge_types is not None and selected_edge_types != set(DEFAULT_EDGE_TYPES):
        full_edges = _build_edges(records)
        full_graph = {"nodes": nodes, "edges": full_edges}
        final_anchor_node_id = _trail_final_anchor_node_id(full_graph)
        node_ids = {
            str(node.get("node_id", node.get("span_id", node.get("id"))))
            for node in nodes
            if node.get("node_id", node.get("span_id", node.get("id"))) is not None
        }
        if final_anchor_node_id is None or final_anchor_node_id not in node_ids:
            raise ValueError("Cannot determine a valid full-graph final anchor for the requested TRAIL edge subset.")
        edges = [edge for edge in full_edges if edge.get("kind") in selected_edge_types]
    else:
        # Preserve the existing default G0 construction and serialization path.
        edges = _build_edges(records, edge_types=ordered_edge_types)
    graph_metadata = {
        "graph_variant": "G0",
        "edge_types": sorted(DEFAULT_EDGE_TYPES),
        "node_retention": "step_and_container_spans_filtered_v1",
        "retained_node_count": len(nodes),
    }
    if edge_types is not None:
        if selected_edge_types != set(DEFAULT_EDGE_TYPES):
            graph_metadata["graph_variant"] = "edge_subset_v1"
            graph_metadata["edge_types"] = ordered_edge_types
            graph_metadata["final_anchor_node_id"] = final_anchor_node_id
        graph_metadata["requested_edge_types"] = ordered_edge_types
    return {
        "trace_id": trace_id,
        "root_span": agent_root.get("span_name"),
        "graph_metadata": graph_metadata,
        "query": nodes[0].get("input") if nodes else None,
        "nodes": nodes,
        "edges": edges,
        "error_anchors": _build_error_anchors(labels),
        "raw_labels": {
            "trace_labels": labels,
            "scores": labels.get("scores"),
        },
    }


def _convert_local_json_record(
    record: Dict[str, Any],
    root_name: str,
    edge_types: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    if "nodes" in record and "edges" in record:
        if edge_types is None:
            return record
        selected = normalize_edge_types(edge_types, supported=DEFAULT_EDGE_TYPES)
        selected_order = [edge_type for edge_type in DEFAULT_EDGE_TYPES if edge_type in selected]
        if any(not isinstance(edge, dict) or not edge.get("kind") for edge in (record.get("edges") or [])):
            raise ValueError("Converted TRAIL graph contains a malformed edge; cannot apply edge selection safely.")
        full_anchor = None
        if selected != set(DEFAULT_EDGE_TYPES):
            node_ids = {
                str(node.get("node_id", node.get("span_id", node.get("id"))))
                for node in (record.get("nodes") or [])
                if isinstance(node, dict)
                and node.get("node_id", node.get("span_id", node.get("id"))) is not None
            }
            original_metadata = record.get("graph_metadata") or {}
            original_anchor = original_metadata.get("final_anchor_node_id") if isinstance(original_metadata, dict) else None
            if original_anchor is not None:
                if str(original_anchor) not in node_ids:
                    raise ValueError("Preconverted TRAIL graph final_anchor_node_id is not a retained node.")
                # A stored anchor belongs to the original full graph. In an
                # already filtered edge-subgraph, the remaining edges can
                # create a different sink; never recompute or compare against
                # that filtered topology.
                full_anchor = str(original_anchor)
            else:
                graph_variant = str(original_metadata.get("graph_variant", "")).lower()
                recorded_types = original_metadata.get("edge_types")
                requested_types = original_metadata.get("requested_edge_types")

                def declares_full_edge_types(value: Any) -> bool:
                    return value is None or (
                        isinstance(value, (list, tuple, set))
                        and all(isinstance(item, str) for item in value)
                        and set(value) == set(DEFAULT_EDGE_TYPES)
                    )

                is_marked_full_graph = (
                    graph_variant == "g0"
                    and declares_full_edge_types(requested_types)
                    and declares_full_edge_types(recorded_types)
                )
                if is_marked_full_graph:
                    # The input explicitly identifies itself as an unfiltered
                    # G0 graph, so compute the anchor before applying the new
                    # edge selection.
                    full_anchor = _trail_final_anchor_node_id(record)
                else:
                    raise ValueError(
                        "Cannot safely select edges from a preconverted TRAIL graph without "
                        "a retained final_anchor_node_id or verified G0 metadata."
                    )
                if full_anchor is None or full_anchor not in node_ids:
                    raise ValueError("Cannot determine a valid full-graph final anchor before filtering the TRAIL graph.")
        graph = dict(record)
        graph["edges"] = [
            edge for edge in (record.get("edges") or [])
            if isinstance(edge, dict) and edge.get("kind") in selected
        ]
        metadata = dict(record.get("graph_metadata") or {})
        metadata["graph_variant"] = (
            "G0" if selected == set(DEFAULT_EDGE_TYPES) else "edge_subset_v1"
        )
        metadata["edge_types"] = sorted(DEFAULT_EDGE_TYPES) if selected == set(DEFAULT_EDGE_TYPES) else selected_order
        metadata["requested_edge_types"] = selected_order
        if selected != set(DEFAULT_EDGE_TYPES):
            metadata["final_anchor_node_id"] = full_anchor
        graph["graph_metadata"] = metadata
        return graph

    if "trace" in record:
        return convert_record(record, root_name=root_name, edge_types=edge_types)

    if "spans" in record:
        return convert_record(
            {
                "trace": record,
                "labels": record.get("labels", {}),
            },
            root_name=root_name,
        edge_types=edge_types,
        )

    raise ValueError(
        "TRAIL JSON records must contain graph nodes/edges, a trace field, or spans."
    )


# File-level conversion entrypoints
def convert_json(
    input_path: str,
    output_path: Optional[str] = None,
    max_rows: Optional[int] = None,
    edge_types: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    if edge_types is not None:
        selected = normalize_edge_types(edge_types, supported=DEFAULT_EDGE_TYPES)
        edge_types = tuple(edge_type for edge_type in DEFAULT_EDGE_TYPES if edge_type in selected)
    records = load_json_records(input_path)
    if max_rows is not None:
        records = records[:max_rows]

    root_name = "CodeAgent.run" if "swe_bench" in input_path else "answer_single_question"
    graphs = [
        _convert_local_json_record(record, root_name=root_name, edge_types=edge_types)
        for record in records
        if isinstance(record, dict)
    ]

    if output_path:
        write_jsonl(output_path, graphs)
    return graphs


def convert_parquet(
    input_path: str,
    output_path: Optional[str] = None,
    max_rows: Optional[int] = None,
    edge_types: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    if edge_types is not None:
        selected = normalize_edge_types(edge_types, supported=DEFAULT_EDGE_TYPES)
        edge_types = tuple(edge_type for edge_type in DEFAULT_EDGE_TYPES if edge_type in selected)
    df = pd.read_parquet(input_path)
    graphs: List[Dict[str, Any]] = []
    if max_rows is not None:
        df = df.head(max_rows)
    if "swe_bench" in input_path:
        root_name = "CodeAgent.run"
    else:
        root_name = "answer_single_question"

    for _, row in df.iterrows():
        graphs.append(convert_record(row.to_dict(), root_name=root_name, edge_types=edge_types))

    if output_path:
        write_jsonl(output_path, graphs)

    return graphs

"""Convert MAADBench execution traces into label-free action graphs.

The converter deliberately copies only observable inputs, model outputs, and
tool results. Verification labels, expected tool behavior, room answers,
latency/token statistics, and failure summaries stay out of diagnosis graphs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from ...common.io import write_jsonl
from ...common.graph_utils import normalize_edge_types


EDGE_TYPES = ("sequential_flow", "execution_flow", "data_flow")
PIPELINE_TRANSITIONS = {
    ("SELECT_PUZZLE", "OBSERVE_PLANNED_CLUE"),
    ("SELECT_PUZZLE", "OBSERVE_CLUE"),
    ("OBSERVE_CLUE", "SOLVE_CLUE"),
    ("OBSERVE_PLANNED_CLUE", "SOLVE_CLUE"),
    ("SOLVE_CLUE", "OBSERVE_ITEM"),
    ("OBSERVE_ITEM", "APPLY_DELTA"),
    ("APPLY_DELTA", "OBSERVE_PUZZLE"),
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", _text(value).lower())).strip()


def _observed_tool_execution(step: dict[str, Any]) -> dict[str, Any]:
    value = step.get("tool_execution")
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in ("status", "result", "error") if key in value}


def _classify_step(step: dict[str, Any], *, apply_delta_is_action: bool) -> str:
    action = str(step.get("action") or "").upper()
    return "action" if action == "UNIT_TOOL_CALL" or (apply_delta_is_action and action == "APPLY_DELTA") else "reasoning"


def _make_query(trace: list[dict[str, Any]]) -> str:
    for step in trace:
        value = step.get("input")
        if value is not None and _text(value).strip():
            return _text(value)
    return "MAADBench escape-room task (initial input missing)"


def _make_node(step: dict[str, Any], index: int, trace_id: str, *, apply_delta_is_action: bool) -> dict[str, Any]:
    action = str(step.get("action") or "UNKNOWN")
    output_message = _text(step.get("output_message", step.get("message", "")))
    output_structured = step.get("output_structured", step.get("structured", {}))
    return {
        # Keep benchmark filenames (which encode escaped/failed status) out
        # of node identifiers because node_id is shown to the diagnosing model.
        "node_id": f"step_{index}",
        "trace_id": trace_id,
        "role": "tool" if action.upper() == "UNIT_TOOL_CALL" else "assistant",
        "name": str(step.get("agent") or ""),
        "semantic_role": _classify_step(step, apply_delta_is_action=apply_delta_is_action),
        "step_index": index,
        "action_type": action,
        "action_id": step.get("action_id"),
        "puzzle_id": step.get("puzzle_id"),
        "input": step.get("input"),
        "output_message": output_message,
        "output_structured": output_structured,
        "tool_execution": _observed_tool_execution(step),
        # This is only a conversion-time edge hint; it is removed before save.
        "_parent_action_id": step.get("parent_action_id"),
    }


def _string_leaves(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [part for child in value.values() for part in _string_leaves(child)]
    if isinstance(value, (list, tuple)):
        return [part for child in value for part in _string_leaves(child)]
    if isinstance(value, str):
        normalized = _normalized(value)
        if 30 <= len(normalized) <= 10000 and len(normalized.split()) >= 5:
            return [normalized]
    return []


def _data_match(source: dict[str, Any], target: dict[str, Any]) -> Optional[dict[str, Any]]:
    target_text = _normalized(target.get("input"))
    if not target_text:
        return None
    candidates = _string_leaves(source.get("output_structured"))
    candidates += _string_leaves(source.get("output_message"))
    for candidate in sorted(set(candidates), key=len, reverse=True):
        if candidate in target_text:
            return {
                "rule": "generated_text_reused",
                "match_length": len(candidate),
                "matched_preview": candidate[:160],
                "confidence": "high",
            }
    return None


def _build_edges(nodes: list[dict[str, Any]], *, edge_types: Optional[Iterable[str]], max_data_hops: int = 20) -> list[dict[str, Any]]:
    enabled = normalize_edge_types(edge_types, supported=EDGE_TYPES)
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(source: dict[str, Any], target: dict[str, Any], kind: str, evidence: dict[str, Any]) -> None:
        if kind not in enabled or source["node_id"] == target["node_id"]:
            return
        key = (source["node_id"], target["node_id"], kind)
        if key not in seen:
            seen.add(key)
            edges.append({"src": key[0], "dst": key[1], "kind": kind, "evidence": evidence})

    query, steps = nodes[0], nodes[1:]
    for previous, current in zip(nodes, nodes[1:]):
        add(previous, current, "sequential_flow", {"rule": "trace_order"})
    if steps:
        add(query, steps[0], "execution_flow", {"rule": "initial_task_input"})
    for previous, current in zip(steps, steps[1:]):
        transition = (str(previous.get("action_type", "")), str(current.get("action_type", "")))
        if previous.get("puzzle_id") == current.get("puzzle_id") and transition in PIPELINE_TRANSITIONS:
            add(previous, current, "execution_flow", {
                "rule": "known_action_transition", "transition": list(transition),
                "confidence": "protocol_heuristic",
            })

    action_to_node: dict[str, dict[str, Any]] = {}
    for node in steps:
        action_id = node.get("action_id")
        if action_id is not None:
            action_to_node.setdefault(str(action_id), node)
    for index, node in enumerate(steps):
        parent_id = node.get("_parent_action_id")
        parent = action_to_node.get(str(parent_id)) if parent_id is not None else None
        if node.get("action_type") == "UNIT_TOOL_CALL" and parent is not None:
            if node["step_index"] < parent["step_index"]:
                add(node, parent, "execution_flow", {
                    "rule": "tool_result_to_parent_completion", "parent_action_id": str(parent_id),
                    "confidence": "explicit_association",
                })
                result = node.get("tool_execution", {}).get("result")
                structured = parent.get("output_structured")
                if isinstance(structured, dict) and result is not None and structured.get("converted_delta") == result:
                    add(node, parent, "data_flow", {
                        "rule": "tool_result_used_as_converted_delta",
                        "parent_action_id": str(parent_id), "confidence": "high",
                    })
            else:
                add(parent, node, "execution_flow", {
                    "rule": "parent_action_to_tool_call", "parent_action_id": str(parent_id),
                    "confidence": "explicit_association",
                })
        if index and node.get("action_type") == "UNIT_TOOL_CALL":
            previous = steps[index - 1]
            if previous.get("puzzle_id") == node.get("puzzle_id") and previous.get("action_type") != "UNIT_TOOL_CALL":
                add(previous, node, "execution_flow", {
                    "rule": "prior_step_context_to_tool", "confidence": "heuristic",
                })

    for target_index in range(1, len(steps)):
        target = steps[target_index]
        for source in steps[max(0, target_index - max_data_hops):target_index]:
            evidence = _data_match(source, target)
            if evidence:
                add(source, target, "data_flow", evidence)
    return edges


def convert_record(data: dict[str, Any], trace_id: str, *, edge_types: Optional[Iterable[str]] = None,
                   apply_delta_is_action: bool = True, max_data_hops: int = 20) -> dict[str, Any]:
    selected_edge_types = normalize_edge_types(edge_types, supported=EDGE_TYPES)
    ordered_edge_types = [kind for kind in EDGE_TYPES if kind in selected_edge_types]
    trace = data.get("trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("Expected a non-empty 'trace' array")
    nodes = [{
        "node_id": "query", "trace_id": trace_id,
        "role": "user", "name": "Task", "semantic_role": "query",
        "metadata": _make_query(trace), "step_index": -1,
        "action_type": "QUERY", "puzzle_id": None,
    }]
    for index, step in enumerate(trace):
        if not isinstance(step, dict):
            raise TypeError(f"trace[{index}] must be an object")
        nodes.append(_make_node(step, index, trace_id, apply_delta_is_action=apply_delta_is_action))
    edges = _build_edges(nodes, edge_types=ordered_edge_types, max_data_hops=max_data_hops)
    for node in nodes:
        node.pop("_parent_action_id", None)

    room = data.get("room") if isinstance(data.get("room"), dict) else {}
    config = data.get("config") if isinstance(data.get("config"), dict) else {}
    llm_config = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    graph = {
        "trace_id": str(trace_id),
        "query": nodes[0]["metadata"],
        "dataset": "MAADBench-Full",
        "source_model": llm_config.get("model"),
        "room_id": room.get("room_id"),
        "difficulty": room.get("difficulty"),
        "nodes": nodes,
        "edges": edges,
    }
    if edge_types is not None:
        full_selection = selected_edge_types == set(EDGE_TYPES)
        graph["graph_metadata"] = {
            "graph_variant": "full_edge_set_v1" if full_selection else "edge_subset_v1",
            "edge_types": list(ordered_edge_types),
            "requested_edge_types": list(ordered_edge_types),
            "node_retention": "query_plus_all_original_actions_v1",
            "retained_node_count": len(nodes),
        }
    return graph


def convert_to_graphs(input_path: str, output_path: str, max_graphs: Optional[int] = None,
                      edge_types: Optional[Iterable[str]] = None) -> list[dict[str, Any]]:
    ordered_edge_types = None
    if edge_types is not None:
        selected = normalize_edge_types(edge_types, supported=EDGE_TYPES)
        ordered_edge_types = [kind for kind in EDGE_TYPES if kind in selected]
    root = Path(input_path).expanduser()
    if root.is_file():
        candidates = [root]
        trace_root = root.parent
    elif root.is_dir():
        candidates = sorted(path for path in root.rglob("*.json") if "EscapeRoom" in path.name)
        trace_root = root
    else:
        raise FileNotFoundError(f"MAADBench input path does not exist: {root}")
    if max_graphs is not None:
        if max_graphs < 0:
            raise ValueError("max_graphs must be non-negative")
        candidates = candidates[:max_graphs]
    if not candidates:
        raise FileNotFoundError(f"No EscapeRoom trace JSON files found under: {root}")

    graphs = []
    for path in candidates:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("trace"), list):
            continue
        trace_id = path.relative_to(trace_root).with_suffix("").as_posix()
        graph = convert_record(data, trace_id, edge_types=ordered_edge_types)
        graph["file_name"] = path.name
        graphs.append(graph)
    if not graphs:
        raise ValueError(f"No valid MAADBench traces found under: {root}")
    write_jsonl(output_path, graphs)
    return graphs


"""Runner adapter for the strict Test-7 Who&When implementation.

The diagnosis algorithm and graph construction are ported from Test-7 Cell 2
and Cell 6. This module reuses only the project's API transport, checkpoint
loop, and evaluation-compatible record envelope.
"""

import re
from typing import Any, Dict, Iterable, List, Optional

from ..common.io import load_graphs, write_jsonl
from ..common.llm import LLMServer
from ..common.run_loop import run_graph_diagnosis
from .diagnoser import ReverseDependencyDFS
from .graph_converter import process_json_directory


def convert_to_graphs(
    input_dir: str,
    output_path: str,
    max_graphs: Optional[int] = None,
    edge_types: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Use the exact Test-7 converter; edge types are not configurable there."""
    if edge_types is not None:
        raise ValueError(
            "Test-7 legacy Who&When uses its original fixed edge construction; "
            "do not pass edge_types."
        )
    graphs = process_json_directory(input_dir, output_path=None)
    if max_graphs is not None:
        graphs = graphs[:max_graphs]
    write_jsonl(output_path, graphs)
    return graphs


def _step_number_from_node(root_node: Dict[str, Any]) -> str:
    raw_step = root_node.get("step_index")
    if raw_step is None:
        return "-1"
    if isinstance(raw_step, (int, float)):
        return str(int(raw_step))
    if isinstance(raw_step, str):
        match = re.search(r"\d+", raw_step)
        if match:
            return match.group(0)
    return "-1"


def _extract_node_from_cascade(
    analysis_results: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    cascade = analysis_results.get("cascade_failure") or {}
    root_ids = cascade.get("root_cause_nodes") or []
    diagnoses = cascade.get("backward_diagnosis") or []
    if not root_ids or not diagnoses:
        return None
    root_id_set = {str(node_id) for node_id in root_ids}
    matched = [
        item for item in diagnoses
        if isinstance(item, dict) and str(item.get("node_id")) in root_id_set
    ]
    if not matched:
        return None
    matched.sort(key=lambda item: item.get("step_index", 10**18))
    best = matched[0]
    return {
        "node_id": best.get("node_id"),
        "role": best.get("role"),
        "name": best.get("name"),
        "step_index": best.get("step_index"),
    }


def extract_root_cause_info(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    default_result = {
        "node_id": "None",
        "agent_role": "None",
        "agent_name": "None",
        "step_number": "-1",
        "valid": False,
        "source": "none",
    }
    if not isinstance(analysis_results, dict):
        return default_result

    root_node = analysis_results.get("root_cause_node")
    source = "root_cause_node"
    if not isinstance(root_node, dict):
        root_node = _extract_node_from_cascade(analysis_results)
        source = "cascade_failure.root_cause_nodes"
    if not isinstance(root_node, dict):
        return default_result

    return {
        "node_id": str(root_node.get("node_id", "None")),
        "agent_role": root_node.get("role") or "None",
        # Test-7 evaluation used the raw name first, then raw role.
        "agent_name": root_node.get("name") or "None",
        "step_number": _step_number_from_node(root_node),
        "valid": True,
        "source": source,
    }


def build_eval_block(
    graph: Dict[str, Any],
    analysis_results: Dict[str, Any],
    graph_index: int,
    include_full_analysis: bool = False,
) -> Dict[str, Any]:
    root_info = extract_root_cause_info(analysis_results)
    cascade = analysis_results.get("cascade_failure") or {}
    status = str(analysis_results.get("status", "")).upper()
    record = {
        "dataset_family": "who_when",
        "status": "FAILED" if status in {"ERROR", "FAILED"} else "OK",
        "graph_index": graph_index,
        "filename": graph.get("file_name"),
        "step_number": root_info["step_number"],
        "trace_id": graph.get("trace_id"),
        "agent_name": root_info["agent_name"],
        "agent_role": root_info["agent_role"],
        "node_id": root_info["node_id"],
        "valid": root_info["valid"],
        "root_source": root_info["source"],
        "root_cause_found": analysis_results.get("root_cause_found", False),
        "failure_established": cascade.get("failure_established"),
        "cascade_root_cause_nodes": cascade.get("root_cause_nodes", []),
        "error_chain": analysis_results.get("error_chain", []),
        "traversal_path": analysis_results.get("traversal_path", []),
        "cascade_failure": cascade,
    }
    if status in {"ERROR", "FAILED"}:
        record["error"] = str(
            analysis_results.get("error")
            or analysis_results.get("reason")
            or "Test-7 diagnosis failed"
        )
    if include_full_analysis:
        record["analysis_results"] = analysis_results
    return record


def make_diagnoser(
    settings: Any,
    llm_server: Optional[LLMServer] = None,
):
    llm_server = llm_server or LLMServer(
        api_key=settings.api_key,
        proxy=settings.proxy,
        url=settings.llm_url,
        model=settings.llm_model,
        resolve_env=False,
    )

    def diagnose(graph: Dict[str, Any]) -> Dict[str, Any]:
        # Test-7 has one initial C1 hypothesis and stack-based DFS.
        engine = ReverseDependencyDFS(
            graph_data=graph,
            model_name=settings.llm_model,
            api_key=settings.api_key,
            cascade_enabled=settings.cascade_enabled,
            llm_server=llm_server,
            include_ground_truth=settings.include_ground_truth,
        )
        return engine.find_root_cause()

    return diagnose


def run(
    *,
    graph_path: str,
    checkpoint_file: str,
    settings: Any,
    end: Optional[int] = None,
) -> List[Dict[str, Any]]:
    graphs = load_graphs(graph_path)
    llm_server = LLMServer(
        api_key=settings.api_key,
        proxy=settings.proxy,
        url=settings.llm_url,
        model=settings.llm_model,
        resolve_env=False,
    )
    diagnose_one = make_diagnoser(settings, llm_server=llm_server)
    return run_graph_diagnosis(
        graphs=graphs,
        checkpoint_file=checkpoint_file,
        diagnose_one=diagnose_one,
        llm_server=llm_server,
        build_record=lambda graph, diagnosis, idx: build_eval_block(
            graph,
            diagnosis,
            idx,
            include_full_analysis=settings.include_full_analysis,
        ),
        end=end,
        include_ground_truth=settings.include_ground_truth,
        include_eval_labels=settings.include_eval_labels,
    )

"""Single Who&When runner: current graph conversion + Original/Combined diagnosis.

The Original ReverseDependencyDFS implementation is preserved in diagnoser.py.
The CombinedReverseDependencyDFS implementation is appended to that same file.
"""
import re
from typing import Any, Dict, Iterable, List, Optional

from ...common.io import load_graphs, write_jsonl
from ...common.llm import LLMServer
from ...common.run_loop import run_graph_diagnosis
from .diagnoser import (
    ReverseDependencyDFS as OriginalReverseDependencyDFS,
    CombinedReverseDependencyDFS,
    canonical_agent_id as original_canonical_agent_id,
    _combined_canonical_agent_id,
)
from .graph_converter import process_json_path


DIAGNOSIS_MODES = {"separate", "combined"}
GRAPH_CONVERTER = "current"


def convert_to_graphs(
    input_dir: str,
    output_path: str,
    max_graphs: Optional[int] = None,
    edge_types: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Convert Who&When inputs with the single current Graph Converter."""
    graphs = process_json_path(
        input_dir,
        output_path=None,
        edge_types=edge_types,
    )
    if max_graphs is not None:
        graphs = graphs[:max_graphs]
    write_jsonl(output_path, graphs)
    return graphs


def step_number_from_node(root_node: Dict[str, Any]) -> str:
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


def build_original_eval_block(
    graph: Dict[str, Any],
    analysis_results: Dict[str, Any],
    graph_index: int,
    include_full_analysis: bool = False,
) -> Dict[str, Any]:
    """Keep the Original ZIP runner's prediction extraction semantics."""
    default = {
        "node_id": "None",
        "agent_role": "None",
        "agent_name": "None",
        "step_number": "-1",
        "valid": False,
        "source": "none",
    }
    root_node = (
        analysis_results.get("root_cause_node")
        if isinstance(analysis_results, dict)
        else None
    )
    if isinstance(root_node, dict):
        root_info = {
            "node_id": str(root_node.get("node_id", "None")),
            "agent_role": root_node.get("role") or "None",
            "agent_name": original_canonical_agent_id(root_node),
            "step_number": step_number_from_node(root_node),
            "valid": True,
            "source": "root_cause_node",
        }
    else:
        root_info = default

    cascade = (
        analysis_results.get("cascade_failure") or {}
        if isinstance(analysis_results, dict)
        else {}
    )
    record = {
        "dataset_family": "who_when",
        "status": "OK",
        "graph_index": graph_index,
        "filename": graph.get("file_name"),
        "trace_id": graph.get("trace_id"),
        "step_number": root_info["step_number"],
        "agent_name": root_info["agent_name"],
        "agent_role": root_info["agent_role"],
        "node_id": root_info["node_id"],
        "valid": root_info["valid"],
        "root_source": root_info["source"],
        "root_cause_found": (
            analysis_results.get("root_cause_found", False)
            if isinstance(analysis_results, dict)
            else False
        ),
        "failure_established": cascade.get("failure_established"),
        "error_chain": (
            analysis_results.get("error_chain", [])
            if isinstance(analysis_results, dict)
            else []
        ),
        "traversal_path": (
            analysis_results.get("traversal_path", [])
            if isinstance(analysis_results, dict)
            else []
        ),
        "cascade_failure": cascade,
    }
    if include_full_analysis:
        record["analysis_results"] = analysis_results
    return record


def build_combined_eval_block(
    graph: Dict[str, Any],
    analysis_results: Dict[str, Any],
    graph_index: int,
    include_full_analysis: bool = False,
) -> Dict[str, Any]:
    """Record builder for the current structured Combined diagnosis."""
    if not isinstance(analysis_results, dict):
        analysis_results = {}
    root_node = analysis_results.get("root_cause_node")
    if isinstance(root_node, dict):
        root_info = {
            "node_id": str(root_node.get("node_id", "None")),
            "agent_role": root_node.get("role") or "None",
            "agent_name": _combined_canonical_agent_id(root_node),
            "step_number": step_number_from_node(root_node),
            "valid": True,
            "source": "root_cause_node",
        }
    else:
        root_info = {
            "node_id": "None",
            "agent_role": "None",
            "agent_name": "None",
            "step_number": "-1",
            "valid": False,
            "source": "none",
        }

    cascade = analysis_results.get("cascade_failure") or {}
    stage1_error_count = analysis_results.get("stage1_error_count", 0)
    diagnosis_failed = str(analysis_results.get("status", "")).upper() in {
        "ERROR", "FAILED"
    }
    record = {
        "dataset_family": "who_when",
        "status": "FAILED" if stage1_error_count or diagnosis_failed else "OK",
        "graph_index": graph_index,
        "filename": graph.get("file_name"),
        "trace_id": graph.get("trace_id"),
        "step_number": root_info["step_number"],
        "agent_name": root_info["agent_name"],
        "agent_role": root_info["agent_role"],
        "node_id": root_info["node_id"],
        "valid": root_info["valid"],
        "root_source": root_info["source"],
        "root_cause_found": analysis_results.get("root_cause_found", False),
        "stage1_error_count": stage1_error_count,
        "failure_established": cascade.get("failure_established"),
        "error_chain": analysis_results.get("error_chain", []),
        "traversal_path": analysis_results.get("traversal_path", []),
        "cascade_failure": cascade,
    }
    if stage1_error_count:
        record["error"] = f"Stage 1 had {stage1_error_count} model or parsing errors"
        errors = analysis_results.get("stage1_errors") or []
        record["stage1_error_samples"] = [str(x)[:500] for x in errors[:5]]
    elif diagnosis_failed:
        record["error"] = str(
            analysis_results.get("error")
            or analysis_results.get("reason")
            or "Combined diagnosis failed"
        )
    if include_full_analysis:
        record["analysis_results"] = analysis_results
    return record


def make_diagnoser(
    settings: Any,
    llm_server: Optional[LLMServer] = None,
):
    diagnosis_mode = getattr(settings, "cascade_call_mode", "separate")
    if diagnosis_mode not in DIAGNOSIS_MODES:
        raise ValueError(
            f"Unknown Who&When cascade_call_mode {diagnosis_mode!r}; "
            f"choose one of {sorted(DIAGNOSIS_MODES)}."
        )
    llm_server = llm_server or LLMServer(
        api_key=settings.api_key,
        proxy=settings.proxy,
        url=settings.llm_url,
        model=settings.llm_model,
        resolve_env=False,
    )

    def diagnose(graph: Dict[str, Any]) -> Dict[str, Any]:
        if diagnosis_mode == "separate":
            # Original ZIP baseline: preserve its own prompts, DFS, candidate
            # generation and Stage-2 selection. Do not activate its optional
            # legacy combined_cascade_enabled path here.
            diagnosis_graph = graph
            if not getattr(settings, "include_ground_truth", True):
                diagnosis_graph = dict(graph)
                diagnosis_graph.pop("ground_truth", None)
            engine = OriginalReverseDependencyDFS(
                graph_data=diagnosis_graph,
                model_name=settings.llm_model,
                api_key=settings.api_key,
                openai_client=llm_server.client,
                cascade_enabled=settings.cascade_enabled,
                traversal_mode=settings.traversal_mode,
                llm_server=llm_server,
                combined_cascade_enabled=False,
                separate_order=getattr(settings, "separate_order", "local_then_cascade"),
            )
        else:
            engine = CombinedReverseDependencyDFS(
                graph_data=graph,
                model_name=settings.llm_model,
                api_key=settings.api_key,
                openai_client=llm_server.client,
                cascade_enabled=settings.cascade_enabled,
                traversal_mode=settings.traversal_mode,
                llm_server=llm_server,
                combined_cascade_enabled=True,
                cascade_call_mode="combined",
                prompt_context_policy=getattr(
                    settings, "prompt_context_policy", "legacy"
                ),
                max_prompt_upstream_chars=getattr(
                    settings, "max_prompt_upstream_chars", 6000
                ),
                max_causes=getattr(settings, "max_causes", 8),
                max_stage2_candidates=getattr(
                    settings, "max_stage2_candidates", 20
                ),
                include_ground_truth=getattr(
                    settings, "include_ground_truth", True
                ),
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
    diagnosis_mode = getattr(settings, "cascade_call_mode", "separate")
    if diagnosis_mode not in DIAGNOSIS_MODES:
        raise ValueError(
            f"Unsupported Who&When cascade_call_mode: {diagnosis_mode!r}"
        )

    graphs = load_graphs(graph_path)
    llm_server = LLMServer(
        api_key=settings.api_key,
        proxy=settings.proxy,
        url=settings.llm_url,
        model=settings.llm_model,
        resolve_env=False,
    )
    diagnose_one = make_diagnoser(settings, llm_server=llm_server)
    record_builder = (
        build_original_eval_block
        if diagnosis_mode == "separate"
        else build_combined_eval_block
    )

    def build_record(graph, diagnosis, idx):
        record = record_builder(
            graph,
            diagnosis,
            idx,
            include_full_analysis=settings.include_full_analysis,
        )
        record["who_when_diagnosis_mode"] = diagnosis_mode
        record["separate_order"] = (
            getattr(settings, "separate_order", "local_then_cascade")
            if settings.cascade_enabled and diagnosis_mode == "separate" else "not_applicable"
        )
        record["requested_edge_types"] = settings.edge_types if settings.edge_types is not None else None
        record["who_when_graph_converter"] = GRAPH_CONVERTER
        return record

    return run_graph_diagnosis(
        graphs=graphs,
        checkpoint_file=checkpoint_file,
        diagnose_one=diagnose_one,
        llm_server=llm_server,
        build_record=build_record,
        end=end,
        include_ground_truth=settings.include_ground_truth,
        include_eval_labels=settings.include_eval_labels,
    )

import re
from typing import Any, Dict, Iterable, List, Optional

from ...common.io import load_graphs, write_jsonl
from ...common.llm import LLMServer
from ...common.run_loop import run_graph_diagnosis
from independent_apps.shared.settings import RuntimeSettings
from .diagnoser import ReverseDependencyDFS, canonical_agent_id
from .graph_converter import process_json_path


def convert_to_graphs(
    input_dir: str,
    output_path: str,
    max_graphs: Optional[int] = None,
    edge_types: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    graphs = process_json_path(input_dir, output_path=None, edge_types=edge_types)
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


def extract_root_cause_info(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    default = {
        "node_id": "None",
        "agent_role": "None",
        "agent_name": "None",
        "step_number": "-1",
        "valid": False,
        "source": "none",
    }
    if not isinstance(analysis_results, dict):
        return default

    root_node = analysis_results.get("root_cause_node")
    if not isinstance(root_node, dict):
        return default

    return {
        "node_id": str(root_node.get("node_id", "None")),
        "agent_role": root_node.get("role") or "None",
        "agent_name": canonical_agent_id(root_node),
        "step_number": step_number_from_node(root_node),
        "valid": True,
        "source": "root_cause_node",
    }


def build_eval_block(
    graph: Dict[str, Any],
    analysis_results: Dict[str, Any],
    graph_index: int,
    include_full_analysis: bool = False,
) -> Dict[str, Any]:
    root_info = extract_root_cause_info(analysis_results)
    cascade = analysis_results.get("cascade_failure") or {}
    stage1_error_count = analysis_results.get("stage1_error_count", 0)
    diagnosis_failed = str(analysis_results.get("status", "")).upper() in {"ERROR", "FAILED"}
    block = {
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
        block["error"] = f"Stage 1 had {stage1_error_count} model or parsing errors"
        stage1_errors = analysis_results.get("stage1_errors") or []
        block["stage1_error_samples"] = [str(error)[:500] for error in stage1_errors[:5]]
    elif diagnosis_failed:
        block["error"] = str(
            analysis_results.get("error") or analysis_results.get("reason") or "Diagnosis failed"
        )
    if include_full_analysis:
        block["analysis_results"] = analysis_results
    return block


def make_diagnoser(settings: RuntimeSettings, llm_server: Optional[LLMServer] = None):
    llm_server = llm_server or LLMServer(
        api_key=settings.api_key,
        proxy=settings.proxy,
        url=settings.llm_url,
        model=settings.llm_model,
        resolve_env=False,
    )

    def diagnose(graph: Dict[str, Any]) -> Dict[str, Any]:
        cascade_call_mode = settings.cascade_call_mode
        engine = ReverseDependencyDFS(
            graph_data=graph,
            model_name=settings.llm_model,
            api_key=settings.api_key,
            openai_client=llm_server.client,
            cascade_enabled=settings.cascade_enabled,
            traversal_mode=settings.traversal_mode,
            llm_server=llm_server,
            combined_cascade_enabled=cascade_call_mode == "combined",
            cascade_call_mode=cascade_call_mode,
            prompt_context_policy=settings.prompt_context_policy,
            max_prompt_upstream_chars=settings.max_prompt_upstream_chars,
            max_causes=settings.max_causes,
            max_stage2_candidates=settings.max_stage2_candidates,
            include_ground_truth=settings.include_ground_truth,
        )
        return engine.find_root_cause()

    return diagnose


def run(
    *,
    graph_path: str,
    checkpoint_file: str,
    settings: RuntimeSettings,
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

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlsplit

from ...common.io import load_checkpoint, load_graphs, write_jsonl
from ...common.llm import LLMServer
from ...common.run_loop import run_graph_diagnosis
from ...common.traversal import resolve_traversal_mode
from independent_apps.shared.settings import RuntimeSettings
from . import diagnoser as trail_diagnoser
from .graph_converter import convert_json, convert_parquet


def convert_to_graphs(
    input_path: str,
    output_path: str,
    max_graphs: Optional[int] = None,
    edge_types: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    if Path(input_path).suffix.lower() in {".json", ".jsonl"}:
        graphs = convert_json(
            input_path, output_path=output_path, max_rows=max_graphs, edge_types=edge_types,
        )
        return graphs

    graphs = convert_parquet(
        input_path, output_path=None, max_rows=max_graphs, edge_types=edge_types,
    )
    write_jsonl(output_path, graphs)
    return graphs


def make_config(settings: RuntimeSettings) -> trail_diagnoser.DiagnoserConfig:
    return trail_diagnoser.DiagnoserConfig(
        node_id_key="node_id",
        edge_src_key="source",
        edge_dst_key="target",
        llm_model=settings.llm_model,
        cascade_enabled=settings.cascade_enabled,
        cascade_call_mode=settings.cascade_call_mode,
        separate_order=settings.separate_order,
        prompt_context_policy=settings.prompt_context_policy,
        max_causes=settings.max_causes,
        cause_state_mode=settings.trail_cause_state_mode,
        trail_schedule_scope=settings.trail_schedule_scope,
        final_hypothesis_policy=settings.trail_final_hypothesis_policy,
        enable_scoring=settings.enable_scoring,
        use_llm_error_synthesis=settings.use_llm_error_synthesis,
        traversal_mode=resolve_traversal_mode(settings.traversal_mode, default="bfs"),
        include_ground_truth_in_node_prompt=False,
        max_current_node_string_length=settings.max_current_node_string_length,
        max_context_node_string_length=settings.max_upstream_node_string_length,
        max_prompt_upstream_chars=settings.max_prompt_upstream_chars,
        max_downstream_node_string_length=settings.max_downstream_node_string_length,
    )


# Frozen implementation identities from the pre-opt-in baseline. The default
# TRAIL path must remain resumable even though this module now contains
# opt-in scheduling and hypothesis/context policies.
_LEGACY_TRAIL_DIAGNOSER_SHA256 = "85C975E50FAF8F699AE4BA7EE84FF44CA902AE060C35E9309E22FA680DC05DA6"
_LEGACY_DIAGNOSIS_ENGINE_SHA256 = "2BF78AF210C4D996A58C0EC88B513E8E24733DCFC8D50EEE0F22D6DC978806B2"


def _trail_source_identity_hashes(settings: RuntimeSettings, config: Any) -> tuple[str, str]:
    """Return prompt/engine identities while preserving the frozen legacy arm."""
    prompt_source_sha = hashlib.sha256(Path(trail_diagnoser.__file__).read_bytes()).hexdigest()
    diagnosis_engine_path = (
        Path(trail_diagnoser.__file__).resolve().parents[2]
        / "common" / "diagnosis" / "engine.py"
    )
    engine_source_sha = hashlib.sha256(diagnosis_engine_path.read_bytes()).hexdigest()
    is_frozen_legacy_arm = (
        settings.trail_schedule_scope == "legacy"
        and settings.trail_final_hypothesis_policy == "gated"
        and config.prompt_context_policy == "legacy"
    )
    if is_frozen_legacy_arm:
        return _LEGACY_TRAIL_DIAGNOSER_SHA256, _LEGACY_DIAGNOSIS_ENGINE_SHA256
    return prompt_source_sha, engine_source_sha



def make_diagnoser(settings: RuntimeSettings, llm_server: Optional[LLMServer] = None):
    llm_server = llm_server or LLMServer(
        api_key=settings.api_key,
        proxy=settings.proxy,
        url=settings.llm_url,
        model=settings.llm_model,
        resolve_env=False,
        max_retries=0,
    )
    config = make_config(settings)

    def diagnose(graph: Dict[str, Any]) -> Dict[str, Any]:
        engine = trail_diagnoser.AgentGraphDiagnoser(
            graph,
            config=config,
            llm_server=llm_server,
        )
        return engine.run_diagnosis()

    return diagnose


def build_record(graph: Dict[str, Any], diagnosis: Dict[str, Any], graph_index: int) -> Dict[str, Any]:
    if not isinstance(diagnosis, dict):
        diagnosis = {"raw_result": diagnosis}
    diagnosis.setdefault("dataset_family", "trail")
    diagnosis.setdefault("trace_id", graph.get("trace_id"))
    diagnosis.setdefault("graph_index", graph_index)
    diagnosis.setdefault("status", "OK")
    return diagnosis


def is_local_prediction_complete(record: Dict[str, Any]) -> bool:
    """Return whether a checkpoint contains a reusable TRAIL Local Prediction.

    New records use the explicit Local status. Historical records retain the
    old top-level success rule; in particular, an old PARTIAL record is never
    promoted because its Local/Cause Path split was not recorded.
    """
    if not isinstance(record, dict):
        return False
    if "local_prediction_status" in record:
        return str(record.get("local_prediction_status", "")).upper() == "SUCCESS"
    return str(record.get("status", "")).upper() in {"SUCCESS", "OK", "COMPLETED"}


def effective_separate_order(settings: RuntimeSettings) -> str:
    if not settings.cascade_enabled or settings.cascade_call_mode != "separate":
        return "not_applicable"
    order = str(settings.separate_order)
    if order not in {"cascade_then_local", "local_then_cascade"}:
        raise ValueError(f"Unsupported Separate execution order: {order}")
    return order


def validate_separate_order_checkpoint(checkpoint_file: str, settings: RuntimeSettings) -> None:
    """Prevent completed predictions from crossing Separate order versions."""
    requested = effective_separate_order(settings)
    if requested == "not_applicable" or not Path(checkpoint_file).is_file():
        return
    for index, record in enumerate(load_checkpoint(checkpoint_file)):
        if not is_local_prediction_complete(record):
            continue
        cascade = record.get("cascade_failure") if isinstance(record, dict) else None
        runtime = record.get("runtime_options") if isinstance(record, dict) else None
        cascade = cascade if isinstance(cascade, dict) else {}
        runtime = runtime if isinstance(runtime, dict) else {}
        recorded_mode = str(
            record.get("diagnosis_mode")
            or cascade.get("cascade_call_mode")
            or runtime.get("cascade_call_mode")
            or ""
        )
        if recorded_mode != "separate":
            raise ValueError(
                f"Checkpoint {checkpoint_file} has a completed record with diagnosis_mode="
                f"{recorded_mode or 'UNRECORDED'}; requested Separate order {requested}. "
                "Choose a new checkpoint path."
            )
        recorded_order = (
            record.get("separate_order")
            or cascade.get("separate_order")
            or runtime.get("separate_order")
        )
        # Pre-versioned Separate checkpoints used Local -> Cascade.
        if recorded_order is None:
            recorded_order = "local_then_cascade"
        if recorded_order != requested:
            raise ValueError(
                f"Checkpoint {checkpoint_file} record {index} uses separate_order="
                f"{recorded_order}; requested {requested}. Choose a new checkpoint path."
            )


def failed_record_metadata(
    exc: BaseException,
    graph: Dict[str, Any],
    *,
    cascade_enabled: bool,
    cascade_call_mode: str = "combined",
    separate_order: str = "cascade_then_local",
) -> Dict[str, Any]:
    """Describe incomplete Local/Cause Path work without fabricating errors."""
    partial = getattr(exc, "shared_diagnosis_partial", None)
    partial = partial if isinstance(partial, dict) else {}
    completed = [str(item) for item in partial.get("completed_node_ids", [])]
    unexecuted = [str(item) for item in partial.get("unexecuted_node_ids", [])]
    failed_node = partial.get("failed_node_id")
    failed_node = str(failed_node) if failed_node is not None else None
    stage = str(partial.get("failure_stage") or "diagnosis")
    current_local_available = partial.get("current_local_inspection") is not None
    local_status = "PARTIAL" if completed or current_local_available else "FAILED"
    local_failed_nodes = (
        [failed_node]
        if failed_node and stage in {
            "local_inspection", "combined_inspection", "local_after_cascade_inspection",
        }
        else []
    )
    expected = list(dict.fromkeys(completed + unexecuted + ([failed_node] if failed_node else [])))
    cause_status = "PARTIAL" if cascade_enabled else "NOT_RUN"
    cause_reasons = (
        [f"DIAGNOSIS_INTERRUPTED_AT_{stage.upper()}"] if cascade_enabled else []
    )
    top_status = "PARTIAL" if local_status == "PARTIAL" else "FAILED"
    diagnosis_mode = cascade_call_mode if cascade_enabled else "disabled"
    effective_order = (
        separate_order
        if cascade_enabled and cascade_call_mode == "separate"
        else "not_applicable"
    )
    return {
        "status": top_status,
        "diagnosis_mode": diagnosis_mode,
        "separate_order": effective_order,
        "local_prediction_status": local_status,
        "cause_path_coverage_status": cause_status,
        "cause_path_partial_reasons": cause_reasons,
        "local_prediction_coverage": {
            "expected_node_ids": expected,
            "completed_node_ids": completed,
            "failed_node_ids": local_failed_nodes,
            "unvisited_node_ids": unexecuted,
            "current_local_inspection_available": current_local_available,
            "final_errors_assembled": False,
        },
        "cause_path_coverage": {
            "status": cause_status,
            "partial_reasons": cause_reasons,
            "not_run_reason": "CASCADE_DISABLED" if not cascade_enabled else None,
        },
    }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    *,
    graph_path: str,
    checkpoint_file: str,
    settings: RuntimeSettings,
    end: Optional[int] = None,
) -> List[Dict[str, Any]]:
    graphs = load_graphs(graph_path)
    validate_separate_order_checkpoint(checkpoint_file, settings)


    request_token_budget_config = None
    if settings.trail_request_token_budget_enabled:
        request_token_budget_config = {
            "context_limit": settings.trail_request_context_limit,
            "hard_projected_limit": settings.trail_request_hard_projected_limit,
            "completion_reserve": settings.trail_request_completion_reserve,
            "safety_factor": settings.trail_request_safety_factor,
            "tokenizer": settings.trail_request_tokenizer,
        }
    llm_server = LLMServer(
        api_key=settings.api_key,
        proxy=settings.proxy,
        url=settings.llm_url,
        model=settings.llm_model,
        resolve_env=False,
        max_retries=0,
        structured_output_token_limit=settings.trail_max_completion_tokens,
        api_request_limit=settings.trail_api_request_limit,
        request_token_budget_config=request_token_budget_config,
    )
    diagnose_one = make_diagnoser(settings, llm_server=llm_server)
    config = make_config(settings)
    prompt_sha, edge_policy_source_sha = _trail_source_identity_hashes(settings, config)
    schema_list = [
            trail_diagnoser.CombinedNodeInspection.model_json_schema(),
            trail_diagnoser.CombinedNodeInspectionH1.model_json_schema(),
            trail_diagnoser.FinalFailureAnalysis.model_json_schema(),
            trail_diagnoser.SingleNodeLocalInspection.model_json_schema(),
            trail_diagnoser.CascadeNodeInspection.model_json_schema(),
            trail_diagnoser.CascadeNodeInspectionH1.model_json_schema(),
            trail_diagnoser.TRAILDiagnosisOutput.model_json_schema(),
        ]
    schema_material = json.dumps(
        schema_list,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    schema_sha = hashlib.sha256(schema_material.encode("utf-8")).hexdigest()
    endpoint = str(settings.llm_url or "")
    endpoint_host = urlsplit(endpoint).hostname or ""
    endpoint_sha = hashlib.sha256(endpoint.encode("utf-8")).hexdigest() if endpoint else ""
    source_file_sha = _sha256_file(graph_path)
    trace_id_list_sha = hashlib.sha256(
        "\n".join(str(graph.get("trace_id", "")) for graph in graphs).encode("utf-8")
    ).hexdigest()
    canonical_graph = json.dumps(
        graphs, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    )
    trace_graph_sha = hashlib.sha256(canonical_graph.encode("utf-8")).hexdigest()
    evaluation_path = Path(trail_diagnoser.__file__).resolve().parents[2] / "evaluation" / "trail.py"
    evaluation_sha = _sha256_file(str(evaluation_path))
    official_path = Path(__file__).resolve().parents[3] / "audit" / "codex_second_pass" / "references" / "trail_calculate_scores.py"
    official_sha = _sha256_file(str(official_path)) if official_path.exists() else None
    edge_kinds = sorted({
        str(edge.get("kind"))
        for graph in graphs for edge in graph.get("edges") or []
        if isinstance(edge, dict) and edge.get("kind")
    })
    fingerprint_material = {
        "dataset": "TRAIL",
        "source_graph_file_sha256": source_file_sha,
        "trace_id_list_sha256": trace_id_list_sha,
        "trace_graph_hash": trace_graph_sha,
        "edge_types": edge_kinds,
        "node_retention": "step_and_container_spans_filtered_v1",
        "cause_state_mode": settings.trail_cause_state_mode,
        "schedule_scope": settings.trail_schedule_scope,
        "final_hypothesis_policy": settings.trail_final_hypothesis_policy,
        "h1_edge_policy": "dependency_only" if settings.trail_cause_state_mode == "H1" else None,
        "edge_policy_implementation_sha256": edge_policy_source_sha,
        "model": settings.llm_model,
        "endpoint_host": endpoint_host,
        "endpoint_sha256": endpoint_sha,
        "temperature": None,
        "seed": None,
        "sdk_max_retries": 0,
        "max_completion_tokens": settings.trail_max_completion_tokens,
        "request_token_budget": request_token_budget_config,
        "api_request_limit": settings.trail_api_request_limit,
        "max_token_retries_per_logical_request": 1,
        "cascade_enabled": settings.cascade_enabled,
        "cascade_call_mode": config.cascade_call_mode,
        "separate_order": effective_separate_order(settings),
        "prompt_context_policy": config.prompt_context_policy,
        "traversal_mode": config.traversal_mode,
        "max_string_length": config.max_string_length,
        "max_current_node_string_length": config.max_current_node_string_length,
        "max_context_node_string_length": config.max_context_node_string_length,
        "max_downstream_node_string_length": config.max_downstream_node_string_length,
        "max_prompt_upstream_chars": config.max_prompt_upstream_chars,
        "max_causes": config.max_causes,
        "max_cause_path_states": config.max_cause_path_states,
        "enable_scoring": config.enable_scoring,
        "use_llm_error_synthesis": config.use_llm_error_synthesis,
        "include_ground_truth_in_node_prompt": False,
        "keep_node_fields": sorted(config.keep_node_fields or []),
        "data_flow_kinds": sorted(config.data_flow_kinds),
        "execution_flow_kinds": sorted(config.execution_flow_kinds),
        "parent_kinds": sorted(config.parent_kinds),
        "prompt_source_sha256": prompt_sha,
        "schema_sha256": schema_sha,
        "evaluation_protocol": "patronus-ai/trail-benchmark@0ffbed9db859b4a66250dc783fa4dccf86869595",
        "local_evaluator_sha256": evaluation_sha,
        "official_evaluator_sha256": official_sha,
        "ground_truth_in_diagnosis": False,
    }
    if (
        settings.trail_cause_state_mode == "H1"
        and settings.cascade_enabled
        and config.cascade_call_mode == "separate"
        and effective_separate_order(settings) == "cascade_then_local"
    ):
        fingerprint_material["cascade_local_evidence_validation_version"] = (
            "trail_h1_cascade_local_evidence_v1"
        )
    if settings.trail_schedule_scope == "legacy":
        # Keep historical default fingerprints resumable; the new all_nodes
        # arm gets a distinct identity.
        fingerprint_material.pop("schedule_scope", None)
    if settings.trail_final_hypothesis_policy == "gated":
        # Keep pre-policy default run identities unchanged.
        fingerprint_material.pop("final_hypothesis_policy", None)
    run_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return run_graph_diagnosis(
        graphs=graphs,
        checkpoint_file=checkpoint_file,
        diagnose_one=diagnose_one,
        llm_server=llm_server,
        build_record=build_record,
        end=end,
        # TRAIL diagnostics are always blind to evaluation labels and answers.
        include_ground_truth=False,
        include_eval_labels=settings.include_eval_labels,
        run_fingerprint=run_fingerprint,
        completion_predicate=is_local_prediction_complete,
        failed_record_metadata=lambda exc, graph: failed_record_metadata(
            exc,
            graph,
            cascade_enabled=settings.cascade_enabled,
            cascade_call_mode=settings.cascade_call_mode,
            separate_order=settings.separate_order,
        ),
    )


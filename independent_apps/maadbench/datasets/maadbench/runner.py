from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit
from ...common.io import load_checkpoint, load_graphs
from ...common.llm import LLMServer
from ...common.run_loop import run_graph_diagnosis
from ...common.traversal import resolve_traversal_mode
from ...common.graph_utils import normalize_edge_types
from independent_apps.shared.settings import RuntimeSettings
from .diagnoser import (
    MAADJudge,
    _MAAD_PATH_CONFLICT_ANALYSIS_VERSION,
    _MAAD_PATH_METHOD_VERSION,
    diagnose_graph,
)
from .graph_converter import convert_to_graphs
from .graph_converter import EDGE_TYPES
from .split import prepare_official_dev_split, prepare_official_test_split, source_matches_trace_id


"""MAADBench conversion-to-diagnosis adapter for the shared run loop."""


_MAAD_CASCADE_LOCAL_EVIDENCE_VALIDATION_VERSION = "maad_path_cascade_local_evidence_v1"


def _api_identity(settings: RuntimeSettings) -> dict[str, Any]:
    endpoint = urlsplit(str(settings.llm_url or ""))
    host = (endpoint.hostname or "").lower()
    is_azure = host.endswith(".openai.azure.com") or host.endswith(".services.ai.azure.com")
    return {
        "api_provider": "azure_openai" if is_azure else "openai_compatible",
        "api_endpoint_host": host or None,
        "azure_deployment": str(settings.llm_model) if is_azure else None,
        "api_version": (
            os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"
        ) if is_azure else None,
    }


def _uses_cascade_local_prevalidation(settings: RuntimeSettings, cause_state_mode: Optional[str] = None) -> bool:
    options = settings.dataset_options or {}
    mode = cause_state_mode or str(options.get("maadbench_cause_state_mode", "legacy"))
    return bool(
        mode == "path"
        and settings.cascade_enabled
        and settings.cascade_call_mode == "separate"
        and settings.separate_order == "cascade_then_local"
    )


def _validate_checkpoint_config(
    checkpoint_file: str,
    settings: RuntimeSettings,
    graph_identity: Optional[dict[str, Any]] = None,
) -> None:
    """Fail closed on model/split changes and checkpoints from archived profiles."""
    path = Path(checkpoint_file).expanduser()
    if not path.is_file():
        return
    options = settings.dataset_options or {}
    cause_state_mode = str(options.get("maadbench_cause_state_mode", "legacy"))
    max_path_states = int(options.get("maadbench_max_cause_path_states", 4096))
    prevalidation_version = (
        _MAAD_CASCADE_LOCAL_EVIDENCE_VALIDATION_VERSION
        if _uses_cascade_local_prevalidation(settings, cause_state_mode)
        else "not_applicable"
    )
    expected = {
        "model": str(settings.llm_model),
        "diagnosis_scope": str(options.get("maadbench_diagnosis_scope", "all")),
        "dev_seed": int(options.get("maadbench_dev_seed", int(options.get("maadbench_seed", 5)) + 101)),
        "dev_size": float(options.get("maadbench_dev_size", 0.2)),
        "dev_trace_limit": options.get("maadbench_dev_trace_limit"),
        "dev_pilot_seed": int(options.get("maadbench_dev_pilot_seed", 107)),
        "context": str(settings.prompt_context_policy),
        "traversal": resolve_traversal_mode(settings.traversal_mode, default="bfs"),
        "cascade_enabled": bool(settings.cascade_enabled),
        "cascade_call_mode": str(settings.cascade_call_mode if settings.cascade_enabled else "disabled"),
        "separate_order": (
            str(settings.separate_order)
            if settings.cascade_enabled and settings.cascade_call_mode == "separate"
            else "not_applicable"
        ),
        "graph_identity": graph_identity,
        "cause_state_mode": cause_state_mode,
        "max_cause_path_states": max_path_states if cause_state_mode == "path" else "not_applicable",
        "cause_path_policy_version": _MAAD_PATH_METHOD_VERSION if cause_state_mode == "path" else "not_applicable",
        "conflict_analysis_version": _MAAD_PATH_CONFLICT_ANALYSIS_VERSION if cause_state_mode == "path" else "not_applicable",
        "method_version": _MAAD_PATH_METHOD_VERSION if cause_state_mode == "path" else "not_applicable",
        "path_method_source_identity": (
            _path_method_source_identity(
                cascade_local_prevalidation=prevalidation_version != "not_applicable",
            ) if cause_state_mode == "path" else "not_applicable"
        ),
        "cascade_local_evidence_validation_version": prevalidation_version,
        **_api_identity(settings),
    }
    archived_profile_fields = {
        "prompt_profile": ({"prompt_profile": "baseline"}, {"maadbench_prompt_profile": "baseline"}),
        "prompt_version": ({"prompt_version": "baseline"}, {"maadbench_prompt_version": "baseline"}),
        "local_schema_variant": ({"local_schema_variant": "legacy_roles"}, {"maadbench_local_schema_variant": "legacy_roles"}),
        "node_context_format": ({"node_context_format": "keyed"}, {"maadbench_node_context_format": "keyed"}),
    }
    for index, record in enumerate(load_checkpoint(str(path))):
        if not isinstance(record, dict) or str(record.get("status", "")).upper() in {"PENDING", "FAILED", "ERROR", "TIMEOUT"}:
            continue
        runtime = record.get("runtime_options") or {}
        for field, (record_allowed, runtime_allowed) in archived_profile_fields.items():
            value = record.get(field)
            if value is not None and record_allowed.get(field) != value:
                raise ValueError(f"Checkpoint {path} uses archived MAADBench {field}={value!r}; choose a new path.")
            runtime_key = next(iter(runtime_allowed))
            runtime_value = runtime.get(runtime_key)
            if runtime_value is not None and runtime_allowed[runtime_key] != runtime_value:
                raise ValueError(f"Checkpoint {path} uses archived MAADBench {runtime_key}={runtime_value!r}; choose a new path.")
        if bool(record.get("value_provenance_enabled", runtime.get("maadbench_value_provenance", False))):
            raise ValueError(f"Checkpoint {path} enables archived MAADBench value provenance; choose a new path.")
        if bool(record.get("post_diagnosis_reconciliation_enabled", runtime.get("maadbench_post_diagnosis_reconciliation", False))):
            raise ValueError(f"Checkpoint {path} enables archived MAADBench reconciliation; choose a new path.")
        found = {
            "model": str(record.get("model") or ""),
            "diagnosis_scope": str(record.get("diagnosis_scope") or runtime.get("maadbench_diagnosis_scope") or "all"),
            "dev_seed": int(runtime.get("maadbench_dev_seed", int(runtime.get("maadbench_seed", 5)) + 101)),
            "dev_size": float(runtime.get("maadbench_dev_size", 0.2)),
            "dev_trace_limit": runtime.get("maadbench_dev_trace_limit"),
            "dev_pilot_seed": int(runtime.get("maadbench_dev_pilot_seed", 107)),
            "context": str(record.get("prompt_context_policy") or runtime.get("prompt_context_policy") or "legacy"),
            "traversal": str(record.get("traversal_mode") or runtime.get("traversal_mode") or "bfs"),
            "cascade_enabled": bool(record.get("cascade_enabled", runtime.get("cascade_enabled", True))),
            "cascade_call_mode": str(record.get("diagnosis_mode") or runtime.get("cascade_call_mode") or "combined"),
            "separate_order": str(
                record.get("separate_order")
                or runtime.get("separate_order")
                or (
                    "local_then_cascade"
                    if str(record.get("diagnosis_mode") or runtime.get("cascade_call_mode") or "combined") == "separate"
                    else "not_applicable"
                )
            ),
            "graph_identity": record.get("graph_identity") or runtime.get("graph_identity"),
            "cause_state_mode": str(
                record.get("cause_state_mode")
                or runtime.get("maadbench_cause_state_mode")
                or "legacy"
            ),
            "max_cause_path_states": (
                int(runtime.get("maadbench_max_cause_path_states", 4096))
                if str(record.get("cause_state_mode") or runtime.get("maadbench_cause_state_mode") or "legacy") == "path"
                else "not_applicable"
            ),
            "cause_path_policy_version": str(
                record.get("cause_path_policy_version")
                or runtime.get("cause_path_policy_version", "maad_path_v1")
                if str(record.get("cause_state_mode") or runtime.get("maadbench_cause_state_mode") or "legacy") == "path"
                else "not_applicable"
            ),
            "conflict_analysis_version": str(
                record.get("conflict_analysis_version")
                or runtime.get("conflict_analysis_version", "legacy_state_signature_v1")
                if str(record.get("cause_state_mode") or runtime.get("maadbench_cause_state_mode") or "legacy") == "path"
                else "not_applicable"
            ),
            "method_version": str(
                record.get("method_version")
                or runtime.get("method_version", "maad_path_v1")
                if str(record.get("cause_state_mode") or runtime.get("maadbench_cause_state_mode") or "legacy") == "path"
                else "not_applicable"
            ),
            "path_method_source_identity": (
                runtime.get("path_method_source_identity", "unverified")
                if str(record.get("cause_state_mode") or runtime.get("maadbench_cause_state_mode") or "legacy") == "path"
                else "not_applicable"
            ),
            "cascade_local_evidence_validation_version": str(
                record.get("cascade_local_evidence_validation_version")
                or runtime.get("cascade_local_evidence_validation_version")
                or "not_applicable"
            ),
            "api_provider": str(runtime.get("api_provider") or ""),
            "api_endpoint_host": runtime.get("api_endpoint_host"),
            "azure_deployment": runtime.get("azure_deployment"),
            "api_version": runtime.get("api_version"),
        }
        if not found["cascade_enabled"]:
            found["cascade_call_mode"] = "disabled"
        if found != expected:
            raise ValueError(
                f"Checkpoint {path} has a different model, split, traversal, or diagnosis configuration at record {index}. "
                f"Existing={found}; requested={expected}. Choose a new checkpoint path."
            )


def _eval_run_dir(checkpoint_file: str) -> Path:
    path = Path(checkpoint_file).expanduser()
    return path.parent / f"{path.stem}_official_eval"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _edge_variant_graph_identity(graphs: list[dict[str, Any]], graph_path: str) -> Optional[dict[str, Any]]:
    requested_by_graph: list[tuple[str, ...]] = []
    for index, graph in enumerate(graphs):
        metadata = graph.get("graph_metadata") or {}
        requested = metadata.get("requested_edge_types")
        if requested is None:
            requested_by_graph.append(())
            continue
        selected = normalize_edge_types(requested, supported=EDGE_TYPES)
        ordered = tuple(kind for kind in EDGE_TYPES if kind in selected)
        actual = {
            edge.get("kind") for edge in graph.get("edges") or []
            if isinstance(edge, dict)
        }
        if actual - selected:
            raise ValueError(
                f"MAADBench edge variant graph {index} contains unselected kinds: {sorted(actual - selected)}"
            )
        requested_by_graph.append(ordered)

    if not any(requested_by_graph):
        return None
    if any(not requested for requested in requested_by_graph):
        raise ValueError("MAADBench graph file mixes edge-variant and legacy graphs; refusing diagnosis.")
    if len(set(requested_by_graph)) != 1:
        raise ValueError("MAADBench graph file mixes requested edge subsets; refusing diagnosis.")
    return {
        "graph_file_sha256": _sha256_file(graph_path),
        "requested_edge_types": list(requested_by_graph[0]),
    }


def _canonical_graph_identity(graphs: list[dict[str, Any]], graph_path: str) -> dict[str, Any]:
    """Strong graph identity for opt-in policies, including ordinary full-edge graphs."""
    canonical = json.dumps(graphs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    trace_ids = [str(graph.get("trace_id", "")) for graph in graphs]
    requested_sets = []
    for graph in graphs:
        metadata = graph.get("graph_metadata") or {}
        requested = metadata.get("requested_edge_types")
        selected = normalize_edge_types(requested, supported=EDGE_TYPES) if requested is not None else set(EDGE_TYPES)
        ordered = [kind for kind in EDGE_TYPES if kind in selected]
        actual = {
            str(edge.get("kind")) for edge in graph.get("edges") or []
            if isinstance(edge, dict) and edge.get("kind") is not None
        }
        unexpected = actual - selected
        if unexpected:
            raise ValueError(
                "MAADBench graph contains edges outside its requested graph identity: "
                f"{sorted(unexpected)}"
            )
        requested_sets.append(tuple(ordered))
    if len(set(requested_sets)) > 1:
        raise ValueError("MAADBench graph file mixes edge subsets; refusing path-mode diagnosis.")
    requested = list(requested_sets[0]) if requested_sets else list(EDGE_TYPES)
    return {
        "graph_file_sha256": _sha256_file(graph_path),
        "canonical_graphs_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "trace_ids_sha256": hashlib.sha256(
            json.dumps(trace_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "graph_count": len(graphs),
        "requested_edge_types": requested,
    }


def _path_method_source_identity(*, cascade_local_prevalidation: bool = False) -> dict[str, str]:
    """Fingerprint the opt-in path method, engine, state rules, and schema/prompt source."""
    package_dir = Path(__file__).resolve().parents[2]
    paths = {
        "maadbench_runner": Path(__file__).resolve(),
        "maadbench_diagnoser": Path(__file__).with_name("diagnoser.py").resolve(),
        "shared_engine": package_dir / "common" / "diagnosis" / "engine.py",
        "cause_policies": package_dir / "common" / "diagnosis" / "policies.py",
        "path_state": package_dir / "common" / "diagnosis" / "path_state.py",
    }
    identity = {name: _sha256_file(str(path)) for name, path in paths.items()}
    identity["cause_path_policy_version"] = _MAAD_PATH_METHOD_VERSION
    identity["conflict_analysis_version"] = _MAAD_PATH_CONFLICT_ANALYSIS_VERSION
    if cascade_local_prevalidation:
        identity["cascade_local_evidence_validation_version"] = (
            _MAAD_CASCADE_LOCAL_EVIDENCE_VALIDATION_VERSION
        )
    return identity


def _maad_path_local_prediction_complete(record: dict[str, Any]) -> bool:
    if str(record.get("cause_state_mode", "")).lower() != "path":
        return False
    if str(record.get("local_prediction_status", "")).upper() != "SUCCESS":
        return False
    coverage = record.get("coverage")
    predictions = record.get("action_predictions")
    if not isinstance(coverage, dict) or not isinstance(predictions, list):
        return False
    expected = coverage.get("expected_local_action_count")
    completed = coverage.get("completed_local_action_count")
    if not isinstance(expected, int) or expected < 1 or completed != expected or len(predictions) != expected:
        return False
    steps: list[int] = []
    nodes: list[str] = []
    for item in predictions:
        if not isinstance(item, dict) or item.get("step_index") is None or not isinstance(item.get("has_error"), bool):
            return False
        try:
            score = float(item.get("local_anomaly_score", item.get("anomaly_score")))
        except (TypeError, ValueError):
            return False
        if not 0.0 <= score <= 1.0:
            return False
        steps.append(int(item["step_index"]))
        nodes.append(str(item.get("node_id", "")))
    return (
        sorted(steps) == list(range(expected))
        and len(set(steps)) == expected
        and all(nodes)
        and len(set(nodes)) == expected
    )


def _maad_checkpoint_complete(record: dict[str, Any]) -> bool:
    status = str(record.get("status", "")).upper()
    if str(record.get("cause_state_mode", "")).lower() == "path":
        if status not in {"SUCCESS", "PARTIAL"} or not _maad_path_local_prediction_complete(record):
            return False
        cause_status = str(record.get("cause_path_coverage_status", "")).upper()
        return (status == "SUCCESS" and cause_status == "COMPLETE") or (
            status == "PARTIAL" and cause_status == "PARTIAL"
        )
    return status in {"SUCCESS", "OK", "COMPLETED"}


def _failed_path_run_metadata(exc: BaseException, graph: dict[str, Any]) -> dict[str, Any]:
    partial = getattr(exc, "shared_diagnosis_partial", None)
    nodes = [
        node for node in graph.get("nodes", [])
        if isinstance(node, dict) and int(node.get("step_index", -1)) >= 0
    ]
    expected_ids = {str(node.get("node_id", "")) for node in nodes}
    completed_ids: set[str] = set()
    if isinstance(partial, dict):
        for item in partial.get("history", []) or []:
            if isinstance(item, dict) and isinstance(item.get("local"), dict):
                completed_ids.add(str(item.get("node_id", "")))
        if isinstance(partial.get("current_local_inspection"), dict):
            completed_ids.add(str(partial.get("failed_node_id", "")))
    completed_ids.intersection_update(expected_ids)
    local_status = "PARTIAL" if completed_ids else "FAILED"
    failure_stage = str(partial.get("failure_stage", "unknown")) if isinstance(partial, dict) else "unknown"
    cause_status = (
        "NOT_RUN"
        if not isinstance(partial, dict)
        or failure_stage in {"final_hypothesis", "initialize_hypotheses"}
        else "PARTIAL"
    )
    reasons = ["RUN_FAILED"]
    if cause_status == "NOT_RUN":
        reasons.append("CASCADE_NOT_STARTED")
    else:
        reasons.append(f"FAILURE_STAGE:{failure_stage}")
    return {
        "cause_state_mode": "path",
        "local_prediction_status": local_status,
        "cause_path_coverage_status": cause_status,
        "cause_path_partial_reasons": reasons,
        "coverage": {
            "expected_local_action_count": len(nodes),
            "completed_local_action_count": len(completed_ids),
            "missing_local_node_ids": sorted(expected_ids - completed_ids),
            "local_prediction_status": local_status,
            "cause_path_coverage_status": cause_status,
            "partial": True,
        },
    }


def make_diagnoser(
    settings: RuntimeSettings,
    llm_server: Optional[LLMServer] = None,
    *,
    debug_output_dir: Optional[str] = None,
):
    server = llm_server or LLMServer(
        api_key=settings.api_key,
        proxy=settings.proxy,
        url=settings.llm_url,
        model=settings.llm_model,
        max_retries=0,
        resolve_env=False,
    )
    dump_visible_context = bool((settings.dataset_options or {}).get("maadbench_dump_visible_context", False))
    judge = MAADJudge(server, dump_visible_context_dir=debug_output_dir if dump_visible_context else None)

    def diagnose(graph: Dict[str, Any]) -> Dict[str, Any]:
        options = settings.dataset_options or {}
        return diagnose_graph(
            graph,
            judge,
            cascade_enabled=settings.cascade_enabled,
            cascade_call_mode=settings.cascade_call_mode,
            separate_order=settings.separate_order,
            traversal_mode=resolve_traversal_mode(settings.traversal_mode, default="bfs"),
            prompt_context_policy=settings.prompt_context_policy,
            max_prompt_upstream_chars=settings.max_prompt_upstream_chars,
            max_causes=settings.max_causes,
            cause_state_mode=str(options.get("maadbench_cause_state_mode", "legacy")),
            max_path_states=int(options.get("maadbench_max_cause_path_states", 4096)),
            dump_visible_context=dump_visible_context,
            debug_output_dir=debug_output_dir if dump_visible_context else None,
        )

    return diagnose, server


def _selected_test_graph_indices(
    graphs: list[dict[str, Any]],
    input_path: str,
    checkpoint_file: str,
    options: dict[str, Any],
) -> set[int]:
    _official, _args, _train, test, _output_dir = prepare_official_test_split(
        input_path, str(_eval_run_dir(checkpoint_file)), options,
    )
    source_files = [str(value).replace("\\", "/") for value in test.source_file]
    selected: set[int] = set()
    unmatched = []
    ambiguous = []
    for source in source_files:
        matches = [
            index for index, graph in enumerate(graphs)
            if source_matches_trace_id(str(graph.get("trace_id", "")), source)
        ]
        if not matches:
            unmatched.append(source)
        elif len(matches) > 1:
            ambiguous.append((source, matches))
        else:
            selected.add(matches[0])
    if unmatched or ambiguous:
        raise ValueError(
            "MAADBench official split could not be matched uniquely to converted graphs. "
            f"Unmatched={unmatched[:5]}, ambiguous={ambiguous[:3]}. "
            "Check --input, source model, domain, and trace type."
        )
    return selected


def _selected_dev_graph_indices(
    graphs: list[dict[str, Any]],
    input_path: str,
    checkpoint_file: str,
    options: dict[str, Any],
) -> set[int]:
    _official, _args, _train, dev, _output_dir = prepare_official_dev_split(
        input_path, str(_eval_run_dir(checkpoint_file)), options,
    )
    source_files = [str(value).replace("\\", "/") for value in dev.source_file]
    selected: set[int] = set()
    unmatched = []
    ambiguous = []
    for source in source_files:
        matches = [
            index for index, graph in enumerate(graphs)
            if source_matches_trace_id(str(graph.get("trace_id", "")), source)
        ]
        if not matches:
            unmatched.append(source)
        elif len(matches) > 1:
            ambiguous.append((source, matches))
        else:
            selected.add(matches[0])
    if unmatched or ambiguous:
        raise ValueError(
            "MAADBench official-train development split could not be matched uniquely to converted graphs. "
            f"Unmatched={unmatched[:5]}, ambiguous={ambiguous[:3]}. "
            "Check --input, source model, domain, and trace type."
        )
    return selected


def build_record(graph: dict[str, Any], diagnosis: dict[str, Any], graph_index: int,
                 settings: RuntimeSettings, scope: str,
                 graph_identity: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    record = dict(diagnosis)
    record.setdefault("dataset_family", "maadbench")
    record.setdefault("trace_id", graph.get("trace_id"))
    record.setdefault("graph_index", graph_index)
    record["diagnosis_scope"] = scope
    record["runtime_options"] = {
        "traversal_mode": resolve_traversal_mode(settings.traversal_mode, default="bfs"),
        "cascade_enabled": settings.cascade_enabled,
        "cascade_call_mode": settings.cascade_call_mode if settings.cascade_enabled else "disabled",
        "separate_order": (
            settings.separate_order
            if settings.cascade_enabled and settings.cascade_call_mode == "separate"
            else "not_applicable"
        ),
        "prompt_context_policy": settings.prompt_context_policy,
        "max_prompt_upstream_chars": settings.max_prompt_upstream_chars,
        "maadbench_dump_visible_context": bool((settings.dataset_options or {}).get("maadbench_dump_visible_context", False)),
        "maadbench_diagnosis_scope": scope,
        "maadbench_dev_seed": int((settings.dataset_options or {}).get("maadbench_dev_seed", int((settings.dataset_options or {}).get("maadbench_seed", 5)) + 101)),
        "maadbench_dev_size": float((settings.dataset_options or {}).get("maadbench_dev_size", 0.2)),
        "maadbench_dev_trace_limit": (settings.dataset_options or {}).get("maadbench_dev_trace_limit"),
        "maadbench_dev_pilot_seed": int((settings.dataset_options or {}).get("maadbench_dev_pilot_seed", 107)),
        **_api_identity(settings),
    }
    if graph_identity is not None:
        record["graph_identity"] = graph_identity
        record["runtime_options"]["graph_identity"] = graph_identity
    if str(record.get("cause_state_mode", "legacy")) == "path":
        local_status = str(record.get("local_prediction_status", "FAILED")).upper()
        cause_status = str(record.get("cause_path_coverage_status", "PARTIAL")).upper()
        record["status"] = (
            "SUCCESS" if local_status == "SUCCESS" and cause_status == "COMPLETE"
            else "PARTIAL" if local_status in {"SUCCESS", "PARTIAL"}
            else "FAILED"
        )
        record["cause_state_mode"] = "path"
        record["runtime_options"].update({
            "maadbench_cause_state_mode": "path",
            "maadbench_max_cause_path_states": int(
                (settings.dataset_options or {}).get("maadbench_max_cause_path_states", 4096)
            ),
            "cause_path_policy_version": _MAAD_PATH_METHOD_VERSION,
            "conflict_analysis_version": _MAAD_PATH_CONFLICT_ANALYSIS_VERSION,
            "method_version": _MAAD_PATH_METHOD_VERSION,
            "include_ground_truth": bool(settings.include_ground_truth),
            "include_eval_labels": bool(settings.include_eval_labels),
        })
        record["method_version"] = _MAAD_PATH_METHOD_VERSION
        record["cause_path_policy_version"] = _MAAD_PATH_METHOD_VERSION
        record["conflict_analysis_version"] = _MAAD_PATH_CONFLICT_ANALYSIS_VERSION
        uses_prevalidation = _uses_cascade_local_prevalidation(settings, "path")
        record["runtime_options"]["path_method_source_identity"] = _path_method_source_identity(
            cascade_local_prevalidation=uses_prevalidation,
        )
        if uses_prevalidation:
            record["cascade_local_evidence_validation_version"] = (
                _MAAD_CASCADE_LOCAL_EVIDENCE_VALIDATION_VERSION
            )
            record["runtime_options"]["cascade_local_evidence_validation_version"] = (
                _MAAD_CASCADE_LOCAL_EVIDENCE_VALIDATION_VERSION
            )
    else:
        record["status"] = "PARTIAL" if (record.get("coverage") or {}).get("partial") else "SUCCESS"
    return record


def run(
    *,
    graph_path: str,
    checkpoint_file: str,
    settings: RuntimeSettings,
    input_path: Optional[str] = None,
    end: Optional[int] = None,
) -> List[Dict[str, Any]]:
    graphs = load_graphs(graph_path)
    options = settings.dataset_options or {}
    cause_state_mode = str(options.get("maadbench_cause_state_mode", "legacy"))
    graph_identity = (
        _canonical_graph_identity(graphs, graph_path)
        if cause_state_mode == "path"
        else _edge_variant_graph_identity(graphs, graph_path)
    )
    _validate_checkpoint_config(checkpoint_file, settings, graph_identity=graph_identity)
    options = settings.dataset_options or {}
    scope = str(options.get("maadbench_diagnosis_scope", "all"))
    if scope not in {"all", "official_test", "official_dev"}:
        raise ValueError("MAADBench diagnosis scope must be 'all', 'official_test', or 'official_dev'")
    selected = None
    if scope in {"official_test", "official_dev"}:
        if not input_path:
            raise ValueError("Official-split diagnosis needs the raw MAADBench --input path")
        if scope == "official_test":
            selected = _selected_test_graph_indices(graphs, input_path, checkpoint_file, options)
        else:
            selected = _selected_dev_graph_indices(graphs, input_path, checkpoint_file, options)

    llm_server = LLMServer(
        api_key=settings.api_key,
        proxy=settings.proxy,
        url=settings.llm_url,
        model=settings.llm_model,
        max_retries=0,
        resolve_env=False,
    )
    options = settings.dataset_options or {}
    debug_dir = None
    if bool(options.get("maadbench_dump_visible_context", False)):
        checkpoint_path = Path(checkpoint_file).expanduser()
        debug_dir = str(checkpoint_path.parent / f"{checkpoint_path.stem}_visible_context")
    diagnose_one, _server = make_diagnoser(
        settings, llm_server=llm_server, debug_output_dir=debug_dir,
    )
    fingerprint_material = None
    if graph_identity is not None:
        fingerprint_material = {
            "graph_identity": graph_identity,
            "model": str(settings.llm_model),
            "diagnosis_scope": scope,
            "dev_seed": int(options.get("maadbench_dev_seed", int(options.get("maadbench_seed", 5)) + 101)),
            "dev_size": float(options.get("maadbench_dev_size", 0.2)),
            "dev_trace_limit": options.get("maadbench_dev_trace_limit"),
            "dev_pilot_seed": int(options.get("maadbench_dev_pilot_seed", 107)),
            "context": str(settings.prompt_context_policy),
            "traversal": resolve_traversal_mode(settings.traversal_mode, default="bfs"),
            "cascade_enabled": bool(settings.cascade_enabled),
            "cascade_call_mode": str(settings.cascade_call_mode if settings.cascade_enabled else "disabled"),
            "separate_order": (
                str(settings.separate_order)
                if settings.cascade_enabled and settings.cascade_call_mode == "separate"
                else "not_applicable"
            ),
            **_api_identity(settings),
        }
        if cause_state_mode == "path":
            fingerprint_material.update({
                "cause_state_mode": "path",
                "max_cause_path_states": int(options.get("maadbench_max_cause_path_states", 4096)),
                "cause_path_policy_version": _MAAD_PATH_METHOD_VERSION,
                "conflict_analysis_version": _MAAD_PATH_CONFLICT_ANALYSIS_VERSION,
                "method_version": _MAAD_PATH_METHOD_VERSION,
                "scheduler": "all_nodes_bfs_path_states_v1",
                "source_identity": _path_method_source_identity(),
                "max_causes": int(settings.max_causes),
                "max_prompt_upstream_chars": int(settings.max_prompt_upstream_chars),
                "include_ground_truth": bool(settings.include_ground_truth),
                "include_eval_labels": bool(settings.include_eval_labels),
            })
            if _uses_cascade_local_prevalidation(settings, cause_state_mode):
                fingerprint_material["cascade_local_evidence_validation_version"] = (
                    _MAAD_CASCADE_LOCAL_EVIDENCE_VALIDATION_VERSION
                )
                fingerprint_material["source_identity"] = _path_method_source_identity(
                    cascade_local_prevalidation=True,
                )
    run_fingerprint = (
        hashlib.sha256(json.dumps(
            fingerprint_material, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        if fingerprint_material is not None else None
    )
    return run_graph_diagnosis(
        graphs=graphs,
        checkpoint_file=checkpoint_file,
        diagnose_one=diagnose_one,
        llm_server=llm_server,
        build_record=lambda graph, diagnosis, index: build_record(
            graph, diagnosis, index, settings, scope, graph_identity=graph_identity,
        ),
        end=end,
        selected_graph_indices=selected,
        include_ground_truth=settings.include_ground_truth,
        include_eval_labels=settings.include_eval_labels,
        run_fingerprint=run_fingerprint,
        completion_predicate=_maad_checkpoint_complete,
        failed_record_metadata=(
            _failed_path_run_metadata if cause_state_mode == "path" else None
        ),
    )

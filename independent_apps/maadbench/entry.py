"""MAADBench-only CLI execution and official evaluation routing."""

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from independent_apps.shared.settings import RuntimeSettings
from .common.graph_utils import normalize_edge_types
from .common.llm import _safe_endpoint
from .evaluation.maadbench import evaluate_maadbench_files
from .registry import MAAD_EDGE_TYPES, OUTPUT_DIR, get_dataset, run_maadbench as run_maadbench_dataset


def _configure_settings(args: Any, settings: RuntimeSettings) -> RuntimeSettings:
    if settings.prompt_context_policy not in {"legacy", "ranked_budget"}:
        raise ValueError("prompt context policy must be 'legacy' or 'ranked_budget' for MAADBench.")
    if args.maadbench_max_cause_path_states < 1:
        raise ValueError("--maadbench-max-cause-path-states must be positive.")
    if args.maadbench_cause_state_mode == "path" and not settings.cascade_enabled:
        raise ValueError("--maadbench-cause-state-mode=path requires Cascade to be enabled.")
    if args.edge_types is not None:
        selected = normalize_edge_types(args.edge_types, supported=MAAD_EDGE_TYPES)
        settings.edge_types = [kind for kind in MAAD_EDGE_TYPES if kind in selected]
    settings.dataset_options = {
        "maadbench_diagnosis_scope": args.maadbench_diagnosis_scope,
        "maadbench_cause_state_mode": args.maadbench_cause_state_mode,
        "maadbench_max_cause_path_states": args.maadbench_max_cause_path_states,
        "maadbench_evaluation_profile": args.maadbench_evaluation_profile,
        "maadbench_dump_visible_context": args.maadbench_dump_visible_context,
        "maadbench_domain": args.maadbench_domain,
        "maadbench_trace_type": args.maadbench_trace_type,
        "maadbench_source_models": args.maadbench_source_model,
        "maadbench_source_temperatures": args.maadbench_source_temperature,
        "maadbench_seed": args.maadbench_seed,
        "maadbench_test_size": args.maadbench_test_size,
        "maadbench_dev_size": args.maadbench_dev_size,
        "maadbench_dev_seed": args.maadbench_dev_seed,
        "maadbench_dev_trace_limit": args.maadbench_dev_trace_limit,
        "maadbench_dev_pilot_seed": args.maadbench_dev_pilot_seed,
        "maadbench_unknown_status": args.maadbench_unknown_status,
        "maadbench_threshold_policy": args.maadbench_threshold_policy,
        "maadbench_anomaly_fraction": args.maadbench_anomaly_fraction,
        "maadbench_official_root": args.maadbench_official_root,
        "maadbench_evaluation_score_source": args.maadbench_evaluation_score_source,
    }
    return settings


def _resolve_dataset_file(path_value: Optional[str], filename: str) -> str:
    path = Path(path_value or str(OUTPUT_DIR))
    if path.is_dir() or path.suffix.lower() not in {".json", ".jsonl"}:
        path = path / filename
    return str(path)


def _resolve_evaluation_output(output: Optional[str], dataset: str) -> str:
    path = Path(output or str(OUTPUT_DIR))
    if path.is_dir() or path.suffix.lower() != ".json":
        path = path / f"{dataset}_evaluation.json"
    return str(path)


def _checkpoint_filename(settings: RuntimeSettings) -> str:
    safe_model = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in (settings.llm_model or "model")
    )
    traversal = "bfs" if settings.traversal_mode == "auto" else settings.traversal_mode
    if not settings.cascade_enabled:
        cascade = "off"
    elif settings.cascade_call_mode == "separate":
        cascade = f"separate-{settings.separate_order}"
    else:
        cascade = settings.cascade_call_mode
    edge_types = settings.edge_types or list(MAAD_EDGE_TYPES)
    return (
        f"maadbench_{safe_model}_{traversal}_cascade-{cascade}_"
        f"{settings.prompt_context_policy}_{'_'.join(edge_types)}_results.json"
    )


def _evaluation_params(args: Any, settings: RuntimeSettings, paths: Dict[str, str]) -> Dict[str, Any]:
    runtime_settings = asdict(settings)
    runtime_settings["separate_order"] = (
        settings.separate_order
        if settings.cascade_enabled and settings.cascade_call_mode == "separate"
        else "not_applicable"
    )
    runtime_settings["api_key_configured"] = bool(runtime_settings.pop("api_key"))
    for key in ("llm_url", "proxy"):
        if runtime_settings.get(key):
            runtime_settings[key] = _safe_endpoint(runtime_settings[key])
    if runtime_settings["edge_types"] is None:
        runtime_settings["edge_types"] = list(MAAD_EDGE_TYPES)
    options = settings.dataset_options
    return {
        "dataset": "maadbench",
        "family": "maadbench",
        "runtime_settings": runtime_settings,
        **paths,
        "run_options": {
            "end": args.end,
            "skip_convert": args.skip_convert,
            "convert_only": args.convert_only,
            "evaluate_only": args.evaluate_only,
            "maadbench_diagnosis_scope": options["maadbench_diagnosis_scope"],
            "maadbench_cause_state_mode": options["maadbench_cause_state_mode"],
            "maadbench_evaluation_profile": options["maadbench_evaluation_profile"],
            "maadbench_evaluation_score_source": options["maadbench_evaluation_score_source"],
        },
    }


def _evaluate(prediction_file: str, reference_path: str, output_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    print(f"Evaluation prediction/checkpoint file: {prediction_file}")
    print(f"Evaluation report path: {output_path}")
    metrics = evaluate_maadbench_files(
        prediction_file,
        reference_path,
        output_path=output_path,
        params=params,
        options=params.get("runtime_settings", {}).get("dataset_options", {}),
    )
    print("Evaluation metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Evaluation report saved to: {output_path}")
    return metrics


def run_maadbench(dataset: str, settings: RuntimeSettings, args: Any) -> Any:
    if dataset != "maadbench":
        raise ValueError(f"MAADBench entry cannot run dataset {dataset!r}.")
    if not args.evaluate_only and not args.convert_only:
        os.environ.setdefault("TRAIL_API_REQUEST_TYPE", "maadbench_diagnosis")
    settings = _configure_settings(args, settings)
    spec = get_dataset(dataset)
    resolved_input = args.input or spec.input_path
    resolved_graph = _resolve_dataset_file(args.graph_path, f"{dataset}_graphs.jsonl")
    if args.skip_convert and args.edge_types is not None and args.graph_path is None:
        raise ValueError("--skip-convert with --edge-types requires an explicit --graph-path for metadata verification.")
    is_subset = settings.edge_types is not None and set(settings.edge_types) != set(MAAD_EDGE_TYPES)
    if is_subset and not args.skip_convert and args.graph_path is None:
        raise ValueError("A non-full edge variant requires an explicit --graph-path; the default graph is protected.")
    if is_subset and not args.convert_only and args.checkpoint_file is None:
        raise ValueError("A non-full edge variant requires an explicit --checkpoint-file.")
    if is_subset and args.evaluate and args.evaluation_output is None:
        raise ValueError("A non-full edge variant requires an explicit --evaluation-output.")
    if args.evaluate_only and args.edge_types is not None:
        raise ValueError("--edge-types is a conversion option and cannot be used with --evaluate-only.")

    resolved_checkpoint = _resolve_dataset_file(args.checkpoint_file, _checkpoint_filename(settings))
    should_evaluate = args.evaluate or args.evaluate_only
    evaluation_output = _resolve_evaluation_output(args.evaluation_output, dataset) if should_evaluate else None

    result = []
    if not args.evaluate_only:
        result = run_maadbench_dataset(
            dataset=dataset,
            settings=settings,
            input_path=resolved_input,
            graph_path=resolved_graph,
            checkpoint_file=resolved_checkpoint,
            end=args.end,
            skip_convert=args.skip_convert,
            convert_only=args.convert_only,
            edge_types=settings.edge_types,
        )
    if should_evaluate:
        params = _evaluation_params(
            args,
            settings,
            {
                "graph_file": resolved_graph,
                "prediction_file": resolved_checkpoint,
                "evaluation_output_path": evaluation_output,
                "input_path": resolved_input,
            },
        )
        _evaluate(resolved_checkpoint, resolved_input, evaluation_output, params)
    return result

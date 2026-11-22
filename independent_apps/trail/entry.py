"""TRAIL-only CLI execution and official evaluation routing."""

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from independent_apps.shared.settings import RuntimeSettings
from .common.graph_utils import normalize_edge_types
from .common.llm import _safe_endpoint
from .evaluation.trail import evaluate_trail_files
from .registry import OUTPUT_DIR, TRAIL_EDGE_TYPES, get_dataset, run_trail as run_trail_dataset


def _configure_settings(args: Any, settings: RuntimeSettings) -> RuntimeSettings:
    allowed_context = {"legacy", "ranked_budget", "prioritized_two_kinds"}
    if settings.prompt_context_policy not in allowed_context:
        raise ValueError("prompt context policy must be 'legacy', 'ranked_budget', or 'prioritized_two_kinds'.")
    settings.trail_schedule_scope = args.trail_schedule_scope or settings.trail_schedule_scope
    settings.trail_final_hypothesis_policy = args.trail_final_hypothesis_policy or settings.trail_final_hypothesis_policy
    settings.trail_cause_state_mode = args.trail_cause_state_mode or "H0"
    settings.trail_request_token_budget_enabled = bool(args.trail_request_token_budget)
    if args.edge_types is not None:
        selected = normalize_edge_types(args.edge_types, supported=TRAIL_EDGE_TYPES)
        settings.edge_types = [kind for kind in TRAIL_EDGE_TYPES if kind in selected]
    for name in (
        "trail_request_context_limit",
        "trail_request_hard_projected_limit",
        "trail_request_completion_reserve",
    ):
        value = getattr(args, name)
        if value is not None:
            if int(value) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive.")
            setattr(settings, name, int(value))
    if args.trail_request_safety_factor is not None:
        if args.trail_request_safety_factor < 1:
            raise ValueError("--trail-request-safety-factor must be at least 1.")
        settings.trail_request_safety_factor = float(args.trail_request_safety_factor)
    if args.trail_request_tokenizer is not None:
        if not args.trail_request_tokenizer.strip():
            raise ValueError("--trail-request-tokenizer must not be empty.")
        settings.trail_request_tokenizer = args.trail_request_tokenizer.strip()
    if args.trail_max_completion_tokens is not None:
        if args.trail_max_completion_tokens <= 0:
            raise ValueError("--trail-max-completion-tokens must be positive.")
        settings.trail_max_completion_tokens = args.trail_max_completion_tokens
    if args.trail_api_request_limit is not None:
        if args.trail_api_request_limit <= 0:
            raise ValueError("--trail-api-request-limit must be positive.")
        settings.trail_api_request_limit = args.trail_api_request_limit
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


def _evaluation_params(args: Any, settings: RuntimeSettings, dataset: str, paths: Dict[str, str]) -> Dict[str, Any]:
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
        runtime_settings["edge_types"] = list(TRAIL_EDGE_TYPES)
    return {
        "dataset": dataset,
        "family": "trail",
        "runtime_settings": runtime_settings,
        **paths,
        "run_options": {
            "end": args.end,
            "skip_convert": args.skip_convert,
            "convert_only": args.convert_only,
            "evaluate_only": args.evaluate_only,
            "maadbench_diagnosis_scope": None,
            "maadbench_cause_state_mode": None,
            "maadbench_evaluation_profile": None,
            "maadbench_evaluation_score_source": None,
        },
    }


def _evaluate(prediction_file: str, graph_file: str, output_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    print(f"Evaluation prediction/checkpoint file: {prediction_file}")
    print(f"Evaluation report path: {output_path}")
    metrics = evaluate_trail_files(prediction_file, graph_file, output_path=output_path, params=params)
    print("Evaluation metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Evaluation report saved to: {output_path}")
    return metrics


def run_trail(dataset: str, settings: RuntimeSettings, args: Any) -> Any:
    settings = _configure_settings(args, settings)
    spec = get_dataset(dataset)
    resolved_input = args.input or spec.input_path
    resolved_graph = _resolve_dataset_file(args.graph_path, f"{dataset}_graphs.jsonl")
    if args.skip_convert and args.edge_types is not None and args.graph_path is None:
        raise ValueError("--skip-convert with --edge-types requires an explicit --graph-path for metadata verification.")
    is_subset = settings.edge_types is not None and set(settings.edge_types) != set(TRAIL_EDGE_TYPES)
    if is_subset and not args.skip_convert and args.graph_path is None:
        raise ValueError("A non-full edge variant requires an explicit --graph-path; the default graph is protected.")
    if is_subset and not args.convert_only and args.checkpoint_file is None:
        raise ValueError("A non-full edge variant requires an explicit --checkpoint-file.")
    if is_subset and args.evaluate and args.evaluation_output is None:
        raise ValueError("A non-full edge variant requires an explicit --evaluation-output.")
    if args.evaluate_only and args.edge_types is not None:
        raise ValueError("--edge-types is a conversion option and cannot be used with --evaluate-only.")

    if settings.cascade_enabled and settings.cascade_call_mode == "separate":
        default_checkpoint = f"{dataset}_results_separate-{settings.separate_order}.json"
    else:
        default_checkpoint = f"{dataset}_results.json"
    resolved_checkpoint = _resolve_dataset_file(args.checkpoint_file, default_checkpoint)
    should_evaluate = args.evaluate or args.evaluate_only
    evaluation_output = _resolve_evaluation_output(args.evaluation_output, dataset) if should_evaluate else None

    result = []
    if not args.evaluate_only:
        result = run_trail_dataset(
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
            dataset,
            {
                "graph_file": resolved_graph,
                "prediction_file": resolved_checkpoint,
                "evaluation_output_path": evaluation_output,
                "input_path": resolved_input,
            },
        )
        _evaluate(resolved_checkpoint, resolved_graph, evaluation_output, params)
    return result

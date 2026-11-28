"""Who&When command execution: current converter + Original/Combined diagnosis."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, Optional

from independent_apps.shared.settings import RuntimeSettings
from .common.llm import _safe_endpoint
from .common.graph_utils import normalize_edge_types
from .datasets.who_when.graph_converter import EDGE_TYPES
from .evaluation.who_when import evaluate_who_when_files
from .registry import OUTPUT_DIR, get_dataset, run_who_when as run_who_when_dataset


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


def _evaluation_params(
    args: Any,
    settings: RuntimeSettings,
    dataset: str,
    paths: Dict[str, str],
) -> Dict[str, Any]:
    runtime_settings = asdict(settings)
    runtime_settings["separate_order"] = (
        settings.separate_order if settings.cascade_enabled and settings.cascade_call_mode == "separate"
        else "not_applicable"
    )
    runtime_settings["api_key_configured"] = bool(runtime_settings.pop("api_key"))
    for key in ("llm_url", "proxy"):
        if runtime_settings.get(key):
            runtime_settings[key] = _safe_endpoint(runtime_settings[key])
    if runtime_settings["edge_types"] is None:
        runtime_settings["edge_types"] = ["sequential_flow", "execution_flow", "data_flow"]
    diagnosis_mode = settings.cascade_call_mode
    return {
        "dataset": dataset,
        "family": "who_when",
        "who_when_implementation": (
            "Original ZIP ReverseDependencyDFS"
            if diagnosis_mode == "separate"
            else "structured Combined ReverseDependencyDFS"
        ),
        "who_when_diagnosis_mode": diagnosis_mode,
        "who_when_graph_converter": "current",
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


def _evaluate(prediction_file: str, reference_path: str, output_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    print(f"Evaluation prediction/checkpoint file: {prediction_file}")
    print(f"Evaluation report path: {output_path}")
    metrics = evaluate_who_when_files(prediction_file, reference_path, output_path=output_path, params=params)
    print("Evaluation metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Evaluation report saved to: {output_path}")
    return metrics



def run_who_when(dataset: str, settings: RuntimeSettings, args: Any) -> Any:
    """Run Original ZIP or structured Combined Who&When diagnosis."""

    spec = get_dataset(dataset)
    resolved_input = args.input or spec.input_path
    should_evaluate = args.evaluate or args.evaluate_only

    # Who&When uses the shared cascade-call-mode as its only strategy selector:
    # separate -> Original ZIP diagnosis; combined -> structured Combined.
    # Preserve the historical Original default when the CLI option is omitted.
    if getattr(args, "cascade_call_mode", None) is None:
        settings.cascade_call_mode = "separate"
    diagnosis_mode = settings.cascade_call_mode
    if diagnosis_mode not in {"separate", "combined"}:
        raise ValueError(
            "Who&When cascade_call_mode must be 'separate' or 'combined'."
        )

    # The common default is Cascade-first for other projects. Preserve Who&When
    # Local-first when callers omit the option (including the existing batch script).
    settings.separate_order = getattr(args, "separate_order", None) or "local_then_cascade"
    if settings.edge_types is not None:
        selected = normalize_edge_types(settings.edge_types, supported=EDGE_TYPES)
        if len(settings.edge_types) != len(selected):
            raise ValueError("Who&When edge types must not contain duplicates.")
        settings.edge_types = [kind for kind in EDGE_TYPES if kind in selected]

    requested_max_causes = getattr(args, "max_causes", None)
    configured_max_causes = (
        requested_max_causes
        if requested_max_causes is not None
        else getattr(settings, "max_causes", 8)
    )

    if (
        isinstance(configured_max_causes, bool)
        or not isinstance(configured_max_causes, int)
        or configured_max_causes < 1
    ):
        raise ValueError(
            "Who&When max_causes must be a positive integer."
        )

    settings.max_causes = configured_max_causes

    if settings.prompt_context_policy not in {
        "legacy", "ranked_budget"
    }:
        raise ValueError(
            "Who&When prompt context policy must be "
            "'legacy' or 'ranked_budget'."
        )

    # The current Graph Converter is fixed. Both diagnosis modes share one Graph;
    # Checkpoints and Evaluation reports remain isolated by call mode.
    edge_suffix = ("_" + "+".join(sorted(settings.edge_types))) if settings.edge_types is not None and set(settings.edge_types) != set(EDGE_TYPES) else ""
    order_suffix = ("_cascade_then_local" if settings.cascade_enabled and diagnosis_mode == "separate" and settings.separate_order == "cascade_then_local" else "")
    run_tag = f"{diagnosis_mode}{order_suffix}_current_graph{edge_suffix}"
    resolved_graph = _resolve_dataset_file(
        args.graph_path,
        f"{dataset}_current{edge_suffix}_graphs.jsonl",
    )

    resolved_checkpoint = _resolve_dataset_file(
        args.checkpoint_file,
        f"{dataset}_{run_tag}_results.json",
    )

    evaluation_output = (
        _resolve_evaluation_output(
            args.evaluation_output, f"{dataset}_{run_tag}"
        )
        if should_evaluate
        else None
    )

    diagnosis_labels = {
        "separate": "Original ZIP ReverseDependencyDFS baseline",
        "combined": "structured Combined ReverseDependencyDFS",
    }
    print(f"[WHO&WHEN DIAGNOSIS] {diagnosis_labels[diagnosis_mode]}")
    print(f"[WHO&WHEN CASCADE CALL MODE] {diagnosis_mode}")
    if settings.cascade_enabled and diagnosis_mode == "separate":
        print(f"[WHO&WHEN SEPARATE ORDER] {settings.separate_order}")
    print("[WHO&WHEN EDGE TYPES]", settings.edge_types or list(EDGE_TYPES))
    print("[WHO&WHEN GRAPH CONVERTER] current (fixed)")
    print(
        "[WHO&WHEN GROUND TRUTH]",
        "visible" if settings.include_ground_truth else "hidden"
    )

    result = []

    if not args.evaluate_only:
        result = run_who_when_dataset(
            dataset=dataset,
            settings=settings,
            input_path=resolved_input,
            graph_path=resolved_graph,
            checkpoint_file=resolved_checkpoint,
            end=args.end,
            skip_convert=args.skip_convert,
            convert_only=args.convert_only,
        )

    if should_evaluate:
        paths = {
            "graph_file": resolved_graph,
            "prediction_file": resolved_checkpoint,
            "evaluation_output_path": evaluation_output,
            "reference_data_path": resolved_input,
        }

        params = _evaluation_params(
            args, settings, dataset, paths
        )

        _evaluate(
            resolved_checkpoint,
            resolved_input,
            evaluation_output,
            params,
        )

    return result

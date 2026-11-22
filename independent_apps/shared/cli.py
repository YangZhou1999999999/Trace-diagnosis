"""One thin CLI for selecting and routing to an owning dataset project."""

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from .settings import load_api_config, settings_from_args


DATASET_ROUTES: Dict[str, Tuple[str, str]] = {
    "who_when_algorithm": ("independent_apps.who_when", "run_who_when"),
    "who_when_handcrafted": ("independent_apps.who_when", "run_who_when"),
    "trail_gaia": ("independent_apps.trail", "run_trail"),
    "trail_swe_bench": ("independent_apps.trail", "run_trail"),
    "maadbench": ("independent_apps.maadbench", "run_maadbench"),
}
DATASET_IDS = tuple(DATASET_ROUTES)
PROJECT_DATASETS: Dict[str, Tuple[str, ...]] = {
    "who_when": ("who_when_algorithm", "who_when_handcrafted"),
    "trail": ("trail_gaia", "trail_swe_bench"),
    "maadbench": ("maadbench",),
}
DEFAULT_API_CONFIG = Path(__file__).resolve().parents[4] / "api_config.json"


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default=None, help="Override the selected dataset's input path.")
    parser.add_argument("--graph-path", default=None, help="Graph file or output directory.")
    parser.add_argument("--checkpoint-file", default=None, help="Checkpoint file or output directory.")
    parser.add_argument("--end", type=int, default=None, help="Limit the number of records diagnosed.")
    parser.add_argument("--skip-convert", action="store_true", help="Use an existing Graph file.")
    parser.add_argument("--convert-only", action="store_true", help="Only convert input data to Graphs.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate after diagnosis.")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate an existing prediction file only.")
    parser.add_argument("--evaluation-output", "--eval-output", default=None, help="Evaluation report path or directory.")
    parser.add_argument("--llm-model", default=None, help="Model identifier; defaults to the API config model or project default.")
    parser.add_argument("--api-config", default=str(DEFAULT_API_CONFIG), help="External JSON API configuration path.")
    parser.add_argument("--proxy", default=None, help="HTTP proxy override.")
    parser.add_argument("--no-cascade", action="store_true", help="Disable Cascade diagnosis.")
    parser.add_argument("--cascade-call-mode", choices=["combined", "separate"], default=None)
    parser.add_argument("--no-ground-truth", action="store_true", help="Exclude labels from model-visible data.")
    parser.add_argument("--include-eval-labels", action="store_true")
    parser.add_argument("--max-causes", type=int, default=8)
    parser.add_argument("--traversal-mode", choices=["auto", "bfs", "dfs"], default=None)
    parser.add_argument("--max-prompt-upstream-chars", type=int, default=None)


def _register_project_arguments(parser: argparse.ArgumentParser, dataset: str) -> None:
    if dataset not in DATASET_ROUTES:
        return
    project, _ = DATASET_ROUTES[dataset]
    options_module = importlib.import_module(f"{project}.cli_options")
    options_module.add_cli_arguments(parser, dataset)


def build_parser(
    dataset: Optional[str] = None,
    *,
    required_dataset: bool = True,
    dataset_choices: Sequence[str] = DATASET_IDS,
    default_dataset: Optional[str] = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Graph diagnosis launcher. Dataset behavior is implemented by the selected project."
    )
    parser.add_argument(
        "--dataset",
        required=required_dataset,
        choices=dataset_choices,
        default=default_dataset,
        help="Dataset implementation to run.",
    )
    _add_common_arguments(parser)
    _register_project_arguments(parser, dataset or "")
    return parser


def build_parser_for_dataset(dataset: str, *, default_dataset: Optional[str] = None) -> argparse.ArgumentParser:
    """Build the common parser plus only the selected project's options."""
    project_package = DATASET_ROUTES.get(dataset, (None, None))[0]
    project = project_package.rsplit(".", 1)[-1] if project_package else None
    choices = PROJECT_DATASETS.get(project, DATASET_IDS)
    return build_parser(
        dataset,
        required_dataset=default_dataset is None,
        dataset_choices=choices,
        default_dataset=default_dataset,
    )


def _dataset_from_argv(argv: Optional[Sequence[str]]) -> Optional[str]:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--dataset")
    parsed, _ = probe.parse_known_args(argv)
    return parsed.dataset


def route_dataset(dataset: str, settings: Any, args: argparse.Namespace) -> Any:
    """Dispatch only; no converter, diagnoser, runner, checkpoint, or evaluator calls here."""
    if dataset not in DATASET_ROUTES:
        valid = ", ".join(DATASET_IDS)
        raise ValueError(f"Unknown dataset {dataset!r}. Valid datasets: {valid}")
    module_name, function_name = DATASET_ROUTES[dataset]
    entry_module = importlib.import_module(f"{module_name}.entry")
    return getattr(entry_module, function_name)(dataset, settings, args)


def main(argv: Optional[Sequence[str]] = None) -> Any:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    dataset = _dataset_from_argv(effective_argv)
    parser = build_parser(dataset, required_dataset=True, dataset_choices=DATASET_IDS)
    args = parser.parse_args(effective_argv)

    if args.convert_only and (args.evaluate or args.evaluate_only):
        parser.error("--convert-only cannot be combined with --evaluate or --evaluate-only.")

    needs_api_config = not args.evaluate_only and not args.convert_only
    if needs_api_config:
        os.environ["TRAIL_API_CONFIG_PATH"] = str(Path(args.api_config).expanduser().resolve())
    try:
        api_config = load_api_config(args.api_config) if needs_api_config else None
        settings = settings_from_args(args, api_config)
    except ValueError as exc:
        parser.error(str(exc))

    return route_dataset(args.dataset, settings, args)

"""Who&When-only dataset selection and execution."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

from independent_apps.shared.settings import RuntimeSettings
from .datasets.who_when import runner as who_when_runner


BASE_DIR = Path(os.environ.get("TRACE_DIAGNOSIS_BASE_DIR") or Path(__file__).resolve().parents[2]).expanduser()
PROJECT_OUTPUT_DIR = Path(os.environ.get("WHO_WHEN_OUTPUT_DIR") or BASE_DIR / "outputs" / "independent" / "who_when").expanduser()
OUTPUT_DIR = PROJECT_OUTPUT_DIR


def _output_path(*parts: str) -> str:
    return str(PROJECT_OUTPUT_DIR.joinpath(*parts))


def _base_path(*parts: str) -> str:
    return str(BASE_DIR.joinpath(*parts))


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_path: str
    graph_path: str
    checkpoint_file: str
    convert: Callable[..., List[dict]]
    run: Callable[..., List[dict]]


DATASETS: Dict[str, DatasetSpec] = {
    "who_when_algorithm": DatasetSpec(
        name="who_when_algorithm",
        input_path=_base_path("Who&When", "Algorithm-Generated"),
        graph_path=_output_path("who_when_algorithm_graphs.jsonl"),
        checkpoint_file=_output_path("who_when_algorithm_results.json"),
        convert=who_when_runner.convert_to_graphs,
        run=who_when_runner.run,
    ),
    "who_when_handcrafted": DatasetSpec(
        name="who_when_handcrafted",
        input_path=_base_path("Who&When", "Hand-Crafted"),
        graph_path=_output_path("who_when_handcrafted_graphs.jsonl"),
        checkpoint_file=_output_path("who_when_handcrafted_results.json"),
        convert=who_when_runner.convert_to_graphs,
        run=who_when_runner.run,
    ),
}


def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        valid = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown Who&When dataset {name!r}. Valid datasets: {valid}")
    return DATASETS[name]


def run_who_when(
    *,
    dataset: str,
    settings: RuntimeSettings,
    input_path: Optional[str] = None,
    graph_path: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
    end: Optional[int] = None,
    skip_convert: bool = False,
    convert_only: bool = False,
) -> List[dict]:
    """Convert and run only one of the two Who&When subsets."""
    spec = get_dataset(dataset)
    resolved_input = input_path or spec.input_path
    resolved_graph = graph_path or spec.graph_path
    resolved_checkpoint = checkpoint_file or spec.checkpoint_file
    if not skip_convert:
        print(f"Converting dataset {dataset} to graphs...")
        graphs = spec.convert(
            resolved_input, resolved_graph, end, settings.edge_types,
        )
        print(f"Graph conversion complete: {len(graphs)} records -> {resolved_graph}")
    if convert_only:
        return []
    return spec.run(
        graph_path=resolved_graph,
        checkpoint_file=resolved_checkpoint,
        settings=settings,
        end=end,
    )

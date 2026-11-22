"""MAADBench-only dataset selection, graph verification, and execution."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from independent_apps.shared.settings import RuntimeSettings
from .common.graph_utils import normalize_edge_types
from .common.io import load_graphs
from .datasets.maadbench import runner as maadbench_runner


MAAD_EDGE_TYPES = ("sequential_flow", "execution_flow", "data_flow")
BASE_DIR = Path(os.environ.get("TRACE_DIAGNOSIS_BASE_DIR") or Path(__file__).resolve().parents[2]).expanduser()
MAADBENCH_DATA_DIR = Path(os.environ.get("MAADBENCH_DATA_DIR") or BASE_DIR.joinpath("Data")).expanduser()
PROJECT_OUTPUT_DIR = Path(os.environ.get("MAADBENCH_OUTPUT_DIR") or BASE_DIR / "outputs" / "independent" / "maadbench").expanduser()
OUTPUT_DIR = PROJECT_OUTPUT_DIR


def _output_path(*parts: str) -> str:
    return str(PROJECT_OUTPUT_DIR.joinpath(*parts))


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_path: str
    graph_path: str
    checkpoint_file: str
    convert: Callable[..., List[dict]]
    run: Callable[..., List[dict]]


DATASETS: Dict[str, DatasetSpec] = {
    "maadbench": DatasetSpec(
        name="maadbench",
        input_path=str(MAADBENCH_DATA_DIR / "gpt-4.1"),
        graph_path=_output_path("maadbench_graphs.jsonl"),
        checkpoint_file=_output_path("maadbench_results.json"),
        convert=maadbench_runner.convert_to_graphs,
        run=maadbench_runner.run,
    ),
}


def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        valid = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown MAADBench dataset {name!r}. Valid datasets: {valid}")
    return DATASETS[name]


def run_maadbench(
    *, dataset: str, settings: RuntimeSettings, input_path: Optional[str] = None,
    graph_path: Optional[str] = None, checkpoint_file: Optional[str] = None,
    end: Optional[int] = None, skip_convert: bool = False,
    convert_only: bool = False, edge_types: Optional[Sequence[str]] = None,
) -> List[dict]:
    spec = get_dataset(dataset)
    resolved_input = input_path or spec.input_path
    resolved_graph = graph_path or spec.graph_path
    resolved_checkpoint = checkpoint_file or spec.checkpoint_file
    requested_types = edge_types if edge_types is not None else settings.edge_types
    normalized_types = None
    if requested_types is not None:
        selected = normalize_edge_types(requested_types, supported=MAAD_EDGE_TYPES)
        normalized_types = [kind for kind in MAAD_EDGE_TYPES if kind in selected]
    is_subset = normalized_types is not None and set(normalized_types) != set(MAAD_EDGE_TYPES)

    if is_subset and not skip_convert and graph_path is None:
        raise ValueError("A non-full edge variant requires an explicit, separate --graph-path.")
    if is_subset and not convert_only and checkpoint_file is None:
        raise ValueError("A non-full edge variant requires an explicit, separate --checkpoint-file.")

    if skip_convert and requested_types is not None:
        if not os.path.isfile(resolved_graph):
            raise FileNotFoundError(f"Cannot verify --edge-types because graph file does not exist: {resolved_graph}")
        graphs = load_graphs(resolved_graph)
        if not graphs:
            raise ValueError("Cannot verify --edge-types against an empty graph file.")
        expected = set(normalized_types or ())
        for index, graph in enumerate(graphs):
            metadata = graph.get("graph_metadata") or {}
            requested = metadata.get("requested_edge_types")
            if requested is None:
                raise ValueError(f"Graph {index} lacks requested-edge metadata; refusing to reuse it with --skip-convert.")
            if set(requested) != expected:
                raise ValueError(f"Graph {index} was built for edge types {requested}, not {normalized_types}.")
            actual = {edge.get("kind") for edge in graph.get("edges") or [] if isinstance(edge, dict)}
            if actual - expected:
                raise ValueError(f"Graph {index} contains unselected edge kinds: {sorted(actual - expected)}.")

    if is_subset and not skip_convert and os.path.abspath(resolved_graph) == os.path.abspath(spec.graph_path):
        raise ValueError("Refusing to write a non-full edge variant over the dataset's default graph path.")
    if not skip_convert:
        print(f"Converting all dataset {dataset} to graphs before applying run scope...")
        # Split matching uses full source indices. `end` limits diagnosis, not conversion.
        graphs = spec.convert(resolved_input, resolved_graph, None, normalized_types)
        print(f"Graph conversion complete: {len(graphs)} records -> {resolved_graph}")
    if convert_only:
        return []
    return spec.run(
        graph_path=resolved_graph, checkpoint_file=resolved_checkpoint,
        settings=settings, end=end, input_path=resolved_input,
    )

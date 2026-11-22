"""Build reproducible materialized edge-subgraph variants for TRAIL and MAADBench.

This module only converts raw traces with the production converters, then
filters the resulting edge lists. It never runs diagnosis or contacts a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from ..common.graph_utils import normalize_edge_types
from ..common.io import write_jsonl
from ..datasets.trail import graph_converter as trail_converter


DATASETS = {
    "trail_gaia": ("trail", trail_converter.DEFAULT_EDGE_TYPES),
    "trail_swe_bench": ("trail", trail_converter.DEFAULT_EDGE_TYPES),
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(path: Path, family: str) -> tuple[str, int, str]:
    if path.is_file():
        return path.name, 1, _sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if family == "maadbench":
        candidates = sorted(p for p in path.rglob("*.json") if "EscapeRoom" in p.name)
    else:
        candidates = [path]
    digest = hashlib.sha256()
    count = 0
    for source in candidates:
        relative = source.relative_to(path).as_posix() if source != path else source.name
        file_hash = _sha256_file(source)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        count += 1
    if count == 0:
        raise FileNotFoundError(f"No source files found under: {path}")
    return path.name, count, digest.hexdigest()


def _canonical_graph_hash(graphs: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for graph in graphs:
        payload = json.dumps(
            graph, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")
        digest.update(payload)
        digest.update(b"\n")
    return digest.hexdigest()


def _load_full_graphs(dataset: str, input_path: Path) -> list[dict[str, Any]]:
    if input_path.suffix.lower() in {".parquet", ".pq"}:
        graphs = trail_converter.convert_parquet(str(input_path), output_path=None, max_rows=None)
    else:
        graphs = trail_converter.convert_json(str(input_path), output_path=None, max_rows=None)
    if not graphs:
        raise ValueError("The production converter returned no graphs.")
    return graphs


def _variant_name(edge_types: Iterable[str]) -> str:
    return "+".join(edge_types)


def _trail_final_anchor_node_id(graph: dict[str, Any]) -> str | None:
    """Use the converter's shared full-graph target rule."""
    return trail_converter._trail_final_anchor_node_id(graph)


def _filtered_graph(
    graph: dict[str, Any],
    *,
    selected: set[str],
    ordered_selected: list[str],
    supported: tuple[str, ...],
    family: str,
) -> dict[str, Any]:
    original_edges = graph.get("edges") or []
    if any(not isinstance(edge, dict) or not edge.get("kind") for edge in original_edges):
        raise ValueError(f"Graph {graph.get('trace_id')!r} contains a malformed edge.")
    unexpected = {
        edge.get("kind") for edge in original_edges
        if isinstance(edge, dict) and edge.get("kind") not in supported
    }
    if unexpected:
        raise ValueError(f"Full graph contains unsupported edge kinds: {sorted(unexpected)}")
    filtered = dict(graph)
    filtered["edges"] = [
        edge for edge in original_edges
        if isinstance(edge, dict) and edge.get("kind") in selected
    ]
    metadata = dict(graph.get("graph_metadata") or {})
    full_selection = selected == set(supported)
    if family == "trail" and full_selection:
        metadata.setdefault("graph_variant", "G0")
        metadata["edge_types"] = sorted(supported)
    else:
        metadata["graph_variant"] = "edge_subset_v1" if not full_selection else "full_edge_set_v1"
        metadata["edge_types"] = list(ordered_selected)
    metadata["requested_edge_types"] = list(ordered_selected)
    if family == "trail":
        final_anchor = _trail_final_anchor_node_id(graph)
        if final_anchor is not None:
            metadata["final_anchor_node_id"] = final_anchor
    filtered["graph_metadata"] = metadata
    return filtered


def generate_edge_variants(dataset: str, input_path: str, output_dir: str) -> dict[str, Any]:
    """Materialize all seven non-empty edge subsets and a deterministic manifest."""
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; choose from {sorted(DATASETS)}")
    family, supported = DATASETS[dataset]
    source = Path(input_path).expanduser().resolve()
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite an existing edge-variant output directory: {target}")

    full_graphs = _load_full_graphs(dataset, source)
    source_identifier, source_file_count, source_sha256 = _source_identity(source, family)
    trace_ids = [str(graph.get("trace_id", "")) for graph in full_graphs]
    if any(not trace_id for trace_id in trace_ids):
        raise ValueError("Every converted graph must have a non-empty trace_id.")
    canonical_full_sha256 = _canonical_graph_hash(full_graphs)

    variants = [
        combo
        for size in range(1, len(supported) + 1)
        for combo in combinations(supported, size)
    ]

    target.mkdir(parents=True, exist_ok=False)
    manifest_variants: list[dict[str, Any]] = []
    graph_source_sha = _sha256_file(
        Path(trail_converter.__file__)
    )
    for variant in variants:
        variant_name = _variant_name(variant)
        output_path = target / f"edges-{variant_name}.jsonl"
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite variant graph: {output_path}")
        selected = set(variant)
        variant_graphs = [
            _filtered_graph(
                graph,
                selected=selected,
                ordered_selected=list(variant),
                supported=tuple(supported),
                family=family,
            )
            for graph in full_graphs
        ]
        for original, variant_graph in zip(full_graphs, variant_graphs):
            if variant_graph.get("trace_id") != original.get("trace_id"):
                raise AssertionError("Edge filtering changed Trace order or identity.")
            if variant_graph.get("nodes") != original.get("nodes"):
                raise AssertionError("Edge filtering changed nodes or node order.")
            expected_edges = [edge for edge in (original.get("edges") or []) if edge.get("kind") in selected]
            if variant_graph["edges"] != expected_edges:
                raise AssertionError("Edge filtering changed edge order or evidence.")
        write_jsonl(str(output_path), variant_graphs)

        per_graph_counts = []
        totals = {kind: 0 for kind in supported}
        for graph in variant_graphs:
            counts = {kind: 0 for kind in supported}
            for edge in graph.get("edges") or []:
                counts[edge["kind"]] += 1
            for kind, count in counts.items():
                totals[kind] += count
            per_graph_counts.append({"trace_id": graph["trace_id"], "edge_counts": counts})
        manifest_variants.append({
            "variant_name": variant_name,
            "requested_edge_kinds": list(variant),
            "graph_file": output_path.name,
            "graph_file_sha256": _sha256_file(output_path),
            "graph_count": len(variant_graphs),
            "node_count_total": sum(len(graph.get("nodes") or []) for graph in variant_graphs),
            "edge_counts_total": totals,
            "per_trace_edge_counts": per_graph_counts,
        })

    trace_hash = _sha256_bytes("\n".join(trace_ids).encode("utf-8"))
    node_sequence_hash = _sha256_bytes("\n".join(
        json.dumps(
            {"trace_id": graph["trace_id"], "node_ids": [node.get("node_id") for node in graph.get("nodes") or []]},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        for graph in full_graphs
    ).encode("utf-8"))
    manifest = {
        "manifest_version": "edge_variants_manifest_v1",
        "dataset": dataset,
        "dataset_family": family,
        "source_identifier": source_identifier,
        "source_file_count": source_file_count,
        "source_sha256": source_sha256,
        "canonical_full_graph_sha256": canonical_full_sha256,
        "supported_edge_kinds": list(supported),
        "trace_count": len(full_graphs),
        "trace_ids_sha256": trace_hash,
        "node_id_sequences_sha256": node_sequence_hash,
        "converter_module_sha256": graph_source_sha,
        "node_retention_policy": (
            "step_and_container_spans_filtered_v1" if family == "trail"
            else "query_plus_all_original_actions_v1"
        ),
        "variants": manifest_variants,
    }
    manifest_path = target / "edge_variants_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--input", required=True, help="Raw Parquet/JSON/JSONL (TRAIL) or source directory/file (MAADBench).")
    parser.add_argument("--output-dir", required=True, help="New, non-existing directory for seven graph files and the manifest.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = generate_edge_variants(args.dataset, args.input, args.output_dir)
    print(f"Generated {len(manifest['variants'])} graph variants for {manifest['trace_count']} traces.")
    print(f"Manifest: {Path(args.output_dir).expanduser().resolve() / 'edge_variants_manifest.json'}")
    for variant in manifest["variants"]:
        print(f"{variant['variant_name']}: {variant['edge_counts_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

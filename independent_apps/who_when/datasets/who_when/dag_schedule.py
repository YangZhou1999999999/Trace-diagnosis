"""Deterministic downstream-first schedule for a Who&When diagnostic DAG."""

from __future__ import annotations

import heapq
from typing import Any, Dict, List, Mapping, Sequence, Tuple


class CausalGraphCycleError(ValueError):
    def __init__(self, diagnostics: Dict[str, Any]):
        self.diagnostics = diagnostics
        nodes = diagnostics.get("cycle_or_blocked_node_ids", [])
        super().__init__(
            "Who&When diagnostic graph has a directed cycle or nodes blocked by one: "
            + ", ".join(map(str, nodes[:12]))
            + ". Inspect graph_audit and correct edge direction before diagnosis."
        )


def downstream_first_order(
    nodes_map: Mapping[str, Dict[str, Any]],
    reverse_adj: Mapping[str, Sequence[Dict[str, Any]]],
    forward_adj: Mapping[str, Sequence[Dict[str, Any]]],
    terminal_node_id: str,
) -> Tuple[List[str], Dict[str, Any]]:
    """Return each terminal ancestor after all its downstream successors.

    The schedule is independent of a BFS/DFS display setting. A reachable
    cycle is rejected rather than visited once and silently losing states.
    """
    if terminal_node_id not in nodes_map:
        raise ValueError(f"Terminal node {terminal_node_id!r} is not in the graph")

    reachable = {terminal_node_id}
    stack = [terminal_node_id]
    while stack:
        current = stack.pop()
        for edge in reverse_adj.get(current, ()):
            source = str(edge.get("src"))
            if source in nodes_map and source not in reachable:
                reachable.add(source)
                stack.append(source)

    successors: Dict[str, set[str]] = {node_id: set() for node_id in reachable}
    predecessors: Dict[str, set[str]] = {node_id: set() for node_id in reachable}
    for source in reachable:
        for edge in forward_adj.get(source, ()):
            destination = str(edge.get("dst"))
            if destination in reachable:
                successors[source].add(destination)
                predecessors[destination].add(source)

    remaining = {node_id: len(successors[node_id]) for node_id in reachable}
    ready: List[Tuple[int, str]] = []
    for node_id, degree in remaining.items():
        if degree == 0:
            raw_step = nodes_map[node_id].get("step_index")
            try:
                step = int(raw_step)
            except (TypeError, ValueError):
                step = 0
            heapq.heappush(ready, (-step, node_id))

    order = []
    while ready:
        _neg_step, node_id = heapq.heappop(ready)
        order.append(node_id)
        for upstream in sorted(predecessors[node_id]):
            remaining[upstream] -= 1
            if remaining[upstream] == 0:
                raw_step = nodes_map[upstream].get("step_index")
                try:
                    step = int(raw_step)
                except (TypeError, ValueError):
                    step = 0
                heapq.heappush(ready, (-step, upstream))

    blocked = sorted(reachable - set(order))
    diagnostics = {
        "terminal_node_id": terminal_node_id,
        "graph_node_count": len(nodes_map),
        "reachable_node_count": len(reachable),
        "unreachable_node_count": len(nodes_map) - len(reachable),
        "unreachable_node_ids": sorted(set(nodes_map) - reachable),
        "diagnostic_edge_count": sum(len(edges) for edges in successors.values()),
        "is_dag": not blocked,
        "cycle_or_blocked_node_ids": blocked,
        "traversal_coverage_definition": "terminal_ancestors_including_terminal",
    }
    if blocked:
        raise CausalGraphCycleError(diagnostics)
    return order, diagnostics

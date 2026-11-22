"""Dataset-neutral reverse graph diagnosis orchestration.

Prompt construction, schema validation, graph conversion and output conversion
remain in dataset adapters. This module owns node scheduling, Local/Cascade
call selection, incoming Cause collection, per-node state transitions and the
shared diagnostic history.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

from .contracts import (
    CascadeFirstSeparateAdapter,
    CauseStatePolicy,
    DiagnosisAdapter,
    EdgePolicy,
    EngineRunResult,
    GraphContractError,
    NodeDiagnosis,
    _field,
    _plain,
)
from .policies import LegacyCauseStatePolicy, MaadLegacyCauseStatePolicy
from ..traversal import TraversalMode, resolve_traversal_mode


class SharedDiagnosisEngine:
    """The sole reverse-graph diagnosis loop for benchmark adapters."""

    def __init__(self, *, max_causes: int = 8) -> None:
        if max_causes < 0:
            raise ValueError("max_causes must be non-negative")
        self.max_causes = max_causes

    @classmethod
    def backward_schedule(
        cls,
        node_ids: Sequence[str],
        edges: Sequence[Mapping[str, Any]],
        *,
        entry_node_id: str,
        traversal_mode: str = "bfs",
        schedule_mode: Literal["reachable", "all_nodes", "reverse_topological", "reverse_topological_all_nodes"] = "reachable",
        edge_priority: Optional[Mapping[str, int]] = None,
        parent_order_by_target: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> tuple[list[str], list[str]]:
        """Public scheduler shared by adapters and the production `run()` loop."""
        return cls._build_schedule(
            node_ids, edges,
            entry_node_id=entry_node_id,
            traversal_mode=traversal_mode,
            schedule_mode=schedule_mode,
            edge_priority=edge_priority,
            parent_order_by_target=parent_order_by_target,
        )

    @staticmethod
    def _cyclic_components(node_ids: Sequence[str], edges: Sequence[Mapping[str, Any]]) -> list[list[str]]:
        """Return only nodes in actual SCC cycles (not Kahn's downstream residue)."""
        adjacency: dict[str, list[str]] = {str(node): [] for node in node_ids}
        for edge in edges:
            src, dst = str(edge.get("src", "")), str(edge.get("dst", ""))
            if src in adjacency and dst in adjacency:
                adjacency[src].append(dst)
        index = 0
        stack: list[str] = []
        on_stack: set[str] = set()
        indices: dict[str, int] = {}
        lowlink: dict[str, int] = {}
        cyclic: list[list[str]] = []

        def visit(node: str) -> None:
            nonlocal index
            indices[node] = index
            lowlink[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)
            for neighbor in adjacency[node]:
                if neighbor not in indices:
                    visit(neighbor)
                    lowlink[node] = min(lowlink[node], lowlink[neighbor])
                elif neighbor in on_stack:
                    lowlink[node] = min(lowlink[node], indices[neighbor])
            if lowlink[node] != indices[node]:
                return
            component: list[str] = []
            while stack:
                current = stack.pop()
                on_stack.remove(current)
                component.append(current)
                if current == node:
                    break
            if len(component) > 1 or any(current == node for current in adjacency[node]):
                cyclic.append(sorted(component))

        for node in adjacency:
            if node not in indices:
                visit(node)
        return sorted(cyclic)

    @classmethod
    def validate_graph(
        cls,
        raw_nodes: Sequence[Mapping[str, Any]],
        edges: Sequence[Mapping[str, Any]],
        *,
        entry_node_id: str,
        edge_policy: EdgePolicy,
    ) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """Validate source graph, then return traversal/propagation projections."""
        mapping_nodes = [node for node in raw_nodes if isinstance(node, Mapping)]
        raw_ids = [str(node.get("node_id")) for node in mapping_nodes if node.get("node_id") is not None]
        duplicate_ids = sorted({node_id for node_id in raw_ids if raw_ids.count(node_id) > 1})
        missing_id_nodes = sum(1 for node in mapping_nodes if node.get("node_id") is None)
        malformed_node_count = len(raw_nodes) - len(mapping_nodes)
        all_ids = list(dict.fromkeys(raw_ids))
        valid = set(all_ids)
        malformed_edges = 0
        missing_endpoints: list[dict[str, Any]] = []
        self_loops: list[dict[str, Any]] = []
        unknown_kinds: set[str] = set()
        normalized: list[dict[str, Any]] = []
        for index, edge in enumerate(edges):
            if not isinstance(edge, Mapping):
                malformed_edges += 1
                continue
            src, dst = str(edge.get("src", "")), str(edge.get("dst", ""))
            kind = str(edge.get("kind", edge.get("type", "unknown"))).lower()
            if src not in valid or dst not in valid:
                missing_endpoints.append({"edge_index": index, "src_exists": src in valid, "dst_exists": dst in valid, "kind": kind})
                continue
            item = dict(edge)
            item["src"], item["dst"], item["kind"] = src, dst, kind
            normalized.append(item)
            if kind not in {str(value).lower() for value in edge_policy.known_kinds}:
                unknown_kinds.add(kind)
            if src == dst:
                self_loops.append({"src": src, "dst": dst, "kind": kind, "edge_index": index})

        traversable = [edge for edge in normalized if edge_policy.allows_traversal(edge)]
        propagating = [edge for edge in traversable if edge_policy.allows_propagation(edge)]
        reachable: set[str] = set()
        frontier = deque([entry_node_id]) if entry_node_id in valid else deque()
        parents: dict[str, list[str]] = defaultdict(list)
        for edge in traversable:
            parents[edge["dst"]].append(edge["src"])
        while frontier:
            current = frontier.popleft()
            if current in reachable:
                continue
            reachable.add(current)
            frontier.extend(parent for parent in parents.get(current, ()) if parent not in reachable)
        traversal_cycles = cls._cyclic_components(all_ids, traversable)
        propagation_cycles = cls._cyclic_components(all_ids, propagating)
        reachable_cycles = sorted({node for comp in traversal_cycles for node in comp if node in reachable})
        reachable_self_loops = [item for item in self_loops if item["src"] in reachable and edge_policy.allows_traversal({"kind": item["kind"]})]
        pair_kinds: dict[tuple[str, str], set[str]] = defaultdict(set)
        for edge in normalized:
            pair_kinds[(edge["src"], edge["dst"])].add(edge["kind"])
        multi_kind_pairs = [
            {"src": src, "dst": dst, "kinds": sorted(kinds)}
            for (src, dst), kinds in sorted(pair_kinds.items()) if len(kinds) > 1
        ]
        report = {
            "edge_policy": edge_policy.name,
            "edge_policy_fingerprint": edge_policy.fingerprint(),
            "cycle_strategy": edge_policy.cycle_strategy,
            "raw_node_count": len(raw_nodes),
            "node_count_with_id": len(raw_ids),
            "duplicate_node_ids": duplicate_ids,
            "missing_node_id_count": missing_id_nodes,
            "malformed_node_count": malformed_node_count,
            "malformed_edge_count": malformed_edges,
            "missing_endpoint_edges": missing_endpoints,
            "self_loop_edges": self_loops,
            "unknown_edge_kinds": sorted(unknown_kinds),
            "all_cycle_components_traversal": traversal_cycles,
            "reachable_cycle_nodes_traversal": reachable_cycles,
            "nonreachable_cycle_nodes_traversal": sorted({node for comp in traversal_cycles for node in comp if node not in reachable}),
            "all_cycle_components_propagation": propagation_cycles,
            "traversable_edge_count": len(traversable),
            "propagation_edge_count": len(propagating),
            "excluded_traversal_edge_count": len(normalized) - len(traversable),
            "excluded_propagation_edge_count": len(traversable) - len(propagating),
            "prompt_context_kinds": edge_policy.prompt_context_kinds,
            "evidence_strength": dict(edge_policy.evidence_strength),
            "multi_kind_src_dst_pairs": multi_kind_pairs,
        }
        contract_issues = bool(duplicate_ids or missing_id_nodes or malformed_node_count or malformed_edges or missing_endpoints)
        if contract_issues:
            raise GraphContractError("Graph has malformed node/edge structure", report)
        if edge_policy.cycle_strategy == "fail_closed" and (reachable_cycles or reachable_self_loops):
            report["action"] = "fail_closed_before_llm"
            raise GraphContractError("Reachable traversal projection is cyclic under fail_closed policy", report)
        report["action"] = "bounded_finite_schedule" if reachable_cycles or reachable_self_loops else "validated_dag_schedule"
        return all_ids, traversable, propagating, report

    @staticmethod
    def _build_schedule(
        node_ids: Sequence[str],
        edges: Sequence[Mapping[str, Any]],
        *,
        entry_node_id: str,
        traversal_mode: str,
        schedule_mode: Literal["reachable", "all_nodes", "reverse_topological", "reverse_topological_all_nodes"],
        edge_priority: Optional[Mapping[str, int]] = None,
        parent_order_by_target: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> tuple[list[str], list[str]]:
        mode = resolve_traversal_mode(traversal_mode, default="bfs")
        valid = set(node_ids)
        parents: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        children: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        pair_rank: dict[tuple[str, str], tuple[int, int]] = {}
        node_rank = {node_id: index for index, node_id in enumerate(node_ids)}
        for edge_index, edge in enumerate(edges):
            src, dst = str(edge.get("src", "")), str(edge.get("dst", ""))
            if src in valid and dst in valid and src != dst:
                if src not in parents[dst]:
                    parents[dst].append(src)
                if dst not in children[src]:
                    children[src].append(dst)
                key = (src, dst)
                priority = (edge_priority or {}).get(str(edge.get("kind", "")).lower(), 3)
                candidate = (priority, edge_index)
                if key not in pair_rank or candidate < pair_rank[key]:
                    pair_rank[key] = candidate
        for target in parents:
            explicit_order = {
                str(source): index
                for index, source in enumerate((parent_order_by_target or {}).get(target, ()))
            }
            if explicit_order:
                parents[target].sort(key=lambda source: (
                    explicit_order.get(source, len(explicit_order)),
                    pair_rank.get((source, target), (3, 0)),
                ))
            else:
                parents[target].sort(key=lambda source: (
                    pair_rank.get((source, target), (3, 0))[0],
                    node_rank.get(source, 0),
                    pair_rank.get((source, target), (3, 0))[1],
                ))
        reachable: set[str] = set()
        frontier = deque([entry_node_id]) if entry_node_id in valid else deque()
        while frontier:
            current = frontier.popleft() if mode == "bfs" else frontier.pop()
            if current in reachable:
                continue
            reachable.add(current)
            neighbors = list(parents.get(current, ()))
            if mode == "dfs":
                neighbors.reverse()
            frontier.extend(item for item in neighbors if item not in reachable)

        if schedule_mode == "reverse_topological_all_nodes":
            active_nodes = set(node_ids)
            indegree = {node: 0 for node in active_nodes}
            for child in active_nodes:
                for parent in parents[child]:
                    if parent in active_nodes:
                        indegree[parent] += 1
            ready = deque(node for node in node_ids if node in active_nodes and indegree[node] == 0)
            order: list[str] = []
            while ready:
                current = ready.popleft() if mode == "bfs" else ready.pop()
                order.append(current)
                for parent in parents[current]:
                    if parent not in indegree:
                        continue
                    indegree[parent] -= 1
                    if indegree[parent] == 0:
                        ready.append(parent)
            residual = {node for node, degree in indegree.items() if degree > 0}
            cycles = sorted({
                node for component in SharedDiagnosisEngine._cyclic_components(
                    list(active_nodes),
                    [edge for edge in edges
                     if str(edge.get("src")) in active_nodes and str(edge.get("dst")) in active_nodes],
                ) for node in component
            })
            if residual:
                order.extend(node for node in node_ids if node in residual and node not in order)
            return order, cycles

        if schedule_mode == "reverse_topological":
            indegree = {node: 0 for node in reachable}
            for child in reachable:
                for parent in parents[child]:
                    if parent in reachable:
                        indegree[parent] += 1
            ready = deque(node for node in node_ids if node in reachable and indegree[node] == 0)
            order: list[str] = []
            while ready:
                current = ready.popleft() if mode == "bfs" else ready.pop()
                order.append(current)
                for parent in parents[current]:
                    if parent not in indegree:
                        continue
                    indegree[parent] -= 1
                    if indegree[parent] == 0:
                        ready.append(parent)
            residual = sorted(node for node, degree in indegree.items() if degree > 0)
            cycles = sorted({node for component in SharedDiagnosisEngine._cyclic_components(list(reachable), [edge for edge in edges if str(edge.get("src")) in reachable and str(edge.get("dst")) in reachable]) for node in component})
            if residual:
                # Preserve a finite diagnosis pass and expose the cyclic set. A
                # caller that requires a DAG can fail closed on this field.
                order.extend(node for node in node_ids if node in residual and node not in order)
            return order, cycles

        order = []
        seen: set[str] = set()
        frontier = deque([entry_node_id]) if entry_node_id in valid else deque()
        while frontier:
            current = frontier.popleft() if mode == "bfs" else frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            order.append(current)
            neighbors = [item for item in parents.get(current, ()) if item not in seen]
            if mode == "dfs":
                neighbors.reverse()
            frontier.extend(neighbors)
        if schedule_mode == "all_nodes":
            for node_id in node_ids:
                if node_id not in seen:
                    order.append(node_id)
                    seen.add(node_id)
        return order, []

    def run(
        self,
        *,
        graph: Mapping[str, Any],
        adapter: DiagnosisAdapter,
        cause_policy: CauseStatePolicy,
        entry_node_id: str,
        traversal_mode: str = "bfs",
        schedule_mode: Literal["reachable", "all_nodes", "reverse_topological", "reverse_topological_all_nodes"] = "reachable",
        node_ids: Optional[Sequence[str]] = None,
        edge_priority: Optional[Mapping[str, int]] = None,
        parent_order_by_target: Optional[Mapping[str, Sequence[str]]] = None,
        edge_policy: Optional[EdgePolicy] = None,
        worklist_visit_budget: Optional[int] = None,
    ) -> EngineRunResult:
        raw_nodes = list(graph.get("nodes") or [])
        nodes = [node for node in raw_nodes if isinstance(node, dict)]
        raw_edges = list(graph.get("edges") or [])
        policy = edge_policy or EdgePolicy.legacy()
        if policy.cycle_strategy not in {"bounded_worklist", "fail_closed"}:
            raise ValueError("EdgePolicy.cycle_strategy must be bounded_worklist or fail_closed")
        all_ids, traversal_edges, propagation_edges, graph_validation = self.validate_graph(
            raw_nodes, raw_edges,
            entry_node_id=str(entry_node_id),
            edge_policy=policy,
        )
        node_map = {str(node.get("node_id")): node for node in nodes if node.get("node_id") is not None}
        ordered_ids = [str(item) for item in (node_ids or list(node_map)) if str(item) in node_map]
        if len(set(ordered_ids)) != len(ordered_ids):
            raise ValueError("SharedDiagnosisEngine received duplicate node IDs")
        if entry_node_id not in node_map:
            raise ValueError(f"Diagnosis entry node {entry_node_id!r} is missing")
        cycles = list(graph_validation["reachable_cycle_nodes_traversal"])
        if schedule_mode in {"all_nodes", "reverse_topological_all_nodes"}:
            cycles = sorted({
                node for component in graph_validation["all_cycle_components_traversal"]
                for node in component
            })
        schedule, cycle_nodes = self.backward_schedule(
            ordered_ids, traversal_edges,
            entry_node_id=entry_node_id,
            traversal_mode=traversal_mode,
            schedule_mode=schedule_mode,
            edge_priority=edge_priority,
            parent_order_by_target=parent_order_by_target,
        )
        # Scheduler's legacy return reports only cycles in the selected reverse
        # topological order. The preflight report is authoritative for every mode.
        cycle_nodes = cycles
        if policy.cycle_strategy == "fail_closed" and cycle_nodes:
            graph_validation["action"] = "fail_closed_before_llm"
            raise GraphContractError("Reachable traversal projection contains a cycle", graph_validation)

        configure_policy = getattr(adapter, "configure_edge_policy", None)
        if callable(configure_policy):
            configure_policy(policy)

        separate_order = "not_applicable"
        inspect_local_with_cascade = None
        if adapter.cascade_enabled and adapter.cascade_call_mode == "separate":
            # Legacy/custom adapters that do not declare the new option retain
            # the historical Local -> Cascade order.
            separate_order = getattr(adapter, "separate_order", "local_then_cascade")
            if separate_order not in {"local_then_cascade", "cascade_then_local"}:
                raise ValueError(f"Unsupported Separate execution order: {separate_order}")
            if separate_order == "cascade_then_local":
                inspect_local_with_cascade = getattr(adapter, "inspect_local_with_cascade", None)
                if not callable(inspect_local_with_cascade):
                    raise ValueError(
                        "cascade_then_local requires adapter.inspect_local_with_cascade(prepared, cascade)"
                    )

        raw_seeds = adapter.initialize_hypotheses(node_map[entry_node_id]) if adapter.cascade_enabled else ()
        seed_causes = [cause_policy.seed(item, seed_node_id=entry_node_id) for item in raw_seeds]
        incoming_by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
        outgoing_by_node: dict[str, list[dict[str, Any]]] = {}
        if seed_causes:
            incoming_by_node[entry_node_id].extend(seed_causes)
            outgoing_by_node[entry_node_id] = [dict(item) for item in seed_causes]

        diagnoses: dict[str, NodeDiagnosis] = {}
        history: list[dict[str, Any]] = []
        execution_order: list[str] = []
        node_positions = {node_id: index for index, node_id in enumerate(schedule)}
        pending = deque(schedule)
        pending_nodes = set(schedule)
        visit_count: dict[str, int] = defaultdict(int)
        processed_state_ids: dict[str, set[Any]] = defaultdict(set)
        local_cache: dict[str, Any] = {}
        supports_revisit = bool(getattr(cause_policy, "reprocess_new_states", False))
        if worklist_visit_budget is not None and int(worklist_visit_budget) < 1:
            raise ValueError("worklist_visit_budget must be positive")
        actual_visit_budget = (
            int(worklist_visit_budget) if worklist_visit_budget is not None
            else (max(1, len(schedule) * 8) if cycle_nodes and policy.cycle_strategy == "bounded_worklist" else None)
        )
        propagation_decisions: list[dict[str, Any]] = []

        def attach_partial_progress(
            exc: BaseException,
            *,
            current_node_id: str,
            stage: str,
            current_local: Any = None,
            current_cascade: Any = None,
        ) -> None:
            """Keep completed nodes inspectable when an adapter request is blocked."""
            try:
                diagnosis_mode = (
                    adapter.cascade_call_mode if adapter.cascade_enabled else "disabled"
                )
                first_visit = visit_count[current_node_id] == 0
                effective_order = (
                    separate_order if diagnosis_mode == "separate" else "not_applicable"
                )
                local_from_cache = (
                    current_local is not None
                    and local_cache.get(current_node_id) is current_local
                )
                local_conditioned_on_cascade = (
                    effective_order == "cascade_then_local"
                    and stage == "local_after_cascade_inspection"
                )
                cascade_recomputed = stage == "cascade_inspection" and not first_visit
                local_stage = (
                    "local_after_cascade" if effective_order == "cascade_then_local"
                    else "local_before_cascade" if diagnosis_mode == "separate"
                    else "combined" if diagnosis_mode == "combined"
                    else "local_only"
                )
                cascade_stage = (
                    "cascade_recomputed" if cascade_recomputed
                    else "cascade_before_local" if effective_order == "cascade_then_local"
                    else "cascade_after_local" if diagnosis_mode == "separate"
                    else "combined" if diagnosis_mode == "combined"
                    else "not_run"
                )
                setattr(exc, "shared_diagnosis_partial", {
                    "status": "PARTIAL",
                    "failure_stage": stage,
                    "diagnosis_mode": diagnosis_mode,
                    "separate_order": effective_order,
                    "local_stage": local_stage,
                    "cascade_stage": cascade_stage,
                    "first_visit": first_visit,
                    "local_from_cache": local_from_cache,
                    "cascade_recomputed": cascade_recomputed,
                    "local_conditioned_on_cascade": local_conditioned_on_cascade,
                    "local_reconditioned_on_new_cascade": False,
                    "failed_node_id": current_node_id,
                    "execution_order": list(execution_order),
                    "completed_node_ids": list(diagnoses),
                    "history": [_plain(item) for item in history],
                    "current_local_inspection": _plain(current_local) if current_local is not None else None,
                    "current_cascade_inspection": _plain(current_cascade) if current_cascade is not None else None,
                    "pending_node_ids": list(pending),
                    "unexecuted_node_ids": [item for item in schedule if item not in diagnoses],
                    "incoming_causes": _plain(incoming_by_node.get(current_node_id, ())),
                })
            except Exception:
                # Preserve the original provider/budget exception if audit
                # metadata cannot be materialized.
                pass

        def state_identity(state: Mapping[str, Any]) -> Any:
            identity = getattr(cause_policy, "identity", None)
            if callable(identity):
                return identity(state)
            cause = state.get("cause") if isinstance(state.get("cause"), Mapping) else state
            return str(cause.get("cause_id", "")), str(state.get("incoming_path_id", ""))

        def consume_policy_audit() -> list[dict[str, Any]]:
            consume = getattr(cause_policy, "consume_audit", None)
            return consume() if callable(consume) else []

        while pending:
            if actual_visit_budget is not None and sum(visit_count.values()) >= actual_visit_budget:
                graph_validation["worklist_visit_budget_exhausted"] = True
                graph_validation["worklist_visit_budget"] = actual_visit_budget
                graph_validation["unprocessed_pending_nodes"] = list(pending)
                break
            node_id = pending.popleft()
            pending_nodes.discard(node_id)
            all_incoming = cause_policy.merge_incoming(incoming_by_node.get(node_id, ()))
            state_audit = consume_policy_audit()
            first_visit = visit_count[node_id] == 0
            if first_visit:
                incoming = all_incoming
            elif supports_revisit:
                incoming = [
                    state for state in all_incoming
                    if state_identity(state) not in processed_state_ids[node_id]
                ]
                if not incoming:
                    continue
            else:
                continue

            position = node_positions.get(node_id, len(node_positions))
            prepared = adapter.prepare_node(node_id, node_map[node_id], incoming, position)
            if isinstance(prepared, dict):
                prepared["first_visit"] = first_visit

            local_from_cache = False
            cascade_recomputed = False
            local_conditioned_on_cascade = False
            if not adapter.cascade_enabled:
                diagnosis_mode = "disabled"
            else:
                diagnosis_mode = adapter.cascade_call_mode
            effective_separate_order = (
                separate_order if diagnosis_mode == "separate" else "not_applicable"
            )

            if not first_visit and node_id in local_cache:
                local = local_cache[node_id]
                local_from_cache = True
                if adapter.cascade_enabled:
                    cascade_input = None if effective_separate_order == "cascade_then_local" else local
                    try:
                        cascade = adapter.inspect_cascade(prepared, cascade_input)
                    except Exception as exc:
                        attach_partial_progress(
                            exc, current_node_id=node_id, stage="cascade_inspection",
                            current_local=local,
                        )
                        raise
                    cascade_recomputed = True
                else:
                    cascade = None
                diagnosis = NodeDiagnosis(local=local, cascade=cascade)
                local_stage = "cached_local"
                cascade_stage = "cascade_recomputed" if cascade is not None else "not_run"
            elif not adapter.cascade_enabled:
                try:
                    local = adapter.inspect_local(prepared)
                except Exception as exc:
                    attach_partial_progress(
                        exc, current_node_id=node_id, stage="local_inspection",
                    )
                    raise
                diagnosis = NodeDiagnosis(local=local, cascade=None)
                local_stage = "local_only"
                cascade_stage = "not_run"
            elif adapter.cascade_call_mode == "combined":
                try:
                    combined = adapter.inspect_combined(prepared)
                except Exception as exc:
                    attach_partial_progress(
                        exc, current_node_id=node_id, stage="combined_inspection",
                    )
                    raise
                if not isinstance(combined, NodeDiagnosis):
                    raise TypeError("inspect_combined() must return NodeDiagnosis")
                if combined.cascade is None:
                    raise ValueError("Combined Local/Cascade diagnosis returned no cascade result")
                diagnosis = combined
                local_stage = "combined"
                cascade_stage = "combined"
            elif separate_order == "cascade_then_local":
                try:
                    cascade = adapter.inspect_cascade(prepared, None)
                    if cascade is None:
                        raise ValueError("Cascade inspection returned no result")
                except Exception as exc:
                    attach_partial_progress(
                        exc, current_node_id=node_id, stage="cascade_inspection",
                    )
                    raise
                try:
                    local = inspect_local_with_cascade(prepared, cascade)
                except Exception as exc:
                    attach_partial_progress(
                        exc,
                        current_node_id=node_id,
                        stage="local_after_cascade_inspection",
                        current_cascade=cascade,
                    )
                    raise
                diagnosis = NodeDiagnosis(local=local, cascade=cascade)
                local_stage = "local_after_cascade"
                cascade_stage = "cascade_before_local"
                local_conditioned_on_cascade = True
            else:
                try:
                    local = adapter.inspect_local(prepared)
                except Exception as exc:
                    attach_partial_progress(
                        exc, current_node_id=node_id, stage="local_inspection",
                    )
                    raise
                try:
                    cascade = adapter.inspect_cascade(prepared, local)
                except Exception as exc:
                    attach_partial_progress(
                        exc, current_node_id=node_id, stage="cascade_inspection", current_local=local,
                    )
                    raise
                diagnosis = NodeDiagnosis(local=local, cascade=cascade)
                local_stage = "local_before_cascade"
                cascade_stage = "cascade_after_local"

            finalize = getattr(adapter, "finalize_diagnosis", None)
            if callable(finalize):
                finalized = finalize(prepared, diagnosis)
                if finalized is not None:
                    diagnosis = finalized
                    if not isinstance(diagnosis, NodeDiagnosis):
                        raise TypeError("finalize_diagnosis() must return NodeDiagnosis or None")

            diagnosis.metadata.update({
                "diagnosis_mode": diagnosis_mode,
                "separate_order": effective_separate_order,
                "local_stage": local_stage,
                "cascade_stage": cascade_stage,
                "first_visit": first_visit,
                "local_from_cache": local_from_cache,
                "cascade_recomputed": cascade_recomputed,
                "local_conditioned_on_cascade": local_conditioned_on_cascade,
                # H1 revisits intentionally reuse the first Local judgment.
                "local_reconditioned_on_new_cascade": False,
            })

            diagnoses[node_id] = diagnosis
            local_cache[node_id] = diagnosis.local
            visit_count[node_id] += 1
            execution_order.append(node_id)
            processed_state_ids[node_id].update(state_identity(state) for state in incoming)

            validation_audit: list[dict[str, Any]] = []
            if adapter.cascade_enabled:
                validate_delta = getattr(adapter, "prepare_cascade_delta", None)
                if callable(validate_delta):
                    prepared_delta = validate_delta(prepared, incoming, diagnosis)
                    if isinstance(prepared_delta, tuple) and len(prepared_delta) == 2:
                        diagnosis.cascade, validation_audit = prepared_delta
                    elif prepared_delta is not None:
                        diagnosis.cascade = prepared_delta
            if diagnosis.cascade is not None:
                updated = cause_policy.apply_delta(
                    incoming, diagnosis.cascade, node_id=node_id, max_causes=self.max_causes,
                )
            else:
                updated = []
            state_audit.extend(validation_audit)
            state_audit.extend(consume_policy_audit())
            diagnosis.metadata["first_visit"] = first_visit
            # Keep metadata and history as separate list objects. Edge-level
            # audit events are appended to both surfaces after the node record
            # is created; sharing this list used to duplicate each event.
            diagnosis.metadata["cause_state_audit"] = list(state_audit)
            previous_outgoing = outgoing_by_node.get(node_id, [])
            combine = getattr(cause_policy, "combine_outgoing", None)
            if supports_revisit and callable(combine):
                outgoing_by_node[node_id] = combine(previous_outgoing, updated)
            elif supports_revisit:
                outgoing_by_node[node_id] = cause_policy.merge_incoming([*previous_outgoing, *updated])
            else:
                outgoing_by_node[node_id] = updated
            node_audit = adapter.after_node(node_id, prepared, diagnosis, incoming, updated)
            default_history = {
                "node_id": node_id,
                "position": position,
                "visit": visit_count[node_id],
                "active_before": _plain(incoming),
                "local": _plain(diagnosis.local),
                "cascade": _plain(diagnosis.cascade),
                "active_after": _plain(updated),
                "cause_state_audit": list(state_audit),
            }
            if node_audit and isinstance(node_audit.get("history_record"), Mapping):
                history_record = dict(node_audit["history_record"])
                if state_audit:
                    history_record.setdefault("cause_state_audit", []).extend(state_audit)
                history.append(history_record)
                diagnosis.metadata.update({
                    key: value for key, value in node_audit.items() if key != "history_record"
                })
            else:
                history.append({**default_history, **(node_audit or {})})

            if not adapter.cascade_enabled:
                continue
            propagation = cause_policy.propagating(updated)
            if not propagation:
                continue
            for edge in traversal_edges:
                if str(edge.get("dst", "")) != node_id:
                    continue
                parent = str(edge.get("src", ""))
                if parent not in node_map or parent not in schedule or parent == node_id:
                    continue
                if not policy.allows_propagation(edge):
                    propagation_decisions.append({
                        "node_id": node_id, "parent_id": parent, "kind": edge.get("kind"),
                        "cause_ids": [str(_field(item.get("cause"), "cause_id", item.get("cause_id", ""))) for item in propagation],
                        "decision": "BLOCKED_BY_EDGE_POLICY",
                        "evidence_strength": dict(policy.evidence_strength).get(str(edge.get("kind", "")).lower(), "unspecified"),
                    })
                    continue
                propagate = getattr(cause_policy, "propagate_to_parent", None)
                if callable(propagate):
                    new_states = propagate(propagation, node_id=node_id, parent_id=parent, edge=edge)
                    edge_audit = consume_policy_audit()
                    if edge_audit:
                        diagnosis.metadata.setdefault("cause_state_audit", []).extend(edge_audit)
                        if history:
                            history[-1].setdefault("cause_state_audit", []).extend(edge_audit)
                else:
                    new_states = [dict(item) for item in propagation]
                if not new_states:
                    continue
                propagation_decisions.append({
                    "node_id": node_id, "parent_id": parent, "kind": edge.get("kind"),
                    "cause_ids": [str(_field(item.get("cause"), "cause_id", item.get("cause_id", ""))) for item in new_states],
                    "decision": "STATE_PROPAGATED",
                    "evidence_strength": dict(policy.evidence_strength).get(str(edge.get("kind", "")).lower(), "unspecified"),
                })
                incoming_by_node[parent].extend(dict(item) for item in new_states)
                if supports_revisit and visit_count[parent] > 0 and parent not in pending_nodes:
                    pending.append(parent)
                    pending_nodes.add(parent)

        path_budget = getattr(cause_policy, "max_path_states", None)
        path_count = getattr(cause_policy, "path_state_count", None)
        history_audits = [
            audit for item in history for audit in (item.get("cause_state_audit") or [])
            if isinstance(audit, Mapping)
        ]
        cycle_terminated_paths = sum(1 for item in history_audits if item.get("status") == "CYCLE_PATH_TERMINATED")
        path_state_budget_exhausted = any(item.get("status") == "PATH_STATE_BUDGET_TRUNCATED" for item in history_audits)
        blocked_nodes = [node for node in schedule if visit_count[node] == 0]
        partial_coverage = bool(
            blocked_nodes or path_state_budget_exhausted
            or graph_validation.get("worklist_visit_budget_exhausted")
            or cycle_nodes
        )
        graph_validation["propagation_decisions"] = propagation_decisions
        graph_validation["execution_order"] = list(execution_order)
        graph_validation["node_visit_count"] = dict(visit_count)
        graph_validation["processed_cause_path_states"] = sum(len(values) for values in processed_state_ids.values())
        graph_validation["cycle_terminated_paths"] = cycle_terminated_paths
        graph_validation["path_state_budget"] = path_budget
        graph_validation["path_state_budget_exhausted"] = path_state_budget_exhausted
        graph_validation["partial_coverage"] = partial_coverage
        graph_validation["blocked_nodes"] = blocked_nodes
        return EngineRunResult(
            schedule=schedule,
            diagnoses=diagnoses,
            incoming_causes={key: cause_policy.merge_incoming(value) for key, value in incoming_by_node.items()},
            outgoing_causes=outgoing_by_node,
            history=history,
            seed_causes=seed_causes,
            scheduler=schedule_mode,
            cycle_nodes=cycle_nodes,
            execution_order=execution_order,
            graph_validation=graph_validation,
            node_visit_count=dict(visit_count),
            processed_cause_path_states=sum(len(values) for values in processed_state_ids.values()),
            cycle_terminated_paths=cycle_terminated_paths,
            path_state_budget=path_budget,
            path_state_budget_exhausted=path_state_budget_exhausted,
            partial_coverage=partial_coverage,
            blocked_nodes=blocked_nodes,
            propagation_decisions=propagation_decisions,
        )

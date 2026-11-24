"""Who&When Stage 1: inspect a DAG after downstream branch facts arrive."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from .causal_state import (
    BranchState,
    advance_branch,
    branch_prompt_summary,
    cause_record_at_node,
    compatible_causes,
    compress_incoming,
    seed_branches,
    summarize_branches,
)
from .dag_schedule import downstream_first_order
from .evidence_audit import audit_quote
from .schemas import FailureCause


def _copy_state(state: BranchState) -> BranchState:
    return replace(state, cause=FailureCause(**state.cause.model_dump()))


def _relation_for(cascade: Dict[str, Any], cause_id: str) -> Dict[str, Any]:
    relations = cascade.get("cause_relations")
    if isinstance(relations, list):
        for item in relations:
            if isinstance(item, dict) and str(item.get("cause_id")) == cause_id:
                return item
        return {"relation": "UNCERTAIN", "explanation": "No per-cause relation returned."}
    return cascade


def diagnose_dag(
    engine: Any,
    start_node_id: str,
    traversal_path: List[str],
    error_chain: List[Dict[str, Any]],
    initial_active_causes: List[FailureCause],
) -> Tuple[Optional[Dict[str, Any]], List[FailureCause], List[Dict[str, Any]]]:
    order, diagnostics = downstream_first_order(
        engine.nodes_map, engine.reverse_adj, engine.forward_adj, start_node_id
    )
    engine.graph_diagnostics = diagnostics
    incoming: Dict[str, List[BranchState]] = {node_id: [] for node_id in order}
    incoming[start_node_id] = seed_branches(initial_active_causes)
    candidates_by_node: Dict[str, Dict[str, Any]] = {}

    for node_id in order:
        before = compress_incoming(incoming[node_id], node_id)
        for state in before:
            engine.branch_provenance_records[state.state_id] = state.audit()
        active = [state for state in before if state.active]
        current_node = engine.nodes_map[node_id]
        predecessors = engine._get_sorted_predecessors(node_id)
        predecessor_nodes = [engine.nodes_map[src_id] for src_id, _kind in predecessors]
        prompt_predecessors = engine._select_prompt_predecessors(predecessor_nodes)
        successors = engine._get_sorted_successors(node_id)
        successor_nodes = [engine.nodes_map[dst_id] for dst_id, _kind in successors]

        local, updated, cascade = engine._inspect_current_node_with_cascade(
            current_node=current_node,
            predecessor_nodes=prompt_predecessors,
            successor_nodes=successor_nodes,
            active_causes=[FailureCause(**state.cause.model_dump()) for state in active],
            branch_paths=[state.path for state in active],
            downstream_branch_summary=branch_prompt_summary(before),
        )
        if len(updated) != len(active):
            raise RuntimeError(
                f"Stage 1 returned {len(updated)} states for {len(active)} active paths at {node_id}"
            )

        after: List[BranchState] = []
        active_index = 0
        for state in before:
            if not state.active:
                after.append(advance_branch(state, node_id=node_id))
                continue
            item = _relation_for(cascade, state.cause.cause_id)
            progressed = advance_branch(
                state,
                node_id=node_id,
                relation=str(item.get("relation") or "UNCERTAIN"),
                explanation=str(item.get("explanation") or ""),
                relation_confidence=item.get("confidence"),
                evidence=item.get("evidence"),
                root_candidate=bool(item.get("is_root_candidate")),
            )
            # The LLM adapter and the branch transition use the same
            # conservative policy; flag drift rather than silently merging it.
            expected = updated[active_index]
            if (progressed.cause.support_status, progressed.cause.should_propagate_upstream) != (
                expected.support_status, expected.should_propagate_upstream
            ):
                raise RuntimeError(f"Branch transition disagrees with adapter at {node_id}/{state.cause.cause_id}")
            active_index += 1
            after.append(progressed)

        after = [
            replace(
                state,
                state_id=f"{node_id}:out:{index}",
                origin_state_ids=(before[index].state_id,),
            )
            for index, state in enumerate(after)
        ]
        for state in after:
            engine.branch_provenance_records[state.state_id] = state.audit()
        facts = [cause_record_at_node(old, new, node_id) for old, new in zip(before, after)]
        for fact in facts:
            fact.update(audit_quote(fact.get("evidence"), current_node))
        summary = summarize_branches(after)
        engine.branch_states_by_node[node_id] = {
            "before": [state.audit() for state in before],
            "after": [state.audit() for state in after],
            "summary": summary,
        }
        engine.cause_state_summaries_by_node[node_id] = summary
        engine.active_causes_by_node[node_id] = compatible_causes(after, initial_active_causes)
        engine.visited.add(node_id)
        traversal_path.append(node_id)

        is_error = bool(local.get("is_error"))
        error_role = str(local.get("error_role") or "UNCERTAIN").upper()
        local_root_candidate = bool(local.get("is_root_candidate"))
        cascade_root_candidate = bool(cascade.get("is_root_candidate"))
        cascade_relation = str(cascade.get("relation") or "UNCERTAIN").upper()

        # Original Who&When candidate admission: retain only a Local error or
        # a Cascade RootCandidate. Do not add generic propagation/exposure nodes.
        should_add = bool(is_error or cascade_root_candidate)
        direct_added = False
        if should_add:
            candidate_error_role = error_role
            if not is_error and cascade_root_candidate and cascade_relation == "INTRODUCES":
                candidate_error_role = "INTRODUCES_ERROR"
            candidate_role = str(local.get("candidate_role") or "NO_RELEVANT_EVIDENCE").upper()
            if cascade_root_candidate and cascade_relation == "INTRODUCES":
                candidate_role = "INTRODUCER_CANDIDATE"
            elif candidate_role == "NO_RELEVANT_EVIDENCE":
                candidate_role = {
                    "INTRODUCES_ERROR": "INTRODUCER_CANDIDATE",
                    "PROPAGATES_ERROR": "PROPAGATION_EVIDENCE",
                    "VERIFIES_PRIOR_ERROR": "VERIFICATION_EVIDENCE",
                    "TERMINATES_AFTER_ERROR": "EXPOSURE_EVIDENCE",
                }.get(candidate_error_role, "NO_RELEVANT_EVIDENCE")

            item = engine._make_candidate_item(
                node_id=node_id,
                node=current_node,
                local_eval=local,
                cascade_eval=cascade,
                candidate_role=candidate_role,
                candidate_source=(
                    "LOCAL_ERROR_AND_CASCADE_ROOT" if is_error and cascade_root_candidate
                    else "LOCAL_ERROR" if is_error else "CASCADE_ROOT_CANDIDATE"
                ),
                error_role=candidate_error_role,
                error_reason=str(local.get("reason") or cascade.get("explanation") or ""),
                is_root_candidate=bool(local_root_candidate or cascade_root_candidate),
                is_soft_candidate=False,
            )
            item["cause_records"] = facts
            item["candidate_support"] = "ORIGINAL_DIRECT_CANDIDATE"
            item["cause_ids"] = sorted({fact["cause_id"] for fact in facts if fact["active_before"]})
            candidates_by_node[node_id] = item
            direct_added = True

        engine.diagnosis_history.append({
            "node_id": node_id,
            "step_index": current_node.get("step_index"),
            "agent_id": str(current_node.get("agent_id") or current_node.get("name") or current_node.get("role") or "Unknown"),
            "local": local,
            "cascade": cascade,
            "active_causes_before": [state.cause.model_dump() for state in active],
            "active_causes_after": [state.cause.model_dump() for state in after if state.active],
            "branch_states_before": [state.audit() for state in before],
            "branch_states_after": [state.audit() for state in after],
            "cause_records": facts,
            "candidate_audit": {
                "direct_candidate_added": direct_added,
                "candidate_count_after_node": len(candidates_by_node),
                "upstream_candidates_added": [],
                "graph_producer_expansion_enabled": False,
            },
        })

        for predecessor_id, _kind in predecessors:
            if predecessor_id in incoming:
                incoming[predecessor_id].extend(_copy_state(state) for state in after)

    candidates = list(candidates_by_node.values())
    error_chain[:] = [
        {
            "node_id": item["node_id"],
            "agent_id": item["agent_id"],
            "step_index": item["step_index"],
            "error_role": item["error_role"],
            "candidate_role": item["candidate_role"],
            "is_root_candidate": item["is_root_candidate"],
            "is_soft_candidate": item["is_soft_candidate"],
            "candidate_source": item["candidate_source"],
            "candidate_support": item.get("candidate_support"),
            "cause_ids": item.get("cause_ids", []),
            "error_reason": item["error_reason"],
        }
        for item in sorted(candidates, key=lambda item: (item.get("step_index") or 0, item["node_id"]))
    ]
    coverage = {
        "reachable_nodes": diagnostics["reachable_node_count"],
        "locally_inspected_nodes": len(traversal_path),
        "candidate_nodes": len(candidates),
        "node_cause_path_records": sum(len(item.get("cause_records") or []) for item in candidates),
        "compressed_branch_state_records": len(engine.branch_provenance_records),
        "represented_path_count_at_initial_node": sum(
            state.path_count for state in compress_incoming(
                incoming[order[-1]], order[-1]
            )
        ) if order else 0,
        "unreachable_node_ids": diagnostics["unreachable_node_ids"],
    }
    return engine._finalize_stage1_candidate_pipeline(
        candidates_by_node=candidates_by_node,
        error_chain=error_chain,
        coverage=coverage,
        fallback_node_id=engine.final_evidence_node_id or start_node_id,
        diagnosis_mode="LOCAL_PLUS_CASCADE",
    )


def diagnose_local_only_dag(
    engine: Any,
    start_node_id: str,
    traversal_path: List[str],
    error_chain: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], List[FailureCause], List[Dict[str, Any]]]:
    """Run the same validated downstream-first schedule without cascade calls."""
    from .diagnoser import _combined_canonical_agent_id as canonical_agent_id

    order, diagnostics = downstream_first_order(
        engine.nodes_map, engine.reverse_adj, engine.forward_adj, start_node_id
    )
    engine.graph_diagnostics = diagnostics
    candidates_by_node: Dict[str, Dict[str, Any]] = {}

    for node_id in order:
        current_node = engine.nodes_map[node_id]
        predecessors = engine._get_sorted_predecessors(node_id)
        predecessor_nodes = [engine.nodes_map[src_id] for src_id, _kind in predecessors]
        prompt_predecessors = engine._select_prompt_predecessors(predecessor_nodes)
        successors = engine._get_sorted_successors(node_id)
        successor_nodes = [engine.nodes_map[dst_id] for dst_id, _kind in successors]

        local = engine._inspect_current_node_local(
            current_node=current_node,
            predecessor_nodes=prompt_predecessors,
            successor_nodes=successor_nodes,
        )
        engine.visited.add(node_id)
        traversal_path.append(node_id)

        if local.get("is_error"):
            error_chain.append({
                "node_id": node_id,
                "role": current_node.get("role"),
                "name": current_node.get("name"),
                "agent_id": canonical_agent_id(current_node),
                "step_index": current_node.get("step_index"),
                "is_error": True,
                "error_reason": local.get("reason", ""),
            })

        candidate_role = str(local.get("candidate_role") or "NO_RELEVANT_EVIDENCE").upper()
        error_role = str(local.get("error_role") or "UNCERTAIN").upper()
        eligible = bool(
            local.get("is_error")
            or local.get("is_root_candidate")
            or candidate_role != "NO_RELEVANT_EVIDENCE"
            or node_id == engine.final_evidence_node_id
        )
        direct_added = False
        if eligible:
            if candidate_role == "NO_RELEVANT_EVIDENCE":
                candidate_role = "EXPOSURE_EVIDENCE"
            if (
                bool(local.get("is_error"))
                and error_role == "INTRODUCES_ERROR"
                and exact_local_evidence
            ):
                support = "EVIDENCE_SUPPORTED"
            elif bool(local.get("is_error")):
                support = "LOCAL_ONLY"
            else:
                support = "SOFT"
            item = engine._make_candidate_item(
                node_id=node_id,
                node=current_node,
                local_eval=local,
                cascade_eval=None,
                candidate_role=candidate_role,
                candidate_source="LOCAL_ONLY_LOCAL_EVIDENCE",
                error_role=error_role,
                error_reason=str(local.get("reason") or ""),
                is_root_candidate=bool(local.get("is_root_candidate")),
                is_soft_candidate=True,
            )
            exact_local_evidence = any(
                isinstance(error, dict)
                and error.get("evidence_match_status") == "EXACT"
                for error in local.get("errors") or []
            )

            item["cause_records"] = []
            item["cause_ids"] = []
            item["candidate_support"] = support
            candidates_by_node[node_id] = item
            direct_added = True

        engine.diagnosis_history.append({
            "node_id": node_id,
            "step_index": current_node.get("step_index"),
            "agent_id": canonical_agent_id(current_node),
            "local": local,
            "cascade": None,
            "active_causes_before": [],
            "active_causes_after": [],
            "branch_states_before": [],
            "branch_states_after": [],
            "cause_records": [],
            "candidate_audit": {
                "direct_candidate_added": direct_added,
                "candidate_count_after_node": len(candidates_by_node),
                "graph_producer_expansion_enabled": True,
                "producer_expansion_eligible": False,
                "producer_candidates_considered": [],
                "producer_candidates_newly_added": [],
                "existing_candidates_merged": [],
                "producer_candidates_skipped": [],
                "producer_paths": [],
            },
        })

    error_chain.sort(key=lambda item: (item.get("step_index") or 0, item["node_id"]))
    coverage = {
        "reachable_nodes": diagnostics["reachable_node_count"],
        "locally_inspected_nodes": len(traversal_path),
        "candidate_nodes": len(candidates_by_node),
        "node_cause_path_records": 0,
        "compressed_branch_state_records": 0,
        "represented_path_count_at_initial_node": 0,
        "unreachable_node_ids": diagnostics["unreachable_node_ids"],
        "diagnosis_mode": "LOCAL_ONLY_WITH_STAGE2",
    }
    return engine._finalize_stage1_candidate_pipeline(
        candidates_by_node=candidates_by_node,
        error_chain=error_chain,
        coverage=coverage,
        fallback_node_id=engine.final_evidence_node_id or start_node_id,
        diagnosis_mode="LOCAL_ONLY_WITH_STAGE2",
    )

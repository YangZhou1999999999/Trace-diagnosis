"""Branch-preserving causal states for Who&When backward diagnosis.

Confidence fields are model judgments, not Bayesian posterior probabilities. A
supported path and a refuted path remain two facts; neither overwrites the other.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .schemas import FailureCause


@dataclass(frozen=True)
class BranchState:
    cause: FailureCause
    path: Tuple[str, ...] = ()
    state_id: str = ""
    origin_state_ids: Tuple[str, ...] = ()
    path_count: int = 1
    active: bool = True
    stop_reason: Optional[str] = None
    relation_confidence: Optional[float] = None
    relation_history: Tuple[Dict[str, Any], ...] = ()

    def audit(self) -> Dict[str, Any]:
        return {
            **self.cause.model_dump(),
            "path": list(self.path),
            "state_id": self.state_id,
            "origin_state_ids": list(self.origin_state_ids),
            "path_count": self.path_count,
            "active": self.active,
            "stop_reason": self.stop_reason,
            "relation_confidence": self.relation_confidence,
            "relation_history": list(self.relation_history),
        }


def seed_branches(causes: Iterable[FailureCause]) -> List[BranchState]:
    return [
        BranchState(cause=FailureCause(**cause.model_dump()), state_id=f"seed:{cause.cause_id}")
        for cause in causes
    ]


def compress_incoming(states: Iterable[BranchState], node_id: str) -> List[BranchState]:
    """Merge equivalent states only within the same immediate downstream edge.

    The original child state IDs and exact count of represented concrete paths
    remain in the provenance DAG. Supported and refuted paths never coalesce.
    """
    groups: Dict[Tuple[Any, ...], List[BranchState]] = {}
    for state in states:
        cause = state.cause
        downstream = state.path[0] if state.path else "<seed>"
        signature = (
            downstream, cause.cause_id, state.active, state.stop_reason,
            cause.support_status, cause.causal_status, cause.localized_node_id,
            cause.root_node_id, cause.evidence, cause.should_propagate_upstream,
        )
        groups.setdefault(signature, []).append(state)

    compressed = []
    for index, signature in enumerate(sorted(groups, key=lambda key: tuple(str(value) for value in key))):
        members = sorted(groups[signature], key=lambda state: (state.path, state.state_id))
        exemplar = members[0]
        compressed.append(replace(
            exemplar,
            state_id=f"{node_id}:in:{index}",
            origin_state_ids=tuple(sorted({state.state_id for state in members if state.state_id})),
            path_count=sum(state.path_count for state in members),
        ))
    return compressed


def advance_branch(
    state: BranchState,
    *,
    node_id: str,
    relation: str = "UNCERTAIN",
    explanation: str = "",
    relation_confidence: Optional[float] = None,
    evidence: Optional[str] = None,
    root_candidate: bool = False,
) -> BranchState:
    """Apply the original Who&When Cause transition to one DAG branch.

    The original Notebook continues upstream for RECOVERS, UNRELATED and
    UNCERTAIN; only REFUTES stops a live cause. Relation confidence is folded
    into Cause confidence as in the original implementation. A root-candidate
    promotion is additionally guarded so it can only occur on INTRODUCES.
    """
    path = (node_id,) + state.path
    if not state.active:
        return replace(
            state,
            path=path,
            state_id="",
            origin_state_ids=(state.state_id,),
        )

    cause = FailureCause(**state.cause.model_dump())
    relation = str(relation or "UNCERTAIN").upper()
    allowed_relations = {
        "INTRODUCES", "PROPAGATES", "AMPLIFIES", "RECOVERS",
        "UNRELATED", "REFUTES", "UNCERTAIN",
    }
    if relation not in allowed_relations:
        relation = "UNCERTAIN"

    try:
        confidence = (
            float(relation_confidence)
            if relation_confidence is not None
            else 0.5
        )
    except (TypeError, ValueError):
        confidence = 0.5
    if confidence != confidence or confidence in (float("inf"), float("-inf")):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    # Retain the original Notebook's confidence update. The hypothesis
    # confidence is therefore the maximum confidence observed along this path.
    cause.confidence = max(float(cause.confidence), confidence)

    stop_reason = None
    if relation in {"INTRODUCES", "PROPAGATES", "AMPLIFIES"}:
        cause.support_status = "SUPPORTED"
        cause.should_propagate_upstream = True

        if relation == "INTRODUCES":
            cause.causal_status = "PROXIMATE_CAUSE"
        elif cause.causal_status == "UNKNOWN":
            cause.causal_status = "PROXIMATE_CAUSE"

        if cause.localized_node_id is None:
            cause.localized_node_id = node_id
        if evidence and not cause.evidence:
            cause.evidence = evidence

    elif relation == "RECOVERS":
        # Original Notebook behavior: this is evidence about the cause, but
        # it does not terminate the backward search for its origin.
        cause.support_status = "SUPPORTED"
        cause.should_propagate_upstream = True

    elif relation == "REFUTES":
        cause.support_status = "REFUTED"
        cause.should_propagate_upstream = False
        stop_reason = "REFUTES"

    elif relation == "UNRELATED":
        if cause.support_status != "SUPPORTED":
            cause.support_status = "HYPOTHESIS"
        # Original Notebook behavior: continue searching upstream.
        cause.should_propagate_upstream = True

    else:  # UNCERTAIN: preserve support state and continue upstream.
        cause.should_propagate_upstream = True

    # A model cannot promote a cause on PROPAGATES/RECOVERS/etc. even if it
    # incorrectly sets is_root_candidate=true in its response.
    promoted_root_candidate = bool(
        root_candidate
        and relation == "INTRODUCES"
        and cause.support_status != "REFUTED"
    )
    if promoted_root_candidate:
        cause.causal_status = "ROOT_CANDIDATE"
        cause.support_status = "SUPPORTED"
        cause.should_propagate_upstream = True

    # This is the original propagation rule: all relations except REFUTES
    # remain active. Keep the explicit guard as a final invariant.
    if relation != "REFUTES":
        cause.should_propagate_upstream = True

    active = bool(
        cause.should_propagate_upstream
        and cause.support_status != "REFUTED"
    )
    if not active and stop_reason is None:
        stop_reason = "REFUTES"

    history = state.relation_history + ({
        "node_id": node_id,
        "cause_id": cause.cause_id,
        "relation": relation,
        "explanation": explanation,
        "relation_confidence": confidence,
        "evidence": evidence,
        "is_root_candidate": promoted_root_candidate,
        "support_status_after": cause.support_status,
        "active_after": active,
    },)
    return BranchState(
        cause=cause,
        path=path,
        state_id="",
        origin_state_ids=(state.state_id,),
        path_count=state.path_count,
        active=active,
        stop_reason=stop_reason,
        relation_confidence=confidence,
        relation_history=history,
    )


def summarize_branches(states: Iterable[BranchState]) -> List[Dict[str, Any]]:
    """Return a conservative, deterministic per-cause view for Stage 2 audit."""
    grouped: Dict[str, List[BranchState]] = {}
    for state in states:
        grouped.setdefault(state.cause.cause_id, []).append(state)

    summaries = []
    for cause_id in sorted(grouped):
        branches = sorted(grouped[cause_id], key=lambda state: state.path)
        statuses = {state.cause.support_status for state in branches}
        if "SUPPORTED" in statuses and "REFUTED" in statuses:
            aggregate = "CONFLICTED"
        elif "SUPPORTED" in statuses:
            aggregate = "SUPPORTED"
        elif "REFUTED" in statuses:
            aggregate = "REFUTED"
        else:
            aggregate = "HYPOTHESIS"
        summaries.append({
            "cause_id": cause_id,
            "support_summary": aggregate,
            "branch_count": len(branches),
            "active_branch_count": sum(state.active for state in branches),
            "supported_branch_count": sum(state.cause.support_status == "SUPPORTED" for state in branches),
            "refuted_branch_count": sum(state.cause.support_status == "REFUTED" for state in branches),
            "stopped_branch_count": sum(not state.active for state in branches),
            "paths": [list(state.path) for state in branches],
            "represented_path_count": sum(state.path_count for state in branches),
        })
    return summaries


def branch_prompt_summary(states: Iterable[BranchState]) -> Dict[str, Any]:
    """Bounded prompt view; full provenance stays in checkpoint audit."""
    branches = list(states)
    summaries = [
        {key: value for key, value in summary.items() if key != "paths"}
        for summary in summarize_branches(branches)
    ]
    representative_by_kind: Dict[Tuple[str, str, bool, Optional[str]], BranchState] = {}
    for state in sorted(branches, key=lambda item: (item.cause.cause_id, item.path)):
        kind = (
            state.cause.cause_id,
            state.cause.support_status,
            state.active,
            state.stop_reason,
        )
        representative_by_kind.setdefault(kind, state)
    representatives = [
        {
            "cause_id": state.cause.cause_id,
            "support_status": state.cause.support_status,
            "active": state.active,
            "stop_reason": state.stop_reason,
            "downstream_node_id": state.path[0] if state.path else None,
            "representative_path": list(state.path),
            "hypothesis_confidence": state.cause.confidence,
            "latest_relation_confidence": state.relation_confidence,
        }
        for state in representative_by_kind.values()
    ]
    return {"cause_summaries": summaries, "state_representatives": representatives}


def compatible_causes(
    states: Iterable[BranchState], initial: Iterable[FailureCause]
) -> List[FailureCause]:
    """Legacy per-node projection; conflicts remain HYPOTHESIS, with detail in audit."""
    branches = list(states)
    seed_by_id = {cause.cause_id: cause for cause in initial}
    projected = []
    for summary in summarize_branches(branches):
        cause_id = summary["cause_id"]
        seed = seed_by_id.get(cause_id)
        if seed is None:
            continue
        cause = FailureCause(**seed.model_dump())
        status = summary["support_summary"]
        cause.support_status = status if status in {"SUPPORTED", "REFUTED"} else "HYPOTHESIS"
        cause.causal_status = "PROXIMATE_CAUSE" if status == "SUPPORTED" else "UNKNOWN"
        cause.should_propagate_upstream = bool(summary["active_branch_count"])
        projected.append(cause)
    return projected


def cause_record_at_node(
    before: BranchState, after: BranchState, node_id: str
) -> Dict[str, Any]:
    """A Stage 1 node x cause x path fact, including stopped paths."""
    latest = after.relation_history[-1] if after.relation_history else {}
    evaluated_here = latest.get("node_id") == node_id and before.active
    return {
        "node_id": node_id,
        "cause_id": after.cause.cause_id,
        "path": list(after.path),
        "downstream_path": list(before.path),
        "state_id": after.state_id,
        "origin_state_ids": list(after.origin_state_ids),
        "path_count": after.path_count,
        "relation": latest.get("relation") if evaluated_here else "NOT_EVALUATED_STOPPED",
        "explanation": latest.get("explanation") if evaluated_here else None,
        "evidence": latest.get("evidence") if evaluated_here else None,
        "relation_confidence": latest.get("relation_confidence") if evaluated_here else None,
        "is_root_candidate": bool(latest.get("is_root_candidate")) if evaluated_here else False,
        "support_status": after.cause.support_status,
        "active_before": before.active,
        "active_after": after.active,
        "stop_reason": after.stop_reason,
    }

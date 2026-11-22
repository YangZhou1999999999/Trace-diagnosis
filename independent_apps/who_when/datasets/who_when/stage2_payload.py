"""Small, deterministic Stage 2 candidate-coverage accounting helpers.

Coverage counts candidate records in the ranked Stage 1 pool. A node can
appear more than once when its per-cause records have not been consolidated;
multiset matching prevents one presented node ID from claiming all copies.
"""

from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence


def _candidate_node_id(item: Mapping[str, Any], rank: int) -> str:
    value = item.get("node_id")
    if value in (None, ""):
        node = item.get("node")
        if isinstance(node, Mapping):
            value = node.get("node_id")
    if value in (None, ""):
        return f"<missing-node-id:{rank}>"
    return str(value)


def build_candidate_coverage(
    candidates: Sequence[Mapping[str, Any]],
    presented_ids: Sequence[str],
    max_candidates: int,
) -> Dict[str, Any]:
    """Audit how much of the ranked pool was actually shown to one selector.

    Raises if the caller claims that a nonexistent or over-limit candidate was
    presented. The result never implies that omitted candidates were compared.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    all_ids: List[str] = [
        _candidate_node_id(item, rank)
        for rank, item in enumerate(candidates, start=1)
    ]
    shown_ids = [str(node_id) for node_id in presented_ids]
    if len(shown_ids) > max_candidates:
        raise ValueError("presented candidate count exceeds configured cap")

    remaining = Counter(shown_ids)
    omitted_ids: List[str] = []
    for node_id in all_ids:
        if remaining[node_id]:
            remaining[node_id] -= 1
        else:
            omitted_ids.append(node_id)
    if any(remaining.values()):
        raise ValueError("presented candidate IDs are not in ranked Stage 1 pool")

    total = len(all_ids)
    presented = len(shown_ids)
    omitted = len(omitted_ids)
    return {
        "total": total,
        "presented": presented,
        "omitted": omitted,
        "policy": "CAPPED",
        "coverage_ratio": presented / total if total else 1.0,
        "presented_node_ids": shown_ids,
        "omitted_node_ids": omitted_ids,
    }


def select_cause_records_for_prompt(
    records: Sequence[Mapping[str, Any]], max_records: int = 16
) -> tuple[List[Mapping[str, Any]], Dict[str, Any]]:
    """Bound each candidate's path facts while showing every cause if possible."""
    if max_records < 1:
        raise ValueError("max_records must be positive")
    values = list(records)
    if len(values) <= max_records:
        selected = values
    else:
        def priority(index: int) -> tuple[int, str, int]:
            item = values[index]
            if item.get("relation") == "INTRODUCES" and item.get("support_status") == "SUPPORTED":
                rank = 0
            elif item.get("relation") == "REFUTES" or item.get("support_status") == "REFUTED":
                rank = 1
            else:
                rank = 2
            return rank, str(item.get("cause_id") or ""), index

        chosen = set()
        for cause_id in sorted({str(item.get("cause_id") or "") for item in values}):
            options = [index for index, item in enumerate(values) if str(item.get("cause_id") or "") == cause_id]
            if options and len(chosen) < max_records:
                chosen.add(min(options, key=priority))
        for index in sorted(range(len(values)), key=priority):
            if len(chosen) >= max_records:
                break
            chosen.add(index)
        selected = [values[index] for index in sorted(chosen)]
    coverage = {
        "total": len(values),
        "presented": len(selected),
        "omitted": len(values) - len(selected),
        "policy": "CAPPED_REPRESENTATIVE" if len(values) > max_records else "ALL",
        "max_records_per_candidate": max_records,
        "all_cause_ids": sorted({str(item.get("cause_id") or "") for item in values}),
        "presented_cause_ids": sorted({str(item.get("cause_id") or "") for item in selected}),
    }
    return selected, coverage

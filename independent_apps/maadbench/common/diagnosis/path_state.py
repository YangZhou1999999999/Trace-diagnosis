"""Shared bounded cause-path state transitions extracted from TRAIL H1.

The functions are dataset-neutral state mechanics; edge and evidence policy stay
in adapters.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

def make_path_id(previous_path_id: str, src: str, dst: str, edge_kind: str) -> str:
    material = "\0".join((str(previous_path_id), str(src), str(dst), str(edge_kind)))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def reconcile_cause_path_states(
    incoming_states: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
    stopped_cause_ids: Sequence[str] = (),
    *,
    node_id: str | None = None,
    localized_field: str = "localized_span_id",
    root_field: str = "root_span_id",
    relation_node_field: str = "span_id",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply H1 outputs to exactly one cause × incoming path at a time.

    Missing or ambiguous path IDs are not broadcast. The state is preserved as
    uncertain and an audit record is returned for the caller.
    """
    active = [
        {"incoming_path_id": str(item.get("incoming_path_id", "")),
         "cause": dict(item.get("cause") or {})}
        for item in incoming_states
        if isinstance(item, Mapping) and isinstance(item.get("cause"), Mapping)
    ]
    path_counts: Dict[str, int] = {}
    for item in active:
        cause_id = str(item["cause"].get("cause_id", ""))
        path_counts[cause_id] = path_counts.get(cause_id, 0) + 1

    active_keys = {
        (str(item["cause"].get("cause_id", "")), item["incoming_path_id"])
        for item in active
    }
    active_path_ids = {item["incoming_path_id"] for item in active}
    descriptions_by_id: Dict[str, set] = {}
    for item in active:
        cause = item["cause"]
        descriptions_by_id.setdefault(str(cause.get("cause_id", "")), set()).add(
            re.sub(r"\s+", " ", str(cause.get("description", "")).strip()).casefold()
        )

    relations_by_key: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    updates_by_key: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    pending_new_updates: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    audit: List[Dict[str, Any]] = []
    for item in updates:
        if not isinstance(item, Mapping):
            continue
        nested_cause = item.get("cause")
        cause_id = str(
            item.get("cause_id")
            or (nested_cause.get("cause_id") if isinstance(nested_cause, Mapping) else "")
            or ""
        )
        path_id = str(item.get("incoming_path_id", ""))
        if not path_id and path_counts.get(cause_id) == 1:
            path_id = next(
                (x["incoming_path_id"] for x in active
                 if str(x["cause"].get("cause_id", "")) == cause_id),
                "",
            )
        key = (cause_id, path_id)
        if key in active_keys:
            if key in updates_by_key:
                audit.append({"kind": "update", "cause_id": cause_id,
                              "incoming_path_id": path_id, "status": "DUPLICATE_UPDATE_KEY"})
                continue
            updates_by_key[key] = item
        elif path_id in active_path_ids and isinstance(nested_cause, Mapping):
            if key in pending_new_updates:
                audit.append({"kind": "update", "cause_id": cause_id,
                              "incoming_path_id": path_id, "status": "DUPLICATE_NEW_CAUSE_KEY"})
                continue
            pending_new_updates[key] = item
        else:
            audit.append({"kind": "update", "cause_id": cause_id,
                          "incoming_path_id": path_id or None,
                          "status": "IGNORED_UNMATCHED_PATH"})

    for item in relations:
        if not isinstance(item, Mapping):
            continue
        cause_id = str(item.get("cause_id", ""))
        path_id = str(item.get("incoming_path_id", ""))
        key = (cause_id, path_id)
        if key in active_keys:
            if key in relations_by_key:
                audit.append({"kind": "relation", "cause_id": cause_id,
                              "incoming_path_id": path_id, "status": "DUPLICATE_RELATION_KEY"})
                continue
            relations_by_key[key] = item
            continue

        pending_update = pending_new_updates.get(key)
        if path_id not in active_path_ids or pending_update is None:
            audit.append({"kind": "relation", "cause_id": cause_id,
                          "incoming_path_id": path_id or None,
                          "status": "IGNORED_UNMATCHED_PATH"})
            continue
        if str(item.get("relation", "")).upper() != "INTRODUCES":
            audit.append({"kind": "relation", "cause_id": cause_id,
                          "incoming_path_id": path_id,
                          "status": "NEW_CAUSE_REQUIRES_INTRODUCES"})
            continue
        new_cause = pending_update.get("cause")
        description = re.sub(
            r"\s+", " ", str(new_cause.get("description", "")).strip()
        ).casefold()
        known_descriptions = descriptions_by_id.get(cause_id, set())
        if known_descriptions and description not in known_descriptions:
            audit.append({"kind": "update", "cause_id": cause_id,
                          "incoming_path_id": path_id,
                          "status": "CAUSE_ID_SEMANTIC_CONFLICT"})
            continue
        active.append({"incoming_path_id": path_id, "cause": dict(new_cause)})
        active_keys.add(key)
        path_counts[cause_id] = path_counts.get(cause_id, 0) + 1
        descriptions_by_id.setdefault(cause_id, set()).add(description)
        updates_by_key[key] = pending_update
        relations_by_key[key] = item
        audit.append({"kind": "update", "cause_id": cause_id,
                      "incoming_path_id": path_id,
                      "status": "NEW_CAUSE_ACCEPTED_PENDING_EVIDENCE_CHECK"})

    for key in pending_new_updates:
        if key not in active_keys:
            audit.append({"kind": "update", "cause_id": key[0],
                          "incoming_path_id": key[1],
                          "status": "NEW_CAUSE_WITHOUT_ACCEPTED_INTRODUCES_RELATION"})

    result: List[Dict[str, Any]] = []
    stop_ids = {str(value) for value in stopped_cause_ids}
    stop_relations = {"REFUTES", "RECOVERS", "UNRELATED"}
    for state in active:
        cause = state["cause"]
        cause_id = str(cause.get("cause_id", ""))
        path_id = state["incoming_path_id"]
        relation_item = relations_by_key.get((cause_id, path_id))
        update_item = updates_by_key.get((cause_id, path_id))
        if update_item is not None:
            updated = update_item.get("cause")
            if isinstance(updated, Mapping):
                # Keep the state identity and the earliest known localization.
                replacement = dict(updated)
                replacement["cause_id"] = cause_id
                previous_description = str(cause.get("description", ""))
                updated_description = str(replacement.get("description", ""))
                if (
                    previous_description.strip()
                    and updated_description.strip()
                    and re.sub(r"\s+", " ", previous_description.strip()).casefold()
                    != re.sub(r"\s+", " ", updated_description.strip()).casefold()
                ):
                    audit.append({
                        "kind": "update",
                        "cause_id": cause_id,
                        "incoming_path_id": path_id,
                        "status": "CAUSE_DESCRIPTION_CHANGED_SAME_ID",
                        "reason": "preserve_original_description; a materially new mechanism needs a new cause_id",
                    })
                    replacement["description"] = previous_description
                if cause.get(localized_field) is not None:
                    replacement[localized_field] = cause[localized_field]
                if cause.get(root_field) is not None:
                    replacement[root_field] = cause[root_field]
                cause = replacement

        relation = str((relation_item or {}).get("relation", "UNCERTAIN")).upper()
        cause["cause_id"] = cause_id
        claimed_root = bool((relation_item or {}).get("is_root_cause"))
        current_span_id = str((relation_item or {}).get(relation_node_field) or node_id or "")
        if cause.get("causal_status") == "ROOT_CAUSE" and not claimed_root:
            audit.append({
                "kind": "root_cause", "cause_id": cause_id,
                "incoming_path_id": path_id,
                "status": "ROOT_CAUSE_REJECTED_WITHOUT_PATH_RELATION",
            })
            cause["causal_status"] = "UNKNOWN"
            cause[root_field] = None
            if cause.get("support_status") != "REFUTED":
                cause["should_propagate_upstream"] = True
        root_evidence_valid = bool(
            claimed_root
            and relation == "INTRODUCES"
            and cause.get("support_status") == "SUPPORTED"
            and str(cause.get(localized_field) or "") == current_span_id
            and isinstance(cause.get("evidence"), str)
            and bool(str(cause.get("evidence")).strip())
        )
        if relation == "REFUTES":
            cause["support_status"] = "REFUTED"
            if cause.get("causal_status") == "ROOT_CAUSE":
                cause["causal_status"] = "UNKNOWN"
                cause[root_field] = None
            if claimed_root:
                audit.append({
                    "kind": "root_cause", "cause_id": cause_id,
                    "incoming_path_id": path_id,
                    "status": "ROOT_CAUSE_REJECTED_REFUTED_PATH",
                })
            cause["should_propagate_upstream"] = False
        elif relation in stop_relations:
            # RECOVERS and UNRELATED stop only this path and do not erase the
            # cause's support record or states arriving through other paths.
            cause["should_propagate_upstream"] = False
        elif claimed_root and root_evidence_valid:
            cause["support_status"] = "SUPPORTED"
            cause["causal_status"] = "ROOT_CAUSE"
            cause[root_field] = current_span_id
            cause["should_propagate_upstream"] = False
        elif claimed_root:
            audit.append({
                "kind": "root_cause", "cause_id": cause_id,
                "incoming_path_id": path_id,
                "status": "ROOT_CAUSE_REJECTED_UNSUPPORTED",
                "reason": "requires_INTRODUCES_SUPPORTED_exact_span_evidence_and_matching_localized_span",
            })
            if cause.get("causal_status") == "ROOT_CAUSE":
                cause["causal_status"] = "UNKNOWN"
                cause[root_field] = None
            if cause.get("support_status") != "REFUTED":
                cause["should_propagate_upstream"] = True
        elif cause_id in stop_ids and path_counts.get(cause_id) == 1:
            cause["should_propagate_upstream"] = False
        result.append({"incoming_path_id": path_id, "cause": cause})
    for cause_id in sorted(stop_ids):
        if path_counts.get(cause_id, 0) > 1:
            audit.append({
                "kind": "stopped_cause_id",
                "cause_id": cause_id,
                "status": "STOP_ID_AMBIGUOUS_IGNORED_USE_PATH_RELATIONS",
                "path_count": path_counts[cause_id],
            })
    return result, audit


def prune_cause_path_states(
    states: Sequence[Mapping[str, Any]], max_causes: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Limit distinct active causes without treating model confidence as probability."""
    priority = {"SUPPORTED": 0, "HYPOTHESIS": 1, "REFUTED": 2}
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for item in states:
        if not isinstance(item, Mapping) or not isinstance(item.get("cause"), Mapping):
            continue
        copied = {"incoming_path_id": str(item.get("incoming_path_id", "")),
                  "cause": dict(item["cause"])}
        cause_id = str(copied["cause"].get("cause_id", ""))
        by_id.setdefault(cause_id, []).append(copied)
    ordered_ids = sorted(
        by_id,
        key=lambda cause_id: (
            min(priority.get(str(x["cause"].get("support_status", "HYPOTHESIS")), 3)
                for x in by_id[cause_id]),
            cause_id,
        ),
    )
    keep_ids = set(ordered_ids[:max(0, int(max_causes))])
    kept = [item for cause_id in ordered_ids if cause_id in keep_ids for item in by_id[cause_id]]
    pruned = [
        {"cause_id": cause_id, "reason": "max_causes_exceeded",
         "importance_order": rank + 1,
         "support_statuses": sorted({str(x["cause"].get("support_status", "HYPOTHESIS"))
                                     for x in by_id[cause_id]})}
        for rank, cause_id in enumerate(ordered_ids) if cause_id not in keep_ids
    ]
    return kept, pruned

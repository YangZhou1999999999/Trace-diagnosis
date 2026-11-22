"""Cause-state compatibility policies used by the shared diagnosis engine.

Policies preserve a dataset's frozen legacy semantics while the engine owns
the executable traversal, invocation, transition and history loop.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Literal, Mapping, Sequence

from ..cascade import normalize_cascade_relation
from .contracts import _field, _plain
from .path_state import make_path_id, prune_cause_path_states, reconcile_cause_path_states


def _as_dict(value: Any) -> dict[str, Any]:
    data = _plain(value)
    return dict(data) if isinstance(data, Mapping) else {}


def _text_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _support_rank(value: Any) -> int:
    return {"HYPOTHESIS": 0, "REFUTED": -1, "SUPPORTED": 1}.get(str(value), 0)


def _causal_rank(value: Any) -> int:
    return {"UNKNOWN": 0, "PROXIMATE_CAUSE": 1, "ROOT_CAUSE": 2}.get(str(value), 0)


def _normalize_trail_state(raw: Mapping[str, Any], node_id: str | None = None) -> dict[str, Any]:
    cause = dict(raw)
    support = str(cause.get("support_status", "HYPOTHESIS"))
    causal = str(cause.get("causal_status", "UNKNOWN"))
    if support == "REFUTED":
        cause["should_propagate_upstream"] = False
        return cause
    if causal == "ROOT_CAUSE":
        cause["support_status"] = "SUPPORTED"
        cause["should_propagate_upstream"] = False
        if cause.get("root_span_id") is None and node_id is not None:
            cause["root_span_id"] = node_id
    if cause.get("support_status") == "SUPPORTED" and cause.get("localized_span_id") is None and node_id is not None:
        cause["localized_span_id"] = node_id
    if causal == "PROXIMATE_CAUSE":
        cause["support_status"] = "SUPPORTED"
        if cause.get("localized_span_id") is None and node_id is not None:
            cause["localized_span_id"] = node_id
    return cause


class LegacyCauseStatePolicy:
    """Configurable compatibility policy used only during migration replay.

    `merge_identity` and `stop_relations` encode legacy behavior; scheduling,
    state storage, call orchestration and history remain in SharedDiagnosisEngine.
    Dataset adapters may subclass this policy for field aliases, but must not
    implement their own graph loop.
    """

    def __init__(
        self,
        *,
        merge_identity: Literal["cause_id", "cause_id_or_description"] = "cause_id",
        stop_relations: Iterable[str] = ("REFUTES", "RECOVERS", "UNRELATED"),
        root_relation_stops: bool = True,
        allow_new_causes: bool = True,
    ) -> None:
        self.merge_identity = merge_identity
        self.stop_relations = {normalize_cascade_relation(item) for item in stop_relations}
        self.root_relation_stops = root_relation_stops
        self.allow_new_causes = allow_new_causes

    @staticmethod
    def _cause_dict(cause: Any) -> dict[str, Any]:
        data = _plain(cause)
        return dict(data) if isinstance(data, Mapping) else {}

    def seed(self, cause: Any, *, seed_node_id: str) -> dict[str, Any]:
        item = self._cause_dict(cause)
        item.setdefault("cause_id", f"C{seed_node_id}")
        item.setdefault("support_status", "HYPOTHESIS")
        item.setdefault("causal_status", "UNKNOWN")
        item.setdefault("should_propagate_upstream", True)
        return item

    def _identity(self, cause: Mapping[str, Any]) -> tuple[str, str]:
        cause_id = str(cause.get("cause_id") or "")
        if self.merge_identity == "cause_id_or_description" and not cause_id:
            return ("description", str(cause.get("description") or "").strip().casefold())
        return ("cause_id", cause_id)

    def merge_incoming(self, states: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for raw in states:
            cause = self._cause_dict(raw)
            if not cause:
                continue
            key = self._identity(cause)
            if not key[1]:
                continue
            old = merged.get(key)
            if old is None:
                merged[key] = cause
            else:
                old_conf = float(old.get("confidence", 0.0) or 0.0)
                new_conf = float(cause.get("confidence", 0.0) or 0.0)
                if new_conf > old_conf:
                    merged[key] = cause
        return list(merged.values())

    def apply_delta(
        self,
        previous: Sequence[dict[str, Any]],
        cascade: Any,
        *,
        node_id: str,
        max_causes: int,
    ) -> list[dict[str, Any]]:
        by_id = {str(item.get("cause_id")): self._cause_dict(item) for item in previous if item.get("cause_id")}
        stopped = {str(item) for item in (_field(cascade, "stopped_cause_ids", []) or [])}
        relations = _plain(_field(cascade, "cause_relations", []) or [])
        updates = _plain(_field(cascade, "cause_updates", []) or [])
        for relation in relations:
            if not isinstance(relation, Mapping):
                continue
            cause_id = str(relation.get("cause_id") or "")
            name = normalize_cascade_relation(relation.get("relation"))
            if name in self.stop_relations or (self.root_relation_stops and relation.get("is_root_cause")):
                stopped.add(cause_id)
        for update in updates:
            if not isinstance(update, Mapping):
                continue
            cause = self._cause_dict(update)
            cause_id = str(cause.get("cause_id") or "")
            if not cause_id:
                continue
            if cause_id not in by_id and not self.allow_new_causes:
                continue
            old = by_id.get(cause_id, {})
            if old.get("localized_node_id") and not cause.get("localized_node_id"):
                cause["localized_node_id"] = old["localized_node_id"]
            if old.get("localized_span_id") and not cause.get("localized_span_id"):
                cause["localized_span_id"] = old["localized_span_id"]
            if old.get("root_node_id") and not cause.get("root_node_id"):
                cause["root_node_id"] = old["root_node_id"]
            if old.get("root_span_id") and not cause.get("root_span_id"):
                cause["root_span_id"] = old["root_span_id"]
            if cause.get("support_status") == "SUPPORTED" and not (cause.get("localized_node_id") or cause.get("localized_span_id")):
                key = "localized_span_id" if "span_id" in cause or "localized_span_id" in old else "localized_node_id"
                cause[key] = node_id
            if cause.get("causal_status") == "ROOT_CAUSE":
                key = "root_span_id" if "span_id" in cause or "root_span_id" in old else "root_node_id"
                cause[key] = cause.get(key) or node_id
                cause["support_status"] = "SUPPORTED"
                cause["should_propagate_upstream"] = False
                stopped.add(cause_id)
            if cause.get("support_status") == "REFUTED":
                stopped.add(cause_id)
            by_id[cause_id] = cause
        for cause_id in stopped:
            if cause_id in by_id:
                by_id[cause_id]["should_propagate_upstream"] = False
        return list(by_id.values())[:max_causes]

    def propagating(self, causes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for raw in causes:
            cause = self._cause_dict(raw)
            if cause.get("should_propagate_upstream", True) and cause.get("support_status") != "REFUTED":
                result.append(cause)
        return result

class MaadLegacyCauseStatePolicy(LegacyCauseStatePolicy):
    """Exact cause merge rules used by the pre-extraction MAAD diagnosis loop."""

    def __init__(self, *, max_causes: int = 8) -> None:
        super().__init__(merge_identity="cause_id", stop_relations=("REFUTES", "RECOVERS", "UNRELATED"))
        self.max_causes = max(0, int(max_causes))

    def merge_incoming(self, states: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for raw in states:
            cause = self._cause_dict(raw)
            cause_id = str(cause.get("cause_id") or "")
            if not cause_id or not cause.get("should_propagate_upstream", True) or cause.get("support_status") == "REFUTED":
                continue
            old = merged.get(cause_id)
            if old is None or float(cause.get("confidence", 0.0) or 0.0) > float(old.get("confidence", 0.0) or 0.0):
                merged[cause_id] = cause
        return list(merged.values())[:self.max_causes]

    def apply_delta(
        self,
        previous: Sequence[dict[str, Any]],
        cascade: Any,
        *,
        node_id: str,
        max_causes: int,
    ) -> list[dict[str, Any]]:
        by_id = {
            str(item.get("cause_id")): self._cause_dict(item)
            for item in previous if item.get("cause_id")
        }
        stopped = {str(item) for item in (_field(cascade, "stopped_cause_ids", []) or [])}
        relations = _plain(_field(cascade, "cause_relations", []) or [])
        updates = _plain(_field(cascade, "cause_updates", []) or [])
        for relation in relations:
            if not isinstance(relation, Mapping):
                continue
            cause_id = str(relation.get("cause_id") or "")
            relation_name = normalize_cascade_relation(relation.get("relation"))
            if relation_name in {"REFUTES", "RECOVERS", "UNRELATED"}:
                stopped.add(cause_id)
            if relation.get("is_root_cause") and cause_id in by_id:
                cause = by_id[cause_id]
                cause["causal_status"] = "ROOT_CAUSE"
                cause["root_node_id"] = node_id
                cause["support_status"] = "SUPPORTED"
                cause["should_propagate_upstream"] = False
        for raw in updates:
            if not isinstance(raw, Mapping):
                continue
            cause = self._cause_dict(raw)
            cause_id = str(cause.get("cause_id") or "")
            if not cause_id:
                continue
            old = by_id.get(cause_id)
            if old:
                if old.get("localized_node_id"):
                    cause["localized_node_id"] = old["localized_node_id"]
                if old.get("root_node_id") and not cause.get("root_node_id"):
                    cause["root_node_id"] = old["root_node_id"]
                if old.get("support_status") == "SUPPORTED" and cause.get("support_status") == "HYPOTHESIS":
                    cause["support_status"] = "SUPPORTED"
                if old.get("causal_status") == "ROOT_CAUSE" and cause.get("causal_status") != "ROOT_CAUSE":
                    cause["causal_status"] = "ROOT_CAUSE"
            if cause.get("support_status") == "SUPPORTED" and not cause.get("localized_node_id"):
                cause["localized_node_id"] = node_id
            if cause.get("causal_status") == "ROOT_CAUSE":
                cause["root_node_id"] = cause.get("root_node_id") or node_id
                cause["support_status"] = "SUPPORTED"
                cause["should_propagate_upstream"] = False
            if cause.get("support_status") == "REFUTED":
                stopped.add(cause_id)
            by_id[cause_id] = cause
        for cause_id in stopped:
            if cause_id in by_id:
                by_id[cause_id]["should_propagate_upstream"] = False
        return list(by_id.values())[:max_causes]


class TrailLegacyCauseStatePolicy(LegacyCauseStatePolicy):
    """Dictionary port of the pre-migration TRAIL H0 Cause semantics."""

    def __init__(self, *, max_causes: int = 8) -> None:
        super().__init__(merge_identity="cause_id", stop_relations=("REFUTES", "RECOVERS", "UNRELATED"))
        self.max_causes = max(0, int(max_causes))

    @staticmethod
    def _merge_record(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
        old_cause = _normalize_trail_state(old)
        cause = _normalize_trail_state(new)
        if cause.get("support_status") != "REFUTED" and old_cause.get("support_status") != "REFUTED":
            if _support_rank(old_cause.get("support_status")) > _support_rank(cause.get("support_status")):
                cause["support_status"] = old_cause.get("support_status")
            if _causal_rank(old_cause.get("causal_status")) > _causal_rank(cause.get("causal_status")):
                cause["causal_status"] = old_cause.get("causal_status")
        for field in ("localized_span_id", "root_span_id", "evidence"):
            if old_cause.get(field) and (field == "localized_span_id" or not cause.get(field)):
                cause[field] = old_cause[field]
        cause["confidence"] = max(float(old_cause.get("confidence", 0.0) or 0.0), float(cause.get("confidence", 0.0) or 0.0))
        if cause.get("category") is None:
            cause["category"] = old_cause.get("category")
        return _normalize_trail_state(cause)

    def seed(self, cause: Any, *, seed_node_id: str) -> dict[str, Any]:
        item = _normalize_trail_state(_as_dict(cause))
        item.setdefault("cause_id", f"C{seed_node_id}")
        item.setdefault("support_status", "HYPOTHESIS")
        item.setdefault("causal_status", "UNKNOWN")
        item.setdefault("should_propagate_upstream", True)
        return item

    def merge_incoming(self, states: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        by_id: dict[str, int] = {}
        by_text: dict[str, int] = {}
        for raw in states:
            cause = _as_dict(raw)
            cause_id = str(cause.get("cause_id") or "")
            text = _text_key(cause.get("description"))
            index = by_id.get(cause_id) if cause_id else None
            if index is None and text:
                index = by_text.get(text)
            if index is None:
                index = len(merged)
                merged.append(cause)
                if cause_id:
                    by_id[cause_id] = index
                if text:
                    by_text[text] = index
            else:
                merged[index] = self._merge_record(merged[index], cause)
        return [item for item in merged if item.get("support_status") != "REFUTED"][:self.max_causes]

    def apply_delta(
        self, previous: Sequence[dict[str, Any]], cascade: Any, *, node_id: str, max_causes: int,
    ) -> list[dict[str, Any]]:
        existing = [_as_dict(item) for item in previous]
        by_id = {str(item.get("cause_id")): item for item in existing if item.get("cause_id")}
        by_text = {_text_key(item.get("description")): item for item in existing if item.get("description")}
        used_ids = set(by_id)
        relations = _plain(_field(cascade, "cause_relations", []) or [])
        stopped = {str(value) for value in (_field(cascade, "stopped_cause_ids", []) or [])}
        relation_by_id: dict[str, Mapping[str, Any]] = {}
        for relation in relations:
            if not isinstance(relation, Mapping):
                continue
            cause_id = str(relation.get("cause_id") or "")
            relation_by_id[cause_id] = relation

        # Match the Notebook's relation_by_id comprehension: duplicate IDs use
        # the final Relation. Only that effective Relation may stop a Cause;
        # caller-provided stopped_cause_ids remain independent hard stops.
        for cause_id, relation in relation_by_id.items():
            relation_name = normalize_cascade_relation(relation.get("relation"))
            if relation_name in {"REFUTES", "RECOVERS", "UNRELATED"}:
                stopped.add(cause_id)

        updates = _plain(_field(cascade, "cause_updates", []) or [])
        result: list[dict[str, Any]] = []
        result_ids: set[str] = set()
        returned_existing_ids: set[str] = set()
        next_id = 1
        while f"C{next_id}" in used_ids:
            next_id += 1

        for raw in updates:
            if not isinstance(raw, Mapping):
                continue
            cause = _as_dict(raw)
            key = _text_key(cause.get("description"))
            matched = by_id.get(str(cause.get("cause_id") or "")) or (by_text.get(key) if key else None)
            if matched is not None:
                cause["cause_id"] = matched["cause_id"]
                cause_id = str(cause["cause_id"])
                returned_existing_ids.add(cause_id)
                if matched.get("localized_span_id") is not None:
                    cause["localized_span_id"] = matched["localized_span_id"]
                if matched.get("root_span_id") is not None and cause.get("root_span_id") is None:
                    cause["root_span_id"] = matched["root_span_id"]
                if cause.get("evidence") is None:
                    cause["evidence"] = matched.get("evidence")
                if cause.get("category") is None:
                    cause["category"] = matched.get("category")
                if cause.get("support_status") != "REFUTED":
                    if _support_rank(matched.get("support_status")) > _support_rank(cause.get("support_status")):
                        cause["support_status"] = matched.get("support_status")
                    if _causal_rank(matched.get("causal_status")) > _causal_rank(cause.get("causal_status")):
                        cause["causal_status"] = matched.get("causal_status")
            else:
                while f"C{next_id}" in used_ids or f"C{next_id}" in result_ids:
                    next_id += 1
                cause["cause_id"] = f"C{next_id}"
                next_id += 1

            cause_id = str(cause["cause_id"])
            explicitly_stopped = cause_id in stopped
            relation = relation_by_id.get(cause_id)
            if relation is not None:
                if normalize_cascade_relation(relation.get("relation")) in {"REFUTES", "RECOVERS", "UNRELATED"}:
                    continue
                if relation.get("is_root_cause"):
                    cause["support_status"] = "SUPPORTED"
                    cause["causal_status"] = "ROOT_CAUSE"
                    cause["should_propagate_upstream"] = False
            if explicitly_stopped:
                cause["should_propagate_upstream"] = False
            cause = _normalize_trail_state(cause, node_id)
            if explicitly_stopped and cause.get("causal_status") != "ROOT_CAUSE":
                continue
            if cause.get("support_status") == "REFUTED" or cause_id in result_ids:
                continue
            result_ids.add(cause_id)
            result.append(cause)

        for old in existing:
            cause_id = str(old.get("cause_id") or "")
            if cause_id in returned_existing_ids or cause_id in result_ids or cause_id in stopped:
                continue
            relation = relation_by_id.get(cause_id)
            if relation is not None and normalize_cascade_relation(relation.get("relation")) in {"REFUTES", "RECOVERS", "UNRELATED"}:
                continue
            retained = _normalize_trail_state(old)
            if retained.get("support_status") == "REFUTED":
                continue
            result_ids.add(cause_id)
            result.append(retained)
        return result[:max_causes]

    def propagating(self, causes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            _as_dict(item) for item in causes
            if item.get("should_propagate_upstream", True) and item.get("support_status") != "REFUTED"
        ]


class TrailPathCauseStatePolicy:
    """Shared-core H1 policy: retain state by Cause × incoming path."""

    reprocess_new_states = True

    def __init__(self, *, max_path_states: int = 4096, max_causes: int = 8) -> None:
        self.max_path_states = max(0, int(max_path_states))
        self.max_causes = max(0, int(max_causes))
        self.path_nodes: dict[str, tuple[str, ...]] = {"FINAL": ()}
        self.path_state_count = 0
        self._state_keys: set[tuple[str, str]] = set()
        self._audit: list[dict[str, Any]] = []

    @staticmethod
    def identity(state: Mapping[str, Any]) -> tuple[str, str]:
        cause = state.get("cause") if isinstance(state.get("cause"), Mapping) else {}
        return str(cause.get("cause_id", "")), str(state.get("incoming_path_id", ""))

    def seed(self, cause: Any, *, seed_node_id: str) -> dict[str, Any]:
        item = _as_dict(cause)
        item.setdefault("should_propagate_upstream", True)
        if not self.path_nodes["FINAL"]:
            self.path_nodes["FINAL"] = (str(seed_node_id),)
        state = {"incoming_path_id": "FINAL", "cause": item}
        key = self.identity(state)
        if key not in self._state_keys:
            self._state_keys.add(key)
            self.path_state_count += 1
        return state

    def merge_incoming(self, states: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        unique: dict[tuple[str, str], dict[str, Any]] = {}
        for raw in states:
            state = _as_dict(raw)
            cause = state.get("cause")
            if not isinstance(cause, Mapping):
                continue
            key = self.identity(state)
            if key[0] and key[1]:
                unique.setdefault(key, {"incoming_path_id": key[1], "cause": dict(cause)})
        kept, pruned = prune_cause_path_states(list(unique.values()), self.max_causes)
        self._audit.extend({"kind": "prune", **item} for item in pruned)
        return kept

    def apply_delta(
        self, previous: Sequence[dict[str, Any]], cascade: Any, *, node_id: str, max_causes: int,
    ) -> list[dict[str, Any]]:
        relations = _plain(_field(cascade, "cause_relations", []) or [])
        updates = _plain(_field(cascade, "cause_updates", []) or [])
        stopped = _plain(_field(cascade, "stopped_cause_ids", []) or [])
        states, audit = reconcile_cause_path_states(
            previous, relations, updates, stopped, node_id=node_id,
        )
        states, pruned = prune_cause_path_states(states, min(self.max_causes, max_causes))
        self._audit.extend(audit)
        self._audit.extend({"kind": "prune", **item} for item in pruned)
        return states

    def propagating(self, causes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for raw in causes:
            state = _as_dict(raw)
            cause = state.get("cause") if isinstance(state.get("cause"), Mapping) else {}
            if (
                cause.get("support_status") != "REFUTED"
                and cause.get("causal_status") != "ROOT_CAUSE"
                and cause.get("should_propagate_upstream", True)
            ):
                result.append(state)
        return result

    def combine_outgoing(self, old: Sequence[dict[str, Any]], new: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        merged = {self.identity(item): _as_dict(item) for item in old}
        for item in new:
            merged[self.identity(item)] = _as_dict(item)
        return list(merged.values())

    def propagate_to_parent(
        self, causes: Sequence[dict[str, Any]], *, node_id: str, parent_id: str,
        edge: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        result = []
        edge_kind = str(edge.get("kind", edge.get("type", "unknown")))
        for raw in causes:
            state = _as_dict(raw)
            cause = state.get("cause") if isinstance(state.get("cause"), Mapping) else {}
            old_path = str(state.get("incoming_path_id", ""))
            old_nodes = self.path_nodes.get(old_path, (str(node_id),))
            if str(parent_id) in old_nodes:
                self._audit.append({
                    "cause_id": cause.get("cause_id"), "incoming_path_id": old_path,
                    "edge": {"src": parent_id, "dst": node_id, "kind": edge_kind},
                    "status": "CYCLE_PATH_TERMINATED",
                })
                continue
            next_path = make_path_id(old_path, str(node_id), str(parent_id), edge_kind)
            key = (str(cause.get("cause_id", "")), next_path)
            if key in self._state_keys:
                continue
            if self.path_state_count >= self.max_path_states:
                self._audit.append({
                    "cause_id": cause.get("cause_id"), "incoming_path_id": next_path,
                    "status": "PATH_STATE_BUDGET_TRUNCATED", "limit": self.max_path_states,
                })
                continue
            self._state_keys.add(key)
            self.path_state_count += 1
            self.path_nodes[next_path] = tuple(old_nodes) + (str(parent_id),)
            result.append({"incoming_path_id": next_path, "cause": dict(cause)})
        return result

    def consume_audit(self) -> list[dict[str, Any]]:
        audit, self._audit = self._audit, []
        return audit


class MaadPathCauseStatePolicy(TrailPathCauseStatePolicy):
    """Opt-in MAAD Cause×Path policy using MAAD's node-location fields."""

    def __init__(self, *, max_path_states: int = 4096, max_causes: int = 8) -> None:
        super().__init__(max_path_states=max_path_states, max_causes=max_causes)
        self._known_cause_ids: set[str] = set()

    def seed(self, cause: Any, *, seed_node_id: str) -> dict[str, Any]:
        state = super().seed(cause, seed_node_id=seed_node_id)
        cause_data = state.get("cause") if isinstance(state.get("cause"), Mapping) else {}
        cause_id = str(cause_data.get("cause_id", ""))
        if cause_id:
            self._known_cause_ids.add(cause_id)
        return state

    def register_local_seed(self, cause: Any, *, node_id: str) -> dict[str, Any] | None:
        """Register a validated evidence-backed cause with a traceable local path."""
        cause_data = _as_dict(cause)
        cause_id = str(cause_data.get("cause_id", ""))
        if not cause_id or cause_id in self._known_cause_ids:
            self._audit.append({
                "kind": "local_seed", "cause_id": cause_id or None,
                "node_id": str(node_id), "status": "LOCAL_SEED_CAUSE_ID_MISSING_OR_COLLIDING",
            })
            return None
        path_id = f"LOCAL_SEED:{node_id}"
        key = (cause_id, path_id)
        if key not in self._state_keys and self.path_state_count >= self.max_path_states:
            self._audit.append({
                "kind": "local_seed", "cause_id": cause_id,
                "incoming_path_id": path_id, "status": "PATH_STATE_BUDGET_TRUNCATED",
                "limit": self.max_path_states,
            })
            return None
        if key not in self._state_keys:
            self._state_keys.add(key)
            self.path_state_count += 1
        self.path_nodes[path_id] = (str(node_id),)
        self._known_cause_ids.add(cause_id)
        return {"incoming_path_id": path_id, "cause": cause_data}

    def apply_delta(
        self, previous: Sequence[dict[str, Any]], cascade: Any, *, node_id: str, max_causes: int,
    ) -> list[dict[str, Any]]:
        relations = _plain(_field(cascade, "cause_relations", []) or [])
        updates = _plain(_field(cascade, "cause_updates", []) or [])
        stopped = _plain(_field(cascade, "stopped_cause_ids", []) or [])
        states, audit = reconcile_cause_path_states(
            previous, relations, updates, stopped, node_id=node_id,
            localized_field="localized_node_id", root_field="root_node_id",
        )
        states, pruned = prune_cause_path_states(states, min(self.max_causes, max_causes))
        self._known_cause_ids.update(
            str((item.get("cause") or {}).get("cause_id", ""))
            for item in states if isinstance(item.get("cause"), Mapping)
        )
        self._audit.extend(audit)
        self._audit.extend({"kind": "prune", **item} for item in pruned)
        return states

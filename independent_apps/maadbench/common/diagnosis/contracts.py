"""Shared diagnosis data contracts and adapter protocols.

This module contains types shared by the engine and dataset-specific policies.
It deliberately has no dependency on the executable diagnosis loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable, Literal, Mapping, Optional, Protocol, Sequence


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value

def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)

@dataclass
class NodeDiagnosis:
    """One adapter-produced node response before shared Cause state transition."""

    local: Any
    cascade: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class EngineRunResult:
    """Shared engine state returned to a dataset output adapter."""

    schedule: list[str]
    diagnoses: dict[str, NodeDiagnosis]
    incoming_causes: dict[str, list[dict[str, Any]]]
    outgoing_causes: dict[str, list[dict[str, Any]]]
    history: list[dict[str, Any]]
    seed_causes: list[dict[str, Any]]
    scheduler: str
    cycle_nodes: list[str] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)
    graph_validation: dict[str, Any] = field(default_factory=dict)
    node_visit_count: dict[str, int] = field(default_factory=dict)
    processed_cause_path_states: int = 0
    cycle_terminated_paths: int = 0
    path_state_budget: Optional[int] = None
    path_state_budget_exhausted: bool = False
    partial_coverage: bool = False
    blocked_nodes: list[str] = field(default_factory=list)
    propagation_decisions: list[dict[str, Any]] = field(default_factory=list)

@dataclass(frozen=True)
class EdgePolicy:
    """Versioned permissions for traversing, propagating and showing graph edges.

    Traversal is a scheduling permission; propagation only moves a Cause state;
    prompt context is adapter-owned visibility. None means all edge kinds for
    traversal/context, while None for propagation preserves legacy behavior.
    Edge evidence strength is descriptive and never establishes a diagnosis.
    """

    name: str = "legacy_all_edges_v1"
    traversal_kinds: Optional[tuple[str, ...]] = None
    propagation_kinds: Optional[tuple[str, ...]] = None
    prompt_context_kinds: Optional[tuple[str, ...]] = None
    evidence_strength: tuple[tuple[str, str], ...] = (
        ("contains", "hierarchy_only"),
        ("sequential_flow", "temporal_order_only"),
        ("execution_flow", "dependency_candidate_not_proof"),
        ("data_flow", "dependency_candidate_not_proof"),
    )
    known_kinds: tuple[str, ...] = (
        "contains", "sequential_flow", "execution_flow", "data_flow", "dependency", "parent"
    )
    cycle_strategy: Literal["bounded_worklist", "fail_closed"] = "bounded_worklist"

    @classmethod
    def legacy(cls) -> "EdgePolicy":
        return cls()

    @classmethod
    def trail_dependency_only(cls, *, cycle_strategy: str = "bounded_worklist") -> "EdgePolicy":
        return cls(
            name="trail_dependency_only_v1",
            traversal_kinds=None,
            propagation_kinds=("execution_flow", "data_flow", "dependency"),
            prompt_context_kinds=None,
            cycle_strategy=cycle_strategy,  # type: ignore[arg-type]
        )

    @classmethod

    def allows_traversal(self, edge: Mapping[str, Any]) -> bool:
        return self._allows(edge, self.traversal_kinds)

    def allows_propagation(self, edge: Mapping[str, Any]) -> bool:
        return self._allows(edge, self.propagation_kinds)

    def allows_prompt_context(self, edge: Mapping[str, Any]) -> bool:
        return self._allows(edge, self.prompt_context_kinds)

    @staticmethod
    def _allows(edge: Mapping[str, Any], kinds: Optional[tuple[str, ...]]) -> bool:
        return kinds is None or str(edge.get("kind", edge.get("type", "unknown"))).lower() in {
            str(kind).lower() for kind in kinds
        }

    def fingerprint(self) -> str:
        material = {
            "name": self.name,
            "traversal_kinds": self.traversal_kinds,
            "propagation_kinds": self.propagation_kinds,
            "prompt_context_kinds": self.prompt_context_kinds,
            "evidence_strength": self.evidence_strength,
            "known_kinds": self.known_kinds,
            "cycle_strategy": self.cycle_strategy,
        }
        payload = json.dumps(material, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

class GraphContractError(ValueError):
    """The selected graph projection cannot be executed under its policy."""

    def __init__(self, message: str, report: Mapping[str, Any]):
        super().__init__(message)
        self.report = dict(report)

class CauseStatePolicy(Protocol):
    """Small semantic differences for replaying a legacy dataset implementation."""

    def seed(self, cause: Any, *, seed_node_id: str) -> dict[str, Any]: ...

    def merge_incoming(self, states: Sequence[dict[str, Any]]) -> list[dict[str, Any]]: ...

    def apply_delta(
        self,
        previous: Sequence[dict[str, Any]],
        cascade: Any,
        *,
        node_id: str,
        max_causes: int,
    ) -> list[dict[str, Any]]: ...

    def propagating(self, causes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]: ...

class DiagnosisAdapter(Protocol):
    """Dataset-owned prompts and output serialization; not a diagnosis loop."""

    cascade_enabled: bool
    cascade_call_mode: Literal["combined", "separate"]

    def initialize_hypotheses(self, entry_node: dict[str, Any]) -> Sequence[Any]: ...

    def prepare_node(
        self,
        node_id: str,
        node: dict[str, Any],
        incoming_causes: Sequence[dict[str, Any]],
        position: int,
    ) -> Any: ...

    def inspect_local(self, prepared: Any) -> Any: ...

    def inspect_cascade(self, prepared: Any, local: Any) -> Any: ...

    def inspect_combined(self, prepared: Any) -> NodeDiagnosis: ...

    def after_node(
        self,
        node_id: str,
        prepared: Any,
        diagnosis: NodeDiagnosis,
        active_before: Sequence[dict[str, Any]],
        active_after: Sequence[dict[str, Any]],
    ) -> Optional[dict[str, Any]]: ...

class CascadeFirstSeparateAdapter(DiagnosisAdapter, Protocol):
    """Optional adapter extension for Separate Cascade -> Local execution.

    The base protocol intentionally remains compatible with existing adapters.
    The engine requires this extension only when ``separate_order`` explicitly
    selects ``cascade_then_local``.
    """

    separate_order: Literal["cascade_then_local"]

    def inspect_local_with_cascade(self, prepared: Any, cascade: Any) -> Any: ...

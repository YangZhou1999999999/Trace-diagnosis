"""Shared graph diagnosis orchestration for TRAIL and MAADBench."""

from .engine import (
    CascadeFirstSeparateAdapter,
    EdgePolicy,
    EngineRunResult,
    GraphContractError,
    LegacyCauseStatePolicy,
    MaadLegacyCauseStatePolicy,
    NodeDiagnosis,
    SharedDiagnosisEngine,
)
from .policies import MaadPathCauseStatePolicy, TrailLegacyCauseStatePolicy, TrailPathCauseStatePolicy
from .path_state import (
    make_path_id,
    prune_cause_path_states,
    reconcile_cause_path_states,
)

__all__ = [
    "CascadeFirstSeparateAdapter", "EdgePolicy", "EngineRunResult", "GraphContractError", "LegacyCauseStatePolicy", "MaadLegacyCauseStatePolicy",
    "NodeDiagnosis", "SharedDiagnosisEngine",
    "MaadPathCauseStatePolicy", "TrailLegacyCauseStatePolicy", "TrailPathCauseStatePolicy",
    "make_path_id", "prune_cause_path_states", "reconcile_cause_path_states",
]

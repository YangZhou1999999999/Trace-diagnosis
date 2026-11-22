"""Structured Stage 1 response schemas for Who&When diagnosis."""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ...common.cascade import CascadeRelation
from ...common.schemas import NodeLocalInspectionBase


NodeStatus = Literal["CORRECT", "ROOT_CAUSE", "PASSIVE_ERROR", "UNCERTAIN"]
ErrorRole = Literal[
    "INTRODUCES_ERROR",
    "PROPAGATES_ERROR",
    "VERIFIES_PRIOR_ERROR",
    "TERMINATES_AFTER_ERROR",
    "NO_ERROR",
    "UNCERTAIN",
]
CandidateRole = Literal[
    "INTRODUCER_CANDIDATE",
    "PROPAGATION_EVIDENCE",
    "VERIFICATION_EVIDENCE",
    "EXPOSURE_EVIDENCE",
    "UPSTREAM_PRODUCER",
    "NO_RELEVANT_EVIDENCE",
]
SupportStatus = Literal["HYPOTHESIS", "SUPPORTED", "REFUTED"]
CausalStatus = Literal[
    "UNKNOWN",
    "PROXIMATE_CAUSE",
    "ROOT_CANDIDATE",
    "ROOT_CAUSE",
]
FailureAssessment = Literal["MISMATCH", "POTENTIAL_MISMATCH", "UNKNOWN"]


class WhoWhenErrorItem(BaseModel):
    """A concrete local error attributed to the current Who&When node."""

    model_config = ConfigDict(extra="forbid")

    location: Optional[str] = Field(default=None, description="Current node_id.")
    evidence: str = Field(description="Exact supporting substring from this node.")
    description: str = Field(description="Why this evidence is an error.")


class SingleNodeLocalInspection(NodeLocalInspectionBase[WhoWhenErrorItem]):
    """Structured local node check used by the default Who&When Stage 1."""

    model_config = ConfigDict(extra="forbid")

    has_error: bool = Field(description="Whether this node itself contains an error.")
    error_role: ErrorRole
    is_root_candidate: bool
    candidate_role: CandidateRole


class CombinedLocalErrorItem(BaseModel):
    """Compact evidence item for the combined local/cascade response."""

    model_config = ConfigDict(extra="forbid")

    evidence: str = Field(description="Exact supporting substring from this node.")
    description: str = Field(description="Why this evidence is an error.")


class CombinedLocalInspection(BaseModel):
    """Local judgment returned alongside cascade relations in one parse call."""

    model_config = ConfigDict(extra="forbid")

    has_error: bool
    errors: List[CombinedLocalErrorItem] = Field(default_factory=list)
    status: NodeStatus
    is_root_candidate: bool
    candidate_role: CandidateRole
    error_role: ErrorRole
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class CombinedCauseRelation(BaseModel):
    """One active cause's relation to the current node."""

    model_config = ConfigDict(extra="forbid")

    cause_id: str
    relation: CascadeRelation
    is_root_candidate: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Optional[str] = Field(
        default=None,
        description="Exact supporting substring from the current node, if any.",
    )
    explanation: str = ""


class CombinedCascadeInspection(BaseModel):
    """Per-cause cascade judgments for the combined response."""

    model_config = ConfigDict(extra="forbid")

    status: NodeStatus
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    cause_relations: List[CombinedCauseRelation] = Field(default_factory=list)


class CombinedNodeInspection(BaseModel):
    """Independent local and cascade judgments in one structured response."""

    model_config = ConfigDict(extra="forbid")

    local: CombinedLocalInspection
    cascade: CombinedCascadeInspection


class FailureCause(BaseModel):
    """Who&When failure hypothesis propagated through the reverse graph."""

    model_config = ConfigDict(extra="forbid")

    cause_id: str = Field(description="Stable ID such as C1 or C2.")
    description: str = Field(description="Hypothesis explaining the downstream failure.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    support_status: SupportStatus = Field(
        default="HYPOTHESIS",
        description="HYPOTHESIS is unverified; SUPPORTED has concrete evidence; REFUTED is rejected.",
    )
    causal_status: CausalStatus = Field(
        default="UNKNOWN",
        description=(
            "ROOT_CANDIDATE remains eligible for Stage 2 comparison and upstream "
            "search; ROOT_CAUSE is a confirmed terminal cause."
        ),
    )
    should_propagate_upstream: bool = Field(
        default=True,
        description="Continue searching for this cause in upstream steps.",
    )
    localized_node_id: Optional[str] = Field(
        default=None,
        description="First node_id with concrete support; preserve once set for this cause.",
    )
    root_node_id: Optional[str] = Field(
        default=None,
        description="Node identified as this cause's root, if established.",
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Concrete supporting text from a node, when available.",
    )


class FinalFailureHypothesis(BaseModel):
    """Initial final-answer assessment before backward Stage 1 traversal."""

    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(description="Concise final-answer failure hypothesis.")
    assessment: FailureAssessment = Field(
        description="How clearly the final answer conflicts with the task or reference."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Optional[str] = Field(
        default=None,
        description="Evidence from the query, final result, or ground truth.",
    )
    ground_truth_used: bool = Field(
        description="Whether the reference answer informed this assessment."
    )
    causes: List[FailureCause] = Field(
        default_factory=list,
        description="Optional initial causes for backward propagation.",
    )


class FinalFailureAnalysis(BaseModel):
    """TRAIL-style final-node seed for Who&When backward attribution."""

    model_config = ConfigDict(extra="forbid")

    has_failure: bool = Field(
        description="Whether the final result has an observable task failure."
    )
    failure_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence that the final result fails, separate from each cause's confidence.",
    )
    final_error: str = Field(
        description="What is wrong with the final answer, or why failure is unproven."
    )
    causes: List[FailureCause] = Field(default_factory=list)
    reasoning: str = Field(description="Evidence-based final failure assessment.")


class CascadeAttribution(BaseModel):
    """Attribution of one active cause at the current step."""

    model_config = ConfigDict(extra="forbid")

    relation: CascadeRelation
    explanation: str = Field(description="Why this relation follows from this step.")
    is_root_candidate: bool = Field(
        description="Whether this step introduces this specific cause."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Optional[str] = Field(
        default=None,
        description="Exact supporting substring from the current step, if available.",
    )


class CascadeCauseRelation(CascadeAttribution):
    """A cascade attribution tied to a stable active cause ID."""

    cause_id: str = Field(description="ID of the active cause being assessed.")


class CascadeNodeInspection(BaseModel):
    """All active-cause relations at one node, kept separate from local errors."""

    model_config = ConfigDict(extra="forbid")

    status: NodeStatus = Field(description="Causal position of the current step.")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Summary of causal attribution at this step.")
    cause_relations: List[CascadeCauseRelation] = Field(default_factory=list)
    cause_updates: List[FailureCause] = Field(
        default_factory=list,
        description=(
            "Reserved for schema parity. Return an empty list in this Who&When "
            "trial; traversal derives cause state from cause_relations."
        ),
    )
    stopped_cause_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Reserved for schema parity. Return an empty list in this Who&When "
            "trial; traversal derives propagation from cause_relations."
        ),
    )



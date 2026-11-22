"""Response model bases shared by the dataset diagnosers."""

from typing import Generic, List, Literal, TypeVar

from pydantic import BaseModel, Field


NodeLocalStatus = Literal["CORRECT", "ROOT_CAUSE", "PASSIVE_ERROR", "UNCERTAIN"]
ErrorItemT = TypeVar("ErrorItemT", bound=BaseModel)


class NodeLocalInspectionBase(BaseModel, Generic[ErrorItemT]):
    """Fields common to a structured local inspection for any trace dataset."""

    has_error: bool
    errors: List[ErrorItemT] = Field(default_factory=list)
    status: NodeLocalStatus
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

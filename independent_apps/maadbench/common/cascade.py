from typing import Final, Literal


CascadeRelation = Literal[
    "INTRODUCES",
    "PROPAGATES",
    "AMPLIFIES",
    "RECOVERS",
    "UNRELATED",
    "REFUTES",
    "UNCERTAIN",
]

CASCADE_RELATIONS: Final[tuple[str, ...]] = (
    "INTRODUCES",
    "PROPAGATES",
    "AMPLIFIES",
    "RECOVERS",
    "UNRELATED",
    "REFUTES",
    "UNCERTAIN",
)

CASCADE_RELATION_SET: Final[frozenset[str]] = frozenset(CASCADE_RELATIONS)


def normalize_cascade_relation(
    value: object,
    *,
    default: CascadeRelation = "UNCERTAIN",
) -> CascadeRelation:
    relation = str(value or "").strip().upper()
    return relation if relation in CASCADE_RELATION_SET else default

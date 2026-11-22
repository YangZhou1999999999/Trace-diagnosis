from typing import Iterable, List, Literal, TypeVar


TraversalMode = Literal["bfs", "dfs"]
T = TypeVar("T")


def resolve_traversal_mode(
    mode: object,
    *,
    default: TraversalMode,
) -> TraversalMode:
    value = str(mode or "auto").strip().lower()
    if value == "auto":
        return default
    if value not in {"bfs", "dfs"}:
        raise ValueError("traversal_mode must be one of: auto, bfs, dfs")
    return value  # type: ignore[return-value]


def pop_frontier(frontier: List[T], mode: TraversalMode) -> T:
    return frontier.pop(0) if mode == "bfs" else frontier.pop()


def push_frontier(
    frontier: List[T],
    values: Iterable[T],
    mode: TraversalMode,
) -> None:
    ordered = list(values)
    if mode == "dfs":
        ordered.reverse()
    frontier.extend(ordered)

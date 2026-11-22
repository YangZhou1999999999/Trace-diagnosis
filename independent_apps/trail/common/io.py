import json
import os
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_graphs(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        first_nonempty = None
        for line in f:
            if line.strip():
                first_nonempty = line
                break

        if first_nonempty is None:
            return []

        f.seek(0)
        if first_nonempty.lstrip().startswith("["):
            loaded = json.load(f)
            if not isinstance(loaded, list):
                raise ValueError(f"Expected JSON list in {path}")
            return loaded
        return [json.loads(line) for line in f if line.strip()]


def load_json_records(path: str) -> List[Any]:
    """Load a local JSON or JSONL source as a list of records."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON input does not exist: {path}")
    if os.path.isdir(path):
        raise IsADirectoryError(f"Expected a JSON file, got a directory: {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        return []

    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        return records

    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict):
        return [loaded]
    raise ValueError(f"Expected a JSON object/list or JSONL records in {path}")


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def load_checkpoint(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, list):
        raise ValueError(f"Checkpoint must be a JSON list: {path}")
    return loaded


def save_checkpoint(path: str, result: Sequence[Dict[str, Any]]) -> None:
    ensure_parent(path)
    output_dir = os.path.dirname(os.path.abspath(path))
    temp_path = os.path.join(output_dir, f".checkpoint_{uuid.uuid4().hex}.tmp")
    try:
        with open(temp_path, "x", encoding="utf-8") as f:
            json.dump(list(result), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except PermissionError:
        # Fail closed: a direct write could truncate the last good checkpoint
        # if the process is interrupted. Keep the prior file intact instead.
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        raise
    except Exception:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        raise


def processed_success_indices(
    result: Sequence[Dict[str, Any]],
    completion_predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Set[int]:
    if completion_predicate is not None:
        completed: Set[int] = set()
        for index, item in enumerate(result):
            if not isinstance(item, dict):
                continue
            try:
                if completion_predicate(item):
                    completed.add(index)
            except Exception:
                # Resume predicates are advisory; malformed records must be
                # retried rather than causing the checkpoint to be skipped.
                continue
        return completed

    successful_statuses = {"SUCCESS", "OK", "COMPLETED"}
    return {
        i
        for i, item in enumerate(result)
        if isinstance(item, dict)
        and str(item.get("status", "")).upper() in successful_statuses
    }


def checkpoint_set_item(result: List[Dict[str, Any]], index: int, item: Dict[str, Any]) -> None:
    while len(result) <= index:
        result.append({"status": "PENDING"})
    result[index] = item



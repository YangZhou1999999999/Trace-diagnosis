import copy
from typing import Any, Dict


EVAL_LABEL_KEYS = {
    "raw_labels",
    "error_anchors",
    "labels",
    "mistake_agent",
    "mistake_step",
    "mistake_reason",
    "is_correct",
    "is_corrected",
    "system_prompt",
}


def clean_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "<missing>", "unknown"}:
        return ""
    return text


def strip_eval_labels(graph: Dict[str, Any], include_ground_truth: bool = True) -> Dict[str, Any]:
    cleaned = {
        key: copy.deepcopy(value)
        for key, value in graph.items()
        if key not in EVAL_LABEL_KEYS
    }
    if not include_ground_truth:
        cleaned.pop("ground_truth", None)
    return cleaned


def extract_eval_labels(graph: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in graph.items()
        if key in EVAL_LABEL_KEYS
    }


def leakage_audit(graph: Dict[str, Any], include_ground_truth: bool) -> Dict[str, Any]:
    label_keys = sorted(key for key in EVAL_LABEL_KEYS if key in graph)
    return {
        "eval_label_keys_in_source_graph": label_keys,
        "eval_label_keys_removed_before_diagnosis": label_keys,
        "ground_truth_in_source_graph": "ground_truth" in graph,
        "ground_truth_allowed_for_diagnosis": bool(include_ground_truth and "ground_truth" in graph),
    }


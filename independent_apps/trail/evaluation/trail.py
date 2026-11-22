import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .report import save_evaluation_report


ALL_CATEGORIES = [
    "Language-only",
    "Tool-related",
    "Poor Information Retrieval",
    "Incorrect Memory Usage",
    "Tool Output Misinterpretation",
    "Incorrect Problem Identification",
    "Tool Selection Errors",
    "Formatting Errors",
    "Instruction Non-compliance",
    "Tool Definition Issues",
    "Environment Setup Errors",
    "Rate Limiting",
    "Authentication Errors",
    "Service Errors",
    "Resource Not Found",
    "Resource Exhaustion",
    "Timeout Issues",
    "Context Handling Failures",
    "Resource Abuse",
    "Goal Deviation",
    "Task Orchestration",
]


def normalize_category(category: str, all_categories: Sequence[str] = ALL_CATEGORIES) -> str:
    if not category:
        return ""

    lowered = str(category).lower().strip()
    no_spaces = lowered.replace(" ", "")

    for std_cat in all_categories:
        if lowered == std_cat.lower() or no_spaces == std_cat.lower().replace(" ", ""):
            return std_cat

    for std_cat in all_categories:
        if no_spaces in std_cat.lower().replace(" ", ""):
            return std_cat

    return lowered


def _safe_load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_predictions(path: str) -> List[Dict[str, Any]]:
    data = _safe_load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Prediction file must be a JSON list: {path}")

    predictions = []
    for item in data:
        if not isinstance(item, dict):
            predictions.append({})
            continue
        raw_prediction = item.get("diagnosis") or item.get("analysis_results") or item
        prediction = dict(raw_prediction) if isinstance(raw_prediction, dict) else {}
        for key in (
            "status", "local_prediction_status", "cause_path_coverage_status",
            "cause_path_partial_reasons", "local_prediction_coverage",
            "cause_path_coverage", "cascade_failure", "graph_index",
        ):
            if key in item and key not in prediction:
                prediction[key] = item[key]
        if item.get("trace_id") is not None:
            if prediction.get("trace_id") not in {None, item["trace_id"]}:
                raise ValueError("Conflicting wrapper and diagnosis trace_id")
            prediction["trace_id"] = item["trace_id"]
        predictions.append(prediction)
    return predictions


def _valid_scoring_errors(record: Dict[str, Any]) -> bool:
    errors = record.get("errors")
    if not isinstance(errors, list):
        return False
    return all(
        isinstance(error, dict)
        and isinstance(error.get("location"), str)
        and bool(error.get("location"))
        and error.get("category") in ALL_CATEGORIES
        and isinstance(error.get("evidence"), str)
        and isinstance(error.get("description"), str)
        and error.get("impact") in {"HIGH", "MEDIUM", "LOW"}
        for error in errors
    )


def is_official_prediction_eligible(record: Dict[str, Any]) -> bool:
    """Validate Local Prediction eligibility without changing scorer formulas.

    New records may be diagnosis-level PARTIAL when their Local Prediction is
    complete and only Cause Path coverage is partial. Historical records use
    the pre-split top-level success rule; legacy PARTIAL records remain out.
    """
    if not isinstance(record, dict) or not _valid_scoring_errors(record):
        return False
    overall = str(record.get("status", "")).upper()
    if "local_prediction_status" in record:
        return (
            str(record.get("local_prediction_status", "")).upper() == "SUCCESS"
            and overall in {"SUCCESS", "PARTIAL"}
        )
    return overall in {"SUCCESS", "OK", "COMPLETED"}


def _effective_local_status(record: Optional[Dict[str, Any]]) -> str:
    if not isinstance(record, dict):
        return "UNRECORDED"
    if "local_prediction_status" in record:
        return str(record.get("local_prediction_status") or "UNRECORDED").upper()
    status = str(record.get("status", "")).upper()
    if status in {"SUCCESS", "OK", "COMPLETED"}:
        return "SUCCESS"
    if status == "FAILED":
        return "FAILED"
    return "UNRECORDED"


def _effective_cause_path_status(record: Optional[Dict[str, Any]]) -> str:
    if not isinstance(record, dict):
        return "UNRECORDED"
    if "cause_path_coverage_status" in record:
        return str(record.get("cause_path_coverage_status") or "UNRECORDED").upper()
    cascade = record.get("cascade_failure")
    if isinstance(cascade, dict) and cascade.get("enabled") is False:
        return "NOT_RUN"
    status = str(record.get("status", "")).upper()
    if status == "PARTIAL":
        return "PARTIAL"
    if status in {"SUCCESS", "OK", "COMPLETED"} and isinstance(cascade, dict):
        return "COMPLETE"
    return "UNRECORDED"


def _coverage_detail(record: Optional[Dict[str, Any]], trace_id: Any) -> Dict[str, Any]:
    record = record if isinstance(record, dict) else None
    local = record.get("local_prediction_coverage") if record else None
    cause = record.get("cause_path_coverage") if record else None
    cascade = record.get("cascade_failure") if record else None
    scheduler = cascade.get("h1_scheduler") if isinstance(cascade, dict) else None
    local = local if isinstance(local, dict) else {}
    cause = cause if isinstance(cause, dict) else {}
    scheduler = scheduler if isinstance(scheduler, dict) else {}
    local_status = _effective_local_status(record)
    cause_status = _effective_cause_path_status(record)
    eligible = is_official_prediction_eligible(record) if record is not None else False
    return {
        "trace_id": trace_id,
        "status": (record or {}).get("status", "MISSING_PREDICTION"),
        "local_prediction_status": local_status,
        "cause_path_coverage_status": cause_status,
        "cycle_nodes": cause.get("cycle_nodes", scheduler.get("cyclic_nodes", [])),
        "blocked_nodes": cause.get("blocked_nodes", scheduler.get("blocked_nodes", [])),
        "path_state_budget": cause.get("path_state_budget", scheduler.get("path_state_limit")),
        "path_state_budget_exhausted": cause.get(
            "path_state_budget_exhausted", scheduler.get("path_state_budget_exhausted", False),
        ),
        "worklist_visit_budget_exhausted": cause.get(
            "worklist_visit_budget_exhausted",
            (scheduler.get("graph_validation") or {}).get("worklist_visit_budget_exhausted", False),
        ),
        "local_expected_node_count": local.get("expected_count"),
        "local_completed_node_count": local.get("completed_count"),
        "local_failed_node_count": local.get("failed_count"),
        "local_unvisited_node_ids": local.get("unvisited_node_ids", []),
        "official_prediction_eligible": eligible,
    }


def build_trail_coverage_report(
    predictions: Sequence[Dict[str, Any]],
    ground_truths: Sequence[Dict[str, Any]],
    *,
    alignment: str = "auto",
) -> Dict[str, Any]:
    """Build coverage metadata for all reference traces, including missing runs."""
    if alignment not in {"auto", "position"}:
        raise ValueError("alignment must be 'auto' or explicit legacy 'position'")
    details: List[Dict[str, Any]] = []
    if alignment == "position":
        if len(predictions) != len(ground_truths):
            raise ValueError("Prediction/ground-truth length mismatch; select the same trace subset explicitly")
        details = [
            _coverage_detail(pred, gt.get("trace_id", index))
            for index, (pred, gt) in enumerate(zip(predictions, ground_truths))
        ]
    else:
        pred_ids = [pred.get("trace_id") for pred in predictions]
        gt_ids = [gt.get("trace_id") for gt in ground_truths]
        if any(value is not None for value in gt_ids):
            if any(value is None or str(value) == "" for value in gt_ids):
                raise ValueError("Missing Ground Truth trace_id; verify identity or explicitly use alignment='position'")
            gt_ids = [str(value) for value in gt_ids]
            if len(set(gt_ids)) != len(gt_ids):
                raise ValueError("Duplicate TRAIL Ground Truth trace_id")
            gt_id_set = set(gt_ids)
            by_id = {}
            unmatched = []
            seen_prediction_ids = set()
            for index, (prediction, trace_id) in enumerate(zip(predictions, pred_ids)):
                eligible = is_official_prediction_eligible(prediction)
                if trace_id is None or str(trace_id) == "":
                    if eligible:
                        raise ValueError("Eligible prediction is missing trace_id")
                    unmatched.append({
                        "prediction_index": index,
                        "trace_id": None,
                        "status": prediction.get("status"),
                        "reason": "MISSING_TRACE_ID_ON_INELIGIBLE_RECORD",
                    })
                    continue
                trace_id = str(trace_id)
                if trace_id in seen_prediction_ids:
                    raise ValueError("Duplicate TRAIL prediction trace_id")
                seen_prediction_ids.add(trace_id)
                if trace_id not in gt_id_set:
                    if eligible:
                        raise ValueError("Eligible prediction trace_id is absent from Ground Truth: " + trace_id)
                    unmatched.append({
                        "prediction_index": index,
                        "trace_id": trace_id,
                        "status": prediction.get("status"),
                        "reason": "UNKNOWN_TRACE_ID_ON_INELIGIBLE_RECORD",
                    })
                    continue
                by_id[trace_id] = prediction
            for trace_id in gt_ids:
                details.append(_coverage_detail(by_id.get(trace_id), trace_id))
        else:
            eligible_with_ids = [
                item for item in predictions
                if is_official_prediction_eligible(item) and item.get("trace_id") is not None
            ]
            if eligible_with_ids:
                raise ValueError("Prediction trace_id cannot be aligned because Ground Truth has no trace_id")
            if len(predictions) != len(ground_truths):
                raise ValueError("Prediction/ground-truth length mismatch; select the same trace subset explicitly")
            details = [
                _coverage_detail(pred, index)
                for index, pred in enumerate(predictions)
            ]
            unmatched = []
    if alignment == "position":
        unmatched = []

    local_counts = {
        status: sum(item["local_prediction_status"] == status for item in details)
        for status in ("SUCCESS", "PARTIAL", "FAILED", "UNRECORDED")
    }
    cause_counts = {
        status: sum(item["cause_path_coverage_status"] == status for item in details)
        for status in ("COMPLETE", "PARTIAL", "NOT_RUN", "UNRECORDED")
    }
    eligible_count = sum(bool(item["official_prediction_eligible"]) for item in details)
    local_cause_partial = [
        item for item in details
        if item["local_prediction_status"] == "SUCCESS"
        and item["cause_path_coverage_status"] == "PARTIAL"
    ]
    return {
        "total_traces": len(details),
        "prediction_records": len(predictions),
        "local_prediction_success": local_counts["SUCCESS"],
        "local_prediction_partial": local_counts["PARTIAL"],
        "local_prediction_failed": local_counts["FAILED"],
        "local_prediction_unrecorded": local_counts["UNRECORDED"],
        "cause_path_complete": cause_counts["COMPLETE"],
        "cause_path_partial": cause_counts["PARTIAL"],
        "cause_path_not_run": cause_counts["NOT_RUN"],
        "cause_path_unrecorded": cause_counts["UNRECORDED"],
        "local_complete_but_cause_path_partial": local_cause_partial,
        "official_prediction_eligible": eligible_count,
        "official_prediction_excluded": len(details) - eligible_count,
        "unmatched_prediction_record_count": len(unmatched),
        "unmatched_prediction_records": unmatched,
        "traces": details,
    }


def _select_official_pairs(
    predictions: Sequence[Dict[str, Any]],
    ground_truths: Sequence[Dict[str, Any]],
    *,
    alignment: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if alignment not in {"auto", "position"}:
        raise ValueError("alignment must be 'auto' or explicit legacy 'position'")
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    if alignment == "position":
        if len(predictions) != len(ground_truths):
            raise ValueError("Prediction/ground-truth length mismatch; select the same trace subset explicitly")
        pairs = list(zip(predictions, ground_truths))
    else:
        eligible_predictions = [item for item in predictions if is_official_prediction_eligible(item)]
        gt_ids = [gt.get("trace_id") for gt in ground_truths]
        if any(value is not None for value in gt_ids):
            if any(value is None or str(value) == "" for value in gt_ids):
                raise ValueError("Missing Ground Truth trace_id; verify identity or explicitly use alignment='position'")
            gt_ids = [str(value) for value in gt_ids]
            if len(set(gt_ids)) != len(gt_ids):
                raise ValueError("Duplicate TRAIL Ground Truth trace_id")
            gt_by_id = dict(zip(gt_ids, ground_truths))
            eligible_ids = [item.get("trace_id") for item in eligible_predictions]
            if any(value is None or str(value) == "" for value in eligible_ids):
                raise ValueError("Eligible prediction is missing trace_id")
            eligible_ids = [str(value) for value in eligible_ids]
            if len(set(eligible_ids)) != len(eligible_ids):
                raise ValueError("Duplicate TRAIL prediction trace_id")
            unknown = set(eligible_ids) - set(gt_ids)
            if unknown:
                raise ValueError("Prediction trace_id is absent from Ground Truth: " + ", ".join(sorted(unknown)))
            pairs = [
                (pred, gt_by_id[trace_id])
                for pred, trace_id in zip(eligible_predictions, eligible_ids)
            ]
        else:
            if any(item.get("trace_id") is not None for item in eligible_predictions):
                raise ValueError("Prediction trace_id cannot be aligned because Ground Truth has no trace_id")
            if len(predictions) != len(ground_truths):
                raise ValueError("Prediction/ground-truth length mismatch; select the same trace subset explicitly")
            pairs = list(zip(predictions, ground_truths))
    eligible = [(pred, gt) for pred, gt in pairs if is_official_prediction_eligible(pred)]
    return [pair[0] for pair in eligible], [pair[1] for pair in eligible]


def load_ground_truths_from_graphs(graph_jsonl: str) -> List[Dict[str, Any]]:
    ground_truths = []
    for graph in load_jsonl(graph_jsonl):
        raw_labels = graph.get("raw_labels") if isinstance(graph, dict) else {}
        trace_labels = raw_labels.get("trace_labels") if isinstance(raw_labels, dict) else None

        if isinstance(trace_labels, dict):
            gt = dict(trace_labels)
        else:
            gt = {}

        gt.setdefault("errors", graph.get("error_anchors") or [])
        if isinstance(raw_labels, dict) and "scores" in raw_labels:
            gt.setdefault("scores", raw_labels.get("scores") or [])
        if graph.get("trace_id") is not None:
            gt["trace_id"] = graph["trace_id"]
        ground_truths.append(gt)
    return ground_truths


def _category_vectors(
    gt_categories: Iterable[str],
    pred_categories: Iterable[str],
    all_categories: Sequence[str],
) -> Tuple[List[int], List[int]]:
    y_true = [0] * len(all_categories)
    y_pred = [0] * len(all_categories)

    for cat in gt_categories:
        if cat in all_categories:
            y_true[all_categories.index(cat)] = 1

    for cat in pred_categories:
        if cat in all_categories:
            y_pred[all_categories.index(cat)] = 1

    return y_true, y_pred


def _weighted_f1(y_true_rows: Sequence[Sequence[int]], y_pred_rows: Sequence[Sequence[int]]) -> float:
    if not y_true_rows:
        return 0.0

    labels = len(y_true_rows[0])
    weighted_sum = 0.0
    total_support = 0

    for label_idx in range(labels):
        tp = fp = fn = support = 0
        for y_true, y_pred in zip(y_true_rows, y_pred_rows):
            true_val = int(y_true[label_idx])
            pred_val = int(y_pred[label_idx])
            support += true_val
            if true_val and pred_val:
                tp += 1
            elif not true_val and pred_val:
                fp += 1
            elif true_val and not pred_val:
                fn += 1

        if support == 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        weighted_sum += support * f1
        total_support += support

    return weighted_sum / total_support if total_support else 0.0


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return 0.0

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def evaluate_trace_predictions(
    predictions: Sequence[Dict[str, Any]],
    ground_truths: Sequence[Dict[str, Any]],
    *,
    alignment: str = "auto",
) -> Dict[str, float]:
    # Official TRAIL joins generated files to annotation filenames. Preserve
    # that identity at this in-memory boundary without changing its formulas.
    if alignment not in {"auto", "position"}:
        raise ValueError("alignment must be 'auto' or explicit legacy 'position'")
    if len(predictions) != len(ground_truths):
        raise ValueError("Prediction/ground-truth length mismatch; select the same trace subset explicitly")
    if alignment == "auto" and any(
        item.get("trace_id") is not None for item in list(predictions) + list(ground_truths)
    ):
        pred_ids = [item.get("trace_id") for item in predictions]
        gt_ids = [item.get("trace_id") for item in ground_truths]
        if any(value is None or str(value) == "" for value in pred_ids + gt_ids):
            raise ValueError("Missing trace_id; verify identity or explicitly use alignment='position'")
        pred_ids = [str(value) for value in pred_ids]
        gt_ids = [str(value) for value in gt_ids]
        if len(set(pred_ids)) != len(pred_ids) or len(set(gt_ids)) != len(gt_ids):
            raise ValueError("Duplicate TRAIL trace_id")
        if set(pred_ids) != set(gt_ids):
            raise ValueError("Prediction/ground-truth trace_id mismatch")
        by_id = dict(zip(pred_ids, predictions))
        predictions = [by_id[trace_id] for trace_id in gt_ids]

    location_accuracy_sum = 0.0
    joint_accuracy_sum = 0.0
    all_y_true = []
    all_y_pred = []
    gt_overall_scores = []
    pred_overall_scores = []
    files_processed = 0

    for pred, gt in zip(predictions, ground_truths):
        gt_errors = gt.get("errors", []) or []
        pred_errors = pred.get("errors", []) or []

        gt_categories_raw = [error.get("category", "") for error in gt_errors if isinstance(error, dict)]
        gt_locations = [error.get("location", "") for error in gt_errors if isinstance(error, dict)]
        pred_categories_raw = [error.get("category", "") for error in pred_errors if isinstance(error, dict)]
        pred_locations = [error.get("location", "") for error in pred_errors if isinstance(error, dict)]

        gt_categories = [normalize_category(cat) for cat in gt_categories_raw if cat]
        pred_categories = [normalize_category(cat) for cat in pred_categories_raw if cat]

        common_locations = set(gt_locations).intersection(set(pred_locations))
        location_accuracy = len(common_locations) / len(set(gt_locations)) if gt_locations else 0.0
        location_accuracy_sum += location_accuracy

        gt_pairs = [
            (gt_locations[i], gt_categories[i])
            for i in range(min(len(gt_locations), len(gt_categories)))
        ]
        pred_pairs = [
            (pred_locations[i], pred_categories[i])
            for i in range(min(len(pred_locations), len(pred_categories)))
        ]
        common_pairs = set(gt_pairs).intersection(set(pred_pairs))
        joint_accuracy = len(common_pairs) / len(set(gt_pairs)) if gt_pairs else 0.0
        joint_accuracy_sum += joint_accuracy

        y_true, y_pred = _category_vectors(gt_categories, pred_categories, ALL_CATEGORIES)
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        gt_scores = gt.get("scores", [{}])[0] if gt.get("scores") else {}
        pred_scores = pred.get("scores", [{}])[0] if pred.get("scores") else {}
        if isinstance(gt_scores, dict) and isinstance(pred_scores, dict):
            if "overall" in gt_scores and "overall" in pred_scores:
                try:
                    gt_overall_scores.append(float(gt_scores.get("overall", -1) or -1))
                    pred_overall_scores.append(float(pred_scores.get("overall", -1) or -1))
                except (TypeError, ValueError):
                    pass

        files_processed += 1

    if files_processed:
        location_accuracy_avg = location_accuracy_sum / files_processed
        joint_accuracy_avg = joint_accuracy_sum / files_processed
        weighted_f1 = _weighted_f1(all_y_true, all_y_pred)
    else:
        location_accuracy_avg = 0.0
        joint_accuracy_avg = 0.0
        weighted_f1 = 0.0

    rho = _pearson(gt_overall_scores, pred_overall_scores)

    return {
        "Cat. F1": round(float(weighted_f1), 4),
        "Loc. Acc.": round(float(location_accuracy_avg), 4),
        "Joint": round(float(joint_accuracy_avg), 4),
        "rho": round(float(rho), 4),
        "n": files_processed,
    }


def evaluate_trail_files(
    prediction_file: str,
    graph_file: str,
    *,
    output_path: Optional[Union[str, Path]] = None,
    params: Optional[Dict[str, Any]] = None,
    alignment: str = "auto",
) -> Dict[str, float]:
    predictions = load_predictions(prediction_file)
    ground_truths = load_ground_truths_from_graphs(graph_file)
    coverage = build_trail_coverage_report(predictions, ground_truths, alignment=alignment)
    scoring_predictions, scoring_ground_truths = _select_official_pairs(
        predictions, ground_truths, alignment=alignment,
    )
    metrics = evaluate_trace_predictions(
        scoring_predictions, scoring_ground_truths, alignment=alignment,
    )
    total_traces = int(coverage["total_traces"])
    evaluated_traces = int(metrics["n"])
    metrics.update({
        "evaluated_traces": evaluated_traces,
        "total_traces": total_traces,
        "coverage": round(
            evaluated_traces / total_traces, 4
        ) if total_traces else 0.0,
        "evaluation_status": (
            "FULL" if evaluated_traces == total_traces
            else "PARTIAL"
        ),
    })
    if output_path is not None:
        report_params = dict(params or {})
        report_params.update({
            "prediction_file": str(prediction_file),
            "graph_file": str(graph_file),
            "alignment": alignment,
        })
        save_evaluation_report(output_path, "trail", report_params, metrics)
        coverage_path = Path(output_path).with_name(Path(output_path).stem + ".coverage.json")
        temporary_path = coverage_path.with_suffix(coverage_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(coverage, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(coverage_path)
    return metrics


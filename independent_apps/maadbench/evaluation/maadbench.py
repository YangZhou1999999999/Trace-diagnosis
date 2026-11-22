"""Official action-level MAADBench evaluation for graph-LLM anomaly scores."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

from .report import save_evaluation_report
from ..datasets.maadbench.split import (
    prepare_official_dev_split,
    prepare_official_test_split,
)


def _load_predictions(path: str) -> list[dict[str, Any]]:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"MAADBench prediction file does not exist: {file_path}")
    try:
        loaded = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        loaded = [
            json.loads(line)
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if isinstance(loaded, dict):
        return [loaded]
    if not isinstance(loaded, list):
        raise ValueError("MAADBench checkpoint must be a JSON list or JSONL file")
    return [item for item in loaded if isinstance(item, dict)]


def _source_aliases(trace_id: str) -> list[str]:
    source = str(trace_id).replace("\\", "/").strip("/").lower()
    if not source.endswith(".json"):
        source += ".json"
    parts = source.split("/")
    return ["/".join(parts[index:]) for index in range(len(parts))]


def _select_anomaly_score(action: dict[str, Any], score_source: str) -> float:
    """Select an explicitly defined score channel without changing the prediction."""
    if score_source == "final":
        value = action.get("anomaly_score")
    elif score_source == "local":
        local = action.get("local_prediction")
        value = local.get("anomaly_score") if isinstance(local, dict) else None
        if value is None:
            value = action.get("local_anomaly_score")
    else:
        raise ValueError(
            f"Unsupported MAADBench evaluation score source: {score_source!r}; "
            "expected 'final' or 'local'."
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Missing numeric {score_source} anomaly score in the MAADBench action prediction."
        )
    return float(value)


def _is_complete_local_cause_partial_record(record: dict[str, Any]) -> bool:
    """Allow only path-mode records with one well-formed Local row per action."""
    if (
        str(record.get("cause_state_mode", "")).lower() != "path"
        or str(record.get("status", "")).upper() != "PARTIAL"
        or str(record.get("local_prediction_status", "")).upper() != "SUCCESS"
        or str(record.get("cause_path_coverage_status", "")).upper() != "PARTIAL"
    ):
        return False
    coverage = record.get("coverage")
    rows = record.get("action_predictions")
    if not isinstance(coverage, dict) or not isinstance(rows, list):
        return False
    expected = coverage.get("expected_local_action_count")
    completed = coverage.get("completed_local_action_count")
    if not isinstance(expected, int) or expected < 1 or completed != expected or len(rows) != expected:
        return False
    steps: list[int] = []
    node_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("step_index") is None:
            return False
        if not isinstance(row.get("has_error"), bool):
            return False
        try:
            score = float(row.get("local_anomaly_score", row.get("anomaly_score")))
        except (TypeError, ValueError):
            return False
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            return False
        steps.append(int(row["step_index"]))
        node_ids.append(str(row.get("node_id", "")))
    return (
        sorted(steps) == list(range(expected))
        and len(set(steps)) == expected
        and all(node_ids)
        and len(set(node_ids)) == expected
    )


def _is_evaluation_record_eligible(record: dict[str, Any], evaluation_profile: str) -> bool:
    status = str(record.get("status", "")).upper()
    if status in {"SUCCESS", "OK"}:
        return True
    return (
        evaluation_profile == "local_complete_cause_partial"
        and _is_complete_local_cause_partial_record(record)
    )


def evaluate_maadbench_files(
    prediction_file: str,
    input_path: str,
    *,
    output_path: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
    options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Score the available predictions on the official MAADBench test actions.

    Partial checkpoints are reported with explicit coverage; the official split,
    threshold policy, score functions, and prediction logger come from MAADBench.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    run_dir = Path(prediction_file).expanduser()
    eval_dir = run_dir.parent / f"{run_dir.stem}_official_eval"
    resolved_options = dict(options or {})
    diagnosis_scope = str(resolved_options.get("maadbench_diagnosis_scope", "all"))
    if diagnosis_scope == "official_dev":
        official, args, _train, _dev, eval_dir = prepare_official_dev_split(
            input_path, str(eval_dir), resolved_options,
        )
        manifest_path = eval_dir / "dev_manifest.csv"
        evaluation_split = "official_train_dev"
    else:
        official, args, _train, _test, eval_dir = prepare_official_test_split(
            input_path, str(eval_dir), resolved_options,
        )
        manifest_path = eval_dir / "test_manifest.csv"
        evaluation_split = "official_test"
    manifest = pd.read_csv(
        manifest_path,
        dtype={"row_id": str, "source_file": str, "action_id": str},
    )
    if manifest.empty or not manifest.row_id.is_unique:
        raise ValueError("Official MAADBench test manifest is empty or has duplicate row IDs")
    run_options = (params or {}).get("run_options") or {}
    score_source = str(run_options.get("maadbench_evaluation_score_source") or "final")
    if score_source not in {"final", "local"}:
        raise ValueError(
            f"Unsupported MAADBench evaluation score source: {score_source!r}; "
            "expected 'final' or 'local'."
        )
    evaluation_profile = str(
        resolved_options.get("maadbench_evaluation_profile")
        or run_options.get("maadbench_evaluation_profile")
        or "legacy"
    )
    if evaluation_profile not in {"legacy", "local_complete_cause_partial"}:
        raise ValueError(
            "Unsupported MAADBench evaluation profile: "
            f"{evaluation_profile!r}; expected 'legacy' or 'local_complete_cause_partial'."
        )
    # MAADBench's logger joins predictions against this manifest. In dev mode,
    # use the deterministic inner holdout rather than the outer official test.
    args._test_manifest = manifest.copy()

    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    accepted_path_partial_records = 0
    excluded_path_partial_records = 0
    for record in _load_predictions(prediction_file):
        record_status = str(record.get("status", "")).upper()
        path_partial_eligible = (
            evaluation_profile == "local_complete_cause_partial"
            and _is_complete_local_cause_partial_record(record)
        )
        if not _is_evaluation_record_eligible(record, evaluation_profile):
            if str(record.get("cause_state_mode", "")).lower() == "path" and record_status == "PARTIAL":
                excluded_path_partial_records += 1
            continue
        if path_partial_eligible:
            accepted_path_partial_records += 1
        trace_id = str(record.get("trace_id", ""))
        for action in record.get("action_predictions") or []:
            if not isinstance(action, dict) or action.get("step_index") is None:
                continue
            step_index = int(action["step_index"])
            # Validate the selected score channel once while building the action index.
            _select_anomaly_score(action, score_source)
            key = (trace_id, step_index)
            if key in by_key:
                raise ValueError(f"Duplicate MAADBench action prediction: {key}")
            by_key[key] = action

    suffix_index: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for (trace_id, step_index), action in by_key.items():
        for alias in _source_aliases(trace_id):
            suffix_index.setdefault((alias, step_index), []).append(action)

    matched_rows = []
    missing_rows = []
    for row in manifest.itertuples(index=False):
        source = str(row.source_file).replace("\\", "/").strip("/").lower()
        step = int(row.step_num)
        hits = suffix_index.get((source, step), [])
        if len(hits) != 1:
            missing_rows.append({
                "source_file": source,
                "step_num": step,
                "match_count": len(hits),
            })
            continue
        prediction = hits[0]
        expected_id = "" if pd.isna(row.action_id) else str(row.action_id)
        actual_id = "" if prediction.get("action_id") is None else str(prediction.get("action_id"))
        if expected_id != actual_id:
            raise ValueError(
                f"MAADBench action ID mismatch for {source}, step={step}: "
                f"{actual_id!r} != {expected_id!r}"
            )
        score = _select_anomaly_score(prediction, score_source)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Invalid MAADBench anomaly score for {source}, step={step}: {score}")
        has_error = prediction.get("has_error")
        if not isinstance(has_error, bool):
            raise ValueError(f"Missing boolean has_error prediction for {source}, step={step}")
        matched_rows.append({
            "row_id": str(row.row_id),
            "source_file": str(row.source_file),
            "step_num": step,
            "action_id": expected_id,
            "gt_label": int(row.step_label),
            "pred_score": score,
            "score_source": score_source,
            "pred_has_error": int(has_error),
            "prediction_status": action.get("status"),
        })

    if not matched_rows:
        raise ValueError(
            "No predictions matched the official MAADBench test actions. "
            "Check the source model/domain/trace type and diagnosis scope."
        )
    evaluated = pd.DataFrame(matched_rows)
    y_true = evaluated["gt_label"].to_numpy(dtype=int)
    y_scores = evaluated["pred_score"].to_numpy(dtype=float)
    matched_ids = set(evaluated["row_id"].astype(str))
    expected_per_trace = manifest.groupby("source_file").size()
    matched_per_trace = evaluated.groupby("source_file").size()

    total_test_traces = len(expected_per_trace)

    evaluated_traces = sum(
        matched_per_trace.get(source, 0) == count
        for source, count in expected_per_trace.items()
    )
    args._test_manifest = manifest[
        manifest["row_id"].astype(str).isin(matched_ids)
    ].copy().reset_index(drop=True)
    configured_policy = str(args.threshold_policy)
    predictions_by_policy: dict[str, Any] = {}
    threshold_comparison: list[dict[str, Any]] = []
    for policy in ("fixed", "oracle"):
        args.threshold_policy = policy
        policy_pred = np.asarray(
            official.predict_scores(args, y_scores, y_true, probability=False), dtype=int,
        )
        predictions_by_policy[policy] = policy_pred
        policy_auc = float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) == 2 else None
        threshold_comparison.append({
            "policy": policy,
            "f1": float(f1_score(y_true, policy_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, policy_pred)),
            "auc": policy_auc,
            "balanced_accuracy": (
                float(balanced_accuracy_score(y_true, policy_pred))
                if len(np.unique(y_true)) == 2 else None
            ),
            "gt_errors": int(y_true.sum()),
            "predicted_errors": int(policy_pred.sum()),
            "n": int(len(y_true)),
        })
    args.threshold_policy = configured_policy
    y_pred = predictions_by_policy[configured_policy]
    evaluated["pred_label_score"] = y_pred
    evaluated["pred_label_fixed"] = predictions_by_policy["fixed"]
    evaluated["pred_label_oracle"] = predictions_by_policy["oracle"]

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) == 2 else None
    y_direct = evaluated["pred_has_error"].to_numpy(dtype=int)
    evaluated["direct_correct"] = evaluated["gt_label"].to_numpy(dtype=int) == y_direct
    direct_metrics: dict[str, Any] = {
        "f1": float(f1_score(y_true, y_direct, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_direct)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, y_direct))
            if len(np.unique(y_true)) == 2 else None
        ),
        "evaluated_actions": int(len(y_true)),
        "total_test_actions": int(len(manifest)),
        "total_manifest_actions": int(len(manifest)),
        "evaluation_split": evaluation_split,
        "coverage": float(len(evaluated) / len(manifest)),
        "gt_errors": int(y_true.sum()),
        "predicted_errors": int(y_direct.sum()),
    }
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_direct, labels=[0, 1]).ravel()
        direct_metrics["confusion_matrix"] = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }
    else:
        direct_metrics["confusion_matrix"] = None
    coverage = len(evaluated) / len(manifest)
    metrics: dict[str, Any] = {
        "n": int(len(evaluated)),
        "total_test_actions": int(len(manifest)),
        "total_manifest_actions": int(len(manifest)),
        "evaluation_split": evaluation_split,
        "missing_test_actions": int(len(missing_rows)),
        "coverage": float(coverage),
        "evaluation_status": "FULL" if not missing_rows else "PARTIAL",
        "anomaly_ratio": float(y_true.mean()),
        "f1": f1,
        "accuracy": accuracy,
        "auc": auc,
        "balanced_accuracy": balanced_accuracy,
        "threshold_policy": str(args.threshold_policy),
        "score_source": score_source,
        "anomaly_fraction": float(args.anomaly_fraction),
        "direct_error_localization": direct_metrics,
        "threshold_policy_comparison": threshold_comparison,
        "evaluated_traces": int(evaluated_traces),
        "total_test_traces": int(total_test_traces),
        "trace_coverage": (
            evaluated_traces / total_test_traces
            if total_test_traces else 0.0
        ),
    }
    if evaluation_profile == "local_complete_cause_partial":
        metrics["evaluation_profile"] = evaluation_profile
        metrics["eligible_cause_path_partial_records"] = accepted_path_partial_records
        metrics["excluded_cause_path_partial_records"] = excluded_path_partial_records

    action_details_path = eval_dir / "matched_action_details.csv"
    evaluated.to_csv(action_details_path, index=False)
    metrics["action_details_file"] = str(action_details_path)

    official.log_results(
        args, 0, 1, f1, accuracy, float("nan") if auc is None else auc, balanced_accuracy,
        gt_labels=y_true, pred_labels=y_pred, pred_scores=y_scores,
        test_ids=evaluated["row_id"].astype(str).to_numpy(),
    )
    (eval_dir / "summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(eval_dir / "missing_test_actions.csv", index=False)
    if output_path is not None:
        save_evaluation_report(output_path, "maadbench", params, metrics)
    return metrics


import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from ..common.labels import clean_label
from .report import save_evaluation_report


def pick_predicted_agent(item: Dict[str, Any]) -> str:
    return (
        clean_label(item.get("agent_name"))
        or clean_label(item.get("agent_role"))
        or clean_label(item.get("predicted_agent"))
        or "Unknown"
    )


def _step_number_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(int(value))
    text = str(value).strip()
    match = re.search(r"\d+", text)
    return match.group(0) if match else text


def read_actual_data(labeled_json: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        with open(labeled_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        mistake_agent = data.get("mistake_agent")
        mistake_step = data.get("mistake_step")
        if mistake_agent is not None and mistake_step is not None:
            return str(mistake_agent), str(mistake_step)
        print(f"Warning: missing mistake_agent or mistake_step in {labeled_json}")
        return None, None
    except FileNotFoundError:
        print(f"Error: actual data file not found: {labeled_json}")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: could not decode JSON: {labeled_json}")
        return None, None


def _count_reference_json_files(data_path: str) -> int:
    """Match the author evaluator's flat-directory reference denominator."""
    try:
        return sum(
            1 for name in os.listdir(data_path)
            if name.endswith(".json") and os.path.isfile(os.path.join(data_path, name))
        )
    except OSError:
        return 0


def read_predictions(eval_file: str) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(eval_file):
        print(f"Error: evaluation file not found: {eval_file}")
        return {}

    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = {}
    parsed_count = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).upper()
        if status in {"FAILED", "ERROR", "PARTIAL", "CANCELLED"}:
            continue

        analysis = item.get("analysis_results") or item.get("diagnosis") or item
        root_node = analysis.get("root_cause_node") if isinstance(analysis, dict) else None
        root_node = root_node if isinstance(root_node, dict) else {}

        filename = item.get("filename") or item.get("file_name")
        if not filename and isinstance(analysis, dict):
            filename = analysis.get("filename") or analysis.get("file_name")
        if not filename and item.get("trace_id"):
            filename = f"{item['trace_id']}.json"

        step_number = item.get("step_number")
        if step_number is None:
            step_number = item.get("predicted_step")
        if step_number is None and isinstance(analysis, dict):
            step_number = analysis.get("predicted_step")
        if step_number is None:
            step_number = root_node.get("step_index")

        if filename and step_number is not None:
            if filename in predictions:
                raise ValueError(f"Duplicate Who&When prediction filename: {filename}")
            merged = dict(root_node)
            merged.update(item)
            if isinstance(analysis, dict):
                merged.update({
                    "predicted_agent": analysis.get("predicted_agent"),
                    "agent_name": item.get("agent_name") or analysis.get("predicted_agent"),
                })

            predictions[filename] = {
                "predicted_agent": pick_predicted_agent(merged),
                "predicted_step": _step_number_from_value(step_number),
            }
            parsed_count += 1
        else:
            print(f"Warning: missing filename or step_number in item: {item}")

    print(f"Successfully parsed predictions for {parsed_count} files from {eval_file}.")
    return predictions



def evaluate_accuracy(
    predictions: Dict[str, Dict[str, str]],
    data_path: str,
    total_files: Optional[int] = None,
) -> Dict[str, Any]:

    correct_agent = 0
    correct_step = 0
    files_evaluated = 0

    # 完整数据集大小：只用于官方分母和 Coverage
    if total_files is None:
        total_files = _count_reference_json_files(data_path)

    if total_files < 0:
        raise ValueError("total_files must be non-negative")

    for filename, pred in predictions.items():
        labeled_file = os.path.join(data_path, filename)

        if not os.path.isfile(labeled_file):
            print(f"Warning: missing reference: {filename}")
            continue

        actual_agent, actual_step = read_actual_data(labeled_file)

        # 必须有有效 Ground Truth 才能参与评价
        if actual_agent is None or actual_step is None:
            continue

        # 实际完成且能够配对的数量
        files_evaluated += 1

        # 保持原论文的字符串包含匹配规则
        if actual_agent in pred["predicted_agent"]:
            correct_agent += 1

        if str(actual_step) in str(pred["predicted_step"]):
            correct_step += 1

    # 1. 实际完成样本的准确率
    agent_accuracy = (
        correct_agent / files_evaluated * 100
        if files_evaluated else None
    )

    step_accuracy = (
        correct_step / files_evaluated * 100
        if files_evaluated else None
    )

    # 2. 原论文：整个参考数据集作为分母
    official_agent_accuracy = (
        correct_agent / total_files * 100
        if total_files else 0.0
    )

    official_step_accuracy = (
        correct_step / total_files * 100
        if total_files else 0.0
    )

    # 3. 实际覆盖率
    coverage = (
        files_evaluated / total_files
        if total_files else 0.0
    )

    return {
        "Agent Acc.": (
            round(agent_accuracy, 4)
            if agent_accuracy is not None else None
        ),
        "Step Acc.": (
            round(step_accuracy, 4)
            if step_accuracy is not None else None
        ),

        "Paper Official Agent Acc.": round(
            official_agent_accuracy, 4
        ),
        "Paper Official Step Acc.": round(
            official_step_accuracy, 4
        ),

        "Correct Agent": correct_agent,
        "Correct Step": correct_step,
        "Files Evaluated": files_evaluated,
        "Total Files": total_files,
        "Coverage": round(coverage, 4),
        "Metric Protocol": "completed_subset",
    }




def evaluate_who_when_files(
    prediction_file: str,
    data_path: str,
    total_files: Optional[int] = None,
    *,
    output_path: Optional[Union[str, Path]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    predictions = read_predictions(prediction_file)
    metrics = evaluate_accuracy(
        predictions,
        data_path,
        total_files=total_files,
    )
    if output_path is not None:
        report_params = dict(params or {})
        report_params.update({
            "prediction_file": str(prediction_file),
            "reference_data_path": str(data_path),
            "total_files": metrics.get("Total Files", total_files),
        })
        save_evaluation_report(output_path, "who_when", report_params, metrics)
    return metrics

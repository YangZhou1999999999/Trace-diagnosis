import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union


def save_evaluation_report(
    output_path: Union[str, Path],
    dataset: str,
    params: Optional[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Write evaluation parameters and metrics to a JSON report."""
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_params = dict(params or {})
    report_params["evaluation_output_path"] = str(report_path)
    report = {
        "dataset": dataset,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "params": report_params,
        "metrics": dict(metrics),
    }

    temporary_path = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(report_path)
    return report

"""Local quote provenance checks that do not decide semantic correctness."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def _metadata_text(node: Dict[str, Any]) -> str:
    raw = node.get("metadata")
    if isinstance(raw, (dict, list)):
        return json.dumps(raw, ensure_ascii=False, default=str)
    return str(raw or "")


def _visible_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...[TRUNCATED]...\n" + text[-half:]


def audit_quote(
    evidence: Optional[str], node: Dict[str, Any], *, visible_limit: int = 1600
) -> Dict[str, str]:
    """Check whether a quote is attributable to current-node metadata.

    The result is a provenance/visibility annotation, never a verdict that
    the model's local error judgment is semantically right or wrong.
    """
    quote = str(evidence or "")
    if not quote.strip():
        return {
            "evidence_match_status": "MISSING",
            "evidence_source": "model_quote",
            "evidence_audit_reason": "The model supplied no local evidence quote.",
        }

    current = _metadata_text(node)
    if quote in current:
        if quote in _visible_text(current, visible_limit):
            return {
                "evidence_match_status": "EXACT",
                "evidence_source": "current_node.metadata",
                "evidence_audit_reason": "Exact quote appears in the visible current-node metadata.",
            }
        return {
            "evidence_match_status": "NOT_VISIBLE",
            "evidence_source": "current_node.metadata",
            "evidence_audit_reason": "Exact quote is in the full node but outside the truncated prompt span.",
        }

    normalize = lambda value: re.sub(r"\s+", " ", value).strip().casefold()
    reason = (
        "Quote matches current-node metadata only after case/whitespace normalization."
        if normalize(quote) and normalize(quote) in normalize(current)
        else "Quote was not located in current-node metadata; semantic judgment remains unverified."
    )
    return {
        "evidence_match_status": "UNVERIFIED",
        "evidence_source": "model_quote",
        "evidence_audit_reason": reason,
    }

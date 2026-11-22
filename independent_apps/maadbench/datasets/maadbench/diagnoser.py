from __future__ import annotations
import ast
import copy
import json
import re
import time
import hashlib
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field
from ...common.llm import LLMServer
from ...common.diagnosis import (
    MaadLegacyCauseStatePolicy,
    MaadPathCauseStatePolicy,
    NodeDiagnosis,
    SharedDiagnosisEngine,
)
from ...common.request_budget import PersistentRequestBudget, RequestBudgetError
from ...common.traversal import resolve_traversal_mode


class MAADBenchTransportFailure(RuntimeError):
    """Fail-closed signal for a connection failure that should stop the batch."""

    stop_run = True


def _contains_transport_connection_error(error: BaseException) -> bool:
    """Return whether an exception chain contains an SDK/network connection error."""
    seen: set[int] = set()
    current: Optional[BaseException] = error
    connection_error_names = {
        "APIConnectionError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "NetworkError",
        "RemoteProtocolError",
        "ConnectionError",
    }
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in connection_error_names:
            return True
        current = current.__cause__ or current.__context__
    return False


BLOCKED_KEYS = {
    "verification", "failure_report", "ground_truth", "schema_errors", "escaped",
    "eval_result", "anomaly_label", "step_label", "gt_label", "test_label",
    "correct_answer", "reference_answer", "gold_answer", "expected_answer",
    "is_correct", "failure_mode", "failed", "label", "labels",
}


SYSTEM_RULES = """You diagnose multi-agent execution traces from MAADBench.
Use only visible action inputs, actual model outputs, observed tool results, and graph edges.
You do not have oracle answers or verifier labels. Do not infer correctness from hidden
benchmark answers. Edges point from earlier to later: sequential_flow is temporal adjacency,
execution_flow is control/protocol, and data_flow represents observed output reuse. Temporal
proximity alone does not prove causality. A propagated wrong value can still be a local anomaly;
causal origin and anomaly detection are separate judgments. Give concrete evidence and do not
claim unknown puzzle answers are known."""


class MAADErrorItem(BaseModel):
    location: str = Field(description="Exact node_id of the current action where the error is observed.")
    evidence: str = Field(description="Exact substring from the current action supporting the error; empty if unavailable.")
    description: str = Field(description="Concrete explanation of what is wrong with this action.")
    impact: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"


class LocalResult(BaseModel):
    has_error: bool
    errors: list[MAADErrorItem] = Field(default_factory=list)
    anomaly_score: float = Field(ge=0.0, le=1.0, description="Continuous action anomaly score; higher means more anomalous.")
    status: Literal["CORRECT", "ROOT_CAUSE", "PASSIVE_ERROR", "UNCERTAIN"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class Cause(BaseModel):
    cause_id: str
    description: str
    category: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    support_status: Literal["HYPOTHESIS", "SUPPORTED", "REFUTED"] = "HYPOTHESIS"
    causal_status: Literal["UNKNOWN", "PROXIMATE_CAUSE", "ROOT_CAUSE"] = "UNKNOWN"
    should_propagate_upstream: bool = True
    localized_node_id: Optional[str] = None
    root_node_id: Optional[str] = None
    evidence: Optional[str] = None


class MaadPathCause(Cause):
    model_config = ConfigDict(extra="forbid")


class CauseRelation(BaseModel):
    cause_id: str
    relation: Literal["INTRODUCES", "PROPAGATES", "AMPLIFIES", "RECOVERS", "UNRELATED", "REFUTES", "UNCERTAIN"]
    explanation: str
    is_root_cause: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: Optional[str] = None


class CascadeResult(BaseModel):
    status: Literal["CORRECT", "ROOT_CAUSE", "PASSIVE_ERROR", "UNCERTAIN"] = "UNCERTAIN"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""
    cause_relations: list[CauseRelation] = Field(default_factory=list)
    cause_updates: list[Cause] = Field(default_factory=list)
    stopped_cause_ids: list[str] = Field(default_factory=list)


class PathScopedCauseRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cause_id: str
    incoming_path_id: str
    relation: Literal["INTRODUCES", "PROPAGATES", "AMPLIFIES", "RECOVERS", "UNRELATED", "REFUTES", "UNCERTAIN"]
    explanation: str
    is_root_cause: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: Optional[str] = None


class PathScopedCauseUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incoming_path_id: str
    cause: MaadPathCause


class CascadeResultPath(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["CORRECT", "ROOT_CAUSE", "PASSIVE_ERROR", "UNCERTAIN"] = "UNCERTAIN"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""
    cause_relations: list[PathScopedCauseRelation] = Field(default_factory=list)
    cause_updates: list[PathScopedCauseUpdate] = Field(default_factory=list)
    stopped_cause_ids: list[str] = Field(default_factory=list)


class CombinedResult(BaseModel):
    local: LocalResult
    cascade: CascadeResult


class CombinedPathResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local: LocalResult
    cascade: CascadeResultPath


class HypothesisResult(BaseModel):
    has_failure: bool
    final_error: str
    causes: list[Cause] = Field(default_factory=list)
    reasoning: str


def _walk_for_blocked_keys(value: Any, location: str = "graph") -> None:
    if isinstance(value, Mapping):
        leaked = BLOCKED_KEYS.intersection(value)
        if leaked:
            raise ValueError(f"Ground-truth-like field(s) {sorted(leaked)} at {location}")
        for key, child in value.items():
            _walk_for_blocked_keys(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_for_blocked_keys(child, f"{location}[{index}]")


def _string(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def _truncate(value: Any, limit: int, marker: str = " ...[TRUNCATED]... ") -> str:
    rendered = _string(value)
    if len(rendered) <= limit:
        return rendered
    half = limit // 2
    return rendered[:half] + marker + rendered[-half:]


_CASCADE_ID_FIELDS = {"cause_id", "incoming_path_id", "path_id", "root_node_id", "localized_node_id"}
_CASCADE_SECRET_FIELDS = {
    "apikey", "apitoken", "authorization", "authtoken", "accesstoken",
    "refreshtoken", "token", "password", "passwd", "secret", "clientsecret",
    "privatekey", "credential", "credentials", "cookie", "setcookie",
}


def _sanitize_cascade_evidence(value: Any, *, string_limit: int = 1200) -> dict[str, Any]:
    """Serialize bounded Cascade fields while retaining every cause/path identity."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, Mapping):
        raise TypeError("cascade_evidence must be a structured Cascade result")

    def clean(item: Any, field_name: Optional[str] = None) -> Any:
        if isinstance(item, Mapping):
            result = {}
            for key, child in item.items():
                normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
                result[str(key)] = "[REDACTED]" if normalized in _CASCADE_SECRET_FIELDS else clean(child, str(key))
            return result
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        if isinstance(item, str):
            if field_name in _CASCADE_ID_FIELDS:
                return item
            redacted = re.sub(
                r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
                r"\1[REDACTED]", item,
            )
            redacted = re.sub(
                r"(?i)((?:api[_-]?key|api[_-]?token|access[_-]?token|refresh[_-]?token|password|client[_-]?secret)\s*[:=]\s*[\"']?)[^\s,;\"']+",
                r"\1[REDACTED]", redacted,
            )
            redacted = re.sub(
                r"(?i)\b(?:sk-ant-|sk-|ghp_|github_pat_)[A-Za-z0-9_-]{16,}",
                "[REDACTED]", redacted,
            )
            return _truncate(redacted, string_limit)
        return item

    return clean(value)


_MAAD_CASCADE_LOCAL_EVIDENCE_VALIDATION_VERSION = "maad_path_cascade_local_evidence_v1"
_MAAD_REJECTED_ROOT_EXPLANATION = "Unverified root claim omitted; assess current-action evidence independently."


def _maad_path_local_evidence_view(
    cascade: CascadeResultPath,
    audit: list[dict[str, Any]],
    current_node: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a schema-allowlisted Path view with rejected Root claims removed."""
    rejected_reasons: dict[tuple[str, str], str] = {}
    for event in audit:
        if str(event.get("status", "")).startswith("ROOT_CAUSE_REJECTED"):
            key = (str(event.get("cause_id", "")), str(event.get("incoming_path_id", "")))
            rejected_reasons.setdefault(key, str(event.get("status")))
    valid_root_keys = {
        (str(relation.cause_id), str(relation.incoming_path_id))
        for relation in cascade.cause_relations
        if relation.is_root_cause
    }
    if cascade.status == "ROOT_CAUSE" and not valid_root_keys and not rejected_reasons:
        rejected_reasons[("", "")] = "ROOT_CAUSE_REJECTED_WITHOUT_VALIDATED_PATH_EVIDENCE"

    relations = []
    for relation in cascade.cause_relations:
        key = (str(relation.cause_id), str(relation.incoming_path_id))
        rejected = key in rejected_reasons
        relations.append({
            "cause_id": relation.cause_id,
            "incoming_path_id": relation.incoming_path_id,
            "relation": relation.relation,
            "explanation": _MAAD_REJECTED_ROOT_EXPLANATION if rejected else relation.explanation,
            "is_root_cause": bool(relation.is_root_cause and not rejected),
            "confidence": relation.confidence,
            "evidence": relation.evidence,
            "root_claim_validation": (
                "rejected" if rejected else
                "validated_tentative" if key in valid_root_keys else "not_a_root_claim"
            ),
        })

    updates = []
    for update in cascade.cause_updates:
        cause = update.cause
        key = (str(cause.cause_id), str(update.incoming_path_id))
        rejected = key in rejected_reasons
        evidence = cause.evidence
        if rejected and evidence and not _has_exact_visible_action_evidence(current_node, evidence):
            evidence = None
        cause_view = {
            "cause_id": cause.cause_id,
            "description": _MAAD_REJECTED_ROOT_EXPLANATION if rejected else cause.description,
            "category": cause.category,
            "confidence": cause.confidence,
            "support_status": cause.support_status,
            "causal_status": "UNKNOWN" if rejected else cause.causal_status,
            "should_propagate_upstream": (
                cause.support_status != "REFUTED" if rejected else cause.should_propagate_upstream
            ),
            "localized_node_id": (
                cause.localized_node_id
                if not rejected or cause.localized_node_id in (None, str(current_node.get("node_id", "")))
                else None
            ),
            "root_node_id": None if rejected else cause.root_node_id,
            "evidence": evidence,
        }
        updates.append({
            "incoming_path_id": update.incoming_path_id,
            "cause": cause_view,
            "root_claim_validation": (
                "rejected" if rejected else
                "validated_tentative" if key in valid_root_keys else "not_a_root_claim"
            ),
        })

    rejected_claims = [
        {"cause_id": cause_id or None, "incoming_path_id": path_id or None, "reason_code": reason}
        for (cause_id, path_id), reason in sorted(rejected_reasons.items())
    ]
    status = cascade.status
    if status == "ROOT_CAUSE" and not valid_root_keys:
        status = "UNCERTAIN"
    return {
        "validation": {
            "applied": True,
            "version": _MAAD_CASCADE_LOCAL_EVIDENCE_VALIDATION_VERSION,
            "validated_tentative_root_keys": [list(key) for key in sorted(valid_root_keys)],
            "rejected_root_claims": rejected_claims,
        },
        "status": status,
        "confidence": cascade.confidence,
        "reasoning": _MAAD_REJECTED_ROOT_EXPLANATION if rejected_claims else cascade.reasoning,
        "cause_relations": relations,
        "cause_updates": updates,
        "stopped_cause_ids": list(cascade.stopped_cause_ids),
    }


def _bounded_visible_value(value: Any, limit: int, marker: str) -> Any:
    if isinstance(value, str):
        return _truncate(value, limit, marker)
    rendered = _string(value)
    return value if len(rendered) <= limit else _truncate(rendered, limit, marker)


def visible_node(
    node: dict[str, Any],
    limit: int = 3000,
    *,
    truncation_marker: str = " ...[TRUNCATED]... ",
) -> dict[str, Any]:
    """Return the allowlisted Action view, preserving bounded structured values."""
    keys = (
        "node_id", "semantic_role", "step_index", "name", "action_type", "action_id",
        "puzzle_id", "input", "output_message", "output_structured", "tool_execution",
    )
    long_fields = {"input", "output_message", "output_structured", "tool_execution"}
    projected = {
        key: _bounded_visible_value(node[key], limit, truncation_marker)
        if key in long_fields else node[key]
        for key in keys if key in node and node[key] is not None
    }
    content_keys = {"input", "output_message", "output_structured", "tool_execution"}

    def has_action_content() -> bool:
        return any(node.get(key) not in (None, "", {}, [], ()) for key in content_keys)

    if str(node.get("semantic_role") or "").lower() != "query":
        metadata = node.get("metadata")
        parsed: Any = dict(metadata) if isinstance(metadata, Mapping) else None
        if isinstance(metadata, str):
            for parser in (json.loads, ast.literal_eval):
                try:
                    candidate = parser(metadata)
                except (ValueError, SyntaxError, json.JSONDecodeError):
                    continue
                if isinstance(candidate, Mapping):
                    parsed = dict(candidate)
                    break
        if isinstance(parsed, dict):
            _walk_for_blocked_keys(parsed, "node.metadata")
            safe_aliases = {
                "input": ("input",),
                "output_message": ("output_message", "output"),
                "output_structured": ("output_structured", "structured_output"),
                "tool_execution": ("tool_execution", "tool_result"),
            }
            action_content_found = has_action_content()
            for target, aliases in safe_aliases.items():
                if node.get(target) not in (None, "", {}, [], ()):
                    continue
                for alias in aliases:
                    if alias in parsed and parsed[alias] is not None:
                        value = parsed[alias]
                        projected[target] = (
                            _bounded_visible_value(value, limit, truncation_marker)
                            if target in long_fields else value
                        )
                        action_content_found = action_content_found or value not in (None, "", {}, [], ())
                        break
            if not action_content_found:
                raise ValueError(
                    "MAAD Action metadata contains no recognized observable input/output fields; "
                    "reconvert it with the current graph converter instead of silently diagnosing an empty Action."
                )
        elif metadata not in (None, "", {}, [], ()) and not has_action_content():
            raise ValueError(
                "MAAD Action metadata has an unsupported non-empty type or text but no allowlisted input/output fields; "
                "reconvert the trace with the current graph converter instead of silently diagnosing an empty Action."
            )
    return projected


class MAADJudge:
    """MAADBench JSON prompts on top of the shared request transport."""

    def __init__(self, server: LLMServer, dump_visible_context_dir: Optional[str | Path] = None):
        self.server = server
        self.model = str(server.model or "")
        self.request_budget = PersistentRequestBudget.from_environment()
        self.dump_visible_context_dir = (
            Path(dump_visible_context_dir).expanduser()
            if dump_visible_context_dir is not None else None
        )
        self._dump_counter = 0

    @staticmethod
    def _safe_dump_component(value: Any) -> str:
        rendered = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "unknown")).strip("._")
        return rendered[:100] or "unknown"

    def _dump_messages(
        self,
        messages: list[dict[str, str]],
        request: dict[str, Any],
        metadata: Optional[dict[str, Any]],
        attempt: int,
    ) -> None:
        if self.dump_visible_context_dir is None:
            return
        self._dump_counter += 1
        self.dump_visible_context_dir.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        message_stats = [
            {
                "index": index,
                "role": message.get("role"),
                "character_length": len(str(message.get("content", ""))),
                "estimated_tokens_chars_div_4": (len(str(message.get("content", ""))) + 3) // 4,
            }
            for index, message in enumerate(messages)
        ]
        context = dict(metadata or {})
        row = {
            "capture_mode": context.pop("capture_mode", "LIVE_PRE_TRANSPORT"),
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "trace_id": context.pop("trace_id", ""),
            "node_id": context.pop("node_id", ""),
            "stage": context.pop("stage", ""),
            "response_schema": context.pop("response_schema", ""),
            "attempt": attempt,
            "model": request.get("model", self.model),
            "request_parameters": {
                key: value for key, value in request.items()
                if key not in {"messages", "api_key", "authorization", "headers"}
            },
            "context_metadata": context,
            "message_count": len(messages),
            "messages_character_length": len(serialized),
            "messages_estimated_tokens_chars_div_4": (len(serialized) + 3) // 4,
            "message_stats": message_stats,
            "messages_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "messages": messages,
        }
        filename = (
            f"{self._dump_counter:06d}_"
            f"{self._safe_dump_component(row['trace_id'])}_"
            f"{self._safe_dump_component(row['node_id'])}_"
            f"{self._safe_dump_component(row['stage'])}_attempt{attempt}.json"
        )
        (self.dump_visible_context_dir / filename).write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def ask(
        self,
        instruction: str,
        payload: dict[str, Any],
        schema: type[BaseModel],
        *,
        system_rules: str = SYSTEM_RULES,
        debug_metadata: Optional[dict[str, Any]] = None,
    ) -> BaseModel:
        schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        messages = [
            {
                "role": "system",
                "content": (
                    system_rules + "\n\n" + instruction
                    + "\n\nReturn exactly one valid JSON object without markdown. "
                    + "Follow the exact property names and enum values below.\n"
                    + "Pydantic JSON Schema:\n" + schema_text
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ]
        last_error: Optional[Exception] = None
        for attempt, token_budget in enumerate((6000, 10000, 10000), start=1):
            request = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": token_budget,
            }
            try:
                self._dump_messages(messages, request, debug_metadata, attempt)
                try:
                    with self.request_budget.transport_call():
                        response = self.server.create(**request)
                except Exception as exc:
                    if isinstance(exc, RequestBudgetError):
                        raise
                    message = str(exc).lower()
                    unsupported_parameter = "max_completion_tokens" in message and any(
                        word in message for word in ("unsupported", "unknown", "unrecognized", "invalid")
                    )
                    if not unsupported_parameter:
                        raise
                    request["max_tokens"] = request.pop("max_completion_tokens")
                    with self.request_budget.transport_call():
                        response = self.server.create(**request)
                if not response.choices:
                    raise ValueError("The LLM returned an empty choices list")
                choice = response.choices[0]
                content = str(getattr(choice.message, "content", "") or "").strip()
                if getattr(choice, "finish_reason", None) in {"length", "max_tokens"}:
                    raise ValueError("The LLM JSON response was truncated at the token limit")
                if not content:
                    raise ValueError("The LLM returned empty message content")
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I).strip()
                try:
                    decoded = json.loads(content)
                except json.JSONDecodeError:
                    start = content.find("{")
                    if start < 0:
                        raise
                    decoded, _end = json.JSONDecoder().raw_decode(content[start:])
                if not isinstance(decoded, dict):
                    raise ValueError("Expected one JSON object")
                return schema.model_validate(decoded)
            except Exception as exc:
                if isinstance(exc, RequestBudgetError):
                    raise
                if isinstance(exc, MAADBenchTransportFailure):
                    raise
                if _contains_transport_connection_error(exc):
                    # The SDK has already exhausted its own transport retries. This
                    # is not a malformed model response, so do not spend additional
                    # requests on JSON repair or continue with the remaining traces.
                    raise MAADBenchTransportFailure(
                        "LLM transport connection failed after SDK retries; stopping the remaining diagnosis run. "
                        f"({type(exc).__name__})"
                    ) from exc
                last_error = exc
                if attempt == 3:
                    break
                messages.append({
                    "role": "user",
                    "content": (
                        f"Previous response failed {schema.__name__} validation: {str(exc)[:700]}. "
                        "Return a complete corrected JSON object matching every required field."
                    ),
                })
                time.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError(f"MAADBench response failed JSON/schema validation: {last_error}") from last_error


EDGE_PRIORITY = {"data_flow": 0, "execution_flow": 1, "sequential_flow": 2}


def _candidate_context(
    node_id: str,
    edges: list[dict[str, Any]],
    node_map: dict[str, dict[str, Any]],
    *,
    upstream: bool,
    truncation_marker: str = " ...[TRUNCATED]... ",
) -> list[dict[str, Any]]:
    by_node: dict[str, dict[str, Any]] = {}
    for edge_index, edge in enumerate(edges):
        if (edge.get("dst") if upstream else edge.get("src")) != node_id:
            continue
        other = str(edge.get("src") if upstream else edge.get("dst"))
        if other not in node_map:
            continue
        candidate = by_node.get(other)
        if candidate is None:
            candidate = {
                "node_id": other,
                "relations": [],
                "edge_evidence": [],
                "node": visible_node(node_map[other], 700, truncation_marker=truncation_marker),
                "_priority": 3,
                "_order": edge_index,
            }
            by_node[other] = candidate
        kind = str(edge.get("kind") or "").lower()
        if kind not in candidate["relations"]:
            candidate["relations"].append(kind)
        if edge.get("evidence") is not None:
            candidate["edge_evidence"].append(edge.get("evidence"))
        candidate["_priority"] = min(candidate["_priority"], EDGE_PRIORITY.get(kind, 3))
    candidates = sorted(by_node.values(), key=lambda item: (item["_priority"], item["_order"]))
    for candidate in candidates:
        ordered_relations = sorted(candidate["relations"], key=lambda kind: EDGE_PRIORITY.get(kind, 3))
        candidate["edge"] = ordered_relations[0] if len(ordered_relations) == 1 else ordered_relations
        candidate.pop("relations", None)
        candidate.pop("_priority", None)
        candidate.pop("_order", None)
    return candidates


def _select_upstream_context(
    node_id: str,
    edges: list[dict[str, Any]],
    node_map: dict[str, dict[str, Any]],
    *,
    policy: str,
    char_budget: int,
    truncation_marker: str = " ...[TRUNCATED]... ",
) -> list[dict[str, Any]]:
    candidates = _candidate_context(
        node_id, edges, node_map, upstream=True, truncation_marker=truncation_marker,
    )
    if policy == "legacy":
        selected = []
        seen = set()
        for item in candidates:
            node_key = item["node_id"]
            if node_key in seen:
                continue
            seen.add(node_key)
            primary_edge = item["edge"][0] if isinstance(item["edge"], list) else item["edge"]
            selected.append({"edge": primary_edge, "node": item["node"]})
            if len(selected) >= 3:
                break
        return selected
    selected = []
    budget = max(0, char_budget)
    for item in candidates:
        compact = {"edge": item["edge"], "edge_evidence": item["edge_evidence"], "node": item["node"]}
        if len(json.dumps(selected + [compact], ensure_ascii=False, default=str)) <= budget:
            selected.append(compact)
    return selected


def _select_downstream_context(
    node_id: str,
    edges: list[dict[str, Any]],
    node_map: dict[str, dict[str, Any]],
    *,
    truncation_marker: str = " ...[TRUNCATED]... ",
) -> list[dict[str, Any]]:
    return [
        {"edge": item["edge"][0] if isinstance(item["edge"], list) else item["edge"], "node": item["node"]}
        for item in _candidate_context(
            node_id, edges, node_map, upstream=False, truncation_marker=truncation_marker,
        )[:3]
    ]


def _backward_order(action_nodes: list[dict[str, Any]], edges: list[dict[str, Any]], traversal_mode: str) -> list[str]:
    """Compatibility wrapper for the shared reverse scheduler."""
    node_ids = [str(node["node_id"]) for node in reversed(action_nodes)]
    ordered, _cycles = SharedDiagnosisEngine.backward_schedule(
        node_ids,
        edges,
        entry_node_id=str(action_nodes[-1]["node_id"]),
        traversal_mode=traversal_mode,
        schedule_mode="all_nodes",
        edge_priority=EDGE_PRIORITY,
    )
    return ordered


def _canonicalize_cause_node_refs(cause: Cause, action_id_to_node_id: dict[str, str]) -> Cause:
    """Map model-returned action_id aliases back to opaque graph node IDs."""
    normalized = cause.model_copy(deep=True)
    for field_name in ("localized_node_id", "root_node_id"):
        value = getattr(normalized, field_name)
        if value is not None and value in action_id_to_node_id:
            setattr(normalized, field_name, action_id_to_node_id[value])
    return normalized


def _canonicalize_local_error_locations(
    local: LocalResult | LocalResultV2,
    action_id_to_node_id: dict[str, str],
) -> None:
    """Ensure error locations reference node_id, never the distinct action_id."""
    for error in local.errors:
        if error.location in action_id_to_node_id:
            error.location = action_id_to_node_id[error.location]


def _has_exact_visible_action_evidence(visible: dict[str, Any], evidence: Any) -> bool:
    if not isinstance(evidence, str) or not evidence.strip():
        return False
    content_fields = ("input", "output_message", "output_structured", "tool_execution")
    return any(
        evidence in _string(visible[field])
        for field in content_fields
        if visible.get(field) not in (None, "", {}, [], ())
    )


_MAAD_PATH_CONFLICT_ANALYSIS_VERSION = "branch_conflicts_v2"
_MAAD_PATH_METHOD_VERSION = "maad_path_v2"
_MAAD_PATH_CONFLICT_REPORT_LIMIT = 256
_MAAD_PATH_VARIATION_STATE_LIMIT = 256


def _analyze_maad_path_conflicts(
    states: list[dict[str, Any]],
    path_nodes: Mapping[str, tuple[str, ...]],
    cause_path_audit: list[dict[str, Any]],
    node_map: Mapping[str, dict[str, Any]],
    *,
    partial_coverage: bool = False,
    max_conflict_records: int = _MAAD_PATH_CONFLICT_REPORT_LIMIT,
) -> dict[str, Any]:
    """Summarize path-state evolution and bounded, verified sibling conflicts.

    Path IDs are opaque. Branch ancestry is derived only from the stored node
    sequences in ``path_nodes``; state changes along an ancestor chain never
    count as independent-branch conflicts.
    """
    rejected_roots = {
        (str(event.get("cause_id", "")), str(event.get("incoming_path_id", "")))
        for event in cause_path_audit
        if str(event.get("status", "")).startswith("ROOT_CAUSE_REJECTED")
    }
    by_cause: dict[str, list[dict[str, Any]]] = {}
    for state in states:
        cause = state.get("cause") if isinstance(state.get("cause"), Mapping) else {}
        cause_id = str(cause.get("cause_id", ""))
        if cause_id:
            by_cause.setdefault(cause_id, []).append(state)

    variations: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    unresolved_path_ids: set[str] = set()
    total_conflict_pairs = 0
    total_conflict_groups = 0
    truncated = False
    variation_states_truncated = False

    def verified_state(state: dict[str, Any]) -> Optional[str]:
        cause = state.get("cause") if isinstance(state.get("cause"), Mapping) else {}
        cause_id = str(cause.get("cause_id", ""))
        path_id = str(state.get("incoming_path_id", ""))
        support = str(cause.get("support_status", "")).upper()
        causal = str(cause.get("causal_status", "")).upper()
        if support == "REFUTED":
            return "REFUTED"
        if support != "SUPPORTED":
            return None
        localized = str(cause.get("localized_node_id") or "")
        evidence = cause.get("evidence")
        if causal == "ROOT_CAUSE":
            root = str(cause.get("root_node_id") or "")
            if (
                (cause_id, path_id) not in rejected_roots
                and root
                and root == localized
                and root in node_map
                and _has_exact_visible_action_evidence(node_map[root], evidence)
            ):
                return "ROOT_CAUSE"
            return None
        if (
            localized in node_map
            and _has_exact_visible_action_evidence(node_map[localized], evidence)
        ):
            return "SUPPORTED"
        return None

    def summarize_state(state: dict[str, Any], node_sequence: tuple[str, ...]) -> dict[str, Any]:
        cause = state.get("cause") if isinstance(state.get("cause"), Mapping) else {}
        return {
            "incoming_path_id": str(state.get("incoming_path_id", "")),
            "node_ids": list(node_sequence),
            "last_node_id": node_sequence[-1] if node_sequence else None,
            "support_status": str(cause.get("support_status", "UNKNOWN")),
            "causal_status": str(cause.get("causal_status", "UNKNOWN")),
            "localized_node_id": cause.get("localized_node_id"),
            "root_node_id": cause.get("root_node_id"),
            "evidence_present": bool(cause.get("evidence")),
            "verified_state": verified_state(state),
        }

    for cause_id in sorted(by_cause):
        cause_states = sorted(
            by_cause[cause_id],
            key=lambda state: str(state.get("incoming_path_id", "")),
        )
        seed_ids = {
            str(state.get("incoming_path_id", ""))
            for state in cause_states
            if str(state.get("incoming_path_id", "")) == "FINAL"
            or str(state.get("incoming_path_id", "")).startswith("LOCAL_SEED:")
        }
        origin_groups: dict[str, list[tuple[dict[str, Any], tuple[str, ...]]]] = {}
        unresolved_for_cause: list[str] = []
        for state in cause_states:
            path_id = str(state.get("incoming_path_id", ""))
            raw_sequence = path_nodes.get(path_id)
            sequence = tuple(str(node_id) for node_id in raw_sequence) if raw_sequence else ()
            if not sequence:
                unresolved_for_cause.append(path_id)
                unresolved_path_ids.add(path_id)
                continue
            if path_id in seed_ids:
                matching_origins = [path_id]
            else:
                matching_origins = []
                for seed_id in sorted(seed_ids):
                    seed_sequence = path_nodes.get(seed_id) or ()
                    normalized_seed = tuple(str(node_id) for node_id in seed_sequence)
                    if normalized_seed and sequence[:len(normalized_seed)] == normalized_seed:
                        matching_origins.append(seed_id)
            if len(matching_origins) != 1:
                unresolved_for_cause.append(path_id)
                unresolved_path_ids.add(path_id)
                continue
            origin_id = matching_origins[0]
            origin_groups.setdefault(origin_id, []).append((state, sequence))

        if unresolved_for_cause:
            variations.append({
                "cause_id": cause_id,
                "origin_id": None,
                "classification": "unavailable",
                "classification_status": "PARTIAL",
                "unresolved_path_ids": sorted(set(unresolved_for_cause)),
                "state_count": len(unresolved_for_cause),
                "states": [],
            })

        for origin_id in sorted(origin_groups):
            entries = origin_groups[origin_id]
            signatures = {
                (
                    str((state.get("cause") or {}).get("support_status", "UNKNOWN")),
                    str((state.get("cause") or {}).get("causal_status", "UNKNOWN")),
                    (state.get("cause") or {}).get("localized_node_id"),
                    (state.get("cause") or {}).get("root_node_id"),
                    bool((state.get("cause") or {}).get("should_propagate_upstream", True)),
                )
                for state, _sequence in entries
            }
            if len(signatures) <= 1:
                continue

            sequences = sorted({sequence for _state, sequence in entries})
            proper_prefixes = {
                sequence[:index]
                for sequence in sequences
                for index in range(1, len(sequence))
            }
            terminal_sequences = [sequence for sequence in sequences if sequence not in proper_prefixes]
            by_sequence: dict[tuple[str, ...], list[dict[str, Any]]] = {}
            for state, sequence in entries:
                if sequence in terminal_sequences:
                    by_sequence.setdefault(sequence, []).append(state)

            trie: dict[str, Any] = {"children": {}, "terminal": []}
            for sequence in terminal_sequences:
                cursor = trie
                for node_id in sequence:
                    cursor = cursor["children"].setdefault(node_id, {"children": {}, "terminal": []})
                cursor["terminal"].extend(by_sequence.get(sequence, []))

            branch_points = 0
            cause_conflict_count = 0
            cause_conflict_group_count = 0

            def walk_trie(cursor: dict[str, Any], prefix: tuple[str, ...]) -> dict[str, Any]:
                nonlocal branch_points, cause_conflict_count, cause_conflict_group_count
                nonlocal total_conflict_pairs, total_conflict_groups, truncated
                counts = {"SUPPORTED": 0, "ROOT_CAUSE": 0, "REFUTED": 0}
                representatives: dict[str, dict[str, Any]] = {}
                for state in sorted(
                    cursor["terminal"],
                    key=lambda item: str(item.get("incoming_path_id", "")),
                ):
                    label = verified_state(state)
                    if label is not None:
                        counts[label] += 1
                        representatives.setdefault(label, state)
                child_summaries = []
                for child_id in sorted(cursor["children"]):
                    child_summary = walk_trie(
                        cursor["children"][child_id], prefix + (str(child_id),),
                    )
                    child_summaries.append((str(child_id), child_summary))
                    for label in counts:
                        counts[label] += child_summary["counts"][label]
                        if label in child_summary["representatives"]:
                            representatives.setdefault(label, child_summary["representatives"][label])

                if len(child_summaries) > 1:
                    branch_points += 1
                    specs = (
                        ("SUPPORTED_VS_REFUTED", "SUPPORTED"),
                        ("VALID_ROOT_VS_REFUTED", "ROOT_CAUSE"),
                    )
                    for conflict_kind, positive_label in specs:
                        total_positive = sum(item[1]["counts"][positive_label] for item in child_summaries)
                        total_refuted = sum(item[1]["counts"]["REFUTED"] for item in child_summaries)
                        within_branch = sum(
                            item[1]["counts"][positive_label] * item[1]["counts"]["REFUTED"]
                            for item in child_summaries
                        )
                        pair_count = total_positive * total_refuted - within_branch
                        if pair_count <= 0:
                            continue
                        cause_conflict_count += pair_count
                        cause_conflict_group_count += 1
                        positive_children = [
                            child for child in child_summaries
                            if child[1]["counts"][positive_label]
                        ]
                        refuted_children = [
                            child for child in child_summaries
                            if child[1]["counts"]["REFUTED"]
                        ]
                        positive_child = positive_children[0] if positive_children else None
                        refuted_child = next(
                            (child for child in refuted_children
                             if positive_child is not None and child[0] != positive_child[0]),
                            None,
                        )
                        if refuted_child is None and refuted_children:
                            refuted_child = refuted_children[0]
                            positive_child = next(
                                (child for child in positive_children if child[0] != refuted_child[0]),
                                None,
                            )
                        if positive_child is None or refuted_child is None:
                            continue
                        total_conflict_pairs += pair_count
                        total_conflict_groups += 1
                        if len(conflicts) >= max_conflict_records:
                            truncated = True
                            continue
                        positive_state = positive_child[1]["representatives"][positive_label]
                        refuted_state = refuted_child[1]["representatives"]["REFUTED"]
                        conflict = {
                            "cause_id": cause_id,
                            "origin_id": origin_id,
                            "branching_point_node_id": prefix[-1] if prefix else None,
                            "common_prefix_node_ids": list(prefix),
                            "conflict_kind": conflict_kind,
                            "conflicting_path_pair_count": pair_count,
                            "path_a": summarize_state(
                                positive_state,
                                tuple(str(n) for n in path_nodes.get(str(positive_state.get("incoming_path_id", "")), ())),
                            ),
                            "path_b": summarize_state(
                                refuted_state,
                                tuple(str(n) for n in path_nodes.get(str(refuted_state.get("incoming_path_id", "")), ())),
                            ),
                            "audit_basis": {
                                "path_lineage_source": "MaadPathCauseStatePolicy.path_nodes",
                                "state_source": "SharedDiagnosisEngine outgoing path states",
                                "positive_state_validation": (
                                    "exact localized Action evidence and no root rejection"
                                    if positive_label == "ROOT_CAUSE"
                                    else "accepted SUPPORTED state with exact localized Action evidence"
                                ),
                                "refuted_state_validation": "accepted path-scoped REFUTED state",
                            },
                        }
                        conflicts.append(conflict)

                return {"counts": counts, "representatives": representatives}

            walk_trie(trie, ())
            state_rows = [
                summarize_state(state, sequence)
                for state, sequence in sorted(
                    entries,
                    key=lambda item: (item[1], str(item[0].get("incoming_path_id", ""))),
                )
            ]
            state_rows_truncated = len(state_rows) > _MAAD_PATH_VARIATION_STATE_LIMIT
            variation_states_truncated = variation_states_truncated or state_rows_truncated
            if branch_points:
                classification = "divergent_conflicting" if cause_conflict_count else "divergent_nonconflicting"
            elif len(sequences) == 1:
                classification = "same_node_route_variation"
            else:
                classification = "linear_evolution"
            variations.append({
                "cause_id": cause_id,
                "origin_id": origin_id,
                "classification": classification,
                "classification_status": "PARTIAL" if partial_coverage else "COMPLETE",
                "state_count": len(state_rows),
                "distinct_state_signature_count": len(signatures),
                "branch_point_count": branch_points,
                "branch_conflict_group_count": cause_conflict_group_count,
                "states": state_rows[:_MAAD_PATH_VARIATION_STATE_LIMIT],
                "truncated": state_rows_truncated,
                "omitted_state_count": max(0, len(state_rows) - _MAAD_PATH_VARIATION_STATE_LIMIT),
            })

    analysis_status = (
        "PARTIAL"
        if partial_coverage or unresolved_path_ids or truncated or variation_states_truncated
        else "COMPLETE"
    )
    return {
        "version": _MAAD_PATH_CONFLICT_ANALYSIS_VERSION,
        "status": analysis_status,
        "cause_path_state_variations": variations,
        "cause_path_branch_conflicts": conflicts,
        "cause_path_conflicts": conflicts,
        "truncated": truncated,
        "variation_states_truncated": variation_states_truncated,
        "max_reported_conflict_records": max_conflict_records,
        "total_conflict_group_count": total_conflict_groups,
        "reported_conflict_group_count": len(conflicts),
        "omitted_conflict_group_count": max(0, total_conflict_groups - len(conflicts)),
        "observed_conflicting_path_pair_count": total_conflict_pairs,
        "unresolved_path_state_count": len(unresolved_path_ids),
    }


def diagnose_graph(
    graph: dict[str, Any],
    judge: MAADJudge,
    *,
    cascade_enabled: bool = True,
    cascade_call_mode: Literal["combined", "separate"] = "combined",
    separate_order: Literal["cascade_then_local", "local_then_cascade"] = "cascade_then_local",
    traversal_mode: str = "bfs",
    prompt_context_policy: Literal["legacy", "ranked_budget"] = "legacy",
    max_prompt_upstream_chars: int = 6000,
    max_causes: int = 8,
    cause_state_mode: Literal["legacy", "path"] = "legacy",
    max_path_states: int = 4096,
    dump_visible_context: bool = False,
    debug_output_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Run the single supported MAAD Action diagnosis through SharedDiagnosisEngine."""
    _walk_for_blocked_keys(graph)
    raw_nodes = graph.get("nodes") or []
    if not raw_nodes:
        raise ValueError("Graph has no nodes")
    node_map = {str(node["node_id"]): node for node in raw_nodes}
    if len(node_map) != len(raw_nodes):
        raise ValueError("Graph contains duplicate node IDs")
    action_nodes = sorted(
        [node for node in raw_nodes if int(node.get("step_index", -1)) >= 0],
        key=lambda node: int(node["step_index"]),
    )
    if not action_nodes:
        raise ValueError("Graph contains no original MAADBench action nodes")
    if [int(node["step_index"]) for node in action_nodes] != list(range(len(action_nodes))):
        raise ValueError("Original action step_index values must be exactly 0..N-1")
    if cascade_call_mode not in {"combined", "separate"}:
        raise ValueError("cascade_call_mode must be 'combined' or 'separate'")
    if separate_order not in {"cascade_then_local", "local_then_cascade"}:
        raise ValueError("separate_order must be 'cascade_then_local' or 'local_then_cascade'")
    if prompt_context_policy not in {"legacy", "ranked_budget"}:
        raise ValueError("prompt_context_policy must be 'legacy' or 'ranked_budget'")
    if cause_state_mode not in {"legacy", "path"}:
        raise ValueError("cause_state_mode must be 'legacy' or 'path'")
    if max_path_states < 1:
        raise ValueError("max_path_states must be positive")
    if cause_state_mode == "path" and not cascade_enabled:
        raise ValueError("MAADBench path cause state requires cascade_enabled=True")
    cascade_prevalidation_active = bool(
        cause_state_mode == "path"
        and cascade_enabled
        and cascade_call_mode == "separate"
        and separate_order == "cascade_then_local"
    )
    if dump_visible_context:
        if not isinstance(judge, MAADJudge):
            raise ValueError("Visible-context dumping requires MAADJudge so it can capture final messages before transport")
        if debug_output_dir is None:
            raise ValueError("debug_output_dir is required when dump_visible_context=True")
        judge.dump_visible_context_dir = Path(debug_output_dir).expanduser()

    edges = [
        edge for edge in (graph.get("edges") or [])
        if str(edge.get("src")) in node_map and str(edge.get("dst")) in node_map
    ]
    query_source = graph.get("query")
    if not query_source:
        query_source = next((
            node.get("metadata") for node in raw_nodes
            if str(node.get("semantic_role") or "").lower() == "query"
            and isinstance(node.get("metadata"), str) and node.get("metadata").strip()
        ), "")
    query = _truncate(query_source or "", 2200)
    ordered_ids = _backward_order(action_nodes, edges, traversal_mode)
    by_id = {str(node["node_id"]): node for node in action_nodes}
    action_id_to_node_id = {
        str(node["action_id"]): str(node["node_id"])
        for node in action_nodes if node.get("action_id") is not None
    }

    def ask_model(instruction: str, payload: dict[str, Any], schema: type[BaseModel], *, stage: str, node_id: str) -> BaseModel:
        metadata = {
            "trace_id": str(graph.get("trace_id", "")),
            "node_id": node_id,
            "stage": stage,
            "capture_mode": "LIVE_PRE_TRANSPORT",
        }
        if isinstance(judge, MAADJudge):
            return judge.ask(instruction, payload, schema, system_rules=SYSTEM_RULES, debug_metadata=metadata)
        return judge.ask(instruction, payload, schema, system_rules=SYSTEM_RULES)

    hypotheses = HypothesisResult(
        has_failure=False,
        final_error="Cascade disabled",
        causes=[],
        reasoning="Cascade inspection was not requested.",
    )
    seed_causes: list[Cause] = []
    known_cause_ids: set[str] = set()
    state_policy = (
        MaadPathCauseStatePolicy(max_path_states=max_path_states, max_causes=max_causes)
        if cause_state_mode == "path"
        else MaadLegacyCauseStatePolicy(max_causes=max_causes)
    )

    def initialize_hypotheses(_entry_node: dict[str, Any]) -> list[Cause]:
        nonlocal hypotheses, seed_causes
        final = by_id[ordered_ids[0]]
        upstream = _select_upstream_context(
            str(final["node_id"]), edges, node_map,
            policy=prompt_context_policy,
            char_budget=max_prompt_upstream_chars,
        )
        payload = {
            "query": query,
            "last_node": visible_node(final),
            "upstream": upstream,
        }
        instruction = (
            "FINAL HYPOTHESIS STAGE. Inspect the final observed action, not an oracle success label. "
            "Return has_failure, final_error, causes, reasoning. Create 0-4 concrete candidate causes "
            "using IDs C1, C2, ... only when visible evidence suggests a downstream failure. "
            "Each cause must include cause_id, description, confidence, support_status, causal_status, "
            "should_propagate_upstream, localized_node_id, root_node_id, evidence."
        )
        hypotheses = ask_model(
            instruction, payload, HypothesisResult,
            stage="final_hypothesis", node_id=str(final["node_id"]),
        )
        hypotheses.causes = [
            _canonicalize_cause_node_refs(cause, action_id_to_node_id)
            for cause in hypotheses.causes
        ]
        if cause_state_mode == "path":
            # Final hypotheses are candidates, not verified localizations. Keep
            # the initial Cause×Path seed neutral until an action supplies an
            # explicit path-scoped relation and evidence.
            hypotheses.causes = [
                cause.model_copy(update={
                    "support_status": "HYPOTHESIS",
                    "causal_status": "UNKNOWN",
                    "localized_node_id": None,
                    "root_node_id": None,
                    "should_propagate_upstream": True,
                })
                for cause in hypotheses.causes
            ]
        seed_causes = [cause for cause in hypotheses.causes if cause.support_status != "REFUTED"][:max_causes]
        known_cause_ids.update(str(cause.cause_id) for cause in seed_causes)
        return seed_causes

    def prepare_node(node_id: str, node: dict[str, Any], incoming: list[dict[str, Any]], _position: int) -> dict[str, Any]:
        upstream = _select_upstream_context(
            node_id, edges, node_map,
            policy=prompt_context_policy,
            char_budget=max_prompt_upstream_chars,
        )
        downstream = _select_downstream_context(node_id, edges, node_map)
        payload = {
            "query": query,
            "current_node": visible_node(node),
            "upstream": upstream,
            "downstream": downstream,
        }
        incoming_path_states = []
        if cause_state_mode == "path":
            incoming_path_states = [
                {
                    "incoming_path_id": str(item.get("incoming_path_id", "")),
                    "cause": Cause.model_validate(item.get("cause") or {}).model_dump(),
                }
                for item in incoming if isinstance(item, Mapping)
            ]
            payload["active_downstream_cause_paths"] = incoming_path_states
            incoming_causes = [Cause.model_validate(item.get("cause") or {}) for item in incoming_path_states]
        else:
            incoming_causes = [Cause.model_validate(item) for item in incoming]
        local_instruction = (
            "NODE-LOCAL INSPECTION. Judge this action independently from later outcomes. "
            "Return has_error, errors, anomaly_score, status, confidence, reasoning. "
            "anomaly_score is a continuous [0,1] action ranking score for AUROC, not a calibrated probability. "
            "Each error must include location equal to the current node_id, exact evidence, description, and impact."
        )
        cascade_instruction = (
            "CASCADE INSPECTION. For each active downstream cause, return a cause_relations entry using "
            "INTRODUCES, PROPAGATES, AMPLIFIES, RECOVERS, UNRELATED, REFUTES, or UNCERTAIN. "
            "Return status, confidence, reasoning, cause_relations, cause_updates, stopped_cause_ids. "
            "Update only changed causes and preserve cause_id. Use the exact current node_id for locations. "
            "If there are no active causes, do not invent one unless this node independently introduces a supported anomaly."
        )
        if cause_state_mode == "path":
            cascade_instruction = (
                "PATH-SCOPED CASCADE INSPECTION. Treat each (cause_id, incoming_path_id) as an "
                "independent cause path. Return each cause_relations item and cause_updates item with "
                "the exact incoming_path_id from active_downstream_cause_paths; never broadcast by cause_id. "
                "Unknown, missing, or ambiguous path IDs are invalid. stopped_cause_ids is legacy and "
                "must be empty when a cause has multiple incoming paths. For a genuinely new cause with "
                "no incoming path, use incoming_path_id=LOCAL_SEED:<current node_id>, a unique cause_id, "
                "relation=INTRODUCES, support_status=SUPPORTED, and exact evidence from the current "
                "visible action. A seed is accepted only after deterministic validation. A Root claim also "
                "requires INTRODUCES, a matching path update, and exact current-action evidence. Graph "
                "connectivity alone is not evidence of causality. Return the path-scoped schema."
            )
        return {
            "node": node,
            "node_id": node_id,
            "incoming": incoming_causes,
            "incoming_path_states": incoming_path_states,
            "common_payload": payload,
            "local_instruction": local_instruction,
            "cascade_instruction": cascade_instruction,
        }

    def inspect_local(prepared: dict[str, Any]) -> LocalResult:
        local = ask_model(
            prepared["local_instruction"], prepared["common_payload"], LocalResult,
            stage="local_only" if not cascade_enabled else "local_separate",
            node_id=prepared["node_id"],
        )
        _canonicalize_local_error_locations(local, action_id_to_node_id)
        return local

    def inspect_local_with_cascade(
        prepared: dict[str, Any], cascade: CascadeResult | CascadeResultPath,
    ) -> LocalResult:
        if cascade is None:
            raise TypeError("Cascade-conditioned Local inspection requires a CascadeResult")
        cascade_for_local: Any = cascade
        if cascade_prevalidation_active:
            incoming_snapshot = copy.deepcopy(prepared.get("incoming_path_states") or [])
            preview, validation_audit, pending_seeds = preview_cascade_delta(
                prepared, incoming_snapshot, cascade,
            )
            prepared["_cascade_local_prevalidation"] = {
                "cascade": preview,
                "audit": validation_audit,
                "pending_local_seeds": pending_seeds,
                "incoming_snapshot": incoming_snapshot,
                "path_state_count_before": state_policy.path_state_count,
                "known_cause_ids_before": set(known_cause_ids),
            }
            cascade_for_local = _maad_path_local_evidence_view(
                preview,
                validation_audit,
                prepared["common_payload"]["current_node"],
            )
        payload = {
            **prepared["common_payload"],
            "cascade_evidence": _sanitize_cascade_evidence(cascade_for_local),
        }
        instruction = prepared["local_instruction"] + """

CASCADE-CONDITIONED LOCAL ERROR DETECTION. The cascade_evidence field is tentative
causal evidence about downstream failure paths, not ground truth. Use it as additional
context while independently deciding whether the CURRENT action itself has an observable
anomaly. INTRODUCES/PROPAGATES/ROOT_CAUSE does not automatically imply has_error=true;
REFUTES/RECOVERS/UNRELATED does not rule out an independent local anomaly. Inspect the
current action's own input, output, and tool result. Do not copy a causal claim as a local
error without current-action evidence. If no active causes are present, still perform
normal local inspection. A Cascade status of CORRECT does not mean the current action is
locally correct. Return the unchanged LocalResult schema.
"""
        local = ask_model(
            instruction, payload, LocalResult,
            stage="local_after_cascade", node_id=prepared["node_id"],
        )
        _canonicalize_local_error_locations(local, action_id_to_node_id)
        return local

    def inspect_cascade(prepared: dict[str, Any], local: Optional[LocalResult]) -> CascadeResult | CascadeResultPath:
        payload = {
            **prepared["common_payload"],
            "new_cause_id_hint": f"S{prepared['node'].get('step_index')}",
        }
        if cause_state_mode == "path":
            cascade = ask_model(
                prepared["cascade_instruction"], payload, CascadeResultPath,
                stage="cascade_path_separate", node_id=prepared["node_id"],
            )
            cascade.cause_updates = [
                item.model_copy(update={
                    "cause": _canonicalize_cause_node_refs(item.cause, action_id_to_node_id),
                })
                for item in cascade.cause_updates
            ]
        else:
            payload["active_downstream_hypotheses"] = [
                cause.model_dump() for cause in prepared["incoming"]
            ]
            cascade = ask_model(
                prepared["cascade_instruction"], payload, CascadeResult,
                stage="cascade_separate", node_id=prepared["node_id"],
            )
            cascade.cause_updates = [
                _canonicalize_cause_node_refs(cause, action_id_to_node_id)
                for cause in cascade.cause_updates
            ]
        return cascade

    def inspect_combined(prepared: dict[str, Any]) -> NodeDiagnosis:
        payload = {
            **prepared["common_payload"],
            "new_cause_id_hint": f"S{prepared['node'].get('step_index')}",
        }
        if cause_state_mode == "path":
            combined_model = CombinedPathResult
            stage = "combined_path_local_cascade"
            payload["active_downstream_cause_paths"] = prepared["incoming_path_states"]
        else:
            combined_model = CombinedResult
            stage = "combined_local_cascade"
            payload["active_downstream_hypotheses"] = [
                cause.model_dump() for cause in prepared["incoming"]
            ]
        combined = ask_model(
            "Return two objects named local and cascade.\n"
            + prepared["local_instruction"] + "\n" + prepared["cascade_instruction"],
            payload, combined_model,
            stage=stage, node_id=prepared["node_id"],
        )
        _canonicalize_local_error_locations(combined.local, action_id_to_node_id)
        if cause_state_mode == "path":
            combined.cascade.cause_updates = [
                item.model_copy(update={
                    "cause": _canonicalize_cause_node_refs(item.cause, action_id_to_node_id),
                })
                for item in combined.cascade.cause_updates
            ]
        else:
            combined.cascade.cause_updates = [
                _canonicalize_cause_node_refs(cause, action_id_to_node_id)
                for cause in combined.cascade.cause_updates
            ]
        return NodeDiagnosis(local=combined.local, cascade=combined.cascade)

    def _prepare_cascade_delta(
        prepared: dict[str, Any], incoming: list[dict[str, Any]], diagnosis: NodeDiagnosis,
        *, commit_local_seeds: bool = True,
    ):
        if cause_state_mode != "path":
            return None
        cascade = diagnosis.cascade
        if not isinstance(cascade, CascadeResultPath):
            raise TypeError("MAADBench path cause mode requires CascadeResultPath")

        node_id = str(prepared["node_id"])
        visible = prepared["common_payload"]["current_node"]
        active_keys = {
            (
                str((state.get("cause") or {}).get("cause_id", "")),
                str(state.get("incoming_path_id", "")),
            )
            for state in incoming if isinstance(state, Mapping)
        }
        previous_cause_by_key = {
            (
                str((state.get("cause") or {}).get("cause_id", "")),
                str(state.get("incoming_path_id", "")),
            ): dict(state.get("cause") or {})
            for state in incoming
            if isinstance(state, Mapping) and isinstance(state.get("cause"), Mapping)
        }
        active_cause_ids = {cause_id for cause_id, _path_id in active_keys}
        audit: list[dict[str, Any]] = []
        relation_by_key: dict[tuple[str, str], PathScopedCauseRelation] = {}
        updates_by_key: dict[tuple[str, str], PathScopedCauseUpdate] = {}

        for relation in cascade.cause_relations:
            key = (str(relation.cause_id), str(relation.incoming_path_id))
            if key in relation_by_key:
                audit.append({
                    "kind": "relation", "cause_id": key[0], "incoming_path_id": key[1],
                    "status": "DUPLICATE_PATH_RELATION_REJECTED",
                })
                continue
            relation_by_key[key] = relation
        for update in cascade.cause_updates:
            key = (str(update.cause.cause_id), str(update.incoming_path_id))
            if key in updates_by_key:
                audit.append({
                    "kind": "update", "cause_id": key[0], "incoming_path_id": key[1],
                    "status": "DUPLICATE_PATH_UPDATE_REJECTED",
                })
                continue
            updates_by_key[key] = update

        # Path-mode never accepts the unscoped legacy stop list: it addresses
        # only a cause ID and could otherwise stop sibling paths together.
        if cascade.stopped_cause_ids:
            audit.extend({
                "kind": "stopped_cause_id", "cause_id": str(cause_id),
                "status": "UNSCOPED_STOP_REJECTED_USE_PATH_RELATION",
            } for cause_id in cascade.stopped_cause_ids)
            cascade = cascade.model_copy(update={"stopped_cause_ids": []})

        # Updates must have the matching path relation. This prevents a
        # response from changing a Cause while omitting its path judgment.
        for key in list(updates_by_key):
            if key in active_keys and key not in relation_by_key:
                audit.append({
                    "kind": "update", "cause_id": key[0], "incoming_path_id": key[1],
                    "status": "UPDATE_WITHOUT_PATH_RELATION_REJECTED",
                })
                updates_by_key.pop(key, None)

        accepted_seeds: set[tuple[str, str]] = set()
        pending_local_seeds: list[dict[str, Any]] = []
        projected_path_state_count = state_policy.path_state_count
        for key, update in list(updates_by_key.items()):
            cause_id, path_id = key
            if key in active_keys:
                continue
            expected_seed_path = f"LOCAL_SEED:{node_id}"
            relation = relation_by_key.get(key)
            cause = update.cause.model_copy(deep=True)
            if path_id != expected_seed_path:
                audit.append({
                    "kind": "update", "cause_id": cause_id, "incoming_path_id": path_id,
                    "status": "UNKNOWN_OR_UNMATCHED_PATH_REJECTED",
                })
                updates_by_key.pop(key, None)
                continue
            if cause_id in active_cause_ids or cause_id in known_cause_ids:
                audit.append({
                    "kind": "local_seed", "cause_id": cause_id, "incoming_path_id": path_id,
                    "status": "LOCAL_SEED_CAUSE_ID_COLLISION",
                })
                updates_by_key.pop(key, None)
                continue
            exact_evidence = _has_exact_visible_action_evidence(visible, cause.evidence)
            seed_valid = bool(
                relation is not None
                and relation.relation == "INTRODUCES"
                and cause.support_status == "SUPPORTED"
                and exact_evidence
                and cause.localized_node_id in (None, node_id)
            )
            if not seed_valid:
                audit.append({
                    "kind": "local_seed", "cause_id": cause_id, "incoming_path_id": path_id,
                    "status": "LOCAL_SEED_REJECTED_UNSUPPORTED",
                    "reason": "requires_unique_cause_introduces_relation_and_exact_current_action_evidence",
                })
                updates_by_key.pop(key, None)
                relation_by_key.pop(key, None)
                continue

            root_claimed = bool(relation.is_root_cause or cause.causal_status == "ROOT_CAUSE")
            root_valid = bool(
                root_claimed
                and relation.is_root_cause
                and cause.causal_status == "ROOT_CAUSE"
                and cause.root_node_id in (None, node_id)
                and exact_evidence
            )
            cause.localized_node_id = node_id
            if root_claimed and not root_valid:
                cause.causal_status = "UNKNOWN"
                cause.root_node_id = None
                cause.should_propagate_upstream = True
                relation = relation.model_copy(update={"is_root_cause": False})
                relation_by_key[key] = relation
                audit.append({
                    "kind": "root_cause", "cause_id": cause_id, "incoming_path_id": path_id,
                    "status": "ROOT_CAUSE_REJECTED_UNSUPPORTED",
                    "reason": "requires_introduces_supported_exact_current_action_evidence_and_matching_node",
                })
            elif root_valid:
                cause.support_status = "SUPPORTED"
                cause.causal_status = "ROOT_CAUSE"
                cause.root_node_id = node_id
                cause.should_propagate_upstream = False
            else:
                cause.should_propagate_upstream = True

            if commit_local_seeds:
                state = state_policy.register_local_seed(cause, node_id=node_id)
                if state is None:
                    updates_by_key.pop(key, None)
                    relation_by_key.pop(key, None)
                    continue
                incoming.append(state)
                known_cause_ids.add(cause_id)
            else:
                if projected_path_state_count >= state_policy.max_path_states:
                    audit.append({
                        "kind": "local_seed", "cause_id": cause_id,
                        "incoming_path_id": path_id,
                        "status": "PATH_STATE_BUDGET_TRUNCATED",
                        "limit": state_policy.max_path_states,
                    })
                    updates_by_key.pop(key, None)
                    relation_by_key.pop(key, None)
                    continue
                pending_local_seeds.append({
                    "cause": cause.model_copy(deep=True),
                    "node_id": node_id,
                })
                projected_path_state_count += 1
            accepted_seeds.add(key)

        accepted_updates: list[PathScopedCauseUpdate] = []
        for key, relation in relation_by_key.items():
            if key not in active_keys and key not in accepted_seeds:
                audit.append({
                    "kind": "relation", "cause_id": key[0], "incoming_path_id": key[1],
                    "status": "UNKNOWN_OR_UNMATCHED_PATH_REJECTED",
                })
        for key, update in updates_by_key.items():
            if key not in active_keys and key not in accepted_seeds:
                continue
            relation = relation_by_key.get(key)
            cause = update.cause.model_copy(deep=True)
            root_claimed = bool(
                (relation is not None and relation.is_root_cause)
                or cause.causal_status == "ROOT_CAUSE"
            )
            if root_claimed:
                root_valid = bool(
                    relation is not None
                    and relation.relation == "INTRODUCES"
                    and relation.is_root_cause
                    and cause.support_status == "SUPPORTED"
                    and cause.localized_node_id in (None, node_id)
                    and cause.root_node_id in (None, node_id)
                    and _has_exact_visible_action_evidence(visible, cause.evidence)
                )
                if root_valid:
                    cause.localized_node_id = node_id
                    cause.root_node_id = node_id
                    cause.causal_status = "ROOT_CAUSE"
                    cause.support_status = "SUPPORTED"
                    cause.should_propagate_upstream = False
                else:
                    # Reject the unverified root/localization update as a whole.
                    # Preserve only evidence-backed state already held for this
                    # exact path; otherwise restore an unverified candidate.
                    previous_cause = previous_cause_by_key.get(key)
                    if previous_cause is None:
                        cause.support_status = "HYPOTHESIS"
                        cause.localized_node_id = None
                        cause.root_node_id = None
                        cause.evidence = None
                        cause.causal_status = "UNKNOWN"
                    else:
                        cause.support_status = str(
                            previous_cause.get("support_status", "HYPOTHESIS")
                        )
                        cause.localized_node_id = previous_cause.get("localized_node_id")
                        cause.root_node_id = previous_cause.get("root_node_id")
                        cause.evidence = previous_cause.get("evidence")
                        cause.causal_status = str(
                            previous_cause.get("causal_status", "UNKNOWN")
                        )
                    if (
                        cause.support_status != "REFUTED"
                        and cause.causal_status != "ROOT_CAUSE"
                    ):
                        cause.should_propagate_upstream = True
                    if relation is not None:
                        relation = relation.model_copy(update={"is_root_cause": False})
                        relation_by_key[key] = relation
                    audit.append({
                        "kind": "root_cause", "cause_id": key[0], "incoming_path_id": key[1],
                        "status": "ROOT_CAUSE_REJECTED_UNSUPPORTED",
                        "reason": "requires_introduces_supported_exact_current_action_evidence_and_matching_node",
                    })
            accepted_updates.append(update.model_copy(update={"cause": cause}))

        # Keep only relations whose path and cause exactly match a valid incoming or
        # validated local-seed state. Missing path IDs are never broadcast.
        accepted_update_keys = {
            (str(update.cause.cause_id), str(update.incoming_path_id))
            for update in accepted_updates
        }
        for key, relation in list(relation_by_key.items()):
            if (
                relation.is_root_cause
                and (key in active_keys or key in accepted_seeds)
                and key not in accepted_update_keys
            ):
                relation_by_key[key] = relation.model_copy(update={"is_root_cause": False})
                audit.append({
                    "kind": "root_cause", "cause_id": key[0], "incoming_path_id": key[1],
                    "status": "ROOT_CAUSE_REJECTED_WITHOUT_PATH_UPDATE",
                })
        accepted_relation_map = {
            key: relation_by_key[key]
            for key in relation_by_key
            if key in active_keys or key in accepted_seeds
        }
        accepted_updates = [
            update for update in accepted_updates
            if (
                (str(update.cause.cause_id), str(update.incoming_path_id)) in active_keys
                or (str(update.cause.cause_id), str(update.incoming_path_id)) in accepted_seeds
            )
        ]
        # A new cause must have a matching INTRODUCES relation, including local seeds.
        accepted_updates = [
            update for update in accepted_updates
            if (str(update.cause.cause_id), str(update.incoming_path_id)) in active_keys
            or (
                (relation := accepted_relation_map.get((str(update.cause.cause_id), str(update.incoming_path_id))))
                is not None and relation.relation == "INTRODUCES"
            )
        ]
        diagnosis.cascade = cascade.model_copy(update={
            "cause_relations": list(accepted_relation_map.values()),
            "cause_updates": accepted_updates,
        })
        return diagnosis.cascade, audit, pending_local_seeds

    def preview_cascade_delta(
        prepared: dict[str, Any], incoming: list[dict[str, Any]], cascade: CascadeResultPath,
    ) -> tuple[CascadeResultPath, list[dict[str, Any]], list[dict[str, Any]]]:
        """Return a validated view and pending seeds without mutating path state."""
        if cause_state_mode != "path":
            raise RuntimeError("Cascade prevalidation is only available for MAAD Path mode")
        shadow_diagnosis = NodeDiagnosis(
            local=None,
            cascade=cascade.model_copy(deep=True),
        )
        validated, audit, pending = _prepare_cascade_delta(
            prepared, incoming, shadow_diagnosis, commit_local_seeds=False,
        )
        if not isinstance(validated, CascadeResultPath):
            raise TypeError("MAAD Path prevalidation returned an invalid Cascade result")
        return validated, audit, pending

    def prepare_cascade_delta(
        prepared: dict[str, Any], incoming: list[dict[str, Any]], diagnosis: NodeDiagnosis,
    ):
        if cause_state_mode != "path":
            return None
        preview = prepared.get("_cascade_local_prevalidation")
        if preview is not None:
            if incoming != preview["incoming_snapshot"]:
                raise RuntimeError("MAAD Path incoming states changed after Cascade prevalidation")
            if state_policy.path_state_count != preview["path_state_count_before"]:
                raise RuntimeError("MAAD Path budget changed after Cascade prevalidation")
            if known_cause_ids != preview["known_cause_ids_before"]:
                raise RuntimeError("MAAD Path Cause IDs changed after Cascade prevalidation")
            for pending in preview["pending_local_seeds"]:
                seed = state_policy.register_local_seed(
                    pending["cause"], node_id=pending["node_id"],
                )
                if seed is None:
                    raise RuntimeError("A previewed MAAD Path seed could not be committed")
                incoming.append(seed)
                known_cause_ids.add(str(pending["cause"].cause_id))
            prepared.pop("_cascade_local_prevalidation", None)
            diagnosis.cascade = preview["cascade"].model_copy(deep=True)
            diagnosis.metadata.update({
                "cascade_evidence_validation_applied": True,
                "cascade_evidence_validation_version": _MAAD_CASCADE_LOCAL_EVIDENCE_VALIDATION_VERSION,
                "rejected_root_count": len({
                    (str(event.get("cause_id", "")), str(event.get("incoming_path_id", "")))
                    for event in preview["audit"]
                    if str(event.get("status", "")).startswith("ROOT_CAUSE_REJECTED")
                }),
            })
            return preview["cascade"], list(preview["audit"])
        committed, audit, _pending = _prepare_cascade_delta(prepared, incoming, diagnosis)
        diagnosis.metadata.update({
            "cascade_evidence_validation_applied": False,
            "cascade_evidence_validation_version": (
                "not_applicable_combined"
                if diagnosis.metadata.get("diagnosis_mode") == "combined"
                else "not_applicable_without_cascade_first"
            ),
            "rejected_root_count": 0,
        })
        return committed, audit

    predictions: list[dict[str, Any]] = []

    def after_node(node_id: str, prepared: dict[str, Any], diagnosis: NodeDiagnosis,
                   active_before: list[dict[str, Any]], active_after: list[dict[str, Any]]) -> dict[str, Any]:
        local = diagnosis.local
        cascade = diagnosis.cascade
        prediction = {
            "step_index": int(prepared["node"]["step_index"]),
            "node_id": node_id,
            "action_id": prepared["node"].get("action_id"),
            "action_type": prepared["node"].get("action_type"),
            "puzzle_id": prepared["node"].get("puzzle_id"),
            "semantic_role": prepared["node"].get("semantic_role"),
            "anomaly_score": max(0.0, min(1.0, float(local.anomaly_score))),
            "local_pred_label_0_5": int(float(local.anomaly_score) >= 0.5),
            "has_error": bool(local.has_error),
            "confidence": max(0.0, min(1.0, float(local.confidence))),
            "status": local.status,
            "reasoning": local.reasoning,
            "errors": [item.model_dump() for item in local.errors],
            "diagnosis_mode": diagnosis.metadata.get("diagnosis_mode"),
            "separate_order": diagnosis.metadata.get("separate_order"),
            "local_stage": diagnosis.metadata.get("local_stage"),
            "cascade_stage": diagnosis.metadata.get("cascade_stage"),
            "first_visit": diagnosis.metadata.get("first_visit"),
            "local_from_cache": diagnosis.metadata.get("local_from_cache", False),
            "cascade_recomputed": diagnosis.metadata.get("cascade_recomputed", False),
            "local_conditioned_on_cascade": diagnosis.metadata.get("local_conditioned_on_cascade", False),
            "local_reconditioned_on_new_cascade": diagnosis.metadata.get(
                "local_reconditioned_on_new_cascade", False,
            ),
        }
        if cause_state_mode == "path":
            prediction.update({
                "cascade_evidence_validation_applied": diagnosis.metadata.get(
                    "cascade_evidence_validation_applied", False,
                ),
                "cascade_evidence_validation_version": diagnosis.metadata.get(
                    "cascade_evidence_validation_version", "not_applicable",
                ),
                "rejected_root_count": diagnosis.metadata.get("rejected_root_count", 0),
            })
        if cause_state_mode == "path":
            prediction.update({
                "cascade_evidence_validation_applied": diagnosis.metadata.get(
                    "cascade_evidence_validation_applied", False,
                ),
                "cascade_evidence_validation_version": diagnosis.metadata.get(
                    "cascade_evidence_validation_version", "not_applicable",
                ),
                "rejected_root_count": diagnosis.metadata.get("rejected_root_count", 0),
            })
        if cause_state_mode == "path":
            prediction["local_anomaly_score"] = max(0.0, min(1.0, float(local.anomaly_score)))
            if diagnosis.metadata.get("first_visit", True):
                predictions.append(prediction)
        else:
            predictions.append(prediction)

        def dump_states(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if cause_state_mode == "path":
                return [
                    {
                        "incoming_path_id": str(item.get("incoming_path_id", "")),
                        "cause": Cause.model_validate(item.get("cause") or {}).model_dump(),
                    }
                    for item in states if isinstance(item, Mapping)
                ]
            return [Cause.model_validate(item).model_dump() for item in states]

        history_record = {
            "node_id": node_id,
            "step_index": int(prepared["node"]["step_index"]),
            "local": local.model_dump(),
            "cascade": cascade.model_dump() if cascade is not None else None,
            "active_before": dump_states(active_before),
            "active_after": dump_states(active_after),
            "diagnosis_mode": diagnosis.metadata.get("diagnosis_mode"),
            "separate_order": diagnosis.metadata.get("separate_order"),
            "local_stage": diagnosis.metadata.get("local_stage"),
            "cascade_stage": diagnosis.metadata.get("cascade_stage"),
            "first_visit": diagnosis.metadata.get("first_visit"),
            "local_from_cache": diagnosis.metadata.get("local_from_cache", False),
            "cascade_recomputed": diagnosis.metadata.get("cascade_recomputed", False),
            "local_conditioned_on_cascade": diagnosis.metadata.get("local_conditioned_on_cascade", False),
            "local_reconditioned_on_new_cascade": diagnosis.metadata.get(
                "local_reconditioned_on_new_cascade", False,
            ),
        }
        if cause_state_mode == "path":
            history_record["cause_state_mode"] = "path"
            history_record.update({
                "cascade_evidence_validation_applied": diagnosis.metadata.get(
                    "cascade_evidence_validation_applied", False,
                ),
                "cascade_evidence_validation_version": diagnosis.metadata.get(
                    "cascade_evidence_validation_version", "not_applicable",
                ),
                "rejected_root_count": diagnosis.metadata.get("rejected_root_count", 0),
            })
            history_record.update({
                "cascade_evidence_validation_applied": diagnosis.metadata.get(
                    "cascade_evidence_validation_applied", False,
                ),
                "cascade_evidence_validation_version": diagnosis.metadata.get(
                    "cascade_evidence_validation_version", "not_applicable",
                ),
                "rejected_root_count": diagnosis.metadata.get("rejected_root_count", 0),
            })
        return {"history_record": history_record}

    class _MAADSharedAdapter:
        pass

    adapter = _MAADSharedAdapter()
    adapter.cascade_enabled = cascade_enabled
    adapter.cascade_call_mode = cascade_call_mode
    adapter.separate_order = separate_order
    adapter.initialize_hypotheses = initialize_hypotheses
    adapter.prepare_node = prepare_node
    adapter.inspect_local = inspect_local
    adapter.inspect_local_with_cascade = inspect_local_with_cascade
    adapter.inspect_cascade = inspect_cascade
    adapter.inspect_combined = inspect_combined
    adapter.after_node = after_node
    if cause_state_mode == "path":
        adapter.prepare_cascade_delta = prepare_cascade_delta

    action_node_ids = {str(node["node_id"]) for node in action_nodes}
    shared_edges = [
        edge for edge in edges
        if str(edge.get("src")) in action_node_ids and str(edge.get("dst")) in action_node_ids
    ]
    shared_run = SharedDiagnosisEngine(max_causes=max_causes).run(
        graph={"nodes": action_nodes, "edges": shared_edges},
        adapter=adapter,
        cause_policy=state_policy,
        entry_node_id=ordered_ids[0],
        traversal_mode=traversal_mode,
        schedule_mode="all_nodes",
        node_ids=[str(node["node_id"]) for node in reversed(action_nodes)],
        edge_priority=EDGE_PRIORITY,
    )
    if shared_run.schedule != ordered_ids:
        raise RuntimeError(
            "Shared MAAD scheduler differs from the frozen legacy order: "
            f"shared={shared_run.schedule!r}; legacy={ordered_ids!r}"
        )
    all_causes: dict[str, Cause] = {}
    active_cause_path_states: list[dict[str, Any]] = []
    if cause_state_mode == "path":
        by_path_key: dict[tuple[str, str], dict[str, Any]] = {}
        for node_id in shared_run.schedule:
            for raw_state in shared_run.outgoing_causes.get(node_id, []):
                if not isinstance(raw_state, Mapping) or not isinstance(raw_state.get("cause"), Mapping):
                    continue
                cause = Cause.model_validate(raw_state["cause"])
                key = (str(cause.cause_id), str(raw_state.get("incoming_path_id", "")))
                by_path_key[key] = {
                    "incoming_path_id": key[1],
                    "cause": cause.model_dump(),
                    "last_node_id": str(node_id),
                }
        active_cause_path_states = [
            by_path_key[key] for key in sorted(by_path_key)
        ]
        root_cause_path_states = [
            state for state in active_cause_path_states
            if state["cause"].get("causal_status") == "ROOT_CAUSE"
        ]
        cause_path_audit = [
            event for item in shared_run.history for event in item.get("cause_state_audit", [])
            if isinstance(event, Mapping)
        ]
        conflict_analysis = _analyze_maad_path_conflicts(
            active_cause_path_states,
            state_policy.path_nodes,
            cause_path_audit,
            node_map,
            partial_coverage=bool(
                shared_run.partial_coverage
                or shared_run.path_state_budget_exhausted
                or shared_run.cycle_nodes
                or (shared_run.graph_validation or {}).get("worklist_visit_budget_exhausted", False)
            ),
        )
        cause_path_conflicts = conflict_analysis["cause_path_branch_conflicts"]
        root_causes = [state["cause"] for state in root_cause_path_states]
    else:
        for node_id in shared_run.schedule:
            for raw_cause in shared_run.outgoing_causes.get(node_id, []):
                cause = Cause.model_validate(raw_cause)
                previous = all_causes.get(cause.cause_id)
                if previous is None or cause.causal_status == "ROOT_CAUSE" or cause.confidence > previous.confidence:
                    all_causes[cause.cause_id] = cause.model_copy(deep=True)
        root_causes = [cause.model_dump() for cause in all_causes.values() if cause.causal_status == "ROOT_CAUSE"]
    predictions.sort(key=lambda item: item["step_index"])
    history = sorted(shared_run.history, key=lambda item: item["step_index"])
    result = {
        "trace_id": str(graph.get("trace_id", "")),
        "dataset": "MAADBench-Full",
        "diagnosis_mode": "disabled" if not cascade_enabled else cascade_call_mode,
        "separate_order": (
            separate_order
            if cascade_enabled and cascade_call_mode == "separate"
            else "not_applicable"
        ),
        "model": judge.model,
        "traversal_mode": resolve_traversal_mode(traversal_mode, default="bfs"),
        "prompt_context_policy": prompt_context_policy,
        "schema_version": (
            "LocalResultV1_path_cause_v2_cascade_local_prevalidation_v1"
            if cascade_prevalidation_active else
            "LocalResultV1_path_cause_v2"
            if cause_state_mode == "path" else "LocalResultV1_legacy_roles"
        ),
        "cascade_enabled": cascade_enabled,
        "action_predictions": predictions,
        "initial_hypotheses": hypotheses.model_dump() if cascade_enabled else None,
        "root_causes": root_causes,
        "diagnosis_history": history,
        "coverage": {
            "scheduled_nodes": len(shared_run.schedule),
            "diagnosed_nodes": len(history),
            "partial": bool(shared_run.partial_coverage),
            "blocked_nodes": list(shared_run.blocked_nodes),
        },
        "status": "PARTIAL" if shared_run.partial_coverage else "SUCCESS",
    }
    if cause_state_mode == "path":
        if cascade_prevalidation_active:
            result["cascade_evidence_validation_version"] = _MAAD_CASCADE_LOCAL_EVIDENCE_VALIDATION_VERSION
        expected_action_steps = list(range(len(action_nodes)))
        predicted_steps = [
            int(item["step_index"]) for item in predictions
            if isinstance(item, Mapping) and item.get("step_index") is not None
        ]
        predicted_node_ids = [
            str(item.get("node_id", "")) for item in predictions if isinstance(item, Mapping)
        ]
        local_complete = (
            len(predicted_steps) == len(expected_action_steps)
            and sorted(predicted_steps) == expected_action_steps
            and len(set(predicted_node_ids)) == len(action_nodes)
            and set(predicted_node_ids) == action_node_ids
        )
        local_status = "SUCCESS" if local_complete else (
            "PARTIAL" if predictions else "FAILED"
        )
        cause_path_status = "PARTIAL" if shared_run.partial_coverage else "COMPLETE"
        partial_reasons = []
        if shared_run.cycle_nodes:
            partial_reasons.append("CYCLE_DETECTED")
        if shared_run.path_state_budget_exhausted:
            partial_reasons.append("PATH_STATE_BUDGET_EXHAUSTED")
        if (shared_run.graph_validation or {}).get("worklist_visit_budget_exhausted"):
            partial_reasons.append("WORKLIST_VISIT_BUDGET_EXHAUSTED")
        if shared_run.blocked_nodes:
            partial_reasons.append("UNVISITED_ACTION_NODES")
        if not local_complete:
            partial_reasons.append("LOCAL_ACTION_PREDICTION_INCOMPLETE")
        result["local_prediction_status"] = local_status
        result["cause_state_mode"] = "path"
        result["cause_path_coverage_status"] = cause_path_status
        result["cause_path_partial_reasons"] = partial_reasons
        if not local_complete:
            result["status"] = "PARTIAL" if predictions else "FAILED"
        result["coverage"].update({
            "local_prediction_status": local_status,
            "expected_local_action_count": len(action_nodes),
            "completed_local_action_count": len(set(predicted_steps)),
            "missing_local_node_ids": sorted(action_node_ids - set(predicted_node_ids)),
            "duplicate_local_step_indices": sorted({
                step for step in predicted_steps if predicted_steps.count(step) > 1
            }),
            "cause_path_coverage_status": cause_path_status,
            "cause_path_partial_reasons": partial_reasons,
        })
        result.update({
            "max_cause_path_states": max_path_states,
            "cause_path_state_count": state_policy.path_state_count,
            "active_cause_path_states": active_cause_path_states,
            "root_cause_path_states": root_cause_path_states,
            "cause_path_conflicts": cause_path_conflicts,
            "cause_path_branch_conflicts": conflict_analysis["cause_path_branch_conflicts"],
            "cause_path_state_variations": conflict_analysis["cause_path_state_variations"],
            "conflict_analysis_version": _MAAD_PATH_CONFLICT_ANALYSIS_VERSION,
            "cause_path_conflicts_semantics": "strict verified conflicts on independent sibling branches; alias of cause_path_branch_conflicts",
            "cause_path_conflict_analysis_status": conflict_analysis["status"],
            "cause_path_conflict_analysis": {
                key: value for key, value in conflict_analysis.items()
                if key not in {
                    "cause_path_state_variations", "cause_path_branch_conflicts",
                    "cause_path_conflicts",
                }
            },
            "cause_path_audit": cause_path_audit,
            "cause_path_partial": bool(shared_run.partial_coverage),
            "worklist_visit_budget_exhausted": bool(
                (shared_run.graph_validation or {}).get("worklist_visit_budget_exhausted", False)
            ),
            "path_state_budget_exhausted": shared_run.path_state_budget_exhausted,
            "cycle_nodes": list(shared_run.cycle_nodes),
            "method_version": _MAAD_PATH_METHOD_VERSION,
            "cause_path_policy_version": _MAAD_PATH_METHOD_VERSION,
        })
    return result

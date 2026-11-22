import json
import hashlib
import os
import re
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field
from ...common.llm import LLMServer, parse_structured
from ...common.cascade import CascadeRelation
from ...common.schemas import NodeLocalInspectionBase
from ...common.diagnosis import (
    EdgePolicy,
    NodeDiagnosis,
    SharedDiagnosisEngine,
    TrailLegacyCauseStatePolicy,
    TrailPathCauseStatePolicy,
)
from ...common.diagnosis.path_state import (
    make_path_id,
    prune_cause_path_states,
    reconcile_cause_path_states,
)
from ...common.traversal import TraversalMode, pop_frontier, push_frontier, resolve_traversal_mode
from ...evaluation.trail import ALL_CATEGORIES as TRAIL_LEAF_CATEGORIES
try:
    import httpx
except ImportError:
    httpx = None

# =====================================================================
# 0. Client
# =====================================================================

proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
url = os.environ.get("OPENAI_BASE_URL")
api_keys = os.environ.get("OPENAI_API_KEY", "")

http_client = None
if httpx is not None and proxy:
    http_client = httpx.Client(proxy=proxy)

_client: Optional[Any] = None
_llm_server: Optional[LLMServer] = None


def get_llm_server() -> LLMServer:
    global _client, _llm_server
    if _llm_server is None:
        _llm_server = LLMServer(
            api_key=api_keys,
            url=url,
            client=_client,
            http_client=http_client,
            resolve_env=False,
            missing_key_message="OpenAI API key not found. Define global api_keys or set OPENAI_API_KEY.",
            missing_package_message="The openai package is not installed in this Python environment.",
        )
        _client = _llm_server.client
    return _llm_server


# =====================================================================
# 1. TRAIL taxonomy and output models
# =====================================================================

TRAIL_TAXONOMY = r"""
You are an expert TRAIL agent trace evaluator.

Allowed leaf categories:
Language-only,
Tool-related,
Poor Information Retrieval,
Incorrect Memory Usage,
Tool Output Misinterpretation,
Incorrect Problem Identification,
Tool Selection Errors,
Formatting Errors,
Instruction Non-compliance,
Tool Definition Issues,
Environment Setup Errors,
Rate Limiting,
Authentication Errors,
Service Errors,
Resource Not Found,
Resource Exhaustion,
Timeout Issues,
Context Handling Failures,
Resource Abuse,
Goal Deviation,
Task Orchestration

CATEGORY RULES:
Language-only:
Unsupported/fabricated natural language. No fake tool result involved.
Tool-related:
Fake tool execution, fabricated tool output, or invented tool capability.
Poor Information Retrieval:
Wrong, irrelevant, incomplete, outdated, or inappropriate retrieval/query.
Incorrect Memory Usage:
Memory was incorrectly stored, retrieved, updated, or applied.
Tool Output Misinterpretation:
A real tool result exists, but the agent misunderstands or misuses it.
Incorrect Problem Identification:
The agent misunderstood the user's goal, constraints, or success criteria.
Tool Selection Errors:
The wrong tool was selected. Do not use when the tool is correct but the query is bad.
Formatting Errors:
The answer content may be correct, but required output format/schema is violated.
Instruction Non-compliance:
An explicit user/system instruction was ignored. Do not use only because the answer is incorrect.
Tool Definition Issues:
The tool interface or description itself is incorrect.
Environment Setup Errors:
Permission, API key, configuration, or environment prevents execution.
Rate Limiting:
Explicit rate limit evidence, such as HTTP 429.
Authentication Errors:
Explicit authentication failure, such as HTTP 401/403.
Service Errors:
Explicit provider/server failure, such as HTTP 5xx.
Resource Not Found:
Explicit missing resource, such as HTTP 404.
Resource Exhaustion:
Explicit memory/resource/quota exhaustion.
Timeout Issues:
Explicit timeout or deadline failure.
Context Handling Failures:
Previous context/state was lost or mishandled.
Resource Abuse:
Excessive or redundant resource/tool usage.
Goal Deviation:
The task was understood, but execution later drifted away.
Task Orchestration:
Failure in decomposition, coordination, or multi-step planning.

IMPORTANT DISTINCTIONS:
- Wrong information obtained -> Poor Information Retrieval.
- Correct information obtained but misunderstood -> Tool Output Misinterpretation.
- Wrong understanding of task -> Incorrect Problem Identification.
- Correct understanding but later drift -> Goal Deviation.
- Wrong tool -> Tool Selection Errors.
- Fake tool event -> Tool-related.
- Real tool result misused -> Tool Output Misinterpretation.
"""

TRAILCategoryType = Literal[
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

TRAIL_CATEGORY_NAMES: List[str] = list(TRAILCategoryType.__args__)
TRAIL_CATEGORY_LIST_TEXT = "\n".join(
    f"- {category}" for category in TRAIL_CATEGORY_NAMES
)















def _node_contains_exact_evidence(node: Any, evidence: str) -> bool:
    """Return whether exact node-visible evidence occurs in a nested payload."""
    if not evidence:
        return False
    if isinstance(node, str):
        return evidence in node
    if isinstance(node, dict):
        return any(_node_contains_exact_evidence(value, evidence) for value in node.values())
    if isinstance(node, list):
        return any(_node_contains_exact_evidence(value, evidence) for value in node)
    return False


_CASCADE_LOCAL_EVIDENCE_VALIDATION_VERSION = "trail_h1_cascade_local_evidence_v1"
_REJECTED_ROOT_EXPLANATION = "Unverified root claim omitted; assess current-span evidence independently."


def _trail_h1_local_evidence_view(
    cascade: Any,
    audit: List[Dict[str, Any]],
    current_node: Dict[str, Any],
) -> Dict[str, Any]:
    """Build an allowlisted Local view after deterministic H1 delta validation."""
    rejected_reasons: Dict[Tuple[str, str], str] = {}
    for event in audit:
        if str(event.get("status", "")).startswith("ROOT_CAUSE_REJECTED"):
            key = (
                str(event.get("cause_id", "")),
                str(event.get("incoming_path_id", "")),
            )
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
            "explanation": _REJECTED_ROOT_EXPLANATION if rejected else relation.explanation,
            "is_root_cause": bool(relation.is_root_cause and not rejected),
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
        if rejected and evidence and not _node_contains_exact_evidence(current_node, evidence):
            evidence = None
        cause_view = {
            "cause_id": cause.cause_id,
            "description": _REJECTED_ROOT_EXPLANATION if rejected else cause.description,
            "category": cause.category,
            "confidence": cause.confidence,
            "support_status": cause.support_status,
            "causal_status": "UNKNOWN" if rejected else cause.causal_status,
            "should_propagate_upstream": (
                cause.support_status != "REFUTED" if rejected else cause.should_propagate_upstream
            ),
            "localized_span_id": (
                cause.localized_span_id
                if not rejected or cause.localized_span_id in (None, current_node.get("span_id"), current_node.get("node_id"))
                else None
            ),
            "root_span_id": None if rejected else cause.root_span_id,
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

    root_rejections = [
        {
            "cause_id": cause_id or None,
            "incoming_path_id": path_id or None,
            "reason_code": reason,
        }
        for (cause_id, path_id), reason in sorted(rejected_reasons.items())
    ]
    has_unverified_root = bool(root_rejections)
    has_rejected_non_root_relation = any(
        event.get("status") == "NON_ROOT_RELATION_REJECTED"
        for event in audit
    )
    status = cascade.status
    if status == "ROOT_CAUSE" and not valid_root_keys:
        status = "UNCERTAIN"
    return {
        "validation": {
            "applied": True,
            "version": _CASCADE_LOCAL_EVIDENCE_VALIDATION_VERSION,
            "validated_tentative_root_keys": [list(key) for key in sorted(valid_root_keys)],
            "rejected_root_claims": root_rejections,
        },
        "status": status,
        "confidence": cascade.confidence,
        "reasoning": (
            _REJECTED_ROOT_EXPLANATION
            if has_unverified_root or has_rejected_non_root_relation
            else cascade.reasoning
        ),
        "cause_relations": relations,
        "cause_updates": updates,
        "stopped_cause_ids": list(cascade.stopped_cause_ids),
    }


class TRAILErrorItem(BaseModel):
    category: TRAILCategoryType = Field(description="Strict TRAIL leaf category.")
    location: str = Field(description="Exact OpenTelemetry span_id where the error occurred.")
    evidence: str = Field(
        description=(
            "Exact substring copied from this node's stored trace evidence, including "
            "messages, arguments, status, errors, and return values."
        )
    )
    description: str = Field(description="Detailed explanation of what went wrong and why.")
    impact: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="Impact level of this error on the trace execution."
    )


class TRAILScoresItem(BaseModel):
    reliability_score: int = Field(ge=1, le=5)
    reliability_reasoning: str
    security_score: int = Field(ge=1, le=5)
    security_reasoning: str
    instruction_adherence_score: int = Field(ge=1, le=5)
    instruction_adherence_reasoning: str
    plan_opt_score: int = Field(ge=1, le=5)
    plan_opt_reasoning: str
    overall: float = Field(ge=0.0, le=5.0)


class TRAILDiagnosisOutput(BaseModel):
    errors: List[TRAILErrorItem] = Field(default_factory=list)
    scores: List[TRAILScoresItem]


class TRAILScoresOnlyOutput(BaseModel):
    scores: List[TRAILScoresItem]


class SingleNodeLocalInspection(NodeLocalInspectionBase[TRAILErrorItem]):
    has_error: bool = Field(description="True if this node itself contains an agent error.")






# =====================================================================
# 2. Cascade-failure models
# =====================================================================

class FailureCause(BaseModel):
    cause_id: str = Field(description="Stable ID such as C1 or C2.")
    description: str = Field(description="Causal hypothesis explaining the downstream failure.")
    category: Optional[TRAILCategoryType] = Field(
        default=None,
        description="Best matching TRAIL category, if this cause can be mapped to one.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    support_status: Literal["HYPOTHESIS", "SUPPORTED", "REFUTED"] = Field(
        default="HYPOTHESIS",
        description=(
            "HYPOTHESIS: possible cause not yet supported by a concrete span; "
            "SUPPORTED: this cause has concrete evidence at one or more spans; "
            "REFUTED: current evidence no longer supports this cause."
        ),
    )
    causal_status: Literal["UNKNOWN", "PROXIMATE_CAUSE", "ROOT_CAUSE"] = Field(
        default="UNKNOWN",
        description=(
            "UNKNOWN: causal position is not established; PROXIMATE_CAUSE: this cause "
            "is supported at a downstream/proximate span but may have an upstream "
            "mechanism; ROOT_CAUSE: earliest sufficiently supported cause on this "
            "failure path."
        ),
    )
    should_propagate_upstream: bool = Field(
        default=True,
        description=(
            "Whether this cause should continue to be searched for in upstream nodes. "
            "Set false when the cause is refuted, recovered, unrelated to this path, "
            "or already established as a root cause."
        ),
    )
    localized_span_id: Optional[str] = Field(
        default=None,
        description=(
            "First concrete span where this cause was supported. Once set, later "
            "processing must not overwrite it for the same cause_id."
        ),
    )
    root_span_id: Optional[str] = Field(
        default=None,
        description=(
            "Span identified as the root cause for this failure path, if any. This is "
            "separate from localized_span_id because a proximate supported span may "
            "not be the root cause."
        ),
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Exact evidence substring when the cause is supported by a node.",
    )

class CascadeCauseRelation(BaseModel):
    cause_id: str
    relation: CascadeRelation
    explanation: str
    is_root_cause: bool = Field(
        description="True only when this node is the earliest supported cause on this path."
    )


class PathScopedCauseRelation(BaseModel):
    cause_id: str
    incoming_path_id: str
    relation: CascadeRelation
    explanation: str
    is_root_cause: bool = Field(
        description="True only when this node is the earliest supported cause on this incoming path."
    )


class PathScopedCauseUpdate(BaseModel):
    incoming_path_id: str
    cause: FailureCause


class FinalFailureAnalysis(BaseModel):
    has_failure: bool = Field(
        description="Whether the final node/output has an observable failure."
    )
    final_error: str = Field(
        description="What is wrong with the final answer/output, or why no final failure is established."
    )
    causes: List[FailureCause] = Field(default_factory=list)
    reasoning: str


class CascadeNodeInspection(BaseModel):
    status: Literal["CORRECT", "ROOT_CAUSE", "PASSIVE_ERROR", "UNCERTAIN"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    cause_relations: List[CascadeCauseRelation] = Field(default_factory=list)
    cause_updates: List[FailureCause] = Field(
        default_factory=list,
        description=(
            "Delta updates only: include causes whose description, support status, "
            "causal status, propagation flag, evidence, localized span, root span, "
            "category, or confidence changed at this node."
        ),
    )
    stopped_cause_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Cause IDs that should stop on this path because they were refuted, "
            "recovered, unrelated, or established as terminal root causes."
        ),
    )

class CombinedNodeInspection(BaseModel):
    local: SingleNodeLocalInspection
    cascade: CascadeNodeInspection


class CascadeNodeInspectionH1(BaseModel):
    status: Literal["CORRECT", "ROOT_CAUSE", "PASSIVE_ERROR", "UNCERTAIN"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    cause_relations: List[PathScopedCauseRelation] = Field(default_factory=list)
    cause_updates: List[PathScopedCauseUpdate] = Field(default_factory=list)
    stopped_cause_ids: List[str] = Field(default_factory=list)


class CombinedNodeInspectionH1(BaseModel):
    local: SingleNodeLocalInspection
    cascade: CascadeNodeInspectionH1












# =====================================================================
# 3. Config and sanitization
# =====================================================================

class DiagnoserConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_id_key: str = "span_id"
    edge_src_key: str = "src"
    edge_dst_key: str = "dst"
    edge_kind_key: str = "kind"

    data_flow_kinds: Set[str] = {"data_flow", "data", "dependency"}
    execution_flow_kinds: Set[str] = {"execution_flow", "execution", "control_flow"}
    parent_kinds: Set[str] = {"contains", "parent_child", "subspan"}
    traversal_mode: TraversalMode = "bfs"
    cascade_call_mode: Literal["combined", "separate"] = "combined"
    separate_order: Literal["cascade_then_local", "local_then_cascade"] = "cascade_then_local"
    prompt_context_policy: Literal["legacy", "ranked_budget", "prioritized_two_kinds"] = "legacy"
    trail_schedule_scope: Literal["legacy", "all_nodes"] = "legacy"
    final_hypothesis_policy: Literal["gated", "retain_candidates"] = "gated"

    is_final_node_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    is_agent_node_fn: Optional[Callable[[Dict[str, Any]], bool]] = None

    llm_model: str = "gpt-4o-mini"
    max_string_length: int = 1500
    max_current_node_string_length: int = 1000
    max_context_node_string_length: int = 400
    max_downstream_node_string_length: int = 300
    max_prompt_upstream_chars: int = 3000
    max_causes: int = 8
    cause_state_mode: Literal["H0", "H1"] = "H0"
    max_cause_path_states: int = 8
    max_downstream_node: int = 6

    cascade_enabled: bool = True
    enable_scoring: bool = False
    use_llm_error_synthesis: bool = False
    include_ground_truth_in_node_prompt: bool = False

    keep_node_fields: Optional[Set[str]] = {
        "span_id",
        "node_id",
        "id",
        "name",
        "span_name",
        "semantic_role",
        "metadata",
        "input",
        "output",
        "input_messages",
        "output_messages",
    }



def truncate_text(text: Any, max_len: int = 1500) -> Any:
    if not isinstance(text, str):
        return text
    text = _redact_sensitive_text(text)
    if len(text) <= max_len:
        return text
    half = max_len // 2
    return f"{text[:half]}\n... [TRUNCATED {len(text) - max_len} CHARS] ...\n{text[-half:]}"


_SENSITIVE_FIELD_NAMES = {
    "apikey",
    "apitoken",
    "authorization",
    "authtoken",
    "bearertoken",
    "accesstoken",
    "refreshtoken",
    "sessiontoken",
    "idtoken",
    "token",
    "password",
    "passwd",
    "secret",
    "clientsecret",
    "privatekey",
    "credential",
    "credentials",
    "cookie",
    "setcookie",
}


def _is_sensitive_field_name(value: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(value).lower())
    return normalized in _SENSITIVE_FIELD_NAMES


def _redact_sensitive_text(value: str) -> str:
    value = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
        r"\1[REDACTED]",
        value,
    )
    value = re.sub(
        r"(?i)((?:api[_-]?key|api[_-]?token|access[_-]?token|refresh[_-]?token|password|client[_-]?secret)\s*[:=]\s*[\"']?)[^\s,;\"']+",
        r"\1[REDACTED]",
        value,
    )
    return re.sub(
        r"(?i)\b(?:sk-ant-|sk-|ghp_|github_pat_)[A-Za-z0-9_-]{16,}",
        "[REDACTED]",
        value,
    )


def sanitize_nested_value(
    value: Any,
    string_limit: int,
    max_list_items: int = 100,
) -> Any:
    """Sanitize nested trace data without applying the node-level field whitelist."""
    if isinstance(value, str):
        return truncate_text(value, string_limit)
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if item is None:
                continue
            cleaned[key] = (
                "[REDACTED]"
                if _is_sensitive_field_name(key)
                else sanitize_nested_value(item, string_limit, max_list_items)
            )
        return cleaned
    if isinstance(value, list):
        items = value
        if len(items) > 4 and all(isinstance(item, dict) for item in items):
            items = items[:2] + items[-2:]
        elif len(items) > max_list_items:
            half = max_list_items // 2
            items = items[:half] + items[-half:]
        return [
            sanitize_nested_value(item, string_limit, max_list_items)
            for item in items
            if item is not None
        ]
    return value


_CASCADE_ID_FIELDS = {
    "cause_id", "incoming_path_id", "path_id", "root_span_id", "localized_span_id",
}


def sanitize_cascade_evidence(value: Any, string_limit: int) -> Any:
    """Bound and redact structured Cascade output without dropping path records."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif isinstance(value, Mapping):
        value = dict(value)
    else:
        raise TypeError("cascade_evidence must be a structured Cascade result")

    def clean(item: Any, field_name: Optional[str] = None) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): (
                    "[REDACTED]"
                    if _is_sensitive_field_name(key)
                    else clean(child, str(key))
                )
                for key, child in item.items()
            }
        if isinstance(item, list):
            return [clean(child) for child in item]
        if isinstance(item, str):
            if field_name in _CASCADE_ID_FIELDS:
                return item
            return truncate_text(item, string_limit)
        return item

    return clean(value)



def sanitize_node_for_llm(
    node: Dict[str, Any],
    config: DiagnoserConfig,
    max_string_length: Optional[int] = None,
) -> Dict[str, Any]:
    """保留嵌套诊断证据，同时限制节点整体长度。"""
    if not isinstance(node, dict):
        return node

    field_limit = (
        max_string_length
        if max_string_length is not None
        else config.max_string_length
    )

    # 原有字符串长度参数，同时决定单节点总体预算。
    total_limit = min(8000, max(1200, field_limit * 5))

    max_depth = 4
    max_dict_keys = 16
    max_list_items = 8

    important_nested = {
        "content", "text", "value", "message", "messages",
        "role", "status", "status_code", "error",
        "exception", "error_message", "type", "name",
        "arguments", "result", "tool_calls", "tool_name",
        "input", "output", "function", "function_call",
        "code", "traceback",
    }

    def clean(value, depth, *, in_metadata=False):
        if isinstance(value, str):
            return truncate_text(value, field_limit)

        if isinstance(value, dict):
            if depth >= max_depth:
                return "[NESTED CONTENT OMITTED]"

            items = [
                (str(k), v)
                for k, v in value.items()
                if v is not None
            ]

            if in_metadata:
                # 避免与顶层 input/output 重复
                items = [
                    (k, v) for k, v in items
                    if not (
                        (k == "input.value"
                         and node.get("input") is not None)
                        or
                        (k == "output.value"
                         and node.get("output") is not None)
                    )
                ]

            # 优先保留诊断相关字段
            items.sort(
                key=lambda pair: (
                    0 if (
                        pair[0].lower() in important_nested
                        or pair[0].lower().split(".")[-1]
                        in important_nested
                    ) else 1
                )
            )

            result = {}
            for k, v in items[:max_dict_keys]:
                result[k] = (
                    "[REDACTED]"
                    if _is_sensitive_field_name(k)
                    else clean(v, depth + 1)
                )

            if len(items) > max_dict_keys:
                result["_omitted_field_count"] = (
                    len(items) - max_dict_keys
                )

            return result

        if isinstance(value, list):
            items = (
                value[:4] + value[-4:]
                if len(value) > max_list_items
                else value
            )
            result = [
                clean(item, depth + 1)
                for item in items
                if item is not None
            ]

            if len(value) > max_list_items:
                result.append(
                    f"[OMITTED {len(value) - max_list_items}"
                    " LIST ITEMS]"
                )

            return result

        return value

    fields = config.keep_node_fields

    sanitized = {
        str(k): (
            "[REDACTED]"
            if _is_sensitive_field_name(k)
            else clean(v, 0, in_metadata=(k == "metadata"))
        )
        for k, v in node.items()
        if v is not None and (
            fields is None or k in fields
        )
    }

    def as_json(value):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )

    if len(as_json(sanitized)) <= total_limit:
        return sanitized

    # 仅针对异常庞大的节点执行第二级压缩。
    identity_keys = (
        "span_id", "node_id", "id",
        "name", "span_name", "semantic_role",
    )
    compact = {
        k: sanitized[k]
        for k in identity_keys
        if k in sanitized
    }

    evidence_keys = (
        "input", "output", "input_messages",
        "output_messages", "metadata",
    )
    weights = {
        "input": 3,
        "output": 3,
        "input_messages": 1,
        "output_messages": 1,
        "metadata": 1,
    }
    present = [
        k for k in evidence_keys
        if k in sanitized
    ]

    available = max(
        0, total_limit - len(as_json(compact)) - 120
    )
    total_weight = (
        sum(weights[k] for k in present) or 1
    )

    for key in present:
        share = (
            available * weights[key] // total_weight
        )
        original = sanitized[key]

        if len(as_json({key: original})) <= share:
            compact[key] = original
            continue

        source = (
            original if isinstance(original, str)
            else as_json(original)
        )
        left, right = 0, len(source)
        best = None

        while left <= right:
            mid = (left + right) // 2
            candidate = {
                "_excerpt": truncate_text(source, mid),
                "_partial": True,
            }

            if len(as_json({key: candidate})) <= share:
                best = candidate
                left = mid + 1
            else:
                right = mid - 1

        if best is not None:
            compact[key] = best

    compact["_node_context_truncated"] = True

    for key in (
        "metadata", "output_messages",
        "input_messages", "output", "input",
    ):
        if len(as_json(compact)) <= total_limit:
            break
        compact.pop(key, None)

    if len(as_json(compact)) > total_limit:
        raise ValueError(
            "Node identifiers exceeded character budget"
        )

    return compact











def sanitize_context_list(
    context_list: List[Dict[str, Any]],
    config: DiagnoserConfig,
    max_string_length: Optional[int] = None,
) -> List[Dict[str, Any]]:
    cleaned_context = []
    for item in context_list:
        cleaned_item = item.copy()
        if "node" in cleaned_item and isinstance(cleaned_item["node"], dict):
            cleaned_item["node"] = sanitize_node_for_llm(
                cleaned_item["node"],
                config,
                max_string_length,
            )
        cleaned_context.append(cleaned_item)
    return cleaned_context


# =====================================================================
# 4. Helpers
# =====================================================================

def safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not value or not isinstance(value, str):
        return value

    text = value.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return value


def json_for_prompt(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def get_field_with_fallback(
    data: Dict[str, Any], keys: List[str], default: Any = None
) -> Any:
    if not isinstance(data, dict):
        return default
    for k in keys:
        if k in data and data[k] is not None:
            return data[k]
    return default


def extract_user_query_smart(node_or_data: Any) -> Optional[str]:
    if not node_or_data:
        return None

    if isinstance(node_or_data, list):
        for item in node_or_data:
            query = extract_user_query_smart(item)
            if query:
                return query
        return None

    if isinstance(node_or_data, str):
        text = node_or_data.strip()
        parsed = safe_json_loads(text)
        if isinstance(parsed, (dict, list)):
            return extract_user_query_smart(parsed)
        return text if text else None

    if isinstance(node_or_data, dict):
        direct_val = get_field_with_fallback(
            node_or_data,
            ["query", "user_query", "task", "prompt", "input", "content"],
        )
        if isinstance(direct_val, str) and direct_val.strip():
            return direct_val.strip()
        if isinstance(direct_val, (dict, list)):
            sub_query = extract_user_query_smart(direct_val)
            if sub_query:
                return sub_query

        for v in node_or_data.values():
            if isinstance(v, (dict, list)):
                res = extract_user_query_smart(v)
                if res:
                    return res

    return None


def extract_ground_truth_smart(node_or_data: Any) -> Optional[str]:
    if not node_or_data:
        return None

    if isinstance(node_or_data, str):
        parsed = safe_json_loads(node_or_data.strip())
        if isinstance(parsed, (dict, list)):
            return extract_ground_truth_smart(parsed)
        return None

    if isinstance(node_or_data, dict):
        value = get_field_with_fallback(
            node_or_data,
            [
                "ground_truth",
                "expected_answer",
                "correct_answer",
                "reference_answer",
                "gold_answer",
            ],
        )
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (dict, list)):
            text = json_for_prompt(value)
            return text if text.strip() else None

        for v in node_or_data.values():
            if isinstance(v, (dict, list)):
                found = extract_ground_truth_smart(v)
                if found:
                    return found

    return None


def normalize_cause_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def deduplicate_errors(errors: List[TRAILErrorItem]) -> List[TRAILErrorItem]:
    seen: Set[Tuple[str, str, str]] = set()
    result: List[TRAILErrorItem] = []
    for err in errors:
        key = (str(err.location), str(err.category), str(err.evidence))
        if key not in seen:
            seen.add(key)
            result.append(err)
    return result










def merge_causes(
    existing: List[FailureCause],
    returned: List[FailureCause],
    max_causes: int = 8,
) -> List[FailureCause]:
    """Merge incoming causes through the shared legacy state policy."""
    return [
        FailureCause.model_validate(item)
        for item in TrailLegacyCauseStatePolicy(max_causes=max_causes).merge_incoming(
            [cause.model_dump() for cause in [*existing, *returned]]
        )[:max_causes]
    ]




def to_serializable_causes(causes: List[FailureCause]) -> List[Dict[str, Any]]:
    return [c.model_dump() for c in causes]


def _aggregate_final_cause_states(
    outgoing_by_node: Mapping[str, List[Any]],
    *,
    path_scoped: bool,
) -> Tuple[Dict[Tuple[str, str], FailureCause], List[Dict[str, Any]]]:
    """Build the legacy cause summary and retain H1's node/path states.

    The legacy summary remains one item per cause identity for H0 compatibility.
    H1 callers additionally receive path-scoped state records so a state from a
    different branch cannot erase a supported root state in the final output.
    """
    causes_by_identity: Dict[Tuple[str, str], FailureCause] = {}
    path_states_by_identity: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for node_id, node_causes in outgoing_by_node.items():
        for item in node_causes:
            if path_scoped:
                if not isinstance(item, Mapping) or not isinstance(item.get("cause"), Mapping):
                    continue
                cause = FailureCause.model_validate(dict(item["cause"]))
                incoming_path_id = str(item.get("incoming_path_id", ""))
                path_key = (str(node_id), incoming_path_id, cause.cause_id)
                path_states_by_identity[path_key] = {
                    "span_id": str(node_id),
                    "incoming_path_id": incoming_path_id,
                    "cause": cause.model_dump(),
                }
            else:
                cause = FailureCause.model_validate(item)
            key = (cause.cause_id, normalize_cause_text(cause.description))
            causes_by_identity[key] = cause
    return causes_by_identity, list(path_states_by_identity.values())


def empty_score() -> TRAILScoresItem:
    return TRAILScoresItem(
        reliability_score=3,
        reliability_reasoning="No score model output was available.",
        security_score=3,
        security_reasoning="No score model output was available.",
        instruction_adherence_score=3,
        instruction_adherence_reasoning="No score model output was available.",
        plan_opt_score=3,
        plan_opt_reasoning="No score model output was available.",
        overall=3.0,
    )


# =====================================================================
# 5. LLMService
# =====================================================================

class LLMService:
    """TRAIL prompts, structured LLM requests, and response handling."""

    def __init__(
        self,
        config: DiagnoserConfig,
        llm_server: Optional[LLMServer] = None,
    ):
        self.config = config
        self.llm_server = llm_server

    def inspect_node_local(
        self,
        user_query: str,
        node: Dict[str, Any],
        upstream_nodes: List[Dict[str, Any]],
        downstream_nodes: List[Dict[str, Any]],
        cascade_evidence: Optional[Any] = None,
    ) -> SingleNodeLocalInspection:
        config = self.config
        clean_node = sanitize_node_for_llm(
            node,
            config,
            config.max_current_node_string_length or config.max_string_length,
        )
        clean_upstream = sanitize_context_list(
            upstream_nodes,
            config,
            config.max_context_node_string_length or config.max_string_length,
        )
        clean_downstream = sanitize_context_list(
            downstream_nodes,
            config,
            config.max_downstream_node_string_length or config.max_string_length,
        )
        span_id = str(
            get_field_with_fallback(clean_node, ["span_id", "node_id", "id"], "UNKNOWN_SPAN")
        )

        system_prompt = f"""
You are an expert AI Agent Trace Evaluator following TRAIL Dataset guidelines.

Allowed TRAIL leaf categories:
{TRAIL_CATEGORY_LIST_TEXT}

CURRENT TARGET SPAN ID: "{span_id}"

This call is NODE-LOCAL ERROR DETECTION ONLY.
Cascade/root-cause analysis is handled by a separate call and must not suppress
local TRAIL errors.

STRICT GUIDELINES:
1. Evaluate the current node objectively, regardless of whether it is a root cause,
   passive propagation, or unrelated to the final answer.
2. 'location' MUST be exactly "{span_id}".
3. 'evidence' MUST be an exact substring copied from the CURRENT NODE's stored
   trace evidence, including messages, arguments, status, errors, or return values.
   Never paraphrase evidence.
4. Use upstream/downstream context only to understand whether the current node's own
   content/action is wrong.
5. A trace can have multiple errors across different nodes. Report all distinct
   errors in this current node.
6. Use only allowed leaf categories.

"""
        clean_cascade_evidence = None
        if cascade_evidence is not None:
            clean_cascade_evidence = sanitize_cascade_evidence(
                cascade_evidence,
                config.max_current_node_string_length or config.max_string_length,
            )
            system_prompt += """
CASCADE-CONDITIONED LOCAL ERROR DETECTION:
The Cascade result below is tentative causal evidence about downstream failure paths,
not ground truth. Use it as additional context while independently deciding whether
the CURRENT span itself contains an observable TRAIL error. A PROPAGATES or ROOT_CAUSE
judgment does not automatically make this span erroneous. REFUTES, RECOVERS, or
UNRELATED does not rule out an independent local error. Inspect the current span's
own evidence, keep every error location at the current span ID, and use only the
official TRAIL leaf categories. If there are no active causes, still perform normal
local inspection. A Cascade status of CORRECT is not a judgment that this span is
locally correct. Do not copy a causal claim as an error without current-span evidence.
"""
        user_prompt = f"""
USER QUERY:
{user_query}

CURRENT TARGET NODE (Span ID: {span_id}):
{json_for_prompt(clean_node)}

UPSTREAM CONTEXT:
{json_for_prompt(clean_upstream)}

DOWNSTREAM CONTEXT:
{json_for_prompt(clean_downstream)}
"""
        if clean_cascade_evidence is not None:
            user_prompt += f"""

CASCADE EVIDENCE (TENTATIVE, NOT GROUND TRUTH):
{json_for_prompt(clean_cascade_evidence)}
"""


        server = self.llm_server or get_llm_server()
        response = parse_structured(
            server,
            model=config.llm_model,
            response_model=SingleNodeLocalInspection,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        parsed = response.choices[0].message.parsed
        for err in parsed.errors:
            err.location = span_id
        return parsed


    def initialize_failure_hypothesis(
        self,
        user_query: str,
        final_node: Dict[str, Any],
        ground_truth: Optional[str],
    ) -> FinalFailureAnalysis:
        config = self.config
        clean_final = sanitize_node_for_llm(final_node, config)
        span_id = str(
            get_field_with_fallback(clean_final, ["span_id", "node_id", "id"], "UNKNOWN_SPAN")
        )

        gt_mode = (
            "A ground-truth/reference answer is available. Use it as external correctness evidence."
            if ground_truth is not None
            else (
                "NO ground truth is available. Establish failure only from trace-internal "
                "evidence such as the user objective, tool outputs, explicit constraints, "
                "and contradictions in the trace."
            )
        )

        system_prompt = f"""
You are the FINAL-NODE STAGE of a TRAIL cascade-failure diagnosis.

{TRAIL_TAXONOMY}

TARGET SPAN ID: "{span_id}"

GROUND-TRUTH MODE:
{gt_mode}

TASK:
1. Decide whether the final output has an observable failure.
2. If failure exists, produce 1-5 causal hypotheses for backward analysis.
3. Use support_status for evidence support:
   - HYPOTHESIS: possible cause, but no concrete supporting span yet.
   - SUPPORTED: this final node provides concrete evidence for the cause.
   - REFUTED: do not use for initial final-node causes unless disproving a candidate.
4. Use causal_status for causal position:
   - UNKNOWN: causal position is not established.
   - PROXIMATE_CAUSE: final node is a supported/proximate source, but upstream
     causes may still exist.
   - ROOT_CAUSE: final node itself is the earliest sufficiently supported mechanism.
5. Keep localized_span_id null unless the final node itself provides concrete
   support for the cause.
6. Keep root_span_id null unless causal_status="ROOT_CAUSE".
7. If causal_status="ROOT_CAUSE", set support_status="SUPPORTED",
   localized_span_id="{span_id}", root_span_id="{span_id}", and
   should_propagate_upstream=false.
8. If causal_status="PROXIMATE_CAUSE", set support_status="SUPPORTED",
   localized_span_id="{span_id}", root_span_id=null, and
   should_propagate_upstream=true.
9. Otherwise set support_status="HYPOTHESIS", causal_status="UNKNOWN",
   localized_span_id=null, root_span_id=null, and should_propagate_upstream=true.
10. If no failure can be established, return has_failure=false and causes=[].
"""
        if config.final_hypothesis_policy == "retain_candidates":
            system_prompt = system_prompt.replace(
                "2. If failure exists, produce 1-5 causal hypotheses for backward analysis.",
                "2. Whether or not the final output is marked failed, produce only concrete, trace-grounded potential upstream causes; return zero candidates when none are justified.",
            ).replace(
                "10. If no failure can be established, return has_failure=false and causes=[].",
                "10. has_failure=false means no observable final-output failure is established; it does not veto a supported potential upstream cause. If no trace-grounded candidate is justified, return causes=[].",
            )

        user_prompt = f"""
USER QUERY:
{user_query}

GROUND TRUTH / CORRECT ANSWER:
{ground_truth if ground_truth is not None else "NOT PROVIDED"}

FINAL NODE (Span ID: {span_id}):
{json_for_prompt(clean_final)}
"""


        server = self.llm_server or get_llm_server()
        response = parse_structured(
            server,
            model=config.llm_model,
            response_model=FinalFailureAnalysis,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        parsed = response.choices[0].message.parsed
        return parsed

    def inspect_node_combined(
        self,
        user_query: str,
        node: Dict[str, Any],
        upstream_nodes: List[Dict[str, Any]],
        downstream_nodes: List[Dict[str, Any]],
        active_causes: List[FailureCause],
        ground_truth: Optional[str],
        cause_paths: Optional[List[Dict[str, Any]]] = None,
    ) -> CombinedNodeInspection:
        """Run local error detection and cascade attribution in one LLM call.
    The local channel is the only source for benchmark-visible errors. The
    cascade channel is root-cause metadata only and must not rewrite local errors.
    """
        config = self.config
        clean_node = sanitize_node_for_llm(
            node,
            config,
            config.max_current_node_string_length or config.max_string_length,
        )
        clean_upstream = sanitize_context_list(
            upstream_nodes,
            config,
            config.max_context_node_string_length or config.max_string_length,
        )
        clean_downstream = sanitize_context_list(
            downstream_nodes,
            config,
            config.max_downstream_node_string_length or config.max_string_length,
        )
        span_id = str(
            get_field_with_fallback(clean_node, ["span_id", "node_id", "id"], "UNKNOWN_SPAN")
        )
        ground_truth_block = ""
        if config.include_ground_truth_in_node_prompt and ground_truth is not None:
            ground_truth_block = f"""
GROUND TRUTH / CORRECT FINAL ANSWER:
{ground_truth}

Use this only as optional final-failure context. Do not use it to infer,
add, remove, or relabel local.errors unless the CURRENT TARGET NODE itself
contains evidence for that local error.
"""    

        path_scope_rules = ""
        if config.cause_state_mode == "H1" and cause_paths is not None:
            path_scope_rules = """
H1 PATH-SCOPED CAUSE STATE:
- The ACTIVE CAUSES block contains separate incoming_path_id entries. Treat each
  (cause_id, incoming_path_id) independently, even when descriptions match.
- Return a cause_relations item for each path you can assess and echo its exact
  incoming_path_id. Return updates as {"incoming_path_id": "...", "cause": {...}}.
- REFUTES, RECOVERS, and UNRELATED stop only the named incoming path. Never use
  stopped_cause_ids to stop multiple branches; use it only for a cause with one path.
- Do not broadcast a relation or update to another path. Missing evidence means
  UNCERTAIN and preserves that path's prior state.
"""
        system_prompt = f"""
You are an expert AI Agent Trace Evaluator following TRAIL Dataset guidelines.

Allowed TRAIL leaf categories:
{TRAIL_CATEGORY_LIST_TEXT}

CURRENT TARGET SPAN ID: "{span_id}"

This is a COMBINED node inspection with two hard-separated output channels:

- local: benchmark-visible node-local errors. This is the only source for the
  final output.errors list and Local diagnosis metrics.
- cascade: causal/root-cause metadata. This is used only for cascade_failure and
  Cascade diagnosis metrics.

Never let cascade attribution add, remove, suppress, or relabel local.errors.
Never relabel a concrete node-local formatting/protocol/tool error as a higher
level root-cause category merely because it contributed to the final failure.


STEP 1 - NODE-LOCAL ERROR DETECTION:
1. Evaluate the current node objectively, regardless of whether it is a root cause,
   passive propagation, or unrelated to the final answer.
2. Report every distinct error in the current node itself.
3. Do not suppress a node-local error merely because it is unrelated to the final
   cascade failure.
4. local.errors[*].location MUST be exactly "{span_id}".
5. local.errors[*].evidence MUST be an exact substring copied from the CURRENT
   NODE's stored trace evidence, including messages, arguments, status, errors, or
   return values. Never paraphrase evidence.
6. Use upstream/downstream context only to understand whether the current node's own
   content/action is wrong. Do not use final-answer correctness alone as local
   evidence.
7. Use only allowed leaf categories.


STEP 2 - CASCADE ATTRIBUTION:
1. Use the local errors from STEP 1 as candidate evidence only. They are NOT
   automatically causal facts.
2. If ACTIVE DOWNSTREAM FAILURE CAUSES is empty, return cascade.status="CORRECT",
   cascade.cause_relations=[], cascade.cause_updates=[], and
   cascade.stopped_cause_ids=[].
3. If active causes exist, classify the current node's relation to each active cause:
   - INTRODUCES: current node creates the mechanism for the first time.
   - PROPAGATES: current node carries an already-existing cause forward.
   - AMPLIFIES: current node makes the cause worse.
   - RECOVERS: current node corrects or neutralizes the cause.
   - UNRELATED: node is independent of the cause.
   - REFUTES: node contradicts the hypothesis.
   - UNCERTAIN: evidence is insufficient.
4. If a local error clearly explains an active downstream cause, return that
   mechanism in cascade.cause_updates. Preserve the existing cause_id when the
   hypothesis is the same; create a new cause_id only when a genuinely new upstream
   mechanism is supported.
5. If a local error is unrelated to the active downstream failure, keep it out of
   cascade.cause_updates; it must remain recorded in local.errors.
6. Return only delta changes in cascade.cause_updates. Do not repeat unchanged
   active causes; omitted active causes are preserved by code.
7. Put a cause_id in cascade.stopped_cause_ids when this node refutes, recovers,
   makes unrelated, or establishes a terminal root cause for that cause on this path.

Cause state semantics for cascade.cause_updates:
- support_status:
  - HYPOTHESIS: possible cause, but no concrete supporting span yet.
  - SUPPORTED: this cause has concrete evidence at a span.
  - REFUTED: evidence no longer supports this cause.
- causal_status:
  - UNKNOWN: causal position is not established.
  - PROXIMATE_CAUSE: this node is a supported/proximate source, but an earlier
    upstream mechanism may still exist.
  - ROOT_CAUSE: this is the earliest sufficiently supported cause on this path.
- should_propagate_upstream:
  - true when the cause still needs upstream search.
  - false when the cause is ROOT_CAUSE, REFUTED, RECOVERED, or UNRELATED on this path.

Delta update rules:
1. Preserve cause_id for the same hypothesis.
2. Only include a cause in cascade.cause_updates when this node changes that cause:
   support, causal position, propagation, evidence, category, confidence, localized
   span, root span, or description.
3. Never overwrite an existing localized_span_id with a later node. If it is already
   known, preserve it unless you are explicitly replacing/refuting the entire hypothesis.
4. If this node provides concrete support for a cause and no earlier localization is
   known, set support_status="SUPPORTED", causal_status="PROXIMATE_CAUSE",
   localized_span_id="{span_id}", and provide exact evidence from this node.
5. If this node is the earliest sufficiently supported mechanism, set
   support_status="SUPPORTED", causal_status="ROOT_CAUSE", root_span_id="{span_id}",
   should_propagate_upstream=false, include the cause_id in stopped_cause_ids,
   and mark the corresponding cause relation is_root_cause=true.
6. If this node refutes a cause, set support_status="REFUTED",
   should_propagate_upstream=false, include it in cause_updates, and include the
   cause_id in stopped_cause_ids.
7. If this node recovers or is unrelated to a cause, mark the cause relation as
   RECOVERS or UNRELATED and include the cause_id in stopped_cause_ids.
8. Do not invent causes without evidence from the current node or the STEP 1 local errors.
9. Keep normally 0-{config.max_causes} active causes.
{path_scope_rules}
"""

        user_prompt = f"""
USER QUERY:
{user_query}

{ground_truth_block}

CURRENT TARGET NODE (Span ID: {span_id}):
{json_for_prompt(clean_node)}

UPSTREAM CONTEXT:
{json_for_prompt(clean_upstream)}

DOWNSTREAM CONTEXT:
{json_for_prompt(clean_downstream)}

ACTIVE DOWNSTREAM FAILURE CAUSES:
{json_for_prompt(cause_paths if cause_paths is not None else to_serializable_causes(active_causes))}
"""


        server = self.llm_server or get_llm_server()
        response = parse_structured(
            server,
            model=config.llm_model,
            response_model=(
                CombinedNodeInspectionH1
                if config.cause_state_mode == "H1" and cause_paths is not None
                else CombinedNodeInspection
            ),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        parsed = response.choices[0].message.parsed
        for err in parsed.local.errors:
            err.location = span_id
        return parsed

    def inspect_node_cascade(
        self,
        user_query: str,
        node: Dict[str, Any],
        upstream_nodes: List[Dict[str, Any]],
        downstream_nodes: List[Dict[str, Any]],
        active_causes: List[FailureCause],
        ground_truth: Optional[str],
        local_errors: Optional[List[TRAILErrorItem]] = None,
        cause_paths: Optional[List[Dict[str, Any]]] = None,
    ) -> CascadeNodeInspection:
        """Attribute causal state in a separate request from local detection."""
        config = self.config
        clean_node = sanitize_node_for_llm(
            node,
            config,
            config.max_current_node_string_length or config.max_string_length,
        )
        clean_upstream = sanitize_context_list(
            upstream_nodes,
            config,
            config.max_context_node_string_length or config.max_string_length,
        )
        clean_downstream = sanitize_context_list(
            downstream_nodes,
            config,
            config.max_downstream_node_string_length or config.max_string_length,
        )
        span_id = str(
            get_field_with_fallback(clean_node, ["span_id", "node_id", "id"], "UNKNOWN_SPAN")
        )
        ground_truth_block = ""
        if config.include_ground_truth_in_node_prompt and ground_truth is not None:
            ground_truth_block = f"""
GROUND TRUTH / CORRECT FINAL ANSWER:
{ground_truth}
Use this only to understand the final failure context. Do not treat it as evidence
that this current span introduced a cause.
"""
        local_error_block = ""
        if local_errors:
            local_error_block = f"""
INDEPENDENT LOCAL INSPECTION CANDIDATES:
{json_for_prompt([error.model_dump() for error in local_errors])}
Treat these as evidence candidates only, not automatically as causal facts.
"""

        path_scope_rules = ""
        if config.cause_state_mode == "H1" and cause_paths is not None:
            path_scope_rules = """
H1 PATH-SCOPED CAUSE STATE:
- Treat every (cause_id, incoming_path_id) as a separate branch.
- Echo incoming_path_id in each cause_relations item. Wrap cause_updates as
  {"incoming_path_id": "...", "cause": {...}}.
- REFUTES, RECOVERS, and UNRELATED stop only the specified path. A missing relation
  is UNCERTAIN and preserves the state; never broadcast to other paths.
"""
        system_prompt = f"""
You are the CASCADE ATTRIBUTION stage of a TRAIL trace diagnosis.

{TRAIL_TAXONOMY}

TARGET SPAN ID: "{span_id}"

Do not detect, add, remove, or rewrite node-local benchmark errors. Return only
cascade causal metadata for the active downstream causes.

For each active cause, classify this span as INTRODUCES, PROPAGATES, AMPLIFIES,
RECOVERS, UNRELATED, REFUTES, or UNCERTAIN. Use exact evidence from this span when
marking a cause SUPPORTED. Preserve cause IDs for unchanged hypotheses and return
only delta updates. Mark ROOT_CAUSE only when this is the earliest sufficiently
supported mechanism; otherwise keep searching upstream. Stop a cause when it is
refuted, recovered, unrelated, or established as a root cause.

If there are no active causes, return status CORRECT and empty relations, updates,
and stopped IDs. Do not invent causes from a final answer mismatch alone.
{path_scope_rules}
"""
        user_prompt = f"""
USER QUERY:
{user_query}

{ground_truth_block}

CURRENT TARGET NODE (Span ID: {span_id}):
{json_for_prompt(clean_node)}

UPSTREAM CONTEXT:
{json_for_prompt(clean_upstream)}

DOWNSTREAM CONTEXT:
{json_for_prompt(clean_downstream)}

ACTIVE DOWNSTREAM FAILURE CAUSES:
{json_for_prompt(cause_paths if cause_paths is not None else to_serializable_causes(active_causes))}

{local_error_block}
"""
        server = self.llm_server or get_llm_server()
        response = parse_structured(
            server,
            model=config.llm_model,
            response_model=(
                CascadeNodeInspectionH1
                if config.cause_state_mode == "H1" and cause_paths is not None
                else CascadeNodeInspection
            ),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.parsed


    def score_trace_report(
        self,
        user_query: str,
        detected_errors: List[TRAILErrorItem],
        final_failure: FinalFailureAnalysis,
        root_cause_causes: List[FailureCause],
        root_cause_relations: List[Dict[str, Any]],
    ) -> TRAILScoresOnlyOutput:
        config = self.config
        system_prompt = """
You are scoring an AI agent trace after error detection has already been completed.
Do not add, remove, or rewrite errors. Only produce the scores item.
"""

        user_prompt = f"""
USER QUERY:
{user_query}

DETECTED ERRORS:
{json_for_prompt([e.model_dump() for e in detected_errors])}

FINAL FAILURE:
{json_for_prompt(final_failure.model_dump())}

ROOT CAUSE CAUSES:
{json_for_prompt([cause.model_dump() for cause in root_cause_causes])}

ROOT CAUSE RELATIONS:
{json_for_prompt(root_cause_relations)}

Produce one scores item with reliability, security, instruction_adherence,
plan_opt, and overall.
"""

        server = self.llm_server or get_llm_server()
        response = parse_structured(
            server,
            model=config.llm_model,
            response_model=TRAILScoresOnlyOutput,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.parsed

    def synthesize_trail_report(
        self,
        user_query: str,
        detected_errors: List[TRAILErrorItem],
        all_node_diagnoses: List[Dict[str, Any]],
    ) -> TRAILDiagnosisOutput:
        config = self.config
        system_prompt = f"""
You are an expert evaluator synthesizing a final TRAIL trace benchmark output.

{TRAIL_TAXONOMY}

Retain valid fine-grained errors with exact location and evidence, remove only
strict duplicates, and produce one scores item.
"""

        user_prompt = f"""
USER QUERY:
{user_query}

ALL DETECTED SPAN ERRORS:
{json_for_prompt([e.model_dump() for e in detected_errors])}

FULL INSPECTION HISTORY:
{json_for_prompt(all_node_diagnoses)}
"""

        server = self.llm_server or get_llm_server()
        response = parse_structured(
            server,
            model=config.llm_model,
            response_model=TRAILDiagnosisOutput,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.parsed


# =====================================================================
# 6. AgentGraphDiagnoser
# =====================================================================

def _trail_prediction_coverage(
    shared_run: Any, *, cascade_enabled: bool, errors: Any,
    retained_node_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Separate Local Prediction completion from Cause Path completeness.

    SharedDiagnosisEngine already exposes its schedule, successful node
    diagnoses, blocked nodes, cycle nodes, and budget flags. This adapter only
    interprets those existing facts; it does not alter scheduling or state
    propagation.
    """
    schedule = [str(node_id) for node_id in (getattr(shared_run, "schedule", None) or [])]
    diagnoses = getattr(shared_run, "diagnoses", None) or {}
    completed = []
    failed = []
    for node_id in schedule:
        diagnosis = diagnoses.get(node_id)
        if diagnosis is None:
            continue
        if getattr(diagnosis, "local", None) is None:
            failed.append(node_id)
        else:
            completed.append(node_id)
    completed_set = set(completed)
    failed_set = set(failed)
    unvisited = [node_id for node_id in schedule if node_id not in completed_set | failed_set]
    retained = [str(node_id) for node_id in (retained_node_ids or schedule)]
    retained_unvisited = [node_id for node_id in retained if node_id not in completed_set]

    errors_valid = isinstance(errors, list) and all(
        isinstance(item, dict)
        and isinstance(item.get("location"), str)
        and bool(item.get("location"))
        and item.get("category") in TRAIL_LEAF_CATEGORIES
        and isinstance(item.get("evidence"), str)
        and isinstance(item.get("description"), str)
        and item.get("impact") in {"HIGH", "MEDIUM", "LOW"}
        for item in errors
    )
    if not schedule or not completed or not errors_valid:
        local_status = "FAILED"
    elif failed or unvisited:
        local_status = "PARTIAL"
    else:
        local_status = "SUCCESS"

    graph_validation = getattr(shared_run, "graph_validation", None) or {}
    cycle_nodes = [str(node_id) for node_id in (getattr(shared_run, "cycle_nodes", None) or [])]
    blocked_nodes = [str(node_id) for node_id in (getattr(shared_run, "blocked_nodes", None) or [])]
    path_budget_exhausted = bool(getattr(shared_run, "path_state_budget_exhausted", False))
    worklist_budget_exhausted = bool(graph_validation.get("worklist_visit_budget_exhausted", False))
    partial_reasons = []
    if cycle_nodes:
        partial_reasons.append("CYCLE_DETECTED")
    if blocked_nodes:
        partial_reasons.append("BLOCKED_NODES")
    if path_budget_exhausted:
        partial_reasons.append("PATH_STATE_BUDGET_EXHAUSTED")
    if worklist_budget_exhausted:
        partial_reasons.append("WORKLIST_VISIT_BUDGET_EXHAUSTED")
    if bool(getattr(shared_run, "partial_coverage", False)) and not partial_reasons:
        partial_reasons.append("ENGINE_PARTIAL_COVERAGE")

    if not cascade_enabled:
        cause_status = "NOT_RUN"
        partial_reasons = []
    else:
        cause_status = "PARTIAL" if partial_reasons else "COMPLETE"

    if local_status == "FAILED":
        diagnosis_status = "FAILED"
    elif local_status == "PARTIAL" or cause_status == "PARTIAL":
        diagnosis_status = "PARTIAL"
    else:
        diagnosis_status = "SUCCESS"

    return {
        "status": diagnosis_status,
        "local_prediction_status": local_status,
        "cause_path_coverage_status": cause_status,
        "cause_path_partial_reasons": partial_reasons,
        "local_prediction_coverage": {
            "expected_node_ids": schedule,
            "completed_node_ids": completed,
            "failed_node_ids": failed,
            "unvisited_node_ids": unvisited,
            "expected_count": len(schedule),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "unvisited_count": len(unvisited),
            "retained_graph_node_count": len(retained),
            "scheduled_node_count": len(schedule),
            "unique_local_completed_count": len(completed_set),
            "unvisited_retained_node_ids": retained_unvisited,
            "final_errors_assembled": errors_valid,
        },
        "cause_path_coverage": {
            "status": cause_status,
            "partial_reasons": partial_reasons,
            "cycle_nodes": cycle_nodes,
            "blocked_nodes": blocked_nodes,
            "path_state_budget": getattr(shared_run, "path_state_budget", None),
            "path_state_budget_exhausted": path_budget_exhausted,
            "worklist_visit_budget": graph_validation.get("worklist_visit_budget"),
            "worklist_visit_budget_exhausted": worklist_budget_exhausted,
            "processed_cause_path_states": getattr(shared_run, "processed_cause_path_states", 0),
        },
    }

class AgentGraphDiagnoser:
    def __init__(
        self,
        graph: Dict[str, Any],
        config: Optional[DiagnoserConfig] = None,
        llm_server: Optional[LLMServer] = None,
    ):
        self.graph = graph
        self.config = config or DiagnoserConfig()
        self.llm_service = LLMService(self.config, llm_server=llm_server)
        self.cascade_call_mode = self.config.cascade_call_mode
        self.separate_order = self.config.separate_order
        self.prompt_context_policy = self.config.prompt_context_policy
        self.traversal_mode = resolve_traversal_mode(
            self.config.traversal_mode,
            default="bfs",
        )

        raw_nodes: List[Dict[str, Any]] = graph.get("nodes") or []
        self.nodes = sorted(
            raw_nodes,
            key=lambda n: (n.get("timestamp") is None, n.get("timestamp") or 0),
        )
        self.edges: List[Dict[str, Any]] = graph.get("edges") or []

        self.node_map: Dict[str, Dict[str, Any]] = {}
        for node in self.nodes:
            nid = get_field_with_fallback(
                node, [self.config.node_id_key, "span_id", "node_id", "id"]
            )
            if nid is not None:
                self.node_map[str(nid)] = node

        self.incoming_edges: Dict[str, List[Dict[str, Any]]] = {}
        self.outgoing_edges: Dict[str, List[Dict[str, Any]]] = {}

        for edge in self.edges:
            src = get_field_with_fallback(
                edge, [self.config.edge_src_key, "source", "src"]
            )
            dst = get_field_with_fallback(
                edge, [self.config.edge_dst_key, "target", "dst"]
            )
            if src is None or dst is None:
                continue
            src_str, dst_str = str(src), str(dst)
            self.outgoing_edges.setdefault(src_str, []).append(edge)
            self.incoming_edges.setdefault(dst_str, []).append(edge)

    def find_final_node(self) -> Optional[Dict[str, Any]]:
        if not self.nodes:
            return None

        if self.config.is_final_node_fn:
            for node in reversed(self.nodes):
                try:
                    if self.config.is_final_node_fn(node):
                        return node
                except Exception:
                    pass

        for node in reversed(self.nodes):
            role = str(node.get("semantic_role") or "").lower()
            name = str(node.get("name") or node.get("span_name") or "").lower()
            if (
                "output" in role
                or "final" in role
                or "output" in name
                or "writer" in name
            ):
                return node

        out_zero_nodes = []
        for node in self.nodes:
            nid = str(
                get_field_with_fallback(
                    node, [self.config.node_id_key, "span_id", "node_id", "id"]
                )
            )
            if len(self.outgoing_edges.get(nid, [])) == 0:
                out_zero_nodes.append(node)

        if out_zero_nodes:
            return out_zero_nodes[-1]
        return self.nodes[-1]

    def get_user_query(self) -> Optional[str]:
        q = extract_user_query_smart(self.graph)
        if q:
            return q

        for node in self.nodes:
            role = str(
                get_field_with_fallback(node, ["semantic_role", "role", "type"], "")
            ).lower()
            if (
                (self.config.is_agent_node_fn and self.config.is_agent_node_fn(node))
                or role in ("agent", "user", "system_input", "start")
            ):
                node_q = extract_user_query_smart(node)
                if node_q:
                    return node_q

        return extract_user_query_smart(self.nodes[0]) if self.nodes else None


    def _get_legacy_upstream_context(self, node_id: str) -> List[Dict[str, Any]]:
        """Reproduce the old prompt-neighbor policy for controlled ablations."""
        incoming_edges = self.incoming_edges.get(node_id, [])
        seen_src_ids = set()
        if not incoming_edges:
            return []

        data_upstream, exec_upstream, parent_upstream, other_upstream = [], [], [], []

        for edge in incoming_edges:
            kind = str(
                get_field_with_fallback(edge, [self.config.edge_kind_key, "type", "kind"], "")
            ).lower()
            src_id = str(
                get_field_with_fallback(edge, [self.config.edge_src_key, "source", "src"])
            )

            if src_id not in self.node_map or src_id in seen_src_ids:
                continue
            seen_src_ids.add(src_id)

            item = {
                "src_node_id": src_id,
                "relation": kind,
                "node": sanitize_node_for_llm(self.node_map[src_id], self.config),
                "edge": edge,
            }

            if kind in self.config.data_flow_kinds:
                data_upstream.append(item)
            elif kind in self.config.execution_flow_kinds:
                exec_upstream.append(item)
            elif kind in self.config.parent_kinds or kind == "contains":
                parent_upstream.append(item)
            else:
                other_upstream.append(item)

        if data_upstream:
            return data_upstream + exec_upstream
        if exec_upstream:
            return exec_upstream + parent_upstream
        return parent_upstream or other_upstream

    def _get_ranked_upstream_candidates(self, node_id: str) -> List[Dict[str, Any]]:
        """Return every unique predecessor, ranked for prompt-context selection."""
        by_source: Dict[str, Dict[str, Any]] = {}
        for edge in self.incoming_edges.get(node_id, []):
            kind = str(
                get_field_with_fallback(edge, [self.config.edge_kind_key, "type", "kind"], "")
            ).lower()
            src_id = str(
                get_field_with_fallback(edge, [self.config.edge_src_key, "source", "src"])
            )
            if src_id not in self.node_map:
                continue

            candidate = by_source.get(src_id)
            if candidate is None:
                candidate = {
                    "src_node_id": src_id,
                    "relations": [],
                    "edge_evidence": [],
                    "node": sanitize_node_for_llm(
                        self.node_map[src_id],
                        self.config,
                        self.config.max_context_node_string_length,
                    ),
                    "_priority": 3,
                    "_order": len(by_source),
                }
                by_source[src_id] = candidate

            if kind not in candidate["relations"]:
                candidate["relations"].append(kind)
            edge_evidence = edge.get("evidence")
            if edge_evidence is not None:
                candidate["edge_evidence"].append(
                    sanitize_nested_value(
                        edge_evidence,
                        self.config.max_context_node_string_length,
                    )
                )

            if kind in self.config.data_flow_kinds:
                candidate["_priority"] = min(candidate["_priority"], 0)
            elif kind in self.config.execution_flow_kinds:
                candidate["_priority"] = min(candidate["_priority"], 1)
            elif kind in self.config.parent_kinds or kind == "contains":
                candidate["_priority"] = min(candidate["_priority"], 2)

        candidates = sorted(
            by_source.values(),
            key=lambda item: (item["_priority"], item["_order"]),
        )
        for candidate in candidates:
            candidate["relation"] = (
                candidate["relations"][0]
                if len(candidate["relations"]) == 1
                else list(candidate["relations"])
            )
            candidate.pop("_priority", None)
            candidate.pop("_order", None)
        return candidates

    def _get_prioritized_two_kinds_context(
        self, node_id: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Select the two highest-priority nonempty edge-kind groups for prompt context."""
        group_order = ("data_flow", "execution_flow", "parent", "other")
        edges_by_group: Dict[str, List[Tuple[int, Dict[str, Any], str, str]]] = {
            group: [] for group in group_order
        }
        requested_kinds: List[str] = []
        for edge_order, edge in enumerate(self.incoming_edges.get(node_id, [])):
            kind = str(
                get_field_with_fallback(edge, [self.config.edge_kind_key, "type", "kind"], "")
            ).lower()
            if kind not in requested_kinds:
                requested_kinds.append(kind)
            if kind in self.config.data_flow_kinds:
                group = "data_flow"
            elif kind in self.config.execution_flow_kinds:
                group = "execution_flow"
            elif kind in self.config.parent_kinds or kind == "contains":
                group = "parent"
            else:
                group = "other"
            src_id = str(
                get_field_with_fallback(edge, [self.config.edge_src_key, "source", "src"])
            )
            if src_id in self.node_map:
                edges_by_group[group].append((edge_order, edge, kind, src_id))

        selected_groups = [group for group in group_order if edges_by_group[group]][:2]
        selected_edges = [
            item
            for group in selected_groups
            for item in edges_by_group[group]
        ]
        selected_edges.sort(key=lambda item: item[0])
        selected_kinds: List[str] = []
        for group in selected_groups:
            for _edge_order, _edge, kind, _src_id in edges_by_group[group]:
                if kind not in selected_kinds:
                    selected_kinds.append(kind)

        by_source: Dict[str, Dict[str, Any]] = {}
        for _edge_order, edge, kind, src_id in selected_edges:
            candidate = by_source.get(src_id)
            if candidate is None:
                candidate = {
                    "src_node_id": src_id,
                    "relations": [],
                    "edge_evidence": [],
                    "node": sanitize_node_for_llm(
                        self.node_map[src_id], self.config,
                        self.config.max_context_node_string_length,
                    ),
                    "_order": len(by_source),
                }
                by_source[src_id] = candidate
            if kind not in candidate["relations"]:
                candidate["relations"].append(kind)
            evidence = edge.get("evidence")
            if evidence is not None:
                candidate["edge_evidence"].append({
                    "relation": kind,
                    "evidence": sanitize_nested_value(
                        evidence, self.config.max_context_node_string_length,
                    ),
                })

        priority = {group: index for index, group in enumerate(group_order)}
        source_group_priority: Dict[str, int] = {}
        for group in selected_groups:
            for _edge_order, _edge, _kind, src_id in edges_by_group[group]:
                source_group_priority[src_id] = min(
                    source_group_priority.get(src_id, len(group_order)), priority[group],
                )
        candidates = sorted(
            by_source.values(),
            key=lambda item: (source_group_priority.get(item["src_node_id"], len(group_order)), item["_order"]),
        )
        for candidate in candidates:
            candidate["relation"] = (
                candidate["relations"][0]
                if len(candidate["relations"]) == 1
                else list(candidate["relations"])
            )
            candidate.pop("_order", None)

        char_budget = max(0, int(self.config.max_prompt_upstream_chars))
        selected: List[Dict[str, Any]] = []
        omitted: List[Dict[str, Any]] = []
        for candidate in candidates:
            proposal = selected + [candidate]
            if len(json.dumps(proposal, ensure_ascii=False, separators=(",", ":"))) <= char_budget:
                selected = proposal
            else:
                omitted.append({
                    "src_node_id": candidate["src_node_id"],
                    "edge_kinds": list(candidate["relations"]),
                })

        visible_kinds = list(dict.fromkeys(
            kind for candidate in selected for kind in candidate["relations"]
        ))
        omitted_by_kind: Dict[str, int] = {}
        for item in omitted:
            for kind in item["edge_kinds"]:
                omitted_by_kind[kind] = omitted_by_kind.get(kind, 0) + 1
        context_chars = len(json.dumps(selected, ensure_ascii=False, separators=(",", ":")))
        audit = {
            "policy": "prioritized_two_kinds",
            "requested_edge_kinds": requested_kinds,
            "selected_edge_kind_groups": selected_groups,
            "selected_edge_kinds": selected_kinds,
            "visible_edge_kinds": visible_kinds,
            "serialized_characters": context_chars,
            "character_limit": char_budget,
            "omitted_sources": omitted,
            "omitted_edge_count_by_kind": omitted_by_kind,
            "unselected_edge_kinds": [kind for kind in requested_kinds if kind not in selected_kinds],
        }
        return selected, audit

    def get_upstream_context(self, node_id: str) -> List[Dict[str, Any]]:
        """Select prompt neighbors independently from the graph traversal policy."""
        if self.prompt_context_policy == "prioritized_two_kinds":
            return self._get_prioritized_two_kinds_context(node_id)[0]
        if self.prompt_context_policy != "ranked_budget":
            return self._get_legacy_upstream_context(node_id)

        candidates = self._get_ranked_upstream_candidates(node_id)
        char_budget = max(0, self.config.max_prompt_upstream_chars)
        selected: List[Dict[str, Any]] = []
        for candidate in candidates:
            candidate_selection = selected + [candidate]
            if len(json.dumps(candidate_selection, ensure_ascii=False)) > char_budget:
                continue
            selected = candidate_selection
        return selected

    def get_downstream_context(self, node_id: str) -> List[Dict[str, Any]]:
        downstream = []
        for edge in self.outgoing_edges.get(node_id, []):
            dst_id = str(
                get_field_with_fallback(edge, [self.config.edge_dst_key, "target", "dst"])
            )
            if dst_id in self.node_map:
                downstream.append(
                    {
                        "dst_node_id": dst_id,
                        "relation": get_field_with_fallback(
                            edge, [self.config.edge_kind_key, "type", "kind"]
                        ),
                        "node": sanitize_node_for_llm(self.node_map[dst_id], self.config),
                        "edge": edge,
                    }
                )
        return downstream[: self.config.max_downstream_node]

    def get_upstream_traversal_ids(self, node_id: str) -> List[str]:
        """Return all predecessor nodes allowed by the traversal policy."""
        return [
            item["src_node_id"]
            for item in self._get_ranked_upstream_candidates(node_id)
            if item["src_node_id"] in self.node_map
        ]


    def _merge_incoming_cause_states(
        self, states: List[List[FailureCause]]
    ) -> List[FailureCause]:
        merged: List[FailureCause] = []
        for state in states:
            merged = merge_causes(merged, state, max_causes=self.config.max_causes)
        return merged

    def run_diagnosis(
        self,
        user_query: Optional[str] = None,
        target_node_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        user_query = user_query or self.get_user_query()
        if not user_query:
            return {
                "errors": [],
                "scores": [],
                "status": "FAILED",
                "local_prediction_status": "FAILED",
                "cause_path_coverage_status": "NOT_RUN",
                "cause_path_partial_reasons": [],
                "reason": "Unable to extract user query.",
            }

        if target_node_id:
            final_node = self.node_map.get(str(target_node_id))
        elif self.config.trail_schedule_scope == "all_nodes":
            graph_metadata = self.graph.get("graph_metadata") or {}
            final_anchor = graph_metadata.get("final_anchor_node_id") or self.graph.get("final_anchor_node_id")
            if final_anchor is not None:
                final_node = self.node_map.get(str(final_anchor))
                if final_node is None:
                    return {
                        "errors": [], "scores": [], "status": "FAILED",
                        "local_prediction_status": "FAILED",
                        "cause_path_coverage_status": "NOT_RUN",
                        "cause_path_partial_reasons": [],
                        "reason": f"Configured final anchor {final_anchor!r} is not a retained node.",
                    }
            elif str(graph_metadata.get("graph_variant", "")).startswith("edge_subset"):
                return {
                    "errors": [], "scores": [], "status": "FAILED",
                    "local_prediction_status": "FAILED",
                    "cause_path_coverage_status": "NOT_RUN",
                    "cause_path_partial_reasons": [],
                    "reason": "Edge-subgraph all_nodes diagnosis requires a full-graph final_anchor_node_id.",
                }
            else:
                final_node = self.find_final_node()
        else:
            final_node = self.find_final_node()
        if not final_node:
            return {
                "errors": [],
                "scores": [],
                "status": "FAILED",
                "local_prediction_status": "FAILED",
                "cause_path_coverage_status": "NOT_RUN",
                "cause_path_partial_reasons": [],
                "reason": "Terminal node not found.",
            }

        raw_final_id = get_field_with_fallback(
            final_node, [self.config.node_id_key, "span_id", "node_id", "id"]
        )
        if raw_final_id is None:
            return {
                "errors": [],
                "scores": [],
                "status": "FAILED",
                "local_prediction_status": "FAILED",
                "cause_path_coverage_status": "NOT_RUN",
                "cause_path_partial_reasons": [],
                "reason": "Final node missing valid ID.",
            }

        final_node_id = str(raw_final_id)
        # TRAIL is a blind diagnosis benchmark. Never read caller-supplied or
        # recursively discovered answer aliases in this diagnosis path.
        ground_truth = None
        final_failure = FinalFailureAnalysis(
            has_failure=False,
            final_error="Cascade failure analysis disabled.",
            causes=[],
            reasoning="Cascade failure analysis disabled.",
        )
        active_final_causes: List[FailureCause] = []
        hypothesis_seed_audit: List[Dict[str, Any]] = []
        seed_cause_count = 0

        # Hypothesis initialization is invoked inside SharedDiagnosisEngine.run
        # through the dataset adapter below, preserving the original call order.

        h1_mode = self.config.cause_state_mode == "H1"
        h1_schedule: Dict[str, Any] = {"order": [final_node_id], "cycle": False, "cyclic_nodes": []}
        queue: List[str] = []
        queue_members = set(queue)
        visited: Set[str] = set()

        incoming_cause_states: Dict[str, Any] = {}
        propagated_causes_by_node: Dict[str, Any] = {}
        h1_path_nodes: Dict[str, Tuple[str, ...]] = {"FINAL": (final_node_id,)}
        h1_processed_states: Dict[str, Set[Tuple[str, str]]] = {}
        h1_path_audit: List[Dict[str, Any]] = []
        h1_path_state_count = 0
        local_inspection_cache: Dict[str, SingleNodeLocalInspection] = {}

        all_detected_errors: List[TRAILErrorItem] = []
        diagnosed_history: List[Dict[str, Any]] = []
        cascade_relations: List[Dict[str, Any]] = []

        # The shared engine owns initialization, traversal, Local/Cascade
        # scheduling, Cause transitions, joins, propagation and history.
        state_policy = (
            TrailPathCauseStatePolicy(
                max_path_states=self.config.max_cause_path_states,
                max_causes=self.config.max_causes,
            )
            if h1_mode else TrailLegacyCauseStatePolicy(max_causes=self.config.max_causes)
        )
        h1_audit_pending: List[Dict[str, Any]] = []
        selected_edge_policy = (
            EdgePolicy.trail_dependency_only() if h1_mode else EdgePolicy.legacy()
        )

        def _shared_initialize_hypotheses(_entry_node: Dict[str, Any]) -> List[FailureCause]:
            nonlocal final_failure, active_final_causes, seed_cause_count
            final_failure = self.llm_service.initialize_failure_hypothesis(
                user_query=user_query,
                final_node=final_node,
                ground_truth=None,
            )
            candidates = [
                cause.model_dump()
                for cause in final_failure.causes
                if final_failure.has_failure or self.config.final_hypothesis_policy == "retain_candidates"
            ]
            if self.config.final_hypothesis_policy == "retain_candidates":
                visible_final = sanitize_node_for_llm(
                    final_node, self.config,
                    self.config.max_current_node_string_length or self.config.max_string_length,
                )
                normalized_candidates = []
                for raw in candidates:
                    cause = FailureCause.model_validate(raw)
                    if cause.support_status == "REFUTED":
                        hypothesis_seed_audit.append({
                            "cause_id": cause.cause_id, "status": "REFUTED_SEED_FILTERED",
                        })
                        continue
                    exact_final_evidence = bool(
                        cause.evidence
                        and _node_contains_exact_evidence(visible_final, cause.evidence)
                        and cause.localized_span_id in (None, final_node_id)
                    )
                    claims_support = (
                        cause.support_status == "SUPPORTED"
                        or cause.causal_status in {"PROXIMATE_CAUSE", "ROOT_CAUSE"}
                    )
                    if claims_support and not exact_final_evidence:
                        cause = cause.model_copy(update={
                            "support_status": "HYPOTHESIS",
                            "causal_status": "UNKNOWN",
                            "localized_span_id": None,
                            "root_span_id": None,
                            "should_propagate_upstream": True,
                        })
                        hypothesis_seed_audit.append({
                            "cause_id": cause.cause_id,
                            "status": "UNVERIFIED_SEED_DOWNGRADED",
                            "reason": "no exact visible final-node evidence",
                        })
                    elif exact_final_evidence and cause.support_status == "SUPPORTED" and not cause.localized_span_id:
                        cause = cause.model_copy(update={"localized_span_id": final_node_id})
                    normalized_candidates.append(cause.model_dump())
                candidates = normalized_candidates
            active_final_causes = [
                FailureCause.model_validate(item)
                for item in TrailLegacyCauseStatePolicy(max_causes=self.config.max_causes).merge_incoming(
                    candidates
                )[:self.config.max_causes]
            ]
            seed_cause_count = len(active_final_causes)
            if h1_mode:
                visible_final = sanitize_node_for_llm(
                    final_node, self.config,
                    self.config.max_current_node_string_length or self.config.max_string_length,
                )
                for cause in active_final_causes:
                    if cause.causal_status == "ROOT_CAUSE":
                        evidence_valid = bool(
                            cause.support_status == "SUPPORTED"
                            and cause.evidence
                            and _node_contains_exact_evidence(visible_final, cause.evidence)
                        )
                        if not evidence_valid:
                            h1_audit_pending.append({
                                "kind": "root_cause", "cause_id": cause.cause_id,
                                "incoming_path_id": "FINAL", "status": "ROOT_CAUSE_REJECTED_UNSUPPORTED",
                                "reason": "requires_supported_cause_with_exact_visible_final_span_evidence",
                            })
                            cause.causal_status = "UNKNOWN"
                            cause.root_span_id = None
                            cause.support_status = "HYPOTHESIS"
                            cause.should_propagate_upstream = True
                # Keep the serialized FinalFailureAnalysis consistent with the
                # H1 seed state actually handed to the shared engine. A rejected
                # model root claim must not reappear as a confirmed root in the
                # final diagnostic summary.
                normalized_by_id = {cause.cause_id: cause for cause in active_final_causes}
                final_failure = final_failure.model_copy(update={
                    "causes": [
                        normalized_by_id.get(cause.cause_id, cause).model_copy(deep=True)
                        for cause in final_failure.causes
                    ]
                })
            return active_final_causes

        def _shared_prepare_node(
            node_id: str, _node: Dict[str, Any], incoming: List[Dict[str, Any]], position: int,
        ) -> Dict[str, Any]:
            current_node = self.node_map[node_id]
            path_states = [dict(item) for item in incoming] if h1_mode else []
            active_causes = (
                [FailureCause.model_validate(item["cause"]) for item in path_states]
                if h1_mode else [FailureCause.model_validate(item) for item in incoming]
            )
            prompt_context_audit = None
            if self.prompt_context_policy == "prioritized_two_kinds":
                upstream_context, prompt_context_audit = self._get_prioritized_two_kinds_context(node_id)
            else:
                upstream_context = self.get_upstream_context(node_id)
            if selected_edge_policy.prompt_context_kinds is not None:
                allowed_context_kinds = {str(kind).lower() for kind in selected_edge_policy.prompt_context_kinds}
                filtered_context = []
                for item in upstream_context:
                    relation = item.get("relation")
                    relations = relation if isinstance(relation, list) else [relation]
                    if any(str(kind).lower() in allowed_context_kinds for kind in relations if kind is not None):
                        filtered_context.append(item)
                upstream_context = filtered_context
            prepared = {
                "node_id": node_id,
                "node": current_node,
                "upstream": upstream_context,
                "downstream": self.get_downstream_context(node_id),
                "active_causes": active_causes,
                "cause_paths": path_states,
                "position": position,
                "first_visit": True,
            }
            if prompt_context_audit is not None:
                prepared["prompt_context_audit"] = prompt_context_audit
            return prepared

        def _shared_inspect_local(prepared: Dict[str, Any]) -> SingleNodeLocalInspection:
            return self.llm_service.inspect_node_local(
                user_query=user_query,
                node=prepared["node"],
                upstream_nodes=prepared["upstream"],
                downstream_nodes=prepared["downstream"],
            )

        def _shared_inspect_local_with_cascade(
            prepared: Dict[str, Any], cascade: Any,
        ) -> SingleNodeLocalInspection:
            if cascade is None or not hasattr(cascade, "model_dump"):
                raise TypeError("Cascade-conditioned Local inspection requires a structured Cascade result")
            cascade_evidence = cascade
            if h1_mode:
                preview, validation_audit = _shared_prepare_cascade_delta(
                    prepared,
                    list(prepared.get("cause_paths") or []),
                    NodeDiagnosis(local=None, cascade=cascade.model_copy(deep=True)),
                    validate_non_root_relations=True,
                )
                prepared["_cascade_local_prevalidation"] = {
                    "cascade": preview,
                    "audit": validation_audit,
                    "accepted_updates": list(prepared.get("accepted_h1_updates") or []),
                }
                current_visible = sanitize_node_for_llm(
                    prepared["node"], self.config,
                    self.config.max_current_node_string_length or self.config.max_string_length,
                )
                cascade_evidence = _trail_h1_local_evidence_view(
                    preview, validation_audit, current_visible,
                )
            return self.llm_service.inspect_node_local(
                user_query=user_query,
                node=prepared["node"],
                upstream_nodes=prepared["upstream"],
                downstream_nodes=prepared["downstream"],
                cascade_evidence=cascade_evidence,
            )

        def _shared_inspect_cascade(
            prepared: Dict[str, Any], local: Optional[SingleNodeLocalInspection],
        ) -> Any:
            if not prepared["active_causes"]:
                response_type = CascadeNodeInspectionH1 if h1_mode else CascadeNodeInspection
                return response_type(
                    status="CORRECT", confidence=1.0,
                    reasoning="No active downstream causes to attribute.",
                    cause_relations=[], cause_updates=[], stopped_cause_ids=[],
                )
            return self.llm_service.inspect_node_cascade(
                user_query=user_query,
                node=prepared["node"],
                upstream_nodes=prepared["upstream"],
                downstream_nodes=prepared["downstream"],
                active_causes=prepared["active_causes"],
                ground_truth=None,
                local_errors=None,
                cause_paths=(prepared["cause_paths"] if h1_mode else None),
            )

        def _shared_inspect_combined(prepared: Dict[str, Any]) -> NodeDiagnosis:
            response = self.llm_service.inspect_node_combined(
                user_query=user_query,
                node=prepared["node"],
                upstream_nodes=prepared["upstream"],
                downstream_nodes=prepared["downstream"],
                active_causes=prepared["active_causes"],
                ground_truth=None,
                cause_paths=(prepared["cause_paths"] if h1_mode else None),
            )
            return NodeDiagnosis(local=response.local, cascade=response.cascade)

        def _shared_finalize_diagnosis(
            prepared: Dict[str, Any], diagnosis: NodeDiagnosis,
        ) -> NodeDiagnosis:
            current_id = prepared["node_id"]
            local: SingleNodeLocalInspection = diagnosis.local
            if diagnosis.cascade is None:
                diagnosis.cascade = CascadeNodeInspection(
                    status="CORRECT", confidence=1.0,
                    reasoning="Cascade failure analysis disabled.",
                    cause_relations=[], cause_updates=[], stopped_cause_ids=[],
                )
            if prepared.get("first_visit") and local.has_error and local.errors:
                for error in local.errors:
                    error.location = current_id
                    all_detected_errors.append(error)
            diagnosis.local = local
            return diagnosis

        def _shared_prepare_cascade_delta(
            prepared: Dict[str, Any], incoming: List[Dict[str, Any]], diagnosis: NodeDiagnosis,
            *, validate_non_root_relations: bool = False,
        ) -> Tuple[Any, List[Dict[str, Any]]]:
            cascade = diagnosis.cascade
            if not h1_mode or cascade is None:
                return cascade, []
            preview = prepared.pop("_cascade_local_prevalidation", None)
            if preview is not None:
                prepared["accepted_h1_updates"] = list(preview["accepted_updates"])
                rejected_roots = len({
                    (str(event.get("cause_id", "")), str(event.get("incoming_path_id", "")))
                    for event in preview["audit"]
                    if str(event.get("status", "")).startswith("ROOT_CAUSE_REJECTED")
                })
                diagnosis.metadata.update({
                    "cascade_evidence_validation_applied": True,
                    "cascade_evidence_validation_version": _CASCADE_LOCAL_EVIDENCE_VALIDATION_VERSION,
                    "rejected_root_count": rejected_roots,
                })
                return preview["cascade"], list(preview["audit"])
            diagnosis.metadata.update({
                "cascade_evidence_validation_applied": False,
                "cascade_evidence_validation_version": (
                    "not_applicable_combined"
                    if diagnosis.metadata.get("diagnosis_mode") == "combined"
                    else "not_applicable_without_cascade_first"
                ),
                "rejected_root_count": 0,
            })
            relations = [
                {"span_id": prepared["node_id"], **item.model_dump()}
                for item in cascade.cause_relations
            ]
            relation_by_key = {
                (str(item.get("cause_id", "")), str(item.get("incoming_path_id", ""))): item
                for item in relations
            }
            active_keys = {
                (str((item.get("cause") or {}).get("cause_id", "")), str(item.get("incoming_path_id", "")))
                for item in incoming if isinstance(item, dict)
            }
            active_cause_ids = {cause_id for cause_id, _path_id in active_keys}
            active_path_ids = {path_id for _cause_id, path_id in active_keys}
            visible_current = sanitize_node_for_llm(
                prepared["node"], self.config,
                self.config.max_current_node_string_length or self.config.max_string_length,
            )
            accepted: List[Any] = []
            audit: List[Dict[str, Any]] = []
            root_relations: List[Any] = []
            updates_by_key = {
                (
                    str(update.cause.cause_id),
                    str(update.incoming_path_id),
                ): update
                for update in cascade.cause_updates
            }
            valid_root_keys: Set[Tuple[str, str]] = set()
            normalized_updates = list(cascade.cause_updates)
            for relation in cascade.cause_relations:
                key = (str(relation.cause_id), str(relation.incoming_path_id))
                if not relation.is_root_cause:
                    if not validate_non_root_relations:
                        root_relations.append(relation)
                        continue
                    if key in active_keys:
                        root_relations.append(relation)
                        continue

                    # Preserve the existing explicit new-cause path: an
                    # unknown Cause ID may be introduced only on an active
                    # path with a matching SUPPORTED update and exact evidence
                    # in the current visible span. A known Cause on a
                    # different path is never treated as a new introduction.
                    candidate_update = updates_by_key.get(key)
                    candidate = candidate_update.cause if candidate_update is not None else None
                    valid_new_cause = bool(
                        key[0] not in active_cause_ids
                        and str(relation.relation).upper() == "INTRODUCES"
                        and key[1] in active_path_ids
                        and candidate is not None
                        and candidate.support_status == "SUPPORTED"
                        and candidate.evidence
                        and _node_contains_exact_evidence(visible_current, candidate.evidence)
                    )
                    if valid_new_cause:
                        root_relations.append(relation)
                    else:
                        audit.append({
                            "kind": "relation",
                            "cause_id": key[0],
                            "incoming_path_id": key[1],
                            "status": "NON_ROOT_RELATION_REJECTED",
                            "reason": (
                                "CAUSE_PATH_ID_MISMATCH"
                                if key[0] in active_cause_ids
                                else "UNKNOWN_CAUSE_ID"
                            ),
                        })
                    continue
                update = updates_by_key.get(key)
                candidate = update.cause if update is not None else None
                evidence_valid = bool(
                    relation.relation == "INTRODUCES"
                    and key[1] in active_path_ids
                    and candidate is not None
                    and candidate.support_status == "SUPPORTED"
                    and candidate.support_status != "REFUTED"
                    and candidate.evidence
                    and _node_contains_exact_evidence(visible_current, candidate.evidence)
                    and candidate.localized_span_id in (None, prepared["node_id"])
                )
                if evidence_valid:
                    valid_root_keys.add(key)
                    rooted_cause = candidate.model_copy(update={
                        "support_status": "SUPPORTED",
                        "causal_status": "ROOT_CAUSE",
                        "localized_span_id": prepared["node_id"],
                        "root_span_id": prepared["node_id"],
                        "should_propagate_upstream": False,
                    })
                    rooted_update = update.model_copy(update={"cause": rooted_cause})
                    normalized_updates = [rooted_update if item is update else item for item in normalized_updates]
                    updates_by_key[key] = rooted_update
                    root_relations.append(relation)
                else:
                    root_relations.append(relation.model_copy(update={"is_root_cause": False}))
                    audit.append({
                        "kind": "root_cause", "cause_id": key[0],
                        "incoming_path_id": key[1], "status": "ROOT_CAUSE_REJECTED_UNSUPPORTED",
                        "reason": "requires_path_INTRODUCES_supported_update_exact_current_span_evidence",
                    })

            for update in normalized_updates:
                raw = update.model_dump()
                nested = raw.get("cause") or {}
                cause_id = str(nested.get("cause_id", ""))
                path_id = str(raw.get("incoming_path_id", ""))
                key = (cause_id, path_id)
                if nested.get("causal_status") == "ROOT_CAUSE" and key not in valid_root_keys:
                    downgraded_cause = update.cause.model_copy(update={
                        "causal_status": "UNKNOWN",
                        "root_span_id": None,
                        "should_propagate_upstream": update.cause.support_status != "REFUTED",
                    })
                    update = update.model_copy(update={"cause": downgraded_cause})
                    raw = update.model_dump()
                    nested = raw.get("cause") or {}
                    audit.append({
                        "kind": "root_cause", "cause_id": cause_id,
                        "incoming_path_id": path_id,
                        "status": "ROOT_CAUSE_REJECTED_WITHOUT_VALIDATED_RELATION",
                    })
                if key in active_keys:
                    accepted.append(update)
                    continue
                relation = relation_by_key.get(key)
                grounded = bool(
                    path_id in active_path_ids and relation
                    and str(relation.get("relation", "")).upper() == "INTRODUCES"
                    and nested.get("support_status") == "SUPPORTED"
                    and _node_contains_exact_evidence(visible_current, str(nested.get("evidence") or ""))
                )
                if grounded:
                    accepted.append(update)
                else:
                    audit.append({
                        "kind": "update", "cause_id": cause_id,
                        "incoming_path_id": path_id or None,
                        "status": "NEW_OR_UNMATCHED_CAUSE_REJECTED",
                        "reason": "requires_existing_path_INTRODUCES_SUPPORTED_exact_visible_evidence",
                    })
            prepared["accepted_h1_updates"] = accepted
            adjusted = cascade.model_copy(deep=True)
            adjusted.cause_updates = accepted
            adjusted.cause_relations = root_relations
            return adjusted, audit

        def _shared_after_node(
            node_id: str, prepared: Dict[str, Any], diagnosis: NodeDiagnosis,
            active_before: List[Dict[str, Any]], active_after: List[Dict[str, Any]],
        ) -> Dict[str, Any]:
            local = diagnosis.local
            cascade = diagnosis.cascade
            if h1_mode:
                relations = [
                    {"span_id": node_id, **item.model_dump()}
                    for item in cascade.cause_relations
                ]
                cascade_relations.extend(relations)
                state_audit = list(diagnosis.metadata.get("cause_state_audit") or [])
                accepted_updates = prepared.get("accepted_h1_updates", [])
                updated_keys = {
                    (str((item.get("cause") or {}).get("cause_id", "")), str(item.get("incoming_path_id", "")))
                    for item in active_after if isinstance(item, dict)
                }
                before_causes = [item["cause"] for item in active_before if isinstance(item, dict) and isinstance(item.get("cause"), dict)]
                after_causes = [item["cause"] for item in active_after if isinstance(item, dict) and isinstance(item.get("cause"), dict)]
                update_audit = state_audit
                incoming_path_ids = sorted({str(item.get("incoming_path_id", "")) for item in active_before})
            else:
                relations = [{"span_id": node_id, **item.model_dump()} for item in cascade.cause_relations]
                cascade_relations.extend(relations)
                before_causes = active_before
                after_causes = active_after
                update_audit = []
                incoming_path_ids = []
            record = {
                "span_id": node_id,
                "local_has_error": local.has_error,
                "local_status": local.status,
                "local_confidence": local.confidence,
                "local_reasoning": local.reasoning,
                "local_errors": [item.model_dump() for item in local.errors],
                "cascade_status": cascade.status,
                "cascade_confidence": cascade.confidence,
                "cascade_reasoning": cascade.reasoning,
                "active_causes_before": [item.model_dump() if hasattr(item, "model_dump") else item for item in before_causes],
                "cause_relations": [item.model_dump() for item in cascade.cause_relations],
                "cause_updates": [item.model_dump() for item in cascade.cause_updates],
                "stopped_cause_ids": list(cascade.stopped_cause_ids),
                "active_causes_after": [item.model_dump() if hasattr(item, "model_dump") else item for item in after_causes],
                "h1_incoming_path_ids": incoming_path_ids,
                "h1_update_audit": update_audit if h1_mode else [],
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
            if h1_mode:
                record.update({
                    "cascade_evidence_validation_applied": diagnosis.metadata.get(
                        "cascade_evidence_validation_applied", False,
                    ),
                    "cascade_evidence_validation_version": diagnosis.metadata.get(
                        "cascade_evidence_validation_version", "not_applicable",
                    ),
                    "rejected_root_count": diagnosis.metadata.get("rejected_root_count", 0),
                })
            if prepared.get("prompt_context_audit") is not None:
                record["prompt_context_audit"] = prepared["prompt_context_audit"]
            diagnosed_history.append(record)
            return {"history_record": record}

        class _TrailSharedAdapter:
            cascade_enabled = self.config.cascade_enabled
            cascade_call_mode = self.cascade_call_mode
            separate_order = self.config.separate_order
            initialize_hypotheses = staticmethod(_shared_initialize_hypotheses)
            prepare_node = staticmethod(_shared_prepare_node)
            inspect_local = staticmethod(_shared_inspect_local)
            inspect_local_with_cascade = staticmethod(_shared_inspect_local_with_cascade)
            inspect_cascade = staticmethod(_shared_inspect_cascade)
            inspect_combined = staticmethod(_shared_inspect_combined)
            finalize_diagnosis = staticmethod(_shared_finalize_diagnosis)
            prepare_cascade_delta = staticmethod(_shared_prepare_cascade_delta)
            after_node = staticmethod(_shared_after_node)

        normalized_nodes = [
            {**dict(node), "node_id": str(node_id)} for node_id, node in self.node_map.items()
        ]
        normalized_edges: List[Dict[str, Any]] = []
        for edge in self.edges:
            if not isinstance(edge, dict):
                continue
            src = get_field_with_fallback(edge, [self.config.edge_src_key, "source", "src"])
            dst = get_field_with_fallback(edge, [self.config.edge_dst_key, "target", "dst"])
            if src is None or dst is None:
                continue
            normalized_edges.append({
                **edge,
                "src": str(src),
                "dst": str(dst),
                "kind": str(get_field_with_fallback(edge, [self.config.edge_kind_key, "type", "kind"], "unknown")).lower(),
            })

        parent_order_by_target = {
            node_id: self.get_upstream_traversal_ids(node_id)
            for node_id in self.node_map
        }
        if h1_mode:
            shared_edges = normalized_edges
            schedule_mode = (
                "reverse_topological_all_nodes"
                if self.config.trail_schedule_scope == "all_nodes"
                else "reverse_topological"
            )
            node_order = sorted(self.node_map)
            parent_order = None
        else:
            shared_edges = [
                edge for edge in normalized_edges
                if edge["src"] in parent_order_by_target.get(edge["dst"], ())
            ]
            schedule_mode = (
                "all_nodes" if self.config.trail_schedule_scope == "all_nodes"
                else "reachable"
            )
            node_order = list(self.node_map)
            parent_order = parent_order_by_target
        edge_priority: Dict[str, int] = {}
        for kind in self.config.data_flow_kinds:
            edge_priority[str(kind).lower()] = 0
        for kind in self.config.execution_flow_kinds:
            edge_priority[str(kind).lower()] = 1
        for kind in self.config.parent_kinds | {"contains"}:
            edge_priority[str(kind).lower()] = 2

        shared_run = SharedDiagnosisEngine(max_causes=self.config.max_causes).run(
            graph={"nodes": normalized_nodes, "edges": shared_edges},
            adapter=_TrailSharedAdapter(),
            cause_policy=state_policy,
            entry_node_id=final_node_id,
            traversal_mode=self.traversal_mode,
            schedule_mode=schedule_mode,
            node_ids=node_order,
            edge_priority=edge_priority,
            parent_order_by_target=parent_order,
            edge_policy=selected_edge_policy,
        )
        propagated_causes_by_node = shared_run.outgoing_causes
        h1_schedule = {
            "order": shared_run.schedule,
            "cycle": bool(shared_run.cycle_nodes),
            "cyclic_nodes": list(shared_run.cycle_nodes),
            "blocked_nodes": list(shared_run.blocked_nodes),
            "fallback": "bounded_path_worklist" if shared_run.cycle_nodes else None,
            "node_visit_count": shared_run.node_visit_count,
            "processed_cause_path_states": shared_run.processed_cause_path_states,
            "cycle_terminated_paths": shared_run.cycle_terminated_paths,
            "path_state_budget": shared_run.path_state_budget,
            "path_state_budget_exhausted": shared_run.path_state_budget_exhausted,
            "partial_coverage": shared_run.partial_coverage,
            "graph_validation": shared_run.graph_validation,
            "edge_policy": selected_edge_policy.name,
            "edge_policy_fingerprint": selected_edge_policy.fingerprint(),
        }
        if h1_mode:
            h1_path_state_count = state_policy.path_state_count
            h1_path_audit = list(h1_audit_pending) + [
                {"span_id": item.get("node_id"), **audit}
                for item in shared_run.history
                for audit in (item.get("cause_state_audit") or [])
            ]
        all_detected_errors = deduplicate_errors(all_detected_errors)

        root_cause_relations: List[Dict[str, Any]] = []
        root_cause_span_ids: Set[str] = set()
        all_final_causes: Dict[Tuple[str, str], FailureCause] = {}
        all_active_cause_path_states: List[Dict[str, Any]] = []
        root_cause_path_states: List[Dict[str, Any]] = []
        root_cause_causes: List[FailureCause] = []

        if self.config.cascade_enabled:
            root_cause_relations = [
                rel for rel in cascade_relations if rel.get("is_root_cause") is True
            ]
            root_cause_span_ids = {
                str(rel["span_id"]) for rel in root_cause_relations if rel.get("span_id")
            }
            all_final_causes, all_active_cause_path_states = _aggregate_final_cause_states(
                propagated_causes_by_node,
                path_scoped=h1_mode,
            )
            if h1_mode:
                root_cause_path_states = [
                    state for state in all_active_cause_path_states
                    if state["cause"].get("causal_status") == "ROOT_CAUSE"
                ]
                for state in root_cause_path_states:
                    root_span_id = state["cause"].get("root_span_id")
                    if root_span_id:
                        root_cause_span_ids.add(str(root_span_id))
                root_cause_causes = [
                    FailureCause.model_validate(state["cause"])
                    for state in root_cause_path_states
                ]
            else:
                for cause in all_final_causes.values():
                    if cause.causal_status == "ROOT_CAUSE" and cause.root_span_id:
                        root_cause_span_ids.add(str(cause.root_span_id))
                root_cause_causes = [
                    cause
                    for cause in all_final_causes.values()
                    if cause.causal_status == "ROOT_CAUSE"
                ]

        cascade_summary = {
            "enabled": self.config.cascade_enabled,
            "cause_state_mode": self.config.cause_state_mode,
            "h1_edge_policy": "dependency_only" if h1_mode else None,
            "cascade_call_mode": self.cascade_call_mode,
            "separate_order": (
                self.separate_order
                if self.config.cascade_enabled and self.cascade_call_mode == "separate"
                else "not_applicable"
            ),
            "prompt_context_policy": self.prompt_context_policy,
            "max_prompt_upstream_chars": self.config.max_prompt_upstream_chars,
            "traversal_mode": self.traversal_mode,
            "final_node_id": final_node_id,
            "ground_truth": ground_truth,
            "final_failure": final_failure.model_dump(),
            "failure_established": final_failure.has_failure,
            "diagnosis_mode": (
                "GROUND_TRUTH" if ground_truth is not None else "TRACE_INTERNAL_EVIDENCE"
            ),
            "root_cause_spans": sorted(root_cause_span_ids),
            "root_cause_relations": root_cause_relations,
            "root_cause_causes": [
                cause.model_dump() for cause in root_cause_causes
            ],
            "all_active_causes": [
                cause.model_dump() for cause in all_final_causes.values()
            ],
            "all_active_cause_path_states": (
                all_active_cause_path_states if h1_mode else None
            ),
            "root_cause_path_states": root_cause_path_states if h1_mode else None,
            "backward_diagnosis": diagnosed_history,
            "cause_propagation": cascade_relations,
            "h1_scheduler": (
                {
                    "order": h1_schedule.get("order", []),
                    "cycle_detected": h1_schedule.get("cycle", False),
                    "cyclic_nodes": h1_schedule.get("cyclic_nodes", []),
                    "blocked_nodes": h1_schedule.get("blocked_nodes", []),
                    "fallback": h1_schedule.get("fallback"),
                    "path_state_count": h1_path_state_count,
                    "path_state_limit": self.config.max_cause_path_states,
                    "node_visit_count": h1_schedule.get("node_visit_count", {}),
                    "processed_cause_path_states": h1_schedule.get("processed_cause_path_states", 0),
                    "cycle_terminated_paths": h1_schedule.get("cycle_terminated_paths", 0),
                    "path_state_budget": h1_schedule.get("path_state_budget"),
                    "path_state_budget_exhausted": h1_schedule.get("path_state_budget_exhausted", False),
                    "partial_coverage": h1_schedule.get("partial_coverage", False),
                    "graph_validation": h1_schedule.get("graph_validation", {}),
                    "edge_policy": h1_schedule.get("edge_policy"),
                    "edge_policy_fingerprint": h1_schedule.get("edge_policy_fingerprint"),
                    "path_audit": h1_path_audit,
                }
                if h1_mode else None
            ),
        }
        if (
            h1_mode
            and self.config.cascade_enabled
            and self.cascade_call_mode == "separate"
            and self.separate_order == "cascade_then_local"
        ):
            cascade_summary["cascade_local_evidence_validation_version"] = (
                _CASCADE_LOCAL_EVIDENCE_VALIDATION_VERSION
            )
        if self.config.final_hypothesis_policy != "gated":
            cascade_summary.update({
                "final_hypothesis_policy": self.config.final_hypothesis_policy,
                "has_failure": bool(final_failure.has_failure),
                "seed_cause_count": seed_cause_count,
                "hypothesis_seed_audit": hypothesis_seed_audit,
            })

        if not self.config.enable_scoring:
            output_dict = TRAILDiagnosisOutput(
                errors=all_detected_errors,
                scores=[],
            ).model_dump()
        elif self.config.use_llm_error_synthesis:
            final_report = self.llm_service.synthesize_trail_report(
                user_query=user_query,
                detected_errors=all_detected_errors,
                all_node_diagnoses=diagnosed_history,
            )
            output_dict = final_report.model_dump()
        else:
            try:
                scores = self.llm_service.score_trace_report(
                    user_query=user_query,
                    detected_errors=all_detected_errors,
                    final_failure=final_failure,
                    root_cause_causes=root_cause_causes,
                    root_cause_relations=root_cause_relations,
                ).scores
            except Exception:
                scores = [empty_score()]

            output_dict = TRAILDiagnosisOutput(
                errors=all_detected_errors,
                scores=scores,
            ).model_dump()

        coverage = _trail_prediction_coverage(
            shared_run,
            cascade_enabled=self.config.cascade_enabled,
            errors=output_dict.get("errors"),
            retained_node_ids=list(self.node_map),
        )
        output_dict["cascade_failure"] = cascade_summary
        output_dict.update(coverage)
        output_dict["diagnosis_mode"] = (
            self.cascade_call_mode if self.config.cascade_enabled else "disabled"
        )
        output_dict["separate_order"] = (
            self.separate_order
            if self.config.cascade_enabled and self.cascade_call_mode == "separate"
            else "not_applicable"
        )
        if self.config.trail_schedule_scope != "legacy":
            output_dict["trail_schedule_scope"] = self.config.trail_schedule_scope
        return output_dict


import os
import json
import hashlib
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener, urlopen
from urllib.error import HTTPError

from pydantic import BaseModel, ValidationError
from .request_budget import (
    PersistentRequestBudget,
    PersistentTokenCostBudget,
    note_budgeted_http_attempt,
    request_budget_environment_configured,
)


_CLAUDE_UNSUPPORTED_NUMBER_CONSTRAINTS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
)

_AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
_STRUCTURED_AUDIT_LOCK = threading.Lock()
_STRUCTURED_AUDIT_REQUEST_INDEX = 0
_STRUCTURED_AUDIT_RUN_ID = uuid.uuid4().hex
_ACTIVE_STRUCTURED_REQUEST: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "active_structured_request", default=None
)
_CURRENT_TRACE_ID: ContextVar[Optional[str]] = ContextVar("current_trace_id", default=None)
_ACTIVE_API_DIAGNOSTIC: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "active_api_diagnostic", default=None
)


@contextmanager
def trace_audit_context(trace_id: Any):
    """Attach the current graph Trace ID to low-level request audit records."""
    token = _CURRENT_TRACE_ID.set(str(trace_id) if trace_id is not None else None)
    try:
        yield
    finally:
        _CURRENT_TRACE_ID.reset(token)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _schema_from_request_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    response_format = payload.get("response_format")
    if isinstance(response_format, dict):
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, dict):
            return json_schema.get("schema")
    output_config = payload.get("output_config")
    if isinstance(output_config, dict):
        output_format = output_config.get("format")
        if isinstance(output_format, dict):
            return output_format.get("schema")
    return None


def _normalized_prompt_messages(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Normalize OpenAI and Anthropic message envelopes for prompt comparison."""
    messages: list[Dict[str, Any]] = []
    system = payload.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    raw_messages = payload.get("messages")
    if isinstance(raw_messages, list):
        for message in raw_messages:
            if isinstance(message, dict):
                messages.append(
                    {
                        "role": message.get("role", "user"),
                        "content": message.get("content", ""),
                    }
                )
    return messages


def _safe_endpoint(url: Any) -> str:
    """Keep endpoint/version information while redacting URL credentials."""
    parsed = urlsplit(str(url))
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    query = []
    sensitive = ("key", "token", "secret", "auth", "signature", "credential")
    for name, value in parse_qsl(parsed.query, keep_blank_values=True):
        if any(part in name.lower() for part in sensitive):
            value = "<redacted>"
        query.append((name, value))
    return urlunsplit((parsed.scheme, host, parsed.path, urlencode(query), ""))


def _record_wire_request(
    *,
    endpoint: str,
    method: str,
    payload: Dict[str, Any],
    attempt: Optional[int] = None,
) -> None:
    """Save the serialized request body observed immediately before transport."""
    audit_path = os.environ.get("TRAIL_REQUEST_AUDIT_PATH")
    context = _ACTIVE_STRUCTURED_REQUEST.get()
    if not audit_path or context is None:
        return
    schema = _schema_from_request_payload(payload)
    canonical_schema = _canonical_json(schema) if schema is not None else None
    prompt_messages = _normalized_prompt_messages(payload)
    canonical_prompt = _canonical_json(prompt_messages)
    row = {
        "record_type": "wire_request",
        "request_index": context.get("request_index"),
        "run_id": _STRUCTURED_AUDIT_RUN_ID,
        "experiment": os.environ.get("TRAIL_EXPERIMENT_LABEL"),
        "trace_id": _CURRENT_TRACE_ID.get(),
        "attempt": os.environ.get("TRAIL_RUN_ATTEMPT", "1"),
        "http_attempt": attempt,
        "transport": context.get("transport"),
        "model": payload.get("model"),
        "response_model": context.get("response_model"),
        "diagnosis_path": os.environ.get("TRAIL_DIAGNOSIS_PATH", "model_default"),
        "schema_mode": "provider_default",
        "endpoint": _safe_endpoint(endpoint),
        "method": method,
        "request_parameters": {
            key: payload[key]
            for key in (
                "max_tokens",
                "max_completion_tokens",
                "temperature",
                "top_p",
                "top_k",
                "reasoning_effort",
                "effort",
            )
            if key in payload
        },
        "omitted_sampling_parameters": [
            key for key in ("temperature", "top_p", "top_k") if key not in payload
        ],
        "schema_name": context.get("response_model"),
        "schema": schema,
        "schema_canonical_json": canonical_schema,
        "normalized_prompt_messages": prompt_messages,
        "prompt_sha256": hashlib.sha256(canonical_prompt.encode("utf-8")).hexdigest(),
        "schema_sha256": (
            hashlib.sha256(canonical_schema.encode("utf-8")).hexdigest()
            if canonical_schema is not None
            else None
        ),
        # Body intentionally excludes all HTTP headers, including API credentials.
        "request_body": payload,
    }
    _append_jsonl(audit_path, row)


def _record_wire_response(
    *,
    status_code: Optional[int],
    response_body: bytes,
    request_url: Optional[str] = None,
) -> None:
    """Save the raw response body and response metadata before SDK parsing."""
    audit_path = os.environ.get("TRAIL_REQUEST_AUDIT_PATH")
    context = _ACTIVE_STRUCTURED_REQUEST.get()
    if not audit_path or context is None:
        return
    raw_text = response_body.decode("utf-8", errors="replace")
    try:
        response_json = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        response_json = None
    finish_reason = None
    usage = None
    if isinstance(response_json, dict):
        finish_reason = response_json.get("stop_reason")
        usage = response_json.get("usage")
        choices = response_json.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason", finish_reason)
    _append_jsonl(
        audit_path,
        {
            "record_type": "wire_response",
            "request_index": context.get("request_index"),
            "run_id": _STRUCTURED_AUDIT_RUN_ID,
            "experiment": os.environ.get("TRAIL_EXPERIMENT_LABEL"),
            "trace_id": _CURRENT_TRACE_ID.get(),
            "attempt": os.environ.get("TRAIL_RUN_ATTEMPT", "1"),
            "status_code": status_code,
            "endpoint": _safe_endpoint(request_url) if request_url else None,
            "finish_reason": finish_reason,
            "usage": usage,
            "response_json": response_json,
            "raw_response_body": raw_text,
        },
    )


def _diagnostic_redact(value: Any) -> Optional[str]:
    if value is None:
        return None
    message = str(value)
    active = _ACTIVE_API_DIAGNOSTIC.get()
    key = str(active.get("api_key") or "") if active else ""
    if key:
        message = message.replace(key, "<redacted>")
    message = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer <redacted>", message)
    message = re.sub(
        r"(?i)(authorization|api[-_ ]key|x-api-key)(\s*[:=]\s*)[^\s,;]+",
        r"\1\2<redacted>", message,
    )
    return message[:4000]


def _http_diagnostic_request_hook(request: Any) -> None:
    active = _ACTIVE_API_DIAGNOSTIC.get()
    if active is None:
        return
    parsed = urlsplit(str(request.url))
    active.setdefault("http_attempts", []).append({
        "attempt": len(active.get("http_attempts", [])) + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "started_monotonic": time.perf_counter(),
        "method": str(request.method),
        "endpoint_hostname": parsed.hostname,
        "endpoint_path": parsed.path,
        "http_status_code": None,
        "azure_error_code": None,
        "error_message": None,
        "finish_reason": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "duration_seconds": None,
    })


def _http_request_budget_hook(_request: Any) -> None:
    """Stop before sending a wire retry when the persistent cap is exhausted."""
    note_budgeted_http_attempt()


def _http_diagnostic_response_hook(response: Any) -> None:
    active = _ACTIVE_API_DIAGNOSTIC.get()
    if active is None or not active.get("http_attempts"):
        return
    attempt = active["http_attempts"][-1]
    attempt["http_status_code"] = getattr(response, "status_code", None)
    try:
        response.read()
        payload = response.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            attempt["azure_error_code"] = error.get("code")
            attempt["error_message"] = _diagnostic_redact(error.get("message"))
        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            attempt["finish_reason"] = choices[0].get("finish_reason")
        usage = payload.get("usage")
        if isinstance(usage, dict):
            attempt["input_tokens"] = usage.get("prompt_tokens", usage.get("input_tokens"))
            attempt["output_tokens"] = usage.get("completion_tokens", usage.get("output_tokens"))
            attempt["total_tokens"] = usage.get("total_tokens")
    started = attempt.pop("started_monotonic", None)
    if started is not None:
        attempt["duration_seconds"] = round(time.perf_counter() - started, 4)


def _diagnostic_error_category(exc: BaseException, status_code: Any, error_code: Any) -> str:
    try:
        status = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        status = None
    code = str(error_code or "").lower()
    if status == 401:
        return "http_401_unauthorized"
    if status == 403:
        return "http_403_forbidden"
    if status == 404 and code == "deploymentnotfound":
        return "http_404_deployment_not_found"
    if status == 404:
        return "http_404_not_found"
    if status == 429:
        return "http_429_rate_limited"
    if status == 400:
        return "http_400_invalid_request"
    if status is not None and 500 <= status <= 599:
        return "http_5xx_server_error"
    chain: list[BaseException] = []
    current: Optional[BaseException] = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    names = " ".join(type(item).__name__.lower() for item in chain)
    messages = " ".join(str(item).lower() for item in chain)
    if any("gaierror" in type(item).__name__.lower() for item in chain) or any(
        part in messages for part in ("name or service not known", "nodename nor servname", "name resolution")
    ):
        return "dns_failure"
    if any("ssl" in type(item).__name__.lower() for item in chain) or any(
        part in messages for part in ("certificate verify failed", "tls handshake", "ssl:")
    ):
        return "tls_https_failure"
    if "connecttimeout" in names or "connect timeout" in messages:
        return "connection_timeout"
    if "readtimeout" in names or "read timeout" in messages:
        return "read_timeout"
    if "connecterror" in names or "connectionrefused" in names:
        return "tcp_connection_failure"
    if "deploymentnotfound" in code:
        return "deployment_not_found"
    if "badrequest" in names or "invalidrequest" in names:
        return "invalid_request"
    return "unclassified_api_failure"


def _http_request_audit_hook(request: Any) -> None:
    if not os.environ.get("TRAIL_REQUEST_AUDIT_PATH"):
        return
    try:
        raw = request.read()
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        context = _ACTIVE_STRUCTURED_REQUEST.get()
        if context is not None:
            _append_jsonl(
                os.environ["TRAIL_REQUEST_AUDIT_PATH"],
                {
                    "record_type": "wire_capture_error",
                    "request_index": context.get("request_index"),
                    "stage": "request_body",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise RuntimeError("Could not audit the final structured HTTP request body.") from exc
    _record_wire_request(
        endpoint=str(request.url),
        method=str(request.method),
        payload=payload,
    )


def _http_response_audit_hook(response: Any) -> None:
    if not os.environ.get("TRAIL_REQUEST_AUDIT_PATH"):
        return
    response.read()
    _record_wire_response(
        status_code=int(response.status_code),
        response_body=response.content,
        request_url=str(response.request.url),
    )


def _install_http_audit_hooks(http_client: Any) -> None:
    if http_client is None:
        return
    hooks = getattr(http_client, "event_hooks", None)
    if not isinstance(hooks, dict):
        if (
            os.environ.get("TRAIL_REQUEST_AUDIT_PATH")
            or os.environ.get("TRAIL_API_DIAGNOSTIC_PATH")
            or request_budget_environment_configured()
        ):
            raise RuntimeError(
                "Request auditing or budgeting requires an HTTP client exposing request/response event hooks."
            )
        return
    if os.environ.get("TRAIL_REQUEST_AUDIT_PATH") and not getattr(
        http_client, "_trail_request_audit_hooks_installed", False
    ):
        hooks.setdefault("request", []).append(_http_request_audit_hook)
        hooks.setdefault("response", []).append(_http_response_audit_hook)
        try:
            http_client._trail_request_audit_hooks_installed = True
        except Exception:
            pass
    if os.environ.get("TRAIL_API_DIAGNOSTIC_PATH") and not getattr(
        http_client, "_trail_api_diagnostic_hooks_installed", False
    ):
        hooks.setdefault("request", []).append(_http_diagnostic_request_hook)
        hooks.setdefault("response", []).append(_http_diagnostic_response_hook)
        try:
            http_client._trail_api_diagnostic_hooks_installed = True
        except Exception:
            pass
    if request_budget_environment_configured() and not getattr(
        http_client, "_trail_request_budget_hook_installed", False
    ):
        hooks.setdefault("request", []).append(_http_request_budget_hook)
        try:
            http_client._trail_request_budget_hook_installed = True
        except Exception:
            pass
    try:
        http_client._trail_audit_hooks_installed = True
    except Exception:
        pass


def _call_with_structured_context(
    request_index: Optional[int],
    *,
    transport: str,
    response_model: type[BaseModel],
    call: Any,
    request_budget_estimate: Optional[Dict[str, Any]] = None,
) -> Any:
    token = _ACTIVE_STRUCTURED_REQUEST.set(
        {
            "request_index": request_index,
            "transport": transport,
            "response_model": response_model.__name__,
            "request_budget_estimate": request_budget_estimate,
        }
    )
    try:
        return call()
    finally:
        _ACTIVE_STRUCTURED_REQUEST.reset(token)


def _claude_compatible_json_schema(value: Any) -> Any:
    """Remove numeric JSON Schema keywords rejected by Claude-compatible APIs."""
    if isinstance(value, list):
        return [_claude_compatible_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    schema = {key: _claude_compatible_json_schema(item) for key, item in value.items()}
    if schema.get("type") not in {"number", "integer"}:
        return schema

    guidance = []
    for key in _CLAUDE_UNSUPPORTED_NUMBER_CONSTRAINTS:
        if key not in schema:
            continue
        bound = schema.pop(key)
        wording = {
            "minimum": f"at least {bound}",
            "maximum": f"at most {bound}",
            "exclusiveMinimum": f"greater than {bound}",
            "exclusiveMaximum": f"less than {bound}",
            "multipleOf": f"a multiple of {bound}",
        }[key]
        guidance.append(wording)
    if guidance:
        description = str(schema.get("description") or "").strip()
        schema["description"] = " ".join(
            part for part in (description, f"Value must be {' and '.join(guidance)}.") if part
        )
    return schema


def _openai_strict_json_schema(response_model: type[BaseModel]) -> Dict[str, Any]:
    """Return the schema the OpenAI SDK's Pydantic parse path sends."""
    try:
        from openai.lib._pydantic import to_strict_json_schema
    except ImportError:
        # Keep schema-only checks usable in environments without the OpenAI
        # package. At runtime, the SDK converter remains authoritative whenever
        # it is installed; this local mapping mirrors its strict transformations.
        return _strict_schema_mapping(response_model.model_json_schema())
    return to_strict_json_schema(response_model)






def _strict_schema_mapping(value: Any) -> Any:
    """Mirror OpenAI's strict-schema normalization for local schema checks."""
    root = value if isinstance(value, dict) else {}

    def normalize(node: Any, path: Tuple[str, ...] = ()) -> Any:
        if isinstance(node, list):
            return [normalize(item, (*path, str(index))) for index, item in enumerate(node)]
        if not isinstance(node, dict):
            return node

        schema = dict(node)
        for defs_key in ("$defs", "definitions"):
            definitions = schema.get(defs_key)
            if isinstance(definitions, dict):
                schema[defs_key] = {
                    name: normalize(definition, (*path, defs_key, name))
                    for name, definition in definitions.items()
                }

        if schema.get("type") == "object" and "additionalProperties" not in schema:
            schema["additionalProperties"] = False
        properties = schema.get("properties")
        if isinstance(properties, dict):
            schema["required"] = list(properties)
            schema["properties"] = {
                name: normalize(property_schema, (*path, "properties", name))
                for name, property_schema in properties.items()
            }
        items = schema.get("items")
        if isinstance(items, dict):
            schema["items"] = normalize(items, (*path, "items"))
        union = schema.get("anyOf")
        if isinstance(union, list):
            schema["anyOf"] = [
                normalize(item, (*path, "anyOf", str(index)))
                for index, item in enumerate(union)
            ]
        all_of = schema.get("allOf")
        if isinstance(all_of, list):
            if len(all_of) == 1 and isinstance(all_of[0], dict):
                schema.update(normalize(all_of[0], (*path, "allOf", "0")))
                schema.pop("allOf", None)
            else:
                schema["allOf"] = [
                    normalize(item, (*path, "allOf", str(index)))
                    for index, item in enumerate(all_of)
                ]
        if schema.get("default", object()) is None:
            schema.pop("default")

        ref = schema.get("$ref")
        if isinstance(ref, str) and len(schema) > 1 and ref.startswith("#/"):
            resolved: Any = root
            for part in ref[2:].split("/"):
                part = part.replace("~1", "/").replace("~0", "~")
                resolved = resolved[part]
            if isinstance(resolved, dict):
                merged = {**resolved, **schema}
                merged.pop("$ref", None)
                return normalize(merged, path)
        return schema

    return normalize(value)


def _record_structured_request(
    *,
    model: str,
    response_model: type[BaseModel],
    transport: str,
    schema: Any,
    messages: Any,
) -> Optional[int]:
    """Optionally save request metadata, and full prompt/schema only in audit mode."""
    audit_path = os.environ.get("TRAIL_REQUEST_AUDIT_PATH")
    stats_path = os.environ.get("TRAIL_REQUEST_STATS_PATH")
    if not audit_path and not stats_path:
        return None
    global _STRUCTURED_AUDIT_REQUEST_INDEX

    if transport == "anthropic_native":
        envelope = {"type": "json_schema", "schema": schema}
        system_parts = []
        wire_messages = []
        for message in messages or []:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if role in {"system", "developer"}:
                system_parts.append(str(content))
            else:
                wire_messages.append({"role": role, "content": content})
        wire_prompt = {"system": "\n".join(system_parts), "messages": wire_messages}
    else:
        envelope = {"name": response_model.__name__, "strict": True, "schema": schema}
        wire_prompt = {"messages": messages}
    serialized_schema = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))

    with _STRUCTURED_AUDIT_LOCK:
        _STRUCTURED_AUDIT_REQUEST_INDEX += 1
        request_index = _STRUCTURED_AUDIT_REQUEST_INDEX
        common = {
            "request_index": request_index,
            "run_id": _STRUCTURED_AUDIT_RUN_ID,
            "experiment": os.environ.get("TRAIL_EXPERIMENT_LABEL"),
            "trace_id": _CURRENT_TRACE_ID.get(),
            "attempt": os.environ.get("TRAIL_RUN_ATTEMPT", "1"),
            "model": model,
            "provider": "anthropic" if transport == "anthropic_native" else "openai-compatible",
            "response_model": response_model.__name__,
            "schema_name": response_model.__name__,
            "schema_strategy": "provider_default",
            "transport": transport,
            "schema_sha256": hashlib.sha256(serialized_schema.encode("utf-8")).hexdigest(),
            "diagnosis_path": os.environ.get("TRAIL_DIAGNOSIS_PATH", "model_default"),
        }
        if stats_path:
            _append_jsonl(
                stats_path,
                {"record_type": "request", **common},
            )
        if audit_path:
            row = {
                **common,
                "schema_envelope": envelope,
                "prompt_messages": messages,
                "wire_prompt": wire_prompt,
            }
            _append_jsonl(audit_path, row)
        return request_index


def _append_jsonl(path_value: str, row: Dict[str, Any]) -> None:
    path = os.path.abspath(path_value)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as audit_file:
        audit_file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _response_usage(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if isinstance(response, dict):
        usage = response.get("usage")

    def get(name: str) -> Any:
        if isinstance(usage, dict):
            return usage.get(name)
        return getattr(usage, name, None) if usage is not None else None

    input_tokens = get("prompt_tokens")
    if input_tokens is None:
        input_tokens = get("input_tokens")
    output_tokens = get("completion_tokens")
    if output_tokens is None:
        output_tokens = get("output_tokens")
    total_tokens = get("total_tokens")
    try:
        input_tokens = int(input_tokens) if input_tokens is not None else None
    except (TypeError, ValueError):
        input_tokens = None
    try:
        output_tokens = int(output_tokens) if output_tokens is not None else None
    except (TypeError, ValueError):
        output_tokens = None
    try:
        total_tokens = int(total_tokens) if total_tokens is not None else None
    except (TypeError, ValueError):
        total_tokens = None
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "retry_count": None,
        "usage_reported": any(
            value is not None for value in (input_tokens, output_tokens, total_tokens)
        ),
    }


def _server_usage_snapshot(server: Any) -> Optional[Dict[str, Any]]:
    snapshot = getattr(server, "usage_snapshot", None)
    if not callable(snapshot):
        return None
    try:
        value = snapshot()
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _server_usage_delta(
    before: Optional[Dict[str, Any]], after: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if before is None or after is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "retry_count": None,
            "usage_reported": False,
        }
    return {
        "input_tokens": after.get("prompt_tokens", 0) - before.get("prompt_tokens", 0),
        "output_tokens": after.get("completion_tokens", 0) - before.get("completion_tokens", 0),
        "total_tokens": after.get("total_tokens", 0) - before.get("total_tokens", 0),
        "retry_count": after.get("max_token_retry_count", 0)
        - before.get("max_token_retry_count", 0),
        "usage_reported": (
            after.get("reported_usage_count", 0) > before.get("reported_usage_count", 0)
        ),
    }


def _response_usage_with_retry(
    response: Any,
    before: Optional[Dict[str, Any]],
    server: Any,
) -> Dict[str, Any]:
    usage = _response_usage(response)
    delta = _server_usage_delta(before, _server_usage_snapshot(server))
    if delta["usage_reported"]:
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            usage[key] = delta[key]
        usage["usage_reported"] = True
    usage["retry_count"] = delta["retry_count"]
    return usage


def _settle_request_cost_budget(
    server: Any,
    *,
    structured_context: Optional[Dict[str, Any]],
    response: Any,
) -> None:
    estimate = (structured_context or {}).get("request_budget_estimate") or {}
    reservation_id = estimate.get("cost_reservation_id")
    if not reservation_id:
        return
    log_path = os.environ.get("TRAIL_TOKEN_BUDGET_LOG_PATH")
    usage = _response_usage(response) if response is not None else _response_usage(None)
    if usage.get("input_tokens") is None or usage.get("output_tokens") is None:
        if log_path:
            _append_jsonl(log_path, {
                "decision": "COST_RESERVATION_RETAINED",
                "status": "USAGE_UNAVAILABLE",
                "trace_id": _CURRENT_TRACE_ID.get(),
                "request_stage": (structured_context or {}).get("response_model"),
                "cost_reservation_id": reservation_id,
                "projected_cost_reservation_usd": estimate.get("projected_cost_reservation_usd"),
            })
        return
    budget = PersistentTokenCostBudget.from_environment()
    if budget is None:
        raise RuntimeError("Live token-cost budget disappeared after request reservation.")
    actual_cost = budget.estimate_actual(usage["input_tokens"], usage["output_tokens"])
    settlement = budget.settle(reservation_id, actual_cost)
    if log_path:
        _append_jsonl(log_path, {
            "decision": "COST_SETTLED",
            "status": "USAGE_REPORTED",
            "trace_id": _CURRENT_TRACE_ID.get(),
            "request_stage": (structured_context or {}).get("response_model"),
            "cost_reservation_id": reservation_id,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "actual_estimated_cost_usd": round(actual_cost, 8),
            **settlement,
        })


class RequestTokenBudgetExceeded(RuntimeError):
    """A structured request was blocked before transport by the opt-in token gate."""

    stop_run = True

    def __init__(self, details: Dict[str, Any]):
        self.details = dict(details)
        self.request_budget_denial = dict(details)
        if details.get("status") == "CONTEXT_BUDGET_EXCEEDED":
            self.stop_run = False
        super().__init__(
            "CONTEXT_BUDGET_EXCEEDED: projected structured request exceeds the configured "
            f"limit ({details.get('projected_total_context')} > {details.get('projected_limit')})."
        )


def _request_budget_tokenizer(name: str) -> Any:
    try:
        import tiktoken
    except Exception as exc:
        raise RuntimeError("Tokenizer could not be loaded.") from exc
    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:
        raise RuntimeError(f"Tokenizer {name!r} is unavailable.") from exc


def _structured_request_token_estimate(
    *,
    model: str,
    response_model: type[BaseModel],
    schema: Any,
    messages: Any,
    transport: str,
    config: Dict[str, Any],
    requested_completion_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Conservatively count the actual prompt envelope and provider wire schema.

    This is an estimate, not provider usage. It deliberately fails closed when
    enabled if the tokenizer, messages, or schema cannot be counted.
    """
    if not isinstance(messages, list) or not isinstance(schema, (dict, list)):
        raise RuntimeError("Structured messages or wire schema are not countable.")
    tokenizer_name = str(config.get("tokenizer", "o200k_base"))
    encoder = _request_budget_tokenizer(tokenizer_name)
    if transport == "anthropic_native":
        schema_envelope: Any = {"type": "json_schema", "schema": schema}
        prompt_envelope: Any = {
            "system": "\n".join(
                str(item.get("content", "")) for item in messages
                if isinstance(item, dict) and item.get("role") in {"system", "developer"}
            ),
            "messages": [
                item for item in messages
                if isinstance(item, dict) and item.get("role", "user") not in {"system", "developer"}
            ],
        }
    else:
        schema_envelope = {"name": response_model.__name__, "strict": True, "schema": schema}
        prompt_envelope = {"messages": messages}
    try:
        message_tokens = len(encoder.encode(_canonical_json(prompt_envelope)))
        wire_schema_tokens = len(encoder.encode(_canonical_json(schema_envelope)))
    except Exception as exc:
        raise RuntimeError("Prompt or wire schema could not be tokenized.") from exc

    # Role separators and provider framing are not fully represented in the
    # JSON count. Add a fixed, auditable allowance before the 10% safety factor.
    format_overhead_tokens = 16 + 8 * len(messages)
    raw_input_estimate = message_tokens + wire_schema_tokens + format_overhead_tokens
    safety_factor = float(config.get("safety_factor", 1.10))
    completion_reserve = max(
        int(config.get("completion_reserve", 32768)),
        int(requested_completion_tokens or 0),
    )
    projected_input = int(__import__("math").ceil(raw_input_estimate * safety_factor))
    projected_total = projected_input + completion_reserve
    context_limit = int(config.get("context_limit", 272000))
    hard_limit = int(config.get("hard_projected_limit", 231200))
    projected_limit = min(hard_limit, int(context_limit))
    return {
        "model": model,
        "request_stage": os.environ.get("TRAIL_API_REQUEST_STAGE") or response_model.__name__,
        "response_model": response_model.__name__,
        "trace_id": _CURRENT_TRACE_ID.get(),
        "tokenizer": tokenizer_name,
        "estimate_kind": "ESTIMATED",
        "message_tokens": message_tokens,
        "wire_schema_tokens": wire_schema_tokens,
        "format_overhead_tokens": format_overhead_tokens,
        "raw_input_estimate": raw_input_estimate,
        "safety_factor": safety_factor,
        "estimated_input_tokens": projected_input,
        "completion_reserve": completion_reserve,
        "projected_total_context": projected_total,
        "context_limit": context_limit,
        "hard_projected_limit": hard_limit,
        "projected_limit": projected_limit,
        "prompt_sha256": hashlib.sha256(_canonical_json(prompt_envelope).encode("utf-8")).hexdigest(),
        "schema_sha256": hashlib.sha256(_canonical_json(schema_envelope).encode("utf-8")).hexdigest(),
    }


def _check_structured_request_budget(
    server: Any,
    *,
    model: str,
    response_model: type[BaseModel],
    schema: Any,
    messages: Any,
    transport: str,
    requested_completion_tokens: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    config = getattr(server, "request_token_budget_config", None)
    if not config:
        return None
    log_path = os.environ.get("TRAIL_TOKEN_BUDGET_LOG_PATH")
    try:
        estimate = _structured_request_token_estimate(
            model=model,
            response_model=response_model,
            schema=schema,
            messages=messages,
            transport=transport,
            config=dict(config),
            requested_completion_tokens=requested_completion_tokens,
        )
    except Exception as exc:
        details = {
            "decision": "BLOCKED",
            "status": "TOKEN_ESTIMATE_UNAVAILABLE",
            "model": model,
            "request_stage": os.environ.get("TRAIL_API_REQUEST_STAGE") or response_model.__name__,
            "response_model": response_model.__name__,
            "trace_id": _CURRENT_TRACE_ID.get(),
            "tokenizer": config.get("tokenizer", "o200k_base"),
            "prompt_sha256": hashlib.sha256(
                _canonical_json(messages).encode("utf-8", errors="replace")
            ).hexdigest() if isinstance(messages, list) else None,
            "schema_sha256": hashlib.sha256(
                _canonical_json(schema).encode("utf-8", errors="replace")
            ).hexdigest() if isinstance(schema, (dict, list)) else None,
            "reason": type(exc).__name__,
        }
        if log_path:
            _append_jsonl(log_path, details)
        error = RequestTokenBudgetExceeded(details)
        error.__cause__ = exc
        raise error
    estimate["decision"] = "ALLOW" if estimate["projected_total_context"] <= estimate["projected_limit"] else "BLOCKED"
    estimate["status"] = (
        "WITHIN_BUDGET" if estimate["decision"] == "ALLOW" else "CONTEXT_BUDGET_EXCEEDED"
    )
    if estimate["decision"] == "ALLOW":
        try:
            cost_budget = PersistentTokenCostBudget.from_environment()
            if cost_budget is not None:
                reservation_id = uuid.uuid4().hex
                projected_cost = cost_budget.estimate_actual(
                    estimate["estimated_input_tokens"], estimate["completion_reserve"],
                )
                estimate["cost_reservation_id"] = reservation_id
                estimate["projected_cost_reservation_usd"] = round(projected_cost, 8)
                estimate["cost_rates_usd_per_million"] = {
                    "input": cost_budget.input_rate,
                    "output": cost_budget.output_rate,
                }
                estimate["cost_budget"] = cost_budget.reserve(reservation_id, projected_cost)
        except Exception as exc:
            estimate["decision"] = "BLOCKED"
            estimate["status"] = "COST_BUDGET_EXCEEDED"
            estimate["reason"] = type(exc).__name__
            if log_path:
                _append_jsonl(log_path, estimate)
            if getattr(exc, "stop_run", False):
                exc.request_budget_denial = dict(estimate)
                raise
            error = RequestTokenBudgetExceeded(estimate)
            error.__cause__ = exc
            raise error
    if log_path:
        _append_jsonl(log_path, estimate)
    if estimate["decision"] == "BLOCKED":
        raise RequestTokenBudgetExceeded(estimate)
    return estimate


def _structured_error_category(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if (
        isinstance(exc, (ValidationError, ValueError))
        or "validation" in name
        or "parse" in name
        or "json" in name
    ):
        return "validation_failure"
    if "max_tokens" in message or "incomplete" in name or "length" in name:
        return "incomplete_output"
    if "refus" in message:
        return "refusal"
    return "api_failure"


def _record_structured_result(
    request_index: Optional[int],
    *,
    status: str,
    finish_reason: Optional[str] = None,
    error_type: Optional[str] = None,
    error_category: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
) -> None:
    """Append request outcome to metadata stats and/or full prompt audit."""
    audit_path = os.environ.get("TRAIL_REQUEST_AUDIT_PATH")
    stats_path = os.environ.get("TRAIL_REQUEST_STATS_PATH")
    if request_index is None or (not audit_path and not stats_path):
        return
    row = {
        "record_type": "result",
        "request_index": request_index,
        "run_id": _STRUCTURED_AUDIT_RUN_ID,
        "experiment": os.environ.get("TRAIL_EXPERIMENT_LABEL"),
        "attempt": os.environ.get("TRAIL_RUN_ATTEMPT", "1"),
        "status": status,
        "validation_status": (
            "PASS"
            if status == "SUCCESS"
            else "FAIL"
            if error_category == "validation_failure"
            else "NOT_VALIDATED"
        ),
        "finish_reason": finish_reason,
        "error_type": error_type,
        "error_category": error_category,
        "retry_count": (usage or {}).get("retry_count"),
        **(usage or _response_usage(None)),
    }
    with _STRUCTURED_AUDIT_LOCK:
        if stats_path:
            _append_jsonl(stats_path, row)
        if audit_path:
            _append_jsonl(audit_path, row)


def _response_finish_reason(response: Any) -> Optional[str]:
    try:
        value = response.choices[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        return None
    return str(value) if value is not None else None


def _exception_finish_reason(exc: Exception) -> Optional[str]:
    value = getattr(exc, "finish_reason", None)
    return str(value) if value is not None else None


@lru_cache(maxsize=None)
def _claude_compatible_response_model(response_model: type[BaseModel]) -> type[BaseModel]:
    """Preserve local Pydantic validation with a Claude-compatible wire schema."""

    def model_json_schema(cls: type[BaseModel], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return _claude_compatible_json_schema(response_model.model_json_schema(*args, **kwargs))

    return type(
        response_model.__name__,
        (response_model,),
        {
            "__module__": response_model.__module__,
            "model_json_schema": classmethod(model_json_schema),
        },
    )


@lru_cache(maxsize=None)


def response_model_for_llm(
    model: Optional[str], response_model: type[BaseModel]
) -> type[BaseModel]:
    """Return the original model except for Claude's narrower JSON Schema subset."""
    if is_claude_model(model):
        return _claude_compatible_response_model(response_model)
    return response_model


def is_claude_model(model: Optional[str]) -> bool:
    """Identify Claude model IDs, including provider-prefixed names."""
    model_name = (model or "").rsplit("/", 1)[-1].lower()
    return model_name.startswith("claude-")


def parse_structured(
    server: Any,
    *,
    model: str,
    response_model: type[BaseModel],
    **kwargs: Any,
) -> Any:
    """Parse a Pydantic response using the provider-specific schema path."""
    wire_model = response_model_for_llm(model, response_model)
    wire_schema = _openai_strict_json_schema(wire_model)
    budget_estimate = _check_structured_request_budget(
        server, model=model, response_model=response_model, schema=wire_schema,
        messages=kwargs.get("messages"), transport="openai_parse",
        requested_completion_tokens=(
            kwargs.get("max_completion_tokens")
            or kwargs.get("max_tokens")
            or getattr(server, "structured_output_token_limit", None)
        ),
    )
    request_index = _record_structured_request(
        model=model,
        response_model=response_model,
        transport="openai_parse",
        schema=wire_schema,
        messages=kwargs.get("messages"),
    )
    usage_before = _server_usage_snapshot(server)
    try:
        response = _call_with_structured_context(
            request_index,
            transport="openai_parse",
            response_model=response_model,
            request_budget_estimate=budget_estimate,
            call=lambda: server.parse(
                model=model,
                response_format=wire_model,
                **kwargs,
            ),
        )
    except Exception as exc:
        _record_structured_result(
            request_index,
            status="ERROR",
            finish_reason=_exception_finish_reason(exc),
            error_type=type(exc).__name__,
            error_category=_structured_error_category(exc),
            usage=_server_usage_delta(usage_before, _server_usage_snapshot(server)),
        )
        raise
    finish_reason = _response_finish_reason(response)
    _record_structured_result(
        request_index,
        status="SUCCESS",
        finish_reason=finish_reason,
        usage=_response_usage(response),
    )
    return response


def make_http_client(
    proxy: Optional[str], *, enable_audit: bool = False
) -> Optional[Any]:
    if (
        not proxy and not enable_audit and not os.environ.get("TRAIL_API_DIAGNOSTIC_PATH")
        and not request_budget_environment_configured()
    ):
        return None
    try:
        import httpx
    except ImportError:
        try:
            # Newer OpenAI SDK distributions may depend on the namespaced
            # HTTPX2 package instead of the traditional ``httpx`` import.
            import httpx2 as httpx
        except ImportError:
            return None
    kwargs: Dict[str, Any] = {}
    if proxy:
        kwargs["proxy"] = proxy
    return httpx.Client(**kwargs)


class LLMServer:
    """Shared OpenAI-compatible transport used by dataset diagnosers.

    Azure OpenAI and compatible endpoints share this transport interface.
    Dataset-specific services remain responsible for prompts and interpretation.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        proxy: Optional[str] = None,
        url: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        http_client: Optional[Any] = None,
        api_version: Optional[str] = None,
        max_retries: Optional[int] = None,
        structured_output_token_limit: Optional[int] = None,
        api_request_limit: Optional[int] = None,
        request_token_budget_config: Optional[Dict[str, Any]] = None,
        resolve_env: bool = True,
        validate_key_before_import: bool = False,
        missing_key_message: Optional[str] = None,
        missing_package_message: Optional[str] = None,
    ):
        self.model = model
        self.url = base_url or url
        self.proxy = proxy
        for name, value in (
            ("structured_output_token_limit", structured_output_token_limit),
            ("api_request_limit", api_request_limit),
        ):
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be a positive integer when set.")
        self.structured_output_token_limit = (
            int(structured_output_token_limit) if structured_output_token_limit is not None else None
        )
        self.api_request_limit = int(api_request_limit) if api_request_limit is not None else None
        self.request_token_budget_config = (
            dict(request_token_budget_config) if request_token_budget_config is not None else None
        )
        if self.request_token_budget_config is not None:
            budget = self.request_token_budget_config
            for key in ("context_limit", "hard_projected_limit", "completion_reserve"):
                if int(budget.get(key, 0)) < 1:
                    raise ValueError(f"request_token_budget_config.{key} must be positive.")
            factor = float(budget.get("safety_factor", 0))
            if factor < 1:
                raise ValueError("request_token_budget_config.safety_factor must be at least 1.")
            if not str(budget.get("tokenizer", "")).strip():
                raise ValueError("request_token_budget_config.tokenizer must be set.")
        self._api_attempt_lock = threading.Lock()
        self._api_attempt_count = 0
        parsed_endpoint = urlsplit(str(self.url or ""))
        self.endpoint_hostname = (parsed_endpoint.hostname or "").lower() or None
        self.provider = (
            "azure_openai"
            if self.endpoint_hostname and (
                self.endpoint_hostname.endswith(".openai.azure.com")
                or self.endpoint_hostname.endswith(".services.ai.azure.com")
            )
            else "openai_compatible"
        )
        self.api_version = (
            api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or _AZURE_OPENAI_API_VERSION
        ) if self.provider == "azure_openai" else None
        self.azure_deployment = str(model) if self.provider == "azure_openai" and model else None
        self.api_key = api_key
        if max_retries is not None and int(max_retries) < 0:
            raise ValueError("max_retries must be non-negative")
        self.request_retry_limit = int(max_retries) if max_retries is not None else None
        self._usage_lock = threading.Lock()
        self._usage = {
            "api_attempts": 0,
            "request_count": 0,
            "failed_request_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reported_usage_count": 0,
            "request_seconds": 0.0,
            "schema_validation_failure_count": 0,
            "incomplete_output_count": 0,
            "max_token_retry_count": 0,
            "refusal_count": 0,
        }
        if client is not None:
            self.client = client
            self.sdk_max_retries = getattr(client, "max_retries", None)
            self.sdk_timeout = str(getattr(client, "timeout", "SDK default"))
            underlying_http_client = getattr(client, "_client", None)
            _install_http_audit_hooks(underlying_http_client)
            if (
                os.environ.get("TRAIL_REQUEST_AUDIT_PATH")
                or os.environ.get("TRAIL_API_DIAGNOSTIC_PATH")
                or request_budget_environment_configured()
            ) and underlying_http_client is None:
                raise RuntimeError(
                    "Request auditing or budgeting was enabled, but the injected OpenAI client "
                    "does not expose its HTTP transport."
                )
            return

        resolved_api_key = api_key or (os.environ.get("OPENAI_API_KEY") if resolve_env else None)
        if validate_key_before_import and not resolved_api_key:
            raise RuntimeError(
                missing_key_message
                or "API key is missing. Set it in the API config JSON or pass --api-config."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            message = missing_package_message or "The openai package is not installed."
            raise RuntimeError(message) from exc

        if not resolved_api_key:
            raise RuntimeError(
                missing_key_message
                or "API key is missing. Set it in the API config JSON or pass --api-config."
            )

        self.api_key = resolved_api_key

        active_http_client = http_client
        if active_http_client is None:
            active_http_client = make_http_client(
                proxy,
                enable_audit=bool(
                    os.environ.get("TRAIL_REQUEST_AUDIT_PATH")
                    or os.environ.get("TRAIL_API_DIAGNOSTIC_PATH")
                    or request_budget_environment_configured()
                ),
            )
        if request_budget_environment_configured() and active_http_client is None:
            raise RuntimeError(
                "A request budget was configured, but an HTTP event-hook client could not be created; refusing uncounted API calls."
            )
        _install_http_audit_hooks(active_http_client)
        client_kwargs = {
            "api_key": resolved_api_key,
            "http_client": active_http_client,
        }
        if max_retries is not None:
            client_kwargs["max_retries"] = int(max_retries)
        if self.url:
            host = (urlparse(self.url).hostname or "").lower()
        else:
            host = ""

        if self.provider == "azure_openai":
            try:
                from openai import AzureOpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "This Azure OpenAI endpoint requires an OpenAI SDK with AzureOpenAI support."
                ) from exc
            client_kwargs["azure_endpoint"] = self.url.rstrip("/")
            client_kwargs["api_version"] = self.api_version
            self.client = AzureOpenAI(**client_kwargs)
        else:
            if self.url:
                client_kwargs["base_url"] = self.url
            self.client = OpenAI(**client_kwargs)
        self.sdk_max_retries = getattr(self.client, "max_retries", None)
        self.sdk_timeout = str(getattr(self.client, "timeout", "SDK default"))

    @staticmethod
    def _field(value: Any, name: str) -> Any:
        if isinstance(value, dict):
            return value.get(name)
        return getattr(value, name, None)

    @staticmethod
    def _token_count(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def usage_snapshot(self) -> Dict[str, Any]:
        with self._usage_lock:
            return dict(self._usage)

    def _reserve_api_attempt(self) -> None:
        """Count one outgoing API attempt and fail closed at the configured cap."""
        with self._api_attempt_lock:
            if self.api_request_limit is not None and self._api_attempt_count >= self.api_request_limit:
                error = RuntimeError(
                    f"LLM API request budget exhausted ({self._api_attempt_count}/{self.api_request_limit})."
                )
                error.stop_run = True
                raise error
            self._api_attempt_count += 1
        self._record_counter("api_attempts")

    def _record_response(self, response: Any, elapsed: float, failed: bool = False) -> None:
        usage = self._field(response, "usage") if response is not None else None
        prompt = self._token_count(
            self._field(usage, "prompt_tokens")
            if self._field(usage, "prompt_tokens") is not None
            else self._field(usage, "input_tokens")
        )
        completion = self._token_count(
            self._field(usage, "completion_tokens")
            if self._field(usage, "completion_tokens") is not None
            else self._field(usage, "output_tokens")
        )
        total = self._token_count(self._field(usage, "total_tokens"))
        reported = any(value is not None for value in (prompt, completion, total))
        if total is None and (prompt is not None or completion is not None):
            total = (prompt or 0) + (completion or 0)

        with self._usage_lock:
            self._usage["request_count"] += 1
            self._usage["failed_request_count"] += int(failed)
            self._usage["request_seconds"] += elapsed
            self._usage["reported_usage_count"] += int(reported)
            if prompt is not None:
                self._usage["prompt_tokens"] += prompt
            if completion is not None:
                self._usage["completion_tokens"] += completion
            if total is not None:
                self._usage["total_tokens"] += total

    def _record_api_diagnostic(
        self,
        context: Dict[str, Any],
        *,
        response: Any = None,
        exception: Optional[BaseException] = None,
        elapsed: float,
    ) -> None:
        path = os.environ.get("TRAIL_API_DIAGNOSTIC_PATH")
        if not path:
            return
        attempts = context.get("http_attempts", [])
        last = attempts[-1] if attempts else {}
        status = last.get("http_status_code") or getattr(exception, "status_code", None)
        error_code = last.get("azure_error_code") or getattr(exception, "code", None)
        finish_reason = last.get("finish_reason")
        if response is not None:
            finish_reason = _response_finish_reason(response) or finish_reason
        usage = _response_usage(response) if response is not None else {
            "input_tokens": None, "output_tokens": None, "total_tokens": None,
        }
        if exception is not None:
            # Keep diagnostic artifacts useful without copying provider text or
            # response excerpts, which may contain private prompt content.
            message = None
            category = _diagnostic_error_category(exception, status, error_code)
            exception_chain = []
            current: Optional[BaseException] = exception
            seen: set[int] = set()
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                exception_chain.append({"exception_class": type(current).__name__})
                current = current.__cause__ or current.__context__
        else:
            message = None
            category = None
            exception_chain = []
        record = {
            "record_type": "api_request",
            "timestamp": context.get("timestamp"),
            "configuration_file_path": os.environ.get("TRAIL_API_CONFIG_PATH"),
            "provider": self.provider,
            "effective_deployment": context.get("model"),
            "endpoint_hostname": self.endpoint_hostname,
            "api_version": self.api_version,
            "experiment": os.environ.get("TRAIL_EXPERIMENT_LABEL"),
            "run_attempt": os.environ.get("TRAIL_RUN_ATTEMPT", "1"),
            "proxy_enabled": bool(self.proxy),
            "request_type": context.get("request_type"),
            "request_stage": context.get("request_stage"),
            "response_schema": context.get("response_schema"),
            "max_completion_tokens": context.get("max_completion_tokens"),
            "max_tokens": context.get("max_tokens"),
            "trace_id": context.get("trace_id"),
            "http_attempt_count": len(attempts) if attempts else None,
            "retry_count": max(0, len(attempts) - 1) if attempts else None,
            "http_status_code": status,
            "exception_class": type(exception).__name__ if exception is not None else None,
            "azure_error_code": str(error_code) if error_code else None,
            "error_category": category,
            "error_message": _diagnostic_redact(message),
            "exception_chain": exception_chain,
            "duration_seconds": round(elapsed, 4),
            "finish_reason": finish_reason,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "sdk_max_retries": self.sdk_max_retries,
            "sdk_timeout": self.sdk_timeout,
            "http_attempts": [
                {
                    key: value for key, value in attempt.items()
                    if key not in {"started_monotonic", "error_message"}
                }
                for attempt in attempts
            ],
        }
        _append_jsonl(path, record)

    def create(self, **kwargs: Any) -> Any:
        """Forward a normal completion request and record provider usage."""
        if self.request_token_budget_config is not None and self.structured_output_token_limit is not None:
            kwargs.setdefault("max_completion_tokens", self.structured_output_token_limit)
        started = time.perf_counter()
        diagnostic_path = os.environ.get("TRAIL_API_DIAGNOSTIC_PATH")
        context: Optional[Dict[str, Any]] = None
        token = None
        if diagnostic_path:
            context = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": kwargs.get("model", self.model),
                "request_type": os.environ.get("TRAIL_API_REQUEST_TYPE", "chat_completion"),
                "request_stage": os.environ.get("TRAIL_API_REQUEST_STAGE"),
                "response_schema": os.environ.get("TRAIL_API_REQUEST_SCHEMA"),
                "max_completion_tokens": kwargs.get("max_completion_tokens"),
                "max_tokens": kwargs.get("max_tokens"),
                "trace_id": _CURRENT_TRACE_ID.get(),
                "api_key": self.api_key,
                "http_attempts": [],
            }
            token = _ACTIVE_API_DIAGNOSTIC.set(context)
        try:
            self._reserve_api_attempt()
            with PersistentRequestBudget.from_environment().transport_call(count_at_http_event=True):
                response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if not getattr(exc, "stop_run", False):
                self._record_response(None, time.perf_counter() - started, failed=True)
                _settle_request_cost_budget(
                    self, structured_context=_ACTIVE_STRUCTURED_REQUEST.get(),
                    response=getattr(exc, "completion", None),
                )
            if context is not None and not getattr(exc, "stop_run", False):
                self._record_api_diagnostic(
                    context, exception=exc, elapsed=time.perf_counter() - started,
                )
            raise
        else:
            elapsed = time.perf_counter() - started
            self._record_response(response, elapsed)
            _settle_request_cost_budget(
                self, structured_context=_ACTIVE_STRUCTURED_REQUEST.get(), response=response,
            )
            if context is not None:
                self._record_api_diagnostic(context, response=response, elapsed=elapsed)
            return response
        finally:
            if token is not None:
                _ACTIVE_API_DIAGNOSTIC.reset(token)

    def parse(self, **kwargs: Any) -> Any:
        """Forward a structured completion request and record provider usage."""
        if self.structured_output_token_limit is not None:
            kwargs.setdefault("max_completion_tokens", self.structured_output_token_limit)
        started = time.perf_counter()
        diagnostic_path = os.environ.get("TRAIL_API_DIAGNOSTIC_PATH")
        structured_context = _ACTIVE_STRUCTURED_REQUEST.get() or {}
        context: Optional[Dict[str, Any]] = None
        token = None
        if diagnostic_path:
            context = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": kwargs.get("model", self.model),
                "request_type": os.environ.get("TRAIL_API_REQUEST_TYPE", "structured_parse"),
                "request_stage": structured_context.get("response_model"),
                "response_schema": structured_context.get("response_model"),
                "max_completion_tokens": kwargs.get("max_completion_tokens"),
                "trace_id": _CURRENT_TRACE_ID.get(),
                "http_attempts": [],
            }
            token = _ACTIVE_API_DIAGNOSTIC.set(context)
        reserved = False
        try:
            self._reserve_api_attempt()
            reserved = True
            with PersistentRequestBudget.from_environment().transport_call(count_at_http_event=True):
                response = self.client.beta.chat.completions.parse(**kwargs)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            # The OpenAI SDK raises LengthFinishReasonError after receiving a
            # response. Its completion carries the usage and finish_reason that
            # were previously lost by this exception path.
            response = getattr(exc, "completion", None)
            if reserved:
                self._record_response(response, elapsed, failed=True)
                _settle_request_cost_budget(
                    self, structured_context=structured_context, response=response,
                )
            exception_name = type(exc).__name__.lower()
            if reserved and (isinstance(exc, ValidationError) or "parse" in exception_name or "json" in exception_name):
                self._record_counter("schema_validation_failure_count")
            if reserved and ("length" in exception_name or "incomplete" in exception_name):
                self._record_counter("incomplete_output_count")
            if context is not None and reserved:
                self._record_api_diagnostic(context, response=response, exception=exc, elapsed=elapsed)
            raise
        finally:
            if token is not None:
                _ACTIVE_API_DIAGNOSTIC.reset(token)
        finish_reason = None
        try:
            finish_reason = response.choices[0].finish_reason
        except (AttributeError, IndexError, TypeError):
            pass
        if finish_reason in {"length", "max_tokens"}:
            self._record_counter("incomplete_output_count")
        self._record_response(response, time.perf_counter() - started)
        _settle_request_cost_budget(
            self, structured_context=structured_context, response=response,
        )
        if context is not None:
            self._record_api_diagnostic(context, response=response, elapsed=time.perf_counter() - started)
        return response

    def _record_counter(self, name: str, amount: int = 1) -> None:
        with self._usage_lock:
            self._usage[name] = self._usage.get(name, 0) + amount



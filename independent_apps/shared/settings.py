"""The single runtime settings and API-config loader for independent apps.

This module contains configuration data and common credential resolution only.
Dataset validation and execution remain in each owning project.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


llm_model = "DeepSeek-V4.1-Flash"
llm_url = ""
proxy = None
api_keys = ""

cascade_enabled = True
cascade_call_mode = "combined"
separate_order = "cascade_then_local"
prompt_context_policy = "legacy"
max_prompt_upstream_chars = 6000
max_causes = 8
max_stage2_candidates = 20
enable_scoring = False
use_llm_error_synthesis = False

max_current_node_string_length = 1200
max_upstream_node_string_length = 450
max_downstream_node_string_length = 350


@dataclass
class RuntimeSettings:
    llm_model: str = llm_model
    llm_url: Optional[str] = llm_url
    proxy: Optional[str] = proxy
    api_key: str = api_keys
    cascade_enabled: bool = cascade_enabled
    cascade_call_mode: str = cascade_call_mode
    separate_order: str = separate_order
    prompt_context_policy: str = prompt_context_policy
    max_prompt_upstream_chars: int = max_prompt_upstream_chars
    max_causes: int = max_causes
    max_stage2_candidates: int = max_stage2_candidates
    enable_scoring: bool = enable_scoring
    use_llm_error_synthesis: bool = use_llm_error_synthesis
    max_current_node_string_length: int = max_current_node_string_length
    max_upstream_node_string_length: int = max_upstream_node_string_length
    max_downstream_node_string_length: int = max_downstream_node_string_length
    include_ground_truth: bool = True
    include_ground_truth_in_node_prompt: bool = False
    include_full_analysis: bool = False
    include_eval_labels: bool = False
    traversal_mode: str = "auto"
    edge_types: Optional[List[str]] = None
    # Trail-prefixed values are consumed only by the Trail project.
    trail_cause_state_mode: str = "H0"
    trail_schedule_scope: str = "legacy"
    trail_final_hypothesis_policy: str = "gated"
    trail_max_completion_tokens: int = 32768
    trail_api_request_limit: int = None
    trail_request_token_budget_enabled: bool = False
    trail_request_context_limit: int = 272000
    trail_request_hard_projected_limit: int = 231200
    trail_request_completion_reserve: int = 32768
    trail_request_safety_factor: float = 1.10
    trail_request_tokenizer: str = "o200k_base"
    # Dataset-only settings which do not belong to the common interface live here.
    dataset_options: Dict[str, Any] = field(default_factory=dict)


def load_api_config(path: str) -> Dict[str, Optional[str]]:
    config_path = Path(path).expanduser()
    try:
        with config_path.open("r", encoding="utf-8-sig") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read API config JSON at {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"API config at {config_path} must contain a JSON object.")

    api_key = config.get("api_key")
    base_url = config.get("base_url")
    model = config.get("model")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(f"API config at {config_path} must contain a non-empty 'api_key'.")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError(f"API config at {config_path} must contain a non-empty 'base_url'.")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError(f"API config at {config_path} has an invalid 'model'.")

    return {
        "api_key": api_key.strip(),
        "llm_url": base_url.strip(),
        "llm_model": model.strip() if model is not None else None,
    }


def settings_from_args(args: Any, api_config: Optional[Dict[str, Optional[str]]] = None) -> RuntimeSettings:
    """Resolve common runtime values without importing any dataset package."""
    settings = RuntimeSettings()
    if getattr(args, "llm_model", None) is not None:
        settings.llm_model = args.llm_model
    if api_config is not None:
        settings.llm_url = api_config["llm_url"]
        settings.api_key = api_config["api_key"]
        if getattr(args, "llm_model", None) is None and api_config.get("llm_model"):
            settings.llm_model = api_config["llm_model"]
    else:
        settings.llm_url = os.environ.get("OPENAI_BASE_URL") or settings.llm_url
        settings.api_key = os.environ.get("OPENAI_API_KEY") or settings.api_key

    requested_proxy = getattr(args, "proxy", None)
    if requested_proxy is not None:
        settings.proxy = requested_proxy.strip() or None
    elif "OPENAI_PROXY" in os.environ:
        settings.proxy = os.environ["OPENAI_PROXY"].strip() or None

    for argument, setting in (
        ("max_causes", "max_causes"),
        ("max_stage2_candidates", "max_stage2_candidates"),
        ("traversal_mode", "traversal_mode"),
        ("edge_types", "edge_types"),
        ("cascade_call_mode", "cascade_call_mode"),
        ("separate_order", "separate_order"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            setattr(settings, setting, value)

    settings.cascade_enabled = not getattr(args, "no_cascade", False)
    context_policy = getattr(args, "prompt_context_policy", None)
    if context_policy is None:
        context_policy = os.environ.get("TRACE_DIAGNOSIS_PROMPT_CONTEXT_POLICY")
    if context_policy:
        settings.prompt_context_policy = context_policy.strip().lower()

    upstream_chars = getattr(args, "max_prompt_upstream_chars", None)
    if upstream_chars is not None:
        if upstream_chars < 0:
            raise ValueError("--max-prompt-upstream-chars must be non-negative.")
        settings.max_prompt_upstream_chars = upstream_chars

    settings.enable_scoring = getattr(args, "enable_scoring", False)
    settings.use_llm_error_synthesis = getattr(args, "use_llm_error_synthesis", False)
    settings.include_ground_truth = not getattr(args, "no_ground_truth", False)
    settings.include_full_analysis = getattr(args, "include_full_analysis", False)
    settings.include_eval_labels = getattr(args, "include_eval_labels", False)
    return settings

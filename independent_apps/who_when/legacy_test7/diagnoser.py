import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Callable, Tuple, Literal

try:
    import httpx
except ImportError:
    httpx = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# =====================================================================
# 1. Edge priority
# =====================================================================
EDGE_PRIORITY = {
    "execution_flow": 3,
    "sequential_flow": 2,
    "data_flow": 1,
}


def clean_label(x: Any) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    if x.lower() in {"", "none", "null", "<missing>", "unknown"}:
        return ""
    return x


def canonical_agent_id(node: Dict[str, Any]) -> str:
    agent_id = clean_label(node.get("agent_id"))
    if agent_id:
        return agent_id

    name = clean_label(node.get("name"))
    if name:
        return name

    role = clean_label(node.get("role"))
    role_l = role.lower()
    if role_l.startswith("orchestrator"):
        return "Orchestrator"
    if "websurfer" in role_l or "web surfer" in role_l:
        return "WebSurfer"
    if "filesurfer" in role_l or "file surfer" in role_l:
        return "FileSurfer"
    if "computerterminal" in role_l or "computer_terminal" in role_l or "computer terminal" in role_l:
        return "ComputerTerminal"
    if "assistant" in role_l:
        return "Assistant"
    if role:
        return role
    return "Unknown"


def is_tool_like_agent(agent_id: str) -> bool:
    agent = clean_label(agent_id).lower().replace(" ", "_")
    return agent in {
        "computer_terminal",
        "computerterminal",
        "terminal",
        "python",
        "python_tool",
        "browser",
        "tool",
    }


def is_tool_like_node(node: Dict[str, Any]) -> bool:
    if bool(node.get("is_tool_executor")):
        return True

    agent = canonical_agent_id(node)
    role = clean_label(node.get("role")).lower()
    name = clean_label(node.get("name")).lower()
    return (
        is_tool_like_agent(agent)
        or "computer_terminal" in name
        or "terminal" in name
        or role in {"tool", "observation"}
    )


def is_terminal_like_node(node: Dict[str, Any]) -> bool:
    if bool(node.get("is_terminal_node") or node.get("is_final_node")):
        return True

    text = node_text(node).strip().lower()
    semantic_role = clean_label(node.get("semantic_role")).lower()
    role = clean_label(node.get("role")).lower()
    return (
        semantic_role in {"final", "termination", "output"}
        or "termination" in role
        or text in {"terminate", "terminated"}
        or text.startswith("terminate")
    )


def is_user_like_node(node: Dict[str, Any]) -> bool:
    role = clean_label(node.get("role")).lower()
    agent = canonical_agent_id(node).lower()
    return role in {"human", "user"} or agent == "user"


def is_preliminary_or_delegation_like_node(node: Dict[str, Any]) -> bool:
    role = clean_label(node.get("role")).lower()
    agent = canonical_agent_id(node).lower()
    text = node_text(node).strip().lower()

    if "orchestrator" not in role and "orchestrator" not in agent:
        return False

    delegation_markers = {
        "->",
        "delegate",
        "delegat",
        "ask websurfer",
        "ask file",
        "we need",
        "need to",
        "plan",
        "strategy",
        "i will",
        "let's",
        "next step",
    }
    return "thought" in role or any(marker in text for marker in delegation_markers)


def normalize_error_role(value: Any) -> str:
    role = str(value or "").strip().upper()
    allowed = {
        "INTRODUCES_ERROR",
        "PROPAGATES_ERROR",
        "VERIFIES_PRIOR_ERROR",
        "TERMINATES_AFTER_ERROR",
        "NO_ERROR",
        "UNCERTAIN",
    }
    return role if role in allowed else "UNCERTAIN"


def node_text(node: Dict[str, Any]) -> str:
    raw = node.get("metadata")
    if isinstance(raw, (dict, list)):
        return json.dumps(raw, ensure_ascii=False, default=str)
    return str(raw or "")


def truncate_for_prompt(value: Any, max_chars: int = 1200) -> str:
    text = (
        json.dumps(value, ensure_ascii=False, default=str)
        if isinstance(value, (dict, list))
        else str(value or "")
    )
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[TRUNCATED]...\n" + text[-half:]


def extract_json_object_loose(text: str) -> Dict[str, Any]:
    """Parse a JSON object from an LLM response; return {} instead of raising."""
    raw = (text or "").strip()
    if not raw:
        return {}

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    candidates = [raw]
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    start = raw.find("{")
    end = raw.rfind("}")
    if 0 <= start < end:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            continue
    return {}


def loose_enum_field(text: str, field: str, allowed: Set[str], default: str) -> str:
    pattern = rf'"?{re.escape(field)}"?\s*:\s*"?([A-Za-z_]+)'
    match = re.search(pattern, text or "", re.IGNORECASE)
    if not match:
        return default
    value = match.group(1).strip().upper()
    return value if value in allowed else default


def loose_bool_field(text: str, field: str, default: bool = False) -> bool:
    pattern = rf'"?{re.escape(field)}"?\s*:\s*(true|false|yes|no|1|0)'
    match = re.search(pattern, text or "", re.IGNORECASE)
    if not match:
        return default
    return match.group(1).strip().lower() in {"true", "yes", "1"}


def loose_float_field(text: str, field: str, default: float = 0.5) -> float:
    pattern = rf'"?{re.escape(field)}"?\s*:\s*(-?\d+(?:\.\d+)?)'
    match = re.search(pattern, text or "")
    if not match:
        return default
    try:
        return max(0.0, min(1.0, float(match.group(1))))
    except ValueError:
        return default


def loose_string_field(text: str, field: str) -> Optional[str]:
    pattern = rf'"?{re.escape(field)}"?\s*:\s*(null|"([^"]*)"|([^,\n\r}}]+))'
    match = re.search(pattern, text or "", re.IGNORECASE)
    if not match:
        return None
    value = match.group(2) if match.group(2) is not None else match.group(3)
    value = clean_label(value)
    if not value or value.lower() in {"null", "none"}:
        return None
    return value


# =====================================================================
# 2. Cause state
# =====================================================================
SupportStatus = Literal["HYPOTHESIS", "SUPPORTED", "REFUTED"]
CausalStatus = Literal[
    "UNKNOWN",
    "PROXIMATE_CAUSE",
    "ROOT_CANDIDATE",
    "ROOT_CAUSE",
]


@dataclass
class FailureCause:
    cause_id: str
    description: str
    confidence: float = 0.5
    support_status: SupportStatus = "HYPOTHESIS"
    causal_status: CausalStatus = "UNKNOWN"
    should_propagate_upstream: bool = True
    localized_node_id: Optional[str] = None
    root_node_id: Optional[str] = None
    evidence: Optional[str] = None

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


def _support_rank(status: str) -> int:
    return {
        "REFUTED": -1,
        "HYPOTHESIS": 0,
        "SUPPORTED": 1,
    }.get(status, 0)


def _causal_rank(status: str) -> int:
    return {
        "UNKNOWN": 0,
        "PROXIMATE_CAUSE": 1,
        "ROOT_CANDIDATE": 2,
        "ROOT_CAUSE": 3,
    }.get(status, 0)


def _normalize_cause_state(cause: FailureCause) -> FailureCause:
    c = FailureCause(**cause.model_dump())

    if c.support_status == "REFUTED":
        c.should_propagate_upstream = False
        return c

    if c.causal_status == "ROOT_CANDIDATE":
        c.support_status = "SUPPORTED"
        c.should_propagate_upstream = True

    elif c.causal_status == "ROOT_CAUSE":
        c.support_status = "SUPPORTED"
        c.should_propagate_upstream = False

    elif c.causal_status == "PROXIMATE_CAUSE":
        c.support_status = "SUPPORTED"
        c.should_propagate_upstream = True

    return c


def _merge_cause_state(old: FailureCause, new: FailureCause) -> FailureCause:
    merged = FailureCause(**old.model_dump())

    if _support_rank(new.support_status) >= _support_rank(old.support_status):
        merged.support_status = new.support_status

    if _causal_rank(new.causal_status) >= _causal_rank(old.causal_status):
        merged.causal_status = new.causal_status

    merged.confidence = max(old.confidence, new.confidence)

    if merged.localized_node_id is None and new.localized_node_id is not None:
        merged.localized_node_id = new.localized_node_id

    if new.root_node_id is not None:
        merged.root_node_id = new.root_node_id

    if not merged.evidence and new.evidence:
        merged.evidence = new.evidence

    if not merged.description and new.description:
        merged.description = new.description

    merged.should_propagate_upstream = new.should_propagate_upstream
    return _normalize_cause_state(merged)


def merge_causes(
    existing: List[FailureCause],
    incoming: List[FailureCause],
) -> List[FailureCause]:
    by_id: Dict[str, FailureCause] = {}
    order: List[str] = []

    for cause in existing + incoming:
        c = _normalize_cause_state(cause)
        if c.cause_id not in by_id:
            by_id[c.cause_id] = c
            order.append(c.cause_id)
        else:
            by_id[c.cause_id] = _merge_cause_state(by_id[c.cause_id], c)

    return [by_id[cid] for cid in order]


def clone_causes(causes: List[FailureCause]) -> List[FailureCause]:
    return [FailureCause(**cause.model_dump()) for cause in causes]


def ground_truth_available(ground_truth: Any) -> bool:
    if ground_truth is None:
        return False
    if isinstance(ground_truth, str):
        return ground_truth.strip() not in {"", "N/A", "None", "null"}
    return True



class _LLMServerCompletionsShim:
    """Expose the OpenAI chat.completions.create shape while using project LLMServer."""
    def __init__(self, llm_server: Any):
        self._llm_server = llm_server

    def create(self, **kwargs: Any) -> Any:
        return self._llm_server.create(**kwargs)


class _LLMServerChatShim:
    def __init__(self, llm_server: Any):
        self.completions = _LLMServerCompletionsShim(llm_server)


class _LLMServerClientShim:
    def __init__(self, llm_server: Any):
        self.chat = _LLMServerChatShim(llm_server)


# =====================================================================
# 3. Unified LLM Service
# =====================================================================
class LLMService:

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        api_key: Optional[str] = None,
        http_client: Optional[Any] = None,
        client: Optional[Any] = None,
        llm_server: Optional[Any] = None,
    ):
        self.model_name = model_name
        if llm_server is not None:
            # All prompts and response parsing below remain from Test-7.
            # Only the HTTP transport is routed through the project's audited client.
            self.client = _LLMServerClientShim(llm_server)
        elif client is not None:
            self.client = client
        else:
            resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not resolved_api_key:
                raise RuntimeError("OpenAI API Key is missing. Pass api_key or set OPENAI_API_KEY env.")
            if OpenAI is None:
                raise RuntimeError("The openai package is not installed.")
            self.client = OpenAI(api_key=resolved_api_key, http_client=http_client)

    def initialize_active_final_causes(
        self,
        query: Any,
        final_node: Dict[str, Any],
        ground_truth: Any,
    ) -> Tuple[List[FailureCause], Dict[str, Any]]:
        final_step = final_node.get("step_index", "N/A")
        final_content = final_node.get("metadata")
        final_content_text = (
            json.dumps(final_content, ensure_ascii=False, default=str)
            if isinstance(final_content, (dict, list))
            else str(final_content)
        )

        gt_available = ground_truth_available(ground_truth)
        gt_text = json.dumps(ground_truth, ensure_ascii=False, default=str)

        prompt = (
            "You are initializing a downstream failure hypothesis for a who&when "
            "agent-trace diagnosis task. Each trace contains one annotated final-answer "
            "error. We will later traverse the trace backward to find where the failure "
            "was introduced or propagated.\n\n"
            "Compare the USER QUERY with the FINAL NODE RESULT. If GROUND TRUTH is "
            "available, compare the final result against it as the correct target.\n\n"
            "Return JSON only:\n"
            "{\n"
            '  "hypothesis": "one concise failure hypothesis",\n'
            '  "assessment": "MISMATCH|POTENTIAL_MISMATCH|UNKNOWN",\n'
            '  "confidence": 0.0_to_1.0,\n'
            '  "evidence": "brief evidence from the query/final result/ground truth",\n'
            '  "ground_truth_used": true_or_false\n'
            "}\n\n"
            f"USER QUERY:\n{query}\n\n"
            f"FINAL STEP: {final_step}\n"
            f"FINAL NODE RESULT:\n{final_content_text}\n\n"
            f"GROUND TRUTH AVAILABLE: {gt_available}\n"
            f"GROUND TRUTH:\n{gt_text}\n"
        )

        system_prompt = "You are a careful assistant analyzing final-answer discrepancies in multi-agent traces."

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            reply_text = (response.choices[0].message.content or "").strip()

            parsed = extract_json_object_loose(reply_text)

            hypothesis = str(parsed.get("hypothesis", "Potential final-answer discrepancy requiring backward diagnosis."))
            assessment = str(parsed.get("assessment", "UNKNOWN")).upper()
            if assessment not in {"MISMATCH", "POTENTIAL_MISMATCH", "UNKNOWN"}:
                assessment = "UNKNOWN"

            try:
                confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
            except (TypeError, ValueError):
                confidence = 0.5

            evidence = parsed.get("evidence")
            if evidence in {None, "", "null"}:
                evidence = None

            analysis = {
                "hypothesis": hypothesis,
                "assessment": assessment,
                "confidence": confidence,
                "evidence": evidence,
                "ground_truth_used": bool(parsed.get("ground_truth_used", gt_available)),
                "ground_truth_available": gt_available,
                "raw_response": reply_text,
            }

            cause = FailureCause(
                cause_id="C1",
                description=hypothesis,
                confidence=confidence,
                support_status="HYPOTHESIS",
                causal_status="UNKNOWN",
                should_propagate_upstream=True,
                evidence=evidence,
            )

            return [cause], analysis

        except Exception as e:
            print(f"[Initial Cause Evaluator Error] {e}")
            fallback_description = "Potential final-answer failure hypothesis derived from query and result."
            return [
                FailureCause(
                    cause_id="C1",
                    description=fallback_description,
                    confidence=0.2,
                    support_status="HYPOTHESIS",
                    causal_status="UNKNOWN",
                    should_propagate_upstream=True,
                )
            ], {
                "hypothesis": fallback_description,
                "assessment": "UNKNOWN",
                "confidence": 0.2,
                "evidence": None,
                "ground_truth_used": gt_available,
                "ground_truth_available": gt_available,
                "error": str(e),
            }

    def cascade_attribution_evaluator(
        self,
        current_node: Dict[str, Any],
        predecessors: List[Dict[str, Any]],
        local_eval: Dict[str, Any],
        active_causes: List[FailureCause],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        query = context.get("query", "N/A")
        ground_truth = context.get("ground_truth", None)
        idx = current_node.get("step_index", "N/A")

        raw_metadata = current_node.get("metadata")
        curr_content = (
            json.dumps(raw_metadata, ensure_ascii=False, default=str)
            if isinstance(raw_metadata, (dict, list))
            else str(raw_metadata)
        )

        formatted_predecessors = []
        for pred in predecessors:
            pred_idx = pred.get("step_index", "?")
            pred_agent = canonical_agent_id(pred)
            pred_role = clean_label(pred.get("role")) or "unknown"
            pred_content = (
                json.dumps(pred.get("metadata"), ensure_ascii=False, default=str)
                if isinstance(pred.get("metadata"), (dict, list))
                else str(pred.get("metadata"))
            )
            formatted_predecessors.append(
                f"[Step {pred_idx} | {pred_agent} (role: {pred_role})]: {pred_content}"
            )

        predecessor_context = "\n".join(formatted_predecessors) or "None (Initial Step)"
        active_payload = [cause.model_dump() for cause in active_causes]

        prompt = (
            "You are performing a supplementary cascade-attribution analysis for a who&when agent trace.\n\n"
            "Possible relations:\n"
            "- INTRODUCES | PROPAGATES | AMPLIFIES | RECOVERS | UNRELATED | REFUTES | UNCERTAIN\n\n"
            "OUTPUT SCHEMA:\n"
            "{\n"
            '  "relation": "INTRODUCES|PROPAGATES|AMPLIFIES|RECOVERS|UNRELATED|REFUTES|UNCERTAIN",\n'
            '  "explanation": "brief evidence-based explanation",\n'
            '  "is_root_candidate": true_or_false,\n'
            '  "confidence": 0.0_to_1.0,\n'
            '  "evidence": "exact substring or null"\n'
            "}\n\n"
            f"USER QUERY:\n{query}\n\n"
            f"GROUND TRUTH:\n{json.dumps(ground_truth, ensure_ascii=False, default=str)}\n\n"
            f"ACTIVE CAUSES:\n{json.dumps(active_payload, ensure_ascii=False, default=str)}\n\n"
            f"LOCAL DIAGNOSIS:\n{json.dumps(local_eval, ensure_ascii=False, default=str)}\n\n"
            f"DIRECT PREDECESSOR CONTEXT:\n{predecessor_context}\n\n"
            f"CURRENT STEP ({idx}) CONTENT:\n{curr_content}\n"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a careful assistant analyzing causal dependencies."},
                    {"role": "user", "content": prompt},
                ],
            )
            reply_text = (response.choices[0].message.content or "").strip()

            parsed = extract_json_object_loose(reply_text)

            allowed_relations = {
                "INTRODUCES", "PROPAGATES", "AMPLIFIES", "RECOVERS", "UNRELATED", "REFUTES", "UNCERTAIN"
            }
            relation = str(
                parsed.get("relation")
                or loose_enum_field(reply_text, "relation", allowed_relations, "UNCERTAIN")
            ).upper()
            if relation not in {
                "INTRODUCES", "PROPAGATES", "AMPLIFIES", "RECOVERS", "UNRELATED", "REFUTES", "UNCERTAIN"
            }:
                relation = "UNCERTAIN"

            try:
                confidence = max(
                    0.0,
                    min(1.0, float(parsed.get("confidence", loose_float_field(reply_text, "confidence", 0.5)))),
                )
            except (TypeError, ValueError):
                confidence = 0.5

            evidence = parsed.get("evidence") or loose_string_field(reply_text, "evidence")
            if evidence in (None, "", "null"):
                evidence = None

            return {
                "relation": relation,
                "explanation": str(parsed.get("explanation", reply_text)),
                "is_root_candidate": bool(
                    parsed.get("is_root_candidate", loose_bool_field(reply_text, "is_root_candidate", False))
                ),
                "confidence": confidence,
                "evidence": evidence,
                "json_parse_recovered": not bool(parsed),
            }

        except Exception as e:
            print(f"[Cascade Evaluator Error] Failed at step {idx}: {e}")
            return {
                "relation": "UNCERTAIN",
                "explanation": f"Cascade attribution failed: {e}",
                "is_root_candidate": False,
                "confidence": 0.0,
                "evidence": None,
            }

    def select_root_from_candidates(
        self,
        query: Any,
        ground_truth: Any,
        final_node: Dict[str, Any],
        final_failure_analysis: Optional[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        final_payload = {
            "step_index": final_node.get("step_index"),
            "agent_id": canonical_agent_id(final_node),
            "content": truncate_for_prompt(final_node.get("metadata"), 900),
        }

        prompt = (
            "You are the second-stage root-cause selector for a who&when "
            "agent-trace diagnosis task.\n\n"
            "Stage 1 has already traversed the graph backward and generated a "
            "high-recall candidate list. The candidate list may contain false "
            "positives. Your job is NOT to find new candidates. Your job is to "
            "compare the given candidates and select the earliest decisive mistake "
            "that best explains the final failure.\n\n"
            "Selection policy:\n"
            "- Choose the first concrete mistake that made the later failure path likely.\n"
            "- Prefer concrete wrong search/click/navigation/extraction/method/assumption "
            "over generic planning or delegation.\n"
            "- A WebSurfer/FileSurfer step can be the root if it makes the bad search, "
            "click, page choice, file choice, extraction, or retrieval judgment.\n"
            "- An Orchestrator planning/delegation step is root only if it itself states "
            "the wrong target, wrong constraint, wrong source, wrong method, or wrong "
            "assumption. Do not select it merely because it assigned work.\n"
            "- A tool/terminal/user/final/termination step is root only if the candidate "
            "evidence shows that this step itself introduced the mistake.\n"
            "- If several candidates are plausible, choose the earliest one that is both "
            "concrete and causally sufficient, not simply the earliest candidate in time.\n\n"
            "Return JSON only:\n"
            "{\n"
            '  "selected_node_id": "node id from candidates or null",\n'
            '  "selected_step": integer_or_null,\n'
            '  "confidence": 0.0_to_1.0,\n'
            '  "rationale": "brief comparison-based reason",\n'
            '  "selection_mode": "LLM_CANDIDATE_RERANK"\n'
            "}\n\n"
            f"USER QUERY:\n{truncate_for_prompt(query, 1400)}\n\n"
            f"GROUND TRUTH:\n{truncate_for_prompt(ground_truth, 900)}\n\n"
            f"FINAL NODE:\n{json.dumps(final_payload, ensure_ascii=False, default=str)}\n\n"
            f"FINAL FAILURE ANALYSIS:\n{json.dumps(final_failure_analysis or {}, ensure_ascii=False, default=str)}\n\n"
            "CANDIDATES:\n"
            f"{json.dumps(candidates, ensure_ascii=False, default=str)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You compare root-cause candidates and return strict JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            reply_text = (response.choices[0].message.content or "").strip()

            parsed = extract_json_object_loose(reply_text)

            selected_node_id = parsed.get("selected_node_id") or loose_string_field(reply_text, "selected_node_id")
            if selected_node_id in {"", "null", "None"}:
                selected_node_id = None

            try:
                selected_step = parsed.get("selected_step") or loose_string_field(reply_text, "selected_step")
                selected_step = int(selected_step) if selected_step is not None else None
            except (TypeError, ValueError):
                selected_step = None

            try:
                confidence = max(
                    0.0,
                    min(1.0, float(parsed.get("confidence", loose_float_field(reply_text, "confidence", 0.5)))),
                )
            except (TypeError, ValueError):
                confidence = 0.5

            return {
                "selected_node_id": str(selected_node_id) if selected_node_id else None,
                "selected_step": selected_step,
                "confidence": confidence,
                "rationale": str(parsed.get("rationale", reply_text)),
                "selection_mode": "LLM_CANDIDATE_RERANK",
                "raw_response": reply_text,
                "json_parse_recovered": not bool(parsed),
            }

        except Exception as e:
            print(f"[Candidate Root Selector Error] {e}")
            return {
                "selected_node_id": None,
                "selected_step": None,
                "confidence": 0.0,
                "rationale": f"Candidate root selection failed: {e}",
                "selection_mode": "RANK_KEY_FALLBACK",
                "error": str(e),
            }

    def evaluate_local_step(
        self,
        current_node: Dict[str, Any],
        predecessors: List[Dict[str, Any]],
        context: Dict[str, Any],
        successors: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        problem = context.get("query", "N/A")
        ground_truth = context.get("ground_truth", "N/A")
        final_node = context.get("final_node")
        idx = current_node.get("step_index", "N/A")
        agent_role = current_node.get("role", "agent")
        agent_name = canonical_agent_id(current_node)

        formatted_predecessors = []
        for pred in predecessors:
            pred_idx = pred.get("step_index", "?")
            pred_agent = canonical_agent_id(pred)
            pred_role = clean_label(pred.get("role")) or "unknown"
            pred_desc = f"{pred_agent} (role: {pred_role})" if pred_agent != "unknown" else pred_role
            pred_content = truncate_for_prompt(pred.get("metadata"), 700)
            formatted_predecessors.append(f"[Step {pred_idx} | {pred_desc}]: {pred_content}")

        predecessor_context = "\n".join(formatted_predecessors) or "None (Initial Step)"

        formatted_successors = []
        for succ in (successors or [])[:3]:
            succ_idx = succ.get("step_index", "?")
            succ_agent = canonical_agent_id(succ)
            succ_role = clean_label(succ.get("role")) or "unknown"
            succ_desc = f"{succ_agent} (role: {succ_role})" if succ_agent != "unknown" else succ_role
            succ_content = truncate_for_prompt(succ.get("metadata"), 700)
            formatted_successors.append(f"[Step {succ_idx} | {succ_desc}]: {succ_content}")

        successor_context = "\n".join(formatted_successors) or "None"

        if isinstance(final_node, dict):
            final_step = final_node.get("step_index", "N/A")
            final_agent = canonical_agent_id(final_node)
            final_content = truncate_for_prompt(final_node.get("metadata"), 900)
            final_context = f"[Step {final_step} | {final_agent}]: {final_content}"
        else:
            final_context = "Not provided"

        raw_metadata = current_node.get("metadata")
        curr_content = truncate_for_prompt(raw_metadata, 1600)
        display_agent = clean_label(agent_name) or clean_label(agent_role) or "agent"
        display_role = clean_label(agent_role)
        agent_desc = f"{display_agent} (role: {display_role})" if display_role and display_role != display_agent else display_agent

        prompt = (
            f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation.\n"
            f"Problem: {problem}\n"
            f"Ground Truth: {ground_truth}\n\n"
            f"Final answer / terminal evidence:\n{final_context}\n\n"
            f"Prior steps:\n{predecessor_context}\n\n"
            f"This step ({idx}) by '{agent_desc}':\n{curr_content}\n\n"
            f"Immediate later evidence, for judging whether THIS step created an error later propagated:\n{successor_context}\n\n"
            "Determine whether this step contains a LOCAL mistake and whether it is the first decisive source of the failure.\n"
            "Important benchmark policy:\n"
            "- The target label is the mistaken agent step best supported by the trace, not necessarily the last step that exposes the wrong final answer.\n"
            "- INTRODUCES_ERROR: this step makes a concrete wrong decision, assumption, retrieval/navigation choice, code/method choice, data interpretation, or reasoning step that explains the final failure.\n"
            "- PROPAGATES_ERROR: this step repeats or uses an earlier mistake without introducing a new one.\n"
            "- VERIFIES_PRIOR_ERROR: this step merely confirms/endorses an earlier wrong result.\n"
            "- TERMINATES_AFTER_ERROR: this step only terminates after an earlier mistake.\n"
            "- NO_ERROR: this step is correct or only neutral bookkeeping.\n"
            "- UNCERTAIN: insufficient evidence.\n\n"
            "Tool/action step policy:\n"
            "- If a Computer_terminal/tool/action step merely executes code, query, or instructions produced by an earlier reasoning step, do NOT mark it as the root just because the output is wrong; use PROPAGATES_ERROR unless the tool/action step itself introduced the bug.\n"
            "- WebSurfer and FileSurfer are agents, not passive tools. A bad search, click, navigation, file choice, extraction, or retrieval judgment by them can INTRODUCE_ERROR.\n"
            "- If an expert/reasoning step writes faulty code, chooses the wrong method, makes an invalid assumption, or asks the tool to do the wrong thing, that expert/reasoning step is the root candidate.\n"
            "- If a later expert step adds a new concrete wrong assumption or wrong interpretation, it can INTRODUCE_ERROR even if earlier context was imperfect. Use PROPAGATES_ERROR only when it mechanically copies an earlier wrong result.\n"
            "- Do not mark a preliminary high-level step as INTRODUCES_ERROR only because later steps failed; require exact evidence in this step.\n"
            "- Do not mark an Orchestrator planning/delegation step as RootCandidate merely because it asked another agent to search, browse, or verify. Use RootCandidate: Yes only if this step itself names the wrong target, wrong constraint, wrong source, wrong method, or wrong assumption.\n"
            "- If this step only delegates work and a later WebSurfer/FileSurfer/Assistant step makes the concrete bad search, click, extraction, or answer choice, this step is usually NO_ERROR or PROPAGATES_ERROR, not INTRODUCES_ERROR.\n"
            "- RootCandidate must be No for NO_ERROR, PROPAGATES_ERROR, VERIFIES_PRIOR_ERROR, TERMINATES_AFTER_ERROR, and UNCERTAIN.\n\n"
            "Use later evidence only to decide whether THIS step's claim, action, method, or assumption was already wrong. Do not blame this step for mistakes introduced only by a later step.\n\n"
            "Respond ONLY in format:\n"
            "1. Yes/No.\n"
            "2. ErrorRole: INTRODUCES_ERROR|PROPAGATES_ERROR|VERIFIES_PRIOR_ERROR|TERMINATES_AFTER_ERROR|NO_ERROR|UNCERTAIN\n"
            "3. RootCandidate: Yes/No\n"
            "4. Reason: [brief evidence-based explanation]"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
                    {"role": "user", "content": prompt},
                ],
            )
            reply_text = response.choices[0].message.content.strip()

            is_error = False
            reason = "No detailed reason extracted."

            yes_no_match = re.search(r"(?:1[\.\)]?\s*)?(Yes|No)", reply_text, re.IGNORECASE)
            if yes_no_match:
                is_error = yes_no_match.group(1).lower() == "yes"

            role_match = re.search(r"ErrorRole:\s*([A-Z_]+)", reply_text, re.IGNORECASE)
            error_role = normalize_error_role(role_match.group(1) if role_match else None)

            root_match = re.search(r"RootCandidate:\s*(Yes|No)", reply_text, re.IGNORECASE)
            is_root_candidate = bool(root_match and root_match.group(1).lower() == "yes")

            reason_match = re.search(r"(?:4|3|2)\.\s*Reason:\s*(.*)", reply_text, re.IGNORECASE | re.DOTALL)
            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                reason = reply_text

            if error_role == "NO_ERROR":
                is_error = False
            elif error_role == "INTRODUCES_ERROR":
                is_error = True
                is_root_candidate = True
            elif error_role in {
                "PROPAGATES_ERROR",
                "VERIFIES_PRIOR_ERROR",
                "TERMINATES_AFTER_ERROR",
                "UNCERTAIN",
            }:
                is_root_candidate = False

            return {
                "is_error": is_error,
                "error_role": error_role,
                "is_root_candidate": is_root_candidate,
                "reason": reason,
            }

        except Exception as e:
            print(f"[LLM Evaluator Error] Failed at step {idx}: {e}")
            return {
                "is_error": False,
                "error_role": "UNCERTAIN",
                "is_root_candidate": False,
                "reason": f"API Evaluation failed: {str(e)}",
            }


# =====================================================================
# 4. Reverse Dependency DFS (Non-recursive / Iterative Stack)
# =====================================================================
class ReverseDependencyDFS:
    def __init__(
            self,
            graph_data: Dict[str, Any],
            model_name: str = "gpt-5-mini",
            api_key: Optional[str] = None,
            openai_client: Optional[Any] = None,
            cascade_enabled: bool = False,
            custom_local_evaluator: Optional[Callable] = None,
            llm_server: Optional[Any] = None,
            include_ground_truth: bool = True,
        ):
            # -------------------------------------------------------------
            # A. Load graph
            # -------------------------------------------------------------
            self.graph_data = graph_data
            self.trace_id = graph_data.get("trace_id")
            self.query = graph_data.get("query")
            self.ground_truth = graph_data.get("ground_truth") if include_ground_truth else None

            self.nodes_map: Dict[str, Dict[str, Any]] = {
                str(node["node_id"]): node
                for node in graph_data.get("nodes", [])
                if "node_id" in node
            }

            self.reverse_adj: Dict[str, List[Dict[str, Any]]] = {
                node_id: [] for node_id in self.nodes_map
            }
            self.forward_adj: Dict[str, List[Dict[str, Any]]] = {
                node_id: [] for node_id in self.nodes_map
            }

            for edge in graph_data.get("edges", []):
                if "src" not in edge or "dst" not in edge:
                    continue
                src = str(edge["src"])
                dst = str(edge["dst"])
                if dst in self.reverse_adj and src in self.nodes_map:
                    self.reverse_adj[dst].append(edge)
                if src in self.forward_adj and dst in self.nodes_map:
                    self.forward_adj[src].append(edge)

            # -------------------------------------------------------------
            # B. LLM configuration
            # -------------------------------------------------------------
            self.llm_service = LLMService(
                model_name=model_name,
                api_key=api_key,
                client=openai_client,
                llm_server=llm_server,
            )
            self.custom_local_evaluator = custom_local_evaluator
            self.cascade_enabled = bool(cascade_enabled)

            # -------------------------------------------------------------
            # C. Per-run state
            # -------------------------------------------------------------
            self.visited: Set[str] = set()
            self.active_final_causes: List[FailureCause] = []
            self.active_causes_by_node: Dict[str, List[FailureCause]] = {}
            self.local_diagnoses_by_node: Dict[str, Dict[str, Any]] = {}
            self.cascade_diagnoses_by_node: Dict[str, Dict[str, Any]] = {}
            self.cascade_relations: List[Dict[str, Any]] = []
            self.diagnosis_history: List[Dict[str, Any]] = []
            self.final_failure_analysis: Optional[Dict[str, Any]] = None

    def _get_sorted_predecessors(self, current_node_id: str) -> List[Tuple[str, str]]:
        in_edges = self.reverse_adj.get(current_node_id, [])

        src_to_best_edge: Dict[str, str] = {}
        for edge in in_edges:
            src = str(edge["src"])
            kind = str(edge.get("kind", ""))
            if src not in src_to_best_edge:
                src_to_best_edge[src] = kind
            else:
                existing_kind = src_to_best_edge[src]
                if EDGE_PRIORITY.get(kind, 0) > EDGE_PRIORITY.get(existing_kind, 0):
                    src_to_best_edge[src] = kind

        candidates = [
            (src_id, kind)
            for src_id, kind in src_to_best_edge.items()
            if src_id in self.nodes_map
        ]

        candidates.sort(
            key=lambda x: (
                EDGE_PRIORITY.get(x[1], 0),
                self.nodes_map[x[0]].get("step_index", -1),
            ),
            reverse=True,
        )
        return candidates

    def _get_sorted_successors(self, current_node_id: str, limit: int = 3) -> List[Tuple[str, str]]:
        out_edges = self.forward_adj.get(current_node_id, [])

        dst_to_best_edge: Dict[str, str] = {}
        for edge in out_edges:
            dst = str(edge["dst"])
            kind = str(edge.get("kind", ""))
            if dst not in dst_to_best_edge:
                dst_to_best_edge[dst] = kind
            else:
                existing_kind = dst_to_best_edge[dst]
                if EDGE_PRIORITY.get(kind, 0) > EDGE_PRIORITY.get(existing_kind, 0):
                    dst_to_best_edge[dst] = kind

        candidates = [
            (dst_id, kind)
            for dst_id, kind in dst_to_best_edge.items()
            if dst_id in self.nodes_map
        ]

        candidates.sort(
            key=lambda x: (
                EDGE_PRIORITY.get(x[1], 0),
                -self.nodes_map[x[0]].get("step_index", 10**9),
            ),
            reverse=True,
        )
        return candidates[:limit]

    def _resolve_query_and_ground_truth(
        self,
        query: Optional[Any] = None,
        ground_truth: Optional[Any] = None,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        resolved_query = self.query if query is None else query
        resolved_gt = self.ground_truth if ground_truth is None else ground_truth
        return resolved_query, resolved_gt

    def find_final_node(self) -> Optional[Dict[str, Any]]:
        if not self.nodes_map:
            return None

        nodes_by_step_desc = sorted(
            self.nodes_map.values(),
            key=lambda n: n.get("step_index", 0),
            reverse=True,
        )

        for node in nodes_by_step_desc:
            if bool(node.get("is_final_node")):
                return node

        for node in nodes_by_step_desc:
            if bool(node.get("is_terminal_node")):
                return node

        return nodes_by_step_desc[0]

    def _candidate_rank_key(
        self,
        item: Dict[str, Any],
        robust: bool = False,
    ) -> Tuple[int, ...]:
        node = item.get("node") or self.nodes_map.get(str(item.get("node_id")), {})
        local = item.get("local") or {}
        cascade = item.get("cascade") or {}

        relation = str(cascade.get("relation", "")).upper()
        error_role = normalize_error_role(item.get("error_role") or local.get("error_role"))
        step_index = int(node.get("step_index") or item.get("step_index") or 0)

        explicit_root = bool(item.get("is_root_candidate") or local.get("is_root_candidate"))

        relation_rank = {
            "INTRODUCES": 0,
            "PROPAGATES": 1,
            "AMPLIFIES": 1,
            "UNRELATED": 2,
            "RECOVERS": 3,
            "REFUTES": 3,
        }.get(relation, 2)

        error_role_rank = {
            "INTRODUCES_ERROR": 0,
            "PROPAGATES_ERROR": 1,
            "UNCERTAIN": 2,
            "VERIFIES_PRIOR_ERROR": 3,
            "TERMINATES_AFTER_ERROR": 4,
            "NO_ERROR": 5,
        }.get(error_role, 2)

        node_type_rank = 1 if is_tool_like_node(node) or is_terminal_like_node(node) else 0

        if robust:
            if is_user_like_node(node):
                node_type_rank = 3
            elif is_tool_like_node(node) or is_terminal_like_node(node):
                node_type_rank = 2
            elif is_preliminary_or_delegation_like_node(node):
                node_type_rank = 1
            else:
                node_type_rank = 0

            return (
                0 if explicit_root else 1,
                relation_rank,
                error_role_rank,
                node_type_rank,
                1 if is_preliminary_or_delegation_like_node(node) else 0,
                step_index,
            )

        return (
            0 if explicit_root else 1,
            relation_rank,
            error_role_rank,
            node_type_rank,
            step_index,
        )

    def _is_decisive_root_candidate(self, item: Dict[str, Any]) -> bool:
        local = item.get("local") or {}
        cascade = item.get("cascade") or {}
        error_role = normalize_error_role(item.get("error_role") or local.get("error_role"))
        relation = str(cascade.get("relation", "")).upper()
        if error_role in {
            "NO_ERROR",
            "PROPAGATES_ERROR",
            "VERIFIES_PRIOR_ERROR",
            "TERMINATES_AFTER_ERROR",
            "UNCERTAIN",
        }:
            return False
        return (
            bool(item.get("is_root_candidate") or local.get("is_root_candidate"))
            or error_role == "INTRODUCES_ERROR"
            or relation == "INTRODUCES"
        )

    def _select_root_cause_candidate(
        self,
        candidates: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], List[FailureCause], List[Dict[str, Any]]]:
        if not candidates:
            return None, [], []

        ranked = []
        for item in candidates:
            enriched = dict(item)
            ranked.append(enriched)

        decisive = [
            item for item in ranked
            if self._is_decisive_root_candidate(item)
            and normalize_error_role(item.get("error_role")) not in {
                "VERIFIES_PRIOR_ERROR",
                "TERMINATES_AFTER_ERROR",
                "NO_ERROR",
            }
        ]
        introducing_decisive_count = sum(
            1
            for item in decisive
            if normalize_error_role(item.get("error_role")) == "INTRODUCES_ERROR"
        )
        robust_rank = introducing_decisive_count >= 8

        for item in ranked:
            item["root_selection_rank_key"] = self._candidate_rank_key(
                item,
                robust=robust_rank,
            )
            item["root_selection_robust_rank"] = robust_rank

        if decisive:
            best = sorted(
                decisive,
                key=lambda item: self._candidate_rank_key(item, robust=robust_rank),
            )[0]
            ranked.sort(
                key=lambda item: (
                    0 if item is best else 1,
                    self._candidate_rank_key(item, robust=robust_rank),
                )
            )
        else:
            ranked.sort(key=lambda item: self._candidate_rank_key(item, robust=robust_rank))
            best = ranked[0]

        node_id = str(best["node_id"])
        causes = clone_causes(self.active_causes_by_node.get(node_id, []))
        return self.nodes_map.get(node_id), causes, ranked

    def _build_stage2_candidate_payload(
        self,
        candidates: List[Dict[str, Any]],
        max_candidates: int = 20,
    ) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []

        for rank, item in enumerate(candidates[:max_candidates], start=1):
            node = item.get("node") or self.nodes_map.get(str(item.get("node_id")), {})
            node_id = str(item.get("node_id") or node.get("node_id"))
            if not node_id or node_id not in self.nodes_map:
                continue

            predecessor_nodes = [
                self.nodes_map[src_id]
                for src_id, _kind in self._get_sorted_predecessors(node_id)[:3]
                if src_id in self.nodes_map
            ]
            successor_nodes = [
                self.nodes_map[dst_id]
                for dst_id, _kind in self._get_sorted_successors(node_id, limit=3)
                if dst_id in self.nodes_map
            ]

            local = item.get("local") or {}
            cascade = item.get("cascade") or {}
            payload.append(
                {
                    "rank": rank,
                    "node_id": node_id,
                    "step_index": node.get("step_index", item.get("step_index")),
                    "agent_id": canonical_agent_id(node),
                    "role": node.get("role"),
                    "semantic_role": node.get("semantic_role"),
                    "is_tool_like": is_tool_like_node(node),
                    "is_terminal_like": is_terminal_like_node(node),
                    "is_preliminary_or_delegation_like": is_preliminary_or_delegation_like_node(node),
                    "stage1_error_role": item.get("error_role"),
                    "stage1_is_root_candidate": item.get("is_root_candidate"),
                    "stage1_rank_key": item.get("root_selection_rank_key"),
                    "stage1_robust_rank": item.get("root_selection_robust_rank"),
                    "stage1_reason": truncate_for_prompt(item.get("error_reason"), 700),
                    "local_diagnosis": {
                        "is_error": local.get("is_error"),
                        "error_role": local.get("error_role"),
                        "is_root_candidate": local.get("is_root_candidate"),
                        "reason": truncate_for_prompt(local.get("reason"), 700),
                    },
                    "cascade_attribution": {
                        "relation": cascade.get("relation"),
                        "is_root_candidate": cascade.get("is_root_candidate"),
                        "confidence": cascade.get("confidence"),
                        "explanation": truncate_for_prompt(cascade.get("explanation"), 700),
                    },
                    "current_content": truncate_for_prompt(node.get("metadata"), 900),
                    "predecessor_context": [
                        {
                            "step_index": pred.get("step_index"),
                            "agent_id": canonical_agent_id(pred),
                            "role": pred.get("role"),
                            "content": truncate_for_prompt(pred.get("metadata"), 450),
                        }
                        for pred in predecessor_nodes
                    ],
                    "successor_context": [
                        {
                            "step_index": succ.get("step_index"),
                            "agent_id": canonical_agent_id(succ),
                            "role": succ.get("role"),
                            "content": truncate_for_prompt(succ.get("metadata"), 450),
                        }
                        for succ in successor_nodes
                    ],
                }
            )

        return payload

    def _rerank_root_candidate_with_llm(
        self,
        ranked_candidates: List[Dict[str, Any]],
        fallback_node: Optional[Dict[str, Any]],
        fallback_causes: List[FailureCause],
    ) -> Tuple[Optional[Dict[str, Any]], List[FailureCause], Dict[str, Any]]:
        if len(ranked_candidates) < 2:
            return fallback_node, fallback_causes, {
                "enabled": False,
                "selection_mode": "SKIPPED_TOO_FEW_CANDIDATES",
                "selected_node_id": str(fallback_node.get("node_id")) if fallback_node else None,
            }

        candidate_payload = self._build_stage2_candidate_payload(ranked_candidates)
        candidate_ids = {item["node_id"] for item in candidate_payload}
        if len(candidate_payload) < 2:
            return fallback_node, fallback_causes, {
                "enabled": False,
                "selection_mode": "SKIPPED_TOO_FEW_VALID_CANDIDATES",
                "selected_node_id": str(fallback_node.get("node_id")) if fallback_node else None,
            }

        final_node = self.find_final_node() or {}
        selection = self.llm_service.select_root_from_candidates(
            query=self.query,
            ground_truth=self.ground_truth,
            final_node=final_node,
            final_failure_analysis=self.final_failure_analysis,
            candidates=candidate_payload,
        )

        selected_id = selection.get("selected_node_id")
        if selected_id not in candidate_ids or selected_id not in self.nodes_map:
            selection["selection_mode"] = "RANK_KEY_FALLBACK"
            selection["fallback_reason"] = (
                "Stage-2 selector did not return a valid candidate node_id."
            )
            selection["selected_node_id"] = str(fallback_node.get("node_id")) if fallback_node else None
            return fallback_node, fallback_causes, {
                "enabled": True,
                "candidate_count": len(candidate_payload),
                **selection,
            }

        selected_node = self.nodes_map[selected_id]
        selected_causes = clone_causes(self.active_causes_by_node.get(selected_id, fallback_causes))
        return selected_node, selected_causes, {
            "enabled": True,
            "candidate_count": len(candidate_payload),
            **selection,
        }

    def _inspect_current_node_local(
        self,
        current_node: Dict[str, Any],
        predecessor_nodes: List[Dict[str, Any]],
        successor_nodes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        context = {
            "query": self.query,
            "ground_truth": self.ground_truth,
            "final_node": self.find_final_node(),
        }
        
        if self.custom_local_evaluator is not None:
            try:
                local_eval = self.custom_local_evaluator(
                    current_node=current_node,
                    predecessors=predecessor_nodes,
                    successors=successor_nodes or [],
                    context=context,
                )
            except TypeError:
                local_eval = self.custom_local_evaluator(
                    current_node=current_node,
                    predecessors=predecessor_nodes,
                    context=context,
                )
        else:
            local_eval = self.llm_service.evaluate_local_step(
                current_node=current_node,
                predecessors=predecessor_nodes,
                context=context,
                successors=successor_nodes or [],
            )

        local_eval = dict(local_eval or {})
        local_eval["is_error"] = bool(local_eval.get("is_error", False))
        local_eval["error_role"] = normalize_error_role(local_eval.get("error_role"))
        local_eval["is_root_candidate"] = bool(local_eval.get("is_root_candidate", False))
        if local_eval["error_role"] == "INTRODUCES_ERROR":
            local_eval["is_error"] = True
            local_eval["is_root_candidate"] = True
        if local_eval["error_role"] == "NO_ERROR":
            local_eval["is_error"] = False
        if local_eval["error_role"] in {
            "NO_ERROR",
            "PROPAGATES_ERROR",
            "VERIFIES_PRIOR_ERROR",
            "TERMINATES_AFTER_ERROR",
            "UNCERTAIN",
        }:
            local_eval["is_root_candidate"] = False

        self.local_diagnoses_by_node[str(current_node["node_id"])] = local_eval
        return local_eval

    def _inspect_current_node_with_cascade(
        self,
        current_node: Dict[str, Any],
        predecessor_nodes: List[Dict[str, Any]],
        successor_nodes: Optional[List[Dict[str, Any]]],
        active_causes: List[FailureCause],
    ) -> Tuple[Dict[str, Any], List[FailureCause], Dict[str, Any]]:
        local_eval = self._inspect_current_node_local(
            current_node=current_node,
            predecessor_nodes=predecessor_nodes,
            successor_nodes=successor_nodes or [],
        )

        cascade_eval = self.llm_service.cascade_attribution_evaluator(
            current_node=current_node,
            predecessors=predecessor_nodes,
            local_eval=local_eval,
            active_causes=active_causes,
            context={"query": self.query, "ground_truth": self.ground_truth},
        )

        current_id = str(current_node["node_id"])
        self.cascade_diagnoses_by_node[current_id] = cascade_eval

        updated_causes = clone_causes(active_causes)
        if not updated_causes:
            updated_causes, fallback_analysis = self.llm_service.initialize_active_final_causes(
                self.query,
                self.find_final_node() or current_node,
                self.ground_truth,
            )
            if self.final_failure_analysis is None:
                self.final_failure_analysis = fallback_analysis

        relation = str(cascade_eval.get("relation", "UNCERTAIN")).upper()
        confidence = float(cascade_eval.get("confidence", 0.5))
        evidence = cascade_eval.get("evidence")
        local_error_role = normalize_error_role(local_eval.get("error_role"))
        is_root_candidate = bool(cascade_eval.get("is_root_candidate", False))
        if local_error_role in {
            "NO_ERROR",
            "PROPAGATES_ERROR",
            "VERIFIES_PRIOR_ERROR",
            "TERMINATES_AFTER_ERROR",
            "UNCERTAIN",
        }:
            is_root_candidate = False
            cascade_eval["is_root_candidate"] = False
        elif relation == "INTRODUCES" and confidence >= 0.65:
            is_root_candidate = True
            cascade_eval["is_root_candidate"] = True

        next_causes: List[FailureCause] = []

        for cause in updated_causes:
            updated = FailureCause(**cause.model_dump())
            updated.confidence = max(updated.confidence, confidence)

            if relation in {"INTRODUCES", "PROPAGATES", "AMPLIFIES"}:
                updated.support_status = "SUPPORTED"
                updated.should_propagate_upstream = True

                if relation == "INTRODUCES":
                    updated.causal_status = "PROXIMATE_CAUSE"
                elif _causal_rank(updated.causal_status) < _causal_rank("PROXIMATE_CAUSE"):
                    updated.causal_status = "PROXIMATE_CAUSE"

                if updated.localized_node_id is None:
                    updated.localized_node_id = current_id
                if evidence and not updated.evidence:
                    updated.evidence = evidence

            elif relation == "RECOVERS":
                updated.support_status = "SUPPORTED"
                updated.should_propagate_upstream = True

            elif relation == "REFUTES":
                updated.support_status = "REFUTED"
                updated.should_propagate_upstream = False

            elif relation == "UNRELATED":
                if updated.support_status != "SUPPORTED":
                    updated.support_status = "HYPOTHESIS"
                updated.should_propagate_upstream = True

            else:  # UNCERTAIN
                updated.should_propagate_upstream = True

            if is_root_candidate and updated.support_status != "REFUTED":
                updated.causal_status = "ROOT_CANDIDATE"
                updated.support_status = "SUPPORTED"
                updated.should_propagate_upstream = True

            updated = _normalize_cause_state(updated)
            next_causes.append(updated)

            self.cascade_relations.append({
                "node_id": current_id,
                "step_index": current_node.get("step_index"),
                "cause_id": updated.cause_id,
                "relation": relation,
                "explanation": cascade_eval.get("explanation", ""),
                "confidence": confidence,
                "is_root_candidate": is_root_candidate,
                "is_root_cause": False,
                "evidence": evidence,
            })

        return local_eval, next_causes, cascade_eval

    # -----------------------------------------------------------------
    # Iterative Non-Recursive DFS Implementation
    # -----------------------------------------------------------------
    def _dfs_backtrack_iterative(
        self,
        start_node_id: str,
        traversal_path: List[str],
        error_chain: List[Dict[str, Any]],
        initial_active_causes: List[FailureCause],
    ) -> Tuple[Optional[Dict[str, Any]], List[FailureCause], List[Dict[str, Any]]]:
        stack: List[Tuple[str, List[FailureCause]]] = [
            (start_node_id, clone_causes(initial_active_causes))
        ]
        root_candidates: List[Dict[str, Any]] = []

        while stack:
            node_id, current_causes = stack.pop()
            if node_id in self.visited or node_id not in self.nodes_map:
                continue

            self.visited.add(node_id)
            traversal_path.append(node_id)

            current_node = self.nodes_map[node_id]
            predecessors = self._get_sorted_predecessors(node_id)
            predecessor_nodes = [self.nodes_map[src_id] for src_id, _ in predecessors]
            successors = self._get_sorted_successors(node_id)
            successor_nodes = [self.nodes_map[dst_id] for dst_id, _ in successors]
            active_causes_before = clone_causes(current_causes)

            if self.cascade_enabled:
                (
                    eval_result,
                    updated_active_causes,
                    cascade_eval,
                ) = self._inspect_current_node_with_cascade(
                    current_node=current_node,
                    predecessor_nodes=predecessor_nodes,
                    successor_nodes=successor_nodes,
                    active_causes=current_causes,
                )
            else:
                eval_result = self._inspect_current_node_local(
                    current_node=current_node,
                    predecessor_nodes=predecessor_nodes,
                    successor_nodes=successor_nodes,
                )
                updated_active_causes = current_causes
                cascade_eval = None

            if self.cascade_enabled:
                self.active_causes_by_node[node_id] = clone_causes(updated_active_causes)

            is_error = bool(eval_result.get("is_error", False))
            error_reason = eval_result.get("reason", "")
            error_role = normalize_error_role(eval_result.get("error_role"))
            local_root_candidate = bool(eval_result.get("is_root_candidate", False))
            cascade_root_candidate = bool((cascade_eval or {}).get("is_root_candidate", False))

            history_item = {
                "node_id": node_id,
                "step_index": current_node.get("step_index"),
                "agent_id": canonical_agent_id(current_node),
                "local": eval_result,
                "cascade": cascade_eval,
                "active_causes_before": [c.model_dump() for c in active_causes_before],
                "active_causes_after": [c.model_dump() for c in updated_active_causes],
            }
            self.diagnosis_history.append(history_item)

            if is_error:
                error_item = {
                    "node_id": node_id,
                    "role": current_node.get("role"),
                    "name": current_node.get("name"),
                    "agent_id": canonical_agent_id(current_node),
                    "step_index": current_node.get("step_index"),
                    "error_role": error_role,
                    "is_root_candidate": local_root_candidate or cascade_root_candidate,
                    "error_reason": error_reason,
                }
                error_chain.append(error_item)
                root_candidates.append({
                    **error_item,
                    "node": current_node,
                    "local": eval_result,
                    "cascade": cascade_eval,
                })
            elif cascade_root_candidate:
                cascade_error_role = (
                    "INTRODUCES_ERROR"
                    if str((cascade_eval or {}).get("relation", "")).upper() == "INTRODUCES"
                    else error_role
                )
                cascade_error_item = {
                    "node_id": node_id,
                    "role": current_node.get("role"),
                    "name": current_node.get("name"),
                    "agent_id": canonical_agent_id(current_node),
                    "step_index": current_node.get("step_index"),
                    "error_role": cascade_error_role,
                    "is_root_candidate": True,
                    "error_reason": (cascade_eval or {}).get("explanation", ""),
                }
                error_chain.append(cascade_error_item)
                root_candidates.append({
                    **cascade_error_item,
                    "node": current_node,
                    "local": eval_result,
                    "cascade": cascade_eval,
                })

            for next_pred_id, _kind in reversed(predecessors):
                if next_pred_id not in self.visited:
                    stack.append((next_pred_id, clone_causes(updated_active_causes)))

        return self._select_root_cause_candidate(root_candidates)

    # -----------------------------------------------------------------
    # Public entry
    # -----------------------------------------------------------------
    def find_root_cause(
        self,
        query: Optional[Any] = None,
        ground_truth: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if not self.nodes_map:
            return {"status": "error", "reason": "Graph is empty"}

        self.query, self.ground_truth = self._resolve_query_and_ground_truth(
            query=query,
            ground_truth=ground_truth,
        )
        if self.query in (None, ""):
            return {"status": "error", "reason": "Query is missing"}

        terminal_node = self.find_final_node()
        if terminal_node is None:
            return {"status": "error", "reason": "Final node not found"}

        self.visited.clear()
        self.active_final_causes = []
        self.active_causes_by_node.clear()
        self.local_diagnoses_by_node.clear()
        self.cascade_diagnoses_by_node.clear()
        self.cascade_relations.clear()
        self.diagnosis_history.clear()
        self.final_failure_analysis = None

        if self.cascade_enabled:
            (
                self.active_final_causes,
                self.final_failure_analysis,
            ) = self.llm_service.initialize_active_final_causes(
                query=self.query,
                final_node=terminal_node,
                ground_truth=self.ground_truth,
            )

        traversal_path: List[str] = []
        error_chain: List[Dict[str, Any]] = []

        # 执行非递归 DFS
        root_cause_node, final_active_causes, ranked_root_candidates = self._dfs_backtrack_iterative(
            start_node_id=str(terminal_node["node_id"]),
            traversal_path=traversal_path,
            error_chain=error_chain,
            initial_active_causes=clone_causes(self.active_final_causes),
        )

        stage2_root_selection = {
            "enabled": False,
            "selection_mode": "DISABLED",
            "selected_node_id": str(root_cause_node.get("node_id")) if root_cause_node else None,
        }
        if ranked_root_candidates:
            (
                root_cause_node,
                final_active_causes,
                stage2_root_selection,
            ) = self._rerank_root_candidate_with_llm(
                ranked_candidates=ranked_root_candidates,
                fallback_node=root_cause_node,
                fallback_causes=final_active_causes,
            )

        root_confirmation: List[Dict[str, Any]] = []
        if self.cascade_enabled and root_cause_node is not None:
            root_id = str(root_cause_node["node_id"])
            root_state = merge_causes(
                self.active_causes_by_node.get(root_id, []),
                final_active_causes,
            )

            confirmed = []
            for cause in root_state:
                c = FailureCause(**cause.model_dump())
                c.root_node_id = root_id
                c.causal_status = "ROOT_CAUSE"
                c.support_status = "SUPPORTED"
                c.should_propagate_upstream = False
                c = _normalize_cause_state(c)
                confirmed.append(c)

                root_confirmation.append({
                    "cause_id": c.cause_id,
                    "root_node_id": root_id,
                    "causal_status": "ROOT_CAUSE",
                    "policy": "Confirmed after candidate collection and root selection.",
                })

            self.active_causes_by_node[root_id] = confirmed

        predicted_step = root_cause_node.get("step_index") if root_cause_node else None
        predicted_agent = canonical_agent_id(root_cause_node) if root_cause_node else None

        cascade_failure = {
            "enabled": self.cascade_enabled,
            "final_failure_analysis": self.final_failure_analysis,
            "active_final_causes": [c.model_dump() for c in self.active_final_causes],
            "cascade_relations": self.cascade_relations,
            "root_candidate_nodes": [
                str(item["node_id"]) for item in ranked_root_candidates
            ],
            "stage2_root_selection": stage2_root_selection,
            "ranked_root_candidates": [
                {
                    "node_id": item.get("node_id"),
                    "step_index": item.get("step_index"),
                    "agent_id": item.get("agent_id"),
                    "error_role": item.get("error_role"),
                    "is_root_candidate": item.get("is_root_candidate"),
                    "root_selection_rank_key": item.get("root_selection_rank_key"),
                    "root_selection_robust_rank": item.get("root_selection_robust_rank"),
                    "error_reason": item.get("error_reason"),
                }
                for item in ranked_root_candidates
            ],
        }

        return {
            "trace_id": self.trace_id,
            "root_cause_found": root_cause_node is not None,
            "root_cause_node": root_cause_node,
            "predicted_step": predicted_step,
            "predicted_agent": predicted_agent,
            "error_chain": error_chain,
            "traversal_path": traversal_path,
            "query": self.query,
            "ground_truth": self.ground_truth,
            "cascade_enabled": self.cascade_enabled,
            "diagnosis_mode": "LOCAL_PLUS_CASCADE" if self.cascade_enabled else "LOCAL_ONLY",
            "active_final_causes": [c.model_dump() for c in self.active_final_causes],
            "final_failure_analysis": self.final_failure_analysis,
            "active_causes_by_node": {
                node_id: [c.model_dump() for c in causes]
                for node_id, causes in self.active_causes_by_node.items()
            },
            "cascade_relations": self.cascade_relations,
            "root_cause_confirmation": root_confirmation,
            "ranked_root_candidates": ranked_root_candidates,
            "stage2_root_selection": stage2_root_selection,
            "cascade_failure": cascade_failure,
            "root_candidate_nodes": sorted({
                str(item["node_id"])
                for item in self.cascade_relations
                if item.get("is_root_candidate")
            } | {str(item["node_id"]) for item in ranked_root_candidates}),
            "local_diagnoses": self.local_diagnoses_by_node,
            "cascade_diagnoses": self.cascade_diagnoses_by_node,
            "diagnosis_history": self.diagnosis_history,
        }

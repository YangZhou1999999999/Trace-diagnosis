#v2
import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Callable, Tuple, Literal

from ...common.labels import clean_label
from ...common.llm import LLMServer
from ...common.cascade import CASCADE_RELATION_SET, normalize_cascade_relation
from ...common.traversal import pop_frontier, push_frontier, resolve_traversal_mode

# =====================================================================
# 1. Edge priority
# =====================================================================
EDGE_PRIORITY = {
    "execution_flow": 3,
    "sequential_flow": 2,
    "data_flow": 1,
}


def canonical_agent_id(node: Dict[str, Any]) -> str:
    agent_id = clean_label(node.get("agent_id"))
    if agent_id:
        return agent_id

    name = clean_label(node.get("name"))
    if name:
        return name

    role = clean_label(node.get("role"))
    if role:
        return role
    return "Unknown"


def is_exposure_node(node: Dict[str, Any]) -> bool:
    return bool(
        node.get("is_tool_executor")
        or node.get("is_terminal_node")
        or node.get("is_final_node")
    )


def is_reasoning_node(node: Dict[str, Any]) -> bool:
    return clean_label(node.get("semantic_role")).lower() == "reasoning"


def is_terminal_marker_node(node: Dict[str, Any]) -> bool:
    text = node_text(node).strip().lower()
    return bool(node.get("is_terminal_node")) and (
        text in {"terminate", "terminated"} or text.startswith("terminate")
    )


def normalize_candidate_role(value: Any) -> str:
    role = str(value or "").strip().upper()
    allowed = {
        "INTRODUCER_CANDIDATE",
        "PROPAGATION_EVIDENCE",
        "VERIFICATION_EVIDENCE",
        "EXPOSURE_EVIDENCE",
        "UPSTREAM_PRODUCER",
        "NO_RELEVANT_EVIDENCE",
    }
    return role if role in allowed else "NO_RELEVANT_EVIDENCE"


def candidate_role_from_error_role(error_role: str, node: Dict[str, Any]) -> str:
    error_role = normalize_error_role(error_role)
    if error_role == "INTRODUCES_ERROR":
        return "INTRODUCER_CANDIDATE"
    if error_role == "PROPAGATES_ERROR":
        return "PROPAGATION_EVIDENCE"
    if error_role == "VERIFIES_PRIOR_ERROR":
        return "VERIFICATION_EVIDENCE"
    if error_role == "TERMINATES_AFTER_ERROR":
        return "EXPOSURE_EVIDENCE"
    if is_exposure_node(node):
        return "EXPOSURE_EVIDENCE"
    return "NO_RELEVANT_EVIDENCE"


def should_expand_upstream_from_candidate(candidate_role: str, error_role: str, node: Dict[str, Any]) -> bool:
    candidate_role = normalize_candidate_role(candidate_role)
    error_role = normalize_error_role(error_role)
    return (
        candidate_role in {
            "PROPAGATION_EVIDENCE",
            "VERIFICATION_EVIDENCE",
            "EXPOSURE_EVIDENCE",
        }
        or error_role in {
            "PROPAGATES_ERROR",
            "VERIFIES_PRIOR_ERROR",
            "TERMINATES_AFTER_ERROR",
        }
        or is_exposure_node(node)
    )


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
        llm_server: Optional[LLMServer] = None,
    ):
        self.model_name = model_name
        self.llm_server = llm_server or LLMServer(
            api_key=api_key,
            client=client,
            http_client=http_client,
            validate_key_before_import=True,
            missing_key_message="OpenAI API Key is missing. Pass api_key or set OPENAI_API_KEY env.",
        )

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
            response = self.llm_server.create(
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
        local_eval: Optional[Dict[str, Any]],
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

        if local_eval is None:
            prompt += (
                "\nLocal inspection has not been performed yet. Infer the cascade relation "
                "from the active causes, current step and predecessor evidence only. "
                "Do not interpret missing Local results as evidence of correctness or error.\n"
            )

        try:
            response = self.llm_server.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a careful assistant analyzing causal dependencies."},
                    {"role": "user", "content": prompt},
                ],
            )
            reply_text = (response.choices[0].message.content or "").strip()

            parsed = extract_json_object_loose(reply_text)

            allowed_relations = CASCADE_RELATION_SET
            relation = str(
                parsed.get("relation")
                or loose_enum_field(reply_text, "relation", allowed_relations, "UNCERTAIN")
            ).upper()
            relation = normalize_cascade_relation(relation)

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
            "high-recall candidate list. The candidate list contains both direct "
            "LLM-diagnosed evidence and graph-expanded upstream producer candidates. "
            "It may contain false "
            "positives. Your job is NOT to find new candidates. Compare the candidates and "
            "select the directly supported, decisive mistake that best explains the final "
            "failure. Causal order matters, but do not select a candidate merely because it is "
            "earlier.\n\n"
            "Selection policy:\n"
            "- First require evidence in the candidate's own step that it made a concrete wrong "
            "decision, action, assumption, claim, or omission that explains the downstream failure.\n"
            "- Compare expected and observed answers by meaning. Do not treat punctuation, "
            "capitalization, spacing, or a harmless name-format variant as a material error "
            "unless the task explicitly requires that exact form or the variation changes the "
            "referent or meaning. A final-answer wording difference alone is not enough to select "
            "the final step as the cause.\n"
            "- If an earlier candidate is only a broad plan, possible contributor, or speculative "
            "precursor, prefer a later candidate whose own content directly shows the decisive "
            "mistake. Use earliest causal position as a tie-breaker between candidates with "
            "comparably direct evidence, not as the primary selection rule.\n"
            "- The target is the step that introduced the bad action, assumption, "
            "retrieval, extraction, code, calculation, or factual claim. Do not choose "
            "a later step merely because it exposed, repeated, verified, or finalized "
            "the wrong outcome.\n"
            "- CandidateRole tells how Stage 1 collected the item. "
            "INTRODUCER_CANDIDATE and UPSTREAM_PRODUCER are possible roots; "
            "PROPAGATION_EVIDENCE, VERIFICATION_EVIDENCE, and EXPOSURE_EVIDENCE "
            "are usually symptoms unless their own content introduced a new mistake.\n"
            "- Use stage1_error_role and stage1_is_root_candidate as evidence, but "
            "do not treat them as infallible; Stage 1 is high-recall and may contain "
            "false positives.\n"
            "- If a tool/output/verifier/final node exposes a wrong result, compare it "
            "with its upstream producer candidates and prefer the producer when the "
            "producer's method, code, assumption, search, extraction, or interpretation "
            "explains the exposed failure.\n"
            "- Prefer concrete wrong search/click/navigation/extraction/method/assumption "
            "over generic planning or delegation.\n"
            "- A planning/delegation step is root only if it itself states the wrong "
            "target, wrong constraint, wrong source, wrong method, or wrong assumption. "
            "Do not select it merely because it assigned work.\n"
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
            response = self.llm_server.create(
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
            "This is Stage 1 high-recall failure-evidence collection. Determine whether this step is relevant evidence for the failure path. Do not force a final root decision here; a later Stage 2 selector will compare candidates globally.\n"
            "Important benchmark policy:\n"
            "- The target label is the mistaken agent step best supported by the trace, not necessarily the last step that exposes the wrong final answer.\n"
            "- Compare the expected and observed answers by meaning. Do not treat punctuation, capitalization, spacing, or a harmless name-format variant as a material error unless the task explicitly requires that exact form or the variation changes the referent or meaning.\n"
            "- A final-answer wording difference can expose a failure, but it does not by itself establish where the failure was introduced; trace the cause to a supported decision, retrieval, or interpretation in the trace.\n"
            "- INTRODUCES_ERROR: this step makes a concrete wrong decision, assumption, retrieval/navigation choice, code/method choice, data interpretation, or reasoning step that explains the final failure.\n"
            "- A concrete mistake can be an omission at the responsible step: skipping a required check, failing to verify an explicit constraint, not performing the needed interaction/action, or prematurely concluding when this step had enough context to continue correctly.\n"
            "- PROPAGATES_ERROR: this step repeats or uses an earlier mistake without introducing a new one.\n"
            "- VERIFIES_PRIOR_ERROR: this step merely confirms/endorses an earlier wrong result.\n"
            "- TERMINATES_AFTER_ERROR: this step only terminates after an earlier mistake.\n"
            "- NO_ERROR: this step is correct or only neutral bookkeeping.\n"
            "- UNCERTAIN: insufficient evidence.\n\n"
            "CandidateRole policy:\n"
            "- INTRODUCER_CANDIDATE: this step may have introduced the concrete mistake.\n"
            "- PROPAGATION_EVIDENCE: this step carries forward an earlier mistake.\n"
            "- VERIFICATION_EVIDENCE: this step confirms or endorses an earlier mistake.\n"
            "- EXPOSURE_EVIDENCE: this step exposes the failure, e.g. wrong tool output, wrong final answer, or termination after failure.\n"
            "- NO_RELEVANT_EVIDENCE: this step is not useful for the failure path.\n\n"
            "Tool/action step policy:\n"
            "- If a tool/action step merely executes code, query, or instructions produced by an earlier reasoning step, do NOT mark it as the root just because the output is wrong; use PROPAGATES_ERROR unless the tool/action step itself introduced the bug.\n"
            "- A browsing, retrieval, file-handling, or interaction step is an active agent step when it chooses a search, click, page, file, extraction, interaction, or retrieval judgment; a bad choice there can INTRODUCE_ERROR.\n"
            "- If an expert/reasoning step writes faulty code, chooses the wrong method, makes an invalid assumption, or asks the tool to do the wrong thing, that expert/reasoning step is the root candidate.\n"
            "- If a later expert step adds a new concrete wrong assumption or wrong interpretation, it can INTRODUCE_ERROR even if earlier context was imperfect. Use PROPAGATES_ERROR only when it mechanically copies an earlier wrong result.\n"
            "- Do not mark a preliminary high-level step as INTRODUCES_ERROR only because later steps failed; require exact evidence in this step.\n"
            "- Do not mark a planning/delegation step as RootCandidate merely because it asked another agent to search, browse, or verify. Use RootCandidate: Yes only if this step itself names the wrong target, wrong constraint, wrong source, wrong method, or wrong assumption.\n"
            "- If this step only delegates work and a later agent step makes the concrete bad search, click, extraction, or answer choice, this step is usually NO_ERROR or PROPAGATES_ERROR, not INTRODUCES_ERROR.\n"
            "- RootCandidate must be No for NO_ERROR, PROPAGATES_ERROR, VERIFIES_PRIOR_ERROR, TERMINATES_AFTER_ERROR, and UNCERTAIN.\n\n"
            "Use later evidence only to decide whether THIS step's claim, action, method, or assumption was already wrong. Do not blame this step for mistakes introduced only by a later step.\n\n"
            "Respond ONLY in format:\n"
            "1. Yes/No.\n"
            "2. ErrorRole: INTRODUCES_ERROR|PROPAGATES_ERROR|VERIFIES_PRIOR_ERROR|TERMINATES_AFTER_ERROR|NO_ERROR|UNCERTAIN\n"
            "3. RootCandidate: Yes/No\n"
            "4. CandidateRole: INTRODUCER_CANDIDATE|PROPAGATION_EVIDENCE|VERIFICATION_EVIDENCE|EXPOSURE_EVIDENCE|NO_RELEVANT_EVIDENCE\n"
            "5. Reason: [brief evidence-based explanation]"
        )

        if context.get("cascade_evidence") is not None:
            # Cascade-first only: retain the existing Local instructions and output format.
            prompt += (
                "\n\nPreliminary cascade attribution (hypothesis, not ground truth):\n"
                + json.dumps(context["cascade_evidence"], ensure_ascii=False, default=str)
                + "\nUse this as supplementary evidence; independently verify the current step "
                  "against the trace. Do not assume the cascade attribution is correct."
            )

        try:
            response = self.llm_server.create(
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

            candidate_role_match = re.search(r"CandidateRole:\s*([A-Z_]+)", reply_text, re.IGNORECASE)
            candidate_role = normalize_candidate_role(
                candidate_role_match.group(1) if candidate_role_match else None
            )

            reason_match = re.search(r"(?:5|4|3|2)\.\s*Reason:\s*(.*)", reply_text, re.IGNORECASE | re.DOTALL)
            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                reason = reply_text

            if candidate_role == "NO_RELEVANT_EVIDENCE":
                candidate_role = candidate_role_from_error_role(error_role, current_node)

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
                "candidate_role": candidate_role,
                "reason": reason,
            }

        except Exception as e:
            print(f"[LLM Evaluator Error] Failed at step {idx}: {e}")
            return {
                "is_error": False,
                "error_role": "UNCERTAIN",
                "is_root_candidate": False,
                "candidate_role": "NO_RELEVANT_EVIDENCE",
                "reason": f"API Evaluation failed: {str(e)}",
            }

    def evaluate_local_step_with_cascade(
        self,
        current_node: Dict[str, Any],
        predecessors: List[Dict[str, Any]],
        successors: List[Dict[str, Any]],
        active_causes: List[FailureCause],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Evaluate a Who&When step and its active cascade state in one LLM call."""
        query = context.get("query", "N/A")
        ground_truth = context.get("ground_truth", "N/A")
        final_node = context.get("final_node")
        idx = current_node.get("step_index", "N/A")
        current_agent = canonical_agent_id(current_node)
        current_role = clean_label(current_node.get("role")) or "agent"
        current_content = truncate_for_prompt(current_node.get("metadata"), 1600)

        formatted_predecessors = []
        for pred in predecessors:
            pred_idx = pred.get("step_index", "?")
            pred_agent = canonical_agent_id(pred)
            pred_role = clean_label(pred.get("role")) or "unknown"
            pred_content = truncate_for_prompt(pred.get("metadata"), 700)
            formatted_predecessors.append(
                f"[Step {pred_idx} | {pred_agent} (role: {pred_role})]: {pred_content}"
            )

        formatted_successors = []
        for succ in successors[:3]:
            succ_idx = succ.get("step_index", "?")
            succ_agent = canonical_agent_id(succ)
            succ_role = clean_label(succ.get("role")) or "unknown"
            succ_content = truncate_for_prompt(succ.get("metadata"), 700)
            formatted_successors.append(
                f"[Step {succ_idx} | {succ_agent} (role: {succ_role})]: {succ_content}"
            )

        if isinstance(final_node, dict):
            final_context = (
                f"[Step {final_node.get('step_index', 'N/A')} | "
                f"{canonical_agent_id(final_node)}]: "
                f"{truncate_for_prompt(final_node.get('metadata'), 900)}"
            )
        else:
            final_context = "Not provided"

        # Active causes are repeated in every visited node's request. Keep the
        # fields needed for attribution and bound free-text fields so long
        # hypotheses do not dominate the combined prompt.
        active_payload = [
            {
                "cause_id": cause.cause_id,
                "description": truncate_for_prompt(cause.description, 700),
                "confidence": cause.confidence,
                "support_status": cause.support_status,
                "causal_status": cause.causal_status,
                "should_propagate_upstream": cause.should_propagate_upstream,
                "localized_node_id": cause.localized_node_id,
                "evidence": truncate_for_prompt(cause.evidence, 300)
                if cause.evidence
                else None,
            }
            for cause in active_causes
        ]
        prompt = (
            "Evaluate this Who&When trace step and perform cascade attribution together. "
            "Return one JSON object with separate `local` and `cascade` objects.\n\n"
            "The two output channels are independent. Assess LOCAL first from the current "
            "step and task context. Do not let the active-cause relation determine or suppress "
            "the local result: a step can contain a local mistake even when cascade relation "
            "is UNRELATED, and a cascade relation does not by itself prove a local mistake.\n\n"
            "LOCAL EVALUATION RULES (high-recall evidence collection):\n"
            "- Identify the mistaken agent step best supported by the trace, not only the later "
            "step that exposes the wrong final answer. Do not make the final global root choice "
            "here; Stage 2 compares candidates.\n"
            "- Compare the expected and observed answers by meaning. Do not treat punctuation, "
            "capitalization, spacing, or a harmless name-format variant as a material error "
            "unless the task explicitly requires that exact form or the variation changes the "
            "referent or meaning. A final-answer wording difference can expose a failure, but "
            "does not by itself establish where the failure was introduced.\n"
            "- INTRODUCES_ERROR means this step makes a concrete wrong decision, assumption, "
            "retrieval/navigation choice, code/method choice, data interpretation, or reasoning "
            "step that explains the final failure.\n"
            "- A concrete mistake can be an omission at the responsible step: skipping a required "
            "check or verification, failing to follow an explicit constraint, not performing a "
            "needed interaction/action, or prematurely concluding when this step had enough "
            "information to continue correctly. Judge omissions against the query and direct "
            "trace context, not hindsight alone.\n"
            "- PROPAGATES_ERROR repeats or uses an earlier mistake without introducing a new one; "
            "VERIFIES_PRIOR_ERROR merely confirms or endorses it; TERMINATES_AFTER_ERROR only "
            "terminates after it; NO_ERROR is correct or neutral bookkeeping; UNCERTAIN means "
            "evidence is insufficient.\n"
            "- If a tool/action step merely executes code, a query, or instructions produced by an "
            "earlier reasoning step, do not mark it as the root only because its output is wrong; "
            "use PROPAGATES_ERROR unless the tool/action step itself introduced the bug.\n"
            "- A browsing, retrieval, file-handling, or interaction step is an active agent step "
            "when it chooses a search, click, page, file, extraction, or retrieval judgment; a bad "
            "choice there can INTRODUCE_ERROR. If an expert/reasoning step writes faulty code, "
            "chooses the wrong method, makes an invalid assumption, or gives the tool wrong "
            "instructions, that step can be the root candidate. A later expert step can introduce "
            "a new error when it adds a concrete wrong assumption or interpretation.\n"
            "- Do not mark a preliminary planning/delegation step as INTRODUCES_ERROR merely "
            "because later steps failed. It must itself name the wrong target, constraint, source, "
            "method, or assumption. If it only delegates work and a later step makes the concrete "
            "bad choice, the planning step is usually NO_ERROR or PROPAGATES_ERROR.\n"
            "- Use later evidence only to judge whether this step's own claim, action, method, or "
            "assumption was already wrong. Do not blame it for a mistake introduced only later.\n"
            "- RootCandidate must be false for NO_ERROR, PROPAGATES_ERROR, VERIFIES_PRIOR_ERROR, "
            "TERMINATES_AFTER_ERROR, and UNCERTAIN. CandidateRole must be one of "
            "INTRODUCER_CANDIDATE, PROPAGATION_EVIDENCE, VERIFICATION_EVIDENCE, "
            "EXPOSURE_EVIDENCE, NO_RELEVANT_EVIDENCE.\n\n"
            "CASCADE RULES:\n"
            "- Assess EACH active cause independently by its cause_id. Return one relation "
            "for each cause; do not apply a single aggregate relation to all causes. A step can "
            "propagate one cause while being unrelated to another.\n"
            "- relation must be INTRODUCES, PROPAGATES, AMPLIFIES, RECOVERS, UNRELATED, "
            "REFUTES, or UNCERTAIN. Only cite evidence from this current step.\n"
            "- is_root_candidate is true only when this step introduces that specific concrete "
            "cause; it must be false for a local role of NO_ERROR, PROPAGATES_ERROR, "
            "VERIFIES_PRIOR_ERROR, TERMINATES_AFTER_ERROR, or UNCERTAIN.\n"
            "- Return exactly one cause_relations entry for each supplied cause_id, including "
            "UNCERTAIN when evidence is insufficient. Do not invent or omit cause IDs.\n"
            "- If there are no active causes, return an empty cause_relations array. Do not "
            "invent causes.\n\n"
            "Return JSON only in exactly this shape. Boolean values below are placeholders; "
            "choose each value from the evidence:\n"
            "{\n"
            '  "local": {"is_error": true, "error_role": "INTRODUCES_ERROR|PROPAGATES_ERROR|'
            'VERIFIES_PRIOR_ERROR|TERMINATES_AFTER_ERROR|NO_ERROR|UNCERTAIN", '
            '"is_root_candidate": false, "candidate_role": "INTRODUCER_CANDIDATE|'
            'PROPAGATION_EVIDENCE|VERIFICATION_EVIDENCE|EXPOSURE_EVIDENCE|'
            'NO_RELEVANT_EVIDENCE", "reason": "brief evidence-based explanation"},\n'
            '  "cascade": {"cause_relations": [{"cause_id": "cause ID", '
            '"relation": "INTRODUCES|PROPAGATES|AMPLIFIES|RECOVERS|UNRELATED|REFUTES|'
            'UNCERTAIN", "explanation": "brief explanation", "is_root_candidate": false, '
            '"confidence": 0.0, "evidence": null}], "explanation": "brief summary"}\n'
            "}\n\n"
            f"USER QUERY:\n{truncate_for_prompt(query, 1400)}\n\n"
            f"GROUND TRUTH:\n{truncate_for_prompt(ground_truth, 900)}\n\n"
            f"FINAL ANSWER / TERMINAL EVIDENCE:\n{final_context}\n\n"
            f"DIRECT PREDECESSORS:\n{chr(10).join(formatted_predecessors) or 'None (Initial Step)'}\n\n"
            f"IMMEDIATE LATER EVIDENCE:\n{chr(10).join(formatted_successors) or 'None'}\n\n"
            f"ACTIVE CAUSES:\n{json.dumps(active_payload, ensure_ascii=False, default=str)}\n\n"
            f"CURRENT STEP ({idx}) by {current_agent} (role: {current_role}):\n{current_content}"
        )

        try:
            response = self.llm_server.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You carefully evaluate agent trace steps and causal dependencies. Return strict JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            reply_text = (response.choices[0].message.content or "").strip()
            parsed = extract_json_object_loose(reply_text)
            local = parsed.get("local") if isinstance(parsed.get("local"), dict) else {}
            cascade = parsed.get("cascade") if isinstance(parsed.get("cascade"), dict) else {}

            def parsed_bool(value: Any) -> bool:
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return bool(value)
                if isinstance(value, str):
                    lowered = value.strip().lower()
                    if lowered in {"true", "yes", "1"}:
                        return True
                    if lowered in {"false", "no", "0"}:
                        return False
                return False

            local_role = normalize_error_role(local.get("error_role"))
            local_result = {
                "is_error": parsed_bool(local.get("is_error")),
                "error_role": local_role,
                "is_root_candidate": parsed_bool(local.get("is_root_candidate")),
                "candidate_role": normalize_candidate_role(local.get("candidate_role")),
                "reason": str(local.get("reason") or reply_text),
            }

            def normalized_cause_relation(item: Dict[str, Any]) -> Dict[str, Any]:
                relation = normalize_cascade_relation(item.get("relation"))
                try:
                    confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
                except (TypeError, ValueError):
                    confidence = 0.5
                evidence = item.get("evidence")
                if evidence is None or (
                    isinstance(evidence, str)
                    and evidence.strip().lower() in {"", "null", "none"}
                ):
                    evidence = None
                return {
                    "cause_id": str(item.get("cause_id", "")),
                    "relation": relation,
                    "explanation": str(item.get("explanation") or ""),
                    "is_root_candidate": parsed_bool(item.get("is_root_candidate")),
                    "confidence": confidence,
                    "evidence": evidence,
                }

            raw_cause_relations = cascade.get("cause_relations")
            parsed_cause_relations = (
                [
                    normalized_cause_relation(item)
                    for item in raw_cause_relations
                    if isinstance(item, dict)
                ]
                if isinstance(raw_cause_relations, list)
                else []
            )
            relations_by_id = {
                item["cause_id"]: item
                for item in parsed_cause_relations
                if item["cause_id"]
            }

            # A legacy one-relation response is unambiguous only for a single
            # active cause. With multiple causes, missing per-cause output must
            # not silently assign one judgment to every cause.
            legacy_relation: Optional[Dict[str, Any]] = None
            if len(active_causes) == 1 and not parsed_cause_relations:
                legacy_relation = normalized_cause_relation({
                    "cause_id": active_causes[0].cause_id,
                    "relation": cascade.get("relation")
                    or loose_enum_field(reply_text, "relation", CASCADE_RELATION_SET, "UNCERTAIN"),
                    "explanation": cascade.get("explanation") or reply_text,
                    "is_root_candidate": cascade.get("is_root_candidate")
                    if "is_root_candidate" in cascade
                    else loose_bool_field(reply_text, "is_root_candidate", False),
                    "confidence": cascade.get("confidence", loose_float_field(reply_text, "confidence", 0.5)),
                    "evidence": cascade.get("evidence") or loose_string_field(reply_text, "evidence"),
                })

            cause_relations = []
            for cause in active_causes:
                cause_id = str(cause.cause_id)
                relation_item = relations_by_id.get(cause_id)
                if relation_item is None and legacy_relation is not None:
                    relation_item = legacy_relation
                if relation_item is None:
                    relation_item = {
                        "cause_id": cause_id,
                        "relation": "UNCERTAIN",
                        "explanation": "No per-cause attribution was returned.",
                        "is_root_candidate": False,
                        "confidence": 0.0,
                        "evidence": None,
                    }
                cause_relations.append(dict(relation_item, cause_id=cause_id))

            observed_relations = [item["relation"] for item in cause_relations]
            if "INTRODUCES" in observed_relations:
                summary_relation = "INTRODUCES"
            elif observed_relations and len(set(observed_relations)) == 1:
                summary_relation = observed_relations[0]
            else:
                summary_relation = "UNCERTAIN"
            summary_sources = [
                item for item in cause_relations
                if item["relation"] == summary_relation
            ]
            if not summary_sources:
                summary_sources = cause_relations
            summary_explanation = str(
                cascade.get("explanation")
                or " ".join(
                    f"{item['cause_id']}: {item['explanation']}"
                    for item in summary_sources
                    if item.get("explanation")
                )
                or reply_text
            )
            summary_evidence = next(
                (item.get("evidence") for item in summary_sources if item.get("evidence")),
                None,
            )
            summary_confidence = max(
                (item["confidence"] for item in summary_sources),
                default=0.0,
            )

            cascade_result = {
                "relation": summary_relation,
                "cause_relations": cause_relations,
                "explanation": summary_explanation,
                "is_root_candidate": any(
                    item["is_root_candidate"]
                    or (item["relation"] == "INTRODUCES" and item["confidence"] >= 0.65)
                    for item in cause_relations
                ),
                "confidence": summary_confidence,
                "evidence": summary_evidence,
                "json_parse_recovered": not bool(
                    local
                    and cascade
                    and isinstance(raw_cause_relations, list)
                    and all(str(cause.cause_id) in relations_by_id for cause in active_causes)
                ),
            }
            return local_result, cascade_result
        except Exception as exc:
            error = f"Combined evaluation failed: {exc}"
            print(f"[Combined Evaluator Error] Failed at step {idx}: {exc}")
            return (
                {
                    "is_error": False,
                    "error_role": "UNCERTAIN",
                    "is_root_candidate": False,
                    "candidate_role": "NO_RELEVANT_EVIDENCE",
                    "reason": error,
                },
                {
                    "relation": "UNCERTAIN",
                    "explanation": error,
                    "is_root_candidate": False,
                    "confidence": 0.0,
                    "evidence": None,
                    "json_parse_recovered": True,
                },
            )

    def evaluate_base_step(
        self,
        current_node: Dict[str, Any],
        predecessors: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        problem = context.get("query", "N/A")
        ground_truth = context.get("ground_truth", "N/A")
        idx = current_node.get("step_index", "N/A")
        agent_role = current_node.get("role", "agent")
        agent_name = current_node.get("name", "agent")

        formatted_predecessors = []
        for pred in predecessors:
            pred_idx = pred.get("step_index", "?")
            pred_agent = clean_label(pred.get("name")) or clean_label(pred.get("role")) or "unknown"
            pred_role = clean_label(pred.get("role")) or "unknown"
            pred_desc = f"{pred_agent} (role: {pred_role})" if pred_agent != "unknown" else pred_role
            pred_content = (
                json.dumps(pred.get("metadata"), ensure_ascii=False)
                if isinstance(pred.get("metadata"), (dict, list))
                else str(pred.get("metadata"))
            )
            formatted_predecessors.append(f"[Step {pred_idx} | {pred_desc}]: {pred_content}")

        predecessor_context = "\n".join(formatted_predecessors) or "None (Initial Step)"

        raw_metadata = current_node.get("metadata")
        curr_content = (
            json.dumps(raw_metadata, ensure_ascii=False)
            if isinstance(raw_metadata, (dict, list))
            else str(raw_metadata)
        )
        display_agent = clean_label(agent_name) or clean_label(agent_role) or "agent"
        display_role = clean_label(agent_role)

        agent_desc = display_agent
        if display_role and display_role != display_agent:
            agent_desc = f"{display_agent} (role: {display_role})"

        prompt = (
            "You are an AI assistant tasked with evaluating the correctness of each step in an ongoing "
            "multi-agent conversation aimed at solving a real-world problem. "
            f"The problem being addressed is: {problem}\n"
            f"The Answer for the problem is: {ground_truth}\n\n"
            f"Here is the context of directly dependent prior steps:\n{predecessor_context}\n\n"
            f"This step ({idx}) was by '{agent_desc}' . Content:\n{curr_content}\n\n"
            f"Your task is to determine whether this agent's action (Step {idx}) contains an error that "
            "could hinder the problem-solving process or lead to an incorrect solution. "
            "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment.\n"
            "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process.\n"
            "Attention: Respond ONLY in the format:\n"
            "1. Yes/No.\n"
            "2. Reason: [Your explanation here]"
        )

        system_prompt = "You are a helpful assistant skilled in analyzing conversations."

        try:
            response = self.llm_server.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            reply_text = response.choices[0].message.content.strip()

            is_error = False
            reason = "No detailed reason extracted."

            yes_no_match = re.search(r"(?:1[\.\)]?\s*)?(Yes|No)", reply_text, re.IGNORECASE)
            if yes_no_match:
                decision = yes_no_match.group(1).lower()
                is_error = decision == "yes"

            reason_match = re.search(
                r"(?:2[\.\)]?\s*)?Reason:\s*(.*)",
                reply_text,
                re.IGNORECASE | re.DOTALL,
            )
            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                reason = reply_text

            return {
                "is_error": is_error,
                "reason": reason,
            }

        except Exception as e:
            print(f"[LLM Evaluator Error] Failed at step {idx}: {e}")
            return {
                "is_error": False,
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
        llm_server: Optional[LLMServer] = None,
        traversal_mode: str = "dfs",
        combined_cascade_enabled: bool = False,
        separate_order: str = "local_then_cascade",
    ):
        if separate_order not in {"local_then_cascade", "cascade_then_local"}:
            raise ValueError(f"Unsupported Separate order: {separate_order!r}")
        self.separate_order = separate_order
        # -------------------------------------------------------------
        # A. Load graph
        # -------------------------------------------------------------
        self.trace_id = graph_data.get("trace_id")
        self.query = graph_data.get("query")
        self.ground_truth = graph_data.get("ground_truth")

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
        self.combined_cascade_enabled = bool(
            combined_cascade_enabled and custom_local_evaluator is None
        )
        self.traversal_mode = resolve_traversal_mode(traversal_mode, default="dfs")

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
        self.final_evidence_node_id: Optional[str] = None

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

    def find_final_evidence_node(self) -> Optional[Dict[str, Any]]:
        """Return the latest substantive node to use as final-failure evidence."""
        if not self.nodes_map:
            return None

        nodes_by_step_desc = sorted(
            self.nodes_map.values(),
            key=lambda n: n.get("step_index", 0),
            reverse=True,
        )

        for node in nodes_by_step_desc:
            if is_terminal_marker_node(node):
                continue
            text = node_text(node).strip()
            if text:
                return node

        return self.find_final_node()

    def _candidate_rank_key(self, item: Dict[str, Any]) -> Tuple[int, ...]:
        node = item.get("node") or self.nodes_map.get(str(item.get("node_id")), {})
        local = item.get("local") or {}
        cascade = item.get("cascade") or {}

        relation = str(cascade.get("relation", "")).upper()
        error_role = normalize_error_role(item.get("error_role") or local.get("error_role"))
        candidate_role = normalize_candidate_role(item.get("candidate_role") or local.get("candidate_role"))
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

        candidate_role_rank = {
            "INTRODUCER_CANDIDATE": 0,
            "UPSTREAM_PRODUCER": 1,
            "PROPAGATION_EVIDENCE": 2,
            "VERIFICATION_EVIDENCE": 3,
            "EXPOSURE_EVIDENCE": 4,
            "NO_RELEVANT_EVIDENCE": 5,
        }.get(candidate_role, 5)

        return (
            0 if explicit_root else 1,
            candidate_role_rank,
            relation_rank,
            error_role_rank,
            step_index,
        )

    def _is_decisive_root_candidate(self, item: Dict[str, Any]) -> bool:
        local = item.get("local") or {}
        cascade = item.get("cascade") or {}
        error_role = normalize_error_role(item.get("error_role") or local.get("error_role"))
        candidate_role = normalize_candidate_role(item.get("candidate_role") or local.get("candidate_role"))
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
            or candidate_role == "INTRODUCER_CANDIDATE"
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
        for item in ranked:
            item["root_selection_rank_key"] = self._candidate_rank_key(item)

        if decisive:
            best = sorted(
                decisive,
                key=self._candidate_rank_key,
            )[0]
            ranked.sort(
                key=lambda item: (
                    0 if item is best else 1,
                    self._candidate_rank_key(item),
                )
            )
        else:
            ranked.sort(key=self._candidate_rank_key)
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
                    "candidate_role": item.get("candidate_role"),
                    "candidate_source": item.get("candidate_source"),
                    "is_soft_candidate": item.get("is_soft_candidate"),
                    "producer_for_node_id": item.get("producer_for_node_id"),
                    "stage1_error_role": item.get("error_role"),
                    "stage1_is_root_candidate": item.get("is_root_candidate"),
                    "stage1_decisive_root_candidate": self._is_decisive_root_candidate(item),
                    "stage1_rank_key": item.get("root_selection_rank_key"),
                    "stage1_reason": truncate_for_prompt(item.get("error_reason"), 700),
                    "local_diagnosis": {
                        "is_error": local.get("is_error"),
                        "error_role": local.get("error_role"),
                        "candidate_role": local.get("candidate_role"),
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

        final_node = self.find_final_evidence_node() or self.find_final_node() or {}
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
        cascade_evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = {
            "query": self.query,
            "ground_truth": self.ground_truth,
            "final_node": self.find_final_evidence_node() or self.find_final_node(),
        }
        if cascade_evidence is not None:
            context["cascade_evidence"] = cascade_evidence
        
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

        return self._normalize_local_evaluation(current_node, local_eval)

    def _normalize_local_evaluation(
        self,
        current_node: Dict[str, Any],
        local_eval: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        local_eval = dict(local_eval or {})
        local_eval["is_error"] = bool(local_eval.get("is_error", False))
        local_eval["error_role"] = normalize_error_role(local_eval.get("error_role"))
        local_eval["candidate_role"] = normalize_candidate_role(local_eval.get("candidate_role"))
        local_eval["is_root_candidate"] = bool(local_eval.get("is_root_candidate", False))
        if local_eval["candidate_role"] == "NO_RELEVANT_EVIDENCE":
            local_eval["candidate_role"] = candidate_role_from_error_role(
                local_eval["error_role"],
                current_node,
            )
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

    def _inspect_current_node_base(
        self,
        current_node: Dict[str, Any],
        predecessor_nodes: List[Dict[str, Any]],
        successor_nodes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        context = {
            "query": self.query,
            "ground_truth": self.ground_truth,
            "final_node": self.find_final_evidence_node() or self.find_final_node(),
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
            local_eval = self.llm_service.evaluate_base_step(
                current_node=current_node,
                predecessors=predecessor_nodes,
                context=context,
            )

        local_eval = dict(local_eval or {})
        local_eval["is_error"] = bool(local_eval.get("is_error", False))
        local_eval.setdefault("reason", "")

        self.local_diagnoses_by_node[str(current_node["node_id"])] = local_eval
        return local_eval

    def _inspect_current_node_with_cascade(
        self,
        current_node: Dict[str, Any],
        predecessor_nodes: List[Dict[str, Any]],
        successor_nodes: Optional[List[Dict[str, Any]]],
        active_causes: List[FailureCause],
    ) -> Tuple[Dict[str, Any], List[FailureCause], Dict[str, Any]]:
        if self.combined_cascade_enabled:
            local_result, cascade_eval = self.llm_service.evaluate_local_step_with_cascade(
                current_node=current_node,
                predecessors=predecessor_nodes,
                successors=successor_nodes or [],
                active_causes=active_causes,
                context={
                    "query": self.query,
                    "ground_truth": self.ground_truth,
                    "final_node": self.find_final_evidence_node() or self.find_final_node(),
                },
            )
            local_eval = self._normalize_local_evaluation(current_node, local_result)
        elif self.separate_order == "cascade_then_local":
            cascade_eval = self.llm_service.cascade_attribution_evaluator(
                current_node=current_node,
                predecessors=predecessor_nodes,
                local_eval=None,
                active_causes=active_causes,
                context={"query": self.query, "ground_truth": self.ground_truth},
            )
            local_eval = self._inspect_current_node_local(
                current_node=current_node,
                predecessor_nodes=predecessor_nodes,
                successor_nodes=successor_nodes or [],
                cascade_evidence=cascade_eval,
            )
        else:
            
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
                self.find_final_evidence_node() or self.find_final_node() or current_node,
                self.ground_truth,
            )
            if self.final_failure_analysis is None:
                self.final_failure_analysis = fallback_analysis

        relation = str(cascade_eval.get("relation", "UNCERTAIN")).upper()
        confidence = float(cascade_eval.get("confidence", 0.5))
        evidence = cascade_eval.get("evidence")
        local_error_role = normalize_error_role(local_eval.get("error_role"))
        is_root_candidate = bool(cascade_eval.get("is_root_candidate", False))
        per_cause_relations = cascade_eval.get("cause_relations")
        use_per_cause_relations = bool(
            self.combined_cascade_enabled and isinstance(per_cause_relations, list)
        )
        per_cause_by_id = {
            str(item.get("cause_id")): item
            for item in per_cause_relations or []
            if isinstance(item, dict) and item.get("cause_id") is not None
        }
        if local_error_role in {
            "NO_ERROR",
            "PROPAGATES_ERROR",
            "VERIFIES_PRIOR_ERROR",
            "TERMINATES_AFTER_ERROR",
            "UNCERTAIN",
        }:
            is_root_candidate = False
            if not use_per_cause_relations:
                cascade_eval["is_root_candidate"] = False
        elif relation == "INTRODUCES" and confidence >= 0.65:
            is_root_candidate = True
            if not use_per_cause_relations:
                cascade_eval["is_root_candidate"] = True

        next_causes: List[FailureCause] = []
        any_cause_root_candidate = False

        for cause in updated_causes:
            updated = FailureCause(**cause.model_dump())
            cause_eval = per_cause_by_id.get(str(updated.cause_id)) if use_per_cause_relations else None
            if use_per_cause_relations:
                cause_relation = str(
                    cause_eval.get("relation", "UNCERTAIN") if cause_eval else "UNCERTAIN"
                ).upper()
                cause_confidence_value = cause_eval.get("confidence", 0.0) if cause_eval else 0.0
                cause_evidence = cause_eval.get("evidence") if cause_eval else None
                cause_explanation = (
                    cause_eval.get("explanation", "No per-cause attribution was returned.")
                    if cause_eval
                    else "No per-cause attribution was returned."
                )
                cause_is_root_candidate = (
                    bool(cause_eval.get("is_root_candidate", False)) if cause_eval else False
                )
            else:
                cause_relation = relation
                cause_confidence_value = confidence
                cause_evidence = evidence
                cause_explanation = cascade_eval.get("explanation", "")
                cause_is_root_candidate = is_root_candidate
            try:
                cause_confidence = max(
                    0.0,
                    min(
                        1.0,
                        float(cause_confidence_value),
                    ),
                )
            except (TypeError, ValueError):
                cause_confidence = 0.0 if use_per_cause_relations else confidence
            if local_error_role in {
                "NO_ERROR",
                "PROPAGATES_ERROR",
                "VERIFIES_PRIOR_ERROR",
                "TERMINATES_AFTER_ERROR",
                "UNCERTAIN",
            }:
                cause_is_root_candidate = False
            elif cause_relation != "INTRODUCES":
                cause_is_root_candidate = False
            elif cause_relation == "INTRODUCES" and cause_confidence >= 0.65:
                cause_is_root_candidate = True
            if cause_eval is not None:
                cause_eval["is_root_candidate"] = cause_is_root_candidate

            any_cause_root_candidate = any_cause_root_candidate or cause_is_root_candidate
            updated.confidence = max(updated.confidence, cause_confidence)

            if cause_relation in {"INTRODUCES", "PROPAGATES", "AMPLIFIES"}:
                updated.support_status = "SUPPORTED"
                updated.should_propagate_upstream = True

                if cause_relation == "INTRODUCES":
                    updated.causal_status = "PROXIMATE_CAUSE"
                elif _causal_rank(updated.causal_status) < _causal_rank("PROXIMATE_CAUSE"):
                    updated.causal_status = "PROXIMATE_CAUSE"

                if updated.localized_node_id is None:
                    updated.localized_node_id = current_id
                if cause_evidence and not updated.evidence:
                    updated.evidence = cause_evidence

            elif cause_relation == "RECOVERS":
                updated.support_status = "SUPPORTED"
                updated.should_propagate_upstream = True

            elif cause_relation == "REFUTES":
                updated.support_status = "REFUTED"
                updated.should_propagate_upstream = False

            elif cause_relation == "UNRELATED":
                if updated.support_status != "SUPPORTED":
                    updated.support_status = "HYPOTHESIS"
                updated.should_propagate_upstream = True

            else:  # UNCERTAIN
                updated.should_propagate_upstream = True

            if cause_is_root_candidate and updated.support_status != "REFUTED":
                updated.causal_status = "ROOT_CANDIDATE"
                updated.support_status = "SUPPORTED"
                updated.should_propagate_upstream = True

            updated = _normalize_cause_state(updated)
            next_causes.append(updated)

            self.cascade_relations.append({
                "node_id": current_id,
                "step_index": current_node.get("step_index"),
                "cause_id": updated.cause_id,
                "relation": cause_relation,
                "explanation": cause_explanation,
                "confidence": cause_confidence,
                "is_root_candidate": cause_is_root_candidate,
                "is_root_cause": False,
                "evidence": cause_evidence,
            })

        if use_per_cause_relations:
            cascade_eval["is_root_candidate"] = any_cause_root_candidate

        return local_eval, next_causes, cascade_eval

    def _make_candidate_item(
        self,
        node_id: str,
        node: Dict[str, Any],
        local_eval: Optional[Dict[str, Any]],
        cascade_eval: Optional[Dict[str, Any]],
        *,
        candidate_role: str,
        candidate_source: str,
        error_role: Optional[str] = None,
        error_reason: Optional[str] = None,
        is_root_candidate: bool = False,
        is_soft_candidate: bool = False,
        producer_for_node_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        local_eval = dict(local_eval or {})
        role = normalize_error_role(error_role or local_eval.get("error_role"))
        c_role = normalize_candidate_role(candidate_role)
        return {
            "node_id": node_id,
            "role": node.get("role"),
            "name": node.get("name"),
            "agent_id": canonical_agent_id(node),
            "step_index": node.get("step_index"),
            "error_role": role,
            "candidate_role": c_role,
            "candidate_source": candidate_source,
            "is_root_candidate": bool(is_root_candidate),
            "is_soft_candidate": bool(is_soft_candidate),
            "producer_for_node_id": producer_for_node_id,
            "error_reason": error_reason or local_eval.get("reason", ""),
            "node": node,
            "local": local_eval,
            "cascade": cascade_eval,
        }

    def _merge_candidate_item(
        self,
        candidates_by_node: Dict[str, Dict[str, Any]],
        item: Dict[str, Any],
    ) -> None:
        node_id = str(item.get("node_id"))
        if not node_id:
            return
        existing = candidates_by_node.get(node_id)
        if existing is None:
            candidates_by_node[node_id] = item
            return

        role_rank = {
            "INTRODUCER_CANDIDATE": 0,
            "UPSTREAM_PRODUCER": 1,
            "PROPAGATION_EVIDENCE": 2,
            "VERIFICATION_EVIDENCE": 3,
            "EXPOSURE_EVIDENCE": 4,
            "NO_RELEVANT_EVIDENCE": 5,
        }
        err_rank = {
            "INTRODUCES_ERROR": 0,
            "PROPAGATES_ERROR": 1,
            "UNCERTAIN": 2,
            "VERIFIES_PRIOR_ERROR": 3,
            "TERMINATES_AFTER_ERROR": 4,
            "NO_ERROR": 5,
        }

        if role_rank.get(item.get("candidate_role"), 5) < role_rank.get(existing.get("candidate_role"), 5):
            existing["candidate_role"] = item.get("candidate_role")
        if err_rank.get(item.get("error_role"), 5) < err_rank.get(existing.get("error_role"), 5):
            existing["error_role"] = item.get("error_role")
        existing["is_root_candidate"] = bool(existing.get("is_root_candidate") or item.get("is_root_candidate"))
        existing["is_soft_candidate"] = bool(existing.get("is_soft_candidate") and item.get("is_soft_candidate"))
        sources = {
            s
            for s in str(existing.get("candidate_source", "")).split("+")
            if s
        } | {
            s
            for s in str(item.get("candidate_source", "")).split("+")
            if s
        }
        existing["candidate_source"] = "+".join(sorted(sources))
        if not existing.get("producer_for_node_id") and item.get("producer_for_node_id"):
            existing["producer_for_node_id"] = item.get("producer_for_node_id")
        if item.get("error_reason") and len(str(item.get("error_reason"))) > len(str(existing.get("error_reason", ""))):
            existing["error_reason"] = item.get("error_reason")

    def _add_upstream_producer_candidates(
        self,
        evidence_node_id: str,
        candidates_by_node: Dict[str, Dict[str, Any]],
        max_producers: int = 4,
    ) -> None:
        added = 0
        for pred_id, edge_kind in self._get_sorted_predecessors(evidence_node_id):
            if added >= max_producers:
                break
            pred_node = self.nodes_map.get(pred_id)
            if not pred_node or not is_reasoning_node(pred_node):
                continue

            pred_local = self.local_diagnoses_by_node.get(pred_id, {})
            pred_cascade = self.cascade_diagnoses_by_node.get(pred_id)
            reason = (
                "Graph-expanded upstream producer candidate: this reasoning node "
                f"precedes failure evidence at {evidence_node_id} via {edge_kind}."
            )
            item = self._make_candidate_item(
                node_id=pred_id,
                node=pred_node,
                local_eval=pred_local,
                cascade_eval=pred_cascade,
                candidate_role="UPSTREAM_PRODUCER",
                candidate_source="GRAPH_UPSTREAM_PRODUCER",
                error_role=pred_local.get("error_role") or "UNCERTAIN",
                error_reason=reason,
                is_root_candidate=False,
                is_soft_candidate=True,
                producer_for_node_id=evidence_node_id,
            )
            self._merge_candidate_item(candidates_by_node, item)
            added += 1

    # -----------------------------------------------------------------
    # Iterative Non-Recursive DFS Implementation
    # -----------------------------------------------------------------
    def _dfs_backtrack_iterative(
        self,
        start_node_id: str,
        traversal_path: List[str],
        error_chain: List[Dict[str, Any]],
        initial_active_causes: List[FailureCause],
        traversal_mode: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], List[FailureCause], List[Dict[str, Any]]]:
        mode = resolve_traversal_mode(
            traversal_mode or self.traversal_mode,
            default=self.traversal_mode,
        )
        frontier: List[Tuple[str, List[FailureCause]]] = [
            (start_node_id, clone_causes(initial_active_causes))
        ]
        candidates_by_node: Dict[str, Dict[str, Any]] = {}

        while frontier:
            node_id, current_causes = pop_frontier(frontier, mode)
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
            candidate_role = normalize_candidate_role(eval_result.get("candidate_role"))
            local_root_candidate = bool(eval_result.get("is_root_candidate", False))
            cascade_root_candidate = bool((cascade_eval or {}).get("is_root_candidate", False))
            cascade_relation = str((cascade_eval or {}).get("relation", "")).upper()

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

            should_add_direct_candidate = (
                is_error
                or candidate_role != "NO_RELEVANT_EVIDENCE"
                or cascade_root_candidate
                or cascade_relation in {"INTRODUCES", "PROPAGATES", "AMPLIFIES"}
                or node_id == self.final_evidence_node_id
            )

            if should_add_direct_candidate:
                if candidate_role == "NO_RELEVANT_EVIDENCE":
                    if node_id == self.final_evidence_node_id or is_exposure_node(current_node):
                        candidate_role = "EXPOSURE_EVIDENCE"
                    elif cascade_relation == "INTRODUCES":
                        candidate_role = "INTRODUCER_CANDIDATE"
                    elif cascade_relation in {"PROPAGATES", "AMPLIFIES"}:
                        candidate_role = "PROPAGATION_EVIDENCE"

                direct_error_role = error_role
                if cascade_root_candidate and cascade_relation == "INTRODUCES":
                    direct_error_role = "INTRODUCES_ERROR"

                direct_reason = error_reason
                if cascade_root_candidate and (cascade_eval or {}).get("explanation"):
                    direct_reason = (cascade_eval or {}).get("explanation", "")

                direct_item = self._make_candidate_item(
                    node_id=node_id,
                    node=current_node,
                    local_eval=eval_result,
                    cascade_eval=cascade_eval,
                    candidate_role=candidate_role,
                    candidate_source="LOCAL_OR_CASCADE_EVIDENCE",
                    error_role=direct_error_role,
                    error_reason=direct_reason,
                    is_root_candidate=local_root_candidate or cascade_root_candidate,
                    is_soft_candidate=False,
                )
                self._merge_candidate_item(candidates_by_node, direct_item)

                if should_expand_upstream_from_candidate(candidate_role, direct_error_role, current_node):
                    self._add_upstream_producer_candidates(
                        evidence_node_id=node_id,
                        candidates_by_node=candidates_by_node,
                    )

            pending_predecessors = [
                (next_pred_id, clone_causes(updated_active_causes))
                for next_pred_id, _kind in predecessors
                if next_pred_id not in self.visited
            ]
            push_frontier(frontier, pending_predecessors, mode)

        for cid, item in candidates_by_node.items():
            refreshed_local = self.local_diagnoses_by_node.get(cid)
            refreshed_cascade = self.cascade_diagnoses_by_node.get(cid)
            if refreshed_local:
                item["local"] = refreshed_local
                local_role = normalize_error_role(refreshed_local.get("error_role"))
                local_candidate_role = normalize_candidate_role(refreshed_local.get("candidate_role"))
                if local_role == "INTRODUCES_ERROR":
                    item["error_role"] = local_role
                    item["candidate_role"] = "INTRODUCER_CANDIDATE"
                    item["is_root_candidate"] = True
                    item["is_soft_candidate"] = False
                elif item.get("candidate_role") == "UPSTREAM_PRODUCER" and local_candidate_role != "NO_RELEVANT_EVIDENCE":
                    item["candidate_role"] = local_candidate_role
                    item["error_role"] = local_role
                    item["is_root_candidate"] = bool(refreshed_local.get("is_root_candidate"))
                    item["is_soft_candidate"] = False
                if refreshed_local.get("reason"):
                    item["error_reason"] = refreshed_local.get("reason")
            if refreshed_cascade:
                item["cascade"] = refreshed_cascade

        root_candidates = list(candidates_by_node.values())
        error_chain[:] = [
            {
                "node_id": item.get("node_id"),
                "role": item.get("role"),
                "name": item.get("name"),
                "agent_id": item.get("agent_id"),
                "step_index": item.get("step_index"),
                "error_role": item.get("error_role"),
                "candidate_role": item.get("candidate_role"),
                "candidate_source": item.get("candidate_source"),
                "is_root_candidate": item.get("is_root_candidate"),
                "is_soft_candidate": item.get("is_soft_candidate"),
                "producer_for_node_id": item.get("producer_for_node_id"),
                "error_reason": item.get("error_reason"),
            }
            for item in sorted(
                root_candidates,
                key=lambda x: int(x.get("step_index") or 0),
            )
        ]

        return self._select_root_cause_candidate(root_candidates)

    def _find_root_cause_base(
        self,
        terminal_node: Dict[str, Any],
        traversal_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        traversal_path: List[str] = []
        error_chain: List[Dict[str, Any]] = []
        mode = resolve_traversal_mode(
            traversal_mode or self.traversal_mode,
            default=self.traversal_mode,
        )
        frontier: List[str] = [str(terminal_node["node_id"])]

        while frontier:
            node_id = pop_frontier(frontier, mode)
            if node_id in self.visited or node_id not in self.nodes_map:
                continue

            self.visited.add(node_id)
            traversal_path.append(node_id)

            current_node = self.nodes_map[node_id]
            predecessors = self._get_sorted_predecessors(node_id)
            predecessor_nodes = [self.nodes_map[src_id] for src_id, _ in predecessors]

            local_eval = self._inspect_current_node_base(
                current_node=current_node,
                predecessor_nodes=predecessor_nodes,
            )

            self.diagnosis_history.append({
                "node_id": node_id,
                "step_index": current_node.get("step_index"),
                "agent_id": canonical_agent_id(current_node),
                "local": local_eval,
                "cascade": None,
                "active_causes_before": [],
                "active_causes_after": [],
            })

            if bool(local_eval.get("is_error", False)):
                error_chain.append({
                    "node_id": node_id,
                    "role": current_node.get("role"),
                    "name": current_node.get("name"),
                    "agent_id": canonical_agent_id(current_node),
                    "step_index": current_node.get("step_index"),
                    "is_error": True,
                    "error_reason": local_eval.get("reason", ""),
                })

            pending_predecessors = [
                next_pred_id
                for next_pred_id, _kind in predecessors
                if next_pred_id not in self.visited
            ]
            push_frontier(frontier, pending_predecessors, mode)

        ranked_root_candidates = []
        for rank, item in enumerate(
            sorted(error_chain, key=lambda x: int(x.get("step_index") or 0)),
            start=1,
        ):
            node_id = str(item["node_id"])
            ranked_root_candidates.append({
                **item,
                "node": self.nodes_map.get(node_id),
                "local": self.local_diagnoses_by_node.get(node_id, {}),
                "cascade": None,
                "candidate_source": "BASE_LOCAL_EVALUATOR",
                "root_selection_rank": rank,
            })

        root_cause_node = (
            ranked_root_candidates[0]["node"]
            if ranked_root_candidates
            else None
        )
        predicted_step = root_cause_node.get("step_index") if root_cause_node else None
        predicted_agent = canonical_agent_id(root_cause_node) if root_cause_node else None

        stage2_root_selection = {
            "enabled": False,
            "selection_mode": "BASE_LOCAL_ONLY",
            "selected_node_id": str(root_cause_node.get("node_id")) if root_cause_node else None,
        }
        cascade_failure = {
            "enabled": False,
            "final_failure_analysis": None,
            "terminal_node_id": str(terminal_node.get("node_id")) if terminal_node else None,
            "final_evidence_node_id": self.final_evidence_node_id,
            "active_final_causes": [],
            "cascade_relations": [],
            "root_candidate_nodes": [
                str(item["node_id"]) for item in ranked_root_candidates
            ],
            "stage2_root_selection": stage2_root_selection,
            "ranked_root_candidates": [
                {
                    "node_id": item.get("node_id"),
                    "step_index": item.get("step_index"),
                    "agent_id": item.get("agent_id"),
                    "candidate_source": item.get("candidate_source"),
                    "root_selection_rank": item.get("root_selection_rank"),
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
            "terminal_node_id": str(terminal_node.get("node_id")) if terminal_node else None,
            "final_evidence_node_id": self.final_evidence_node_id,
            "cascade_enabled": False,
            "diagnosis_mode": "BASE_LOCAL_ONLY",
            "active_final_causes": [],
            "final_failure_analysis": None,
            "active_causes_by_node": {},
            "cascade_relations": [],
            "root_cause_confirmation": [],
            "ranked_root_candidates": ranked_root_candidates,
            "stage2_root_selection": stage2_root_selection,
            "cascade_failure": cascade_failure,
            "root_candidate_nodes": [
                str(item["node_id"]) for item in ranked_root_candidates
            ],
            "local_diagnoses": self.local_diagnoses_by_node,
            "cascade_diagnoses": {},
            "diagnosis_history": self.diagnosis_history,
        }

    # -----------------------------------------------------------------
    # Public entry
    # -----------------------------------------------------------------
    def find_root_cause(
        self,
        query: Optional[Any] = None,
        ground_truth: Optional[Any] = None,
        traversal_mode: Optional[str] = None,
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
        self.final_evidence_node_id = None
        final_evidence_node = self.find_final_evidence_node() or terminal_node
        self.final_evidence_node_id = str(final_evidence_node.get("node_id")) if final_evidence_node else None
        mode = resolve_traversal_mode(
            traversal_mode or self.traversal_mode,
            default=self.traversal_mode,
        )

        if self.cascade_enabled:
            (
                self.active_final_causes,
                self.final_failure_analysis,
            ) = self.llm_service.initialize_active_final_causes(
                query=self.query,
                final_node=final_evidence_node,
                ground_truth=self.ground_truth,
            )
        elif not self.cascade_enabled:
            return self._find_root_cause_base(terminal_node, traversal_mode=mode)

        traversal_path: List[str] = []
        error_chain: List[Dict[str, Any]] = []

        # 执行非递归 DFS
        root_cause_node, final_active_causes, ranked_root_candidates = self._dfs_backtrack_iterative(
            start_node_id=str(terminal_node["node_id"]),
            traversal_path=traversal_path,
            error_chain=error_chain,
            initial_active_causes=clone_causes(self.active_final_causes),
            traversal_mode=mode,
        )

        stage2_root_selection = {
            "enabled": False,
            "selection_mode": "DISABLED",
            "selected_node_id": str(root_cause_node.get("node_id")) if root_cause_node else None,
        }
        if self.cascade_enabled and ranked_root_candidates:
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
            "combined_cascade_enabled": self.combined_cascade_enabled,
            "final_failure_analysis": self.final_failure_analysis,
            "terminal_node_id": str(terminal_node.get("node_id")) if terminal_node else None,
            "final_evidence_node_id": self.final_evidence_node_id,
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
                    "candidate_role": item.get("candidate_role"),
                    "candidate_source": item.get("candidate_source"),
                    "is_soft_candidate": item.get("is_soft_candidate"),
                    "producer_for_node_id": item.get("producer_for_node_id"),
                    "is_root_candidate": item.get("is_root_candidate"),
                    "root_selection_rank_key": item.get("root_selection_rank_key"),
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
            "terminal_node_id": str(terminal_node.get("node_id")) if terminal_node else None,
            "final_evidence_node_id": self.final_evidence_node_id,
            "cascade_enabled": self.cascade_enabled,
            "combined_cascade_enabled": self.combined_cascade_enabled,
            "diagnosis_mode": (
                "LOCAL_PLUS_COMBINED_CASCADE"
                if self.cascade_enabled and self.combined_cascade_enabled
                else "LOCAL_PLUS_CASCADE" if self.cascade_enabled else "LOCAL_ONLY"
            ),
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

# =============================================================================
# ADDITIVE EXTENSION: structured Combined diagnosis.
# The Original ZIP implementation above is preserved byte-for-byte.
# =============================================================================

# Combined strategy implementation (ported from the current Combined diagnosis implementation)
# The Original ZIP implementation above remains unchanged and is the Separate path.
# -----------------------------------------------------------------------------
import hashlib
import json
import re
from collections import deque
from typing import Any as _CombinedAny, Dict as _CombinedDict, List as _CombinedList, Optional as _CombinedOptional, Set as _CombinedSet, Callable as _CombinedCallable, Tuple as _CombinedTuple, Literal as _CombinedLiteral
from ...common.labels import clean_label as _combined_clean_label
from ...common.llm import LLMServer as _combined_LLMServer, parse_structured as _combined_parse_structured
from ...common.cascade import CASCADE_RELATION_SET as _combined_CASCADE_RELATION_SET, normalize_cascade_relation as _combined_normalize_cascade_relation
from ...common.traversal import resolve_traversal_mode as _combined_resolve_traversal_mode
from .schemas import (
    CombinedNodeInspection as _combined_CombinedNodeInspection,
    CascadeNodeInspection as _combined_CascadeNodeInspection,
    FailureCause as _combined_FailureCause,
    FinalFailureAnalysis as _combined_FinalFailureAnalysis,
    SingleNodeLocalInspection as _combined_SingleNodeLocalInspection,
    WhoWhenErrorItem as _combined_WhoWhenErrorItem,
)
from .evidence_audit import audit_quote as _combined_audit_quote
from .causal_state import (
    BranchState as _combined_BranchState,
    advance_branch as _combined_advance_branch,
    cause_record_at_node as _combined_cause_record_at_node,
    compatible_causes as _combined_compatible_causes,
    seed_branches as _combined_seed_branches,
    summarize_branches as _combined_summarize_branches,
)
from .dag_schedule import CausalGraphCycleError as _combined_CausalGraphCycleError, downstream_first_order as _combined_downstream_first_order
from .stage2_payload import build_candidate_coverage as _combined_build_candidate_coverage, select_cause_records_for_prompt as _combined_select_cause_records_for_prompt

# =====================================================================
# 1. Edge priority
# =====================================================================
_combined_EDGE_PRIORITY = {
    "execution_flow": 3,
    "sequential_flow": 2,
    "data_flow": 1,
}


def _combined_canonical_agent_id(node: Dict[str, Any]) -> str:
    agent_id = _combined_clean_label(node.get("agent_id"))
    if agent_id:
        return agent_id

    name = _combined_clean_label(node.get("name"))
    if name:
        return name

    role = _combined_clean_label(node.get("role"))
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


def _combined_is_tool_like_agent(agent_id: str) -> bool:
    agent = _combined_clean_label(agent_id).lower().replace(" ", "_")
    return agent in {
        "computer_terminal",
        "computerterminal",
        "terminal",
        "python",
        "python_tool",
        "browser",
        "tool",
    }

def _combined_is_tool_like_node(node: Dict[str, Any]) -> bool:
    if bool(node.get("is_tool_executor")):
        return True

    agent = _combined_canonical_agent_id(node)
    role = _combined_clean_label(node.get("role")).lower()
    name = _combined_clean_label(node.get("name")).lower()
    return (
        _combined_is_tool_like_agent(agent)
        or "computer_terminal" in name
        or "terminal" in name
        or role in {"tool", "observation"}
    )

def _combined_is_terminal_like_node(node: Dict[str, Any]) -> bool:
    if bool(node.get("is_terminal_node") or node.get("is_final_node")):
        return True

    text = _combined_node_text(node).strip().lower()
    semantic_role = _combined_clean_label(node.get("semantic_role")).lower()
    role = _combined_clean_label(node.get("role")).lower()
    return (
        semantic_role in {"final", "termination", "output"}
        or "termination" in role
        or text in {"terminate", "terminated"}
        or text.startswith("terminate")
    )

def _combined_is_user_like_node(node: Dict[str, Any]) -> bool:
    role = _combined_clean_label(node.get("role")).lower()
    agent = _combined_canonical_agent_id(node).lower()
    return role in {"human", "user"} or agent == "user"

def _combined_is_preliminary_or_delegation_like_node(node: Dict[str, Any]) -> bool:
    role = _combined_clean_label(node.get("role")).lower()
    agent = _combined_canonical_agent_id(node).lower()
    text = _combined_node_text(node).strip().lower()

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


def _combined_is_exposure_node(node: Dict[str, Any]) -> bool:
    return bool(
        node.get("is_tool_executor")
        or node.get("is_terminal_node")
        or node.get("is_final_node")
    )


def _combined_is_reasoning_node(node: Dict[str, Any]) -> bool:
    return _combined_clean_label(node.get("semantic_role")).lower() == "reasoning"


def _combined_is_terminal_marker_node(node: Dict[str, Any]) -> bool:
    text = _combined_node_text(node).strip().lower()
    return bool(node.get("is_terminal_node")) and (
        text in {"terminate", "terminated"} or text.startswith("terminate")
    )


def _combined_normalize_candidate_role(value: Any) -> str:
    role = str(value or "").strip().upper()
    allowed = {
        "INTRODUCER_CANDIDATE",
        "PROPAGATION_EVIDENCE",
        "VERIFICATION_EVIDENCE",
        "EXPOSURE_EVIDENCE",
        "UPSTREAM_PRODUCER",
        "NO_RELEVANT_EVIDENCE",
    }
    return role if role in allowed else "NO_RELEVANT_EVIDENCE"


def _combined_candidate_role_from_error_role(error_role: str, node: Dict[str, Any]) -> str:
    error_role = _combined_normalize_error_role(error_role)
    if error_role == "INTRODUCES_ERROR":
        return "INTRODUCER_CANDIDATE"
    if error_role == "PROPAGATES_ERROR":
        return "PROPAGATION_EVIDENCE"
    if error_role == "VERIFIES_PRIOR_ERROR":
        return "VERIFICATION_EVIDENCE"
    if error_role == "TERMINATES_AFTER_ERROR":
        return "EXPOSURE_EVIDENCE"
    if _combined_is_exposure_node(node):
        return "EXPOSURE_EVIDENCE"
    return "NO_RELEVANT_EVIDENCE"


def _combined_should_expand_upstream_from_candidate(candidate_role: str, error_role: str, node: Dict[str, Any]) -> bool:
    candidate_role = _combined_normalize_candidate_role(candidate_role)
    error_role = _combined_normalize_error_role(error_role)
    return (
        candidate_role in {
            "PROPAGATION_EVIDENCE",
            "VERIFICATION_EVIDENCE",
            "EXPOSURE_EVIDENCE",
        }
        or error_role in {
            "PROPAGATES_ERROR",
            "VERIFIES_PRIOR_ERROR",
            "TERMINATES_AFTER_ERROR",
        }
        or _combined_is_exposure_node(node)
    )


def _combined_normalize_error_role(value: Any) -> str:
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


def _combined_node_text(node: Dict[str, Any]) -> str:
    raw = node.get("metadata")
    if isinstance(raw, (dict, list)):
        return json.dumps(raw, ensure_ascii=False, default=str)
    return str(raw or "")


def _combined_truncate_for_prompt(value: Any, max_chars: int = 1200) -> str:
    text = (
        json.dumps(value, ensure_ascii=False, default=str)
        if isinstance(value, (dict, list))
        else str(value or "")
    )
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[TRUNCATED]...\n" + text[-half:]


def _combined__format_step_context_node(node: Dict[str, Any], limit: int) -> str:
    return (
        f"[node_id={node.get('node_id')} | Step {node.get('step_index', '?')} | "
        f"{_combined_canonical_agent_id(node)} (role: {_combined_clean_label(node.get('role')) or 'unknown'})] "
        f"{_combined_truncate_for_prompt(node.get('metadata'), limit)}"
    )


def _combined_extract_json_object_loose(text: str) -> Dict[str, Any]:
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


def _combined_loose_enum_field(text: str, field: str, allowed: Set[str], default: str) -> str:
    pattern = rf'"?{re.escape(field)}"?\s*:\s*"?([A-Za-z_]+)'
    match = re.search(pattern, text or "", re.IGNORECASE)
    if not match:
        return default
    value = match.group(1).strip().upper()
    return value if value in allowed else default


def _combined_loose_bool_field(text: str, field: str, default: bool = False) -> bool:
    pattern = rf'"?{re.escape(field)}"?\s*:\s*(true|false|yes|no|1|0)'
    match = re.search(pattern, text or "", re.IGNORECASE)
    if not match:
        return default
    return match.group(1).strip().lower() in {"true", "yes", "1"}


def _combined_loose_float_field(text: str, field: str, default: float = 0.5) -> float:
    pattern = rf'"?{re.escape(field)}"?\s*:\s*(-?\d+(?:\.\d+)?)'
    match = re.search(pattern, text or "")
    if not match:
        return default
    try:
        return max(0.0, min(1.0, float(match.group(1))))
    except ValueError:
        return default


def _combined_loose_string_field(text: str, field: str) -> Optional[str]:
    pattern = rf'"?{re.escape(field)}"?\s*:\s*(null|"([^"]*)"|([^,\n\r}}]+))'
    match = re.search(pattern, text or "", re.IGNORECASE)
    if not match:
        return None
    value = match.group(2) if match.group(2) is not None else match.group(3)
    value = _combined_clean_label(value)
    if not value or value.lower() in {"null", "none"}:
        return None
    return value


# =====================================================================
# 2. Cause state
# =====================================================================
def _combined__support_rank(status: str) -> int:
    return {
        "REFUTED": -1,
        "HYPOTHESIS": 0,
        "SUPPORTED": 1,
    }.get(status, 0)


def _combined__causal_rank(status: str) -> int:
    return {
        "UNKNOWN": 0,
        "PROXIMATE_CAUSE": 1,
        "ROOT_CANDIDATE": 2,
        "ROOT_CAUSE": 3,
    }.get(status, 0)


def _combined__normalize_cause_state(cause: _combined_FailureCause) -> _combined_FailureCause:
    c = _combined_FailureCause(**cause.model_dump())
    if c.support_status == "REFUTED":
        c.should_propagate_upstream = False
    if c.causal_status == "ROOT_CAUSE":
        c.should_propagate_upstream = False
    # A causal label is not independent evidence for support. In particular,
    # a model's premature ROOT_CAUSE label must not create SUPPORTED status.
    return c


def _combined__merge_cause_state(old: _combined_FailureCause, new: _combined_FailureCause) -> _combined_FailureCause:
    merged = _combined_FailureCause(**old.model_dump())
    # Compatibility projection only. The diagnostic traversal uses BranchState
    # and keeps every path fact; it never calls this lossy projection to decide
    # a root. Conflicts therefore project to unresolved, not ranked support.
    if old.support_status != new.support_status:
        merged.support_status = "HYPOTHESIS"
        merged.causal_status = "UNKNOWN"
        merged.localized_node_id = None
        merged.root_node_id = None
    elif old.causal_status != new.causal_status:
        merged.causal_status = "UNKNOWN"
    merged.should_propagate_upstream = (
        old.should_propagate_upstream or new.should_propagate_upstream
    )
    return _combined__normalize_cause_state(merged)


def _combined_merge_causes(
    existing: List[_combined_FailureCause],
    incoming: List[_combined_FailureCause],
) -> List[_combined_FailureCause]:
    by_id: Dict[str, _combined_FailureCause] = {}
    order: List[str] = []

    for cause in existing + incoming:
        c = _combined__normalize_cause_state(cause)
        if c.cause_id not in by_id:
            by_id[c.cause_id] = c
            order.append(c.cause_id)
        else:
            by_id[c.cause_id] = _combined__merge_cause_state(by_id[c.cause_id], c)

    return [by_id[cid] for cid in order]


def _combined_clone_causes(causes: List[_combined_FailureCause]) -> List[_combined_FailureCause]:
    return [_combined_FailureCause(**cause.model_dump()) for cause in causes]


def _combined_ground_truth_available(ground_truth: Any) -> bool:
    if ground_truth is None:
        return False
    if isinstance(ground_truth, str):
        return ground_truth.strip() not in {"", "N/A", "None", "null"}
    return True


def _combined_structured_cascade_to_legacy(
    inspection: _combined_CascadeNodeInspection,
    active_causes: List[_combined_FailureCause],
    current_node: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Summarize per-cause structured judgments for the existing traversal."""
    active_ids = {cause.cause_id for cause in active_causes}
    by_id = {}
    duplicate_ids = []
    unknown_ids = []
    for item in inspection.cause_relations:
        if item.cause_id not in active_ids:
            unknown_ids.append(item.cause_id)
        elif item.cause_id in by_id:
            duplicate_ids.append(item.cause_id)
        else:
            by_id[item.cause_id] = item
    relations = []
    for cause in active_causes:
        item = by_id.get(cause.cause_id)
        if item is None:
            relations.append({
                "cause_id": cause.cause_id,
                "relation": "UNCERTAIN",
                "explanation": "No per-cause attribution was returned.",
                "is_root_candidate": False,
                "confidence": 0.0,
                "evidence": None,
            })
        else:
            relation = item.model_dump()
            if current_node is not None:
                relation.update(_combined_audit_quote(item.evidence, current_node))
            relations.append(relation)

    if any(item["relation"] == "INTRODUCES" for item in relations):
        summary_relation = "INTRODUCES"
    elif relations and len({item["relation"] for item in relations}) == 1:
        summary_relation = relations[0]["relation"]
    else:
        summary_relation = "UNCERTAIN"
    sources = [item for item in relations if item["relation"] == summary_relation] or relations
    return {
        "relation": summary_relation,
        "cause_relations": relations,
        "explanation": inspection.reasoning,
        "is_root_candidate": any(item["is_root_candidate"] for item in relations),
        "confidence": max((item["confidence"] for item in sources), default=inspection.confidence),
        "evidence": next((item["evidence"] for item in sources if item["evidence"]), None),
        "status": inspection.status,
        "cause_updates": [item.model_dump() for item in inspection.cause_updates],
        "stopped_cause_ids": inspection.stopped_cause_ids,
        "json_parse_recovered": any(cause.cause_id not in by_id for cause in active_causes),
        "relation_audit": {
            "duplicate_cause_ids": sorted(set(duplicate_ids)),
            "unknown_cause_ids": sorted(set(unknown_ids)),
            "missing_cause_ids": sorted(active_ids - set(by_id)),
        },
    }


def _combined_structured_local_to_legacy(
    inspection: _combined_SingleNodeLocalInspection,
    current_node: Dict[str, Any],
) -> Dict[str, Any]:
    """Translate a structured local inspection to the traversal's legacy shape."""
    node_id = str(current_node.get("node_id", ""))
    role = _combined_normalize_error_role(inspection.error_role)
    reported_status = inspection.status
    has_error = bool(inspection.has_error)
    errors = []
    conflicts = []
    for error in inspection.errors:
        if error.location not in (None, "", node_id):
            conflicts.append("error_location_mismatch")
        item = error.model_dump()
        item["location"] = node_id
        item.update(_combined_audit_quote(error.evidence, current_node))
        errors.append(item)
    if has_error and not errors:
        conflicts.append("has_error_without_items")
    if errors and not has_error:
        conflicts.append("error_items_without_has_error")
    if inspection.is_root_candidate and (not has_error or role != "INTRODUCES_ERROR"):
        conflicts.append("root_candidate_without_local_introducer")
    if has_error and role == "NO_ERROR":
        conflicts.append("has_error_with_no_error_role")
        role = "UNCERTAIN"
    if has_error:
        if role not in {
            "INTRODUCES_ERROR",
            "PROPAGATES_ERROR",
            "VERIFIES_PRIOR_ERROR",
            "TERMINATES_AFTER_ERROR",
        }:
            role = (
                "INTRODUCES_ERROR"
                if reported_status == "ROOT_CAUSE"
                else "UNCERTAIN"
            )
    elif reported_status == "CORRECT":
        role = "NO_ERROR"
    elif reported_status == "PASSIVE_ERROR":
        if role not in {
            "PROPAGATES_ERROR",
            "VERIFIES_PRIOR_ERROR",
            "TERMINATES_AFTER_ERROR",
        }:
            role = "PROPAGATES_ERROR"
    else:
        role = "UNCERTAIN"

    is_root_candidate = bool(
        inspection.is_root_candidate and has_error and errors
        and role == "INTRODUCES_ERROR"
    )
    if role == "INTRODUCES_ERROR":
        status = "ROOT_CAUSE"
    elif role in {
        "PROPAGATES_ERROR",
        "VERIFIES_PRIOR_ERROR",
        "TERMINATES_AFTER_ERROR",
    }:
        status = "PASSIVE_ERROR"
    elif role == "NO_ERROR":
        status = "CORRECT"
    else:
        status = "UNCERTAIN"

    candidate_role = _combined_normalize_candidate_role(inspection.candidate_role)
    if candidate_role == "NO_RELEVANT_EVIDENCE":
        candidate_role = _combined_candidate_role_from_error_role(role, current_node)
    if role == "INTRODUCES_ERROR":
        candidate_role = "INTRODUCER_CANDIDATE"
    elif role == "PROPAGATES_ERROR":
        candidate_role = "PROPAGATION_EVIDENCE"
    elif role == "VERIFIES_PRIOR_ERROR":
        candidate_role = "VERIFICATION_EVIDENCE"
    elif role == "TERMINATES_AFTER_ERROR":
        candidate_role = "EXPOSURE_EVIDENCE"
    elif role == "NO_ERROR" and not _combined_is_exposure_node(current_node):
        candidate_role = "NO_RELEVANT_EVIDENCE"

    evidence = [item["evidence"] for item in errors if item.get("evidence")]
    reason = inspection.reasoning
    if has_error and role == "INTRODUCES_ERROR" and not evidence and not str(reason).strip():
        conflicts.append("introduces_without_explanation_or_evidence")

    return {
        "is_error": has_error,
        "error_role": role,
        "is_root_candidate": is_root_candidate,
        "candidate_role": candidate_role,
        "reason": reason,
        "status": status,
        "reported_status": reported_status,
        "schema_conflict": status != reported_status or bool(conflicts),
        "confidence": inspection.confidence,
        "evidence": evidence,
        "errors": errors,
        "evidence_audit": {
            "reported_error_count": len(inspection.errors),
            "exact_visible_count": sum(item["evidence_match_status"] == "EXACT" for item in errors),
            "unverified_count": sum(item["evidence_match_status"] != "EXACT" for item in errors),
            "conflicts": sorted(set(conflicts)),
            "semantic_error_judgment_preserved": has_error,
        },
    }


def _combined_structured_combined_to_legacy(
    inspection: _combined_CombinedNodeInspection,
    current_node: Dict[str, Any],
    active_causes: List[_combined_FailureCause],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Translate a combined structured response to existing traversal objects."""
    local_errors = [
        _combined_WhoWhenErrorItem(
            location=str(current_node.get("node_id", "")),
            evidence=error.evidence,
            description=error.description,
        )
        for error in inspection.local.errors
    ]
    local_model = _combined_SingleNodeLocalInspection(
        has_error=inspection.local.has_error,
        errors=local_errors,
        status=inspection.local.status,
        confidence=inspection.local.confidence,
        reasoning=inspection.local.reasoning,
        error_role=inspection.local.error_role,
        is_root_candidate=inspection.local.is_root_candidate,
        candidate_role=inspection.local.candidate_role,
    )
    local_eval = _combined_structured_local_to_legacy(local_model, current_node)

    active_ids = {cause.cause_id for cause in active_causes}
    relations_by_id = {}
    duplicate_relation_ids = []
    unknown_relation_ids = []
    for item in inspection.cascade.cause_relations:
        if item.cause_id not in active_ids:
            unknown_relation_ids.append(item.cause_id)
        elif item.cause_id in relations_by_id:
            duplicate_relation_ids.append(item.cause_id)
        else:
            relations_by_id[item.cause_id] = item
    relation_models = []
    root_candidate_audit = []
    for cause in active_causes:
        item = relations_by_id.get(cause.cause_id)
        if item is None:
            relation = "UNCERTAIN"
            explanation = "No per-cause attribution was returned."
            confidence = 0.0
            evidence = None
            is_root_candidate = False
            root_candidate_audit.append({
                "cause_id": cause.cause_id,
                "raw_is_root_candidate": False,
                "normalized_relation": relation,
                "local_qualifies_as_introducer": bool(
                    local_eval["is_error"]
                    and local_eval["error_role"] == "INTRODUCES_ERROR"
                ),
                "cascade_evidence_present": False,
                "cascade_evidence_exact_substring": False,
                "adapter_kept_candidate": False,
                "rejected_reasons": ["cause_relation_missing"],
            })
        else:
            relation = _combined_normalize_cascade_relation(item.relation)
            explanation = item.explanation
            confidence = item.confidence
            evidence = item.evidence
            provenance = _combined_audit_quote(evidence, current_node)
            local_qualifies = bool(
                local_eval["is_error"]
                and local_eval["error_role"] == "INTRODUCES_ERROR"
                and local_eval["is_root_candidate"]
            )
            evidence_valid = provenance["evidence_match_status"] == "EXACT"
            has_rationale = bool(str(item.explanation).strip() or str(evidence or "").strip())
            is_root_candidate = bool(
                item.is_root_candidate
                and relation == "INTRODUCES"
                and local_qualifies
                and has_rationale
            )
            rejected_reasons = []
            if not item.is_root_candidate:
                rejected_reasons.append("model_did_not_mark_candidate")
            if relation != "INTRODUCES":
                rejected_reasons.append("relation_not_introduces")
            if not local_qualifies:
                rejected_reasons.append("local_not_evidenced_introducer")
            if not has_rationale:
                rejected_reasons.append("cascade_rationale_missing")
            root_candidate_audit.append({
                "cause_id": cause.cause_id,
                "raw_is_root_candidate": bool(item.is_root_candidate),
                "normalized_relation": relation,
                "local_qualifies_as_introducer": local_qualifies,
                "cascade_evidence_present": bool(item.evidence),
                "cascade_evidence_exact_substring": evidence_valid,
                "cascade_evidence_match_status": provenance["evidence_match_status"],
                "cascade_evidence_source": provenance["evidence_source"],
                "cascade_evidence_audit_reason": provenance["evidence_audit_reason"],
                "cascade_has_supporting_rationale": has_rationale,
                "adapter_kept_candidate": is_root_candidate,
                "rejected_reasons": rejected_reasons,
            })
        relation_models.append({
            "cause_id": cause.cause_id,
            "relation": relation,
            "explanation": explanation,
            "is_root_candidate": is_root_candidate,
            "confidence": confidence,
            "evidence": evidence,
            **_combined_audit_quote(evidence, current_node),
        })

    if any(item["relation"] == "INTRODUCES" for item in relation_models):
        summary_relation = "INTRODUCES"
    elif relation_models and len({item["relation"] for item in relation_models}) == 1:
        summary_relation = relation_models[0]["relation"]
    else:
        summary_relation = "UNCERTAIN"
    summary_sources = [
        item for item in relation_models if item["relation"] == summary_relation
    ] or relation_models
    cascade_eval = {
        "relation": summary_relation,
        "cause_relations": relation_models,
        "explanation": inspection.cascade.reasoning,
        "is_root_candidate": any(item["is_root_candidate"] for item in relation_models),
        "confidence": max(
            (item["confidence"] for item in summary_sources),
            default=inspection.cascade.confidence,
        ),
        "evidence": next(
            (item["evidence"] for item in summary_sources if item["evidence"]), None
        ),
        "status": inspection.cascade.status,
        "cause_updates": [],
        "stopped_cause_ids": [],
        "json_parse_recovered": any(
            cause.cause_id not in relations_by_id for cause in active_causes
        ),
        "adapter_audit": {
            "local_reported_error_count": len(inspection.local.errors),
            "local_exact_evidence_count": local_eval["evidence_audit"]["exact_visible_count"],
            "local_unverified_evidence_count": local_eval["evidence_audit"]["unverified_count"],
            "local_error_role": local_eval["error_role"],
            "local_is_error_after_adapter": local_eval["is_error"],
            "local_is_root_candidate_after_adapter": local_eval["is_root_candidate"],
            "cause_root_candidates": root_candidate_audit,
            "duplicate_relation_ids": sorted(set(duplicate_relation_ids)),
            "unknown_relation_ids": sorted(set(unknown_relation_ids)),
        },
    }
    return local_eval, cascade_eval


# =====================================================================
# 3. Unified LLM Service
# =====================================================================
class _combined_LLMService:

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        api_key: Optional[str] = None,
        http_client: Optional[Any] = None,
        client: Optional[Any] = None,
        llm_server: Optional[_combined_LLMServer] = None,
        max_causes: int = 8,
    ):
        if isinstance(max_causes, bool) or not isinstance(max_causes, int) or max_causes < 1:
            raise ValueError("max_causes must be a positive integer")
        self.model_name = model_name
        self.max_causes = max_causes
        self.stage1_errors: List[str] = []
        self.stage1_request_audit: List[Dict[str, Any]] = []
        self.llm_server = llm_server or _combined_LLMServer(
            api_key=api_key,
            client=client,
            http_client=http_client,
            validate_key_before_import=True,
            missing_key_message="OpenAI API Key is missing. Pass api_key or set OPENAI_API_KEY env.",
        )

    def _parse_stage1(self, prompt: str, response_model: Any, system: str) -> Any:
        schema = response_model.model_json_schema()
        schema_json = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        audit_item = {
            "request_index": len(self.stage1_request_audit),
            "model": self.model_name,
            "response_model": response_model.__name__,
            "schema_sha256": hashlib.sha256(schema_json.encode("utf-8")).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "system": system,
            "prompt": prompt,
            "status": "started",
        }
        self.stage1_request_audit.append(audit_item)
        try:
            response = _combined_parse_structured(
                self.llm_server,
                model=self.model_name,
                response_model=response_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Structured Stage 1 returned no parsed response")
            audit_item["status"] = "ok"
            audit_item["parsed_output"] = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
            return parsed
        except Exception as exc:
            audit_item["status"] = "error"
            audit_item["error"] = str(exc)
            raise

    @staticmethod
    def _structured_step_context(
        current_node: Dict[str, Any],
        predecessors: List[Dict[str, Any]],
        successors: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> str:
        final_node = context.get("final_node")
        return (
            f"USER QUERY:\n{_combined_truncate_for_prompt(context.get('query'), 1400)}\n\n"
            f"GROUND TRUTH / CORRECT ANSWER:\n{_combined_truncate_for_prompt(context.get('ground_truth'), 900)}\n\n"
            f"DOWNSTREAM BRANCH STATE SUMMARY:\n{_combined_truncate_for_prompt(context.get('downstream_branch_summary') or {}, 5000)}\n\n"
            f"FINAL OUTPUT / TERMINAL EVIDENCE:\n{_combined__format_step_context_node(final_node, 900) if isinstance(final_node, dict) else 'Not provided'}\n\n"
            f"DIRECT PREDECESSORS:\n{chr(10).join(_combined__format_step_context_node(node, 700) for node in predecessors) or 'None'}\n\n"
            f"IMMEDIATE SUCCESSORS:\n{chr(10).join(_combined__format_step_context_node(node, 700) for node in successors[:3]) or 'None'}\n\n"
            f"CURRENT TARGET STEP:\n{_combined__format_step_context_node(current_node, 1600)}"
        )

    def _normalize_initial_causes(
        self,
        raw_causes: Any,
        *,
        has_failure: bool,
        fallback_description: str,
    ) -> Tuple[List[_combined_FailureCause], List[Dict[str, Any]], bool]:
        """Conservatively canonicalize initial hypotheses; never certify model status."""
        audit: List[Dict[str, Any]] = []
        if not has_failure:
            if raw_causes:
                audit.append({"reason": "causes_ignored_without_failure", "count": len(raw_causes)})
            return [], audit, False
        if not isinstance(raw_causes, list):
            audit.append({"reason": "invalid_causes_container", "value_type": type(raw_causes).__name__})
            raw_causes = []

        kept: List[_combined_FailureCause] = []
        seen_ids: Set[str] = set()
        seen_descriptions: Set[Tuple[str, ...]] = set()
        for index, value in enumerate(raw_causes):
            raw = value.model_dump() if hasattr(value, "model_dump") else value
            if not isinstance(raw, dict):
                audit.append({"index": index, "reason": "invalid_cause", "action": "dropped"})
                continue
            model_id = str(raw.get("cause_id") or "").strip()
            if not model_id:
                audit.append({"index": index, "reason": "missing_model_id", "action": "renumbered"})
            elif model_id.casefold() in seen_ids:
                audit.append({"index": index, "reason": "duplicate_model_id", "model_cause_id": model_id, "action": "renumbered"})
            else:
                seen_ids.add(model_id.casefold())

            description = " ".join(str(raw.get("description") or "").split())
            # Only short noun phrases may be reordered. Reordering a causal
            # sentence can invert who caused what ("A precedes B" versus
            # "B precedes A"), so longer/variable-containing text keeps order.
            content_tokens = tuple(
                token for token in re.findall(r"\w+", description.casefold())
                if token not in {"a", "an", "the"}
            )
            tokens = (
                tuple(sorted(content_tokens))
                if len(content_tokens) == 2 and all(len(token) > 2 for token in content_tokens)
                else content_tokens
            )
            if not tokens:
                audit.append({"index": index, "reason": "empty_description", "action": "dropped"})
                continue
            if tokens in seen_descriptions:
                audit.append({"index": index, "reason": "duplicate_description", "action": "dropped"})
                continue
            seen_descriptions.add(tokens)
            if len(kept) >= self.max_causes:
                audit.append({"index": index, "reason": "max_causes_cap", "action": "dropped"})
                continue
            try:
                confidence = _combined_FailureCause(
                    cause_id=f"C{len(kept) + 1}",
                    description=description,
                    confidence=raw.get("confidence", 0.5),
                ).confidence
            except Exception as exc:
                audit.append({"index": index, "reason": "invalid_confidence", "detail": str(exc)[:200], "action": "dropped"})
                continue
            if raw.get("support_status") not in (None, "HYPOTHESIS") or raw.get("causal_status") not in (None, "UNKNOWN") or raw.get("root_node_id") or raw.get("localized_node_id"):
                audit.append({"index": index, "reason": "premature_causal_status_reset", "action": "reset_to_hypothesis_unknown"})
            if raw.get("evidence"):
                audit.append({"index": index, "reason": "unverified_initial_cause_quote", "action": "kept_in_final_analysis_only"})
            kept.append(_combined_FailureCause(
                cause_id=f"C{len(kept) + 1}",
                description=description,
                confidence=confidence,
                support_status="HYPOTHESIS",
                causal_status="UNKNOWN",
                should_propagate_upstream=True,
                localized_node_id=None,
                root_node_id=None,
                evidence=None,
            ))
            audit.append({"index": index, "reason": "accepted", "model_cause_id": model_id, "cause_id": kept[-1].cause_id})

        if kept:
            return kept, audit, False
        description = " ".join(str(fallback_description or "").split()) or "Potential final-answer discrepancy requiring backward diagnosis."
        fallback = _combined_FailureCause(
            cause_id="C1", description=description, confidence=0.0,
            support_status="HYPOTHESIS", causal_status="UNKNOWN",
        )
        audit.append({"reason": "fallback_hypothesis", "action": "added", "confidence": 0.0})
        return [fallback], audit, True

    def _initialize_active_final_causes_structured(
        self, query: Any, final_node: Dict[str, Any], ground_truth: Any
    ) -> Tuple[List[_combined_FailureCause], Dict[str, Any]]:
        gt_available = _combined_ground_truth_available(ground_truth)
        final_id = str(final_node.get("node_id", ""))
        prompt = (
            "This Who&When trace is annotated with a final-answer error. Compare the final "
            "answer with the user request and, when available, the reference answer. "
            f"Seed 0..{self.max_causes} distinct, testable causal hypotheses for backward inspection; "
            "usually 1 to 5 is enough. Do not add filler hypotheses. Use stable C1..Ck "
            "IDs and distinct mechanism descriptions. Report final-failure confidence "
            "separately from each hypothesis confidence. Do not assign a "
            "root node from final-answer mismatch alone. Use HYPOTHESIS/UNKNOWN and leave "
            "localized_node_id and root_node_id null unless this final step itself gives "
            "concrete evidence. A supported final-step cause is usually PROXIMATE_CAUSE "
            "and should continue upstream. If no failure is established, return no causes.\n\n"
            f"USER QUERY:\n{_combined_truncate_for_prompt(query, 1800)}\n\n"
            f"GROUND TRUTH AVAILABLE: {gt_available}\n"
            f"GROUND TRUTH:\n{_combined_truncate_for_prompt(ground_truth, 1200)}\n\n"
            f"FINAL STEP node_id={final_id}:\n{_combined_truncate_for_prompt(final_node.get('metadata'), 2000)}"
        )
        try:
            parsed: _combined_FinalFailureAnalysis = self._parse_stage1(
                prompt,
                _combined_FinalFailureAnalysis,
                "Assess final-answer failure and return structured Who&When causal hypotheses.",
            )
            causes, cause_audit, fallback_hypothesis = self._normalize_initial_causes(
                parsed.causes,
                has_failure=parsed.has_failure,
                fallback_description=parsed.final_error,
            )
            failure_confidence = parsed.failure_confidence
            analysis = {
                "hypothesis": causes[0].description if causes else parsed.final_error,
                "assessment": "MISMATCH" if parsed.has_failure else "UNKNOWN",
                "confidence": failure_confidence,
                "failure_confidence": failure_confidence,
                "hypothesis_confidences": {cause.cause_id: cause.confidence for cause in causes},
                "hypothesis_count": len(causes),
                "dropped_count": sum(item.get("action") == "dropped" for item in cause_audit),
                "cause_audit": cause_audit,
                "fallback_hypothesis": fallback_hypothesis,
                "evidence": parsed.final_error,
                "ground_truth_used": gt_available,
                "ground_truth_available": gt_available,
                "has_failure": parsed.has_failure,
                "reasoning": parsed.reasoning,
                "causes": [cause.model_dump() for cause in causes],
                "schema_mode": "structured",
            }
            return causes, analysis
        except Exception as exc:
            self.stage1_errors.append(f"final: {exc}")
            print(f"[Structured Final Hypothesis Error] {exc}")
            causes, cause_audit, _ = self._normalize_initial_causes(
                [], has_failure=True,
                fallback_description="Potential final-answer discrepancy requiring backward diagnosis.",
            )
            return causes, {
                "hypothesis": causes[0].description,
                "assessment": "UNKNOWN",
                "confidence": 0.0,
                "failure_confidence": 0.0,
                "hypothesis_confidences": {"C1": 0.0},
                "hypothesis_count": 1,
                "dropped_count": 0,
                "cause_audit": cause_audit,
                "fallback_hypothesis": True,
                "evidence": None,
                "ground_truth_used": gt_available,
                "ground_truth_available": gt_available,
                "schema_mode": "structured",
                "error": str(exc),
            }

    def _evaluate_cascade_structured(
        self,
        current_node: Dict[str, Any],
        predecessors: List[Dict[str, Any]],
        local_eval: Dict[str, Any],
        active_causes: List[_combined_FailureCause],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = (
            "Assess the CURRENT TARGET STEP against EACH active downstream failure cause "
            "independently. Return exactly one cause_relations item for each supplied cause_id; "
            "use UNCERTAIN when evidence is insufficient. INTRODUCES means this step creates "
            "the mechanism; PROPAGATES carries an earlier error; AMPLIFIES worsens it; "
            "RECOVERS corrects it; UNRELATED is independent; REFUTES contradicts the cause. "
            "is_root_candidate may be true only for an INTRODUCES relation backed by this "
            "step's own evidence and a local INTRODUCES_ERROR. Evidence must be an exact "
            "substring of the current step. Keep cause_updates and stopped_cause_ids empty; "
            "the Who&When traversal derives those states from the relations for this trial.\n\n"
            f"LOCAL INSPECTION:\n{json.dumps(local_eval, ensure_ascii=False, default=str)}\n\n"
            f"ACTIVE CAUSES:\n{json.dumps([cause.model_dump() for cause in active_causes], ensure_ascii=False)}\n\n"
            + self._structured_step_context(current_node, predecessors, [], context)
        )
        try:
            parsed: _combined_CascadeNodeInspection = self._parse_stage1(
                prompt,
                _combined_CascadeNodeInspection,
                "Return structured per-cause attribution for a Who&When trace step.",
            )
            return _combined_structured_cascade_to_legacy(parsed, active_causes, current_node)
        except Exception as exc:
            self.stage1_errors.append(f"cascade step {current_node.get('step_index')}: {exc}")
            print(f"[Structured Cascade Evaluator Error] Step {current_node.get('step_index')}: {exc}")
            return {
                "relation": "UNCERTAIN",
                "cause_relations": [],
                "explanation": f"Structured cascade evaluation failed: {exc}",
                "is_root_candidate": False,
                "confidence": 0.0,
                "evidence": None,
                "json_parse_recovered": True,
            }
    def evaluate_cascade_per_cause(
        self,
        current_node: Dict[str, Any],
        predecessors: List[Dict[str, Any]],
        local_eval: Dict[str, Any],
        active_causes: List[_combined_FailureCause],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the structured one-relation-per-cause Cascade contract."""
        return self._evaluate_cascade_structured(
            current_node=current_node,
            predecessors=predecessors,
            local_eval=local_eval,
            active_causes=active_causes,
            context=context,
        )


    def _evaluate_local_step_structured(
        self,
        current_node: Dict[str, Any],
        predecessors: List[Dict[str, Any]],
        successors: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Inspect one node with a Pydantic response model."""
        prompt = (
            "Inspect only the CURRENT TARGET STEP for local evidence on the failure path. "
            "This is high-recall evidence collection; Stage 2 will compare candidates. "
            "Return concrete errors only when the current node itself contains supporting "
            "evidence, and quote each evidence value as an exact substring of the node text.\n\n"
            "Use INTRODUCES_ERROR for a concrete wrong decision, assumption, retrieval or "
            "navigation choice, code/method choice, data interpretation, or reasoning step. "
            "A responsible omission can count when the step had enough context to act or "
            "verify correctly. Use PROPAGATES_ERROR when the node repeats or uses an earlier "
            "mistake, VERIFIES_PRIOR_ERROR when it only endorses it, and "
            "TERMINATES_AFTER_ERROR when it only terminates after it. Use NO_ERROR for "
            "correct or neutral work and UNCERTAIN when evidence is insufficient.\n\n"
            "Do not blame a tool/action step for merely executing an earlier instruction. "
            "A retrieval or interaction choice can introduce an error when the choice itself "
            "is wrong. Do not infer a root cause from the final mismatch alone. "
            "is_root_candidate must be true only for a supported INTRODUCES_ERROR with "
            "exact local evidence. Candidate role must match the error role. A passive error "
            "is still an error on the path but is not a root candidate.\n\n"
            + self._structured_step_context(current_node, predecessors, successors, context)
        )
        try:
            parsed: _combined_SingleNodeLocalInspection = self._parse_stage1(
                prompt,
                _combined_SingleNodeLocalInspection,
                "Inspect this Who&When node and return a structured local error judgment.",
            )
            return _combined_structured_local_to_legacy(parsed, current_node)
        except Exception as exc:
            step = current_node.get("step_index")
            self.stage1_errors.append(f"local step {step}: {exc}")
            print(f"[Structured Local Evaluator Error] Step {step}: {exc}")
            return {
                "is_error": False,
                "error_role": "UNCERTAIN",
                "is_root_candidate": False,
                "candidate_role": "NO_RELEVANT_EVIDENCE",
                "reason": f"Structured local evaluation failed: {exc}",
                "status": "UNCERTAIN",
                "reported_status": "UNCERTAIN",
                "schema_conflict": False,
                "confidence": 0.0,
                "evidence": [],
                "errors": [],
            }

    def _evaluate_combined_structured(
        self,
        current_node: Dict[str, Any],
        predecessors: List[Dict[str, Any]],
        successors: List[Dict[str, Any]],
        active_causes: List[_combined_FailureCause],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Evaluate local error evidence and per-cause cascade in one parse call."""
        active_payload = [
            {
                "cause_id": cause.cause_id,
                "description": _combined_truncate_for_prompt(cause.description, 700),
                "confidence": cause.confidence,
                "support_status": cause.support_status,
                "causal_status": cause.causal_status,
                "should_propagate_upstream": cause.should_propagate_upstream,
                "localized_node_id": cause.localized_node_id,
                "evidence": cause.evidence,
            }
            for cause in active_causes
        ]
        prompt = (
            "Inspect the CURRENT TARGET STEP for a Who&When diagnosis. Return independent "
            "local and cascade judgments. The local result must be based on this step and "
            "trace context; do not let a cascade relation prove or suppress a local error.\n\n"
            "For local inspection, collect concrete evidence with exact substrings from the "
            "current node. INTRODUCES_ERROR means this step made the wrong decision, "
            "assumption, retrieval/navigation choice, code/method choice, data interpretation, "
            "or reasoning step. A responsible omission may count when there was enough context "
            "to act or verify correctly. PROPAGATES_ERROR repeats or uses an earlier mistake; "
            "VERIFIES_PRIOR_ERROR only endorses it; TERMINATES_AFTER_ERROR only terminates "
            "after it; NO_ERROR is correct or neutral; UNCERTAIN means evidence is insufficient. "
            "Do not blame a tool for merely executing an earlier instruction, and do not infer "
            "a root from the final mismatch alone. Root candidate must be true only for a local "
            "INTRODUCES_ERROR supported by exact evidence.\n\n"
            "For cascade, assess every supplied cause_id independently. Return one relation per "
            "cause: INTRODUCES, PROPAGATES, AMPLIFIES, RECOVERS, UNRELATED, REFUTES, or "
            "UNCERTAIN. Evidence must be an exact substring of the current node. A cascade "
            "root candidate is true only when the relation is INTRODUCES, its evidence is "
            "concrete, and the local judgment is INTRODUCES_ERROR. If there are no active "
            "causes, return an empty cause_relations list.\n\n"
            f"ACTIVE CAUSES:\n{json.dumps(active_payload, ensure_ascii=False, default=str)}\n\n"
            + self._structured_step_context(current_node, predecessors, successors, context)
        )
        try:
            parsed: _combined_CombinedNodeInspection = self._parse_stage1(
                prompt,
                _combined_CombinedNodeInspection,
                "Return structured local and per-cause cascade judgments for this Who&When node.",
            )
            return _combined_structured_combined_to_legacy(parsed, current_node, active_causes)
        except Exception as exc:
            step = current_node.get("step_index")
            self.stage1_errors.append(f"combined step {step}: {exc}")
            print(f"[Structured Combined Evaluator Error] Step {step}: {exc}")
            error = f"Structured combined evaluation failed: {exc}"
            return (
                {
                    "is_error": False,
                    "error_role": "UNCERTAIN",
                    "is_root_candidate": False,
                    "candidate_role": "NO_RELEVANT_EVIDENCE",
                    "reason": error,
                    "status": "UNCERTAIN",
                    "reported_status": "UNCERTAIN",
                    "schema_conflict": False,
                    "confidence": 0.0,
                    "evidence": [],
                    "errors": [],
                },
                {
                    "relation": "UNCERTAIN",
                    "cause_relations": [],
                    "explanation": error,
                    "is_root_candidate": False,
                    "confidence": 0.0,
                    "evidence": None,
                    "json_parse_recovered": True,
                },
            )

    def _initialize_active_final_causes_legacy(
        self,
        query: Any,
        final_node: Dict[str, Any],
        ground_truth: Any,
    ) -> Tuple[List[_combined_FailureCause], Dict[str, Any]]:
        final_step = final_node.get("step_index", "N/A")
        final_content = final_node.get("metadata")
        final_content_text = (
            json.dumps(final_content, ensure_ascii=False, default=str)
            if isinstance(final_content, (dict, list))
            else str(final_content)
        )

        gt_available = _combined_ground_truth_available(ground_truth)
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
            response = self.llm_server.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            reply_text = (response.choices[0].message.content or "").strip()

            parsed = _combined_extract_json_object_loose(reply_text)

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

            cause = _combined_FailureCause(
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
                _combined_FailureCause(
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

    def initialize_active_final_causes(
        self,
        query: Any,
        final_node: Dict[str, Any],
        ground_truth: Any,
    ) -> Tuple[List[_combined_FailureCause], Dict[str, Any]]:
        """Use the exact legacy C1 prompt for a one-cause run, otherwise seed multiple hypotheses.

        This keeps the original Notebook baseline reproducible with max_causes=1
        while allowing the current multi-cause experiment when max_causes > 1.
        """
        if self.max_causes == 1:
            return self._initialize_active_final_causes_legacy(
                query=query,
                final_node=final_node,
                ground_truth=ground_truth,
            )
        return self._initialize_active_final_causes_structured(
            query=query,
            final_node=final_node,
            ground_truth=ground_truth,
        )

    def cascade_attribution_evaluator(
        self,
        current_node: Dict[str, Any],
        predecessors: List[Dict[str, Any]],
        local_eval: Dict[str, Any],
        active_causes: List[_combined_FailureCause],
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
            pred_agent = _combined_canonical_agent_id(pred)
            pred_role = _combined_clean_label(pred.get("role")) or "unknown"
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
            response = self.llm_server.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a careful assistant analyzing causal dependencies."},
                    {"role": "user", "content": prompt},
                ],
            )
            reply_text = (response.choices[0].message.content or "").strip()

            parsed = _combined_extract_json_object_loose(reply_text)

            allowed_relations = {
                "INTRODUCES", "PROPAGATES", "AMPLIFIES", "RECOVERS", "UNRELATED", "REFUTES", "UNCERTAIN"
            }
            relation = str(
                parsed.get("relation")
                or _combined_loose_enum_field(reply_text, "relation", allowed_relations, "UNCERTAIN")
            ).upper()
            if relation not in {
                "INTRODUCES", "PROPAGATES", "AMPLIFIES", "RECOVERS", "UNRELATED", "REFUTES", "UNCERTAIN"
            }:
                relation = "UNCERTAIN"

            try:
                confidence = max(
                    0.0,
                    min(1.0, float(parsed.get("confidence", _combined_loose_float_field(reply_text, "confidence", 0.5)))),
                )
            except (TypeError, ValueError):
                confidence = 0.5

            evidence = parsed.get("evidence") or _combined_loose_string_field(reply_text, "evidence")
            if evidence in (None, "", "null"):
                evidence = None

            return {
                "relation": relation,
                "explanation": str(parsed.get("explanation", reply_text)),
                "is_root_candidate": bool(
                    parsed.get("is_root_candidate", _combined_loose_bool_field(reply_text, "is_root_candidate", False))
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
        candidate_coverage: Optional[Dict[str, Any]] = None,
        diagnosis_mode: str = "LOCAL_PLUS_CASCADE",
    ) -> Dict[str, Any]:
        final_payload = {
            "step_index": final_node.get("step_index"),
            "agent_id": _combined_canonical_agent_id(final_node),
            "content": _combined_truncate_for_prompt(final_node.get("metadata"), 900),
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
            f"USER QUERY:\n{_combined_truncate_for_prompt(query, 1400)}\n\n"
            f"GROUND TRUTH:\n{_combined_truncate_for_prompt(ground_truth, 900)}\n\n"
            f"FINAL NODE:\n{json.dumps(final_payload, ensure_ascii=False, default=str)}\n\n"
            f"FINAL FAILURE ANALYSIS:\n{json.dumps(final_failure_analysis or {}, ensure_ascii=False, default=str)}\n\n"
            "CANDIDATES:\n"
            f"{json.dumps(candidates, ensure_ascii=False, default=str)}"
        )

        try:
            response = self.llm_server.create(
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

            parsed = _combined_extract_json_object_loose(reply_text)

            selected_node_id = parsed.get("selected_node_id") or _combined_loose_string_field(reply_text, "selected_node_id")
            if selected_node_id in {"", "null", "None"}:
                selected_node_id = None

            try:
                selected_step = parsed.get("selected_step") or _combined_loose_string_field(reply_text, "selected_step")
                selected_step = int(selected_step) if selected_step is not None else None
            except (TypeError, ValueError):
                selected_step = None

            try:
                confidence = max(
                    0.0,
                    min(1.0, float(parsed.get("confidence", _combined_loose_float_field(reply_text, "confidence", 0.5)))),
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
        agent_name = _combined_canonical_agent_id(current_node)

        formatted_predecessors = []
        for pred in predecessors:
            pred_idx = pred.get("step_index", "?")
            pred_agent = _combined_canonical_agent_id(pred)
            pred_role = _combined_clean_label(pred.get("role")) or "unknown"
            pred_desc = f"{pred_agent} (role: {pred_role})" if pred_agent != "unknown" else pred_role
            pred_content = _combined_truncate_for_prompt(pred.get("metadata"), 700)
            formatted_predecessors.append(f"[Step {pred_idx} | {pred_desc}]: {pred_content}")

        predecessor_context = "\n".join(formatted_predecessors) or "None (Initial Step)"

        formatted_successors = []
        for succ in (successors or [])[:3]:
            succ_idx = succ.get("step_index", "?")
            succ_agent = _combined_canonical_agent_id(succ)
            succ_role = _combined_clean_label(succ.get("role")) or "unknown"
            succ_desc = f"{succ_agent} (role: {succ_role})" if succ_agent != "unknown" else succ_role
            succ_content = _combined_truncate_for_prompt(succ.get("metadata"), 700)
            formatted_successors.append(f"[Step {succ_idx} | {succ_desc}]: {succ_content}")

        successor_context = "\n".join(formatted_successors) or "None"

        if isinstance(final_node, dict):
            final_step = final_node.get("step_index", "N/A")
            final_agent = _combined_canonical_agent_id(final_node)
            final_content = _combined_truncate_for_prompt(final_node.get("metadata"), 900)
            final_context = f"[Step {final_step} | {final_agent}]: {final_content}"
        else:
            final_context = "Not provided"

        raw_metadata = current_node.get("metadata")
        curr_content = _combined_truncate_for_prompt(raw_metadata, 1600)
        display_agent = _combined_clean_label(agent_name) or _combined_clean_label(agent_role) or "agent"
        display_role = _combined_clean_label(agent_role)
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
            response = self.llm_server.create(
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
            error_role = _combined_normalize_error_role(role_match.group(1) if role_match else None)

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

    def evaluate_local_step_with_cascade(
        self,
        current_node: Dict[str, Any],
        predecessors: List[Dict[str, Any]],
        successors: List[Dict[str, Any]],
        active_causes: List[_combined_FailureCause],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Use the structured Stage 1 response contract."""
        return self._evaluate_combined_structured(current_node, predecessors, successors, active_causes, context)

    def evaluate_base_step(
        self,
        current_node: Dict[str, Any],
        predecessors: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use the structured Stage 1 response contract."""
        return self._evaluate_local_step_structured(current_node, predecessors, [], context)


# =====================================================================
# 4. Reverse Dependency DFS (Non-recursive / Iterative Stack)
# =====================================================================
class CombinedReverseDependencyDFS:
    def __init__(
        self,
        graph_data: Dict[str, Any],
        model_name: str = "gpt-5-mini",
        api_key: Optional[str] = None,
        openai_client: Optional[Any] = None,
        cascade_enabled: bool = False,
        custom_local_evaluator: Optional[Callable] = None,
        llm_server: Optional[_combined_LLMServer] = None,
        traversal_mode: str = "dfs",
        combined_cascade_enabled: bool = False,
        cascade_call_mode: Optional[str] = None,
        prompt_context_policy: str = "legacy",
        max_prompt_upstream_chars: int = 6000,
        max_causes: int = 8,
        max_stage2_candidates: int = 20,
        include_ground_truth: bool = True,
    ):
        # -------------------------------------------------------------
        # A. Load graph
        # -------------------------------------------------------------
        self.trace_id = graph_data.get("trace_id")
        self.query = graph_data.get("query")
        self.include_ground_truth = bool(include_ground_truth)
        self.ground_truth = graph_data.get("ground_truth") if self.include_ground_truth else None

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
        self.llm_service = _combined_LLMService(
            model_name=model_name,
            api_key=api_key,
            client=openai_client,
            llm_server=llm_server,
            max_causes=max_causes,
        )
        self.max_causes = self.llm_service.max_causes
        if (
            isinstance(max_stage2_candidates, bool)
            or not isinstance(max_stage2_candidates, int)
            or max_stage2_candidates < 1
        ):
            raise ValueError("max_stage2_candidates must be a positive integer")
        self.max_stage2_candidates = max_stage2_candidates
        self.custom_local_evaluator = custom_local_evaluator
        self.cascade_enabled = bool(cascade_enabled)
        if cascade_call_mode is None:
            cascade_call_mode = "combined" if combined_cascade_enabled else "separate"
        if cascade_call_mode not in {"combined", "separate"}:
            raise ValueError("cascade_call_mode must be 'combined' or 'separate'.")
        if prompt_context_policy not in {"legacy", "ranked_budget"}:
            raise ValueError("prompt_context_policy must be 'legacy' or 'ranked_budget'.")
        if max_prompt_upstream_chars < 0:
            raise ValueError("max_prompt_upstream_chars must be non-negative.")
        self.cascade_call_mode = cascade_call_mode
        self.combined_cascade_enabled = bool(
            cascade_call_mode == "combined" and custom_local_evaluator is None
        )
        if cascade_call_mode == "combined" and not self.combined_cascade_enabled:
            # A custom local evaluator cannot share the structured combined
            # response, so the effective execution path is two-stage.
            self.cascade_call_mode = "separate"
        self.prompt_context_policy = prompt_context_policy
        self.max_prompt_upstream_chars = max_prompt_upstream_chars
        self.traversal_mode = _combined_resolve_traversal_mode(traversal_mode, default="dfs")

        # -------------------------------------------------------------
        # C. Per-run state
        # -------------------------------------------------------------
        self.visited: Set[str] = set()
        self.active_final_causes: List[_combined_FailureCause] = []
        self.active_causes_by_node: Dict[str, List[_combined_FailureCause]] = {}
        self.branch_states_by_node: Dict[str, Dict[str, Any]] = {}
        self.branch_provenance_records: Dict[str, Dict[str, Any]] = {}
        self.cause_state_summaries_by_node: Dict[str, List[Dict[str, Any]]] = {}
        self.graph_diagnostics: Dict[str, Any] = {}
        self.stage1_candidate_coverage: Dict[str, Any] = {}
        self.upstream_producer_expansion: Dict[str, Any] = {}
        self.local_diagnoses_by_node: Dict[str, Dict[str, Any]] = {}
        self.cascade_diagnoses_by_node: Dict[str, Dict[str, Any]] = {}
        self.cascade_relations: List[Dict[str, Any]] = []
        self.diagnosis_history: List[Dict[str, Any]] = []
        self.final_failure_analysis: Optional[Dict[str, Any]] = None
        self.final_evidence_node_id: Optional[str] = None

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
                if _combined_EDGE_PRIORITY.get(kind, 0) > _combined_EDGE_PRIORITY.get(existing_kind, 0):
                    src_to_best_edge[src] = kind

        candidates = [
            (src_id, kind)
            for src_id, kind in src_to_best_edge.items()
            if src_id in self.nodes_map
        ]

        candidates.sort(
            key=lambda x: (
                _combined_EDGE_PRIORITY.get(x[1], 0),
                self.nodes_map[x[0]].get("step_index", -1),
            ),
            reverse=True,
        )
        return candidates

    def _select_prompt_predecessors(
        self,
        predecessor_nodes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Select prompt-only neighbors without changing graph traversal."""
        if self.prompt_context_policy == "legacy":
            return predecessor_nodes

        selected: List[Dict[str, Any]] = []
        used_chars = 0
        budget = max(0, self.max_prompt_upstream_chars)
        for node in predecessor_nodes:
            rendered = _combined__format_step_context_node(node, 700)
            added_chars = len(rendered) + (1 if selected else 0)
            if used_chars + added_chars <= budget:
                selected.append(node)
                used_chars += added_chars
        return selected

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
                if _combined_EDGE_PRIORITY.get(kind, 0) > _combined_EDGE_PRIORITY.get(existing_kind, 0):
                    dst_to_best_edge[dst] = kind

        candidates = [
            (dst_id, kind)
            for dst_id, kind in dst_to_best_edge.items()
            if dst_id in self.nodes_map
        ]

        candidates.sort(
            key=lambda x: (
                _combined_EDGE_PRIORITY.get(x[1], 0),
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
        resolved_gt = (
            self.ground_truth if ground_truth is None else ground_truth
        ) if self.include_ground_truth else None
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

    def find_final_evidence_node(self) -> Optional[Dict[str, Any]]:
        """Return the latest substantive node to use as final-failure evidence."""
        if not self.nodes_map:
            return None

        nodes_by_step_desc = sorted(
            self.nodes_map.values(),
            key=lambda n: n.get("step_index", 0),
            reverse=True,
        )

        for node in nodes_by_step_desc:
            if _combined_is_terminal_marker_node(node):
                continue
            text = _combined_node_text(node).strip()
            if text:
                return node

        return self.find_final_node()

    def _candidate_rank_key(
        self,
        item: Dict[str, Any],
        robust: bool = False,
    ) -> Tuple[int, ...]:
        node = item.get("node") or self.nodes_map.get(str(item.get("node_id")), {})
        local = item.get("local") or {}
        cascade = item.get("cascade") or {}

        relation = str(cascade.get("relation", "")).upper()
        error_role = _combined_normalize_error_role(item.get("error_role") or local.get("error_role"))
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

        node_type_rank = 1 if _combined_is_tool_like_node(node) or _combined_is_terminal_like_node(node) else 0

        if robust:
            if _combined_is_user_like_node(node):
                node_type_rank = 3
            elif _combined_is_tool_like_node(node) or _combined_is_terminal_like_node(node):
                node_type_rank = 2
            elif _combined_is_preliminary_or_delegation_like_node(node):
                node_type_rank = 1
            else:
                node_type_rank = 0

            return (
                0 if explicit_root else 1,
                relation_rank,
                error_role_rank,
                node_type_rank,
                1 if _combined_is_preliminary_or_delegation_like_node(node) else 0,
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
        error_role = _combined_normalize_error_role(item.get("error_role") or local.get("error_role"))
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
    ) -> Tuple[Optional[Dict[str, Any]], List[_combined_FailureCause], List[Dict[str, Any]]]:
        if not candidates:
            return None, [], []

        ranked = []
        for item in candidates:
            enriched = dict(item)
            ranked.append(enriched)

        decisive = [
            item for item in ranked
            if self._is_decisive_root_candidate(item)
            and _combined_normalize_error_role(item.get("error_role")) not in {
                "VERIFIES_PRIOR_ERROR",
                "TERMINATES_AFTER_ERROR",
                "NO_ERROR",
            }
        ]
        introducing_decisive_count = sum(
            1
            for item in decisive
            if _combined_normalize_error_role(item.get("error_role")) == "INTRODUCES_ERROR"
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
        causes = _combined_clone_causes(self.active_causes_by_node.get(node_id, []))
        return self.nodes_map.get(node_id), causes, ranked

    def _build_stage2_candidate_payload(
        self,
        candidates: List[Dict[str, Any]],
        max_candidates: Optional[int] = None,
        diagnosis_mode: str = "LOCAL_PLUS_CASCADE",
    ) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        limit = self.max_stage2_candidates if max_candidates is None else max_candidates

        for rank, item in enumerate(candidates[:limit], start=1):
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
                    "agent_id": _combined_canonical_agent_id(node),
                    "role": node.get("role"),
                    "semantic_role": node.get("semantic_role"),
                    "is_tool_like": _combined_is_tool_like_node(node),
                    "is_terminal_like": _combined_is_terminal_like_node(node),
                    "is_preliminary_or_delegation_like": _combined_is_preliminary_or_delegation_like_node(node),
                    "stage1_error_role": item.get("error_role"),
                    "stage1_is_root_candidate": item.get("is_root_candidate"),
                    "stage1_rank_key": item.get("root_selection_rank_key"),
                    "stage1_robust_rank": item.get("root_selection_robust_rank"),
                    "stage1_reason": _combined_truncate_for_prompt(item.get("error_reason"), 700),
                    "local_diagnosis": {
                        "is_error": local.get("is_error"),
                        "error_role": local.get("error_role"),
                        "is_root_candidate": local.get("is_root_candidate"),
                        "reason": _combined_truncate_for_prompt(local.get("reason"), 700),
                    },
                    "cascade_attribution": {
                        "relation": cascade.get("relation"),
                        "is_root_candidate": cascade.get("is_root_candidate"),
                        "confidence": cascade.get("confidence"),
                        "explanation": _combined_truncate_for_prompt(cascade.get("explanation"), 700),
                    },
                    "current_content": _combined_truncate_for_prompt(node.get("metadata"), 900),
                    "predecessor_context": [
                        {
                            "step_index": pred.get("step_index"),
                            "agent_id": _combined_canonical_agent_id(pred),
                            "role": pred.get("role"),
                            "content": _combined_truncate_for_prompt(pred.get("metadata"), 450),
                        }
                        for pred in predecessor_nodes
                    ],
                    "successor_context": [
                        {
                            "step_index": succ.get("step_index"),
                            "agent_id": _combined_canonical_agent_id(succ),
                            "role": succ.get("role"),
                            "content": _combined_truncate_for_prompt(succ.get("metadata"), 450),
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
        fallback_causes: List[_combined_FailureCause],
        diagnosis_mode: str = "LOCAL_PLUS_CASCADE",
    ) -> Tuple[Optional[Dict[str, Any]], List[_combined_FailureCause], Dict[str, Any]]:
        if len(ranked_candidates) < 2:
            coverage = _combined_build_candidate_coverage(ranked_candidates, [], self.max_stage2_candidates)
            coverage["policy"] = "SKIPPED_TOO_FEW_CANDIDATES"
            return fallback_node, fallback_causes, {
                "enabled": False,
                "selection_mode": "SKIPPED_TOO_FEW_CANDIDATES",
                "selected_node_id": str(fallback_node.get("node_id")) if fallback_node else None,
                "candidate_coverage": coverage,
            }

        candidate_payload = self._build_stage2_candidate_payload(
            ranked_candidates, diagnosis_mode=diagnosis_mode
        )
        per_candidate_record_coverage = []
        coverage = _combined_build_candidate_coverage(
            ranked_candidates,
            [item["node_id"] for item in candidate_payload],
            self.max_stage2_candidates,
        )
        candidate_ids = {item["node_id"] for item in candidate_payload}
        if len(candidate_payload) < 2:
            coverage = _combined_build_candidate_coverage(ranked_candidates, [], self.max_stage2_candidates)
            coverage["policy"] = "SKIPPED_TOO_FEW_VALID_CANDIDATES"
            return fallback_node, fallback_causes, {
                "enabled": False,
                "selection_mode": "SKIPPED_TOO_FEW_VALID_CANDIDATES",
                "selected_node_id": str(fallback_node.get("node_id")) if fallback_node else None,
                "candidate_coverage": coverage,
            }

        final_node = self.find_final_node() or {}
        selection = self.llm_service.select_root_from_candidates(
            query=self.query,
            ground_truth=self.ground_truth,
            final_node=final_node,
            final_failure_analysis=self.final_failure_analysis,
            candidates=candidate_payload,
            candidate_coverage=coverage,
            diagnosis_mode=diagnosis_mode,
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
                "candidate_coverage": coverage,
                "per_candidate_record_coverage": per_candidate_record_coverage,
                "candidate_node_ids": [item["node_id"] for item in candidate_payload],
                **selection,
            }

        selected_node = self.nodes_map[selected_id]
        selected_causes = _combined_clone_causes(self.active_causes_by_node.get(selected_id, fallback_causes))
        return selected_node, selected_causes, {
            "enabled": True,
            "candidate_count": len(candidate_payload),
            "candidate_coverage": coverage,
            "per_candidate_record_coverage": per_candidate_record_coverage,
            "candidate_node_ids": [item["node_id"] for item in candidate_payload],
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

        return self._normalize_local_evaluation(current_node, local_eval)

    def _normalize_local_evaluation(
        self,
        current_node: Dict[str, Any],
        local_eval: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Normalize Local results using the original Who&When role policy."""
        local_eval = dict(local_eval or {})
        local_eval["is_error"] = bool(local_eval.get("is_error", False))
        local_eval["error_role"] = _combined_normalize_error_role(local_eval.get("error_role"))
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

        local_eval["candidate_role"] = _combined_normalize_candidate_role(
            local_eval.get("candidate_role")
        )
        if local_eval["candidate_role"] == "NO_RELEVANT_EVIDENCE":
            local_eval["candidate_role"] = _combined_candidate_role_from_error_role(
                local_eval["error_role"], current_node
            )

        self.local_diagnoses_by_node[str(current_node["node_id"])] = local_eval
        return local_eval

    def _inspect_current_node_base(
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
            local_eval = self.llm_service.evaluate_base_step(
                current_node=current_node,
                predecessors=predecessor_nodes,
                context=context,
            )

        local_eval = dict(local_eval or {})
        local_eval["is_error"] = bool(local_eval.get("is_error", False))
        local_eval.setdefault("reason", "")

        self.local_diagnoses_by_node[str(current_node["node_id"])] = local_eval
        return local_eval

    def _inspect_current_node_with_cascade(
        self,
        current_node: Dict[str, Any],
        predecessor_nodes: List[Dict[str, Any]],
        successor_nodes: Optional[List[Dict[str, Any]]],
        active_causes: List[_combined_FailureCause],
        branch_paths: Optional[List[Tuple[str, ...]]] = None,
        downstream_branch_summary: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[_combined_FailureCause], Dict[str, Any]]:
        """Inspect one node while preserving cause identity and branch alignment.

        A DAG join can deliver multiple active BranchState objects for the same
        cause_id. The model is asked once per distinct cause, but the returned
        relation is applied independently to every concrete incoming branch.
        """
        # Build one prompt representative per cause_id. Prefer the most
        # informative current branch state; never replace it with the initial
        # seed hypothesis. The full per-branch state remains in the summary.
        prompt_causes_by_id: Dict[str, _combined_FailureCause] = {}
        ordered_active = sorted(
            active_causes,
            key=lambda cause: (
                str(cause.cause_id),
                cause.support_status != "SUPPORTED",
                cause.causal_status == "UNKNOWN",
                cause.localized_node_id is None,
                cause.root_node_id is None,
            ),
        )
        for cause in ordered_active:
            prompt_causes_by_id.setdefault(
                str(cause.cause_id),
                _combined_FailureCause(**cause.model_dump()),
            )
        prompt_causes = [prompt_causes_by_id[cid] for cid in sorted(prompt_causes_by_id)]

        context = {
            "query": self.query,
            "ground_truth": self.ground_truth,
            "final_node": self.find_final_node(),
            "downstream_branch_summary": downstream_branch_summary or {},
        }

        if self.combined_cascade_enabled:
            local_result, cascade_eval = self.llm_service.evaluate_local_step_with_cascade(
                current_node=current_node,
                predecessors=predecessor_nodes,
                successors=successor_nodes or [],
                active_causes=prompt_causes,
                context=context,
            )
            local_eval = self._normalize_local_evaluation(current_node, local_result)
        else:
            local_eval = self._inspect_current_node_local(
                current_node=current_node,
                predecessor_nodes=predecessor_nodes,
                successor_nodes=successor_nodes or [],
            )
            if len(prompt_causes) > 1:
                # The original single-Cause Cascade prompt is retained for the
                # legacy C1 baseline. Multiple causes require independent
                # cause_relations rather than one global relation copied to all.
                cascade_eval = self.llm_service.evaluate_cascade_per_cause(
                    current_node=current_node,
                    predecessors=predecessor_nodes,
                    local_eval=local_eval,
                    active_causes=prompt_causes,
                    context=context,
                )
            else:
                cascade_eval = self.llm_service.cascade_attribution_evaluator(
                    current_node=current_node,
                    predecessors=predecessor_nodes,
                    local_eval=local_eval,
                    active_causes=prompt_causes,
                    context=context,
                )

        current_id = str(current_node["node_id"])
        self.cascade_diagnoses_by_node[current_id] = cascade_eval

        # Keep the output list aligned with active_causes (one entry per
        # incoming branch), not prompt_causes (one entry per distinct cause).
        per_cause_relations = cascade_eval.get("cause_relations")
        use_per_cause_relations = isinstance(per_cause_relations, list)
        per_cause_by_id: Dict[str, Dict[str, Any]] = {}
        duplicate_ids = []
        for item in per_cause_relations or []:
            if not isinstance(item, dict) or item.get("cause_id") is None:
                continue
            cause_id = str(item["cause_id"])
            if cause_id in per_cause_by_id:
                duplicate_ids.append(cause_id)
            else:
                per_cause_by_id[cause_id] = item

        active_ids = {str(cause.cause_id) for cause in active_causes}
        cascade_eval.setdefault("relation_audit", {})
        cascade_eval["relation_audit"].update({
            "duplicate_cause_ids": sorted(set(duplicate_ids)),
            "unknown_cause_ids": sorted(set(per_cause_by_id) - active_ids),
            "missing_cause_ids": sorted(active_ids - set(per_cause_by_id)) if use_per_cause_relations else [],
            "prompt_unique_cause_count": len(prompt_causes),
            "active_branch_count": len(active_causes),
        })

        next_causes: List[_combined_FailureCause] = []
        any_root_candidate = False
        for index, cause in enumerate(active_causes):
            item = per_cause_by_id.get(str(cause.cause_id)) if use_per_cause_relations else cascade_eval
            if item is None:
                item = {
                    "relation": "UNCERTAIN",
                    "explanation": "No per-cause attribution was returned.",
                    "is_root_candidate": False,
                    "confidence": 0.0,
                    "evidence": None,
                }

            cause_relation = str(item.get("relation") or "UNCERTAIN").upper()
            if cause_relation not in _combined_CASCADE_RELATION_SET:
                cause_relation = "UNCERTAIN"
            explanation = str(item.get("explanation") or "")
            evidence = item.get("evidence")
            try:
                relation_confidence = max(
                    0.0,
                    min(1.0, float(item.get("confidence", 0.0))),
                )
            except (TypeError, ValueError):
                relation_confidence = 0.0

            model_root_candidate = bool(item.get("is_root_candidate", False))
            local_error_role = _combined_normalize_error_role(local_eval.get("error_role"))
            local_introducer = bool(
                local_eval.get("is_error")
                and local_error_role == "INTRODUCES_ERROR"
                and local_eval.get("is_root_candidate")
            )

            if self.combined_cascade_enabled or use_per_cause_relations:
                # Structured contracts require an explicit per-cause root
                # judgment, an INTRODUCES relation, and local introducer support.
                root_candidate = bool(
                    model_root_candidate
                    and cause_relation == "INTRODUCES"
                    and local_introducer
                    and (explanation.strip() or str(evidence or "").strip())
                )
            else:
                # Preserve the original Separate single-Cause policy: Local
                # non-introducer roles suppress Cascade root promotion; an
                # INTRODUCES relation with confidence >= .65 can promote it.
                root_candidate = model_root_candidate
                if local_error_role in {
                    "NO_ERROR", "PROPAGATES_ERROR", "VERIFIES_PRIOR_ERROR",
                    "TERMINATES_AFTER_ERROR", "UNCERTAIN",
                }:
                    root_candidate = False
                elif cause_relation == "INTRODUCES" and relation_confidence >= 0.65:
                    root_candidate = True
                elif cause_relation != "INTRODUCES":
                    root_candidate = False

            item["is_root_candidate"] = root_candidate
            any_root_candidate = any_root_candidate or root_candidate

            path = (
                branch_paths[index]
                if branch_paths is not None and index < len(branch_paths)
                else ()
            )
            initial_state = _combined_BranchState(
                cause=_combined_FailureCause(**cause.model_dump()),
                path=tuple(path),
                active=bool(
                    cause.should_propagate_upstream
                    and cause.support_status != "REFUTED"
                ),
                stop_reason=None if cause.should_propagate_upstream else "PRIOR_STOP",
            )
            advanced = _combined_advance_branch(
                initial_state,
                node_id=current_id,
                relation=cause_relation,
                explanation=explanation,
                relation_confidence=relation_confidence,
                evidence=evidence,
                root_candidate=root_candidate,
            )
            next_causes.append(advanced.cause)

            self.cascade_relations.append({
                "node_id": current_id,
                "step_index": current_node.get("step_index"),
                "cause_id": cause.cause_id,
                "path": list(advanced.path),
                "relation": cause_relation,
                "explanation": explanation,
                "confidence": relation_confidence,
                "hypothesis_confidence": cause.confidence,
                "is_root_candidate": root_candidate,
                "is_root_cause": False,
                "evidence": evidence,
                "support_status_after": advanced.cause.support_status,
                "active_after": advanced.active,
                "stop_reason": advanced.stop_reason,
                **_combined_audit_quote(evidence, current_node),
            })

        cascade_eval["is_root_candidate"] = any_root_candidate
        if not use_per_cause_relations:
            relation = str(cascade_eval.get("relation") or "UNCERTAIN").upper()
            if relation == "INTRODUCES" and not any_root_candidate:
                cascade_eval.setdefault("relation_audit", {})["root_not_promoted_from_confidence"] = True
        return local_eval, next_causes, cascade_eval

    def _make_candidate_item(
        self,
        node_id: str,
        node: Dict[str, Any],
        local_eval: Optional[Dict[str, Any]],
        cascade_eval: Optional[Dict[str, Any]],
        *,
        candidate_role: str,
        candidate_source: str,
        error_role: Optional[str] = None,
        error_reason: Optional[str] = None,
        is_root_candidate: bool = False,
        is_soft_candidate: bool = False,
        producer_for_node_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        local_eval = dict(local_eval or {})
        role = _combined_normalize_error_role(error_role or local_eval.get("error_role"))
        c_role = _combined_normalize_candidate_role(candidate_role)
        return {
            "node_id": node_id,
            "role": node.get("role"),
            "name": node.get("name"),
            "agent_id": _combined_canonical_agent_id(node),
            "step_index": node.get("step_index"),
            "error_role": role,
            "candidate_role": c_role,
            "candidate_source": candidate_source,
            "is_root_candidate": bool(is_root_candidate),
            "is_soft_candidate": bool(is_soft_candidate),
            "producer_for_node_id": producer_for_node_id,
            "error_reason": error_reason or local_eval.get("reason", ""),
            "node": node,
            "local": local_eval,
            "cascade": cascade_eval,
        }

    def _merge_candidate_item(
        self,
        candidates_by_node: Dict[str, Dict[str, Any]],
        item: Dict[str, Any],
    ) -> str:
        node_id = str(item.get("node_id"))
        if not node_id:
            return "skipped"
        existing = candidates_by_node.get(node_id)
        if existing is None:
            item.setdefault("producer_for_node_ids", [])
            item.setdefault("producer_source_evidence_node_ids", [])
            item.setdefault("producer_paths", [])
            item.setdefault("producer_edge_kinds", [])
            candidates_by_node[node_id] = item
            return "added"

        existing_sources = {
            source for source in str(existing.get("candidate_source") or "").split("+") if source
        }
        incoming_sources = {
            source for source in str(item.get("candidate_source") or "").split("+") if source
        }
        existing_is_direct = "GRAPH_UPSTREAM_PRODUCER" not in existing_sources
        incoming_is_direct = "GRAPH_UPSTREAM_PRODUCER" not in incoming_sources

        role_rank = {
            "INTRODUCER_CANDIDATE": 0,
            "PROPAGATION_EVIDENCE": 1,
            "UPSTREAM_PRODUCER": 2,
            "VERIFICATION_EVIDENCE": 3,
            "EXPOSURE_EVIDENCE": 4,
            "NO_RELEVANT_EVIDENCE": 5,
        }
        err_rank = {
            "INTRODUCES_ERROR": 0,
            "PROPAGATES_ERROR": 1,
            "UNCERTAIN": 2,
            "VERIFIES_PRIOR_ERROR": 3,
            "TERMINATES_AFTER_ERROR": 4,
            "NO_ERROR": 5,
        }

        # Graph expansion enriches a direct Local/Cascade candidate, but must
        # not replace its direct role, diagnosis, or evidence.
        if not (existing_is_direct and not incoming_is_direct):
            if role_rank.get(item.get("candidate_role"), 5) < role_rank.get(existing.get("candidate_role"), 5):
                existing["candidate_role"] = item.get("candidate_role")
            if err_rank.get(item.get("error_role"), 5) < err_rank.get(existing.get("error_role"), 5):
                existing["error_role"] = item.get("error_role")
        existing["is_root_candidate"] = bool(existing.get("is_root_candidate") or item.get("is_root_candidate"))
        existing["is_soft_candidate"] = bool(existing.get("is_soft_candidate") and item.get("is_soft_candidate"))
        existing["candidate_source"] = "+".join(sorted(existing_sources | incoming_sources))
        if not existing.get("producer_for_node_id") and item.get("producer_for_node_id"):
            existing["producer_for_node_id"] = item.get("producer_for_node_id")
        for field in ("producer_for_node_ids", "producer_source_evidence_node_ids", "producer_edge_kinds"):
            existing[field] = sorted({
                str(value)
                for value in list(existing.get(field) or []) + list(item.get(field) or [])
                if value not in (None, "")
            })
        merged_paths = list(existing.get("producer_paths") or []) + list(item.get("producer_paths") or [])
        unique_paths: Dict[str, Dict[str, Any]] = {}
        for path in merged_paths:
            if isinstance(path, dict):
                key = json.dumps(path, sort_keys=True, ensure_ascii=False, default=str)
                unique_paths.setdefault(key, path)
        existing["producer_paths"] = [unique_paths[key] for key in sorted(unique_paths)]
        if (
            not existing_is_direct
            and item.get("error_reason")
            and len(str(item.get("error_reason"))) > len(str(existing.get("error_reason", "")))
        ):
            existing["error_reason"] = item.get("error_reason")
        support_rank = {"EVIDENCE_SUPPORTED": 0, "LOCAL_ONLY": 1, "SOFT": 2}
        old_support = existing.get("candidate_support")
        new_support = item.get("candidate_support")
        if old_support is None or support_rank.get(new_support, 3) < support_rank.get(old_support, 3):
            existing["candidate_support"] = new_support
        return "merged"

    def _add_upstream_producer_candidates(
        self,
        evidence_node_id: str,
        candidates_by_node: Dict[str, Dict[str, Any]],
        max_producers: int = 4,
        max_hops: int = 3,
    ) -> Dict[str, Any]:
        """Find nearest inspected reasoning producers through dependency edges."""
        audit: Dict[str, Any] = {
            "evidence_node_id": evidence_node_id,
            "max_hops": max_hops,
            "max_producers": max_producers,
            "eligible": True,
            "producer_candidates_considered": [],
            "producer_candidates_newly_added": [],
            "existing_candidates_merged": [],
            "producer_candidates_skipped": [],
            "producer_paths": [],
        }
        if max_hops < 1 or max_producers < 1:
            audit["eligible"] = False
            audit["producer_candidates_skipped"].append({
                "reason": "invalid_expansion_limit",
                "max_hops": max_hops,
                "max_producers": max_producers,
            })
            return audit

        allowed_kinds = {"execution_flow", "data_flow"}
        queue = deque([(evidence_node_id, (evidence_node_id,), tuple())])
        seen_walks = set()
        producers: Dict[str, List[Dict[str, Any]]] = {}
        while queue:
            current_id, reverse_nodes, reverse_edges = queue.popleft()
            depth = len(reverse_edges)
            incoming = [
                edge for edge in self.reverse_adj.get(current_id, [])
                if str(edge.get("kind") or "").strip().lower() in allowed_kinds
            ]
            raw_incoming = self.reverse_adj.get(current_id, [])
            if raw_incoming and not incoming:
                audit["producer_candidates_skipped"].append({
                    "node_id": current_id,
                    "reason": "no_eligible_dependency_edge",
                    "excluded_edge_kinds": sorted({
                        str(edge.get("kind") or "").strip().lower()
                        for edge in raw_incoming
                    }),
                    "path_node_ids": list(reversed(reverse_nodes)),
                })
            incoming.sort(key=lambda edge: (
                -_combined_EDGE_PRIORITY.get(str(edge.get("kind") or "").strip().lower(), 0),
                -int(
                    self.nodes_map.get(str(edge.get("src")), {}).get("step_index")
                    if self.nodes_map.get(str(edge.get("src")), {}).get("step_index") is not None
                    else -1
                ),
                str(edge.get("src")),
                str(edge.get("kind") or "").lower(),
            ))
            if not incoming and depth == 0:
                audit["producer_candidates_skipped"].append({
                    "reason": "no_dependency_edge",
                    "edge_kinds_considered": sorted(allowed_kinds),
                })
            for edge in incoming:
                pred_id = str(edge.get("src"))
                kind = str(edge.get("kind") or "").strip().lower()
                if pred_id not in self.nodes_map:
                    audit["producer_candidates_skipped"].append({
                        "node_id": pred_id,
                        "reason": "missing_graph_node",
                        "edge": {"src": pred_id, "dst": current_id, "kind": kind},
                    })
                    continue
                if pred_id in reverse_nodes:
                    audit["producer_candidates_skipped"].append({
                        "node_id": pred_id,
                        "reason": "cycle_guard",
                        "path_node_ids": list(reverse_nodes),
                    })
                    continue
                next_nodes = reverse_nodes + (pred_id,)
                next_edges = reverse_edges + ({"src": pred_id, "dst": current_id, "kind": kind},)
                walk_key = (
                    pred_id,
                    next_nodes,
                    tuple((value["src"], value["dst"], value["kind"]) for value in next_edges),
                )
                if walk_key in seen_walks:
                    continue
                seen_walks.add(walk_key)
                next_depth = depth + 1
                pred_node = self.nodes_map[pred_id]
                if _combined_is_reasoning_node(pred_node):
                    record = {
                        "node_id": pred_id,
                        "evidence_node_id": evidence_node_id,
                        "hop_count": next_depth,
                        "path_node_ids": list(reversed(next_nodes)),
                        "edges": list(reversed(next_edges)),
                        "edge_kinds": list(dict.fromkeys(
                            path_edge["kind"] for path_edge in reversed(next_edges)
                        )),
                        "locally_inspected": pred_id in self.local_diagnoses_by_node,
                    }
                    audit["producer_candidates_considered"].append(record)
                    if pred_id not in self.local_diagnoses_by_node:
                        audit["producer_candidates_skipped"].append({
                            **record,
                            "reason": "producer_not_locally_inspected",
                        })
                        continue
                    if pred_id not in producers and len(producers) >= max_producers:
                        audit["producer_candidates_skipped"].append({
                            **record,
                            "reason": "per_evidence_producer_limit",
                        })
                        continue
                    path_key = json.dumps(
                        {key: record[key] for key in ("path_node_ids", "edges")},
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    paths = producers.setdefault(pred_id, [])
                    if all(
                        json.dumps({key: old[key] for key in ("path_node_ids", "edges")},
                                   sort_keys=True, ensure_ascii=False) != path_key
                        for old in paths
                    ):
                        paths.append(record)
                    # A reasoning Producer terminates this backward branch.
                    continue

                if next_depth < max_hops:
                    queue.append((pred_id, next_nodes, next_edges))
                else:
                    audit["producer_candidates_skipped"].append({
                        "node_id": pred_id,
                        "reason": "maximum_hops_reached_before_reasoning_producer",
                        "path_node_ids": list(reversed(next_nodes)),
                        "edges": list(reversed(next_edges)),
                    })

        for producer_id, paths in producers.items():
            producer_node = self.nodes_map[producer_id]
            local = self.local_diagnoses_by_node[producer_id]
            cascade = self.cascade_diagnoses_by_node.get(producer_id)
            actual_history = next(
                (row for row in self.diagnosis_history if str(row.get("node_id")) == producer_id),
                {},
            )
            item = self._make_candidate_item(
                node_id=producer_id,
                node=producer_node,
                local_eval=local,
                cascade_eval=cascade,
                candidate_role="UPSTREAM_PRODUCER",
                candidate_source="GRAPH_UPSTREAM_PRODUCER",
                error_role=local.get("error_role") or "UNCERTAIN",
                error_reason=(
                    "Graph-connected upstream reasoning candidate for failure evidence; "
                    "the relationship alone does not establish that this node is erroneous."
                ),
                is_root_candidate=False,
                is_soft_candidate=True,
            )
            item["candidate_support"] = "SOFT"
            item["cause_records"] = list(actual_history.get("cause_records") or [])
            item["producer_for_node_id"] = evidence_node_id
            item["producer_for_node_ids"] = [evidence_node_id]
            item["producer_source_evidence_node_ids"] = [evidence_node_id]
            item["producer_paths"] = sorted(paths, key=lambda path: (
                path["hop_count"], path["path_node_ids"], path["edge_kinds"]
            ))
            item["producer_edge_kinds"] = sorted({
                kind for path in paths for kind in path["edge_kinds"]
            })
            outcome = self._merge_candidate_item(candidates_by_node, item)
            record = {
                "node_id": producer_id,
                "evidence_node_id": evidence_node_id,
                "path_count": len(paths),
                "producer_paths": item["producer_paths"],
            }
            audit["producer_paths"].extend(item["producer_paths"])
            if outcome == "added":
                audit["producer_candidates_newly_added"].append(record)
            else:
                audit["existing_candidates_merged"].append(record)

        audit["unique_producer_count"] = len(producers)
        return audit

    def _expand_upstream_producer_candidates(
        self,
        candidates_by_node: Dict[str, Dict[str, Any]],
        *,
        max_hops: int = 3,
        max_producers_per_evidence: int = 4,
    ) -> Dict[str, Any]:
        """Apply the single shared Producer expansion stage to direct candidates."""
        summary: Dict[str, Any] = {
            "enabled": True,
            "eligible_evidence_nodes": [],
            "producer_candidates_considered": [],
            "producer_candidates_newly_added": [],
            "existing_candidates_merged": [],
            "producer_candidates_skipped": [],
            "producer_paths": [],
            "max_hops": max_hops,
            "max_producers_per_evidence_node": max_producers_per_evidence,
            "per_evidence": [],
        }
        direct_candidates = list(candidates_by_node.values())
        direct_candidate_count = len(direct_candidates)
        history_by_node = {
            str(row.get("node_id")): row
            for row in self.diagnosis_history
            if row.get("node_id") is not None
        }
        for row in self.diagnosis_history:
            audit_row = row.setdefault("candidate_audit", {})
            audit_row["graph_producer_expansion_enabled"] = True
            audit_row.setdefault("producer_expansion_eligible", False)
            audit_row.setdefault("producer_candidates_considered", [])
            audit_row.setdefault("producer_candidates_newly_added", [])
            audit_row.setdefault("existing_candidates_merged", [])
            audit_row.setdefault("producer_candidates_skipped", [])
            audit_row.setdefault("producer_paths", [])
            audit_row.setdefault("upstream_candidates_added", [])
        for candidate in direct_candidates:
            node_id = str(candidate.get("node_id"))
            candidate_role = _combined_normalize_candidate_role(candidate.get("candidate_role"))
            error_role = _combined_normalize_error_role(candidate.get("error_role"))
            direct_root = self._is_decisive_root_candidate(candidate)
            is_best_effort_fallback = (
                "BEST_EFFORT_NO_CANDIDATE" in str(candidate.get("candidate_source") or "")
            )
            eligible = bool(
                candidate_role not in {"INTRODUCER_CANDIDATE", "UPSTREAM_PRODUCER"}
                and not direct_root
                and not is_best_effort_fallback
                and _combined_should_expand_upstream_from_candidate(
                    candidate_role,
                    error_role,
                    self.nodes_map.get(node_id, {}),
                )
            )
            history_row = history_by_node.get(node_id)
            audit_row = history_row.get("candidate_audit") if history_row else None
            if not isinstance(audit_row, dict):
                audit_row = {}
                if history_row is not None:
                    history_row["candidate_audit"] = audit_row
            audit_row["graph_producer_expansion_enabled"] = True
            audit_row["producer_expansion_eligible"] = eligible
            audit_row.setdefault("producer_candidates_considered", [])
            audit_row.setdefault("producer_candidates_newly_added", [])
            audit_row.setdefault("existing_candidates_merged", [])
            audit_row.setdefault("producer_candidates_skipped", [])
            audit_row.setdefault("producer_paths", [])
            audit_row.setdefault("upstream_candidates_added", [])
            if not eligible:
                skipped = {
                    "evidence_node_id": node_id,
                    "reason": (
                        "best_effort_fallback_is_not_evidence"
                        if is_best_effort_fallback
                        else
                        "decisive_or_introducer_candidate"
                        if candidate_role == "INTRODUCER_CANDIDATE" or direct_root
                        else "producer_candidates_are_not_recursively_expanded"
                        if candidate_role == "UPSTREAM_PRODUCER"
                        else "candidate_not_eligible_evidence"
                    ),
                }
                summary["producer_candidates_skipped"].append(skipped)
                audit_row["producer_candidates_skipped"].append(skipped)
                continue

            summary["eligible_evidence_nodes"].append(node_id)
            per_evidence = self._add_upstream_producer_candidates(
                evidence_node_id=node_id,
                candidates_by_node=candidates_by_node,
                max_producers=max_producers_per_evidence,
                max_hops=max_hops,
            )
            summary["per_evidence"].append(per_evidence)
            for key in (
                "producer_candidates_considered",
                "producer_candidates_newly_added",
                "existing_candidates_merged",
                "producer_candidates_skipped",
                "producer_paths",
            ):
                summary[key].extend(per_evidence.get(key, []))
            audit_row["producer_candidates_considered"].extend(
                per_evidence.get("producer_candidates_considered", [])
            )
            audit_row["producer_candidates_newly_added"].extend(
                per_evidence.get("producer_candidates_newly_added", [])
            )
            audit_row["existing_candidates_merged"].extend(
                per_evidence.get("existing_candidates_merged", [])
            )
            audit_row["producer_candidates_skipped"].extend(
                per_evidence.get("producer_candidates_skipped", [])
            )
            audit_row["producer_paths"].extend(per_evidence.get("producer_paths", []))
            audit_row["upstream_candidates_added"].extend(
                record.get("node_id")
                for record in per_evidence.get("producer_candidates_newly_added", [])
                if record.get("node_id")
            )

        for row in self.diagnosis_history:
            row.setdefault("candidate_audit", {})["candidate_count_after_expansion"] = len(candidates_by_node)
        summary["unique_candidate_count_after_expansion"] = len(candidates_by_node)
        return summary

    def _finalize_stage1_candidate_pipeline(
        self,
        *,
        candidates_by_node: Dict[str, Dict[str, Any]],
        error_chain: List[Dict[str, Any]],
        coverage: Dict[str, Any],
        fallback_node_id: str,
        diagnosis_mode: str,
    ) -> Tuple[Optional[Dict[str, Any]], List[_combined_FailureCause], List[Dict[str, Any]]]:
        """Finalize only candidates admitted by the original Local/Cascade rules."""
        direct_candidate_count = len(candidates_by_node)
        coverage = dict(coverage)
        coverage["direct_candidate_nodes"] = direct_candidate_count
        coverage["candidate_nodes"] = direct_candidate_count
        producer_audit = {
            "enabled": False,
            "reason": "Disabled for the restored original Who&When candidate policy.",
            "eligible_evidence_nodes": [],
            "producer_candidates_considered": [],
            "producer_candidates_newly_added": [],
            "existing_candidates_merged": [],
            "producer_candidates_skipped": [],
            "producer_paths": [],
        }
        coverage["producer_expansion"] = producer_audit
        self.upstream_producer_expansion = producer_audit
        self.stage1_candidate_coverage = coverage
        return self._select_root_cause_candidate(list(candidates_by_node.values()))

    # -----------------------------------------------------------------
    # Iterative Non-Recursive DFS Implementation
    # -----------------------------------------------------------------
    def _dfs_backtrack_iterative(
        self,
        start_node_id: str,
        traversal_path: List[str],
        error_chain: List[Dict[str, Any]],
        initial_active_causes: List[_combined_FailureCause],
        traversal_mode: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], List[_combined_FailureCause], List[Dict[str, Any]]]:
        """Compatibility entry; Stage 1 uses a downstream-first DAG schedule."""
        from .causal_traversal import diagnose_dag

        return diagnose_dag(
            self, start_node_id, traversal_path, error_chain, initial_active_causes
        )

    def _find_root_cause_base(
        self,
        terminal_node: Dict[str, Any],
        traversal_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Backward-compatible entry into the shared Local-only DAG path."""
        del terminal_node
        cascade_enabled = self.cascade_enabled
        self.cascade_enabled = False
        try:
            return self.find_root_cause(
                query=self.query,
                ground_truth=self.ground_truth,
                traversal_mode=traversal_mode,
            )
        finally:
            self.cascade_enabled = cascade_enabled

    # -----------------------------------------------------------------
    # Public entry
    # -----------------------------------------------------------------
    def find_root_cause(
        self,
        query: Optional[Any] = None,
        ground_truth: Optional[Any] = None,
        traversal_mode: Optional[str] = None,
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
        self.branch_states_by_node.clear()
        self.branch_provenance_records.clear()
        self.cause_state_summaries_by_node.clear()
        self.graph_diagnostics.clear()
        self.stage1_candidate_coverage.clear()
        self.upstream_producer_expansion.clear()
        self.local_diagnoses_by_node.clear()
        self.cascade_diagnoses_by_node.clear()
        self.cascade_relations.clear()
        self.diagnosis_history.clear()
        self.llm_service.stage1_errors.clear()
        self.final_failure_analysis = None
        self.final_evidence_node_id = None
        final_evidence_node = terminal_node
        self.final_evidence_node_id = str(final_evidence_node.get("node_id")) if final_evidence_node else None
        mode = _combined_resolve_traversal_mode(
            traversal_mode or self.traversal_mode,
            default=self.traversal_mode,
        )
        try:
            _order, self.graph_diagnostics = _combined_downstream_first_order(
                self.nodes_map,
                self.reverse_adj,
                self.forward_adj,
                str(terminal_node["node_id"]),
            )
        except _combined_CausalGraphCycleError as exc:
            return {
                "status": "ERROR",
                "reason": str(exc),
                "trace_id": self.trace_id,
                "root_cause_found": False,
                "root_cause_node": None,
                "predicted_step": None,
                "predicted_agent": None,
                "graph_diagnostics": exc.diagnostics,
                "stage1_error_count": 0,
                "stage1_errors": [],
            }

        # Final Failure Analysis seeds the candidate-selection context in both
        # modes. In Local-only, these remain unverified hypotheses and are not
        # propagated through the graph.
        (
            self.active_final_causes,
            self.final_failure_analysis,
        ) = self.llm_service.initialize_active_final_causes(
            query=self.query,
            final_node=final_evidence_node,
            ground_truth=self.ground_truth,
        )

        traversal_path: List[str] = []
        error_chain: List[Dict[str, Any]] = []

        if self.cascade_enabled:
            # Full Method: downstream-first DAG processing retains all branch
            # facts at joins and applies Cause × Path transitions.
            root_cause_node, final_active_causes, ranked_root_candidates = self._dfs_backtrack_iterative(
                start_node_id=str(terminal_node["node_id"]),
                traversal_path=traversal_path,
                error_chain=error_chain,
                initial_active_causes=_combined_clone_causes(self.active_final_causes),
                traversal_mode=mode,
            )
            diagnosis_mode = (
                "LOCAL_PLUS_COMBINED_CASCADE"
                if self.combined_cascade_enabled else "LOCAL_PLUS_CASCADE"
            )
        else:
            from .causal_traversal import diagnose_local_only_dag

            root_cause_node, final_active_causes, ranked_root_candidates = diagnose_local_only_dag(
                self,
                start_node_id=str(terminal_node["node_id"]),
                traversal_path=traversal_path,
                error_chain=error_chain,
            )
            diagnosis_mode = "LOCAL_ONLY_WITH_STAGE2"

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
                diagnosis_mode=diagnosis_mode,
            )

        predicted_step = root_cause_node.get("step_index") if root_cause_node else None
        predicted_agent = _combined_canonical_agent_id(root_cause_node) if root_cause_node else None
        selected_node_id = str(root_cause_node["node_id"]) if root_cause_node else None
        selected_item = next(
            (item for item in ranked_root_candidates if str(item.get("node_id")) == selected_node_id),
            None,
        )
        selected_records = list((selected_item or {}).get("cause_records") or [])
        eligible = {
            str(item.get("cause_id"))
            for item in selected_records
            if isinstance(item, dict)
            and item.get("cause_id")
            and item.get("support_status") != "REFUTED"
            and item.get("relation") not in {"REFUTES", "NOT_EVALUATED_STOPPED"}
        }
        model_selected = stage2_root_selection.get("selected_cause_ids") or []
        model_ids = [str(cid) for cid in model_selected if str(cid) in eligible]
        if not self.cascade_enabled:
            selected_cause_ids = []
        elif model_ids:
            selected_cause_ids = sorted(set(model_ids))
        else:
            introducers = {
                str(item["cause_id"])
                for item in selected_records
                if isinstance(item, dict)
                and item.get("relation") == "INTRODUCES"
                and item.get("support_status") == "SUPPORTED"
            }
            selected_cause_ids = sorted(introducers or {
                str(item["cause_id"])
                for item in selected_records
                if isinstance(item, dict)
                and item.get("support_status") == "SUPPORTED"
                and item.get("relation") not in {"REFUTES", "NOT_EVALUATED_STOPPED"}
            })
        root_confirmation: List[Dict[str, Any]] = []
        root_prediction = {
            "node_id": selected_node_id,
            "step_index": predicted_step,
            "agent_id": predicted_agent,
            "diagnosis_mode": diagnosis_mode,
            "selection_mode": stage2_root_selection.get("selection_mode"),
            "selected_cause_ids": selected_cause_ids,
            "fallback_rationale": stage2_root_selection.get("fallback_reason"),
            "candidate_coverage": stage2_root_selection.get("candidate_coverage"),
            "evidence_status_unchanged_by_selection": True,
        }

        cascade_failure = {
            "enabled": self.cascade_enabled,
            "cascade_attribution_performed": self.cascade_enabled,
            "combined_cascade_enabled": self.combined_cascade_enabled,
            "cascade_call_mode": self.cascade_call_mode,
            "prompt_context_policy": self.prompt_context_policy,
            "max_prompt_upstream_chars": self.max_prompt_upstream_chars,
            "traversal_mode": mode,
            "final_failure_analysis": self.final_failure_analysis,
            "terminal_node_id": str(terminal_node.get("node_id")) if terminal_node else None,
            "final_evidence_node_id": self.final_evidence_node_id,
            "active_final_causes": (
                [c.model_dump() for c in self.active_final_causes]
                if self.cascade_enabled else []
            ),
            "initial_cause_hypotheses": [c.model_dump() for c in self.active_final_causes],
            "cascade_relations": self.cascade_relations,
            "branch_states_by_node": self.branch_states_by_node,
            "branch_provenance_records": self.branch_provenance_records,
            "cause_state_summaries_by_node": self.cause_state_summaries_by_node,
            "graph_diagnostics": self.graph_diagnostics,
            "stage1_candidate_coverage": self.stage1_candidate_coverage,
            "candidate_coverage": stage2_root_selection.get("candidate_coverage"),
            "root_prediction": root_prediction,
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
                    "candidate_role": item.get("candidate_role"),
                    "candidate_source": item.get("candidate_source"),
                    "is_soft_candidate": item.get("is_soft_candidate"),
                    "producer_for_node_id": item.get("producer_for_node_id"),
                    "producer_for_node_ids": item.get("producer_for_node_ids", []),
                    "producer_source_evidence_node_ids": item.get("producer_source_evidence_node_ids", []),
                    "producer_paths": item.get("producer_paths", []),
                    "producer_edge_kinds": item.get("producer_edge_kinds", []),
                    "is_root_candidate": item.get("is_root_candidate"),
                    "root_selection_rank_key": item.get("root_selection_rank_key"),
                    "error_reason": item.get("error_reason"),
                    "candidate_support": item.get("candidate_support"),
                    "cause_records": item.get("cause_records", []),
                }
                for item in ranked_root_candidates
            ],
        }

        return {
            "trace_id": self.trace_id,
            "stage1_error_count": len(self.llm_service.stage1_errors),
            "stage1_errors": list(self.llm_service.stage1_errors),
            "root_cause_found": root_cause_node is not None,
            "root_cause_node": root_cause_node,
            "predicted_step": predicted_step,
            "predicted_agent": predicted_agent,
            "root_prediction": root_prediction,
            "selection_mode": stage2_root_selection.get("selection_mode"),
            "selected_cause_ids": selected_cause_ids,
            "candidate_coverage": stage2_root_selection.get("candidate_coverage"),
            "error_chain": error_chain,
            "traversal_path": traversal_path,
            "query": self.query,
            "ground_truth": self.ground_truth,
            "terminal_node_id": str(terminal_node.get("node_id")) if terminal_node else None,
            "final_evidence_node_id": self.final_evidence_node_id,
            "cascade_enabled": self.cascade_enabled,
            "cascade_call_mode": self.cascade_call_mode,
            "combined_cascade_enabled": self.combined_cascade_enabled,
            "prompt_context_policy": self.prompt_context_policy,
            "max_prompt_upstream_chars": self.max_prompt_upstream_chars,
            "traversal_mode": mode,
            "diagnosis_mode": (
                diagnosis_mode
            ),
            "cascade_attribution_performed": self.cascade_enabled,
            "active_final_causes": (
                [c.model_dump() for c in self.active_final_causes]
                if self.cascade_enabled else []
            ),
            "initial_cause_hypotheses": [c.model_dump() for c in self.active_final_causes],
            "final_failure_analysis": self.final_failure_analysis,
            "active_causes_by_node": {
                node_id: [c.model_dump() for c in causes]
                for node_id, causes in self.active_causes_by_node.items()
            },
            "branch_states_by_node": self.branch_states_by_node,
            "branch_provenance_records": self.branch_provenance_records,
            "cause_state_summaries_by_node": self.cause_state_summaries_by_node,
            "graph_diagnostics": self.graph_diagnostics,
            "stage1_candidate_coverage": self.stage1_candidate_coverage,
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
            "stage1_request_audit": self.llm_service.stage1_request_audit,
        }


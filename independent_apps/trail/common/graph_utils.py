import json
import re
from difflib import SequenceMatcher
from typing import Any, Collection, Iterable, List, Optional, Set, Tuple


DEFAULT_STOP_ATOMS = {
    "true",
    "false",
    "null",
    "none",
    "ok",
    "success",
    "error",
    "message",
    "role",
    "content",
    "type",
    "json",
}


def normalize_edge_types(
    edge_types: Optional[Iterable[str]],
    *,
    supported: Iterable[str],
) -> Set[str]:
    supported_set = set(supported)
    if edge_types is None:
        return supported_set

    requested = {str(edge_type).strip().lower() for edge_type in edge_types}
    invalid = requested - supported_set
    if invalid:
        valid_text = ", ".join(sorted(supported_set))
        invalid_text = ", ".join(sorted(invalid))
        raise ValueError(f"Unsupported edge type(s): {invalid_text}. Choose from: {valid_text}")
    if not requested:
        raise ValueError("At least one edge type must be selected; omit edge_types to keep all edges.")
    return requested

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False)
        except Exception:
            text = str(text)

    text = re.sub(r"[#*`_>-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def extract_str_from_io(raw_val: Any) -> str:
    if raw_val is None:
        return ""

    if isinstance(raw_val, str):
        s = raw_val.strip()
        if s and s[0] in "{[":
            try:
                parsed = json.loads(s)
                return extract_str_from_io(parsed)
            except Exception:
                return s
        return s

    if isinstance(raw_val, dict):
        parts = []
        for k, v in raw_val.items():
            parts.append(str(k))
            parts.append(extract_str_from_io(v))
        return " ".join(parts)

    if isinstance(raw_val, list):
        return " ".join(extract_str_from_io(x) for x in raw_val)

    return str(raw_val)


def extract_atoms(
    raw_val: Any,
    *,
    stop_atoms: Collection[str] = DEFAULT_STOP_ATOMS,
) -> List[str]:
    atoms = []

    def walk(x):
        if x is None:
            return
        if isinstance(x, str):
            s = x.strip()
            if len(s) > 1 and s[0] in "{[":
                try:
                    parsed = json.loads(s)
                    walk(parsed)
                    return
                except Exception:
                    pass
            norm_str = normalize_text(s)
            if norm_str:
                atoms.append(norm_str)
        elif isinstance(x, dict):
            for k, v in x.items():
                norm_k = normalize_text(str(k))
                if norm_k and len(norm_k) > 3 and norm_k not in stop_atoms:
                    atoms.append(norm_k)
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for item in x:
                walk(item)
        else:
            norm_str = normalize_text(str(x))
            if norm_str:
                atoms.append(norm_str)

    walk(raw_val)
    return atoms


def is_data_flow_present(
    src_out_raw: Any,
    dst_in_raw: Any,
    *,
    stop_atoms: Collection[str] = DEFAULT_STOP_ATOMS,
    token_overlap_threshold: float = 0.5,
) -> Tuple[bool, float, str]:
    src_atoms = {
        a for a in extract_atoms(src_out_raw, stop_atoms=stop_atoms)
        if len(a) >= 3 and a.lower() not in stop_atoms
    }
    dst_atoms = {
        a for a in extract_atoms(dst_in_raw, stop_atoms=stop_atoms)
        if len(a) >= 3 and a.lower() not in stop_atoms
    }

    if not src_atoms or not dst_atoms:
        return False, 0.0, "none"

    exact = src_atoms & dst_atoms
    if exact:
        return True, 1.0, "exact_atom_match"

    src_text = normalize_text(extract_str_from_io(src_out_raw))
    dst_text = normalize_text(extract_str_from_io(dst_in_raw))
    if not src_text or not dst_text:
        return False, 0.0, "none"

    src_tokens_clean = {
        t for t in set(src_text.split()) if len(t) > 2 and t.lower() not in stop_atoms
    }
    dst_tokens_clean = {
        t for t in set(dst_text.split()) if len(t) > 2 and t.lower() not in stop_atoms
    }

    token_overlap = src_tokens_clean & dst_tokens_clean
    if token_overlap:
        score = len(token_overlap) / max(1, min(len(src_tokens_clean), len(dst_tokens_clean)))
        if score >= token_overlap_threshold:
            return True, round(score, 4), "token_overlap"

    if len(src_text) < 500 and len(dst_text) < 500:
        ratio = SequenceMatcher(None, src_text, dst_text).ratio()
        if ratio >= 0.8:
            return True, round(ratio, 4), "fuzzy_ratio"

    return False, 0.0, "none"

"""Strict + loose graders for Task B (deal extraction -> JSON record).

Mirrors Task A's grader API (extract_.../format_valid/grade/grade_loose) but for a
five-field JSON record: company, round, raise, valuation, founders. The locked
contract is in TASKB_PREREG.md. Both graders normalize key order, founder order, and
number formats (correct parsing, NOT lenience); strict vs loose differ ONLY in the
per-field match criterion. The score is per-field: the fraction of the 5 fields that
match (0.0-1.0). All comparison logic is pure and unit-testable without a model.
"""

from __future__ import annotations

import difflib
import json
import re
import string

EXPECTED_KEYS = ("company", "round", "raise", "valuation", "founders")


# --------------------------------------------------------------------------- #
# JSON extraction (handles code fences, leading/trailing prose, nested braces). #
# --------------------------------------------------------------------------- #
def _balanced_span_from(text, start):
    """Return the balanced {...} span beginning at `start` (string-aware), or None."""
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def extract_json(text):
    """Pull the first parseable JSON object out of free-form model output, or None."""
    if not text or not isinstance(text, str):
        return None
    i = 0
    while True:
        start = text.find("{", i)
        if start == -1:
            return None
        span = _balanced_span_from(text, start)
        if span is None:
            return None  # unclosed brace from here on
        try:
            obj = json.loads(span)
        except (json.JSONDecodeError, ValueError):
            obj = None
        if isinstance(obj, dict):
            return obj
        i = start + len(span)


def format_valid(text):
    """1.0 iff the output parses to a JSON object with all five expected keys."""
    obj = extract_json(text)
    if obj is None:
        return 0.0
    return 1.0 if all(k in obj for k in EXPECTED_KEYS) else 0.0


# --------------------------------------------------------------------------- #
# Pure normalization / parsing helpers.                                         #
# --------------------------------------------------------------------------- #
_PUNCT = string.punctuation


def normalize_str(s):
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    if s is None:
        return ""
    s = re.sub(r"\s+", " ", str(s).lower()).strip()
    return s.strip(_PUNCT + " ")


_MULT = {"k": 1e3, "thousand": 1e3,
         "m": 1e6, "mm": 1e6, "mil": 1e6, "million": 1e6,
         "b": 1e9, "bn": 1e9, "billion": 1e9}
_NUM_RE = re.compile(r"^([-+]?\d*\.?\d+)\s*([a-z]+)?$")


def parse_number(value):
    """Parse $12M / 12 million / $12,000,000 / 12000000 / 12000000.0 -> float, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = value.strip().lower().replace("$", "").replace("usd", "").replace(",", "").strip()
    m = _NUM_RE.match(s)
    if not m:
        return None
    num = float(m.group(1))
    suffix = m.group(2)
    if suffix:
        if suffix not in _MULT:
            return None
        num *= _MULT[suffix]
    return num


def _ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def token_set_ratio(a, b):
    """A pure (stdlib-only) token-set fuzzy ratio in [0, 1] over normalized strings."""
    a, b = normalize_str(a), normalize_str(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    t1, t2 = set(a.split()), set(b.split())
    inter = " ".join(sorted(t1 & t2))
    c1 = (inter + " " + " ".join(sorted(t1 - t2))).strip()
    c2 = (inter + " " + " ".join(sorted(t2 - t1))).strip()
    cands = [_ratio(a, b), _ratio(c1, c2)]
    if inter:
        cands += [_ratio(inter, c1), _ratio(inter, c2)]
    return max(cands)


def _canon_round(s):
    """Canonicalize a round name so 'A' == 'Series A' and 'Pre-Seed' == 'Preseed'."""
    s = normalize_str(s).replace("-", " ")
    s = re.sub(r"\bseries\b", "", s)
    return re.sub(r"\s+", "", s)


def _as_founder_list(value):
    """Coerce a founders value (list, or a 'A and B' string) into a list of names."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        return [p.strip() for p in re.split(r",|\band\b|&|;|/", value) if p.strip()]
    return [str(value)]


# --------------------------------------------------------------------------- #
# Per-field matchers (return 1.0 / 0.0).                                        #
# --------------------------------------------------------------------------- #
def _str_eq_strict(model, gold):
    return 1.0 if normalize_str(model) == normalize_str(gold) else 0.0


def _num_eq_strict(model, gold):
    pm, pg = parse_number(model), parse_number(gold)
    if pm is None or pg is None:
        return 0.0
    return 1.0 if round(pm) == round(pg) else 0.0  # whole-dollar exactness


def _founders_eq_strict(model, gold):
    sm = {normalize_str(x) for x in _as_founder_list(model)} - {""}
    sg = {normalize_str(x) for x in _as_founder_list(gold)} - {""}
    return 1.0 if sm == sg else 0.0


def _name_loose(model, gold, threshold=0.80):
    return 1.0 if token_set_ratio(model, gold) >= threshold else 0.0


def _round_loose(model, gold):
    return 1.0 if _canon_round(model) == _canon_round(gold) else 0.0


def _num_loose(model, gold, tol=0.10):
    pm, pg = parse_number(model), parse_number(gold)
    if pm is None or pg is None:
        return 0.0
    if pg == 0:
        return 1.0 if pm == 0 else 0.0
    return 1.0 if abs(pm - pg) <= tol * abs(pg) else 0.0


def _founders_loose(model, gold, threshold=0.80):
    m = [normalize_str(x) for x in _as_founder_list(model) if normalize_str(x)]
    g = [normalize_str(x) for x in _as_founder_list(gold) if normalize_str(x)]
    if not g or len(m) != len(g):
        return 0.0
    used = [False] * len(m)
    for gf in g:
        matched = False
        for i, mf in enumerate(m):
            if not used[i] and token_set_ratio(gf, mf) >= threshold:
                used[i] = matched = True
                break
        if not matched:
            return 0.0
    return 1.0


# --------------------------------------------------------------------------- #
# Public scores.                                                                #
# --------------------------------------------------------------------------- #
def grade(text, gold):
    """STRICT per-field score in [0, 1] = fraction of the 5 fields matching exactly."""
    obj = extract_json(text)
    if obj is None:
        return 0.0
    score = (
        _str_eq_strict(obj.get("company"), gold.get("company"))
        + _str_eq_strict(obj.get("round"), gold.get("round"))
        + _num_eq_strict(obj.get("raise"), gold.get("raise"))
        + _num_eq_strict(obj.get("valuation"), gold.get("valuation"))
        + _founders_eq_strict(obj.get("founders"), gold.get("founders"))
    )
    return score / 5.0


def grade_loose(text, gold):
    """LOOSE per-field score in [0, 1] (pre-registered tolerances)."""
    obj = extract_json(text)
    if obj is None:
        return 0.0
    score = (
        _name_loose(obj.get("company"), gold.get("company"))
        + _round_loose(obj.get("round"), gold.get("round"))
        + _num_loose(obj.get("raise"), gold.get("raise"))
        + _num_loose(obj.get("valuation"), gold.get("valuation"))
        + _founders_loose(obj.get("founders"), gold.get("founders"))
    )
    return score / 5.0


def all_five_exact(text, gold):
    """Diagnostic: 1.0 iff the strict per-field score is a perfect 1.0, else 0.0."""
    return 1.0 if grade(text, gold) == 1.0 else 0.0

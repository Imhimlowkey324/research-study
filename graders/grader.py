"""Strict grader for Task A (deal-math ownership %).

The grader scores one model output against one gold answer and returns a
binary reward: ``1.0`` for a match, ``0.0`` otherwise. It is deliberately
*strict* and *pre-registered* -- the match rule below is fixed in advance so
the reward signal is well defined for an RLVR run.

How the model output is normalized
----------------------------------
Model outputs are free-form text, so before comparing we extract a single
number from them:

1. **Find the final answer.** If the text contains an explicit answer cue
   (``answer is``, ``answer:``, ``ownership:``, ``result =``, ``equals``,
   ``=`` ...), the number immediately following the *last* such cue is taken
   (optionally through a hedge word like "about"/"approximately"). Otherwise
   the *last* number in the text is taken. Rationale: models that reason first
   usually state the answer last, and the task instruction tells them to reply
   with only the number -- so a bare trailing distractor with no cue is, by
   this documented rule, treated as the answer (and will simply be scored
   wrong if it isn't the gold).

2. **Strip surrounding noise.** Words, a leading "Ownership:", a trailing
   percent sign or the word "percent", trailing periods, and leading/trailing
   whitespace are all ignored -- only the numeric token is kept.

3. **Resolve commas (documented choice).** A comma followed by exactly three
   digits is a *thousands separator* and is removed (``3,000,000`` -> 3000000,
   ``2,016`` -> 2016), matching the dataset's own number formatting. A single
   comma followed by one or two digits is treated as a *decimal separator*
   (``16,13`` -> ``16.13``) so European-style decimals are still graded
   correctly. The period is always the decimal point.

4. **No parseable number -> 0.0.** A non-answer (empty string, "I don't know",
   prose with no digits) scores 0.0 rather than raising.

Match rule (strict, no tolerance)
----------------------------------
Both the extracted number and the gold are rounded to two decimals with
round-half-up (``ROUND_HALF_UP``) via :class:`decimal.Decimal`, independent of
binary float representation. They must then be *exactly* equal.

So ``16.13`` matches gold ``16.13``; ``16.1`` and ``16`` do **not**. At the
third-decimal boundary the documented behavior is ``16.125`` -> ``16.13``
(matches gold ``16.13``) and ``16.135`` -> ``16.14`` (does **not** match).
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

# A numeric token: optional sign, digits with optional thousands commas, an
# optional decimal part -- or a bare ".5". A trailing sentence period is left
# out because the decimal part requires digits after the dot.
_NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?|[-+]?\.\d+"
_NUMBER_RE = re.compile(_NUMBER)

# "answer cue" + (optional linker) + (optional hedge) + the number it labels.
_ANSWER_CUE_RE = re.compile(
    r"(?:final\s+answer|answer|ownership(?:\s+percentage)?|result|equals?|=)"
    r"\s*(?:is|are|was|of|to|:|=)?"
    r"\s*(?:about|approximately|roughly|around|nearly|~)?"
    r"\s*(" + _NUMBER + r")",
    re.IGNORECASE,
)

_TWO_PLACES = Decimal("0.01")


def _normalize_token(token: str) -> str | None:
    """Turn a raw numeric token into a clean decimal string, or None.

    Resolves commas (thousands vs. decimal) and strips signs/punctuation. The
    result is parseable by both ``float`` and ``Decimal``.
    """
    s = token.strip().rstrip(".,")
    if not s:
        return None

    sign = ""
    if s[0] in "+-":
        sign = "-" if s[0] == "-" else ""
        s = s[1:]

    if "," in s and "." in s:
        # e.g. "1,234.56" -> American grouping; commas are thousands separators.
        s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if (len(parts) == 2 and 1 <= len(parts[1]) <= 2
                and parts[0].isdigit() and parts[1].isdigit()):
            s = parts[0] + "." + parts[1]   # decimal comma, e.g. "16,13"
        else:
            s = s.replace(",", "")          # thousands separators

    candidate = sign + s
    try:
        float(candidate)
    except ValueError:
        return None
    return candidate


def _extract_number_str(text: str | None) -> str | None:
    """Select the model's final-answer number and return its normalized string."""
    if not text:
        return None

    cues = list(_ANSWER_CUE_RE.finditer(text))
    if cues:
        norm = _normalize_token(cues[-1].group(1))
        if norm is not None:
            return norm

    for token in reversed(_NUMBER_RE.findall(text)):
        norm = _normalize_token(token)
        if norm is not None:
            return norm
    return None


def _round2(value: float | str) -> Decimal | None:
    """Round to two decimals, half-up, via Decimal (no float drift)."""
    try:
        return Decimal(str(value)).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return None


def extract_answer(text: str | None) -> float | None:
    """Find the model's final-answer number; return it as a float, or None.

    This is the public "find the number in the model's output" step. It is the
    basis for the format-validity metric (:func:`format_valid`) and shares the
    same underlying extraction logic that :func:`grade` uses internally.
    Returns ``None`` when the text contains no parseable number.
    """
    norm = _extract_number_str(text)
    return float(norm) if norm is not None else None


# Backwards-compatible alias for the previous public name.
extract_number = extract_answer


def grade(model_output: str | None, gold: float) -> float:
    """Strict binary score: 1.0 if the model's answer matches gold, else 0.0.

    Parameters
    ----------
    model_output : the raw, free-form model output text.
    gold         : the gold ownership percentage (a float, e.g. 16.13).
    """
    extracted = _extract_number_str(model_output)
    if extracted is None:
        return 0.0
    e, g = _round2(extracted), _round2(gold)
    if e is None or g is None:
        return 0.0
    return 1.0 if e == g else 0.0


def format_valid(text: str | None) -> float:
    """Format-validity metric: 1.0 if the model produced a number, else 0.0.

    One of the study's three metrics (correctness, format-validity, Pass@k).
    It asks only whether the model emitted a parseable number at all -- it does
    NOT check whether that number is correct.
    """
    return 1.0 if extract_answer(text) is not None else 0.0


def accuracy(model_outputs, golds) -> float:
    """Mean strict score over a batch of (output, gold) pairs."""
    outputs = list(model_outputs)
    gold_list = list(golds)
    if len(outputs) != len(gold_list):
        raise ValueError("model_outputs and golds must be the same length")
    if not outputs:
        return 0.0
    return sum(grade(o, g) for o, g in zip(outputs, gold_list)) / len(outputs)

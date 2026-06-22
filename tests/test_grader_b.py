"""Adversarial tests for the Task B JSON grader — the cases that break JSON graders.

Built and run BEFORE the generator (Principle 7): a brittle grader makes the study
fiction. Pure Python, no GPU.
"""

import json

import pytest

from graders.grader_b import (
    _canon_round,
    all_five_exact,
    extract_json,
    format_valid,
    grade,
    grade_loose,
    normalize_str,
    parse_number,
    token_set_ratio,
)

GOLD = {
    "company": "Northwind",
    "round": "Series A",
    "raise": 12_000_000,
    "valuation": 60_000_000,
    "founders": ["Alice Johnson", "Bob Lee"],
}


def _out(**overrides):
    """A model JSON string = the gold with some fields overridden."""
    return json.dumps({**GOLD, **overrides})


# --------------------------------------------------------------------------- #
# Invariances: key order, founder order, number formats                        #
# --------------------------------------------------------------------------- #
def test_key_order_invariance():
    a = json.dumps({"company": "Northwind", "round": "Series A", "raise": 12_000_000,
                    "valuation": 60_000_000, "founders": ["Alice Johnson", "Bob Lee"]})
    b = json.dumps({"founders": ["Alice Johnson", "Bob Lee"], "valuation": 60_000_000,
                    "raise": 12_000_000, "round": "Series A", "company": "Northwind"})
    assert grade(a, GOLD) == 1.0
    assert grade(b, GOLD) == 1.0


def test_founder_order_invariance():
    assert grade(_out(founders=["Alice Johnson", "Bob Lee"]), GOLD) == 1.0
    assert grade(_out(founders=["Bob Lee", "Alice Johnson"]), GOLD) == 1.0


@pytest.mark.parametrize("raise_val", ["$12M", "12 million", "$12,000,000", 12_000_000, "12000000"])
def test_number_formats_all_equal_under_strict(raise_val):
    assert grade(_out(**{"raise": raise_val}), GOLD) == 1.0   # 'raise' is a keyword


def test_parse_number_formats():
    for v in ["$12M", "12 million", "$12,000,000", 12_000_000, "12000000"]:
        assert parse_number(v) == 12_000_000.0
    assert round(parse_number("$1.2B")) == 1_200_000_000
    assert parse_number("$500K") == 500_000.0
    assert parse_number("not a number") is None
    assert parse_number(None) is None


# --------------------------------------------------------------------------- #
# Loose tolerances (the dial) pass where strict fails                          #
# --------------------------------------------------------------------------- #
def test_name_typo_passes_loose_fails_strict():
    out = _out(company="Northwynd")        # one-character typo
    assert grade(out, GOLD) == 0.8         # strict: company miss -> 4/5
    assert grade_loose(out, GOLD) == 1.0   # loose: fuzzy name match


def test_number_within_10pct_passes_loose_fails_strict():
    out = _out(**{"raise": 12_500_000})    # +4.2%, within 10%
    assert grade(out, GOLD) == 0.8
    assert grade_loose(out, GOLD) == 1.0


def test_number_outside_10pct_fails_loose():
    out = _out(**{"raise": 20_000_000})    # +66%
    assert grade_loose(out, GOLD) == 0.8


def test_round_abbreviation_passes_loose_fails_strict():
    out = _out(round="A")                  # "A" for "Series A"
    assert grade(out, GOLD) == 0.8
    assert grade_loose(out, GOLD) == 1.0


def test_canon_round_synonyms():
    assert _canon_round("Series A") == _canon_round("A")
    assert _canon_round("Pre-Seed") == _canon_round("Preseed") == _canon_round("Pre Seed")
    assert _canon_round("Seed") != _canon_round("Series A")


def test_token_set_ratio_thresholds():
    assert token_set_ratio("Northwynd", "Northwind") >= 0.80
    assert token_set_ratio("Acme Robotics", "Acme Robotic") >= 0.80
    assert token_set_ratio("Totally Different", "Northwind") < 0.80


# --------------------------------------------------------------------------- #
# Case / whitespace / trailing punctuation                                     #
# --------------------------------------------------------------------------- #
def test_case_whitespace_punctuation_handled():
    out = json.dumps({**GOLD, "company": "  northwind ", "round": "series a.",
                      "founders": ["alice johnson", "bob lee"]})
    assert grade(out, GOLD) == 1.0


def test_normalize_str_basics():
    assert normalize_str("  Series A. ") == "series a"
    assert normalize_str("Northwind,") == "northwind"


# --------------------------------------------------------------------------- #
# Malformed / fenced / prose-wrapped output                                    #
# --------------------------------------------------------------------------- #
def test_fenced_json_extracted():
    text = f"Sure, here you go:\n```json\n{json.dumps(GOLD)}\n```\nHope that helps!"
    assert grade(text, GOLD) == 1.0
    assert format_valid(text) == 1.0


def test_leading_and_trailing_prose_ignored():
    assert grade("The extracted record is " + json.dumps(GOLD) + ". Done.", GOLD) == 1.0


def test_unparseable_scores_zero():
    assert grade("I can't help with that.", GOLD) == 0.0
    assert grade("{not valid json", GOLD) == 0.0
    assert extract_json("no json here") is None
    assert format_valid("no json here") == 0.0
    assert format_valid("{not valid json") == 0.0


def test_missing_key_is_format_invalid_but_still_graded():
    partial = json.dumps({"company": "Northwind", "round": "Series A", "raise": 12_000_000})
    assert format_valid(partial) == 0.0          # missing valuation + founders
    assert grade(partial, GOLD) == 0.6           # 3 present & correct, 2 missing


# --------------------------------------------------------------------------- #
# Per-field grading: partial credit, never-1.0-when-wrong, exactly-1.0-correct  #
# --------------------------------------------------------------------------- #
def test_partial_three_of_five_is_point_six():
    out = _out(valuation=99_999_999, founders=["Someone Else"])
    assert grade(out, GOLD) == 0.6


def test_fully_correct_is_one_and_all_five_exact():
    assert grade(json.dumps(GOLD), GOLD) == 1.0
    assert grade_loose(json.dumps(GOLD), GOLD) == 1.0
    assert all_five_exact(json.dumps(GOLD), GOLD) == 1.0


def test_wrong_answer_never_scores_one():
    out = json.dumps({"company": "Zephyr", "round": "Series Z", "raise": 1,
                      "valuation": 2, "founders": ["Nobody"]})
    assert grade(out, GOLD) == 0.0
    assert grade_loose(out, GOLD) < 1.0
    assert all_five_exact(out, GOLD) == 0.0


def test_loose_is_superset_of_strict_on_a_mixed_output():
    # company exact, round abbrev, raise within 10%, valuation exact, founder typo
    out = json.dumps({"company": "Northwind", "round": "A", "raise": 12_400_000,
                      "valuation": 60_000_000, "founders": ["Alice Johnsen", "Bob Lee"]})
    assert grade(out, GOLD) <= grade_loose(out, GOLD)
    assert grade_loose(out, GOLD) == 1.0

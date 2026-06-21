"""Tests for the strict Task A ownership-% grader.

Two non-negotiable properties are asserted across many cases:
  * it must NEVER mark a correct answer wrong, and
  * it must NEVER mark a wrong answer right.
"""

import pytest

from graders.grader import accuracy, extract_number, format_valid, grade, grade_loose


# --------------------------------------------------------------------------- #
# Exact correct                                                                #
# --------------------------------------------------------------------------- #
def test_exact_correct():
    assert grade("16.13", 16.13) == 1.0


# --------------------------------------------------------------------------- #
# Correct, wrapped in noise                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "16.13%",
    "about 16.13",
    "The answer is 16.13.",
    "Ownership: 16.13%",
    " 16.13 \n",
    "16.13 percent",
    "\n\n16.13\n",
    "the ownership percentage is approximately 16.13%.",
    "Answer = 16.13",
])
def test_correct_with_noise(text):
    assert grade(text, 16.13) == 1.0


# --------------------------------------------------------------------------- #
# Whole-number golds and trailing-zero formatting                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", ["20", "20.0", "20.00", "20%", "Ownership: 20.00%"])
def test_whole_number_golds(text):
    assert grade(text, 20.0) == 1.0


# --------------------------------------------------------------------------- #
# Genuinely wrong (no tolerance)                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,gold", [
    ("16.1", 16.13),
    ("16", 16.13),
    ("61.3", 16.13),
    ("16.14", 16.13),
    ("16.12", 16.13),
    ("0.1613", 16.13),     # fraction instead of percent
    ("1613", 16.13),       # decimal point dropped
    ("19.99", 20.0),
    ("20.1", 20.0),
    ("21.7", 21.67),
    ("21.6", 21.67),
])
def test_genuinely_wrong(text, gold):
    assert grade(text, gold) == 0.0


# --------------------------------------------------------------------------- #
# Distractor traps: the model echoes a non-answer number from the prompt       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "2016",            # founding year
    "3",               # number of investors / $3M ARR significand
    "$3,000,000",      # a raise/ARR amount
    "750,000",         # a raise amount
    "40 people",       # headcount
])
def test_distractor_numbers_are_wrong(text):
    assert grade(text, 21.67) == 0.0


def test_distractor_before_cued_answer():
    # The real answer is cued; the leading distractor must be ignored.
    assert grade("Founded in 2016, the answer is 21.67%.", 21.67) == 1.0
    assert grade("The firm had 40 employees; ownership: 21.67%.", 21.67) == 1.0


def test_distractor_after_answer_without_cue_uses_last_number():
    # Documented behavior: with no cue, the LAST number wins. Here that is the
    # correct answer, so the trailing distractor scenario must still resolve it.
    assert grade("Founded 2016, so 21.67", 21.67) == 1.0


# --------------------------------------------------------------------------- #
# Non-answers                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "",
    "   ",
    "I don't know",
    "the ownership percentage",
    "It depends on several factors that the question does not pin down.",
    None,
])
def test_non_answers(text):
    assert grade(text, 16.13) == 0.0


# --------------------------------------------------------------------------- #
# Rounding edges (explicit, documented: round-half-up to 2 dp)                 #
# --------------------------------------------------------------------------- #
def test_rounding_half_up_boundary():
    # 16.125 -> 16.13 (half rounds up) -> matches gold 16.13
    assert grade("16.125", 16.13) == 1.0
    # 16.135 -> 16.14 (half rounds up) -> does NOT match gold 16.13
    assert grade("16.135", 16.13) == 0.0
    # Just below / above the boundary, for completeness.
    assert grade("16.134", 16.13) == 1.0
    assert grade("16.126", 16.13) == 1.0
    assert grade("16.124", 16.13) == 0.0
    assert grade("16.144", 16.13) == 0.0


# --------------------------------------------------------------------------- #
# Comma handling (documented)                                                  #
# --------------------------------------------------------------------------- #
def test_comma_as_decimal_separator():
    # European-style decimal comma is graded correctly.
    assert grade("16,13", 16.13) == 1.0
    assert grade("ownership is 16,13%", 16.13) == 1.0


def test_comma_as_thousands_separator():
    # A grouped number is not mistaken for a small decimal.
    assert extract_number("3,000,000") == 3000000.0
    assert extract_number("2,016") == 2016.0
    assert grade("1,234.56", 16.13) == 0.0


# --------------------------------------------------------------------------- #
# Final-answer selection                                                       #
# --------------------------------------------------------------------------- #
def test_last_cue_wins_when_model_corrects_itself():
    text = "The answer is 5. Wait, that's wrong -- the answer is 50."
    assert grade(text, 50.0) == 1.0
    assert grade(text, 5.0) == 0.0


# --------------------------------------------------------------------------- #
# Direct extraction checks                                                     #
# --------------------------------------------------------------------------- #
def test_extract_number_values():
    assert extract_number("The answer is 16.13.") == 16.13
    assert extract_number("about 16.13") == 16.13
    assert extract_number("2016") == 2016.0
    assert extract_number("16,13") == 16.13
    assert extract_number("") is None
    assert extract_number("I don't know") is None
    assert extract_number(None) is None


# --------------------------------------------------------------------------- #
# accuracy() helper                                                            #
# --------------------------------------------------------------------------- #
def test_accuracy_helper():
    outs = ["16.13", "16", "The answer is 20."]
    golds = [16.13, 16.13, 20.0]
    assert accuracy(outs, golds) == pytest.approx(2 / 3)
    assert accuracy([], []) == 0.0
    with pytest.raises(ValueError):
        accuracy(["16.13"], [16.13, 20.0])


# --------------------------------------------------------------------------- #
# The two non-negotiable properties, swept over many golds and forms           #
# --------------------------------------------------------------------------- #
GOLDS = [5.0, 10.0, 16.13, 20.0, 21.67, 29.33, 33.33, 50.0, 12.7]


def _correct_forms(gold: float):
    g2 = f"{gold:.2f}"          # always 2 dp, e.g. "20.00"
    gg = f"{gold:g}"            # trimmed, e.g. "20"
    return [
        gg, g2,
        f"{gg}%", f"{g2}%",
        f"about {gg}",
        f"The answer is {g2}.",
        f"Ownership: {g2}%",
        f"  {gg}\n",
        f"the final answer is {gg} percent",
    ]


def test_never_marks_a_correct_answer_wrong():
    for gold in GOLDS:
        for text in _correct_forms(gold):
            assert grade(text, gold) == 1.0, (text, gold)


def test_never_marks_a_wrong_answer_right():
    # For each gold, every OTHER gold's value is a wrong answer, plus some
    # near-miss perturbations -- none may score 1.0.
    for gold in GOLDS:
        wrongs = [f"{other:g}" for other in GOLDS if other != gold]
        wrongs += [
            f"{gold + 0.01:.2f}",
            f"{gold - 0.01:.2f}",
            f"{gold + 0.1:.1f}",
            f"{round(gold)}" if abs(gold - round(gold)) > 0.005 else f"{gold + 1:g}",
            "2016", "3", "750,000",
        ]
        for text in wrongs:
            assert grade(text, gold) == 0.0, (text, gold)


# --------------------------------------------------------------------------- #
# format_valid(): the format-validity metric -- "did the model emit a number?" #
# (independent of whether that number is correct)                              #
# --------------------------------------------------------------------------- #
def test_format_valid_clean_number():
    assert format_valid("16.13") == 1.0


def test_format_valid_number_wrapped_in_words():
    assert format_valid("The answer is about 16.13%.") == 1.0


def test_format_valid_i_dont_know():
    assert format_valid("I don't know") == 0.0


def test_format_valid_empty_string():
    assert format_valid("") == 0.0


def test_format_valid_is_independent_of_correctness():
    # A wrong number is still a VALID FORMAT -- a number was produced.
    assert format_valid("2016") == 1.0
    assert grade("2016", 16.13) == 0.0


@pytest.mark.parametrize("text", ["", "   ", None, "the ownership percentage", "no digits here"])
def test_format_valid_is_zero_when_no_number(text):
    assert format_valid(text) == 0.0


# --------------------------------------------------------------------------- #
# grade_loose(): the reward dial -- correct within +/- LOOSE_TOLERANCE (0.50)  #
# --------------------------------------------------------------------------- #
def test_loose_exact_match():
    assert grade_loose("16.13", 16.13) == 1.0


@pytest.mark.parametrize("text", ["16.50", "15.70", "16.13", "16.00", "16.30"])
def test_loose_inside_band(text):
    assert grade_loose(text, 16.13) == 1.0


@pytest.mark.parametrize("text", ["16.70", "15.60", "20", "10.0"])
def test_loose_outside_band(text):
    assert grade_loose(text, 16.13) == 0.0


def test_loose_boundary_is_inclusive():
    # Exactly 0.50 away on each side -> inclusive -> 1.0.
    assert grade_loose("15.63", 16.13) == 1.0
    assert grade_loose("16.63", 16.13) == 1.0
    # Just past the 0.50 boundary -> 0.0.
    assert grade_loose("15.62", 16.13) == 0.0
    assert grade_loose("16.64", 16.13) == 0.0


def test_the_dial_changes_something():
    # Non-negotiable: an input that is WRONG under strict but ACCEPTED loose.
    assert grade("16.40", 16.13) == 0.0
    assert grade_loose("16.40", 16.13) == 1.0


@pytest.mark.parametrize("text", ["2016", "750,000", "61.3", "0.1613"])
def test_loose_distractor_wrong_under_both(text):
    assert grade(text, 16.13) == 0.0
    assert grade_loose(text, 16.13) == 0.0


@pytest.mark.parametrize("text", ["", "   ", None, "I don't know", "the ownership percentage"])
def test_loose_non_answer(text):
    assert grade_loose(text, 16.13) == 0.0

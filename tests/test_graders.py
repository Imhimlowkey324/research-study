"""Tests for the graders package."""

from graders.grader import accuracy, exact_match, within_tolerance


def test_exact_match():
    assert exact_match(4.0, 4.0) == 1.0
    assert exact_match(4.0, 5.0) == 0.0


def test_within_tolerance():
    assert within_tolerance(1.0, 1.0 + 1e-9) == 1.0
    assert within_tolerance(1.0, 1.5) == 0.0


def test_accuracy_fraction():
    assert accuracy([1.0, 2.0, 3.0], [1.0, 2.0, 0.0]) == 2 / 3


def test_accuracy_empty_is_zero():
    assert accuracy([], []) == 0.0


def test_accuracy_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        accuracy([1.0, 2.0], [1.0])

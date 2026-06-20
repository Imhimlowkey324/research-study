"""Graders score candidate answers against the expected results.

Each grader takes an expected value and a candidate (actual) value and returns
a numeric score. Add new graders here as the study grows.
"""

from __future__ import annotations


def exact_match(expected: float, actual: float) -> float:
    """Return 1.0 if the answer is exactly right, else 0.0."""
    return 1.0 if expected == actual else 0.0


def within_tolerance(expected: float, actual: float, tol: float = 1e-6) -> float:
    """Return 1.0 if ``actual`` is within ``tol`` of ``expected``, else 0.0."""
    return 1.0 if abs(expected - actual) <= tol else 0.0


def accuracy(expected: list[float], actual: list[float]) -> float:
    """Return the fraction of exact matches across the whole dataset."""
    if not expected:
        return 0.0
    if len(expected) != len(actual):
        raise ValueError("expected and actual must be the same length")
    hits = sum(exact_match(e, a) for e, a in zip(expected, actual))
    return hits / len(expected)

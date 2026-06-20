"""Tests for the data_generation package."""

from data_generation.generate import generate_samples, save_samples


def test_generate_is_reproducible():
    assert generate_samples(5, seed=42) == generate_samples(5, seed=42)


def test_generate_respects_count():
    assert len(generate_samples(7)) == 7


def test_expected_matches_prompt():
    # The expected value should be the sum encoded in the prompt.
    for sample in generate_samples(20):
        a, b = (int(x) for x in sample.prompt.split(" + "))
        assert sample.expected == a + b


def test_save_samples_writes_one_line_per_sample(tmp_path):
    samples = generate_samples(3)
    out = save_samples(samples, tmp_path / "out.jsonl")
    assert out.exists()
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3

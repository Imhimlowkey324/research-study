"""Generate synthetic data for the research study.

This module produces the raw dataset the study runs on. Replace the example
logic below with your real data-generation process (sampling, simulation,
API calls, etc.). Everything is kept reproducible via a fixed random seed.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Sample:
    """A single item in the study dataset."""

    id: int
    prompt: str
    expected: float


def generate_samples(n: int, seed: int = 0) -> list[Sample]:
    """Create ``n`` reproducible synthetic samples.

    Passing the same ``seed`` always yields the same data, which keeps the
    study reproducible across runs and machines.
    """
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        samples.append(Sample(id=i, prompt=f"{a} + {b}", expected=float(a + b)))
    return samples


def save_samples(samples: list[Sample], path: str | Path) -> Path:
    """Write samples to disk as JSON Lines and return the output path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample)) + "\n")
    return path


if __name__ == "__main__":
    data = generate_samples(10)
    out = save_samples(data, "data/samples.jsonl")
    print(f"Wrote {len(data)} samples to {out}")

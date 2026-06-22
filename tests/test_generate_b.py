"""Pure-Python tests for the Task B data generator (no GPU)."""

import json
from collections import Counter

from data_generation.generate_b import (
    EASY_OOD_TEMPLATES,
    EASY_TRAIN_TEMPLATES,
    HARD_OOD_TEMPLATES,
    HARD_TRAIN_TEMPLATES,
    TASK_B_INSTRUCTION,
    _recoverable,
    build_all_b,
)
from graders.grader_b import format_valid, grade

SEED = 7


def _build():
    return build_all_b(SEED, n_easy=60, n_hard=60, n_ood_easy=15, n_ood_hard=15)


def test_determinism_same_seed():
    assert _build() == _build()


def test_counts_and_difficulty_split():
    d = _build()
    assert len(d["train_easy"]) == 60
    assert len(d["train_hard"]) == 60
    assert len(d["ood_test"]) == 30
    assert sum(it["difficulty"] == "easy" for it in d["ood_test"]) == 15
    assert sum(it["difficulty"] == "hard" for it in d["ood_test"]) == 15


def test_item_shape():
    d = _build()
    for it in d["train_easy"][:5] + d["train_hard"][:5] + d["ood_test"][:5]:
        assert it["prompt"].endswith(TASK_B_INSTRUCTION)
        ans = it["answer"]
        assert set(ans) == {"company", "round", "raise", "valuation", "founders"}
        assert isinstance(ans["founders"], list) and 1 <= len(ans["founders"]) <= 3
        assert isinstance(ans["raise"], int) and isinstance(ans["valuation"], int)
        assert ans["valuation"] > ans["raise"]   # pre-money > raise


def test_train_and_ood_templates_are_disjoint():
    train = set(EASY_TRAIN_TEMPLATES) | set(HARD_TRAIN_TEMPLATES)
    ood = set(EASY_OOD_TEMPLATES) | set(HARD_OOD_TEMPLATES)
    assert train.isdisjoint(ood)


def test_train_and_ood_prompts_are_disjoint():
    d = _build()
    train_prompts = {it["prompt"] for it in d["train_easy"] + d["train_hard"]}
    ood_prompts = {it["prompt"] for it in d["ood_test"]}
    assert train_prompts.isdisjoint(ood_prompts)


def test_gold_recoverable_and_self_consistent():
    d = _build()
    sample = d["train_easy"][:12] + d["train_hard"][:12] + d["ood_test"][:12]
    for it in sample:
        # the gold values are actually present/parseable in the prose
        assert _recoverable(it["answer"], it["info"]["prose"]), it["info"]["prose"]
        # a perfect extraction (the gold itself) scores a clean 1.0 and is format-valid
        gold_json = json.dumps(it["answer"])
        assert grade(gold_json, it["answer"]) == 1.0
        assert format_valid(gold_json) == 1.0


def _max_freq(values):
    counts = Counter(values)
    return max(counts.values()) / len(values)


def test_no_single_value_dominates():
    d = _build()
    for split in ("train_easy", "train_hard"):
        items = d[split]
        n = len(items)
        assert _max_freq([it["answer"]["company"] for it in items]) < 0.5
        assert _max_freq([it["answer"]["round"] for it in items]) < 0.5
        assert _max_freq([it["answer"]["raise"] for it in items]) < 0.5
        assert _max_freq([it["answer"]["valuation"] for it in items]) < 0.5
        name_counts = Counter()
        for it in items:
            for name in set(it["answer"]["founders"]):
                name_counts[name] += 1
        assert max(name_counts.values()) / n < 0.5

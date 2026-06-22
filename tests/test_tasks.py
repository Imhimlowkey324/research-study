"""Tests for the task router and Task-B engine wiring (pure Python, no GPU).

The critical test is the REGRESSION: task='A' resolves to exactly today's Task-A
wiring, so the in-flight Task-A runs are unaffected.
"""

import json
import re
from pathlib import Path

import pytest

import study_config as cfg
from data_generation.generate import build_all
from data_generation.generate_b import build_all_b
from graders import grader as grader_a
from graders import grader_b
from training.run_real import expected_repo_ids, missing_runs, repo_id
from training.tasks import task_spec
from training.train_grpo import build_reward_func, to_grpo_dataset

CONDITIONS = [("strict", "easy"), ("strict", "easy_hard"),
              ("loose", "easy"), ("loose", "easy_hard")]
SEEDS = [0, 1, 2]


# --------------------------------------------------------------------------- #
# REGRESSION: task='A' is byte-for-byte today's Task-A wiring (A untouched)     #
# --------------------------------------------------------------------------- #
def test_task_spec_A_is_unchanged_task_A_wiring():
    spec = task_spec("A")
    assert spec.build_all is build_all
    assert spec.grade is grader_a.grade
    assert spec.grade_loose is grader_a.grade_loose
    assert spec.extract is grader_a.extract_answer
    assert spec.format_valid is grader_a.format_valid
    assert spec.system_prompt == cfg.SYSTEM_PROMPT
    assert spec.repo_prefix == "rlvr-taskA-"
    assert spec.encode_gold(16.13) == 16.13      # identity codec
    assert spec.decode_gold(16.13) == 16.13


def test_repo_id_A_names_unchanged():
    # exact legacy names — must equal what the in-flight runs already use
    assert repo_id("strict", "easy", 0, "alice") == "alice/rlvr-taskA-strict-easy-seed0"
    assert repo_id("loose", "easy_hard", 2, "alice", task="A") == \
        "alice/rlvr-taskA-loose-easy_hard-seed2"


def test_to_grpo_dataset_A_unchanged():
    items = [{"prompt": "Compute the ownership.", "answer": 12.5, "difficulty": "easy"}]
    rows = to_grpo_dataset(items)                # default task='A'
    assert rows[0]["prompt"][0] == {"role": "system", "content": cfg.SYSTEM_PROMPT}
    assert rows[0]["answer"] == 12.5             # identity codec, float as-is


def test_task_A_snapshot_excludes_task_B_constants():
    snap = cfg.snapshot()
    assert "SYSTEM_PROMPT" in snap and "TRAIN_EXAMPLES" in snap
    assert "SYSTEM_PROMPT_B" not in snap
    assert "MAX_COMPLETION_LENGTH_B" not in snap


# --------------------------------------------------------------------------- #
# task='B' wiring                                                              #
# --------------------------------------------------------------------------- #
def test_task_spec_B_wiring():
    spec = task_spec("B")
    assert spec.build_all is build_all_b
    assert spec.grade is grader_b.grade
    assert spec.grade_loose is grader_b.grade_loose
    assert spec.extract is grader_b.extract_json
    assert spec.format_valid is grader_b.format_valid
    assert spec.system_prompt == cfg.SYSTEM_PROMPT_B
    assert spec.repo_prefix == "rlvr-taskB-"


def test_task_spec_rejects_unknown():
    with pytest.raises(ValueError):
        task_spec("C")


def test_gold_codec_B_roundtrips():
    spec = task_spec("B")
    gold = {"company": "Acme", "round": "Series A", "raise": 5000000,
            "valuation": 20000000, "founders": ["Jo Lee"]}
    encoded = spec.encode_gold(gold)
    assert isinstance(encoded, str)              # JSON string in the dataset column
    assert spec.decode_gold(encoded) == gold     # dict -> str -> dict


def test_system_prompt_b_matches_prereg():
    text = Path(__file__).resolve().parent.parent.joinpath("TASKB_PREREG.md").read_text(
        encoding="utf-8")
    match = re.search(r"```text\n(.*?)\n```", text, re.DOTALL)
    assert match is not None
    assert match.group(1).strip() == cfg.SYSTEM_PROMPT_B


# --------------------------------------------------------------------------- #
# Task-B reward wrapper (the engine routes B end-to-end in code)               #
# --------------------------------------------------------------------------- #
GOLD_B = {"company": "Northwind", "round": "Series A", "raise": 12000000,
          "valuation": 60000000, "founders": ["Alice Johnson", "Bob Lee"]}
GOLD_B_STR = json.dumps(GOLD_B)   # how the gold is stored in the dataset 'answer' column


def _chat(text):
    return [{"role": "assistant", "content": text}]


def test_reward_B_strict_perfect_and_partial():
    rf = build_reward_func("strict", "B")
    perfect = json.dumps(GOLD_B)
    partial = json.dumps({**GOLD_B, "valuation": 1, "founders": ["Wrong"]})   # 3 of 5
    rewards = rf(completions=[perfect, partial], answer=[GOLD_B_STR, GOLD_B_STR])
    assert rewards == [1.0, 0.6]


def test_reward_B_handles_chat_list_and_string():
    rf = build_reward_func("strict", "B")
    s = json.dumps(GOLD_B)
    rewards = rf(completions=[s, _chat(s)], answer=[GOLD_B_STR, GOLD_B_STR])
    assert rewards == [1.0, 1.0]


def test_reward_B_near_miss_loose_beats_strict():
    strict = build_reward_func("strict", "B")
    loose = build_reward_func("loose", "B")
    # company typo + raise within 10% -> strict misses 2 fields, loose gets all 5
    out = json.dumps({**GOLD_B, "company": "Northwynd", "raise": 12500000})
    assert strict(completions=[out], answer=[GOLD_B_STR]) == [0.6]
    assert loose(completions=[out], answer=[GOLD_B_STR]) == [1.0]


def test_reward_B_unparseable_is_zero():
    rf = build_reward_func("strict", "B")
    assert rf(completions=["I cannot help with that."], answer=[GOLD_B_STR]) == [0.0]


# --------------------------------------------------------------------------- #
# Task-B repo names + missing set                                             #
# --------------------------------------------------------------------------- #
def test_repo_id_B_twelve_distinct_and_disjoint_from_A():
    b_names = [repo_id(r, d, s, "alice", task="B") for (r, d) in CONDITIONS for s in SEEDS]
    assert len(b_names) == 12 and len(set(b_names)) == 12
    assert b_names[0] == "alice/rlvr-taskB-strict-easy-seed0"
    a_names = [repo_id(r, d, s, "alice", task="A") for (r, d) in CONDITIONS for s in SEEDS]
    assert set(b_names).isdisjoint(set(a_names))


def test_expected_repo_ids_and_missing_set_B():
    expected = expected_repo_ids("alice", task="B")
    assert len(expected) == 12
    done = expected[:5]
    assert missing_runs(done, expected) == set(expected[5:])

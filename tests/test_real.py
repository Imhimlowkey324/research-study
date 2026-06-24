"""Pure-Python tests for the real-run wrapper (no model, no GPU, no Hub).

Importing training.run_real must not import torch or huggingface_hub (deferred),
so these run on a plain machine.
"""

import pytest

from study_config import snapshot
from training.train_grpo import NaNLossError
from training.run_real import (
    _classify_error,
    _fmt_hm,
    _parse_reward_line,
    batch_repo_id,
    batch_status_label,
    build_results,
    missing_runs,
    repo_id,
    reward_trended_up,
    should_stop,
    summarize_batch,
)

CONDITIONS = [("strict", "easy"), ("strict", "easy_hard"),
              ("loose", "easy"), ("loose", "easy_hard")]
SEEDS = [0, 1, 2]


# --------------------------------------------------------------------------- #
# repo_id                                                                      #
# --------------------------------------------------------------------------- #
def test_repo_id_deterministic_and_exact():
    assert repo_id("strict", "easy", 0, "alice") == "alice/rlvr-taskA-strict-easy-seed0"
    assert repo_id("strict", "easy", 0, "alice") == repo_id("strict", "easy", 0, "alice")


def test_repo_id_twelve_distinct_names():
    names = [repo_id(r, d, s, "alice") for (r, d) in CONDITIONS for s in SEEDS]
    assert len(names) == 12
    assert len(set(names)) == 12     # all unique per (condition, seed)


# --------------------------------------------------------------------------- #
# reward_trended_up                                                            #
# --------------------------------------------------------------------------- #
def _hist(values):
    return [{"step": i, "reward": v} for i, v in enumerate(values)]


def test_reward_trend_rising():
    t = reward_trended_up(_hist([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
    assert t["trended_up"] is True
    assert t["last_q_mean"] > t["first_q_mean"]
    assert t["max_reward"] == 0.7
    assert t["delta"] > 0


def test_reward_trend_flat():
    t = reward_trended_up(_hist([0.5] * 8))
    assert t["trended_up"] is False
    assert t["delta"] == 0.0


def test_reward_trend_falling():
    t = reward_trended_up(_hist([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]))
    assert t["trended_up"] is False
    assert t["delta"] < 0


def test_reward_trend_short_and_empty_do_not_crash():
    short = reward_trended_up(_hist([0.3]))      # n=1
    assert short["trended_up"] is False
    empty = reward_trended_up([])
    assert empty["trended_up"] is False
    assert empty["max_reward"] is None


# --------------------------------------------------------------------------- #
# missing_runs                                                                 #
# --------------------------------------------------------------------------- #
def test_missing_runs_partial():
    expected = [repo_id(r, d, s, "bob") for (r, d) in CONDITIONS for s in SEEDS]
    done = expected[:5]
    assert missing_runs(done, expected) == set(expected[5:])
    assert len(missing_runs(done, expected)) == 7


def test_missing_runs_all_done_is_empty():
    expected = [repo_id(r, d, s, "bob") for (r, d) in CONDITIONS for s in SEEDS]
    assert missing_runs(expected, expected) == set()


# --------------------------------------------------------------------------- #
# build_results assembly                                                       #
# --------------------------------------------------------------------------- #
def test_build_results_includes_snapshot_and_trend():
    snap = snapshot()
    trend = reward_trended_up(_hist([0.1, 0.5]))
    history = [{"step": 1, "reward": 0.1}, {"step": 2, "reward": 0.5}]
    results = build_results("strict", "easy", 0, "ok", "abc1234", 123.4, 5.6,
                            history, trend, "alice/rlvr-taskA-strict-easy-seed0", snap)
    assert results["reward_mode"] == "strict"
    assert results["difficulty"] == "easy"
    assert results["seed"] == 0
    assert results["status"] == "ok"
    assert results["commit"] == "abc1234"
    assert results["runtime_sec"] == 123.4
    assert results["peak_mem_gb"] == 5.6
    assert results["reward_history"] == history
    assert results["reward_trend"] == trend
    assert results["adapter_repo"] == "alice/rlvr-taskA-strict-easy-seed0"
    # the full frozen config is captured for the record
    assert results["config_snapshot"] == snap
    assert "SYSTEM_PROMPT" in results["config_snapshot"]
    assert "TRAIN_EXAMPLES" in results["config_snapshot"]


# --------------------------------------------------------------------------- #
# Incremental-capture + status helpers (underpin crash safety)                 #
# --------------------------------------------------------------------------- #
def test_parse_reward_line():
    assert _parse_reward_line("  step 5: reward mean=0.3333 std=0.47") == \
        {"step": 5, "reward": 0.3333, "reward_std": 0.47}
    assert _parse_reward_line("  step 0: reward mean=0.0000 std=None") == \
        {"step": 0, "reward": 0.0, "reward_std": None}
    assert _parse_reward_line("saved adapter to /kaggle/working") is None


def test_fmt_hm():
    assert _fmt_hm(0) == "0:00"
    assert _fmt_hm(3661) == "1:01"
    assert _fmt_hm(7320) == "2:02"


def test_classify_error():
    assert _classify_error(NaNLossError("loss became nan")) == "nan"
    assert _classify_error(RuntimeError("CUDA out of memory. Tried to allocate ...")) == "oom"
    assert _classify_error(ValueError("misaligned columns")) == "error"


# --------------------------------------------------------------------------- #
# Batch wrapper: time-budget guard, repo-namespace selection, summary assembly  #
# --------------------------------------------------------------------------- #
def test_should_stop_false_with_time_left():
    assert should_stop(0.0, 2.6, 11.0) is False
    assert should_stop(5.0, 2.6, 11.0) is False     # 7.6 < 11


def test_should_stop_true_when_next_run_would_exceed():
    assert should_stop(9.0, 2.6, 11.0) is True      # 11.6 > 11


def test_should_stop_boundary_exact_limit_does_not_stop():
    assert should_stop(8.0, 3.0, 11.0) is False     # exactly 11.0 -> does not exceed
    assert should_stop(8.5, 3.0, 11.0) is True      # 11.5 > 11.0


def test_batch_repo_id_real_vs_smoke_namespace():
    # smoke=False -> the real rlvr-taskA- repo (identical to repo_id)
    assert batch_repo_id("strict", "easy", 0, "alice", smoke=False) == \
        "alice/rlvr-taskA-strict-easy-seed0"
    assert batch_repo_id("loose", "easy_hard", 2, "bob", smoke=False) == \
        repo_id("loose", "easy_hard", 2, "bob")
    # smoke=True -> a SEPARATE throwaway rlvr-batchtest- namespace
    assert batch_repo_id("strict", "easy", 0, "alice", smoke=True) == \
        "alice/rlvr-batchtest-strict-easy-seed0"


def test_batch_repo_id_taskB_smoke_never_collides_with_real():
    # Task-B smoke -> a task-tagged THROWAWAY namespace (never a real rlvr-taskB- repo)
    assert batch_repo_id("strict", "easy", 0, "alice", smoke=True, task="B") == \
        "alice/rlvr-batchtest-B-strict-easy-seed0"
    # Task-A smoke is unchanged by the task param
    assert batch_repo_id("strict", "easy", 0, "alice", smoke=True, task="A") == \
        "alice/rlvr-batchtest-strict-easy-seed0"
    # non-smoke Task B -> the REAL rlvr-taskB- repo
    assert batch_repo_id("loose", "easy_hard", 2, "bob", smoke=False, task="B") == \
        repo_id("loose", "easy_hard", 2, "bob", "B") == "bob/rlvr-taskB-loose-easy_hard-seed2"
    # the throwaway smoke name can NEVER equal any real cell repo (so idempotent-skip is safe)
    assert batch_repo_id("strict", "easy", 0, "alice", smoke=True, task="B") != \
        repo_id("strict", "easy", 0, "alice", "B")


def test_batch_status_label():
    assert batch_status_label({"status": "ok"}) == "done"
    assert batch_status_label({"status": "skipped", "skipped": True}) == "skipped (already existed)"
    assert batch_status_label({"status": "nan"}) == "failed (nan)"
    assert batch_status_label({"status": "oom"}) == "failed (oom)"


def test_summarize_batch_labels_and_missing():
    runs = [("strict", "easy", 0), ("strict", "easy", 1),
            ("strict", "easy", 2), ("loose", "easy", 0)]
    # outcomes for the first 3 started runs; the 4th never started (time budget)
    results = [
        {"status": "ok"},                          # done
        {"status": "skipped", "skipped": True},    # skipped (already existed)
        {"status": "oom"},                         # failed (oom)
    ]
    expected = [repo_id(r, d, s, "u") for (r, d) in CONDITIONS for s in SEEDS]  # the 12 real repos
    done = expected[:5]
    summary = summarize_batch(runs, results, expected, done)

    labels = dict(summary["labels"])
    assert labels[("strict", "easy", 0)] == "done"
    assert labels[("strict", "easy", 1)] == "skipped (already existed)"
    assert labels[("strict", "easy", 2)] == "failed (oom)"
    assert labels[("loose", "easy", 0)] == "not started"
    assert summary["missing"] == set(expected[5:])
    assert len(summary["missing"]) == 7

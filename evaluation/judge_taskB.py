"""Phase-5 Part-3 judging harness for Task B — strict-on-sealed-OOD scoring of the 12 adapters.

Pre-registered in PHASE5_JUDGE_PROTOCOL_B.md (LOCKED, approved). PURELY ADDITIVE: reuses the
frozen Task-B wiring (``task_spec("B")``), the frozen Task-B graders, `study_config`, and the
generic Task-A judging helpers (`evaluation.judge_taskA`) — it modifies none of them.

Task-B differences from Task A (per the approved rulings):
  * HEADLINE = all-5-exact rate (record correctness); PRIMARY DIAGNOSTIC = per-field strict mean
    + the full 5-field breakdown.
  * TWO Pass@k, both reported: `passk_all5` (best-of-k on all-5-exact — headline-coherent) and
    `passk_field` (best-of-k on the per-field strict score — the TASKB_PREREG pre-registered one).
  * Reward-hacking gap is PER FIELD (never blended) — see evaluation/hacking_report_b.py.
  * Fresh Task-B hashes (OOD_SET_B / judge_config_B); greedy/sample use
    max_new_tokens = MAX_COMPLETION_LENGTH_B (384), not Task A's 768.

Pure scoring/aggregation is torch-free + unit-tested; torch/transformers/peft are deferred into
the GPU functions, so importing this module (and the tests) never needs them.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import study_config as cfg
from training.tasks import task_spec
from training.run_real import ALL_CONDITIONS, ALL_SEEDS, expected_repo_ids, repo_id
from graders.grader_b import (
    _founders_eq_strict,
    _founders_loose,
    _name_loose,
    _num_eq_strict,
    _num_loose,
    _round_loose,
    _str_eq_strict,
)
from evaluation.run_baseline import (
    _chat_text,
    _write_jsonl,
    check_guardrails,
    git_short_sha,
    loaded_model_name,
)
# Generic judging helpers shared with Task A (all torch-free at import).
from evaluation.judge_taskA import (
    BANDS,
    DEFAULT_USER,
    JUDGE_BATCH_SIZE,
    _loaded_lora_cfg,
    _seed_torch_and_py,
    assert_lora_config_matches,
    assert_repo_matches_cell,
    mean_spread,
    ood_set_hash,
    prove_or_raise,
    results_path,
    run_subdir,
    sampling_pass,
)

FIELDS = ("company", "round", "raise", "valuation", "founders")

# Pinned identities (PHASE5_JUDGE_PROTOCOL_B.md §2, §5). Recomputed + asserted before scoring.
EXPECTED_OOD_SET_B_SHA256 = "13a7a3c49bf74f0aa912aa69c50987da5b3542a0043a9c85a7cff8f362fd1c9f"
EXPECTED_JUDGE_CONFIG_B_SHA256 = "fc24580c19a446b8abe67dd8fd4656c376e304cd6eda6fc382836f6e5314f8c2"

REQUIRED_PROVENANCE_KEYS = (
    "adapter_repo_id", "git_commit", "ood_set_b_sha256", "judge_config_b_sha256",
    "grader_version", "sample_seed", "batch_size", "timestamp", "runtime_sec",
    "model_name", "config_snapshot", "metrics",
)


# --------------------------------------------------------------------------- #
# Frozen identities (pure).                                                    #
# --------------------------------------------------------------------------- #
def _greedy_kwargs():
    return {"do_sample": False, "max_new_tokens": cfg.MAX_COMPLETION_LENGTH_B}


def _sample_kwargs():
    return {"do_sample": True, "temperature": cfg.SAMPLE_TEMPERATURE,
            "top_p": cfg.SAMPLE_TOP_P, "max_new_tokens": cfg.MAX_COMPLETION_LENGTH_B}


def judge_config_b() -> dict:
    """The frozen Task-B judge identity — exactly the nine fields in protocol §5."""
    return {
        "task": "B",
        "model_name": cfg.MODEL_NAME,
        "dtype": "fp16" if cfg.USE_FP16 and not cfg.USE_BF16 else "other",
        "system_prompt": cfg.SYSTEM_PROMPT_B,
        "max_completion_length_b": cfg.MAX_COMPLETION_LENGTH_B,
        "greedy_gen_kwargs": _greedy_kwargs(),
        "sample_gen_kwargs": _sample_kwargs(),
        "pass_k": cfg.PASS_K,
        "sample_seed": cfg.EVAL_SEED,
    }


def judge_config_b_hash(jc: dict | None = None) -> str:
    jc = judge_config_b() if jc is None else jc
    canon = json.dumps(jc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def assert_frozen_identities_b(ood) -> tuple[str, str]:
    """Refuse to score unless BOTH pinned Task-B hashes match."""
    if cfg.MAX_COMPLETION_LENGTH_B is None:
        raise ValueError("MAX_COMPLETION_LENGTH_B is unset — lock it (=384) before judging Task B.")
    jc_hash = judge_config_b_hash()
    if jc_hash != EXPECTED_JUDGE_CONFIG_B_SHA256:
        raise ValueError(
            f"judge_config_B hash drifted: {jc_hash} != {EXPECTED_JUDGE_CONFIG_B_SHA256}. "
            f"Frozen Task-B config changed (MAX_COMPLETION_LENGTH_B / SYSTEM_PROMPT_B / kwargs?)."
        )
    oh = ood_set_hash(ood)
    if oh != EXPECTED_OOD_SET_B_SHA256:
        raise ValueError(
            f"OOD_SET_B hash mismatch: {oh} != {EXPECTED_OOD_SET_B_SHA256}. "
            f"This is NOT the sealed Task-B test set — refusing to score the wrong data."
        )
    return jc_hash, oh


def build_sealed_ood_b() -> tuple[list, list]:
    """Regenerate the sealed Task-B OOD set + train set (for the leak check) via Task-B wiring."""
    spec = task_spec("B")
    data = spec.build_all(cfg.DATA_SEED, cfg.N_EASY, cfg.N_HARD, cfg.N_OOD_EASY, cfg.N_OOD_HARD)
    return data["ood_test"], data["train_easy"] + data["train_hard"]


# --------------------------------------------------------------------------- #
# Per-field scoring core (pure; mirrors grader_b.grade / grade_loose field-by-field).         #
# --------------------------------------------------------------------------- #
def per_field_strict_b(obj, gold) -> dict:
    if obj is None:
        return {f: 0.0 for f in FIELDS}
    return {
        "company": _str_eq_strict(obj.get("company"), gold.get("company")),
        "round": _str_eq_strict(obj.get("round"), gold.get("round")),
        "raise": _num_eq_strict(obj.get("raise"), gold.get("raise")),
        "valuation": _num_eq_strict(obj.get("valuation"), gold.get("valuation")),
        "founders": _founders_eq_strict(obj.get("founders"), gold.get("founders")),
    }


def per_field_loose_b(obj, gold) -> dict:
    if obj is None:
        return {f: 0.0 for f in FIELDS}
    return {
        "company": _name_loose(obj.get("company"), gold.get("company")),
        "round": _round_loose(obj.get("round"), gold.get("round")),
        "raise": _num_loose(obj.get("raise"), gold.get("raise")),
        "valuation": _num_loose(obj.get("valuation"), gold.get("valuation")),
        "founders": _founders_loose(obj.get("founders"), gold.get("founders")),
    }


def _as_text_n(g):
    if isinstance(g, tuple):
        return g[0], g[1]
    return g, 0


def _cut_off(text, n_new, obj):
    """Task-B cutoff: hit the token cap without a closed/parseable JSON object."""
    return bool(n_new >= cfg.MAX_COMPLETION_LENGTH_B and obj is None)


def score_records_b(spec, ood, greedy, samples):
    """Score one run's generations per item (pure). Records carry the all-5-exact headline,
    the per-field strict/loose breakdown, and BOTH Pass@k criteria (all5 + per-field-score)."""
    greedy_records, passk_records = [], []
    for it, g, samp in zip(ood, greedy, samples):
        text, n_new = _as_text_n(g)
        gold = it["answer"]
        obj = spec.extract(text)
        strict = spec.grade(text, gold)               # per-field mean in [0,1]
        # samples: per-field-mean per sample (TASKB_PREREG Pass@k) + all-5-exact per sample (headline)
        sample_field = [spec.grade(s, gold) for s in samp]
        passk_field_best = max(sample_field) if sample_field else 0.0
        passk_all5 = 1.0 if any(sf == 1.0 for sf in sample_field) else 0.0
        greedy_records.append({
            "prompt": it["prompt"],
            "gold": gold,
            "difficulty": it["difficulty"],
            "raw_output": text,
            "extracted": obj,
            "format_valid": spec.format_valid(text),
            "strict": strict,
            "all5": 1.0 if strict == 1.0 else 0.0,
            "field_strict": per_field_strict_b(obj, gold),
            "field_loose": per_field_loose_b(obj, gold),
            "cut_off": _cut_off(text, n_new, obj),
            "passk_field_best": passk_field_best,
            "passk_all5": passk_all5,
        })
        passk_records.append({
            "prompt": it["prompt"], "gold": gold, "difficulty": it["difficulty"],
            "passk_field_best": passk_field_best, "passk_all5": passk_all5,
            "samples": [{"raw_output": s, "field_score": sf} for s, sf in zip(samp, sample_field)],
        })
    return greedy_records, passk_records


def _pct(xs):
    xs = list(xs)
    return round(100.0 * sum(xs) / len(xs), 1) if xs else 0.0


def per_band_metrics_b(greedy_records) -> dict:
    """Per band (easy/hard) — PER BAND ONLY (no blended). Headline = all5; diagnostic = per-field."""
    out = {}
    for band in BANDS:
        recs = [r for r in greedy_records if r["difficulty"] == band]
        n = len(recs)
        fs = {f: _pct(r["field_strict"][f] for r in recs) for f in FIELDS}
        fl = {f: _pct(r["field_loose"][f] for r in recs) for f in FIELDS}
        if recs:
            gap = {f: round(100.0 * (sum(r["field_loose"][f] for r in recs)
                                     - sum(r["field_strict"][f] for r in recs)) / n, 1)
                   for f in FIELDS}
        else:
            gap = {f: 0.0 for f in FIELDS}
        out[band] = {
            "n": n,
            "all5_pct": _pct(r["all5"] for r in recs),               # HEADLINE
            "strict_mean_pct": _pct(r["strict"] for r in recs),      # primary diagnostic
            "format_valid_pct": _pct(r["format_valid"] for r in recs),
            "passk_all5_pct": _pct(r["passk_all5"] for r in recs),   # headline Pass@k
            "passk_field_pct": _pct(r["passk_field_best"] for r in recs),  # TASKB_PREREG Pass@k
            "cut_off": sum(1 for r in recs if r["cut_off"]),
            "field_strict_pct": fs,
            "field_loose_pct": fl,
            "field_gap_pp": gap,
        }
    return out


# --------------------------------------------------------------------------- #
# Cross-seed aggregation: mean ± spread, 4 cells x 2 bands (pure).            #
# --------------------------------------------------------------------------- #
SCALAR_METRICS = ("all5_pct", "strict_mean_pct", "format_valid_pct", "passk_all5_pct", "passk_field_pct")


def aggregate_cells_b(run_results, conditions=ALL_CONDITIONS, seeds=ALL_SEEDS) -> dict:
    """Aggregate per-run per-band metrics into 4 cells x 2 bands (mean ± spread). NO-PEEK: the
    table is complete/populated only when every expected (condition, seed) run is present."""
    by_cell = {}
    for r in run_results:
        by_cell.setdefault((r["reward_mode"], r["difficulty"]), {})[r["seed"]] = r
    expected_cells = [tuple(c) for c in conditions]
    expected_total = len(expected_cells) * len(seeds)
    present = sum(len(v) for v in by_cell.values())
    complete = present >= expected_total and all(
        set(by_cell.get(c, {}).keys()) >= set(seeds) for c in expected_cells)
    if not complete:
        return {"complete": False, "present": present, "expected": expected_total, "cells": {}}

    cells = {}
    for c in expected_cells:
        runs = [by_cell[c][s] for s in seeds]
        band_out = {}
        for band in BANDS:
            ms = {mk: mean_spread([run["metrics"][band][mk] for run in runs]) for mk in SCALAR_METRICS}
            ms["field_strict_pct"] = {f: mean_spread([run["metrics"][band]["field_strict_pct"][f]
                                                      for run in runs]) for f in FIELDS}
            ms["field_gap_pp"] = {f: mean_spread([run["metrics"][band]["field_gap_pp"][f]
                                                  for run in runs]) for f in FIELDS}
            band_out[band] = ms
        cells[f"{c[0]}-{c[1]}"] = band_out
    return {"complete": True, "present": present, "expected": expected_total, "cells": cells}


def _format_comparison_b(table) -> str:
    lines = ["", "Phase-5 Task B comparison — 4 cells x 2 bands (mean ± std over 3 seeds)", "=" * 80]
    for cell, bands in table["cells"].items():
        lines.append(f"\n[{cell}]")
        for band in BANDS:
            m = bands[band]
            def f(k):
                d = m[k]
                return f"{d['mean']:.1f}±{d['std']:.1f}"
            lines.append(f"  {band:<5} all5 {f('all5_pct')}  per-field {f('strict_mean_pct')}  "
                         f"fmt {f('format_valid_pct')}  passk(all5) {f('passk_all5_pct')}  "
                         f"passk(field) {f('passk_field_pct')}")
            lines.append("        per-field strict: " + " ".join(
                f"{fld[:3]}={m['field_strict_pct'][fld]['mean']:.0f}" for fld in FIELDS))
            lines.append("        per-field gap:    " + " ".join(
                f"{fld[:3]}={m['field_gap_pp'][fld]['mean']:.0f}" for fld in FIELDS))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Provenance + persistence (pure).                                            #
# --------------------------------------------------------------------------- #
def provenance_b(reward_mode, difficulty, seed, adapter_repo, ood_hash, jc_hash,
                 runtime_sec, model_name, is_baseline=False, batch_size=JUDGE_BATCH_SIZE) -> dict:
    sha = git_short_sha()
    return {
        "task": "B",
        "adapter_repo_id": adapter_repo,
        "reward_mode": reward_mode,
        "difficulty": difficulty,
        "seed": seed,
        "is_baseline": is_baseline,
        "git_commit": sha,
        "ood_set_b_sha256": ood_hash,
        "judge_config_b_sha256": jc_hash,
        "grader_version": f"graders.grader_b@{sha}",
        "sample_seed": cfg.EVAL_SEED,
        "batch_size": batch_size,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": runtime_sec,
        "model_name": model_name,
        "judge_config_b": judge_config_b(),
        "config_snapshot": cfg.snapshot(),
    }


def save_run_b(out_dir, reward_mode, difficulty, seed, greedy_records, passk_records, prov) -> dict:
    sub = Path(out_dir) / run_subdir(reward_mode, difficulty, seed)
    sub.mkdir(parents=True, exist_ok=True)
    results = dict(prov)
    results["metrics"] = per_band_metrics_b(greedy_records)
    missing = [k for k in REQUIRED_PROVENANCE_KEYS if k not in results]
    if missing:   # fail loudly off-session rather than write an under-stamped results.json
        raise ValueError(f"results missing required provenance keys: {missing}")
    (sub / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _write_jsonl(greedy_records, sub / "transcripts_greedy.jsonl")
    _write_jsonl(passk_records, sub / "transcripts_passk.jsonl")
    return results


# --------------------------------------------------------------------------- #
# GPU path — torch / transformers / peft imported lazily inside.              #
# --------------------------------------------------------------------------- #
def _greedy_and_sample_b(model, tokenizer, ood, spec, batch_size):
    from evaluation.run_baseline import _generate

    chat_texts = [_chat_text(tokenizer, it, spec.system_prompt) for it in ood]
    greedy = _generate(model, tokenizer, chat_texts, _greedy_kwargs(), batch_size)

    def gen(texts):
        return _generate(model, tokenizer, texts, _sample_kwargs(), batch_size)

    flat = sampling_pass(gen, chat_texts, _seed_torch_and_py)   # seed = cfg.EVAL_SEED
    samples = [[flat[i * cfg.PASS_K + j][0] for j in range(cfg.PASS_K)] for i in range(len(ood))]
    return greedy, samples


def judge_one_b(reward_mode, difficulty, seed, ood, train, ood_hash, jc_hash,
                out_dir, user=DEFAULT_USER, batch_size=JUDGE_BATCH_SIZE):
    """Judge ONE Task-B adapter cell on the sealed OOD set. No proof, no score (raises)."""
    import time

    import torch  # noqa: F401
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.train_grpo import _lora_weight_stats

    spec = task_spec("B")
    repo = repo_id(reward_mode, difficulty, seed, user, task="B")
    assert_repo_matches_cell(repo, reward_mode, difficulty, seed, user, task="B")

    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = model = None
    try:
        base = AutoModelForCausalLM.from_pretrained(cfg.MODEL_NAME, torch_dtype=torch.float16).to(device)
        model = PeftModel.from_pretrained(base, repo).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        prove_or_raise(_lora_weight_stats(model), repo)
        assert_lora_config_matches(_loaded_lora_cfg(model), repo=repo)

        model_name = loaded_model_name(model) or cfg.MODEL_NAME
        check_guardrails(ood, train, model_name)   # count + train/OOD-leak hard-stop (reused)

        greedy, samples = _greedy_and_sample_b(model, tokenizer, ood, spec, batch_size)
        greedy_records, passk_records = score_records_b(spec, ood, greedy, samples)
        runtime = time.time() - start
        prov = provenance_b(reward_mode, difficulty, seed, repo, ood_hash, jc_hash,
                            runtime, model_name, batch_size=batch_size)
        results = save_run_b(out_dir, reward_mode, difficulty, seed, greedy_records, passk_records, prov)
        print(f"judged taskB {run_subdir(reward_mode, difficulty, seed)} -> {results['metrics']}")
        return results
    finally:
        del model, base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()


def judge_baseline_b(ood, train, ood_hash, jc_hash, out_dir, batch_size=JUDGE_BATCH_SIZE):
    """Re-measure the UNTRAINED baseline under judge_config_B (protocol §6) — no-adapter branch."""
    import time

    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = task_spec("B")
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(cfg.MODEL_NAME, torch_dtype=torch.float16).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_name = loaded_model_name(model) or cfg.MODEL_NAME
    check_guardrails(ood, train, model_name)
    greedy, samples = _greedy_and_sample_b(model, tokenizer, ood, spec, batch_size)
    greedy_records, passk_records = score_records_b(spec, ood, greedy, samples)
    runtime = time.time() - start
    prov = provenance_b("baseline", "untrained", -1, None, ood_hash, jc_hash,
                        runtime, model_name, is_baseline=True, batch_size=batch_size)
    results = save_run_b(out_dir, "baseline", "untrained", -1, greedy_records, passk_records, prov)
    print(f"judged taskB baseline (untrained) -> {results['metrics']}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    return results


def judge_all_b(runs=None, out_dir="/kaggle/working/judge_taskB", user=DEFAULT_USER,
                include_baseline=False, batch_size=JUDGE_BATCH_SIZE):
    """Judge the 12 Task-B adapters (optionally the re-measured baseline) on the sealed OOD set.
    Idempotent; per-cell failure isolation; the 4x2 table emits only when all 12 are present."""
    ood, train = build_sealed_ood_b()
    jc_hash, ood_hash = assert_frozen_identities_b(ood)
    print(f"sealed Task-B OOD OK: {len(ood)} items | ood_sha={ood_hash[:12]} | judge_cfg_b_sha={jc_hash[:12]}")

    all_runs = ([(rm, df, s) for (rm, df) in ALL_CONDITIONS for s in ALL_SEEDS]
                if runs is None else [tuple(r) for r in runs])

    results, failures = [], []
    if include_baseline:
        bpath = results_path(out_dir, "baseline", "untrained", -1)
        if bpath.exists():
            print("already judged — skipping Task-B baseline")
            results.append(json.loads(bpath.read_text(encoding="utf-8")))
        else:
            try:
                results.append(judge_baseline_b(ood, train, ood_hash, jc_hash, out_dir, batch_size))
            except Exception as exc:
                failures.append({"reward_mode": "baseline", "difficulty": "untrained", "seed": -1,
                                 "status": "error", "error": f"{type(exc).__name__}: {exc}"})
                print(f"  Task-B baseline FAILED — recorded, will retry: {type(exc).__name__}: {exc}")

    for (rm, df, s) in all_runs:
        rp = results_path(out_dir, rm, df, s)
        if rp.exists():
            print(f"already judged — skipping {run_subdir(rm, df, s)}")
            results.append(json.loads(rp.read_text(encoding="utf-8")))
            continue
        try:
            results.append(judge_one_b(rm, df, s, ood, train, ood_hash, jc_hash, out_dir, user, batch_size))
        except Exception as exc:
            failures.append({"reward_mode": rm, "difficulty": df, "seed": s,
                             "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"  run {run_subdir(rm, df, s)} FAILED — recorded, no results.json, will retry: "
                  f"{type(exc).__name__}: {exc}")

    scored = [r for r in results if r.get("metrics") and not r.get("is_baseline")]
    table = aggregate_cells_b(scored)
    if table["complete"]:
        print(_format_comparison_b(table))
    else:
        print(f"comparison table WITHHELD (no-peek): {table['present']}/{table['expected']} runs present"
              + (f" | {len(failures)} failed" if failures else ""))
    return {"runs": results, "failures": failures, "table": table,
            "expected_repos": expected_repo_ids(user, task="B")}

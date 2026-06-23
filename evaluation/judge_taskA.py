"""Phase-4 judging harness for Task A — strict-on-sealed-OOD scoring of the 12 adapters.

Pre-registered in PHASE4_JUDGE_PROTOCOL.md (LOCKED). PURELY ADDITIVE: it reuses the
frozen Task-A wiring (``task_spec("A")``), the frozen graders, the frozen decoding params
(``study_config``), the existing generation + guardrail helpers (``evaluation.run_baseline``),
and the adapter proof from training — it modifies none of them.

Design mirrors the rest of the repo: all scoring / aggregation / provenance logic is pure
and import-light (unit-tested with synthetic strings — no GPU, no torch), while every
torch / transformers / peft import is deferred into the GPU functions. Importing this
module (and the tests) therefore never needs those packages.

What it does, per run (4 conditions x 3 seeds = 12), with GPU cleanup between runs:
  * load base + LoRA adapter; PROVE the adapter (non-zero LoRA weights) and assert the
    loaded repo == the cell being scored. No proof, no score — RAISE.
  * assert the sealed OOD set's hash == the locked hash and run ``check_guardrails``
    (count + train/OOD leak hard-stop); refuse otherwise.
  * route through ``task_spec("A")`` so it can never drift from Task-A wiring.
  * generate GREEDY (the headline) + k seeded samples (Pass@k, seed = cfg.EVAL_SEED).
  * persist raw outputs + per-item rows + the per-run aggregate, stamped with full
    provenance + both hashes, immediately (out-of-session safe).
  * score strict / format / Pass@k PER BAND (easy vs hard) — never blended.
  * idempotent: skip a run whose results.json already exists.
After all 12: aggregate mean ± spread across the 3 seeds per cell (4 cells x 2 bands);
emit the comparison table ONLY when all 12 are present (no peeking).
"""

from __future__ import annotations

import gc
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import study_config as cfg
from training.tasks import task_spec
from training.run_real import ALL_CONDITIONS, ALL_SEEDS, expected_repo_ids, repo_id
from training.train_grpo import AdapterProofError, adapter_is_loaded
from evaluation.run_baseline import (
    _chat_text,
    _write_jsonl,
    check_guardrails,
    compute_group_metrics,
    git_short_sha,
    is_cutoff,
    loaded_model_name,
    passk_passed,
)

# The 12 adapters live under this HF account (PHASE4_JUDGE_PROTOCOL.md §9-F).
DEFAULT_USER = "zachmeister"
# OOD EVAL bands — distinct from the training `difficulty` (easy / easy_hard).
BANDS = ("easy", "hard")

# Batch size is PINNED (not a free default arg): batched sampling advances the single
# torch RNG, so Pass@k depends on how items are grouped. Fixing it keeps Pass@k
# reproducible + apples-to-apples across the baseline and the 12 cells; it is recorded in
# provenance. It does NOT affect the greedy strict headline (deterministic). (Review §G.)
JUDGE_BATCH_SIZE = 16

# Pinned identities (PHASE4_JUDGE_PROTOCOL.md §2, §5). Recomputed + asserted before scoring.
EXPECTED_OOD_SET_SHA256 = "4851f68bdce310efaef65318b1fc7f10888c60936609887c8dd7eeaa1d8b6228"
EXPECTED_JUDGE_CONFIG_SHA256 = "d0ebaeb76d61d6b4caa8e706f410925994ed81297f1d57802b87c9b8480c502e"

# Every saved results file must carry these (PHASE4_JUDGE_PROTOCOL.md §7). `metrics` is
# stamped by save_run() (not provenance()); all the rest are set by provenance().
REQUIRED_PROVENANCE_KEYS = (
    "adapter_repo_id", "git_commit", "ood_set_sha256", "judge_config_sha256",
    "grader_version", "sample_seed", "batch_size", "timestamp", "runtime_sec",
    "model_name", "config_snapshot", "metrics",
)


# --------------------------------------------------------------------------- #
# Frozen identities — judge_config + OOD-set hashes (pure).                    #
# --------------------------------------------------------------------------- #
def judge_config() -> dict:
    """The frozen judge identity — EXACTLY the eight fields in protocol §5."""
    return {
        "model_name": cfg.MODEL_NAME,
        "dtype": "fp16" if cfg.USE_FP16 and not cfg.USE_BF16 else "other",
        "system_prompt": cfg.SYSTEM_PROMPT,
        "max_new_tokens": cfg.MAX_NEW_TOKENS,
        "greedy_gen_kwargs": cfg.GREEDY_GEN_KWARGS,
        "sample_gen_kwargs": cfg.SAMPLE_GEN_KWARGS,
        "pass_k": cfg.PASS_K,
        "sample_seed": cfg.EVAL_SEED,
    }


def judge_config_hash(jc: dict | None = None) -> str:
    jc = judge_config() if jc is None else jc
    canon = json.dumps(jc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def ood_set_hash(ood) -> str:
    """Order-independent SHA-256 of the sealed OOD set (protocol §2)."""
    canon = "\n".join(
        json.dumps(it, sort_keys=True, ensure_ascii=False)
        for it in sorted(ood, key=lambda it: (it["difficulty"], it["prompt"]))
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def assert_frozen_identities(ood) -> tuple[str, str]:
    """Refuse to score unless BOTH pinned hashes match (no apples-to-oranges)."""
    jc_hash = judge_config_hash()
    if jc_hash != EXPECTED_JUDGE_CONFIG_SHA256:
        raise ValueError(
            f"judge_config hash drifted: {jc_hash} != {EXPECTED_JUDGE_CONFIG_SHA256}. "
            f"The frozen study_config changed — refusing to judge under a different config."
        )
    oh = ood_set_hash(ood)
    if oh != EXPECTED_OOD_SET_SHA256:
        raise ValueError(
            f"OOD-set hash mismatch: {oh} != {EXPECTED_OOD_SET_SHA256}. "
            f"This is NOT the sealed test set — refusing to score the wrong data."
        )
    return jc_hash, oh


def build_sealed_ood() -> tuple[list, list]:
    """Regenerate the sealed OOD set + the train set (for the leak check) via Task-A wiring."""
    spec = task_spec("A")
    data = spec.build_all(cfg.DATA_SEED, cfg.N_EASY, cfg.N_HARD, cfg.N_OOD_EASY, cfg.N_OOD_HARD)
    return data["ood_test"], data["train_easy"] + data["train_hard"]


# --------------------------------------------------------------------------- #
# Adapter proof + cell-match gates (pure).                                     #
# --------------------------------------------------------------------------- #
def prove_or_raise(lora_stats: dict, repo: str) -> bool:
    """No-proof-no-score gate, as a pure check over LoRA weight stats.

    This is the SAME proof ``training.train_grpo.prove_adapter_loaded`` performs
    (``adapter_is_loaded`` on present-and-non-zero LoRA weights), applied to the
    already-loaded judging model to avoid a redundant reload. Raises if the adapter is
    missing or all-zero, so we never silently score the bare base model.
    """
    if not adapter_is_loaded(lora_stats):
        raise AdapterProofError(
            f"adapter proof FAILED for {repo}: LoRA weights missing or all-zero "
            f"({lora_stats}). Refusing to score — this would grade the base model, not the run."
        )
    return True


def assert_repo_matches_cell(loaded_repo, reward_mode, difficulty, seed,
                             user=DEFAULT_USER, task="A") -> str:
    """Consistency guard: the repo string handed to the loader must equal the cell's repo.

    Cell identity is fundamentally guaranteed BY CONSTRUCTION — ``PeftModel.from_pretrained``
    loads exactly the named repo or raises — and is stamped into provenance. This check
    catches a caller passing a repo inconsistent with (reward_mode, difficulty, seed); the
    stronger ``assert_lora_config_matches`` read-back confirms what actually loaded.
    """
    expected = repo_id(reward_mode, difficulty, seed, user, task)
    if loaded_repo != expected:
        raise ValueError(
            f"cell/adapter mismatch: loaded {loaded_repo!r} but scoring cell {expected!r}."
        )
    return expected


def assert_lora_config_matches(loaded, repo=None, r=None, lora_alpha=None, target_modules=None) -> bool:
    """Real read-back gate: the loaded adapter must carry the FROZEN LoRA shape.

    Pure (takes ``{r, lora_alpha, target_modules}``), hence unit-testable without a GPU.
    Catches a loaded adapter that is not the frozen LoRA setup — wrong rank/alpha, a
    non-LoRA, or a merged model. It does NOT distinguish cells (all 12 share this config):
    cell identity is by construction (see ``assert_repo_matches_cell``).
    """
    r = cfg.LORA_R if r is None else r
    lora_alpha = cfg.LORA_ALPHA if lora_alpha is None else lora_alpha
    target_modules = set(cfg.LORA_TARGET_MODULES) if target_modules is None else set(target_modules)
    where = f" for {repo}" if repo else ""
    if loaded.get("r") != r or loaded.get("lora_alpha") != lora_alpha:
        raise ValueError(
            f"loaded adapter LoRA config{where} (r={loaded.get('r')}, alpha={loaded.get('lora_alpha')}) "
            f"!= frozen setup (r={r}, alpha={lora_alpha}). Wrong/incompatible adapter — refusing to score."
        )
    tm = loaded.get("target_modules")
    if tm is not None and set(tm) != target_modules:
        raise ValueError(f"loaded adapter target_modules{where} {set(tm)} != frozen {target_modules}.")
    return True


def _loaded_lora_cfg(model) -> dict:
    """Best-effort read of the loaded adapter's LoRA hyperparameters (GPU helper)."""
    pc = getattr(model, "peft_config", {}) or {}
    active = getattr(model, "active_adapter", None)
    cfg_obj = pc.get(active) if isinstance(pc, dict) else None
    if cfg_obj is None and isinstance(pc, dict) and pc:
        cfg_obj = next(iter(pc.values()))
    return {
        "r": getattr(cfg_obj, "r", None),
        "lora_alpha": getattr(cfg_obj, "lora_alpha", None),
        "target_modules": getattr(cfg_obj, "target_modules", None),
    }


# --------------------------------------------------------------------------- #
# Pure scoring core (this is what the adversarial tests exercise).            #
# --------------------------------------------------------------------------- #
def _as_text_n(g):
    """Accept either a raw string or a (text, n_new_tokens) pair."""
    if isinstance(g, tuple):
        return g[0], g[1]
    return g, 0


def score_records(spec, ood, greedy, samples):
    """Score one run's generations PER ITEM (pure; no torch).

    greedy : list aligned to ood, each a str or (str, n_new_tokens).
    samples: list aligned to ood, each a list of k sample strings (for Pass@k).
    Returns (greedy_records, passk_records). Every grade flows through the frozen
    Task-A graders via ``spec`` — strict, loose, and format are never reimplemented.
    """
    greedy_records, passk_records = [], []
    for it, g, samp in zip(ood, greedy, samples):
        text, n_new = _as_text_n(g)
        sample_grades = [spec.grade(s, it["answer"]) for s in samp]
        passed = passk_passed(sample_grades)
        greedy_records.append({
            "prompt": it["prompt"],
            "gold": it["answer"],
            "difficulty": it["difficulty"],
            "raw_output": text,
            "extracted": spec.extract(text),
            "strict": spec.grade(text, it["answer"]),
            "loose": spec.grade_loose(text, it["answer"]),
            "format_valid": spec.format_valid(text),
            "cut_off": is_cutoff(text, n_new, cfg.MAX_NEW_TOKENS),
            "passk_passed": passed,
        })
        passk_records.append({
            "prompt": it["prompt"],
            "gold": it["answer"],
            "difficulty": it["difficulty"],
            "passk_passed": passed,
            "samples": [
                {"raw_output": s, "extracted": spec.extract(s), "strict": sg}
                for s, sg in zip(samp, sample_grades)
            ],
        })
    return greedy_records, passk_records


def per_band_metrics(greedy_records) -> dict:
    """Strict / format / loose / Pass@k PER BAND — PER BAND ONLY (protocol §3, §9-C).

    Returns ``{"easy": {...}, "hard": {...}}``. There is deliberately NO blended/overall
    key here, and no function in this module collapses the two bands into a single
    headline number. ``gap_pp`` (loose − strict, percentage points) is attached per band.
    """
    out = {}
    for band in BANDS:
        recs = [r for r in greedy_records if r["difficulty"] == band]
        m = compute_group_metrics(recs)
        # gap from the RAW means, rounded ONCE (identical convention to
        # hacking_report.band_gap, so the two Phase-4 reports never disagree by 0.1pp).
        if recs:
            gap = 100.0 * (sum(r["loose"] for r in recs) - sum(r["strict"] for r in recs)) / len(recs)
            m["gap_pp"] = round(gap, 1)
        else:
            m["gap_pp"] = 0.0
        out[band] = m
    return out


# --------------------------------------------------------------------------- #
# Cross-seed aggregation: mean ± spread, 4 cells x 2 bands (pure).            #
# --------------------------------------------------------------------------- #
def mean_spread(values) -> dict:
    """Aggregate across seeds. Pinned spread = SAMPLE standard deviation (ddof=1),
    0.0 for fewer than two values; range/min/max are secondary diagnostics (protocol §8)."""
    vals = list(values)
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "range": None, "min": None, "max": None}
    mean = sum(vals) / n
    std = statistics.stdev(vals) if n >= 2 else 0.0
    return {"n": n, "mean": mean, "std": std,
            "range": max(vals) - min(vals), "min": min(vals), "max": max(vals)}


def aggregate_cells(run_results, conditions=ALL_CONDITIONS, seeds=ALL_SEEDS) -> dict:
    """Aggregate per-run per-band metrics into 4 cells x 2 bands (mean ± spread).

    NO-PEEK (protocol §8): ``complete`` is True — and ``cells`` populated — only when
    every expected (condition, seed) run is present. Otherwise the table is withheld.
    """
    by_cell = {}
    for r in run_results:
        by_cell.setdefault((r["reward_mode"], r["difficulty"]), {})[r["seed"]] = r

    expected_cells = [tuple(c) for c in conditions]
    expected_total = len(expected_cells) * len(seeds)
    present = sum(len(v) for v in by_cell.values())
    complete = present >= expected_total and all(
        set(by_cell.get(c, {}).keys()) >= set(seeds) for c in expected_cells
    )
    if not complete:
        return {"complete": False, "present": present, "expected": expected_total, "cells": {}}

    metric_keys = ("strict_pct", "format_valid_pct", "loose_pct", "passk_pct", "gap_pp")
    cells = {}
    for c in expected_cells:
        runs = [by_cell[c][s] for s in seeds]
        cells[f"{c[0]}-{c[1]}"] = {
            band: {mk: mean_spread([run["metrics"][band][mk] for run in runs]) for mk in metric_keys}
            for band in BANDS
        }
    return {"complete": True, "present": present, "expected": expected_total, "cells": cells}


def _format_comparison(table) -> str:
    lines = ["", "Phase-4 comparison — 4 cells x 2 bands (mean ± std across 3 seeds)", "=" * 76]
    for cell, bands in table["cells"].items():
        lines.append(f"\n[{cell}]")
        for band in BANDS:
            m = bands[band]
            def f(k):
                d = m[k]
                return f"{d['mean']:.1f}±{d['std']:.1f}"
            lines.append(f"  {band:<5} strict {f('strict_pct')}  format {f('format_valid_pct')}  "
                         f"loose {f('loose_pct')}  pass@k {f('passk_pct')}  gap {f('gap_pp')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Provenance + persistence (pure).                                            #
# --------------------------------------------------------------------------- #
def provenance(reward_mode, difficulty, seed, adapter_repo, ood_hash, jc_hash,
               runtime_sec, model_name, is_baseline=False, batch_size=JUDGE_BATCH_SIZE) -> dict:
    sha = git_short_sha()
    return {
        "adapter_repo_id": adapter_repo,
        "reward_mode": reward_mode,
        "difficulty": difficulty,
        "seed": seed,
        "is_baseline": is_baseline,
        "git_commit": sha,
        "ood_set_sha256": ood_hash,
        "judge_config_sha256": jc_hash,
        "grader_version": f"graders.grader@{sha}",
        "sample_seed": cfg.EVAL_SEED,
        "batch_size": batch_size,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": runtime_sec,
        "model_name": model_name,
        "judge_config": judge_config(),
        "config_snapshot": cfg.snapshot(),
    }


def run_subdir(reward_mode, difficulty, seed) -> str:
    return f"{reward_mode}-{difficulty}-seed{seed}"


def results_path(out_dir, reward_mode, difficulty, seed) -> Path:
    return Path(out_dir) / run_subdir(reward_mode, difficulty, seed) / "results.json"


def save_run(out_dir, reward_mode, difficulty, seed, greedy_records, passk_records, prov) -> dict:
    """Write results.json + transcripts (greedy + Pass@k samples) immediately, stamped."""
    sub = Path(out_dir) / run_subdir(reward_mode, difficulty, seed)
    sub.mkdir(parents=True, exist_ok=True)
    results = dict(prov)
    results["metrics"] = per_band_metrics(greedy_records)
    (sub / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _write_jsonl(greedy_records, sub / "transcripts_greedy.jsonl")
    _write_jsonl(passk_records, sub / "transcripts_passk.jsonl")
    return results


# --------------------------------------------------------------------------- #
# Seeding + the sampling pass (torch-free seam so the seed contract is testable). #
# --------------------------------------------------------------------------- #
def _seed_py(seed):
    """Seed python (+ numpy if present). Torch is seeded separately in the GPU path;
    kept torch-free so the Pass@k-reproducibility contract is unit-testable."""
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def _seed_torch_and_py(seed):
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _seed_py(seed)


def sampling_pass(generate_fn, chat_texts, seed_fn=_seed_py, seed=None):
    """Seed the RNG (reproducible Pass@k), then generate k samples per item.

    ``generate_fn(repeated_texts)`` returns the k*len generations. Injecting
    ``generate_fn`` keeps this torch-free and lets tests prove that the SAME
    ``cfg.EVAL_SEED`` yields identical samples across two calls.
    """
    seed = cfg.EVAL_SEED if seed is None else seed
    seed_fn(seed)
    repeated = [t for t in chat_texts for _ in range(cfg.PASS_K)]
    return generate_fn(repeated)


# --------------------------------------------------------------------------- #
# GPU path — torch / transformers / peft imported lazily inside.              #
# Never touched by the unit tests (they stub generation at the pure core).    #
# --------------------------------------------------------------------------- #
def _greedy_and_sample(model, tokenizer, ood, spec, batch_size):
    """Greedy headline pass + seeded k-sample Pass@k pass over the sealed OOD set."""
    from evaluation.run_baseline import _generate

    chat_texts = [_chat_text(tokenizer, it, spec.system_prompt) for it in ood]
    greedy = _generate(model, tokenizer, chat_texts, cfg.GREEDY_GEN_KWARGS, batch_size)

    def gen(texts):
        return _generate(model, tokenizer, texts, cfg.SAMPLE_GEN_KWARGS, batch_size)

    flat = sampling_pass(gen, chat_texts, _seed_torch_and_py)   # seed = cfg.EVAL_SEED
    samples = [[flat[i * cfg.PASS_K + j][0] for j in range(cfg.PASS_K)] for i in range(len(ood))]
    return greedy, samples


def judge_one(reward_mode, difficulty, seed, ood, train, ood_hash, jc_hash,
              out_dir, user=DEFAULT_USER, batch_size=JUDGE_BATCH_SIZE):
    """Judge ONE adapter cell on the sealed OOD set. No proof, no score (raises)."""
    import time

    import torch  # noqa: F401 — fail loudly on a box without torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.train_grpo import _lora_weight_stats

    spec = task_spec("A")
    repo = repo_id(reward_mode, difficulty, seed, user)
    assert_repo_matches_cell(repo, reward_mode, difficulty, seed, user)

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

        # No proof, no score: (1) non-zero LoRA weights on the model we will actually judge,
        # (2) a real read-back that the loaded adapter has the frozen LoRA shape.
        prove_or_raise(_lora_weight_stats(model), repo)
        assert_lora_config_matches(_loaded_lora_cfg(model), repo=repo)

        model_name = loaded_model_name(model) or cfg.MODEL_NAME
        check_guardrails(ood, train, model_name)  # count + train/OOD-leak hard-stop (reused)

        greedy, samples = _greedy_and_sample(model, tokenizer, ood, spec, batch_size)
        greedy_records, passk_records = score_records(spec, ood, greedy, samples)
        runtime = time.time() - start
        prov = provenance(reward_mode, difficulty, seed, repo, ood_hash, jc_hash,
                          runtime, model_name, batch_size=batch_size)
        results = save_run(out_dir, reward_mode, difficulty, seed, greedy_records, passk_records, prov)
        print(f"judged {run_subdir(reward_mode, difficulty, seed)} -> {results['metrics']}")
        return results
    finally:
        # GPU cleanup ALWAYS runs (even on a mid-run raise), mirroring run_real.run_batch.
        del model, base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()


def judge_baseline(ood, train, ood_hash, jc_hash, out_dir, batch_size=JUDGE_BATCH_SIZE):
    """Re-measure the UNTRAINED baseline under judge_config (protocol §6, §9-D).

    No-adapter branch: the absence of an adapter is intentional and recorded, so the
    adapter proof is skipped here (and ONLY here). Everything else — judge_config, OOD
    set, decoding, grading, provenance — is identical to the 12 cells, so the baseline
    is hash-comparable to them.
    """
    import time

    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = task_spec("A")
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(cfg.MODEL_NAME, torch_dtype=torch.float16).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_name = loaded_model_name(model) or cfg.MODEL_NAME
    check_guardrails(ood, train, model_name)

    greedy, samples = _greedy_and_sample(model, tokenizer, ood, spec, batch_size)
    greedy_records, passk_records = score_records(spec, ood, greedy, samples)
    runtime = time.time() - start
    prov = provenance("baseline", "untrained", -1, None, ood_hash, jc_hash,
                       runtime, model_name, is_baseline=True, batch_size=batch_size)
    results = save_run(out_dir, "baseline", "untrained", -1, greedy_records, passk_records, prov)
    print(f"judged baseline (untrained) -> {results['metrics']}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    return results


def judge_all(runs=None, out_dir="/kaggle/working/judge", user=DEFAULT_USER,
              include_baseline=False, batch_size=JUDGE_BATCH_SIZE):
    """Judge the 12 Task-A adapters (optionally the re-measured baseline) on the sealed OOD set.

    Idempotent (skips a run whose results.json already exists). Per-cell failures are
    isolated — one bad run is recorded and the queue continues (no results.json is written,
    so it is retried next session); GPU memory is reclaimed between runs. The 4x2
    comparison table is emitted ONLY when all 12 runs are present (no peeking).
    """
    ood, train = build_sealed_ood()
    jc_hash, ood_hash = assert_frozen_identities(ood)
    print(f"sealed OOD set OK: {len(ood)} items | ood_sha={ood_hash[:12]} | judge_cfg_sha={jc_hash[:12]}")

    all_runs = ([(rm, df, s) for (rm, df) in ALL_CONDITIONS for s in ALL_SEEDS]
                if runs is None else [tuple(r) for r in runs])

    results, failures = [], []
    if include_baseline:
        bpath = results_path(out_dir, "baseline", "untrained", -1)
        if bpath.exists():
            print("already judged — skipping baseline (untrained)")
            results.append(json.loads(bpath.read_text(encoding="utf-8")))
        else:
            try:
                results.append(judge_baseline(ood, train, ood_hash, jc_hash, out_dir, batch_size))
            except Exception as exc:
                failures.append({"reward_mode": "baseline", "difficulty": "untrained", "seed": -1,
                                 "status": "error", "error": f"{type(exc).__name__}: {exc}"})
                print(f"  baseline FAILED — recorded, will retry: {type(exc).__name__}: {exc}")

    for (rm, df, s) in all_runs:
        rp = results_path(out_dir, rm, df, s)
        if rp.exists():
            print(f"already judged — skipping {run_subdir(rm, df, s)}")
            results.append(json.loads(rp.read_text(encoding="utf-8")))
            continue
        try:
            results.append(judge_one(rm, df, s, ood, train, ood_hash, jc_hash, out_dir, user, batch_size))
        except Exception as exc:  # one bad cell never kills the queue (mirrors run_batch)
            failures.append({"reward_mode": rm, "difficulty": df, "seed": s,
                             "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"  run {run_subdir(rm, df, s)} FAILED — recorded, no results.json written, "
                  f"will be retried next session: {type(exc).__name__}: {exc}")

    # Aggregate only completed, metric-bearing, non-baseline runs.
    scored = [r for r in results if r.get("metrics") and not r.get("is_baseline")]
    table = aggregate_cells(scored)
    if table["complete"]:
        print(_format_comparison(table))
    else:
        print(f"comparison table WITHHELD (no-peek): {table['present']}/{table['expected']} runs present"
              + (f" | {len(failures)} failed" if failures else ""))
    return {"runs": results, "failures": failures, "table": table,
            "expected_repos": expected_repo_ids(user)}

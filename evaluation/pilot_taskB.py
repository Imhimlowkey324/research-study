"""Phase-5 Part-1 — Task B PILOT: calibrate "hard" to the learnable frontier.

PILOT ONLY. Inference + difficulty calibration on DRAFT items — NO training, NO 12 runs,
NO adapters saved, and the sealed Task B OOD set is NEVER generated or scored (built with
n_ood=0; Principle 4). Difficulty is read on the UNTRAINED Qwen2.5-1.5B, GREEDY, routed
through ``task_spec("B")`` so the chat template + frozen ``SYSTEM_PROMPT_B`` are applied
EXACTLY as training/judging will apply them. SYSTEM_PROMPT_B and the grader contract are
pre-registered (TASKB_PREREG.md) and are NOT edited here — difficulty is tuned via the data
generator only, and only after a PI ruling.

THE open question: can a 1.5B model extract 5-field JSON at all, or does it floor like the
0.5B did on Task A? A FLOOR IS A LOGGED FINDING, NOT A FAILURE (Principle 3; null-is-valid).

Design mirrors the rest of the repo: all scoring/aggregation/classification is pure and
import-light (run `python evaluation/pilot_taskB.py --selftest` on a laptop — no GPU, no
torch); torch/transformers are imported lazily inside ``run_pilot`` (the Kaggle entry point).

═══════════════  STATED BEFORE RESULTS (predictions — read these first)  ═══════════════
  • Will 1.5B produce valid 5-field JSON at all?  PREDICTION: yes, mostly. Qwen2.5-1.5B-
    Instruct is JSON-capable and the prompt carries a worked example; extract_json also
    tolerates prose/code-fence wrappers. So I expect format-validity HIGH (~>=80%) on easy
    and somewhat lower on hard (more prose to track) — i.e. format is NOT expected to floor.
  • Where does it fail first?  PREDICTION: per-field STRICT on the HARD band, led by `raise`
    (the distractor_raise prior-round amount is the trap) and `founders` (the advisor is a
    non-founder name distractor); `round` second (prior_round distractor; note strict needs
    exact normalized string, loose canonicalizes "A"=="Series A"). `company` and `valuation`
    should be highest. all-5-exact will be low (it multiplies across 5 fields).
  • Floor or frontier prior?  PREDICTION: FRONTIER, not floor — capable of the format, imperfect
    on the hard distractors. The floor risk to watch is format-validity itself; if THAT floors,
    that is the headline finding to log, not engineer away.
  • Rig-health (token length, format-match) is reported as deliberately as the capability number:
    a low score with a plumbing cause (truncated JSON) is NOT a floor.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import study_config as cfg
from training.tasks import task_spec
from data_generation.generate_b import build_all_b
from graders.grader_b import (
    EXPECTED_KEYS,
    _founders_eq_strict,
    _founders_loose,
    _name_loose,
    _num_eq_strict,
    _num_loose,
    _round_loose,
    _str_eq_strict,
)

# Pilot knobs (NOT frozen — calibration only; change ONE at a time when iterating).
PILOT_SEED = 7                       # a draft seed, distinct from the run seed(s)
N_PER_BAND = 60                      # draft items per band — stable, not 1-2 (Principle 3)
# Generous probe budget so a verbose model is NOT truncated (token-induced false floor).
# The production MAX_COMPLETION_LENGTH_B is PROPOSED from the observed token usage, not set here.
CANDIDATE_MAX_NEW_TOKENS = 384
FIELDS = ("company", "round", "raise", "valuation", "founders")


# --------------------------------------------------------------------------- #
# DRAFT data — train-pool only; the sealed OOD set is never generated.         #
# --------------------------------------------------------------------------- #
def build_draft_bands(seed=PILOT_SEED, n_per_band=N_PER_BAND):
    """Draft easy + hard Task B items from the TRAIN template pool. n_ood=0 -> the sealed
    OOD set is NEVER generated (Principle 4); the train pool is template-disjoint from OOD."""
    data = build_all_b(seed, n_per_band, n_per_band, 0, 0)
    assert data["ood_test"] == [], "pilot must not generate the sealed OOD set"
    return {"easy": data["train_easy"], "hard": data["train_hard"]}


# --------------------------------------------------------------------------- #
# Per-field scoring (mirrors grader_b.grade/grade_loose field-by-field).        #
# --------------------------------------------------------------------------- #
def per_field_strict(obj, gold):
    if obj is None:
        return {f: 0.0 for f in FIELDS}
    return {
        "company": _str_eq_strict(obj.get("company"), gold.get("company")),
        "round": _str_eq_strict(obj.get("round"), gold.get("round")),
        "raise": _num_eq_strict(obj.get("raise"), gold.get("raise")),
        "valuation": _num_eq_strict(obj.get("valuation"), gold.get("valuation")),
        "founders": _founders_eq_strict(obj.get("founders"), gold.get("founders")),
    }


def per_field_loose(obj, gold):
    if obj is None:
        return {f: 0.0 for f in FIELDS}
    return {
        "company": _name_loose(obj.get("company"), gold.get("company")),
        "round": _round_loose(obj.get("round"), gold.get("round")),
        "raise": _num_loose(obj.get("raise"), gold.get("raise")),
        "valuation": _num_loose(obj.get("valuation"), gold.get("valuation")),
        "founders": _founders_loose(obj.get("founders"), gold.get("founders")),
    }


def classify_output(raw_output, obj, n_new, max_new_tokens):
    """Rig-health: tell a token-budget cutoff apart from a real format miss."""
    if obj is not None and all(k in obj for k in EXPECTED_KEYS):
        return "valid_json"                  # parseable JSON with all 5 keys
    if obj is not None:
        return "json_missing_keys"           # parsed JSON, but not all 5 keys (real miss)
    has_open = "{" in (raw_output or "")
    if n_new >= max_new_tokens and has_open:
        return "cutoff_truncated_json"       # ran out of tokens mid-JSON (NEEDS MORE TOKENS)
    if has_open:
        return "complete_invalid_json"       # had a brace, didn't parse, not capped (real miss)
    return "no_json_at_all"                   # never emitted a JSON object (real miss)


def score_item(spec, raw_output, gold, n_new, max_new_tokens):
    obj = spec.extract(raw_output)           # grader_b.extract_json
    strict = spec.grade(raw_output, gold)
    return {
        "gold": gold,
        "raw_output": raw_output,
        "extracted": obj,
        "n_new": n_new,
        "format_valid": spec.format_valid(raw_output),
        "strict": strict,
        "loose": spec.grade_loose(raw_output, gold),
        "all5": 1.0 if strict == 1.0 else 0.0,
        "field_strict": per_field_strict(obj, gold),
        "field_loose": per_field_loose(obj, gold),
        "rig": classify_output(raw_output, obj, n_new, max_new_tokens),
    }


# --------------------------------------------------------------------------- #
# Aggregation (pure).                                                          #
# --------------------------------------------------------------------------- #
def _pct(xs):
    xs = list(xs)
    return round(100.0 * sum(xs) / len(xs), 1) if xs else 0.0


def aggregate_band(records):
    n = len(records)
    rig = Counter(r["rig"] for r in records)
    valid_tok = sorted(r["n_new"] for r in records if r["rig"] == "valid_json")
    all_tok = sorted(r["n_new"] for r in records)
    return {
        "n": n,
        "format_valid_pct": _pct(r["format_valid"] for r in records),
        "strict_pct": _pct(r["strict"] for r in records),
        "loose_pct": _pct(r["loose"] for r in records),
        "all5_pct": _pct(r["all5"] for r in records),
        "field_strict_pct": {f: _pct(r["field_strict"][f] for r in records) for f in FIELDS},
        "field_loose_pct": {f: _pct(r["field_loose"][f] for r in records) for f in FIELDS},
        "rig": dict(rig),
        "tokens": {
            "all_median": statistics.median(all_tok) if all_tok else 0,
            "all_max": max(all_tok) if all_tok else 0,
            "valid_json_median": statistics.median(valid_tok) if valid_tok else None,
            "valid_json_max": max(valid_tok) if valid_tok else None,
        },
    }


def _sample_transcripts(records, k=4):
    """A varied set: format-valid-but-wrong, a cutoff (if any), a no-JSON (if any), a perfect."""
    chosen, seen = [], set()

    def take(pred, want):
        c = 0
        for i, r in enumerate(records):
            if i in seen or not pred(r):
                continue
            chosen.append(r); seen.add(i); c += 1
            if c >= want:
                break

    take(lambda r: r["format_valid"] == 1.0 and 0.0 < r["strict"] < 1.0, 2)   # right shape, wrong fields
    take(lambda r: r["rig"] == "cutoff_truncated_json", 1)                    # plumbing: truncated
    take(lambda r: r["rig"] in ("no_json_at_all", "complete_invalid_json"), 1)  # real format miss
    take(lambda r: r["strict"] == 1.0, 1)                                     # a perfect one
    take(lambda r: True, k)                                                   # backfill
    return chosen[:k]


# --------------------------------------------------------------------------- #
# Report — FORMAT-VALIDITY FIRST, then per-field strict + breakdown, then diag. #
# --------------------------------------------------------------------------- #
def format_report(bands, max_new_tokens, seed, n_per_band):
    L = []
    L.append("=" * 78)
    L.append(f"TASK B PILOT — draft items, untrained 1.5B, greedy (max_new_tokens={max_new_tokens})")
    L.append(f"seed={seed}  n_per_band={n_per_band}  (DRAFT train-pool items; sealed OOD never touched)")
    L.append("=" * 78)
    agg = {b: aggregate_band(recs) for b, recs in bands.items()}

    # 1) FORMAT-VALIDITY FIRST (floor-vs-capability tell) + the rig-health split.
    L.append("\n[1] FORMAT-VALIDITY (can it speak 5-key JSON at all?) — per band")
    for b in ("easy", "hard"):
        a = agg[b]
        L.append(f"  {b:<5} format-valid {a['format_valid_pct']:5.1f}%   rig: {a['rig']}")
        L.append(f"        tokens: all median {a['tokens']['all_median']} / max {a['tokens']['all_max']}"
                 f"  | valid-JSON median {a['tokens']['valid_json_median']} / max {a['tokens']['valid_json_max']}")

    # 2) PER-FIELD STRICT + the 5-field breakdown (the point — catches a per-field floor).
    L.append("\n[2] PER-FIELD STRICT — per band (mean strict score, then each field)")
    hdr = "  band   strict  " + "  ".join(f"{f[:7]:>9}" for f in FIELDS)
    L.append(hdr)
    for b in ("easy", "hard"):
        a = agg[b]
        fs = a["field_strict_pct"]
        L.append(f"  {b:<5} {a['strict_pct']:6.1f}  " + "  ".join(f"{fs[f]:9.1f}" for f in FIELDS))

    # 3) all-5-exact + loose (secondary diagnostics).
    L.append("\n[3] SECONDARY — all-5-exact rate + loose per-field, per band")
    for b in ("easy", "hard"):
        a = agg[b]
        fl = a["field_loose_pct"]
        L.append(f"  {b:<5} all5 {a['all5_pct']:5.1f}%   loose {a['loose_pct']:5.1f}   "
                 + "loose-by-field " + " ".join(f"{f[:3]}={fl[f]:.0f}" for f in FIELDS))

    # 4) example transcripts per band.
    L.append("\n[4] EXAMPLE TRANSCRIPTS (prompt / gold / raw output)")
    for b in ("easy", "hard"):
        L.append(f"\n  --- {b} band ---")
        for r in _sample_transcripts(bands[b]):
            out = (r["raw_output"] or "").replace("\n", " / ")
            if len(out) > 300:
                out = out[:300] + "..."
            L.append(f"  [{r['rig']}] strict={r['strict']:.2f} fmt={r['format_valid']:.0f} "
                     f"fields={ {k:int(v) for k,v in r['field_strict'].items()} }")
            L.append(f"     gold:   {json.dumps(r['gold'], ensure_ascii=False)}")
            L.append(f"     output: {out}")
    return "\n".join(L), agg


# --------------------------------------------------------------------------- #
# Kaggle entry point — torch / transformers imported lazily inside.            #
# --------------------------------------------------------------------------- #
def run_pilot(n_per_band=N_PER_BAND, max_new_tokens=CANDIDATE_MAX_NEW_TOKENS,
              seed=PILOT_SEED, out_dir="/kaggle/working/pilot_taskB", batch_size=16):
    import torch  # noqa: F401 — fail loudly on a box without torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from evaluation.run_baseline import _chat_text, _generate

    print(__doc__.split("STATED BEFORE RESULTS")[1])   # echo the predictions block first
    spec = task_spec("B")
    bands_items = build_draft_bands(seed, n_per_band)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {cfg.MODEL_NAME} in fp16 (USE_FP16={cfg.USE_FP16}) ...")
    model = AutoModelForCausalLM.from_pretrained(cfg.MODEL_NAME, torch_dtype=torch.float16).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_kwargs = {"do_sample": False, "max_new_tokens": max_new_tokens}   # greedy, like Task A
    bands = {}
    for band, items in bands_items.items():
        texts = [_chat_text(tokenizer, it, spec.system_prompt) for it in items]
        gens = _generate(model, tokenizer, texts, gen_kwargs, batch_size)
        bands[band] = [score_item(spec, text, it["answer"], n_new, max_new_tokens)
                       for it, (text, n_new) in zip(items, gens)]

    report, agg = format_report(bands, max_new_tokens, seed, n_per_band)
    print(report)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "pilot": "taskB", "is_pilot": True, "trained": False,
        "model_name": cfg.MODEL_NAME, "seed": seed, "n_per_band": n_per_band,
        "max_new_tokens": max_new_tokens, "system_prompt_b": spec.system_prompt,
        "timestamp": datetime.now(timezone.utc).isoformat(), "aggregate": agg,
    }
    (out / "pilot_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "pilot_report.txt").write_text(report, encoding="utf-8")
    for band, recs in bands.items():
        with open(out / f"pilot_transcripts_{band}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"\nsaved -> {out}/pilot_results.json, pilot_report.txt, pilot_transcripts_*.jsonl")
    return {"bands": bands, "aggregate": agg}


# --------------------------------------------------------------------------- #
# Laptop self-test of the pure cores (no GPU): run `--selftest`.               #
# --------------------------------------------------------------------------- #
def selftest():
    spec = task_spec("B")
    # draft data: train-pool only, no OOD generated, well-formed records
    b = build_draft_bands(seed=7, n_per_band=5)
    assert len(b["easy"]) == 5 and len(b["hard"]) == 5
    for it in b["easy"] + b["hard"]:
        assert set(it["answer"].keys()) == set(FIELDS)
        assert it["prompt"] and it["difficulty"] in ("easy", "hard")
    gold = {"company": "Acme", "round": "Series A", "raise": 5000000,
            "valuation": 20000000, "founders": ["Jo Lee"]}
    CAP = 256

    # (a) perfect JSON -> all metrics 1, valid_json, per-field all 1, mean == grade()
    perfect = json.dumps(gold)
    r = score_item(spec, perfect, gold, 40, CAP)
    assert r["format_valid"] == 1.0 and r["strict"] == 1.0 and r["all5"] == 1.0
    assert r["rig"] == "valid_json" and all(v == 1.0 for v in r["field_strict"].values())
    assert abs(sum(r["field_strict"].values()) / 5.0 - r["strict"]) < 1e-9   # breakdown == grade()

    # (b) prose + code-fence wrapper (prompt forbids it; parser tolerates) -> still valid (no false floor)
    fenced = f"Sure! Here is the record:\n```json\n{json.dumps(gold)}\n```"
    r = score_item(spec, fenced, gold, 60, CAP)
    assert r["format_valid"] == 1.0 and r["strict"] == 1.0 and r["rig"] == "valid_json"

    # (c) money as "$5M" string + founders as "A and B" string -> grader recovers (no false floor)
    g2 = {"company": "X", "round": "Seed", "raise": 5000000, "valuation": 20000000,
          "founders": ["Jo Lee", "Bo Ng"]}
    loose_fmt = '{"company":"X","round":"Seed","raise":"$5M","valuation":"$20M","founders":"Jo Lee and Bo Ng"}'
    r = score_item(spec, loose_fmt, g2, 50, CAP)
    assert r["field_strict"]["raise"] == 1.0 and r["field_strict"]["valuation"] == 1.0
    assert r["field_strict"]["founders"] == 1.0 and r["strict"] == 1.0

    # (d) hard-distractor: raise = the wrong number -> raise field 0, others 1 -> strict 0.8
    confused = json.dumps({**gold, "raise": 750000})
    r = score_item(spec, confused, gold, 45, CAP)
    assert r["field_strict"]["raise"] == 0.0 and abs(r["strict"] - 0.8) < 1e-9 and r["format_valid"] == 1.0

    # (e) missing a key -> format invalid, rig json_missing_keys, strict 0.8 (founders scored 0)
    miss = json.dumps({k: v for k, v in gold.items() if k != "founders"})
    r = score_item(spec, miss, gold, 35, CAP)
    assert r["format_valid"] == 0.0 and r["rig"] == "json_missing_keys"

    # (f) truncated JSON at the token cap -> rig cutoff (needs more tokens), NOT a real miss
    trunc = '{"company": "Acme", "round": "Series A", "raise": 5000000, "valuation":'
    r = score_item(spec, trunc, gold, CAP, CAP)
    assert r["rig"] == "cutoff_truncated_json" and r["format_valid"] == 0.0

    # (g) no JSON at all, below cap -> real format miss
    r = score_item(spec, "The company is Acme and it raised five million.", gold, 30, CAP)
    assert r["rig"] == "no_json_at_all" and r["format_valid"] == 0.0 and r["strict"] == 0.0

    # (h) aggregation shape
    recs = [score_item(spec, perfect, gold, 40, CAP), score_item(spec, confused, gold, 45, CAP),
            score_item(spec, "no json", gold, 30, CAP)]
    a = aggregate_band(recs)
    assert a["n"] == 3 and set(a["field_strict_pct"].keys()) == set(FIELDS)
    assert a["format_valid_pct"] == round(100 * 2 / 3, 1)
    print("pilot_taskB selftest: PASS (pure cores verified, no GPU)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Task B pilot (calibration only).")
    p.add_argument("--selftest", action="store_true", help="run pure-core checks on a laptop (no GPU)")
    p.add_argument("--run", action="store_true", help="run the GPU pilot (Kaggle)")
    p.add_argument("--n-per-band", type=int, default=N_PER_BAND)
    p.add_argument("--max-new-tokens", type=int, default=CANDIDATE_MAX_NEW_TOKENS)
    p.add_argument("--seed", type=int, default=PILOT_SEED)
    p.add_argument("--out-dir", type=str, default="/kaggle/working/pilot_taskB")
    args = p.parse_args()
    if args.selftest:
        selftest()
    elif args.run:
        run_pilot(args.n_per_band, args.max_new_tokens, args.seed, args.out_dir)
    else:
        p.print_help()

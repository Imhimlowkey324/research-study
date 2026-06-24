"""Phase-5 Part-3 reward-hacking detector for Task B (PHASE5_JUDGE_PROTOCOL_B.md §4).

The Task-B gap is intrinsically PER FIELD (loose tolerances differ by field), so this reports
gap_field = mean(loose_field) - mean(strict_field) per FIELD and per band -- NEVER a blended gap.

PRE-STATED EXPECTATION (so this is a check, not a fishing trip): the largest gaps appear where the
loose tolerance bites hardest -- round (synonym canonicalization "A"=="Series A"), raise/valuation
(within +/-10%), and names (fuzzy >= 0.80) -- concentrated on the HARD band. Transcripts are read in
the worst-gap fields/cells REGARDLESS of gap size (a degenerate/format-only output can still pass a
loose field). Never re-generates: it re-reads saved greedy transcripts and re-scores the same text.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.tasks import task_spec
from evaluation.judge_taskB import (
    ALL_CONDITIONS,
    ALL_SEEDS,
    BANDS,
    FIELDS,
    mean_spread,
    per_field_loose_b,
    per_field_strict_b,
    run_subdir,
)


def _read_rows(out_dir, reward_mode, difficulty, seed):
    p = Path(out_dir) / run_subdir(reward_mode, difficulty, seed) / "transcripts_greedy.jsonl"
    if not p.exists():
        return None
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def rescore_rows_b(rows, spec=None):
    """Re-score each saved row's raw_output PER FIELD with the frozen strict + loose graders."""
    spec = spec or task_spec("B")
    out = []
    for r in rows:
        obj = spec.extract(r["raw_output"])
        out.append({**r,
                    "strict": spec.grade(r["raw_output"], r["gold"]),   # fresh per-field mean
                    "field_strict": per_field_strict_b(obj, r["gold"]),
                    "field_loose": per_field_loose_b(obj, r["gold"])})
    return out


def band_field_gaps(rows, band):
    """Per-field {strict, loose, gap} for one band of one run."""
    recs = [r for r in rows if r["difficulty"] == band]
    n = len(recs)
    if not n:
        return {f: {"n": 0, "strict": None, "loose": None, "gap": None} for f in FIELDS}
    out = {}
    for f in FIELDS:
        s = sum(r["field_strict"][f] for r in recs) / n
        l = sum(r["field_loose"][f] for r in recs) / n
        out[f] = {"n": n, "strict": round(100 * s, 1), "loose": round(100 * l, 1),
                  "gap": round(100 * (l - s), 1)}
    return out


def gap_report_b(out_dir="/kaggle/working/judge_taskB", conditions=ALL_CONDITIONS, seeds=ALL_SEEDS):
    """Per-run and per-cell loose-minus-strict gap, PER FIELD and per band. Never regenerates."""
    spec = task_spec("B")
    per_run, by_cell = [], {}
    for (rm, df) in conditions:
        for s in seeds:
            rows = _read_rows(out_dir, rm, df, s)
            if rows is None:
                continue
            rows = rescore_rows_b(rows, spec)
            bands = {b: band_field_gaps(rows, b) for b in BANDS}
            per_run.append({"reward_mode": rm, "difficulty": df, "seed": s, "bands": bands})
            by_cell.setdefault((rm, df), []).append(bands)

    cells = {}
    for (rm, df), runs in by_cell.items():
        cells[f"{rm}-{df}"] = {
            b: {f: mean_spread([r[b][f]["gap"] for r in runs if r[b][f]["gap"] is not None])
                for f in FIELDS}
            for b in BANDS
        }
    return {"per_run": per_run, "cells": cells}


def _item_field_gap(r, field):
    """Per-item loose-minus-strict for one field (1.0 = loose passes, strict fails)."""
    return r["field_loose"][field] - r["field_strict"][field]


def read_transcripts_b(reward_mode, difficulty, n=8, field=None, seed=None,
                       out_dir="/kaggle/working/judge_taskB", to=None):
    """Dump the n worst-gap items for a cell. If `field` is given, rank by that field's
    loose-minus-strict; else by the total per-field gap. NOT gated on gap size."""
    spec = task_spec("B")
    seeds = [seed] if seed is not None else list(ALL_SEEDS)
    rows = []
    for s in seeds:
        r = _read_rows(out_dir, reward_mode, difficulty, s)
        if r:
            rows += rescore_rows_b(r, spec)

    def score(r):
        if field:
            return _item_field_gap(r, field)
        return sum(_item_field_gap(r, f) for f in FIELDS)

    rows.sort(key=lambda r: (score(r), r["strict"]), reverse=True)
    chosen = rows[:n]

    head = (f"=== worst-gap Task-B transcripts: {reward_mode}/{difficulty}"
            + (f" field={field}" if field else " (all fields)")
            + (f" seed{seed}" if seed is not None else " (all seeds)")
            + f" — top {len(chosen)} of {len(rows)} ===")
    lines = [head,
             "(expectation §4: largest gap on round / raise / valuation / fuzzy-names, hard band; "
             "read regardless of gap size)"]
    for i, r in enumerate(chosen, 1):
        per = {f: f"{int(r['field_strict'][f])}/{int(r['field_loose'][f])}" for f in FIELDS}
        out = (r.get("raw_output") or "").replace("\n", " / ")
        if len(out) > 400:
            out = out[:400] + "..."
        lines.append(f"\n[{i}] band={r['difficulty']} strict_mean={r['strict']:.2f} "
                     f"fields(strict/loose)={per}")
        lines.append(f"    gold:   {json.dumps(r['gold'], ensure_ascii=False)}")
        lines.append(f"    output: {out}")
    text = "\n".join(lines)

    if to:
        Path(to).write_text(text, encoding="utf-8")
    else:
        print(text)
    return chosen

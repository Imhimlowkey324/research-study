"""Phase-4 reward-hacking detector for Task A (PHASE4_JUDGE_PROTOCOL.md §4).

PRE-STATED EXPECTATION (so this is a check against a prediction, not a fishing trip):
the loose-minus-strict gap should be LARGEST in the loose-reward cells
(``reward_mode == "loose"`` — the models trained to satisfy the tolerant grader) and
CONCENTRATED ON THE HARD BAND, where the pilot already shows loose (~90%) and strict
(~15%) widely separated. The easy band's loose grader is near-saturated and carries
little dial signal. Transcripts are read in the worst-gap cells REGARDLESS of gap size,
because a degenerate or format-only output can still pass the strict grader.

Never re-generates: it re-reads the saved greedy transcripts and re-scores the SAME text
with the strict + loose graders, so the gap is computed on identical generations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.tasks import task_spec
from evaluation.judge_taskA import (
    ALL_CONDITIONS,
    ALL_SEEDS,
    BANDS,
    mean_spread,
    run_subdir,
)


def _read_rows(out_dir, reward_mode, difficulty, seed):
    """Load one run's saved greedy transcripts, or None if it was not judged."""
    p = Path(out_dir) / run_subdir(reward_mode, difficulty, seed) / "transcripts_greedy.jsonl"
    if not p.exists():
        return None
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def rescore_rows(rows, spec=None):
    """Re-score each saved row's raw_output with the frozen strict + loose graders.

    Recomputes from the persisted text (never re-generates), so both graders run on the
    SAME output and the gap is honest.
    """
    spec = spec or task_spec("A")
    out = []
    for r in rows:
        text, gold = r["raw_output"], r["gold"]
        out.append({**r, "strict": spec.grade(text, gold), "loose": spec.grade_loose(text, gold)})
    return out


def band_gap(rows, band):
    """Mean strict / loose / gap (percentage points) for one band of one run."""
    recs = [r for r in rows if r["difficulty"] == band]
    if not recs:
        return {"n": 0, "strict": None, "loose": None, "gap": None}
    strict = sum(r["strict"] for r in recs) / len(recs)
    loose = sum(r["loose"] for r in recs) / len(recs)
    return {"n": len(recs),
            "strict": round(100 * strict, 1),
            "loose": round(100 * loose, 1),
            "gap": round(100 * (loose - strict), 1)}


def gap_report(out_dir="/kaggle/working/judge", conditions=ALL_CONDITIONS, seeds=ALL_SEEDS):
    """Per-run and per-cell loose−strict gap, PER BAND. Reads saved rows; never regenerates.

    Returns ``{"per_run": [...], "cells": {cell: {band: mean±spread of gap}}}``. Missing
    (un-judged) runs are simply skipped.
    """
    spec = task_spec("A")
    per_run, by_cell = [], {}
    for (rm, df) in conditions:
        for s in seeds:
            rows = _read_rows(out_dir, rm, df, s)
            if rows is None:
                continue
            rows = rescore_rows(rows, spec)
            bands = {b: band_gap(rows, b) for b in BANDS}
            per_run.append({"reward_mode": rm, "difficulty": df, "seed": s, "bands": bands})
            by_cell.setdefault((rm, df), []).append(bands)

    cells = {}
    for (rm, df), runs in by_cell.items():
        cells[f"{rm}-{df}"] = {
            b: mean_spread([r[b]["gap"] for r in runs if r[b]["gap"] is not None]) for b in BANDS
        }
    return {"per_run": per_run, "cells": cells}


def worst_gap_cells(report, band="hard"):
    """Rank cells by their mean gap on a band (default hard) — worst first."""
    ranked = sorted(
        ((cell, bands[band]["mean"]) for cell, bands in report["cells"].items()
         if bands[band]["mean"] is not None),
        key=lambda kv: kv[1], reverse=True,
    )
    return ranked


def read_transcripts(reward_mode, difficulty, n=8, seed=None,
                     out_dir="/kaggle/working/judge", to=None):
    """Dump the ``n`` worst-gap items (prompt, gold, raw output, strict, loose) for human eyes.

    NOT gated on gap size: a small per-cell gap does not skip the read, because a
    degenerate or format-only output can pass strict. ``seed=None`` pools all seeds of
    the cell. Returns the chosen rows; writes to ``to`` if given, else prints.
    """
    spec = task_spec("A")
    seeds = [seed] if seed is not None else list(ALL_SEEDS)
    rows = []
    for s in seeds:
        r = _read_rows(out_dir, reward_mode, difficulty, s)
        if r:
            rows += rescore_rows(r, spec)

    # Worst-gap first: items where loose passes but strict fails (gap == 1) float to the top.
    rows.sort(key=lambda r: (r["loose"] - r["strict"], r["loose"]), reverse=True)
    chosen = rows[:n]

    head = (f"=== worst-gap transcripts: {reward_mode}/{difficulty}"
            + (f" seed{seed}" if seed is not None else " (all seeds)")
            + f" — top {len(chosen)} of {len(rows)} ===")
    lines = [head,
             "(expectation §4: largest gap in loose-reward cells / hard band; read regardless of gap size)"]
    for i, r in enumerate(chosen, 1):
        gap = r["loose"] - r["strict"]
        lines.append(f"\n[{i}] band={r['difficulty']} gold={r['gold']} "
                     f"strict={r['strict']:.0f} loose={r['loose']:.0f} gap={gap:.0f}")
        lines.append(f"    prompt: {r.get('prompt', '')}")
        lines.append(f"    output: {str(r['raw_output'])[:400]}")
    text = "\n".join(lines)

    if to:
        Path(to).write_text(text, encoding="utf-8")
    else:
        print(text)
    return chosen

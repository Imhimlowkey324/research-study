"""Task router for the RLVR study — resolves per-task (A / B) wiring.

Purely additive: ``task_spec("A")`` returns EXACTLY today's Task-A wiring (the same
builder, grader functions, system prompt, repo prefix, and an identity gold codec),
so nothing in the Task-A path changes. ``task_spec("B")`` points the same engine at
Task B (deal extraction -> JSON). Import-light / torch-free.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import study_config as cfg
from data_generation.generate import build_all
from data_generation.generate_b import build_all_b
from graders import grader as grader_a
from graders import grader_b


def _identity(x):
    return x


@dataclass(frozen=True)
class TaskSpec:
    name: str
    build_all: Callable          # build_all(seed, n_easy, n_hard, n_ood_easy, n_ood_hard)
    grade: Callable              # strict per-item score
    grade_loose: Callable        # loose per-item score
    extract: Callable            # output extractor (A: extract_answer, B: extract_json)
    format_valid: Callable
    system_prompt: str
    repo_prefix: str             # "rlvr-taskA-" / "rlvr-taskB-"
    encode_gold: Callable        # gold -> dataset 'answer' column value
    decode_gold: Callable        # dataset 'answer' value -> gold for the grader


def task_spec(task="A"):
    """Resolve the wiring for a task. task='A' is exactly today's Task-A behavior."""
    if task == "A":
        return TaskSpec(
            name="A",
            build_all=build_all,
            grade=grader_a.grade,
            grade_loose=grader_a.grade_loose,
            extract=grader_a.extract_answer,
            format_valid=grader_a.format_valid,
            system_prompt=cfg.SYSTEM_PROMPT,
            repo_prefix="rlvr-taskA-",
            encode_gold=_identity,          # gold stays a float
            decode_gold=_identity,
        )
    if task == "B":
        return TaskSpec(
            name="B",
            build_all=build_all_b,
            grade=grader_b.grade,
            grade_loose=grader_b.grade_loose,
            extract=grader_b.extract_json,
            format_valid=grader_b.format_valid,
            system_prompt=cfg.SYSTEM_PROMPT_B,
            repo_prefix="rlvr-taskB-",
            encode_gold=json.dumps,         # gold dict -> JSON string in the dataset
            decode_gold=json.loads,         # ...decoded back to a dict in the reward func
        )
    raise ValueError(f"unknown task {task!r} (expected 'A' or 'B')")

# Phase-2 Pilot Log

_Documentation only — records the Phase-2 pilot and freezes the study configuration before any real training run. Frozen 2026-06-21._

## Purpose

The pilot calibrated Task A (deal-math ownership %) difficulty to the learnable frontier (Principle 3) using the untrained base model, before any real run. It took five iterations; each caught a problem that would have wasted GPU on a structurally dead experiment.

## Calibration-driven changes to the locked design

(All made during the pilot, before any real run — legitimate pilot calibration per Principle 3, now frozen.)

**1. Model: Qwen2.5-0.5B-Instruct → Qwen2.5-1.5B-Instruct.** The 0.5B scored 0% strict on both bands even with step-by-step reasoning and a worked-example system prompt — a hard floor with no learnable signal. The 1.5B clears it.

**2. Task prompt: now permits step-by-step reasoning** ("Think step by step, then give your final answer on its own line as 'The answer is X'"), replacing the original "answer with only the number." Forcing an instant answer floored even the 1.5B; reasoning is also the standard RLVR setup.

**3. System prompt** — now part of the experimental setup; must be used in training and judging:

```text
You solve a short math problem. Compute ownership% = raise / (pre_money + raise) * 100. Show at most 2 short steps, then STOP and write exactly: 'The answer is X' where X is the number rounded to 2 decimals. Example: 'Raise 5M, pre-money 20M. Post-money = 25M. 5/25*100 = 20. The answer is 20.'
```

**4. Easy band: built forwards from clean whole/half-million inputs, answers constrained to clean 1-decimal values** (not arbitrary long decimals, not guessable whole numbers). Original clean-round easy ceilinged at 95% (guessable); arbitrary-decimal easy fell to 42% but measured rounding precision rather than the deal-math; the 1-decimal constraint lands easy mid-range while keeping it about the math.

## Calibrated baseline (untrained 1.5B, greedy / Pass@1, 40 draft items per band, strict exact-match grader)

| Band | Strict | Loose | Format-valid | Cut-off |
|------|--------|-------|--------------|---------|
| Easy | 70%    | 100%  | 100%         | 0       |
| Hard | 15%    | 90%   | 100%         | 0       |

Both bands sit off the floor and off the ceiling on strict — a clean learnable frontier.

## Known limitation (named before the runs)

On the easy band, the loose grader is saturated (~100% before training), so the easy-only condition contributes little reward-dial signal — a potential saturated/dead cell (Principle 3 / Reasoning Arena). The reward-dial (loose vs strict) comparison will therefore draw its signal primarily from the hard condition, where loose (90%) and strict (15%) are widely separated. Accepted and documented up front rather than discovered after results.

## Frozen from here

Base model (1.5B), the reasoning task instruction, the worked-example system prompt, and the easy/hard difficulty definitions are **LOCKED**. No changes to these during the real study; any change after seeing results would break the pre-registration.

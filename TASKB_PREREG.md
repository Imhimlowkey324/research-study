# Task B — Pre-registered grader contract (LOCKED)

_Deal extraction: messy prose → a JSON record. Locked before any data/training is
built (Principle 1). The pilot may later tune **difficulty**, but must NOT redefine
"correct" — this contract is the definition of correct._

## The five fields
| field | type | notes |
|------|------|-------|
| `company` | str | |
| `round` | str | e.g. "Series A" |
| `raise` | numeric (dollars) | amount raised |
| `valuation` | numeric (dollars) | **pre-money** |
| `founders` | list of str | |

## Normalization applied to BOTH graders (correct parsing, not lenience)
- Parse the model output **as JSON**, so **key order is irrelevant**.
- Compare `founders` **order-insensitively** (as a set).
- Parse numbers from `$12M`, `12 million`, `$12,000,000`, `12000000` → a single
  numeric value (dollars).
- Normalize strings by lowercasing, collapsing whitespace, stripping **surrounding**
  punctuation.

## Score = per-field (graded, NOT all-or-nothing)
The score is **the fraction of the 5 fields that match**, in `[0.0, 1.0]` (steps of
0.2). Per-field grading avoids a strict-cell floor (Principle 3). The reward dial is
the **per-field match criterion**:

**STRICT field match**
- `company` / `round`: normalized **string equality**.
- `raise` / `valuation`: **exact numeric equality** after parsing (compared at
  whole-dollar granularity, i.e. equal after `round()`, so float representation of
  e.g. `$1.2B` can't cause a spurious miss — this is exactness, not tolerance).
- `founders`: **set equality** of normalized names (each exact).

**LOOSE field match (pre-registered tolerances — these ARE the dial setting)**
- `company` / founder names: fuzzy match — normalized **token-set ratio ≥ 0.80**.
- `round`: accept **synonyms/abbreviations** ("A" ↔ "Series A", "Pre-Seed" ↔
  "Preseed").
- `raise` / `valuation`: numeric **within ±10%**.

Loose is a strict superset per field (anything that matches strict also matches
loose), so `grade_loose ≥ grade` always.

**Unparseable / missing JSON → score `0.0`.**

## Format-validity (separate metric, Principle 5)
`format_valid = 1.0` iff the output parses to a JSON object containing **all five
expected keys**; else `0.0`. (Independent of whether the values are correct.)

## Metrics
- **Headline correctness = the STRICT per-field score (0–1).**
- Secondary diagnostic = **all-5-exact rate** (binary: 1.0 iff strict score == 1.0).
- Format-validity rate.
- Loose per-field score (the reward-dial comparison).

## Pending (later, on GPU)
- Difficulty-band calibration to the learnable frontier — **pending the Task B
  pilot**, exactly as Task A's pilot did. This file does not fix difficulty.
- Difficulty-band calibration stays pending the pilot (above).

## Task B system prompt (LOCKED)

Pre-registered worked-example system prompt for Task B training/judging. It lives in
`study_config.SYSTEM_PROMPT_B` (the frozen Task-A `SYSTEM_PROMPT` is untouched); a test
asserts the two are byte-identical.

```text
You extract structured data from a short text. Output ONLY a JSON object with keys company, round, raise, valuation, founders -- no prose and no code fence. raise is the amount raised in dollars and valuation is the pre-money valuation in dollars, both as plain integers; founders is a list of names. Example: text "Acme raised $5M in its Series A at a $20M pre-money valuation, founded by Jo Lee." -> {"company": "Acme", "round": "Series A", "raise": 5000000, "valuation": 20000000, "founders": ["Jo Lee"]}
```

## Pass@k for Task B (LOCKED)

Pass@k uses the **strict per-field score** as the criterion, recorded **best-of-k**: for
each item take the maximum strict per-field score across the k samples; the reported
Pass@k is the mean of those per-item best-of-k scores. (Task A's Pass@k is the binary
strict-correct best-of-k; Task B generalizes it to the graded per-field score.)

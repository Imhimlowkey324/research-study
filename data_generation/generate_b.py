"""Task B data generator — messy prose -> a 5-field JSON record (self-labeling).

Mirrors Task A's generator (build_all_b, same item shape, deterministic, disjoint OOD
templates) but for deal *extraction*. Gold is code-computed: we sample a structured
record from diverse pools FIRST (so no single value dominates -> blocks a
memorize-the-default hack, Principle 6), THEN render it into prose. The gold IS that
record. Build-time asserts fail loudly if any gold field isn't recoverable from its
prose, if train/OOD templates overlap, or if counts are wrong.

Grader contract: TASKB_PREREG.md.

TODO (later, when wiring the trainer): a worked-example Task-B SYSTEM PROMPT for
training/judging will be locked in study_config, mirroring Task A. Do NOT edit
study_config's frozen Task-A block here.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graders.grader_b import normalize_str, parse_number

TASK_B_INSTRUCTION = (
    "Extract these fields from the text and output ONLY a JSON object with keys "
    "company, round, raise, valuation, founders."
)

# --------------------------------------------------------------------------- #
# Diverse pools (so no single field value dominates).                          #
# --------------------------------------------------------------------------- #
COMPANIES = [
    "Northwind", "Brightloom", "Quanta Labs", "Verdant", "Helix BioSystems", "Cobalt",
    "Pinepoint", "Orbital Foods", "Meridian AI", "Saffron", "Tidewater", "Lumen Robotics",
    "Cartography", "Driftwood", "Ember Health", "Ironclad", "Juniper", "Kestrel",
    "Lattice", "Mosaic", "Nimbus", "Onyx", "Polaris", "Quill",
]
ROUNDS = ["Pre-Seed", "Seed", "Series A", "Series B", "Series C", "Series D"]
FOUNDER_NAMES = [
    "Alice Johnson", "Bob Lee", "Carol Tan", "David Park", "Elena Ruiz", "Frank Obi",
    "Grace Kim", "Hassan Ali", "Ivy Chen", "Jack Moore", "Kira Novak", "Liam Walsh",
    "Maya Singh", "Noah Brooks", "Omar Haddad", "Priya Nair", "Quinn Adams", "Rosa Diaz",
]
# Advisors are a DISJOINT pool, so a hard-band advisor distractor is never a founder.
ADVISOR_NAMES = [
    "Walter Crane", "Sylvia Mond", "Theo Vance", "Uma Patel", "Victor Long",
    "Wendy Cho", "Xander Reed", "Yara Salah", "Zoe Frost", "Gabriel Stone",
]
RAISE_AMOUNTS = [
    500_000, 750_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000, 5_000_000,
    8_000_000, 10_000_000, 12_000_000, 15_000_000, 20_000_000, 25_000_000, 40_000_000, 50_000_000,
]
VAL_MULTIPLES = [3, 4, 5, 6, 8, 10]

# --------------------------------------------------------------------------- #
# Templates. Train and OOD pools are DISJOINT (different sentence skeletons).   #
# Placeholders: {company} {round} {raise} {valuation} {founders} and, hard:     #
# {distractor_raise} {advisor} {prior_round}.                                   #
# --------------------------------------------------------------------------- #
EASY_TRAIN_TEMPLATES = [
    "{company} raised {raise} in its {round} round at a pre-money valuation of {valuation}. "
    "The company was founded by {founders}.",
    "Founded by {founders}, {company} closed a {round} round of {raise} at a {valuation} "
    "pre-money valuation.",
    "{company}'s {round} round brought in {raise} on a pre-money valuation of {valuation}; "
    "it was started by {founders}.",
    "In its {round} round, {company} raised {raise} at a pre-money valuation of {valuation}. "
    "{founders} founded the company.",
    "{company}, founded by {founders}, secured {raise} in a {round} round priced at a "
    "{valuation} pre-money valuation.",
]
EASY_OOD_TEMPLATES = [
    "The {round} financing for {company} came to {raise}, set against a {valuation} pre-money "
    "valuation. {founders} launched it.",
    "{founders} are behind {company}, which took in {raise} during a {round} round valued at "
    "{valuation} pre-money.",
    "At a {valuation} pre-money valuation, {company} pulled together a {round} round worth "
    "{raise}; the founders are {founders}.",
    "{company} -- the brainchild of {founders} -- announced a {round} round of {raise} with a "
    "pre-money valuation of {valuation}.",
    "A {round} round at {company} raised {raise} (pre-money valuation: {valuation}). It was "
    "co-founded by {founders}.",
]
# Hard = STRONGER distractors that BITE (Phase-5 re-pilot, one dial = distractor STRENGTH).
# The prior-round amount ({distractor_raise}) is placed where a careless extractor would grab
# it -- often BEFORE the current raise, in parallel "raised X" phrasing -- and the {advisor}
# sits in a founder-adjacent clause. But every item STILL explicitly ties the gold to the
# current/latest {round} round and to "founded by {founders}", so a careful reader recovers
# every field unambiguously (disambiguation the model can get wrong, NOT ambiguity the prose
# fails to resolve). Disambiguation is purely TEMPORAL (earlier/prior/previous vs current/
# latest/more recently) -- no magnitude claim that could be false. WHICH distractor bites and
# WHERE varies across templates, so there is no fixed positional tell to exploit. Train and
# OOD skeletons stay disjoint. (Easy templates + _fill_hard are untouched.)
HARD_TRAIN_TEMPLATES = [
    "{company} raised {distractor_raise} in an earlier {prior_round} round; more recently, its "
    "{round} round brought in {raise} at a {valuation} pre-money valuation. It was founded by "
    "{founders}, and {advisor} advises the board.",
    "Founded by {founders} -- with {advisor} serving as an advisor -- {company} closed its {round} "
    "round at {raise}, on a {valuation} pre-money valuation, after the {distractor_raise} of its "
    "previous {prior_round} round.",
    "After a {prior_round} round that had brought in {distractor_raise}, {company} pressed ahead: "
    "its {round} round raised {raise} at a {valuation} pre-money valuation. The founders are "
    "{founders}; {advisor} is an advisor.",
    "{company} had raised {distractor_raise} back in its {prior_round} days. Advised by {advisor} "
    "and founded by {founders}, it went on to land {raise} in its {round} round, at a {valuation} "
    "pre-money valuation.",
    "{company}'s {round} round came in at {raise}, with a {valuation} pre-money valuation, following "
    "the {distractor_raise} it raised in its {prior_round} round. The team: founders {founders}, "
    "plus advisor {advisor}.",
]
HARD_OOD_TEMPLATES = [
    "{company} secured {distractor_raise} in a {prior_round} round some time ago; its current "
    "{round} round, however, came to {raise} at a {valuation} pre-money valuation. The company was "
    "founded by {founders}, with {advisor} on its advisory board.",
    "Built by {founders} and counseled by advisor {advisor}, {company} priced its {round} round at "
    "{raise} with a {valuation} pre-money valuation, after the {distractor_raise} from its "
    "{prior_round} round.",
    "Its {prior_round} round had drawn {distractor_raise}, but {company} was just getting started: "
    "the {round} round pulled {raise} at a {valuation} pre-money valuation. Founders {founders} ran "
    "the company; {advisor} advised it.",
    "Long ago {company} took in {distractor_raise} during a {prior_round} round. Guided by advisor "
    "{advisor} and started by {founders}, it later secured {raise} in its {round} round, at a "
    "{valuation} pre-money valuation.",
    "The {round} round at {company} landed {raise} on a {valuation} pre-money valuation, compared "
    "with the {distractor_raise} of its {prior_round} chapter. Behind it: founders {founders}, "
    "alongside advisor {advisor}.",
]


# --------------------------------------------------------------------------- #
# Rendering helpers.                                                            #
# --------------------------------------------------------------------------- #
def _money_str(value, style):
    if style == "suffix_M":
        return f"${value / 1e6:g}M"
    if style == "suffix_B":
        return f"${value / 1e9:g}B"
    if style == "suffix_K":
        return f"${value / 1e3:g}K"
    if style == "word_million":
        return f"${value / 1e6:g} million"
    if style == "commas":
        return f"${value:,}"
    if style == "plain":
        return str(value)
    raise ValueError(style)


def _money_styles_for(value):
    styles = ["commas", "plain"]
    if value >= 1_000_000_000:
        styles.append("suffix_B")
    elif value >= 1_000_000:
        styles += ["suffix_M", "word_million"]
    elif value >= 1000 and value % 1000 == 0:
        styles.append("suffix_K")
    return styles


def _render_money(value, rng, mixed):
    styles = _money_styles_for(value)
    if mixed:
        return _money_str(value, rng.choice(styles))
    for pref in ("suffix_M", "suffix_B", "suffix_K", "commas"):
        if pref in styles:
            return _money_str(value, pref)
    return _money_str(value, "plain")


def _render_founders(names):
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


# --------------------------------------------------------------------------- #
# Recoverability check (reuses the grader's parsing, so it is self-consistent). #
# --------------------------------------------------------------------------- #
_MONEY_RE = re.compile(
    r"\$?\s?\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand|mm|bn|[mbk])?", re.IGNORECASE
)


def _all_numbers_in(text):
    out = set()
    for m in _MONEY_RE.finditer(text):
        v = parse_number(m.group(0))
        if v is not None:
            out.add(round(v))
    return out


def _recoverable(record, prose):
    """Every gold field must be recoverable from the prose (names/round appear, numbers parse)."""
    n = normalize_str(prose)
    if normalize_str(record["company"]) not in n:
        return False
    if normalize_str(record["round"]) not in n:
        return False
    for f in record["founders"]:
        if normalize_str(f) not in n:
            return False
    nums = _all_numbers_in(prose)
    return round(record["raise"]) in nums and round(record["valuation"]) in nums


# --------------------------------------------------------------------------- #
# Record sampling + item construction.                                          #
# --------------------------------------------------------------------------- #
def _sample_record(rng):
    raise_amt = rng.choice(RAISE_AMOUNTS)
    n = rng.randint(1, 3)
    return {
        "company": rng.choice(COMPANIES),
        "round": rng.choice(ROUNDS),
        "raise": raise_amt,
        "valuation": raise_amt * rng.choice(VAL_MULTIPLES),  # pre-money > raise, clean multiple
        "founders": rng.sample(FOUNDER_NAMES, n),
    }


def _fill_easy(record, rng):
    fill = {
        "company": record["company"],
        "round": record["round"],
        "raise": _render_money(record["raise"], rng, mixed=False),
        "valuation": _render_money(record["valuation"], rng, mixed=False),
        "founders": _render_founders(record["founders"]),
    }
    info = {"raise_str": fill["raise"], "valuation_str": fill["valuation"], "distractors": []}
    return fill, info


def _fill_hard(record, rng):
    distractor = rng.choice(RAISE_AMOUNTS)
    while distractor in (record["raise"], record["valuation"]):
        distractor = rng.choice(RAISE_AMOUNTS)
    advisor = rng.choice(ADVISOR_NAMES)
    prior_round = rng.choice([r for r in ROUNDS if r != record["round"]])
    fill = {
        "company": record["company"],
        "round": record["round"],
        "raise": _render_money(record["raise"], rng, mixed=True),
        "valuation": _render_money(record["valuation"], rng, mixed=True),
        "founders": _render_founders(record["founders"]),
        "distractor_raise": _render_money(distractor, rng, mixed=True),
        "advisor": advisor,
        "prior_round": prior_round,
    }
    info = {"raise_str": fill["raise"], "valuation_str": fill["valuation"],
            "distractors": [fill["distractor_raise"], advisor], "prior_round": prior_round}
    return fill, info


def _generate_b(n, difficulty, templates, rng):
    maker = _fill_easy if difficulty == "easy" else _fill_hard
    items, seen, tries = [], set(), 0
    budget = n * 200 + 1000
    while len(items) < n and tries < budget:
        tries += 1
        record = _sample_record(rng)
        fill, info = maker(record, rng)
        prose = rng.choice(templates).format(**fill)
        prompt = prose + " " + TASK_B_INSTRUCTION
        if prompt in seen:
            continue
        # Build-time sanity: gold must be recoverable from its own prose.
        assert _recoverable(record, prose), f"gold not recoverable: {record} || {prose}"
        seen.add(prompt)
        info["prose"] = prose
        items.append({"prompt": prompt, "answer": record, "difficulty": difficulty, "info": info})
    while len(items) < n:  # rare top-up if the unique space was exhausted
        record = _sample_record(rng)
        fill, info = maker(record, rng)
        prose = rng.choice(templates).format(**fill)
        assert _recoverable(record, prose)
        info["prose"] = prose
        items.append({"prompt": prose + " " + TASK_B_INSTRUCTION, "answer": record,
                      "difficulty": difficulty, "info": info})
    return items


def build_all_b(seed, n_easy, n_hard, n_ood_easy, n_ood_hard):
    """Build all three Task B datasets deterministically from a single seed."""
    train_easy = _generate_b(n_easy, "easy", EASY_TRAIN_TEMPLATES, random.Random(seed + 1))
    train_hard = _generate_b(n_hard, "hard", HARD_TRAIN_TEMPLATES, random.Random(seed + 2))

    rng_ood = random.Random(seed + 3)
    ood_easy = _generate_b(n_ood_easy, "easy", EASY_OOD_TEMPLATES, rng_ood)
    ood_hard = _generate_b(n_ood_hard, "hard", HARD_OOD_TEMPLATES, rng_ood)
    ood = ood_easy + ood_hard
    rng_ood.shuffle(ood)

    # Build-time asserts (fail loudly on a bad generator).
    train_templates = set(EASY_TRAIN_TEMPLATES) | set(HARD_TRAIN_TEMPLATES)
    ood_templates = set(EASY_OOD_TEMPLATES) | set(HARD_OOD_TEMPLATES)
    assert train_templates.isdisjoint(ood_templates), "train/OOD template pools overlap"
    assert len(train_easy) == n_easy and len(train_hard) == n_hard
    assert len(ood) == n_ood_easy + n_ood_hard

    return {"train_easy": train_easy, "train_hard": train_hard, "ood_test": ood}

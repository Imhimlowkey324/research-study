# FROZEN config — mirrors PILOT_LOG.md. Do not change after runs begin.
#
# Single source of truth imported by every run (this baseline now; training and
# judging later) so settings can never drift apart.

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Worked-example system prompt — copied VERBATIM from the ```text block in
# PILOT_LOG.md. Must stay byte-identical; tests/test_eval.py asserts it against
# the file. Do not retype or paraphrase.
SYSTEM_PROMPT = "You solve a short math problem. Compute ownership% = raise / (pre_money + raise) * 100. Show at most 2 short steps, then STOP and write exactly: 'The answer is X' where X is the number rounded to 2 decimals. Example: 'Raise 5M, pre-money 20M. Post-money = 25M. 5/25*100 = 20. The answer is 20.'"

# Worked-example Task B system prompt (extraction -> JSON). Mirrors Task A's terse
# style; pre-registered in TASKB_PREREG.md (a test asserts they match). The frozen
# Task-A SYSTEM_PROMPT above is NOT touched.
SYSTEM_PROMPT_B = (
    'You extract structured data from a short text. Output ONLY a JSON object with keys '
    'company, round, raise, valuation, founders -- no prose and no code fence. raise is the '
    'amount raised in dollars and valuation is the pre-money valuation in dollars, both as '
    'plain integers; founders is a list of names. '
    'Example: text "Acme raised $5M in its Series A at a $20M pre-money valuation, founded by '
    'Jo Lee." -> {"company": "Acme", "round": "Series A", "raise": 5000000, "valuation": '
    '20000000, "founders": ["Jo Lee"]}'
)

# Task-B-only override. LOCKED at 384 by the Phase-5 Part-2 pilot/protocol ruling: it is
# both the Task-B training completion length AND the Task-B judging max_new_tokens (pilot max
# tokens seen = 105, so 384 has wide headroom). Excluded from snapshot() (ends with _B), so no
# Task-A value changes. Feeds judge_config_B -> JUDGE_CONFIG_B_SHA256 (PHASE5_JUDGE_PROTOCOL_B.md).
MAX_COMPLETION_LENGTH_B = 384

# Generation budget.
MAX_NEW_TOKENS = 768

# Greedy decode — the correctness / format-validity pass.
GREEDY_GEN_KWARGS = {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS}

# Sampling decode — the Pass@k pass.
PASS_K = 4
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P = 0.95
SAMPLE_GEN_KWARGS = {
    "do_sample": True,
    "temperature": SAMPLE_TEMPERATURE,
    "top_p": SAMPLE_TOP_P,
    "max_new_tokens": MAX_NEW_TOKENS,
}

# Data — frozen seed + sizes (must match how the JSONL data is generated).
DATA_SEED = 0
N_EASY = 500
N_HARD = 500
N_OOD_EASY = 100
N_OOD_HARD = 100

# Evaluation — seed set before the sampling pass so Pass@k is reproducible.
EVAL_SEED = 0


# DRAFT training hyperparameters — validated by the Part 2 smoke test, finalized
# before the 12 runs.

# LoRA adapter (peft).
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Attention + MLP projection layers of Qwen2.5 (targeted explicitly).
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# GRPO (TRL GRPOConfig).
NUM_GENERATIONS = 4                   # group size G (trimmed 6->4 to fit the 16GB T4)
LEARNING_RATE = 1e-5
KL_BETA = 0.04                        # KL coefficient (GRPOConfig.beta)
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 384           # trimmed 512->384 to fit the 16GB T4
GRAD_CHECKPOINTING = True
GRAD_CLIP = 1.0                       # max_grad_norm
GRADIENT_ACCUMULATION_STEPS = 6       # 4->6 so effective batch (4 gen x 6) ~ prior (6 gen x 4)
TRAIN_EXAMPLES = 256                  # total training items; held FIXED across the
                                      # easy vs easy_hard conditions (only the mix changes)
MAX_STEPS = 200                       # draft cap for a real run
LOGGING_STEPS = 1

# Precision: fp16 (the T4 is Turing — do NOT use bf16). NOTE: fp16 + RL can be
# unstable; GRAD_CLIP is on and the trainer watches for NaN/inf loss.
USE_FP16 = True
USE_BF16 = False

# Tiny throwaway overrides for the Part 2 smoke test (dress rehearsal only).
SMOKE = {
    "max_steps": 8,
    "num_generations": 4,
    "max_completion_length": 256,
    "train_examples": 24,             # easy-only
    "logging_steps": 1,
    "gradient_accumulation_steps": 1,
}


def snapshot() -> dict:
    """All frozen Task-A config constants, for printing on screen and saving with results.

    Task-B-specific constants (suffixed ``_B``, e.g. SYSTEM_PROMPT_B) are excluded so the
    Task-A config snapshot stays byte-identical now that Task B constants also exist.
    """
    return {
        name: value
        for name, value in globals().items()
        if name.isupper() and not name.startswith("_") and not name.endswith("_B")
    }

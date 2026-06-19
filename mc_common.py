"""Shared, tokenizer-free helpers for the MMLU uncertainty experiment.

Pure logic only: prompt formatting and output parsing. No torch, no tokenizer,
no model or dataset loading happens here, so every function in this module is
safe to exercise on a login node. Helpers that need a tokenizer or torch
(letter token ids, think-block splitting, token entropy) live with the step
that first uses them, inside a SLURM job.
"""

import re

LETTERS = "ABCD"

# What we ask the model to do. The strict two-line ending is what
# parse_answer_confidence keys off of, so keep the format exact.
SYSTEM_INSTRUCTION = (
    "You are answering a multiple-choice question. Think step by step. "
    "After your reasoning, end your reply with EXACTLY these two lines and "
    "nothing after them:\n"
    "Answer: <one of A, B, C, D>\n"
    "Confidence: <an integer from 0 to 100>"
)

# Match the answer letter and the confidence integer anywhere in the text.
# Case-insensitive so "answer: b" still parses; \d{1,3} captures 0-100 (and
# stray larger values, which we clamp). We take the LAST occurrence of each:
# the model reasons out loud and may write a decoy "Answer:" mid-reasoning,
# but the real answer lines are last (per SYSTEM_INSTRUCTION).
_ANSWER_RE = re.compile(r"Answer:\s*([ABCD])", re.IGNORECASE)
_CONF_RE = re.compile(r"Confidence:\s*(\d{1,3})", re.IGNORECASE)


def build_question_prompt(question, choices):
    """Render a question and its four choices as an A./B./C./D. block."""
    lines = [question, ""]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def build_messages(question, choices):
    """Build the chat messages list (system + user) for one question.

    Returns plain dicts; the chat template is applied later inside the job,
    where the tokenizer is available.
    """
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": build_question_prompt(question, choices)},
    ]


def parse_answer_confidence(text):
    """Extract (letter, confidence) from a model reply.

    Tolerant of casing and surrounding text. Confidence is clamped to
    [0, 100]. Either element is None if it is absent, so a missing answer
    line yields (None, ...) instead of raising.
    """
    answer_matches = _ANSWER_RE.findall(text)
    conf_matches = _CONF_RE.findall(text)

    letter = answer_matches[-1].upper() if answer_matches else None
    confidence = None
    if conf_matches:
        confidence = max(0, min(100, int(conf_matches[-1])))
    return letter, confidence


def is_correct(pred_letter, gold_idx):
    """Compare a predicted letter to the gold choice index (0-3).

    Returns None when there is no prediction to score.
    """
    if pred_letter is None:
        return None
    return pred_letter == LETTERS[gold_idx]

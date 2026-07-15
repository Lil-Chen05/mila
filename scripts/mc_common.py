"""Shared, tokenizer-free helpers for the MMLU uncertainty experiment.

Pure logic only: prompt formatting, output parsing, and answer-token location
(the latter over already-decoded token strings, never raw ids). No torch, no
tokenizer, no model or dataset loading happens here, so every function in this
module is safe to exercise on a login node. Helpers that need a tokenizer or
torch (letter token ids, think-block splitting, token entropy) live with the
step that first uses them, inside a SLURM job.
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

# Match the answer letter and the confidence integer anywhere in the text,
# tolerant of the markdown the model sometimes wraps the labels in
# (**Answer:** B, *Confidence:* 90, `Answer`: C) and of whitespace drift
# ("Answer : B"). _EMPH is an optional run of markdown emphasis chars that may
# appear before/after the label word and after the colon. Group 1 stays JUST
# the letter / number, so m.start(1) lands on the letter itself --
# find_answer_token relies on that. Case-insensitive so "answer: b" still
# parses; \d{1,3} captures 0-100 (stray larger values are clamped). We take the
# LAST occurrence of each: the model reasons out loud and may write a decoy
# "Answer:" mid-reasoning, but the real answer lines are last (per
# SYSTEM_INSTRUCTION).
_EMPH = r"[*_`]*"  # optional markdown run: ** * _ `
ANSWER_RE = re.compile(rf"{_EMPH}Answer{_EMPH}\s*:\s*{_EMPH}\s*([ABCD])", re.IGNORECASE)
CONF_RE = re.compile(rf"{_EMPH}Confidence{_EMPH}\s*:\s*{_EMPH}\s*(\d{{1,3}})", re.IGNORECASE)


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
    answer_matches = ANSWER_RE.findall(text)
    conf_matches = CONF_RE.findall(text)

    letter = answer_matches[-1].upper() if answer_matches else None
    confidence = None
    if conf_matches:
        confidence = max(0, min(100, int(conf_matches[-1])))
    return letter, confidence


def find_answer_token(decoded_tokens):
    """Locate the answer-letter token by anchoring to the LAST "Answer:" marker.

    Pure over the decoded token STRINGS -- no torch, no tokenizer -- so it stays
    in this login-safe module. Two spaces, kept separate: a fuzzy TEXT match
    finds the marker (it may span several tokens or fuse with whitespace), then
    an EXACT char->token map -- built from the SAME joined pieces we matched
    against, so there is no whole-vs-piecewise spacing mismatch -- recovers the
    token index whose logits produced the letter. Uses the shared ANSWER_RE, so
    it inherits the markdown / whitespace tolerance and the group-1-is-the-letter
    invariant. Not gated on </think>, so an answer committed inside the think
    block is found too. Returns (token_index_or_None, located_letter_or_None).
    """
    spans = []
    pos = 0
    for s in decoded_tokens:
        spans.append((pos, pos + len(s)))
        pos += len(s)
    recon = "".join(decoded_tokens)
    matches = list(ANSWER_RE.finditer(recon))
    if not matches:
        return None, None
    m = matches[-1]
    cpos = m.start(1)                 # char index of the letter itself
    located_letter = m.group(1).upper()
    for i, (a, b) in enumerate(spans):
        if a <= cpos < b:
            return i, located_letter
    return None, located_letter


def is_correct(pred_letter, gold_idx):
    """Compare a predicted letter to the gold choice index (0-3).

    Returns None when there is no prediction to score.
    """
    if pred_letter is None:
        return None
    return pred_letter == LETTERS[gold_idx]

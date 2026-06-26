"""Login-safe pytest for the hardened answer/confidence parser.

Pure-Python string tests over mc_common -- NO model, NO torch, NO tokenizer, NO
dataset is imported, so this is safe to run on a login node:

    uv run pytest test_mc_common.py -q

Covers the markdown / whitespace tolerance of ANSWER_RE/CONF_RE via
parse_answer_confidence, the last-match (decoy) discipline, the None-on-absent
discipline, the [0, 100] clamp, and the char->token span mapping in
find_answer_token.
"""

import pytest

from mc_common import parse_answer_confidence, find_answer_token


# --- per-shape parser: each asserts the FULL (letter, confidence) tuple --------
# Shapes the model actually emits: markdown emphasis around the label, drift
# whitespace before the colon, lowercase + surrounding spaces, backticks. The
# baseline plain shape is included as the last row.
@pytest.mark.parametrize(
    "text, letter, conf",
    [
        ("**Answer:** B\n**Confidence:** 95", "B", 95),
        ("*Answer:* C\n*Confidence:* 60", "C", 60),
        ("**Answer: D**\n**Confidence: 40**", "D", 40),
        ("Answer : A\nConfidence : 80", "A", 80),
        (" answer: b \n confidence: 33 ", "B", 33),   # lowercase -> .upper()
        ("`Answer:` B\n`Confidence:` 50", "B", 50),
        ("Answer: C\nConfidence: 75", "C", 75),        # baseline
    ],
)
def test_parse_shapes(text, letter, conf):
    assert parse_answer_confidence(text) == (letter, conf)


# --- unbalanced markdown: permissive on purpose, pin it (don't forbid it) ------
def test_unbalanced_markdown_still_parses():
    # opening ** but closing single * -- we WANT this to still resolve to B.
    letter, _ = parse_answer_confidence("**Answer:* B")
    assert letter == "B"


# --- decoy: pins last-match, not merely found-vs-None --------------------------
DECOY = (
    "Working through it: my first instinct was Answer: A with Confidence: 10.\n"
    "But re-checking the units flips the conclusion.\n"
    "Answer: C\n"
    "Confidence: 88"
)


def test_decoy_takes_last_match():
    # A first-match regression would yield ("A", 10) -- a wrong, non-None answer.
    assert parse_answer_confidence(DECOY) == ("C", 88)


# --- combined q06 shape: last-match AND markdown together ----------------------
Q06 = (
    "Hmm, Answer: A seems plausible at first glance.\n"
    "After eliminating distractors, the remaining option is correct.\n"
    "**Answer:** B\n"
    "**Confidence:** 72"
)


def test_combined_q06_shape():
    assert parse_answer_confidence(Q06) == ("B", 72)


# --- None-discipline: documented (None, ...) shapes, exact 2-tuple -------------
def test_no_match_returns_none_none():
    assert parse_answer_confidence("there is no answer or confidence here") == (None, None)


def test_answer_only():
    assert parse_answer_confidence("Answer: B") == ("B", None)


def test_confidence_only():
    assert parse_answer_confidence("Confidence: 90") == (None, 90)


# --- confidence clamp: upper bound --------------------------------------------
def test_confidence_clamped_to_100():
    assert parse_answer_confidence("Answer: A\nConfidence: 250") == ("A", 100)


# --- find_answer_token: realistic multi-token spans, assert (idx, letter) ------
# decoded_tokens concatenate to the reconstructed string (the precondition the
# char->token span map relies on); the marker spans several tokens before the
# letter, so this is not a single-token cheat.
@pytest.mark.parametrize(
    "toks, idx, letter",
    [
        (["Answer", ":", " B"], 2, "B"),                  # bare      -> "Answer: B"
        (["Answer", " :", " B"], 2, "B"),                 # drift     -> "Answer : B"
        (["**", "Answer", ":", "**", " B"], 4, "B"),      # markdown  -> "**Answer:** B"
    ],
)
def test_find_answer_token_lands_on_letter(toks, idx, letter):
    result = find_answer_token(toks)
    assert result == (idx, letter)
    assert letter in toks[idx]   # returned index decodes to the token holding the letter


def test_find_answer_token_no_marker():
    assert find_answer_token(["no", " marker", " here"]) == (None, None)

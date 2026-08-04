"""Pure SmolLM3 adapter contract tests; no model or real tokenizer is loaded."""

from __future__ import annotations

import copy

import pytest

from part1_smollm3_adapter import (
    FORCED_CLOSE_INDUCER,
    REASONING_CLOSE_TAG,
    REASONING_OPEN_TAG,
    SmolLM3AdapterError,
    locate_reasoning_tokens,
    parse_forced_output,
    parse_natural_output,
    preflight_tokenizer_contract,
    render_question_prompt,
    validate_context_budget,
)


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.encodings = {
            REASONING_OPEN_TAG: [10],
            REASONING_CLOSE_TAG: [11],
            "<|assistant|>\n": [30, 31],
            FORCED_CLOSE_INDUCER: [11, 12],
            f"{FORCED_CLOSE_INDUCER} A": [11, 12, 20],
            f"{FORCED_CLOSE_INDUCER} B": [11, 12, 21],
            f"{FORCED_CLOSE_INDUCER} C": [11, 12, 22],
            f"{FORCED_CLOSE_INDUCER} D": [11, 12, 23],
        }

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.calls.append(("encode", text, add_special_tokens))
        return list(self.encodings[text])

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(("chat", copy.deepcopy(messages), kwargs))
        suffix = "<|assistant|>\n" if kwargs["add_generation_prompt"] else ""
        return "rendered-chat" + suffix


def question() -> dict:
    return {
        "question": "What is 2 + 2?",
        "choices": ["1", "2", "4", "5"],
    }


def test_render_question_prompt_uses_thinking_chat_template_and_fixed_answer_contract() -> None:
    tokenizer = FakeTokenizer()
    prompt = render_question_prompt(tokenizer, question())

    assert prompt == "rendered-chat<|assistant|>\n"
    chat_call = next(call for call in tokenizer.calls if call[0] == "chat")
    messages = chat_call[1]
    assert chat_call[2] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }
    content = messages[0]["content"]
    assert "A. 1" in content and "D. 5" in content
    assert "Answer: <A|B|C|D>" in content
    assert "Confidence: <integer 0-100>" in content


def test_tokenizer_preflight_persists_exact_boundary_tokens_without_mutating_pad() -> None:
    tokenizer = FakeTokenizer()
    contract = preflight_tokenizer_contract(tokenizer)

    assert contract == {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": None,
        "reasoning_open_tag": REASONING_OPEN_TAG,
        "reasoning_open_token_ids": [10],
        "reasoning_open_tag_origin": "generated_output",
        "assistant_generation_suffix_text": "<|assistant|>\n",
        "assistant_generation_suffix_token_ids": [30, 31],
        "reasoning_close_tag": REASONING_CLOSE_TAG,
        "reasoning_close_token_ids": [11],
        "inducer_text": FORCED_CLOSE_INDUCER,
        "inducer_token_ids": [11, 12],
        "ad_token_convention": "inducer_boundary_space_uppercase_single_token",
        "ad_raw_token_sequences": {"A": [20], "B": [21], "C": [22], "D": [23]},
        "ad_token_ids": [20, 21, 22, 23],
    }
    assert tokenizer.pad_token_id is None
    assert all(call[-1] is False for call in tokenizer.calls if call[0] == "encode")


def test_tokenizer_preflight_rejects_non_single_or_non_distinct_choice_tokens() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.encodings[f"{FORCED_CLOSE_INDUCER} D"] = [11, 12, 22]
    with pytest.raises(SmolLM3AdapterError, match="distinct"):
        preflight_tokenizer_contract(tokenizer)

    tokenizer = FakeTokenizer()
    tokenizer.encodings[f"{FORCED_CLOSE_INDUCER} A"] = [11, 12, 20, 30]
    with pytest.raises(SmolLM3AdapterError, match="single token"):
        preflight_tokenizer_contract(tokenizer)


def test_reasoning_boundaries_support_open_tag_in_generation_or_prompt() -> None:
    generated_open = locate_reasoning_tokens(
        generated_token_ids=[10, 100, 101, 11, 200],
        prompt_token_ids=[1, 2],
        open_token_ids=[10],
        close_token_ids=[11],
    )
    assert generated_open.reasoning_status == "closed"
    assert generated_open.reasoning_indices == (1, 2)
    assert generated_open.reasoning_token_ids == (100, 101)
    assert generated_open.reasoning_boundaries == {
        "generated_start": 1,
        "generated_end_exclusive": 3,
    }
    assert generated_open.close_tag_information == {
        "found": True,
        "generated_start": 3,
        "generated_end_exclusive": 4,
    }

    prompt_open = locate_reasoning_tokens(
        generated_token_ids=[100, 11, 200],
        prompt_token_ids=[1, 10],
        open_token_ids=[10],
        close_token_ids=[11],
    )
    assert prompt_open.reasoning_indices == (0,)
    assert prompt_open.reasoning_token_ids == (100,)


def test_reasoning_boundaries_preserve_missing_close_and_zero_reasoning() -> None:
    missing = locate_reasoning_tokens(
        generated_token_ids=[10, 100, 101],
        prompt_token_ids=[],
        open_token_ids=[10],
        close_token_ids=[11],
    )
    assert missing.reasoning_status == "missing_close"
    assert missing.reasoning_token_ids == (100, 101)
    assert missing.close_tag_information["found"] is False

    empty = locate_reasoning_tokens(
        generated_token_ids=[11, 200],
        prompt_token_ids=[10],
        open_token_ids=[10],
        close_token_ids=[11],
    )
    assert empty.reasoning_status == "no_reasoning"
    assert empty.reasoning_token_ids == ()


def test_natural_parser_uses_only_adjacent_terminal_post_close_pair() -> None:
    output = (
        "<think>Maybe Answer: B\nConfidence: 12</think>\n"
        "Some transition.\nAnswer: C\nConfidence: 80\n"
    )
    parsed = parse_natural_output(output)
    assert parsed.answer == "C"
    assert parsed.answer_parse_status == "parsed"
    assert parsed.raw_confidence_text == "80"
    assert parsed.raw_parsed_confidence == 80
    assert parsed.normalized_confidence == 0.8
    assert parsed.terminal_answer_block_text == "Answer: C\nConfidence: 80"

    unrelated = parse_natural_output(
        "<think>x</think>\nAnswer: A\nExplanation in between\nConfidence: 99"
    )
    assert unrelated.answer == "A"
    assert unrelated.answer_parse_status == "parsed"
    assert unrelated.confidence_parse_status == "missing"
    assert unrelated.raw_parsed_confidence is None


def test_natural_parser_missing_close_never_promotes_answer_like_text() -> None:
    parsed = parse_natural_output("<think>unfinished Answer: D\nConfidence: 90")
    assert parsed.reasoning_close_found is False
    assert parsed.answer is None
    assert parsed.answer_parse_status == "missing"
    assert parsed.confidence_parse_status == "missing"
    assert parsed.diagnostic_answer_like_text == "Answer: D"


def test_confidence_out_of_range_and_forced_suffix_pairing_are_preserved() -> None:
    out_of_range = parse_natural_output(
        "<think>x</think>\nAnswer: B\nConfidence: 101"
    )
    assert out_of_range.confidence_parse_status == "out_of_range"
    assert out_of_range.raw_confidence_text == "101"
    assert out_of_range.raw_parsed_confidence == 101
    assert out_of_range.normalized_confidence is None

    forced = parse_forced_output(" C\nConfidence: 75")
    assert forced.answer == "C"
    assert forced.raw_parsed_confidence == 75
    assert forced.terminal_answer_block_text == "Answer: C\nConfidence: 75"


def test_malformed_answer_and_confidence_are_preserved_without_coercion() -> None:
    parsed = parse_natural_output(
        "<think>x</think>\nAnswer: maybe B\nConfidence: probably 80"
    )
    assert parsed.answer is None
    assert parsed.answer_parse_status == "out_of_domain"
    assert parsed.raw_confidence_text == "probably 80"
    assert parsed.raw_parsed_confidence is None
    assert parsed.confidence_parse_status == "malformed"
    assert parsed.normalized_confidence is None


def test_context_budget_checks_natural_and_checkpoint_limits() -> None:
    assert validate_context_budget(
        model_context_window=9000,
        prompt_token_count=700,
        natural_max_new_tokens=8192,
        longest_checkpoint_prefix_tokens=8000,
        inducer_token_count=10,
        checkpoint_max_new_tokens=32,
    )["natural_required_tokens"] == 8892

    with pytest.raises(SmolLM3AdapterError, match="natural.*context"):
        validate_context_budget(
            model_context_window=8192,
            prompt_token_count=1,
            natural_max_new_tokens=8192,
            longest_checkpoint_prefix_tokens=10,
            inducer_token_count=2,
            checkpoint_max_new_tokens=32,
        )
    with pytest.raises(SmolLM3AdapterError, match="checkpoint.*context"):
        validate_context_budget(
            model_context_window=100,
            prompt_token_count=1,
            natural_max_new_tokens=10,
            longest_checkpoint_prefix_tokens=80,
            inducer_token_count=10,
            checkpoint_max_new_tokens=32,
        )

"""Pure preflight assembly tests; GPU execution remains unverified."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest

import part1_smollm3_preflight as preflight
from part1_smollm3_preflight import (
    build_effective_generation_settings,
    resolve_context_window,
)
from part1_contract import canonical_json_bytes


class Config:
    max_position_embeddings = 65536


class Tokenizer:
    model_max_length = 32768


class JsonConfig:
    def __init__(self, in_memory: dict, serialized: str):
        self.in_memory = in_memory
        self.serialized = serialized

    def to_dict(self) -> dict:
        return self.in_memory

    def to_json_string(self, *, use_diff: bool) -> str:
        assert use_diff is False
        return self.serialized


def test_model_config_hash_uses_json_serialization_to_canonicalize_integer_keys() -> None:
    config = JsonConfig(
        {"id2label": {0: "LABEL_0", 1: "LABEL_1"}},
        '{"id2label":{"0":"LABEL_0","1":"LABEL_1"}}',
    )

    with pytest.raises(TypeError, match="keys must be strings"):
        canonical_json_bytes(config.to_dict())

    assert preflight.model_config_sha256(config) == hashlib.sha256(
        canonical_json_bytes({"id2label": {"0": "LABEL_0", "1": "LABEL_1"}})
    ).hexdigest()


@pytest.mark.parametrize(
    ("serialized", "message"),
    [
        ("not JSON", "valid JSON"),
        ("[]", "JSON object"),
    ],
)
def test_model_config_hash_rejects_invalid_or_non_object_json(
    serialized: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        preflight.model_config_sha256(JsonConfig({}, serialized))


def test_context_window_uses_smallest_finite_model_and_tokenizer_limit() -> None:
    assert resolve_context_window(Config(), Tokenizer()) == 32768
    Tokenizer.model_max_length = 10**30
    assert resolve_context_window(Config(), Tokenizer()) == 65536


def test_context_window_fails_when_neither_source_is_usable() -> None:
    class Empty:
        pass

    with pytest.raises(ValueError, match="context window"):
        resolve_context_window(Empty(), Empty())


def test_effective_settings_preserve_requested_values_and_special_tokens() -> None:
    natural, checkpoint = build_effective_generation_settings(
        {"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": None}
    )
    assert natural["do_sample"] is True
    assert natural["temperature"] == 0.6
    assert natural["max_new_tokens"] == 8192
    assert natural["batch_size"] == 1
    assert natural["pad_token_id"] is None
    assert checkpoint["do_sample"] is False
    assert checkpoint["max_new_tokens"] == 32
    assert checkpoint["batch_size"] == 1


def questions() -> list[dict]:
    return [
        {
            "question_id": f"{index:064x}",
            "sample_index": index,
            "subject": "high_school_mathematics",
            "question": "LATER_LONGEST" if index == 437 else f"short-{index}",
            "choices": ["one", "two", "three", "four"],
        }
        for index in range(500)
    ]


class PopulationTokenizer:
    assistant_suffix = "<|assistant|>\n"

    def __init__(self, *, bad_suffix_index: int | None = None):
        self.calls = 0
        self.render_calls = 0
        self.bad_suffix_index = bad_suffix_index

    def apply_chat_template(self, messages, **_kwargs):
        index = self.render_calls
        self.render_calls += 1
        suffix = "<|wrong|>\n" if index == self.bad_suffix_index else self.assistant_suffix
        return messages[0]["content"] + suffix

    def __call__(self, prompt, *, add_special_tokens):
        assert add_special_tokens is False
        self.calls += 1
        length = 200 if "LATER_LONGEST" in prompt else 10
        suffix = [90, 91] if prompt.endswith(self.assistant_suffix) else [90, 92]
        return {"input_ids": [[*range(length - len(suffix)), *suffix]]}


def token_contract() -> dict:
    return {
        "reasoning_open_token_ids": [99],
        "reasoning_open_tag_origin": "generated_output",
        "assistant_generation_suffix_text": PopulationTokenizer.assistant_suffix,
        "assistant_generation_suffix_token_ids": [90, 91],
        "inducer_token_ids": [7, 8],
    }


def test_all_500_prompts_are_validated_and_later_longest_controls_context() -> None:
    from part1_smollm3_preflight import validate_all_question_prompts

    tokenizer = PopulationTokenizer()
    report = validate_all_question_prompts(
        questions(),
        tokenizer=tokenizer,
        token_contract=token_contract(),
        model_context_window=9000,
    )

    assert tokenizer.calls == 500
    assert report["validated_prompt_count"] == 500
    assert report["maximum_prompt_token_count"] == 200
    assert report["worst_sample_identity"] == {
        "question_id": f"{437:064x}",
        "sample_index": 437,
        "subject": "high_school_mathematics",
    }
    assert report["natural_required_tokens"] == 200 + 8192
    assert report["checkpoint_required_tokens"] == 200 + 8192 + 2 + 32


def test_generated_reasoning_open_is_not_required_in_prompt_suffix() -> None:
    from part1_smollm3_preflight import validate_all_question_prompts

    class GeneratedOpenTokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return messages[0]["content"] + "<|assistant|>\n"

        def __call__(self, prompt, *, add_special_tokens):
            assert add_special_tokens is False
            assert prompt.endswith("<|assistant|>\n")
            return {"input_ids": [[1, 2, 90, 91]]}

    report = validate_all_question_prompts(
        questions(),
        tokenizer=GeneratedOpenTokenizer(),
        token_contract={
            "reasoning_open_token_ids": [99],
            "assistant_generation_suffix_text": "<|assistant|>\n",
            "assistant_generation_suffix_token_ids": [90, 91],
            "inducer_token_ids": [7, 8],
        },
        model_context_window=9000,
    )

    assert report["validated_prompt_count"] == 500


def test_later_prompt_with_wrong_reasoning_open_suffix_fails_preflight() -> None:
    from part1_smollm3_preflight import validate_all_question_prompts

    with pytest.raises(ValueError, match="sample_index 499.*assistant-generation suffix"):
        validate_all_question_prompts(
            questions(),
            tokenizer=PopulationTokenizer(bad_suffix_index=499),
            token_contract=token_contract(),
            model_context_window=9000,
        )


def test_greedy_open_validation_persists_evidence_and_fails_closed() -> None:
    from part1_smollm3_preflight import validate_greedy_reasoning_open

    assert validate_greedy_reasoning_open(
        expected_open_token_ids=[99], observed_token_id=99
    ) == {
        "reasoning_open_tag_origin": "generated_output",
        "expected_reasoning_open_token_ids": [99],
        "observed_greedy_next_token_id": 99,
        "matches_expected": True,
    }

    with pytest.raises(ValueError, match="greedy next token.*98.*expected.*99"):
        validate_greedy_reasoning_open(
            expected_open_token_ids=[99], observed_token_id=98
        )

    with pytest.raises(ValueError, match="exactly one token"):
        validate_greedy_reasoning_open(
            expected_open_token_ids=[99, 100], observed_token_id=99
        )


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_dirty_git_provenance_hash_includes_untracked_scoped_path_and_bytes(
    tmp_path: Path,
) -> None:
    from part1_smollm3_preflight import _git_provenance

    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Synthetic Test")
    _git(tmp_path, "config", "user.email", "synthetic@example.invalid")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "tracked.py").write_text("tracked\n", encoding="utf-8")
    _git(tmp_path, "add", "scripts/tracked.py")
    _git(tmp_path, "commit", "-qm", "synthetic base")

    base_commit, clean_hash = _git_provenance(tmp_path)
    untracked = tmp_path / "scripts" / "untracked.py"
    untracked.write_text("first content\n", encoding="utf-8")
    same_base, first_hash = _git_provenance(tmp_path)
    untracked.write_text("second content\n", encoding="utf-8")
    _, second_hash = _git_provenance(tmp_path)

    assert same_base == base_commit
    assert first_hash != clean_hash
    assert second_hash != first_hash

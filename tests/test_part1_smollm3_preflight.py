"""Pure preflight assembly tests; GPU execution remains unverified."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from part1_smollm3_preflight import (
    build_effective_generation_settings,
    resolve_context_window,
)


class Config:
    max_position_embeddings = 65536


class Tokenizer:
    model_max_length = 32768


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
    def __init__(self, *, bad_suffix_index: int | None = None):
        self.calls = 0
        self.bad_suffix_index = bad_suffix_index

    def apply_chat_template(self, messages, **_kwargs):
        return messages[0]["content"]

    def __call__(self, prompt, *, add_special_tokens):
        assert add_special_tokens is False
        index = self.calls
        self.calls += 1
        length = 200 if "LATER_LONGEST" in prompt else 10
        suffix = 98 if index == self.bad_suffix_index else 99
        return {"input_ids": [[*range(length - 1), suffix]]}


def test_all_500_prompts_are_validated_and_later_longest_controls_context() -> None:
    from part1_smollm3_preflight import validate_all_question_prompts

    tokenizer = PopulationTokenizer()
    report = validate_all_question_prompts(
        questions(),
        tokenizer=tokenizer,
        token_contract={"reasoning_open_token_ids": [99], "inducer_token_ids": [7, 8]},
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


def test_later_prompt_with_wrong_reasoning_open_suffix_fails_preflight() -> None:
    from part1_smollm3_preflight import validate_all_question_prompts

    with pytest.raises(ValueError, match="sample_index 499.*reasoning open tag"):
        validate_all_question_prompts(
            questions(),
            tokenizer=PopulationTokenizer(bad_suffix_index=499),
            token_contract={
                "reasoning_open_token_ids": [99],
                "inducer_token_ids": [7, 8],
            },
            model_context_window=9000,
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

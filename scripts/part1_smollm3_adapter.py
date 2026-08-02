"""SmolLM3-specific Part 1 prompt, token-boundary, and loading adapter.

All heavyweight imports are deferred to GPU-only functions. Pure helpers are
safe to import and test on a login node with explicit fake token sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence


MODEL_REPOSITORY = "HuggingFaceTB/SmolLM3-3B"
TOKENIZER_REPOSITORY = MODEL_REPOSITORY
ADAPTER_VERSION = "part1-smollm3-adapter-v1"
PROMPT_VERSION = "part1-smollm3-mcq-v1"
PARSER_VERSION = "part1-terminal-block-v1"
INDUCER_VERSION = "part1-smollm3-forced-close-v1"
REASONING_OPEN_TAG = "<think>"
REASONING_CLOSE_TAG = "</think>"
FORCED_CLOSE_INDUCER = f"{REASONING_CLOSE_TAG}\nAnswer:"
ANSWER_TOKEN_CONVENTION = "inducer_boundary_space_uppercase_single_token"
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_ANSWER_LINE = re.compile(r"(?m)^[ \t]*Answer:[ \t]*(?P<answer>[^\r\n]*?)[ \t]*$")
_ANSWER_LIKE = re.compile(r"Answer:[ \t]*(?P<answer>[^\r\n]*?)(?=[ \t]*\r?$)", re.MULTILINE)
_ADJACENT_CONFIDENCE = re.compile(
    r"\r?\n[ \t]*Confidence:[ \t]*(?P<confidence>[^\r\n]*?)[ \t]*(?=\r?\n|\Z)"
)
_INTEGER = re.compile(r"^[+-]?\d+$")


class SmolLM3AdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReasoningLocation:
    reasoning_status: str
    reasoning_indices: tuple[int, ...]
    reasoning_token_ids: tuple[int, ...]
    reasoning_boundaries: dict[str, int]
    close_tag_information: dict[str, Any]


@dataclass(frozen=True)
class TerminalParse:
    reasoning_close_found: bool
    terminal_answer_block_text: str | None
    terminal_answer_block_span: dict[str, int] | None
    answer: str | None
    answer_parse_status: str
    raw_confidence_text: str | None
    raw_parsed_confidence: int | None
    normalized_confidence: float | None
    confidence_parse_status: str
    diagnostic_answer_like_text: str | None


def _find_subsequence(values: Sequence[int], pattern: Sequence[int], start: int = 0) -> int | None:
    if not pattern:
        raise SmolLM3AdapterError("token boundary sequence must not be empty")
    final_start = len(values) - len(pattern)
    for index in range(start, final_start + 1):
        if list(values[index : index + len(pattern)]) == list(pattern):
            return index
    return None


def locate_reasoning_tokens(
    *,
    generated_token_ids: Sequence[int],
    prompt_token_ids: Sequence[int],
    open_token_ids: Sequence[int],
    close_token_ids: Sequence[int],
) -> ReasoningLocation:
    """Locate reasoning IDs without re-tokenizing decoded text."""

    generated = tuple(int(token_id) for token_id in generated_token_ids)
    prompt = tuple(int(token_id) for token_id in prompt_token_ids)
    opening = tuple(int(token_id) for token_id in open_token_ids)
    closing = tuple(int(token_id) for token_id in close_token_ids)
    if not opening or not closing:
        raise SmolLM3AdapterError("reasoning tag token sequences must not be empty")

    if len(prompt) >= len(opening) and prompt[-len(opening) :] == opening:
        reasoning_start = 0
    else:
        opening_start = _find_subsequence(generated, opening)
        reasoning_start = opening_start + len(opening) if opening_start is not None else 0

    close_start = _find_subsequence(generated, closing, reasoning_start)
    reasoning_end = len(generated) if close_start is None else close_start
    indices = tuple(range(reasoning_start, reasoning_end))
    reasoning_ids = tuple(generated[index] for index in indices)
    if close_start is None:
        status = "missing_close"
        close_information: dict[str, Any] = {
            "found": False,
            "generated_start": None,
            "generated_end_exclusive": None,
        }
    else:
        status = "closed" if reasoning_ids else "no_reasoning"
        close_information = {
            "found": True,
            "generated_start": close_start,
            "generated_end_exclusive": close_start + len(closing),
        }
    return ReasoningLocation(
        reasoning_status=status,
        reasoning_indices=indices,
        reasoning_token_ids=reasoning_ids,
        reasoning_boundaries={
            "generated_start": reasoning_start,
            "generated_end_exclusive": reasoning_end,
        },
        close_tag_information=close_information,
    )


def _question_text(record: Mapping[str, Any]) -> str:
    question = record.get("question")
    choices = record.get("choices")
    if not isinstance(question, str) or not question.strip():
        raise SmolLM3AdapterError("question text must be nonempty")
    if (
        not isinstance(choices, (list, tuple))
        or len(choices) != 4
        or any(not isinstance(choice, str) for choice in choices)
    ):
        raise SmolLM3AdapterError("question must contain exactly four string choices")
    choice_lines = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip("ABCD", choices, strict=True)
    )
    return (
        f"Question: {question}\n\n{choice_lines}\n\n"
        "Reason carefully, then finish with exactly this two-line block:\n"
        "Answer: <A|B|C|D>\n"
        "Confidence: <integer 0-100>"
    )


def render_question_prompt(tokenizer: Any, record: Mapping[str, Any]) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": _question_text(record)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise SmolLM3AdapterError("SmolLM3 chat template returned an empty prompt")
    return rendered


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(encoded, (list, tuple)) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in encoded
    ):
        raise SmolLM3AdapterError(f"tokenizer returned invalid token IDs for {text!r}")
    return list(encoded)


def preflight_tokenizer_contract(tokenizer: Any) -> dict[str, Any]:
    """Resolve exact token conventions without mutating tokenizer specials."""

    open_ids = _encode_without_special_tokens(tokenizer, REASONING_OPEN_TAG)
    close_ids = _encode_without_special_tokens(tokenizer, REASONING_CLOSE_TAG)
    inducer_ids = _encode_without_special_tokens(tokenizer, FORCED_CLOSE_INDUCER)
    if not open_ids or not close_ids or not inducer_ids:
        raise SmolLM3AdapterError("reasoning and inducer token sequences must not be empty")

    raw_sequences: dict[str, list[int]] = {}
    for letter in "ABCD":
        boundary_ids = _encode_without_special_tokens(
            tokenizer, f"{FORCED_CLOSE_INDUCER} {letter}"
        )
        if boundary_ids[: len(inducer_ids)] != inducer_ids:
            raise SmolLM3AdapterError(
                f"choice {letter} changes inducer tokenization at the exact answer boundary"
            )
        suffix = boundary_ids[len(inducer_ids) :]
        if len(suffix) != 1:
            raise SmolLM3AdapterError(
                f"choice {letter} must be one single token at the exact inducer boundary"
            )
        raw_sequences[letter] = suffix
    token_ids = [raw_sequences[letter][0] for letter in "ABCD"]
    if len(set(token_ids)) != 4:
        raise SmolLM3AdapterError("A-D answer tokens must be four distinct token IDs")

    return {
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "reasoning_open_tag": REASONING_OPEN_TAG,
        "reasoning_open_token_ids": open_ids,
        "reasoning_close_tag": REASONING_CLOSE_TAG,
        "reasoning_close_token_ids": close_ids,
        "inducer_text": FORCED_CLOSE_INDUCER,
        "inducer_token_ids": inducer_ids,
        "ad_token_convention": ANSWER_TOKEN_CONVENTION,
        "ad_raw_token_sequences": raw_sequences,
        "ad_token_ids": token_ids,
    }


def _missing_parse(*, diagnostic: str | None = None) -> TerminalParse:
    return TerminalParse(
        reasoning_close_found=False,
        terminal_answer_block_text=None,
        terminal_answer_block_span=None,
        answer=None,
        answer_parse_status="missing",
        raw_confidence_text=None,
        raw_parsed_confidence=None,
        normalized_confidence=None,
        confidence_parse_status="missing",
        diagnostic_answer_like_text=diagnostic,
    )


def _parse_terminal_region(
    region: str, *, region_offset: int, reasoning_close_found: bool
) -> TerminalParse:
    matches = list(_ANSWER_LINE.finditer(region))
    if not matches:
        return TerminalParse(
            reasoning_close_found=reasoning_close_found,
            terminal_answer_block_text=None,
            terminal_answer_block_span=None,
            answer=None,
            answer_parse_status="missing",
            raw_confidence_text=None,
            raw_parsed_confidence=None,
            normalized_confidence=None,
            confidence_parse_status="missing",
            diagnostic_answer_like_text=None,
        )
    answer_match = matches[-1]
    raw_answer = answer_match.group("answer").strip()
    if not raw_answer:
        answer = None
        answer_status = "missing"
    elif raw_answer in "ABCD" and len(raw_answer) == 1:
        answer = raw_answer
        answer_status = "parsed"
    else:
        answer = None
        answer_status = "out_of_domain"

    confidence_match = _ADJACENT_CONFIDENCE.match(region, answer_match.end())
    if confidence_match is None:
        raw_confidence = None
        parsed_confidence = None
        normalized = None
        confidence_status = "missing"
        block_end = answer_match.end()
    else:
        raw_confidence = confidence_match.group("confidence").strip()
        block_end = confidence_match.end()
        if not raw_confidence or _INTEGER.fullmatch(raw_confidence) is None:
            parsed_confidence = None
            normalized = None
            confidence_status = "malformed"
        else:
            parsed_confidence = int(raw_confidence)
            if 0 <= parsed_confidence <= 100:
                normalized = parsed_confidence / 100
                confidence_status = "parsed"
            else:
                normalized = None
                confidence_status = "out_of_range"

    block_text = region[answer_match.start() : block_end].strip()
    return TerminalParse(
        reasoning_close_found=reasoning_close_found,
        terminal_answer_block_text=block_text,
        terminal_answer_block_span={
            "start": region_offset + answer_match.start(),
            "end_exclusive": region_offset + block_end,
        },
        answer=answer,
        answer_parse_status=answer_status,
        raw_confidence_text=raw_confidence,
        raw_parsed_confidence=parsed_confidence,
        normalized_confidence=normalized,
        confidence_parse_status=confidence_status,
        diagnostic_answer_like_text=None,
    )


def parse_natural_output(decoded_output: str) -> TerminalParse:
    close_start = decoded_output.find(REASONING_CLOSE_TAG)
    if close_start < 0:
        answer_matches = list(_ANSWER_LIKE.finditer(decoded_output))
        diagnostic = None
        if answer_matches:
            raw = answer_matches[-1].group("answer").strip()
            diagnostic = f"Answer: {raw}" if raw else "Answer:"
        return _missing_parse(diagnostic=diagnostic)
    region_start = close_start + len(REASONING_CLOSE_TAG)
    return _parse_terminal_region(
        decoded_output[region_start:],
        region_offset=region_start,
        reasoning_close_found=True,
    )


def parse_forced_output(decoded_forced_output: str) -> TerminalParse:
    reconstructed = f"Answer:{decoded_forced_output}"
    return _parse_terminal_region(
        reconstructed,
        region_offset=0,
        reasoning_close_found=True,
    )


def validate_context_budget(
    *,
    model_context_window: int,
    prompt_token_count: int,
    natural_max_new_tokens: int,
    longest_checkpoint_prefix_tokens: int,
    inducer_token_count: int,
    checkpoint_max_new_tokens: int,
) -> dict[str, int]:
    values = {
        "model_context_window": model_context_window,
        "prompt_token_count": prompt_token_count,
        "natural_max_new_tokens": natural_max_new_tokens,
        "longest_checkpoint_prefix_tokens": longest_checkpoint_prefix_tokens,
        "inducer_token_count": inducer_token_count,
        "checkpoint_max_new_tokens": checkpoint_max_new_tokens,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()):
        raise SmolLM3AdapterError("context budget values must be nonnegative integers")
    natural_required = prompt_token_count + natural_max_new_tokens
    checkpoint_required = (
        longest_checkpoint_prefix_tokens + inducer_token_count + checkpoint_max_new_tokens
    )
    if natural_required > model_context_window:
        raise SmolLM3AdapterError(
            f"natural generation context requires {natural_required}, exceeds {model_context_window}"
        )
    if checkpoint_required > model_context_window:
        raise SmolLM3AdapterError(
            f"checkpoint generation context requires {checkpoint_required}, exceeds {model_context_window}"
        )
    return {
        "model_context_window": model_context_window,
        "natural_required_tokens": natural_required,
        "checkpoint_required_tokens": checkpoint_required,
    }


def require_model_commit_sha(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_SHA.fullmatch(value) is None:
        raise SmolLM3AdapterError(f"{label} must be an immutable lowercase 40-character commit SHA")
    return value


def load_model_and_tokenizer(*, model_revision: str, tokenizer_revision: str):
    """Load one bfloat16 SmolLM3 instance; call only inside a GPU SLURM job."""

    model_revision = require_model_commit_sha(model_revision, label="model_revision")
    tokenizer_revision = require_model_commit_sha(
        tokenizer_revision, label="tokenizer_revision"
    )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SmolLM3AdapterError(
            "SmolLM3 execution requires exactly one visible CUDA GPU inside SLURM"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_REPOSITORY,
        revision=tokenizer_revision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPOSITORY,
        revision=model_revision,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer

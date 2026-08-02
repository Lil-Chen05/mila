#!/usr/bin/env python3
"""GPU-only SmolLM3 preflight and non-production smoke manifest creation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from part1_checkpoints import CHECKPOINT_GENERATION_SETTINGS
from part1_contract import canonical_json_bytes
from part1_generation import NATURAL_GENERATION_SETTINGS
from part1_manifests import load_manifest_bundle
from part1_model_run import build_smoke_model_run_manifest
from part1_smollm3_adapter import (
    ADAPTER_VERSION,
    INDUCER_VERSION,
    MODEL_REPOSITORY,
    PARSER_VERSION,
    PROMPT_VERSION,
    TOKENIZER_REPOSITORY,
    load_model_and_tokenizer,
    preflight_tokenizer_contract,
    render_question_prompt,
    require_model_commit_sha,
    validate_context_budget,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_ROOT = REPOSITORY_ROOT / "manifests" / "part1"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "part1-smoke"


def resolve_context_window(model_config: Any, tokenizer: Any) -> int:
    candidates: list[int] = []
    for value in (
        getattr(model_config, "max_position_embeddings", None),
        getattr(model_config, "n_positions", None),
        getattr(tokenizer, "model_max_length", None),
    ):
        if (
            not isinstance(value, bool)
            and isinstance(value, int)
            and 0 < value < 10**9
        ):
            candidates.append(value)
    if not candidates:
        raise ValueError("could not resolve a finite model/tokenizer context window")
    return min(candidates)


def build_effective_generation_settings(
    token_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    specials = {
        "bos_token_id": token_contract.get("bos_token_id"),
        "eos_token_id": token_contract.get("eos_token_id"),
        "pad_token_id": token_contract.get("pad_token_id"),
        "batch_size": 1,
    }
    return (
        {**NATURAL_GENERATION_SETTINGS, **specials},
        {
            **CHECKPOINT_GENERATION_SETTINGS,
            "return_dict_in_generate": True,
            "output_logits": True,
            **specials,
        },
    )


def _single_prompt_token_ids(encoded: Mapping[str, Any]) -> list[int]:
    raw = encoded.get("input_ids")
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        raise ValueError("preflight tokenizer did not return input_ids")
    if raw and isinstance(raw[0], (list, tuple)):
        if len(raw) != 1:
            raise ValueError("preflight prompt must tokenize to batch size one")
        raw = raw[0]
    token_ids = list(raw)
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        for token_id in token_ids
    ):
        raise ValueError("preflight tokenizer returned invalid prompt token IDs")
    return token_ids


def validate_all_question_prompts(
    records: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    token_contract: Mapping[str, Any],
    model_context_window: int,
) -> dict[str, Any]:
    """Validate every fixed-manifest prompt and the population worst-case budget."""

    if len(records) != 500:
        raise ValueError("SmolLM3 prompt preflight requires exactly 500 questions")
    open_ids = list(token_contract["reasoning_open_token_ids"])
    if not open_ids:
        raise ValueError("reasoning open token IDs must be nonempty")
    maximum_prompt_tokens = -1
    worst_record: Mapping[str, Any] | None = None
    for record in records:
        prompt = render_question_prompt(tokenizer, record)
        token_ids = _single_prompt_token_ids(
            tokenizer(prompt, add_special_tokens=False)
        )
        if token_ids[-len(open_ids) :] != open_ids:
            raise ValueError(
                "prompt at sample_index "
                f"{record.get('sample_index')} does not end at the expected reasoning open tag"
            )
        if len(token_ids) > maximum_prompt_tokens:
            maximum_prompt_tokens = len(token_ids)
            worst_record = record
    assert worst_record is not None
    context_report: dict[str, Any] = validate_context_budget(
        model_context_window=model_context_window,
        prompt_token_count=maximum_prompt_tokens,
        natural_max_new_tokens=NATURAL_GENERATION_SETTINGS["max_new_tokens"],
        longest_checkpoint_prefix_tokens=(
            maximum_prompt_tokens + NATURAL_GENERATION_SETTINGS["max_new_tokens"]
        ),
        inducer_token_count=len(token_contract["inducer_token_ids"]),
        checkpoint_max_new_tokens=CHECKPOINT_GENERATION_SETTINGS["max_new_tokens"],
    )
    context_report.update(
        validated_prompt_count=len(records),
        maximum_prompt_token_count=maximum_prompt_tokens,
        worst_sample_identity={
            "question_id": worst_record["question_id"],
            "sample_index": worst_record["sample_index"],
            "subject": worst_record["subject"],
        },
    )
    return context_report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_revision(repository: str, requested_revision: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id=repository, revision=requested_revision)
    return require_model_commit_sha(getattr(info, "sha", None), label=repository)


def _git_provenance(repository_root: Path = REPOSITORY_ROOT) -> tuple[str, str]:
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    require_model_commit_sha(base, label="base Git commit")
    diff = subprocess.run(
        [
            "git",
            "diff",
            "--binary",
            "HEAD",
            "--",
            "scripts",
            "jobs",
            "configs",
            "schemas",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=repository_root,
        capture_output=True,
        check=True,
    ).stdout
    untracked_output = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "scripts",
            "jobs",
            "configs",
            "schemas",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=repository_root,
        capture_output=True,
        check=True,
    ).stdout
    untracked_paths = sorted(path for path in untracked_output.split(b"\0") if path)
    digest = hashlib.sha256()
    digest.update(b"part1-scoped-dirty-git-v2\0")
    digest.update(len(diff).to_bytes(8, "big"))
    digest.update(diff)
    for relative_bytes in untracked_paths:
        relative = Path(os.fsdecode(relative_bytes))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Git reported an unsafe untracked scoped path")
        source = repository_root / relative
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(
                f"untracked scoped provenance target is not a regular file: {relative}"
            )
        content = source.read_bytes()
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return base, digest.hexdigest()


def _compatible_json_write(path: Path, value: Mapping[str, Any]) -> str:
    content = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise RuntimeError(f"existing immutable JSON output is incompatible: {path}")
        return "identical_existing"
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return "published"


def _tokenizer_hashes(tokenizer: Any, temporary_root: Path) -> dict[str, str]:
    tokenizer_root = temporary_root / "tokenizer"
    tokenizer.save_pretrained(tokenizer_root)
    return {
        str(path.relative_to(tokenizer_root)): _sha256_file(path)
        for path in sorted(tokenizer_root.rglob("*"))
        if path.is_file()
    }


def run_preflight(
    *,
    manifest_root: Path,
    output_root: Path,
    requested_model_revision: str,
    requested_tokenizer_revision: str,
) -> dict[str, Any]:
    bundle = load_manifest_bundle(
        questions_path=manifest_root / "questions.jsonl",
        question_manifest_path=manifest_root / "questions.manifest.json",
        study_manifest_path=manifest_root / "study_manifest.json",
    )
    model_revision = _resolve_revision(MODEL_REPOSITORY, requested_model_revision)
    tokenizer_revision = _resolve_revision(
        TOKENIZER_REPOSITORY, requested_tokenizer_revision
    )
    model, tokenizer = load_model_and_tokenizer(
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
    )

    import torch

    token_contract = preflight_tokenizer_contract(tokenizer)
    context_window = resolve_context_window(model.config, tokenizer)
    context_report = validate_all_question_prompts(
        bundle.records,
        tokenizer=tokenizer,
        token_contract=token_contract,
        model_context_window=context_window,
    )
    first_question = bundle.records[0]
    prompt = render_question_prompt(tokenizer, first_question)
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("preflight prompt must tokenize to batch size one")
    prompt_token_ids = input_ids[0].tolist()
    open_ids = token_contract["reasoning_open_token_ids"]
    if prompt_token_ids[-len(open_ids) :] != open_ids:
        raise ValueError("thinking chat template does not end at the expected reasoning open tag")
    device_inputs = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.inference_mode():
        forward = model(**device_inputs)
    if forward.logits.ndim != 3 or forward.logits.shape[0] != 1:
        raise ValueError("one-model GPU forward preflight returned an invalid logits shape")
    if not bool(torch.isfinite(forward.logits[:, -1, :].float()).all().item()):
        raise ValueError("GPU forward preflight produced non-finite answer-step logits")

    effective_natural, effective_checkpoint = build_effective_generation_settings(
        token_contract
    )
    lock_hash = _sha256_file(REPOSITORY_ROOT / "uv.lock")
    temporary_parent = Path(os.environ.get("SLURM_TMPDIR", output_root))
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="part1-preflight-", dir=temporary_parent) as temp:
        tokenizer_hashes = _tokenizer_hashes(tokenizer, Path(temp))
    environment = {
        "python": platform.python_version(),
        "transformers": importlib.metadata.version("transformers"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_model": torch.cuda.get_device_name(0),
        "visible_gpu_count": torch.cuda.device_count(),
        "dtype": "bfloat16",
        "batch_size": 1,
        "uv_lock_sha256": lock_hash,
        "model_config_sha256": hashlib.sha256(
            canonical_json_bytes(model.config.to_dict())
        ).hexdigest(),
        "tokenizer_files_sha256": tokenizer_hashes,
        "model_context_window": context_window,
    }
    report: dict[str, Any] = {
        "schema_name": "part1_smollm3_preflight",
        "schema_version": "1.0.0",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": model_revision,
        "tokenizer_repository": TOKENIZER_REPOSITORY,
        "tokenizer_revision": tokenizer_revision,
        "question_manifest_hash": bundle.question_manifest["question_manifest_hash"],
        "study_id": bundle.study_manifest["study_id"],
        "study_manifest_hash": bundle.study_manifest["study_manifest_hash"],
        "token_contract": token_contract,
        "effective_natural_generation": effective_natural,
        "effective_checkpoint_generation": effective_checkpoint,
        "context_validation": context_report,
        "forward_validation": {
            "batch_size": int(forward.logits.shape[0]),
            "sequence_length": int(forward.logits.shape[1]),
            "vocabulary_size": int(forward.logits.shape[2]),
            "finite_last_step_logits": True,
            "model_training": bool(model.training),
        },
        "environment_versions": environment,
        "component_versions": {
            "adapter": ADAPTER_VERSION,
            "prompt": PROMPT_VERSION,
            "parser": PARSER_VERSION,
            "inducer": INDUCER_VERSION,
        },
    }
    if report["forward_validation"]["model_training"]:
        raise ValueError("model.eval() was not effective during preflight")

    preflight_path = output_root / "preflight" / "preflight.json"
    preflight_publication = _compatible_json_write(preflight_path, report)
    base_commit, diff_hash = _git_provenance()
    smoke_paths: dict[str, str] = {}
    smoke_publication: dict[str, str] = {}
    for scope in ("smoke_a", "smoke_b", "reproducibility"):
        manifest = build_smoke_model_run_manifest(
            study_manifest=bundle.study_manifest,
            preflight_report=report,
            execution_scope=scope,
            base_git_commit=base_commit,
            diff_hash=diff_hash,
        )
        path = output_root / "model-runs" / scope / "model_run_manifest.json"
        smoke_publication[scope] = _compatible_json_write(path, manifest)
        smoke_paths[scope] = str(path)

    return {
        "status": "passed",
        "preflight_path": str(preflight_path),
        "preflight_publication": preflight_publication,
        "smoke_model_run_manifest_paths": smoke_paths,
        "smoke_model_run_manifest_publication": smoke_publication,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--tokenizer-revision", default="main")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run_preflight(
            manifest_root=args.manifest_root,
            output_root=args.output_root,
            requested_model_revision=args.model_revision,
            requested_tokenizer_revision=args.tokenizer_revision,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

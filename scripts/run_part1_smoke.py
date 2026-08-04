#!/usr/bin/env python3
"""Run one bounded non-production Part 1 smoke plan inside a GPU SLURM job."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

from part1_checkpoints import (
    CHECKPOINT_GENERATION_SETTINGS,
    CheckpointGenerationCapture,
    build_alias_checkpoint_terminal_result,
    build_checkpoint_infrastructure_failure_result,
    build_checkpoint_probe_plans,
    build_checkpoint_terminal_result,
)
from part1_contract import (
    FIXED_SUBJECTS,
    audit_event_id,
    attempt_id,
    derive_generation_seed,
    validate_instance,
)
from part1_generation import (
    NATURAL_GENERATION_SETTINGS,
    NaturalGenerationCapture,
    build_natural_infrastructure_failure_result,
    build_natural_terminal_result,
)
from part1_failure_policy import classify_failure
from part1_manifests import load_manifest_bundle
from part1_model_run import validate_preflight_model_run_compatibility
from part1_runtime import (
    FreshProcessRequired,
    LockMetadata,
    LockedShardSession,
    WorkSpec,
    prepare_resume,
)
from part1_smollm3_adapter import (
    load_model_and_tokenizer,
    preflight_tokenizer_contract,
    render_question_prompt,
)
from part1_storage_estimate import assess_free_space, estimate_part1_storage


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_ROOT = REPOSITORY_ROOT / "manifests" / "part1"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "part1-smoke"


def select_smoke_work(
    records: Sequence[Mapping[str, Any]], *, execution_scope: str
) -> list[tuple[Mapping[str, Any], int]]:
    if len(records) != 500:
        raise ValueError("smoke selection requires the validated 500-question manifest")
    for sample_index, record in enumerate(records):
        expected_subject = FIXED_SUBJECTS[sample_index // 100]
        if record.get("sample_index") != sample_index:
            raise ValueError("question manifest sample_index order is invalid")
        if record.get("subject") != expected_subject:
            raise ValueError("question manifest subject block order is invalid")
    if execution_scope == "smoke_a":
        return [(records[0], run_id) for run_id in range(10)]
    if execution_scope == "smoke_b":
        return [(records[index * 100], 0) for index in range(5)]
    raise ValueError(f"unsupported smoke execution scope: {execution_scope}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _execution_context() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def _attempt_event(
    *,
    work: WorkSpec,
    shard_id: str,
    attempt_number: int,
    event_type: str,
    terminal_record_id: str | None,
    event_sequence: int | None = None,
    outcome_category: str | None = None,
    error_details: Mapping[str, Any] | None = None,
    retry_classification: str | None = None,
    retry_decision: str | None = None,
    backoff_seconds: int | None = None,
) -> dict[str, Any]:
    attempt_value = attempt_id(
        work.study_id,
        work.model_run_id,
        work.question_id,
        work.run_id,
        attempt_number,
        checkpoint_id=work.checkpoint_id,
    )
    sequence = (
        0 if event_type == "attempt_started" else 1
    ) if event_sequence is None else event_sequence
    event = {
        "schema_name": "part1_audit_event",
        "schema_version": "1.0.0",
        "event_id": audit_event_id(attempt_value, event_type, sequence),
        "event_scope": "attempt",
        "study_id": work.study_id,
        "model_run_id": work.model_run_id,
        "shard_id": shard_id,
        "question_id": work.question_id,
        "run_id": work.run_id,
        "checkpoint_id": work.checkpoint_id,
        "attempt_id": attempt_value,
        "attempt_number": attempt_number,
        "event_sequence": sequence,
        "event_type": event_type,
        "event_timestamp": _now(),
        "execution_context": _execution_context(),
        "outcome_category": outcome_category,
        "error_details": dict(error_details) if error_details is not None else None,
        "retry_classification": retry_classification,
        "retry_decision": retry_decision,
        "backoff_seconds": backoff_seconds,
        "related_lock_owner": None,
        "terminal_record_id": terminal_record_id,
        "operator_reason": None,
    }
    validate_instance("audit_event", event)
    return event


def _generation_kwargs(settings: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "max_new_tokens",
        "return_dict_in_generate",
        "output_logits",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    }
    return {key: value for key, value in settings.items() if key in allowed}


def _entropy_trace_from_gpu_logits(logits: Sequence[Any], generated_count: int) -> tuple[float, ...]:
    import torch

    if len(logits) != generated_count:
        raise ValueError("generate output logits and tokens are not aligned")
    values: list[float] = []
    for step in logits:
        raw = step[0].detach().to(dtype=torch.float32)
        log_probabilities = torch.log_softmax(raw, dim=-1)
        entropy = -(log_probabilities.exp() * log_probabilities).sum()
        value = float(entropy.item())
        if not torch.isfinite(entropy).item() or value < 0:
            raise ValueError("raw pre-warper entropy is non-finite")
        values.append(value)
    return tuple(values)


def _stop_reason(generated_ids: Sequence[int], tokenizer: Any, max_new_tokens: int) -> str:
    if len(generated_ids) >= max_new_tokens:
        return "max_new_tokens"
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_values = set(eos if isinstance(eos, (list, tuple)) else [eos])
    return "eos" if generated_ids and generated_ids[-1] in eos_values else "other"


def _execute_natural(
    *,
    model: Any,
    tokenizer: Any,
    question: Mapping[str, Any],
    run_id: int,
    seed: int,
    attempt_number: int,
    model_manifest: Mapping[str, Any],
    token_contract: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    prompt = render_question_prompt(tokenizer, question)
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    prompt_ids = encoded["input_ids"][0].tolist()
    generation_device = torch.device(model.device)
    rng_devices = [generation_device] if generation_device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                **_generation_kwargs(model_manifest["effective_natural_generation"]),
            )
    if not hasattr(output, "logits") or output.logits is None:
        raise ValueError("generate output did not expose raw pre-warper logits")
    generated = output.sequences[0, len(prompt_ids) :].tolist()
    entropies = _entropy_trace_from_gpu_logits(output.logits, len(generated))
    decoded = tokenizer.decode(
        generated,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    capture = NaturalGenerationCapture(
        rendered_prompt=prompt,
        prompt_token_ids=tuple(prompt_ids),
        generated_token_ids=tuple(generated),
        decoded_output=decoded,
        raw_prewarper_logits=(),
        stop_reason=_stop_reason(
            generated, tokenizer, NATURAL_GENERATION_SETTINGS["max_new_tokens"]
        ),
        precomputed_entropy_nats=entropies,
    )
    identity = {
        "study_id": model_manifest["study_id"],
        "model_run_id": model_manifest["model_run_id"],
        "model_run_manifest_hash": model_manifest["model_run_manifest_hash"],
        "question_manifest_hash": model_manifest["question_manifest_hash"],
        "question_id": question["question_id"],
        "sample_index": question["sample_index"],
        "subject": question["subject"],
        "gold_letter": question["gold_letter"],
    }
    return build_natural_terminal_result(
        identity=identity,
        run_id=run_id,
        generation_seed=seed,
        terminal_attempt_number=attempt_number,
        capture=capture,
        token_contract=token_contract,
        decode_reasoning=lambda ids: tokenizer.decode(
            list(ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    )


def _execute_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    parent: Mapping[str, Any],
    plan: Any,
    token_contract: Mapping[str, Any],
    gold_letter: str,
    attempt_number: int,
) -> dict[str, Any]:
    import torch

    input_ids = torch.tensor(
        [plan.model_input_token_ids], dtype=torch.long, device=model.device
    )
    attention_mask = torch.ones_like(input_ids)
    settings = {
        **CHECKPOINT_GENERATION_SETTINGS,
        "return_dict_in_generate": True,
        "output_logits": True,
        "bos_token_id": token_contract.get("bos_token_id"),
        "eos_token_id": token_contract.get("eos_token_id"),
        "pad_token_id": token_contract.get("pad_token_id"),
    }
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **_generation_kwargs(settings),
        )
    generated = output.sequences[0, input_ids.shape[1] :].tolist()
    if not hasattr(output, "logits") or output.logits is None:
        raise ValueError("checkpoint generate output did not expose raw logits")
    if len(output.logits) != len(generated):
        raise ValueError("checkpoint output logits and generated tokens are not aligned")
    answer_step_logits = (
        tuple(output.logits[0][0].detach().to(dtype=torch.float32).cpu().tolist())
        if generated
        else None
    )
    capture = CheckpointGenerationCapture(
        forced_generated_token_ids=tuple(generated),
        decoded_forced_output=tokenizer.decode(
            generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        raw_prewarper_logits=(),
        answer_step_raw_logits=answer_step_logits,
    )
    return build_checkpoint_terminal_result(
        parent=parent,
        plan=plan,
        capture=capture,
        token_contract=token_contract,
        gold_letter=gold_letter,
        terminal_attempt_number=attempt_number,
    )


def _completion_event_for_result(
    session: LockedShardSession,
    *,
    work: WorkSpec,
    attempt_number: int,
    result: Mapping[str, Any],
    completion_event_sequence: int = 1,
) -> dict[str, Any]:
    expected_attempt = attempt_id(
        work.study_id,
        work.model_run_id,
        work.question_id,
        work.run_id,
        attempt_number,
        checkpoint_id=work.checkpoint_id,
    )
    if expected_attempt != result["terminal_attempt_id"]:
        raise ValueError("prepared result attempt identity differs from durable start")
    outcome = result.get(
        "natural_execution_outcome", result.get("checkpoint_execution_outcome")
    )
    category: str | None = None
    details: Mapping[str, Any] | None = None
    retry_classification: str | None = None
    retry_decision: str | None = None
    backoff_seconds: int | None = None
    if outcome == "terminal_infrastructure_failure":
        raw_details = result.get("terminal_error_details")
        if not isinstance(raw_details, Mapping):
            raise ValueError("terminal infrastructure result is missing error details")
        category_value = raw_details.get("category")
        if not isinstance(category_value, str):
            raise ValueError("terminal infrastructure result is missing a failure category")
        category = category_value
        details = raw_details
        policy = classify_failure(category, attempt_number)
        retry_classification = policy.classification
        retry_decision = policy.retry_decision
        backoff_seconds = policy.backoff_seconds
    return _attempt_event(
        work=work,
        shard_id=session.owner.shard_id,
        attempt_number=attempt_number,
        event_type="attempt_completed",
        terminal_record_id=work.terminal_record_id,
        event_sequence=completion_event_sequence,
        outcome_category=category,
        error_details=details,
        retry_classification=retry_classification,
        retry_decision=retry_decision,
        backoff_seconds=backoff_seconds,
    )


def _preflight_result_commit(
    session: LockedShardSession,
    *,
    work: WorkSpec,
    attempt_number: int,
    result: Mapping[str, Any],
    completion_event_sequence: int = 1,
) -> None:
    completion = _completion_event_for_result(
        session,
        work=work,
        attempt_number=attempt_number,
        result=result,
        completion_event_sequence=completion_event_sequence,
    )
    session.store.preflight_terminal_commit(result, completion)


def _commit_result(
    session: LockedShardSession,
    *,
    work: WorkSpec,
    attempt_number: int,
    result: Mapping[str, Any],
    completion_event_sequence: int = 1,
) -> None:
    completion = _completion_event_for_result(
        session,
        work=work,
        attempt_number=attempt_number,
        result=result,
        completion_event_sequence=completion_event_sequence,
    )
    session.store.commit_terminal_result(result, completion)


def _start_attempt(
    session: LockedShardSession, *, work: WorkSpec, attempt_number: int
) -> None:
    session.store.append_audit_event(
        _attempt_event(
            work=work,
            shard_id=session.owner.shard_id,
            attempt_number=attempt_number,
            event_type="attempt_started",
            terminal_record_id=None,
        )
    )


def _categorize_execution_exception(exc: Exception) -> str:
    explicit = getattr(exc, "part1_failure_category", None)
    if explicit is not None:
        classify_failure(str(explicit), 1)
        return str(explicit)
    type_name = f"{type(exc).__module__}.{type(exc).__qualname__}".lower()
    message = str(exc).lower()
    if "cuda" in type_name or "cuda" in message:
        return "transient_cuda_runtime_failure"
    if isinstance(exc, OSError):
        return "temporary_filesystem_failure"
    if isinstance(exc, (AssertionError, KeyError, TypeError, ValueError)):
        return "unsupported_model_or_tokenizer_behaviour"
    return "transient_worker_failure"


def _execution_error_details(exc: Exception, category: str) -> dict[str, str]:
    return {
        "category": category,
        "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "message": str(exc) or repr(exc),
    }


def _terminal_failure_reference(
    work: WorkSpec, attempt_number: int, event_sequence: int
) -> str:
    attempt_value = attempt_id(
        work.study_id,
        work.model_run_id,
        work.question_id,
        work.run_id,
        attempt_number,
        checkpoint_id=work.checkpoint_id,
    )
    return audit_event_id(attempt_value, "attempt_completed", event_sequence)


def _next_attempt_event_sequence(
    session: LockedShardSession, work: WorkSpec, attempt_number: int
) -> int:
    attempt_value = attempt_id(
        work.study_id,
        work.model_run_id,
        work.question_id,
        work.run_id,
        attempt_number,
        checkpoint_id=work.checkpoint_id,
    )
    events = session.store.build_index().events_by_attempt.get(attempt_value, ())
    return max((int(event["event_sequence"]) for event in events), default=-1) + 1


def _run_work_lifecycle(
    session: LockedShardSession,
    *,
    work: WorkSpec,
    execute_attempt: Callable[[int], Mapping[str, Any]],
    build_terminal_failure: Callable[
        [int, str, str, Mapping[str, Any]], Mapping[str, Any]
    ],
    sleep: Callable[[float], Any] = time.sleep,
) -> tuple[str, bool]:
    """Resume one logical work item through policy-authorized terminal state."""

    while True:
        decision = prepare_resume(
            session.store,
            [work],
            event_timestamp=_now(),
            execution_context=_execution_context(),
        )[work]
        if decision.status == "completed":
            return "completed", False
        if decision.status == "terminal":
            return "terminal", False
        if decision.status == "terminalization_required":
            attempt_number = decision.attempts_consumed
            category = decision.failure_category or "interrupted_process"
            completion_sequence = _next_attempt_event_sequence(
                session, work, attempt_number
            )
            details = {
                "category": category,
                "exception_type": "interrupted_process",
                "message": "resume requires terminalization of an exhausted interrupted attempt",
            }
            terminal_result = build_terminal_failure(
                attempt_number,
                category,
                _terminal_failure_reference(
                    work, attempt_number, completion_sequence
                ),
                details,
            )
            _commit_result(
                session,
                work=work,
                attempt_number=attempt_number,
                result=terminal_result,
                completion_event_sequence=completion_sequence,
            )
            return "terminal", True
        if decision.status != "retryable" or decision.next_attempt_number is None:
            raise RuntimeError(f"work is not executable: {decision}")

        attempt_number = decision.next_attempt_number
        _start_attempt(session, work=work, attempt_number=attempt_number)
        try:
            result = execute_attempt(attempt_number)
            _preflight_result_commit(
                session,
                work=work,
                attempt_number=attempt_number,
                result=result,
            )
        except Exception as exc:
            category = _categorize_execution_exception(exc)
            policy = classify_failure(category, attempt_number)
            details = _execution_error_details(exc, category)
            if policy.retry_decision == "retry":
                failure_event = _attempt_event(
                    work=work,
                    shard_id=session.owner.shard_id,
                    attempt_number=attempt_number,
                    event_type="attempt_failed",
                    terminal_record_id=None,
                    outcome_category=category,
                    error_details=details,
                    retry_classification=policy.classification,
                    retry_decision=policy.retry_decision,
                    backoff_seconds=policy.backoff_seconds,
                )
                session.store.append_audit_event(failure_event)
                if category == "transient_cuda_runtime_failure":
                    raise FreshProcessRequired(
                        "transient CUDA retry requires worker termination and a fresh CUDA process"
                    ) from exc
                assert policy.backoff_seconds is not None
                sleep(policy.backoff_seconds)
                continue
            terminal_result = build_terminal_failure(
                attempt_number,
                category,
                _terminal_failure_reference(work, attempt_number, 1),
                details,
            )
            _commit_result(
                session,
                work=work,
                attempt_number=attempt_number,
                result=terminal_result,
            )
            return "terminal", True

        _commit_result(
            session,
            work=work,
            attempt_number=attempt_number,
            result=result,
        )
        return "completed", True


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def run_smoke(
    *,
    execution_scope: str,
    manifest_root: Path,
    preflight_path: Path,
    model_run_manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    bundle = load_manifest_bundle(
        questions_path=manifest_root / "questions.jsonl",
        question_manifest_path=manifest_root / "questions.manifest.json",
        study_manifest_path=manifest_root / "study_manifest.json",
    )
    preflight = _load_json(preflight_path)
    model_manifest = _load_json(model_run_manifest_path)
    if model_manifest.get("execution_scope") != execution_scope:
        raise ValueError("model-run manifest execution scope differs from requested smoke")
    if model_manifest.get("production") is not False:
        raise ValueError("bounded smoke runner refuses production model-run manifests")
    validate_preflight_model_run_compatibility(
        preflight_report=preflight,
        model_manifest=model_manifest,
        study_manifest=bundle.study_manifest,
        question_manifest=bundle.question_manifest,
    )
    token_contract = preflight["token_contract"]
    work_selection = select_smoke_work(bundle.records, execution_scope=execution_scope)
    output_root.mkdir(parents=True, exist_ok=True)
    estimate = estimate_part1_storage(
        question_count=len(work_selection),
        natural_runs_per_question=1,
        checkpoints_per_natural=11,
    )
    free_space = assess_free_space(
        estimate,
        free_bytes=shutil.disk_usage(output_root).free,
    )
    if free_space["status"] == "insufficient":
        raise OSError(free_space["warning"])

    model, tokenizer = load_model_and_tokenizer(
        model_revision=model_manifest["model_revision"],
        tokenizer_revision=model_manifest["tokenizer_revision"],
    )
    if preflight_tokenizer_contract(tokenizer) != token_contract:
        raise ValueError("runtime tokenizer contract differs from GPU preflight")
    shard_id = "shard-000"
    shard_root = output_root / execution_scope / model_manifest["model_run_id"] / shard_id
    owner = LockMetadata(
        lock_id=uuid.uuid4().hex,
        study_id=model_manifest["study_id"],
        model_run_id=model_manifest["model_run_id"],
        shard_id=shard_id,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        acquired_at=_now(),
    )
    completed_natural = 0
    completed_checkpoints = 0
    with LockedShardSession.acquire(
        shard_root,
        owner=owner,
        model_run_manifest_hash=model_manifest["model_run_manifest_hash"],
    ) as session:
        for question, run_id in work_selection:
            seed = derive_generation_seed(
                base_seed=model_manifest["base_generation_seed"],
                canonical_model_identity=model_manifest["canonical_model_identity"],
                question_id=question["question_id"],
                run_id=run_id,
            )
            natural_work = WorkSpec.natural(
                model_manifest["study_id"],
                model_manifest["model_run_id"],
                model_manifest["model_run_manifest_hash"],
                question["question_id"],
                run_id,
                seed=seed,
            )
            natural_identity = {
                "study_id": model_manifest["study_id"],
                "model_run_id": model_manifest["model_run_id"],
                "model_run_manifest_hash": model_manifest[
                    "model_run_manifest_hash"
                ],
                "question_manifest_hash": model_manifest["question_manifest_hash"],
                "question_id": question["question_id"],
                "sample_index": question["sample_index"],
                "subject": question["subject"],
                "gold_letter": question["gold_letter"],
            }

            def execute_natural_attempt(attempt_number: int) -> Mapping[str, Any]:
                return _execute_natural(
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    run_id=run_id,
                    seed=seed,
                    attempt_number=attempt_number,
                    model_manifest=model_manifest,
                    token_contract=token_contract,
                )

            def build_natural_failure(
                attempt_number: int,
                category: str,
                reference: str,
                details: Mapping[str, Any],
            ) -> Mapping[str, Any]:
                return build_natural_infrastructure_failure_result(
                    identity=natural_identity,
                    run_id=run_id,
                    generation_seed=seed,
                    terminal_attempt_number=attempt_number,
                    prompt_hash=model_manifest["prompt_hash"],
                    failure_category=category,
                    infrastructure_failure_reference=reference,
                    error_details=details,
                )

            natural_status, natural_published = _run_work_lifecycle(
                session,
                work=natural_work,
                execute_attempt=execute_natural_attempt,
                build_terminal_failure=build_natural_failure,
            )
            if natural_published:
                completed_natural += 1
            if natural_status != "completed":
                continue

            index = session.store.build_index()
            parent_key = (
                model_manifest["study_id"],
                model_manifest["model_run_id"],
                question["question_id"],
                run_id,
            )
            parent = index.natural_terminal_by_key[parent_key]
            plans = build_checkpoint_probe_plans(
                parent,
                inducer_token_ids=token_contract["inducer_token_ids"],
                inducer_version=model_manifest["inducer_version"],
            )
            for plan in plans:
                checkpoint_work = WorkSpec.checkpoint(
                    model_manifest["study_id"],
                    model_manifest["model_run_id"],
                    model_manifest["model_run_manifest_hash"],
                    question["question_id"],
                    run_id,
                    plan.checkpoint_id,
                    seed=seed,
                )

                def execute_checkpoint_attempt(
                    attempt_number: int,
                ) -> Mapping[str, Any]:
                    if plan.is_alias:
                        owner_id = plan.alias_metadata["owner_checkpoint_id"]
                        owner_key = (*parent_key, owner_id)
                        owner_record = session.store.build_index().checkpoint_terminal_by_key.get(
                            owner_key
                        )
                        if owner_record is None:
                            raise RuntimeError("alias owner checkpoint is not durable")
                        return build_alias_checkpoint_terminal_result(
                            parent=parent,
                            owner_record=owner_record,
                            alias_plan=plan,
                            terminal_attempt_number=attempt_number,
                        )
                    return _execute_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        parent=parent,
                        plan=plan,
                        token_contract=token_contract,
                        gold_letter=question["gold_letter"],
                        attempt_number=attempt_number,
                    )

                def build_checkpoint_failure(
                    attempt_number: int,
                    category: str,
                    reference: str,
                    details: Mapping[str, Any],
                ) -> Mapping[str, Any]:
                    return build_checkpoint_infrastructure_failure_result(
                        parent=parent,
                        plan=plan,
                        terminal_attempt_number=attempt_number,
                        failure_category=category,
                        infrastructure_failure_reference=reference,
                        error_details=details,
                        inducer_text=token_contract.get(
                            "inducer_text", "</think>\nAnswer:"
                        ),
                    )

                _checkpoint_status, checkpoint_published = _run_work_lifecycle(
                    session,
                    work=checkpoint_work,
                    execute_attempt=execute_checkpoint_attempt,
                    build_terminal_failure=build_checkpoint_failure,
                )
                if checkpoint_published:
                    completed_checkpoints += 1

    return {
        "status": "completed",
        "execution_scope": execution_scope,
        "model_run_id": model_manifest["model_run_id"],
        "shard_root": str(shard_root),
        "new_natural_results": completed_natural,
        "new_checkpoint_results": completed_checkpoints,
        "storage_estimate": estimate,
        "free_space_assessment": free_space,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-scope", choices=("smoke_a", "smoke_b"), required=True)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument(
        "--preflight",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "preflight" / "preflight.json",
    )
    parser.add_argument("--model-run-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run_smoke(
            execution_scope=args.execution_scope,
            manifest_root=args.manifest_root,
            preflight_path=args.preflight,
            model_run_manifest_path=args.model_run_manifest,
            output_root=args.output_root,
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

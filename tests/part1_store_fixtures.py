"""Synthetic Part 1 storage fixtures; never load model or dataset code."""

from __future__ import annotations

from copy import deepcopy

from part1_contract import (
    attempt_id,
    audit_event_id,
    checkpoint_record_id,
    natural_record_id,
)
from part1_failure_policy import classify_failure


STUDY_ID = "b" * 64
MODEL_RUN_ID = "c" * 64
MODEL_RUN_MANIFEST_HASH = "d" * 64
QUESTION_MANIFEST_HASH = "e" * 64
QUESTION_ID = "f" * 64
SHARD_ID = "shard-000"


def natural_result(*, attempt_number: int = 1, outcome: str = "complete") -> dict:
    complete = outcome == "complete"
    terminal_attempt_id = attempt_id(
        STUDY_ID,
        MODEL_RUN_ID,
        QUESTION_ID,
        0,
        attempt_number,
    )
    return {
        "schema_name": "part1_natural_terminal_result",
        "schema_version": "1.0.0",
        "raw_record_id": natural_record_id(STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0),
        "study_id": STUDY_ID,
        "model_run_id": MODEL_RUN_ID,
        "model_run_manifest_hash": MODEL_RUN_MANIFEST_HASH,
        "question_manifest_hash": QUESTION_MANIFEST_HASH,
        "question_id": QUESTION_ID,
        "sample_index": 0,
        "subject": "high_school_mathematics",
        "run_id": 0,
        "generation_seed": 123,
        "seed_algorithm_version": "part1-seed-v1",
        "terminal_attempt_number": attempt_number,
        "terminal_attempt_id": terminal_attempt_id,
        "infrastructure_failure_reference": None if complete else "audit-event:failure",
        "prompt_hash": "1" * 64,
        "rendered_prompt": "Prompt" if complete else None,
        "prompt_token_ids": [1, 2] if complete else None,
        "generated_token_ids": [10, 11] if complete else None,
        "decoded_output": "<think>x</think> Answer: C" if complete else None,
        "reasoning_text": "x" if complete else None,
        "reasoning_boundaries": {"start": 1, "end": 2} if complete else None,
        "close_tag_information": {"found": True, "token_start": 2, "token_end": 3}
        if complete
        else None,
        "stop_reason": "eos" if complete else "error",
        "generated_token_count": 2 if complete else None,
        "reasoning_token_count": 1 if complete else None,
        "per_token_entropy_nats": [1.1234567890123457, 0.9876543210987654]
        if complete
        else None,
        "mean_reasoning_entropy_nats": 1.1234567890123457 if complete else None,
        "tail_reasoning_entropy_nats": 1.1234567890123457 if complete else None,
        "terminal_answer_block_text": "Answer: C" if complete else None,
        "terminal_answer_block_span": {"start": 20, "end": 29} if complete else None,
        "natural_answer": "C" if complete else None,
        "raw_confidence_text": "80" if complete else None,
        "raw_parsed_confidence": 80 if complete else None,
        "normalized_confidence": 0.8 if complete else None,
        "natural_correct": True if complete else None,
        "diagnostic_answer_like_text": None,
        "checkpoint_eligible": complete,
        "checkpoint_ids": [f"cp-{index:02d}" for index in range(11)] if complete else None,
        "natural_execution_outcome": outcome,
        "reasoning_status": "closed" if complete else "malformed",
        "answer_parse_status": "parsed" if complete else "missing",
        "confidence_parse_status": "parsed" if complete else "missing",
        "component_versions": {"adapter": "smollm3-v1", "parser": "v1"},
        "terminal_error_details": None
        if complete
        else {"category": "temporary_filesystem_failure", "message": "disk busy"},
    }


def checkpoint_result(*, attempt_number: int = 1, checkpoint_id: str = "cp-05") -> dict:
    raw_id = natural_record_id(STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0)
    terminal_attempt_id = attempt_id(
        STUDY_ID,
        MODEL_RUN_ID,
        QUESTION_ID,
        0,
        attempt_number,
        checkpoint_id=checkpoint_id,
    )
    return {
        "schema_name": "part1_checkpoint_terminal_result",
        "schema_version": "1.0.0",
        "checkpoint_record_id": checkpoint_record_id(
            STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0, checkpoint_id
        ),
        "parent_raw_record_id": raw_id,
        "study_id": STUDY_ID,
        "model_run_id": MODEL_RUN_ID,
        "model_run_manifest_hash": MODEL_RUN_MANIFEST_HASH,
        "question_manifest_hash": QUESTION_MANIFEST_HASH,
        "question_id": QUESTION_ID,
        "sample_index": 0,
        "subject": "high_school_mathematics",
        "run_id": 0,
        "checkpoint_id": checkpoint_id,
        "natural_seed": 123,
        "terminal_attempt_number": attempt_number,
        "terminal_attempt_id": terminal_attempt_id,
        "infrastructure_failure_reference": None,
        "requested_checkpoint_index": 5,
        "requested_fraction": 0.5,
        "k_keep": 1,
        "actual_fraction": 0.5,
        "shared_probe_id": "2" * 64,
        "is_alias": False,
        "alias_metadata": {"owner_checkpoint_id": checkpoint_id, "members": [checkpoint_id]},
        "prefix_hash": "3" * 64,
        "inducer_version": "smollm3-forced-close-v1",
        "inducer_text": "</think>\nAnswer:",
        "forced_generated_token_ids": [67, 21],
        "decoded_forced_output": " C Confidence: 80",
        "terminal_answer_block_text": "Answer: C\nConfidence: 80",
        "forced_answer": "C",
        "raw_confidence_text": "80",
        "raw_parsed_confidence": 80,
        "normalized_confidence": 0.8,
        "checkpoint_local_correct": True,
        "answer_token_index": 0,
        "answer_token_id": 67,
        "token_convention": "leading_space_uppercase_single_token",
        "ad_token_ids": [65, 66, 67, 68],
        "ad_logits_float32": [0.10000000149011612, 0.20000000298023224, 0.699999988079071, 0.0],
        "ad_probabilities_float32": [0.20000000298023224, 0.20000000298023224, 0.5, 0.10000000149011612],
        "answer_entropy_nats": 1.2206072645530175,
        "full_vocabulary_answer_step_entropy_nats": 4.123456789012345,
        "maximum_ad_probability": 0.5,
        "agrees_with_natural_answer": True,
        "checkpoint_execution_outcome": "complete",
        "checkpoint_model_output_status": "valid",
        "answer_parse_status": "parsed",
        "confidence_parse_status": "parsed",
        "answer_token_status": "located",
        "entropy_status": "computed",
        "component_versions": {"adapter": "smollm3-v1", "inducer": "v1"},
        "terminal_error_details": None,
    }


def attempt_event(record: dict, event_type: str, sequence: int) -> dict:
    is_checkpoint = record["schema_name"] == "part1_checkpoint_terminal_result"
    record_id = record["checkpoint_record_id"] if is_checkpoint else record["raw_record_id"]
    is_failure = event_type in {"attempt_failed", "attempt_interrupted"}
    attempt_number = record["terminal_attempt_number"]
    execution_outcome = record.get(
        "natural_execution_outcome", record.get("checkpoint_execution_outcome")
    )
    terminal_completion = event_type == "attempt_completed" and (
        execution_outcome == "terminal_infrastructure_failure"
    )
    terminal_category = (
        record["terminal_error_details"]["category"] if terminal_completion else None
    )
    terminal_policy = (
        classify_failure(terminal_category, attempt_number) if terminal_category else None
    )
    failure_policy = (
        classify_failure(
            "interrupted_process"
            if event_type == "attempt_interrupted"
            else "temporary_filesystem_failure",
            attempt_number,
        )
        if is_failure
        else None
    )
    event = {
        "schema_name": "part1_audit_event",
        "schema_version": "1.0.0",
        "event_id": audit_event_id(record["terminal_attempt_id"], event_type, sequence),
        "event_scope": "attempt",
        "study_id": record["study_id"],
        "model_run_id": record["model_run_id"],
        "shard_id": SHARD_ID,
        "question_id": record["question_id"],
        "run_id": record["run_id"],
        "checkpoint_id": record.get("checkpoint_id"),
        "attempt_id": record["terminal_attempt_id"],
        "attempt_number": record["terminal_attempt_number"],
        "event_sequence": sequence,
        "event_type": event_type,
        "event_timestamp": "2026-07-31T00:00:00Z",
        "execution_context": {"hostname": "node", "pid": 123},
        "outcome_category": terminal_category or (
            "interrupted_process"
            if event_type == "attempt_interrupted"
            else "temporary_filesystem_failure" if event_type == "attempt_failed" else None
        ),
        "error_details": None,
        "retry_classification": (
            terminal_policy.classification
            if terminal_policy
            else "retryable" if is_failure else None
        ),
        "retry_decision": (
            terminal_policy.retry_decision
            if terminal_policy
            else "exhausted"
            if is_failure and attempt_number == 3
            else "retry"
            if is_failure
            else None
        ),
        "backoff_seconds": (
            terminal_policy.backoff_seconds
            if terminal_policy
            else failure_policy.backoff_seconds if failure_policy else None
        ),
        "related_lock_owner": None,
        "terminal_record_id": record_id if event_type == "attempt_completed" else None,
        "operator_reason": None,
    }
    return event


def shard_event(event_type: str, sequence: int) -> dict:
    return {
        "schema_name": "part1_audit_event",
        "schema_version": "1.0.0",
        "event_id": audit_event_id(
            None,
            event_type,
            sequence,
            study_id_value=STUDY_ID,
            model_run_id_value=MODEL_RUN_ID,
            shard_id=SHARD_ID,
        ),
        "event_scope": "shard",
        "study_id": STUDY_ID,
        "model_run_id": MODEL_RUN_ID,
        "shard_id": SHARD_ID,
        "question_id": None,
        "run_id": None,
        "checkpoint_id": None,
        "attempt_id": None,
        "attempt_number": None,
        "event_sequence": sequence,
        "event_type": event_type,
        "event_timestamp": "2026-07-31T00:00:00Z",
        "execution_context": {"hostname": "node", "pid": 123},
        "outcome_category": "synthetic_fixture",
        "error_details": None,
        "retry_classification": None,
        "retry_decision": None,
        "backoff_seconds": None,
        "related_lock_owner": None,
        "terminal_record_id": None,
        "operator_reason": None,
    }


def with_attempt(record: dict, number: int) -> dict:
    changed = deepcopy(record)
    checkpoint_id = changed.get("checkpoint_id")
    changed["terminal_attempt_number"] = number
    changed["terminal_attempt_id"] = attempt_id(
        changed["study_id"],
        changed["model_run_id"],
        changed["question_id"],
        changed["run_id"],
        number,
        checkpoint_id=checkpoint_id,
    )
    return changed

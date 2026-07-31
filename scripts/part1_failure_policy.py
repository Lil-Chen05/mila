"""Shared, login-safe Phase 1 infrastructure failure policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


MAX_TOTAL_ATTEMPTS = 3
RETRYABLE_CATEGORIES = frozenset(
    {
        "interrupted_process",
        "temporary_filesystem_failure",
        "transient_worker_failure",
        "transient_cuda_runtime_failure",
    }
)
TERMINAL_CATEGORIES = frozenset(
    {
        "invalid_configuration",
        "schema_incompatibility",
        "manifest_incompatibility",
        "tokenizer_preflight_incompatibility",
        "deterministic_context_overflow",
        "reproducible_cuda_oom",
        "unsupported_model_or_tokenizer_behaviour",
        "corrupt_immutable_manifest",
    }
)


@dataclass(frozen=True)
class FailurePolicyDecision:
    classification: str
    retry_decision: str


def classify_failure(category: str, attempt_number: int) -> FailurePolicyDecision:
    if not 1 <= attempt_number <= MAX_TOTAL_ATTEMPTS:
        raise ValueError("attempt_number must be 1 through 3")
    if category in TERMINAL_CATEGORIES:
        return FailurePolicyDecision("terminal", "do_not_retry")
    if category in RETRYABLE_CATEGORIES:
        if attempt_number == MAX_TOTAL_ATTEMPTS:
            return FailurePolicyDecision("retryable", "exhausted")
        return FailurePolicyDecision("retryable", "retry")
    raise ValueError(f"unknown infrastructure failure category: {category}")


def validate_failure_event_policy(event: Mapping[str, Any]) -> None:
    category = event.get("outcome_category")
    if not isinstance(category, str):
        raise ValueError("failure policy requires a known outcome_category")
    expected = classify_failure(category, int(event["attempt_number"]))
    if event.get("retry_classification") != expected.classification:
        raise ValueError("failure retry classification differs from policy")
    if event.get("retry_decision") != expected.retry_decision:
        raise ValueError("failure retry decision differs from policy")

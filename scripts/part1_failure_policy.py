"""Shared, login-safe Phase 1 infrastructure failure policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


MAX_TOTAL_ATTEMPTS = 3
ATTEMPT_NUMBERS = (1, 2, 3)
BACKOFF_SECONDS = (0, 30, 120)
RETRYABLE_CATEGORY_ORDER = (
        "interrupted_process",
        "temporary_filesystem_failure",
        "transient_worker_failure",
        "transient_cuda_runtime_failure",
)
RETRYABLE_CATEGORIES = frozenset(RETRYABLE_CATEGORY_ORDER)
TERMINAL_CATEGORY_ORDER = (
        "invalid_configuration",
        "schema_incompatibility",
        "manifest_incompatibility",
        "tokenizer_preflight_incompatibility",
        "deterministic_context_overflow",
        "reproducible_cuda_oom",
        "unsupported_model_or_tokenizer_behaviour",
        "corrupt_immutable_manifest",
)
TERMINAL_CATEGORIES = frozenset(TERMINAL_CATEGORY_ORDER)


@dataclass(frozen=True)
class FailurePolicyDecision:
    classification: str
    retry_decision: str
    backoff_seconds: int | None


def classify_failure(category: str, attempt_number: int) -> FailurePolicyDecision:
    if not 1 <= attempt_number <= MAX_TOTAL_ATTEMPTS:
        raise ValueError("attempt_number must be 1 through 3")
    if category in TERMINAL_CATEGORIES:
        return FailurePolicyDecision("terminal", "do_not_retry", None)
    if category in RETRYABLE_CATEGORIES:
        if attempt_number == MAX_TOTAL_ATTEMPTS:
            return FailurePolicyDecision("retryable", "exhausted", None)
        return FailurePolicyDecision(
            "retryable", "retry", BACKOFF_SECONDS[attempt_number]
        )
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
    if event.get("backoff_seconds") != expected.backoff_seconds:
        raise ValueError("failure backoff_seconds differs from policy")
    if event.get("event_type") == "attempt_failed" and expected.retry_decision != "retry":
        raise ValueError(
            "attempt_failed is retry evidence; terminalization requires a terminal result"
        )

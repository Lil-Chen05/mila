"""Smoke budget/selection tests; real generation remains GPU-unverified."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from part1_contract import FIXED_SUBJECTS
from part1_store import Part1ShardStore
from part1_store_fixtures import (
    MODEL_RUN_ID,
    MODEL_RUN_MANIFEST_HASH,
    QUESTION_ID,
    SHARD_ID,
    STUDY_ID,
    attempt_event,
    checkpoint_result,
    natural_result,
)
from run_part1_smoke import select_smoke_work


def records() -> list[dict]:
    return [
        {
            "sample_index": sample_index,
            "subject": FIXED_SUBJECTS[sample_index // 100],
            "question_id": f"{sample_index:064x}",
        }
        for sample_index in range(500)
    ]


def test_smoke_a_is_one_fixed_question_all_ten_runs() -> None:
    work = select_smoke_work(records(), execution_scope="smoke_a")
    assert [(item["sample_index"], run_id) for item, run_id in work] == [
        (0, run_id) for run_id in range(10)
    ]
    assert len(work) * 11 == 110


def test_smoke_b_is_first_question_per_subject_one_run() -> None:
    work = select_smoke_work(records(), execution_scope="smoke_b")
    assert [(item["sample_index"], item["subject"], run_id) for item, run_id in work] == [
        (index * 100, subject, 0)
        for index, subject in enumerate(FIXED_SUBJECTS)
    ]
    assert len(work) * 11 == 55


def test_smoke_selection_fails_if_manifest_order_or_count_is_wrong() -> None:
    with pytest.raises(ValueError, match="500"):
        select_smoke_work(records()[:-1], execution_scope="smoke_a")
    bad = records()
    bad[100]["subject"] = FIXED_SUBJECTS[0]
    with pytest.raises(ValueError, match="subject"):
        select_smoke_work(bad, execution_scope="smoke_b")


def test_execute_natural_scopes_global_rng_without_passing_generator() -> None:
    import torch

    import run_part1_smoke
    from part1_generation import NATURAL_GENERATION_SETTINGS

    class FakeTokenizer:
        eos_token_id = 2

        def apply_chat_template(self, _messages, **_kwargs):
            return "rendered prompt"

        def __call__(self, _prompt, **_kwargs):
            input_ids = torch.tensor([[1, 2]], dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

        def decode(self, token_ids, **_kwargs):
            values = list(token_ids)
            if values in ([100], [101]):
                return "sampled reasoning"
            return "<think>sampled reasoning</think>\nAnswer: C\nConfidence: 80"

    class FakeModel:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.generation_kwargs = []

        def generate(self, *, input_ids, attention_mask, **kwargs):
            if "generator" in kwargs:
                raise TypeError("generator must not be passed to generate")
            self.generation_kwargs.append(kwargs)
            sampled = int(torch.multinomial(torch.tensor([0.5, 0.5]), 1).item())
            generated = torch.tensor(
                [[10, 100 + sampled, 11, 20, 21, 2]], dtype=torch.long
            )
            sequences = torch.cat((input_ids, generated), dim=1)
            logits = tuple(torch.tensor([[0.0, 0.0]]) for _ in generated[0])
            return SimpleNamespace(sequences=sequences, logits=logits)

    question = {
        "question": "What is 2 + 2?",
        "choices": ["1", "2", "4", "5"],
        "question_id": "e" * 64,
        "sample_index": 0,
        "subject": "high_school_mathematics",
        "gold_letter": "C",
    }
    model_manifest = {
        "study_id": "a" * 64,
        "model_run_id": "b" * 64,
        "model_run_manifest_hash": "c" * 64,
        "question_manifest_hash": "d" * 64,
        "effective_natural_generation": NATURAL_GENERATION_SETTINGS,
    }
    token_contract = {
        "reasoning_open_token_ids": [10],
        "reasoning_close_token_ids": [11],
    }
    model = FakeModel()
    tokenizer = FakeTokenizer()

    def execute():
        return run_part1_smoke._execute_natural(
            model=model,
            tokenizer=tokenizer,
            question=question,
            run_id=0,
            seed=123,
            attempt_number=1,
            model_manifest=model_manifest,
            token_contract=token_contract,
        )

    torch.manual_seed(987)
    state_before_first = torch.random.get_rng_state().clone()
    first = execute()
    assert torch.equal(torch.random.get_rng_state(), state_before_first)

    torch.rand(3)
    state_before_second = torch.random.get_rng_state().clone()
    second = execute()
    assert torch.equal(torch.random.get_rng_state(), state_before_second)

    assert first["generated_token_ids"] == second["generated_token_ids"]
    assert model.generation_kwargs == [NATURAL_GENERATION_SETTINGS] * 2


def session(tmp_path):
    store = Part1ShardStore(
        tmp_path / "shard",
        shard_id=SHARD_ID,
        study_id=STUDY_ID,
        model_run_id=MODEL_RUN_ID,
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
        unsafe_for_tests=True,
    )
    store.initialize_provenance_header()
    return SimpleNamespace(store=store, owner=SimpleNamespace(shard_id=SHARD_ID))


def natural_work():
    from part1_runtime import WorkSpec

    return WorkSpec.natural(
        STUDY_ID,
        MODEL_RUN_ID,
        MODEL_RUN_MANIFEST_HASH,
        QUESTION_ID,
        0,
        seed=123,
    )


def failure_result(attempt_number, category, reference, details):
    result = natural_result(
        attempt_number=attempt_number,
        outcome="terminal_infrastructure_failure",
    )
    result["infrastructure_failure_reference"] = reference
    result["terminal_error_details"] = dict(details, category=category)
    return result


def test_smoke_lifecycle_retries_policy_authorized_filesystem_failure(tmp_path) -> None:
    import run_part1_smoke

    active = session(tmp_path)
    attempts = []
    backoffs = []

    def execute(attempt_number):
        attempts.append(attempt_number)
        if attempt_number == 1:
            raise OSError("synthetic filesystem outage")
        return natural_result(attempt_number=attempt_number)

    status, published = run_part1_smoke._run_work_lifecycle(
        active,
        work=natural_work(),
        execute_attempt=execute,
        build_terminal_failure=failure_result,
        sleep=backoffs.append,
    )

    assert (status, published) == ("completed", True)
    assert attempts == [1, 2]
    assert backoffs == [30]
    events = active.store.inspect().audit_events
    assert [event["event_type"] for event in events] == [
        "attempt_started",
        "attempt_failed",
        "attempt_started",
        "attempt_completed",
    ]
    assert events[1]["outcome_category"] == "temporary_filesystem_failure"
    assert events[1]["retry_classification"] == "retryable"
    assert events[1]["retry_decision"] == "retry"
    assert events[1]["backoff_seconds"] == 30


def test_smoke_lifecycle_terminalizes_nonretryable_execution_failure(tmp_path) -> None:
    import run_part1_smoke

    active = session(tmp_path)

    def execute(_attempt_number):
        raise ValueError("synthetic unsupported output contract")

    status, published = run_part1_smoke._run_work_lifecycle(
        active,
        work=natural_work(),
        execute_attempt=execute,
        build_terminal_failure=failure_result,
        sleep=lambda _seconds: None,
    )

    assert (status, published) == ("terminal", True)
    inspection = active.store.inspect()
    assert inspection.natural_results[0]["terminal_error_details"]["category"] == (
        "unsupported_model_or_tokenizer_behaviour"
    )
    completion = inspection.audit_events[-1]
    assert completion["event_type"] == "attempt_completed"
    assert completion["retry_classification"] == "terminal"
    assert completion["retry_decision"] == "do_not_retry"


def test_smoke_lifecycle_terminalizes_deterministic_commit_preflight_rejection(
    tmp_path,
) -> None:
    import run_part1_smoke

    active = session(tmp_path)
    real_preflight = active.store.preflight_terminal_commit
    executions = []

    def reject_successful_result(record, completion_event):
        if record.get("natural_execution_outcome") == "complete":
            raise ValueError("synthetic schema/scientific preflight rejection")
        return real_preflight(record, completion_event)

    active.store.preflight_terminal_commit = reject_successful_result

    status, published = run_part1_smoke._run_work_lifecycle(
        active,
        work=natural_work(),
        execute_attempt=lambda attempt: (
            executions.append(attempt) or natural_result(attempt_number=attempt)
        ),
        build_terminal_failure=failure_result,
        sleep=lambda _seconds: None,
    )

    assert (status, published) == ("terminal", True)
    assert executions == [1]
    inspection = active.store.inspect()
    assert inspection.natural_results[0]["terminal_error_details"]["category"] == (
        "unsupported_model_or_tokenizer_behaviour"
    )
    assert [event["event_type"] for event in inspection.audit_events] == [
        "attempt_started",
        "attempt_completed",
    ]


def test_smoke_lifecycle_cuda_retry_requires_a_fresh_process(tmp_path) -> None:
    import run_part1_smoke
    from part1_runtime import FreshProcessRequired

    class SyntheticCudaRuntimeError(RuntimeError):
        pass

    active = session(tmp_path)
    with pytest.raises(FreshProcessRequired, match="fresh CUDA process"):
        run_part1_smoke._run_work_lifecycle(
            active,
            work=natural_work(),
            execute_attempt=lambda _attempt: (_ for _ in ()).throw(
                SyntheticCudaRuntimeError("synthetic CUDA launch failure")
            ),
            build_terminal_failure=failure_result,
            sleep=lambda _seconds: None,
        )

    inspection = active.store.inspect()
    assert inspection.natural_results == ()
    assert [event["event_type"] for event in inspection.audit_events] == [
        "attempt_started",
        "attempt_failed",
    ]
    assert inspection.audit_events[-1]["outcome_category"] == (
        "transient_cuda_runtime_failure"
    )


def test_smoke_lifecycle_publishes_required_terminalization_without_execution(
    tmp_path,
) -> None:
    import run_part1_smoke

    active = session(tmp_path)
    for attempt_number in (1, 2):
        record = natural_result(attempt_number=attempt_number)
        active.store.append_audit_event(attempt_event(record, "attempt_started", 0))
        active.store.append_audit_event(attempt_event(record, "attempt_failed", 1))
    third = natural_result(attempt_number=3)
    active.store.append_audit_event(attempt_event(third, "attempt_started", 0))

    status, published = run_part1_smoke._run_work_lifecycle(
        active,
        work=natural_work(),
        execute_attempt=lambda _attempt: pytest.fail("terminalization regenerated work"),
        build_terminal_failure=failure_result,
        sleep=lambda _seconds: None,
    )

    assert (status, published) == ("terminal", True)
    inspection = active.store.inspect()
    assert inspection.natural_results[0]["terminal_error_details"]["category"] == (
        "interrupted_process"
    )
    third_events = [
        event for event in inspection.audit_events if event["attempt_number"] == 3
    ]
    assert [event["event_type"] for event in third_events] == [
        "attempt_started",
        "attempt_interrupted",
        "attempt_completed",
    ]


def test_smoke_lifecycle_terminalizes_final_completion_without_result(tmp_path) -> None:
    import run_part1_smoke

    active = session(tmp_path)
    for attempt_number in (1, 2):
        record = natural_result(attempt_number=attempt_number)
        active.store.append_audit_event(attempt_event(record, "attempt_started", 0))
        active.store.append_audit_event(attempt_event(record, "attempt_failed", 1))
    third = natural_result(attempt_number=3)
    active.store.append_audit_event(attempt_event(third, "attempt_started", 0))
    active.store.append_audit_event(attempt_event(third, "attempt_completed", 1))

    status, published = run_part1_smoke._run_work_lifecycle(
        active,
        work=natural_work(),
        execute_attempt=lambda _attempt: pytest.fail("terminalization regenerated work"),
        build_terminal_failure=failure_result,
        sleep=lambda _seconds: None,
    )

    assert (status, published) == ("terminal", True)
    inspection = active.store.inspect()
    assert inspection.natural_results[0]["terminal_error_details"]["category"] == (
        "interrupted_process"
    )
    third_events = [
        event for event in inspection.audit_events if event["attempt_number"] == 3
    ]
    assert [event["event_type"] for event in third_events] == [
        "attempt_started",
        "attempt_completed",
        "attempt_interrupted",
        "attempt_completed",
    ]
    assert [event["event_sequence"] for event in third_events] == [0, 1, 2, 3]


@pytest.mark.parametrize("checkpoint", [False, True])
def test_smoke_lifecycle_never_regenerates_completed_work(tmp_path, checkpoint) -> None:
    import run_part1_smoke
    from part1_runtime import WorkSpec

    active = session(tmp_path)
    parent = natural_result()
    active.store.append_audit_event(attempt_event(parent, "attempt_started", 0))
    active.store.commit_terminal_result(parent, attempt_event(parent, "attempt_completed", 1))
    if checkpoint:
        record = checkpoint_result()
        active.store.append_audit_event(attempt_event(record, "attempt_started", 0))
        active.store.commit_terminal_result(
            record,
            attempt_event(record, "attempt_completed", 1),
        )
        work = WorkSpec.checkpoint(
            STUDY_ID,
            MODEL_RUN_ID,
            MODEL_RUN_MANIFEST_HASH,
            QUESTION_ID,
            0,
            "cp-05",
            seed=123,
        )
    else:
        work = natural_work()

    status, published = run_part1_smoke._run_work_lifecycle(
        active,
        work=work,
        execute_attempt=lambda _attempt: pytest.fail("completed work was regenerated"),
        build_terminal_failure=failure_result,
        sleep=lambda _seconds: None,
    )
    assert (status, published) == ("completed", False)

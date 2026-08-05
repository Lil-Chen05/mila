"""Production shard selection and orchestration tests (synthetic execution only)."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

import pytest

from part1_checkpoints import (
    CheckpointGenerationCapture,
    build_checkpoint_terminal_result,
)
from part1_contract import FIXED_SUBJECTS
from part1_contract import derive_generation_seed
from part1_generation import (
    NaturalGenerationCapture,
    build_natural_infrastructure_failure_result,
    build_natural_terminal_result,
)
from part1_manifests import load_manifest_bundle
from part1_model_run import (
    build_production_model_run_manifest,
    build_smoke_model_run_manifest,
)
from part1_store import Part1ShardStore
from test_part1_model_run import preflight as synthetic_preflight


def records() -> list[dict]:
    return [
        {
            "sample_index": sample_index,
            "subject": FIXED_SUBJECTS[sample_index // 100],
            "question_id": f"{sample_index:064x}",
        }
        for sample_index in range(500)
    ]


def test_select_shard_work_partitions_every_natural_key_exactly_once() -> None:
    from run_part1_shard import select_shard_work

    selected = [
        item
        for shard_index in range(500)
        for item in select_shard_work(
            records(), shard_index=shard_index, shard_count=500
        )
    ]

    keys = [(question["sample_index"], run_id) for question, run_id in selected]
    assert len(keys) == 5_000
    assert len(set(keys)) == 5_000
    assert set(keys) == {
        (sample_index, run_id)
        for sample_index in range(500)
        for run_id in range(10)
    }
    assert all(
        len(
            select_shard_work(
                records(), shard_index=shard_index, shard_count=500
            )
        )
        == 10
        for shard_index in range(500)
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop(), "500"),
        (lambda value: value[1].update(sample_index=2), "sample_index"),
        (lambda value: value[100].update(subject=FIXED_SUBJECTS[0]), "subject"),
    ],
)
def test_select_shard_work_rejects_noncanonical_manifest(mutation, message: str) -> None:
    from run_part1_shard import select_shard_work

    value = records()
    mutation(value)
    with pytest.raises(ValueError, match=message):
        select_shard_work(value, shard_index=0, shard_count=500)


@pytest.mark.parametrize(
    ("shard_index", "shard_count"),
    [(-1, 500), (500, 500), (0, 0), (True, 500), (0, True)],
)
def test_select_shard_work_rejects_invalid_partition(
    shard_index: object, shard_count: object
) -> None:
    from run_part1_shard import select_shard_work

    with pytest.raises(ValueError, match="shard"):
        select_shard_work(
            records(), shard_index=shard_index, shard_count=shard_count
        )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def production_fixture(tmp_path: Path) -> dict:
    repository = tmp_path / "repository"
    manifest_root = repository / "manifests" / "part1"
    shutil.copytree(REPOSITORY_ROOT / "manifests" / "part1", manifest_root)
    shutil.copy2(REPOSITORY_ROOT / "uv.lock", repository / "uv.lock")
    (repository / ".gitignore").write_text(
        "results/part1/\nresults/part1-smoke/\n", encoding="utf-8"
    )
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Part 1 Test")
    _git(repository, "config", "user.email", "part1-test@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "fixture")
    head = _git(repository, "rev-parse", "HEAD")

    bundle = load_manifest_bundle(
        questions_path=manifest_root / "questions.jsonl",
        question_manifest_path=manifest_root / "questions.manifest.json",
        study_manifest_path=manifest_root / "study_manifest.json",
    )
    preflight = copy.deepcopy(synthetic_preflight())
    for field in ("study_id", "study_manifest_hash", "question_manifest_hash"):
        preflight[field] = bundle.study_manifest[field]
    preflight["environment_versions"]["uv_lock_sha256"] = hashlib.sha256(
        (repository / "uv.lock").read_bytes()
    ).hexdigest()
    preflight_path = repository / "results" / "part1-smoke" / "preflight" / "preflight.json"
    _write_json(preflight_path, preflight)
    manifest = build_production_model_run_manifest(
        study_manifest=bundle.study_manifest,
        preflight_report=preflight,
        final_git_commit=head,
        output_root=Path("results/part1"),
    )
    manifest_path = (
        repository
        / "results"
        / "part1"
        / manifest["model_run_id"]
        / "model_run_manifest.json"
    )
    _write_json(manifest_path, manifest)
    return {
        "repository": repository,
        "manifest_root": manifest_root,
        "preflight_path": preflight_path,
        "preflight": preflight,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "bundle": bundle,
    }


def _fake_natural(**kwargs) -> dict:
    question = kwargs["question"]
    manifest = kwargs["model_manifest"]
    capture = NaturalGenerationCapture(
        rendered_prompt="synthetic prompt",
        prompt_token_ids=(1, 2),
        generated_token_ids=(10, 99, 11, 20, 2),
        decoded_output="<think>x</think>\nAnswer: A\nConfidence: 80",
        raw_prewarper_logits=(),
        stop_reason="eos",
        precomputed_entropy_nats=(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    return build_natural_terminal_result(
        identity={
            "study_id": manifest["study_id"],
            "model_run_id": manifest["model_run_id"],
            "model_run_manifest_hash": manifest["model_run_manifest_hash"],
            "question_manifest_hash": manifest["question_manifest_hash"],
            "question_id": question["question_id"],
            "sample_index": question["sample_index"],
            "subject": question["subject"],
            "gold_letter": question["gold_letter"],
        },
        run_id=kwargs["run_id"],
        generation_seed=kwargs["seed"],
        terminal_attempt_number=kwargs["attempt_number"],
        capture=capture,
        token_contract=kwargs["token_contract"],
        decode_reasoning=lambda _ids: "x",
    )


def _fake_abnormal_natural(**kwargs) -> dict:
    question = kwargs["question"]
    manifest = kwargs["model_manifest"]
    capture = NaturalGenerationCapture(
        rendered_prompt="synthetic prompt",
        prompt_token_ids=(1, 2),
        generated_token_ids=(10, 99, 11, 2),
        decoded_output="<think>x</think>\nNo terminal answer block",
        raw_prewarper_logits=(),
        stop_reason="eos",
        precomputed_entropy_nats=(1.0, 1.0, 1.0, 1.0),
    )
    return build_natural_terminal_result(
        identity={
            "study_id": manifest["study_id"],
            "model_run_id": manifest["model_run_id"],
            "model_run_manifest_hash": manifest["model_run_manifest_hash"],
            "question_manifest_hash": manifest["question_manifest_hash"],
            "question_id": question["question_id"],
            "sample_index": question["sample_index"],
            "subject": question["subject"],
            "gold_letter": question["gold_letter"],
        },
        run_id=kwargs["run_id"],
        generation_seed=kwargs["seed"],
        terminal_attempt_number=kwargs["attempt_number"],
        capture=capture,
        token_contract=kwargs["token_contract"],
        decode_reasoning=lambda _ids: "x",
    )


def _fake_checkpoint(**kwargs) -> dict:
    token_contract = kwargs["token_contract"]
    logits = tuple(float(index) / 10 for index in range(24))
    capture = CheckpointGenerationCapture(
        forced_generated_token_ids=(20, 2),
        decoded_forced_output=" A\nConfidence: 80",
        raw_prewarper_logits=(),
        answer_step_raw_logits=logits,
    )
    return build_checkpoint_terminal_result(
        parent=kwargs["parent"],
        plan=kwargs["plan"],
        capture=capture,
        token_contract=token_contract,
        gold_letter=kwargs["gold_letter"],
        terminal_attempt_number=kwargs["attempt_number"],
    )


def _runner_arguments(fixture: dict) -> dict:
    return {
        "execution_scope": "production",
        "shard_index": 0,
        "shard_count": 500,
        "repository_root": fixture["repository"],
        "manifest_root": fixture["manifest_root"],
        "preflight_path": fixture["preflight_path"],
        "model_run_manifest_path": fixture["manifest_path"],
        "load_runtime": lambda **_kwargs: (object(), object()),
        "inspect_tokenizer": lambda _tokenizer: fixture["preflight"]["token_contract"],
        "execute_natural": _fake_natural,
        "execute_checkpoint": _fake_checkpoint,
    }


def _store(fixture: dict) -> Part1ShardStore:
    manifest = fixture["manifest"]
    return Part1ShardStore(
        fixture["repository"] / manifest["output_paths"]["raw_shards"] / "shard-000",
        shard_id="shard-000",
        study_id=manifest["study_id"],
        model_run_id=manifest["model_run_id"],
        model_run_manifest_hash=manifest["model_run_manifest_hash"],
        unsafe_for_tests=True,
    )


def _terminal_failure_execution(**_kwargs):
    raise ValueError("synthetic unsupported model behavior")


def _stream_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def _populate_extra_natural(fixture: dict, *, record_index: int = 1) -> None:
    from part1_runtime import LockMetadata, LockedShardSession, WorkSpec
    from run_part1_smoke import _run_work_lifecycle

    manifest = fixture["manifest"]
    question = fixture["bundle"].records[record_index]
    seed = derive_generation_seed(
        base_seed=manifest["base_generation_seed"],
        canonical_model_identity=manifest["canonical_model_identity"],
        question_id=question["question_id"],
        run_id=0,
    )
    owner = LockMetadata(
        lock_id="9" * 32,
        study_id=manifest["study_id"],
        model_run_id=manifest["model_run_id"],
        shard_id="shard-000",
        hostname=socket.gethostname(),
        pid=os.getpid(),
        slurm_job_id=None,
        slurm_array_task_id=None,
        acquired_at="2026-08-05T00:00:00Z",
    )
    with LockedShardSession.acquire(
        _store(fixture).root,
        owner=owner,
        model_run_manifest_hash=manifest["model_run_manifest_hash"],
    ) as session:
        work = WorkSpec.natural(
            manifest["study_id"],
            manifest["model_run_id"],
            manifest["model_run_manifest_hash"],
            question["question_id"],
            0,
            seed=seed,
        )
        identity = {
            "study_id": manifest["study_id"],
            "model_run_id": manifest["model_run_id"],
            "model_run_manifest_hash": manifest["model_run_manifest_hash"],
            "question_manifest_hash": manifest["question_manifest_hash"],
            "question_id": question["question_id"],
            "sample_index": question["sample_index"],
            "subject": question["subject"],
            "gold_letter": question["gold_letter"],
        }

        def terminal_failure(attempt, category, reference, details):
            return build_natural_infrastructure_failure_result(
                identity=identity,
                run_id=0,
                generation_seed=seed,
                terminal_attempt_number=attempt,
                prompt_hash=manifest["prompt_hash"],
                failure_category=category,
                infrastructure_failure_reference=reference,
                error_details=details,
            )

        _run_work_lifecycle(
            session,
            work=work,
            execute_attempt=lambda attempt: _fake_natural(
                model=object(),
                tokenizer=object(),
                question=question,
                run_id=0,
                seed=seed,
                attempt_number=attempt,
                model_manifest=manifest,
                token_contract=fixture["preflight"]["token_contract"],
            ),
            build_terminal_failure=terminal_failure,
        )


def test_checkpoint_only_resume_uses_durable_natural_without_regeneration(
    tmp_path: Path,
) -> None:
    from part1_runtime import FreshProcessRequired
    from run_part1_shard import run_part1_shard

    fixture = production_fixture(tmp_path)
    interrupted = False

    class SyntheticCudaFailure(RuntimeError):
        part1_failure_category = "transient_cuda_runtime_failure"

    def interrupt_first_checkpoint(**kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise SyntheticCudaFailure("synthetic fresh-process boundary")
        return _fake_checkpoint(**kwargs)

    first_args = _runner_arguments(fixture)
    first_args["execute_natural"] = _fake_abnormal_natural
    first_args["execute_checkpoint"] = interrupt_first_checkpoint
    with pytest.raises(FreshProcessRequired, match="fresh CUDA process"):
        run_part1_shard(**first_args)

    interrupted_state = _store(fixture).build_index()
    assert len(interrupted_state.natural_terminal_by_key) == 1
    assert len(interrupted_state.checkpoint_terminal_by_key) == 0

    resume_args = _runner_arguments(fixture)
    def resume_natural(**kwargs):
        if kwargs["run_id"] == 0:
            pytest.fail("checkpoint-only resume regenerated a durable natural")
        raise SyntheticCudaFailure("stop after checkpoint-only resume evidence")

    resume_args["execute_natural"] = resume_natural
    with pytest.raises(FreshProcessRequired, match="fresh CUDA process"):
        run_part1_shard(**resume_args)

    final_store = _store(fixture)
    final_state = final_store.build_index()
    assert len(final_state.natural_terminal_by_key) == 1
    assert len(final_state.checkpoint_terminal_by_key) == 11
    assert not final_store.finalization_path.exists()
    assert final_state.hierarchy_errors == ()
    assert final_state.lifecycle_errors == ()

    abnormal = next(iter(final_state.natural_terminal_by_key.values()))
    assert abnormal["natural_execution_outcome"] == "complete"
    assert abnormal["answer_parse_status"] == "missing"
    assert abnormal["checkpoint_eligible"] is True

    finish_args = _runner_arguments(fixture)

    def finish_naturals(**kwargs):
        if kwargs["run_id"] == 0:
            pytest.fail("finalization regenerated the durable abnormal natural")
        return _terminal_failure_execution(**kwargs)

    finish_args["execute_natural"] = finish_naturals
    finish_args["execute_checkpoint"] = lambda **_kwargs: pytest.fail(
        "finalization regenerated a durable checkpoint"
    )
    report = run_part1_shard(**finish_args)
    finalized_state = final_store.build_index()
    assert report["status"] == "completed"
    assert len(finalized_state.natural_terminal_by_key) == 10
    assert len(finalized_state.checkpoint_terminal_by_key) == 11
    assert final_store.finalization_path.exists()

    resubmit = _runner_arguments(fixture)
    resubmit["load_runtime"] = lambda **_kwargs: pytest.fail(
        "compatible finalized shard loaded model"
    )
    assert run_part1_shard(**resubmit)["status"] == "already_finalized"

    final_store.finalization_path.unlink()
    checkpoints = [
        json.loads(line)
        for line in final_store.checkpoint_results_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    checkpoints = [
        record
        for record in checkpoints
        if not (record["run_id"] == 0 and record["checkpoint_id"] == "cp-06")
    ]
    final_store.checkpoint_results_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in checkpoints
        ),
        encoding="utf-8",
    )
    events = [
        json.loads(line)
        for line in final_store.audit_events_path.read_text(encoding="utf-8").splitlines()
    ]
    events = [
        event
        for event in events
        if not (event.get("run_id") == 0 and event.get("checkpoint_id") == "cp-06")
    ]
    final_store.audit_events_path.write_text(
        "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )
    missing_args = _runner_arguments(fixture)
    missing_args["execute_natural"] = lambda **_kwargs: pytest.fail(
        "missing-checkpoint resume regenerated a natural"
    )
    missing_args["execute_checkpoint"] = lambda **_kwargs: (_ for _ in ()).throw(
        SyntheticCudaFailure("synthetic missing-checkpoint stop")
    )
    with pytest.raises(FreshProcessRequired, match="fresh CUDA process"):
        run_part1_shard(**missing_args)
    assert not final_store.finalization_path.exists()


def test_terminal_natural_failures_are_final_and_checkpoint_ineligible(
    tmp_path: Path,
) -> None:
    from run_part1_shard import run_part1_shard

    fixture = production_fixture(tmp_path)
    args = _runner_arguments(fixture)
    args["execute_natural"] = lambda **_kwargs: (_ for _ in ()).throw(
        ValueError("synthetic unsupported model behavior")
    )
    args["execute_checkpoint"] = lambda **_kwargs: pytest.fail(
        "checkpoint executed for failed natural"
    )

    run_part1_shard(**args)

    store = _store(fixture)
    state = store.build_index()
    assert len(state.natural_terminal_by_key) == 10
    assert {
        record["natural_execution_outcome"]
        for record in state.natural_terminal_by_key.values()
    } == {"terminal_infrastructure_failure"}
    assert len(state.checkpoint_terminal_by_key) == 0
    assert store.finalization_path.exists()

    args = _runner_arguments(fixture)
    args["load_runtime"] = lambda **_kwargs: pytest.fail("finalized shard loaded model")
    report = run_part1_shard(**args)
    assert report["status"] == "already_finalized"
    assert not (store.root / ".writer.lock").exists()


def test_live_lock_refuses_automatic_recovery_before_model_load(tmp_path: Path) -> None:
    from run_part1_shard import run_part1_shard
    from part1_runtime import LockMetadata, LockedShardSession, StaleRecoveryRefused

    fixture = production_fixture(tmp_path)
    manifest = fixture["manifest"]
    root = _store(fixture).root
    owner = LockMetadata(
        lock_id="1" * 32,
        study_id=manifest["study_id"],
        model_run_id=manifest["model_run_id"],
        shard_id="shard-000",
        hostname=__import__("socket").gethostname(),
        pid=__import__("os").getpid(),
        slurm_job_id=None,
        slurm_array_task_id=None,
        acquired_at="2026-08-05T00:00:00Z",
    )
    held = LockedShardSession.acquire(
        root,
        owner=owner,
        model_run_manifest_hash=manifest["model_run_manifest_hash"],
    )
    args = _runner_arguments(fixture)
    args["load_runtime"] = lambda **_kwargs: pytest.fail("locked shard loaded model")
    try:
        with pytest.raises(StaleRecoveryRefused, match="live"):
            run_part1_shard(**args)
    finally:
        held.close()


def test_unknown_lock_refuses_before_model_load(tmp_path: Path) -> None:
    from run_part1_shard import run_part1_shard
    from part1_runtime import LockMetadata, LockedShardSession, StaleRecoveryRefused

    fixture = production_fixture(tmp_path)
    manifest = fixture["manifest"]
    owner = LockMetadata(
        lock_id="3" * 32,
        study_id=manifest["study_id"],
        model_run_id=manifest["model_run_id"],
        shard_id="shard-000",
        hostname="remote-worker.invalid",
        pid=12345,
        slurm_job_id=None,
        slurm_array_task_id=None,
        acquired_at="2026-08-05T00:00:00Z",
    )
    held = LockedShardSession.acquire(
        _store(fixture).root,
        owner=owner,
        model_run_manifest_hash=manifest["model_run_manifest_hash"],
    )
    args = _runner_arguments(fixture)
    args["load_runtime"] = lambda **_kwargs: pytest.fail("unknown lock loaded model")
    try:
        with pytest.raises(StaleRecoveryRefused, match="uncertain"):
            run_part1_shard(**args)
    finally:
        held.close()


def test_conclusive_dead_lock_takeover_resumes_safely(tmp_path: Path) -> None:
    from part1_runtime import FreshProcessRequired, LockMetadata, LockedShardSession
    from run_part1_shard import run_part1_shard

    fixture = production_fixture(tmp_path)
    manifest = fixture["manifest"]
    owner = LockMetadata(
        lock_id="4" * 32,
        study_id=manifest["study_id"],
        model_run_id=manifest["model_run_id"],
        shard_id="shard-000",
        hostname=socket.gethostname(),
        pid=99_999_999,
        slurm_job_id=None,
        slurm_array_task_id=None,
        acquired_at="2026-08-05T00:00:00Z",
    )
    LockedShardSession.acquire(
        _store(fixture).root,
        owner=owner,
        model_run_manifest_hash=manifest["model_run_manifest_hash"],
    )

    class SyntheticCudaFailure(RuntimeError):
        part1_failure_category = "transient_cuda_runtime_failure"

    args = _runner_arguments(fixture)
    args["execute_natural"] = lambda **_kwargs: (_ for _ in ()).throw(
        SyntheticCudaFailure("synthetic post-takeover stop")
    )
    with pytest.raises(FreshProcessRequired, match="fresh CUDA process"):
        run_part1_shard(**args)

    inspection = _store(fixture).inspect()
    assert any(
        event["event_type"] == "stale_lock_recovered"
        for event in inspection.audit_events
    )
    assert not (_store(fixture).root / ".writer.lock").exists()


def test_incomplete_finalized_shard_is_fatal_before_model_load(tmp_path: Path) -> None:
    from part1_runtime import LockMetadata, LockedShardSession
    from run_part1_shard import run_part1_shard

    fixture = production_fixture(tmp_path)
    manifest = fixture["manifest"]
    root = _store(fixture).root
    owner = LockMetadata(
        lock_id="2" * 32,
        study_id=manifest["study_id"],
        model_run_id=manifest["model_run_id"],
        shard_id="shard-000",
        hostname="synthetic",
        pid=1,
        slurm_job_id=None,
        slurm_array_task_id=None,
        acquired_at="2026-08-05T00:00:00Z",
    )
    with LockedShardSession.acquire(
        root,
        owner=owner,
        model_run_manifest_hash=manifest["model_run_manifest_hash"],
    ) as session:
        session.store.finalize()
    args = _runner_arguments(fixture)
    args["load_runtime"] = lambda **_kwargs: pytest.fail("incomplete shard loaded model")

    with pytest.raises(RuntimeError, match="finalized.*incomplete|coverage"):
        run_part1_shard(**args)


def test_scope_and_output_path_refusals_happen_before_model_load(tmp_path: Path) -> None:
    from run_part1_shard import run_part1_shard

    fixture = production_fixture(tmp_path)
    args = _runner_arguments(fixture)
    args["output_root"] = fixture["repository"] / "results" / "wrong-root"
    args["load_runtime"] = lambda **_kwargs: pytest.fail("wrong path loaded model")
    with pytest.raises(ValueError, match="output root|raw_shards"):
        run_part1_shard(**args)

    smoke = build_smoke_model_run_manifest(
        study_manifest=fixture["bundle"].study_manifest,
        preflight_report=fixture["preflight"],
        execution_scope="phase3_smoke",
        base_git_commit=_git(fixture["repository"], "rev-parse", "HEAD"),
        diff_hash=hashlib.sha256(b"").hexdigest(),
    )
    smoke_path = fixture["repository"] / "results" / "part1-smoke" / "model-runs" / (
        "phase3_smoke/model_run_manifest.json"
    )
    _write_json(smoke_path, smoke)
    args = _runner_arguments(fixture)
    args.update(
        execution_scope="phase3_smoke",
        model_run_manifest_path=smoke_path,
        output_root=fixture["repository"] / fixture["manifest"]["output_paths"]["raw_shards"],
        load_runtime=lambda **_kwargs: pytest.fail("aliased smoke path loaded model"),
    )
    with pytest.raises(ValueError, match="smoke.*(production|canonical)|separate"):
        run_part1_shard(**args)


def test_extra_valid_natural_fails_closed_without_mutating_active_shard(
    tmp_path: Path,
) -> None:
    from run_part1_shard import run_part1_shard

    fixture = production_fixture(tmp_path)
    _populate_extra_natural(fixture)
    store = _store(fixture)
    before = _stream_bytes(store.root)
    args = _runner_arguments(fixture)
    args["load_runtime"] = lambda **_kwargs: pytest.fail(
        "contaminated shard loaded the model"
    )

    with pytest.raises(RuntimeError, match="extra|incompatible|assigned"):
        run_part1_shard(**args)

    assert _stream_bytes(store.root) == before
    assert not store.finalization_path.exists()


def test_malformed_active_stream_fails_closed_without_mutation(tmp_path: Path) -> None:
    from part1_runtime import LockMetadata, LockedShardSession
    from run_part1_shard import run_part1_shard

    fixture = production_fixture(tmp_path)
    manifest = fixture["manifest"]
    store = _store(fixture)
    owner = LockMetadata(
        lock_id="8" * 32,
        study_id=manifest["study_id"],
        model_run_id=manifest["model_run_id"],
        shard_id="shard-000",
        hostname=socket.gethostname(),
        pid=os.getpid(),
        slurm_job_id=None,
        slurm_array_task_id=None,
        acquired_at="2026-08-05T00:00:00Z",
    )
    with LockedShardSession.acquire(
        store.root,
        owner=owner,
        model_run_manifest_hash=manifest["model_run_manifest_hash"],
    ):
        pass
    store.natural_results_path.write_bytes(b'{"malformed":')
    before = _stream_bytes(store.root)
    args = _runner_arguments(fixture)
    args["load_runtime"] = lambda **_kwargs: pytest.fail("malformed shard loaded model")

    with pytest.raises(RuntimeError, match="malformed|corrupt|stream|trailing"):
        run_part1_shard(**args)

    assert _stream_bytes(store.root) == before
    assert not store.finalization_path.exists()


def test_lifecycle_incompatible_active_stream_fails_closed(tmp_path: Path) -> None:
    from part1_contract import audit_event_id
    from run_part1_shard import run_part1_shard

    fixture = production_fixture(tmp_path)
    _populate_extra_natural(fixture, record_index=0)
    store = _store(fixture)
    events = [
        json.loads(line)
        for line in store.audit_events_path.read_text(encoding="utf-8").splitlines()
    ]
    duplicate_start = dict(events[0])
    duplicate_start["event_sequence"] = 2
    duplicate_start["event_id"] = audit_event_id(
        duplicate_start["attempt_id"], "attempt_started", 2
    )
    with store.audit_events_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(duplicate_start, sort_keys=True, separators=(",", ":")) + "\n"
        )
    before = _stream_bytes(store.root)
    args = _runner_arguments(fixture)
    args["load_runtime"] = lambda **_kwargs: pytest.fail(
        "lifecycle-incompatible shard loaded model"
    )

    with pytest.raises(RuntimeError, match="lifecycle|incompatible|corrupt"):
        run_part1_shard(**args)

    assert _stream_bytes(store.root) == before
    assert not store.finalization_path.exists()


def test_output_roots_reject_symlink_and_noncanonical_smoke_override(
    tmp_path: Path,
) -> None:
    from run_part1_shard import run_part1_shard

    fixture = production_fixture(tmp_path)
    raw_root = fixture["repository"] / fixture["manifest"]["output_paths"][
        "raw_shards"
    ]
    outside = tmp_path / "outside"
    outside.mkdir()
    raw_root.parent.mkdir(parents=True, exist_ok=True)
    raw_root.symlink_to(outside, target_is_directory=True)
    args = _runner_arguments(fixture)
    args["load_runtime"] = lambda **_kwargs: pytest.fail("symlinked path loaded model")
    with pytest.raises((ValueError, RuntimeError), match="symlink|directory|unsafe"):
        run_part1_shard(**args)
    assert not (outside / "shard-000").exists()

    raw_root.unlink()
    raw_root.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError), match="directory|unsafe"):
        run_part1_shard(**args)
    raw_root.unlink()
    smoke = build_smoke_model_run_manifest(
        study_manifest=fixture["bundle"].study_manifest,
        preflight_report=fixture["preflight"],
        execution_scope="phase3_smoke",
        base_git_commit=_git(fixture["repository"], "rev-parse", "HEAD"),
        diff_hash=hashlib.sha256(b"").hexdigest(),
    )
    smoke_path = fixture["repository"] / (
        "results/part1-smoke/model-runs/phase3_smoke/model_run_manifest.json"
    )
    _write_json(smoke_path, smoke)
    args = _runner_arguments(fixture)
    args.update(
        execution_scope="phase3_smoke",
        model_run_manifest_path=smoke_path,
        output_root=fixture["repository"] / "results/part1-smoke/phase3_smoke/arbitrary/raw_shards",
        load_runtime=lambda **_kwargs: pytest.fail("noncanonical smoke path loaded model"),
    )
    with pytest.raises(ValueError, match="canonical|exact|smoke"):
        run_part1_shard(**args)


def test_output_root_rejects_unsafe_ancestor_components(tmp_path: Path) -> None:
    from run_part1_shard import run_part1_shard

    fixture = production_fixture(tmp_path)
    external_manifest = tmp_path / "model_run_manifest.json"
    shutil.copy2(fixture["manifest_path"], external_manifest)
    run_root = fixture["manifest_path"].parent
    shutil.rmtree(run_root)
    outside = tmp_path / "ancestor-target"
    outside.mkdir()
    run_root.symlink_to(outside, target_is_directory=True)
    args = _runner_arguments(fixture)
    args["model_run_manifest_path"] = external_manifest
    args["load_runtime"] = lambda **_kwargs: pytest.fail(
        "symlinked ancestor loaded model"
    )

    with pytest.raises(ValueError, match="symlink|directory"):
        run_part1_shard(**args)
    assert not (outside / "raw_shards").exists()

    run_root.unlink()
    run_root.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        run_part1_shard(**args)
    assert not (outside / "raw_shards").exists()


def test_finalized_marker_type_json_and_identity_are_validated_before_no_load(
    tmp_path: Path,
) -> None:
    from run_part1_shard import run_part1_shard

    fixture = production_fixture(tmp_path)
    args = _runner_arguments(fixture)
    args["execute_natural"] = _terminal_failure_execution
    args["execute_checkpoint"] = lambda **_kwargs: pytest.fail(
        "checkpoint executed for failed natural"
    )
    run_part1_shard(**args)
    marker = _store(fixture).finalization_path
    valid_bytes = marker.read_bytes()
    valid_marker = json.loads(valid_bytes)
    missing_field = dict(valid_marker)
    missing_field.pop("study_id")

    invalid_values = [
        b"not json",
        json.dumps(
            {
                **valid_marker,
                "model_run_id": "0" * 64,
            }
        ).encode(),
        json.dumps(
            {
                **valid_marker,
                "extra": True,
            }
        ).encode(),
        json.dumps(missing_field).encode(),
        json.dumps({**valid_marker, "store_version": "wrong"}).encode(),
        json.dumps({**valid_marker, "finalized_at": "not-a-time"}).encode(),
    ]
    for value in invalid_values:
        marker.write_bytes(value)
        bad_args = _runner_arguments(fixture)
        bad_args["load_runtime"] = lambda **_kwargs: pytest.fail(
            "invalid finalized marker loaded model"
        )
        with pytest.raises(RuntimeError, match="finalized|marker"):
            run_part1_shard(**bad_args)

    marker.unlink()
    marker.mkdir()
    with pytest.raises(RuntimeError, match="finalized|marker|regular"):
        run_part1_shard(**_runner_arguments(fixture))
    marker.rmdir()
    target = tmp_path / "marker-target"
    target.write_bytes(valid_bytes)
    marker.symlink_to(target)
    with pytest.raises(RuntimeError, match="finalized|marker|symlink"):
        run_part1_shard(**_runner_arguments(fixture))
    marker.unlink()
    marker.write_bytes(valid_bytes)
    final_args = _runner_arguments(fixture)
    final_args["load_runtime"] = lambda **_kwargs: pytest.fail(
        "valid finalized marker loaded model"
    )
    assert run_part1_shard(**final_args)["status"] == "already_finalized"

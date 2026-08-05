"""Production shard selection and orchestration tests (synthetic execution only)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from part1_checkpoints import (
    CheckpointGenerationCapture,
    build_checkpoint_terminal_result,
)
from part1_contract import FIXED_SUBJECTS
from part1_generation import NaturalGenerationCapture, build_natural_terminal_result
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
    with pytest.raises(ValueError, match="smoke.*production|separate"):
        run_part1_shard(**args)

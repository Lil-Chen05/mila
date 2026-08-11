"""Read-only bounded-smoke coverage tests using real temporary shard stores."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from part1_checkpoints import build_checkpoint_probe_plans
from part1_contract import derive_generation_seed
from part1_generation import build_natural_infrastructure_failure_result
from part1_model_run import build_smoke_model_run_manifest
from part1_runtime import LockMetadata, LockedShardSession, WorkSpec
from part1_store import Part1ShardStore
from run_part1_shard import select_shard_work
from run_part1_smoke import _run_work_lifecycle, select_smoke_work
from test_part1_model_run import preflight as synthetic_preflight
from test_run_part1_shard import _fake_checkpoint, _fake_natural, production_fixture


def fingerprint_tree(root: Path) -> tuple[tuple[str, str, str | int, int | None], ...]:
    """Capture every source path/type/content state without following symlinks."""

    entries: list[tuple[str, str, str | int, int | None]] = []
    for path in sorted((root, *root.rglob("*")), key=lambda item: item.as_posix()):
        relative = "." if path == root else path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if path.is_symlink():
            entries.append((relative, "symlink", os.readlink(path), None))
        elif path.is_dir():
            entries.append((relative, "directory", 0, None))
        elif path.is_file():
            data = path.read_bytes()
            entries.append((relative, "file", hashlib.sha256(data).hexdigest(), len(data)))
        else:
            entries.append((relative, "other", mode, None))
    return tuple(entries)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def smoke_fixture(tmp_path: Path, *, scope: str, terminal_failure: bool = False) -> dict:
    fixture = production_fixture(tmp_path)
    repository = fixture["repository"]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, text=True, capture_output=True
    ).stdout.strip()
    preflight = synthetic_preflight()
    preflight.update({field: fixture["bundle"].study_manifest[field] for field in ("study_id", "study_manifest_hash", "question_manifest_hash")})
    preflight["environment_versions"]["uv_lock_sha256"] = hashlib.sha256((repository / "uv.lock").read_bytes()).hexdigest()
    manifest = build_smoke_model_run_manifest(
        study_manifest=fixture["bundle"].study_manifest,
        preflight_report=preflight,
        execution_scope=scope,
        base_git_commit=head,
        diff_hash="0" * 64,
    )
    manifest_path = repository / "results/part1-smoke/model-runs" / scope / "model_run_manifest.json"
    _write_json(manifest_path, manifest)
    shard_root = (
        repository / "results/part1-smoke/phase3_smoke" / manifest["model_run_id"] / "raw_shards/shard-000"
        if scope == "phase3_smoke"
        else repository / "results/part1-smoke" / scope / manifest["model_run_id"] / "shard-000"
    )
    selected = (
        select_shard_work(fixture["bundle"].records, shard_index=0, shard_count=500)
        if scope == "phase3_smoke"
        else select_smoke_work(fixture["bundle"].records, execution_scope=scope)
    )
    owner = LockMetadata(
        lock_id="1" * 32, study_id=manifest["study_id"], model_run_id=manifest["model_run_id"],
        shard_id="shard-000", hostname="test-host", pid=123, slurm_job_id=None,
        slurm_array_task_id=None, acquired_at="2026-08-11T00:00:00Z",
    )
    with LockedShardSession.acquire(shard_root, owner=owner, model_run_manifest_hash=manifest["model_run_manifest_hash"]) as session:
        for question, run_id in selected:
            seed = derive_generation_seed(base_seed=manifest["base_generation_seed"], canonical_model_identity=manifest["canonical_model_identity"], question_id=question["question_id"], run_id=run_id)
            work = WorkSpec.natural(manifest["study_id"], manifest["model_run_id"], manifest["model_run_manifest_hash"], question["question_id"], run_id, seed=seed)
            identity = {"study_id": manifest["study_id"], "model_run_id": manifest["model_run_id"], "model_run_manifest_hash": manifest["model_run_manifest_hash"], "question_manifest_hash": manifest["question_manifest_hash"], "question_id": question["question_id"], "sample_index": question["sample_index"], "subject": question["subject"], "gold_letter": question["gold_letter"]}

            def natural(attempt: int, *, question=question, run_id=run_id, seed=seed):
                if terminal_failure:
                    raise ValueError("synthetic terminal failure")
                return _fake_natural(model=object(), tokenizer=object(), question=question, run_id=run_id, seed=seed, attempt_number=attempt, model_manifest=manifest, token_contract=preflight["token_contract"])

            def natural_failure(attempt: int, category: str, reference: str, details: dict) -> dict:
                return build_natural_infrastructure_failure_result(
                    identity=identity, run_id=run_id, generation_seed=seed, terminal_attempt_number=attempt,
                    prompt_hash=manifest["prompt_hash"], failure_category=category,
                    infrastructure_failure_reference=reference, error_details=details,
                )

            status, _ = _run_work_lifecycle(session, work=work, execute_attempt=natural, build_terminal_failure=natural_failure, sleep=lambda _seconds: None)
            if status != "completed":
                continue
            parent = session.store.build_index().natural_terminal_by_key[(manifest["study_id"], manifest["model_run_id"], question["question_id"], run_id)]
            for plan in build_checkpoint_probe_plans(parent, inducer_token_ids=preflight["token_contract"]["inducer_token_ids"], inducer_version=manifest["inducer_version"]):
                checkpoint_work = WorkSpec.checkpoint(manifest["study_id"], manifest["model_run_id"], manifest["model_run_manifest_hash"], question["question_id"], run_id, plan.checkpoint_id, seed=seed)
                _run_work_lifecycle(
                    session, work=checkpoint_work,
                    execute_attempt=lambda attempt, plan=plan, parent=parent, question=question: _fake_checkpoint(model=object(), tokenizer=object(), parent=parent, plan=plan, token_contract=preflight["token_contract"], gold_letter=question["gold_letter"], attempt_number=attempt),
                    build_terminal_failure=lambda *_args: pytest.fail("unexpected fallback"), sleep=lambda _seconds: None,
                )
        session.store.finalize()
    return {**fixture, "manifest": manifest, "manifest_path": manifest_path, "shard_root": shard_root, "selected": selected}


@pytest.mark.parametrize(("scope", "expected_natural", "expected_checkpoint"), [("smoke_a", 10, 110), ("smoke_b", 5, 55), ("phase3_smoke", 10, 110)])
def test_valid_bounded_smoke_is_complete_and_byte_identical(tmp_path: Path, scope: str, expected_natural: int, expected_checkpoint: int) -> None:
    from part1_smoke_coverage import build_smoke_coverage_report

    fixture = smoke_fixture(tmp_path, scope=scope)
    before = fingerprint_tree(fixture["shard_root"])
    report = build_smoke_coverage_report(repository_root=fixture["repository"], model_run_manifest_path=fixture["manifest_path"], shard_root=fixture["shard_root"])

    assert report["is_valid"] is True
    assert report["coverage_complete"] is True
    assert report["summary"]["natural_partition"]["complete"] == expected_natural
    assert report["summary"]["checkpoint_partition"]["complete"] == expected_checkpoint
    assert report["summary"]["natural_run_ids"] == (list(range(10)) if scope != "smoke_b" else [0])
    assert report["summary"]["checkpoint_indices"] == list(range(11))
    assert fingerprint_tree(fixture["shard_root"]) == before


@pytest.mark.parametrize("mutation", ["wrong_path", "manifest_drift", "unfinalized", "active_lock", "pending_takeover", "invalid_tail", "duplicate_natural", "duplicate_checkpoint", "missing_natural", "missing_checkpoint", "unexpected_natural", "unexpected_checkpoint", "lifecycle"])
def test_smoke_validator_fails_closed_for_one_bad_condition(tmp_path: Path, mutation: str) -> None:
    from part1_smoke_coverage import build_smoke_coverage_report

    fixture = smoke_fixture(tmp_path, scope="smoke_a")
    shard_root = fixture["shard_root"]
    if mutation == "wrong_path":
        requested_shard = shard_root.parent / "other"
    else:
        requested_shard = shard_root
        if mutation == "manifest_drift":
            manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
            manifest["model_run_manifest_hash"] = "f" * 64
            _write_json(fixture["manifest_path"], manifest)
        elif mutation == "unfinalized":
            (shard_root / ".finalized").unlink()
        elif mutation == "active_lock":
            (shard_root / ".writer.lock").write_text("active\n", encoding="utf-8")
        elif mutation == "pending_takeover":
            (shard_root / ".writer-lock-recovery.claim").write_text("pending\n", encoding="utf-8")
        elif mutation == "invalid_tail":
            with (shard_root / "natural_results.jsonl").open("ab") as handle:
                handle.write(b'{"invalid"')
        elif mutation == "duplicate_natural":
            path = shard_root / "natural_results.jsonl"
            path.write_bytes(path.read_bytes() + path.read_bytes().splitlines(keepends=True)[0])
        elif mutation == "duplicate_checkpoint":
            path = shard_root / "checkpoint_results.jsonl"
            path.write_bytes(path.read_bytes() + path.read_bytes().splitlines(keepends=True)[0])
        elif mutation == "missing_natural":
            path = shard_root / "natural_results.jsonl"
            path.write_bytes(b"".join(path.read_bytes().splitlines(keepends=True)[1:]))
        elif mutation == "missing_checkpoint":
            path = shard_root / "checkpoint_results.jsonl"
            path.write_bytes(b"".join(path.read_bytes().splitlines(keepends=True)[1:]))
        elif mutation == "unexpected_natural":
            path = shard_root / "natural_results.jsonl"
            record = json.loads(path.read_bytes().splitlines()[0])
            record["run_id"] = 99
            path.write_bytes(path.read_bytes() + json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        elif mutation == "unexpected_checkpoint":
            path = shard_root / "checkpoint_results.jsonl"
            record = json.loads(path.read_bytes().splitlines()[0])
            record["checkpoint_id"] = "cp-99"
            path.write_bytes(path.read_bytes() + json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        elif mutation == "lifecycle":
            (shard_root / "audit_events.jsonl").write_bytes(b"")
    before = fingerprint_tree(shard_root)
    with pytest.raises((RuntimeError, ValueError), match="(?:canonical|finalized|lock|tail|failed|coverage|source|identity|checkpoint|natural)"):
        build_smoke_coverage_report(repository_root=fixture["repository"], model_run_manifest_path=fixture["manifest_path"], shard_root=requested_shard)
    assert fingerprint_tree(shard_root) == before


def test_terminal_natural_failure_has_eleven_ineligible_checkpoints(tmp_path: Path) -> None:
    from part1_smoke_coverage import build_smoke_coverage_report

    fixture = smoke_fixture(tmp_path, scope="smoke_a", terminal_failure=True)
    report = build_smoke_coverage_report(repository_root=fixture["repository"], model_run_manifest_path=fixture["manifest_path"], shard_root=fixture["shard_root"])

    assert report["is_valid"] is True
    assert report["coverage_complete"] is True
    assert report["paper_analysis_ready"] is False
    assert report["summary"]["natural_partition"]["terminal_infrastructure_failure"] == 10
    assert report["summary"]["checkpoint_partition"]["ineligible"] == 110


def test_later_validator_only_git_head_does_not_invalidate_smoke_provenance(tmp_path: Path) -> None:
    from part1_smoke_coverage import build_smoke_coverage_report

    fixture = smoke_fixture(tmp_path, scope="smoke_a")
    later = fixture["repository"] / "validator-only-note.txt"
    later.write_text("later read-only validator documentation\n", encoding="utf-8")
    subprocess.run(["git", "add", later.name], cwd=fixture["repository"], check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "later validator docs"], cwd=fixture["repository"], check=True)

    report = build_smoke_coverage_report(repository_root=fixture["repository"], model_run_manifest_path=fixture["manifest_path"], shard_root=fixture["shard_root"])

    assert report["is_valid"] is True


def test_smoke_validator_detects_source_mutation_at_final_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import part1_smoke_coverage

    fixture = smoke_fixture(tmp_path, scope="smoke_a")
    original = part1_smoke_coverage._partition
    mutated = False

    def mutate_once(*args, **kwargs):
        nonlocal mutated
        result = original(*args, **kwargs)
        if not mutated:
            mutated = True
            with (fixture["shard_root"] / "audit_events.jsonl").open("ab") as handle:
                handle.write(b"\n")
        return result

    monkeypatch.setattr(part1_smoke_coverage, "_partition", mutate_once)
    with pytest.raises(RuntimeError, match="inputs changed"):
        part1_smoke_coverage.build_smoke_coverage_report(repository_root=fixture["repository"], model_run_manifest_path=fixture["manifest_path"], shard_root=fixture["shard_root"])

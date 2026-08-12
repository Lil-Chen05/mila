"""Tests for the acceptance-to-production bootstrap submission."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest


def test_bootstrap_submits_gate_with_exact_afterok_and_records_both_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import submit_part1_unattended as module

    commit = "b" * 40
    monkeypatch.setattr(module, "_git_state", lambda _root: (commit, (), ("notes.md",)))
    calls: list[list[str]] = []
    outputs = iter(("4101\n", "4102\n"))

    def fake_run(arguments: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, stdout=next(outputs), stderr="")

    receipt = module.submit_unattended(
        repository_root=tmp_path,
        run_command=fake_run,
        submitted_at="2026-08-12T00:00:00Z",
    )

    relative_receipt = f"results/part1-submission/{commit}/bootstrap_receipt.json"
    assert receipt["status"] == "submitted"
    assert receipt["jobs"] == {"acceptance": "4101", "production_gate": "4102"}
    assert receipt["gate_dependency"] == "afterok:4101"
    assert receipt["unrelated_untracked_paths"] == ["notes.md"]
    assert calls == [
        [
            "sbatch",
            "--parsable",
            f"--job-name=part1-full-acceptance-{commit[:12]}",
            f"--comment=part1:{commit}:full_acceptance",
            "jobs/part1_full_acceptance.sh",
        ],
        [
            "sbatch",
            "--parsable",
            f"--job-name=part1-production-gate-{commit[:12]}",
            f"--comment=part1:{commit}:production_gate",
            "--dependency=afterok:4101",
            f"--export=ALL,ACCEPTANCE_JOB_ID=4101,BOOTSTRAP_RECEIPT={relative_receipt}",
            "jobs/part1_production_gate.sh",
        ],
    ]
    path = tmp_path / relative_receipt
    assert json.loads(path.read_text(encoding="utf-8")) == receipt


def test_bootstrap_refuses_dirty_or_repeated_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import submit_part1_unattended as module

    monkeypatch.setattr(module, "_git_state", lambda _root: ("b" * 40, ("scripts/x.py",), ()))
    with pytest.raises(RuntimeError, match="clean tracked worktree"):
        module.submit_unattended(
            repository_root=tmp_path,
            run_command=lambda *_args, **_kwargs: pytest.fail("sbatch must not run"),
        )

    monkeypatch.setattr(module, "_git_state", lambda _root: ("b" * 40, (), ()))
    first_outputs = iter(("4201\n", "4202\n"))
    module.submit_unattended(
        repository_root=tmp_path,
        run_command=lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 0, stdout=next(first_outputs), stderr=""
        ),
    )
    with pytest.raises(RuntimeError, match="receipt already exists"):
        module.submit_unattended(
            repository_root=tmp_path,
            run_command=lambda *_args, **_kwargs: pytest.fail("duplicate sbatch"),
        )


def test_gate_rejects_incompatible_bootstrap_receipt(tmp_path: Path) -> None:
    from submit_part1_production_chain import submit_production_chain

    commit = "b" * 40
    bootstrap = (
        tmp_path
        / "results/part1-submission"
        / commit
        / "bootstrap_receipt.json"
    )
    bootstrap.parent.mkdir(parents=True)
    bootstrap.write_text(
        json.dumps(
            {
                "status": "submitted",
                "gate_dependency": "afterok:9999",
                "final_production_git_commit": commit,
                "jobs": {"acceptance": "1001", "production_gate": "1002"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="bootstrap receipt is incompatible"):
        submit_production_chain(
            model_manifest={
                "model_run_id": "a" * 64,
                "final_production_git_commit": commit,
            },
            acceptance_job_id="1001",
            gate_job_id="1002",
            bootstrap_receipt_path=bootstrap,
            repository_root=tmp_path,
        )

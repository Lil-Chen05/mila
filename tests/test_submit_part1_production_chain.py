"""Fail-closed tests for unattended Part 1 production submission."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest


def _manifest() -> dict[str, str]:
    return {
        "model_run_id": "a" * 64,
        "final_production_git_commit": "b" * 40,
    }


def _bootstrap(tmp_path: Path) -> Path:
    path = tmp_path / "results/part1-submission" / ("b" * 40) / "bootstrap_receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "part1-submission-bootstrap-v2",
                "acceptance_mode": "focused_readiness_v1",
                "status": "submitted",
                "gate_dependency": "afterok:1001",
                "final_production_git_commit": "b" * 40,
                "jobs": {"acceptance": "1001", "production_gate": "1002"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_submission_chain_rejects_legacy_full_shape_bootstrap(
    tmp_path: Path,
) -> None:
    from submit_part1_production_chain import submit_production_chain

    manifest = _manifest()
    bootstrap = _bootstrap(tmp_path)
    payload = json.loads(bootstrap.read_text(encoding="utf-8"))
    payload["schema_version"] = "part1-submission-bootstrap-v1"
    payload.pop("acceptance_mode")
    bootstrap.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="bootstrap receipt is incompatible"):
        submit_production_chain(
            model_manifest=manifest,
            repository_root=tmp_path,
            run_command=lambda *_args, **_kwargs: pytest.fail("sbatch must not run"),
            acceptance_job_id="1001",
            gate_job_id="1002",
            bootstrap_receipt_path=bootstrap,
        )


def test_submission_chain_records_exact_jobs_and_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import submit_part1_production_chain as module

    manifest = _manifest()
    bootstrap = _bootstrap(tmp_path)
    (tmp_path / "results/part1" / manifest["model_run_id"]).mkdir(parents=True)
    monkeypatch.setattr(
        module,
        "build_launch_plan",
        lambda **_kwargs: {
            "is_valid": True,
            "command_executed": False,
            "model_run_id": manifest["model_run_id"],
            "resources": {"initial_concurrency": 16},
        },
    )
    calls: list[list[str]] = []
    job_ids = iter(("1101\n", "1102\n", "1103\n", "1104\n"))

    def fake_run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, stdout=next(job_ids), stderr="")

    receipt = module.submit_production_chain(
        model_manifest=manifest,
        repository_root=tmp_path,
        run_command=fake_run,
        submitted_at="2026-08-12T00:00:00Z",
        acceptance_job_id="1001",
        gate_job_id="1002",
        bootstrap_receipt_path=bootstrap,
    )

    assert receipt["status"] == "submitted"
    assert receipt["acceptance_mode"] == "focused_readiness_v1"
    assert receipt["jobs"] == {
        "acceptance": "1001",
        "production_gate": "1002",
        "generation": "1101",
        "validation": "1102",
        "merge": "1103",
        "analysis": "1104",
    }
    assert calls == [
        [
            "sbatch",
            "--parsable",
            f"--job-name=part1-generation-{manifest['model_run_id'][:12]}",
            f"--comment=part1:{manifest['model_run_id']}:generation",
            f"--export=ALL,MODEL_RUN_ID={manifest['model_run_id']}",
            "--array=0-499%16",
            "jobs/part1_generate_array.sh",
        ],
        [
            "sbatch",
            "--parsable",
            f"--job-name=part1-validation-{manifest['model_run_id'][:12]}",
            f"--comment=part1:{manifest['model_run_id']}:validation",
            "--dependency=afterany:1101",
            f"--export=ALL,MODEL_RUN_ID={manifest['model_run_id']}",
            "jobs/part1_validate.sh",
        ],
        [
            "sbatch",
            "--parsable",
            f"--job-name=part1-merge-{manifest['model_run_id'][:12]}",
            f"--comment=part1:{manifest['model_run_id']}:merge",
            "--dependency=afterok:1102",
            f"--export=ALL,MODEL_RUN_ID={manifest['model_run_id']}",
            "jobs/part1_merge.sh",
        ],
        [
            "sbatch",
            "--parsable",
            f"--job-name=part1-analysis-{manifest['model_run_id'][:12]}",
            f"--comment=part1:{manifest['model_run_id']}:analysis",
            "--dependency=afterok:1103",
            f"--export=ALL,MODEL_RUN_ID={manifest['model_run_id']},BOOTSTRAP_REPLICATES=5000",
            "jobs/part1_analyze.sh",
        ],
    ]
    path = tmp_path / "results/part1" / manifest["model_run_id"] / "submission_receipt.json"
    assert json.loads(path.read_text(encoding="utf-8")) == receipt


def test_submission_chain_refuses_existing_receipt_before_sbatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import submit_part1_production_chain as module

    manifest = _manifest()
    bootstrap = _bootstrap(tmp_path)
    receipt = tmp_path / "results/part1" / manifest["model_run_id"] / "submission_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "build_launch_plan",
        lambda **_kwargs: {
            "is_valid": True,
            "command_executed": False,
            "model_run_id": manifest["model_run_id"],
            "resources": {"initial_concurrency": 16},
        },
    )

    with pytest.raises(RuntimeError, match="receipt already exists"):
        module.submit_production_chain(
            model_manifest=manifest,
            repository_root=tmp_path,
            run_command=lambda *_args, **_kwargs: pytest.fail("sbatch must not run"),
            acceptance_job_id="1001",
            gate_job_id="1002",
            bootstrap_receipt_path=bootstrap,
        )


def test_submission_failure_retains_partial_job_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import submit_part1_production_chain as module

    manifest = _manifest()
    bootstrap = _bootstrap(tmp_path)
    (tmp_path / "results/part1" / manifest["model_run_id"]).mkdir(parents=True)
    monkeypatch.setattr(
        module,
        "build_launch_plan",
        lambda **_kwargs: {
            "is_valid": True,
            "command_executed": False,
            "model_run_id": manifest["model_run_id"],
            "resources": {"initial_concurrency": 16},
        },
    )
    count = 0

    def fake_run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal count
        count += 1
        if count == 1:
            return subprocess.CompletedProcess(arguments, 0, stdout="2201\n", stderr="")
        raise subprocess.CalledProcessError(1, arguments, stderr="submission rejected")

    with pytest.raises(subprocess.CalledProcessError):
        module.submit_production_chain(
            model_manifest=manifest,
            repository_root=tmp_path,
            run_command=fake_run,
            submitted_at="2026-08-12T00:00:00Z",
            acceptance_job_id="1001",
            gate_job_id="1002",
            bootstrap_receipt_path=bootstrap,
        )

    path = tmp_path / "results/part1" / manifest["model_run_id"] / "submission_receipt.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["status"] == "submission_failed"
    assert saved["jobs"] == {
        "acceptance": "1001",
        "production_gate": "1002",
        "generation": "2201",
    }
    assert saved["failed_stage"] == "validation"
    assert saved["in_flight"]["stage"] == "validation"


@pytest.mark.parametrize("output", ["123;bad cluster\n", "123\nextra\n", "abc\n"])
def test_job_id_parser_rejects_malformed_output(output: str) -> None:
    from submit_part1_production_chain import _parse_job_id

    with pytest.raises(RuntimeError, match="invalid job ID"):
        _parse_job_id(output)


def test_concurrent_invocations_submit_only_one_generation_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import submit_part1_production_chain as module

    manifest = _manifest()
    bootstrap = _bootstrap(tmp_path)
    (tmp_path / "results/part1" / manifest["model_run_id"]).mkdir(parents=True)
    monkeypatch.setattr(
        module,
        "build_launch_plan",
        lambda **_kwargs: {
            "is_valid": True,
            "command_executed": False,
            "model_run_id": manifest["model_run_id"],
            "resources": {"initial_concurrency": 16},
        },
    )
    start = threading.Barrier(2)
    calls: list[list[str]] = []
    lock = threading.Lock()

    def fake_run(arguments: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        with lock:
            calls.append(arguments)
            job_id = str(3000 + len(calls))
        return subprocess.CompletedProcess(arguments, 0, stdout=job_id + "\n", stderr="")

    def invoke() -> object:
        start.wait()
        try:
            return module.submit_production_chain(
                model_manifest=manifest,
                repository_root=tmp_path,
                run_command=fake_run,
                acceptance_job_id="1001",
                gate_job_id="1002",
                bootstrap_receipt_path=bootstrap,
            )
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: invoke(), range(2)))

    assert sum(isinstance(outcome, dict) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, RuntimeError) for outcome in outcomes) == 1
    generation_calls = [arguments for arguments in calls if "--array=0-499%16" in arguments]
    assert len(generation_calls) == 1

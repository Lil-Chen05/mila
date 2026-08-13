"""Login-safe production launch-plan and Part 1 job contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from part1_storage_estimate import estimate_part1_storage
from test_run_part1_shard import production_fixture


def test_launch_plan_reports_exact_unrun_workload_and_storage(tmp_path: Path) -> None:
    from part1_launch_plan import build_launch_plan

    fixture = production_fixture(tmp_path)
    plan = build_launch_plan(
        model_manifest=fixture["manifest"],
        repository_root=fixture["repository"],
    )

    assert plan["is_valid"] is True
    assert plan["command_executed"] is False
    assert plan["workload"] == {
        "questions": 500,
        "natural_runs": 5_000,
        "checkpoint_keys": 55_000,
        "shards": 500,
        "questions_per_shard": 1,
        "natural_runs_per_task": 10,
    }
    assert plan["resources"] == {
        "gpu": "l40s:1",
        "wall_time": "12:00:00",
        "initial_concurrency": 16,
    }
    assert plan["storage_estimates"]["expected_2048_tokens"] == (
        estimate_part1_storage(expected_generated_tokens=2048)
    )
    assert plan["storage_estimates"]["cap_8192_tokens"] == (
        estimate_part1_storage(expected_generated_tokens=8192)
    )
    assert plan["submission_command"] == (
        "sbatch --export=ALL,MODEL_RUN_ID="
        f"{fixture['manifest']['model_run_id']} "
        "--array=0-499%16 jobs/part1_generate_array.sh"
    )


def test_launch_plan_refuses_dirty_commit_or_dependency_lock(tmp_path: Path) -> None:
    from part1_launch_plan import build_launch_plan

    fixture = production_fixture(tmp_path)
    (fixture["repository"] / "uv.lock").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean|lock"):
        build_launch_plan(
            model_manifest=fixture["manifest"],
            repository_root=fixture["repository"],
        )


def test_part1_jobs_have_exact_resources_uv_resolution_and_future_clis() -> None:
    root = Path(__file__).resolve().parents[1]
    jobs = {
        name: (root / "jobs" / name).read_text(encoding="utf-8")
        for name in (
            "materialize_part1_mmlu.sh",
            "part1_smollm3_preflight.sh",
            "part1_reproducibility.sh",
            "part1_smoke_a.sh",
            "part1_smoke_b.sh",
            "part1_generate_array.sh",
            "part1_phase3_smoke.sh",
            "part1_validate.sh",
            "part1_merge.sh",
            "part1_analyze.sh",
        )
    }
    for content in jobs.values():
        assert "command -v uv" in content
        assert '$HOME/.local/bin/uv' in content
        assert '[[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]' in content
        assert '[[ ! -x "$UV_BIN" ]]' in content
        assert 'srun "$UV_BIN" run python' in content
    for name in ("part1_generate_array.sh", "part1_phase3_smoke.sh"):
        assert "#SBATCH --gpus-per-task=l40s:1" in jobs[name]
        assert "#SBATCH --cpus-per-task=4" in jobs[name]
        assert "#SBATCH --mem=32G" in jobs[name]
        assert "#SBATCH --time=12:00:00" in jobs[name]
        assert 'export HF_HOME="$SCRATCH/hf_cache"' in jobs[name]
    assert '${MODEL_RUN_ID:?' in jobs["part1_generate_array.sh"]
    assert '${SLURM_ARRAY_TASK_ID:?' in jobs["part1_generate_array.sh"]
    assert "sbatch --array" not in jobs["part1_generate_array.sh"]
    assert "--shard-index 0" in jobs["part1_phase3_smoke.sh"]
    for name, cli, wall_time in (
        ("part1_validate.sh", "scripts/validate_part1_results.py", "12:00:00"),
        ("part1_merge.sh", "scripts/merge_part1_results.py", "1-00:00:00"),
        ("part1_analyze.sh", "scripts/analyze_part1.py", "1-12:00:00"),
    ):
        assert "#SBATCH --gpus-per-task" not in jobs[name]
        assert "#SBATCH --cpus-per-task=4" in jobs[name]
        assert f"#SBATCH --time={wall_time}" in jobs[name]
        assert cli in jobs[name]


def test_launch_readiness_job_is_cpu_only_and_excludes_full_shape_marker() -> None:
    root = Path(__file__).resolve().parents[1]
    content = (root / "jobs/part1_launch_readiness.sh").read_text(encoding="utf-8")

    assert "#SBATCH --gpus-per-task" not in content
    assert "#SBATCH --cpus-per-task=4" in content
    assert "#SBATCH --mem=32G" in content
    assert "#SBATCH --time=1:00:00" in content
    assert "#SBATCH --output=logs/part1-launch-readiness-%j.out" in content
    assert 'srun "$UV_BIN" run pytest' in content
    assert '-m "not part1_full_acceptance"' in content
    assert "tests/test_submit_part1_unattended.py" in content
    assert "tests/test_submit_part1_production_chain.py" in content
    assert "tests/test_part1_launch_plan.py" in content
    assert "--basetemp" not in content


def test_production_gate_is_cpu_only_and_submits_only_after_manifest_creation() -> None:
    root = Path(__file__).resolve().parents[1]
    content = (root / "jobs/part1_production_gate.sh").read_text(encoding="utf-8")

    assert "#SBATCH --gpus-per-task" not in content
    assert "#SBATCH --cpus-per-task=1" in content
    assert "#SBATCH --mem=4G" in content
    assert "#SBATCH --time=00:30:00" in content
    assert '${ACCEPTANCE_JOB_ID:?' in content
    assert '${BOOTSTRAP_RECEIPT:?' in content
    assert '${SLURM_JOB_ID:?' in content
    assert content.count("scripts/validate_part1_smoke_results.py") == 1
    assert 'SCOPE="phase3_smoke"' in content
    assert "for SCOPE in smoke_a smoke_b phase3_smoke" not in content
    smoke_position = content.index("scripts/validate_part1_smoke_results.py")
    create_position = content.index("scripts/create_part1_model_run_manifest.py")
    plan_position = content.index("scripts/part1_launch_plan.py")
    submit_position = content.index("scripts/submit_part1_production_chain.py")
    assert smoke_position < create_position < plan_position < submit_position
    assert '--acceptance-job-id "$ACCEPTANCE_JOB_ID"' in content
    assert '--gate-job-id "$SLURM_JOB_ID"' in content
    assert '--bootstrap-receipt "$BOOTSTRAP_RECEIPT"' in content

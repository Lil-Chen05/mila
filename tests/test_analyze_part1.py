"""CLI contract tests for the production Part 1 analysis entry point."""

from __future__ import annotations

import json
from pathlib import Path


def test_cli_defaults_to_final_and_prints_one_compact_success_object(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import analyze_part1

    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return Path("results/part1/run/analysis/final-r5000"), {
            "analysis_id": "a" * 64,
            "analysis_manifest_hash": "b" * 64,
            "merge_id": "c" * 64,
            "merge_manifest_hash": "d" * 64,
            "coverage_report_id": "e" * 64,
            "bootstrap_replicates": 5000,
            "bootstrap_mode": "final",
            "paper_analysis_ready": True,
        }

    monkeypatch.setattr(analyze_part1, "analyze_production", run)
    manifest = tmp_path / "model_run_manifest.json"
    assert analyze_part1.main(["--model-run-manifest", str(manifest)]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out.count("\n") == 1
    payload = json.loads(output.out)
    assert payload == {
        "status": "published",
        "mode": "final",
        "analysis_directory": "results/part1/run/analysis/final-r5000",
        "analysis_id": "a" * 64,
        "analysis_manifest_hash": "b" * 64,
        "merge_id": "c" * 64,
        "merge_manifest_hash": "d" * 64,
        "coverage_report_id": "e" * 64,
        "bootstrap_replicates": 5000,
        "paper_analysis_ready": True,
    }
    assert captured["bootstrap_replicates"] == 5000
    assert captured["repository_root"] == Path(__file__).resolve().parents[1]


def test_cli_accepts_only_1000_or_5000_and_never_exposes_fixture_mode(
    monkeypatch, capsys
) -> None:
    import analyze_part1

    monkeypatch.setattr(
        analyze_part1,
        "analyze_production",
        lambda **kwargs: (
            Path("results/part1/run/analysis/development-r1000"),
            {
                "analysis_id": "a" * 64,
                "analysis_manifest_hash": "b" * 64,
                "merge_id": "c" * 64,
                "merge_manifest_hash": "d" * 64,
                "coverage_report_id": "e" * 64,
                "bootstrap_replicates": kwargs["bootstrap_replicates"],
                "bootstrap_mode": "development",
                "paper_analysis_ready": True,
            },
        ),
    )
    assert analyze_part1.main(
        ["--model-run-manifest", "manifest.json", "--bootstrap-replicates", "1000"]
    ) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "development"
    parser = analyze_part1.build_parser()
    assert "small-fixture" not in parser.format_help()
    for value in ("1", "999", "5001"):
        try:
            parser.parse_args(
                ["--model-run-manifest", "manifest.json", "--bootstrap-replicates", value]
            )
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"invalid bootstrap count accepted: {value}")


def test_cli_failure_is_one_compact_stderr_object_and_never_claims_success(
    monkeypatch, capsys
) -> None:
    import analyze_part1

    def fail(**_kwargs):
        raise ValueError("coverage is not paper ready")

    monkeypatch.setattr(analyze_part1, "analyze_production", fail)
    assert analyze_part1.main(["--model-run-manifest", "manifest.json"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.count("\n") == 1
    assert json.loads(output.err) == {
        "status": "error",
        "error_type": "ValueError",
        "message": "coverage is not paper ready",
    }


def test_cli_argument_failure_also_uses_the_compact_json_error_contract(capsys) -> None:
    import analyze_part1

    assert analyze_part1.main(
        ["--model-run-manifest", "manifest.json", "--bootstrap-replicates", "7"]
    ) == 2
    output = capsys.readouterr()
    assert output.out == ""
    payload = json.loads(output.err)
    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"
    assert "invalid choice" in payload["message"]

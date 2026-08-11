"""JSON-only CLI coverage for the bounded read-only smoke validator."""

from __future__ import annotations

import json
from pathlib import Path

from test_part1_smoke_coverage import fingerprint_tree, smoke_fixture


def test_cli_prints_compact_success_json_without_writing(tmp_path: Path, capsys) -> None:
    from validate_part1_smoke_results import main

    fixture = smoke_fixture(tmp_path, scope="smoke_b")
    before = fingerprint_tree(fixture["shard_root"])
    result = main(["--repository-root", str(fixture["repository"]), "--model-run-manifest", str(fixture["manifest_path"]), "--shard-root", str(fixture["shard_root"])])

    assert result == 0
    report = json.loads(capsys.readouterr().out)
    assert report["is_valid"] is True
    assert report["mutation_performed"] is False
    assert fingerprint_tree(fixture["shard_root"]) == before


def test_cli_reports_failure_to_stderr_and_returns_two(tmp_path: Path, capsys) -> None:
    from validate_part1_smoke_results import main

    fixture = smoke_fixture(tmp_path, scope="smoke_a")
    fixture["shard_root"].joinpath(".finalized").unlink()
    result = main(["--repository-root", str(fixture["repository"]), "--model-run-manifest", str(fixture["manifest_path"]), "--shard-root", str(fixture["shard_root"])])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    report = json.loads(captured.err)
    assert report["status"] == "failed"
    assert report["mutation_performed"] is False

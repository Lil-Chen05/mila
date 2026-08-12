"""Read-only bounded-smoke coverage tests using real temporary shard stores."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Callable

import pytest

from part1_checkpoints import build_checkpoint_probe_plans
from part1_contract import audit_event_id, derive_generation_seed
from part1_generation import build_natural_infrastructure_failure_result
from part1_model_run import build_smoke_model_run_manifest
from run_part1_shard import select_shard_work
from run_part1_smoke import select_smoke_work
from part1_store_fixtures import attempt_event
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


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            for value in values
        )
    )


def _mutate_first_jsonl(path: Path, mutate: Callable[[dict], None]) -> None:
    records = [json.loads(line) for line in path.read_bytes().splitlines()]
    mutate(records[0])
    _write_jsonl(path, records)


def _mutate_all_jsonl(path: Path, mutate: Callable[[dict], None]) -> None:
    records = [json.loads(line) for line in path.read_bytes().splitlines()]
    for record in records:
        mutate(record)
    _write_jsonl(path, records)


def _add_valid_recovery_evidence(fixture: dict) -> tuple[Path, Path]:
    shard_root = fixture["shard_root"]
    manifest = fixture["manifest"]
    stream = shard_root / "natural_results.jsonl"
    stream_bytes = stream.read_bytes()
    recovered = b'{"synthetic-invalid-tail"'
    recovered_hash = hashlib.sha256(recovered).hexdigest()
    quarantine_name = f"natural_results.{recovered_hash}.trailing-bytes.bin"
    event = {
        "schema_name": "part1_audit_event",
        "schema_version": "1.0.0",
        "event_id": audit_event_id(
            None,
            "trailing_line_recovered",
            0,
            study_id_value=manifest["study_id"],
            model_run_id_value=manifest["model_run_id"],
            shard_id="shard-000",
        ),
        "event_scope": "shard",
        "study_id": manifest["study_id"],
        "model_run_id": manifest["model_run_id"],
        "shard_id": "shard-000",
        "question_id": None,
        "run_id": None,
        "checkpoint_id": None,
        "attempt_id": None,
        "attempt_number": None,
        "event_sequence": 0,
        "event_type": "trailing_line_recovered",
        "event_timestamp": "2026-08-11T00:00:00Z",
        "execution_context": {"hostname": "test-host", "pid": 123},
        "outcome_category": "invalid_final_line",
        "error_details": {
            "stream": "natural_results",
            "recovered_byte_count": len(recovered),
            "recovered_bytes_sha256": recovered_hash,
            "quarantine_artifact": quarantine_name,
            "original_size": len(stream_bytes) + len(recovered),
            "valid_prefix_size": len(stream_bytes),
            "valid_prefix_sha256": hashlib.sha256(stream_bytes).hexdigest(),
        },
        "retry_classification": None,
        "retry_decision": None,
        "backoff_seconds": None,
        "related_lock_owner": None,
        "terminal_record_id": None,
        "operator_reason": None,
    }
    audit = shard_root / "audit_events.jsonl"
    audit.write_bytes(
        audit.read_bytes()
        + json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    journal = shard_root / "recovery_journal" / f"{event['event_id']}.json"
    journal.parent.mkdir()
    journal.write_bytes(json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    quarantine = shard_root / "quarantine" / quarantine_name
    quarantine.parent.mkdir()
    quarantine.write_bytes(recovered)
    return journal, quarantine


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
    natural_results: list[dict] = []
    checkpoint_results: list[dict] = []
    audit_events: list[dict] = []
    for question, run_id in selected:
        seed = derive_generation_seed(
            base_seed=manifest["base_generation_seed"],
            canonical_model_identity=manifest["canonical_model_identity"],
            question_id=question["question_id"],
            run_id=run_id,
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
        if terminal_failure:
            parent = build_natural_infrastructure_failure_result(
                identity=identity,
                run_id=run_id,
                generation_seed=seed,
                terminal_attempt_number=1,
                prompt_hash=manifest["prompt_hash"],
                failure_category="invalid_configuration",
                infrastructure_failure_reference=f"synthetic:natural:{run_id}",
                error_details={"category": "invalid_configuration", "synthetic": True},
            )
        else:
            parent = _fake_natural(
                model=object(), tokenizer=object(), question=question, run_id=run_id,
                seed=seed, attempt_number=1, model_manifest=manifest,
                token_contract=preflight["token_contract"],
            )
            # The smoke compatibility contract stores the immutable prompt
            # contract hash, while this generic fake helper hashes its literal
            # synthetic rendered prompt.
            parent["prompt_hash"] = manifest["prompt_hash"]
        natural_results.append(parent)
        audit_events.extend(
            (attempt_event(parent, "attempt_started", 0), attempt_event(parent, "attempt_completed", 1))
        )
        if parent["natural_execution_outcome"] != "complete":
            continue
        for plan in build_checkpoint_probe_plans(
            parent,
            inducer_token_ids=preflight["token_contract"]["inducer_token_ids"],
            inducer_version=manifest["inducer_version"],
        ):
            checkpoint = _fake_checkpoint(
                model=object(), tokenizer=object(), parent=parent, plan=plan,
                token_contract=preflight["token_contract"], gold_letter=question["gold_letter"],
                attempt_number=1,
            )
            checkpoint_results.append(checkpoint)
            audit_events.extend(
                (attempt_event(checkpoint, "attempt_started", 0), attempt_event(checkpoint, "attempt_completed", 1))
            )

    shard_root.mkdir(parents=True)
    _write_json(
        shard_root / ".shard-provenance.json",
        {
            "schema_name": "part1_shard_provenance",
            "schema_version": "1.0.0",
            "study_id": manifest["study_id"],
            "model_run_id": manifest["model_run_id"],
            "model_run_manifest_hash": manifest["model_run_manifest_hash"],
            "shard_id": "shard-000",
        },
    )
    _write_jsonl(shard_root / "natural_results.jsonl", natural_results)
    _write_jsonl(shard_root / "checkpoint_results.jsonl", checkpoint_results)
    _write_jsonl(shard_root / "audit_events.jsonl", audit_events)
    (shard_root / ".writer.guard").write_bytes(b"")
    _write_json(
        shard_root / ".finalized",
        {
            "store_version": "part1-store-v1",
            "shard_id": "shard-000",
            "study_id": manifest["study_id"],
            "model_run_id": manifest["model_run_id"],
            "finalized_at": "2026-08-11T00:00:00Z",
        },
    )
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


def test_success_hashes_exact_complete_input_without_mutation(tmp_path: Path) -> None:
    from part1_smoke_coverage import build_smoke_coverage_report

    fixture = smoke_fixture(tmp_path, scope="smoke_b")
    journal, quarantine = _add_valid_recovery_evidence(fixture)
    before = fingerprint_tree(fixture["repository"])

    report = build_smoke_coverage_report(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["manifest_path"],
        shard_root=fixture["shard_root"],
    )

    expected_paths = sorted(
        path.relative_to(fixture["repository"]).as_posix()
        for path in (
            fixture["shard_root"] / ".shard-provenance.json",
            fixture["shard_root"] / "natural_results.jsonl",
            fixture["shard_root"] / "checkpoint_results.jsonl",
            fixture["shard_root"] / "audit_events.jsonl",
            fixture["shard_root"] / ".finalized",
            fixture["shard_root"] / ".writer.guard",
            journal,
            quarantine,
        )
    )
    hashes = report["summary"]["stable_source_hashes"]
    assert [entry["relative_path"] for entry in hashes] == expected_paths
    for entry in hashes:
        data = (fixture["repository"] / entry["relative_path"]).read_bytes()
        assert entry == {
            "relative_path": entry["relative_path"],
            "sha256": hashlib.sha256(data).hexdigest(),
            "byte_size": len(data),
        }
    assert fingerprint_tree(fixture["repository"]) == before


@pytest.mark.parametrize("mutation", ["wrong_path", "wrong_scope", "manifest_drift", "unfinalized", "active_lock", "pending_takeover", "invalid_tail", "duplicate_natural", "duplicate_checkpoint", "missing_natural", "missing_checkpoint", "unexpected_natural", "unexpected_checkpoint", "hierarchy", "checkpoint_plan", "lifecycle"])
def test_smoke_validator_fails_closed_for_one_bad_condition(tmp_path: Path, mutation: str) -> None:
    from part1_smoke_coverage import build_smoke_coverage_report

    fixture = smoke_fixture(tmp_path, scope="smoke_a")
    shard_root = fixture["shard_root"]
    if mutation == "wrong_path":
        requested_shard = shard_root.parent / "other"
    else:
        requested_shard = shard_root
        if mutation == "wrong_scope":
            manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
            manifest["execution_scope"] = "production"
            _write_json(fixture["manifest_path"], manifest)
        elif mutation == "manifest_drift":
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
        elif mutation == "hierarchy":
            _mutate_first_jsonl(
                shard_root / "checkpoint_results.jsonl",
                lambda record: record.update(parent_raw_record_id="f" * 64),
            )
        elif mutation == "checkpoint_plan":
            _mutate_first_jsonl(
                shard_root / "checkpoint_results.jsonl",
                lambda record: record.update(requested_fraction=0.123),
            )
        elif mutation == "lifecycle":
            (shard_root / "audit_events.jsonl").write_bytes(b"")
    before = fingerprint_tree(shard_root)
    with pytest.raises((RuntimeError, ValueError)):
        build_smoke_coverage_report(repository_root=fixture["repository"], model_run_manifest_path=fixture["manifest_path"], shard_root=requested_shard)
    assert fingerprint_tree(shard_root) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "natural_prompt_hash",
        "natural_component_adapter",
        "natural_component_prompt",
        "natural_component_parser",
        "checkpoint_inducer_text",
        "checkpoint_token_convention",
        "checkpoint_ad_token_ids",
        "checkpoint_component_adapter",
        "checkpoint_component_parser",
        "checkpoint_component_inducer",
    ],
)
def test_terminal_records_require_exact_manifest_compatibility(
    tmp_path: Path, mutation: str
) -> None:
    from part1_smoke_coverage import build_smoke_coverage_report

    fixture = smoke_fixture(tmp_path, scope="smoke_b")
    natural_path = fixture["shard_root"] / "natural_results.jsonl"
    checkpoint_path = fixture["shard_root"] / "checkpoint_results.jsonl"
    if mutation == "natural_prompt_hash":
        _mutate_first_jsonl(natural_path, lambda record: record.update(prompt_hash="0" * 64))
    elif mutation.startswith("natural_component_"):
        component = mutation.removeprefix("natural_component_")
        _mutate_first_jsonl(
            natural_path,
            lambda record: record["component_versions"].update({component: "different-v1"}),
        )
    elif mutation == "checkpoint_inducer_text":
        _mutate_all_jsonl(
            checkpoint_path, lambda record: record.update(inducer_text="different")
        )
    elif mutation == "checkpoint_token_convention":
        _mutate_first_jsonl(checkpoint_path, lambda record: record.update(token_convention="different-v1"))
    elif mutation == "checkpoint_ad_token_ids":
        def change_ids(record: dict) -> None:
            changed = list(record["ad_token_ids"])
            changed[-1] += 100
            record["ad_token_ids"] = changed
        _mutate_first_jsonl(checkpoint_path, change_ids)
    else:
        component = mutation.removeprefix("checkpoint_component_")
        _mutate_first_jsonl(
            checkpoint_path,
            lambda record: record["component_versions"].update({component: "different-v1"}),
        )

    with pytest.raises(ValueError, match="manifest|compatibility|validation"):
        build_smoke_coverage_report(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["manifest_path"],
            shard_root=fixture["shard_root"],
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "finalization_missing_timestamp",
        "finalization_invalid_timestamp",
        "finalization_extra_field",
        "writer_guard_directory",
        "takeover_event_directory",
        "recovery_non_json",
        "recovery_nested_json",
        "recovery_invalid_semantics",
        "recovery_semantic_mismatch",
        "quarantine_file",
        "lock_history_file",
    ],
)
def test_smoke_validator_requires_canonical_finalization_and_optional_layout(
    tmp_path: Path, mutation: str
) -> None:
    from part1_smoke_coverage import build_smoke_coverage_report

    fixture = smoke_fixture(tmp_path, scope="smoke_b")
    shard_root = fixture["shard_root"]
    marker_path = shard_root / ".finalized"
    if mutation.startswith("finalization_"):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if mutation == "finalization_missing_timestamp":
            marker.pop("finalized_at")
        elif mutation == "finalization_invalid_timestamp":
            marker["finalized_at"] = "2026-08-11T00:00:00+00:00"
        else:
            marker["unexpected"] = True
        _write_json(marker_path, marker)
    elif mutation == "writer_guard_directory":
        (shard_root / ".writer.guard").unlink()
        (shard_root / ".writer.guard").mkdir()
        (shard_root / ".writer.guard" / "child").write_bytes(b"unsafe")
    elif mutation == "takeover_event_directory":
        (shard_root / ".writer-lock-takeover-event.json").mkdir()
        (shard_root / ".writer-lock-takeover-event.json" / "child").write_bytes(b"unsafe")
    elif mutation.startswith("recovery_"):
        if mutation == "recovery_semantic_mismatch":
            journal_path, _quarantine = _add_valid_recovery_evidence(fixture)
            event = json.loads(journal_path.read_text(encoding="utf-8"))
            event["error_details"]["original_size"] += 1
            journal_path.write_bytes(
                json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            audit_path = shard_root / "audit_events.jsonl"
            events = [json.loads(line) for line in audit_path.read_bytes().splitlines()]
            events[-1] = event
            _write_jsonl(audit_path, events)
        else:
            journal = shard_root / "recovery_journal"
            journal.mkdir()
            if mutation == "recovery_non_json":
                (journal / "unexpected.bin").write_bytes(b"ignored by glob")
            elif mutation == "recovery_nested_json":
                (journal / "nested").mkdir()
                (journal / "nested" / ("f" * 64 + ".json")).write_bytes(b"{}")
            else:
                (journal / ("f" * 64 + ".json")).write_bytes(b"{}")
    elif mutation == "quarantine_file":
        (shard_root / "quarantine").write_bytes(b"must be a directory")
    else:
        (shard_root / ".lock_history").write_bytes(b"must be a directory")

    with pytest.raises((RuntimeError, ValueError), match="finalization|layout|source|recovery|directory|file|canonical|failed"):
        build_smoke_coverage_report(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["manifest_path"],
            shard_root=fixture["shard_root"],
        )


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


def test_smoke_validator_uses_store_snapshot_apis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_smoke_coverage

    fixture = smoke_fixture(tmp_path, scope="smoke_b")
    calls = {"inspect": 0, "index": 0, "recovery": 0}
    store_type = part1_smoke_coverage.Part1ShardStore
    original_inspect = store_type.inspect_from_snapshot
    original_index = store_type.build_index_from_snapshot
    original_recovery = store_type.recovery_journal_events_from_snapshot

    def inspect(store, **kwargs):
        calls["inspect"] += 1
        return original_inspect(store, **kwargs)

    def index(store, inspection, **kwargs):
        calls["index"] += 1
        return original_index(store, inspection, **kwargs)

    def recovery(store, entries):
        calls["recovery"] += 1
        return original_recovery(store, entries)

    monkeypatch.setattr(store_type, "inspect_from_snapshot", inspect)
    monkeypatch.setattr(store_type, "build_index_from_snapshot", index)
    monkeypatch.setattr(store_type, "recovery_journal_events_from_snapshot", recovery)

    part1_smoke_coverage.build_smoke_coverage_report(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["manifest_path"],
        shard_root=fixture["shard_root"],
    )

    # One index build belongs to the independent public validate_shard gate;
    # the second consumes the authoritative inspection/recovery snapshot.
    assert calls == {"inspect": 1, "index": 2, "recovery": 1}


def test_smoke_validator_rejects_source_change_between_inventory_and_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_smoke_coverage

    fixture = smoke_fixture(tmp_path, scope="smoke_b")
    audit_path = fixture["shard_root"] / "audit_events.jsonl"
    original_read_bytes = Path.read_bytes
    changed = False

    def read_bytes(path: Path) -> bytes:
        nonlocal changed
        data = original_read_bytes(path)
        if path == audit_path and not changed:
            changed = True
            events = [json.loads(line) for line in data.splitlines()]
            events[0]["event_timestamp"] = "2026-08-11T00:00:01Z"
            _write_jsonl(audit_path, events)
        return data

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(RuntimeError, match="changed during validation"):
        part1_smoke_coverage.build_smoke_coverage_report(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["manifest_path"],
            shard_root=fixture["shard_root"],
        )


@pytest.mark.parametrize(
    "appearing_state",
    ["unexpected_source", "active_lock", "pending_claim", "safe_component_replacement"],
)
def test_smoke_validator_reinventories_tree_and_lock_state_at_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    appearing_state: str,
) -> None:
    import part1_smoke_coverage

    fixture = smoke_fixture(tmp_path, scope="smoke_b")
    shard_root = fixture["shard_root"]
    original = part1_smoke_coverage._partition
    changed = False

    def mutate_once(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if changed:
            return result
        changed = True
        if appearing_state == "unexpected_source":
            (shard_root / "appeared.bin").write_bytes(b"appeared")
        elif appearing_state == "active_lock":
            (shard_root / ".writer.lock").write_bytes(b"appeared")
        elif appearing_state == "pending_claim":
            (shard_root / ".writer-lock-recovery.claim").write_bytes(b"appeared")
        else:
            moved = shard_root.parent / "shard-original"
            shard_root.rename(moved)
            shard_root.symlink_to(moved, target_is_directory=True)
        return result

    monkeypatch.setattr(part1_smoke_coverage, "_partition", mutate_once)
    with pytest.raises(RuntimeError, match="changed during validation"):
        part1_smoke_coverage.build_smoke_coverage_report(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["manifest_path"],
            shard_root=fixture["shard_root"],
        )


@pytest.mark.parametrize("changed_input", ["manifest_bytes", "git_head"])
def test_smoke_validator_rechecks_manifest_inputs_and_git_head_at_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_input: str
) -> None:
    import part1_smoke_coverage

    fixture = smoke_fixture(tmp_path, scope="smoke_b")
    original = part1_smoke_coverage._partition
    changed = False

    def mutate_once(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            changed = True
            if changed_input == "manifest_bytes":
                fixture["manifest_path"].write_bytes(
                    fixture["manifest_path"].read_bytes() + b" "
                )
            else:
                note = fixture["repository"] / "head-change.txt"
                note.write_text("change HEAD during validation\n", encoding="utf-8")
                subprocess.run(["git", "add", note.name], cwd=fixture["repository"], check=True)
                subprocess.run(
                    ["git", "commit", "--quiet", "-m", "change during validation"],
                    cwd=fixture["repository"], check=True,
                )
        return result

    monkeypatch.setattr(part1_smoke_coverage, "_partition", mutate_once)
    with pytest.raises(RuntimeError, match="changed during validation"):
        part1_smoke_coverage.build_smoke_coverage_report(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["manifest_path"],
            shard_root=fixture["shard_root"],
        )

"""CPU-only full-shape acceptance for the Part 1 production data path."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pytest

from part1_contract import (
    FIXED_CHECKPOINT_FRACTIONS,
    FIXED_PRIMARY_AUROC_FEATURE_REGISTRY,
    FIXED_SUBJECTS,
    attempt_id,
    audit_event_id,
    checkpoint_record_id,
    derive_generation_seed,
    natural_record_id,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXED_TIMESTAMP = "2026-08-11T00:00:00Z"
MAX_ACCEPTANCE_BYTES = 8 * 1024**3
MAX_COVERAGE_SECONDS = 4 * 60 * 60
MAX_MERGE_SECONDS = 4 * 60 * 60
MAX_ANALYSIS_SECONDS = 8 * 60 * 60


def _json_bytes(value: Mapping[str, Any], *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + suffix
    ).encode("utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_json_bytes(row, newline=True))


def _wrong_letter(gold_letter: str) -> str:
    return "ABCD"[("ABCD".index(gold_letter) + 1) % 4]


def _signal_run(sample_index: int, run_id: int) -> int:
    # The first question's runs 0 and 2 are the deliberate repeated trajectories.
    return 0 if sample_index == 0 and run_id in {0, 2} else run_id


def _natural_result(fixture: Mapping[str, Any], question: Mapping[str, Any], run_id: int) -> dict[str, Any]:
    from part1_generation import NaturalGenerationCapture, build_natural_terminal_result

    manifest = fixture["manifest"]
    token_contract = fixture["preflight"]["token_contract"]
    sample_index = question["sample_index"]
    signal_run = _signal_run(sample_index, run_id)
    gold = question["gold_letter"]
    answer = gold if run_id % 2 == 0 else _wrong_letter(gold)
    confidence_percent = 25 + 10 * ((sample_index + signal_run) % 7)
    reasoning_tokens = tuple(90 + (index % 5) for index in range(20))
    generated = (
        *token_contract["reasoning_open_token_ids"],
        *reasoning_tokens,
        *token_contract["reasoning_close_token_ids"],
        token_contract["ad_token_ids"]["ABCD".index(answer)],
        token_contract["eos_token_id"],
    )
    entropy_base = 0.35 + 0.025 * ((sample_index + signal_run) % 13)
    entropies = tuple(
        entropy_base + 0.003 * token_index for token_index in range(len(generated))
    )
    seed = derive_generation_seed(
        base_seed=manifest["base_generation_seed"],
        canonical_model_identity=manifest["canonical_model_identity"],
        question_id=question["question_id"],
        run_id=run_id,
    )
    capture = NaturalGenerationCapture(
        rendered_prompt=f"Synthetic canonical prompt {sample_index}",
        prompt_token_ids=(1, 2, 3),
        generated_token_ids=generated,
        decoded_output=(
            f"<think>synthetic trajectory {sample_index}:{signal_run}</think>\n"
            f"Answer: {answer}\nConfidence: {confidence_percent}"
        ),
        raw_prewarper_logits=(),
        stop_reason="eos",
        precomputed_entropy_nats=entropies,
    )
    result = build_natural_terminal_result(
        identity={
            "study_id": manifest["study_id"],
            "model_run_id": manifest["model_run_id"],
            "model_run_manifest_hash": manifest["model_run_manifest_hash"],
            "question_manifest_hash": manifest["question_manifest_hash"],
            "question_id": question["question_id"],
            "sample_index": sample_index,
            "subject": question["subject"],
            "gold_letter": gold,
        },
        run_id=run_id,
        generation_seed=seed,
        terminal_attempt_number=1,
        capture=capture,
        token_contract=token_contract,
        decode_reasoning=lambda _token_ids: f"synthetic trajectory {sample_index}:{signal_run}",
    )
    # Generation normally receives the preflight-pinned prompt contract. This
    # direct fixture bypasses runtime rendering, so bind that immutable contract
    # explicitly; production coverage recomputes and validates it.
    result["prompt_hash"] = manifest["prompt_hash"]
    return result


def _retarget_natural_result(
    fixture: Mapping[str, Any],
    question: Mapping[str, Any],
    run_id: int,
    template: Mapping[str, Any],
) -> dict[str, Any]:
    """Clone one prevalidated row while recomputing every run/content field."""

    manifest = fixture["manifest"]
    token_contract = fixture["preflight"]["token_contract"]
    sample_index = question["sample_index"]
    signal_run = _signal_run(sample_index, run_id)
    gold = question["gold_letter"]
    answer = gold if run_id % 2 == 0 else _wrong_letter(gold)
    confidence_percent = 25 + 10 * ((sample_index + signal_run) % 7)
    generated = list(template["generated_token_ids"])
    generated[-2] = token_contract["ad_token_ids"]["ABCD".index(answer)]
    entropy_base = 0.35 + 0.025 * ((sample_index + signal_run) % 13)
    entropies = [
        entropy_base + 0.003 * token_index for token_index in range(len(generated))
    ]
    reasoning_values = entropies[1:21]
    seed = derive_generation_seed(
        base_seed=manifest["base_generation_seed"],
        canonical_model_identity=manifest["canonical_model_identity"],
        question_id=question["question_id"],
        run_id=run_id,
    )
    output = dict(template)
    output.update(
        raw_record_id=natural_record_id(
            manifest["study_id"],
            manifest["model_run_id"],
            question["question_id"],
            run_id,
        ),
        run_id=run_id,
        generation_seed=seed,
        terminal_attempt_id=attempt_id(
            manifest["study_id"],
            manifest["model_run_id"],
            question["question_id"],
            run_id,
            1,
        ),
        generated_token_ids=generated,
        decoded_output=(
            f"<think>synthetic trajectory {sample_index}:{signal_run}</think>\n"
            f"Answer: {answer}\nConfidence: {confidence_percent}"
        ),
        reasoning_text=f"synthetic trajectory {sample_index}:{signal_run}",
        per_token_entropy_nats=entropies,
        mean_reasoning_entropy_nats=sum(reasoning_values) / len(reasoning_values),
        tail_reasoning_entropy_nats=sum(reasoning_values[-2:]) / 2,
        terminal_answer_block_text=f"Answer: {answer}\nConfidence: {confidence_percent}",
        natural_answer=answer,
        raw_confidence_text=str(confidence_percent),
        raw_parsed_confidence=confidence_percent,
        normalized_confidence=confidence_percent / 100,
        natural_correct=answer == gold,
    )
    return output


def _checkpoint_answer(gold_letter: str, *, mode: int, checkpoint_index: int) -> str:
    if mode == 0:
        return gold_letter
    if mode == 1:
        return _wrong_letter(gold_letter)
    if mode == 2:
        return gold_letter if checkpoint_index < 5 else _wrong_letter(gold_letter)
    return gold_letter if checkpoint_index % 2 == 0 else _wrong_letter(gold_letter)


def _checkpoint_results(
    fixture: Mapping[str, Any],
    question: Mapping[str, Any],
    natural: Mapping[str, Any],
    template: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from part1_checkpoints import (
        build_checkpoint_probe_plans,
        choice_answer_metrics,
    )

    manifest = fixture["manifest"]
    token_contract = fixture["preflight"]["token_contract"]
    sample_index = question["sample_index"]
    run_id = natural["run_id"]
    signal_run = _signal_run(sample_index, run_id)
    mode = (sample_index + signal_run) % 4
    plans = build_checkpoint_probe_plans(
        natural,
        inducer_token_ids=token_contract["inducer_token_ids"],
        inducer_version=manifest["inducer_version"],
    )
    assert len(plans) == 11
    assert all(not plan.is_alias for plan in plans)
    output: list[dict[str, Any]] = []
    for plan in plans:
        answer = _checkpoint_answer(
            question["gold_letter"],
            mode=mode,
            checkpoint_index=plan.requested_checkpoint_index,
        )
        confidence_percent = 20 + 10 * (
            (sample_index + signal_run + plan.requested_checkpoint_index) % 8
        )
        vocabulary_size = max(token_contract["ad_token_ids"]) + 1
        logits = [0.01 * index for index in range(vocabulary_size)]
        answer_token = token_contract["ad_token_ids"]["ABCD".index(answer)]
        logits[answer_token] = 0.8 + 0.15 * (
            1 + (sample_index + signal_run + plan.requested_checkpoint_index) % 5
        )
        metrics = choice_answer_metrics(
            logits, ad_token_ids=token_contract["ad_token_ids"]
        )
        checkpoint = dict(template)
        checkpoint.update(
            checkpoint_record_id=checkpoint_record_id(
                natural["study_id"],
                natural["model_run_id"],
                natural["question_id"],
                natural["run_id"],
                plan.checkpoint_id,
            ),
            parent_raw_record_id=natural["raw_record_id"],
            question_id=natural["question_id"],
            sample_index=natural["sample_index"],
            subject=natural["subject"],
            run_id=natural["run_id"],
            checkpoint_id=plan.checkpoint_id,
            natural_seed=natural["generation_seed"],
            terminal_attempt_id=attempt_id(
                natural["study_id"],
                natural["model_run_id"],
                natural["question_id"],
                natural["run_id"],
                1,
                checkpoint_id=plan.checkpoint_id,
            ),
            requested_checkpoint_index=plan.requested_checkpoint_index,
            requested_fraction=plan.requested_fraction,
            k_keep=plan.k_keep,
            actual_fraction=plan.actual_fraction,
            shared_probe_id=plan.shared_probe_id,
            is_alias=plan.is_alias,
            alias_metadata=plan.alias_metadata,
            prefix_hash=plan.prefix_hash,
            inducer_version=plan.inducer_version,
            forced_generated_token_ids=[answer_token, token_contract["eos_token_id"]],
            decoded_forced_output=f" {answer}\nConfidence: {confidence_percent}",
            terminal_answer_block_text=f"Answer: {answer}\nConfidence: {confidence_percent}",
            forced_answer=answer,
            raw_confidence_text=str(confidence_percent),
            raw_parsed_confidence=confidence_percent,
            normalized_confidence=confidence_percent / 100,
            checkpoint_local_correct=answer == question["gold_letter"],
            answer_token_index=0,
            answer_token_id=answer_token,
            ad_logits_float32=metrics["ad_logits_float32"],
            ad_probabilities_float32=metrics["ad_probabilities_float32"],
            answer_entropy_nats=metrics["answer_entropy_nats"],
            full_vocabulary_answer_step_entropy_nats=metrics[
                "full_vocabulary_answer_step_entropy_nats"
            ],
            maximum_ad_probability=metrics["maximum_ad_probability"],
            agrees_with_natural_answer=answer == natural["natural_answer"],
        )
        output.append(checkpoint)
    return output


def _checkpoint_template(
    fixture: Mapping[str, Any],
    question: Mapping[str, Any],
    natural: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and schema-check one row per question; clones are reader-validated."""

    from part1_checkpoints import (
        CheckpointGenerationCapture,
        build_checkpoint_probe_plans,
        build_checkpoint_terminal_result,
    )

    manifest = fixture["manifest"]
    token_contract = fixture["preflight"]["token_contract"]
    plan = build_checkpoint_probe_plans(
        natural,
        inducer_token_ids=token_contract["inducer_token_ids"],
        inducer_version=manifest["inducer_version"],
    )[0]
    answer = _checkpoint_answer(
        question["gold_letter"],
        mode=question["sample_index"] % 4,
        checkpoint_index=0,
    )
    vocabulary_size = max(token_contract["ad_token_ids"]) + 1
    logits = [0.01 * index for index in range(vocabulary_size)]
    answer_token = token_contract["ad_token_ids"]["ABCD".index(answer)]
    logits[answer_token] = 0.95
    capture = CheckpointGenerationCapture(
        forced_generated_token_ids=(answer_token, token_contract["eos_token_id"]),
        decoded_forced_output=f" {answer}\nConfidence: 20",
        raw_prewarper_logits=(),
        answer_step_raw_logits=tuple(logits),
    )
    return build_checkpoint_terminal_result(
        parent=natural,
        plan=plan,
        capture=capture,
        token_contract=token_contract,
        gold_letter=question["gold_letter"],
        terminal_attempt_number=1,
    )


def _attempt_event(
    record: Mapping[str, Any], *, shard_id: str, event_type: str, sequence: int
) -> dict[str, Any]:
    checkpoint = record["schema_name"] == "part1_checkpoint_terminal_result"
    terminal_record_id = (
        record["checkpoint_record_id"] if checkpoint else record["raw_record_id"]
    )
    attempt = record["terminal_attempt_id"]
    return {
        "schema_name": "part1_audit_event",
        "schema_version": "1.0.0",
        "event_id": audit_event_id(attempt, event_type, sequence),
        "event_scope": "attempt",
        "study_id": record["study_id"],
        "model_run_id": record["model_run_id"],
        "shard_id": shard_id,
        "question_id": record["question_id"],
        "run_id": record["run_id"],
        "checkpoint_id": record.get("checkpoint_id"),
        "attempt_id": attempt,
        "attempt_number": 1,
        "event_sequence": sequence,
        "event_type": event_type,
        "event_timestamp": FIXED_TIMESTAMP,
        "execution_context": {"hostname": "synthetic-cpu", "pid": 1},
        "outcome_category": None,
        "error_details": None,
        "retry_classification": None,
        "retry_decision": None,
        "backoff_seconds": None,
        "related_lock_owner": None,
        "terminal_record_id": terminal_record_id if event_type == "attempt_completed" else None,
        "operator_reason": None,
    }


def _write_shard(
    fixture: Mapping[str, Any], question: Mapping[str, Any]
) -> None:
    manifest = fixture["manifest"]
    shard_id = f"shard-{question['sample_index']:03d}"
    shard_root = fixture["raw_root"] / shard_id
    shard_root.mkdir(parents=True)
    natural_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    natural_template = _natural_result(fixture, question, 0)
    checkpoint_template = _checkpoint_template(
        fixture, question, natural_template
    )
    for run_id in range(10):
        natural = (
            natural_template
            if run_id == 0
            else _retarget_natural_result(
                fixture, question, run_id, natural_template
            )
        )
        natural_rows.append(natural)
        audit_rows.extend(
            (
                _attempt_event(natural, shard_id=shard_id, event_type="attempt_started", sequence=0),
                _attempt_event(natural, shard_id=shard_id, event_type="attempt_completed", sequence=1),
            )
        )
        for checkpoint in _checkpoint_results(
            fixture, question, natural, checkpoint_template
        ):
            checkpoint_rows.append(checkpoint)
            audit_rows.extend(
                (
                    _attempt_event(
                        checkpoint,
                        shard_id=shard_id,
                        event_type="attempt_started",
                        sequence=0,
                    ),
                    _attempt_event(
                        checkpoint,
                        shard_id=shard_id,
                        event_type="attempt_completed",
                        sequence=1,
                    ),
                )
            )
    (shard_root / ".shard-provenance.json").write_bytes(
        _json_bytes(
            {
                "schema_name": "part1_shard_provenance",
                "schema_version": "1.0.0",
                "study_id": manifest["study_id"],
                "model_run_id": manifest["model_run_id"],
                "model_run_manifest_hash": manifest["model_run_manifest_hash"],
                "shard_id": shard_id,
            }
        )
    )
    _write_jsonl(shard_root / "natural_results.jsonl", natural_rows)
    _write_jsonl(shard_root / "checkpoint_results.jsonl", checkpoint_rows)
    _write_jsonl(shard_root / "audit_events.jsonl", audit_rows)
    (shard_root / ".finalized").write_bytes(
        _json_bytes(
            {
                "store_version": "part1-store-v1",
                "shard_id": shard_id,
                "study_id": manifest["study_id"],
                "model_run_id": manifest["model_run_id"],
                "finalized_at": FIXED_TIMESTAMP,
            }
        )
    )


def write_full_shape_fixture(tmp_path: Path) -> dict[str, Any]:
    from test_run_part1_shard import production_fixture

    fixture = production_fixture(tmp_path)
    config_target = fixture["repository"] / "configs" / "part1" / "analysis.json"
    config_target.parent.mkdir(parents=True)
    shutil.copy2(REPOSITORY_ROOT / "configs" / "part1" / "analysis.json", config_target)
    raw_root = fixture["repository"] / fixture["manifest"]["output_paths"]["raw_shards"]
    raw_root.mkdir(parents=True)
    fixture["raw_root"] = raw_root
    for question in fixture["bundle"].records:
        _write_shard(fixture, question)
    return fixture


def run_coverage_publication(fixture: Mapping[str, Any]) -> Path:
    from part1_coverage import build_coverage_report, publish_coverage_report

    report = build_coverage_report(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["manifest_path"],
        validation_started_at=FIXED_TIMESTAMP,
        validation_completed_at="2026-08-11T00:00:01Z",
    )
    assert report["is_valid"] is True
    assert report["coverage_complete"] is True
    assert report["paper_analysis_ready"] is True
    assert report["summary"]["natural_partition"]["complete"] == 5_000
    assert report["summary"]["checkpoint_partition"]["complete"] == 55_000
    coverage_path = (
        fixture["repository"]
        / fixture["manifest"]["output_paths"]["validation"]
        / "coverage_report.json"
    )
    publish_coverage_report(
        report,
        coverage_path,
        repository_root=fixture["repository"],
    )
    return coverage_path


def run_merge_publication(
    fixture: Mapping[str, Any], coverage_path: Path
) -> Path:
    from part1_merge import merge_part1_results

    merge_path, manifest = merge_part1_results(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["manifest_path"],
        coverage_report_path=coverage_path,
    )
    fixture["merge_manifest"] = manifest
    return merge_path


def run_analysis_publication(
    fixture: Mapping[str, Any],
    coverage_path: Path,
    merge_path: Path,
    *,
    bootstrap_replicates: int,
) -> Path:
    from part1_analysis import analyze_production, load_production_analysis_source

    assert coverage_path.is_file()
    assert merge_path.is_dir()
    loaded = load_production_analysis_source(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["manifest_path"],
    )
    assert len(loaded.natural_rows) == 5_000
    assert len(loaded.checkpoint_rows) == 55_000
    fixture["loaded_source"] = loaded
    analysis_path, manifest = analyze_production(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["manifest_path"],
        bootstrap_replicates=bootstrap_replicates,
    )
    fixture["analysis_manifest"] = manifest
    return analysis_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tree_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _report_phase(
    label: str,
    *,
    phase_started: float,
    run_started: float,
    durations: dict[str, float],
) -> float:
    now = time.perf_counter()
    durations[label] = now - phase_started
    print(
        f"part1_full_acceptance phase={label} "
        f"phase_seconds={now - phase_started:.3f} "
        f"elapsed_seconds={now - run_started:.3f}",
        file=sys.__stderr__,
        flush=True,
    )
    return now


def assert_full_acceptance(
    fixture: Mapping[str, Any],
    coverage_path: Path,
    merge_path: Path,
    analysis_path: Path,
) -> None:
    from part1_analysis import TABLE_SPECS, load_production_analysis_source, validate_analysis_directory
    from part1_merge import validate_merge_directory

    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert coverage["summary"]["expected"] == {
        "questions": 500,
        "shards": 500,
        "natural_logical_keys": 5_000,
        "checkpoint_logical_keys": 55_000,
    }
    assert coverage["summary"]["observed"]["shards"] == 500
    assert coverage["summary"]["natural_partition"]["complete"] == 5_000
    assert coverage["summary"]["checkpoint_partition"]["complete"] == 55_000
    assert coverage["summary"]["structural_errors"] == []
    assert len(coverage["source_files"]) > 2_000

    merge = validate_merge_directory(
        merge_path, expected_manifest=fixture["merge_manifest"]
    )
    assert merge["outputs"]["natural_results"]["row_count"] == 5_000
    assert merge["outputs"]["checkpoint_results"]["row_count"] == 55_000
    assert merge["outputs"]["audit_events"]["row_count"] == 120_000

    source = load_production_analysis_source(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["manifest_path"],
    )
    natural_keys = {(row["question_id"], row["run_id"]) for row in source.natural_rows}
    expected_keys = {
        (question["question_id"], run_id)
        for question in fixture["bundle"].records
        for run_id in range(10)
    }
    assert natural_keys == expected_keys
    for subject in FIXED_SUBJECTS:
        targets = {
            row["natural_correct"] for row in source.natural_rows if row["subject"] == subject
        }
        assert targets == {False, True}

    manifest = validate_analysis_directory(
        analysis_path, expected_manifest=fixture["analysis_manifest"]
    )
    assert manifest["bootstrap_replicates"] == 1_000
    assert manifest["bootstrap_mode"] == "development"
    assert set(manifest["tables"]) == {
        filename for filename, _columns in TABLE_SPECS.values()
    }
    provenance = {
        key: manifest[key]
        for key in (
            "study_id",
            "study_manifest_hash",
            "question_manifest_hash",
            "model_run_id",
            "model_run_manifest_hash",
            "merge_id",
            "merge_manifest_hash",
            "coverage_report_id",
        )
    }
    for metadata_name in manifest["tables"].values():
        metadata = json.loads((analysis_path / metadata_name).read_text(encoding="utf-8"))
        assert metadata["source_provenance"] == provenance
        assert metadata["bootstrap_replicates"] == 1_000

    trajectories = _read_csv(analysis_path / TABLE_SPECS["trajectory_features"][0])
    assert len(trajectories) == 5_000
    assert {row["subject"] for row in trajectories} == set(FIXED_SUBJECTS)
    repeated = [
        row
        for row in trajectories
        if row["sample_index"] == "0" and row["run_id"] in {"0", "2"}
    ]
    assert len(repeated) == 2
    ignored = {"raw_record_id", "run_id", "generation_seed"}
    assert {
        key: value for key, value in repeated[0].items() if key not in ignored
    } == {
        key: value for key, value in repeated[1].items() if key not in ignored
    }
    for feature in FIXED_PRIMARY_AUROC_FEATURE_REGISTRY:
        values = {row[feature] for row in trajectories}
        assert "" not in values
        assert len(values) > 1

    primary = _read_csv(analysis_path / TABLE_SPECS["primary_auroc"][0])
    for feature in FIXED_PRIMARY_AUROC_FEATURE_REGISTRY:
        pooled = [
            row
            for row in primary
            if row["feature"] == feature
            and row["target"] == "natural_correct"
            and row["grouping"] == "pooled"
        ]
        subjects = {
            row["subject"]
            for row in primary
            if row["feature"] == feature
            and row["target"] == "natural_correct"
            and row["grouping"] == "subject"
        }
        assert len(pooled) == 1
        assert pooled[0]["sample_size"] == "5000"
        assert subjects == set(FIXED_SUBJECTS)

    calibration = _read_csv(analysis_path / TABLE_SPECS["calibration_metrics"][0])
    natural_rows = [
        row
        for row in calibration
        if row["calibration_family"] == "natural_confidence"
        and row["target"] == "natural_correct"
        and row["grouping"] in {"pooled", "subject"}
    ]
    assert {row["grouping"] for row in natural_rows} == {"pooled", "subject"}
    assert {row["subject"] for row in natural_rows if row["grouping"] == "subject"} == set(
        FIXED_SUBJECTS
    )
    for family in ("checkpoint_confidence", "maximum_ad_probability"):
        rows = [
            row
            for row in calibration
            if row["calibration_family"] == family
            and row["target"] == "checkpoint_local_correct"
            and row["grouping"] in {"pooled", "subject"}
        ]
        assert {float(row["requested_fraction"]) for row in rows} == set(
            FIXED_CHECKPOINT_FRACTIONS
        )
        assert {
            float(row["requested_fraction"])
            for row in rows
            if row["is_main_checkpoint"] == "true"
        } == {0.0, 0.5, 1.0}
        assert all(row["sample_size"] == "5000" for row in rows)

    within = _read_csv(analysis_path / TABLE_SPECS["within_question_distribution"][0])
    first_question = [row for row in within if row["question_id"] == repeated[0]["question_id"]]
    assert len(first_question) == len(FIXED_PRIMARY_AUROC_FEATURE_REGISTRY)
    assert all(row["correct_run_count"] == "5" for row in first_question)
    assert all(row["incorrect_run_count"] == "5" for row in first_question)


def _oracle_row(
    subject: str, question_id: str, run_id: int, correct: bool, value: float
) -> dict[str, Any]:
    row = {
        "subject": subject,
        "question_id": question_id,
        "run_id": run_id,
        "natural_correct": correct,
    }
    row.update({feature: value for feature in FIXED_PRIMARY_AUROC_FEATURE_REGISTRY})
    return row


def test_bootstrap_draw_multiplicity_oracle() -> None:
    from part1_bootstrap import expand_question_draws, question_draw_plan_from_indices

    subject = FIXED_SUBJECTS[0]
    frame = [
        {"subject": subject, "question_id": question_id}
        for question_id in ("q0", "q1", "q2")
    ]
    plan = question_draw_plan_from_indices(
        frame, np.array([[0, 0, 2]]), seed=42, small_fixture=True
    )
    rows = [
        {"subject": subject, "question_id": question_id, "value": question_id}
        for question_id in ("q0", "q1", "q2")
    ]
    expanded = expand_question_draws(
        plan, rows, replicate_id=0, max_rows=3
    )
    assert [row["question_id"] for row in expanded] == ["q0", "q0", "q2"]


def test_macro_invalidity_oracle() -> None:
    from part1_bootstrap import question_draw_plan_from_indices
    from part1_statistics import primary_auroc_analysis

    rows: list[dict[str, Any]] = []
    frame: list[dict[str, str]] = []
    for subject_index, subject in enumerate(FIXED_SUBJECTS):
        false_q = f"{subject}-false"
        true_q = f"{subject}-true"
        rows.extend(
            (
                _oracle_row(subject, false_q, 0, False, 0.1),
                _oracle_row(subject, true_q, 0, True, 0.9),
            )
        )
        frame.extend(
            (
                {"subject": subject, "question_id": false_q},
                {"subject": subject, "question_id": true_q},
            )
        )
    plan = question_draw_plan_from_indices(
        frame,
        np.array([[0, 0, 0, 1, 0, 1, 0, 1, 0, 1]]),
        seed=42,
    )
    result = primary_auroc_analysis(rows, plan)
    feature = FIXED_PRIMARY_AUROC_FEATURE_REGISTRY[0]
    macro = next(
        row
        for row in result["metric_rows"]
        if row["feature"] == feature and row["grouping"] == "macro"
    )
    pooled = next(
        row
        for row in result["metric_rows"]
        if row["feature"] == feature and row["grouping"] == "pooled"
    )
    assert (macro["valid_replicates"], macro["invalid_replicates"]) == (0, 1)
    assert (pooled["valid_replicates"], pooled["invalid_replicates"]) == (1, 0)


def test_within_question_paired_difference_oracle() -> None:
    from part1_bootstrap import build_question_draw_plan
    from part1_statistics import within_question_analysis

    subject = FIXED_SUBJECTS[0]
    rows = [
        _oracle_row(subject, "q0", 0, True, 8.0),
        _oracle_row(subject, "q0", 1, True, 6.0),
        _oracle_row(subject, "q0", 2, False, 2.0),
        _oracle_row(subject, "q1", 0, True, 5.0),
        _oracle_row(subject, "q1", 1, False, 1.0),
    ]
    plan = build_question_draw_plan(
        [
            {"subject": subject, "question_id": "q0"},
            {"subject": subject, "question_id": "q1"},
        ],
        replicates=2,
        small_fixture=True,
    )
    result = within_question_analysis(rows, plan, allow_small_fixture=True)
    feature = FIXED_PRIMARY_AUROC_FEATURE_REGISTRY[0]
    distribution = [
        row for row in result["distribution_rows"] if row["feature"] == feature
    ]
    assert [row["paired_difference"] for row in distribution] == [5.0, 4.0]
    summary = next(
        row for row in result["summary_rows"] if row["feature"] == feature
    )
    assert summary["mean_paired_difference"] == 4.5


def test_switch_adjacency_and_stabilization_oracle() -> None:
    from part1_trajectories import extract_trajectory_features
    from test_part1_trajectories import _checkpoint, _natural

    natural = _natural()
    checkpoints = [
        _checkpoint(natural, 0, answer="C"),
        _checkpoint(natural, 2, answer="A"),
        _checkpoint(natural, 3, answer="C"),
        *[_checkpoint(natural, index, answer="C") for index in range(4, 11)],
    ]
    row = extract_trajectory_features(natural, checkpoints)
    assert row["answer_switch_count"] == 1
    assert row["valid_transition_count"] == 8
    assert row["stabilization_fraction"] == 0.3

    missing_endpoint = extract_trajectory_features(natural, checkpoints[:-1])
    assert missing_endpoint["stabilization_fraction"] is None
    assert missing_endpoint["stabilization_reason"] == "checkpoint_1.0_missing"


def test_direct_fixture_shard_passes_the_production_scanner(tmp_path: Path) -> None:
    from part1_coverage import scan_production_shard
    from test_run_part1_shard import production_fixture

    fixture = production_fixture(tmp_path)
    raw_root = fixture["repository"] / fixture["manifest"]["output_paths"]["raw_shards"]
    raw_root.mkdir(parents=True)
    fixture["raw_root"] = raw_root
    question = fixture["bundle"].records[0]
    _write_shard(fixture, question)

    scan = scan_production_shard(
        repository_root=fixture["repository"],
        shard_root=raw_root / "shard-000",
        shard_index=0,
        question=question,
        model_manifest=fixture["manifest"],
    )
    natural = [item for rows in scan.natural_observations.values() for item in rows]
    checkpoints = [
        item for rows in scan.checkpoint_observations.values() for item in rows
    ]
    assert scan.structural_errors == ()
    assert len(natural) == 10
    assert all(item.defect is None for item in natural)
    assert len(checkpoints) == 110
    assert all(item.defect is None for item in checkpoints)


@pytest.mark.part1_full_acceptance
def test_full_shape_raw_to_analysis_acceptance(tmp_path: Path) -> None:
    started = time.perf_counter()
    phase_started = started
    phase_durations: dict[str, float] = {}
    fixture = write_full_shape_fixture(tmp_path)
    phase_started = _report_phase(
        "fixture_written",
        phase_started=phase_started,
        run_started=started,
        durations=phase_durations,
    )
    coverage_path = run_coverage_publication(fixture)
    phase_started = _report_phase(
        "coverage_published",
        phase_started=phase_started,
        run_started=started,
        durations=phase_durations,
    )
    merge_path = run_merge_publication(fixture, coverage_path)
    phase_started = _report_phase(
        "merge_published",
        phase_started=phase_started,
        run_started=started,
        durations=phase_durations,
    )
    analysis_path = run_analysis_publication(
        fixture,
        coverage_path,
        merge_path,
        bootstrap_replicates=1_000,
    )
    phase_started = _report_phase(
        "analysis_published",
        phase_started=phase_started,
        run_started=started,
        durations=phase_durations,
    )
    assert_full_acceptance(fixture, coverage_path, merge_path, analysis_path)
    _report_phase(
        "assertions_complete",
        phase_started=phase_started,
        run_started=started,
        durations=phase_durations,
    )
    elapsed = time.perf_counter() - started
    temporary_bytes = _tree_bytes(tmp_path)
    print(
        f"part1_full_acceptance elapsed_seconds={elapsed:.3f} "
        f"temporary_bytes={temporary_bytes}",
        flush=True,
    )
    assert phase_durations["coverage_published"] <= MAX_COVERAGE_SECONDS
    assert phase_durations["merge_published"] <= MAX_MERGE_SECONDS
    assert phase_durations["analysis_published"] <= MAX_ANALYSIS_SECONDS
    assert temporary_bytes <= MAX_ACCEPTANCE_BYTES

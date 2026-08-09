"""Manifest-bound Part 1 analysis computation and atomic artifact publication."""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mila-matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from part1_bootstrap import build_question_draw_plan
from part1_contract import (
    FIXED_CHECKPOINT_FRACTIONS,
    FIXED_PRIMARY_AUROC_FEATURE_REGISTRY,
    FIXED_SUBJECTS,
    canonical_json_bytes,
    load_config,
    model_run_id,
    model_run_manifest_hash,
    validate_fixed_model_requested_contract,
    validate_instance,
)
from part1_coverage import (
    _global_snapshot_errors,
    build_coverage_report,
    coverage_report_id,
    validate_coverage_report_semantics,
)
from part1_manifests import load_manifest_bundle
from part1_merge import (
    PublicationDurabilityError,
    PublicationStateIndeterminateError,
    TABLE_FILENAMES as MERGE_TABLE_FILENAMES,
    _exclusive_rename_at,
    _fsync_directory_descriptor,
    _load_regular_json,
    _read_regular_file_at,
    _remove_own_stage,
    _require_no_symlink_components,
    decode_merge_table,
    validate_merge_directory,
    validate_merge_directory_at,
)
from part1_runtime import validate_manifest_compatibility
from part1_statistics import (
    checkpoint_calibration_analysis,
    natural_calibration_analysis,
    primary_auroc_analysis,
    secondary_checkpoint_auroc_analysis,
    within_question_analysis,
)
from part1_trajectories import build_trajectory_rows


ANALYSIS_FORMAT_VERSION = "part1-analysis-artifacts-v1"
ANALYSIS_IDENTITY_VERSION = "part1-analysis-identity-v1"
ANALYSIS_MANIFEST_HASH_VERSION = "part1-analysis-manifest-hash-v1"
CSV_SERIALIZATION_VERSION = "part1-analysis-csv-v1"
PLOT_VERSION = "part1-analysis-plots-v1"
BOOTSTRAP_SEED = 42
PRODUCTION_BOOTSTRAP_REPLICATES = frozenset((1_000, 5_000))
MAIN_CHECKPOINT_FRACTIONS = frozenset((0.0, 0.5, 1.0))
FIXTURE_EVIDENCE_LIMITS = (
    "Synthetic fixtures validate analysis control flow, fixed formulas already covered "
    "by Task 6, provenance, serialization, plot selection, and publication; they do "
    "not validate real SmolLM3 tokenization, logits, generation, or empirical conclusions."
)

ANALYSIS_CONFIG_ORACLE: dict[str, Any] = {
    "schema_name": "part1_analysis_config",
    "config_version": "1.0.0",
    "analysis_contract_version": "part1-analysis-v1",
    "primary_target": "natural_correct",
    "primary_auroc_feature_registry": list(FIXED_PRIMARY_AUROC_FEATURE_REGISTRY),
    "bootstrap_seed": 42,
    "development_bootstrap_replicates": 1000,
    "final_bootstrap_replicates": 5000,
    "confidence_interval_percent": 95,
    "minimum_valid_bootstrap_fraction": 0.95,
    "ece_bins": 10,
    "main_checkpoint_fractions": [0.0, 0.5, 1.0],
    "all_checkpoint_fractions": list(FIXED_CHECKPOINT_FRACTIONS),
}

INTERVAL_COLUMNS = (
    "requested_replicates",
    "valid_replicates",
    "invalid_replicates",
    "valid_fraction",
    "confidence_level",
    "percentile_method",
    "lower",
    "upper",
    "interval_valid",
    "interval_reason",
    "warning",
)
METRIC_COLUMNS = (
    "analysis_label",
    "feature",
    "predictor",
    "target",
    "cohort_definition",
    "total_candidate_rows",
    "target_missing_count",
    "predictor_missing_count",
    "sample_size",
    "positive_count",
    "negative_count",
    "grouping",
    "subject",
    "point_estimate",
    "point_estimate_status",
    "point_undefined_reason",
    *INTERVAL_COLUMNS,
)
TRAJECTORY_FEATURE_COLUMNS = (
    "study_id",
    "model_run_id",
    "model_run_manifest_hash",
    "question_manifest_hash",
    "question_id",
    "sample_index",
    "subject",
    "run_id",
    "raw_record_id",
    "natural_execution_outcome",
    "stop_reason",
    "reasoning_status",
    "answer_parse_status",
    "confidence_parse_status",
    "checkpoint_eligible",
    "natural_answer",
    "natural_correct",
    *tuple(FIXED_PRIMARY_AUROC_FEATURE_REGISTRY),
    "answer_switch_count",
    "valid_transition_count",
    "transition_evaluability_status",
    "transition_evaluability_reason",
    "first_natural_answer_appearance_fraction",
    "first_natural_answer_appearance_status",
    "first_natural_answer_appearance_reason",
    "left_correct_answer",
    "left_correct_answer_status",
    "left_correct_answer_reason",
    "later_recovered_correct_answer",
    "later_recovered_correct_answer_status",
    "later_recovered_correct_answer_reason",
    "forced_endpoint_agrees_with_natural",
    "forced_endpoint_agrees_with_natural_status",
    "forced_endpoint_agrees_with_natural_reason",
    "stabilization_fraction",
    "stabilization_status",
    "stabilization_reason",
    "checkpoint_calibration",
    "feature_missing_reasons",
)
TRAJECTORY_EVENT_COLUMNS = (
    "grouping",
    "subject",
    "trajectory_count",
    "switch_count_available",
    "switch_count_unavailable",
    "switch_count_sum",
    "switch_count_mean",
    "first_appearance_found",
    "first_appearance_not_found",
    "first_appearance_unavailable",
    "first_appearance_mean_fraction",
    "left_correct_true",
    "left_correct_false",
    "left_correct_unavailable",
    "later_recovery_true",
    "later_recovery_false",
    "later_recovery_not_applicable",
    "later_recovery_unavailable",
    "endpoint_agreement_true",
    "endpoint_agreement_false",
    "endpoint_agreement_unavailable",
    "stabilization_computed",
    "stabilization_unavailable",
    "stabilization_mean_fraction",
)
RELIABILITY_COLUMNS = (
    "calibration_family",
    "analysis_label",
    "feature",
    "predictor",
    "target",
    "cohort_definition",
    "grouping",
    "subject",
    "requested_fraction",
    "is_main_checkpoint",
    "bin_index",
    "bin_lower",
    "bin_upper",
    "upper_inclusive",
    "count",
    "mean_confidence",
    "empirical_accuracy",
    "absolute_gap",
    "weighted_ece_contribution",
)
WITHIN_SUMMARY_COLUMNS = (
    "analysis_label",
    "feature",
    "target",
    "cohort_definition",
    "qualifying_question_count",
    "mean_paired_difference",
    "median_paired_difference",
    *INTERVAL_COLUMNS,
)
WITHIN_DISTRIBUTION_COLUMNS = (
    "analysis_label",
    "study_id",
    "model_run_id",
    "subject",
    "question_id",
    "feature",
    "target",
    "correct_run_count",
    "incorrect_run_count",
    "correct_run_mean",
    "incorrect_run_mean",
    "paired_difference",
)

TABLE_SPECS: dict[str, tuple[str, tuple[str, ...]]] = {
    "trajectory_features": ("trajectory_features.csv", TRAJECTORY_FEATURE_COLUMNS),
    "trajectory_events": ("trajectory_events.csv", TRAJECTORY_EVENT_COLUMNS),
    "primary_auroc": ("primary_auroc.csv", METRIC_COLUMNS),
    "secondary_checkpoint_auroc": (
        "secondary_checkpoint_auroc.csv",
        (*METRIC_COLUMNS, "requested_fraction", "is_main_checkpoint"),
    ),
    "calibration_metrics": (
        "calibration_metrics.csv",
        ("calibration_family", *METRIC_COLUMNS, "requested_fraction", "is_main_checkpoint"),
    ),
    "reliability_bins": ("reliability_bins.csv", RELIABILITY_COLUMNS),
    "within_question_summary": (
        "within_question_summary.csv",
        WITHIN_SUMMARY_COLUMNS,
    ),
    "within_question_distribution": (
        "within_question_distribution.csv",
        WITHIN_DISTRIBUTION_COLUMNS,
    ),
}
PLOT_FILENAMES = (
    "primary_auroc.png",
    "checkpoint_ece.png",
    "natural_reliability.png",
    "checkpoint_reliability_main.png",
    "within_question_paired_differences.png",
    "switching_stabilization.png",
)
EXPECTED_ARTIFACT_NAMES = (
    "analysis_manifest.json",
    "analysis_summary.json",
    *tuple(filename for filename, _columns in TABLE_SPECS.values()),
    *tuple(
        f"{filename.removesuffix('.csv')}.metadata.json"
        for filename, _columns in TABLE_SPECS.values()
    ),
    *PLOT_FILENAMES,
)


@dataclass(frozen=True, slots=True)
class AnalysisSource:
    repository_root: Path
    model_manifest: Mapping[str, Any]
    merge_manifest: Mapping[str, Any]
    coverage_report: Mapping[str, Any]
    analysis_config: Mapping[str, Any]
    question_frame: tuple[Mapping[str, Any], ...]
    natural_rows: tuple[Mapping[str, Any], ...]
    checkpoint_rows: tuple[Mapping[str, Any], ...]
    small_fixture: bool
    revalidate_inputs: Callable[[], None]


@dataclass(frozen=True, slots=True)
class AnalysisComputation:
    analysis_id: str
    bootstrap_mode: str
    trajectory_rows: tuple[dict[str, Any], ...]
    analyses: Mapping[str, Mapping[str, Any]]
    tables: Mapping[str, tuple[dict[str, Any], ...]]
    summary: Mapping[str, Any]


def _plain_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _domain_hash(identity_type: str, version: str, payload: Mapping[str, Any]) -> str:
    return _sha256(
        canonical_json_bytes(
            {
                "identity_type": identity_type,
                "identity_version": version,
                "payload": dict(payload),
            }
        )
    )


def validate_analysis_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config, Mapping) or canonical_json_bytes(
        dict(config)
    ) != canonical_json_bytes(ANALYSIS_CONFIG_ORACLE):
        raise ValueError("analysis config differs from the fixed executable oracle")


def analysis_config_hash(config: Mapping[str, Any]) -> str:
    validate_analysis_config(config)
    return _domain_hash(
        "analysis_config_hash", "part1-analysis-config-hash-v1", dict(config)
    )


def _same_json(left: Any, right: Any) -> bool:
    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


def _terminal_failure_count(coverage: Mapping[str, Any]) -> int:
    summary = coverage.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("coverage summary is missing")
    natural = summary.get("natural_partition")
    checkpoint = summary.get("checkpoint_partition")
    if not isinstance(natural, Mapping) or not isinstance(checkpoint, Mapping):
        raise ValueError("coverage terminal partitions are missing")
    values = (
        natural.get("terminal_infrastructure_failure"),
        checkpoint.get("terminal_infrastructure_failure"),
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("coverage terminal failure counts must be nonnegative integers")
    return int(values[0] + values[1])


def _validate_source_contract(source: AnalysisSource, bootstrap_replicates: int) -> None:
    if type(source.small_fixture) is not bool:
        raise ValueError("small_fixture must be boolean")
    if type(bootstrap_replicates) is not int or bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be a positive integer")
    if not source.small_fixture and bootstrap_replicates not in PRODUCTION_BOOTSTRAP_REPLICATES:
        raise ValueError("production bootstrap replicates must be exactly 1000 or 5000")
    validate_analysis_config(source.analysis_config)
    model = source.model_manifest
    merge = source.merge_manifest
    coverage = source.coverage_report
    for field in (
        "study_id",
        "study_manifest_hash",
        "question_manifest_hash",
        "model_run_id",
        "model_run_manifest_hash",
    ):
        if not _same_json(model.get(field), merge.get(field)):
            raise ValueError(f"analysis model/merge provenance differs for {field}")
    if not _same_json(
        merge.get("coverage_report_id"), coverage.get("validation_report_id")
    ):
        raise ValueError("analysis merge/coverage provenance differs")
    if coverage.get("paper_analysis_ready") is not True or _terminal_failure_count(
        coverage
    ) != 0:
        raise ValueError(
            "paper-final analysis requires paper_analysis_ready and zero terminal "
            "infrastructure failures"
        )


def _mode_and_child(replicates: int) -> tuple[str, str]:
    mode = "final" if replicates == 5_000 else "development"
    return mode, f"{mode}-r{replicates}"


def _analysis_identity_payload(
    source: AnalysisSource, *, bootstrap_replicates: int, bootstrap_mode: str
) -> dict[str, Any]:
    return {
        "study_id": source.model_manifest["study_id"],
        "study_manifest_hash": source.model_manifest["study_manifest_hash"],
        "question_manifest_hash": source.model_manifest["question_manifest_hash"],
        "model_run_id": source.model_manifest["model_run_id"],
        "model_run_manifest_hash": source.model_manifest["model_run_manifest_hash"],
        "merge_id": source.merge_manifest["merge_id"],
        "merge_manifest_hash": source.merge_manifest["merge_manifest_hash"],
        "coverage_report_id": source.coverage_report["validation_report_id"],
        "analysis_config_hash": analysis_config_hash(source.analysis_config),
        "analysis_contract_version": source.analysis_config[
            "analysis_contract_version"
        ],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_mode": bootstrap_mode,
        "analysis_format_version": ANALYSIS_FORMAT_VERSION,
        "csv_serialization_version": CSV_SERIALIZATION_VERSION,
        "plot_version": PLOT_VERSION,
    }


def analysis_id(
    source: AnalysisSource, *, bootstrap_replicates: int, bootstrap_mode: str
) -> str:
    return _domain_hash(
        "analysis_id",
        ANALYSIS_IDENTITY_VERSION,
        _analysis_identity_payload(
            source,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_mode=bootstrap_mode,
        ),
    )


def _event_summary_row(
    rows: Sequence[Mapping[str, Any]], *, grouping: str, subject: str | None
) -> dict[str, Any]:
    switch_values = [
        row["answer_switch_count"]
        for row in rows
        if row.get("transition_evaluability_status") == "evaluated"
    ]
    appearances = [
        row["first_natural_answer_appearance_fraction"]
        for row in rows
        if row.get("first_natural_answer_appearance_status") == "found"
    ]
    stabilizations = [
        row["stabilization_fraction"]
        for row in rows
        if row.get("stabilization_status") == "computed"
    ]
    count_status = lambda field, value: sum(row.get(field) == value for row in rows)
    return {
        "grouping": grouping,
        "subject": subject,
        "trajectory_count": len(rows),
        "switch_count_available": len(switch_values),
        "switch_count_unavailable": len(rows) - len(switch_values),
        "switch_count_sum": int(sum(switch_values)),
        "switch_count_mean": (
            None if not switch_values else float(sum(switch_values) / len(switch_values))
        ),
        "first_appearance_found": count_status(
            "first_natural_answer_appearance_status", "found"
        ),
        "first_appearance_not_found": count_status(
            "first_natural_answer_appearance_status", "not_found"
        ),
        "first_appearance_unavailable": count_status(
            "first_natural_answer_appearance_status", "unavailable"
        ),
        "first_appearance_mean_fraction": (
            None if not appearances else float(sum(appearances) / len(appearances))
        ),
        "left_correct_true": sum(row.get("left_correct_answer") is True for row in rows),
        "left_correct_false": sum(row.get("left_correct_answer") is False for row in rows),
        "left_correct_unavailable": count_status("left_correct_answer_status", "unavailable"),
        "later_recovery_true": sum(
            row.get("later_recovered_correct_answer") is True for row in rows
        ),
        "later_recovery_false": sum(
            row.get("later_recovered_correct_answer") is False
            and row.get("later_recovered_correct_answer_status") == "evaluated"
            for row in rows
        ),
        "later_recovery_not_applicable": count_status(
            "later_recovered_correct_answer_status", "not_applicable"
        ),
        "later_recovery_unavailable": count_status(
            "later_recovered_correct_answer_status", "unavailable"
        ),
        "endpoint_agreement_true": sum(
            row.get("forced_endpoint_agrees_with_natural") is True for row in rows
        ),
        "endpoint_agreement_false": sum(
            row.get("forced_endpoint_agrees_with_natural") is False for row in rows
        ),
        "endpoint_agreement_unavailable": count_status(
            "forced_endpoint_agrees_with_natural_status", "unavailable"
        ),
        "stabilization_computed": count_status("stabilization_status", "computed"),
        "stabilization_unavailable": count_status("stabilization_status", "unavailable"),
        "stabilization_mean_fraction": (
            None
            if not stabilizations
            else float(sum(stabilizations) / len(stabilizations))
        ),
    }


def summarize_trajectory_events(
    trajectory_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = [_event_summary_row(trajectory_rows, grouping="pooled", subject=None)]
    for subject in FIXED_SUBJECTS:
        output.append(
            _event_summary_row(
                [row for row in trajectory_rows if row.get("subject") == subject],
                grouping="subject",
                subject=subject,
            )
        )
    return output


def _calibration_family(predictor: str) -> str:
    if predictor == "natural_verbalized_confidence":
        return "natural_confidence"
    if predictor == "checkpoint_normalized_confidence":
        return "checkpoint_confidence"
    if predictor == "checkpoint_maximum_ad_probability":
        return "maximum_ad_probability"
    raise ValueError(f"unsupported calibration predictor: {predictor}")


def _with_calibration_metadata(
    rows: Sequence[Mapping[str, Any]], *, checkpoint: bool
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        row["calibration_family"] = _calibration_family(str(row["predictor"]))
        if not checkpoint:
            row["requested_fraction"] = None
            row["is_main_checkpoint"] = None
        output.append(row)
    return output


def _plot_series(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    primary = [
        {
            "feature": row["feature"],
            "point_estimate": row["point_estimate"],
            "lower": row["lower"],
            "upper": row["upper"],
        }
        for row in tables["primary_auroc"]
        if row["grouping"] == "pooled"
    ]
    checkpoint_ece = [
        {
            "calibration_family": row["calibration_family"],
            "requested_fraction": row["requested_fraction"],
            "is_main_checkpoint": row["is_main_checkpoint"],
            "point_estimate": row["point_estimate"],
        }
        for row in tables["calibration_metrics"]
        if row["grouping"] == "pooled" and row["requested_fraction"] is not None
    ]
    natural_reliability = [
        {
            "bin_index": row["bin_index"],
            "mean_confidence": row["mean_confidence"],
            "empirical_accuracy": row["empirical_accuracy"],
            "count": row["count"],
        }
        for row in tables["reliability_bins"]
        if row["grouping"] == "pooled"
        and row["calibration_family"] == "natural_confidence"
    ]
    checkpoint_reliability = [
        {
            "calibration_family": row["calibration_family"],
            "requested_fraction": row["requested_fraction"],
            "bin_index": row["bin_index"],
            "mean_confidence": row["mean_confidence"],
            "empirical_accuracy": row["empirical_accuracy"],
            "count": row["count"],
        }
        for row in tables["reliability_bins"]
        if row["grouping"] == "pooled" and row["is_main_checkpoint"] is True
    ]
    within = [
        {
            "feature": row["feature"],
            "mean_paired_difference": row["mean_paired_difference"],
            "median_paired_difference": row["median_paired_difference"],
        }
        for row in tables["within_question_summary"]
    ]
    events = [dict(row) for row in tables["trajectory_events"]]
    return {
        "primary_auroc": primary,
        "checkpoint_ece": checkpoint_ece,
        "natural_reliability": natural_reliability,
        "checkpoint_reliability_main": checkpoint_reliability,
        "within_question": within,
        "switching_stabilization": events,
    }


def compute_analysis(
    source: AnalysisSource, *, bootstrap_replicates: int
) -> AnalysisComputation:
    """Run the fixed analyses once from already validated immutable source rows."""

    _validate_source_contract(source, bootstrap_replicates)
    trajectory_rows = build_trajectory_rows(source.natural_rows, source.checkpoint_rows)
    if len(trajectory_rows) != len(source.natural_rows):
        raise ValueError("trajectory construction must produce one row per natural key")
    draw_plan = build_question_draw_plan(
        source.question_frame,
        replicates=bootstrap_replicates,
        seed=BOOTSTRAP_SEED,
        small_fixture=source.small_fixture,
    )
    fixture_options = {"allow_small_fixture": source.small_fixture}
    primary = primary_auroc_analysis(trajectory_rows, draw_plan, **fixture_options)
    natural = natural_calibration_analysis(trajectory_rows, draw_plan, **fixture_options)
    checkpoint = checkpoint_calibration_analysis(
        trajectory_rows, draw_plan, **fixture_options
    )
    secondary = secondary_checkpoint_auroc_analysis(
        trajectory_rows, draw_plan, **fixture_options
    )
    within = within_question_analysis(trajectory_rows, draw_plan, **fixture_options)
    analyses = {
        "primary_auroc": primary,
        "natural_calibration": natural,
        "checkpoint_calibration": checkpoint,
        "secondary_checkpoint_auroc": secondary,
        "within_question": within,
    }
    calibration_metrics = _with_calibration_metadata(
        natural["metric_rows"], checkpoint=False
    ) + _with_calibration_metadata(checkpoint["metric_rows"], checkpoint=True)
    reliability = _with_calibration_metadata(
        natural["reliability_rows"], checkpoint=False
    ) + _with_calibration_metadata(checkpoint["reliability_rows"], checkpoint=True)
    tables: dict[str, tuple[dict[str, Any], ...]] = {
        "trajectory_features": tuple(dict(row) for row in trajectory_rows),
        "trajectory_events": tuple(summarize_trajectory_events(trajectory_rows)),
        "primary_auroc": tuple(dict(row) for row in primary["metric_rows"]),
        "secondary_checkpoint_auroc": tuple(
            dict(row) for row in secondary["metric_rows"]
        ),
        "calibration_metrics": tuple(calibration_metrics),
        "reliability_bins": tuple(reliability),
        "within_question_summary": tuple(dict(row) for row in within["summary_rows"]),
        "within_question_distribution": tuple(
            dict(row) for row in within["distribution_rows"]
        ),
    }
    mode, child = _mode_and_child(bootstrap_replicates)
    identity = analysis_id(
        source, bootstrap_replicates=bootstrap_replicates, bootstrap_mode=mode
    )
    output_root = source.model_manifest["output_paths"]["analysis"]
    summary = {
        "schema_name": "part1_analysis_summary",
        "schema_version": "1.0.0",
        "analysis_id": identity,
        **_analysis_identity_payload(
            source,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_mode=mode,
        ),
        "source_natural_row_count": len(source.natural_rows),
        "source_checkpoint_row_count": len(source.checkpoint_rows),
        "trajectory_row_count": len(trajectory_rows),
        "paper_analysis_ready": source.coverage_report["paper_analysis_ready"],
        "terminal_infrastructure_failure_count": _terminal_failure_count(
            source.coverage_report
        ),
        "repetition_filter_applied": False,
        "successful_abnormal_output_policy": "preserved",
        "evidence_scope": FIXTURE_EVIDENCE_LIMITS if source.small_fixture else "production",
        "analysis_directory": f"{output_root}/{child}",
        "output_tables": {
            key: f"{output_root}/{child}/{filename}"
            for key, (filename, _columns) in TABLE_SPECS.items()
        },
        "output_plots": [f"{output_root}/{child}/{name}" for name in PLOT_FILENAMES],
        "primary_main_rows": [
            dict(row)
            for row in primary["metric_rows"]
            if row["grouping"] in {"pooled", "macro"}
        ],
        "natural_calibration_main_rows": [
            dict(row)
            for row in calibration_metrics
            if row["calibration_family"] == "natural_confidence"
            and row["grouping"] in {"pooled", "macro"}
        ],
        "checkpoint_calibration_predictor_families": [
            "checkpoint_confidence",
            "maximum_ad_probability",
        ],
        "checkpoint_calibration_all_fractions": list(FIXED_CHECKPOINT_FRACTIONS),
        "checkpoint_calibration_main_rows": [
            dict(row)
            for row in calibration_metrics
            if row["requested_fraction"] in MAIN_CHECKPOINT_FRACTIONS
            and row["grouping"] in {"pooled", "macro"}
        ],
        "all_checkpoint_fractions_table": TABLE_SPECS["calibration_metrics"][0],
        "within_question_summaries": [dict(row) for row in within["summary_rows"]],
        "switching_stabilization_summaries": [
            dict(row) for row in tables["trajectory_events"]
        ],
        "plot_series": _plot_series(tables),
    }
    return AnalysisComputation(
        analysis_id=identity,
        bootstrap_mode=mode,
        trajectory_rows=tuple(dict(row) for row in trajectory_rows),
        analyses=analyses,
        tables=tables,
        summary=summary,
    )


def _encode_csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return _plain_json_bytes(value).decode("utf-8")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("analysis CSV cannot encode non-finite values")
        return json.dumps(value, allow_nan=False, separators=(",", ":"))
    if isinstance(value, str):
        return value
    raise ValueError(f"analysis CSV cannot encode {type(value).__name__}")


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(columns),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    expected = set(columns)
    for row in rows:
        if set(row) != expected:
            missing = sorted(expected.difference(row))
            extra = sorted(set(row).difference(expected))
            raise ValueError(f"analysis table row fields differ: missing={missing}; extra={extra}")
        writer.writerow({column: _encode_csv_cell(row[column]) for column in columns})
    return buffer.getvalue().encode("utf-8")


def _source_provenance(source: AnalysisSource) -> dict[str, Any]:
    return {
        "study_id": source.model_manifest["study_id"],
        "study_manifest_hash": source.model_manifest["study_manifest_hash"],
        "question_manifest_hash": source.model_manifest["question_manifest_hash"],
        "model_run_id": source.model_manifest["model_run_id"],
        "model_run_manifest_hash": source.model_manifest["model_run_manifest_hash"],
        "merge_id": source.merge_manifest["merge_id"],
        "merge_manifest_hash": source.merge_manifest["merge_manifest_hash"],
        "coverage_report_id": source.coverage_report["validation_report_id"],
    }


def _write_fsynced(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _render_plots(stage: Path, computation: AnalysisComputation) -> None:
    series = computation.summary["plot_series"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "figure.dpi": 100,
            "savefig.dpi": 100,
        }
    )

    def save(filename: str, draw: Callable[[Any], None]) -> None:
        figure, axes = plt.subplots(figsize=(7, 4))
        draw(axes)
        figure.tight_layout()
        figure.savefig(
            stage / filename,
            format="png",
            metadata={"Software": PLOT_VERSION},
        )
        plt.close(figure)
        descriptor = os.open(stage / filename, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def numeric(value: Any) -> float:
        return float("nan") if value is None else float(value)

    save(
        "primary_auroc.png",
        lambda axes: (
            axes.plot(
                range(len(series["primary_auroc"])),
                [numeric(row["point_estimate"]) for row in series["primary_auroc"]],
                marker="o",
            ),
            axes.set_title("Primary pooled AUROC"),
            axes.set_xlabel("Fixed feature order"),
            axes.set_ylabel("AUROC"),
            axes.set_ylim(0.0, 1.0),
        ),
    )

    def checkpoint_ece(axes: Any) -> None:
        for family in ("checkpoint_confidence", "maximum_ad_probability"):
            rows = [
                row
                for row in series["checkpoint_ece"]
                if row["calibration_family"] == family
            ]
            axes.plot(
                [row["requested_fraction"] for row in rows],
                [numeric(row["point_estimate"]) for row in rows],
                marker="o",
                label=family,
            )
            main = [row for row in rows if row["is_main_checkpoint"]]
            axes.scatter(
                [row["requested_fraction"] for row in main],
                [numeric(row["point_estimate"]) for row in main],
                marker="D",
            )
        axes.set_title("Checkpoint ECE at all requested fractions")
        axes.set_xlabel("Requested fraction")
        axes.set_ylabel("ECE")
        axes.legend()

    save("checkpoint_ece.png", checkpoint_ece)

    def reliability_plot(axes: Any, rows: Sequence[Mapping[str, Any]], title: str) -> None:
        axes.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black")
        available = [
            row
            for row in rows
            if row["mean_confidence"] is not None and row["empirical_accuracy"] is not None
        ]
        axes.scatter(
            [row["mean_confidence"] for row in available],
            [row["empirical_accuracy"] for row in available],
        )
        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(0.0, 1.0)
        axes.set_title(title)
        axes.set_xlabel("Mean confidence")
        axes.set_ylabel("Empirical accuracy")

    save(
        "natural_reliability.png",
        lambda axes: reliability_plot(
            axes, series["natural_reliability"], "Natural confidence reliability"
        ),
    )
    save(
        "checkpoint_reliability_main.png",
        lambda axes: reliability_plot(
            axes,
            series["checkpoint_reliability_main"],
            "Checkpoint reliability at main fractions",
        ),
    )
    save(
        "within_question_paired_differences.png",
        lambda axes: (
            axes.axhline(0.0, color="black", linewidth=0.8),
            axes.bar(
                range(len(series["within_question"])),
                [
                    numeric(row["mean_paired_difference"])
                    for row in series["within_question"]
                ],
            ),
            axes.set_title("Within-question paired differences"),
            axes.set_xlabel("Fixed feature order"),
            axes.set_ylabel("Correct mean - incorrect mean"),
        ),
    )
    save(
        "switching_stabilization.png",
        lambda axes: (
            axes.bar(
                range(len(series["switching_stabilization"])),
                [
                    numeric(row["switch_count_mean"])
                    for row in series["switching_stabilization"]
                ],
                label="mean switches",
            ),
            axes.plot(
                range(len(series["switching_stabilization"])),
                [
                    numeric(row["stabilization_mean_fraction"])
                    for row in series["switching_stabilization"]
                ],
                marker="o",
                label="mean stabilization fraction",
            ),
            axes.set_title("Switching and stabilization summary"),
            axes.set_xlabel("Pooled then fixed subject order"),
            axes.legend(),
        ),
    )


def _artifact_entry(path: Path, *, kind: str) -> dict[str, Any]:
    data = path.read_bytes()
    return {"kind": kind, "sha256": _sha256(data), "byte_size": len(data)}


def _analysis_manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload["analysis_manifest_hash"] = ""
    return _domain_hash(
        "analysis_manifest_hash", ANALYSIS_MANIFEST_HASH_VERSION, payload
    )


def validate_analysis_manifest(manifest: Mapping[str, Any]) -> None:
    expected_fields = {
        "schema_name",
        "schema_version",
        "analysis_id",
        "analysis_manifest_hash",
        "study_id",
        "study_manifest_hash",
        "question_manifest_hash",
        "model_run_id",
        "model_run_manifest_hash",
        "merge_id",
        "merge_manifest_hash",
        "coverage_report_id",
        "analysis_config_hash",
        "analysis_contract_version",
        "bootstrap_seed",
        "bootstrap_replicates",
        "bootstrap_mode",
        "analysis_format_version",
        "csv_serialization_version",
        "plot_version",
        "paper_analysis_ready",
        "terminal_infrastructure_failure_count",
        "repetition_filter_applied",
        "tables",
        "plots",
        "artifacts",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_fields:
        raise ValueError("analysis manifest fields differ from the fixed contract")
    if (
        manifest["schema_name"] != "part1_analysis_manifest"
        or manifest["schema_version"] != "1.0.0"
        or manifest["analysis_contract_version"] != "part1-analysis-v1"
        or manifest["analysis_format_version"] != ANALYSIS_FORMAT_VERSION
        or manifest["csv_serialization_version"] != CSV_SERIALIZATION_VERSION
        or manifest["plot_version"] != PLOT_VERSION
        or type(manifest["bootstrap_seed"]) is not int
        or manifest["bootstrap_seed"] != BOOTSTRAP_SEED
        or type(manifest["bootstrap_replicates"]) is not int
        or manifest["bootstrap_replicates"] <= 0
        or manifest["bootstrap_mode"]
        != ("final" if manifest["bootstrap_replicates"] == 5_000 else "development")
        or manifest["paper_analysis_ready"] is not True
        or type(manifest["terminal_infrastructure_failure_count"]) is not int
        or manifest["terminal_infrastructure_failure_count"] != 0
        or manifest["repetition_filter_applied"] is not False
    ):
        raise ValueError("analysis manifest fixed semantics differ")
    identity_fields = (
        "analysis_id",
        "analysis_manifest_hash",
        "study_id",
        "study_manifest_hash",
        "question_manifest_hash",
        "model_run_id",
        "model_run_manifest_hash",
        "merge_id",
        "merge_manifest_hash",
        "coverage_report_id",
        "analysis_config_hash",
    )
    if any(
        not isinstance(manifest[field], str)
        or len(manifest[field]) != 64
        or any(character not in "0123456789abcdef" for character in manifest[field])
        for field in identity_fields
    ):
        raise ValueError("analysis manifest contains an invalid SHA-256 identity")
    identity_payload = {
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
            "analysis_config_hash",
            "analysis_contract_version",
            "bootstrap_seed",
            "bootstrap_replicates",
            "bootstrap_mode",
            "analysis_format_version",
            "csv_serialization_version",
            "plot_version",
        )
    }
    if manifest["analysis_id"] != _domain_hash(
        "analysis_id", ANALYSIS_IDENTITY_VERSION, identity_payload
    ):
        raise ValueError("analysis identity does not recompute")
    if manifest["analysis_manifest_hash"] != _analysis_manifest_hash(manifest):
        raise ValueError("analysis manifest hash does not recompute")
    expected_tables = {
        filename: f"{filename.removesuffix('.csv')}.metadata.json"
        for filename, _columns in TABLE_SPECS.values()
    }
    if manifest["tables"] != expected_tables or manifest["plots"] != list(
        PLOT_FILENAMES
    ):
        raise ValueError("analysis table/plot inventory differs")
    expected_artifacts = set(EXPECTED_ARTIFACT_NAMES).difference(
        {"analysis_manifest.json"}
    )
    if not isinstance(manifest["artifacts"], Mapping) or set(
        manifest["artifacts"]
    ) != expected_artifacts:
        raise ValueError("analysis artifact inventory is incomplete")
    for name, entry in manifest["artifacts"].items():
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"kind", "sha256", "byte_size"}
            or entry["kind"]
            not in {"table", "table_metadata", "plot", "summary"}
            or not isinstance(entry["sha256"], str)
            or len(entry["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in entry["sha256"])
            or type(entry["byte_size"]) is not int
            or entry["byte_size"] <= 0
        ):
            raise ValueError(f"analysis artifact inventory entry is invalid: {name}")


def _write_stage(
    stage: Path, source: AnalysisSource, computation: AnalysisComputation
) -> dict[str, Any]:
    _write_fsynced(stage / "analysis_summary.json", _plain_json_bytes(computation.summary))
    tables: dict[str, str] = {}
    nested_columns_by_table = {
        "trajectory_features": ["checkpoint_calibration", "feature_missing_reasons"]
    }
    for table_name, (filename, columns) in TABLE_SPECS.items():
        rows = computation.tables[table_name]
        data = _csv_bytes(rows, columns)
        _write_fsynced(stage / filename, data)
        metadata_name = f"{filename.removesuffix('.csv')}.metadata.json"
        metadata = {
            "schema_name": "part1_analysis_table_metadata",
            "schema_version": "1.0.0",
            "serialization_version": CSV_SERIALIZATION_VERSION,
            "table_name": table_name,
            "table_filename": filename,
            "table_sha256": _sha256(data),
            "table_byte_size": len(data),
            "row_count": len(rows),
            "ordered_columns": list(columns),
            "nested_json_columns": nested_columns_by_table.get(table_name, []),
            "source_provenance": _source_provenance(source),
            "analysis_id": computation.analysis_id,
            "analysis_contract_version": source.analysis_config[
                "analysis_contract_version"
            ],
            "analysis_config_hash": analysis_config_hash(source.analysis_config),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": computation.summary["bootstrap_replicates"],
            "bootstrap_mode": computation.bootstrap_mode,
        }
        _write_fsynced(stage / metadata_name, _plain_json_bytes(metadata))
        tables[filename] = metadata_name
    _render_plots(stage, computation)
    artifacts: dict[str, dict[str, Any]] = {}
    for name in sorted(set(EXPECTED_ARTIFACT_NAMES).difference({"analysis_manifest.json"})):
        kind = (
            "table"
            if name.endswith(".csv")
            else "table_metadata"
            if name.endswith(".metadata.json")
            else "plot"
            if name.endswith(".png")
            else "summary"
        )
        artifacts[name] = _artifact_entry(stage / name, kind=kind)
    manifest: dict[str, Any] = {
        "schema_name": "part1_analysis_manifest",
        "schema_version": "1.0.0",
        "analysis_id": computation.analysis_id,
        "analysis_manifest_hash": "",
        **_analysis_identity_payload(
            source,
            bootstrap_replicates=computation.summary["bootstrap_replicates"],
            bootstrap_mode=computation.bootstrap_mode,
        ),
        "paper_analysis_ready": True,
        "terminal_infrastructure_failure_count": 0,
        "repetition_filter_applied": False,
        "tables": tables,
        "plots": list(PLOT_FILENAMES),
        "artifacts": artifacts,
    }
    manifest["analysis_manifest_hash"] = _analysis_manifest_hash(manifest)
    _write_fsynced(stage / "analysis_manifest.json", _plain_json_bytes(manifest))
    return manifest


def _read_regular_at(directory_descriptor: int, name: str) -> bytes:
    if Path(name).name != name:
        raise ValueError(f"unsafe analysis artifact name: {name}")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
    except OSError as exc:
        raise ValueError(
            f"analysis artifact is missing, symlinked, or unreadable: {name}"
        ) from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(f"analysis artifact is nonregular: {name}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _json_object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict) or data != _plain_json_bytes(value):
        raise ValueError(f"{label} is not a canonical JSON object")
    return value


def _validate_analysis_directory_descriptor(
    directory_descriptor: int,
    *,
    expected_manifest: Mapping[str, Any] | None = None,
    expected_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    entries = os.listdir(directory_descriptor)
    if set(entries) != set(EXPECTED_ARTIFACT_NAMES) or len(entries) != len(
        EXPECTED_ARTIFACT_NAMES
    ):
        raise ValueError("analysis directory has missing or extra contents")
    manifest = _json_object(
        _read_regular_at(directory_descriptor, "analysis_manifest.json"),
        label="analysis manifest",
    )
    validate_analysis_manifest(manifest)
    if expected_manifest is not None and not _same_json(manifest, dict(expected_manifest)):
        raise ValueError("analysis directory differs from the expected manifest")
    for name, entry in manifest["artifacts"].items():
        data = _read_regular_at(directory_descriptor, name)
        if (
            entry["sha256"] != _sha256(data)
            or type(entry["byte_size"]) is not int
            or entry["byte_size"] != len(data)
            or len(data) == 0
        ):
            raise ValueError(f"analysis artifact bytes differ for {name}")
    summary = _json_object(
        _read_regular_at(directory_descriptor, "analysis_summary.json"),
        label="analysis summary",
    )
    if expected_summary is not None and not _same_json(summary, dict(expected_summary)):
        raise ValueError("analysis summary differs from the computed result")
    for field in (
        "analysis_id",
        "study_id",
        "study_manifest_hash",
        "question_manifest_hash",
        "model_run_id",
        "model_run_manifest_hash",
        "merge_id",
        "merge_manifest_hash",
        "coverage_report_id",
        "analysis_config_hash",
        "bootstrap_seed",
        "bootstrap_replicates",
        "bootstrap_mode",
    ):
        if not _same_json(summary.get(field), manifest.get(field)):
            raise ValueError(f"analysis summary/manifest provenance differs for {field}")
    checkpoint_series = summary.get("plot_series", {}).get("checkpoint_ece")
    if not isinstance(checkpoint_series, list) or len(checkpoint_series) != 22:
        raise ValueError("checkpoint ECE plot series must contain both all-11 families")
    if {
        row.get("requested_fraction") for row in checkpoint_series
    } != set(FIXED_CHECKPOINT_FRACTIONS) or {
        row.get("requested_fraction")
        for row in checkpoint_series
        if row.get("is_main_checkpoint") is True
    } != set(MAIN_CHECKPOINT_FRACTIONS):
        raise ValueError("checkpoint ECE plot fractions or main markers differ")
    for table_name, (filename, columns) in TABLE_SPECS.items():
        data = _read_regular_at(directory_descriptor, filename)
        if data.startswith(b"#"):
            raise ValueError(f"analysis CSV contains a provenance comment: {filename}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"analysis CSV is not UTF-8: {filename}") from exc
        reader = csv.DictReader(io.StringIO(text, newline=""))
        rows = list(reader)
        if list(reader.fieldnames or ()) != list(columns):
            raise ValueError(f"analysis CSV columns differ: {filename}")
        if data.count(b"\n") != len(rows) + 1:
            raise ValueError(f"analysis CSV framing differs: {filename}")
        metadata_name = manifest["tables"][filename]
        metadata = _json_object(
            _read_regular_at(directory_descriptor, metadata_name),
            label=f"analysis table metadata {table_name}",
        )
        if (
            metadata.get("table_name") != table_name
            or metadata.get("table_filename") != filename
            or metadata.get("table_sha256") != _sha256(data)
            or metadata.get("table_byte_size") != len(data)
            or metadata.get("row_count") != len(rows)
            or metadata.get("ordered_columns") != list(columns)
            or metadata.get("serialization_version") != CSV_SERIALIZATION_VERSION
            or metadata.get("analysis_id") != manifest["analysis_id"]
            or metadata.get("analysis_config_hash") != manifest["analysis_config_hash"]
        ):
            raise ValueError(f"analysis table sidecar differs: {filename}")
        nested = metadata.get("nested_json_columns")
        if not isinstance(nested, list) or any(column not in columns for column in nested):
            raise ValueError(f"analysis table nested JSON metadata differs: {filename}")
        for row in rows:
            for column in nested:
                try:
                    json.loads(row[column])
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"analysis nested JSON cell is invalid: {filename}:{column}"
                    ) from exc
    for plot in PLOT_FILENAMES:
        data = _read_regular_at(directory_descriptor, plot)
        if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) <= 100:
            raise ValueError(f"analysis plot is not a nonempty PNG: {plot}")
    return manifest


def validate_analysis_directory(
    directory: Path,
    *,
    expected_manifest: Mapping[str, Any] | None = None,
    expected_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    directory = Path(os.path.abspath(directory))
    parent_descriptor = os.open(
        directory.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        try:
            descriptor = os.open(
                directory.name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise ValueError(
                f"analysis directory is missing, symlinked, or non-directory: {directory}"
            ) from exc
        try:
            return _validate_analysis_directory_descriptor(
                descriptor,
                expected_manifest=expected_manifest,
                expected_summary=expected_summary,
            )
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _ensure_directory(path: Path) -> None:
    missing: list[Path] = []
    current = path
    while not os.path.lexists(current):
        missing.append(current)
        current = current.parent
    _require_no_symlink_components(current)
    if not current.is_dir():
        raise ValueError(f"analysis publication ancestor is not a directory: {current}")
    for component in reversed(missing):
        component.mkdir()
        parent_descriptor = os.open(component.parent, os.O_RDONLY)
        try:
            _fsync_directory_descriptor(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    _require_no_symlink_components(path)


def publish_analysis(
    source: AnalysisSource,
    *,
    bootstrap_replicates: int,
    fault_hook: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Compute, stage, validate, and atomically publish one immutable analysis."""

    source.revalidate_inputs()
    computation = compute_analysis(source, bootstrap_replicates=bootstrap_replicates)
    _mode, child = _mode_and_child(bootstrap_replicates)
    analysis_relative = source.model_manifest["output_paths"]["analysis"]
    expected_analysis = f"results/part1/{source.model_manifest['model_run_id']}/analysis"
    if analysis_relative != expected_analysis:
        raise ValueError("production analysis output path is not canonical")
    root = Path(os.path.abspath(source.repository_root)) / analysis_relative
    _ensure_directory(root)
    target = root / child
    parent_descriptor = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    prefix = f".{child}.stage-"
    stage = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    stage_descriptor = os.open(
        stage.name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_descriptor,
    )
    hook = fault_hook or (lambda _boundary: None)
    published = False
    renamed = False
    try:
        hook("stage_created")
        manifest = _write_stage(stage, source, computation)
        hook("artifacts_written")
        _fsync_directory_descriptor(stage_descriptor)
        hook("stage_fsynced")
        validate_analysis_directory(
            stage,
            expected_manifest=manifest,
            expected_summary=computation.summary,
        )
        hook("reload_complete")
        hook("before_input_revalidation")
        hook("before_exclusive_rename")
        source.revalidate_inputs()
        if os.path.lexists(target):
            existing = validate_analysis_directory(
                target,
                expected_manifest=manifest,
                expected_summary=computation.summary,
            )
            return target, existing
        try:
            _exclusive_rename_at(parent_descriptor, stage.name, target.name)
        except FileExistsError:
            source.revalidate_inputs()
            existing = validate_analysis_directory(
                target,
                expected_manifest=manifest,
                expected_summary=computation.summary,
            )
            return target, existing
        renamed = True
        try:
            hook("after_exclusive_rename")
            _fsync_directory_descriptor(parent_descriptor)
        except BaseException as durability_error:
            try:
                _exclusive_rename_at(parent_descriptor, target.name, stage.name)
                renamed = False
                _fsync_directory_descriptor(parent_descriptor)
            except BaseException as rollback_error:
                published = True
                raise PublicationStateIndeterminateError(
                    "analysis post-rename durability failed and rollback was not durable; "
                    f"final path may remain at {target}: error={durability_error}; "
                    f"rollback={rollback_error}"
                ) from durability_error
            raise PublicationDurabilityError(
                f"analysis publication durability failed and was rolled back: {target}"
            ) from durability_error
        published = True
        return target, manifest
    finally:
        try:
            if not published and not renamed:
                _remove_own_stage(
                    stage,
                    root,
                    prefix,
                    parent_descriptor=parent_descriptor,
                    stage_descriptor=stage_descriptor,
                    fault_hook=hook,
                )
        finally:
            try:
                os.close(stage_descriptor)
            finally:
                os.close(parent_descriptor)


def _read_analysis_config(repository_root: Path) -> tuple[dict[str, Any], bytes, Path]:
    path = repository_root / "configs" / "part1" / "analysis.json"
    _require_no_symlink_components(path)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("analysis config must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"analysis config is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("analysis config must be a JSON object")
    validate_analysis_config(value)
    return value, data, path


def load_production_analysis_source(
    *, repository_root: Path, model_run_manifest_path: Path
) -> AnalysisSource:
    """Strictly load canonical production manifests and lossless merged rows."""

    repository_root = Path(os.path.abspath(repository_root))
    manifest_path = Path(model_run_manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = repository_root / manifest_path
    manifest_path = Path(os.path.abspath(manifest_path))
    model, model_bytes = _load_regular_json(
        manifest_path, label="production model-run manifest"
    )
    validate_instance("model_run_manifest", model)
    validate_fixed_model_requested_contract(model)
    if (
        model.get("schema_version") != "1.1.0"
        or model.get("production") is not True
        or model.get("execution_scope") != "production"
        or model.get("clean_tracked_worktree") is not True
        or model.get("model_run_id") != model_run_id(model)
        or model.get("model_run_manifest_hash") != model_run_manifest_hash(model)
    ):
        raise ValueError("analysis requires a canonical production schema-1.1 manifest")
    expected_manifest_path = (
        repository_root
        / "results"
        / "part1"
        / model["model_run_id"]
        / "model_run_manifest.json"
    )
    if manifest_path != expected_manifest_path:
        raise ValueError("production model-run manifest path is not canonical")
    expected_paths = {
        key: f"results/part1/{model['model_run_id']}/{key if key != 'raw_shards' else 'raw_shards'}"
        for key in ("raw_shards", "validation", "merged", "analysis")
    }
    if not _same_json(model.get("output_paths"), expected_paths):
        raise ValueError("production output paths are not canonical")

    config, config_bytes, config_path = _read_analysis_config(repository_root)
    merged_path = repository_root / model["output_paths"]["merged"]
    merge = validate_merge_directory(merged_path)
    for field in (
        "study_id",
        "study_manifest_hash",
        "question_manifest_hash",
        "model_run_id",
        "model_run_manifest_hash",
    ):
        if not _same_json(model.get(field), merge.get(field)):
            raise ValueError(f"model/merge provenance differs for {field}")

    coverage_relative = merge["coverage_report"]["relative_path"]
    expected_coverage_relative = f"results/part1/{model['model_run_id']}/validation/coverage_report.json"
    if coverage_relative != expected_coverage_relative:
        raise ValueError("merge coverage report path is not canonical")
    coverage_path = repository_root / coverage_relative
    coverage, coverage_bytes = _load_regular_json(coverage_path, label="coverage report")
    if (
        _sha256(coverage_bytes) != merge["coverage_report"]["sha256"]
        or len(coverage_bytes) != merge["coverage_report"]["byte_size"]
        or coverage.get("validation_report_id") != merge.get("coverage_report_id")
        or coverage.get("validation_report_id") != coverage_report_id(coverage)
    ):
        raise ValueError("coverage bytes or identity differ from merge provenance")
    validate_instance("validation_report", coverage)
    validate_coverage_report_semantics(coverage)
    if coverage.get("paper_analysis_ready") is not True or _terminal_failure_count(
        coverage
    ) != 0:
        raise ValueError(
            "paper-final analysis requires paper_analysis_ready and zero terminal "
            "infrastructure failures"
        )
    rebuilt = build_coverage_report(
        repository_root=repository_root,
        model_run_manifest_path=manifest_path,
        validation_started_at=coverage["validation_started_at"],
        validation_completed_at=coverage["validation_completed_at"],
    )
    if not _same_json(rebuilt, coverage):
        raise ValueError("coverage report differs from current immutable source/Git snapshot")

    bundle = load_manifest_bundle(
        questions_path=repository_root / "manifests/part1/questions.jsonl",
        question_manifest_path=repository_root
        / "manifests/part1/questions.manifest.json",
        study_manifest_path=repository_root / "manifests/part1/study_manifest.json",
    )
    validate_manifest_compatibility(bundle.study_manifest, model)
    if len(bundle.records) != 500 or [row["sample_index"] for row in bundle.records] != list(
        range(500)
    ):
        raise ValueError("tracked question bundle is not the fixed ordered 500-question frame")

    parent_descriptor = os.open(
        merged_path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        validate_merge_directory_at(
            parent_descriptor, merged_path.name, expected_manifest=merge
        )
        merged_descriptor = os.open(
            merged_path.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        try:
            recovered: dict[str, tuple[dict[str, Any], ...]] = {}
            for kind in ("natural_results", "checkpoint_results"):
                data = _read_regular_file_at(
                    merged_descriptor, MERGE_TABLE_FILENAMES[kind]
                )
                output = merge["outputs"][kind]
                if len(data) != output["byte_size"] or _sha256(data) != output["sha256"]:
                    raise ValueError(f"merged {kind} bytes differ during analysis read")
                table = pq.read_table(pa.BufferReader(data))
                if table.num_rows != output["row_count"]:
                    raise ValueError(f"merged {kind} row count differs during analysis read")
                recovered[kind] = tuple(decode_merge_table(kind, table))
        finally:
            os.close(merged_descriptor)
    finally:
        os.close(parent_descriptor)
    validate_merge_directory(merged_path, expected_manifest=merge)

    expected_git = coverage["summary"]["observed_git_commit"]
    expected_clean = coverage["summary"]["clean_tracked_worktree"]

    def revalidate() -> None:
        current_model, current_model_bytes = _load_regular_json(
            manifest_path, label="production model-run manifest"
        )
        if current_model_bytes != model_bytes or not _same_json(current_model, model):
            raise ValueError("production model-run manifest changed during analysis")
        current_config, current_config_bytes, _ = _read_analysis_config(repository_root)
        if current_config_bytes != config_bytes or not _same_json(current_config, config):
            raise ValueError("analysis config changed during analysis")
        validate_merge_directory(merged_path, expected_manifest=merge)
        current_coverage, current_coverage_bytes = _load_regular_json(
            coverage_path, label="coverage report"
        )
        if current_coverage_bytes != coverage_bytes or not _same_json(
            current_coverage, coverage
        ):
            raise ValueError("coverage report changed during analysis")
        errors = _global_snapshot_errors(
            repository_root=repository_root,
            source_files=coverage["source_files"],
            expected_git_commit=expected_git,
            expected_clean_tracked=expected_clean,
        )
        if errors:
            raise ValueError("analysis immutable source/Git snapshot changed: " + "; ".join(errors[:5]))

    revalidate()
    question_frame = tuple(
        {"subject": row["subject"], "question_id": row["question_id"]}
        for row in bundle.records
    )
    return AnalysisSource(
        repository_root=repository_root,
        model_manifest=copy.deepcopy(model),
        merge_manifest=copy.deepcopy(merge),
        coverage_report=copy.deepcopy(coverage),
        analysis_config=copy.deepcopy(config),
        question_frame=question_frame,
        natural_rows=recovered["natural_results"],
        checkpoint_rows=recovered["checkpoint_results"],
        small_fixture=False,
        revalidate_inputs=revalidate,
    )


def analyze_production(
    *,
    repository_root: Path,
    model_run_manifest_path: Path,
    bootstrap_replicates: int = 5_000,
) -> tuple[Path, dict[str, Any]]:
    if bootstrap_replicates not in PRODUCTION_BOOTSTRAP_REPLICATES:
        raise ValueError("production bootstrap replicates must be exactly 1000 or 5000")
    source = load_production_analysis_source(
        repository_root=repository_root,
        model_run_manifest_path=model_run_manifest_path,
    )
    return publish_analysis(source, bootstrap_replicates=bootstrap_replicates)


__all__ = [
    "AnalysisSource",
    "AnalysisComputation",
    "ANALYSIS_CONFIG_ORACLE",
    "EXPECTED_ARTIFACT_NAMES",
    "FIXTURE_EVIDENCE_LIMITS",
    "analysis_config_hash",
    "validate_analysis_config",
    "validate_analysis_manifest",
    "summarize_trajectory_events",
    "compute_analysis",
    "validate_analysis_directory",
    "publish_analysis",
    "load_production_analysis_source",
    "analyze_production",
]

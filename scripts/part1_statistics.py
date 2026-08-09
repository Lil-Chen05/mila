"""Fixed, pure statistical analyses for Part 1 trajectory rows."""

from __future__ import annotations

from collections import defaultdict
import math
from numbers import Real
from statistics import median
from typing import Any, Mapping, Sequence

from part1_bootstrap import expand_question_draws, percentile_interval
from part1_contract import (
    FIXED_CHECKPOINT_FRACTIONS,
    FIXED_PRIMARY_AUROC_FEATURE_REGISTRY,
    FIXED_SUBJECTS,
)


PRIMARY_TARGET = "natural_correct"
PRIMARY_FEATURE_REGISTRY = tuple(FIXED_PRIMARY_AUROC_FEATURE_REGISTRY)
CHECKPOINT_PREDICTORS = (
    "checkpoint_normalized_confidence",
    "checkpoint_maximum_ad_probability",
)
MAIN_CHECKPOINT_FRACTIONS = frozenset((0.0, 0.5, 1.0))
COHORT_DEFINITION = "boolean_target_and_finite_predictor"


def _finite_real(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
    )


def rank_auroc(targets: Sequence[bool], scores: Sequence[Real]) -> float | None:
    """Return average-rank Mann--Whitney AUROC, or None for one target class."""

    if len(targets) != len(scores) or not targets:
        raise ValueError("targets and scores must have equal nonzero length")
    if any(type(target) is not bool for target in targets):
        raise ValueError("AUROC targets must be actual booleans")
    if any(not _finite_real(score) for score in scores):
        raise ValueError("AUROC scores must be finite real numbers and not booleans")

    positive_count = sum(targets)
    negative_count = len(targets) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: float(scores[index]))
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        value = float(scores[order[start]])
        while end < len(order) and float(scores[order[end]]) == value:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    positive_rank_sum = sum(
        rank for rank, target in zip(ranks, targets, strict=True) if target
    )
    statistic = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return float(statistic / (positive_count * negative_count))


def reliability_ece(
    targets: Sequence[bool], confidences: Sequence[Real]
) -> dict[str, Any]:
    """Compute fixed ten-bin, count-weighted ECE and all reliability bins."""

    if len(targets) != len(confidences):
        raise ValueError("targets and confidences must have equal length")
    if any(type(target) is not bool for target in targets):
        raise ValueError("calibration targets must be actual booleans")
    values: list[float] = []
    for confidence in confidences:
        if not _finite_real(confidence) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("calibration confidences must be finite real numbers in [0,1]")
        values.append(float(confidence))

    sample_size = len(values)
    bin_members: list[list[int]] = [[] for _ in range(10)]
    for index, confidence in enumerate(values):
        bin_index = min(9, int(confidence * 10.0))
        bin_members[bin_index].append(index)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for bin_index, members in enumerate(bin_members):
        count = len(members)
        if count:
            mean_confidence = sum(values[index] for index in members) / count
            accuracy = sum(targets[index] for index in members) / count
            gap = abs(mean_confidence - accuracy)
            contribution = gap * count / sample_size
            ece += contribution
        else:
            mean_confidence = None
            accuracy = None
            gap = None
            contribution = 0.0
        bins.append(
            {
                "bin_index": bin_index,
                "bin_lower": bin_index / 10.0,
                "bin_upper": (bin_index + 1) / 10.0,
                "upper_inclusive": bin_index == 9,
                "count": count,
                "mean_confidence": (
                    None if mean_confidence is None else float(mean_confidence)
                ),
                "empirical_accuracy": None if accuracy is None else float(accuracy),
                "absolute_gap": None if gap is None else float(gap),
                "weighted_ece_contribution": float(contribution),
            }
        )
    return {
        "sample_size": sample_size,
        "ece": None if sample_size == 0 else float(ece),
        "bins": bins,
    }


def _cohort(
    rows: Sequence[Mapping[str, Any]], predictor: str, target: str
) -> dict[str, Any]:
    targets: list[bool] = []
    predictors: list[float] = []
    target_missing = 0
    predictor_missing = 0
    for row in rows:
        target_value = row.get(target)
        if target_value is None:
            target_missing += 1
            continue
        if type(target_value) is not bool:
            raise ValueError(f"{target} must be an actual boolean or None")
        predictor_value = row.get(predictor)
        if predictor_value is None:
            predictor_missing += 1
            continue
        if not _finite_real(predictor_value):
            raise ValueError(f"{predictor} must be a finite real number or None")
        targets.append(target_value)
        predictors.append(float(predictor_value))
    return {
        "total_candidate_rows": len(rows),
        "target_missing_count": target_missing,
        "predictor_missing_count": predictor_missing,
        "sample_size": len(targets),
        "positive_count": sum(targets),
        "negative_count": len(targets) - sum(targets),
        "targets": targets,
        "predictors": predictors,
    }


def _point_reason(sample_size: int, positive_count: int, negative_count: int) -> str | None:
    if sample_size == 0:
        return "no_eligible_observations"
    if positive_count == 0 or negative_count == 0:
        return "single_target_class"
    return None


def _base_metric_row(
    *,
    analysis_label: str,
    predictor: str,
    target: str,
    grouping: str,
    subject: str | None,
    cohort: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "analysis_label": analysis_label,
        "feature": predictor,
        "predictor": predictor,
        "target": target,
        "cohort_definition": COHORT_DEFINITION,
        "total_candidate_rows": int(cohort["total_candidate_rows"]),
        "target_missing_count": int(cohort["target_missing_count"]),
        "predictor_missing_count": int(cohort["predictor_missing_count"]),
        "sample_size": int(cohort["sample_size"]),
        "positive_count": int(cohort["positive_count"]),
        "negative_count": int(cohort["negative_count"]),
        "grouping": grouping,
        "subject": subject,
    }


def _auroc_metric_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    analysis_label: str,
    predictor: str,
    target: str,
    grouping: str,
    subject: str | None,
) -> dict[str, Any]:
    cohort = _cohort(rows, predictor, target)
    reason = _point_reason(
        cohort["sample_size"], cohort["positive_count"], cohort["negative_count"]
    )
    estimate = (
        None
        if reason is not None
        else rank_auroc(cohort["targets"], cohort["predictors"])
    )
    output = _base_metric_row(
        analysis_label=analysis_label,
        predictor=predictor,
        target=target,
        grouping=grouping,
        subject=subject,
        cohort=cohort,
    )
    output.update(
        {
            "point_estimate": estimate,
            "point_estimate_status": "defined" if estimate is not None else "undefined",
            "point_undefined_reason": reason,
        }
    )
    return output


def _macro_row(
    subject_rows: Sequence[Mapping[str, Any]],
    *,
    analysis_label: str,
    predictor: str,
    target: str,
) -> dict[str, Any]:
    estimates = [row["point_estimate"] for row in subject_rows]
    defined = len(subject_rows) == len(FIXED_SUBJECTS) and all(
        estimate is not None for estimate in estimates
    )
    estimate = float(sum(estimates) / len(FIXED_SUBJECTS)) if defined else None
    output = {
        "analysis_label": analysis_label,
        "feature": predictor,
        "predictor": predictor,
        "target": target,
        "cohort_definition": COHORT_DEFINITION,
        "total_candidate_rows": sum(row["total_candidate_rows"] for row in subject_rows),
        "target_missing_count": sum(row["target_missing_count"] for row in subject_rows),
        "predictor_missing_count": sum(
            row["predictor_missing_count"] for row in subject_rows
        ),
        "sample_size": sum(row["sample_size"] for row in subject_rows),
        "positive_count": sum(row["positive_count"] for row in subject_rows),
        "negative_count": sum(row["negative_count"] for row in subject_rows),
        "point_estimate": estimate,
        "point_estimate_status": "defined" if defined else "undefined",
        "point_undefined_reason": None if defined else "incomplete_subject_macro",
        "grouping": "macro",
        "subject": None,
    }
    return output


def _replicate_ids(draw_plan: Sequence[Mapping[str, Any]]) -> list[int]:
    replicate_ids = sorted({draw.get("replicate_id") for draw in draw_plan})
    if not replicate_ids or any(type(value) is not int or value < 0 for value in replicate_ids):
        raise ValueError("draw plan requires nonnegative integer replicate IDs")
    return replicate_ids


def _bootstrap_auroc_rows(
    rows: Sequence[Mapping[str, Any]],
    draw_plan: Sequence[Mapping[str, Any]],
    *,
    analysis_label: str,
    predictors: Sequence[str],
    target: str,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    expanded = expand_question_draws(draw_plan, rows)
    replicate_ids = _replicate_ids(draw_plan)
    output: list[dict[str, Any]] = []
    for predictor in predictors:
        extra = {} if metadata is None else dict(metadata[predictor])
        for replicate_id in replicate_ids:
            replicate_rows = [
                row for row in expanded if row["replicate_id"] == replicate_id
            ]
            pooled = _auroc_metric_row(
                replicate_rows,
                analysis_label=analysis_label,
                predictor=predictor,
                target=target,
                grouping="pooled",
                subject=None,
            )
            subject_rows = [
                _auroc_metric_row(
                    [row for row in replicate_rows if row["subject"] == subject],
                    analysis_label=analysis_label,
                    predictor=predictor,
                    target=target,
                    grouping="subject",
                    subject=subject,
                )
                for subject in FIXED_SUBJECTS
            ]
            macro = _macro_row(
                subject_rows,
                analysis_label=analysis_label,
                predictor=predictor,
                target=target,
            )
            for metric in (pooled, *subject_rows, macro):
                output.append(
                    {
                        "analysis_label": analysis_label,
                        "feature": predictor,
                        "predictor": predictor,
                        "target": target,
                        "replicate_id": replicate_id,
                        "grouping": metric["grouping"],
                        "subject": metric["subject"],
                        "sample_size": metric["sample_size"],
                        "positive_count": metric["positive_count"],
                        "negative_count": metric["negative_count"],
                        "point_estimate": metric["point_estimate"],
                        "invalid_reason": metric["point_undefined_reason"],
                        **extra,
                    }
                )
    return output


def _attach_intervals(
    metric_rows: list[dict[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    requested_replicates: int,
) -> None:
    for metric in metric_rows:
        estimates = [
            row["point_estimate"]
            for row in bootstrap_rows
            if row["predictor"] == metric["predictor"]
            and row["grouping"] == metric["grouping"]
            and row["subject"] == metric["subject"]
            and row.get("requested_fraction") == metric.get("requested_fraction")
        ]
        metric.update(
            percentile_interval(
                estimates, requested_replicates=requested_replicates
            )
        )


def primary_auroc_analysis(
    rows: Sequence[Mapping[str, Any]],
    draw_plan: Sequence[Mapping[str, Any]],
    *,
    target: str = PRIMARY_TARGET,
    feature_registry: Sequence[str] = PRIMARY_FEATURE_REGISTRY,
) -> dict[str, Any]:
    """Compute the fixed eleven-feature primary AUROC table and shared bootstrap."""

    if target != PRIMARY_TARGET:
        raise ValueError("primary AUROC target must be natural_correct")
    if tuple(feature_registry) != PRIMARY_FEATURE_REGISTRY:
        raise ValueError("primary AUROC feature registry mismatch")
    metric_rows: list[dict[str, Any]] = []
    for feature in PRIMARY_FEATURE_REGISTRY:
        pooled = _auroc_metric_row(
            rows,
            analysis_label="primary_auroc",
            predictor=feature,
            target=target,
            grouping="pooled",
            subject=None,
        )
        subjects = [
            _auroc_metric_row(
                [row for row in rows if row.get("subject") == subject],
                analysis_label="primary_auroc",
                predictor=feature,
                target=target,
                grouping="subject",
                subject=subject,
            )
            for subject in FIXED_SUBJECTS
        ]
        metric_rows.extend((pooled, *subjects, _macro_row(
            subjects,
            analysis_label="primary_auroc",
            predictor=feature,
            target=target,
        )))
    bootstrap_rows = _bootstrap_auroc_rows(
        rows,
        draw_plan,
        analysis_label="primary_auroc",
        predictors=PRIMARY_FEATURE_REGISTRY,
        target=target,
    )
    _attach_intervals(metric_rows, bootstrap_rows, len(_replicate_ids(draw_plan)))
    return {
        "analysis_label": "primary_auroc",
        "target": target,
        "feature_registry": list(PRIMARY_FEATURE_REGISTRY),
        "metric_rows": metric_rows,
        "bootstrap_rows": bootstrap_rows,
    }


def _calibration_metric_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    analysis_label: str,
    predictor: str,
    target: str,
    grouping: str,
    subject: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cohort = _cohort(rows, predictor, target)
    reliability = reliability_ece(cohort["targets"], cohort["predictors"])
    metric = _base_metric_row(
        analysis_label=analysis_label,
        predictor=predictor,
        target=target,
        grouping=grouping,
        subject=subject,
        cohort=cohort,
    )
    estimate = reliability["ece"]
    metric.update(
        {
            "point_estimate": estimate,
            "point_estimate_status": "defined" if estimate is not None else "undefined",
            "point_undefined_reason": (
                None if estimate is not None else "no_eligible_observations"
            ),
        }
    )
    bin_rows = [
        {
            "analysis_label": analysis_label,
            "feature": predictor,
            "predictor": predictor,
            "target": target,
            "cohort_definition": COHORT_DEFINITION,
            "grouping": grouping,
            "subject": subject,
            **bin_row,
        }
        for bin_row in reliability["bins"]
    ]
    return metric, bin_rows


def _bootstrap_calibration_rows(
    rows: Sequence[Mapping[str, Any]],
    draw_plan: Sequence[Mapping[str, Any]],
    *,
    analysis_label: str,
    predictors: Sequence[str],
    target: str,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    expanded = expand_question_draws(draw_plan, rows)
    output: list[dict[str, Any]] = []
    for predictor in predictors:
        extra = {} if metadata is None else dict(metadata[predictor])
        for replicate_id in _replicate_ids(draw_plan):
            replicate_rows = [
                row for row in expanded if row["replicate_id"] == replicate_id
            ]
            pooled, _ = _calibration_metric_row(
                replicate_rows,
                analysis_label=analysis_label,
                predictor=predictor,
                target=target,
                grouping="pooled",
                subject=None,
            )
            subjects = [
                _calibration_metric_row(
                    [row for row in replicate_rows if row["subject"] == subject],
                    analysis_label=analysis_label,
                    predictor=predictor,
                    target=target,
                    grouping="subject",
                    subject=subject,
                )[0]
                for subject in FIXED_SUBJECTS
            ]
            macro = _macro_row(
                subjects,
                analysis_label=analysis_label,
                predictor=predictor,
                target=target,
            )
            for metric in (pooled, *subjects, macro):
                output.append(
                    {
                        "analysis_label": analysis_label,
                        "feature": predictor,
                        "predictor": predictor,
                        "target": target,
                        "replicate_id": replicate_id,
                        "grouping": metric["grouping"],
                        "subject": metric["subject"],
                        "sample_size": metric["sample_size"],
                        "point_estimate": metric["point_estimate"],
                        "invalid_reason": metric["point_undefined_reason"],
                        **extra,
                    }
                )
    return output


def _calibration_analysis(
    rows: Sequence[Mapping[str, Any]],
    draw_plan: Sequence[Mapping[str, Any]],
    *,
    analysis_label: str,
    predictors: Sequence[str],
    target: str,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    metric_rows, reliability_rows = _calibration_point_tables(
        rows,
        analysis_label=analysis_label,
        predictors=predictors,
        target=target,
        metadata=metadata,
    )
    bootstrap_rows = _bootstrap_calibration_rows(
        rows,
        draw_plan,
        analysis_label=analysis_label,
        predictors=predictors,
        target=target,
        metadata=metadata,
    )
    _attach_intervals(metric_rows, bootstrap_rows, len(_replicate_ids(draw_plan)))
    return {
        "analysis_label": analysis_label,
        "target": target,
        "predictors": list(predictors),
        "metric_rows": metric_rows,
        "reliability_rows": reliability_rows,
        "bootstrap_rows": bootstrap_rows,
    }


def _calibration_point_tables(
    rows: Sequence[Mapping[str, Any]],
    *,
    analysis_label: str,
    predictors: Sequence[str],
    target: str,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    for predictor in predictors:
        extra = {} if metadata is None else dict(metadata[predictor])
        pooled, pooled_bins = _calibration_metric_row(
            rows,
            analysis_label=analysis_label,
            predictor=predictor,
            target=target,
            grouping="pooled",
            subject=None,
        )
        subject_metrics: list[dict[str, Any]] = []
        subject_bins: list[dict[str, Any]] = []
        for subject in FIXED_SUBJECTS:
            metric, bins = _calibration_metric_row(
                [row for row in rows if row.get("subject") == subject],
                analysis_label=analysis_label,
                predictor=predictor,
                target=target,
                grouping="subject",
                subject=subject,
            )
            subject_metrics.append(metric)
            subject_bins.extend(bins)
        macro = _macro_row(
            subject_metrics,
            analysis_label=analysis_label,
            predictor=predictor,
            target=target,
        )
        for metric in (pooled, *subject_metrics, macro):
            metric.update(extra)
            metric_rows.append(metric)
        for bin_row in (*pooled_bins, *subject_bins):
            bin_row.update(extra)
            reliability_rows.append(bin_row)
    return metric_rows, reliability_rows


def natural_calibration_analysis(
    rows: Sequence[Mapping[str, Any]],
    draw_plan: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Calibrate natural verbalized confidence only against natural correctness."""

    return _calibration_analysis(
        rows,
        draw_plan,
        analysis_label="natural_calibration",
        predictors=("natural_verbalized_confidence",),
        target=PRIMARY_TARGET,
    )


def _flatten_checkpoint_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        slots = row.get("checkpoint_calibration")
        if not isinstance(slots, list) or len(slots) != 11:
            raise ValueError("trajectory row must contain eleven checkpoint calibration slots")
        for index, expected_fraction in enumerate(FIXED_CHECKPOINT_FRACTIONS):
            slot = slots[index]
            if not isinstance(slot, Mapping):
                raise ValueError("checkpoint calibration slots must be mappings")
            if slot.get("requested_checkpoint_index") != index or not _finite_real(
                slot.get("requested_fraction")
            ) or float(slot["requested_fraction"]) != expected_fraction:
                raise ValueError("checkpoint calibration slots must use canonical fractions")
            output = {
                "study_id": row.get("study_id"),
                "model_run_id": row.get("model_run_id"),
                "subject": row.get("subject"),
                "question_id": row.get("question_id"),
                "run_id": row.get("run_id"),
                "requested_checkpoint_index": index,
                "requested_fraction": expected_fraction,
                "checkpoint_local_correct": slot.get("checkpoint_local_correct"),
                "checkpoint_normalized_confidence": slot.get("normalized_confidence"),
                "checkpoint_maximum_ad_probability": slot.get(
                    "maximum_ad_probability"
                ),
            }
            for bootstrap_field in ("replicate_id", "draw_index", "draw_id"):
                if bootstrap_field in row:
                    output[bootstrap_field] = row[bootstrap_field]
            flat.append(output)
    return flat


def _checkpoint_metadata(
    predictors: Sequence[str], fraction: float
) -> dict[str, dict[str, Any]]:
    return {
        predictor: {
            "requested_fraction": fraction,
            "is_main_checkpoint": fraction in MAIN_CHECKPOINT_FRACTIONS,
        }
        for predictor in predictors
    }


def checkpoint_calibration_analysis(
    rows: Sequence[Mapping[str, Any]],
    draw_plan: Sequence[Mapping[str, Any]],
    *,
    predictors: Sequence[str] = CHECKPOINT_PREDICTORS,
) -> dict[str, Any]:
    """Compute both fixed checkpoint-local calibration families by fraction."""

    if tuple(predictors) != CHECKPOINT_PREDICTORS:
        raise ValueError("checkpoint calibration predictors are fixed and exclude entropy")
    flat = _flatten_checkpoint_rows(rows)
    expanded = expand_question_draws(draw_plan, rows)
    expanded_flat = _flatten_checkpoint_rows(expanded)
    combined_metrics: list[dict[str, Any]] = []
    combined_reliability: list[dict[str, Any]] = []
    combined_bootstrap: list[dict[str, Any]] = []
    for fraction in FIXED_CHECKPOINT_FRACTIONS:
        fraction_rows = [row for row in flat if row["requested_fraction"] == fraction]
        fraction_expanded = [
            row for row in expanded_flat if row["requested_fraction"] == fraction
        ]
        metadata = _checkpoint_metadata(predictors, fraction)
        point_metrics, point_reliability = _calibration_point_tables(
            fraction_rows,
            analysis_label="checkpoint_calibration",
            predictors=predictors,
            target="checkpoint_local_correct",
            metadata=metadata,
        )
        bootstrap = _bootstrap_calibration_from_expanded(
            fraction_expanded,
            draw_plan,
            analysis_label="checkpoint_calibration",
            predictors=predictors,
            target="checkpoint_local_correct",
            metadata=metadata,
        )
        _attach_intervals(
            point_metrics, bootstrap, len(_replicate_ids(draw_plan))
        )
        combined_metrics.extend(point_metrics)
        combined_reliability.extend(point_reliability)
        combined_bootstrap.extend(bootstrap)
    return {
        "analysis_label": "checkpoint_calibration",
        "target": "checkpoint_local_correct",
        "predictors": list(predictors),
        "metric_rows": combined_metrics,
        "reliability_rows": combined_reliability,
        "bootstrap_rows": combined_bootstrap,
    }


def _bootstrap_calibration_from_expanded(
    expanded_rows: Sequence[Mapping[str, Any]],
    draw_plan: Sequence[Mapping[str, Any]],
    *,
    analysis_label: str,
    predictors: Sequence[str],
    target: str,
    metadata: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for predictor in predictors:
        for replicate_id in _replicate_ids(draw_plan):
            replicate_rows = [
                row for row in expanded_rows if row.get("replicate_id") == replicate_id
            ]
            pooled, _ = _calibration_metric_row(
                replicate_rows,
                analysis_label=analysis_label,
                predictor=predictor,
                target=target,
                grouping="pooled",
                subject=None,
            )
            subjects = [
                _calibration_metric_row(
                    [row for row in replicate_rows if row.get("subject") == subject],
                    analysis_label=analysis_label,
                    predictor=predictor,
                    target=target,
                    grouping="subject",
                    subject=subject,
                )[0]
                for subject in FIXED_SUBJECTS
            ]
            macro = _macro_row(
                subjects,
                analysis_label=analysis_label,
                predictor=predictor,
                target=target,
            )
            for metric in (pooled, *subjects, macro):
                output.append(
                    {
                        "analysis_label": analysis_label,
                        "feature": predictor,
                        "predictor": predictor,
                        "target": target,
                        "replicate_id": replicate_id,
                        "grouping": metric["grouping"],
                        "subject": metric["subject"],
                        "sample_size": metric["sample_size"],
                        "point_estimate": metric["point_estimate"],
                        "invalid_reason": metric["point_undefined_reason"],
                        **metadata[predictor],
                    }
                )
    return output


def secondary_checkpoint_auroc_analysis(
    rows: Sequence[Mapping[str, Any]],
    draw_plan: Sequence[Mapping[str, Any]],
    *,
    predictors: Sequence[str] = CHECKPOINT_PREDICTORS,
) -> dict[str, Any]:
    """Compute separately labelled checkpoint-local AUROC by requested fraction."""

    if tuple(predictors) != CHECKPOINT_PREDICTORS:
        raise ValueError("secondary checkpoint AUROC predictors are fixed")
    flat = _flatten_checkpoint_rows(rows)
    expanded_flat = _flatten_checkpoint_rows(expand_question_draws(draw_plan, rows))
    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for fraction in FIXED_CHECKPOINT_FRACTIONS:
        metadata = _checkpoint_metadata(predictors, fraction)
        fraction_rows = [row for row in flat if row["requested_fraction"] == fraction]
        fraction_expanded = [
            row for row in expanded_flat if row["requested_fraction"] == fraction
        ]
        for predictor in predictors:
            pooled = _auroc_metric_row(
                fraction_rows,
                analysis_label="secondary_checkpoint_local_auroc",
                predictor=predictor,
                target="checkpoint_local_correct",
                grouping="pooled",
                subject=None,
            )
            subjects = [
                _auroc_metric_row(
                    [row for row in fraction_rows if row.get("subject") == subject],
                    analysis_label="secondary_checkpoint_local_auroc",
                    predictor=predictor,
                    target="checkpoint_local_correct",
                    grouping="subject",
                    subject=subject,
                )
                for subject in FIXED_SUBJECTS
            ]
            macro = _macro_row(
                subjects,
                analysis_label="secondary_checkpoint_local_auroc",
                predictor=predictor,
                target="checkpoint_local_correct",
            )
            for metric in (pooled, *subjects, macro):
                metric.update(metadata[predictor])
                metric_rows.append(metric)
        fraction_bootstrap = _bootstrap_auroc_from_expanded(
            fraction_expanded,
            draw_plan,
            predictors=predictors,
            metadata=metadata,
        )
        bootstrap_rows.extend(fraction_bootstrap)
    _attach_intervals(metric_rows, bootstrap_rows, len(_replicate_ids(draw_plan)))
    return {
        "analysis_label": "secondary_checkpoint_local_auroc",
        "target": "checkpoint_local_correct",
        "predictors": list(predictors),
        "metric_rows": metric_rows,
        "bootstrap_rows": bootstrap_rows,
    }


def _bootstrap_auroc_from_expanded(
    expanded_rows: Sequence[Mapping[str, Any]],
    draw_plan: Sequence[Mapping[str, Any]],
    *,
    predictors: Sequence[str],
    metadata: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    label = "secondary_checkpoint_local_auroc"
    target = "checkpoint_local_correct"
    for predictor in predictors:
        for replicate_id in _replicate_ids(draw_plan):
            replicate_rows = [
                row for row in expanded_rows if row.get("replicate_id") == replicate_id
            ]
            pooled = _auroc_metric_row(
                replicate_rows,
                analysis_label=label,
                predictor=predictor,
                target=target,
                grouping="pooled",
                subject=None,
            )
            subjects = [
                _auroc_metric_row(
                    [row for row in replicate_rows if row.get("subject") == subject],
                    analysis_label=label,
                    predictor=predictor,
                    target=target,
                    grouping="subject",
                    subject=subject,
                )
                for subject in FIXED_SUBJECTS
            ]
            macro = _macro_row(
                subjects,
                analysis_label=label,
                predictor=predictor,
                target=target,
            )
            for metric in (pooled, *subjects, macro):
                output.append(
                    {
                        "analysis_label": label,
                        "feature": predictor,
                        "predictor": predictor,
                        "target": target,
                        "replicate_id": replicate_id,
                        "grouping": metric["grouping"],
                        "subject": metric["subject"],
                        "sample_size": metric["sample_size"],
                        "positive_count": metric["positive_count"],
                        "negative_count": metric["negative_count"],
                        "point_estimate": metric["point_estimate"],
                        "invalid_reason": metric["point_undefined_reason"],
                        **metadata[predictor],
                    }
                )
    return output


def _question_order_key(pair: tuple[str, str]) -> tuple[int, str, str]:
    subject, question_id = pair
    try:
        subject_index = FIXED_SUBJECTS.index(subject)
    except ValueError:
        subject_index = len(FIXED_SUBJECTS)
    return subject_index, subject, question_id


def within_question_analysis(
    rows: Sequence[Mapping[str, Any]],
    draw_plan: Sequence[Mapping[str, Any]],
    *,
    feature_registry: Sequence[str] = PRIMARY_FEATURE_REGISTRY,
) -> dict[str, Any]:
    """Compute equal-question paired differences and their shared-plan bootstrap."""

    if tuple(feature_registry) != PRIMARY_FEATURE_REGISTRY:
        raise ValueError("within-question feature registry mismatch")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    question_subject: dict[str, str] = {}
    for row in rows:
        subject = row.get("subject")
        question_id = row.get("question_id")
        if not isinstance(subject, str) or not subject or not isinstance(question_id, str) or not question_id:
            raise ValueError("within-question rows require subject and question_id")
        previous_subject = question_subject.setdefault(question_id, subject)
        if previous_subject != subject:
            raise ValueError("within-question rows violate subject/question consistency")
        target = row.get(PRIMARY_TARGET)
        if target is not None and type(target) is not bool:
            raise ValueError("natural_correct must be an actual boolean or None")
        for feature in PRIMARY_FEATURE_REGISTRY:
            value = row.get(feature)
            if value is not None and not _finite_real(value):
                raise ValueError(f"{feature} must be a finite real number or None")
        grouped[(subject, question_id)].append(row)

    distribution_rows: list[dict[str, Any]] = []
    for feature in PRIMARY_FEATURE_REGISTRY:
        for pair in sorted(grouped, key=_question_order_key):
            question_rows = grouped[pair]
            boolean_targets = [
                row[PRIMARY_TARGET]
                for row in question_rows
                if type(row.get(PRIMARY_TARGET)) is bool
            ]
            if True not in boolean_targets or False not in boolean_targets:
                continue
            correct_values = [
                float(row[feature])
                for row in question_rows
                if row.get(PRIMARY_TARGET) is True and row.get(feature) is not None
            ]
            incorrect_values = [
                float(row[feature])
                for row in question_rows
                if row.get(PRIMARY_TARGET) is False and row.get(feature) is not None
            ]
            if not correct_values or not incorrect_values:
                continue
            correct_mean = sum(correct_values) / len(correct_values)
            incorrect_mean = sum(incorrect_values) / len(incorrect_values)
            first = question_rows[0]
            distribution_rows.append(
                {
                    "analysis_label": "within_question_paired_difference",
                    "study_id": first.get("study_id"),
                    "model_run_id": first.get("model_run_id"),
                    "subject": pair[0],
                    "question_id": pair[1],
                    "feature": feature,
                    "target": PRIMARY_TARGET,
                    "correct_run_count": len(correct_values),
                    "incorrect_run_count": len(incorrect_values),
                    "correct_run_mean": float(correct_mean),
                    "incorrect_run_mean": float(incorrect_mean),
                    "paired_difference": float(correct_mean - incorrect_mean),
                }
            )

    expanded_differences = expand_question_draws(draw_plan, distribution_rows)
    replicate_ids = _replicate_ids(draw_plan)
    bootstrap_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for feature in PRIMARY_FEATURE_REGISTRY:
        feature_distribution = [
            row for row in distribution_rows if row["feature"] == feature
        ]
        differences = [row["paired_difference"] for row in feature_distribution]
        for replicate_id in replicate_ids:
            drawn = [
                row["paired_difference"]
                for row in expanded_differences
                if row["feature"] == feature
                and row["replicate_id"] == replicate_id
            ]
            estimate = None if not drawn else float(sum(drawn) / len(drawn))
            bootstrap_rows.append(
                {
                    "analysis_label": "within_question_paired_difference",
                    "feature": feature,
                    "target": PRIMARY_TARGET,
                    "replicate_id": replicate_id,
                    "drawn_qualifying_question_count": len(drawn),
                    "point_estimate": estimate,
                    "invalid_reason": (
                        None if estimate is not None else "no_qualifying_drawn_questions"
                    ),
                }
            )
        feature_bootstrap = [
            row["point_estimate"] for row in bootstrap_rows if row["feature"] == feature
        ]
        summary = {
            "analysis_label": "within_question_paired_difference",
            "feature": feature,
            "target": PRIMARY_TARGET,
            "cohort_definition": "mixed_boolean_correctness_and_finite_feature_on_both_sides",
            "qualifying_question_count": len(differences),
            "mean_paired_difference": (
                None if not differences else float(sum(differences) / len(differences))
            ),
            "median_paired_difference": (
                None if not differences else float(median(differences))
            ),
        }
        summary.update(
            percentile_interval(
                feature_bootstrap, requested_replicates=len(replicate_ids)
            )
        )
        summary_rows.append(summary)
    return {
        "analysis_label": "within_question_paired_difference",
        "target": PRIMARY_TARGET,
        "feature_registry": list(PRIMARY_FEATURE_REGISTRY),
        "distribution_rows": distribution_rows,
        "bootstrap_draw_rows": expanded_differences,
        "summary_rows": summary_rows,
        "bootstrap_rows": bootstrap_rows,
    }


__all__ = [
    "rank_auroc",
    "reliability_ece",
    "primary_auroc_analysis",
    "natural_calibration_analysis",
    "checkpoint_calibration_analysis",
    "secondary_checkpoint_auroc_analysis",
    "within_question_analysis",
]

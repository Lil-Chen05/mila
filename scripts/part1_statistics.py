"""Fixed point estimates and compact weighted bootstrap analyses for Part 1."""

from __future__ import annotations

from collections import defaultdict
import math
from numbers import Integral, Real
from statistics import median
from typing import Any, Callable, Mapping, Sequence

from part1_bootstrap import QuestionDrawPlan, percentile_interval
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

_MetricKey = tuple[str, str, str | None, float | None]
_QuestionWeights = Mapping[tuple[str, str], int]


def _finite_real(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    if isinstance(value, Integral):
        return True
    try:
        return bool(math.isfinite(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _validate_targets_scores_weights(
    targets: Sequence[bool], scores: Sequence[Real], weights: Sequence[int]
) -> list[int]:
    if len(targets) != len(scores) or len(targets) != len(weights) or not targets:
        raise ValueError("targets, scores, and weights must have equal nonzero length")
    if any(type(target) is not bool for target in targets):
        raise ValueError("AUROC targets must be actual booleans")
    if any(not _finite_real(score) for score in scores):
        raise ValueError("AUROC scores must be finite real numbers and not booleans")
    output: list[int] = []
    for weight in weights:
        if isinstance(weight, bool) or not isinstance(weight, Integral) or weight < 0:
            raise ValueError("AUROC requires nonnegative integer weights")
        output.append(int(weight))
    return output


def weighted_rank_auroc(
    targets: Sequence[bool], scores: Sequence[Real], weights: Sequence[int]
) -> float | None:
    """Exact native-score weighted AUROC with half credit for tied pairs."""

    integer_weights = _validate_targets_scores_weights(targets, scores, weights)
    active = [index for index, weight in enumerate(integer_weights) if weight > 0]
    positive_total = sum(
        integer_weights[index] for index in active if targets[index]
    )
    negative_total = sum(
        integer_weights[index] for index in active if not targets[index]
    )
    if positive_total == 0 or negative_total == 0:
        return None
    try:
        order = sorted(active, key=lambda index: scores[index])
    except (TypeError, ValueError) as error:
        raise ValueError("AUROC scores must be mutually comparable") from error

    twice_concordant = 0
    negative_below = 0
    start = 0
    while start < len(order):
        end = start + 1
        group_score = scores[order[start]]
        try:
            while end < len(order) and bool(scores[order[end]] == group_score):
                end += 1
        except (TypeError, ValueError) as error:
            raise ValueError("AUROC scores must support exact tie equality") from error
        positive_weight = sum(
            integer_weights[index]
            for index in order[start:end]
            if targets[index]
        )
        negative_weight = sum(
            integer_weights[index]
            for index in order[start:end]
            if not targets[index]
        )
        twice_concordant += 2 * positive_weight * negative_below
        twice_concordant += positive_weight * negative_weight
        negative_below += negative_weight
        start = end
    return float(twice_concordant / (2 * positive_total * negative_total))


def rank_auroc(targets: Sequence[bool], scores: Sequence[Real]) -> float | None:
    """Average-rank/Mann--Whitney AUROC using exact native score ordering."""

    return weighted_rank_auroc(targets, scores, [1] * len(targets))


def weighted_reliability_ece(
    targets: Sequence[bool], confidences: Sequence[Real], weights: Sequence[int]
) -> dict[str, Any]:
    """Fixed ten-bin reliability/ECE using nonnegative integer multiplicities."""

    if len(targets) != len(confidences) or len(targets) != len(weights):
        raise ValueError("targets, confidences, and weights must have equal length")
    if any(type(target) is not bool for target in targets):
        raise ValueError("calibration targets must be actual booleans")
    values: list[float] = []
    integer_weights: list[int] = []
    for confidence in confidences:
        if not _finite_real(confidence) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("calibration confidences must be finite real numbers in [0,1]")
        values.append(float(confidence))
    for weight in weights:
        if isinstance(weight, bool) or not isinstance(weight, Integral) or weight < 0:
            raise ValueError("calibration requires nonnegative integer weights")
        integer_weights.append(int(weight))

    sample_size = sum(integer_weights)
    bin_indices: list[list[int]] = [[] for _ in range(10)]
    for index, confidence in enumerate(values):
        bin_indices[min(9, int(confidence * 10.0))].append(index)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for bin_index, members in enumerate(bin_indices):
        count = sum(integer_weights[index] for index in members)
        if count:
            mean_confidence = sum(
                values[index] * integer_weights[index] for index in members
            ) / count
            accuracy = sum(
                int(targets[index]) * integer_weights[index] for index in members
            ) / count
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
                "count": int(count),
                "mean_confidence": (
                    None if mean_confidence is None else float(mean_confidence)
                ),
                "empirical_accuracy": None if accuracy is None else float(accuracy),
                "absolute_gap": None if gap is None else float(gap),
                "weighted_ece_contribution": float(contribution),
            }
        )
    return {
        "sample_size": int(sample_size),
        "ece": None if sample_size == 0 else float(ece),
        "bins": bins,
    }


def reliability_ece(
    targets: Sequence[bool], confidences: Sequence[Real]
) -> dict[str, Any]:
    """Fixed ten-bin reliability/ECE with unit observation weights."""

    return weighted_reliability_ece(targets, confidences, [1] * len(targets))


def _canonical_plan_pairs(draw_plan: QuestionDrawPlan) -> set[tuple[str, str]]:
    if not isinstance(draw_plan, QuestionDrawPlan):
        raise ValueError("draw_plan must be a compact QuestionDrawPlan")
    return {
        (subject, question_id)
        for subject, group in zip(
            draw_plan.subjects, draw_plan.question_ids_by_subject, strict=True
        )
        for question_id in group
    }


def _validate_source_rows(
    rows: Sequence[Mapping[str, Any]], draw_plan: QuestionDrawPlan
) -> None:
    canonical_pairs = _canonical_plan_pairs(draw_plan)
    question_subject: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("analysis rows must be mappings")
        subject = row.get("subject")
        question_id = row.get("question_id")
        if not isinstance(subject, str) or not subject:
            raise ValueError("analysis rows require a nonempty subject")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("analysis rows require a nonempty question_id")
        pair = (subject, question_id)
        if pair not in canonical_pairs:
            raise ValueError("analysis row question is absent from the compact draw plan")
        previous = question_subject.setdefault(question_id, subject)
        if previous != subject:
            raise ValueError("analysis rows violate subject/question consistency")


def _cohort(
    rows: Sequence[Mapping[str, Any]],
    predictor: str,
    target: str,
    question_weights: _QuestionWeights | None = None,
) -> dict[str, Any]:
    targets: list[bool] = []
    predictors: list[Real] = []
    weights: list[int] = []
    total_candidate = 0
    target_missing = 0
    predictor_missing = 0
    for row in rows:
        weight = (
            1
            if question_weights is None
            else int(question_weights[(str(row["subject"]), str(row["question_id"]))])
        )
        total_candidate += weight
        target_value = row.get(target)
        if target_value is None:
            target_missing += weight
            continue
        if type(target_value) is not bool:
            raise ValueError(f"{target} must be an actual boolean or None")
        predictor_value = row.get(predictor)
        if predictor_value is None:
            predictor_missing += weight
            continue
        if not _finite_real(predictor_value):
            raise ValueError(f"{predictor} must be a finite real number or None")
        targets.append(target_value)
        predictors.append(predictor_value)
        weights.append(weight)
    positive_count = sum(
        weight for target_value, weight in zip(targets, weights, strict=True) if target_value
    )
    sample_size = sum(weights)
    return {
        "total_candidate_rows": int(total_candidate),
        "target_missing_count": int(target_missing),
        "predictor_missing_count": int(predictor_missing),
        "sample_size": int(sample_size),
        "positive_count": int(positive_count),
        "negative_count": int(sample_size - positive_count),
        "targets": targets,
        "predictors": predictors,
        "weights": weights,
    }


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
    question_weights: _QuestionWeights | None = None,
) -> dict[str, Any]:
    cohort = _cohort(rows, predictor, target, question_weights)
    estimate = (
        None
        if cohort["sample_size"] == 0
        else weighted_rank_auroc(
            cohort["targets"], cohort["predictors"], cohort["weights"]
        )
    )
    if cohort["sample_size"] == 0:
        reason = "no_eligible_observations"
    elif estimate is None:
        reason = "single_target_class"
    else:
        reason = None
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


def _calibration_metric_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    analysis_label: str,
    predictor: str,
    target: str,
    grouping: str,
    subject: str | None,
    question_weights: _QuestionWeights | None = None,
    include_bins: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cohort = _cohort(rows, predictor, target, question_weights)
    reliability = weighted_reliability_ece(
        cohort["targets"], cohort["predictors"], cohort["weights"]
    )
    estimate = reliability["ece"]
    metric = _base_metric_row(
        analysis_label=analysis_label,
        predictor=predictor,
        target=target,
        grouping=grouping,
        subject=subject,
        cohort=cohort,
    )
    metric.update(
        {
            "point_estimate": estimate,
            "point_estimate_status": "defined" if estimate is not None else "undefined",
            "point_undefined_reason": (
                None if estimate is not None else "no_eligible_observations"
            ),
        }
    )
    if not include_bins:
        return metric, []
    bins = [
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
    return metric, bins


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
    return {
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


def _group_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    analysis_label: str,
    predictor: str,
    target: str,
    metric_builder: Callable[..., Any],
    question_weights: _QuestionWeights | None = None,
    include_bins: bool = True,
    rows_by_subject: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def build(
        group_rows: Sequence[Mapping[str, Any]], grouping: str, subject: str | None
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        result = metric_builder(
            group_rows,
            analysis_label=analysis_label,
            predictor=predictor,
            target=target,
            grouping=grouping,
            subject=subject,
            question_weights=question_weights,
            **({"include_bins": include_bins} if metric_builder is _calibration_metric_row else {}),
        )
        return result if isinstance(result, tuple) else (result, [])

    pooled, pooled_bins = build(rows, "pooled", None)
    subject_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = list(pooled_bins)
    for subject in FIXED_SUBJECTS:
        metric, bins = build(
            (
                rows_by_subject.get(subject, ())
                if rows_by_subject is not None
                else [row for row in rows if row.get("subject") == subject]
            ),
            "subject",
            subject,
        )
        subject_rows.append(metric)
        reliability_rows.extend(bins)
    macro = _macro_row(
        subject_rows,
        analysis_label=analysis_label,
        predictor=predictor,
        target=target,
    )
    return [pooled, *subject_rows, macro], reliability_rows


def _index_rows_by_subject(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    indexed = {subject: [] for subject in FIXED_SUBJECTS}
    for row in rows:
        subject = row.get("subject")
        if subject in indexed:
            indexed[str(subject)].append(row)
    return indexed


def _metric_spec(
    rows: Sequence[Mapping[str, Any]],
    *,
    predictor: str,
    target: str,
    metadata: Mapping[str, Any] | None = None,
    rows_by_subject: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    return {
        "rows": rows,
        "rows_by_subject": (
            _index_rows_by_subject(rows) if rows_by_subject is None else rows_by_subject
        ),
        "predictor": predictor,
        "target": target,
        "metadata": {} if metadata is None else dict(metadata),
    }


def _metric_key(metric: Mapping[str, Any]) -> _MetricKey:
    return (
        str(metric["predictor"]),
        str(metric["grouping"]),
        None if metric["subject"] is None else str(metric["subject"]),
        metric.get("requested_fraction"),
    )


def _stream_metric_estimates(
    specs: Sequence[Mapping[str, Any]],
    draw_plan: QuestionDrawPlan,
    *,
    analysis_label: str,
    metric_builder: Callable[..., Any],
) -> dict[_MetricKey, list[float | None]]:
    estimates: dict[_MetricKey, list[float | None]] = defaultdict(list)
    for replicate_id in range(draw_plan.replicates):
        question_weights = draw_plan.question_multiplicities(replicate_id)
        for spec in specs:
            metrics, _ = _group_metric_rows(
                spec["rows"],
                analysis_label=analysis_label,
                predictor=str(spec["predictor"]),
                target=str(spec["target"]),
                metric_builder=metric_builder,
                question_weights=question_weights,
                include_bins=False,
                rows_by_subject=spec["rows_by_subject"],
            )
            metadata = spec.get("metadata", {})
            for metric in metrics:
                metric.update(metadata)
                estimates[_metric_key(metric)].append(metric["point_estimate"])
    return dict(estimates)


def _attach_intervals(
    metric_rows: list[dict[str, Any]],
    estimates: Mapping[_MetricKey, Sequence[float | None]],
    requested_replicates: int,
) -> None:
    for metric in metric_rows:
        metric.update(
            percentile_interval(
                estimates.get(_metric_key(metric), ()),
                requested_replicates=requested_replicates,
            )
        )


def _bootstrap_metadata(draw_plan: QuestionDrawPlan) -> dict[str, Any]:
    return {
        "representation": "compact_question_multiplicity_weights",
        "replicates": draw_plan.replicates,
        "logical_draw_count": draw_plan.logical_draw_count,
        "selected_index_cell_count": draw_plan.selected_index_cell_count,
        "selected_index_storage_bytes": draw_plan.estimated_storage_bytes,
        "replicate_diagnostic_rows_retained": False,
        "draw_rows_retained": False,
    }


def primary_auroc_analysis(
    rows: Sequence[Mapping[str, Any]],
    draw_plan: QuestionDrawPlan,
    *,
    target: str = PRIMARY_TARGET,
    feature_registry: Sequence[str] = PRIMARY_FEATURE_REGISTRY,
) -> dict[str, Any]:
    """Fixed eleven-feature primary AUROC with compact weighted bootstrap."""

    if target != PRIMARY_TARGET:
        raise ValueError("primary AUROC target must be natural_correct")
    if tuple(feature_registry) != PRIMARY_FEATURE_REGISTRY:
        raise ValueError("primary AUROC feature registry mismatch")
    _validate_source_rows(rows, draw_plan)
    metric_rows: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []
    rows_by_subject = _index_rows_by_subject(rows)
    for feature in PRIMARY_FEATURE_REGISTRY:
        metrics, _ = _group_metric_rows(
            rows,
            analysis_label="primary_auroc",
            predictor=feature,
            target=target,
            metric_builder=_auroc_metric_row,
            rows_by_subject=rows_by_subject,
        )
        metric_rows.extend(metrics)
        specs.append(
            _metric_spec(
                rows,
                predictor=feature,
                target=target,
                rows_by_subject=rows_by_subject,
            )
        )
    estimates = _stream_metric_estimates(
        specs,
        draw_plan,
        analysis_label="primary_auroc",
        metric_builder=_auroc_metric_row,
    )
    _attach_intervals(metric_rows, estimates, draw_plan.replicates)
    return {
        "analysis_label": "primary_auroc",
        "target": target,
        "feature_registry": list(PRIMARY_FEATURE_REGISTRY),
        "metric_rows": metric_rows,
        "bootstrap": _bootstrap_metadata(draw_plan),
    }


def _calibration_point_tables(
    specs: Sequence[Mapping[str, Any]], *, analysis_label: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    for spec in specs:
        metrics, bins = _group_metric_rows(
            spec["rows"],
            analysis_label=analysis_label,
            predictor=str(spec["predictor"]),
            target=str(spec["target"]),
            metric_builder=_calibration_metric_row,
            rows_by_subject=spec["rows_by_subject"],
        )
        metadata = spec.get("metadata", {})
        for metric in metrics:
            metric.update(metadata)
            metric_rows.append(metric)
        for bin_row in bins:
            bin_row.update(metadata)
            reliability_rows.append(bin_row)
    return metric_rows, reliability_rows


def _calibration_analysis(
    specs: Sequence[Mapping[str, Any]],
    draw_plan: QuestionDrawPlan,
    *,
    analysis_label: str,
    predictors: Sequence[str],
    target: str,
) -> dict[str, Any]:
    metric_rows, reliability_rows = _calibration_point_tables(
        specs, analysis_label=analysis_label
    )
    estimates = _stream_metric_estimates(
        specs,
        draw_plan,
        analysis_label=analysis_label,
        metric_builder=_calibration_metric_row,
    )
    _attach_intervals(metric_rows, estimates, draw_plan.replicates)
    return {
        "analysis_label": analysis_label,
        "target": target,
        "predictors": list(predictors),
        "metric_rows": metric_rows,
        "reliability_rows": reliability_rows,
        "bootstrap": _bootstrap_metadata(draw_plan),
    }


def natural_calibration_analysis(
    rows: Sequence[Mapping[str, Any]], draw_plan: QuestionDrawPlan
) -> dict[str, Any]:
    """Calibrate natural confidence only against natural correctness."""

    _validate_source_rows(rows, draw_plan)
    predictor = "natural_verbalized_confidence"
    return _calibration_analysis(
        [_metric_spec(rows, predictor=predictor, target=PRIMARY_TARGET)],
        draw_plan,
        analysis_label="natural_calibration",
        predictors=(predictor,),
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
            if (
                type(slot.get("requested_checkpoint_index")) is not int
                or slot["requested_checkpoint_index"] != index
                or type(slot.get("requested_fraction")) is not float
                or slot["requested_fraction"] != expected_fraction
            ):
                raise ValueError("checkpoint calibration slots must use canonical fractions")
            flat.append(
                {
                    "study_id": row.get("study_id"),
                    "model_run_id": row.get("model_run_id"),
                    "subject": row.get("subject"),
                    "question_id": row.get("question_id"),
                    "run_id": row.get("run_id"),
                    "requested_checkpoint_index": index,
                    "requested_fraction": expected_fraction,
                    "checkpoint_local_correct": slot.get("checkpoint_local_correct"),
                    "checkpoint_normalized_confidence": slot.get(
                        "normalized_confidence"
                    ),
                    "checkpoint_maximum_ad_probability": slot.get(
                        "maximum_ad_probability"
                    ),
                }
            )
    return flat


def _checkpoint_specs(
    flat_rows: Sequence[Mapping[str, Any]], predictors: Sequence[str]
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for fraction in FIXED_CHECKPOINT_FRACTIONS:
        fraction_rows = [
            row for row in flat_rows if row["requested_fraction"] == fraction
        ]
        rows_by_subject = _index_rows_by_subject(fraction_rows)
        for predictor in predictors:
            specs.append(
                _metric_spec(
                    fraction_rows,
                    predictor=predictor,
                    target="checkpoint_local_correct",
                    metadata={
                        "requested_fraction": fraction,
                        "is_main_checkpoint": fraction in MAIN_CHECKPOINT_FRACTIONS,
                    },
                    rows_by_subject=rows_by_subject,
                )
            )
    return specs


def checkpoint_calibration_analysis(
    rows: Sequence[Mapping[str, Any]],
    draw_plan: QuestionDrawPlan,
    *,
    predictors: Sequence[str] = CHECKPOINT_PREDICTORS,
) -> dict[str, Any]:
    """Both fixed checkpoint-local calibration families by logical fraction."""

    if tuple(predictors) != CHECKPOINT_PREDICTORS:
        raise ValueError("checkpoint calibration predictors are fixed and exclude entropy")
    _validate_source_rows(rows, draw_plan)
    flat = _flatten_checkpoint_rows(rows)
    return _calibration_analysis(
        _checkpoint_specs(flat, predictors),
        draw_plan,
        analysis_label="checkpoint_calibration",
        predictors=predictors,
        target="checkpoint_local_correct",
    )


def secondary_checkpoint_auroc_analysis(
    rows: Sequence[Mapping[str, Any]],
    draw_plan: QuestionDrawPlan,
    *,
    predictors: Sequence[str] = CHECKPOINT_PREDICTORS,
) -> dict[str, Any]:
    """Separately labelled checkpoint-local AUROC by logical fraction."""

    if tuple(predictors) != CHECKPOINT_PREDICTORS:
        raise ValueError("secondary checkpoint AUROC predictors are fixed")
    _validate_source_rows(rows, draw_plan)
    specs = _checkpoint_specs(_flatten_checkpoint_rows(rows), predictors)
    metric_rows: list[dict[str, Any]] = []
    for spec in specs:
        metrics, _ = _group_metric_rows(
            spec["rows"],
            analysis_label="secondary_checkpoint_local_auroc",
            predictor=str(spec["predictor"]),
            target="checkpoint_local_correct",
            metric_builder=_auroc_metric_row,
            rows_by_subject=spec["rows_by_subject"],
        )
        for metric in metrics:
            metric.update(spec["metadata"])
            metric_rows.append(metric)
    estimates = _stream_metric_estimates(
        specs,
        draw_plan,
        analysis_label="secondary_checkpoint_local_auroc",
        metric_builder=_auroc_metric_row,
    )
    _attach_intervals(metric_rows, estimates, draw_plan.replicates)
    return {
        "analysis_label": "secondary_checkpoint_local_auroc",
        "target": "checkpoint_local_correct",
        "predictors": list(predictors),
        "metric_rows": metric_rows,
        "bootstrap": _bootstrap_metadata(draw_plan),
    }


def _question_order_key(pair: tuple[str, str]) -> tuple[int, str, str]:
    subject, question_id = pair
    try:
        subject_index = FIXED_SUBJECTS.index(subject)
    except ValueError:
        subject_index = len(FIXED_SUBJECTS)
    return subject_index, subject, question_id


def within_question_analysis(
    rows: Sequence[Mapping[str, Any]],
    draw_plan: QuestionDrawPlan,
    *,
    feature_registry: Sequence[str] = PRIMARY_FEATURE_REGISTRY,
) -> dict[str, Any]:
    """Equal-question paired differences with compact multiplicity bootstrap."""

    if tuple(feature_registry) != PRIMARY_FEATURE_REGISTRY:
        raise ValueError("within-question feature registry mismatch")
    _validate_source_rows(rows, draw_plan)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        target = row.get(PRIMARY_TARGET)
        if target is not None and type(target) is not bool:
            raise ValueError("natural_correct must be an actual boolean or None")
        for feature in PRIMARY_FEATURE_REGISTRY:
            value = row.get(feature)
            if value is not None and not _finite_real(value):
                raise ValueError(f"{feature} must be a finite real number or None")
        grouped[(str(row["subject"]), str(row["question_id"]))].append(row)

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

    estimates: dict[str, list[float | None]] = {
        feature: [] for feature in PRIMARY_FEATURE_REGISTRY
    }
    by_feature = {
        feature: [row for row in distribution_rows if row["feature"] == feature]
        for feature in PRIMARY_FEATURE_REGISTRY
    }
    for replicate_id in range(draw_plan.replicates):
        question_weights = draw_plan.question_multiplicities(replicate_id)
        for feature, feature_rows in by_feature.items():
            weighted_sum = 0.0
            weight_total = 0
            for row in feature_rows:
                weight = question_weights[(row["subject"], row["question_id"])]
                weighted_sum += row["paired_difference"] * weight
                weight_total += weight
            estimates[feature].append(
                None if weight_total == 0 else float(weighted_sum / weight_total)
            )

    summary_rows: list[dict[str, Any]] = []
    for feature in PRIMARY_FEATURE_REGISTRY:
        differences = [row["paired_difference"] for row in by_feature[feature]]
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
                estimates[feature], requested_replicates=draw_plan.replicates
            )
        )
        summary_rows.append(summary)
    return {
        "analysis_label": "within_question_paired_difference",
        "target": PRIMARY_TARGET,
        "feature_registry": list(PRIMARY_FEATURE_REGISTRY),
        "distribution_rows": distribution_rows,
        "summary_rows": summary_rows,
        "bootstrap": _bootstrap_metadata(draw_plan),
    }


__all__ = [
    "weighted_rank_auroc",
    "rank_auroc",
    "weighted_reliability_ece",
    "reliability_ece",
    "primary_auroc_analysis",
    "natural_calibration_analysis",
    "checkpoint_calibration_analysis",
    "secondary_checkpoint_auroc_analysis",
    "within_question_analysis",
]

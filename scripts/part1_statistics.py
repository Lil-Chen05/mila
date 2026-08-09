"""Fixed point estimates and compact weighted bootstrap analyses for Part 1."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from numbers import Integral, Real
from statistics import median
from typing import Any, Callable, Mapping, Sequence

import numpy as np

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

_STRUCTURAL_HOOK: Callable[[Sequence[str]], None] | None = None


def _emit_structural(event: str) -> None:
    if _STRUCTURAL_HOOK is not None:
        _STRUCTURAL_HOOK([event])


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


def _validate_analysis_plan(
    draw_plan: QuestionDrawPlan, *, allow_small_fixture: bool
) -> None:
    if not isinstance(draw_plan, QuestionDrawPlan):
        raise ValueError("draw_plan must be a compact QuestionDrawPlan")
    if type(allow_small_fixture) is not bool:
        raise ValueError("allow_small_fixture must be boolean")
    if draw_plan.small_fixture and not allow_small_fixture:
        raise ValueError("small-fixture draw plan requires explicit analysis opt-in")


@dataclass(frozen=True, slots=True)
class _SourceIndex:
    rows: tuple[Mapping[str, Any], ...]
    question_indices: np.ndarray[Any, Any]
    subject_indices: np.ndarray[Any, Any]
    question_lookup: Mapping[tuple[str, str], int]
    rows_by_subject: Mapping[str, tuple[Mapping[str, Any], ...]]


def _compile_source_index(
    rows: Sequence[Mapping[str, Any]],
    draw_plan: QuestionDrawPlan,
    *,
    allow_small_fixture: bool,
) -> _SourceIndex:
    _validate_analysis_plan(
        draw_plan, allow_small_fixture=allow_small_fixture
    )
    question_lookup: dict[tuple[str, str], int] = {}
    question_subject: dict[str, str] = {}
    subject_lookup = {subject: index for index, subject in enumerate(FIXED_SUBJECTS)}
    offset = 0
    for subject, group in zip(
        draw_plan.subjects, draw_plan.question_ids_by_subject, strict=True
    ):
        for question_id in group:
            question_lookup[(subject, question_id)] = offset
            question_subject[question_id] = subject
            offset += 1

    immutable_rows = tuple(rows)
    question_indices = np.empty(len(immutable_rows), dtype=np.int32)
    subject_indices = np.empty(len(immutable_rows), dtype=np.int8)
    by_subject: dict[str, list[Mapping[str, Any]]] = {
        subject: [] for subject in FIXED_SUBJECTS
    }
    observed_question_subject: dict[str, str] = {}
    for row_index, row in enumerate(immutable_rows):
        _emit_structural("source_row_indexed")
        if not isinstance(row, Mapping):
            raise ValueError("analysis rows must be mappings")
        subject = row.get("subject")
        question_id = row.get("question_id")
        if not isinstance(subject, str) or not subject:
            raise ValueError("analysis rows require a nonempty subject")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("analysis rows require a nonempty question_id")
        pair = (subject, question_id)
        if pair not in question_lookup:
            raise ValueError("analysis row question is absent from the compact draw plan")
        previous = observed_question_subject.setdefault(question_id, subject)
        if previous != subject or question_subject[question_id] != subject:
            raise ValueError("analysis rows violate subject/question consistency")
        question_indices[row_index] = question_lookup[pair]
        subject_index = subject_lookup.get(subject)
        if subject_index is None:
            subject_indices[row_index] = -1
        else:
            subject_indices[row_index] = subject_index
            by_subject[subject].append(row)
    question_indices.setflags(write=False)
    subject_indices.setflags(write=False)
    return _SourceIndex(
        rows=immutable_rows,
        question_indices=question_indices,
        subject_indices=subject_indices,
        question_lookup=question_lookup,
        rows_by_subject={key: tuple(value) for key, value in by_subject.items()},
    )


def _cohort(
    rows: Sequence[Mapping[str, Any]],
    predictor: str,
    target: str,
    question_weights: _QuestionWeights | None = None,
) -> dict[str, Any]:
    _emit_structural("point_cohort_scan")
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
    _emit_structural("point_group_compile")
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
    source_positions: Sequence[int] | None = None,
) -> dict[str, Any]:
    return {
        "rows": rows,
        "rows_by_subject": (
            _index_rows_by_subject(rows) if rows_by_subject is None else rows_by_subject
        ),
        "predictor": predictor,
        "target": target,
        "metadata": {} if metadata is None else dict(metadata),
        "source_positions": source_positions,
    }


@dataclass(frozen=True, slots=True)
class _CompiledAurocGroup:
    all_question_indices: np.ndarray[Any, Any]
    target_missing_question_indices: np.ndarray[Any, Any]
    predictor_missing_question_indices: np.ndarray[Any, Any]
    eligible_question_indices: np.ndarray[Any, Any]
    sorted_targets: np.ndarray[Any, Any]
    tie_boundaries: tuple[tuple[int, int], ...]

    def evaluate(self, question_weights: np.ndarray[Any, Any]) -> float | None:
        if len(self.eligible_question_indices) == 0:
            return None
        weights = question_weights[self.eligible_question_indices]
        positive_total = int(weights[self.sorted_targets].sum())
        negative_total = int(weights[~self.sorted_targets].sum())
        if positive_total == 0 or negative_total == 0:
            return None
        twice_concordant = 0
        negative_below = 0
        for start, end in self.tie_boundaries:
            group_weights = weights[start:end]
            group_targets = self.sorted_targets[start:end]
            positive_weight = int(group_weights[group_targets].sum())
            negative_weight = int(group_weights[~group_targets].sum())
            twice_concordant += 2 * positive_weight * negative_below
            twice_concordant += positive_weight * negative_weight
            negative_below += negative_weight
        return float(twice_concordant / (2 * positive_total * negative_total))


@dataclass(frozen=True, slots=True)
class _CompiledCalibrationGroup:
    all_question_indices: np.ndarray[Any, Any]
    target_missing_question_indices: np.ndarray[Any, Any]
    predictor_missing_question_indices: np.ndarray[Any, Any]
    eligible_question_indices: np.ndarray[Any, Any]
    targets: np.ndarray[Any, Any]
    confidences: np.ndarray[Any, Any]
    bin_indices: np.ndarray[Any, Any]

    def evaluate(self, question_weights: np.ndarray[Any, Any]) -> float | None:
        if len(self.eligible_question_indices) == 0:
            return None
        weights = question_weights[self.eligible_question_indices]
        sample_size = int(weights.sum())
        if sample_size == 0:
            return None
        counts = np.zeros(10, dtype=np.int64)
        target_sums = np.zeros(10, dtype=np.int64)
        np.add.at(counts, self.bin_indices, weights)
        np.add.at(target_sums, self.bin_indices, weights * self.targets)
        confidence_sums = np.bincount(
            self.bin_indices,
            weights=weights * self.confidences,
            minlength=10,
        )
        occupied = counts > 0
        gaps = np.zeros(10, dtype=float)
        gaps[occupied] = np.abs(
            confidence_sums[occupied] / counts[occupied]
            - target_sums[occupied] / counts[occupied]
        )
        return float(np.sum(gaps * counts) / sample_size)


@dataclass(frozen=True, slots=True)
class _CompiledMetricSpec:
    predictor: str
    target: str
    metadata: Mapping[str, Any]
    groups: tuple[_CompiledAurocGroup | _CompiledCalibrationGroup, ...]


@dataclass(slots=True)
class _GroupAccumulator:
    all_question_indices: list[int]
    target_missing_question_indices: list[int]
    predictor_missing_question_indices: list[int]
    eligible_question_indices: list[int]
    targets: list[bool]
    predictors: list[Real]


def _new_accumulator() -> _GroupAccumulator:
    return _GroupAccumulator([], [], [], [], [], [])


def _compile_auroc_group(accumulator: _GroupAccumulator) -> _CompiledAurocGroup:
    _emit_structural("auroc_order_compiled")
    active = list(range(len(accumulator.predictors)))
    try:
        order = sorted(active, key=lambda index: accumulator.predictors[index])
    except (TypeError, ValueError) as error:
        raise ValueError("AUROC scores must be mutually comparable") from error
    boundaries: list[tuple[int, int]] = []
    start = 0
    while start < len(order):
        end = start + 1
        group_score = accumulator.predictors[order[start]]
        try:
            while end < len(order) and bool(
                accumulator.predictors[order[end]] == group_score
            ):
                end += 1
        except (TypeError, ValueError) as error:
            raise ValueError("AUROC scores must support exact tie equality") from error
        boundaries.append((start, end))
        start = end
    question_indices = np.asarray(
        [accumulator.eligible_question_indices[index] for index in order],
        dtype=np.int32,
    )
    targets = np.asarray(
        [accumulator.targets[index] for index in order], dtype=bool
    )
    question_indices.setflags(write=False)
    targets.setflags(write=False)
    return _CompiledAurocGroup(
        all_question_indices=np.asarray(accumulator.all_question_indices, dtype=np.int32),
        target_missing_question_indices=np.asarray(
            accumulator.target_missing_question_indices, dtype=np.int32
        ),
        predictor_missing_question_indices=np.asarray(
            accumulator.predictor_missing_question_indices, dtype=np.int32
        ),
        eligible_question_indices=question_indices,
        sorted_targets=targets,
        tie_boundaries=tuple(boundaries),
    )


def _compile_calibration_group(
    accumulator: _GroupAccumulator,
) -> _CompiledCalibrationGroup:
    _emit_structural("calibration_bins_compiled")
    confidences = np.asarray(accumulator.predictors, dtype=float)
    bins = np.minimum(9, (confidences * 10.0).astype(np.int8))
    question_indices = np.asarray(
        accumulator.eligible_question_indices, dtype=np.int32
    )
    targets = np.asarray(accumulator.targets, dtype=np.int64)
    for array in (confidences, bins, question_indices, targets):
        array.setflags(write=False)
    return _CompiledCalibrationGroup(
        all_question_indices=np.asarray(accumulator.all_question_indices, dtype=np.int32),
        target_missing_question_indices=np.asarray(
            accumulator.target_missing_question_indices, dtype=np.int32
        ),
        predictor_missing_question_indices=np.asarray(
            accumulator.predictor_missing_question_indices, dtype=np.int32
        ),
        eligible_question_indices=question_indices,
        targets=targets,
        confidences=confidences,
        bin_indices=bins,
    )


def _compile_metric_spec(
    source: _SourceIndex,
    spec: Mapping[str, Any],
    *,
    metric_kind: str,
) -> _CompiledMetricSpec:
    predictor = str(spec["predictor"])
    target = str(spec["target"])
    positions = spec.get("source_positions")
    if positions is None:
        positions = range(len(source.rows))
    accumulators = [_new_accumulator() for _ in range(6)]
    for position in positions:
        _emit_structural("spec_source_row_compiled")
        row = source.rows[position]
        question_index = int(source.question_indices[position])
        subject_index = int(source.subject_indices[position])
        group_indices = [0]
        if 0 <= subject_index < len(FIXED_SUBJECTS):
            group_indices.append(subject_index + 1)
        target_value = row.get(target)
        predictor_value = row.get(predictor)
        if target_value is not None and type(target_value) is not bool:
            raise ValueError(f"{target} must be an actual boolean or None")
        if predictor_value is not None and not _finite_real(predictor_value):
            raise ValueError(f"{predictor} must be a finite real number or None")
        for group_index in group_indices:
            accumulator = accumulators[group_index]
            accumulator.all_question_indices.append(question_index)
            if target_value is None:
                accumulator.target_missing_question_indices.append(question_index)
            elif predictor_value is None:
                accumulator.predictor_missing_question_indices.append(question_index)
            else:
                accumulator.eligible_question_indices.append(question_index)
                accumulator.targets.append(target_value)
                accumulator.predictors.append(predictor_value)
    if metric_kind == "auroc":
        groups = tuple(_compile_auroc_group(value) for value in accumulators)
    elif metric_kind == "calibration":
        groups = tuple(_compile_calibration_group(value) for value in accumulators)
    else:
        raise ValueError("unsupported compiled metric kind")
    return _CompiledMetricSpec(
        predictor=predictor,
        target=target,
        metadata=dict(spec.get("metadata", {})),
        groups=groups,
    )


def _metric_key(metric: Mapping[str, Any]) -> _MetricKey:
    return (
        str(metric["predictor"]),
        str(metric["grouping"]),
        None if metric["subject"] is None else str(metric["subject"]),
        metric.get("requested_fraction"),
    )


def _stream_metric_estimates(
    specs: Sequence[_CompiledMetricSpec],
    draw_plan: QuestionDrawPlan,
) -> dict[_MetricKey, list[float | None]]:
    estimates: dict[_MetricKey, list[float | None]] = defaultdict(list)
    for replicate_id in range(draw_plan.replicates):
        question_weights = draw_plan.question_multiplicity_vector(replicate_id)
        for spec in specs:
            values = [group.evaluate(question_weights) for group in spec.groups]
            macro = (
                float(sum(values[1:]) / len(FIXED_SUBJECTS))
                if all(value is not None for value in values[1:])
                else None
            )
            grouping_values = [
                ("pooled", None, values[0]),
                *[
                    ("subject", subject, values[index + 1])
                    for index, subject in enumerate(FIXED_SUBJECTS)
                ],
                ("macro", None, macro),
            ]
            fraction = spec.metadata.get("requested_fraction")
            for grouping, subject, value in grouping_values:
                estimates[(spec.predictor, grouping, subject, fraction)].append(value)
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
    allow_small_fixture: bool = False,
) -> dict[str, Any]:
    """Fixed eleven-feature primary AUROC with compact weighted bootstrap."""

    if target != PRIMARY_TARGET:
        raise ValueError("primary AUROC target must be natural_correct")
    if tuple(feature_registry) != PRIMARY_FEATURE_REGISTRY:
        raise ValueError("primary AUROC feature registry mismatch")
    source = _compile_source_index(
        rows, draw_plan, allow_small_fixture=allow_small_fixture
    )
    metric_rows: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []
    for feature in PRIMARY_FEATURE_REGISTRY:
        metrics, _ = _group_metric_rows(
            source.rows,
            analysis_label="primary_auroc",
            predictor=feature,
            target=target,
            metric_builder=_auroc_metric_row,
            rows_by_subject=source.rows_by_subject,
        )
        metric_rows.extend(metrics)
        specs.append(
            _metric_spec(
                source.rows,
                predictor=feature,
                target=target,
                rows_by_subject=source.rows_by_subject,
            )
        )
    compiled_specs = [
        _compile_metric_spec(source, spec, metric_kind="auroc") for spec in specs
    ]
    estimates = _stream_metric_estimates(
        compiled_specs, draw_plan
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
    compiled_specs: Sequence[_CompiledMetricSpec],
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
        compiled_specs, draw_plan
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
    rows: Sequence[Mapping[str, Any]],
    draw_plan: QuestionDrawPlan,
    *,
    allow_small_fixture: bool = False,
) -> dict[str, Any]:
    """Calibrate natural confidence only against natural correctness."""

    source = _compile_source_index(
        rows, draw_plan, allow_small_fixture=allow_small_fixture
    )
    predictor = "natural_verbalized_confidence"
    specs = [
        _metric_spec(
            source.rows,
            predictor=predictor,
            target=PRIMARY_TARGET,
            rows_by_subject=source.rows_by_subject,
        )
    ]
    return _calibration_analysis(
        specs,
        [_compile_metric_spec(source, specs[0], metric_kind="calibration")],
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
        positions = [
            index
            for index, row in enumerate(flat_rows)
            if row["requested_fraction"] == fraction
        ]
        fraction_rows = [flat_rows[index] for index in positions]
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
                    source_positions=positions,
                )
            )
    return specs


def checkpoint_calibration_analysis(
    rows: Sequence[Mapping[str, Any]],
    draw_plan: QuestionDrawPlan,
    *,
    predictors: Sequence[str] = CHECKPOINT_PREDICTORS,
    allow_small_fixture: bool = False,
) -> dict[str, Any]:
    """Both fixed checkpoint-local calibration families by logical fraction."""

    if tuple(predictors) != CHECKPOINT_PREDICTORS:
        raise ValueError("checkpoint calibration predictors are fixed and exclude entropy")
    flat = _flatten_checkpoint_rows(rows)
    source = _compile_source_index(
        flat, draw_plan, allow_small_fixture=allow_small_fixture
    )
    specs = _checkpoint_specs(source.rows, predictors)
    return _calibration_analysis(
        specs,
        [
            _compile_metric_spec(source, spec, metric_kind="calibration")
            for spec in specs
        ],
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
    allow_small_fixture: bool = False,
) -> dict[str, Any]:
    """Separately labelled checkpoint-local AUROC by logical fraction."""

    if tuple(predictors) != CHECKPOINT_PREDICTORS:
        raise ValueError("secondary checkpoint AUROC predictors are fixed")
    flat = _flatten_checkpoint_rows(rows)
    source = _compile_source_index(
        flat, draw_plan, allow_small_fixture=allow_small_fixture
    )
    specs = _checkpoint_specs(source.rows, predictors)
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
    compiled_specs = [
        _compile_metric_spec(source, spec, metric_kind="auroc") for spec in specs
    ]
    estimates = _stream_metric_estimates(
        compiled_specs, draw_plan
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
    allow_small_fixture: bool = False,
) -> dict[str, Any]:
    """Equal-question paired differences with compact multiplicity bootstrap."""

    if tuple(feature_registry) != PRIMARY_FEATURE_REGISTRY:
        raise ValueError("within-question feature registry mismatch")
    source = _compile_source_index(
        rows, draw_plan, allow_small_fixture=allow_small_fixture
    )
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in source.rows:
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
    compiled_by_feature: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = {}
    for feature, feature_rows in by_feature.items():
        _emit_structural("within_question_compiled")
        question_indices = np.asarray(
            [
                source.question_lookup[(str(row["subject"]), str(row["question_id"]))]
                for row in feature_rows
            ],
            dtype=np.int32,
        )
        paired_differences = np.asarray(
            [row["paired_difference"] for row in feature_rows], dtype=float
        )
        question_indices.setflags(write=False)
        paired_differences.setflags(write=False)
        compiled_by_feature[feature] = (question_indices, paired_differences)
    for replicate_id in range(draw_plan.replicates):
        question_weights = draw_plan.question_multiplicity_vector(replicate_id)
        for feature, (question_indices, paired_differences) in compiled_by_feature.items():
            feature_weights = question_weights[question_indices]
            weight_total = int(feature_weights.sum())
            estimates[feature].append(
                None
                if weight_total == 0
                else float(np.dot(paired_differences, feature_weights) / weight_total)
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

"""Deterministic subject-stratified question bootstrap primitives for Part 1."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from part1_contract import FIXED_SUBJECTS


DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_DEVELOPMENT_REPLICATES = 1_000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_MINIMUM_VALID_FRACTION = 0.95

_BOOTSTRAP_FIELDS = ("replicate_id", "draw_index", "draw_id")


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _subject_question_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, list[str]]]:
    subjects: list[str] = []
    questions_by_subject: dict[str, list[str]] = defaultdict(list)
    pairs: set[tuple[str, str]] = set()
    question_subject: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("question frame rows must be mappings")
        subject = _require_nonempty_string(row.get("subject"), "subject")
        question_id = _require_nonempty_string(row.get("question_id"), "question_id")
        pair = (subject, question_id)
        if pair in pairs:
            raise ValueError("question frame requires unique subject/question rows")
        previous_subject = question_subject.setdefault(question_id, subject)
        if previous_subject != subject:
            raise ValueError("question frame violates subject/question consistency")
        pairs.add(pair)
        if subject not in questions_by_subject:
            subjects.append(subject)
        questions_by_subject[subject].append(question_id)
    if not rows:
        raise ValueError("question frame must not be empty")
    return subjects, dict(questions_by_subject)


def build_question_draw_plan(
    question_frame: Sequence[Mapping[str, Any]],
    *,
    replicates: int = DEFAULT_DEVELOPMENT_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    small_fixture: bool = False,
) -> list[dict[str, Any]]:
    """Build one reusable, explicit draw plan from unique authoritative questions."""

    if type(replicates) is not int or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(small_fixture) is not bool:
        raise ValueError("small_fixture must be boolean")
    subjects, questions_by_subject = _subject_question_rows(question_frame)
    if not small_fixture and subjects != list(FIXED_SUBJECTS):
        raise ValueError("question frame must have exact fixed subject presence and order")

    generator = np.random.default_rng(seed)
    plan: list[dict[str, Any]] = []
    for replicate_id in range(replicates):
        for subject_index, subject in enumerate(subjects):
            question_ids = questions_by_subject[subject]
            selected_indices = generator.integers(
                0, len(question_ids), size=len(question_ids)
            )
            for draw_index, selected_index in enumerate(selected_indices.tolist()):
                plan.append(
                    {
                        "replicate_id": replicate_id,
                        "subject": subject,
                        "draw_index": draw_index,
                        "draw_id": (
                            f"bootstrap-r{replicate_id:06d}-s{subject_index:02d}"
                            f"-d{draw_index:06d}"
                        ),
                        "question_id": question_ids[int(selected_index)],
                    }
                )
    return plan


def _validate_draw_plan(
    draw_plan: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    by_pair: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    draw_ids: set[str] = set()
    question_subject: dict[str, str] = {}
    previous_order: tuple[int, int] | None = None
    for draw in draw_plan:
        if not isinstance(draw, Mapping):
            raise ValueError("draw plan rows must be mappings")
        replicate_id = draw.get("replicate_id")
        draw_index = draw.get("draw_index")
        if type(replicate_id) is not int or replicate_id < 0:
            raise ValueError("replicate_id must be a nonnegative integer")
        if type(draw_index) is not int or draw_index < 0:
            raise ValueError("draw_index must be a nonnegative integer")
        draw_id = _require_nonempty_string(draw.get("draw_id"), "draw_id")
        if draw_id in draw_ids:
            raise ValueError("draw plan requires globally unique draw_id values")
        draw_ids.add(draw_id)
        subject = _require_nonempty_string(draw.get("subject"), "subject")
        question_id = _require_nonempty_string(draw.get("question_id"), "question_id")
        previous_subject = question_subject.setdefault(question_id, subject)
        if previous_subject != subject:
            raise ValueError("draw plan violates subject/question consistency")
        order = (replicate_id, len(draw_ids) - 1)
        if previous_order is not None and order[0] < previous_order[0]:
            raise ValueError("draw plan must have deterministic replicate order")
        previous_order = order
        by_pair[(subject, question_id)].append(draw)
    if not draw_plan:
        raise ValueError("draw plan must not be empty")
    return dict(by_pair)


def expand_question_draws(
    draw_plan: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Duplicate every row associated with a selected question once per draw ID."""

    draws_by_pair = _validate_draw_plan(draw_plan)
    rows_by_pair: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    question_subject: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("analysis rows must be mappings")
        for field in _BOOTSTRAP_FIELDS:
            if field in row:
                raise ValueError(f"analysis row contains reserved bootstrap field {field}")
        subject = _require_nonempty_string(row.get("subject"), "subject")
        question_id = _require_nonempty_string(row.get("question_id"), "question_id")
        previous_subject = question_subject.setdefault(question_id, subject)
        if previous_subject != subject:
            raise ValueError("analysis rows violate subject/question consistency")
        rows_by_pair[(subject, question_id)].append(row)

    plan_question_subject = {
        question_id: subject for subject, question_id in draws_by_pair
    }
    for question_id, subject in question_subject.items():
        plan_subject = plan_question_subject.get(question_id)
        if plan_subject is not None and plan_subject != subject:
            raise ValueError("draw plan and analysis rows violate subject/question consistency")

    expanded: list[dict[str, Any]] = []
    for draw in draw_plan:
        pair = (draw["subject"], draw["question_id"])
        for source in rows_by_pair.get(pair, []):
            output = deepcopy(dict(source))
            output.update(
                {
                    "replicate_id": int(draw["replicate_id"]),
                    "draw_index": int(draw["draw_index"]),
                    "draw_id": str(draw["draw_id"]),
                }
            )
            expanded.append(output)
    return expanded


def percentile_interval(
    estimates: Sequence[Real | None],
    *,
    requested_replicates: int | None = None,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    minimum_valid_fraction: float = DEFAULT_MINIMUM_VALID_FRACTION,
) -> dict[str, Any]:
    """Summarize finite replicate estimates with a linear percentile interval."""

    if requested_replicates is None:
        requested_replicates = len(estimates)
    if type(requested_replicates) is not int or requested_replicates <= 0:
        raise ValueError("requested_replicates must be a positive integer")
    if requested_replicates < len(estimates):
        raise ValueError("requested_replicates cannot be smaller than estimates")
    if isinstance(confidence_level, bool) or not isinstance(confidence_level, Real):
        raise ValueError("confidence_level must be a real number in (0,1)")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be a real number in (0,1)")
    if isinstance(minimum_valid_fraction, bool) or not isinstance(
        minimum_valid_fraction, Real
    ):
        raise ValueError("minimum_valid_fraction must be a real number in [0,1]")
    if not 0.0 <= float(minimum_valid_fraction) <= 1.0:
        raise ValueError("minimum_valid_fraction must be a real number in [0,1]")

    valid: list[float] = []
    for estimate in estimates:
        if estimate is None:
            continue
        if isinstance(estimate, bool) or not isinstance(estimate, Real):
            raise ValueError("estimate must be a finite real number or None")
        numeric = float(estimate)
        if not math.isfinite(numeric):
            raise ValueError("estimate must be a finite real number or None")
        valid.append(numeric)

    valid_count = len(valid)
    invalid_count = requested_replicates - valid_count
    valid_fraction = valid_count / requested_replicates
    interval_valid = valid_count > 0 and valid_fraction >= float(minimum_valid_fraction)
    if interval_valid:
        tail = (1.0 - float(confidence_level)) / 2.0
        lower, upper = np.percentile(
            np.asarray(valid, dtype=float),
            [tail * 100.0, (1.0 - tail) * 100.0],
            method="linear",
        ).tolist()
        reason = None
    else:
        lower = None
        upper = None
        reason = "insufficient_valid_bootstrap_replicates"
    return {
        "requested_replicates": int(requested_replicates),
        "valid_replicates": int(valid_count),
        "invalid_replicates": int(invalid_count),
        "valid_fraction": float(valid_fraction),
        "confidence_level": float(confidence_level),
        "percentile_method": "linear",
        "lower": None if lower is None else float(lower),
        "upper": None if upper is None else float(upper),
        "interval_valid": bool(interval_valid),
        "interval_reason": reason,
        "warning": reason,
    }


__all__ = [
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_DEVELOPMENT_REPLICATES",
    "build_question_draw_plan",
    "expand_question_draws",
    "percentile_interval",
]

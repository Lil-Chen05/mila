"""Compact subject-stratified question bootstrap primitives for Part 1."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from part1_contract import FIXED_SUBJECTS


DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_DEVELOPMENT_REPLICATES = 1_000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_MINIMUM_VALID_FRACTION = 0.95
MAX_AUDIT_MATERIALIZED_ROWS = 100_000

_BOOTSTRAP_FIELDS = ("replicate_id", "draw_index", "draw_id")


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _canonical_questions(
    rows: Sequence[Mapping[str, Any]], *, small_fixture: bool
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    if type(small_fixture) is not bool:
        raise ValueError("small_fixture must be boolean")
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
        previous = question_subject.setdefault(question_id, subject)
        if previous != subject:
            raise ValueError("each question_id must belong to exactly one subject")
        pairs.add(pair)
        if subject not in questions_by_subject:
            subjects.append(subject)
        questions_by_subject[subject].append(question_id)
    if not rows:
        raise ValueError("question frame must not be empty")
    if not small_fixture and subjects != list(FIXED_SUBJECTS):
        raise ValueError("question frame must have exact fixed subject presence and order")
    return tuple(subjects), tuple(
        tuple(questions_by_subject[subject]) for subject in subjects
    )


def _index_dtype(maximum_question_count: int) -> np.dtype[Any]:
    if maximum_question_count <= np.iinfo(np.uint8).max + 1:
        return np.dtype(np.uint8)
    if maximum_question_count <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


@dataclass(frozen=True, slots=True)
class QuestionDrawPlan:
    """Immutable compact selected-question indices with lazy draw audit views."""

    subjects: tuple[str, ...]
    question_ids_by_subject: tuple[tuple[str, ...], ...]
    replicates: int
    seed: int
    small_fixture: bool
    _dtype_string: str
    _selected_index_bytes: bytes

    def __post_init__(self) -> None:
        if not self.subjects or len(self.subjects) != len(self.question_ids_by_subject):
            raise ValueError("compact plan subjects and question groups must align")
        if len(set(self.subjects)) != len(self.subjects):
            raise ValueError("compact plan subjects must be unique")
        if type(self.replicates) is not int or self.replicates <= 0:
            raise ValueError("replicates must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.small_fixture) is not bool:
            raise ValueError("small_fixture must be boolean")
        if not self.small_fixture and self.subjects != tuple(FIXED_SUBJECTS):
            raise ValueError("compact plan must have exact fixed subject presence and order")
        seen_questions: set[str] = set()
        for subject, question_ids in zip(
            self.subjects, self.question_ids_by_subject, strict=True
        ):
            _require_nonempty_string(subject, "subject")
            if not question_ids or len(set(question_ids)) != len(question_ids):
                raise ValueError("compact plan question IDs must be fixed, nonempty, and unique")
            for question_id in question_ids:
                _require_nonempty_string(question_id, "question_id")
                if question_id in seen_questions:
                    raise ValueError("each question_id must belong to exactly one subject")
                seen_questions.add(question_id)
        try:
            dtype = np.dtype(self._dtype_string)
        except TypeError as error:
            raise ValueError("compact plan index dtype is invalid") from error
        if dtype.kind != "u":
            raise ValueError("compact plan selected indices require an unsigned integer dtype")
        expected_bytes = self.selected_index_cell_count * dtype.itemsize
        if len(self._selected_index_bytes) != expected_bytes:
            raise ValueError("compact plan selected index storage has invalid shape")
        selected = self.selected_indices
        offset = 0
        for question_ids in self.question_ids_by_subject:
            count = len(question_ids)
            segment = selected[:, offset : offset + count]
            if np.any(segment >= count):
                raise ValueError("selected index out of range for subject")
            offset += count

    @property
    def total_question_count(self) -> int:
        return sum(len(question_ids) for question_ids in self.question_ids_by_subject)

    @property
    def selected_index_cell_count(self) -> int:
        return self.replicates * self.total_question_count

    @property
    def logical_draw_count(self) -> int:
        return self.selected_index_cell_count

    @property
    def estimated_storage_bytes(self) -> int:
        return len(self._selected_index_bytes)

    @staticmethod
    def estimated_selected_index_bytes(replicates: int, total_questions: int) -> int:
        if type(replicates) is not int or replicates <= 0:
            raise ValueError("replicates must be a positive integer")
        if type(total_questions) is not int or total_questions <= 0:
            raise ValueError("total_questions must be a positive integer")
        return replicates * total_questions * np.dtype(np.uint32).itemsize

    @property
    def selected_indices(self) -> np.ndarray[Any, np.dtype[np.unsignedinteger[Any]]]:
        return np.frombuffer(
            self._selected_index_bytes, dtype=np.dtype(self._dtype_string)
        ).reshape(self.replicates, self.total_question_count)

    def _validate_replicate_id(self, replicate_id: int) -> None:
        if type(replicate_id) is not int or not 0 <= replicate_id < self.replicates:
            raise ValueError("replicate_id is outside the compact draw plan")

    def iter_draw_rows(self, replicate_id: int) -> Iterator[dict[str, Any]]:
        """Yield deterministic audit rows for one requested replicate only."""

        self._validate_replicate_id(replicate_id)
        selected = self.selected_indices[replicate_id]
        offset = 0
        for subject_index, (subject, question_ids) in enumerate(
            zip(self.subjects, self.question_ids_by_subject, strict=True)
        ):
            for draw_index in range(len(question_ids)):
                selected_index = int(selected[offset + draw_index])
                yield {
                    "replicate_id": replicate_id,
                    "subject": subject,
                    "draw_index": draw_index,
                    "draw_id": (
                        f"bootstrap-r{replicate_id:06d}-s{subject_index:02d}"
                        f"-d{draw_index:06d}"
                    ),
                    "question_id": question_ids[selected_index],
                }
            offset += len(question_ids)

    def question_multiplicities(
        self, replicate_id: int
    ) -> dict[tuple[str, str], int]:
        """Return exact integer selection multiplicity for every canonical question."""

        self._validate_replicate_id(replicate_id)
        multiplicities = self.question_multiplicity_vector(replicate_id)
        output: dict[tuple[str, str], int] = {}
        offset = 0
        for subject, question_ids in zip(
            self.subjects, self.question_ids_by_subject, strict=True
        ):
            counts = multiplicities[offset : offset + len(question_ids)]
            for question_id, count in zip(question_ids, counts.tolist(), strict=True):
                output[(subject, question_id)] = int(count)
            offset += len(question_ids)
        return output

    def question_multiplicity_vector(self, replicate_id: int) -> np.ndarray[Any, Any]:
        """Return one compact integer vector aligned to canonical question order."""

        self._validate_replicate_id(replicate_id)
        selected = self.selected_indices[replicate_id]
        output = np.empty(self.total_question_count, dtype=np.int64)
        offset = 0
        for question_ids in self.question_ids_by_subject:
            count = len(question_ids)
            output[offset : offset + count] = np.bincount(
                selected[offset : offset + count].astype(np.int64),
                minlength=count,
            )
            offset += count
        return output

    def materialize_draw_rows(self, *, max_rows: int) -> list[dict[str, Any]]:
        if type(max_rows) is not int or max_rows <= 0:
            raise ValueError("max_rows must be a positive integer")
        if self.logical_draw_count > MAX_AUDIT_MATERIALIZED_ROWS:
            raise ValueError("compact plan exceeds the fixed audit ceiling")
        if self.logical_draw_count > max_rows:
            raise ValueError("compact plan exceeds explicit audit materialization limit")
        return [
            row
            for replicate_id in range(self.replicates)
            for row in self.iter_draw_rows(replicate_id)
        ]


def question_draw_plan_from_indices(
    question_frame: Sequence[Mapping[str, Any]],
    selected_indices: np.ndarray[Any, Any],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    small_fixture: bool = False,
) -> QuestionDrawPlan:
    """Construct a compact plan from explicit local subject indices."""

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    subjects, questions = _canonical_questions(
        question_frame, small_fixture=small_fixture
    )
    array = np.asarray(selected_indices)
    total_questions = sum(len(group) for group in questions)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] != total_questions:
        raise ValueError("selected indices must have shape (replicates, total_questions)")
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError("selected indices must be an integer array")
    offset = 0
    for group in questions:
        count = len(group)
        segment = array[:, offset : offset + count]
        if np.any(segment < 0) or np.any(segment >= count):
            raise ValueError("selected index out of range for subject")
        offset += count
    dtype = _index_dtype(max(len(group) for group in questions))
    owned = np.ascontiguousarray(array, dtype=dtype)
    return QuestionDrawPlan(
        subjects=subjects,
        question_ids_by_subject=questions,
        replicates=int(owned.shape[0]),
        seed=seed,
        small_fixture=small_fixture,
        _dtype_string=dtype.str,
        _selected_index_bytes=owned.tobytes(order="C"),
    )


def build_question_draw_plan(
    question_frame: Sequence[Mapping[str, Any]],
    *,
    replicates: int = DEFAULT_DEVELOPMENT_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    small_fixture: bool = False,
) -> QuestionDrawPlan:
    """Build one compact reusable plan using the canonical RNG loop order."""

    if type(replicates) is not int or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    subjects, questions = _canonical_questions(
        question_frame, small_fixture=small_fixture
    )
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    total_questions = sum(len(group) for group in questions)
    selected = np.empty((replicates, total_questions), dtype=np.int64)
    generator = np.random.default_rng(seed)
    for replicate_id in range(replicates):
        offset = 0
        for group in questions:
            count = len(group)
            selected[replicate_id, offset : offset + count] = generator.integers(
                0, count, size=count
            )
            offset += count
    question_frame_canonical = [
        {"subject": subject, "question_id": question_id}
        for subject, group in zip(subjects, questions, strict=True)
        for question_id in group
    ]
    return question_draw_plan_from_indices(
        question_frame_canonical,
        selected,
        seed=seed,
        small_fixture=small_fixture,
    )


def question_draw_plan_from_rows(
    question_frame: Sequence[Mapping[str, Any]],
    draw_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    small_fixture: bool = False,
) -> QuestionDrawPlan:
    """Explicitly validate and compact legacy per-draw rows for fixtures/audits."""

    subjects, questions = _canonical_questions(
        question_frame, small_fixture=small_fixture
    )
    if not draw_rows:
        raise ValueError("legacy draw rows must not be empty")
    replicate_values: list[int] = []
    draw_ids: set[str] = set()
    triples: set[tuple[int, str, int]] = set()
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    question_lookup = {
        (subject, question_id): index
        for subject, group in zip(subjects, questions, strict=True)
        for index, question_id in enumerate(group)
    }
    actual_order: list[tuple[int, str, int]] = []
    for row in draw_rows:
        if not isinstance(row, Mapping):
            raise ValueError("legacy draw rows must be mappings")
        replicate_id = row.get("replicate_id")
        draw_index = row.get("draw_index")
        if type(replicate_id) is not int or replicate_id < 0:
            raise ValueError("legacy draw rows require nonnegative replicate IDs")
        if type(draw_index) is not int or draw_index < 0:
            raise ValueError("legacy draw rows require nonnegative draw indices")
        replicate_values.append(replicate_id)
        subject = _require_nonempty_string(row.get("subject"), "subject")
        question_id = _require_nonempty_string(row.get("question_id"), "question_id")
        if (subject, question_id) not in question_lookup:
            raise ValueError("legacy draw row question is absent from the question frame")
        draw_id = _require_nonempty_string(row.get("draw_id"), "draw_id")
        if draw_id in draw_ids:
            raise ValueError("legacy draw rows require unique draw_id values")
        draw_ids.add(draw_id)
        triple = (replicate_id, subject, draw_index)
        if triple in triples:
            raise ValueError("legacy draw rows require contiguous draw indices without duplicates")
        triples.add(triple)
        groups[(replicate_id, subject)].append(draw_index)
        actual_order.append(triple)
    replicate_ids = sorted(set(replicate_values))
    if replicate_ids != list(range(len(replicate_ids))):
        raise ValueError("legacy draw rows require contiguous replicate IDs")
    for replicate_id in replicate_ids:
        for subject, group in zip(subjects, questions, strict=True):
            indices = groups.get((replicate_id, subject), [])
            if len(indices) != len(group):
                raise ValueError("legacy draw rows require consistent per-subject draw counts")
            if sorted(indices) != list(range(len(group))):
                raise ValueError("legacy draw rows require contiguous draw indices")
    expected_order = [
        (replicate_id, subject, draw_index)
        for replicate_id in replicate_ids
        for subject, group in zip(subjects, questions, strict=True)
        for draw_index in range(len(group))
    ]
    if actual_order != expected_order:
        raise ValueError("legacy draw rows require canonical replicate/subject/draw order")
    selected = np.empty(
        (len(replicate_ids), sum(len(group) for group in questions)), dtype=np.int64
    )
    cursor = 0
    for row in draw_rows:
        subject = str(row["subject"])
        question_id = str(row["question_id"])
        selected[int(row["replicate_id"]), cursor % selected.shape[1]] = question_lookup[
            (subject, question_id)
        ]
        cursor += 1
    canonical_frame = [
        {"subject": subject, "question_id": question_id}
        for subject, group in zip(subjects, questions, strict=True)
        for question_id in group
    ]
    return question_draw_plan_from_indices(
        canonical_frame,
        selected,
        seed=seed,
        small_fixture=small_fixture,
    )


def expand_question_draws(
    draw_plan: QuestionDrawPlan,
    rows: Sequence[Mapping[str, Any]],
    *,
    replicate_id: int,
    max_rows: int,
) -> list[dict[str, Any]]:
    """Materialize one explicitly bounded replicate for audit/testing only."""

    if not isinstance(draw_plan, QuestionDrawPlan):
        raise ValueError("draw_plan must be a compact QuestionDrawPlan")
    if type(max_rows) is not int or max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")
    rows_by_pair: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    canonical_pairs = {
        (subject, question_id)
        for subject, group in zip(
            draw_plan.subjects, draw_plan.question_ids_by_subject, strict=True
        )
        for question_id in group
    }
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("analysis rows must be mappings")
        for field in _BOOTSTRAP_FIELDS:
            if field in row:
                raise ValueError(f"analysis row contains reserved bootstrap field {field}")
        pair = (
            _require_nonempty_string(row.get("subject"), "subject"),
            _require_nonempty_string(row.get("question_id"), "question_id"),
        )
        if pair not in canonical_pairs:
            raise ValueError("analysis row question is absent from the compact draw plan")
        rows_by_pair[pair].append(row)
    logical_rows = sum(
        len(rows_by_pair[(draw["subject"], draw["question_id"])])
        for draw in draw_plan.iter_draw_rows(replicate_id)
    )
    if logical_rows > max_rows:
        raise ValueError("expanded rows exceed explicit audit materialization limit")
    if logical_rows > MAX_AUDIT_MATERIALIZED_ROWS:
        raise ValueError("expanded rows exceed the fixed audit ceiling")
    expanded: list[dict[str, Any]] = []
    for draw in draw_plan.iter_draw_rows(replicate_id):
        pair = (draw["subject"], draw["question_id"])
        for source in rows_by_pair[pair]:
            output = dict(source)
            output.update(
                {
                    "replicate_id": draw["replicate_id"],
                    "draw_index": draw["draw_index"],
                    "draw_id": draw["draw_id"],
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
    "MAX_AUDIT_MATERIALIZED_ROWS",
    "QuestionDrawPlan",
    "build_question_draw_plan",
    "question_draw_plan_from_indices",
    "question_draw_plan_from_rows",
    "expand_question_draws",
    "percentile_interval",
]

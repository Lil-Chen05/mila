"""Login-safe tests for deterministic 200q analysis helpers.

These tests use small in-memory tables only: no model, tokenizer, dataset, or
saved experiment result is loaded.
"""

import pandas as pd
import pytest

from analysis import analyze_200q


def test_select_individual_trace_qids_is_balanced_diverse_and_reproducible():
    rows = pd.DataFrame(
        [
            {"qid": 0, "subject": "physics", "final_correct": True},
            {"qid": 1, "subject": "physics", "final_correct": True},
            {"qid": 2, "subject": "history", "final_correct": True},
            {"qid": 3, "subject": "law", "final_correct": True},
            {"qid": 4, "subject": "biology", "final_correct": True},
            {"qid": 5, "subject": "economics", "final_correct": False},
            {"qid": 6, "subject": "economics", "final_correct": False},
            {"qid": 7, "subject": "philosophy", "final_correct": False},
            {"qid": 8, "subject": "chemistry", "final_correct": False},
            {"qid": 9, "subject": "psychology", "final_correct": False},
        ]
    )

    assert hasattr(analyze_200q, "select_individual_trace_qids")
    first = analyze_200q.select_individual_trace_qids(rows, per_group=3, seed=7)
    second = analyze_200q.select_individual_trace_qids(rows, per_group=3, seed=7)

    assert first == second
    assert set(first) == {True, False}
    assert all(len(first[label]) == 3 for label in (True, False))

    by_qid = rows.set_index("qid")
    for label in (True, False):
        chosen = by_qid.loc[first[label]]
        assert chosen["final_correct"].eq(label).all()
        assert chosen["subject"].nunique() == 3


def test_select_correlated_signal_pairs_ranks_unique_nonexcluded_pairs():
    signals = ["a", "b", "c", "final_correct"]
    corr = pd.DataFrame(
        [
            [1.0, -0.8, 0.4, 0.9],
            [-0.8, 1.0, 0.6, 0.2],
            [0.4, 0.6, 1.0, -0.7],
            [0.9, 0.2, -0.7, 1.0],
        ],
        index=signals,
        columns=signals,
    )

    assert hasattr(analyze_200q, "select_correlated_signal_pairs")
    selected = analyze_200q.select_correlated_signal_pairs(
        corr,
        n_pairs=2,
        excluded_signals={"final_correct"},
        excluded_pairs={frozenset(("a", "b"))},
    )

    assert selected == [("b", "c", 0.6), ("a", "c", 0.4)]


def test_summarize_h_ans_distributions_drops_missing_and_uses_threshold():
    rows = pd.DataFrame(
        {
            "final_correct": [True, True, True, False, False],
            "H_ans_prior": [0.0, 0.1, float("nan"), 0.02, 0.08],
        }
    )

    assert hasattr(analyze_200q, "summarize_h_ans_distributions")
    summary = analyze_200q.summarize_h_ans_distributions(
        rows, ["H_ans_prior"], near_zero_threshold=0.05
    ).set_index("final_correct")

    assert summary.loc[True, "n"] == 2
    assert summary.loc[False, "n"] == 2
    assert summary.loc[True, "mean"] == pytest.approx(0.05)
    assert summary.loc[False, "median"] == pytest.approx(0.05)
    assert summary.loc[True, "near_zero_frac"] == pytest.approx(0.5)
    assert summary.loc[False, "near_zero_frac"] == pytest.approx(0.5)

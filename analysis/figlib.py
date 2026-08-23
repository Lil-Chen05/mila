"""Shared helpers for the final-r5000 report figures (login-safe, CPU-only).

Pure pandas/numpy/matplotlib over the fixed CSVs, the merged parquet, and the
recovered-confidence artifacts. Never loads a model or dataset. All four report
figures import from here so data construction, bootstrap, and house style stay
consistent and are defined once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "results/part1/6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c"
MERGED = RUN / "merged"
FIXED = RUN / "analysis/final-r5000"
CONF = REPO / "analysis/final-r5000/confidence"
OUT = REPO / "analysis/final-r5000/figures"
CACHE = REPO / "analysis/final-r5000/cache"

FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_DECILES = 10
BOOT_SEED = 42

# Subjects: canonical id -> short paper label.
SUBJECTS = {
    "high_school_mathematics": "Mathematics",
    "high_school_physics": "Physics",
    "high_school_chemistry": "Chemistry",
    "high_school_biology": "Biology",
    "high_school_psychology": "Psychology",
}
SUBJECT_ORDER = list(SUBJECTS.values())
SUBJECT_COLORS = {
    "Mathematics": "#4C72B0",
    "Physics": "#DD8452",
    "Chemistry": "#55A868",
    "Biology": "#C44E52",
    "Psychology": "#8172B3",
}
# Outcome colours (colour-blind safe).
COL_CORRECT = "#2166AC"
COL_INCORRECT = "#B2182B"


def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.8,
    })


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _reasoning_slice(entropy, boundaries_json):
    b = json.loads(boundaries_json) if isinstance(boundaries_json, str) else None
    if not b or "generated_start" not in b:
        return None
    s, e = b["generated_start"], b["generated_end_exclusive"]
    arr = np.asarray(entropy[s:e], dtype=float)
    return arr if arr.size else None


def decile_means(entropy_slice, n=N_DECILES):
    """Per-run mean reasoning entropy in each of ``n`` equal token-progress bins.

    Positions are binned by normalized token progress; runs shorter than ``n``
    tokens leave later bins as NaN (handled by nan-aware aggregation upstream).
    """
    L = entropy_slice.size
    out = np.full(n, np.nan)
    if L == 0:
        return out
    idx = np.minimum((np.arange(L) * n // L), n - 1)
    for b in range(n):
        m = entropy_slice[idx == b]
        if m.size:
            out[b] = m.mean()
    return out


def load_natural(with_traces=False):
    """Natural-run table. ``with_traces`` adds the per-token entropy slice."""
    cols = ["raw_record_id", "question_id", "subject", "run_id", "natural_correct",
            "answer_parse_status", "reasoning_status", "mean_reasoning_entropy_nats",
            "tail_reasoning_entropy_nats", "reasoning_token_count"]
    if with_traces:
        cols += ["per_token_entropy_nats", "reasoning_boundaries"]
    df = pd.read_parquet(MERGED / "natural_results.parquet", columns=cols)
    df["subject_label"] = df["subject"].map(SUBJECTS)
    return df


def load_checkpoints():
    cols = ["parent_raw_record_id", "subject", "requested_fraction", "answer_entropy_nats",
            "maximum_ad_probability", "forced_answer", "checkpoint_local_correct",
            "ad_probabilities_float32", "answer_parse_status"]
    df = pd.read_parquet(MERGED / "checkpoint_results.parquet", columns=cols)
    df["subject_label"] = df["subject"].map(SUBJECTS)
    return df


def load_recovered_confidence(level="checkpoint"):
    fn = ("recovered_confidence_checkpoint.parquet" if level == "checkpoint"
          else "recovered_confidence_natural.parquet")
    return pd.read_parquet(CONF / fn)


def evaluable_ids(nat=None):
    """raw_record_ids of the primary evaluable cohort (parsed A-D natural answer)."""
    nat = load_natural() if nat is None else nat
    return set(nat.loc[nat["natural_correct"].notna(), "raw_record_id"])


# --------------------------------------------------------------------------
# Question-cluster bootstrap (subject-stratified, matching the fixed method)
# --------------------------------------------------------------------------
def question_cluster_bootstrap(values, question_id, subject, stat_fn,
                               n_boot=2000, seed=BOOT_SEED):
    """Percentile CI band by resampling questions within subject with replacement.

    ``values`` is an (n_runs, k) array (k columns = bins/fractions). ``stat_fn``
    maps a value sub-array to a length-k vector (e.g. nan-mean over runs). Returns
    (point, lo, hi) each length k. Preserves within-question run dependence.
    """
    values = np.asarray(values, dtype=float)
    qid = np.asarray(question_id)
    subj = np.asarray(subject)
    point = stat_fn(values)

    # Group row indices by question, and questions by subject.
    q_to_rows = {}
    for i, q in enumerate(qid):
        q_to_rows.setdefault(q, []).append(i)
    questions = np.array(list(q_to_rows.keys()))
    q_subj = np.array([subj[q_to_rows[q][0]] for q in questions])
    rows_per_q = [np.asarray(q_to_rows[q]) for q in questions]

    rng = np.random.default_rng(seed)
    subj_levels = np.unique(q_subj)
    q_idx_by_subj = {s: np.where(q_subj == s)[0] for s in subj_levels}

    boot = np.empty((n_boot, point.size), dtype=float)
    for b in range(n_boot):
        picked = []
        for s in subj_levels:
            pool = q_idx_by_subj[s]
            draw = rng.integers(0, pool.size, pool.size)
            for qi in pool[draw]:
                picked.append(rows_per_q[qi])
        rows = np.concatenate(picked)
        boot[b] = stat_fn(values[rows])
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    return point, lo, hi


def nanmean_rows(v):
    return np.nanmean(v, axis=0)


# --------------------------------------------------------------------------
# Prefix reasoning entropy (info available THROUGH each checkpoint)
# --------------------------------------------------------------------------
def build_prefix_entropy(ev, cp=None):
    """Per-run mean reasoning entropy over the first ``k_keep`` NATURAL reasoning
    tokens retained at each checkpoint fraction (never forced-probe tokens).

    Returns an (n_runs, 11) array aligned to ``ev`` rows and ``FRACTIONS``.
    Fraction 0.0 (k_keep=0) is NaN by construction (pre-reasoning baseline).
    """
    if cp is None:
        cp = pd.read_parquet(MERGED / "checkpoint_results.parquet",
                             columns=["parent_raw_record_id", "requested_fraction", "k_keep"])
    kk = (cp.pivot_table(index="parent_raw_record_id", columns="requested_fraction",
                         values="k_keep", aggfunc="max")
            .reindex(index=ev["raw_record_id"], columns=FRACTIONS))
    kk_arr = kk.to_numpy(dtype=float)
    out = np.full((len(ev), len(FRACTIONS)), np.nan)
    for i, row in enumerate(ev.itertuples(index=False)):
        sl = _reasoning_slice(row.per_token_entropy_nats, row.reasoning_boundaries)
        if sl is None:
            continue
        for j in range(len(FRACTIONS)):
            k = kk_arr[i, j]
            if np.isfinite(k) and k >= 1:
                out[i, j] = sl[:int(k)].mean()
    return out


# --------------------------------------------------------------------------
# AUROC and question-cluster bootstrap for discrimination curves
# --------------------------------------------------------------------------
def _avg_ranks(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0    # average rank (1-based)
        i = j + 1
    return ranks


def fast_auroc(scores, y):
    """AUROC with higher score -> positive class. ``y`` boolean. NaN-safe caller."""
    y = np.asarray(y, dtype=bool)
    n_pos = int(y.sum()); n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = _avg_ranks(np.asarray(scores, dtype=float))
    return (r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auroc_curve_bootstrap(P, y, qid, subj, n_boot=2000, seed=BOOT_SEED):
    """AUROC per column of oriented predictor matrix ``P`` (higher -> correct),
    with subject-stratified question-cluster percentile CIs. NaN predictors are
    dropped per column. Returns (point, lo, hi, n) each length = P.shape[1]."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=bool)
    qid = np.asarray(qid); subj = np.asarray(subj)
    k = P.shape[1]

    def col_aurocs(rows):
        res = np.full(k, np.nan)
        yr = y[rows]
        for j in range(k):
            s = P[rows, j]
            m = np.isfinite(s)
            if m.sum() >= 2:
                res[j] = fast_auroc(s[m], yr[m])
        return res

    all_rows = np.arange(P.shape[0])
    point = col_aurocs(all_rows)
    n = np.array([int(np.isfinite(P[:, j]).sum()) for j in range(k)])

    # question -> rows, subject grouping
    q_to_rows = {}
    for i, q in enumerate(qid):
        q_to_rows.setdefault(q, []).append(i)
    questions = np.array(list(q_to_rows.keys()))
    q_subj = np.array([subj[q_to_rows[q][0]] for q in questions])
    rows_per_q = [np.asarray(q_to_rows[q]) for q in questions]
    rng = np.random.default_rng(seed)
    subj_levels = np.unique(q_subj)
    q_idx_by_subj = {s: np.where(q_subj == s)[0] for s in subj_levels}

    boot = np.empty((n_boot, k), dtype=float)
    for b in range(n_boot):
        picked = []
        for s in subj_levels:
            pool = q_idx_by_subj[s]
            draw = rng.integers(0, pool.size, pool.size)
            picked.extend(rows_per_q[qi] for qi in pool[draw])
        boot[b] = col_aurocs(np.concatenate(picked))
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    return point, lo, hi, n

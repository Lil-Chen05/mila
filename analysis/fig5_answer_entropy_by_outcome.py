"""Exploratory: answer-choice entropy across reasoning, split by whether the
NATURAL final answer was correct.

Question: does the A-D distribution concentrate as reasoning progresses even on
trajectories that ultimately yield an incorrect natural final answer?

Figure 5 of the report. Reuses analysis/figlib.py for loading, house style, and
the subject-stratified question-cluster bootstrap, so it matches Figures 1-4. The
figure and its values JSON go to analysis/final-r5000/figures/ and are mirrored to
report/figures/ for the LaTeX build; the full per-position estimate table stays in
analysis/final-r5000/diagnostics/.

Wording rule followed throughout: a trajectory is never called correct or
incorrect; groups are "trajectories yielding a correct/incorrect natural final
answer". Falling answer-choice entropy is a statement about the model's A-D
distribution under the forced-close probe, not about the reasoning being right.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as F

N_BOOT = 5000
SEED = F.BOOT_SEED
OUT = F.OUT                                   # analysis/final-r5000/figures
DIAG = F.REPO / "analysis/final-r5000/diagnostics"
FR = np.array(F.FRACTIONS)
K = len(F.FRACTIONS)

ROWS = []


def rec(analysis, group, fraction, statistic, est, lo, hi, n, nq, notes=""):
    ROWS.append({
        "analysis": analysis, "group": group, "fraction": fraction,
        "statistic": statistic, "n": n, "n_questions": nq,
        "estimate": None if est is None or not np.isfinite(est) else float(est),
        "ci_lower": None if lo is None or not np.isfinite(lo) else float(lo),
        "ci_upper": None if hi is None or not np.isfinite(hi) else float(hi),
        "n_boot": N_BOOT, "seed": SEED, "notes": notes,
    })


# ---------------------------------------------------------------------------
def build():
    """Evaluable cohort with an (n, 11) answer-choice entropy matrix."""
    nat = F.load_natural()
    ev = nat[nat["natural_correct"].notna()].reset_index(drop=True).copy()
    ev["natural_correct"] = ev["natural_correct"].astype(bool)
    assert len(ev) == 3550 and ev["question_id"].nunique() == 472

    cp = pd.read_parquet(F.MERGED / "checkpoint_results.parquet",
                         columns=["parent_raw_record_id", "requested_fraction",
                                  "answer_entropy_nats"])
    A = (cp.pivot_table(index="parent_raw_record_id", columns="requested_fraction",
                        values="answer_entropy_nats", aggfunc="mean")
           .reindex(index=ev["raw_record_id"], columns=F.FRACTIONS).to_numpy(float))
    return ev, A


def group_means_bootstrap(ev, A):
    """Per-fraction nan-mean for each outcome group, with a shared subject-
    stratified question-cluster bootstrap so the two curves and their difference
    come from the same resamples."""
    y = ev["natural_correct"].to_numpy(bool).astype(float)
    values = np.column_stack([A, y])

    def stat(v):
        a, yy = v[:, :K], v[:, K] > 0.5
        out = np.full(3 * K, np.nan)
        with np.errstate(invalid="ignore"):
            if yy.sum():
                out[:K] = np.nanmean(a[yy], axis=0)
            if (~yy).sum():
                out[K:2 * K] = np.nanmean(a[~yy], axis=0)
        out[2 * K:] = out[K:2 * K] - out[:K]        # incorrect minus correct
        return out

    return F.question_cluster_bootstrap(
        values, ev["question_id"].to_numpy(), ev["subject"].to_numpy(),
        stat, n_boot=N_BOOT, seed=SEED)


def change_bootstrap(ev, A, i0, i1, label):
    """Bootstrap CI for the change in mean entropy between two fractions,
    per group, using the same clustered resamples."""
    y = ev["natural_correct"].to_numpy(bool).astype(float)
    values = np.column_stack([A[:, i0], A[:, i1], y])

    def stat(v):
        yy = v[:, 2] > 0.5
        out = np.full(2, np.nan)
        with np.errstate(invalid="ignore"):
            out[0] = np.nanmean(v[yy, 1]) - np.nanmean(v[yy, 0])
            out[1] = np.nanmean(v[~yy, 1]) - np.nanmean(v[~yy, 0])
        return out

    pt, lo, hi = F.question_cluster_bootstrap(
        values, ev["question_id"].to_numpy(), ev["subject"].to_numpy(),
        stat, n_boot=N_BOOT, seed=SEED)
    for j, g in enumerate(("correct", "incorrect")):
        rec("entropy_change", g, f"{FR[i0]:.1f}->{FR[i1]:.1f}", label,
            pt[j], lo[j], hi[j],
            int(np.isfinite(A[:, [i0, i1]]).all(axis=1).sum()),
            ev["question_id"].nunique())
    return pt, lo, hi


# ---------------------------------------------------------------------------
def mixed_outcome(ev, A):
    """Within-question comparison on the 84 mixed-outcome questions.

    For each qualifying question and fraction, average over that question's
    correct-answer runs and over its incorrect-answer runs, then average the
    per-question differences with equal weight per question. Holding the
    question fixed removes between-question difficulty differences.
    """
    y = ev["natural_correct"].to_numpy(bool)
    qid = ev["question_id"].to_numpy()
    g = pd.DataFrame({"q": qid, "y": y}).groupby("q")["y"].agg(["sum", "count"])
    mixed = set(g[(g["sum"] > 0) & (g["sum"] < g["count"])].index)
    m = np.array([q in mixed for q in qid])
    sub = ev[m].reset_index(drop=True)
    Asub = A[m]
    nq = sub["question_id"].nunique()

    # per-question group means -> (n_questions, 11) for each group
    qs = sub["question_id"].to_numpy()
    ys = sub["natural_correct"].to_numpy(bool)
    order = sorted(set(qs))
    qc = np.full((len(order), K), np.nan)
    qi = np.full((len(order), K), np.nan)
    import warnings
    for r, q in enumerate(order):
        sel = qs == q
        with warnings.catch_warnings():      # all-NaN group at some fractions
            warnings.simplefilter("ignore", RuntimeWarning)
            qc[r] = np.nanmean(Asub[sel & ys], axis=0)
            qi[r] = np.nanmean(Asub[sel & ~ys], axis=0)

    qsubj = np.array([sub.loc[sub["question_id"] == q, "subject"].iloc[0] for q in order])
    values = np.column_stack([qc, qi])

    def stat(v):
        c, i = v[:, :K], v[:, K:]
        out = np.full(3 * K, np.nan)
        with np.errstate(invalid="ignore"):
            out[:K] = np.nanmean(c, axis=0)
            out[K:2 * K] = np.nanmean(i, axis=0)
            out[2 * K:] = np.nanmean(i - c, axis=0)   # paired within question
        return out

    pt, lo, hi = F.question_cluster_bootstrap(
        values, np.array(order), qsubj, stat, n_boot=N_BOOT, seed=SEED)
    n_pairs = np.isfinite(qc - qi).sum(axis=0)
    return pt, lo, hi, nq, len(sub), n_pairs, (qc, qi)


def endpoint_concentration(ev):
    """How concentrated is the A-D distribution at the endpoint, per group?

    Also checks the endpoint degeneracy: at fraction 1.0 the forced-close probe
    reproduces the natural final answer by construction, so these numbers are an
    endpoint confidence readout, not an independent prediction.
    """
    na = pd.read_parquet(F.MERGED / "natural_results.parquet",
                         columns=["raw_record_id", "natural_answer"])
    cp = pd.read_parquet(F.MERGED / "checkpoint_results.parquet",
                         columns=["parent_raw_record_id", "requested_fraction",
                                  "forced_answer", "maximum_ad_probability",
                                  "answer_entropy_nats"])
    e = cp[cp["requested_fraction"] == 1.0]
    m = (ev.merge(na, on="raw_record_id")
           .merge(e, left_on="raw_record_id", right_on="parent_raw_record_id", how="left"))
    ok = m.dropna(subset=["forced_answer"])
    match = float((ok["forced_answer"] == ok["natural_answer"]).mean())
    out = {"endpoint_matches_natural_answer_fraction": match,
           "endpoint_match_n": int(len(ok))}
    print(f"  endpoint forced answer equals natural answer: "
          f"{100*match:.1f}% of {len(ok)} (degeneracy by construction)")
    for lab, sub in (("correct", m[m["natural_correct"]]),
                     ("incorrect", m[~m["natural_correct"]])):
        d = sub.dropna(subset=["maximum_ad_probability"])
        de = sub.dropna(subset=["answer_entropy_nats"])
        g = {"n": int(len(d)),
             "median_max_ad_probability": float(d["maximum_ad_probability"].median()),
             "pct_max_ad_prob_ge_0.90": float(100 * (d["maximum_ad_probability"] >= 0.9).mean()),
             "pct_max_ad_prob_ge_0.99": float(100 * (d["maximum_ad_probability"] >= 0.99).mean()),
             "pct_answer_entropy_below_0.1_nats": float(100 * (de["answer_entropy_nats"] < 0.1).mean())}
        out[lab] = g
        rec("endpoint_concentration", lab, "1.0", "pct_max_ad_prob_ge_0.90",
            g["pct_max_ad_prob_ge_0.90"], None, None, g["n"],
            ev["question_id"].nunique(), notes="descriptive, no CI")
        rec("endpoint_concentration", lab, "1.0", "pct_answer_entropy_below_0.1_nats",
            g["pct_answer_entropy_below_0.1_nats"], None, None, g["n"],
            ev["question_id"].nunique(), notes="descriptive, no CI")
        print(f"  {lab+'-answer':18s} n={g['n']:4d}  median max A-D prob "
              f"{g['median_max_ad_probability']:.4f}  "
              f">=0.90 {g['pct_max_ad_prob_ge_0.90']:.1f}%  "
              f"entropy<0.1 nats {g['pct_answer_entropy_below_0.1_nats']:.1f}%")
    return out


# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev, A = build()
    y = ev["natural_correct"].to_numpy(bool)
    NQ = ev["question_id"].nunique()

    n_c = np.isfinite(A[y]).sum(axis=0)
    n_i = np.isfinite(A[~y]).sum(axis=0)

    print("=" * 84)
    print("COHORT: evaluable natural trajectories (natural_correct.notna())")
    print(f"  n = {len(ev)}   questions = {NQ}   "
          f"correct-answer runs = {int(y.sum())}   incorrect-answer runs = {int((~y).sum())}")
    print("  predictor: answer_entropy_nats (forced-close A-D probe), nats, max = ln 4 = "
          f"{np.log(4):.4f}")

    pt, lo, hi = group_means_bootstrap(ev, A)
    mc, mi, md = pt[:K], pt[K:2 * K], pt[2 * K:]
    lc, li, ld = lo[:K], lo[K:2 * K], lo[2 * K:]
    hc, hi_, hd = hi[:K], hi[K:2 * K], hi[2 * K:]

    print("\n" + "=" * 84)
    print("MEAN ANSWER-CHOICE ENTROPY BY NORMALIZED REASONING POSITION")
    print(f"  {'f':>4s} | {'correct mean':>13s} {'95% CI':>18s} {'n':>5s} "
          f"| {'incorrect mean':>14s} {'95% CI':>18s} {'n':>5s} | {'diff (I-C)':>11s}")
    for j in range(K):
        print(f"  {FR[j]:4.1f} | {mc[j]:13.4f} [{lc[j]:.4f},{hc[j]:.4f}] {n_c[j]:5d} "
              f"| {mi[j]:14.4f} [{li[j]:.4f},{hi_[j]:.4f}] {n_i[j]:5d} "
              f"| {md[j]:+.4f} [{ld[j]:+.4f},{hd[j]:+.4f}]")
        rec("mean_answer_entropy", "correct", f"{FR[j]:.1f}", "mean",
            mc[j], lc[j], hc[j], int(n_c[j]), NQ)
        rec("mean_answer_entropy", "incorrect", f"{FR[j]:.1f}", "mean",
            mi[j], li[j], hi_[j], int(n_i[j]), NQ)
        rec("mean_answer_entropy_difference", "incorrect_minus_correct",
            f"{FR[j]:.1f}", "difference", md[j], ld[j], hd[j],
            int(n_c[j] + n_i[j]), NQ)

    # ---- beginning, endpoint, total change, early vs late -----------------
    print("\n" + "=" * 84)
    print("BEGINNING, ENDPOINT, AND SHAPE OF THE DECLINE")
    i_peak_c = int(np.nanargmax(mc)); i_peak_i = int(np.nanargmax(mi))
    summary = {}
    for gname, m, ip in (("correct", mc, i_peak_c), ("incorrect", mi, i_peak_i)):
        beg, end, peak = m[0], m[-1], m[ip]
        half = m[5]
        total = end - beg
        share_first = (half - beg) / total if total != 0 else np.nan
        from_peak = end - peak
        share_first_peak = (half - peak) / from_peak if from_peak != 0 else np.nan
        summary[gname] = {
            "f0.0": float(beg), "peak_fraction": float(FR[ip]), "peak": float(peak),
            "f0.5": float(half), "f1.0": float(end),
            "total_change_0_to_1": float(total),
            "change_from_peak_to_1": float(from_peak),
            "share_of_0to1_change_by_f0.5": float(share_first),
            "share_of_peak_to_1_change_by_f0.5": float(share_first_peak),
            "endpoint_as_fraction_of_ln4": float(end / np.log(4)),
        }
        print(f"  {gname}-answer trajectories:")
        print(f"     f=0.0 {beg:.4f} | peak {peak:.4f} at f={FR[ip]:.1f} | "
              f"f=0.5 {half:.4f} | f=1.0 {end:.4f}")
        print(f"     total change 0.0->1.0 = {total:+.4f} nats  "
              f"({100*end/np.log(4):.1f}% of ln4 remaining at endpoint)")
        print(f"     decline from peak to 1.0 = {from_peak:+.4f} nats; "
              f"share completed by f=0.5 = {100*share_first_peak:.1f}%")

    for i0, i1, lab in ((0, 10, "change_f0.0_to_f1.0"),
                        (2, 10, "change_f0.2_to_f1.0"),
                        (0, 5, "change_f0.0_to_f0.5"),
                        (5, 10, "change_f0.5_to_f1.0")):
        p, l, h = change_bootstrap(ev, A, i0, i1, lab)
        print(f"  {lab:22s} correct {p[0]:+.4f} [{l[0]:+.4f},{h[0]:+.4f}]   "
              f"incorrect {p[1]:+.4f} [{l[1]:+.4f},{h[1]:+.4f}]")

    print("\n" + "=" * 84)
    print("ENDPOINT CONCENTRATION OF THE A-D DISTRIBUTION")
    endpoint = endpoint_concentration(ev)

    # ---- secondary: mixed-outcome questions --------------------------------
    print("\n" + "=" * 84)
    ppt, plo, phi, nq_mix, n_mix, n_pairs, _ = mixed_outcome(ev, A)
    print(f"WITHIN-QUESTION (MIXED-OUTCOME) COMPARISON: "
          f"{nq_mix} questions, {n_mix} runs")
    print(f"  {'f':>4s} | {'correct':>9s} {'incorrect':>10s} | "
          f"{'paired diff (I-C)':>19s} {'95% CI':>20s} {'pairs':>6s}")
    for j in range(K):
        print(f"  {FR[j]:4.1f} | {ppt[j]:9.4f} {ppt[K+j]:10.4f} | "
              f"{ppt[2*K+j]:19.4f} [{plo[2*K+j]:+.4f},{phi[2*K+j]:+.4f}] {n_pairs[j]:6d}")
        rec("mixed_outcome_mean", "correct", f"{FR[j]:.1f}", "mean",
            ppt[j], plo[j], phi[j], int(n_pairs[j]), nq_mix,
            notes="equal weight per question")
        rec("mixed_outcome_mean", "incorrect", f"{FR[j]:.1f}", "mean",
            ppt[K+j], plo[K+j], phi[K+j], int(n_pairs[j]), nq_mix,
            notes="equal weight per question")
        rec("mixed_outcome_paired_difference", "incorrect_minus_correct",
            f"{FR[j]:.1f}", "paired_difference", ppt[2*K+j], plo[2*K+j], phi[2*K+j],
            int(n_pairs[j]), nq_mix, notes="within-question paired, equal weight per question")

    make_figure(FR, (mc, lc, hc), (mi, li, hi_), n_c, n_i,
                (ppt[:K], ppt[K:2*K]), nq_mix)

    DIAG.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ROWS).to_csv(DIAG / "answer_entropy_by_outcome.csv", index=False)
    (OUT / "fig5_values.json").write_text(json.dumps({
        "fraction": FR.tolist(),
        "correct": {"mean": mc.tolist(), "lo": lc.tolist(), "hi": hc.tolist(),
                    "n": n_c.tolist()},
        "incorrect": {"mean": mi.tolist(), "lo": li.tolist(), "hi": hi_.tolist(),
                      "n": n_i.tolist()},
        "difference_incorrect_minus_correct": {"point": md.tolist(), "lo": ld.tolist(),
                                               "hi": hd.tolist()},
        "mixed_outcome": {"n_questions": int(nq_mix), "n_runs": int(n_mix),
                          "correct_mean": ppt[:K].tolist(),
                          "incorrect_mean": ppt[K:2*K].tolist(),
                          "paired_difference": ppt[2*K:].tolist(),
                          "paired_lo": plo[2*K:].tolist(),
                          "paired_hi": phi[2*K:].tolist()},
        "endpoint_concentration": endpoint,
    }, indent=2))
    (DIAG / "answer_entropy_by_outcome_summary.json").write_text(json.dumps({
        "question": "does the A-D distribution concentrate as reasoning progresses "
                    "even on trajectories yielding an incorrect natural final answer",
        "provenance": {
            "checkpoint_table": str(F.MERGED / "checkpoint_results.parquet"),
            "trajectory_table": str(F.MERGED / "natural_results.parquet"),
            "loader": "analysis/figlib.py::load_natural + direct read_parquet",
            "cohort_filter": "natural_correct.notna()",
            "entropy_column": "answer_entropy_nats",
            "position_column": "requested_fraction",
            "grouping_column": "natural_correct",
            "bootstrap": "figlib.question_cluster_bootstrap, subject-stratified, "
                         "resamples whole questions",
            "n_boot": N_BOOT, "seed": SEED,
            "max_possible_entropy_nats": float(np.log(4)),
        },
        "cohort": {"n": int(len(ev)), "questions": int(NQ),
                   "correct_answer_runs": int(y.sum()),
                   "incorrect_answer_runs": int((~y).sum()),
                   "mixed_outcome_questions": int(nq_mix),
                   "mixed_outcome_runs": int(n_mix)},
        "coverage_by_fraction": {"fractions": FR.tolist(),
                                 "n_correct": n_c.tolist(), "n_incorrect": n_i.tolist()},
        "shape_summary": summary,
        "endpoint_concentration": endpoint,
        "caveats": [
            "At fraction 1.0 the forced-close probe reproduces the natural final "
            "answer by construction, so the endpoint is an endpoint confidence "
            "readout rather than an independent prediction.",
            "Answer-entropy coverage varies by fraction (3148 to 3502 of 3550); "
            "missingness is not random.",
            "Falling answer-choice entropy describes the A-D distribution under the "
            "probe, not the correctness of the reasoning.",
        ],
        "estimates": ROWS,
    }, indent=2))
    print("\nwrote", OUT / "fig5_values.json")
    print("wrote", DIAG / "answer_entropy_by_outcome.csv")
    print("wrote", DIAG / "answer_entropy_by_outcome_summary.json")


# ---------------------------------------------------------------------------
def make_figure(fx, corr, incorr, n_c, n_i, mixed_means, nq_mix):
    """Single-panel figure matching the house style of Figures 1-4.

    Per-position coverage (``n_c``/``n_i``) is reported in the caption and in
    fig5_values.json rather than in a second panel.
    """
    F.set_style()
    mc, lc, hc = corr
    mi, li, hi_ = incorr
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)

    for m, l, h, colour, lab in (
            (mc, lc, hc, F.COL_CORRECT, "Correct natural final answer"),
            (mi, li, hi_, F.COL_INCORRECT, "Incorrect natural final answer")):
        ax.plot(fx, m, "-o", color=colour, ms=4, label=lab)
        ax.fill_between(fx, l, h, color=colour, alpha=0.16, lw=0)

    # Within-question means on the mixed-outcome questions, shown for comparison;
    # the pooled curves remain the primary series.
    ax.plot(fx, mixed_means[0], "--", color=F.COL_CORRECT, lw=1.2, alpha=0.8,
            label=f"Correct, within {nq_mix} mixed-outcome questions")
    ax.plot(fx, mixed_means[1], "--", color=F.COL_INCORRECT, lw=1.2, alpha=0.8,
            label=f"Incorrect, within {nq_mix} mixed-outcome questions")

    ax.axhline(np.log(4), color="0.55", ls=":", lw=1)
    ax.set_ylabel("Answer-choice entropy (nats)")
    ax.set_xlabel("Normalized reasoning progress")
    ax.set_ylim(0, 1.45)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks(fx)
    # Legend sits in the empty upper-right band, where both curves have already
    # descended, so it never covers a line or its interval.
    # Anchored just below the ln 4 reference line so the dotted rule does not run
    # through the legend text.
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93), frameon=False,
              fontsize=8.5, labelspacing=0.35, handletextpad=0.6)
    ax.set_title("Answer-Choice Entropy by Final-Answer Correctness", fontsize=11.5)

    out = OUT / "fig5_answer_entropy_by_outcome.png"
    fig.savefig(out); fig.savefig(out.with_suffix(".pdf"))
    # Keep the LaTeX build in sync (report/main.tex has \graphicspath{{figures/}}).
    rep = F.REPO / "report/figures/fig5_answer_entropy_by_outcome.png"
    rep.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(rep); fig.savefig(rep.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", out, "and", rep)


if __name__ == "__main__":
    main()

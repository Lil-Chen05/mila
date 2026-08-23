"""Diagnostic: reasoning-trajectory length vs. mean reasoning-token entropy.

Read-only sensitivity analysis on the SAME evaluable natural cohort used by the
paper's Results (natural_correct.notna(); n=3550 over 472 questions). Answers:
how strongly is reasoning length associated with mean reasoning entropy, and how
much of the entropy -> natural-correctness discrimination is length-related?

Reuses analysis/figlib.py for data loading, AUROC, orientation, and the
subject-stratified question-cluster bootstrap. Pure pandas/numpy/matplotlib;
login-node safe (no torch, no model, no dataset). Writes only into
analysis/final-r5000/diagnostics/. Nothing existing is modified.

Residualization here is a SENSITIVITY analysis. It does not establish a causal
effect of entropy independent of length.
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

N_BOOT = 5000                      # matches primary_auroc.metadata.json
SEED = F.BOOT_SEED                 # 42
OUT = F.REPO / "analysis/final-r5000/diagnostics"

EXPECTED_N = 3550
EXPECTED_Q = 472
EXPECTED_MEAN_ENTROPY_AUROC = 0.7025143951373459


# ---------------------------------------------------------------------------
# cohort
# ---------------------------------------------------------------------------
def load_cohort():
    nat = F.load_natural()
    ev = nat[nat["natural_correct"].notna()].reset_index(drop=True).copy()
    ev["natural_correct"] = ev["natural_correct"].astype(bool)
    ev["length"] = ev["reasoning_token_count"].astype(float)
    ev["log_length"] = np.log(ev["length"])
    ev["mean_entropy"] = ev["mean_reasoning_entropy_nats"].astype(float)

    checks = {
        "n_trajectories": int(len(ev)),
        "n_questions": int(ev["question_id"].nunique()),
        "n_correct": int(ev["natural_correct"].sum()),
        "n_incorrect": int((~ev["natural_correct"]).sum()),
        "length_missing": int(ev["length"].isna().sum()),
        "entropy_missing": int(ev["mean_entropy"].isna().sum()),
        "length_non_positive": int((ev["length"] <= 0).sum()),
        "entropy_non_finite": int((~np.isfinite(ev["mean_entropy"])).sum()),
    }
    assert checks["n_trajectories"] == EXPECTED_N, checks
    assert checks["n_questions"] == EXPECTED_Q, checks
    assert checks["length_missing"] == 0 and checks["entropy_missing"] == 0, checks
    assert checks["length_non_positive"] == 0, checks

    y = ev["natural_correct"].to_numpy(bool)
    got = F.fast_auroc(-ev["mean_entropy"].to_numpy(float), y)
    assert abs(got - EXPECTED_MEAN_ENTROPY_AUROC) < 1e-12, (
        f"mean-entropy AUROC {got!r} does not reproduce the published "
        f"{EXPECTED_MEAN_ENTROPY_AUROC!r}; cohort or orientation has drifted")
    checks["reproduced_mean_entropy_auroc"] = float(got)
    return ev, checks


# ---------------------------------------------------------------------------
# statistics (pure numpy; the project has no scipy/statsmodels by design)
# ---------------------------------------------------------------------------
def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc = x - x.mean(); yc = y - y.mean()
    d = np.sqrt((xc * xc).sum() * (yc * yc).sum())
    return float(xc @ yc / d) if d > 0 else np.nan


def spearman(x, y):
    """Tie-corrected Spearman via the project's own average-rank routine."""
    return pearson(F._avg_ranks(np.asarray(x, float)),
                   F._avg_ranks(np.asarray(y, float)))


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def logistic_irls(X, y, ridge=1e-6, max_iter=60, tol=1e-9):
    """Newton/IRLS logistic fit. Returns coefficients, or NaNs if it fails."""
    y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    R = ridge * np.eye(X.shape[1]); R[0, 0] = 0.0
    for _ in range(max_iter):
        eta = np.clip(X @ b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * w[:, None]) + R
        g = X.T @ (y - p) - R @ b
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return np.full(X.shape[1], np.nan)
        b = b + step
        if np.max(np.abs(step)) < tol:
            return b
    return b


def design(ev, subject_dummies, with_correct=True, interaction=False):
    """[1, log_length, (natural_correct), subject dummies, (interaction)].

    Subjects are treatment-coded against Mathematics (first in
    figlib.SUBJECT_ORDER), matching the project's canonical subject order.
    """
    cols = [np.ones(len(ev)), ev["log_length"].to_numpy(float)]
    names = ["intercept", "log_length"]
    c = ev["natural_correct"].to_numpy(float)
    if with_correct:
        cols.append(c); names.append("natural_correct")
    for lab in F.SUBJECT_ORDER[1:]:
        cols.append((ev["subject_label"] == lab).to_numpy(float))
        names.append(f"subject[{lab}]")
    if interaction:
        cols.append(ev["log_length"].to_numpy(float) * c)
        names.append("log_length_x_natural_correct")
    return np.column_stack(cols), names


# ---------------------------------------------------------------------------
# bootstrap wrappers (subject-stratified question-cluster, via figlib)
# ---------------------------------------------------------------------------
def boot_ci(ev, values, stat_fn, n_boot=N_BOOT):
    """figlib.question_cluster_bootstrap on an arbitrary vector statistic."""
    pt, lo, hi = F.question_cluster_bootstrap(
        values, ev["question_id"].to_numpy(), ev["subject"].to_numpy(),
        stat_fn, n_boot=n_boot, seed=SEED)
    return np.atleast_1d(pt), np.atleast_1d(lo), np.atleast_1d(hi)


def corr_ci(ev, xcol, ycol, kind="pearson", n_boot=N_BOOT):
    fn = pearson if kind == "pearson" else spearman
    v = np.column_stack([ev[xcol].to_numpy(float), ev[ycol].to_numpy(float)])
    pt, lo, hi = boot_ci(ev, v, lambda a: np.array([fn(a[:, 0], a[:, 1])]), n_boot)
    return float(pt[0]), float(lo[0]), float(hi[0])


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------
ROWS = []


def rec(analysis, subset, statistic, estimate, lo, hi, n, nq, **extra):
    ROWS.append({
        "analysis": analysis, "subset": subset, "statistic": statistic,
        "n": int(n), "n_questions": int(nq),
        "estimate": None if estimate is None or not np.isfinite(estimate) else float(estimate),
        "ci_lower": None if lo is None or not np.isfinite(lo) else float(lo),
        "ci_upper": None if hi is None or not np.isfinite(hi) else float(hi),
        "n_boot": N_BOOT, "seed": SEED,
        "notes": extra.get("notes", ""),
    })


def nq(df):
    return df["question_id"].nunique()


def fmt(p, lo, hi):
    return f"{p:+.4f} [{lo:+.4f}, {hi:+.4f}]"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev, checks = load_cohort()
    y = ev["natural_correct"].to_numpy(bool)
    n, NQ = len(ev), nq(ev)
    print("=" * 78)
    print("COHORT (identical to the paper's evaluable natural cohort)")
    for k, v in checks.items():
        print(f"  {k:34s} {v}")

    # ---- 2. length descriptives -----------------------------------------
    def desc(s):
        s = pd.Series(s)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        return {"n": int(s.size), "mean": float(s.mean()), "median": float(s.median()),
                "sd": float(s.std(ddof=1)), "q1": float(q1), "q3": float(q3),
                "iqr": float(q3 - q1), "min": float(s.min()), "max": float(s.max()),
                "skew": float(s.skew())}

    length_desc = {
        "all": desc(ev["length"]),
        "correct": desc(ev.loc[y, "length"]),
        "incorrect": desc(ev.loc[~y, "length"]),
        "log_all": desc(ev["log_length"]),
    }
    print("\nREASONING-TOKEN COUNT")
    print(f"  {'subset':10s} {'n':>5s} {'mean':>9s} {'median':>8s} {'sd':>9s} "
          f"{'IQR':>16s} {'min':>6s} {'max':>7s}")
    for k in ("all", "correct", "incorrect"):
        d = length_desc[k]
        print(f"  {k:10s} {d['n']:5d} {d['mean']:9.1f} {d['median']:8.1f} {d['sd']:9.1f} "
              f"{d['q1']:7.0f}-{d['q3']:<8.0f} {d['min']:6.0f} {d['max']:7.0f}")
    print(f"  skew(raw)={length_desc['all']['skew']:.2f}  "
          f"skew(log)={length_desc['log_all']['skew']:.2f}  -> log transform used below")

    # median length difference by outcome, question-cluster CI
    v = np.column_stack([ev["length"].to_numpy(float), y.astype(float)])
    def med_diff(a):
        m = a[:, 1] > 0.5
        if m.sum() < 2 or (~m).sum() < 2:
            return np.array([np.nan])
        return np.array([np.median(a[~m, 0]) - np.median(a[m, 0])])
    p, lo, hi = boot_ci(ev, v, med_diff)
    rec("length_descriptive", "incorrect_minus_correct", "median_length_difference_tokens",
        p[0], lo[0], hi[0], n, NQ)
    print(f"  median length, incorrect minus correct: {p[0]:+.1f} "
          f"[{lo[0]:+.1f}, {hi[0]:+.1f}] tokens")

    # ---- 3. primary correlations ----------------------------------------
    print("\n" + "=" * 78)
    print(f"PRIMARY CORRELATIONS: reasoning length vs mean reasoning entropy "
          f"(n={n}, questions={NQ})")
    specs = [
        ("pearson_raw_length_vs_mean_entropy", "length", "pearson"),
        ("pearson_log_length_vs_mean_entropy", "log_length", "pearson"),
        ("spearman_length_vs_mean_entropy", "length", "spearman"),
    ]
    primary = {}
    for name, xcol, kind in specs:
        r, lo, hi = corr_ci(ev, xcol, "mean_entropy", kind)
        primary[name] = (r, lo, hi)
        rec("primary_correlation", "all_evaluable", name, r, lo, hi, n, NQ)
        print(f"  {name:42s} r={fmt(r, lo, hi)}")

    # secondary: within-subject (subject-demeaned) log-length vs entropy. The
    # pooled correlation mixes between-subject and within-subject structure;
    # demeaning isolates the within-subject part.
    dm = ev.copy()
    for c in ("log_length", "mean_entropy"):
        dm[c] = dm[c] - dm.groupby("subject_label")[c].transform("mean")
    r, lo, hi = corr_ci(dm, "log_length", "mean_entropy")
    rec("secondary_correlation", "all_evaluable",
        "pearson_within_subject_log_length_vs_mean_entropy", r, lo, hi, n, NQ,
        notes="both variables centred within subject")
    print(f"  {'pearson_within_subject_log_len_vs_ent':42s} r={fmt(r, lo, hi)}  (secondary)")

    # secondary: tail entropy, same cohort
    ev["tail_entropy"] = ev["tail_reasoning_entropy_nats"].astype(float)
    r, lo, hi = corr_ci(ev, "log_length", "tail_entropy")
    rec("secondary_correlation", "all_evaluable",
        "pearson_log_length_vs_tail_entropy", r, lo, hi, n, NQ)
    print(f"  {'pearson_log_length_vs_tail_entropy':42s} r={fmt(r, lo, hi)}  (secondary)")

    # ---- 4. by correctness ----------------------------------------------
    print("\n" + "=" * 78)
    print("LOG-LENGTH vs MEAN ENTROPY, SPLIT BY NATURAL CORRECTNESS")
    by_outcome = {}
    for lab, sub in (("correct", ev[y]), ("incorrect", ev[~y])):
        s = sub.reset_index(drop=True)
        r, lo, hi = corr_ci(s, "log_length", "mean_entropy")
        by_outcome[lab] = (r, lo, hi, len(s), nq(s))
        rec("correlation_by_outcome", lab, "pearson_log_length_vs_mean_entropy",
            r, lo, hi, len(s), nq(s))
        print(f"  {lab:10s} n={len(s):5d} questions={nq(s):4d}  r={fmt(r, lo, hi)}")

    # ---- 5. by subject ---------------------------------------------------
    print("\n" + "=" * 78)
    print("LOG-LENGTH vs MEAN ENTROPY, BY SUBJECT")
    print(f"  {'subject':14s} {'n':>5s} {'quest':>6s} {'incorr':>7s} {'r':>8s}  95% CI")
    by_subject = {}
    for lab in F.SUBJECT_ORDER:
        s = ev[ev["subject_label"] == lab].reset_index(drop=True)
        r, lo, hi = corr_ci(s, "log_length", "mean_entropy")
        n_inc = int((~s["natural_correct"]).sum())
        by_subject[lab] = (r, lo, hi, len(s), nq(s), n_inc)
        rec("correlation_by_subject", lab, "pearson_log_length_vs_mean_entropy",
            r, lo, hi, len(s), nq(s), notes=f"n_incorrect={n_inc}")
        print(f"  {lab:14s} {len(s):5d} {nq(s):6d} {n_inc:7d} {r:8.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]")

    # ---- 6. adjusted regression -----------------------------------------
    print("\n" + "=" * 78)
    print("ADJUSTED OLS: mean_entropy ~ log(length) + natural_correct + C(subject)")
    X, names = design(ev, True)
    Y = ev["mean_entropy"].to_numpy(float)
    vals = np.column_stack([X, Y])
    pt, lo, hi = boot_ci(ev, vals, lambda a: ols(a[:, :-1], a[:, -1]))
    adj = {}
    for j, nm in enumerate(names):
        adj[nm] = (float(pt[j]), float(lo[j]), float(hi[j]))
        rec("adjusted_ols", "all_evaluable", f"coef:{nm}", pt[j], lo[j], hi[j], n, NQ,
            notes="mean_entropy ~ log_length + natural_correct + C(subject); "
                  "subject reference = Mathematics")
        print(f"  {nm:34s} {fmt(pt[j], lo[j], hi[j])}")
    resid_full = Y - X @ pt
    r2 = 1 - (resid_full ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    print(f"  in-sample R^2 = {r2:.4f}   n = {n}   questions = {NQ}")
    rec("adjusted_ols", "all_evaluable", "r_squared", r2, None, None, n, NQ)

    # unadjusted slope, for comparison
    Xu = np.column_stack([np.ones(n), ev["log_length"].to_numpy(float)])
    ptu, lou, hiu = boot_ci(ev, np.column_stack([Xu, Y]),
                            lambda a: ols(a[:, :-1], a[:, -1]))
    rec("unadjusted_ols", "all_evaluable", "coef:log_length", ptu[1], lou[1], hiu[1], n, NQ,
        notes="mean_entropy ~ log_length only")
    print(f"  unadjusted log_length slope       {fmt(ptu[1], lou[1], hiu[1])}")

    # secondary interaction model
    Xi, names_i = design(ev, True, interaction=True)
    pti, loi, hii = boot_ci(ev, np.column_stack([Xi, Y]),
                            lambda a: ols(a[:, :-1], a[:, -1]))
    j = names_i.index("log_length_x_natural_correct")
    rec("interaction_ols", "all_evaluable", "coef:log_length_x_natural_correct",
        pti[j], loi[j], hii[j], n, NQ,
        notes="secondary; mean_entropy ~ log_length * natural_correct + C(subject)")
    print(f"  [secondary] interaction           {fmt(pti[j], loi[j], hii[j])}")
    interaction = (float(pti[j]), float(loi[j]), float(hii[j]))

    # ---- 7. reverse confounding check -----------------------------------
    print("\n" + "=" * 78)
    print("REVERSE CONFOUNDING CHECK (sensitivity analysis, not causal)")
    print("  orientation: higher score -> predicts natural_correct, as in the Results")
    ll = ev["log_length"].to_numpy(float)
    me = ev["mean_entropy"].to_numpy(float)
    Xr = np.column_stack([np.ones(n), ll])
    br = ols(Xr, me)                       # entropy ~ log length, NO correctness
    resid = me - Xr @ br
    ev["entropy_resid"] = resid
    print(f"  residualizing fit: mean_entropy = {br[0]:.4f} + {br[1]:.4f} * log_length")

    P = np.column_stack([-me, -ll, -resid])
    labels = ["mean_reasoning_entropy", "log_reasoning_length",
              "length_residualized_mean_entropy"]
    pt, lo, hi, nn = F.auroc_curve_bootstrap(
        P, y, ev["question_id"].to_numpy(), ev["subject"].to_numpy(),
        n_boot=N_BOOT, seed=SEED)
    aurocs = {}
    for j, lab in enumerate(labels):
        aurocs[lab] = (float(pt[j]), float(lo[j]), float(hi[j]), int(nn[j]))
        rec("auroc", lab, "auroc_natural_correct", pt[j], lo[j], hi[j], nn[j], NQ,
            notes="predictor negated so higher predicts correct")
        print(f"  AUROC {lab:34s} {pt[j]:.4f} [{lo[j]:.4f}, {hi[j]:.4f}] n={nn[j]}")

    # paired differences vs mean entropy, same cohort (fig4 pattern)
    def paired(rows):
        yr = y[rows]
        base = F.fast_auroc(P[rows, 0], yr)
        return np.array([base - F.fast_auroc(P[rows, j], yr) for j in range(P.shape[1])])
    dpt, dlo, dhi = boot_ci(ev, np.arange(n).reshape(-1, 1),
                            lambda a: paired(a[:, 0].astype(int)))
    for j in (1, 2):
        rec("auroc_paired_difference", f"mean_entropy_minus_{labels[j]}",
            "auroc_difference", dpt[j], dlo[j], dhi[j], n, NQ)
        print(f"  paired diff  mean entropy - {labels[j]:34s} "
              f"{fmt(dpt[j], dlo[j], dhi[j])}")
    paired_diffs = {labels[j]: (float(dpt[j]), float(dlo[j]), float(dhi[j]))
                    for j in (1, 2)}

    # variant: residualize on log length AND subject (the specification used in
    # RESULTS_STORY_AUDIT.md), reported so the two numbers can be reconciled.
    Xs, _ = design(ev, True, with_correct=False)   # [1, log_length, subject dummies]
    resid_s = me - Xs @ ols(Xs, me)
    spt, slo, shi, snn = F.auroc_curve_bootstrap(
        (-resid_s).reshape(-1, 1), y, ev["question_id"].to_numpy(),
        ev["subject"].to_numpy(), n_boot=N_BOOT, seed=SEED)
    rec("auroc", "length_and_subject_residualized_mean_entropy",
        "auroc_natural_correct", spt[0], slo[0], shi[0], int(snn[0]), NQ,
        notes="variant: residualized on log_length + C(subject), no correctness")
    print(f"  [variant] residualized on log length + subject: "
          f"{spt[0]:.4f} [{slo[0]:.4f}, {shi[0]:.4f}]")
    aurocs["length_and_subject_residualized_mean_entropy"] = (
        float(spt[0]), float(slo[0]), float(shi[0]), int(snn[0]))

    # robustness: refit the residualization inside each bootstrap replicate
    def resid_auroc(rows):
        m = me[rows]; l = ll[rows]
        Xb = np.column_stack([np.ones(rows.size), l])
        rr = m - Xb @ ols(Xb, m)
        return np.array([F.fast_auroc(-rr, y[rows])])
    rpt, rlo, rhi = boot_ci(ev, np.arange(n).reshape(-1, 1),
                            lambda a: resid_auroc(a[:, 0].astype(int)))
    rec("auroc", "length_residualized_mean_entropy_refit",
        "auroc_natural_correct", rpt[0], rlo[0], rhi[0], n, NQ,
        notes="robustness: residualization refit within each bootstrap replicate")
    print(f"  [robustness] residualization refit per replicate: "
          f"{rpt[0]:.4f} [{rlo[0]:.4f}, {rhi[0]:.4f}]")

    # ---- 8. joint logistic (in-sample descriptive) -----------------------
    print("\n" + "=" * 78)
    print("JOINT LOGISTIC (IN-SAMPLE DESCRIPTIVE MODEL, NOT A VALIDATED PREDICTOR)")
    print("  natural_correct ~ mean_entropy + log(length) + C(subject)")
    cols = [np.ones(n), me, ll]
    lnames = ["intercept", "mean_entropy", "log_length"]
    for lab in F.SUBJECT_ORDER[1:]:
        cols.append((ev["subject_label"] == lab).to_numpy(float))
        lnames.append(f"subject[{lab}]")
    XL = np.column_stack(cols)
    bl = logistic_irls(XL, y.astype(float))
    lpt, llo, lhi = boot_ci(ev, np.column_stack([XL, y.astype(float)]),
                            lambda a: logistic_irls(a[:, :-1], a[:, -1]))
    logit_out = {}
    for j, nmj in enumerate(lnames):
        logit_out[nmj] = (float(lpt[j]), float(llo[j]), float(lhi[j]))
        rec("joint_logistic_in_sample", "all_evaluable", f"coef:{nmj}",
            lpt[j], llo[j], lhi[j], n, NQ,
            notes="in-sample descriptive; ridge=1e-6; not cross-validated")
        print(f"  {nmj:34s} {fmt(lpt[j], llo[j], lhi[j])}")
    lin_auc = F.fast_auroc(XL @ bl, y)
    apt, alo, ahi = F.auroc_curve_bootstrap(
        (XL @ bl).reshape(-1, 1), y, ev["question_id"].to_numpy(),
        ev["subject"].to_numpy(), n_boot=N_BOOT, seed=SEED)[:3]
    rec("joint_logistic_in_sample", "all_evaluable", "in_sample_auroc",
        lin_auc, alo[0], ahi[0], n, NQ,
        notes="IN-SAMPLE ONLY; coefficients fit on the full cohort then scored on it")
    print(f"  in-sample AUROC (NOT validated)    {lin_auc:.4f} "
          f"[{alo[0]:.4f}, {ahi[0]:.4f}]")

    # ---- 9. figures ------------------------------------------------------
    make_figures(ev, y)

    # ---- 10. outputs -----------------------------------------------------
    tbl = pd.DataFrame(ROWS)
    tbl.to_csv(OUT / "length_entropy_estimates.csv", index=False)
    payload = {
        "title": "Reasoning length vs mean reasoning-token entropy (diagnostic)",
        "provenance": {
            "trajectory_table": str(F.MERGED / "natural_results.parquet"),
            "loader": "analysis/figlib.py::load_natural",
            "cohort_filter": "natural_correct.notna()  (as in fig2/fig4)",
            "length_column": "reasoning_token_count",
            "entropy_column": "mean_reasoning_entropy_nats",
            "tail_entropy_column": "tail_reasoning_entropy_nats",
            "correctness_column": "natural_correct",
            "subject_column": "subject",
            "question_column": "question_id",
            "trajectory_column": "raw_record_id",
            "log": "natural logarithm (np.log)",
            "bootstrap": "subject-stratified question-cluster percentile bootstrap "
                         "via figlib.question_cluster_bootstrap / auroc_curve_bootstrap",
            "n_boot": N_BOOT, "seed": SEED,
            "auroc_orientation": "predictor negated so higher predicts natural_correct",
            "residualization_caveat": "sensitivity analysis only; not causal evidence "
                                      "that entropy acts independently of length",
        },
        "cohort_checks": checks,
        "length_descriptives": length_desc,
        "estimates": ROWS,
    }
    (OUT / "length_entropy_summary.json").write_text(json.dumps(payload, indent=2))
    print("\n" + "=" * 78)
    print("wrote", OUT / "length_entropy_estimates.csv")
    print("wrote", OUT / "length_entropy_summary.json")
    return dict(checks=checks, length=length_desc, primary=primary,
                by_outcome=by_outcome, by_subject=by_subject, adjusted=adj,
                interaction=interaction, aurocs=aurocs, paired=paired_diffs,
                logistic=logit_out, in_sample_auroc=float(lin_auc))


# ---------------------------------------------------------------------------
# figures (diagnostic only; not added to the report)
# ---------------------------------------------------------------------------
def make_figures(ev, y):
    F.set_style()
    ll = ev["log_length"].to_numpy(float)
    me = ev["mean_entropy"].to_numpy(float)

    # --- Figure A: log length vs mean entropy ---------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    # The minority class (378 incorrect vs 3172 correct) needs a higher alpha to
    # be visible at all; both remain well below opaque.
    for mask, colour, size, alpha in ((y, F.COL_CORRECT, 9, 0.09),
                                      (~y, F.COL_INCORRECT, 12, 0.38)):
        ax.scatter(ll[mask], me[mask], s=size, alpha=alpha, color=colour,
                   linewidths=0, rasterized=True)
    grid = np.linspace(ll.min(), ll.max(), 200)
    for mask, colour, lab in ((y, F.COL_CORRECT, "Correct"),
                              (~y, F.COL_INCORRECT, "Incorrect")):
        b = ols(np.column_stack([np.ones(mask.sum()), ll[mask]]), me[mask])
        ax.plot(grid, b[0] + b[1] * grid, color=colour, lw=2.0, label=lab)
    b = ols(np.column_stack([np.ones(ll.size), ll]), me)
    ax.plot(grid, b[0] + b[1] * grid, color="0.25", lw=1.6, ls="--", label="All")
    ax.set_xlabel("Log reasoning-token count")
    ax.set_ylabel("Mean reasoning-token entropy (nats)")
    ax.set_title("Reasoning Length and Mean Reasoning-Token Entropy")
    ax.legend(loc="lower right", ncol=3, frameon=False)
    out = OUT / "figA_length_vs_entropy.png"
    fig.savefig(out); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", out)

    # --- Figure B: length distribution by outcome ------------------------
    # Length is strongly right-skewed (skew 3.9), so the distribution is shown on
    # a log token axis. Densities are computed over log10 tokens so that area is
    # proportional on the axis actually drawn.
    L = ev["length"].to_numpy(float)
    g = np.log10(L)
    bins = np.linspace(g.min(), g.max(), 42)
    fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    for mask, colour, lab in ((y, F.COL_CORRECT, "Correct"),
                              (~y, F.COL_INCORRECT, "Incorrect")):
        ax.hist(g[mask], bins=bins, density=True, histtype="stepfilled",
                color=colour, alpha=0.28, lw=0)
        ax.hist(g[mask], bins=bins, density=True, histtype="step",
                color=colour, lw=1.8, label=f"{lab} (n={int(mask.sum())})")
        ax.axvline(np.median(g[mask]), color=colour, ls=":", lw=1.6)
    ticks = [150, 300, 500, 1000, 2000, 4000, 8000]
    ax.set_xticks(np.log10(ticks))
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel("Reasoning-token count (log scale)")
    ax.set_ylabel("Density per log10 token")
    ax.set_title("Reasoning Length by Natural Final-Answer Correctness")
    ax.legend(loc="upper right", frameon=False)
    out = OUT / "figB_length_by_outcome.png"
    fig.savefig(out); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()

"""Figure 4 (RQ3): head-to-head discrimination + calibration.

4A  Dot-and-whisker of pooled AUROC (natural correctness) across signal
    families, 95% CI, n annotated. Ordering is DESCRIPTIVE -- we do not read
    non-distinguishability off overlapping CIs; a paired AUROC-difference
    analysis (printed + saved) backs any head-to-head statement in the text.
    Includes a simple trajectory-length baseline (inverse log reasoning-token
    count) so the reader can see that a length heuristic matches mean reasoning
    entropy. Its paired difference vs mean entropy is saved to the values JSON
    for the text, deliberately not drawn on the figure.
4B  Endpoint reliability diagram for the probability-like signals: the
    selected-answer probability (== max A-D probability here) and the repaired
    verbalized confidence. Calibrated against final-answer correctness. Entropy
    is deliberately excluded (not a probability).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figlib as F

N_BOOT = 2000
LETTERS = "ABCD"


# ---- data assembly -------------------------------------------------------
def load_run_table():
    """Per-run predictors on the evaluable cohort, plus target + cluster keys."""
    tf = pd.read_csv(F.FIXED / "trajectory_features.csv",
                     usecols=["raw_record_id", "question_id", "subject", "natural_correct",
                              "negative_mean_reasoning_entropy", "negative_tail_reasoning_entropy",
                              "negative_answer_entropy_fraction_1.0", "negative_answer_switch_count",
                              "negative_stabilization_fraction"])
    tf = tf[tf["natural_correct"].notna()].copy()
    tf["natural_correct"] = tf["natural_correct"].astype(bool)
    rc = F.load_recovered_confidence("natural")[["raw_record_id", "confidence_status_final",
                                                 "confidence_value_final"]]
    tf = tf.merge(rc, on="raw_record_id", how="left")
    tf["verbalized_confidence"] = np.where(tf["confidence_status_final"].eq("missing"), np.nan,
                                           tf["confidence_value_final"].astype("float"))
    return tf


BASELINE_N_BOOT = 5000          # matches the fixed export's replicate count
BASELINE_EXPECTED = (0.7061, 0.6594, 0.7516)


def length_baseline():
    """AUROC of inverse log reasoning-token count for natural correctness.

    Same evaluable cohort (natural_correct.notna(), n=3550) and same orientation
    as every other row: the predictor is negated so that higher plotted values
    mean stronger evidence of correctness. Row order matches
    trajectory_features.csv, so the question-cluster draws align with the rest
    of the analysis. Returns ((auc, lo, hi, n), paired_diff_vs_mean_entropy).
    """
    ev = F.load_natural()
    ev = ev[ev["natural_correct"].notna()].reset_index(drop=True)
    y = ev["natural_correct"].to_numpy(bool)
    qid = ev["question_id"].to_numpy(); subj = ev["subject"].to_numpy()
    inv_log_len = -np.log(ev["reasoning_token_count"].to_numpy(float))
    neg_entropy = -ev["mean_reasoning_entropy_nats"].to_numpy(float)

    P = np.column_stack([neg_entropy, inv_log_len])
    pt, lo, hi, n = F.auroc_curve_bootstrap(P, y, qid, subj,
                                            n_boot=BASELINE_N_BOOT, seed=F.BOOT_SEED)
    auc = (float(pt[1]), float(lo[1]), float(hi[1]), int(n[1]))
    for got, want, lab in zip(auc[:3], BASELINE_EXPECTED, ("auroc", "lo", "hi")):
        assert round(got, 4) == want, f"length baseline {lab} {got:.6f} != {want}"

    # Paired difference (mean entropy - length baseline) on this cohort; saved
    # to the values JSON for the text, not drawn.
    diff = paired_diff_vs_ref(P, y, qid, subj, ref=0,
                              n_boot=BASELINE_N_BOOT, seed=F.BOOT_SEED)[1]
    return auc, diff


def build_endpoint_calibration():
    """Apples-to-apples endpoint calibration inputs on a single common cohort.

    Both probability-like quantities are elicited by the SAME forced endpoint
    (fraction-1.0) probe and refer to the SAME selected endpoint answer:
      - selected-answer probability p(forced answer);
      - repaired endpoint CHECKPOINT verbalized confidence / 100.
    Target = NATURAL final-answer correctness (the paper's reliability definition,
    same as Figure 4A). Restricted to trajectories with an evaluable natural
    answer and both endpoint measures available. On this cohort the endpoint
    forced answer equals the natural final answer, so both quantities are valid
    confidence readouts for the answer being scored (asserted below).
    Returns (p_sel, conf, y_natural, n_common, endpoint_match_fraction).
    """
    nat = pd.read_parquet(F.MERGED / "natural_results.parquet",
                          columns=["raw_record_id", "natural_correct", "natural_answer"])
    nat = nat[nat["natural_correct"].notna()]

    cp = pd.read_parquet(F.MERGED / "checkpoint_results.parquet",
                         columns=["checkpoint_record_id", "parent_raw_record_id",
                                  "requested_fraction", "forced_answer", "ad_probabilities_float32"])
    e = cp[cp["requested_fraction"] == 1.0].copy()

    def psel(r):
        p, fa = r.ad_probabilities_float32, r.forced_answer
        if p is None or not isinstance(fa, str) or fa not in LETTERS:
            return np.nan
        return float(p[LETTERS.index(fa)])
    e["p_sel"] = e.apply(psel, axis=1)

    rc = F.load_recovered_confidence("checkpoint")
    rc = rc[rc["requested_fraction"] == 1.0][["checkpoint_record_id",
                                              "confidence_status_final", "confidence_value_final"]]
    e = e.merge(rc, on="checkpoint_record_id", how="left")
    e["conf"] = np.where(e["confidence_status_final"].eq("missing"), np.nan,
                         e["confidence_value_final"].astype("float") / 100.0)

    m = nat.merge(e[["parent_raw_record_id", "forced_answer", "p_sel", "conf"]],
                  left_on="raw_record_id", right_on="parent_raw_record_id", how="inner")
    common = m.dropna(subset=["p_sel", "conf"])
    match = float((common["forced_answer"] == common["natural_answer"]).mean())
    return (common["p_sel"].to_numpy(float), common["conf"].to_numpy(float),
            common["natural_correct"].to_numpy(bool), int(len(common)), match)


def reliability(pred, y, n_bins=10):
    pred = np.asarray(pred, float); y = np.asarray(y, bool)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, n_bins - 1)
    conf = np.full(n_bins, np.nan); acc = np.full(n_bins, np.nan); cnt = np.zeros(n_bins, int)
    for b in range(n_bins):
        m = idx == b
        cnt[b] = m.sum()
        if m.sum():
            conf[b] = pred[m].mean(); acc[b] = y[m].mean()
    ece = np.nansum(cnt / cnt.sum() * np.abs(acc - conf))
    return conf, acc, cnt, float(ece)


def auroc_ci(pred, y, qid, subj):
    pt, lo, hi, n = F.auroc_curve_bootstrap(pred.reshape(-1, 1), y,
                                            qid, subj, n_boot=N_BOOT)
    return float(pt[0]), float(lo[0]), float(hi[0]), int(n[0])


# ---- figure --------------------------------------------------------------
def main():
    F.set_style()
    F.OUT.mkdir(parents=True, exist_ok=True)
    tf = load_run_table()
    y = tf["natural_correct"].to_numpy(bool)
    qid = tf["question_id"].to_numpy(); subj = tf["subject"].to_numpy()

    pa = pd.read_csv(F.FIXED / "primary_auroc.csv")
    pool = pa[pa["grouping"] == "pooled"].set_index("feature")

    def fixed(feat):
        r = pool.loc[feat]
        return float(r["point_estimate"]), float(r["lower"]), float(r["upper"]), int(r["sample_size"])

    # Repaired confidence AUROC computed here; others from fixed export.
    conf_pred = tf["verbalized_confidence"].to_numpy(float)
    conf_auc = auroc_ci(conf_pred, y, qid, subj)

    length_auc, length_diff = length_baseline()

    # (label, (auc, lo, hi, n)) ordered from highest to lowest AUROC.
    rows = [
        ("Inverse log trajectory length (baseline)", length_auc),
        ("Mean reasoning-token entropy", fixed("negative_mean_reasoning_entropy")),
        ("Endpoint answer-choice entropy", fixed("negative_answer_entropy_fraction_1.0")),
        ("Verbalized confidence", conf_auc),
        ("Tail reasoning-token entropy", fixed("negative_tail_reasoning_entropy")),
        ("Stabilization", fixed("negative_stabilization_fraction")),
        ("Answer switching", fixed("negative_answer_switch_count")),
    ]

    # ---- Single-panel discrimination forest plot ----
    # Height scales with the row count so spacing matches the previous version.
    fig, ax = plt.subplots(figsize=(8.5, 4.1), constrained_layout=True)
    mc = "#2C3E50"
    ypos = list(range(len(rows)))[::-1]          # first listed row at the top
    for yp, (lab, (auc, lo, hi, n)) in zip(ypos, rows):
        ax.plot([lo, hi], [yp, yp], color=mc, lw=1.8, solid_capstyle="round", zorder=2)
        # The baseline keeps the same colour and size but an open face, a
        # restrained cue that it is a reference rather than a measured signal.
        face = "white" if lab.endswith("(baseline)") else mc
        ax.plot(auc, yp, "o", color=mc, mfc=face, ms=6, mec=mc if face == "white" else "white",
                mew=1.4 if face == "white" else 0.8, zorder=3)
        ax.text(0.878, yp, f"{auc:.3f}", va="center", ha="right", fontsize=9, color="0.15")
    ax.axvline(0.5, color="0.55", ls="--", lw=1, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0.45, 0.90)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("AUROC")

    out = F.OUT / "fig4_signal_comparison.png"
    fig.savefig(out); fig.savefig(out.with_suffix(".pdf"))
    # Keep the LaTeX build in sync (report/main.tex has \graphicspath{{figures/}}).
    rep = F.REPO / "report/figures/fig4_signal_comparison.png"
    rep.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(rep); fig.savefig(rep.with_suffix(".pdf"))
    plt.close(fig)

    # Endpoint calibration values retained for the record (RQ3 text / JSON), not
    # shown in the main figure.
    psel, conf_end, y_end, n_common, endpoint_match = build_endpoint_calibration()
    _, _, _, ece_p = reliability(psel, y_end)
    _, _, _, ece_conf = reliability(conf_end, y_end)

    # ---- paired AUROC-difference CIs (mean entropy vs each) on common cohort ----
    feats = {"Mean reasoning entropy": "negative_mean_reasoning_entropy",
             "Endpoint answer-choice entropy": "negative_answer_entropy_fraction_1.0",
             "Verbalized confidence": None, "Answer switching": "negative_answer_switch_count",
             "Stabilization": "negative_stabilization_fraction"}
    common = tf.copy()
    pred_cols = {}
    for lab, feat in feats.items():
        pred_cols[lab] = (common["verbalized_confidence"].to_numpy(float) if feat is None
                          else common[feat].to_numpy(float))
    mask = np.all([np.isfinite(v) for v in pred_cols.values()], axis=0)
    cc = common[mask]
    ccy = cc["natural_correct"].to_numpy(bool)
    ccq = cc["question_id"].to_numpy(); ccs = cc["subject"].to_numpy()
    P = np.column_stack([pred_cols[l][mask] for l in feats])   # already oriented (neg_* + confidence)
    paired = paired_diff_vs_ref(P, ccy, ccq, ccs, ref=0)
    diff_report = {list(feats)[j]: paired[j] for j in range(1, len(feats))}

    (F.OUT / "fig4_values.json").write_text(json.dumps({
        "auroc": {r[0]: {"auc": r[1][0], "lo": r[1][1], "hi": r[1][2], "n": r[1][3]} for r in rows},
        "calibration": {"target": "natural final-answer correctness",
                        "common_cohort_n": n_common,
                        "endpoint_matches_natural_fraction": endpoint_match,
                        "selected_answer_prob_ECE": ece_p,
                        "endpoint_checkpoint_confidence_ECE": ece_conf},
        "paired_diff_common_cohort_n": int(mask.sum()),
        "paired_diff_mean_entropy_minus": diff_report,
        "length_baseline": {
            "predictor": "inverse log reasoning_token_count",
            "cohort": "evaluable natural trajectories (natural_correct.notna())",
            "n": length_auc[3], "n_boot": BASELINE_N_BOOT, "seed": F.BOOT_SEED,
            "auroc": length_auc[0], "lo": length_auc[1], "hi": length_auc[2],
            "paired_diff_mean_entropy_minus_length": length_diff,
            "note": "paired difference reported in text only, not drawn on the figure",
        },
    }, indent=2))
    print("wrote", out, "and", rep)
    print(f"length baseline AUROC {length_auc[0]:.4f} "
          f"[{length_auc[1]:.4f},{length_auc[2]:.4f}] n={length_auc[3]}")
    print(f"paired diff mean entropy - length baseline "
          f"{length_diff['diff']:+.4f} [{length_diff['lo']:+.4f}, {length_diff['hi']:+.4f}] "
          "(text only, not plotted)")
    print(f"verbalized confidence AUROC {conf_auc[0]:.4f} [{conf_auc[1]:.4f},{conf_auc[2]:.4f}] n={conf_auc[3]}")
    print(f"calibration ECE: selected-prob {ece_p:.3f} | confidence {ece_conf:.3f}")
    print(f"common cohort n={int(mask.sum())}; paired AUROC diff (mean entropy - X):")
    for k, v in diff_report.items():
        print(f"   - {k:32s} {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]")


def paired_diff_vs_ref(P, y, qid, subj, ref=0, n_boot=N_BOOT, seed=F.BOOT_SEED):
    """Question-cluster bootstrap CI for AUROC(ref) - AUROC(col) per column."""
    P = np.asarray(P, float); y = np.asarray(y, bool)
    k = P.shape[1]

    def diffs(rows):
        yr = y[rows]
        base = F.fast_auroc(P[rows, ref], yr)
        return np.array([base - F.fast_auroc(P[rows, j], yr) for j in range(k)])
    point = diffs(np.arange(P.shape[0]))

    q_to_rows = {}
    for i, q in enumerate(qid):
        q_to_rows.setdefault(q, []).append(i)
    questions = list(q_to_rows); q_subj = np.array([subj[q_to_rows[q][0]] for q in questions])
    rows_per_q = [np.asarray(q_to_rows[q]) for q in questions]
    rng = np.random.default_rng(seed)
    subj_levels = np.unique(q_subj)
    by_subj = {s: np.where(q_subj == s)[0] for s in subj_levels}
    boot = np.empty((n_boot, k))
    for b in range(n_boot):
        picked = []
        for s in subj_levels:
            pool = by_subj[s]
            picked.extend(rows_per_q[qi] for qi in pool[rng.integers(0, pool.size, pool.size)])
        boot[b] = diffs(np.concatenate(picked))
    lo = np.nanpercentile(boot, 2.5, axis=0); hi = np.nanpercentile(boot, 97.5, axis=0)
    return {j: {"diff": float(point[j]), "lo": float(lo[j]), "hi": float(hi[j])} for j in range(k)}


if __name__ == "__main__":
    main()

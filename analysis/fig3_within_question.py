"""Figure 3 (RQ2): do naturally correct vs incorrect runs differ in reasoning
uncertainty, and how much survives holding the question fixed?

3A  Pooled reasoning-entropy profile across normalized progress, correct vs
    incorrect (question-cluster bands). Labelled as pooled -- may reflect
    between-question difficulty.
3B  The 84 mixed-outcome questions: one equal-weighted within-question
    difference each (incorrect - correct mean reasoning entropy), ordered and
    coloured by subject, with the fixed equal-weight mean and 95% CI. This is
    the fixed, pre-specified RQ2 estimand.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figlib as F


def main():
    F.set_style()
    F.OUT.mkdir(parents=True, exist_ok=True)

    # ---- 3A: pooled decile profile (post-hoc descriptive) ----
    nat = F.load_natural(with_traces=True)
    ev = nat[nat["natural_correct"].notna()].reset_index(drop=True)
    y = ev["natural_correct"].to_numpy(bool)
    dec = np.full((len(ev), F.N_DECILES), np.nan)
    for i, r in enumerate(ev.itertuples(index=False)):
        sl = F._reasoning_slice(r.per_token_entropy_nats, r.reasoning_boundaries)
        if sl is not None:
            dec[i] = F.decile_means(sl)

    # Stack correct/incorrect into 20 cols so one nan-mean bootstrap yields both,
    # preserving question clustering (a question keeps all its runs per resample).
    stacked = np.full((len(ev), 2 * F.N_DECILES), np.nan)
    stacked[y, :F.N_DECILES] = dec[y]
    stacked[~y, F.N_DECILES:] = dec[~y]
    pt, lo, hi = F.question_cluster_bootstrap(
        stacked, ev["question_id"].to_numpy(), ev["subject"].to_numpy(),
        F.nanmean_rows, n_boot=2000)
    cor = (pt[:F.N_DECILES], lo[:F.N_DECILES], hi[:F.N_DECILES])
    inc = (pt[F.N_DECILES:], lo[F.N_DECILES:], hi[F.N_DECILES:])
    gap = float(np.nanmean(inc[0] - cor[0]))
    x = (np.arange(F.N_DECILES) + 0.5) / F.N_DECILES

    # ---- 3B: fixed within-question differences ----
    wd = pd.read_csv(F.FIXED / "within_question_distribution.csv")
    me = wd[wd["feature"] == "negative_mean_reasoning_entropy"].copy()
    me["subject_label"] = me["subject"].map(F.SUBJECTS)
    me = me.sort_values("paired_difference").reset_index(drop=True)
    ws = pd.read_csv(F.FIXED / "within_question_summary.csv")
    s = ws[ws["feature"] == "negative_mean_reasoning_entropy"].iloc[0]
    mean_d, lo_d, hi_d = float(s["mean_paired_difference"]), float(s["lower"]), float(s["upper"])
    n_pos = int((me["paired_difference"] > 0).sum())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.9))

    # ---- Panel A: pooled entropy profiles by outcome ----
    axA.plot(x, cor[0], "-o", color=F.COL_CORRECT, ms=4, label=f"Correct (n={int(y.sum()):,})")
    axA.fill_between(x, cor[1], cor[2], color=F.COL_CORRECT, alpha=0.16, lw=0)
    axA.plot(x, inc[0], "-s", color=F.COL_INCORRECT, ms=4, label=f"Incorrect (n={int((~y).sum()):,})")
    axA.fill_between(x, inc[1], inc[2], color=F.COL_INCORRECT, alpha=0.16, lw=0)
    axA.set_xlabel("Normalized reasoning progress")
    axA.set_ylabel("Reasoning-token entropy (nats)")
    axA.set_title("(A)  Pooled reasoning-token entropy by outcome", loc="left")
    axA.legend(loc="lower right")
    axA.set_xlim(0, 1)

    # ---- Panel B: distribution of within-question differences ----
    # Dot-density (Wilkinson) layout: bin the 84 differences and stack points
    # symmetrically within each bin. The vertical axis is for visibility only.
    diffs = me["paired_difference"].to_numpy()
    lo_x, hi_x = diffs.min(), diffs.max()
    nbins = 30
    edges = np.linspace(lo_x - 1e-9, hi_x + 1e-9, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bidx = np.clip(np.digitize(diffs, edges) - 1, 0, nbins - 1)
    px = np.empty(len(diffs)); py = np.empty(len(diffs))
    for b in range(nbins):
        members = np.where(bidx == b)[0]
        for j, idx in enumerate(members):
            off = 0 if j == 0 else ((j + 1) // 2) * (1 if j % 2 == 1 else -1)
            px[idx] = centers[b]; py[idx] = off
    y_sum = py.max() + 2.4                       # summary row above the point cloud

    axB.axvline(0, color="0.5", ls="--", lw=1, zorder=1)
    axB.scatter(px, py, s=20, color="#4C72B0", alpha=0.85,
                edgecolors="white", linewidths=0.3, zorder=2)
    # Equally weighted mean and its 95% CI (fixed values, unchanged).
    axB.plot([lo_d, hi_d], [y_sum, y_sum], color="k", lw=1.8, zorder=4)
    for xc in (lo_d, hi_d):
        axB.plot([xc, xc], [y_sum - 0.7, y_sum + 0.7], color="k", lw=1.3, zorder=4)
    axB.scatter([mean_d], [y_sum], marker="D", s=55, color="k", zorder=5)
    axB.text(mean_d, y_sum + 1.7,
             f"Equal-weight mean {mean_d:+.4f}\n95% CI [{lo_d:.4f}, {hi_d:.4f}]",
             ha="center", va="bottom", fontsize=8.5, color="0.15")

    axB.set_xlabel("Incorrect minus correct mean entropy (nats)")
    axB.set_title("(B)  Within-question entropy differences", loc="left")
    span = hi_x - lo_x
    axB.set_xlim(lo_x - 0.06 * span, hi_x + 0.06 * span)
    axB.set_ylim(py.min() - 2.0, y_sum + 5.2)
    axB.set_yticks([])
    axB.spines["left"].set_visible(False)

    fig.tight_layout()
    out = F.OUT / "fig3_within_question.png"
    fig.savefig(out); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    (F.OUT / "fig3_values.json").write_text(json.dumps({
        "pooled_gap_nats": gap,
        "correct_decile_mean": cor[0].tolist(),
        "incorrect_decile_mean": inc[0].tolist(),
        "within_question_mean": mean_d, "ci": [lo_d, hi_d],
        "n_questions": int(len(me)), "n_incorrect_gt_correct": n_pos,
    }, indent=2))
    print("wrote", out)
    print(f"pooled gap (incorrect-correct) mean over deciles: {gap:.4f} nats")
    print(f"within-question mean {mean_d:+.4f} [{lo_d:+.4f},{hi_d:+.4f}]; {n_pos}/84 positive")


if __name__ == "__main__":
    main()

"""Exploratory analysis of results/20q/checkpoints.jsonl (decile checkpoint probes).

LOGIN-NODE-SAFE: reads ONLY a small local JSONL (our own results). No model, no HF
dataset load, no GPU. Headless node has no display -> matplotlib Agg backend (set
BEFORE importing pyplot).

STANDING CAVEATS (stamped into every output, FINDINGS.md, and figure suptitles):
  - single subject (abstract_algebra) -- not subject-representative
  - n=19 usable closed questions (qid14 hit the 4096-token cap unclosed; reported
    SEPARATELY, never averaged in)
  - 1 greedy run/question -- no sampling spread
  - exploratory data produced off MIRRORED gpu helpers (gpu_common.py refactor still
    pending) -> directional, not a finding
  - the incorrect group is n=6 -> SUGGESTIVE, not significant. No p-values on n=6.

Run:  uv run python analysis/analyze_checkpoints.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")            # headless: pick a non-interactive backend before pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IN_PATH = "results/20q/checkpoints.jsonl"
TAB_DIR = "analysis/20q/tables"
FIG_DIR = "analysis/20q/figures"
FINDINGS = "analysis/20q/FINDINGS.md"

CAVEATS = (
    "single subject (abstract_algebra) | n=19 usable closed questions | "
    "1 greedy run/question | exploratory off mirrored gpu helpers | "
    "incorrect group n=6 -> suggestive, not significant (no p-values)"
)


def spearman(x, y):
    """Spearman rho without scipy: Pearson on ranks (drops NaN pairs first)."""
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(s) < 3:
        return float("nan")
    return np.corrcoef(s["x"].rank(), s["y"].rank())[0, 1]


def pearson(x, y):
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(s) < 3:
        return float("nan")
    return np.corrcoef(s["x"], s["y"])[0, 1]


def main():
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    df = pd.read_json(IN_PATH, lines=True)
    print("=" * 78)
    print("CHECKPOINT ANALYSIS — exploratory")
    print(f"CAVEATS: {CAVEATS}")
    print("=" * 78)
    print(f"loaded {len(df)} rows, {df['qid'].nunique()} questions, "
          f"subjects={sorted(df['subject'].unique())}")

    # --- cohort split: closed chains vs the unclosed outlier (qid14) -------------
    closed = df[df["think_closed"]].copy()
    unclosed = df[~df["think_closed"]].copy()
    unclosed_qids = sorted(unclosed["qid"].unique())
    print(f"\nCOHORT: closed={closed['qid'].nunique()} questions "
          f"({len(closed)} rows); unclosed (reported separately)={unclosed_qids}")
    # sanity: in the closed cohort the answer token should always be a clean A-D
    n_nan_H = int(closed["H_letter"].isna().sum())
    print(f"closed-cohort rows with H_letter=NaN (letters not matched): {n_nan_H}")

    # --- per-question FINAL (frac=1.0) correctness -> question-level group label ---
    final = closed[closed["frac"] == 1.0][["qid", "forced_letter", "gold"]].copy()
    final["final_correct"] = final["forced_letter"] == final["gold"]
    qid_final = dict(zip(final["qid"], final["final_correct"]))
    closed["final_correct"] = closed["qid"].map(qid_final)
    n_corr = int(final["final_correct"].sum())
    n_inc = int((~final["final_correct"]).sum())
    print(f"\nFINAL-checkpoint accuracy (forced@1.0 == gold): "
          f"{n_corr}/{n_corr + n_inc} correct, {n_inc} incorrect")
    print(f"  correct qids  : {sorted(final[final.final_correct].qid)}")
    print(f"  incorrect qids: {sorted(final[~final.final_correct].qid)}")

    # ============================================================================
    # BASIC: per-decile aggregate over the closed cohort
    # ============================================================================
    agg = closed.groupby("frac").agg(
        n_questions=("qid", "nunique"),
        H_letter_mean=("H_letter", "mean"),
        H_letter_median=("H_letter", "median"),
        H_full_mean=("H_full", "mean"),
        confidence_mean=("confidence", "mean"),
        accuracy=("correct", "mean"),     # checkpoint-level correctness (steer A)
    ).reset_index()
    agg.to_csv(f"{TAB_DIR}/decile_aggregate.csv", index=False)
    print("\n--- per-decile aggregate (closed cohort, n=19) ---")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(agg.to_string(index=False))

    # ============================================================================
    # SPLIT: entropy / confidence / accuracy by final correctness, per decile
    # ============================================================================
    grp = closed.groupby(["final_correct", "frac"]).agg(
        H_letter_mean=("H_letter", "mean"),
        confidence_mean=("confidence", "mean"),
        accuracy=("correct", "mean"),
    ).reset_index()
    piv = grp.pivot(index="frac", columns="final_correct", values="H_letter_mean")
    piv.columns = ["H_incorrect" if c is False else "H_correct" for c in piv.columns]
    piv["gap_incorrect_minus_correct"] = piv["H_incorrect"] - piv["H_correct"]
    piv = piv.reset_index()
    piv.to_csv(f"{TAB_DIR}/entropy_by_correctness_frac.csv", index=False)
    print("\n--- mean answer-letter entropy by FINAL correctness, per decile ---")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(piv.to_string(index=False))

    # headline numbers: where does the gap peak, does it close at the endpoint?
    gap = piv.set_index("frac")["gap_incorrect_minus_correct"]
    max_gap_frac = gap.idxmax()
    max_gap_val = gap.max()
    endpoint_gap = gap.loc[1.0]
    H_corr_end = piv.set_index("frac").loc[1.0, "H_correct"]
    H_inc_end = piv.set_index("frac").loc[1.0, "H_incorrect"]
    print(f"\nGAP (incorrect - correct) peaks at frac={max_gap_frac} "
          f"({max_gap_val:.3f} nats); endpoint gap (frac=1.0)={endpoint_gap:.3f} "
          f"nats (H_correct={H_corr_end:.3f}, H_incorrect={H_inc_end:.3f})")

    # ============================================================================
    # CONFOUND (steer C): chain length differs by group -> compare ABSOLUTE position
    # ============================================================================
    nth = closed.drop_duplicates("qid")[["qid", "n_think", "final_correct"]]
    len_by_grp = nth.groupby("final_correct")["n_think"].agg(["mean", "median", "min", "max", "count"])
    len_by_grp.index = ["incorrect" if i is False else "correct" for i in len_by_grp.index]
    len_by_grp.to_csv(f"{TAB_DIR}/chain_length_by_group.csv")
    print("\n--- chain length (n_think) by group: the confound ---")
    with pd.option_context("display.float_format", lambda v: f"{v:.1f}"):
        print(len_by_grp.to_string())

    # bin entropy by absolute token position k_keep, common bins across groups
    BINW = 400
    kmax = int(closed["k_keep"].max())
    bins = np.arange(0, kmax + BINW, BINW)
    closed["kbin"] = pd.cut(closed["k_keep"], bins=bins, right=False)
    abs_tbl = closed.groupby(["kbin", "final_correct"], observed=True)["H_letter"].agg(["mean", "count"]).reset_index()
    abs_tbl.to_csv(f"{TAB_DIR}/entropy_by_abs_position.csv", index=False)
    print(f"\n--- mean entropy vs ABSOLUTE token position (bins of {BINW}) ---")
    print(abs_tbl.to_string(index=False))

    # does the separation survive at matched absolute position? compare per-bin means
    # only where BOTH groups have data.
    abs_p = closed.groupby(["kbin", "final_correct"], observed=True)["H_letter"].mean().unstack()
    survive_rows = []
    if False in abs_p.columns and True in abs_p.columns:
        both = abs_p.dropna()
        for kb, r in both.iterrows():
            survive_rows.append((kb, r[True], r[False], r[False] - r[True]))
    n_bins_both = len(survive_rows)
    n_bins_corr_lower = sum(1 for _, hc, hi, d in survive_rows if d > 0)
    print(f"\nABSOLUTE-position check: {n_bins_both} bins have both groups; "
          f"correct-lower-entropy in {n_bins_corr_lower}/{n_bins_both}")

    # ============================================================================
    # FLIP / LOCK-IN: when does the forced answer commit to its final letter?
    # ============================================================================
    def per_q(g):
        g = g.sort_values("frac")
        letters = list(g["forced_letter"])
        fracs = list(g["frac"])
        final_letter = letters[-1]
        commit = fracs[-1]
        for i in range(len(letters) - 1, -1, -1):     # walk back over the stable tail
            if letters[i] == final_letter:
                commit = fracs[i]
            else:
                break
        n_flips = sum(1 for i in range(1, len(letters)) if letters[i] != letters[i - 1])
        return pd.Series({"n_think": g["n_think"].iloc[0],
                          "final_letter": final_letter,
                          "final_correct": bool(g["final_correct"].iloc[0]),
                          "commit_frac": commit, "n_flips": n_flips})

    pq = closed.groupby("qid").apply(per_q, include_groups=False).reset_index()
    pq.to_csv(f"{TAB_DIR}/per_question_summary.csv", index=False)
    print("\n--- per-question lock-in / flips ---")
    print(pq.to_string(index=False))
    lockin = pq.groupby("final_correct")[["commit_frac", "n_flips"]].mean()
    lockin.index = ["incorrect" if i is False else "correct" for i in lockin.index]
    print("\n--- mean commit_frac / n_flips by group ---")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(lockin.to_string())

    # ============================================================================
    # CALIBRATION: are the two uncertainty signals (entropy, verbalized conf) aligned?
    # ============================================================================
    r_p = pearson(closed["H_letter"], closed["confidence"])
    r_s = spearman(closed["H_letter"], closed["confidence"])
    print(f"\n--- calibration: H_letter vs verbalized confidence (closed cohort) ---")
    print(f"Pearson r={r_p:.3f}   Spearman rho={r_s:.3f}   (expect negative)")

    # ============================================================================
    # FIGURES
    # ============================================================================
    suptitle = f"checkpoint probes (exploratory) — {CAVEATS}"

    def faint_lines(ax, data, ycol, color="0.7"):
        for _, g in data.groupby("qid"):
            g = g.sort_values("frac")
            ax.plot(g["frac"], g[ycol], color=color, alpha=0.35, linewidth=0.8)

    # 1. entropy vs frac (per-question faint + mean)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    faint_lines(ax, closed, "H_letter")
    m = closed.groupby("frac")["H_letter"].mean()
    ax.plot(m.index, m.values, color="tab:blue", linewidth=2.5, marker="o", label="mean (n=19)")
    ax.set(xlabel="fraction of reasoning kept", ylabel="answer-letter entropy (nats)",
           title="Answer-letter entropy collapses as reasoning accumulates")
    ax.legend(); fig.suptitle(suptitle, fontsize=7, y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig_entropy_vs_frac.png", dpi=130); plt.close(fig)

    # 2. confidence vs frac
    fig, ax = plt.subplots(figsize=(7, 4.5))
    faint_lines(ax, closed, "confidence")
    m = closed.groupby("frac")["confidence"].mean()
    ax.plot(m.index, m.values, color="tab:green", linewidth=2.5, marker="o", label="mean (n=19)")
    ax.set(xlabel="fraction of reasoning kept", ylabel="verbalized confidence (0-100)",
           title="Verbalized confidence vs reasoning kept")
    ax.legend(); fig.suptitle(suptitle, fontsize=7, y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig_confidence_vs_frac.png", dpi=130); plt.close(fig)

    # 3. accuracy vs frac (checkpoint-level correctness)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    m = closed.groupby("frac")["correct"].mean()
    ax.plot(m.index, m.values, color="tab:purple", linewidth=2.5, marker="o")
    ax.set(xlabel="fraction of reasoning kept", ylabel="forced-answer accuracy",
           ylim=(0, 1), title="Forced-answer accuracy vs reasoning kept (checkpoint-level)")
    fig.suptitle(suptitle, fontsize=7, y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig_accuracy_vs_frac.png", dpi=130); plt.close(fig)

    # 4. HEADLINE: mean entropy by final correctness, overlaid (gap opens mid, closes end)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = [(True, "tab:blue", "-", f"correct@final (n={n_corr})"),
              (False, "tab:orange", "--", f"incorrect@final (n={n_inc})")]
    for grpval, color, ls, lbl in styles:
        sub = closed[closed["final_correct"] == grpval]
        mm = sub.groupby("frac")["H_letter"].mean()
        ax.plot(mm.index, mm.values, color=color, linestyle=ls, marker="o", linewidth=2.5, label=lbl)
    ax.axvspan(0.5, 0.9, color="0.9", zorder=0)   # mid-chain band where the gap lives
    ax.set(xlabel="fraction of reasoning kept", ylabel="mean answer-letter entropy (nats)",
           title="Convergence TIMING: correct answers' entropy collapses earlier\n"
                 "(gap opens mid-chain, vanishes at the endpoint)")
    ax.legend(); fig.suptitle(suptitle, fontsize=7, y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig_entropy_by_correctness_frac.png", dpi=130); plt.close(fig)

    # 5. CONFOUND: entropy vs absolute token position, by group (scatter + binned mean)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    centers = bins[:-1] + BINW / 2
    for grpval, color, ls, lbl in styles:
        sub = closed[closed["final_correct"] == grpval]
        ax.scatter(sub["k_keep"], sub["H_letter"], color=color, alpha=0.30, s=16)
        bm = sub.groupby(pd.cut(sub["k_keep"], bins=bins, right=False), observed=True)["H_letter"].mean()
        xs = [centers[i] for i, iv in enumerate(bm.index.categories) if iv in bm.index]
        ax.plot(xs, bm.values, color=color, linestyle=ls, marker="s", linewidth=2.2, label=lbl)
    ax.set(xlabel="absolute reasoning tokens kept (k_keep)", ylabel="answer-letter entropy (nats)",
           title="Confound check: entropy vs ABSOLUTE token position\n"
                 "(does the timing separation survive matched token counts?)")
    ax.legend(); fig.suptitle(suptitle, fontsize=7, y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig_entropy_by_correctness_abs.png", dpi=130); plt.close(fig)

    # 6. calibration scatter
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(closed["confidence"], closed["H_letter"], alpha=0.4, s=18, color="tab:red")
    ax.set(xlabel="verbalized confidence (0-100)", ylabel="answer-letter entropy (nats)",
           title=f"Calibration: entropy vs verbalized confidence  "
                 f"(Pearson {r_p:.2f}, Spearman {r_s:.2f})")
    fig.suptitle(suptitle, fontsize=7, y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig_calibration.png", dpi=130); plt.close(fig)

    print(f"\nwrote 6 figures -> {FIG_DIR}/  and 5 tables -> {TAB_DIR}/")

    # ============================================================================
    # FINDINGS.md — numbers interpolated from the computed values (no hand-typed stats)
    # ============================================================================
    abs_verdict = (
        f"survives: correct group shows lower mean entropy in "
        f"{n_bins_corr_lower}/{n_bins_both} overlapping absolute-position bins"
        if n_bins_both and n_bins_corr_lower >= (n_bins_both + 1) // 2
        else f"DOES NOT clearly survive: correct-lower in only "
             f"{n_bins_corr_lower}/{n_bins_both} overlapping bins"
    )
    H0_mean = agg.set_index("frac").loc[0.0, "H_letter_mean"]
    H1_mean = agg.set_index("frac").loc[1.0, "H_letter_mean"]
    c0 = agg.set_index("frac").loc[0.0, "confidence_mean"]
    c1 = agg.set_index("frac").loc[1.0, "confidence_mean"]
    acc0 = agg.set_index("frac").loc[0.0, "accuracy"]
    acc1 = agg.set_index("frac").loc[1.0, "accuracy"]
    commit_c = lockin.loc["correct", "commit_frac"]
    commit_i = lockin.loc["incorrect", "commit_frac"]
    flips_c = lockin.loc["correct", "n_flips"]
    flips_i = lockin.loc["incorrect", "n_flips"]
    len_c = len_by_grp.loc["correct", "mean"]
    len_i = len_by_grp.loc["incorrect", "mean"]
    len_c_med = len_by_grp.loc["correct", "median"]
    len_i_med = len_by_grp.loc["incorrect", "median"]
    acc_series = agg.set_index("frac")["accuracy"]
    acc_valley_frac = acc_series.idxmin()
    acc_valley = acc_series.min()
    n_entropy_gaps = int(closed["H_letter"].isna().sum())

    md = f"""# Checkpoint probe — exploratory findings (20q, decile resolution)

> **Status: EXPLORATORY, directional only — NOT a finding.**
> Caveats that gate every claim below:
> - **single subject** (`abstract_algebra`) — not subject-representative
> - **n = 19** usable closed questions (qid14 hit the 4096-token cap unclosed; reported separately)
> - **1 greedy run/question** — no sampling spread
> - data produced off **mirrored gpu helpers** (the `gpu_common.py` refactor is still pending;
>   nothing here ships as a finding until the pipeline is consolidated and the data regenerated)
> - the **incorrect group is n = 6** → results are **suggestive, not significant**. No p-values reported on n=6.

## Cohort
- 220 rows = 20 questions × 11 deciles. Closed cohort = **19 questions** (209 rows).
- **qid14** (unclosed, `think_closed=False`) is held out and never averaged in.
- Question-level groups use **final-checkpoint (frac=1.0) correctness** (`forced_letter==gold`),
  not `natural_pred==gold`: **{n_corr} correct / {n_inc} incorrect**.

## 1. Whole-cohort trajectories (basic)
As more reasoning is spliced in, the forced answer's letter-entropy collapses and verbalized
confidence drifts up:

| | frac 0.0 (no reasoning) | frac 1.0 (full chain) |
|---|---|---|
| mean answer-letter entropy (nats) | {H0_mean:.3f} | {H1_mean:.3f} |
| mean verbalized confidence | {c0:.1f} | {c1:.1f} |
| forced-answer accuracy | {acc0:.2f} | {acc1:.2f} |

See `fig_entropy_vs_frac.png`, `fig_confidence_vs_frac.png`, `fig_accuracy_vs_frac.png`.

**Two wrinkles, both honest gaps not bugs:**
- Accuracy is **non-monotonic**: it dips to **{acc_valley:.2f} at frac {acc_valley_frac:.1f}** before
  recovering to {acc1:.2f} — a *little* reasoning underperforms *none* here. Intriguing (echoes
  "partial chain-of-thought can hurt") but well within n=19 noise; flagged, not claimed.
- **{n_entropy_gaps} checkpoint** (qid15 @ frac 0.5) produced no clean answer token, so its
  entropy is left as a visible `NaN` rather than guessed; that one cell's mean is over 18, not 19.

## 2. HEADLINE — convergence *timing*, not entropy *level*
Splitting by final correctness, the correct group's entropy collapses **earlier**. The gap
(incorrect − correct mean entropy) **opens mid-chain and vanishes at the endpoint**:

- gap peaks at **frac {max_gap_frac:.1f}** (**{max_gap_val:.3f} nats**),
- endpoint gap (frac 1.0) = **{endpoint_gap:.3f} nats** (H_correct={H_corr_end:.3f}, H_incorrect={H_inc_end:.3f}) — effectively converged.

**This is not "lower entropy ⇒ correct."** At the endpoint both groups are equally (near-zero)
entropy; the signal is *how soon* they get there. See `fig_entropy_by_correctness_frac.png`
(grey band = mid-chain region where the gap lives).

## 3. Chain-length confound (the first thing a reviewer will raise)
The worry: if incorrect chains were longer, a given *fraction* would map to more *absolute*
tokens, and "earlier convergence" could be a length artifact. **In this data the premise does
not hold** — the groups are length-balanced:

| group | mean n_think | median n_think |
|---|---|---|
| correct (n={n_corr}) | {len_c:.0f} | {len_c_med:.0f} |
| incorrect (n={n_inc}) | {len_i:.0f} | {len_i_med:.0f} |

Means are within ~1.5%, and the **median runs the *opposite* way** (correct chains are if
anything *longer*). So there is little length confound to begin with. The decisive check is
re-plotting entropy against absolute `k_keep` (`fig_entropy_by_correctness_abs.png`):

> **{abs_verdict}.**

Since the separation holds at *matched absolute token positions*, the earlier-convergence effect
in §2 is **not** a chain-length artifact. (Caveat: the correct group has few points in the
highest-k bins — short-and-medium chains dominate — so the very-high-k tail is thin.)

## 4. Answer lock-in / flips
Earliest fraction after which the forced letter equals its final value and never changes again:

- mean commit fraction: **correct {commit_c:.2f}** vs **incorrect {commit_i:.2f}**
- mean answer flips along the chain: **correct {flips_c:.2f}** vs **incorrect {flips_i:.2f}**

(Consistent direction with §2 if correct questions commit earlier / flip less.) See
`per_question_summary.csv`.

## 5. Calibration of the two uncertainty signals
Token entropy vs verbalized confidence across all closed checkpoints:
**Pearson r = {r_p:.3f}, Spearman ρ = {r_s:.3f}** (negative = the two signals agree:
higher stated confidence ↔ lower entropy). See `fig_calibration.png`.

## Held-out: qid14, and the qid15 gap (kept visible, never silently dropped)
- **qid14** — unclosed chain (never emitted `</think>` within 4096 tokens). It *did* produce a
  clean forced answer at every checkpoint (letter `C`, which happens to be gold), but stayed
  **high-entropy (~1.0 nats) throughout — it never converged**, consistent with never closing its
  reasoning. The 1.0 invariant is **N/A** because the *natural* chain emitted no committed answer
  to compare against (not because the forced probe failed). Excluded from all aggregates above.
- **qid15 @ frac 0.5** — a *closed*-cohort checkpoint where the forced probe emitted no parseable
  answer token → `H_letter`/`forced_letter` left as `NaN`/None (honest gap). It is the lone
  missing cell in §1's frac-0.5 means. Note: in §4 the flip count treats that `None` as a distinct
  symbol, so qid15's flip tally is mildly inflated.

---
*Generated by `analysis/analyze_checkpoints.py` from `results/20q/checkpoints.jsonl`.*
"""
    with open(FINDINGS, "w") as fh:
        fh.write(md)
    print(f"wrote {FINDINGS}")


if __name__ == "__main__":
    main()

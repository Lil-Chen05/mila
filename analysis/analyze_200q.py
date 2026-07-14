"""Analysis of the 200q run: signal x signal correlations + outlier forensics.

Inputs (all small local files -- LOGIN-NODE-SAFE, no model/HF dataset/GPU):
  results/checkpoints_200q.jsonl         one row per (qid, decile checkpoint)
  results/chain_token_entropy_200q.jsonl one row per qid, full per-token entropy trace
  results/questions_200q.json            question text (exported by dump_questions.py)

GK's asks driving this script:
  1. lots of plots showing correlations between the signals
     (reasoning-token entropy ~ answer entropy ~ verbalized confidence ~ correctness)
  2. outlier forensics: where is the gap between verbalized confidence and
     accuracy / token entropy biggest? inspect those datapoints manually.

Cohort discipline (approved framing):
  - grouping label = FINAL-checkpoint correctness (forced_letter@frac1.0 == gold)
  - closed cohort only in aggregates; truncated (16k cap) and skipped
    (n_think=2 no-reasoning) cohorts reported separately, never averaged in.

Run:  uv run python analysis/analyze_200q.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")            # headless: non-interactive backend before pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_TAG = "200q"
CKPT_PATH = f"results/checkpoints_{RUN_TAG}.jsonl"
ENT_PATH = f"results/chain_token_entropy_{RUN_TAG}.jsonl"
Q_PATH = f"results/questions_{RUN_TAG}.json"
TAB_DIR = "analysis/tables"
FIG_DIR = "analysis/figures"
FINDINGS = f"analysis/FINDINGS_{RUN_TAG}.md"
OUTLIERS_MD = f"analysis/OUTLIERS_{RUN_TAG}.md"

N_QUESTIONS = 200
N_PROFILE_BINS = 20          # normalized-position bins for token-entropy profiles
TAIL_FRAC = 0.10             # "late chain" = last 10% of think tokens
CAVEATS = ("1 greedy run/question | grouping = final-checkpoint correctness | "
           "truncated (16k) and skipped (no-reasoning) cohorts held out of aggregates")


# ---------- torch-free stats helpers -------------------------------------------
def pearson(x, y):
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(s) < 3:
        return float("nan")
    return np.corrcoef(s["x"], s["y"])[0, 1]


def spearman(x, y):
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(s) < 3:
        return float("nan")
    return np.corrcoef(s["x"].rank(), s["y"].rank())[0, 1]


def auroc(scores, labels):
    """Mann-Whitney AUROC: P(score_correct > score_incorrect). No sklearn."""
    s = pd.DataFrame({"s": scores, "y": labels}).dropna()
    pos, neg = s[s["y"].astype(bool)], s[~s["y"].astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = s["s"].rank()                     # average ranks handle ties
    u = ranks[s["y"].astype(bool)].sum() - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def profile_bins(trace, n_bins=N_PROFILE_BINS):
    """Average a variable-length entropy trace into n_bins normalized-position bins."""
    trace = np.asarray(trace, dtype=float)
    edges = np.linspace(0, len(trace), n_bins + 1).astype(int)
    return np.array([trace[a:b].mean() if b > a else np.nan
                     for a, b in zip(edges[:-1], edges[1:])])


def qsnippet(qrow, width=180):
    text = qrow["question"].replace("\n", " ")
    return text[:width] + ("..." if len(text) > width else "")


def main():
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    ckpt = pd.read_json(CKPT_PATH, lines=True)
    ents = [json.loads(l) for l in open(ENT_PATH)]
    ent = pd.DataFrame([{k: v for k, v in r.items() if k != "per_token_entropy"}
                        for r in ents])
    traces = {r["qid"]: r["per_token_entropy"] for r in ents}
    questions = {q["qid"]: q for q in json.load(open(Q_PATH))}

    print("=" * 80)
    print(f"200q ANALYSIS — {CAVEATS}")
    print("=" * 80)
    print(f"checkpoint rows: {len(ckpt)}   qids: {ckpt['qid'].nunique()}   "
          f"entropy records: {len(ent)}   questions file: {len(questions)}")

    # ---------- cohort accounting (every one of the 200 ends up somewhere) -------
    present = set(ent["qid"])
    skipped_qids = sorted(set(range(N_QUESTIONS)) - present)
    skipped = pd.DataFrame([{"qid": q, "subject": questions[q]["subject"],
                             "question": qsnippet(questions[q])} for q in skipped_qids])
    skipped.to_csv(f"{TAB_DIR}/skipped_no_reasoning_{RUN_TAG}.csv", index=False)

    closed_qids = set(ent[ent["think_closed"]]["qid"])
    trunc_qids = sorted(present - closed_qids)
    print(f"\nCOHORTS: closed={len(closed_qids)}  truncated@16k={len(trunc_qids)}  "
          f"skipped(no-reasoning)={len(skipped_qids)}  "
          f"total={len(closed_qids) + len(trunc_qids) + len(skipped_qids)}/{N_QUESTIONS}")
    print("skipped subjects:", dict(skipped["subject"].value_counts()) if len(skipped) else {})

    # ---------- question-level signal table --------------------------------------
    def at_frac(f, col):
        sub = ckpt[ckpt["frac"] == f][["qid", col]]
        return dict(zip(sub["qid"], sub[col]))

    fl_end = at_frac(1.0, "forced_letter")
    sig = ent[["qid", "subject", "gold", "n_think", "think_closed",
               "natural_confidence", "mean_think_entropy"]].copy()
    sig = sig.rename(columns={"mean_think_entropy": "H_think_mean",
                              "natural_confidence": "conf_natural"})
    sig["H_think_tail"] = sig.apply(
        lambda r: float(np.mean(traces[r["qid"]][max(0, int(r["n_think"] * (1 - TAIL_FRAC))):r["n_think"]])),
        axis=1)
    sig["H_ans_prior"] = sig["qid"].map(at_frac(0.0, "H_letter"))   # no-reasoning baseline
    sig["H_ans_mid"] = sig["qid"].map(at_frac(0.5, "H_letter"))
    sig["H_ans_end"] = sig["qid"].map(at_frac(1.0, "H_letter"))
    sig["conf_end"] = sig["qid"].map(at_frac(1.0, "confidence"))
    sig["final_letter"] = sig["qid"].map(fl_end)
    sig["final_correct"] = np.where(sig["final_letter"].isna(), np.nan,
                                    sig["final_letter"] == sig["gold"])
    sig.to_csv(f"{TAB_DIR}/signals_per_question_{RUN_TAG}.csv", index=False)

    closed = sig[sig["think_closed"] & sig["final_correct"].notna()].copy()
    closed["final_correct"] = closed["final_correct"].astype(bool)
    n_corr = int(closed["final_correct"].sum())
    n_inc = len(closed) - n_corr
    acc = n_corr / len(closed)
    print(f"\nclosed cohort with final answer: {len(closed)}  "
          f"({n_corr} correct / {n_inc} incorrect, acc={acc:.3f})")

    # ============================================================================
    # 1. CALIBRATION: verbalized confidence vs empirical accuracy
    # ============================================================================
    bins = [0, 70, 80, 85, 90, 95, 100.001]
    labels = ["<70", "70-79", "80-84", "85-89", "90-94", "95-100"]
    cal = closed.dropna(subset=["conf_end"]).copy()
    cal["conf_bin"] = pd.cut(cal["conf_end"], bins=bins, labels=labels, right=False)
    cal_tbl = cal.groupby("conf_bin", observed=True).agg(
        n=("qid", "count"), accuracy=("final_correct", "mean"),
        conf_mean=("conf_end", "mean")).reset_index()
    cal_tbl["gap"] = cal_tbl["conf_mean"] / 100 - cal_tbl["accuracy"]
    cal_tbl.to_csv(f"{TAB_DIR}/calibration_bins_{RUN_TAG}.csv", index=False)
    ece = float((cal_tbl["gap"].abs() * cal_tbl["n"]).sum() / cal_tbl["n"].sum())
    print("\n--- calibration (endpoint verbalized confidence vs accuracy) ---")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(cal_tbl.to_string(index=False))
    print(f"ECE (bin-weighted |conf-acc|): {ece:.3f}")

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 6), height_ratios=[3, 1], sharex=True)
    xs = np.arange(len(cal_tbl))
    ax.bar(xs, cal_tbl["accuracy"], width=0.6, color="tab:blue", label="empirical accuracy")
    ax.plot(xs, cal_tbl["conf_mean"] / 100, color="tab:red", marker="o", linewidth=2,
            label="mean stated confidence")
    for x, (a, c) in enumerate(zip(cal_tbl["accuracy"], cal_tbl["conf_mean"] / 100)):
        ax.annotate("", xy=(x, a), xytext=(x, c),
                    arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set(ylabel="accuracy / confidence", ylim=(0, 1.05),
           title=f"Verbalized confidence is mis-calibrated (ECE={ece:.2f}):\n"
                 f"stated ~90% vs actual {acc:.0%} overall")
    ax.legend(loc="lower right")
    ax2.bar(xs, cal_tbl["n"], width=0.6, color="0.6")
    ax2.set(xticks=xs, xticklabels=cal_tbl["conf_bin"], xlabel="stated confidence bin",
            ylabel="n")
    fig.suptitle(CAVEATS, fontsize=7, y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig{RUN_TAG}_calibration_curve.png", dpi=130)
    plt.close(fig)

    # ============================================================================
    # 2. SIGNAL x SIGNAL CORRELATION MATRIX
    # ============================================================================
    SIGNALS = ["H_think_mean", "H_think_tail", "H_ans_prior", "H_ans_mid",
               "H_ans_end", "conf_end", "final_correct"]
    corr_p = pd.DataFrame(index=SIGNALS, columns=SIGNALS, dtype=float)
    corr_s = pd.DataFrame(index=SIGNALS, columns=SIGNALS, dtype=float)
    cnum = closed[SIGNALS].astype(float)
    for a in SIGNALS:
        for b in SIGNALS:
            corr_p.loc[a, b] = pearson(cnum[a], cnum[b])
            corr_s.loc[a, b] = spearman(cnum[a], cnum[b])
    corr_p.to_csv(f"{TAB_DIR}/correlations_pearson_{RUN_TAG}.csv")
    corr_s.to_csv(f"{TAB_DIR}/correlations_spearman_{RUN_TAG}.csv")
    print("\n--- Spearman correlation matrix (question level, closed cohort) ---")
    with pd.option_context("display.float_format", lambda v: f"{v:.2f}"):
        print(corr_s.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, mat, name in ((axes[0], corr_p, "Pearson"), (axes[1], corr_s, "Spearman")):
        im = ax.imshow(mat.values.astype(float), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(SIGNALS)), SIGNALS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(SIGNALS)), SIGNALS, fontsize=8)
        for i in range(len(SIGNALS)):
            for j in range(len(SIGNALS)):
                ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"{name} (n={len(closed)})")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(f"Signal correlations — {CAVEATS}", fontsize=8)
    fig.savefig(f"{FIG_DIR}/fig{RUN_TAG}_signal_correlation_heatmap.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # ============================================================================
    # 3. WHICH SIGNAL PREDICTS CORRECTNESS? (AUROC per signal)
    # ============================================================================
    # sign convention: orient every signal so "higher = predicts correct"
    auroc_specs = [
        ("verbalized confidence (end)", closed["conf_end"]),
        ("-mean think-token entropy", -closed["H_think_mean"]),
        ("-tail think-token entropy", -closed["H_think_tail"]),
        ("-answer entropy @0.0 (prior)", -closed["H_ans_prior"]),
        ("-answer entropy @0.5 (mid)", -closed["H_ans_mid"]),
        ("-answer entropy @1.0 (end)", -closed["H_ans_end"]),
    ]
    aur = pd.DataFrame([{"signal": name, "auroc": auroc(s, closed["final_correct"]),
                         "n": int(pd.DataFrame({"s": s}).dropna().shape[0])}
                        for name, s in auroc_specs])
    aur.to_csv(f"{TAB_DIR}/auroc_{RUN_TAG}.csv", index=False)
    print("\n--- AUROC: P(signal ranks a correct question above an incorrect one) ---")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(aur.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(aur["signal"], aur["auroc"], color=["tab:red"] + ["tab:blue"] * 5)
    ax.axvline(0.5, color="0.3", linestyle="--", label="chance")
    ax.set(xlabel="AUROC for predicting final correctness", xlim=(0.3, 1.0),
           title="Which uncertainty signal actually knows?")
    ax.legend(); fig.suptitle(CAVEATS, fontsize=7, y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig{RUN_TAG}_auroc_bars.png", dpi=130)
    plt.close(fig)

    # ============================================================================
    # 4. SCATTERS (colored by correctness) + confidence distributions
    # ============================================================================
    scatters = [
        ("H_think_mean", "conf_end", "mean think-token entropy (nats)",
         "verbalized confidence (end)", "thinkH_vs_conf"),
        ("H_think_mean", "H_ans_end", "mean think-token entropy (nats)",
         "answer entropy @1.0 (nats)", "thinkH_vs_ansH"),
        ("H_ans_mid", "conf_end", "answer entropy @0.5 (nats)",
         "verbalized confidence (end)", "ansHmid_vs_conf"),
    ]
    for xc, yc, xl, yl, tag in scatters:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for val, color, lbl in ((True, "tab:blue", f"correct (n={n_corr})"),
                                (False, "tab:orange", f"incorrect (n={n_inc})")):
            sub = closed[closed["final_correct"] == val]
            ax.scatter(sub[xc], sub[yc], color=color, alpha=0.55, s=22, label=lbl)
        rho = spearman(closed[xc], closed[yc])
        ax.set(xlabel=xl, ylabel=yl, title=f"{xl}  vs  {yl}   (Spearman {rho:.2f})")
        ax.legend(); fig.suptitle(CAVEATS, fontsize=7, y=1.0)
        fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig{RUN_TAG}_scatter_{tag}.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins_c = np.arange(50, 105, 5)
    for val, color, lbl in ((True, "tab:blue", f"correct (n={n_corr})"),
                            (False, "tab:orange", f"incorrect (n={n_inc})")):
        sub = closed[closed["final_correct"] == val]["conf_end"].dropna()
        ax.hist(sub, bins=bins_c, alpha=0.55, color=color, label=lbl, density=True)
    ax.set(xlabel="verbalized confidence (end)", ylabel="density",
           title="Stated confidence barely separates correct from incorrect")
    ax.legend(); fig.suptitle(CAVEATS, fontsize=7, y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig{RUN_TAG}_confidence_hist.png", dpi=130)
    plt.close(fig)

    # ============================================================================
    # 5. TRAJECTORY REPLICATION at n=150: answer entropy by frac + abs position
    # ============================================================================
    ck = ckpt[ckpt["qid"].isin(closed["qid"])].copy()
    ck["final_correct"] = ck["qid"].map(dict(zip(closed["qid"], closed["final_correct"])))
    styles = [(True, "tab:blue", "-", f"correct@final (n={n_corr})"),
              (False, "tab:orange", "--", f"incorrect@final (n={n_inc})")]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for val, color, ls, lbl in styles:
        mm = ck[ck["final_correct"] == val].groupby("frac")["H_letter"].mean()
        ax.plot(mm.index, mm.values, color=color, linestyle=ls, marker="o",
                linewidth=2.5, label=lbl)
    ax.set(xlabel="fraction of reasoning kept", ylabel="mean answer-letter entropy (nats)",
           title="Replication at n=150 / 51 subjects: does correct still converge earlier?")
    ax.legend(); fig.suptitle(CAVEATS, fontsize=7, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig{RUN_TAG}_entropy_by_correctness_frac.png", dpi=130)
    plt.close(fig)

    gap_tbl = ck.groupby(["final_correct", "frac"])["H_letter"].mean().unstack(0)
    gap_tbl["gap"] = gap_tbl[False] - gap_tbl[True]
    gap_tbl.to_csv(f"{TAB_DIR}/entropy_by_correctness_frac_{RUN_TAG}.csv")
    max_gap_frac = float(gap_tbl["gap"].idxmax())
    max_gap = float(gap_tbl["gap"].max())
    end_gap = float(gap_tbl["gap"].loc[1.0])
    print(f"\nH_letter gap (incorrect-correct): peak {max_gap:.3f} nats at frac "
          f"{max_gap_frac:.1f}; endpoint gap {end_gap:.3f} nats")

    # median n_think is ~560, so bins must resolve sub-chain-length structure:
    # 100-token bins over the dense 0-2000 region (beyond that both groups thin out)
    BINW = 100
    kb = np.arange(0, 2000 + BINW, BINW)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    survive = []
    binned = {}
    for val, color, ls, lbl in styles:
        sub = ck[(ck["final_correct"] == val) & (ck["k_keep"] <= 2000)]
        bm = sub.groupby(pd.cut(sub["k_keep"], bins=kb, right=False),
                         observed=True)["H_letter"].mean()
        binned[val] = bm
        xs = [iv.left + BINW / 2 for iv in bm.index]
        ax.plot(xs, bm.values, color=color, linestyle=ls, marker="s", linewidth=2.2, label=lbl)
    both = pd.DataFrame(binned).dropna()
    survive = [(idx, r[True], r[False]) for idx, r in both.iterrows()]
    n_lower = sum(1 for _, hc, hi in survive if hi > hc)
    ax.set(xlabel="absolute reasoning tokens kept (k_keep)",
           ylabel="mean answer-letter entropy (nats)",
           title=f"Confound check at ABSOLUTE position: correct lower in "
                 f"{n_lower}/{len(survive)} shared bins")
    ax.legend(); fig.suptitle(CAVEATS, fontsize=7, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig{RUN_TAG}_entropy_by_correctness_abs.png", dpi=130)
    plt.close(fig)
    print(f"absolute-position check: correct-lower-entropy in {n_lower}/{len(survive)} shared bins")

    # ============================================================================
    # 6. TOKEN-ENTROPY PROFILES along the reasoning chain (the NEW signal)
    # ============================================================================
    prof = {True: [], False: []}
    for _, r in closed.iterrows():
        prof[r["final_correct"]].append(profile_bins(traces[r["qid"]][:r["n_think"]]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = (np.arange(N_PROFILE_BINS) + 0.5) / N_PROFILE_BINS
    for val, color, ls, lbl in styles:
        arr = np.vstack(prof[val])
        mean = np.nanmean(arr, axis=0)
        lo, hi = np.nanpercentile(arr, [25, 75], axis=0)
        ax.plot(xs, mean, color=color, linestyle=ls, linewidth=2.5, label=lbl)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
    ax.set(xlabel="normalized position in reasoning chain",
           ylabel="token entropy (nats)",
           title="Reasoning-token entropy along the chain (mean, IQR band)")
    ax.legend(); fig.suptitle(CAVEATS, fontsize=7, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig{RUN_TAG}_token_entropy_profiles.png", dpi=130)
    plt.close(fig)

    # ============================================================================
    # 7. RUNAWAY (16k-cap) FORENSICS: repetition-loop signature?
    # ============================================================================
    run_rows = []
    for q in trunc_qids:
        tr = np.asarray(traces[q], dtype=float)
        head, tail = tr[:1024], tr[-1024:]
        run_rows.append({"qid": q, "subject": ent.set_index("qid").loc[q, "subject"],
                         "n_gen": len(tr), "head_mean": head.mean(),
                         "tail_mean": tail.mean(),
                         "tail_frac_below_0.05": float((tail < 0.05).mean())})
    runaway = pd.DataFrame(run_rows)
    runaway.to_csv(f"{TAB_DIR}/runaway_stats_{RUN_TAG}.csv", index=False)
    if len(runaway):
        print("\n--- runaway chains (hit 16k cap): head vs tail entropy ---")
        with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
            print(runaway.to_string(index=False))

        W = 128
        sample = trunc_qids[:6]
        fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=False, sharey=True)
        for ax, q in zip(axes.flat, sample):
            tr = np.asarray(traces[q], dtype=float)
            roll = np.convolve(tr, np.ones(W) / W, mode="valid")
            ax.plot(roll, linewidth=0.8, color="tab:red")
            ax.set_title(f"qid {q} ({ent.set_index('qid').loc[q, 'subject']})", fontsize=8)
        fig.suptitle(f"Runaway chains: rolling(±{W}) token entropy — loop = flat/periodic tail",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/fig{RUN_TAG}_runaway_traces.png", dpi=130)
        plt.close(fig)

    # ============================================================================
    # 8. OUTLIER FORENSICS (GK bullet 2): shortlists + manual-inspection gallery
    # ============================================================================
    wrong = closed[~closed["final_correct"]].copy()
    over = wrong.sort_values(["conf_end", "H_ans_end"],
                             ascending=[False, True]).head(10)
    over.to_csv(f"{TAB_DIR}/outliers_overconfident_wrong_{RUN_TAG}.csv", index=False)

    # signals disagree: stated certain but answer distribution NOT collapsed
    dis = closed[(closed["conf_end"] >= 90)].sort_values("H_ans_end", ascending=False).head(10)
    dis.to_csv(f"{TAB_DIR}/outliers_signal_disagree_{RUN_TAG}.csv", index=False)

    def gallery(df, title, why):
        parts = [f"## {title}\n\n{why}\n"]
        for _, r in df.iterrows():
            q = questions[r["qid"]]
            choices = "\n".join(f"  - {'ABCD'[i]}. {c}" for i, c in enumerate(q["choices"]))
            parts.append(
                f"### qid {r['qid']} — {r['subject']}\n"
                f"**gold {r['gold']} / model {r['final_letter']} "
                f"({'RIGHT' if r['final_correct'] else 'WRONG'})** | "
                f"conf={r['conf_end']:.0f} | H_ans_end={r['H_ans_end']:.3f} | "
                f"H_ans_mid={r['H_ans_mid']:.3f} | H_think_mean={r['H_think_mean']:.3f} | "
                f"n_think={r['n_think']}\n\n"
                f"> {qsnippet(q, 400)}\n\n{choices}\n")
        return "\n".join(parts)

    md = (f"# Outlier gallery — {RUN_TAG}\n\n*{CAVEATS}*\n\n"
          + gallery(over, "Most overconfident wrong answers",
                    "Sorted by stated confidence (desc), then answer entropy (asc): "
                    "the model is wrong, *says* it is sure, and its answer "
                    "distribution agrees it is sure — the worst calibration failures.")
          + "\n" + gallery(dis, "Signals disagree: stated certain, distribution uncertain",
                           "conf_end >= 90 but highest answer-entropy at the endpoint: "
                           "verbalized confidence and token entropy point opposite ways."))
    with open(OUTLIERS_MD, "w") as fh:
        fh.write(md)
    print(f"\nwrote outlier gallery -> {OUTLIERS_MD}")
    print("top overconfident-wrong:",
          [(int(r.qid), r.subject, f"conf={r.conf_end:.0f}") for r in over.itertuples()][:5])

    # ============================================================================
    # 9. FINDINGS
    # ============================================================================
    aur_i = aur.set_index("signal")["auroc"]
    corr_hthink_conf = spearman(closed["H_think_mean"], closed["conf_end"])
    corr_hthink_hans = spearman(closed["H_think_mean"], closed["H_ans_end"])
    corr_hans_conf = spearman(closed["H_ans_mid"], closed["conf_end"])
    tail_loop = float(runaway["tail_frac_below_0.05"].mean()) if len(runaway) else float("nan")
    findings = f"""# 200q findings — three uncertainty signals on the same chains

*{CAVEATS}*

## Cohorts (all 200 accounted for)
- **closed & answered: {len(closed)}** ({n_corr} correct / {n_inc} incorrect, acc={acc:.2f})
- truncated at 16k cap: {len(trunc_qids)} (runaway chains, forensics below)
- skipped, model wrote NO reasoning (n_think=2): {len(skipped_qids)} — all long-passage
  humanities/social subjects; see `tables/skipped_no_reasoning_{RUN_TAG}.csv`

## 1. Calibration (GK bullet 1+2 anchor)
Stated confidence ~90 everywhere; actual accuracy {acc:.2f}. **ECE={ece:.2f}.**
See `fig{RUN_TAG}_calibration_curve.png`; the per-bin table is
`tables/calibration_bins_{RUN_TAG}.csv`.

## 2. Which signal knows? (AUROC, higher = better ranks correct above incorrect)
| signal | AUROC |
|---|---|
| verbalized confidence (end) | {aur_i['verbalized confidence (end)']:.3f} |
| mean think-token entropy | {aur_i['-mean think-token entropy']:.3f} |
| tail think-token entropy | {aur_i['-tail think-token entropy']:.3f} |
| answer entropy @0.5 (mid) | {aur_i['-answer entropy @0.5 (mid)']:.3f} |
| answer entropy @1.0 (end) | {aur_i['-answer entropy @1.0 (end)']:.3f} |

## 3. Signal x signal (Spearman, question level)
- think-token entropy ~ verbalized confidence: **{corr_hthink_conf:.2f}**
- think-token entropy ~ answer entropy (end): **{corr_hthink_hans:.2f}**
- answer entropy (mid) ~ verbalized confidence: **{corr_hans_conf:.2f}**
Full matrices: `tables/correlations_*_{RUN_TAG}.csv`, heatmap figure.

## 4. Timing replication (20q headline at n={len(closed)}, 51 subjects)
Gap (incorrect - correct mean answer entropy) peaks at frac {max_gap_frac:.1f}
({max_gap:.3f} nats), endpoint gap {end_gap:.3f} nats. Absolute-position check:
correct-lower in only {n_lower}/{len(survive)} shared {BINW}-token bins — at n=150
with wildly varying chain lengths, the separation lives in NORMALIZED position
(fraction of chain), not absolute token count: entropy collapse timing appears to
scale with chain length. Needs discussion with GK — this reframes (not refutes)
the 20q timing story.

## 5. Runaway chains ({len(trunc_qids)} at the 16k cap)
Mean fraction of final-1024 tokens with entropy < 0.05 nats: **{tail_loop:.2f}**
(repetition-loop signature if high). Traces: `fig{RUN_TAG}_runaway_traces.png`.

## 6. Outliers for manual inspection
`OUTLIERS_{RUN_TAG}.md` — top overconfident-wrong + signals-disagree galleries
with full question text.

---
*Generated by `analysis/analyze_200q.py`.*
"""
    with open(FINDINGS, "w") as fh:
        fh.write(findings)
    print(f"wrote {FINDINGS}")
    print(f"\nfigures -> {FIG_DIR}/fig{RUN_TAG}_*.png   tables -> {TAB_DIR}/*_{RUN_TAG}.csv")


if __name__ == "__main__":
    main()

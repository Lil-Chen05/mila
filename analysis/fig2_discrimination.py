"""Figure 2 (RQ1): discrimination of eventual natural-answer correctness across
the reasoning trajectory.

For each checkpoint fraction f, AUROC (oriented so >0.5 = better) of the
information available THROUGH f for discriminating the natural final answer:
  - prefix (cumulative) reasoning entropy  [starts at f=0.1];
  - checkpoint answer-choice entropy       [11 fractions];
  - repaired verbalized confidence         [11 fractions].
95% subject-stratified question-cluster bootstrap. A thin strip shows per-signal
coverage (n) by fraction. Endpoint (f=1.0) forced answer equals the natural
answer by construction (conditional continuation) -- flagged in the caption.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figlib as F

N_BOOT = 2000


def build():
    nat = F.load_natural(with_traces=True)
    ev = nat[nat["natural_correct"].notna()].reset_index(drop=True)
    y = ev["natural_correct"].to_numpy(bool)

    cp_all = pd.read_parquet(F.MERGED / "checkpoint_results.parquet",
                             columns=["parent_raw_record_id", "requested_fraction",
                                      "k_keep", "answer_entropy_nats"])
    prefix = F.build_prefix_entropy(ev, cp_all[["parent_raw_record_id", "requested_fraction", "k_keep"]])

    ev_ids = set(ev["raw_record_id"])
    cp = cp_all[cp_all["parent_raw_record_id"].isin(ev_ids)]
    ans = (cp.pivot_table(index="parent_raw_record_id", columns="requested_fraction",
                          values="answer_entropy_nats", aggfunc="mean")
             .reindex(index=ev["raw_record_id"], columns=F.FRACTIONS).to_numpy(float))

    conf = F.load_recovered_confidence("checkpoint")
    conf = conf[conf["parent_raw_record_id"].isin(ev_ids)].copy()
    conf["cval"] = conf["confidence_value_final"].astype("Float64")
    confm = (conf.pivot_table(index="parent_raw_record_id", columns="requested_fraction",
                              values="cval", aggfunc="mean")
               .reindex(index=ev["raw_record_id"], columns=F.FRACTIONS).to_numpy(float))

    qid = ev["question_id"].to_numpy(); subj = ev["subject"].to_numpy()
    # Orient so higher -> predicts correct: negate entropies; confidence as-is.
    curves = {
        "prefix_reasoning_entropy": F.auroc_curve_bootstrap(-prefix, y, qid, subj, N_BOOT),
        "answer_choice_entropy":    F.auroc_curve_bootstrap(-ans,    y, qid, subj, N_BOOT),
        "verbalized_confidence":    F.auroc_curve_bootstrap(confm,   y, qid, subj, N_BOOT),
    }
    return curves


def main():
    F.set_style()
    F.OUT.mkdir(parents=True, exist_ok=True)
    curves = build()
    fx = np.array(F.FRACTIONS)

    styles = {
        "prefix_reasoning_entropy": ("#3B6EA5", "Prefix reasoning entropy"),
        "answer_choice_entropy":    ("#C56A2C", "Answer-choice entropy"),
        "verbalized_confidence":    ("#4C8C4A", "Verbalized confidence"),
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)

    for key, (pt, lo, hi, n) in curves.items():
        c, lab = styles[key]
        m = np.isfinite(pt)
        ax.plot(fx[m], pt[m], "-o", color=c, ms=4, label=lab)
        ax.fill_between(fx[m], lo[m], hi[m], color=c, alpha=0.16, lw=0)

    ax.axhline(0.5, color="0.5", ls="--", lw=1)
    ax.text(0.005, 0.5, "chance", fontsize=8, color="0.5", va="bottom", ha="left")
    ax.set_ylabel("AUROC for natural final-answer correctness")
    ax.set_xlabel("Reasoning progress (0 = pre-reasoning, 1 = full trajectory)")
    ax.set_ylim(0.45, 0.82)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks(fx)
    # Legend sits in the empty band below every curve (all AUROCs stay above ~0.51),
    # so it never covers a line or its confidence interval.
    ax.legend(loc="lower center", ncol=3, frameon=False, fontsize=9,
              columnspacing=1.2, handletextpad=0.5, borderaxespad=0.6)
    ax.set_title("Discrimination of Natural Final-Answer Correctness Across Reasoning",
                 fontsize=11.5)

    out = F.OUT / "fig2_discrimination.png"
    fig.savefig(out); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    payload = {k: {"fraction": fx.tolist(), "auroc": v[0].tolist(),
                   "lo": v[1].tolist(), "hi": v[2].tolist(), "n": v[3].tolist()}
               for k, v in curves.items()}
    (F.OUT / "fig2_values.json").write_text(json.dumps(payload, indent=2))
    print("wrote", out)
    for k, v in curves.items():
        print(f"{k:26s}", np.round(v[0], 4).tolist())


if __name__ == "__main__":
    main()

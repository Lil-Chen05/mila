"""Figure 1 (RQ1): temporal evolution of the three reliability signals.

Stacked, shared-x panels over normalized reasoning progress (pooled/marginal on
the evaluable natural cohort):
  1A reasoning-token entropy, per-decile of token progress (natural trajectory);
  1B checkpoint answer-choice entropy, 11 forced-answer probes;
  1C repaired verbalized confidence, 11 forced-answer probes.
95% subject-stratified question-cluster bootstrap bands. Coverage annotated.

Different clocks: 1A is measured ALONG the natural trajectory; 1B/1C are elicited
at forced-answer checkpoints. Fraction 0.0 is a pre-reasoning question baseline.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figlib as F


def build_matrices():
    nat = F.load_natural(with_traces=True)
    ev = nat[nat["natural_correct"].notna()].copy()          # evaluable cohort
    ev = ev.reset_index(drop=True)

    # 1A: per-run decile means of reasoning-token entropy.
    dec = np.full((len(ev), F.N_DECILES), np.nan)
    for i, row in enumerate(ev.itertuples(index=False)):
        sl = F._reasoning_slice(row.per_token_entropy_nats, row.reasoning_boundaries)
        if sl is not None:
            dec[i] = F.decile_means(sl)

    # Checkpoint-derived per-run x fraction matrices (answer entropy, confidence).
    cp = F.load_checkpoints()
    cp = cp[cp["parent_raw_record_id"].isin(set(ev["raw_record_id"]))]
    ans = (cp.pivot_table(index="parent_raw_record_id", columns="requested_fraction",
                          values="answer_entropy_nats", aggfunc="mean")
             .reindex(index=ev["raw_record_id"], columns=F.FRACTIONS))

    conf = F.load_recovered_confidence("checkpoint")
    conf = conf[conf["parent_raw_record_id"].isin(set(ev["raw_record_id"]))].copy()
    conf["cval"] = conf["confidence_value_final"].astype("Float64")
    confm = (conf.pivot_table(index="parent_raw_record_id", columns="requested_fraction",
                             values="cval", aggfunc="mean")
               .reindex(index=ev["raw_record_id"], columns=F.FRACTIONS))

    meta = ev[["raw_record_id", "question_id", "subject"]]
    return meta, dec, ans.to_numpy(dtype=float), confm.to_numpy(dtype=float)


def band(values, meta):
    return F.question_cluster_bootstrap(values, meta["question_id"].to_numpy(),
                                        meta["subject"].to_numpy(), F.nanmean_rows)


def coverage(mat):
    return np.mean(~np.isnan(mat), axis=0)


def main():
    F.set_style()
    F.OUT.mkdir(parents=True, exist_ok=True)
    meta, dec, ans, conf = build_matrices()
    n_runs = len(meta)

    dec_pt, dec_lo, dec_hi = band(dec, meta)
    ans_pt, ans_lo, ans_hi = band(ans, meta)
    con_pt, con_lo, con_hi = band(conf, meta)

    decile_x = (np.arange(F.N_DECILES) + 0.5) / F.N_DECILES
    fx = np.array(F.FRACTIONS)
    ans_cov = coverage(ans)
    con_cov = coverage(conf)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.0), sharex=True,
                             constrained_layout=True)

    # 1A reasoning-token entropy (natural trajectory).
    ax = axes[0]
    ax.plot(decile_x, dec_pt, "-o", color="#3B6EA5", ms=4)
    ax.fill_between(decile_x, dec_lo, dec_hi, color="#3B6EA5", alpha=0.18, lw=0)
    ax.set_ylabel(r"Reasoning-token" "\n" r"entropy (nats) $\downarrow$")
    ax.set_title("(A)  Mean reasoning-token entropy", loc="left")
    ax.set_ylim(0, 0.7)

    # 1B answer-choice entropy (forced probes; f=0 is the pre-reasoning baseline).
    ax = axes[1]
    ax.plot(fx, ans_pt, "-o", color="#C56A2C", ms=4)
    ax.fill_between(fx, ans_lo, ans_hi, color="#C56A2C", alpha=0.18, lw=0)
    ax.set_ylabel(r"Answer-choice" "\n" r"entropy (nats) $\downarrow$")
    ax.set_title("(B)  Mean answer-choice entropy", loc="left")
    ax.set_ylim(0, 0.7)

    # 1C verbalized confidence (forced probes).
    ax = axes[2]
    ax.plot(fx, con_pt, "-o", color="#4C8C4A", ms=4)
    ax.fill_between(fx, con_lo, con_hi, color="#4C8C4A", alpha=0.18, lw=0)
    ax.set_ylabel(r"Verbalized" "\n" r"confidence (%) $\uparrow$")
    ax.set_title("(C)  Mean verbalized confidence", loc="left")
    # Full 0-100 range so the panel shows that verbalized confidence stays high.
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    axes[2].set_xlabel("Normalized reasoning progress")
    axes[2].set_xlim(-0.03, 1.03)
    axes[2].set_xticks(fx)

    out = F.OUT / "fig1_signal_evolution.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    # Dump plotted values for number cross-check.
    payload = {
        "n_evaluable_runs": int(n_runs),
        "decile_reasoning_entropy": {"x": decile_x.tolist(), "mean": dec_pt.tolist(),
                                     "lo": dec_lo.tolist(), "hi": dec_hi.tolist()},
        "answer_entropy": {"fraction": fx.tolist(), "mean": ans_pt.tolist(),
                           "lo": ans_lo.tolist(), "hi": ans_hi.tolist(),
                           "coverage": ans_cov.tolist()},
        "verbalized_confidence": {"fraction": fx.tolist(), "mean": con_pt.tolist(),
                                  "lo": con_lo.tolist(), "hi": con_hi.tolist(),
                                  "coverage": con_cov.tolist()},
    }
    (F.OUT / "fig1_values.json").write_text(json.dumps(payload, indent=2))
    print("wrote", out)
    print("reasoning entropy decile means:", np.round(dec_pt, 3).tolist())
    print("answer entropy per fraction   :", np.round(ans_pt, 3).tolist())
    print("confidence per fraction       :", np.round(con_pt, 1).tolist())
    print("confidence coverage per frac  :", np.round(con_cov, 3).tolist())


if __name__ == "__main__":
    main()

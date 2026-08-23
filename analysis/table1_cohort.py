"""Table 1: final natural-run cohort and evaluability by subject.

Foundation for every AUROC: completion was 100% but natural correctness is
available for only 71% of runs, unevenly by subject, and only 84 questions carry
both outcomes (the within-question base cohort). Emits LaTeX (booktabs), CSV, and
a PNG preview. Login-safe: reads the fixed trajectory_features.csv only.
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

import figlib as F

TABLES = REPO_TABLES = F.REPO / "analysis/final-r5000/tables"


def build():
    tf = pd.read_csv(F.FIXED / "trajectory_features.csv",
                     usecols=["subject", "question_id", "natural_correct"])
    tf["subj"] = tf["subject"].map(F.SUBJECTS)

    def mixed(g):
        ev = g.dropna(subset=["natural_correct"])
        return int((ev.groupby("question_id")["natural_correct"].nunique() == 2).sum())

    rows = []
    for s in F.SUBJECT_ORDER + ["Total"]:
        g = tf if s == "Total" else tf[tf["subj"] == s]
        att = len(g); ev = int(g["natural_correct"].notna().sum())
        cor = int((g["natural_correct"] == True).sum())
        inc = int((g["natural_correct"] == False).sum())
        rows.append({"Subject": s,
                     "Evaluable": f"{ev:,} ({100 * ev / att:.1f})",
                     "CorrIncorr": f"{cor:,} / {inc:,}",
                     "Accuracy": round(100 * cor / ev, 1),
                     "Mixed": mixed(g)})
    return pd.DataFrame(rows)


def to_latex(t):
    head = (r"\begin{table*}[t]\centering""\n"
            r"\begin{tabular}{lrrr r}""\n\\toprule\n"
            r"Subject & Evaluable, $n$ (\%) & Correct / Incorrect & Acc.\ (\%) & "
            r"Mixed-outcome Q \\""\n\\midrule\n")
    body = []
    for _, r in t.iterrows():
        rule = r"\midrule " if r["Subject"] == "Total" else ""
        name = r"\textbf{Total}" if r["Subject"] == "Total" else r["Subject"]
        body.append(f"{rule}{name} & {r.Evaluable} & {r.CorrIncorr} & {r.Accuracy} & {r.Mixed} \\\\")
    tail = ("\n\\bottomrule\n\\end{tabular}\n"
            r"\caption{Final natural-run cohort and evaluability by subject. Each "
            r"subject contributed 100 questions $\times$ 10 stochastic trajectories "
            r"(1{,}000 runs; 5{,}000 total), and every run completed, but a natural "
            r"final answer could be parsed for only 3{,}550 (71.0\%), unevenly "
            r"across subjects. \emph{Acc.} is accuracy \emph{among evaluable runs} "
            r"only; because evaluability is subject- and length-selected, "
            r"per-subject accuracies are not directly comparable (e.g.\ Mathematics "
            r"is 98.3\% on a cohort missing 40\% of its runs). \emph{Mixed-outcome "
            r"Q} counts questions with at least one naturally correct and one "
            r"naturally incorrect evaluable run --- the within-question base cohort "
            r"for RQ2.}""\n"
            r"\label{tab:cohort}""\n\\end{table*}""\n")
    return head + "\n".join(body) + tail


def to_png(t):
    F.set_style()
    fig, ax = plt.subplots(figsize=(8.4, 2.4)); ax.axis("off")
    cols = ["Subject", "Evaluable", "CorrIncorr", "Accuracy", "Mixed"]
    disp = t[cols].copy()
    disp.columns = ["Subject", "Evaluable, n (%)", "Correct / Incorrect", "Acc. (%)",
                    "Mixed-outcome Q"]
    tbl = ax.table(cellText=disp.values, colLabels=disp.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
    for j in range(len(disp.columns)):
        tbl[0, j].set_text_props(weight="bold"); tbl[0, j].set_facecolor("#EDEDED")
        tbl[len(disp), j].set_text_props(weight="bold")
    ax.set_title("Table 1. Final natural-run cohort and evaluability by subject",
                 loc="left", fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(TABLES / "table1_cohort.png", dpi=200)
    plt.close(fig)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    t = build()
    t.to_csv(TABLES / "table1_cohort.csv", index=False)
    (TABLES / "table1_cohort.tex").write_text(to_latex(t))
    to_png(t)
    print(t.to_string(index=False))
    print("\nwrote", TABLES / "table1_cohort.tex", "(+ .csv, .png)")


if __name__ == "__main__":
    main()

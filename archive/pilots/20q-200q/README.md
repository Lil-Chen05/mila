# Historical 20q/200q MMLU proxy

This directory is a frozen snapshot of the exploratory experiment that led to
the final Part 1 design. It used one greedy trajectory per question and was
run first on 20 questions and then on 200 questions. Its results were useful
for developing the checkpoint probe, but they are not evidence for the final
report.

The directory preserves the original scripts, SLURM jobs, tests, notes,
tracked results, and analyses together. Their paths still describe the old
repository layout, so they are retained for inspection rather than supported
as runnable entry points. Do not load models or datasets on a login node.

The final study instead uses 500 fixed MMLU questions, ten stochastic natural
runs per question, and eleven greedy checkpoint probes per successful run.
See the repository root README and `docs/part1/` for the maintained study.

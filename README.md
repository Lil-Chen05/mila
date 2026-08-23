# Beyond Endpoint Confidence

This repository contains the completed COMP 400 study **Beyond Endpoint
Confidence: Characterizing Uncertainty Across Reasoning Trajectories**. It
examines how reasoning-token entropy, answer-choice entropy, and verbalized
confidence evolve during reasoning, and how their relationship with natural
final-answer correctness changes across and within questions.

Read the reviewed paper: **[report/main.pdf](report/main.pdf)**. The paper
source is [report/main.tex](report/main.tex), and the numerical audit is
[analysis/final-r5000/RESULTS_VALIDATION_SUMMARY.md](analysis/final-r5000/RESULTS_VALIDATION_SUMMARY.md).

## Study design and cohort

- Model: `HuggingFaceTB/SmolLM3-3B` in thinking mode.
- Data: 500 MMLU test questions—100 each from high-school mathematics,
  physics, chemistry, biology, and psychology—sampled without replacement with
  seed 42.
- Generation: 10 independent stochastic natural trajectories per question
  (5,000 total), followed by greedy forced-answer probes at 11 fractions from
  0.0 through 1.0 for each successful trajectory.
- Analysis cohort: 3,550 trajectories with an evaluable natural A–D answer
  (3,172 correct and 378 incorrect). The within-question analysis is restricted
  to 84 questions containing both correct and incorrect evaluable runs.

The 3,550-run cohort is 71% of all generated trajectories. Evaluability varies
substantially by subject and with trajectory length, so reported accuracy and
correctness analyses are conditional on evaluability rather than estimates for
the full 5,000-run cohort.

## Main findings

- The signals evolve differently: reasoning-token entropy rises early and then
  plateaus, answer-choice entropy falls as the model commits, and verbalized
  confidence remains high and compressed.
- Prefix reasoning entropy has its strongest discrimination point estimate
  relatively early, around 20–30% of the trajectory (AUROC about 0.77), but
  point-estimate ordering alone is not a pairwise significance result.
- The pooled correct–incorrect reasoning-entropy gap is about 0.13 nats, but it
  shrinks to a small and heterogeneous within-question difference of 0.0113
  nats (95% CI [0.0002, 0.0225]) across the 84 mixed-outcome questions. This
  suggests that benchmark-wide separation partly reflects differences between
  questions.
- The model becomes increasingly concentrated on an answer for both correct
  and incorrect trajectories. Increasing answer certainty therefore does not
  establish correctness.
- Endpoint probability and verbalized confidence are both overconfident on the
  common endpoint cohort. Calibration and discrimination should be interpreted
  as distinct properties.

These conclusions are limited to one 3B model, five high-school MMLU subjects,
multiple-choice probing, and the evaluable subset. See the paper for the full
methods, confidence intervals, and limitations.

## Result provenance

Results in `analysis/final-r5000/` use three explicit provenance labels:

- **fixed**: original immutable production-analysis exports;
- **repaired**: verbalized confidence recovered by a narrow deterministic
  parser repair for an attached end-of-turn sentinel; and
- **reconstructed intended analysis**: prefix reasoning entropy calculated to
  recover the intended checkpoint-level analysis while reproducing fixed
  endpoint anchors.

Do not combine these categories or describe repaired/reconstructed outputs as
unchanged fixed exports. The detailed rationale and checks are in the
[results validation summary](analysis/final-r5000/RESULTS_VALIDATION_SUMMARY.md).

## Repository layout

```text
report/                 final paper source, compiled PDF, bibliography, figures
analysis/final-r5000/   validated final tables, figures, diagnostics, audit
manifests/part1/        tracked immutable 500-question and study manifests
docs/part1/             protocol, schema, validation, runbook, and history
scripts/                maintained pipeline and analysis implementation
jobs/                   historical/maintained Mila SLURM entry points
tests/                  synthetic and pure-logic verification
configs/part1/          versioned execution and analysis templates
schemas/part1/          JSON Schema contracts
archive/                frozen pilots and warm-up experiments
```

Early 20-question and 200-question pilots are preserved under
[archive/](archive/README.md). They are historical context, not evidence for
the final study.

## Safe verification and paper build

The following commands are login-node safe because they do not load model
weights or a Hugging Face dataset:

```bash
uv run python scripts/validate_part1_manifests.py
uv run python scripts/part1_dry_run.py
uv run pytest -q -m "not part1_full_acceptance"
```

To rebuild the paper, with `latexmk`, pdfLaTeX, and BibTeX installed:

```bash
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Never add `--dataset-cache` to the manifest validator on a Mila login node;
that option loads a materialized Hugging Face dataset. Do not rerun production
generation or recovery jobs. See [AGENTS.md](AGENTS.md) and
[HANDOFF.md](HANDOFF.md) before operational work.

## Data availability, citation, and license

The immutable question/study manifests, final numerical outputs, publication
figures, paper source, and compiled paper are tracked here. Raw production
shards, merged production Parquet files, model weights, and materialized MMLU
data are not committed because of size, storage, and provenance constraints;
their immutable IDs, hashes, counts, and recovery history are documented in
[HANDOFF.md](HANDOFF.md) and
[docs/part1/OPERATIONS_HISTORY.md](docs/part1/OPERATIONS_HISTORY.md).

If you use this work, cite the report using [CITATION.cff](CITATION.cff). This
repository currently includes no software or data license; availability of the
source does not itself grant reuse rights.

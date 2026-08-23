# AGENTS.md — completed-project maintenance contract

## Current state

Part 1 is complete. Production generation, canonical merge, the immutable
`final-r5000` analysis, and the final paper have all been published and
validated. The paper is `report/main.pdf`; the current concise handoff is
`HANDOFF.md`; the full recovery chronology is
`docs/part1/OPERATIONS_HISTORY.md`.

There is no active production or recovery chain. Do not rerun generation,
validation, merge, recovery, or analysis jobs; resubmit historical SLURM jobs;
replace receipts; rewrite the preserved failed standard validation report; or
mutate published production artifacts. Historical scripts remain for
reproducibility and audit, not as a standing authorization to execute them.

## Scientific contract

The completed study asks whether uncertainty observed throughout a reasoning
trajectory predicts the correctness of the naturally generated final answer.
Its fixed design is:

- model: `HuggingFaceTB/SmolLM3-3B`, revision
  `a07cc9a04f16550a088caea529712d1d335b0ac1`, in thinking mode;
- data: 500 `cais/mmlu` test questions, 100 without replacement from each of
  `high_school_mathematics`, `high_school_physics`,
  `high_school_chemistry`, `high_school_biology`, and
  `high_school_psychology`, in that order;
- seeds: question sampling, base generation, and bootstrap seed 42, with
  per-run seeds from the documented canonical SHA-256 derivation;
- generation: exactly 10 stochastic natural runs per question (`run_id` 0–9),
  temperature 0.6, top-p 0.95, top-k 50, and at most 8192 new tokens;
- probing: 11 greedy forced-answer checkpoints at fractions 0.0 through 1.0;
- primary target: `natural_correct`; checkpoint-local correctness and endpoint
  agreement remain secondary measurements; and
- measurements: full-vocabulary reasoning-token entropy, renormalized A–D
  answer entropy, verbalized confidence, maximum A–D probability, switching,
  stabilization, and abnormal-output statuses.

The authoritative contracts are `docs/part1/DECISIONS.md`, `SCHEMA.md`,
`PLAN.md`, `VALIDATION.md`, and `RUNBOOK.md`. Do not silently change metrics,
parsing, cohorts, manifests, identities, hashes, retry semantics, or analysis
claims. A future extension requires explicit approval, a versioned contract,
new immutable provenance, and independent verification; it must not overwrite
Part 1.

## Result and provenance language

The completed cohort contains 5,000 generated trajectories, 3,550 evaluable
natural answers (3,172 correct and 378 incorrect), and 84 mixed-outcome
questions used for within-question analysis. Correctness results are
conditional on evaluability.

Preserve these labels whenever results are discussed:

- **fixed** — original immutable production-analysis exports;
- **repaired** — verbalized confidence recovered by the documented narrow,
  deterministic parser repair; and
- **reconstructed intended analysis** — prefix reasoning entropy recovered for
  the intended checkpoint-level analysis while reproducing fixed anchors.

Do not overstate descriptive point-estimate orderings, the small heterogeneous
within-question effect, calibration comparisons, or generalization beyond this
model and multiple-choice cohort. `report/main.tex` and
`analysis/final-r5000/RESULTS_VALIDATION_SUMMARY.md` are the interpretation
sources of record; do not change final scientific metrics during maintenance.

## Cluster safety (non-negotiable)

- Never load a model, tokenizer, or Hugging Face dataset on a Mila login node,
  including one-question, CPU, or exploratory loads. Use a compute-node SLURM
  job for any model or dataset access.
- Pure logic—schema checks, serialization, parsing, prompt formatting, and
  synthetic tests—may run on a login node if it imports no execution path that
  loads model weights or data.
- `$SLURM_TMPDIR` is ephemeral. Persistent manifests, results, logs, and
  receipts belong under the home directory or `$SCRATCH`.
- Use one model on one GPU, `model.eval()`, bfloat16 weights, and batch size 1.
  Do not batch questions, natural runs, checkpoints, or probes.
- CPU-only preparation and analysis jobs omit `--gpus-per-task`. GPU jobs use
  only the resources they require.

## Repository and environment conventions

- Use `uv`: add dependencies with `uv add <package>` and run Python entry points
  with `uv run`. Do not use pip or instruct users to activate `.venv` manually.
- Run from the repository root. Python entry points live in `scripts/`, SLURM
  wrappers in `jobs/`, shared contracts/helpers in `scripts/part1_*.py`, and
  synthetic/pure tests in `tests/`.
- On compute nodes, set `HF_HOME=$SCRATCH/hf_cache` before any Hugging Face load.
- Dataset acquisition must use streaming plus a bounded `take`, then
  `save_to_disk`; never full-download and then `.select()`. Jobs load the
  materialized cache rather than downloading again.
- The tracked manifest bundle under `manifests/part1/` is immutable. Generated
  operational manifests, raw shards, merged Parquet, caches, weights, and SLURM
  logs remain untracked in their documented persistent/ignored locations.

## Maintenance workflow

- Preserve unrelated and untracked user work. Avoid unrelated refactors and do
  not modify archived pilots as though they were current code.
- Use the smallest scoped change, update the relevant contract documentation,
  and run verification proportional to risk. Model/data tests belong in SLURM;
  login-node checks must remain pure/synthetic.
- Keep `.superpowers/` as local state excluded through `.git/info/exclude`; do
  not add it to the shared `.gitignore`.
- Do not commit large data, raw production shards, model weights, Arrow/Parquet
  artifacts, operational model-run manifests, or `slurm-*.out` logs.
- Before any future immutable publication, require a clean tracked worktree,
  reviewed provenance, and an explicit new authorization. Never reuse a Part 1
  receipt, model-run ID, or recovery path for new work.

When a request conflicts with cluster safety, immutable provenance, the final
paper, or the fixed scientific contract, stop and raise the conflict rather
than working around it.

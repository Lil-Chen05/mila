# AGENTS.md — Working Agreement for This Project

## PRODUCTION RECOVERY — READ HANDOFF FIRST

Full-shape acceptance job `10347033` timed out after 12 hours and its exact
`afterok` gate `10347034` was cancelled without launching production. The user
explicitly waived another full-shape rehearsal. Production must use the focused
readiness bootstrap documented in root `HANDOFF.md`; do not bypass its gate or
submit individual production jobs manually. Once the recovery chain is active,
do not commit, pull, or switch the Mila checkout away from its recorded commit.

## What this project is
This repository contains Part 1 of an uncertainty-trajectory research experiment
on the Mila SLURM cluster. It studies whether uncertainty measured throughout a
reasoning trajectory predicts the correctness of a reasoning model's naturally
generated final answer.

- Current model: `HuggingFaceTB/SmolLM3-3B` in thinking mode. Implement only
  SmolLM3 during the current four-prompt sequence; later models will use the
  same immutable study design through model-specific adapters. Do not implement
  later models or the later answerable-versus-unanswerable study now.
- Dataset: `cais/mmlu`, test split. The fixed question manifest contains exactly
  500 questions: 100 without replacement from each of
  `high_school_mathematics`, `high_school_physics`,
  `high_school_chemistry`, `high_school_biology`, and
  `high_school_psychology`, in that order, with question-sampling seed 42.
- Question sampling, base generation, and bootstrapping all use their specified
  seed value 42; per-run generation seeds use the documented canonical SHA-256
  derivation rather than Python's built-in `hash()`.
- Each model-question pair has exactly 10 stochastic natural runs (`run_id`
  0–9), not an extra greedy natural run. Natural generation uses `do_sample=True`,
  temperature 0.6, top-p 0.95, top-k 50, and at most 8192 new tokens.
- Each successful natural run is probed greedily at the 11 requested checkpoint
  fractions 0.0 through 1.0 in increments of 0.1. Abnormal model output remains
  data when execution succeeded; it is not automatically retried or excluded.
- The primary target is `natural_correct`. Checkpoint-local correctness and
  checkpoint-1.0 agreement with the natural answer are secondary measurements;
  disagreement is valid and must not invalidate or retry a record.
- Measurements include raw full-vocabulary entropy over natural reasoning
  tokens, renormalized A–D checkpoint entropy, verbalized confidence, maximum
  A–D probability, switching and stabilization, and abnormal-output statuses.

The detailed scientific and engineering contracts belong in `docs/part1/`,
especially `PLAN.md`, `DECISIONS.md`, `STATUS.md`, `SCHEMA.md`, `RUNBOOK.md`,
and `VALIDATION.md`. They define metrics, parsing, manifests, canonical
identities, calibration, bootstrapping, retry policy, persistence, and
validation. Do not silently change those contracts or add primary analysis
features without explicit approval.

## HOW YOU MUST WORK
- Follow the current four-prompt sequence: Prompt 1 establishes the repository
  audit, documentation, plan, and working contract; Prompt 2 implements Phase 1;
  Prompt 3 implements Phase 2; Prompt 4 implements Phase 3.
- Each phase prompt authorizes all work explicitly listed in that prompt. Do not
  request approval for every command or small edit. Stay within the current
  phase, stop at its completion, report results, and wait for the next prompt.
- Use separate repository-analysis, implementation, verification, and
  documentation roles where available. Only the implementation role edits
  production code; verification remains independent; documentation follows
  verified implementation. The root agent coordinates shared-file changes.
- Avoid unrelated refactors. Treat pre-existing unrelated and untracked files as
  user-owned: do not modify, delete, ignore, or commit them without direction.
- The final bounded Phase 3 smoke has completed, but its hardened CLI-validator
  acceptance is still pending. Do not create the production model-run manifest,
  claim production readiness, or launch production before every remaining gate
  in `docs/part1/RUNBOOK.md` passes.

## THREE IMPLEMENTATION PHASES
1. **Phase 1 — foundations:** schemas, canonical serialization and identities,
   persistence, locking, failure handling, and resumability. Do not create an
   immutable production model-run manifest in this phase.
2. **Phase 2 — generation:** the tracked fixed question manifest and study
   manifest, SmolLM3 adapter, deterministic per-run seeding, stochastic natural
   generation, and greedy checkpoint probes. Smoke manifests and outputs remain
   separate from production artifacts.
3. **Phase 3 — completion:** analysis, validation, merging, production
   model-run-manifest lifecycle, SLURM readiness, documentation, and final
   bounded smoke tests. Create the operational production model-run manifest
   only after the final production commit from a clean tracked worktree.

## CURRENT PRODUCTION AUTHORIZATION
- On 2026-08-11, the user explicitly authorized the full production run after
  readiness is established. This authorization does not waive any validation,
  review, clean-tree, provenance, or manifest gate.
- The approved production array target is `0-499%16`, with a four-day deadline.
  The scientific protocol is unchanged: 500 fixed questions, 10 stochastic
  natural runs per question, and 11 requested checkpoints per successful
  natural run.
- Authorization is not evidence of readiness or launch. Until the hardened
  CLI validator, final verification/review/commit, production-manifest, and
  launch-readiness gates pass in order, the production manifest and array must
  remain absent.
- The unattended path queues only the CPU focused-readiness suite with
  `acceptance_mode=focused_readiness_v1` and its exact `afterok` production
  gate initially. The gate must validate the current-format Phase 3 smoke and
  the clean final commit before it may create the production manifest or submit
  the `%16` array. Smoke A/B remain immutable historical acceptance evidence.
  Partial submission receipts block automatic resubmission.

Use two immutable provenance levels: a tracked, model-independent study
manifest and one operational model-run manifest per model revision and adapter.
Tracked question manifests must live outside ignored data directories. Generated
model-run manifests and raw outputs belong under an explicitly ignored results
directory. Stable hashes must use documented canonical payloads and exclude
self-referential or mutable operational fields.

## CLUSTER SAFETY RULES (non-negotiable)
- NEVER load a model or dataset on a login node—not even one question, on CPU,
  or as a quick test. All model and dataset loading happens inside a SLURM job
  on a compute node.
- Pure logic such as string parsing, prompt formatting, serialization, and
  helper functions with no model or dataset loading may be tested on the login
  node. Anything that loads weights or a dataset goes in a job.
- Compute nodes are ephemeral; `$SLURM_TMPDIR` is wiped when a job ends.
  Anything that must persist goes in the home directory or `$SCRATCH`.
- Request GPUs only when needed. Data preparation and analysis jobs are CPU-only
  and omit `--gpus-per-task`.
- Use one model on one GPU, `model.eval()`, bfloat16 weights, and batch size 1.
  Do not batch questions, natural runs, checkpoints, or checkpoint probes.

## ENVIRONMENT CONVENTIONS
- The dependency manager is `uv`. Use `uv add <pkg>` to add dependencies; never
  use pip and never suggest activating `.venv` manually.
- Run commands from the repository root. Python entry points live at
  `scripts/<task>.py`, SLURM launchers at `jobs/<task>.sh`, shared helpers at
  `scripts/mc_common.py`, and pure unit tests under `tests/`.
- Run scripts with `uv run python scripts/<task>.py`; in jobs use
  `srun uv run python scripts/<task>.py`.
- Cache model downloads in scratch, not home. Job scripts must set
  `export HF_HOME=$SCRATCH/hf_cache` before any Hugging Face load.

## DATA AND STORAGE
- Fetch datasets with streaming plus a bounded `take`, then use `save_to_disk`
  for persistent materialization. Never full-download a dataset and then call
  `.select()`.
- Jobs load materialized data from disk rather than downloading it again.
- The immutable 500-question manifest must be tracked outside the ignored
  `data/` directory and reused unchanged by every eventual model.
- Do not rely on `$SLURM_TMPDIR` for manifests, results, logs, or any other
  artifact that must survive the job.

## GIT CONVENTIONS
- `slurm-*.out` files are ignored on purpose. Do not add or commit them.
- Commit code, configuration, tracked manifests, documentation, and small result
  files. Do not commit downloaded datasets, model weights, large Arrow/Parquet
  artifacts, raw production shards, or operational model-run manifests.
- Keep `.superpowers/` as local tool state excluded through `.git/info/exclude`;
  never edit its contents or add it to the shared `.gitignore`.
- Use clear, scoped commit messages. Before production generation, require a
  clean tracked worktree except for intentionally local exclusions, and record
  the final production Git commit in the model-run provenance.

## SBATCH BASELINE
Tune resources per task; CPU-only jobs omit the GPU line:

```bash
#!/bin/bash
#SBATCH --job-name=<task>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1   # GPU steps only
#SBATCH --mem=16G
#SBATCH --time=1:00:00
export HF_HOME=$SCRATCH/hf_cache
srun uv run python scripts/<task>.py
```

## WHEN UNSURE
- If a product, model, library, or cluster detail may be version-specific, verify
  it against the pinned implementation or authoritative documentation rather
  than guessing.
- If a request appears to conflict with cluster safety, provenance, the current
  phase boundary, or the documented scientific contract, stop and raise the
  conflict instead of working around it.

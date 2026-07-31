# Part 1 runbook

## Status and safety boundary

This is the lifecycle contract established in Prompt 1. Commands whose entry
points do not yet exist are labelled **planned** and must not be run until their
implementation phase passes [VALIDATION.md](VALIDATION.md). Current executable
20q/200q commands are historical and do not implement this runbook.

Never load a model, tokenizer, or Hugging Face dataset on a login node—not for a
single row, CPU test, or preflight. Dataset materialization runs in a CPU SLURM
job. Model/tokenizer preflight and generation run in GPU SLURM jobs. Pure logic,
small local JSON validation, merge planning, and tests that import no model or
dataset are login-safe.

This four-prompt sequence never launches the full 500-question experiment. It
permits only bounded smoke jobs explicitly authorized by Prompt 3 or Prompt 4.
This runbook intentionally provides no production submission command; a later
explicit production authorization is required.

Scientific settings are fixed in [DECISIONS.md](DECISIONS.md), record meanings
in [SCHEMA.md](SCHEMA.md), phase gates in [PLAN.md](PLAN.md), and current state in
[STATUS.md](STATUS.md).

## Persistent storage and environment

- Run repository commands from the repository root.
- Use `uv`; never use `pip` or manually activate `.venv`.
- Python entry points use `uv run python scripts/<task>.py`; SLURM jobs use
  `srun uv run python scripts/<task>.py`.
- Every job that can load Hugging Face artifacts exports
  `HF_HOME=$SCRATCH/hf_cache` before the load.
- `$SLURM_TMPDIR` is ephemeral. Manifests, journals, raw records, audit events,
  validation reports, merged data, logs needed for audit, and analysis outputs
  must live in home or `$SCRATCH` at an explicitly configured persistent root.
- GPU generation uses one model on one GPU, `model.eval()`, bfloat16 weights,
  and batch size one. It never batches questions, natural runs, checkpoints, or
  probes.
- CPU-only materialization, validation, merge, and analysis jobs omit
  `--gpus-per-task`.

The exact persistent Part 1 root, Mila-compatible lock mechanism, and stale-lock
policy are Phase 1 decisions. No operational command may default to an
ephemeral or ambiguous path. Phase 1 acceptance requires the selected path and
policy to be explicit in configuration and tested.

## Artifact classes

| Class | Planned location | Git policy |
|---|---|---|
| Question JSONL and sidecar | `manifests/part1/` | tracked and immutable after Phase 2 commit |
| Study manifest | `manifests/part1/study_manifest.json` | tracked and immutable after Phase 2 commit |
| Smoke model-run manifests/output | separate persistent smoke root under `results/part1-smoke/` or an equally explicit configured root | generated and ignored; never mixed with production |
| Production model-run manifest | `results/part1/<model_run_id>/model_run_manifest.json` | generated after final production commit and ignored |
| Raw shards/events/journals | under the matching smoke or production model-run root | generated, persistent, ignored |
| Merged/validated/analysis outputs | under the matching model-run root, with small approved summaries optionally tracked later | never overwrite valid artifacts before validation |
| MMLU materialization/cache | configured persistent `data/` and `$SCRATCH/hf_cache` | ignored; not the tracked scientific manifest |

New ignore rules must be narrow enough to leave tracked question/study
manifests visible and to coexist with tracked historical 20q/200q results.

## Lifecycle overview

1. Lock Phase 1 contracts and storage.
2. Materialize and verify the fixed questions in a CPU compute job.
3. Commit the question and study manifests.
4. Preflight the pinned SmolLM3 model/tokenizer and adapter on a GPU compute
   node.
5. Run only authorized, isolated bounded smokes; exercise failure and resume.
6. Complete validators, atomic merge, analysis, SLURM launchers, and docs.
7. Commit final tracked production artifacts and require a clean tracked
   worktree.
8. Generate the immutable operational production model-run manifest under the
   ignored persistent results root and confirm it does not dirty the worktree.
9. Stop. Do not submit the full experiment without later explicit approval.

## 1. Phase 1 storage/configuration preflight

Before Phase 2, the operator must be able to run login-safe pure tests:

```bash
uv run pytest -q
```

The Phase 1 implementation must also provide a pure configuration validation
that resolves the persistent root without loading model or dataset. The planned
interface is:

```bash
uv run python scripts/validate_part1_config.py --mode smoke
```

**Planned; not implemented in Prompt 1.** If Phase 1 selects another focused
entry point, update this runbook after verification. It must print the resolved
persistent root, schema/identity versions, lock policy, retry policy, and
whether the selected mode is smoke or production. It must fail closed on an
ephemeral/unset root.

## 2. Question materialization

Question materialization loads `cais/mmlu`; therefore it always runs through a
CPU SLURM job:

```bash
sbatch jobs/prepare_part1_questions.sh
```

**Planned for Phase 2; not currently implemented.** The job must:

1. export `HF_HOME=$SCRATCH/hf_cache` before dataset access;
2. use streaming and bounded selection rather than full download followed by
   `.select()`;
3. select exactly 100 without replacement for each fixed subject with seed 42;
4. preserve the specified subject and seeded-selection order;
5. save any reusable materialization to persistent `data/`, never
   `$SLURM_TMPDIR`;
6. export the small tracked JSONL/sidecar outside `data/`; and
7. validate counts, source revision, order, uniqueness, record hashes, and
   complete manifest hash before the files are accepted.

The generated tracked manifest must be reviewed and committed before any smoke
model-run manifest can reference it. Rerunning materialization against the same
immutable revision must reproduce the exact bytes and hash or fail the gate.

## 3. Study manifest creation

The planned Phase 2 pure command is:

```bash
uv run python scripts/create_study_manifest.py
```

It may read only the small tracked question manifest and local configuration;
it must not import/load a model, tokenizer, or dataset. It writes
`manifests/part1/study_manifest.json`, validates every fixed decision, and
recomputes its IDs/hashes. Commit it with the question manifest after review.

If the implemented filename differs, update this planned command only after its
tests pass. Never edit an immutable study manifest in place; an approved
scientific change creates a new version and identity.

## 4. SmolLM3 compute-node preflight

Preflight loads model/tokenizer artifacts and therefore requires a single-GPU
compute job:

```bash
sbatch jobs/preflight_smol_lm3.sh
```

**Planned for Phase 2; not currently implemented.** The job resolves and emits a
small preflight report containing:

- immutable model and tokenizer revisions;
- environment versions and bfloat16/evaluation mode;
- resolved requested/effective natural and checkpoint settings;
- thinking opening/closing tag text and token IDs;
- prompt rendering/version evidence;
- `</think>\nAnswer:` inducer text and exact token sequence;
- A–D candidate encodings, selected convention, and answer-token-location test;
- a small logits/entropy sanity check using only the Prompt-3-authorized bounded
  input, if that prompt authorizes it.

Preflight failure blocks smoke generation. Do not fall back to a mutable model
revision, guessed token ID, or login-node inspection.

## 5. Smoke manifest and bounded smoke

Create a non-production smoke model-run manifest in the separate ignored smoke
root. It references the committed question/study manifests and verified
preflight facts but covers only the explicitly authorized small smoke subset.

If tracked code is dirty, record:

```bash
git rev-parse HEAD
git diff --binary --no-ext-diff
```

The implementation stores the base commit and a documented SHA-256 hash of the
diff bytes, and marks the smoke `production = false`. The diff itself must not
be inferred from an undocumented later worktree state.

The planned submission is:

```bash
sbatch jobs/smoke_part1.sh
```

**Planned and allowed only when Prompt 3 or Prompt 4 explicitly authorizes that
bounded smoke.** Its manifest and output root must be visibly different from
production. The smoke must exercise at least one complete run, all eleven
checkpoint identities, alias behavior using a short/zero synthetic or bounded
case when feasible, abnormal-output retention, failure journaling, and resume.
It must never expand implicitly to all 500 questions or ten runs for all
questions.

## 6. Submission and shard ownership

Before any later production submission, verify all of the following:

- final tracked generation code, manifests, tests, launchers, and docs are
  committed;
- `git status --porcelain` is empty for tracked/unignored state;
- model-run manifest hash, study/question hashes, model/tokenizer revisions,
  final Git commit, persistent paths, and environment versions validate;
- every planned logical `(question_id, run_id)` has exactly one shard owner;
- each shard is bound to exactly one model-run manifest;
- retry/lock configuration is explicit;
- storage capacity and SLURM resources have been checked without loading the
  model on the login node.

This runbook does not authorize or show the full production `sbatch` command.
During the four prompts, stop after bounded smoke validation.

## 7. Resume and retry

Resume first acquires the Phase 1 lock at the configured persistent root, reads
durable events/terminal records, validates their identities/hashes, and computes
missing logical work. It never infers success from stdout alone.

The planned smoke resume interface is:

```bash
sbatch jobs/smoke_part1.sh --export=ALL,RESUME=1
```

**Planned; exact launcher arguments are locked during Phase 2/3.** Required
semantics are:

- completed natural runs are not regenerated, including capped, unclosed,
  short, malformed, repetitive, or natural/checkpoint-disagreement records;
- completed checkpoint identities are not reprobed; missing eligible
  checkpoint work may resume using the same natural-chain record;
- only failures classified retryable by the finite Phase 1 policy are retried;
- every retry uses the same generation seed and a new attempt ID;
- retry exhaustion writes a terminal infrastructure failure;
- duplicate workers cannot publish two terminal records for one logical
  identity; and
- stale locks are recovered only by the documented verified policy, never by
  deleting lock files speculatively.

## 8. Validation

Validate raw shards before merge. The planned CPU-only job is:

```bash
sbatch jobs/validate_part1.sh
```

It must not load a model or dataset. It verifies schema/nullability,
provenance/hash compatibility, record/event uniqueness, terminal coverage,
token/entropy alignment, eleven checkpoint identities and aliases, seed
derivation, measurement ranges, and shard ownership. It reports successful
abnormal model behavior as data rather than an infrastructure failure.

Validation writes a new report and never modifies raw records or immutable
manifests. A failed report blocks merge.

## 9. Merge

The planned CPU-only job is:

```bash
sbatch jobs/merge_part1.sh
```

It validates every input completely before publication, writes to a temporary
artifact in the same persistent filesystem, fsyncs as required by the Phase 1
contract, and atomically renames only after completeness/hash checks pass. A
missing shard, question/run/checkpoint identity, duplicate, mixed manifest, or
invalid terminal record is fatal. Failure must leave any previously valid
merged artifact unchanged and must not leave an incomplete file at the final
path.

## 10. Analysis

The planned CPU-only job is:

```bash
sbatch jobs/analyze_part1.sh
```

Analysis runs only from a validated merged artifact and immutable manifests. It
uses the fixed registry/targets, per-fraction calibration, subject-stratified
question bootstrap with multiplicity, within-question paired differences, and
alias-aware switching/stabilization. Development uses 1,000 replicates; final
results use 5,000. Machine-readable outputs retain all eleven fractions and
bootstrap validity counts. Main summaries show fractions 0.0, 0.5, and 1.0.

Historical `analysis/analyze_checkpoints.py` and `analysis/analyze_200q.py` must
not be run on Part 1 production data.

## 11. Final production-manifest ordering

After Phase 3 tests and the authorized final bounded smoke pass:

1. Commit final generation code, validation/merge/analysis code, launchers,
   tracked question/study manifests, and documentation.
2. Verify the tracked worktree is clean except for intentional local exclusions:

   ```bash
   git status --porcelain
   git rev-parse HEAD
   ```

3. Use the recorded commit plus verified Phase 2 preflight facts to generate the
   operational production model-run manifest under the ignored persistent
   results root. The planned pure command is:

   ```bash
   uv run python scripts/create_model_run_manifest.py --mode production
   ```

4. Re-run `git status --porcelain`; creating the operational manifest must not
   dirty the tracked worktree.
5. Validate the manifest and output root, then stop. Do not launch full
   production generation during this sequence.

Generating the manifest before the final commit, recording a dirty commit, or
creating it in a tracked path invalidates the production lifecycle and requires
a fresh model-run manifest identity after the gate is satisfied.

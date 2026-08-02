# Part 1 Phase 2 Local Preparation Implementation Plan

> Execute this plan in the existing clean dedicated Codex checkout. Preserve
> the user's one-commit requirement; do not submit jobs or access Mila.

**Goal:** Deliver one reviewed commit containing all locally implementable
Phase 2 code, tests, jobs, and paused-state documentation, with CPU dataset
materialization as the next required action.

**Architecture:** A login-safe manifest/science layer is separated from thin
CPU/GPU entry points that lazily import datasets, transformers, and torch.
Finalized manifest content is validated in staging and existing destinations
are preflighted before any atomic rename. SmolLM3-specific behavior stays in an
adapter, while general planning/parsing/metric helpers remain model-neutral.

**Tooling:** Python 3.12, `uv`, pytest, JSON Schema Draft 2020-12, Hugging Face
Datasets/Hub/Transformers (execution jobs only), PyTorch (GPU execution only),
and SLURM.

---

## Task 1: Migrate the dataset and manifest contracts

**Files:**

- Modify: `configs/part1/dataset_materialization.json`
- Modify: `schemas/part1/question_manifest.schema.json`
- Modify: `schemas/part1/study_manifest.schema.json`
- Modify: `scripts/part1_contract.py`
- Modify: `tests/test_part1_contract.py`

Write failing tests for the explicit per-subject config strategy and dataset
revision fields in study identity, then update schemas, fixed oracles, and hash
field registries. Run the focused contract tests red then green.

## Task 2: Build the login-safe manifest bundle

**Files:**

- Create: `scripts/part1_manifests.py`
- Create: `tests/test_part1_manifests.py`

Write failing tests for row normalization, 500-row order/count/uniqueness,
record hashes, question/study construction, serialization/reload, existing-file
compatibility, failure-before-publication, identical reruns, and divergent
targets. Implement the smallest login-safe APIs needed by the CPU job and
validator. Run focused tests after each red–green increment.

## Task 3: Add the CPU-only materialization and validation entry points

**Files:**

- Create: `scripts/materialize_part1_mmlu.py`
- Create: `scripts/validate_part1_manifests.py`
- Create: `jobs/materialize_part1_mmlu.sh`
- Modify: `.gitignore`
- Create/modify: focused CLI tests under `tests/`

Test dependency injection with fake Hub/dataset functions before implementing
the lazy real imports. Stage the cache and manifests, validate the bounded
saved dataset, preflight all targets, and publish atomically. Ensure the job has
no GPU request, exports `HF_HOME=$SCRATCH/hf_cache`, writes its log under
`logs/`, and contains no Git command.

## Task 4: Implement locally testable adapter and science helpers

**Files:**

- Create: `scripts/part1_smollm3_adapter.py`
- Create: `scripts/part1_generation.py`
- Create: `scripts/part1_checkpoints.py`
- Create: `scripts/part1_storage_estimate.py`
- Create: `tests/test_part1_smollm3_adapter.py`
- Create: `tests/test_part1_generation.py`
- Create: `tests/test_part1_checkpoints.py`
- Create: `tests/test_part1_storage_estimate.py`

Use fake tokenizers/token sequences only after each desired pure behavior has a
failing test. Cover exact terminal block pairing, missing-close/no-reasoning,
out-of-range confidence, token/entropy alignment, final-10% entropy, ten unique
seeds, ties-even placements, short-chain aliases, A–D probability/entropy math,
and storage safeguards. Lazy GPU functions must fail clearly without the
required execution context and must not auto-map PAD to EOS.

## Task 5: Prepare unsubmitted GPU preflight and smoke execution

**Files:**

- Create: `scripts/part1_smollm3_preflight.py`
- Create: `scripts/run_part1_smoke.py`
- Create: `jobs/part1_smollm3_preflight.sh`
- Create: `jobs/part1_smoke_a.sh`
- Create: `jobs/part1_smoke_b.sh`
- Create: `tests/fixtures/part1_synthetic/README.md`
- Create: synthetic fixture JSON/JSONL files and validation tests

Test CLI argument validation and smoke selection/planning without importing a
real model. Prepare fail-closed job scripts for one-GPU, batch-one execution.
Keep Smoke A and B separate from production and do not create a production
model-run manifest.

## Task 6: Update the six Part 1 documents

**Files:**

- Modify: `docs/part1/DECISIONS.md`
- Modify: `docs/part1/PLAN.md`
- Modify: `docs/part1/RUNBOOK.md`
- Modify: `docs/part1/SCHEMA.md`
- Modify: `docs/part1/STATUS.md`
- Modify: `docs/part1/VALIDATION.md`

Record the prepared/paused state, exact CPU command, expected paths and logs,
post-job validation, publication/recovery semantics, evidence labels, prepared
GPU commands, and every remaining Mila/GPU gate. Leave hashes and results as
pending placeholders, not fabricated values.

## Task 7: Verify, review, and commit

Run:

```bash
UV_CACHE_DIR=/private/tmp/mila-uv-cache uv run pytest -q --tb=short
bash -n jobs/*.sh
uv run python scripts/validate_part1_manifests.py --help
uv run python scripts/materialize_part1_mmlu.py --help
git diff --check
git status --short
```

Inspect the complete diff, obtain an independent requirements/code review,
address all Critical/Important findings, rerun the full verification suite, and
create exactly one scoped local preparation commit. Do not add generated Mila
artifacts and do not push.

# Part 1 Phase 3 Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish Part 1 production provenance, resumable SLURM orchestration,
coverage validation, atomic merge, and the fixed manifest-driven analyses,
then create the ignored production model-run manifest without launching the
500-question experiment.

**Architecture:** Keep the Phase 2 SmolLM3 generation and checkpoint kernels
unchanged and add a manifest-bound production shard orchestrator around those
proven functions. Treat raw shards as append-only authoritative inputs;
coverage validation creates the compatibility gate, merge atomically publishes
reproducible Parquet derivatives with embedded provenance, and one CPU-only
analysis entry point produces machine-readable tables, plots, and sidecars.
Production identity is created only after the final tracked commit and a clean
tracked-worktree check.

**Tech Stack:** Python 3.12, `uv`, pytest, JSON Schema Draft 2020-12, NumPy,
pandas, PyArrow (already locked through `datasets`), Matplotlib, POSIX file
locking, Git, and Mila SLURM.

## Global Constraints

- Do not launch the complete 500-question experiment.
- Do not alter the fixed model, prompts, sampling settings, checkpoint
  fractions, targets, calibration pairs, bootstrap rules, or feature registry.
- Never load a model, tokenizer, or dataset on a login node.
- One L40S, one model, one process, batch size one; do not batch questions,
  natural runs, or checkpoint probes.
- Reuse Phase 2 Smoke A/B as real-model evidence. Because the new production
  orchestrator is generation-path code, run exactly one new bounded Phase 3
  smoke through that orchestrator in a new non-production root; never exceed
  one question by ten runs.
- All new feature and defect work follows red-green-refactor. Synthetic records
  prove control flow and statistics only, never real SmolLM3 behavior.
- Preserve append-only raw streams, exclusive locking, terminal-result
  authority, three-attempt retry policy, checkpoint-only resume, and successful
  abnormal-output retention.
- `.superpowers/` stays excluded through `.git/info/exclude` and uncommitted.
- Do not modify or commit unrelated user-owned files.
- The final production model-run manifest is ignored operational state. Create
  it only after tracked work is committed and the tracked worktree is clean.
- Prefer exact JSON integers/booleans and canonical SHA-256 identities; Python
  `hash()` is forbidden.

---

### Task 1: Production model-run provenance and clean-worktree gate

**Files:**

- Modify: `schemas/part1/model_run_manifest.schema.json`
- Modify: `scripts/part1_contract.py`
- Modify: `scripts/part1_model_run.py`
- Create: `scripts/create_part1_model_run_manifest.py`
- Modify: `tests/test_part1_contract.py`
- Modify: `tests/test_part1_model_run.py`
- Create: `tests/test_create_part1_model_run_manifest.py`

**Interfaces:**

- Consumes: validated tracked question/study manifests, Phase 2
  `preflight.json`, `uv.lock`, and `git rev-parse HEAD`.
- Produces:
  `build_production_model_run_manifest(*, study_manifest, preflight_report,
  final_git_commit, output_root) -> dict[str, Any]` and
  `publish_production_model_run_manifest(...) -> Path`.
- Production-only fields are explicit BOS/EOS/PAD IDs, model context window,
  dependency lock SHA-256, clean tracked-worktree confirmation, and relative
  output paths for shards/validation/merge/analysis. Output paths are excluded
  from both stable identity payloads, as required by the existing provenance
  contract; the remaining production fields participate in the complete hash.
  Keep Phase 2 schema `1.0.0` smoke manifests valid with unchanged IDs/hashes;
  emit the richer production schema `1.1.0`, whose model-run identity also
  selects the explicit special-token IDs and context window.

- [ ] **Step 1: Write failing production-manifest tests.**

  Add tests that construct the existing complete preflight fixture and assert:

  ```python
  manifest = build_production_model_run_manifest(
      study_manifest=study(),
      preflight_report=preflight(),
      final_git_commit="1" * 40,
      output_root=Path("results/part1"),
  )
  assert manifest["production"] is True
  assert manifest["execution_scope"] == "production"
  assert manifest["bos_token_id"] is None
  assert manifest["eos_token_id"] == 2
  assert manifest["pad_token_id"] is None
  assert manifest["model_context_window"] == 65536
  assert manifest["dependency_lock_sha256"] == "e" * 64
  assert manifest["clean_tracked_worktree"] is True
  assert manifest["output_paths"]["raw_shards"].startswith(
      f"results/part1/{manifest['model_run_id']}/"
  )
  ```

  Also assert production rejects missing/invalid fields, a dirty tracked
  worktree, a non-HEAD commit, a preflight/lock mismatch, symlinked targets,
  divergent existing bytes, and output-path mutation changing either stable
  identity. Add `phase3_smoke` to the non-production scope enum and prove old
  Smoke A/B manifests still recompute unchanged.

- [ ] **Step 2: Run the focused tests and confirm RED.**

  Run:

  ```bash
  UV_CACHE_DIR=/private/tmp/mila-uv-cache uv run pytest -q \
    tests/test_part1_contract.py tests/test_part1_model_run.py \
    tests/test_create_part1_model_run_manifest.py --tb=short
  ```

  Expected: failures for the missing production builder/CLI and required
  production fields, while all pre-existing cases remain green.

- [ ] **Step 3: Implement the minimal production builder and atomic publisher.**

  Extend model-run hashing with a production-only selected field tuple. Build
  the stable identity first, derive the model-run ID, then populate relative
  output paths containing that ID. The CLI must use read-only Git commands,
  validate `uv.lock`, write a same-directory fsynced temporary file, and publish
  with identical-only semantics. It must recheck tracked cleanliness and HEAD
  immediately before publication and must not invoke `git add` or `git commit`.
  Execution-relevant untracked files under `scripts/`, `jobs/`, `configs/`,
  `schemas/`, or `manifests/` block production creation; unrelated untracked
  files are reported and left untouched.

- [ ] **Step 4: Run focused tests GREEN and commit.**

  Run the command from Step 2 and `git diff --check`, then commit only this
  slice as `feat(part1): add production model-run provenance`.

---

### Task 2: Production shard orchestrator and SLURM launch plan

**Files:**

- Create: `scripts/run_part1_shard.py`
- Create: `scripts/part1_launch_plan.py`
- Create: `jobs/part1_generate_array.sh`
- Create: `jobs/part1_phase3_smoke.sh`
- Modify: `jobs/part1_smoke_a.sh`
- Modify: `jobs/part1_smoke_b.sh`
- Modify: `jobs/part1_smollm3_preflight.sh`
- Modify: `jobs/part1_reproducibility.sh`
- Create: `jobs/part1_validate.sh`
- Create: `jobs/part1_merge.sh`
- Create: `jobs/part1_analyze.sh`
- Create: `tests/test_run_part1_shard.py`
- Create: `tests/test_part1_launch_plan.py`

**Interfaces:**

- Consumes: one compatible model-run manifest, the fixed question bundle,
  preflight artifact, `SLURM_ARRAY_TASK_ID`, and existing Phase 2 execution and
  lifecycle functions from `run_part1_smoke.py`.
- Produces:
  `select_shard_work(records, shard_index, shard_count)`,
  `run_part1_shard(...)`, and `build_launch_plan(...)`.
- The fixed readiness plan is 500 shards, one question (ten natural runs) per
  shard, array `0-499%1`, one L40S per task, and a 12-hour wall time. This is
  based on Smoke A's exact same one-question/ten-run workload completing in
  01:26:28; 12 hours preserves the already proven job envelope and gives about
  eightfold headroom for question-dependent trajectory length. Smoke B's five
  one-run workload completed in 00:25:14 and supports the same order of
  magnitude. Verify Mila `MaxArraySize >= 500`; otherwise stop and revise the
  plan before changing the shard count.

- [ ] **Step 1: Write failing sharding, resume, finalization, and shell tests.**

  Tests must prove every `(question_id, run_id)` appears exactly once across
  500 shards, each natural uses 11 checkpoint IDs, finalized compatible shards
  return success without lock acquisition, incomplete shards resume only
  missing work, active locks fail closed, terminal natural infrastructure
  failures make their checkpoints ineligible, and a full array resubmission is
  safe. Patch only the model-loading boundary for orchestration tests; assert on
  durable real `Part1ShardStore` records rather than mock call counts.

  Shell assertions must require:

  ```text
  #SBATCH --gpus-per-task=l40s:1
  #SBATCH --time=12:00:00
  export HF_HOME="$SCRATCH/hf_cache"
  --array=0-499%1 (in the documented submission command)
  ```

  Each GPU and CPU job resolves `uv` from `command -v uv`, then the executable
  `$HOME/.local/bin/uv`, and fails clearly if neither exists.

- [ ] **Step 2: Run focused tests and confirm RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/mila-uv-cache uv run pytest -q \
    tests/test_run_part1_shard.py tests/test_part1_launch_plan.py --tb=short
  ```

- [ ] **Step 3: Implement the minimal orchestrator.**

  Import and reuse `_execute_natural`, `_execute_checkpoint`, and
  `_run_work_lifecycle` without changing their model inputs or generation
  settings. Validate scope/path separation before any model load. Use
  `LockedShardSession`, preserve fresh-process CUDA behavior, validate expected
  terminal coverage for the selected shard, and finalize only a complete
  shard. A finalized compatible shard is an idempotent no-op; a finalized
  incompatible shard is fatal.

- [ ] **Step 4: Run GREEN, run shell syntax checks, and commit.**

  ```bash
  bash -n jobs/part1_generate_array.sh jobs/part1_phase3_smoke.sh \
    jobs/part1_smoke_a.sh jobs/part1_smoke_b.sh \
    jobs/part1_smollm3_preflight.sh jobs/part1_reproducibility.sh
  ```

  Commit as `feat(part1): add resumable production array runner`.

---

### Task 3: Machine-readable production coverage validator

**Files:**

- Create: `scripts/part1_coverage.py`
- Create: `scripts/validate_part1_results.py`
- Modify: `schemas/part1/validation_report.schema.json`
- Create: `tests/test_part1_coverage.py`
- Create: `tests/test_validate_part1_results.py`

**Interfaces:**

- Consumes: tracked manifests, one model-run manifest, expected shard count,
  and raw shard roots.
- Produces:
  `build_coverage_report(...) -> dict[str, Any]` and an atomically published
  `validation/coverage_report.json`.
- Natural partition keys are `complete`,
  `terminal_infrastructure_failure`, `retryable_incomplete`, `missing`,
  `duplicate`, `schema_incompatible`, and `manifest_incompatible`.
  Checkpoint keys are `complete`, `terminal_infrastructure_failure`,
  `retryable_incomplete`, `ineligible`, `missing`, and `duplicate`.

- [ ] **Step 1: Write failing partition and incompatibility tests.**

  Build complete schema-valid raw shards with `Part1ShardStore` fixtures, then
  independently mutate one condition per test. Assert nominal counts of 5,000
  natural and 55,000 checkpoint logical keys for production, explicit 11-key
  ineligibility after natural infrastructure failure, every requested
  model-output attribute combination, duplicate/schema/manifest distinctions,
  lifecycle-derived retryable/incomplete counts, and rejection of historical
  20q/200q or mixed-manifest roots.

- [ ] **Step 2: Run focused tests RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/mila-uv-cache uv run pytest -q \
    tests/test_part1_coverage.py tests/test_validate_part1_results.py --tb=short
  ```

- [ ] **Step 3: Implement read-only validation and atomic report writing.**

  Recompute all IDs, seeds, parent/alias relationships, schema compatibility,
  and source stream SHA-256 values. Report `structurally_valid`,
  `coverage_complete`, and `paper_analysis_ready` separately. Terminal
  infrastructure failures may be structurally valid and mergeable but set
  `paper_analysis_ready=false` unless a later documented decision accepts them.
  Missing, duplicate, retryable, schema-incompatible, or manifest-incompatible
  work makes both coverage and structural validation fail.

- [ ] **Step 4: Run GREEN and commit.**

  Commit as `feat(part1): add complete workload coverage validation`.

---

### Task 4: Validate-before-publish atomic merge

**Files:**

- Create: `scripts/part1_merge.py`
- Create: `scripts/merge_part1_results.py`
- Create: `tests/test_part1_merge.py`
- Create: `tests/test_merge_part1_results.py`

**Interfaces:**

- Consumes: a structurally valid coverage report and exactly the source streams
  named by that report.
- Produces: atomically published `natural_results.parquet`,
  `checkpoint_results.parquet`, `audit_events.parquet`, and
  `merge_manifest.json` under the model run's merged path.

- [ ] **Step 1: Write failing merge tests.**

  Assert exact logical ordering, source-shard hash coverage, embedded Parquet
  metadata for study/model/question/model-manifest/coverage identities,
  byte-identical rerun behavior, no replacement of divergent finalized output,
  and rejection of a changed source after validation, an invalid coverage
  report, duplicate keys, mixed model runs, and old 20q/200q inputs.

- [ ] **Step 2: Run focused tests RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/mila-uv-cache uv run pytest -q \
    tests/test_part1_merge.py tests/test_merge_part1_results.py --tb=short
  ```

- [ ] **Step 3: Implement staged Parquet publication.**

  Require every source shard to be finalized, rehash all sources before
  reading, validate rows again, sort deterministically,
  write and fsync a sibling staging directory, reload all three Parquet tables,
  verify their embedded metadata and row counts, then publish with one
  same-filesystem directory rename. Never mutate or remove raw shards.

- [ ] **Step 4: Run GREEN and commit.**

  Commit as `feat(part1): add provenance-bound atomic merge`.

---

### Task 5: Trajectory feature extraction

**Files:**

- Create: `scripts/part1_trajectories.py`
- Create: `tests/test_part1_trajectories.py`

**Interfaces:**

- Consumes: compatible merged natural and checkpoint rows.
- Produces: one analysis row per natural run with all eleven fixed primary
  feature values plus first natural-answer appearance, leaving-correct,
  recovery, endpoint agreement, valid-transition counts, and explicit
  missingness.

- [ ] **Step 1: Write failing trajectory tests.**

  Cover aliases, `A -> missing -> B` (no switch), valid adjacent physical
  changes, first natural-answer appearance, leaving a correct answer and later
  recovery, invalid checkpoint 1.0, missing/malformed later suffix values,
  short/zero reasoning, and stabilization at 0.0, an interior fraction, 1.0,
  and null.

- [ ] **Step 2: Run RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/mila-uv-cache uv run pytest -q \
    tests/test_part1_trajectories.py --tb=short
  ```

- [ ] **Step 3: Implement the smallest pure extractor.**

  Iterate requested checkpoints in order, skip aliases without creating a
  transition, and clear adjacency only on an invalid non-aliased physical
  checkpoint. Derive the stable suffix relative to the valid logical 1.0
  answer. Never synthesize behavior from an absent record.

- [ ] **Step 4: Run GREEN and commit.**

  Commit as `feat(part1): derive fixed trajectory features`.

---

### Task 6: AUROC, calibration, bootstrap, and within-question statistics

**Files:**

- Create: `scripts/part1_statistics.py`
- Create: `scripts/part1_bootstrap.py`
- Create: `tests/test_part1_statistics.py`
- Create: `tests/test_part1_bootstrap.py`

**Interfaces:**

- Consumes: trajectory rows plus the fixed 500-question subject frame.
- Produces: pooled/per-subject/macro primary AUROC rows, ECE and explicit
  reliability-bin rows, subject-stratified draw plans, percentile intervals,
  and within-question paired-difference rows/summaries.

- [ ] **Step 1: Write failing exact-statistic tests.**

  Use hand-computable fixtures to pin rank-based AUROC including ties, all
  eleven primary features targeting only `natural_correct`, ten equal-width
  ECE bins with 1.0 in bin ten and empty bins retained, checkpoint-local ECE
  targets, no entropy ECE, arithmetic five-subject macro values, and
  checkpoint-local AUROC only under a secondary label.

- [ ] **Step 2: Write failing bootstrap/paired tests.**

  Assert repeated question draw IDs preserve multiplicity, seed 42 is stable,
  one-class pooled/subject replicates are invalid, macro AUROC is invalid when
  any subject is invalid, fewer than 95% valid replicates suppresses the
  interval with a warning, and within-question output equals `correct mean -
  incorrect mean` with equal question weights and repeated draw multiplicity.

- [ ] **Step 3: Run RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/mila-uv-cache uv run pytest -q \
    tests/test_part1_statistics.py tests/test_part1_bootstrap.py --tb=short
  ```

- [ ] **Step 4: Implement pure NumPy/pandas statistics.**

  Use an average-rank AUROC implementation, one shared subject-stratified draw
  plan per configured replicate count, 2.5/97.5 percentiles, and no set-based
  reconstruction. Report requested/valid/invalid replicate counts for every
  interval and complete cohort/missingness/sample-size fields for every metric.

- [ ] **Step 5: Run GREEN and commit.**

  Commit as `feat(part1): implement fixed uncertainty analyses`.

---

### Task 7: One manifest-driven analysis entry point and atomic outputs

**Files:**

- Create: `scripts/part1_analysis.py`
- Create: `scripts/analyze_part1.py`
- Create: `tests/test_part1_analysis.py`
- Create: `tests/test_analyze_part1.py`

**Interfaces:**

- Consumes: compatible merged tables, merge manifest, coverage report, tracked
  study/question manifests, analysis config, and `--bootstrap-replicates`
  (`1000` development default, `5000` final).
- Produces: `analysis_summary.json`, CSV tables plus adjacent
  `<table>.metadata.json` sidecars, and plots for primary AUROC, all-11-fraction
  checkpoint ECE, main-fraction reliability, within-question differences, and
  trajectory summaries.

- [ ] **Step 1: Write failing integration/output tests.**

  Assert manifest grouping fields are present in every table, no comment lines
  appear in CSV, each CSV has a compatible sidecar with source hashes, all
  eleven ECE fractions occur in machine-readable data and plots, 0.0/0.5/1.0
  are marked main, outputs publish atomically, and absent or incompatible data
  fail before any final output appears. Also prove automatic repetition
  exclusion does not exist.

- [ ] **Step 2: Run RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/mila-uv-cache uv run pytest -q \
    tests/test_part1_analysis.py tests/test_analyze_part1.py --tb=short
  ```

- [ ] **Step 3: Implement orchestration and plots.**

  Validate every input identity, build trajectories once, reuse one draw plan,
  write all tables/plots to a sibling stage, reload JSON/CSV and verify plot
  files are nonempty, then publish the complete analysis directory atomically.
  Refuse paper-final analysis when coverage has unaccepted terminal
  infrastructure failure, while still allowing structural merge/diagnostics.

- [ ] **Step 4: Run GREEN and commit.**

  Commit as `feat(part1): add manifest-driven production analysis`.

---

### Task 8: Independent verification, bounded Phase 3 smoke, and documentation

**Files:**

- Modify: `AGENTS.md`
- Modify: `docs/part1/PLAN.md`
- Modify: `docs/part1/DECISIONS.md`
- Modify: `docs/part1/STATUS.md`
- Modify: `docs/part1/SCHEMA.md`
- Modify: `docs/part1/RUNBOOK.md`
- Modify: `docs/part1/VALIDATION.md`

**Interfaces:**

- Consumes: all reviewed Task 1-7 commits, Phase 2 Smoke A/B evidence, and one
  new bounded `phase3_smoke` result if Task 2 changed the generation path.
- Produces: finalized operator documentation, reviewed readiness evidence, one
  final tracked commit, and then the ignored production model-run manifest.

- [ ] **Step 1: Run local verification and independent code review.**

  Run the full test suite, focused crash/locking/resume tests, manifest
  validator, compilation, JSON parsing, CLI help, shell syntax, forbidden
  import/job scans, `git diff --check`, and an independent whole-branch review.
  Fix every Critical/Important finding test-first and rerun affected tests.

- [ ] **Step 2: Deploy only the reviewed commit to Mila and run one new bounded smoke.**

  Generate a new `phase3_smoke` manifest and output root, submit
  `jobs/part1_phase3_smoke.sh`, and run no other GPU job. Validate accounting,
  terminal log, manifest compatibility, lifecycle, exact bounded shape, and
  absence of production artifacts. If only docs change afterward, do not rerun
  the smoke; any later generation/config change requires a replacement smoke.

- [ ] **Step 3: Reuse Phase 2 real smokes with new read-only tooling.**

  Run the Phase 3 coverage validator against Smoke A and Smoke B on Mila and
  confirm their existing exact 10/110 and 5/55 terminal shapes without
  interpreting their small-sample AUROC/ECE values as scientific evidence.

- [ ] **Step 4: Run a synthetic end-to-end coverage/merge/analysis acceptance.**

  Use temporary schema-valid shards with both target classes and all five
  subjects. Confirm coverage passes, merge publishes, all primary AUROC targets
  are `natural_correct`, ECE targets are correct, bootstrap multiplicity/macro
  invalidity pass, paired differences match the oracle, and switching/
  stabilization cases match the oracle.

- [ ] **Step 5: Finalize the seven tracked documentation files and AGENTS.md.**

  Record exact commands, paths, status semantics, timing-based 500-shard/
  12-hour choice, launch/resume procedures, validation/merge/analysis gates,
  bounded smoke evidence, and the distinction between launch readiness and
  final-paper readiness. Do not add a command that launches the full experiment
  outside the explicitly documented, still-unrun launch command.

- [ ] **Step 6: Commit tracked readiness artifacts and prove the gate.**

  Re-run full verification, commit the final tracked tree, and require:

  ```bash
  git status --porcelain --untracked-files=no
  ```

  to be empty. Report unrelated untracked files without modifying them.

- [ ] **Step 7: Create the ignored production model-run manifest.**

  Run the production-manifest CLI from the final commit. Verify its IDs/hashes,
  the exact final commit, clean-worktree confirmation, output paths, dependency
  lock hash, and that Git remains clean. Run a production dry-run/coverage check
  that reports exactly 5,000 missing natural keys before launch; do not merge or
  analyze absent production data and do not submit the array.

- [ ] **Step 8: Report readiness and stop.**

  Report `READY` only for production launch if every code, provenance, smoke,
  and clean-worktree gate passes. Explicitly report that the experiment remains
  not ready for final paper analysis until the unlaunched production array has
  complete coverage, merge, and final 5,000-replicate analysis.

---

## Required final commands

The implemented runbook must expose these exact command forms (with the actual
model-run ID substituted after manifest creation):

```bash
uv run python scripts/validate_part1_manifests.py
uv run python scripts/create_part1_model_run_manifest.py
uv run python scripts/part1_launch_plan.py --model-run-manifest results/part1/<model-run-id>/model_run_manifest.json
sbatch --array=0-499%1 jobs/part1_generate_array.sh
sbatch --array=0-499%1 jobs/part1_generate_array.sh  # safe resume; not run in Phase 3
sbatch --export=ALL,MODEL_RUN_ID=<model-run-id> jobs/part1_validate.sh
sbatch --export=ALL,MODEL_RUN_ID=<model-run-id> jobs/part1_merge.sh
sbatch --export=ALL,MODEL_RUN_ID=<model-run-id>,BOOTSTRAP_REPLICATES=5000 jobs/part1_analyze.sh
```

The first three are readiness commands. The array launch/resume commands must
be printed but remain unrun. Validation, merge, and analysis become successful
production commands only after complete production output exists.

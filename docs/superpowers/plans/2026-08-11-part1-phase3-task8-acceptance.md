# Part 1 Phase 3 Task 8 Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed read-only validator for the three bounded smoke
scopes and one explicit full-shape synthetic coverage-to-analysis acceptance,
then use the validator on Mila without changing generation code or submitting
production work.

**Architecture:** Keep the strict production coverage implementation unchanged.
Add one smoke-only validator that derives expected work from existing selection
and checkpoint-planning functions, inspects immutable shards through
`Part1ShardStore`, and emits JSON only to stdout. Add a separately marked,
production-shaped synthetic acceptance that invokes the real coverage, merge,
and analysis publication APIs without monkeypatching their validation gates.

**Tech Stack:** Python 3.12, `uv`, pytest, JSON/JSONL, NumPy, pandas, PyArrow,
Matplotlib, existing Part 1 manifest/store/coverage/merge/analysis modules.

## Global Constraints

- Do not change natural generation, checkpoint generation, model inputs,
  scientific settings, identities, or production coverage totals.
- Never load a model, tokenizer, or dataset in these tests or on a login node.
- Existing smoke roots are immutable inputs; validation takes no writer lock and
  creates, deletes, or replaces no file.
- Support only `smoke_a`, `smoke_b`, and `phase3_smoke`.
- Production coverage remains exactly 500 questions, 5,000 natural keys, and
  55,000 checkpoint keys.
- Synthetic fixtures establish software/statistical control flow only, never
  real SmolLM3 behavior.
- Follow red-green-refactor and commit each reviewed task independently.
- Do not submit any SLURM job in Tasks 1 or 2.

---

### Task 1: Dedicated read-only bounded-smoke coverage validator

**Files:**

- Create: `scripts/part1_smoke_coverage.py`
- Create: `scripts/validate_part1_smoke_results.py`
- Create: `tests/test_part1_smoke_coverage.py`
- Create: `tests/test_validate_part1_smoke_results.py`

**Interfaces:**

- Consume the tracked manifest bundle, canonical smoke model-run manifest, and
  canonical finalized shard root.
- Produce
  `build_smoke_coverage_report(*, repository_root: Path,
  model_run_manifest_path: Path, shard_root: Path) -> dict[str, Any]`.
- The CLI requires `--model-run-manifest` and `--shard-root`, accepts an
  optional `--repository-root`, prints one compact JSON object, exits `0` only
  for a complete valid bounded workload, and never writes a report file.
- Reuse `select_smoke_work` for Smoke A/B and call `select_shard_work` with the
  validated records, `shard_index=0`, and `shard_count=500` for Phase 3 smoke;
  reuse canonical generation-seed derivation and
  `build_checkpoint_probe_plans` for checkpoint identities.

- [ ] **Step 1: Write the failing selection, completeness, and mutation tests.**

  Build real temporary `Part1ShardStore` fixtures using existing schema-valid
  fake natural/checkpoint helpers. Assert:

  ```python
  report = build_smoke_coverage_report(
      repository_root=fixture["repository"],
      model_run_manifest_path=fixture["manifest_path"],
      shard_root=fixture["shard_root"],
  )
  assert report["is_valid"] is True
  assert report["coverage_complete"] is True
  assert report["summary"]["natural_partition"]["complete"] == expected_natural
  assert report["summary"]["checkpoint_partition"]["complete"] == expected_checkpoint
  assert report["summary"]["natural_run_ids"] == expected_run_ids
  assert report["summary"]["checkpoint_indices"] == list(range(11))
  assert fingerprint_tree(fixture["shard_root"]) == before
  ```

  Parameterize exact shapes: Smoke A `10/110`, Smoke B `5/55`, and Phase 3
  smoke `10/110`. Add one-condition tests for wrong scope/path, manifest/hash
  drift, unfinalized shard, active lock, pending takeover, invalid tail,
  duplicate/unexpected/missing natural or checkpoint key, hierarchy/lifecycle
  failure, terminal natural failure with 11 ineligible checkpoints, and source
  mutation during validation. Fingerprint every path/type/hash/size/symlink
  target before and after and require no new entry.

- [ ] **Step 2: Run focused tests and confirm RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/mila-uv-cache uv run pytest -q \
    tests/test_part1_smoke_coverage.py \
    tests/test_validate_part1_smoke_results.py --tb=short
  ```

  Expected: import/CLI failures because both new modules are absent.

- [ ] **Step 3: Implement the minimal fail-closed validator.**

  Validate regular non-symlink paths and all manifest identities first. Derive
  the canonical path from execution scope and model-run ID and reject every
  override. Snapshot authoritative source bytes, inspect/build the store index,
  invoke `validate_shard`, derive expected natural and checkpoint keys, build
  exhaustive bounded partitions, and rehash/revalidate every input plus Git
  HEAD at the end. `stable_source_hashes` is sorted by relative POSIX path and
  includes provenance, three JSONL streams, `.finalized`, and retained
  recovery/quarantine evidence. Never instantiate `LockedShardSession`.

- [ ] **Step 4: Implement the JSON-only CLI and prove failures are nonzero.**

  The success object includes execution scope, IDs/hashes, exact partitions,
  run IDs, checkpoint indices, audit count, stable source hashes,
  `structurally_valid`, `coverage_complete`, `paper_analysis_ready`, and
  `mutation_performed=false`. Failure prints a compact object with
  `status=failed`, exception type/message, and `mutation_performed=false` to
  stderr and returns `2`.

- [ ] **Step 5: Run GREEN, regressions, compilation, and commit.**

  ```bash
  UV_CACHE_DIR=/private/tmp/mila-uv-cache uv run pytest -q \
    tests/test_part1_smoke_coverage.py \
    tests/test_validate_part1_smoke_results.py \
    tests/test_part1_coverage.py tests/test_validate_part1_results.py \
    tests/test_part1_store.py tests/test_part1_runtime.py --tb=short
  UV_CACHE_DIR=/private/tmp/mila-uv-cache uv run python -m py_compile \
    scripts/part1_smoke_coverage.py scripts/validate_part1_smoke_results.py
  git diff --check
  ```

  Commit as `feat(part1): add read-only smoke coverage validation`.

---

### Task 2: Connected full-shape synthetic production acceptance

**Files:**

- Create: `tests/test_part1_end_to_end_acceptance.py`
- Modify: `pyproject.toml`

**Interfaces:**

- Register pytest marker `part1_full_acceptance`.
- Consume existing test factories for canonical manifests and schema-valid
  terminal natural/checkpoint/audit records, but invoke real public coverage,
  merge, analysis-loader, and analysis-publication functions.
- Produce no persistent repository artifact; report elapsed seconds and total
  temporary bytes through pytest captured output.

- [ ] **Step 1: Register the marker and write the failing connected test.**

  Construct all 500 canonical question identities, ten runs each, eleven
  checkpoints per complete natural, five ordered 100-question subject blocks,
  and both correctness classes in every subject. Populate finite nonconstant
  values/statuses for every primary feature and all calibration families.
  Give one question two identical valid trajectories/answers under distinct run
  IDs. Mark the test:

  ```python
  @pytest.mark.part1_full_acceptance
  def test_full_shape_raw_to_analysis_acceptance(tmp_path: Path) -> None:
      fixture = write_full_shape_fixture(tmp_path)
      coverage_path = run_coverage_publication(fixture)
      merge_path = run_merge_publication(fixture, coverage_path)
      analysis_path = run_analysis_publication(
          fixture, coverage_path, merge_path, bootstrap_replicates=1_000
      )
      assert_full_acceptance(fixture, coverage_path, merge_path, analysis_path)
  ```

  Invoke `build_coverage_report`/`publish_coverage_report`, real merge staging
  and publication, the real production analysis loader, and real atomic
  analysis publication. Do not monkeypatch compatibility, coverage, merge, or
  analysis validation. The public production calls are
  `build_coverage_report` plus `publish_coverage_report`,
  `merge_part1_results`, `load_production_analysis_source`, and
  `analyze_production` with `bootstrap_replicates=1_000`. Confirm RED at the
  first missing fixture/public handoff, not a typo or dependency error.

- [ ] **Step 2: Add only fast fixture construction needed for GREEN.**

  Write canonical JSONL/provenance/finalization files directly inside the
  temporary fixture to avoid tens of thousands of durability syscalls, then
  make production readers validate every row. Do not add a small-mode flag or
  relax a production invariant. Use exactly 1,000 development bootstrap
  replicates, the smaller of the two values accepted by the production API.

- [ ] **Step 3: Assert non-vacuous published analysis contracts.**

  Reload final artifacts and require every pooled/per-subject primary AUROC row
  with target `natural_correct`; natural-confidence, checkpoint-confidence, and
  maximum-A–D calibration rows at all eleven fractions with prescribed local
  targets; main markers at `0.0/0.5/1.0`; full sidecar/source provenance; all
  5,000 canonical runs in cohorts; and repeated trajectories retained at full
  multiplicity.

  In the same file add hand-computable public-function oracle tests for repeated
  bootstrap draw multiplicity, all-five-subject macro invalidity, exact
  within-question correct-minus-incorrect differences, switch gap adjacency,
  and stabilization/null cases.

- [ ] **Step 4: Run the explicit full acceptance and enforce its envelope.**

  ```bash
  UV_CACHE_DIR=/private/tmp/mila-uv-cache uv run pytest -q \
    -m part1_full_acceptance tests/test_part1_end_to_end_acceptance.py --tb=short -s
  ```

  Require pass, measured runtime at most 30 minutes, and measured temporary
  bytes at most 8 GiB. Exceeding either is a Task 8 blocker, not a skip.

- [ ] **Step 5: Run focused oracle and pipeline regressions, then commit.**

  ```bash
  UV_CACHE_DIR=/private/tmp/mila-uv-cache uv run pytest -q \
    tests/test_part1_coverage.py tests/test_validate_part1_results.py \
    tests/test_part1_merge.py tests/test_merge_part1_results.py \
    tests/test_part1_trajectories.py tests/test_part1_statistics.py \
    tests/test_part1_bootstrap.py tests/test_part1_analysis.py \
    tests/test_analyze_part1.py --tb=short
  git diff --check
  ```

  Commit as `test(part1): add full-shape Phase 3 acceptance`.

---

### Task 3: Independent review, Mila evidence, and Task 8 handoff

**Files:**

- Modify: `docs/part1/PLAN.md`
- Modify: `docs/part1/DECISIONS.md`
- Modify: `docs/part1/STATUS.md`
- Modify: `docs/part1/SCHEMA.md`
- Modify: `docs/part1/RUNBOOK.md`
- Modify: `docs/part1/VALIDATION.md`
- Modify: `AGENTS.md` only for verified current lifecycle/readiness wording;
  never weaken cluster safety silently.

- [ ] **Step 1: Independently review Tasks 1 and 2 and close findings.**

  Require spec compliance, code quality approval, focused test evidence, and no
  generation/config diff after the successful Phase 3 smoke commit.

- [ ] **Step 2: Deploy the reviewed validator-only commits to Mila.**

  Fast-forward by verified Git bundle if GitHub publication remains
  unauthorized. Do not replace or rerun any smoke. Run the read-only validator
  against canonical Smoke A, Smoke B, and Phase 3 smoke paths and capture exact
  JSON, file hashes, and zero-mutation evidence.

- [ ] **Step 3: Return to the parent Phase 3 Task 8 plan.**

  Use the verified smoke reports and full-shape acceptance evidence to finish
  documentation, broad verification, final review/commit, clean-tree proof,
  ignored production-manifest creation, and the missing-work launch plan. Do
  not submit the 500-question array inside this implementation plan.

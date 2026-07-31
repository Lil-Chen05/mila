# Part 1 implementation plan

## Authority and current phase

This plan implements the fixed Part 1 contract in [DECISIONS.md](DECISIONS.md).
The data contracts are in [SCHEMA.md](SCHEMA.md), operational ordering is in
[RUNBOOK.md](RUNBOOK.md), and acceptance evidence belongs in
[VALIDATION.md](VALIDATION.md). Changes to the scientific contract require
explicit approval and a versioned manifest change.

`README.md`, `CLAUDE.md`, `docs/plan.md`, `docs/NOTES_resume.md`,
`docs/TODO_TALKING_POINTS_200Q.md`, the old Superpowers repository-reorganization
specification, and all `results/{20q,200q}` and `analysis/{20q,200q}` artifacts
are historical. They describe pilots and are not instructions or evidence for
the production Part 1 design.

**Current phase: Prompt 1 is complete; stop and await Prompt 2.** Prompt 1
audited the repository, installed the working agreement, and established these
six authoritative documents. No Phase 1 production implementation or immutable
production model-run manifest has been created.

## Scope

Part 1 evaluates whether uncertainty along a naturally sampled reasoning
trajectory predicts `natural_correct`. It uses one immutable, balanced
500-question MMLU manifest, ten natural runs per model-question pair, and eleven
greedy checkpoint probes per successfully executed natural run. The current
four-prompt sequence implements only `HuggingFaceTB/SmolLM3-3B`. Later models
must enter through adapters without changing the study design.

Out of scope for all three phases are later models, the later
answerable-versus-unanswerable study, unapproved primary features, an automatic
repetition detector, and a full 500-question experiment run. Only the bounded
smokes authorized by Prompt 3 or Prompt 4 may generate model output during this
sequence.

## Dependencies and order

The phases are strictly ordered:

1. Phase 1 locks record meaning, identity, and crash-safe storage before any
   new experiment artifacts exist.
2. Phase 2 builds the fixed scientific inputs and SmolLM3 generation path on
   the Phase 1 contracts.
3. Phase 3 builds analysis and production operations only after Phase 2 raw
   records can be validated.

Within every phase, use test-first work units: add a focused failing test,
confirm the expected failure, implement the smallest contract-compliant change,
run the focused test, then run the complete login-safe suite. Model- or
dataset-dependent acceptance checks run only in appropriately resourced SLURM
jobs on compute nodes.

## Phase 1 — foundations (Prompt 2)

### Scope and likely files

Define and test schemas, canonical serialization and identifiers, persistence,
locking, terminal failure handling, and resumability. Do not create a production
model-run manifest or materialize the 500-question production manifest.

Likely additions or focused changes:

- `scripts/part1_contract.py`: schema validation, canonical serialization,
  stable hashes, logical identities, version compatibility, and seed derivation.
- `scripts/part1_store.py`: append-only events, atomic terminal-record
  publication, lock ownership, duplicate suppression, recovery, and resume
  planning.
- `tests/test_part1_contract.py`: schemas, nullability, canonical bytes, golden
  hashes and seeds, identity boundaries, and compatibility.
- `tests/test_part1_store.py`: atomic writes, interrupted writes, lock
  contention/staleness, failure events, retry identity, and idempotent resume.
- `tests/fixtures/part1/`: small synthetic records only; no model or dataset
  loads.
- `.gitignore`: only the narrow operational Part 1 result patterns required by
  the approved storage layout; historical tracked outputs remain untouched.
- These six `docs/part1/` files: update decisions only after verified behavior.

The exact module split may be adjusted during Prompt 2 if the same responsibility
boundaries and test seams are preserved. Existing `scripts/checkpoints.py` is not
the Phase 1 implementation target.

### Test-first work units

1. Schema-version and enum validation, including the natural/checkpoint
   nullability matrix and all eleven checkpoint identities.
2. Canonical bytes and golden SHA-256 identities for question, question
   manifest, study, model run, raw record, event, and generation seed.
3. Manifest compatibility checks and rejection of mixed model-run records in a
   shard.
4. Atomic append/publication and recovery from partial or interrupted writes.
5. Exclusive locking with the chosen Mila-compatible stale-lock policy.
6. Terminal infrastructure-failure events, retry classification, same-seed
   retries, and exact attempt accounting.
7. Resume planning that schedules only missing or retryable logical work and
   never regenerates successful abnormal model output.

### Completion criteria

Phase 1 is complete only when:

- [SCHEMA.md](SCHEMA.md)'s deferred canonical payload and raw-record-granularity
  choices are explicitly locked and covered by golden tests.
- Schema validators reject invalid enum combinations, missing logical identity,
  forbidden values, and incorrect nullability.
- Serialization is deterministic across key order, whitespace, Unicode, and
  process restarts; no identity contains itself or mutable operational fields.
- Persistence tests demonstrate that interruption cannot publish a partial
  terminal record and that resume is idempotent.
- Lock contention and stale-lock behavior are deterministic on the chosen
  persistent Mila filesystem.
- Infrastructure failures remain distinguishable from executed-but-abnormal
  model output; retry attempts reuse the same generation seed.
- The full login-safe test suite passes.
- No model or dataset has been loaded on a login node, no production manifest
  has been created, and no full experiment has been launched.

## Phase 2 — generation (Prompt 3)

### Scope and likely files

Create the tracked fixed question manifest and study manifest; implement the
shared Hugging Face interface and SmolLM3 adapter; implement deterministic
per-run seeding, ten stochastic natural generations, and eleven greedy
checkpoint identities with alias preservation. Smoke artifacts must remain
separate from production artifacts.

Likely additions or focused changes:

- `scripts/prepare_part1_questions.py` and
  `jobs/prepare_part1_questions.sh`: compute-node materialization using streaming
  plus bounded selection, five exact subject quotas, and tracked manifest export.
- `manifests/part1/questions.jsonl`,
  `manifests/part1/questions.manifest.json`, and
  `manifests/part1/study_manifest.json`: immutable tracked scientific inputs.
- `scripts/part1_hf.py`: shared generation interface and common measurement
  logic.
- `scripts/smol_lm3_adapter.py`: thinking boundaries, prompt, terminal parser,
  forced-close inducer, and model-specific token conventions.
- `scripts/run_part1.py` and a bounded smoke launcher under `jobs/`: one model on
  one GPU, batch size one, no cross-question/run/checkpoint batching.
- `tests/test_part1_questions.py`, `tests/test_smol_lm3_adapter.py`, and
  `tests/test_part1_generation.py`: pure fixtures and mocked interfaces on the
  login node; real model/dataset checks only in SLURM smoke jobs.

Existing `scripts/fetch_mmlu.py`, `scripts/checkpoints.py`, and
`jobs/checkpoints.sh` are historical inputs to the rewrite, not protocol-compliant
production entry points.

### Test-first work units

1. Five-subject selection order, 100-per-subject quotas, sample indices 0–499,
   stable question IDs, and manifest hash verification.
2. Study-manifest construction and compatibility with the Phase 1 schema.
3. Adapter parsing of reasoning open/close boundaries and one terminal
   post-close answer/confidence block, including malformed and missing cases.
4. Compute-node preflight of the immutable model/tokenizer revisions, thinking
   tags, A–D token sequences, answer-token location, and requested versus
   effective generation settings.
5. Canonical run seeds for `run_id` 0–9 and stochastic natural-generation
   configuration without an extra greedy run.
6. Raw float32 pre-warper natural entropy, unrounded summaries, and exact
   entropy/token alignment.
7. Token-ID checkpoint placement, all eleven requested identities, shared
   probes for aliases, and greedy probe measurements/statuses.
8. Bounded smoke execution and resume from separate smoke manifests/output
   roots. If code is dirty, record the base commit plus diff hash and label the
   run non-production.

### Completion criteria

Phase 2 is complete only when:

- The tracked question manifest contains exactly 500 unique records in the
  required subject and seeded-selection order, with 100 per subject, and its
  source revision and hashes validate.
- The tracked study manifest encodes every fixed scientific and analysis
  contract and validates against the question manifest.
- SmolLM3 preflight evidence fixes immutable model/tokenizer revisions and the
  exact tag and A–D token conventions.
- Every successful smoke natural run produces aligned raw tokens/entropies and
  exactly eleven requested checkpoint records, including aliases.
- Sampling settings, canonical seed derivation, and same-seed retry reuse are
  verified; checkpoint probes are greedy and deterministic.
- Capped, unclosed, zero/short, missing-answer, and malformed successful outputs
  remain checkpoint-eligible and are not retried as infrastructure failures.
- All login-safe tests and the explicitly authorized bounded SLURM smoke pass.
- Production and smoke manifests/output roots are distinct, and no production
  model-run manifest or full experiment has been created.

## Phase 3 — completion (Prompt 4)

### Scope and likely files

Finish analysis, validation, merge safety, operational production model-run
manifest lifecycle, SLURM readiness, documentation, and final bounded smoke
tests. Only after the final tracked production commit and clean-worktree gate
may the operational production model-run manifest be created.

Likely additions or focused changes:

- `analysis/analyze_part1.py`: fixed primary AUROC, calibration, bootstrap,
  within-question, switching, stabilization, and machine-readable outputs.
- `scripts/validate_part1.py`: schema, provenance, completeness, alias, and
  semantic validation.
- `scripts/merge_part1.py`: validate-before-publish atomic merge with exact
  logical-identity coverage.
- `scripts/create_model_run_manifest.py`: post-commit operational manifest
  creation from verified preflight facts.
- `jobs/run_part1.sh`, `jobs/validate_part1.sh`, `jobs/merge_part1.sh`, and
  `jobs/analyze_part1.sh`: production-ready resource separation and persistent
  paths.
- `tests/test_part1_analysis.py`, `tests/test_part1_validation.py`, and
  `tests/test_part1_merge.py`: synthetic edge cases, bootstrap multiplicity,
  compatibility, and crash safety.
- Narrow `.gitignore` updates for generated operational manifests and raw Part 1
  output, without ignoring tracked manifests or changing historical artifacts.

### Test-first work units

1. ECE bin boundaries, invalid confidence handling, per-fraction reporting, and
   pooled/subject/macro summaries.
2. Fixed AUROC registry orientation and target, subject-stratified bootstrap
   multiplicity, invalid-replicate accounting, and the 95% interval gate.
3. Within-question paired differences with equally weighted questions and
   multiplicity-preserving question bootstrap.
4. Alias-aware switching, missing-value adjacency breaks, first natural-answer
   appearance, final-answer stabilization, recovery, and 1.0 agreement.
5. Complete validator rejection tests for mixed provenance, duplicates,
   omissions, illegal nulls, unrounded/unaligned values, and incompatible
   manifests.
6. Atomic merge tests proving incomplete inputs never replace a valid merged
   artifact.
7. SLURM and lifecycle checks, followed by only the Prompt-4-authorized bounded
   smoke.

### Completion criteria

Phase 3 is complete only when:

- Synthetic analysis tests cover every scientific definition in
  [DECISIONS.md](DECISIONS.md), including bootstrap multiplicity and invalid
  intervals.
- Validation and merge reject incomplete, duplicate, mixed, or incompatible
  data before publishing output.
- GPU jobs use one model on one GPU in bfloat16 evaluation mode with batch size
  one; CPU-only jobs omit GPU requests; all Hugging Face loads use scratch
  cache and occur on compute nodes.
- The final bounded smoke passes end to end through resume, validation, merge,
  and analysis without touching production outputs.
- Final generation code, tracked manifests, tests, and documentation are
  committed; the tracked worktree is clean except for intentional local
  exclusions.
- The production model-run manifest is then generated under an explicitly
  ignored persistent results directory, records the final production commit,
  and does not dirty the tracked worktree.
- The full 500-question run remains unlaunched. Production execution requires a
  later explicit authorization.

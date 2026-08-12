# Part 1 implementation plan

## Authority and current phase

This plan implements the fixed Part 1 contract in [DECISIONS.md](DECISIONS.md).
The executable data contract is summarized in [SCHEMA.md](SCHEMA.md), operator
procedures are in [RUNBOOK.md](RUNBOOK.md), and acceptance evidence is in
[VALIDATION.md](VALIDATION.md). Scientific changes require explicit approval and
a versioned manifest change.

**Current phase: Phase 3 Task 8 is in progress.** CPU materialization and
manifest validation, GPU preflight, Mila filesystem/scheduler checks,
same-environment reproducibility, both Phase 2 bounded GPU smokes, and reviewed
Phase 3 implementation Tasks 1–7 are complete. Final bounded Phase 3 smoke job
`10324103` completed `0:0` in `01:27:13`; its stable shard contains exactly 10
natural results, 110 checkpoint results, and 240 audit events, and direct
integrity validation passed. The hardened CLI validator is implemented,
independently approved, and locally regression-tested; its retained-smoke Mila
execution is pending. The corrected full-shape synthetic acceptance is prepared
as a CPU-only Mila job after a fixture-only prompt-hash defect stopped the first
local attempt at coverage. No production model-run manifest exists, production
readiness has not been established, and the full experiment is unsubmitted.

Historical root documentation, old 20q/200q code, results, and analyses remain
pilot artifacts. They are not instructions or evidence for Part 1.

## Scope and phase order

Part 1 evaluates whether uncertainty along a naturally sampled reasoning
trajectory predicts `natural_correct`. It uses one immutable balanced
500-question MMLU manifest, ten natural runs per model-question pair, and eleven
greedy checkpoint probes per successful natural run. The current sequence
implements only `HuggingFaceTB/SmolLM3-3B`.

The phases remain strictly ordered:

1. Phase 1 fixes record meaning, identity, persistence, failure handling, and
   resume behavior without producing scientific inputs or model output.
2. Phase 2 creates the fixed question/study manifests and SmolLM3 generation
   path, then runs only an explicitly authorized bounded smoke.
3. Phase 3 implements analysis, full validation and merge, SLURM readiness, and
   the post-commit production model-run-manifest lifecycle.

Later models, the later answerable-versus-unanswerable study, and unapproved
primary features remain out of scope. On 2026-08-11 the user explicitly
authorized the unchanged full production run after readiness: the target array
is `0-499%16`, with a four-day deadline. This authorization does not bypass the
ordered Phase 3 acceptance, review, clean-tree, manifest, or readiness gates.

## Phase 1 — foundations, completed

### Implemented artifacts

- Six versioned templates in `configs/part1/`: study protocol, model-run
  execution, dataset materialization, storage, retries, and analysis.
- Eight JSON Schema Draft 2020-12 files in `schemas/part1/`: question record,
  question manifest, study manifest, model-run manifest, natural terminal
  result, checkpoint terminal result, audit event, and validation report.
- `scripts/part1_contract.py`: canonical JSON, domain-separated SHA-256
  identities, deterministic generation seeds, schema validation, and Phase 1
  configuration guards.
- `scripts/part1_store.py`: normalized append-only terminal/event streams,
  lifecycle indexing, crash recovery, shard validation, and finalization.
- `scripts/part1_failure_policy.py`: one shared failure taxonomy, three-attempt
  policy, and exact backoffs.
- `scripts/part1_runtime.py`: immutable shard binding, locking/takeover,
  liveness, manifest compatibility, retry planning, and natural/checkpoint
  resume planning.
- `scripts/part1_dry_run.py` and `scripts/part1_operator_unlock.py`: login-safe
  read-only inspection and explicit audited operator recovery.
- Synthetic tests in `tests/test_part1_contract.py`,
  `tests/test_part1_store.py`, `tests/test_part1_runtime.py`, and
  `tests/part1_store_fixtures.py`.

The implementation was delivered in the scoped commit sequence:

- contracts: `0b7b6f4`, corrected by `8ce4bbf`;
- storage: `10d2f95`, corrected by `34c4c90` and `a850ee2`; and
- runtime: `c066bff`, corrected by `d15ba85` and `68926ec`.

Final independent review corrections are `e6e0dee`, `1ca5661`, `260fb44`,
`12dd6a9`, and `5969080`. They close directory-entry durability, lifecycle,
checkpoint/alias integrity, fixed-contract/null-matrix/timestamp enforcement,
locked-store defaults, retry evidence/count validation, and the exact 10% tail
entropy oracle.

### Locked engineering decisions

- Canonical bytes use `part1-canonical-json-v1`; scientific identity uses
  `part1-identity-v1`; seed derivation uses `part1-seed-v1`.
- Natural and checkpoint outcomes are normalized into
  `natural_results.jsonl` and `checkpoint_results.jsonl`. Lifecycle evidence is
  separate in `audit_events.jsonl`.
- A terminal result is fsynced before `attempt_completed` is appended and
  fsynced. The result is authoritative if the completion event is absent.
- First stream/report/finalization files and every newly created root/directory
  component are directory-fsynced after publication so their names are durable,
  not only their file contents.
- Every attempt is consumed by its durable `attempt_started`. Orphans and
  completion-without-result states are classified on resume.
- Only an incomplete final line may be repaired. Exact invalid bytes are
  quarantined and recovery is journaled before mutation; malformed middle
  records are rejected.
- A shard has an immutable `.shard-provenance.json` binding the study ID,
  model-run ID, complete model-run-manifest hash, and shard ID.
- One stable `.writer.guard` POSIX advisory lock spans lease validation and the
  complete mutation, close, or takeover. `.writer.lock` holds owner metadata;
  takeover state and history are durable and resumable.
- There are exactly three attempts. Retryable categories use the same logical
  identity and seed, with backoffs `[0, 30, 120]`; transient CUDA retry requires
  a fresh process. Final failures are terminalized by result then completion,
  not by a final `attempt_failed` event.
- `attempt_interrupted` is restricted to `interrupted_process`. Executable
  retry eligibility comes from exactly one coherent retry-authorizing closure
  on the latest persisted attempt; caller category/count only check equality,
  and `attempts_consumed` must be a real integer in `[0,3]`.
- Checkpoint publication/indexing recompute ties-to-even placement, actual
  fraction, checkpoint/shared-probe identities, alias ownership/membership, and
  cross-record physical-probe coherence from the persisted natural parent.
- All fixed templates are checked against an independent complete oracle
  (storage roots are the only variable fields). Study science and requested
  model settings are exact; effective settings require every requested key and
  exact value while permitting additional resolved serializable fields.
- Store mutation requires a runtime lock capability by default; only synthetic
  tests may explicitly opt into `unsafe_for_tests=True`.
- Resume is independent at natural-run and checkpoint granularity. A completed
  natural result is never regenerated because checkpoint work is absent.

Full contracts and exact payloads are in [SCHEMA.md](SCHEMA.md); operational
state transitions are in [RUNBOOK.md](RUNBOOK.md).

### Verified completion gate

The Prompt 1 baseline had 21 passing login-safe tests. Phase 1 concluded at
`5969080` with 265 passing login-safe synthetic tests, including the five
terminal-append crash boundaries, storage-recovery and takeover boundary
matrices, and a real two-process local POSIX `flock` regression. Independent
final re-review found no remaining Critical, Important, or Minor Phase 1 code
finding. No real SLURM command, CUDA runtime, model, tokenizer, dataset,
production manifest, or experiment output was used.

The repository-level implementation gate is satisfied, subject to the
operational Phase 2 checks below. Phase 1 does not claim Mila filesystem or
cluster validation and is not production-execution authorization.

## Phase 2 — fixed data and generation (completed)

### Implemented preparation

The scoped preparation commit provides:

1. an explicit per-subject `cais/mmlu` strategy replacing the invalid
   `source_config: "all"` assumption;
2. a CPU job that resolves a 40-character commit SHA, verifies each complete
   test-split count, streams/shuffles with seed 42 and a full-split buffer,
   takes 100 per subject, stages the saved dataset plus all three manifests,
   reloads and validates all content, and publishes the manifest directory in
   one atomic rename;
3. strict identical-only rerun behavior for existing finalized manifests and a
   separate login-safe validator for returned outputs;
4. pure SmolLM3 prompt/tag/token-boundary, terminal parsing, natural entropy,
   checkpoint placement/metrics/alias, seed, schema, smoke-selection, and
   storage-estimation logic;
5. prepared GPU-only model preflight, isolated reproducibility, Smoke A, and
   Smoke B scripts using separate non-production model-run identities; and
6. a synthetic evidence catalog with schema-valid raw-input families that
   explicitly denies real-model and Phase 3 analysis-implementation evidence.

### Completed operational gates

1. CPU job `10284018` materialized and validated the exact 500-question MMLU
   bundle. The three tracked files under `manifests/part1/` were independently
   validated and committed in `2e0bcae`.
2. Single-L40S preflight and the provenance refresh job `10284702` completed
   `0:0` at Git commit
   `e19edf22c8a3f7462ab66d5cae11b38247df5ed9`. The immutable model/tokenizer
   revision is `a07cc9a04f16550a088caea529712d1d335b0ac1`.
3. The persistent Mila filesystem gate passed all four focused tests at
   `results/part1-smoke/mila-filesystem-gate.EUEXDB`. The scheduler liveness
   gate also passed, including the fail-closed distinction between an immediate
   completed/absent return-code-0 job (`DEAD`) and a later purged return-code-1
   query (`UNKNOWN`).
4. Corrected reproducibility job `10284721` completed `0:0` with status passed,
   exact token and parser equality, and exact entropy arrays at tolerance 0.0
   for seed `2552280803631819986`. The RNG correction is commit `e19edf2`; the
   local suite at that commit was 367 passed. The durable report is
   `results/part1-smoke/reproducibility/14c49484a4eebdb79372cb14b3e0076812e983d688c49aa7e3c2280bb44be7c0/reproducibility_report.json`.

### Completed Smoke A and Smoke B gates

Smoke A authoritative job `10284742` completed `0:0` in `01:26:28` on
`cn-l018`. The runner reached its terminal `complete` state. Its one-question
non-production shard has exactly natural run IDs `0`–`9`, all requested
checkpoint indices `0`–`10` for each run, 10 natural results, 110 checkpoint
results, and 240 audit events. Manifest, dry-run, and lifecycle validation all
passed with 0 errors and 0 warnings; no production root was created. All 10
natural records have `natural_execution_outcome=complete`,
`reasoning_status=closed`, `answer_parse_status=parsed`, and
`confidence_parse_status=malformed`. All 110 checkpoint records have
`checkpoint_execution_outcome=complete`; `checkpoint_model_output_status` is
valid/invalid `93/17`, `answer_token_status` is located/missing `109/1`, and
`entropy_status` is computed/unavailable `109/1`. This successful abnormal
output was retained as data.

Smoke B's first submission, job `10292499`, failed in one second before
Python, model, dataset, or artifact creation because non-interactive SSH did
not include `~/.local/bin` in `PATH`, so `srun` could not execute `uv`. It
created no Smoke B root and is diagnostic only, not a model, reproducibility,
or experiment failure. The authoritative resubmission, job `10292530`, used
the unchanged documented sbatch job from a Mila login shell and completed
`0:0` in `00:25:14` on `cn-l072`. Its non-production shard exercised sample
indices `0`, `100`, `200`, `300`, and `400` (one fixed first question per
subject), each at natural `run_id=0` and checkpoint indices `0`–`10`: 5
natural results, 55 checkpoint results, and 120 audit events. The terminal
runner state was `complete`; manifest, dry-run, and lifecycle validation all
passed with 0 errors and 0 warnings; no production root was created. All 5
natural records have `natural_execution_outcome=complete`;
`reasoning_status` is closed/missing_close `4/1`, `answer_parse_status` is
parsed/missing/out_of_domain `3/1/1`, and `confidence_parse_status` is
malformed/missing `4/1`. All 55 checkpoint records have
`checkpoint_execution_outcome=complete`; `checkpoint_model_output_status` is
valid/invalid `42/13`, `answer_token_status` is located/missing `46/9`, and
`entropy_status` is computed/unavailable `46/9`. This successful abnormal
output was retained as data. There were no retries, terminalization,
tail/recovery, lock, or takeover issues.

### Phase 2 completion record

- A purged/error `squeue` response remains `UNKNOWN`, not `DEAD`; post-job
  decisions use `sacct` and retained job evidence.
- Generation integration used `LockedShardSession`; terminal records and audit
  evidence were policy-complete.
- The non-interactive `uv` `PATH` failure is a Phase 3 SLURM-readiness
  hardening item, not a reproducibility or experimental failure.

### Phase 2 completion criteria

- The tracked manifest has exactly 500 unique questions, five ordered blocks of
  100, seed 42, and reproducible hashes. **Passed.**
- The tracked study manifest validates and contains every fixed scientific
  contract. **Passed.**
- SmolLM3 preflight fixes immutable revisions, tags, prompt, A–D tokens, and
  requested/effective settings. **Passed.**
- Every successful smoke natural run has aligned tokens/entropies and all eleven
  requested checkpoint identities, including aliases.
- Successful abnormal output remains data; only infrastructure failures enter
  the retry policy.
- Login-safe tests and the authorized bounded compute smoke pass without
  creating production artifacts.

The dataset, study-manifest, preflight, filesystem, scheduler,
reproducibility, Smoke A, and Smoke B criteria are passed. The full
500-question experiment remained forbidden at the Phase 2 gate. The later
2026-08-11 authorization applies only after every remaining Phase 3 gate
passes.

## Phase 3 — completion

Phase 3 Tasks 1–7 are implemented, independently reviewed, and deployed to Mila
at `a7e1135e85476f5fc43986949f467b10ba450623`. Task 8 is in progress. Its one
authorized final bounded smoke, job `10324103`, completed `0:0` in `01:27:13`.
The terminal artifacts contain exactly 10 natural results, 110 checkpoint
results, and 240 audit events; direct shape, provenance, hierarchy, and
lifecycle integrity validation passed. This is bounded non-production evidence,
not production readiness. Hardened CLI-validator acceptance remains pending,
and no production array has been submitted.

Task 8 uses this fail-closed order:

1. complete and verify the hardened CLI validator, then obtain its acceptance
   result for the bounded Phase 3 smoke and the required read-only Smoke A/B
   coverage;
2. complete the synthetic end-to-end acceptance and reconcile its evidence with
   the tracked validation ledger;
3. finalize the six Part 1 documents plus `AGENTS.md`, run full verification,
   and complete independent review;
4. create the final tracked commit, deploy it to Mila, and require a clean
   tracked worktree;
5. submit the CPU-only full acceptance and an `afterok` production gate;
6. only after acceptance succeeds, the gate revalidates all retained smokes,
   creates the immutable operational production manifest from that exact
   commit, validates the `0-499%16` launch plan, and submits generation;
7. validation runs after the array terminates, while merge and final analysis
   remain `afterok`-gated; every returned job ID is durably recorded; and
8. operate the authorized chain to the four-day deadline without advancing the
   tracked Mila checkout.

No step above is presently evidence that hardened CLI acceptance, production
manifest creation, readiness, or launch has passed.

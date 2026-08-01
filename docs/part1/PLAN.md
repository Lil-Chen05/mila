# Part 1 implementation plan

## Authority and current phase

This plan implements the fixed Part 1 contract in [DECISIONS.md](DECISIONS.md).
The executable data contract is summarized in [SCHEMA.md](SCHEMA.md), operator
procedures are in [RUNBOOK.md](RUNBOOK.md), and acceptance evidence is in
[VALIDATION.md](VALIDATION.md). Scientific changes require explicit approval and
a versioned manifest change.

**Current phase: Phase 1 is implemented and independently verified. Stop before
Phase 2.** Phase 1 added login-safe schemas, configuration templates, canonical
identities and seeds, normalized append-only storage, recovery, exclusive
locking, retry policy, and resumability. It did not materialize a dataset, load
a model or tokenizer, create a production model-run manifest, generate output,
or launch an experiment.

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

The full 500-question experiment, later models, the later
answerable-versus-unanswerable study, and unapproved primary features remain out
of scope.

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

## Phase 2 — generation (next prompt only)

### Authorized scope when Prompt 3 arrives

Phase 2 must:

1. resolve the immutable MMLU revision and materialize the fixed sample in a
   CPU SLURM job using streaming plus bounded selection;
2. create, inspect, and commit the tracked question JSONL/sidecar and
   model-independent study manifest;
3. resolve immutable SmolLM3 model/tokenizer revisions and adapter conventions
   in compute-node preflight;
4. implement token-boundary parsing, stochastic natural generation, raw
   pre-warper entropy, and greedy checkpoint probes;
5. use `LockedShardSession`, policy-complete events, complete manifest hashes,
   canonical seeds, same-seed retries, and a fresh process after transient CUDA
   failure; and
6. run only the Prompt-3-authorized bounded smoke under the separate ignored
   smoke root.

### Phase 2 dependencies and operational checks

- The dataset, model, and tokenizer immutable revisions remain unresolved.
- The fixed question and study manifests do not yet exist.
- SmolLM3 thinking tags, prompt rendering, forced-close token IDs, A–D token
  convention, and effective generation settings require compute-node preflight.
- The stable guard passed a local two-process test, but POSIX `flock` behavior
  on the selected persistent Mila filesystem must be confirmed.
- Directory fsync, no-overwrite hard links, and atomic replacement must also be
  exercised on that exact persistent Mila filesystem before operational use.
- The fail-closed `squeue` probe and array selector are synthetically tested;
  Mila's actual array-job output/absence behavior must be confirmed.
- Generation integration must never bypass `LockedShardSession` or construct
  incomplete events/results by hand.

### Phase 2 completion criteria

- The tracked manifest has exactly 500 unique questions, five ordered blocks of
  100, seed 42, and reproducible hashes.
- The tracked study manifest validates and contains every fixed scientific
  contract.
- SmolLM3 preflight fixes immutable revisions, tags, prompt, A–D tokens, and
  requested/effective settings.
- Every successful smoke natural run has aligned tokens/entropies and all eleven
  requested checkpoint identities, including aliases.
- Successful abnormal output remains data; only infrastructure failures enter
  the retry policy.
- Login-safe tests and the authorized bounded compute smoke pass without
  creating production artifacts.

## Phase 3 — completion

Phase 3 remains deferred. It owns analysis, full raw validation, atomic merge,
SLURM launchers, production lifecycle checks, and final bounded smoke. The
operational production model-run manifest may be created only after final
tracked production code/manifests/docs are committed and the tracked worktree
is clean except for intentional local exclusions. Its creation under the
ignored production root must leave Git clean. The full experiment still
requires separate authorization afterward.

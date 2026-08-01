# Part 1 status

## Executive status

**Prompt 2 / Phase 1 is implemented and independently verified on branch
`codex/phase1-infrastructure`. Stop before Phase 2.** The repository now has
executable schemas, canonical identities and seeds, crash-consistent normalized
storage, exclusive shard ownership, failure policy, retry planning,
resumability, validation reports, and login-safe operator tools.

Phase 1 did not load a model, tokenizer, dataset, CUDA runtime, or invoke real
SLURM. It did not materialize MMLU, create the fixed question/study manifests,
create a production model-run manifest, generate a smoke or production output,
or launch the experiment. The full 500-question run remains unauthorized.

Read this with [PLAN.md](PLAN.md), [DECISIONS.md](DECISIONS.md),
[SCHEMA.md](SCHEMA.md), [RUNBOOK.md](RUNBOOK.md), and
[VALIDATION.md](VALIDATION.md). Older root documentation and 20q/200q artifacts
remain historical.

## Git and audit baseline

- Prompt 1 working agreement: `01ed450` (`docs: update Part 1 working
  contract`).
- Prompt 1 authoritative documentation: `27cbacd` (`docs: establish Part 1
  experiment plan`).
- Prompt 2 preflight re-read all six documents and `AGENTS.md`, inspected Git,
  verified `.superpowers/` remained excluded only through `.git/info/exclude`,
  found the tracked worktree clean, and reran the existing suite: 21 tests
  passed.
- `.superpowers/`, historical outputs, and unrelated user state were not
  modified.

## Phase 1 implementation ledger

### Contracts and schemas

Commits `0b7b6f4` and `8ce4bbf` added:

- six `1.0.0` templates under `configs/part1/` for study protocol, model-run
  execution, dataset materialization, storage, retries, and analysis;
- eight JSON Schema Draft 2020-12 files under `schemas/part1/` for question
  records/manifests, study/model-run manifests, natural/checkpoint terminals,
  audit events, and validation reports;
- `scripts/part1_contract.py` and its synthetic test suite;
- `jsonschema` as a direct `uv` dependency; and
- narrow ignores for generated `results/part1-smoke/` and `results/part1/`
  roots without changing historical tracked result policy.

The locked canonical contract is `part1-canonical-json-v1`; scientific
identities use `part1-identity-v1`; deterministic seeds use `part1-seed-v1`.
All public identities, including shared probes and both audit scopes, have
golden vectors.

### Storage and recovery

Commits `10d2f95`, `34c4c90`, `a850ee2`, `e6e0dee`, and `1ca5661` added and
hardened
`scripts/part1_store.py` plus fixtures/tests. The implementation uses three
separate append-only streams, per-record flush/fsync, durable directory-entry
creation, result-before-completion ordering, terminal-result authority,
reconciliation, exact-byte tail quarantine, immutable recovery journals,
validation reports, duplicate/conflict detection, complete checkpoint-parent
and alias-group validation, and a `.finalized` marker that blocks further
raw-shard mutation.

The public result API requires a matching durable `attempt_started` before it
can create authoritative result bytes. Malformed middle lines are fatal; only
the final physical line can be repaired. A valid EOF JSON object missing only
its newline is repaired by append, not rewrite.

### Locking, retry, and resume

Commits `c066bff`, `d15ba85`, `68926ec`, `e6e0dee`, `1ca5661`, `260fb44`, and
`12dd6a9` added and hardened:

- `scripts/part1_failure_policy.py`;
- `scripts/part1_runtime.py`;
- `scripts/part1_dry_run.py`;
- `scripts/part1_operator_unlock.py`; and
- the locking/retry/resume test matrix.

Every shard is bound by immutable `.shard-provenance.json` to `study_id`,
`model_run_id`, the complete `model_run_manifest_hash`, and `shard_id`.
`.writer.lock` records the active owner, while stable `.writer.guard` advisory
locking spans the complete check-and-mutate/close/takeover critical section.
Takeover state is durable, idempotent, and auditable through the active claim
and immutable `.lock_history/` artifacts. Displaced writers fail ownership
checks.

Automatic stale recovery never uses age and fails closed on ambiguity. Any LIVE
source refuses; conclusive SLURM DEAD suffices for a SLURM owner with an
uncheckable remote PID; same-host PID DEAD can establish a non-SLURM owner is
dead and may resolve unknown scheduler state. Operator recovery requires a
nonblank reason and records `operator_unlock`.

Retry policy has exactly three attempts and backoffs `[0, 30, 120]`.
`attempt_failed` authorizes only a subsequent retry, and
`attempt_interrupted` is restricted to `interrupted_process`.
Final/nonretryable failures publish a terminal infrastructure result followed
by completion; retryable attempt-3 interruption remains
terminalization-required until that publication. Transient CUDA retry requires
worker termination and a fresh process. Every retry preserves seed and logical
identity.

Resume is idempotent at natural and checkpoint granularity. It counts durable
starts, reconciles orphan/completion anomalies, rejects corrupt provenance or
hierarchy, skips terminal keys, and never regenerates a successful natural
chain because a checkpoint is missing. Read-only retry planning derives its
category and consumed-attempt count only from exactly one retry-authorizing
closure on the latest persisted attempt; caller values are equality checks,
and `attempts_consumed` must be a true integer in `[0, 3]`.

The final correction chain also makes the six fixed Phase 1 configuration
oracles independent of their loaded templates, enforces the fixed structured
study and requested model contracts, accepts additional resolved effective
generation fields only when every requested field retains its exact JSON-typed
value, enforces RFC 3339 timestamps and complete result null/status matrices,
and requires a runtime lock capability for store mutation by default. The
fixed tail-entropy oracle is the arithmetic mean over the final
`max(1, ceil(0.10 * n_reasoning))` recognized reasoning tokens, with `null` for
zero reasoning tokens; even a self-consistent rehashed 20% manifest is rejected.

## Manifest hierarchy status

The two-level provenance design is implemented as schema/validation support,
not as concrete Phase 2/3 artifacts:

1. The tracked question manifest and tracked model-independent study manifest
   remain Phase 2 outputs under `manifests/part1/`; neither exists yet.
2. The operational model-run-manifest schema exists. No production instance
   exists. Phase 3 may create one only after the final production commit and a
   clean tracked-worktree gate, under the ignored persistent production root.

Smoke and production roots are distinct and ignored. Phase 1 configuration
rejects production mode, ambiguous roots, explicit/expanded ephemeral roots,
and smoke/production aliasing.

## Verification evidence

The final independent re-review of `5969080` inspected the complete correction
chain through `e6e0dee`, `1ca5661`, `260fb44`, `12dd6a9`, and `5969080`. It
found no remaining Critical, Important, Minor, P0, P1, P2, or P3 Phase 1 code
finding. Fresh reported evidence was:

- `uv run pytest -q` — **265 passed**;
- focused tail/fixed-contract compatibility slice — **12 passed, 168
  deselected**;
- real two-process local POSIX `flock` regression — **1 passed**;
- Python compile and JSON parsing checks — exit 0;
- CLI help and default dry run — exit 0, read-only, valid, and
  non-production;
- diff/scope/static-import scans — clean; and
- tracked worktree clean on `codex/phase1-infrastructure` before this
  documentation refresh.

Coverage includes:

- canonical bytes, all hash identities, exclusions, and stable seed vectors;
- schemas and complete outcome/nullability matrices;
- all five terminal commit crash boundaries;
- authoritative-result and completion-without-result handling;
- orphan counting and terminalization-required states;
- exact-byte tail recovery, recovery crash boundaries, malformed-middle
  rejection, and finalization;
- full-precision persistence and token/entropy/A–D alignment;
- duplicate/conflict and checkpoint-parent rejection;
- ties-to-even checkpoint placement and cross-record physical alias coherence;
- writer contention, liveness, operator recovery, every takeover durability
  boundary, and a real two-process local POSIX `flock` regression;
- directory-entry durability, retry-evidence/count/category authority,
  retry taxonomy/backoff/fresh-process CUDA policy, fixed-contract oracles,
  RFC 3339 timestamps, and complete confidence/null-status matrices; and
- manifest compatibility, resume, idempotent rerun, and smoke/production path
  separation.

All evidence was synthetic and login-safe. It is not model, dataset, GPU,
filesystem-on-Mila, or real-scheduler validation.

## Current output contract

A bound active shard may contain:

```text
<shard-root>/
  .shard-provenance.json
  natural_results.jsonl
  checkpoint_results.jsonl
  audit_events.jsonl
  recovery_journal/<event-id>.json
  quarantine/<stream>.<sha256>.trailing-bytes.bin
  .writer.guard
  .writer.lock
  .writer-lock-recovery.claim
  .lock_history/<claim-id>.claim.json
  .lock_history/<claim-id>.event.json
  .finalized
```

Some entries exist only while a lock/takeover is active or after a particular
recovery. Validation reports are written externally at an explicitly supplied
path; the configured directory name is `validation_reports`.

## Phase 2 blockers and risks

There is no remaining repository-code blocker within Phase 1. The next phase is
gated on:

1. resolving immutable MMLU dataset revision/source-row identity and creating
   the tracked fixed question/study manifests on a compute node where required;
2. resolving immutable model/tokenizer revisions, prompt/tag/token conventions,
   and effective generation settings through SmolLM3 compute-node preflight;
3. confirming directory `fsync`, no-overwrite hard-link publication, atomic
   replacement, and stable POSIX `flock` behavior on the selected Mila
   persistent filesystem;
4. confirming the actual Mila `squeue` array selector/output and completed-job
   absence semantics used by the fail-closed liveness probe;
5. integrating generation only through `LockedShardSession`, complete compatible
   manifests, policy-complete events/results, canonical seeds, and same-seed
   retry; and
6. ensuring transient CUDA retry exits to a fresh process and smoke artifacts
   remain separate from production.

No production model-run manifest or production generation is permitted at this
boundary.

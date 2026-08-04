# Part 1 status

## Executive status

**Phase 2 is paused during bounded Smoke A on branch
`codex/phase1-infrastructure`.** CPU materialization and manifest validation,
single-L40S SmolLM3 preflight, the Mila persistent-filesystem and scheduler
liveness gates, and same-environment GPU reproducibility have passed. Phase 2
is not complete.

At the explicit pause snapshot `2026-08-04T18:52:39-04:00`, Smoke A SLURM job
`10284742` was `RUNNING` for `33:14` on `cn-l018`. Durable files contained 7
natural results, 66 checkpoint results, and 146 audit events. Monitoring stopped
at that snapshot; the job was not cancelled. Its later state is deliberately
unknown here. Smoke A remains **running or awaiting post-job validation** and
must not be marked passed until its terminal SLURM state, runner report, and
completed artifacts are independently validated. Smoke B has not been
submitted, and no other job should be submitted while this pause is in effect.

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
- The execution snapshot is based on Git commit
  `e19edf22c8a3f7462ab66d5cae11b38247df5ed9` (`e19edf2`), which contains the
  scoped RNG correction used by the passing reproducibility job. The fresh
  local suite at this commit was **367 passed**.

## Phase 2 operational ledger

- CPU materialization job `10284018` produced and validated the exact tracked
  500-question bundle: five ordered 100-question subject blocks, without
  replacement, using question-sampling seed 42. The tracked manifest commit is
  `2e0bcae`.
- The current preflight/provenance artifacts were refreshed by job `10284702`,
  which completed `0:0` on one L40S at the current commit. They bind both model
  and tokenizer to immutable revision
  `a07cc9a04f16550a088caea529712d1d335b0ac1`.
- Corrected reproducibility job `10284721` completed `0:0` with status
  `passed`: exact generated-token equality, exact parser equality, and exact
  entropy-array equality at tolerance `0.0`, using canonical seed
  `2552280803631819986`. Its report is
  `results/part1-smoke/reproducibility/14c49484a4eebdb79372cb14b3e0076812e983d688c49aa7e3c2280bb44be7c0/reproducibility_report.json`.
  Earlier job `10284623` was a diagnostic failed attempt before the RNG
  correction and is not the current result.
- The Mila persistent-filesystem gate passed all four focused tests at
  `results/part1-smoke/mila-filesystem-gate.EUEXDB`, including directory
  `fsync`, no-overwrite hard-link publication, atomic replacement, and
  two-process POSIX `flock` behavior.
- The scheduler liveness gate passed for live exact output. An immediately
  completed job produced empty output with return code 0 and was classified
  `DEAD`; a later purged job produced return code 1 and remains `UNKNOWN` under
  the fail-closed policy. A purged `squeue` error must never be treated as
  evidence that an owner is dead.

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

The two-level provenance design now has concrete non-production Phase 2
artifacts while retaining the production gate:

1. The tracked question JSONL, question sidecar, and model-independent study
   manifest exist under `manifests/part1/`, were produced by job `10284018`,
   independently validated, and were committed in `2e0bcae`.
2. Non-production model-run manifests exist under
   `results/part1-smoke/model-runs/` for preflight/reproducibility/smoke use.
   No production instance exists. Phase 3 may create one only after the final
   production commit and a clean tracked-worktree gate, under the ignored
   persistent production root.

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

The evidence listed in this subsection is the historical Phase 1 synthetic
evidence. The separate Phase 2 operational ledger above records the completed
Mila dataset, GPU, filesystem, scheduler, and reproducibility gates.

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
  ..writer.lock.<lock-id>.pending
  .<target-name>.<uuid>.tmp
  .lock_history/<claim-id>.claim.json
  .lock_history/<claim-id>.event.json
  .lock_history/<claim-id>.pending-quarantine
  .lock_history/.<target-name>.<uuid>.tmp
  .finalized
```

Some entries exist only while a lock/takeover is active or after a particular
recovery. The `..writer.lock.<lock-id>.pending` replacement exists only during
a pending takeover and is reused or resumed idempotently after a crash. A
conflicting pending replacement may be preserved as
`.lock_history/<claim-id>.pending-quarantine`, which remains retained evidence.
Runtime exclusive publication may also leave a complete, uniquely named
`.<target-name>.<uuid>.tmp` in the shard root or `.lock_history/` after a
failure or crash following temporary-file fsync but before or around
no-overwrite hard-link publication. Such a temp is non-authoritative orphan
evidence: the authoritative target is absent or complete, never partial. It is
safe to leave because the state machines ignore it, and it must not be removed
manually while any writer or takeover may be active. Cleanup is a deliberate
post-liveness/operator-evidence step, not ordinary recovery.
Validation reports are written externally at an explicitly supplied path; the
configured directory name is `validation_reports`.

## Phase 2 pause boundary, blockers, and risks

The paused Smoke A artifacts are:

```text
job:            10284742
Git commit:     e19edf22c8a3f7462ab66d5cae11b38247df5ed9
shard:          results/part1-smoke/smoke_a/a332786f767d9f84c23ad0ddd057b46d5d3a8d7b266458d9b0352f5bf90ea374/shard-000
log:            logs/part1-smoke-a-10284742.out
model manifest: results/part1-smoke/model-runs/smoke_a/model_run_manifest.json
preflight:      results/part1-smoke/preflight/preflight.json
```

Resume from the Mila repository root without submitting anything:

```bash
ssh mila
cd /home/mila/c/chenje/my-project
squeue --jobs=10284742 --noheader --format=%i,%T,%M,%R
```

If `squeue` has no live row, obtain accounting evidence instead; do not infer
`DEAD` from a purged/error response:

```bash
sacct -j 10284742 \
  --format=JobIDRaw,State,ExitCode,Elapsed,NodeList \
  --noheader --parsable2
tail -n 80 logs/part1-smoke-a-10284742.out
wc -l \
  results/part1-smoke/smoke_a/a332786f767d9f84c23ad0ddd057b46d5d3a8d7b266458d9b0352f5bf90ea374/shard-000/natural_results.jsonl \
  results/part1-smoke/smoke_a/a332786f767d9f84c23ad0ddd057b46d5d3a8d7b266458d9b0352f5bf90ea374/shard-000/checkpoint_results.jsonl \
  results/part1-smoke/smoke_a/a332786f767d9f84c23ad0ddd057b46d5d3a8d7b266458d9b0352f5bf90ea374/shard-000/audit_events.jsonl
uv run python scripts/validate_part1_manifests.py
uv run python scripts/part1_dry_run.py \
  --mode smoke \
  --persistent-root results/part1-smoke \
  --study-manifest manifests/part1/study_manifest.json \
  --model-run-manifest results/part1-smoke/model-runs/smoke_a/model_run_manifest.json \
  --shard-root results/part1-smoke/smoke_a/a332786f767d9f84c23ad0ddd057b46d5d3a8d7b266458d9b0352f5bf90ea374/shard-000
```

For a successful Smoke A, require `COMPLETED` with exit `0:0`, the runner's
terminal JSON summary in the log, 10 natural terminal records and 110 checkpoint
terminal records, all ten run IDs for the same fixed question, and all eleven
requested checkpoint identities for every successful natural run (including
physical aliases). The dry run must confirm compatible manifests and hashes,
valid schemas/hierarchy/lifecycle, no invalid tail or pending recovery, no
terminalization requirement, no unresolved writer/takeover lock, and no
retry-required work. Counts must be stable after the job is terminal. Audit
events must be policy-complete; a line count alone is not sufficient.

Only after that independent post-job validation may Smoke A be marked passed
and Smoke B become eligible. Smoke B has not been submitted and must not be
submitted during this pause. Remaining Mila/GPU work is Smoke A terminal
validation followed, after explicit continuation, by bounded Smoke B and its
validation. Phase 3 remains deferred. The full 500-question experiment is
forbidden, and no production model-run manifest may be created.

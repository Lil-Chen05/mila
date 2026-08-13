# Part 1 status

## Executive status

**Recovery snapshot:** full-shape acceptance job `10347033` reached `TIMEOUT`
after `12:00:20` on 2026-08-12. It wrote the fixture in `58.304s` and completed
strict coverage in `18135.654s`, then timed out during merge. Its fail-closed
gate `10347034` was cancelled because `afterok:10347033` was unsatisfied. No
production model-run manifest, production receipt, or GPU array was created.
Historical operational evidence remains under
`results/part1-submission/96822f88c0a38bac4a35f29e90caf601831e7f50/`.
The user explicitly waived another full-shape synthetic rehearsal. Recovery
uses a one-hour CPU-only focused regression job followed by the existing exact
`afterok` production gate. See root `HANDOFF.md` for continuation.

**Phase 3 implementation is complete through Tasks 1–7 on branch
`codex/phase1-infrastructure`; Task 8 remains in progress after the successful
final bounded Phase 3 smoke.** The reviewed generation, validation, merge,
trajectory, statistics, bootstrap, and analysis code was deployed on Mila at
`a7e1135e85476f5fc43986949f467b10ba450623`. Job `10324103` completed `0:0` in
`01:27:13`. Its generated smoke model-run ID is
`a23087c8c0897bbf9f075b3edaa28c75b5087cffbcd203e2fea4bf16093b6dcf`.
The stable shard contains exactly 10 natural results, 110 checkpoint results,
and 240 audit events, and direct integrity validation passed. The hardened CLI
validator is implemented, independently approved, and passed 57 focused plus
291 regression tests. All completed smokes retain their historical acceptance;
strict live validation targets the actual production shards after generation.
Production readiness has not yet passed.

Pre-submission gates passed on Mila: the tracked tree was clean at the exact
commit above; `uv.lock` matched preflight SHA-256
`9cab4a125cc2bbf880efcd25826c6e4cdf9964889c7c29742cd10e96bc98db36`;
shell syntax and the no-collision checks passed; `MaxArraySize=1001`; manifest
validation returned exactly 500 questions and the expected stable hashes; and
the focused pure suite passed `34 passed in 1148.24s`. No model or dataset was
loaded on the login node.

The first optimized local production-shaped acceptance built all 500 shards but
failed coverage after `02:20:19` because synthetic natural rows carried a
content-derived prompt hash rather than the model-run manifest prompt hash. No
merge or analysis ran. The fixture-only correction passes a real one-shard scan
of 10 natural and 110 checkpoint rows with zero defects, five focused tests,
and 300 pipeline regressions. Corrected job `10347033` subsequently completed
strict coverage but timed out during merge after 12 hours. This established the
real scale cost without revealing a schema or integrity defect. It is retained
as historical evidence and is no longer a launch gate. The replacement
`jobs/part1_launch_readiness.sh` runs focused regressions while explicitly
excluding the full-shape marker; the conditional gate submits the `%16` chain
only after readiness passes. The exact local readiness command passed 797 tests
with one full-shape test deselected in `943.13s`. No production model-run
manifest or production root exists, and the full array has not yet been
submitted.

Recovery readiness job `10357631` exposed one cross-platform test-fixture
assumption: pytest places `tmp_path` under `/tmp` on Mila, so the ephemeral-root
guard correctly fired before the test's expected configured-root mismatch.
This is pure validation-test behavior and does not affect production code or
science. Gate `10357632` remains fail-closed; no production job was submitted.

The next readiness attempt, job `10362018`, passed the first corrected fixture
but exposed the same general issue in another `tmp_path` test. Readiness
`10362018` and gate `10362019` were cancelled after the failure was observed;
no production job was submitted. The job-level correction now binds pytest's
temporary root to the unique persistent-scratch path
`$SCRATCH/part1-launch-readiness-$SLURM_JOB_ID`, rather than modifying each
affected test individually.

A third attempt, readiness `10362106`, demonstrated that a global `$SCRATCH`
pytest root breaks the opposite class of tests that intentionally exercise
ephemeral-path refusal. It was cancelled and gate `10362107` did not run; no
production job was submitted. The final minimal readiness definition therefore
runs only the 16 launch-critical bootstrap, receipt, dependency, collision,
resource, and launch-plan tests. Immutable manifests and the clean commit remain
separately validated by the production gate before any GPU job; completed
smokes remain historical acceptance evidence.

Readiness `10362161` then passed all 16 launch-critical tests in `13.18s`, but
gate `10362162` failed safely before manifest creation because historical Smoke
A/B predate the `.finalized` lifecycle marker required by the hardened current
validator. No production manifest, receipt, or job was created. This prompted
an attempted Phase 3-only live gate; Smoke A/B remained historical evidence and
were not mutated or retrofitted to a newer storage contract.

Readiness `10362197` passed in `10s`; gate `10362198` then found that the
earlier Phase 3 smoke also predates the final prompt-hash contract. It failed
before manifest creation and produced no production receipt or job. Retained
smokes are no longer live launch inputs. Strict validation remains mandatory on
the actual generated production shards before merge and analysis.

The commit containing this checkpoint is the final tracked production
candidate; its exact SHA is captured by the unattended bootstrap receipt and,
only after focused readiness passes, by the production model-run manifest. The fresh
launch-critical local suite passed 257 tests in `570.14s`; the independent
orchestration review approved with no findings. The unrelated untracked
`METHODS_EXPERIMENTAL_DESIGN.md` remains user-owned and excluded.

On 2026-08-11 the user explicitly authorized the post-readiness full production
run with target array `0-499%16` and a four-day deadline. The scientific design
is unchanged at exactly 500 questions × 10 natural runs × 11 requested
checkpoints per successful natural run. Authorization does not waive or satisfy
the remaining gates and is not a claim that a production manifest exists, that
readiness passed, or that launch occurred.

Smoke A authoritative job `10284742` completed `0:0` in `01:26:28` on
`cn-l018`; its terminal runner state was `complete`. Smoke B authoritative job
`10292530` completed `0:0` in `00:25:14` on `cn-l072`; its terminal runner
state was `complete`. Both are non-production and independently passed
manifest, dry-run, and lifecycle validation with 0 errors and 0 warnings.

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
- Smoke A authoritative job `10284742` completed `0:0` in `01:26:28` on
  `cn-l018`, with terminal runner state `complete`, exactly one fixed question,
  natural run IDs `0`–`9`, requested checkpoint indices `0`–`10`, 10 natural
  results, 110 checkpoint results, and 240 audit events. Its manifest, dry run,
  and lifecycle validation passed with 0 errors and 0 warnings. No production
  root was created. All 10 natural records have
  `natural_execution_outcome=complete`, `reasoning_status=closed`,
  `answer_parse_status=parsed`, and `confidence_parse_status=malformed`. All
  110 checkpoint records have `checkpoint_execution_outcome=complete`;
  `checkpoint_model_output_status` is valid/invalid `93/17`,
  `answer_token_status` is located/missing `109/1`, and `entropy_status` is
  computed/unavailable `109/1`. This successful abnormal output was retained
  as data.
- Smoke B initial submission `10292499` failed in one second before Python,
  model, dataset, or artifact creation because a non-interactive SSH `PATH`
  omitted `~/.local/bin` and `srun` could not execute `uv`. It created no Smoke
  B root. This is diagnostic-only SLURM-readiness evidence, not a model,
  reproducibility, or experiment failure.
- Smoke B authoritative resubmission `10292530` used the unchanged documented
  sbatch job from a Mila login shell and completed `0:0` in `00:25:14` on
  `cn-l072`, terminal runner state `complete`. It covered sample indices
  `0`, `100`, `200`, `300`, and `400` (one fixed first question per subject),
  natural `run_id=0`, and requested checkpoint indices `0`–`10`: 5 natural
  results, 55 checkpoint results, and 120 audit events. Manifest, dry-run, and
  lifecycle validation passed with 0 errors and 0 warnings; no production root
  was created. All 5 natural records have `natural_execution_outcome=complete`;
  `reasoning_status` is closed/missing_close `4/1`, `answer_parse_status` is
  parsed/missing/out_of_domain `3/1/1`, and `confidence_parse_status` is
  malformed/missing `4/1`. All 55 checkpoint records have
  `checkpoint_execution_outcome=complete`; `checkpoint_model_output_status` is
  valid/invalid `42/13`, `answer_token_status` is located/missing `46/9`, and
  `entropy_status` is computed/unavailable `46/9`. This successful abnormal
  output was retained as data; there was no retry, terminalization,
  tail/recovery, lock, or takeover issue.

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

## Historical pre-completion Smoke A monitoring record

This is the pre-completion monitoring record, retained for provenance only; it
is not current operational instruction. The then-active Smoke A artifacts were:

```text
job:            10284742
Git commit:     e19edf22c8a3f7462ab66d5cae11b38247df5ed9
shard:          results/part1-smoke/smoke_a/a332786f767d9f84c23ad0ddd057b46d5d3a8d7b266458d9b0352f5bf90ea374/shard-000
log:            logs/part1-smoke-a-10284742.out
model manifest: results/part1-smoke/model-runs/smoke_a/model_run_manifest.json
preflight:      results/part1-smoke/preflight/preflight.json
```

The historical monitoring procedure was:

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

For historical acceptance, Smoke A required `COMPLETED` with exit `0:0`, the runner's
terminal JSON summary in the log, 10 natural terminal records and 110 checkpoint
terminal records, all ten run IDs for the same fixed question, and all eleven
requested checkpoint identities for every successful natural run (including
physical aliases). The dry run must confirm compatible manifests and hashes,
valid schemas/hierarchy/lifecycle, no invalid tail or pending recovery, no
terminalization requirement, no writer/takeover lock, and no
retry-required work. Counts must be stable after the job is terminal. Audit
events must be policy-complete; a line count alone is not sufficient.

That independent post-job validation passed, followed by authoritative Smoke B
validation. The `10292499` launch-path failure remains diagnostic-only Phase 3
SLURM-readiness hardening: non-interactive submission must expose `uv` on
`PATH`. This paragraph is historical Phase 2 evidence; the current Phase 3
checkpoint is recorded at the top of this file. The full 500-question
experiment remains unrun, and no production model-run manifest may be created
until every final Phase 3 gate passes.

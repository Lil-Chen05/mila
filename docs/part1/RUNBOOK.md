# Part 1 runbook

## Status and safety boundary

Phase 1's login-safe infrastructure remains implemented. Phase 2 is paused
during bounded Smoke A. At snapshot `2026-08-04T18:52:39-04:00`, SLURM job
`10284742` was `RUNNING` for `33:14` on `cn-l018`; the shard contained 7 natural
results, 66 checkpoint results, and 146 audit events. Monitoring stopped then,
without cancelling the job. Smoke A is still running or awaiting post-job
validation; it is not passed. Smoke B has not been submitted.

Never load a model, tokenizer, or Hugging Face dataset on a login node. Dataset
materialization runs in a CPU SLURM job. Model/tokenizer preflight and generation
run in a GPU SLURM job. This four-prompt sequence never launches the full
500-question experiment and this runbook provides no production submission
command.

Scientific settings are in [DECISIONS.md](DECISIONS.md), exact record contracts
in [SCHEMA.md](SCHEMA.md), phase gates in [PLAN.md](PLAN.md), current state in
[STATUS.md](STATUS.md), and evidence in [VALIDATION.md](VALIDATION.md).

## Persistent roots and artifact classes

- Run from the repository root with `uv`; never use `pip` or manually activate
  `.venv`.
- `$SLURM_TMPDIR` is ephemeral and forbidden for every persistent artifact.
- Jobs that can load Hugging Face artifacts export
  `HF_HOME=$SCRATCH/hf_cache` before access.
- `results/part1-smoke/` and `results/part1/` are separate, narrow ignored
  roots. Phase 1 allows only smoke-mode configuration validation and performs no
  generation.
- Tracked question JSONL/sidecar and study manifest live under
  `manifests/part1/`. They are authoritative, are not ignored, and were
  validated and committed in `2e0bcae`. The saved dataset under `data/part1/`
  is an ignored reproducible cache.
- Operational model-run manifests and raw shards are generated and ignored.
  No production model-run manifest exists; its Phase 3 creation follows the
  final production commit and clean-worktree gate.

Configuration validation refuses an omitted/mismatched root, smoke/production
aliasing, `/tmp`, `/private/tmp`, literal `$SLURM_TMPDIR`, and paths under its
expanded value.

## CPU dataset bootstrap — completed evidence

CPU materialization job `10284018` already completed and produced the exact
validated 500-question manifest bundle. Its submission command was:

```bash
sbatch jobs/materialize_part1_mmlu.sh
```

Do not run the Python entry point on a login node. The CPU job exports
`HF_HOME=$SCRATCH/hf_cache`, resolves `cais/mmlu@main` to an immutable
40-character commit SHA, and loads the test split separately for these exact
configs and order:

1. `high_school_mathematics`
2. `high_school_physics`
3. `high_school_chemistry`
4. `high_school_biology`
5. `high_school_psychology`

For each config it verifies the complete test-split size, uses streaming,
attaches the source row index before shuffling, shuffles with seed 42 and a
buffer equal to that verified split size, and takes exactly 100. It stages and
reloads the 500-record cache and all manifests before touching final paths.
Subject, revision, count, schema, identity, or hash failure publishes no final
manifest file. A successful first publication renames the complete staged
`part1` manifest directory once on the same filesystem. An existing complete
identical bundle is retained byte-for-byte; a partial or differing bundle is a
hard incompatibility. The job never invokes Git.

Expected successful artifacts:

```text
logs/materialize-part1-mmlu-<job-id>.out
manifests/part1/questions.jsonl
manifests/part1/questions.manifest.json
manifests/part1/study_manifest.json
data/part1/mmlu-<question_manifest_hash>/
```

The final log line is a compact JSON report containing the resolved revision,
verified source split counts, total count, three identities/hashes, publication
states, manifest paths, and cache path. A failure writes a compact JSON error
and exits 2.

The returned bundle was independently validated and committed in `2e0bcae`.
For later integrity checking, run the login-safe validator from the repository
root:

```bash
uv run python scripts/validate_part1_manifests.py
wc -l manifests/part1/questions.jsonl
git diff -- manifests/part1
git status --short
```

On Mila, additionally compare the ignored cache with the authoritative records
using the hash printed by the first command:

```bash
uv run python scripts/validate_part1_manifests.py \
  --dataset-cache data/part1/mmlu-<question_manifest_hash>
```

The materialization job must not be resubmitted as part of Smoke A monitoring.

## GPU sequence — current pause

GPU preflight, including provenance refresh job `10284702`, passed on one L40S
at Git commit `e19edf22c8a3f7462ab66d5cae11b38247df5ed9`. It records immutable
model/tokenizer revision `a07cc9a04f16550a088caea529712d1d335b0ac1` at:

```text
results/part1-smoke/preflight/preflight.json
```

Corrected reproducibility job `10284721` completed `0:0` and passed exact token,
parse, and entropy-array equality at tolerance 0.0 for seed
`2552280803631819986`. Its report is
`results/part1-smoke/reproducibility/14c49484a4eebdb79372cb14b3e0076812e983d688c49aa7e3c2280bb44be7c0/reproducibility_report.json`.
The Mila filesystem gate passed all four focused tests at
`results/part1-smoke/mila-filesystem-gate.EUEXDB`, and the real scheduler
liveness gate is approved with its documented fail-closed purged-job nuance.

Smoke A job `10284742` was already submitted. Do not cancel it, resubmit it, or
submit Smoke B or any other job while monitoring is paused. Do not submit the
full 500-question run.

## Resume Smoke A monitoring and validation

Reconnect and inspect the exact existing job from the Mila repository root:

```bash
ssh mila
cd /home/mila/c/chenje/my-project
squeue --jobs=10284742 --noheader --format=%i,%T,%M,%R
```

If the row is still present, record its current state and allow the existing job
to continue. If the row is absent, query accounting:

```bash
sacct -j 10284742 \
  --format=JobIDRaw,State,ExitCode,Elapsed,NodeList \
  --noheader --parsable2
```

Do not treat an `squeue` error for a purged job as `DEAD`. The validated Mila
semantics are: live exact output is `LIVE`; an immediately completed query with
empty output and return code 0 is `DEAD`; a later purged query returning code 1
is `UNKNOWN` and fails closed. Use `sacct`, the retained log, and process/lock
evidence as appropriate.

The frozen snapshot paths are:

```text
log:            logs/part1-smoke-a-10284742.out
model manifest: results/part1-smoke/model-runs/smoke_a/model_run_manifest.json
preflight:      results/part1-smoke/preflight/preflight.json
shard:          results/part1-smoke/smoke_a/a332786f767d9f84c23ad0ddd057b46d5d3a8d7b266458d9b0352f5bf90ea374/shard-000
Git commit:     e19edf22c8a3f7462ab66d5cae11b38247df5ed9
```

Inspect the retained terminal output and stable line counts:

```bash
tail -n 80 logs/part1-smoke-a-10284742.out
wc -l \
  results/part1-smoke/smoke_a/a332786f767d9f84c23ad0ddd057b46d5d3a8d7b266458d9b0352f5bf90ea374/shard-000/natural_results.jsonl \
  results/part1-smoke/smoke_a/a332786f767d9f84c23ad0ddd057b46d5d3a8d7b266458d9b0352f5bf90ea374/shard-000/checkpoint_results.jsonl \
  results/part1-smoke/smoke_a/a332786f767d9f84c23ad0ddd057b46d5d3a8d7b266458d9b0352f5bf90ea374/shard-000/audit_events.jsonl
```

When the job is terminal, first rerun the tracked-manifest validator, then use
the existing read-only shard/lifecycle inspection. Neither command loads a
dataset, tokenizer, or model:

```bash
uv run python scripts/validate_part1_manifests.py
uv run python scripts/part1_dry_run.py \
  --mode smoke \
  --persistent-root results/part1-smoke \
  --study-manifest manifests/part1/study_manifest.json \
  --model-run-manifest results/part1-smoke/model-runs/smoke_a/model_run_manifest.json \
  --shard-root results/part1-smoke/smoke_a/a332786f767d9f84c23ad0ddd057b46d5d3a8d7b266458d9b0352f5bf90ea374/shard-000
uv run python -c '
import collections, json
from pathlib import Path
root = Path("results/part1-smoke/smoke_a/a332786f767d9f84c23ad0ddd057b46d5d3a8d7b266458d9b0352f5bf90ea374/shard-000")
load = lambda name: [json.loads(line) for line in (root / name).read_text().splitlines()]
naturals = load("natural_results.jsonl")
checkpoints = load("checkpoint_results.jsonl")
expected_runs = set(range(10))
expected_checkpoints = set(range(11))
question_ids = {row["question_id"] for row in naturals}
by_run = collections.defaultdict(set)
for row in checkpoints:
    assert row["question_id"] in question_ids
    by_run[row["run_id"]].add(row["requested_checkpoint_index"])
assert len(naturals) == 10 and len(checkpoints) == 110
assert {row["run_id"] for row in naturals} == expected_runs and len(question_ids) == 1
assert all(row["natural_execution_outcome"] == "complete" for row in naturals)
assert set(by_run) == expected_runs and all(value == expected_checkpoints for value in by_run.values())
assert all(row["checkpoint_execution_outcome"] == "complete" for row in checkpoints)
print({"natural_results": 10, "checkpoint_results": 110, "shape": "passed"})
'
```

Smoke A passes only if all of the following hold:

1. SLURM accounting says `COMPLETED` with exit `0:0`, and the log contains the
   runner's terminal JSON summary rather than only partial progress.
2. Stable files contain exactly 10 natural terminal results and 110 checkpoint
   terminal results. All ten run IDs belong to the same fixed question; each
   successful natural has all eleven requested checkpoint identities, including
   aliases.
3. Manifest IDs/hashes and shard provenance match. All records pass schema,
   array-alignment, hierarchy, duplicate/conflict, and lifecycle checks.
4. There is no invalid or unterminated tail, pending recovery, missing terminal
   closure, terminalization requirement, active writer lock, or pending
   takeover. Audit evidence is policy-complete; its line count alone is not an
   acceptance test.
5. The dry-run report returns `is_valid=true`; the shape check confirms that
   every natural and checkpoint execution completed without infrastructure
   terminalization.

Only after independent review of all five checks may Smoke A be marked passed.
Smoke B then becomes eligible, but must not be submitted until the user resumes
this work explicitly.

## Implemented shard layout

An initialized shard is bound before its first stream append:

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

Files appear only when their state requires them. The
`..writer.lock.<lock-id>.pending` replacement exists only during a pending
takeover and is reused or resumed idempotently after a crash. A conflicting
pending replacement may be preserved as
`.lock_history/<claim-id>.pending-quarantine`, which is retained evidence.
Runtime exclusive publication may also leave a complete, uniquely named
`.<target-name>.<uuid>.tmp` in the shard root or `.lock_history/` after a
failure or crash following temporary-file fsync but before or around
no-overwrite hard-link publication. Such a temp is non-authoritative orphan
evidence: the authoritative target is absent or complete, never partial. Do not
remove it manually while any writer or takeover may be active. It is safe to
leave because the state machines ignore it; cleanup is a deliberate
post-liveness/operator-evidence step, not ordinary recovery.
`.writer.lock` and the active claim are normally removed when their operation
finishes; `.writer.guard` is a stable advisory-lock inode; history, recovery
journals, and quarantine evidence are retained. Validation reports are written
to an explicit external path; the configured report directory name is
`validation_reports`.

`.shard-provenance.json` immutably binds `study_id`, `model_run_id`, complete
`model_run_manifest_hash`, and `shard_id`. Mismatched or missing provenance
blocks use. `.finalized` blocks new lock acquisition/takeover and further raw
stream append or recovery.

## Login-safe Phase 1 commands

### Full pure test suite

```bash
uv run pytest -q
```

This is the primary repository gate. The final independently re-reviewed Phase
1 head, `5969080`, passed all 265 tests. The suite is synthetic and imports no
model/dataset execution path.

### Read-only dry run

Default template/schema validation:

```bash
uv run python scripts/part1_dry_run.py
```

The command emits one compact machine-readable JSON object. On success it
reports `is_valid=true`, `would_create_production_manifest=false`,
`imports_model_or_data_libraries=false`, and `mutation_performed=false`. It
does not create the configured root, manifest, shard, or output. Failure is
machine-readable and exits with status 2.

Options are:

```text
--mode {smoke,production}
--persistent-root PATH
--study-manifest PATH
--model-run-manifest PATH
--shard-root PATH
--work-specs PATH
--retry-request PATH
```

Phase 1 production mode fails closed. `--persistent-root` is a validation
override and must still be a safe, separate smoke root. Study and model-run
manifests must be supplied together. `--work-specs` is a JSON array of complete
`WorkSpec` objects; `--retry-request` is one JSON object with `work`,
`category`, and `attempts_consumed`. Work/retry inputs require both compatible
manifests and a manifest-bound `--shard-root`.

For `--retry-request`, `attempts_consumed` must be a JSON integer from 0 through
3; booleans, floats, strings, negative values, and values above 3 are rejected
without conversion. The caller's category and attempt count are assertions,
not authorities. The report derives the operative category and count from the
maximum persisted `attempt_started` and exactly one coherent retry-authorizing
`attempt_failed` or `attempt_interrupted` closure on that latest attempt. A
pristine key, orphaned latest attempt, ambiguous closure, completed or terminal
key, exhausted state, active lock, pending takeover, or finalized shard is not
retry eligible. Requested and persisted values, latest attempt, closure count,
and blockers remain visible in the machine-readable report.

Example read-only inspection of later synthetic/smoke artifacts:

```bash
uv run python scripts/part1_dry_run.py \
  --study-manifest <study.json> \
  --model-run-manifest <model-run.json> \
  --shard-root <shard-root> \
  --work-specs <work-specs.json> \
  --retry-request <retry-request.json>
```

The report includes compatibility, raw validation, tails/pending recoveries,
lock owner, takeover state/history, finalization, consumed attempts, orphaned
attempts, resume classifications, and retry planning. It never reconciles or
recovers. Corruption, incompatible identity/seed/hash, invalid nested shard
validation, missing provenance, finalized work needing mutation, or ineligible
retry makes the top-level report invalid and exits nonzero.

### Explicit operator unlock

Start a newly reasoned operator takeover:

```bash
uv run python scripts/part1_operator_unlock.py \
  --shard-root <shard-root> \
  --study-id <64-hex-study-id> \
  --model-run-id <64-hex-model-run-id> \
  --model-run-manifest-hash <64-hex-complete-hash> \
  --shard-id <shard-id> \
  --reason "<verified operator justification>"
```

Finish an already durable pending takeover, supplying a new operator reason:

```bash
uv run python scripts/part1_operator_unlock.py \
  --shard-root <shard-root> \
  --finish-pending \
  --reason "<verified operator justification>"
```

`--reason` is always required and must be nonblank. `--finish-pending` derives
the immutable identities from the durable claim, so study/model/shard/hash
arguments are not required in that mode. Success emits the event ID/type/shard
as compact JSON. Do not delete `.writer.lock`, claims, pending files, or history
by hand.

## Writer ownership and stale recovery

Production integration must acquire `LockedShardSession`; mutating
`Part1ShardStore` requires a runtime lock capability by default. Only synthetic
tests may explicitly opt into `unsafe_for_tests=True`. Lock metadata contains
lock ID, study/model-run/shard, hostname, PID, optional SLURM job/array task
IDs, and RFC 3339 acquisition time.

The stable `.writer.guard` uses reentrant process/thread serialization plus
POSIX `flock`. It spans ownership validation and the complete append, tail
recovery, report creation, finalization, close, or takeover. A second writer
waits/refuses and a displaced lease cannot mutate or delete a replacement lock.

Automatic stale recovery follows evidence, never age:

| Owner/evidence | Decision |
|---|---|
| worker LIVE or SLURM LIVE | refuse |
| SLURM owner; scheduler conclusively DEAD | recoverable even when remote PID is unknown |
| SLURM owner; scheduler UNKNOWN; same-host PID DEAD | recoverable |
| non-SLURM owner; same-host PID DEAD | recoverable |
| remote non-SLURM owner, probe error/timeout/missing command, conflicting output, or other ambiguity | refuse |

The default scheduler probe is
`squeue --jobs=<job>[_<array-task>] --noheader --format=%i`, with a timeout. A
successful empty result is DEAD, the exact selector alone is LIVE, and every
error/other result is UNKNOWN. Tests inject the runner; Phase 1 did not call
real Slurm. Confirm this behavior on Mila before relying on automatic takeover.

Takeover durably creates a unique claim and history record before replacement.
Control files are published complete using fsynced same-directory temporary
files and no-overwrite hard links; the authoritative target is therefore absent
or complete, never partial. Pre/post-replacement crashes, partial event append,
event-before-cleanup, and pending-replacement states are resumable.
Conflicting pending bytes fail automatic recovery; a reasoned operator path
quarantines them before continuing. The active claim is removed last.

First creation of a stream/control/report/finalization file fsyncs both the
file and its containing directory. Creating a missing shard-root or history
directory component also fsyncs the parent directory entry. Before a Mila GPU
smoke relies on this protocol, exercise directory `fsync`, no-overwrite hard-link
publication, atomic replacement, and two-process POSIX `flock` on the selected
Mila persistent filesystem.

## Result, event, retry, and resume ordering

For every successful or terminal outcome:

1. `attempt_started` is already durable and has consumed the attempt number.
2. Append the terminal natural/checkpoint result.
3. Flush and fsync the result stream.
4. Append matching `attempt_completed`.
5. Flush and fsync the audit stream.

If a crash occurs after step 3, the result is authoritative: resume must not
retry it, adds `terminal_result_recovered`, and may add the missing completion.
Completion without result and a started attempt without any terminal event are
classified interrupted and consume the attempt.

There are exactly three attempts. Retryable categories are interrupted process,
temporary filesystem failure, transient worker failure, and transient CUDA
runtime failure. Terminal categories are invalid configuration, schema or
manifest incompatibility, tokenizer-preflight incompatibility, deterministic
context overflow, reproducible CUDA OOM, unsupported model/tokenizer behavior,
and corrupt immutable manifest. Backoffs are `[0, 30, 120]` indexed by attempt.

`attempt_failed` is written only when another retry is authorized. A
nonretryable current attempt or exhausted third attempt writes the terminal
infrastructure-failure result followed by completion. An exhausted interruption
is `terminalization_required` until that result is durable. CUDA retry never
continues in the failed process. `attempt_interrupted` is reserved for the
`interrupted_process` category; terminal categories cannot be represented by a
fabricated interruption and instead use result-first terminalization.

Before work, resume finishes durable recovery evidence as appropriate,
reconciles orphaned lifecycle states, validates shard/manifests/hierarchy,
counts starts, and classifies each complete `WorkSpec`. It skips every terminal
key. Checkpoints require a complete eligible parent with exact ID, seed,
provenance, sample/subject, and checkpoint membership. A completed natural
chain is independent of later checkpoint absence/failure. Checkpoint preflight
recomputes ties-to-even placement, actual fraction, prefix/shared-probe
identity, alias membership/ownership, and alias flag from the natural parent;
all aliases of one physical prefix must agree on prefix hash, inducer
version/text, and shared-probe ID.

## Trailing-line recovery and validation

Only the final physical line is recoverable:

1. inspect and reject any malformed complete/middle record;
2. for an invalid tail, persist the exact bytes under `quarantine/`;
3. fsync immutable `recovery_journal/<event-id>.json` evidence;
4. verify prefix/tail lengths and SHA-256 values;
5. truncate only the evidenced invalid bytes, or append only `\n` when a valid
   final JSON object lacks it;
6. append the matching `trailing_line_recovered` audit event; and
7. preserve the journal and quarantine permanently.

Every boundary is idempotently resumable. Pending/invalid recovery journals,
tails, duplicate/conflicting results/events, schema/nullability errors,
scientific-array misalignment, lifecycle/hierarchy errors, or required
terminalization block finalization. A valid result missing only completion is
authoritative but remains a validation warning until reconciled; finalization
requires full terminal/event consistency.

Validation also enforces RFC 3339 event/report timestamps, the exact fixed
configuration/study/requested-model contracts, required requested values in
effective generation settings, confidence boundary/null matrices, and the
structured final-10%-of-reasoning tail-entropy rule. Recomputing IDs and hashes
does not make drifted science compatible.

## Remaining Phase 2/3 lifecycle

1. CPU materialization, tracked-manifest validation/commit, GPU preflight,
   filesystem/scheduler checks, and corrected reproducibility are complete.
2. Resume observation of existing Smoke A job `10284742`; do not resubmit it.
   Validate its terminal accounting, log, counts, provenance, schemas,
   lifecycle, hierarchy, tails, recovery, and lock state before marking it
   passed.
3. After explicit continuation and only if Smoke A passes, run the bounded
   Smoke B job under `results/part1-smoke/` and validate it independently.
4. Phase 3 adds analysis, complete raw validation, validate-before-publish
   merge, SLURM readiness, and its final bounded smoke.
5. Final tracked production artifacts are committed; the tracked worktree must
   be clean.
6. Only then is the operational production model-run manifest generated under
   `results/part1/<model_run_id>/`, after which Git must remain clean.
7. Stop. Full production submission requires separate authorization.

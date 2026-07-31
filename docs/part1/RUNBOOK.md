# Part 1 runbook

## Status and safety boundary

Phase 1's login-safe contract, storage, lock, retry, resume, dry-run, and
operator tools are implemented. Phase 2/3 model, dataset, generation,
validation/merge, analysis, and SLURM entry points remain deferred unless
explicitly marked otherwise below.

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
- Tracked question JSONL/sidecar and study manifest will live under
  `manifests/part1/` after Phase 2. They are not ignored and do not yet exist.
- Operational model-run manifests and raw shards are generated and ignored.
  No production model-run manifest exists; its Phase 3 creation follows the
  final production commit and clean-worktree gate.

Configuration validation refuses an omitted/mismatched root, smoke/production
aliasing, `/tmp`, `/private/tmp`, literal `$SLURM_TMPDIR`, and paths under its
expanded value.

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
  .lock_history/<claim-id>.claim.json
  .lock_history/<claim-id>.event.json
  .lock_history/<claim-id>.pending-quarantine
  .finalized
```

Files appear only when their state requires them. `.writer.lock` and the active
claim are normally removed when their operation finishes; `.writer.guard` is a
stable advisory-lock inode; history, recovery journals, and quarantine evidence
are retained. Validation reports are written to an explicit external path; the
configured report directory name is `validation_reports`.

`.shard-provenance.json` immutably binds `study_id`, `model_run_id`, complete
`model_run_manifest_hash`, and `shard_id`. Mismatched or missing provenance
blocks use. `.finalized` blocks new lock acquisition/takeover and further raw
stream append or recovery.

## Login-safe Phase 1 commands

### Full pure test suite

```bash
uv run pytest -q
```

This is the primary repository gate. Phase 1 concluded with 188 passing tests.
The suite is synthetic and imports no model/dataset execution path.

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

Production integration must acquire `LockedShardSession`; direct unguarded
store use is for internal read/testing only. Lock metadata contains lock ID,
study/model-run/shard, hostname, PID, optional SLURM job/array task IDs, and RFC
3339 acquisition time.

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
files and no-overwrite hard links. Pre/post-replacement crashes, partial event
append, event-before-cleanup, and pending-replacement states are resumable.
Conflicting pending bytes fail automatic recovery; a reasoned operator path
quarantines them before continuing. The active claim is removed last.

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
continues in the failed process.

Before work, resume finishes durable recovery evidence as appropriate,
reconciles orphaned lifecycle states, validates shard/manifests/hierarchy,
counts starts, and classifies each complete `WorkSpec`. It skips every terminal
key. Checkpoints require a complete eligible parent with exact ID, seed,
provenance, sample/subject, and checkpoint membership. A completed natural
chain is independent of later checkpoint absence/failure.

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

## Phase 2/3 lifecycle, still deferred

1. Phase 2 CPU job resolves/materializes the fixed MMLU sample using streaming
   plus bounded selection, then the tracked question/study manifests are
   inspected and committed.
2. Phase 2 GPU compute preflight resolves immutable SmolLM3/tokenizer revisions,
   tags, prompt, inducer, A–D convention, environment, and effective settings.
3. Only an explicitly authorized bounded smoke runs under
   `results/part1-smoke/`, using Phase 1 locks/events/storage/resume.
4. Phase 3 adds full raw validation, validate-before-publish merge, analysis,
   and SLURM launchers, followed by its bounded smoke.
5. Final tracked production artifacts are committed; the tracked worktree must
   be clean.
6. Only then is the operational production model-run manifest generated under
   `results/part1/<model_run_id>/`, after which Git must remain clean.
7. Stop. Full production submission requires separate authorization.

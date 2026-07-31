# Part 1 validation ledger and acceptance matrix

## Evidence rules

Phase 1 validation is pure, synthetic, and login-safe. It may exercise JSON,
filesystem crash boundaries, process locking, liveness decisions with injected
probes, retry planning, and resume logic. It must not load a model, tokenizer,
dataset, torch weights, or CUDA, invoke real Slurm, create an operational
production manifest, or generate experiment output.

Passing Phase 1 tests establishes the repository contract. It does not establish
SmolLM3 behavior, MMLU provenance, Mila filesystem locking, real scheduler
semantics, compute-node execution, or production readiness. Those remain
resource-appropriate Phase 2/3 gates.

## Prompt 1 and Phase 1 ledger

### Baseline preflight

Before production-code changes, the coordinator:

- re-read `AGENTS.md` and all six authoritative Part 1 documents;
- inspected Git status/history/diffs and verified the Prompt 1 commits;
- verified `.superpowers/` remained unmodified and excluded only through
  `.git/info/exclude`;
- compared `STATUS.md` to the implementation; and
- ran `uv run pytest -q`: **21 passed**.

The tracked worktree was clean. No model/data action occurred.

### Contracts gate

Commits `0b7b6f4` and `8ce4bbf` implemented six templates, eight Draft 2020-12
schemas, canonical serialization, all required IDs/hashes, shared probe and
audit scopes, seeds, schema/nullability validation, path guards, and golden
vectors.

Final focused result at this gate:

```text
uv run pytest -q tests/test_part1_contract.py
26 passed
```

JSON parsing, compilation, whitespace, diff, ignore, forbidden-import, and
selected-token-log-probability scans passed. Independent re-review approved the
corrected slice with no remaining P0–P3 finding.

### Storage gate

Commits `10d2f95`, `34c4c90`, and `a850ee2` implemented normalized append-only
streams, durable result/event ordering, full lifecycle indexing, exact-byte
tail recovery, immutable recovery journals, validation reports, hierarchy/
alignment checks, and finalization.

Final focused result at this gate:

```text
uv run pytest -q tests/test_part1_store.py
37 passed
```

Independent final storage review approved the slice. In particular, public
terminal append requires a prior durable matching start and creates no result
bytes when that precondition fails.

### Runtime gate

Commits `c066bff`, `d15ba85`, and `68926ec` implemented guarded sessions,
takeover durability, stale/operator recovery, retry policy, terminalization,
parent/provenance enforcement, resume, dry run, and operator CLI.

The final independent closure review ran:

```text
uv run pytest -q tests/test_part1_runtime.py -k \
  'atomic_exclusive_create or pending_replacement_file or \
   corrupt_pending_replacement or ordinary_post_replacement or \
   every_takeover_claim_state or finalized_dry_run or \
   dry_run_rejects_work_or_retry or posix_flock'
11 passed, 86 deselected

uv run pytest -q
188 passed

git diff --check 68926ec^ 68926ec
exit 0

git status --short --branch
## codex/phase1-infrastructure
```

The reviewer approved `68926ec` with no remaining P0, P1, or P2 finding. The
real two-process local POSIX regression showed that a second process could not
complete takeover while the first held `.writer.guard`.

### CLI/static gates

The implementation reports additionally recorded success for:

```bash
uv run python -m compileall -q \
  scripts/part1_contract.py \
  scripts/part1_store.py \
  scripts/part1_failure_policy.py \
  scripts/part1_runtime.py \
  scripts/part1_dry_run.py \
  scripts/part1_operator_unlock.py

uv run python scripts/part1_dry_run.py --help
uv run python scripts/part1_dry_run.py
uv run python scripts/part1_operator_unlock.py --help

find configs/part1 schemas/part1 -type f -name '*.json' -exec jq empty {} +
```

The default dry run returned success, `is_valid=true`, and
`mutation_performed=false`. No output root or manifest was created.

## Phase 1 acceptance coverage

| Area | Verified behavior |
|---|---|
| Canonical bytes | Recursive key sorting, ordered arrays, line-ending normalization, direct UTF-8 Unicode, compact envelope, finite-only values, no trailing newline, and version separation. |
| Hash identities | Golden vectors for question content/ID, question manifest, study ID/hash, model-run ID/hash, natural/checkpoint/shared-probe records, attempts, and attempt/shard audit events; self/mutable/path/time exclusions. |
| Validation-report identity | Independent immutable target payload; mutable timestamps/results do not change its ID. |
| Seeds | `part1-seed-v1` golden values, deterministic first-eight-byte big-endian conversion/mask, range, and model/question/run/version separation. |
| Schemas | All eight Draft 2020-12 schemas load; required fields, enums, no-extra-property rules, confidence arithmetic, index/fraction relation, and successful/invalid/failure nullability pass negative tests. |
| Lifecycle separation | Only terminal records in result streams; exactly eight event types and two scopes; starts consume sequential attempts; event references/transitions checked. |
| Five commit crash boundaries | Before result append, during result append, after result fsync/before completion, during completion append, and after both fsyncs. |
| Result authority | Durable result without completion is not retried; recovery evidence and optional completion are idempotent. Completion without result is interruption. |
| Orphans/terminalization | Orphan starts are interrupted and counted. Nonretryable/final retryable failures require a terminal result; exhausted interruption remains terminalization-required until published. |
| Raw append/recovery | Per-record fsync; valid bytes never overwritten; only final line repaired; exact-byte quarantine, immutable journals, valid-missing-newline append repair, malformed-middle rejection, and recovery crash matrix. |
| Scientific persistence | Full-precision finite values; generated-token/entropy and answer-token alignment; four A–D vectors/probability sum/entropy/max checks; selected-token log probabilities absent. |
| Duplicates/hierarchy | Duplicate/conflicting terminal/event IDs rejected; one complete eligible natural parent with exact ID/seed/provenance/checkpoint membership required. |
| Finalization/reports | Machine-readable stable-target reports; pending tails/recoveries, lifecycle/hierarchy errors, and terminalization block finalization; `.finalized` blocks mutation. |
| Exclusive writer | Second owner rejection, guarded mutation/report/finalize/close/takeover, displaced-writer refusal, and local two-process POSIX `flock` regression. |
| Takeover durability | Every claim/replacement/event/cleanup crash boundary, partial event, exact pending reuse, conflicting-pending quarantine with operator reason, one-event idempotence, and active-claim-last cleanup. |
| Liveness | Any LIVE refuses; conclusive SLURM DEAD and same-host PID DEAD rules; remote/ambiguous/error/timeout/missing-command refusal; age never used. |
| Operator recovery | Nonblank reason required; prior owner archived; `operator_unlock` durable; `--finish-pending` completes a durable claim. |
| Retry policy | Exact category lists, attempts 1–3, backoffs `[0,30,120]`, same identity/seed, nonretryable current-attempt terminalization, and fresh-process CUDA guard. |
| Resume/idempotence | Natural/checkpoint granularity, completed skip, parent eligibility, orphan reconciliation once, same full resubmission, manifest/hash/seed mismatch rejection, finalized-work refusal. |
| Path separation | Smoke/production roots are separate and narrowly ignored; ephemeral and aliased roots fail closed; historical tracked outputs are unaffected. |

## Durable ordering invariants under validation

For each terminal logical key:

1. exactly one durable `attempt_started` consumes the attempt number;
2. no more than one terminal result exists;
3. result append and fsync precede completion append and fsync;
4. a result is authoritative when completion is absent;
5. `attempt_completed` must reference the exact terminal result and attempt;
6. `attempt_failed` is retry-only; terminal policy closes through a terminal
   infrastructure-failure result and completion; and
7. checkpoint terminals have an exact complete eligible natural parent.

Validation fails on missing starts, completion without a result, orphaned
attempts, contradictory transitions, nonsequential attempts, retry-policy
metadata drift, parent/provenance inconsistency, duplicate results, or pending
terminalization. The single authoritative-result/missing-completion case is a
warning until reconciliation; it never authorizes retry.

## Remaining Phase 2 acceptance matrix

| Area | Environment | Required evidence |
|---|---|---|
| MMLU revision/selection | CPU compute job | Immutable revision, streaming + bounded take, exactly five ordered 100-question blocks, no replacement, indices 0–499, seed 42, stable source-row identities and hashes. |
| Question/study manifests | Compute output + login-safe validator | Tracked outside ignored data, exact science, recomputed IDs/hashes, and no mutable/unresolved source facts. |
| SmolLM3 preflight | Single-GPU compute job | Immutable model/tokenizer revisions, bf16/eval/batch-one, tags/tokens, prompt, inducer, A–D convention, answer-step location, environment, and requested/effective settings. |
| Mila lock operation | Selected persistent filesystem | Confirm two-process POSIX `flock` exclusion on the actual target filesystem. |
| Mila scheduler liveness | Mila shell/compute context | Confirm `squeue --jobs=<job>[_<array>] --noheader --format=%i`, exact live output, completed/absent output, permissions, return codes, and timeout behavior before automatic takeover. |
| Runtime integration | Pure mocks + bounded smoke | All writes through `LockedShardSession`; complete manifests/WorkSpec hashes; policy-complete start/failure/result/completion events; same-seed retry; fresh process after transient CUDA. |
| Natural generation | Authorized bounded GPU smoke | Ten configured stochastic run identities when the smoke scope calls for them, no extra greedy run, fixed settings, raw pre-warper float32 entropy, exact token alignment, abnormal-output retention. |
| Checkpoints | Authorized bounded GPU smoke | Greedy probes, all eleven requested identities, alias sharing, forced outputs, A–D/full-vocabulary metrics and orthogonal statuses. |
| Smoke separation | Pure validation | Non-production manifest/output under ignored smoke root; optional dirty-code base commit/diff hash; no production manifest/output. |

Phase 2 cannot pass on mocked tests alone. It requires only the explicitly
authorized bounded compute preflight/smoke, never the full experiment.

## Phase 3 and production-manifest gate

Analysis, bootstrap/calibration, complete raw validation, validate-before-
publish merge, SLURM launchers, and end-to-end bounded smoke remain Phase 3.
Only after final tracked production artifacts are committed and the tracked
worktree is clean may the operational production model-run manifest be created
under the ignored persistent production root. The manifest must record the
final commit and its creation must leave Git clean. This readiness evidence is
not authorization to launch the 500-question production run.

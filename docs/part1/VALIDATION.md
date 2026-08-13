# Part 1 validation ledger and acceptance matrix

## Full-shape timeout and approved recovery

Final candidate `96822f88c0a38bac4a35f29e90caf601831e7f50` passed Mila
manifest, shell, clean-tree, and unattended-submission preflight. CPU
full-shape job `10347033` completed fixture creation and strict coverage, then
timed out during merge after `12:00:20`. Gate `10347034` was cancelled by exact
`afterok:10347033`; no production manifest or GPU array was created. The user
explicitly waived repeating this synthetic scale rehearsal. A one-hour focused
regression job excluding `part1_full_acceptance` replaces it as the pre-launch
readiness dependency; real post-generation validation remains mandatory.

## Phase 3 post-smoke gate

Phase 3 Tasks 1–7 are implemented and independently reviewed at
`a7e1135e85476f5fc43986949f467b10ba450623`. Final bounded generation smoke job
`10324103` completed `0:0` in `01:27:13`. Its smoke model-run ID is
`a23087c8c0897bbf9f075b3edaa28c75b5087cffbcd203e2fea4bf16093b6dcf`.
Its stable shard contains exactly 10 natural results, 110 checkpoint results,
and 240 audit events, and direct terminal-state, shape, immutable provenance,
schema, hierarchy, lifecycle, and policy-complete audit validation passed. The
hardened CLI-validator acceptance gate remains pending. Therefore Phase 3,
production launch readiness, and final-paper readiness are still explicitly
**not yet passed**.

The submission gate itself passed on Mila:

| Gate | Evidence |
|---|---|
| Exact deployed commit and clean tracked tree | **PASSED** — `a7e1135e85476f5fc43986949f467b10ba450623` |
| Immutable manifest bundle | **PASSED** — 500 total, 100 per subject; question hash `dd379f48322d6eb07c309101361738a965be320b4124bb45bd44723b1abe474d`; study hash `859eecd5e0437a901555d5fd2d99692feccb5257df16de60bfe0fe648626142b` |
| Preflight dependency compatibility | **PASSED** — current and recorded `uv.lock` SHA-256 both `9cab4a125cc2bbf880efcd25826c6e4cdf9964889c7c29742cd10e96bc98db36` |
| Scheduler/collision/shell checks | **PASSED** — `MaxArraySize=1001`, no existing Phase 3 smoke roots, all relevant job scripts pass `bash -n` |
| Focused no-model Mila tests | **PASSED** — 34 passed in 1148.24 seconds |
| Final bounded Phase 3 smoke direct integrity | **PASSED** — job `10324103`, `COMPLETED`/`0:0`, `01:27:13`; terminal runner and exact 10 natural/110 checkpoint/240 audit shape directly validated |
| Hardened CLI-validator acceptance | **PENDING** — must accept the bounded Phase 3 smoke and required read-only Smoke A/B coverage with retained machine-readable evidence |
| Full-shape synthetic acceptance | **WAIVED AFTER SCALE EVIDENCE** — corrected job `10347033` completed full strict coverage with no reported defect, then timed out during merge at 12 hours; it will not be repeated under the four-day deadline |
| Focused recovery readiness, local | **PASSED** — `797 passed, 1 deselected in 943.13s`; the deselected test is exactly `part1_full_acceptance` |
| Focused recovery readiness, Mila first attempt | **FIXTURE FAILURE** — job `10357631` reached an environment-dependent assertion because Mila's pytest `tmp_path` is under forbidden `/tmp`; gate `10357632` remained fail-closed and no production job was submitted |
| Focused recovery readiness, Mila second attempt | **ENVIRONMENT FAILURE** — job `10362018` passed the first corrected fixture, then another `tmp_path` test hit the same forbidden-`/tmp` ordering; readiness `10362018` and gate `10362019` were cancelled, no production was submitted, and the job-level pytest basetemp was moved to unique `$SCRATCH` storage |
| Final launch-critical local suite | **PASSED** — 257 tests in `570.14s`, excluding only the explicit full-shape marker prepared for CPU SLURM |
| Unattended submission review | **PASSED BEFORE RECOVERY** — exact `afterok` bootstrap, exclusive receipts, race/crash reconciliation, and `%16` dependencies were independently approved; focused-readiness recovery requires fresh tests and review before submission |

Continuation starts with the hardened CLI-validator acceptance sequence in
[RUNBOOK.md](RUNBOOK.md), not with another smoke submission. No production
manifest or array may be created until focused readiness, hardened retained-smoke
validation, final documentation, full verification, independent review, final
commit, and the clean-tree gate all pass.

The full-shape flow is retained but no longer resubmitted. Measured runtime
raised the post-generation CPU ceilings to 12 hours for validation, 24 hours
for merge, and 36 hours for final analysis; these are operational resource
changes only and do not alter any metric or scientific contract.

The user explicitly authorized the post-readiness full production run on
2026-08-11, with target array `0-499%16` and a four-day deadline. The protocol
remains exactly 500 questions × 10 natural runs × 11 requested checkpoints per
successful natural run. Authorization is not acceptance evidence: no production
manifest exists, readiness has not passed, and production has not launched.

## Evidence rules

Phase 1 validation is pure, synthetic, and login-safe. It may exercise JSON,
filesystem crash boundaries, process locking, liveness decisions with injected
probes, retry planning, and resume logic. It must not load a model, tokenizer,
dataset, torch weights, or CUDA, invoke real Slurm, create an operational
production manifest, or generate experiment output.

Passing local tests establishes the repository contract. It does not establish
SmolLM3 behavior, MMLU provenance, Mila filesystem locking, real scheduler
semantics, compute-node execution, or production readiness. Those remain
resource-appropriate Phase 2/3 gates.

## Phase 2 evidence and completed smoke record

The login-safe preparation evidence covers:

- migration to explicit per-subject MMLU configs and 40-character resolved
  commit enforcement;
- deterministic five-block/seeded-order normalization, 500-record schema and
  identity/hash construction, staged reload, single-directory atomic
  publication, identical-only reruns, and independent returned-file validation;
- pure SmolLM3 prompt rendering, tag/token-boundary contracts, answer/confidence
  parsing, context arithmetic, natural entropy summaries, ten deterministic
  seeds, ties-to-even checkpoints, alias recovery, A–D metrics, smoke budgets,
  model-run identity separation, reproducibility comparison, and storage
  estimates; and
- Phase 1 append ordering, terminal-result authority, crash recovery, exclusive
  locking, retry counting, checkpoint-only resume, and stale-lock handling.

The fake dataset/tokenizer/logit tests remain control-flow evidence only. They
are now supplemented, but not replaced, by the bounded Mila evidence below.
The catalog at `tests/fixtures/part1_synthetic/catalog.json` provides
schema-valid synthetic raw-input families for the implemented Phase 3 AUROC,
bootstrap, macro-aggregation, within-question, switching, and stabilization
paths. Its evidence is synthetic and does not establish real-model results or
replace the still-pending hardened CLI and focused-readiness gates. Full-shape
synthetic acceptance is waived after job `10347033` supplied scale evidence.

| Gate | Current evidence |
|---|---|
| Local pure/unit/integration suite | **PASSED — 367 tests** at `e19edf22c8a3f7462ab66d5cae11b38247df5ed9` |
| Real MMLU revision, rows, counts, hashes, cache | **PASSED — CPU job `10284018`; exact 500-question bundle** |
| Returned manifest independent validation | **PASSED — tracked manifest commit `2e0bcae`** |
| Real SmolLM3/tokenizer preflight and forward | **PASSED — provenance refresh `10284702`, `0:0`, one L40S; immutable revision `a07cc9a04f16550a088caea529712d1d335b0ac1`** |
| Same-environment reproducibility | **PASSED — job `10284721`, `0:0`; exact tokens, parse, and entropy arrays at tolerance 0.0; seed `2552280803631819986`; report under model run `14c49484a4eebdb79372cb14b3e0076812e983d688c49aa7e3c2280bb44be7c0`** |
| Mila persistent filesystem | **PASSED — four focused tests at `results/part1-smoke/mila-filesystem-gate.EUEXDB`** |
| Mila scheduler liveness | **PASSED with fail-closed nuance — live exact output; immediate completed empty/rc0 is `DEAD`; later purged/rc1 is `UNKNOWN`** |
| Smoke A | **PASSED — authoritative job `10284742`, `COMPLETED`/`0:0`, `01:26:28`, `cn-l018`; terminal runner `complete`, 10 natural/110 checkpoint/240 audit** |
| Smoke B first submission | **DIAGNOSTIC ONLY — job `10292499` failed in 1s before Python/model/data/artifact creation because non-interactive SSH `PATH` omitted `~/.local/bin`, so `srun` could not execute `uv`; no Smoke B root** |
| Smoke B | **PASSED — authoritative job `10292530`, unchanged documented sbatch job from Mila login shell, `COMPLETED`/`0:0`, `00:25:14`, `cn-l072`; terminal runner `complete`, 5 natural/55 checkpoint/120 audit** |

The `2026-08-04T18:52:39-04:00` Smoke A running snapshot is historical only.
Authoritative job `10284742` subsequently completed with exactly one question,
natural run IDs `0`–`9`, checkpoint indices `0`–`10` for every run, and passed
manifest, dry-run, and lifecycle validation with 0 errors and 0 warnings. It
created no production root. All 10 natural records have
`natural_execution_outcome=complete`, `reasoning_status=closed`,
`answer_parse_status=parsed`, and `confidence_parse_status=malformed`. All 110
checkpoint records have `checkpoint_execution_outcome=complete`;
`checkpoint_model_output_status` is valid/invalid `93/17`,
`answer_token_status` is located/missing `109/1`, and `entropy_status` is
computed/unavailable `109/1`. This successful abnormal output was retained as
data.

Authoritative Smoke B job `10292530` covered sample indices `0`, `100`, `200`,
`300`, and `400` (one fixed first question per subject), each with natural
`run_id=0` and checkpoint indices `0`–`10`. It passed manifest, dry-run, and
lifecycle validation with 0 errors and 0 warnings, and created no production
root. All 5 natural records have `natural_execution_outcome=complete`;
`reasoning_status` is closed/missing_close `4/1`, `answer_parse_status` is
parsed/missing/out_of_domain `3/1/1`, and `confidence_parse_status` is
malformed/missing `4/1`. All 55 checkpoint records have
`checkpoint_execution_outcome=complete`; `checkpoint_model_output_status` is
valid/invalid `42/13`, `answer_token_status` is located/missing `46/9`, and
`entropy_status` is computed/unavailable `46/9`. This successful abnormal
output was retained as data; there were no retries, terminalization,
tail/recovery, lock, or takeover issue. Job `10292499` is not
experiment evidence; its non-interactive `uv` `PATH` failure is a Phase 3
SLURM-readiness hardening item.

The following Smoke A paths and commands are retained as the historical
post-job validation record:

```text
results/part1-smoke/smoke_a/a332786f767d9f84c23ad0ddd057b46d5d3a8d7b266458d9b0352f5bf90ea374/shard-000
logs/part1-smoke-a-10284742.out
results/part1-smoke/model-runs/smoke_a/model_run_manifest.json
results/part1-smoke/preflight/preflight.json
```

Do not infer success from partial counts. The historical validation procedure
used `squeue`/`sacct` to establish terminal state, inspected the terminal log,
reran both read-only validators, and inspected stable stream counts:

```bash
ssh mila
cd /home/mila/c/chenje/my-project
squeue --jobs=10284742 --noheader --format=%i,%T,%M,%R
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

Smoke A passed with `COMPLETED`/`0:0`, the runner terminal JSON, exactly
10 natural and 110 checkpoint terminal records, all ten run IDs for the same
fixed question, all eleven checkpoint identities per successful natural
(including aliases), compatible manifests/hashes/provenance, valid schemas and
hierarchy, policy-complete lifecycle/audit evidence, and no invalid tail,
pending recovery, terminalization, retry requirement, active writer lock, or
pending takeover. A purged/error `squeue` query is `UNKNOWN`, not `DEAD`.

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

### Runtime and final-correction gates

Commits `c066bff`, `d15ba85`, and `68926ec` implemented guarded sessions,
takeover durability, stale/operator recovery, retry policy, terminalization,
parent/provenance enforcement, resume, dry run, and operator CLI. Subsequent
independent reviews drove the bounded correction chain:

- `e6e0dee`: directory-entry durability, terminal interruption policy,
  recomputed checkpoint placement/aliases, manifest-bound dry run, fixed
  science/configuration oracles, complete null/status matrices, RFC 3339
  enforcement, and lock-capability-by-default storage;
- `1ca5661`: cross-record alias physical coherence, persisted retry-category
  authority, independent complete six-config oracles, effective-settings
  compatibility, and confidence boundary enforcement;
- `260fb44`: exactly one retry-authorizing closure on the latest persisted
  attempt;
- `12dd6a9`: strict integer `[0, 3]` caller attempt-count validation; and
- `5969080`: exact structured 10% tail-entropy oracle and rejection of fully
  rehashed 20% alternatives.

The final independent re-review ran:

```text
UV_CACHE_DIR=/private/tmp/mila-uv-cache uv run pytest -q \
  tests/test_part1_contract.py tests/test_part1_runtime.py \
  -k 'fixed_tail_entropy_contract or self_consistent_twenty_percent \
      or manifest_compatibility_rejects_fixed_contract_drift \
      or fixed_config_oracle' --tb=short
12 passed, 168 deselected

UV_CACHE_DIR=/private/tmp/mila-uv-cache uv run pytest -q --tb=short
265 passed in 36.18s

UV_CACHE_DIR=/private/tmp/mila-uv-cache uv run pytest -q \
  tests/test_part1_runtime.py::test_posix_flock_blocks_takeover_in_a_second_process \
  --tb=short
1 passed

UV_CACHE_DIR=/private/tmp/mila-uv-cache uv run python \
  scripts/part1_dry_run.py
exit 0; is_valid=true; mutation_performed=false;
would_create_production_manifest=false;
imports_model_or_data_libraries=false

git diff --check 12dd6a9 5969080
exit 0
```

Compile, JSON parsing, scope, result-absence, and static-import scans were also
clean. The reviewer found no remaining Critical, Important, Minor, P0, P1, P2,
or P3 Phase 1 code finding. The real two-process local POSIX regression showed
that a second process could not complete takeover while the first held
`.writer.guard`. All correction work remained synthetic and login-safe.

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
| Fixed configuration oracle | Every tracked field in all six Phase 1 configs is compared to an independent canonical JSON-typed oracle; only storage roots vary; self-referential template drift and boolean/integer lookalikes fail. |
| Fixed science/model oracle | Complete structured study and requested model/generation contracts are exact; effective settings retain every requested key/value while allowing additional canonical resolved fields; direct and fully rehashed drift fail. |
| Tail entropy | Fixed arithmetic mean over the final `max(1,ceil(0.10*n_reasoning))` recognized reasoning tokens, null at zero; historical and structured 20% variants are rejected. |
| Date-time/null matrices | RFC 3339 annotations are procedurally enforced; infrastructure diagnostics, reasoning-summary states, confidence missing/malformed/parsed/out-of-range boundaries, and natural/checkpoint nullability pass positive/negative tests. |
| Lifecycle separation | Only terminal records in result streams; exactly eight event types and two scopes; starts consume sequential attempts; event references/transitions checked. |
| Five commit crash boundaries | Before result append, during result append, after result fsync/before completion, during completion append, and after both fsyncs. |
| Result authority | Durable result without completion is not retried; recovery evidence and optional completion are idempotent. Completion without result is interruption. |
| Orphans/terminalization | Orphan starts are interrupted and counted. Nonretryable/final retryable failures require a terminal result; exhausted interruption remains terminalization-required until published. |
| Raw append/recovery | Per-record fsync; first file and every new directory entry are durably linked; valid bytes never overwritten; only final line repaired; exact-byte quarantine, immutable journals, valid-missing-newline append repair, malformed-middle rejection, and recovery crash matrix. |
| Scientific persistence | Full-precision finite values; generated-token/entropy and answer-token alignment; four A–D vectors/probability sum/entropy/max checks; selected-token log probabilities absent. |
| Duplicates/hierarchy | Duplicate/conflicting terminal/event IDs rejected; one complete eligible natural parent required; ties-to-even placement, actual fraction, IDs, prefix and aliases are recomputed; aliases of one physical prefix must agree across records. |
| Finalization/reports | Machine-readable stable-target reports; pending tails/recoveries, lifecycle/hierarchy errors, and terminalization block finalization; `.finalized` blocks mutation. |
| Exclusive writer | Store mutation requires a runtime lock capability by default; second owner rejection, guarded mutation/report/finalize/close/takeover, displaced-writer refusal, explicit test-only unsafe opt-in, and local two-process POSIX `flock` regression. |
| Takeover durability | Every claim/replacement/event/cleanup crash boundary, partial event, exact pending reuse, conflicting-pending quarantine with operator reason, one-event idempotence, and active-claim-last cleanup. |
| Liveness | Any LIVE refuses; conclusive SLURM DEAD and same-host PID DEAD rules; remote/ambiguous/error/timeout/missing-command refusal; age never used. |
| Operator recovery | Nonblank reason required; prior owner archived; `operator_unlock` durable; `--finish-pending` completes a durable claim. |
| Retry policy | Exact category lists, attempts 1–3, backoffs `[0,30,120]`, same identity/seed, `attempt_interrupted` only for interrupted process, nonretryable current-attempt terminalization, and fresh-process CUDA guard. |
| Retry evidence/input | Operative policy derives from exactly one retry-authorizing closure on the latest persisted attempt; caller category/count are equality checks only; count is a true integer `[0,3]`; pristine/orphan/ambiguous/completed/terminal/exhausted/locked/pending/finalized cases fail closed. |
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

## Phase 2 acceptance matrix

| Area | State | Evidence or remaining requirement |
|---|---|---|
| MMLU revision/selection | **Passed** | Job `10284018`; immutable revision, bounded selection, five ordered 100-question blocks, seed 42, stable identities/hashes. |
| Question/study manifests | **Passed** | Independent validator; tracked commit `2e0bcae`. |
| SmolLM3 preflight | **Passed** | Refresh job `10284702`, one L40S, `0:0`; immutable model/tokenizer revision and full prompt/token/environment checks. |
| Mila filesystem operation | **Passed** | Gate 4 under `results/part1-smoke/mila-filesystem-gate.EUEXDB`. |
| Mila scheduler liveness | **Passed with nuance** | Exact live output and immediate completed/absent semantics approved; purged rc1 remains `UNKNOWN` and fails closed. |
| Reproducibility | **Passed** | Job `10284721`, `0:0`; exact tokens, parse, and entropy arrays at tolerance 0.0. |
| Smoke A runtime integration | **Passed** | Job `10284742`: policy-complete writes, terminal runner `complete`, valid manifest/dry-run/lifecycle reports with 0 errors/warnings, and exact 10/110/240 counts. |
| Natural generation | **Passed** | Ten stochastic run identities `0`–`9`, no extra greedy run, fixed settings, aligned raw pre-warper float32 entropy, and abnormal-output retention. |
| Checkpoints | **Passed** | 110 records: all eleven requested identities per successful natural, including aliases and valid retained metric/status conditions. |
| Smoke separation | **Passed** | Both authorized smokes used non-production manifests/output under the ignored smoke root; no production root was created. |
| Smoke B | **Passed** | Job `10292530`: one fixed first question from each subject, `run_id=0`, all eleven checkpoints, and exact 5/55/120 counts. Job `10292499` is diagnostic-only launch-path evidence. |

Phase 2 is complete: Smoke A and Smoke B were independently validated. The
full experiment remained unrun and unauthorized at that gate. The later
2026-08-11 authorization applies only after the remaining Phase 3 readiness
sequence passes.

## Phase 3 and production-manifest gate

Analysis, bootstrap/calibration, complete raw validation, validate-before-
publish merge, and SLURM readiness hardening are implemented and independently
reviewed. End-to-end bounded Phase 3 smoke job `10324103` completed `0:0` in
`01:27:13`; its exact 10 natural/110 checkpoint/240 audit shape and direct
integrity validation passed. Hardened CLI-validator execution and focused
readiness remain pending; the duplicate synthetic full-shape rehearsal is
explicitly waived. Phase 3 production readiness has not yet passed.

Only after the focused-readiness and retained-smoke gates, final documentation, full verification,
independent review, the final tracked commit, and a clean worktree may the
operational production model-run manifest be created under the ignored
persistent production root. The manifest must record the final commit and its
creation must leave Git clean. Production launch-readiness/launch-plan checks
for `0-499%16` follow manifest creation. The explicit 2026-08-11 user
authorization permits launch only after all gates pass and sets a four-day
deadline; it does not itself prove manifest creation, readiness, or launch.

# Part 1 validation ledger and acceptance matrix

## Final acceptance state

All final gates are closed. Production generation completed with 500 finalized
shards and exact 5,000-natural/55,000-checkpoint shape; recovery merge published
5,000 natural, 55,000 checkpoint, and 120,003 audit rows; publication job
`10391026` validated and atomically promoted the 24-artifact `final-r5000`
analysis; the analysis manifest hash
`ea1e182034bd66fee2c7300e67f4db2a583965474d5e510a2f176136265cce9c`
records `paper_analysis_ready: true`; and the reviewed paper is
[../../report/main.pdf](../../report/main.pdf).

The final correctness cohort is 3,550 evaluable trajectories (3,172 correct,
378 incorrect), including 84 mixed-outcome questions for within-question
analysis. Final outputs retain explicit **fixed**, **repaired** verbalized
confidence, and **reconstructed intended analysis** prefix-entropy provenance.
No historical submission, validation, recovery, merge, or analysis job may be
rerun.

The dated sections below preserve evidence as it was known at each checkpoint.
Any statement that readiness was pending or production absent is historical and
superseded by this final acceptance state. Exact job dependencies, receipts,
hashes, and no-resubmission warnings are consolidated in
[OPERATIONS_HISTORY.md](OPERATIONS_HISTORY.md).

## Publication-only recovery state (2026-08-17)

Direct analysis job `10390517` completed all 5,000 bootstrap calculations and
wrote the complete 24-file analysis stage, then failed final reload on an
ordering-only validator defect: the producer uses the mandated fixed subject
order, while the validator used alphabetical order. Commit
`054d02d4dc4d1a7c24e7806fe3424ab69b899f6f` adds a regression for physics
preceding chemistry under the study contract and corrects only that comparator.

The same commit adds publication-only recovery that validates every artifact,
typed table semantics, hashes, metadata, plots, exact model/analysis/bootstrap
identity, directory inode, and collision state before atomic rename, then
revalidates the published inode and fsyncs the parent. Its focused analysis and
recovery verification passed 104 tests. Mila CPU job `10391026` completed `0:0`
in nine seconds; post-rename validation passed. The final directory contains all
24 expected artifacts, the stage and claim are absent, its manifest hash is
`ea1e182034bd66fee2c7300e67f4db2a583965474d5e510a2f176136265cce9c`, and
both manifest and summary record `paper_analysis_ready: true`. Final-paper
analysis acceptance is **PASSED**.

## Direct full-analysis recovery state (2026-08-17)

Merge finalize job `10385970` successfully published the exact canonical
production datasets: 5,000 natural rows, 55,000 checkpoint rows, and 120,003
audit rows. Final analysis job `10385971` failed `2:0` after `10:21:23` because
the trajectory layer rejected repeated checkpoint labels globally. Labels
`cp-00` through `cp-10` are intentionally scoped to each natural parent; their
global record identities remain unique.

Recovery commit `c2b10f6107ccc43620f9cb865dc3916577f91a3d` retains global
`checkpoint_record_id` uniqueness and per-parent checkpoint label/index
consistency, validates the immutable merge/recovery provenance, and reads each
merged dataset once. The audit Parquet is checked through metadata/schema
without decoding its 120,003 rows. A pre/post artifact snapshot and final cheap
byte/stat revalidation guard publication against input changes. Scientific
metrics, checkpoint definitions, sampling, and the 5,000-bootstrap analysis are
unchanged.

The exclusive receipt records direct analysis job `10390517` as `submitted`
with `no_preflight: true`. It failed `2:0` after completing all calculations
because of the fixed-order validator defect. Publication recovery `10391026`
subsequently validated and published the complete stage; final-paper acceptance
is recorded in the authoritative section above.

## Historical preserved-stage recovery state (2026-08-16)

Recovery commit `95a434ce7ab3d03619f7aad49993dd9745dab533` is deployed at
`/home/mila/c/chenje/my-project-recovery-95a434c`, separate from the clean
production checkout pinned at `ffa998a7ee1f156e150c8da33b258165ee53e032`.
Prior merge `10383206` failed `2:0` after `03:54:36` when validation rejected
legitimate zero-byte `runtime_guard` files; a subsequent GPFS cleanup `EINVAL`
masked that original exception. Dependent analysis `10383207` was cancelled.

The failure occurred after the raw scan and Parquet writes. The preserved
`.merged.stage-ri97qy41` is bound to:

| Artifact | Exact evidence |
|---|---|
| Merge manifest | 921,804 bytes; SHA-256 `16ad6ae082af664a3f3afecee83568f340e839ad2220117e3f03391c4bd10509` |
| Merge identity | `447cfc9125349369f24b3e0e6865c254b516ceb84c70263d9f0a0e36801938e6` |
| Merge-manifest hash | `a2f47af9378a6906c64f4f0ea9ae76d9d2f41c67be913b7c9cca5fe63dcbce03` |
| Natural Parquet | 5,000 rows; SHA-256 `25f1a61104b17fa085fc16c2eb13df67cb01e6477d59d5417d0146c3127986d3` |
| Checkpoint Parquet | 55,000 rows; SHA-256 `16dc510a2ea214f6225f3efec48b927bebaee279788cbd61e7d41b12db63f4f6` |
| Audit Parquet | 120,003 rows; SHA-256 `ac72306de70170ba6a9c8e32e83b6008903e009f9ed91ca1fcdde5cc029d5cd2` |

The user approved focused validation of this exact stage without duplicating
the completed raw-shard scan. The recovery still validates the preserved
manifest, Parquet bytes/counts/schemas, waiver and failed-report provenance,
clean commits, and publication state before atomic rename. Exclusive receipt
`validation/merge_stage_recovery_submission_receipt.json` records finalize
`10385970` and analysis `10385971` with exact `afterok:10385970`; both were
pending at submission. At that checkpoint, finalize was running on `cn-h001`
and analysis was dependency-held. Finalize later published the merge, and
publication recovery `10391026` completed final-paper acceptance.

## Production validation and exact-waiver state (2026-08-15)

Production generation is complete for model run
`6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c` at
commit `ffa998a7ee1f156e150c8da33b258165ee53e032`. Main array `10362272` plus
targeted retry `10362285` yielded 500 finalized shards. Recovery validation
`10381201` observed exactly 500 shards, 5,000 natural records, and 55,000
checkpoint records, with zero warnings, and preserved coverage report ID
`2bfe7cd6908351e3f1d6c9a2eec4f41c9dfa97f124f9da2f70925365490f23db`.

The standard result is **FAILED AND PRESERVED**, not waived into a pass. Its
60,001 structural errors have one shared cause:

| Evidence | Interpretation |
|---|---|
| 5,000 natural `manifest_incompatible` errors | Validator compares the record's content-derived prompt hash with the manifest's distinct global prompt-contract hash. |
| 55,000 checkpoint `retryable_incomplete` errors | Exact cascade from rejection of each otherwise present natural parent. |
| 1 physical-count error | Consequence of excluding all records above from the accepted partition. |
| 0 warnings | No additional warning category was observed. |
| Sample canonical recomputation | Stored natural hash exactly matched recomputation from `prompt_version`, `rendered_prompt`, and `prompt_token_ids`. |

These observations establish full persisted shape and isolate a validator
contract mismatch. They do not, by themselves, establish that every row is
valid. The user-authorized recovery-only waiver therefore must re-read all raw
records and recompute every natural prompt hash during merge. It must retain
the standard schema, lifecycle, hierarchy, duplicate, source-snapshot,
checkpoint compatibility, and exact-count checks, and reject any discrepancy
outside the exact fingerprint above.

Acceptance of the recovery requires all of the following:

- the failed report bytes, SHA-256, validation ID, inventory, error partition,
  warning count, model-run ID, and generation commit match the waiver sidecar;
- all 5,000 natural content-derived prompt hashes recompute exactly;
- all 5,000 natural rows have `natural_execution_outcome=complete` and all
  55,000 checkpoint rows have `checkpoint_execution_outcome=complete`, checked
  separately because the failed report's incompatibility partition masks those
  terminal outcomes;
- all 55,000 checkpoints pass their normal compatibility and parent checks once
  the erroneous manifest-level prompt comparison is excluded;
- the recovery merge publishes exactly three canonical Parquet datasets and
  records the failed report plus waiver provenance; and
- unchanged 5,000-bootstrap analysis publishes the final manifest/summary,
  eight CSV tables, six figures, and `paper_analysis_ready: true`.

Standard validator/merge/analysis behavior remains unchanged, and the failed
coverage report is never rewritten. Recovery commit
`1a4b6039758cd8fd84b68f74c0828e6c5f382dae` passed focused verification and
independent review. It runs from a separate clean worktree while the production
checkout remains clean and pinned at
`ffa998a7ee1f156e150c8da33b258165ee53e032`. Waiver preparation `10383205`,
merge/check `10383206` (`afterok:10383205`), and final analysis `10383207`
(`afterok:10383206`) were submitted through the exclusive launcher. Preparation
completed `0:0` in four seconds and published waiver ID
`dc6d50b0a4712eedac891bb55b794b532ea5e9232f6712b85536c196851f9163`.
Merge subsequently failed as described above and analysis was cancelled. At
that checkpoint final-paper readiness had not passed; the later preserved-stage
and publication recoveries completed it.

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
schema, hierarchy, lifecycle, and policy-complete audit validation passed. At
that historical checkpoint the hardened CLI-validator acceptance gate was
still pending, so Phase 3, production launch readiness, and final-paper
readiness had not yet passed. All later final gates subsequently completed.

The submission gate itself passed on Mila:

| Gate | Evidence |
|---|---|
| Exact deployed commit and clean tracked tree | **PASSED** — `a7e1135e85476f5fc43986949f467b10ba450623` |
| Immutable manifest bundle | **PASSED** — 500 total, 100 per subject; question hash `dd379f48322d6eb07c309101361738a965be320b4124bb45bd44723b1abe474d`; study hash `859eecd5e0437a901555d5fd2d99692feccb5257df16de60bfe0fe648626142b` |
| Preflight dependency compatibility | **PASSED** — current and recorded `uv.lock` SHA-256 both `9cab4a125cc2bbf880efcd25826c6e4cdf9964889c7c29742cd10e96bc98db36` |
| Scheduler/collision/shell checks | **PASSED** — `MaxArraySize=1001`, no existing Phase 3 smoke roots, all relevant job scripts pass `bash -n` |
| Focused no-model Mila tests | **PASSED** — 34 passed in 1148.24 seconds |
| Final bounded Phase 3 smoke direct integrity | **PASSED** — job `10324103`, `COMPLETED`/`0:0`, `01:27:13`; terminal runner and exact 10 natural/110 checkpoint/240 audit shape directly validated |
| Retained real-model smokes | **HISTORICAL EVIDENCE** — Smoke A/B predate `.finalized`; Phase 3 smoke predates the final prompt-hash contract; completed jobs and accepted counts are retained but are not live launch inputs |
| Full-shape synthetic acceptance | **WAIVED AFTER SCALE EVIDENCE** — corrected job `10347033` completed full strict coverage with no reported defect, then timed out during merge at 12 hours; it will not be repeated under the four-day deadline |
| Focused recovery readiness, local | **PASSED** — `797 passed, 1 deselected in 943.13s`; the deselected test is exactly `part1_full_acceptance` |
| Focused recovery readiness, Mila first attempt | **FIXTURE FAILURE** — job `10357631` reached an environment-dependent assertion because Mila's pytest `tmp_path` is under forbidden `/tmp`; gate `10357632` remained fail-closed and no production job was submitted |
| Focused recovery readiness, Mila second attempt | **ENVIRONMENT FAILURE** — job `10362018` passed the first corrected fixture, then another `tmp_path` test hit the same forbidden-`/tmp` ordering; readiness `10362018` and gate `10362019` were cancelled, no production was submitted, and the job-level pytest basetemp was moved to unique `$SCRATCH` storage |
| Focused recovery readiness, Mila third attempt | **ENVIRONMENT INCOMPATIBILITY** — global `$SCRATCH` basetemp caused tests intentionally checking ephemeral-path rejection to fail; readiness `10362106` was cancelled, gate `10362107` did not run, no production was submitted, and readiness was narrowed to the 16 launch-critical orchestration/plan tests |
| Focused recovery readiness, Mila fourth attempt | **READINESS PASSED / LEGACY GATE MISMATCH** — readiness `10362161` passed 16 tests in `13.18s`; gate `10362162` failed before production because Smoke A/B predate required `.finalized` markers; no production manifest, receipt, or job was created, prompting an attempted Phase 3-only live gate |
| Focused recovery readiness, Mila fifth attempt | **READINESS PASSED / HISTORICAL PROMPT MISMATCH** — readiness `10362197` passed in `10s`; gate `10362198` found the earlier Phase 3 smoke predates the final prompt-hash contract and failed before creating any production manifest, receipt, or job; retained smokes are historical evidence rather than live gate inputs |
| First production array startup | **SCHEDULER STARTUP FAILURE** — gate `10362215` submitted array `10362218`; tasks failed before Python/model startup with Slurm exit `64` because default CPU binding was outside task allocations; array cancelled, no scientific shards written, wrapper corrected with `srun --cpu-bind=none` |
| Final launch-critical local suite | **PASSED** — 257 tests in `570.14s`, excluding only the explicit full-shape marker prepared for CPU SLURM |
| Unattended submission review | **PASSED BEFORE RECOVERY** — exact `afterok` bootstrap, exclusive receipts, race/crash reconciliation, and `%16` dependencies were independently approved; focused-readiness recovery requires fresh tests and review before submission |

At that checkpoint, continuation started with focused readiness and the
production gate in [RUNBOOK.md](RUNBOOK.md), not another smoke submission. The
later chain passed its launch gates and strictly validated the production
artifacts; this paragraph is not a resubmission instruction.

The full-shape flow is retained but no longer resubmitted. Measured runtime
raised the post-generation CPU ceilings to 12 hours for validation, 24 hours
for merge, and 36 hours for final analysis; these are operational resource
changes only and do not alter any metric or scientific contract.

The user explicitly authorized the post-readiness full production run on
2026-08-11, with target array `0-499%16` and a four-day deadline. The protocol
remains exactly 500 questions × 10 natural runs × 11 requested checkpoints per
successful natural run. Authorization alone was not acceptance evidence; the
subsequent production and recovery records provide that evidence.

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

## Historical Phase 3 and production-manifest gate

Analysis, bootstrap/calibration, complete raw validation, validate-before-
publish merge, and SLURM readiness hardening are implemented and independently
reviewed. End-to-end bounded Phase 3 smoke job `10324103` completed `0:0` in
`01:27:13`; its exact 10 natural/110 checkpoint/240 audit shape and direct
integrity validation passed. At that checkpoint hardened CLI-validator
execution and focused readiness remained pending; the duplicate synthetic
full-shape rehearsal was explicitly waived. The later final gates passed.

Only after focused readiness, immutable-manifest and clean-commit validation,
with completed smokes retained as historical evidence, final documentation,
independent review, the final tracked commit, and a clean worktree may the
operational production model-run manifest be created under the ignored
persistent production root. The manifest must record the final commit and its
creation must leave Git clean. Production launch-readiness/launch-plan checks
for `0-499%16` follow manifest creation. The explicit 2026-08-11 user
authorization permits launch only after all gates pass and sets a four-day
deadline; it does not itself prove manifest creation, readiness, or launch.

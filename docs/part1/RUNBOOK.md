# Part 1 runbook

## Active direct-analysis recovery (2026-08-17)

The direct full-analysis job is already submitted. Do **not** rerun
`scripts/submit_part1_direct_analysis_recovery.py` or submit the job script
manually. The exclusive receipt is:

```text
/home/mila/c/chenje/my-project/results/part1/6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c/validation/direct_analysis_recovery_receipt.json
```

It binds execution commit
`c2b10f6107ccc43620f9cb865dc3916577f91a3d`, merge-stage recovery ID
`e5e0281eb2f01293d510bd9da10b16195aae7afa7d43c5a8878744173e031de4`,
5,000 bootstrap replicates, and analysis job `10390517`. Use only read-only
monitoring:

```bash
ssh mila
squeue --jobs=10390517 --format='%i|%j|%T|%M|%l|%R'
sacct -j 10390517 --format=JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End,NodeList --parsable2
tail -n 100 /home/mila/c/chenje/my-project-recovery-c2b10f6/logs/part1-direct-analysis-10390517.out
```

At submission reconciliation the job was `PENDING` for cluster capacity. It
subsequently started on `cn-m003` and was `RUNNING` at the latest check. The
initial Matplotlib cache warning used an automatic writable `/tmp` fallback and
does not require intervention. Once it is terminal, require `COMPLETED` and
exit `0:0`, then validate the immutable analysis publication under:

```text
/home/mila/c/chenje/my-project/results/part1/6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c/analysis/final-r5000/
```

Completion requires the final analysis manifest and summary, eight CSV tables
with same-stem metadata, six figures, and `paper_analysis_ready: true`. A
nonzero terminal state is evidence to diagnose; it does not authorize deleting
the merged data, replacing the receipt, or rerunning generation.

## Current merge-stage recovery procedure (2026-08-16)

The active recovery checkout is
`/home/mila/c/chenje/my-project-recovery-95a434c` at
`95a434ce7ab3d03619f7aad49993dd9745dab533`. Keep the production checkout
clean and pinned at `ffa998a7ee1f156e150c8da33b258165ee53e032`.

Merge `10383206` failed `2:0` after `03:54:36`: a legitimate empty
`runtime_guard` was rejected by manifest validation, then cleanup hit GPFS
`EINVAL` and masked the primary diagnostic. Preserve
`.merged.stage-ri97qy41`; it contains the already completed raw-scan result
with exact row counts 5,000 natural, 55,000 checkpoint, and 120,003 audit.
Do not rerun the raw scan.

The focused publication chain has already been submitted. Its exclusive
receipt is:

```text
/home/mila/c/chenje/my-project/results/part1/6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c/validation/merge_stage_recovery_submission_receipt.json
```

Monitor only the recorded jobs; both were pending at submission:

```text
finalize: 10385970
analysis: 10385971  afterok:10385970
```

At the latest check, finalize was running on `cn-h001`; analysis remained
dependency-held.

Use only read-only checks:

```bash
ssh mila
cd /home/mila/c/chenje/my-project-recovery-95a434c
python -m json.tool /home/mila/c/chenje/my-project/results/part1/6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c/validation/merge_stage_recovery_submission_receipt.json
squeue --jobs=10385970,10385971 --format='%i %T %M %R'
sacct -j 10385970,10385971 --format=JobIDRaw,State,ExitCode,Elapsed,NodeList --parsable2
tail -n 80 logs/part1-recover-merge-stage-10385970.out
tail -n 80 logs/part1-analyze-stage-recovery-10385971.out
```

Finalize must validate the exact stage manifest/hash, three output hashes and
row counts, schemas, immutable waiver/report provenance, both clean Git states,
and absence of competing publication before same-parent atomic rename. It does
not duplicate the completed raw-shard scan. On success it writes
`validation/merge_stage_recovery.json`; analysis `10385971` must verify that
sidecar and the published merge before creating `analysis/final-r5000/`.
Do not resubmit, rename, delete, or edit the preserved stage or either receipt.

## Historical prompt-hash recovery procedure (2026-08-15)

Do not submit generation, focused readiness, the production gate, or standard
validation again. Production generation is complete for model run
`6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c` at
`ffa998a7ee1f156e150c8da33b258165ee53e032`. Jobs `10362272` and `10362285`
left 500 finalized shards with 5,000 natural and 55,000 checkpoint records.

Preserve this standard failed report byte-for-byte:

```text
results/part1/6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c/validation/coverage_report.json
validation ID: 2bfe7cd6908351e3f1d6c9a2eec4f41c9dfa97f124f9da2f70925365490f23db
```

It records the exact prompt-hash validator defect described in
[VALIDATION.md](VALIDATION.md). It is not a successful coverage report and must
not be replaced, edited, or relabeled. The user-authorized recovery path uses a
separate waiver sidecar bound to the report's exact bytes and fingerprint.

After the scoped implementation is committed, deploy that recovery commit in a
separate Mila worktree. Keep `/home/mila/c/chenje/my-project` clean and pinned
to the generation commit. Recovery code may execute from the separate worktree,
but its repository/persistent-root argument must point at the pinned production
checkout so it reads the existing manifest, failed report, and raw shards. Do
not call the standard global snapshot check against the recovery checkout.

Before submission:

```bash
ssh mila
cd /home/mila/c/chenje/my-project
git rev-parse HEAD
git status --short
squeue --me
sha256sum results/part1/6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c/validation/coverage_report.json
```

Require the first command to print
`ffa998a7ee1f156e150c8da33b258165ee53e032`, no tracked changes, no active
production jobs, and retain the printed report SHA-256 in the waiver. Then use
only the recovery CLI/job commands added by the scoped recovery commit. Every
CPU `srun` must specify `--cpu-bind=none`, and dependencies must be:

```text
waiver verification -> merge (afterok) -> analysis (afterok)
```

Historical first-waiver submission command (do not run again):

```bash
uv run python scripts/submit_part1_prompt_hash_waiver_recovery.py \
  --production-repository-root /home/mila/c/chenje/my-project \
  --model-run-id 6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c
```

The launcher submits only these CPU jobs:

```text
jobs/part1_prepare_prompt_hash_waiver.sh
jobs/part1_merge_prompt_hash_waiver.sh    afterok:<prepare-job-id>
jobs/part1_analyze_prompt_hash_waiver.sh  afterok:<merge-job-id>
```

It atomically records partial submission state and all returned IDs at:

```text
/home/mila/c/chenje/my-project/results/part1/6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c/validation/prompt_hash_waiver_recovery_receipt.json
```

An existing receipt is exclusive evidence; do not run the launcher again.
Inspect it and its exact job IDs instead:

```bash
python -m json.tool \
  /home/mila/c/chenje/my-project/results/part1/6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c/validation/prompt_hash_waiver_recovery_receipt.json
```

Logs persist under the recovery worktree's `logs/` directory:

```text
logs/part1-waiver-prepare-<prepare-job-id>.out
logs/part1-merge-waiver-<merge-job-id>.out
logs/part1-analyze-waiver-<analysis-job-id>.out
```

The waiver stage must accept only 500 shards, 5,000 natural rows, 55,000
checkpoint rows, zero warnings, the exact 5,000 prompt-contract mismatches, the
exact 55,000 parent cascades, and the aggregate count error. During merge,
recompute all 5,000 natural prompt hashes from `prompt_version`,
`rendered_prompt`, and `prompt_token_ids`; retain all other standard merge
checks. The incompatibility partition in the failed report masks terminal
outcomes, so merge must separately require all 5,000 natural records to have
`natural_execution_outcome=complete` and all 55,000 checkpoint records to have
`checkpoint_execution_outcome=complete`. Record the failed report ID/hash,
waiver artifact ID/hash, generation commit, recovery commit, exact
authorization, and all three job IDs in durable provenance/receipt files.

The chain was submitted once from recovery commit
`1a4b6039758cd8fd84b68f74c0828e6c5f382dae` with these authoritative IDs:

```text
prepare:  10383205
merge:    10383206  afterok:10383205
analysis: 10383207  afterok:10383206
```

Do not run the launcher again. Preparation completed `0:0` in four seconds and
published waiver ID
`dc6d50b0a4712eedac891bb55b794b532ea5e9232f6712b85536c196851f9163`.
Merge later failed on the zero-byte guard validation rule described above, and
analysis was cancelled by its dependency. Retain these IDs and the exclusive
receipt as historical evidence.

Do not declare completion until the recovery chain produces three canonical
Parquet datasets, the final analysis manifest and summary, eight CSV tables,
six figures, and `paper_analysis_ready: true`.

Expected production-root publications are:

```text
results/part1/<model-run-id>/validation/prompt_hash_waiver.json
results/part1/<model-run-id>/merged/merge_manifest.json
results/part1/<model-run-id>/merged/natural_results.parquet
results/part1/<model-run-id>/merged/checkpoint_results.parquet
results/part1/<model-run-id>/merged/audit_events.parquet
results/part1/<model-run-id>/analysis/final-r5000/analysis_manifest.json
results/part1/<model-run-id>/analysis/final-r5000/analysis_summary.json
results/part1/<model-run-id>/analysis/final-r5000/trajectory_features.csv
results/part1/<model-run-id>/analysis/final-r5000/trajectory_events.csv
results/part1/<model-run-id>/analysis/final-r5000/primary_auroc.csv
results/part1/<model-run-id>/analysis/final-r5000/secondary_checkpoint_auroc.csv
results/part1/<model-run-id>/analysis/final-r5000/calibration_metrics.csv
results/part1/<model-run-id>/analysis/final-r5000/reliability_bins.csv
results/part1/<model-run-id>/analysis/final-r5000/within_question_summary.csv
results/part1/<model-run-id>/analysis/final-r5000/within_question_distribution.csv
```

Each CSV has a same-stem `.metadata.json`. The six analysis figures are
`primary_auroc.png`, `checkpoint_ece.png`, `natural_reliability.png`,
`checkpoint_reliability_main.png`,
`within_question_paired_differences.png`, and
`switching_stabilization.png`.

## Historical pre-production recovery checkpoint

Full-shape acceptance `10347033` timed out after `12:00:20` during merge and
gate `10347034` was cancelled by its unsatisfied `afterok` dependency. No
production manifest or GPU job was created. Do not resubmit either historical
job. Use only the focused-readiness bootstrap below and treat its receipts as
the operational source of truth.

## Current Phase 3 continuation checkpoint (2026-08-11)

The reviewed Phase 3 tree is deployed on Mila at
`a7e1135e85476f5fc43986949f467b10ba450623`. All submission preflights passed,
including the current preflight/`uv.lock` match, immutable manifest validation,
shell syntax, clean tracked state, `MaxArraySize=1001`, no existing Phase 3
smoke root, and 34 focused pure tests. Exactly one bounded Phase 3 job was
submitted, and it is now terminal:

```text
job:            10324103
submitted:      2026-08-09T13:55:40-04:00
submission:     sbatch jobs/part1_phase3_smoke.sh
started:        2026-08-09T13:55:58-04:00 on cn-l033
accounting:     COMPLETED, exit 0:0, elapsed 01:27:13
log:            logs/part1-phase3-smoke-10324103.out
model manifest: results/part1-smoke/model-runs/phase3_smoke/model_run_manifest.json
model-run ID:   a23087c8c0897bbf9f075b3edaa28c75b5087cffbcd203e2fea4bf16093b6dcf
shard:          results/part1-smoke/phase3_smoke/a23087c8c0897bbf9f075b3edaa28c75b5087cffbcd203e2fea4bf16093b6dcf/raw_shards/shard-000
stable counts:  10 natural, 110 checkpoint, 240 audit
```

Direct post-job integrity validation passed: the terminal runner state and
stable counts were checked; natural run IDs are `0`–`9`; all 11 requested
checkpoint identities exist for every successful natural run; immutable
manifest/provenance, schema, hierarchy, lifecycle, and policy-complete audit
checks passed; and no production root was created. Abnormal successful output
remains data. Hardened CLI-validator acceptance is a separate remaining gate
and has **not** passed merely because these direct checks passed.

Continue Phase 3 Task 8 in this exact order:

1. finish and verify the hardened CLI validator;
2. retain completed Smoke A/B/Phase 3 jobs and counts as immutable historical
   evidence; validate actual production shards strictly after generation;
3. retain full-shape timeout `10347033` as historical scale evidence; do not
   repeat it;
4. finalize the six Part 1 documents plus `AGENTS.md`, run full verification,
   and complete independent review;
5. create the final tracked commit and confirm the tracked worktree is clean;
6. run `uv run python scripts/create_part1_model_run_manifest.py` so the
   immutable production model-run manifest records that exact commit under the
   ignored production root, then confirm Git remains clean;
7. run the production launch-readiness and launch-plan checks for all 500 shard
   indices using the unchanged 500-question × 10-run × 11-checkpoint protocol
   and approved concurrency `%16`; and
8. only after every prior step passes, use the user's explicit 2026-08-11
   authorization to submit production and operate to the four-day deadline.

For unattended execution, use the reviewed bootstrap launcher from the final
clean deployed commit. It exclusively claims a bootstrap receipt before any
`sbatch`, then submits only the CPU focused-readiness suite and its exact
`afterok` gate:

```bash
uv run python scripts/submit_part1_unattended.py
```

The readiness job is CPU-only with a one-hour ceiling and runs the regression
suite with `-m "not part1_full_acceptance"`; it loads no model or dataset. The
gate cannot run if readiness fails. Do not submit either job script directly. The bootstrap
receipt under `results/part1-submission/<final-commit>/bootstrap_receipt.json`
records both job IDs, the exact `afterok` relationship, the command arguments,
and any in-flight stage for crash reconciliation. Once released, the gate
validates that receipt, tracked manifests, and clean commit, creates the
production model-run manifest from the same clean commit, validates the `%16`
launch plan, submits generation plus dependent validation/merge/final analysis,
and atomically records all six job IDs under
`results/part1/<model-run-id>/submission_receipt.json`. An existing or partial
receipt blocks automatic resubmission. It is safe to disconnect only after the
launcher prints `status=submitted`; confirm its two job IDs with `squeue`.

The launch-plan report prints the exact self-contained production command with
the immutable model-run ID embedded. Its shape, which remains unrun until step
8, is:

```bash
sbatch --export=ALL,MODEL_RUN_ID=<model-run-id> \
  --array=0-499%16 jobs/part1_generate_array.sh
```

Do not execute the literal placeholder. After manifest and launch-plan
validation, copy the exact `submission_command` from the machine-readable
launch-plan report. For an unattended production chain, set `MODEL_RUN_ID` to
that same 64-hex identity and retain every returned job ID:

```bash
GENERATION_JOB_ID=$(sbatch --parsable \
  --export=ALL,MODEL_RUN_ID="$MODEL_RUN_ID" \
  --array=0-499%16 jobs/part1_generate_array.sh)
VALIDATION_JOB_ID=$(sbatch --parsable \
  --dependency=afterany:"$GENERATION_JOB_ID" \
  --export=ALL,MODEL_RUN_ID="$MODEL_RUN_ID" \
  jobs/part1_validate.sh)
MERGE_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:"$VALIDATION_JOB_ID" \
  --export=ALL,MODEL_RUN_ID="$MODEL_RUN_ID" \
  jobs/part1_merge.sh)
ANALYSIS_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:"$MERGE_JOB_ID" \
  --export=ALL,MODEL_RUN_ID="$MODEL_RUN_ID",BOOTSTRAP_REPLICATES=5000 \
  jobs/part1_analyze.sh)
```

`afterany` ensures validation publishes a diagnostic coverage report even when
an array task fails. The `afterok` dependencies prevent merge or analysis from
running on failed validation or merge output. Safe resume uses the same
self-contained `%16` array command only after the prior array is terminal; the
append-only store skips terminal keys and resumes incomplete work.

Once the production manifest binds the final tracked commit, do not commit,
pull, checkout another revision, or otherwise advance the Mila worktree while
the production array or its dependent jobs are active. The runner enforces the
recorded commit. Retain the production model-run ID, submitted job IDs, and live
state in the ignored production result tree and the operator handoff; do not
create a later tracked documentation commit that makes the active checkout
incompatible with the manifest.

The post-generation CPU ceilings are 12 hours for validation, 24 hours for
merge, and 36 hours for final 5,000-bootstrap analysis. These are
kill-prevention ceilings, not expected runtimes. Validation of real outputs,
canonical merge, and final analysis remain mandatory; the waived work is only
the duplicate pre-launch synthetic rehearsal.

This authorization does not establish hardened CLI acceptance, create the
production manifest, pass launch readiness, or prove that submission occurred.

## Status and safety boundary

Phase 1's login-safe infrastructure remains implemented and Phase 2 is
complete. Authoritative Smoke A job `10284742` completed `0:0` in `01:26:28`
on `cn-l018`; authoritative Smoke B job `10292530` completed `0:0` in
`00:25:14` on `cn-l072`. Both non-production smokes reached terminal runner
state `complete` and passed manifest, dry-run, and lifecycle validation with 0
errors and 0 warnings. Phase 3 Tasks 1–7 are implemented and reviewed; Task 8
continues after bounded smoke job `10324103` completed `0:0` in `01:27:13` with
the exact 10 natural/110 checkpoint/240 audit shape and passing direct integrity
validation. Hardened CLI-validator acceptance is still pending. No production
root/model-run manifest exists, launch readiness has not passed, and the full
500-question job remains unrun.

Never load a model, tokenizer, or Hugging Face dataset on a login node. Dataset
materialization runs in a CPU SLURM job. Model/tokenizer preflight and generation
run in a GPU SLURM job. The user explicitly authorized the post-readiness full
run on 2026-08-11, with target concurrency `%16` and a four-day deadline. This
runbook records that eventual command but marks it unrun until every ordered
gate passes. The authorization changes none of the 500 × 10 × 11 protocol.

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

## GPU sequence — completed Phase 2 evidence

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

Smoke A job `10284742` completed `0:0` in `01:26:28` on `cn-l018`, with terminal
runner state `complete`, 10 natural results, 110 checkpoint results, and 240
audit events. It covered exactly one question, natural run IDs `0`–`9`, and
checkpoint indices `0`–`10` for every run. Manifest, dry-run, and lifecycle
validation passed with 0 errors and 0 warnings; no production root was created.
All 10 natural records have `natural_execution_outcome=complete`,
`reasoning_status=closed`, `answer_parse_status=parsed`, and
`confidence_parse_status=malformed`. All 110 checkpoint records have
`checkpoint_execution_outcome=complete`; `checkpoint_model_output_status` is
valid/invalid `93/17`, `answer_token_status` is located/missing `109/1`, and
`entropy_status` is computed/unavailable `109/1`. This successful abnormal
output was retained as data.

Smoke B job `10292499` is historical diagnostic evidence only: it failed in one
second before Python, model, dataset, or artifact creation because a
non-interactive SSH `PATH` omitted `~/.local/bin`, so `srun` could not execute
`uv`. It created no Smoke B root. The unchanged documented sbatch job was then
submitted from a Mila login shell as authoritative job `10292530`, which
completed `0:0` in `00:25:14` on `cn-l072` with terminal runner state
`complete`. It covered sample indices `0`, `100`, `200`, `300`, and `400` (the
fixed first question of each subject), natural `run_id=0`, and checkpoint
indices `0`–`10`: 5 natural results, 55 checkpoint results, and 120 audit
events. Manifest, dry-run, and lifecycle validation passed with 0 errors and 0
warnings; no production root was created. All 5 natural records have
`natural_execution_outcome=complete`; `reasoning_status` is
closed/missing_close `4/1`, `answer_parse_status` is
parsed/missing/out_of_domain `3/1/1`, and `confidence_parse_status` is
malformed/missing `4/1`. All 55 checkpoint records have
`checkpoint_execution_outcome=complete`; `checkpoint_model_output_status` is
valid/invalid `42/13`, `answer_token_status` is located/missing `46/9`, and
`entropy_status` is computed/unavailable `46/9`. This successful abnormal
output was retained as data; there were no retries, terminalization,
tail/recovery, lock, or takeover issues. Do not submit the full 500-question
run.

## Historical Smoke A monitoring and validation procedure

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

All five checks subsequently passed. Smoke B was then independently validated
as recorded above. The one-second `10292499` launch failure is a Phase 3
SLURM-readiness hardening item, not a model, reproducibility, or experimental
failure.

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
directory component also fsyncs the parent directory entry. The Phase 2 Mila
filesystem gate exercised directory `fsync`, no-overwrite hard-link
publication, atomic replacement, and two-process POSIX `flock` on the selected
persistent filesystem.

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

## Remaining Phase 3 lifecycle

1. CPU materialization, tracked-manifest validation/commit, GPU preflight,
   filesystem/scheduler checks, corrected reproducibility, Smoke A, and Smoke
   B are complete.
2. Analysis, complete raw validation, validate-before-publish merge, and SLURM
   launch-readiness machinery are implemented and independently reviewed at
   `a7e1135`; operational launch readiness has not passed.
3. Bounded Phase 3 smoke job `10324103` completed `0:0` in `01:27:13`; its
   exact 10 natural/110 checkpoint/240 audit shape and direct integrity checks
   passed.
4. Run focused readiness and retain completed smokes as historical evidence;
   do not repeat full-shape synthetic acceptance. Strictly validate actual
   production shards after generation.
5. Finalize the six Part 1 documents plus `AGENTS.md`, run full verification and
   independent review, create the final tracked commit, and require a clean
   tracked worktree.
6. Generate the operational production model-run manifest under
   `results/part1/<model_run_id>/` from that exact commit, after which Git must
   remain clean.
7. Pass production launch-readiness/launch-plan validation for `0-499%16` under
   the unchanged 500 × 10 × 11 protocol.
8. Only then use the explicit 2026-08-11 authorization to launch and operate to
   the four-day deadline. The production manifest, readiness result, and launch
   remain absent at this checkpoint.

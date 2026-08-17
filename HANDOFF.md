# Part 1 production recovery handoff

## Active publication-only recovery (2026-08-17)

Commit `054d02d4dc4d1a7c24e7806fe3424ab69b899f6f` is pushed and
deployed cleanly at `/home/mila/c/chenje/my-project-recovery-054d02d`.
CPU job `10391026` was submitted to validate and atomically publish the exact
preserved analysis stage without recomputing the completed 5,000 bootstraps:

```text
/home/mila/c/chenje/my-project/results/part1/6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c/analysis/.final-r5000.stage-5o7fg8le
analysis ID: 2c141766ccd3e77c8692294bcb067c3ea66bfcfd2fd0f18c2ca3d61c45f01bb7
target:      final-r5000
```

Job `10391026` completed `0:0` in nine seconds on `cn-m001`. Its post-rename
validator reported analysis manifest hash
`ea1e182034bd66fee2c7300e67f4db2a583965474d5e510a2f176136265cce9c`
and `paper_analysis_ready: true`. The hidden stage and publication claim are
absent, and the immutable final directory contains all 24 expected artifacts.
Do not resubmit. Historical accounting can be inspected read-only:

```bash
squeue --jobs=10391026 --format='%i|%j|%T|%M|%l|%R'
sacct -j 10391026 --format=JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End,NodeList --parsable2
tail -n 100 /home/mila/c/chenje/my-project-recovery-054d02d/logs/part1-finalize-analysis-10391026.out
```

The prior direct job `10390517` is terminal `FAILED`/`2:0` after `00:54:47`; it completed every
analysis artifact but its final validator incorrectly used alphabetical subject
order instead of the fixed study order. It must not be resubmitted. Part 1
production analysis is now complete and paper-analysis ready.

## Earlier direct-analysis checkpoint (2026-08-17)

The direct full-analysis attempt is historical and must not be resubmitted. Recovery commit
`c2b10f6107ccc43620f9cb865dc3916577f91a3d` is pushed and deployed in the
clean worktree `/home/mila/c/chenje/my-project-recovery-c2b10f6`. Its exclusive
receipt records analysis job `10390517`, 5,000 bootstrap replicates,
`no_preflight: true`, and status `submitted`:

```text
/home/mila/c/chenje/my-project/results/part1/6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c/validation/direct_analysis_recovery_receipt.json
```

Job `10390517` later failed `2:0` after `00:54:47` on the fixed-order validator
defect documented above. Its initial Matplotlib cache warning was unrelated.
Do not run the launcher again, modify either Mila checkout, or replace the
receipt:

```bash
ssh mila
squeue --jobs=10390517 --format='%i|%j|%T|%M|%l|%R'
sacct -j 10390517 --format=JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End,NodeList --parsable2
tail -n 100 /home/mila/c/chenje/my-project-recovery-c2b10f6/logs/part1-direct-analysis-10390517.out
```

The merged production datasets are already atomically published and remain
immutable: 5,000 natural rows, 55,000 checkpoint rows, and 120,003 audit rows.
The previous analysis job `10385971` failed after `10:21:23` because the
trajectory builder incorrectly required checkpoint labels such as `cp-00` to
be globally unique, although the protocol intentionally reuses those labels
for each natural parent. Commit `c2b10f6` preserves global
`checkpoint_record_id` uniqueness and per-parent checkpoint consistency while
removing only that invalid global-label requirement. Its direct recovery path
reads each merged file once, validates the immutable recovery provenance and
artifact bytes, and ran the unchanged final 5,000-bootstrap analysis. The exact
preserved stage was subsequently validated and published by `10391026`.

## Earlier preserved-stage checkpoint (2026-08-16)

Recovery commit `95a434ce7ab3d03619f7aad49993dd9745dab533` is pushed and
deployed at `/home/mila/c/chenje/my-project-recovery-95a434c`. The immutable
production checkout remains clean and pinned at generation commit
`ffa998a7ee1f156e150c8da33b258165ee53e032`.

The prior waiver merge job `10383206` failed `2:0` after `03:54:36`. Its
primary failure was a merge-manifest validation rule that rejected legitimate
zero-byte `runtime_guard` files; a later GPFS cleanup `EINVAL` masked that
diagnostic in the final exception. The job nevertheless completed the expensive
raw read and preserved an exact, fully written stage at
`.merged.stage-ri97qy41`: 5,000 natural, 55,000 checkpoint, and 120,003 audit
rows. Recovery binds the stage manifest SHA-256
`16ad6ae082af664a3f3afecee83568f340e839ad2220117e3f03391c4bd10509`,
merge ID `447cfc9125349369f24b3e0e6865c254b516ceb84c70263d9f0a0e36801938e6`,
merge-manifest hash
`a2f47af9378a6906c64f4f0ea9ae76d9d2f41c67be913b7c9cca5fe63dcbce03`,
and the exact Parquet hashes/counts. Dependent analysis `10383207` was
cancelled without running.

The user approved focused validation and atomic publication of this preserved
stage without duplicating the completed raw-shard scan. The new exclusive
receipt is:

```text
/home/mila/c/chenje/my-project/results/part1/6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c/validation/merge_stage_recovery_submission_receipt.json
```

It records finalize job `10385970` and final analysis job `10385971` with exact
`afterok:10385970`. Both were pending at submission; the latest check has
finalize running on `cn-h001` and analysis dependency-held. Do not resubmit or
alter the preserved stage. Resume with read-only monitoring:

```bash
ssh mila
cd /home/mila/c/chenje/my-project-recovery-95a434c
python -m json.tool /home/mila/c/chenje/my-project/results/part1/6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c/validation/merge_stage_recovery_submission_receipt.json
squeue --jobs=10385970,10385971 --format='%i %T %M %R'
sacct -j 10385970,10385971 --format=JobIDRaw,State,ExitCode,Elapsed,NodeList --parsable2
tail -n 80 logs/part1-recover-merge-stage-10385970.out
tail -n 80 logs/part1-analyze-stage-recovery-10385971.out
```

The remaining content in this section records the generation and first waiver
recovery that led to the preserved stage.

Production generation is complete. The production model-run ID is
`6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c`,
bound to generation commit `ffa998a7ee1f156e150c8da33b258165ee53e032`.
Array `10362272` produced the production shards; two tasks that encountered a
one-time `raw_shards` directory-creation race were completed by targeted retry
`10362285`. All 500 shard directories now contain `.finalized`.

This is strong persistence and shape evidence, but it is not a successful
standard coverage result. Recovery validation `10381201` scanned the completed
outputs for `06:19` and preserved this failed report without publishing a
merge:

```text
results/part1/6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c/validation/coverage_report.json
validation ID: 2bfe7cd6908351e3f1d6c9a2eec4f41c9dfa97f124f9da2f70925365490f23db
observed:      500 shards, 5,000 natural rows, 55,000 checkpoint rows
warnings:      0
```

The report is not ready because the standard validator compares each natural
row's content-derived `prompt_hash` with the model-run manifest's global prompt
contract hash. Those are intentionally different hash domains. A sampled
production row's stored hash exactly matched a fresh canonical recomputation
from its `prompt_version`, rendered prompt, and prompt token IDs. The same
single comparison rejected all 5,000 natural rows; all 55,000 checkpoint
failures are the parent-rejection cascade. The report also contains the
resulting physical-count error, for 60,001 structural errors total. This is a
validator-contract defect, not evidence that the generated records are wrong.

The user authorized an exact, recovery-only prompt-hash waiver. It must preserve
the failed standard report byte-for-byte, accept only the fingerprint above,
recompute and verify every natural row's content-derived prompt hash while the
merge already reads all records, and retain all normal schema, lifecycle,
hierarchy, duplicate, source-snapshot, and exact-count checks. Any unrelated
defect must still fail closed. Because the failed report's incompatibility
partition masks terminal outcomes, recovery merge must additionally require
`natural_execution_outcome=complete` on all 5,000 natural rows and
`checkpoint_execution_outcome=complete` on all 55,000 checkpoint rows before
paper readiness. Standard validation and standard merge/analysis behavior
remain unchanged.

Recovery commit `1a4b6039758cd8fd84b68f74c0828e6c5f382dae` was independently
reviewed and deployed in separate Mila worktree
`/home/mila/c/chenje/my-project-recovery-1a4b603`. The production checkout
remains pinned at `ffa998a7ee1f156e150c8da33b258165ee53e032`. The exclusive
launcher submitted this chain at `2026-08-16T03:28:50Z`:

```text
waiver preparation: 10383205
merge/check:        10383206  afterok:10383205
final analysis:     10383207  afterok:10383206
```

Historically, preparation completed `0:0` in four seconds and published
waiver ID `dc6d50b0a4712eedac891bb55b794b532ea5e9232f6712b85536c196851f9163`;
merge later failed as documented above and analysis was cancelled. Do not
resubmit that chain. Do not rerun generation or the six-hour standard validator.

Historical first-waiver submission command (do not run again):

```bash
uv run python scripts/submit_part1_prompt_hash_waiver_recovery.py \
  --production-repository-root /home/mila/c/chenje/my-project \
  --model-run-id 6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c
```

The launcher exclusively creates and incrementally updates:

```text
/home/mila/c/chenje/my-project/results/part1/6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c/validation/prompt_hash_waiver_recovery_receipt.json
```

Do not resubmit if that receipt exists. The three logs are written under the
recovery worktree's `logs/` as `part1-waiver-prepare-<job-id>.out`,
`part1-merge-waiver-<job-id>.out`, and
`part1-analyze-waiver-<job-id>.out`.

## Read this first

The scientific contract is `AGENTS.md` plus
`docs/part1/{PLAN,DECISIONS,STATUS,SCHEMA,RUNBOOK,VALIDATION}.md`. The unrelated
untracked `METHODS_EXPERIMENTAL_DESIGN.md` is user-owned: do not modify, stage,
commit, delete, or ignore it.

Never load a model, tokenizer, or Hugging Face dataset on a login node. Model
and dataset loading belongs only in SLURM jobs.

## Historical failed-closed chain

The first unattended bootstrap was bound to commit
`96822f88c0a38bac4a35f29e90caf601831e7f50`:

| Role | Job ID | Final state | Evidence |
|---|---:|---|---|
| Full-shape synthetic acceptance | `10347033` | `TIMEOUT` | `12:00:20`; fixture and strict coverage completed, merge incomplete |
| Production gate | `10347034` | `CANCELLED` | exact `afterok:10347033` remained unsatisfied |

No production model-run manifest, production receipt, GPU array, merge, or
analysis was created. Preserve these logs and receipt as historical evidence:

```text
logs/part1-full-acceptance-10347033.out
logs/part1-production-gate-10347034.out
results/part1-submission/96822f88c0a38bac4a35f29e90caf601831e7f50/bootstrap_receipt.json
```

The user explicitly waived another full-shape rehearsal. Do not resubmit job
`10347033` or run `jobs/part1_full_acceptance.sh` as a production prerequisite.

Focused-readiness attempt `10357631` found a pure cross-platform fixture issue:
Mila's pytest `tmp_path` resolves under forbidden `/tmp`, causing the
ephemeral-root error to precede the mismatch error expected by one config test.
Its dependent gate is `10357632`; it did not submit production. The fixture now
uses a non-ephemeral configured-root mismatch and must be verified before a new
recovery commit is submitted.

Readiness attempt `10362018` confirmed the issue was job-wide when another
`tmp_path` test encountered the same `/tmp` guard. Readiness `10362018` and gate
`10362019` were cancelled; no production job was submitted. The minimal
job-level fix sets pytest `--basetemp` to
`$SCRATCH/part1-launch-readiness-$SLURM_JOB_ID`, giving every test a unique
non-ephemeral temporary root without editing tests individually.

Readiness `10362106` proved a global `$SCRATCH` basetemp is also unsuitable:
tests intentionally exercising ephemeral-root refusal require pytest's normal
temporary location. It was cancelled, gate `10362107` did not run, and no
production job was submitted. Final readiness is deliberately limited to the
16 launch-critical tests in `test_submit_part1_unattended.py`,
`test_submit_part1_production_chain.py`, and `test_part1_launch_plan.py`.
Immutable-manifest and clean-commit validation remain mandatory inside the
production gate. All retained smokes remain historical evidence.

Readiness `10362161` passed all 16 launch-critical tests in `13.18s`. Gate
`10362162` then failed before production because Smoke A/B were generated under
the older shard lifecycle and contain no `.finalized` file required by the
hardened current validator. This prompted an attempted Phase 3-only live gate;
Smoke A/B remained immutable historical evidence with their previously
accepted counts and job outcomes.
No legacy artifact is mutated or retrofitted, and no production receipt was
created by gate `10362162`.

Readiness `10362197` passed in `10s`; gate `10362198` then showed that the
earlier Phase 3 smoke also predates the final prompt-hash contract. It failed
before manifest creation and created no production receipt or job. Retained
smokes are therefore historical evidence, not live launch inputs. Actual
production outputs still undergo strict post-generation validation before
merge or analysis.

## Historical pre-generation recovery chain

The following entry point was the approved production-launch path and is now
historical because production generation has completed. Do not resubmit it:

```bash
uv run python scripts/submit_part1_unattended.py
```

It atomically records and submits:

1. a CPU-only focused regression job with a one-hour ceiling and marker
   expression `not part1_full_acceptance`;
2. a production gate with exact `afterok` on focused readiness;
3. after the gate validates immutable manifests, the clean commit, and launch plan,
   generation array `0-499%16`;
4. real-output validation with `afterany` on generation and a 12-hour ceiling;
5. canonical merge with `afterok` on validation and a 24-hour ceiling; and
6. final 5,000-bootstrap analysis with `afterok` on merge and a 36-hour ceiling.

The bootstrap receipt is
`results/part1-submission/<recovery-commit>/bootstrap_receipt.json`. It must use
schema `part1-submission-bootstrap-v2`, record
`acceptance_mode=focused_readiness_v1`, and contain readiness plus gate job IDs.
Once the gate runs, the authoritative production receipt is
`results/part1/<model-run-id>/submission_receipt.json`; it records readiness,
gate, generation, validation, merge, and analysis IDs. Never guess or duplicate
those jobs.

## First commands after reconnecting

```bash
ssh mila
cd /home/mila/c/chenje/my-project

find results/part1-submission -name bootstrap_receipt.json \
  -exec python -m json.tool {} \;
find results/part1 -name submission_receipt.json \
  -exec python -m json.tool {} \;
```

Use the newest receipt's exact IDs with `squeue` and `sacct`. Inspect the
corresponding logs under `logs/`; focused readiness logs use
`part1-launch-readiness-<job-id>.out`.

## Failure handling

- If focused readiness fails, the gate must remain cancelled and no production
  job may exist. Preserve its log and diagnose before resubmission.
- If the gate fails, inspect both receipt trees and reconcile any `in_flight`
  submission by deterministic Slurm name/comment before taking action.
- Validation intentionally runs after any generation outcome. If it reports
  missing or invalid shards, merge and analysis must remain blocked. Completed
  raw shards are append-only and reusable for a targeted resume.
- Once the production manifest exists, do not advance the Mila checkout away
  from its recorded commit while generation or downstream jobs are active.

## Required final artifacts

Successful completion requires `paper_analysis_ready: true`, three merged
Parquet datasets, an analysis manifest and summary, eight CSV tables with
metadata, and six plots: primary AUROC, checkpoint ECE, natural reliability,
checkpoint reliability, within-question paired differences, and
switching/stabilization. At that point only interpretation and report
organization remain.

## Established evidence

- Immutable bundle: 500 questions, 100 per subject; question hash
  `dd379f48322d6eb07c309101361738a965be320b4124bb45bd44723b1abe474d`;
  study hash `859eecd5e0437a901555d5fd2d99692feccb5257df16de60bfe0fe648626142b`.
- Smoke A `10284742`, Smoke B `10292530`, and Phase 3 smoke `10324103`
  completed successfully. Phase 3 produced exactly 10 natural, 110 checkpoint,
  and 240 audit records with direct integrity validation.
- The corrected synthetic fixture passed a real one-shard production scan and
  hundreds of focused/regression tests. Full-shape job `10347033` completed
  strict coverage before timing out during merge; no integrity defect appeared.
- The exact focused-readiness command passed locally with 797 tests and one
  full-shape test deselected in `943.13s`.
- Gate `10362215` submitted production array `10362218`, but tasks failed before
  Python/model startup with Slurm exit `64`: the default CPU bind mask was
  outside each step allocation. The array was cancelled and wrote no scientific
  shards. The generation wrapper now uses `srun --cpu-bind=none`; changed code
  requires a new commit-bound production manifest rather than reuse of the old
  model run.

# Part 1 production recovery handoff

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

## Approved recovery chain

Run only this entry point from a clean, exactly deployed recovery commit:

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

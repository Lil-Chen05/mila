# Part 1 validation ledger and acceptance matrix

## Purpose and evidence rules

This file records Prompt 1 audit evidence and the checks required before each
later phase can be called complete. A green legacy test suite is not evidence
that the new scientific protocol is implemented. Exact scientific behavior is
accepted only through focused tests, schema/hash validation, and—where a model
or dataset is involved—an explicitly authorized bounded SLURM job on a compute
node.

Never run a model, tokenizer, or Hugging Face dataset on a login node. Pure
parsers, serializers, schemas, persistence helpers, synthetic analysis, and
small local JSON checks may be login-safe. See [DECISIONS.md](DECISIONS.md),
[SCHEMA.md](SCHEMA.md), [PLAN.md](PLAN.md), and [RUNBOOK.md](RUNBOOK.md).

Historical 20q/200q outputs and their passing tests remain pilot evidence only.
They do not validate the Part 1 target, sampling, provenance, status,
checkpoint-alias, calibration, or bootstrap contracts.

## Prompt 1 audit ledger

### V1 — Git baseline and scope

- Action: identify baseline, branch state, and recent history.
- Commands:

  ```bash
  git status --short --branch
  git log --oneline --decorate -12
  git rev-parse HEAD
  git rev-parse ce113b5
  ```

- Environment: local repository inspection; no model/dataset load.
- Result: before documentation drafting, `main` was at
  `01ed450be07ac346c148ba0ec1846b5770fd9838`, one commit ahead of
  `origin/main`; baseline `ce113b5060de2d7a2b0266e5527be2df4a57a5e9` was
  `origin/main`; tracked worktree was otherwise clean.
- Warning: this result was captured before the root-reviewed docs-only follow-up
  commit; Git history and current status are authoritative afterward.

### V2 — Prompt 1 working-agreement commit

- Action: verify commit identity and file scope.
- Commands:

  ```bash
  git show --stat --oneline 01ed450
  git show --format=fuller --no-ext-diff 01ed450 -- AGENTS.md
  git diff --stat ce113b5..HEAD
  ```

- Environment: local Git inspection.
- Result: commit
  `01ed450be07ac346c148ba0ec1846b5770fd9838`, subject
  `docs: update Part 1 working contract`, created only `AGENTS.md` with 144
  inserted lines. Independent verification found it correct and scoped.
- Warning: `CLAUDE.md` and `README.md` remain tracked and historically worded;
  `AGENTS.md` plus `docs/part1/` are authoritative for the four-prompt sequence.

### V3 — Local `.superpowers/` exclusion and unrelated state

- Action: verify local exclusion and list unignored untracked files.
- Commands:

  ```bash
  git check-ignore -v .superpowers .superpowers/
  git ls-files --others --exclude-standard
  ```

- Environment: local Git metadata.
- Result: both `.superpowers` forms matched
  `.git/info/exclude:7:.superpowers/`; no unignored untracked files were listed
  before documentation drafting.
- Warning: ignored `.venv/` and `.pytest_cache/` are local environment/cache
  state. `.superpowers/` remains user/tool-owned and was not edited or committed.

### V4 — Architecture and baseline-claim audit

- Action: inspect current scripts, jobs, tests, analyses, root docs, ignores, and
  tracked historical results.
- Representative exact commands:

  ```bash
  rg --files -g '!data/**' -g '!results/**' -g '!.superpowers/**'
  sed -n '1,437p' scripts/checkpoints.py
  sed -n '1,116p' scripts/mc_common.py
  sed -n '1,72p' scripts/merge_shards.py
  sed -n '1,47p' scripts/fetch_mmlu.py
  rg -n 'do_sample|MAX_NEW_TOKENS|MIN_THINK_TOKENS|round\(|MODEL_NAME|open\(' scripts analysis
  rg -n 'final_correct|think_closed|confidence|AUROC|bootstrap' analysis tests
  git ls-files 'results/**' 'analysis/**' 'docs/**' README.md CLAUDE.md
  ```

- Environment: static local inspection; no model/dataset load.
- Result: findings are recorded in [STATUS.md](STATUS.md). All baseline claims
  were verified with one qualification: mutable `MODEL_NAME` exists but is not
  an immutable or persisted identity.
- Warnings: the highest risks are target/cohort mismatch, absent provenance,
  crash-unsafe persistence, parser contamination, lost checkpoint aliases,
  incomplete/rounded uncertainty fields, and non-atomic merge publication.

### V5 — Fresh login-safe test suite

- Action: run all existing pure unit tests after the `AGENTS.md` change.
- Command:

  ```bash
  uv run pytest -q
  ```

- Environment: login-safe local test environment; Python 3.12.13, pytest 9.1.0;
  no model, tokenizer, or dataset load.
- Result: **21 passed in 0.79s** at the final pre-commit root verification gate.
- Warnings:
  - These are legacy tests in `tests/test_mc_common.py` and
    `tests/test_analyze_200q.py`; they cover parser tolerance/token location and
    three small 200q analysis helpers.
  - Some tests pin behavior that the Part 1 contract deliberately replaces,
    including independent last-match parsing and confidence clamping.
  - They do not cover new schemas, hashes, seeds, manifests, persistence,
    locking, resume, stochastic generation, aliases, status/nullability,
    bootstrap, calibration, merge safety, or the primary target.

### V6 — Documentation-only scope

- Action: verify that Prompt 1 documentation work touches only the authorized
  files and contains no placeholder markers or production claims.
- Commands to run after the documentation patch:

  ```bash
  git status --short docs/part1
  rg --files docs/part1
  rg -n '[F]IXME|T[B]D|T[O]DO:' docs/part1
  rg -n '[[:blank:]]+$' docs/part1
  ```

- Expected result: one untracked `docs/part1/` directory containing exactly the
  six authorized Markdown files; placeholder and trailing-whitespace scans
  return no matches.
- Environment: local text/Git validation.
- Warning: root agent, not the documentation agent, owns final review and commit.

## Phase 1 validation matrix — foundations

All Phase 1 tests are pure and login-safe. Test fixtures are synthetic and must
not load a model/tokenizer/dataset.

| Area | Required tests | Acceptance |
|---|---|---|
| Schema presence/types | Minimal valid question, manifest, natural, checkpoint, and event records; each required field removed in turn; wrong types and unknown required enums | Valid fixtures pass; every invalid mutation fails with field-specific diagnostics. |
| Natural status/nullability | Complete/closed, missing-close, zero-reasoning, missing/malformed/out-of-domain answer, missing/malformed/out-of-range confidence, and terminal infrastructure failure | Every required/null combination matches [SCHEMA.md](SCHEMA.md); missing-close forces natural answer/correctness/normalized confidence null. |
| Checkpoint status/nullability | Complete-valid, complete-invalid, missing answer step, entropy failure, out-of-range confidence, and terminal infrastructure failure | Executed invalid output remains `complete`; infrastructure failure has no scientific measurements; illegal combinations fail. |
| Eleven identities/aliases | `n_reasoning` values 0, 1, short values causing ties, and ordinary lengths; ties-to-even cases | Exactly eleven requested identities always exist for complete natural execution; aliases share probe ID/result and retain unique requested indices. |
| Canonical bytes | Same object with different insertion order/whitespace; Unicode and escaping cases; integers/floats/null/arrays; newline behavior | Canonical bytes are byte-identical where semantic identity is identical and differ for every scientific change specified by the payload. |
| Golden identities | Question content/ID, question-manifest, study ID/hash, model-run ID/hash, raw/checkpoint/shared-probe/event IDs | Golden lower-hex SHA-256 values recompute exactly; self fields and mutable operational fields cannot affect them. |
| Seed algorithm | Golden vectors for base seed 42, canonical model ID, stable question ID, run IDs 0 and 9; Unicode/field-boundary collision cases; range extremes | Deterministic exact integers across processes; no delimiter ambiguity; values lie in the locked supported nonnegative PyTorch range; built-in `hash()` is absent. |
| Compatibility | Same study across compatible raw versions; mismatched study/question hash/protocol/analysis version; unknown schema | Only explicitly compatible combinations pass; one shard cannot mix model-run manifests. |
| Atomic persistence | Publish valid record; inject interruption before/during/after flush/rename; truncated journal tail; restart recovery | No partial terminal record appears as complete; prior valid data survives; recovery is deterministic and idempotent. |
| Duplicate suppression | Two writers target one logical run/checkpoint; repeated resume; duplicate terminal event | Exactly one terminal publication wins; duplicate attempts are auditable and cannot duplicate scientific rows. |
| Locking | Contention, owner metadata, timeout, verified stale lock, live lock misclassified as stale | One owner at a time; stale recovery follows the locked Mila policy; live locks are never stolen. |
| Failures/retries | Each locked infrastructure category, nonretryable failure, retry exhaustion, executed abnormal output, checkpoint-1.0 disagreement, repetitive text | Only approved infrastructure categories retry; finite count/backoff is recorded; seed is unchanged; abnormal/disagreement/repetition never retry automatically. |
| Resume planner | Mix of complete runs, partial eligible checkpoints, retryable/exhausted failures, invalid hashes, and successful abnormal outputs | Schedules only legitimate missing/retryable work; rejects corrupt/incompatible state; repeated planning is stable. |

Phase 1 completion additionally requires:

- exact canonical payload/serialization and raw-record-granularity decisions in
  [SCHEMA.md](SCHEMA.md) replaced by versioned, tested choices;
- the selected persistent root and lock/stale policy documented;
- `uv run pytest -q` green from the repository root; and
- confirmation that no production model-run manifest, model load, dataset load,
  or full experiment was performed.

## Phase 2 validation matrix — questions and generation

Pure adapter/placement tests may run on the login node. Any real dataset,
tokenizer, or model check runs in the resource-appropriate SLURM job.

| Area | Environment | Required validation and acceptance |
|---|---|---|
| Question selection | CPU compute node | Pinned `cais/mmlu` test revision; streaming plus bounded selection; exactly 500 records, five ordered blocks of 100, no replacement, indices 0–499, seed 42, reproducible bytes/hashes. |
| Source identity | CPU compute node + pure validator | Every record maps audibly to the pinned source; duplicate question text cannot collapse distinct source rows accidentally. |
| Study manifest | Login-safe pure | All fixed decisions and exact registry are present; question hash recomputes; compatible raw versions/analysis contract are explicit; ID/hash golden checks pass. |
| Parser/boundaries | Login-safe synthetic token/text fixtures | Opening/closing tokens excluded; no-open/no-close rules exact; one terminal post-close block; answer/confidence paired; unclosed answer-like diagnostics never become targets; no minimum-length drop. |
| SmolLM3 preflight | Single-GPU compute node | Immutable model/tokenizer revisions, tag sequences, `</think>\nAnswer:` inducer IDs, A–D encodings, answer-step location, bf16/eval/batch-one, environment, and effective settings are recorded and validated. |
| Run seeds | Login-safe pure plus bounded GPU smoke | Run IDs 0–9 map to ten deterministic seeds; same inputs reproduce seeds; different run IDs are distinct under golden cases; retries reuse seed. |
| Natural sampling | Explicitly authorized bounded single-GPU smoke | Exactly ten configured logical runs in the smoke plan where authorized, `do_sample=True`, temperature 0.6, top-p 0.95, top-k 50, 8,192 cap, no extra greedy natural run, no batching. |
| Raw natural entropy | Bounded GPU smoke + pure validation | Logits used before warpers and cast float32; nats; unrounded values; token/entropy arrays exactly align; mean and `ceil(10%)` tail recompute; no logits/selected-token log-prob stored. |
| Checkpoint probes | Bounded GPU smoke | Greedy, 32-token cap, token-ID prefixes, all eleven identities, alias sharing, raw A–D logits/probabilities/entropy/max and full-vocab entropy at located answer step. |
| Abnormal behavior | Synthetic plus bounded observed cases | Complete capped/missing-close/short/zero/malformed runs remain stored and checkpoint-eligible; only natural infrastructure failure is ineligible. |
| Smoke provenance | Pure validator | Separate smoke root; `production=false`; dirty smoke records base commit/diff hash; no production manifest/output is created. |
| Seed traceability and probe determinism | Pure validation plus bounded authorized cases | Every attempt records the correctly derived seed and retries reuse it; greedy probes are deterministic for identical recorded prefixes/settings. No extra cross-hardware bitwise natural-output invariant is imposed. |

Phase 2 does not pass on mocked tests alone. It requires the explicitly
authorized bounded compute-node preflight/smoke, but never the full 500-question
run.

## Phase 3 validation matrix — analysis and production readiness

| Area | Required synthetic/operational checks | Acceptance |
|---|---|---|
| Fixed AUROC registry | Registry equality, sign orientation, `natural_correct` target, ties/missing data, one-class cohorts | Exactly eleven approved primary features; larger always means more likely correct; checkpoint-local AUROC labelled secondary. |
| Bootstrap multiplicity | A tiny five-subject fixture where a question is drawn 0/1/3 times; explicit draw IDs/weights | All associated runs/checkpoints contribute exactly by draw multiplicity; no `.isin` reconstruction. |
| Bootstrap validity | Subject and pooled one-class replicates; macro with one invalid subject; separate 949-valid and 950-valid cases per 1,000 | Invalid counts reported; macro valid only with all five; interval valid only at least 95%; otherwise point estimate plus warning and null/invalid interval. |
| Bootstrap scales | Development and final configuration tests | 1,000 and 5,000 requested replicates respectively, seed 42, subject-stratified question sampling, 95% percentile interval. |
| Calibration | Exact values on bin edges including 0 and 1; empty bins; out-of-range raw confidence; each fraction | Ten equal-width bins, 1.0 in final bin, explicit empty bins, count-weighted ECE; no entropy ECE, no clamping, no primary fraction pooling. |
| Reporting scopes | Unequal subject sample fixture | Pooled, each subject, and arithmetic five-subject macro are distinct and correct; all eleven fractions retained, main summary shows 0.0/0.5/1.0. |
| Within-question | Questions with all-correct/all-wrong/mixed runs; repeated bootstrap draws | Only mixed questions qualify; correct-minus-incorrect feature means; equal question weight; median/distribution and multiplicity-preserving CI. |
| Switching | Aliases, valid changes, `A -> missing -> B`, malformed gaps | Only adjacent valid non-aliased changes count; missing/malformed breaks adjacency; aliases never add switches. |
| Appearance/stabilization | Leave/return paths, missing natural answer, invalid 1.0, missing later checkpoint, stable suffix | First appearance and final-answer-relative stabilization follow exact definitions; required invalid/missing cases return null. |
| Recovery/agreement | Switch away from correct then recover; natural/1.0 agree/disagree/missing | Correct-away and recovery flags are exact; disagreement is retained secondary data and never validation/retry failure. |
| Raw validator | Missing/duplicate run/checkpoint, mixed manifests, invalid hashes/status/nulls, token mismatch, alias mismatch | Every defect is fatal before merge; valid abnormal outputs pass with explicit status. |
| Merge safety | Missing shard/qid/run/checkpoint, duplicate, injected crash, pre-existing valid merged file | Validate before publish; failure leaves prior valid output unchanged and no incomplete final path; successful merge has recomputed hash/coverage. |
| SLURM safety | Static launcher audit and bounded compute smoke | CPU jobs request no GPU; GPU one model/GPU/batch-one; `HF_HOME` set before HF load; persistent outputs; no `$SLURM_TMPDIR` reliance. |
| End-to-end smoke | Prompt-4-authorized bounded data only | Materialized references through generate/resume/validate/merge/analyze; distinct smoke paths; rerun is idempotent; no production work generated. |
| Production manifest gate | Dirty and clean worktree fixtures; ignored-path check; recorded final commit | Dirty tracked worktree rejected; manifest created only after final commit; correct immutable revisions/settings/env/output; creation leaves Git clean. |

## Production-readiness evidence packet

Before Phase 3 is reported complete, retain or link:

- full login-safe test output;
- CPU question-manifest job ID/log and validation report;
- GPU SmolLM3 preflight report and immutable revisions;
- bounded smoke job IDs/logs, manifest/hash, output root, and diff provenance if
  dirty;
- resume/failure-injection evidence;
- raw validation and atomic merge reports;
- synthetic analysis test output for bootstrap/calibration/switching;
- final tracked Git commit and clean-worktree evidence;
- production model-run manifest hash plus post-creation clean-worktree evidence.

This packet establishes readiness only. It is not authorization to run the full
500-question experiment.

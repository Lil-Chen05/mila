# Part 1 status

## Executive status

**Prompt 1 work is complete and awaiting Prompt 2.** The repository was audited,
the current working agreement and authoritative Part 1 documentation were
committed, and `.superpowers/` was locally excluded. No Phase 1
production pipeline code, immutable production model-run manifest, or new model
or dataset artifact has been created. The full 500-question experiment has not
been launched.

This status is read with [PLAN.md](PLAN.md), [DECISIONS.md](DECISIONS.md),
[SCHEMA.md](SCHEMA.md), [RUNBOOK.md](RUNBOOK.md), and
[VALIDATION.md](VALIDATION.md). `README.md`, `CLAUDE.md`, older `docs/` files,
and 20q/200q results and analyses remain tracked historical artifacts; they do
not define the current protocol.

## Repository baseline and Prompt 1 changes

The audit baseline was commit
`ce113b5060de2d7a2b0266e5527be2df4a57a5e9` (`ce113b5`,
`Add entropy trace and signal distribution analyses`) on `main`, also
`origin/main` at audit time. That tracked revision contained the legacy
greedy 20q/200q experiment, tracked pilot outputs/analysis, an outdated
`README.md` and `CLAUDE.md`, and no tracked `AGENTS.md`.

Prompt 1 committed only the new working agreement:

- commit: `01ed450be07ac346c148ba0ec1846b5770fd9838`
- subject: `docs: update Part 1 working contract`
- scope: new `AGENTS.md`, 144 lines; no production/test/job changes

The commit was independently verified as correct and scoped. At the start of
documentation drafting, `main` was one commit ahead of `origin/main` and the
tracked worktree was clean. The six authoritative `docs/part1/` files were then
root-reviewed and added in a separate docs-only Prompt 1 commit. Its hash is not
embedded here because doing so would make this document self-referential; Git
history is the authority for that commit identity.

`.superpowers/` is local tool state and is now excluded by the exact line
`.superpowers/` in `.git/info/exclude`. It was not edited, deleted, added to
shared `.gitignore`, or committed.

There were no genuinely unrelated unignored untracked files. Ignored `.venv/`
and `.pytest_cache/` entries are local environment/cache state, and the locally
excluded `.superpowers/` is tool state. Tracked historical results are not
untracked files and remain untouched.

## Verified architecture at the baseline

| Area | Verified implementation |
|---|---|
| Model path | `scripts/checkpoints.py` imports Hugging Face Transformers `AutoTokenizer` and `AutoModelForCausalLM`; this is the active model interface. |
| Core execution | `process_question` generates one full natural chain and then greedily probes checkpoints in the same process. Both natural generation and probes currently set `do_sample=False`. |
| Model configuration | `MODEL_NAME` is mutable through the environment and defaults to `HuggingFaceTB/SmolLM3-3B`; it is printed but neither an immutable revision nor a stable persisted identity. |
| Legacy scale | `scripts/fetch_mmlu.py` globally shuffles the `cais/mmlu` test split with seed 42 and takes 200 rows. It does not enforce the five fixed subject quotas/order. |
| Generation limits | Natural maximum defaults to 16,384 tokens; forced checkpoints use 32. Current production contract instead requires natural maximum 8,192. |
| Checkpoints | Eleven requested deciles are defined, but `checkpoint_indices` deduplicates equal `k_keep` values. `process_question` writes only the minimum fraction for each unique prefix, so requested alias identities are lost. |
| Reasoning/output handling | The first `</think>` token closes reasoning. Chains with fewer than eight reasoning tokens are dropped. Missing-close chains can be probed, but current aggregate analysis excludes them. |
| Parser | `mc_common.py` independently chooses the last answer match and last confidence match anywhere in text, clamps confidence, and is not gated on a terminal post-close block. `find_answer_token` likewise is not post-close gated. |
| Entropy | Current helper casts raw logits to float32 and reports nats. Natural per-token entropy is stored for all generated tokens but rounded to four decimals; mean reasoning entropy is unrounded. |
| Missing measurements | Generated natural token IDs and full decoded natural text are not persisted. Raw A–D logits/probabilities, maximum A–D probability, explicit answer-step/location/entropy statuses, tail entropy as a raw field, seeds, and complete provenance are absent. |
| Persistence | Records accumulate in memory and each shard output is opened with `w` only at shard end. A crash can lose the full shard; a rerun overwrites it. There is no atomic journal, lock, terminal-record index, or resume protocol. |
| Failures | Per-question exceptions are printed to stdout and skipped. There is no durable structured failure record, attempt identity, retry classification, or same-seed retry. |
| Identity/provenance | No schemas, schema versions, question/study/model-run manifests, raw/event IDs, canonical hashes, immutable model/tokenizer revisions, run IDs, or generation seeds exist. |
| Sharding/merge | Sharding uses `qid % NUM_SHARDS`. Merge checks missing shard files and ownership but tolerates missing question IDs, writes the destination before all validation finishes, and may leave incomplete output even when it exits nonzero. |
| Analysis | `analysis/analyze_200q.py` targets forced checkpoint-1.0 correctness, not `natural_correct`; aggregates only closed/answered chains and reports truncated/short cohorts separately. It uses legacy custom confidence bins, has no specified stratified bootstrap, and omits the fixed primary registry, within-question sampled-run analysis, maximum A–D probability, and required switching/stabilization rules. |
| Switching | Legacy 20q analysis counts any adjacent stored-value change, including `None`, and calls stable suffix start a commit fraction; it is not alias/missingness compliant. |
| Historical data | `results/20q`, `results/200q`, `analysis/20q`, and `analysis/200q` are tracked pilot artifacts with unversioned legacy records. |

## Baseline claims: verified and qualified

The Prompt 1 baseline claims were verified: Transformers is active;
`scripts/checkpoints.py` combines natural generation and probes; both paths are
greedy; natural full text/token IDs are not persisted; shard writes occur only
after the shard loop; no per-run seed/resume or durable structured failure
exists; legacy analysis uses checkpoint-1.0 correctness; 20q/200q assumptions
remain; confidence is clamped; short chains are dropped; and checkpoint
fractions are deduplicated.

The only qualification is the claim that there is “no model ID”: a mutable
`MODEL_NAME` configuration/default exists and is printed. It is not an immutable
model revision, canonical model identity, manifest field, or value persisted in
raw records, so it does not satisfy the new provenance contract.

The audit also established details not explicit in the baseline claims:

- current natural cap is 16,384, not the required 8,192;
- per-token entropy is rounded to four decimals;
- the parser can pair unrelated answer/confidence matches and accept text inside
  unclosed reasoning;
- checkpoint 1.0 is called an invariant, but disagreement is only reported and
  not asserted; the new protocol correctly treats it as a secondary outcome;
- the current merge can publish incomplete output before reporting failure; and
- the Part 1 uncertainty/status fields and two-level provenance are almost
  entirely absent.

## Completed, current, and outstanding work

### Completed in Prompt 1

- Audited Git state, current scripts, jobs, tests, analyses, root documentation,
  ignores, and historical outputs.
- Verified all baseline claims and the one qualification above.
- Replaced the obsolete working contract with committed `AGENTS.md` while
  preserving cluster, `uv`, storage, and Git safety rules.
- Locally excluded `.superpowers/` without modifying shared `.gitignore`.
- Established the fixed science, schemas, runbook, validation matrix, and exact
  three-phase plan in `docs/part1/`.
- Independently ran the fresh login-safe legacy suite; the final root gate
  confirmed 21 tests passed in 0.79s.

### Current boundary

Stop after the root-reviewed docs-only Prompt 1 commit and wait for Prompt 2.
There are no intended uncommitted project changes at this handoff. Phase 1 is
not authorized until Prompt 2.

### Outstanding by phase

- Phase 1: executable schemas; canonical bytes and IDs; seed algorithm;
  persistence, locking, failure events, retry policy, and resumability.
- Phase 2: immutable dataset/model/tokenizer revisions; fixed question and study
  manifests; SmolLM3 adapter/preflight; ten sampled natural runs; exact raw
  entropy; all eleven checkpoint identities; separate smoke artifacts.
- Phase 3: analysis, bootstrap, calibration, switching/stabilization, validator,
  atomic merge, SLURM readiness, production-manifest lifecycle, and final
  bounded smoke.
- After this four-prompt sequence and only with separate approval: launch the
  full production experiment and add later models.

## Risks, blockers, and deferred decisions

Highest-priority risks are:

1. target/cohort mismatch from using forced checkpoint-1.0 correctness and
   excluding successful abnormal chains instead of targeting
   `natural_correct` with explicit missingness;
2. missing provenance and mutable model identity;
3. crash-unsafe buffered persistence and overwrite-on-rerun;
4. parser contamination across reasoning/answer blocks;
5. loss of checkpoint aliases and therefore incorrect trajectory semantics;
6. missing/unrounded uncertainty fields, including exact token IDs/text, raw A–D
   values, maximum probability, and complete statuses; and
7. merge publication before completeness validation.

There is no Prompt 1 blocker. The following are genuine repository-specific
decisions assigned to later phases, with acceptance criteria in
[SCHEMA.md](SCHEMA.md) and [VALIDATION.md](VALIDATION.md):

- Phase 1: raw-record granularity; canonical payload details; retryable failure
  taxonomy/count/backoff; Mila lock/stale-lock policy; persistent output root;
  atomic journal format; narrow ignore patterns.
- Phase 2: immutable MMLU, model, and tokenizer revisions; source-row identity;
  compute-node reasoning-tag and A–D token preflight.
- Phase 3 or explicit later maintenance: whether to update/deprecate tracked
  `CLAUDE.md` and `README.md`; how narrow new ignore rules coexist with tracked
  historical results.

Until resolved, these are controlled phase gates, not placeholders or authority
to guess.

## Latest successful checks

| Command | Result |
|---|---|
| `uv run pytest -q` | 21 passed in 0.79s at the final pre-commit root gate; login-safe legacy tests only; no model/tokenizer/dataset load. |
| `git show --stat --oneline 01ed450` | Only `AGENTS.md`; 144 insertions. |
| `git diff --stat ce113b5..HEAD` | Only `AGENTS.md` before documentation drafting. |
| `git check-ignore -v .superpowers .superpowers/` | Both matched `.git/info/exclude:7:.superpowers/`. |
| `git ls-files --others --exclude-standard` | No unignored untracked files before documentation drafting. |
| `git status --short --branch` | `main...origin/main [ahead 1]` and otherwise clean before documentation drafting. |

See [VALIDATION.md](VALIDATION.md) for environment, warnings, and future
acceptance checks.

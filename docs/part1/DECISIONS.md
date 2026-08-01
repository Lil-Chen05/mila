# Part 1 scientific and engineering decisions

## Authority

This file records the immutable scientific protocol established in Prompt 1.
[PLAN.md](PLAN.md) assigns implementation work, [SCHEMA.md](SCHEMA.md) defines
the record contract, [RUNBOOK.md](RUNBOOK.md) defines operations, and
[VALIDATION.md](VALIDATION.md) defines acceptance evidence. A change to a fixed
decision requires explicit approval and a versioned study or analysis contract;
it must never be made silently in code.

The 20-question and 200-question code, documentation, results, and analyses are
historical pilots. Their greedy one-run design and cohort rules do not define
Part 1.

## Objective, model scope, and targets

Part 1 asks whether uncertainty measured throughout a reasoning trajectory
predicts whether the model's naturally generated final answer is correct.

- The primary target is `natural_correct`.
- Checkpoint-local forced-answer correctness, checkpoint-1.0 forced-answer
  correctness, agreement between checkpoint 1.0 and the natural answer, and
  execution/model-output attributes are secondary measurements.
- Natural generation is stochastic and checkpoint probing is greedy. A valid
  disagreement between the natural answer and checkpoint 1.0 is data, not an
  invariant violation, retry condition, or invalid record.
- The current implementation scope is only
  `HuggingFaceTB/SmolLM3-3B` in thinking mode. Later models must reuse the same
  study through a shared Hugging Face interface and model-specific adapters.
- Later models and the later answerable-versus-unanswerable MMLU study are out
  of scope for the current four-prompt sequence.

## Fixed question design

Use `cais/mmlu`, test split, with exactly 500 questions. Select 100 without
replacement from each subject in this exact order:

1. `high_school_mathematics`
2. `high_school_physics`
3. `high_school_chemistry`
4. `high_school_biology`
5. `high_school_psychology`

The constants are:

- `question_sampling_seed = 42`
- `base_generation_seed = 42`
- `bootstrap_seed = 42`

Manifest order is subject order above, then seeded-shuffle selection order
within each subject. `sample_index` is sequential from 0 through 499. Every
eventual model receives this exact immutable manifest in this exact order. The
question manifest and sidecar are tracked outside ignored `data/`, at the
planned locations `manifests/part1/questions.jsonl` and
`manifests/part1/questions.manifest.json`.

The immutable MMLU dataset revision is a repository-specific decision deferred
to Phase 2. Acceptance requires a resolved immutable revision, recorded source
configuration and split, exact quota/order checks, and reproducible manifest
hashes.

## Natural generation

Each model-question pair receives exactly ten natural runs with `run_id` 0
through 9. There is no additional greedy natural run. Every run uses the same
semantic prompt; only its deterministic seed changes.

Requested settings are fixed:

| Setting | Value |
|---|---:|
| `do_sample` | `True` |
| `temperature` | `0.6` |
| `top_p` | `0.95` |
| `top_k` | `50` |
| `max_new_tokens` | `8192` |
| `return_dict_in_generate` | `True` |
| `output_logits` | `True` |
| checkpoint `max_new_tokens` | `32` |

The model runs in evaluation mode with bfloat16 weights, one model on one GPU,
and batch size one. Questions, natural runs, checkpoints, and checkpoint probes
are never batched.

Both requested settings and effective settings after resolving the pinned
model's `GenerationConfig` are persisted in the model-run manifest. The exact
model and tokenizer revisions are resolved during Phase 2 preflight and must be
immutable before any production model-run identity is constructed.

### Per-run seed

The run seed is derived from a versioned canonical SHA-256 serialization of:

1. base generation seed;
2. canonical model ID;
3. stable question ID;
4. run ID.

Python's built-in `hash()` is forbidden. Phase 1 locked `part1-seed-v1`: hash
the canonical JSON payload containing `seed_algorithm_version`,
`base_seed`, `canonical_model_identity`, `question_id`, and `run_id`; interpret
the first eight SHA-256 digest bytes as an unsigned big-endian integer and mask
with `2**63 - 1`. The result is in `[0, 2**63 - 1]`. Golden vectors cover model,
question, run, and version separation. A retry reuses the original seed.

## Numerical measurements

Persist scientific values at their computed precision. Rounding is permitted
only for human-readable tables, Markdown summaries, plotted labels, and
figures. All entropy calculations cast the relevant raw logits to float32, use
natural logarithms, and report nats.

### Natural reasoning-token entropy

The primary natural token entropy is full-vocabulary entropy of raw next-token
logits before temperature scaling, top-p filtering, top-k filtering, or any
other sampling warper. Sampling chooses the generated token; sampling warpers
must not alter the distribution used for this metric.

For every executed natural chain,
`per_token_entropy_nats[i]` corresponds exactly to
`generated_token_ids[i]`; equal lengths are asserted before a terminal record
is published. Full-vocabulary logits are not stored. Selected-token log
probability is neither calculated nor stored in Part 1.

Recognized reasoning tokens follow the adapter boundary rules below.

- Mean reasoning entropy is their arithmetic mean and is null when
  `n_reasoning = 0`.
- Tail reasoning entropy is the arithmetic mean of the final
  `max(1, ceil(0.10 * n_reasoning))` recognized reasoning tokens and is null
  when `n_reasoning = 0`.

### Checkpoint answer distribution

At the identified answer-token generation step:

- persist raw A–D logits using the token convention selected at adapter
  preflight;
- compute probabilities using float32 softmax renormalized over those four
  logits only;
- compute four-choice entropy from those probabilities;
- compute maximum A–D probability from those probabilities; and
- compute full-vocabulary entropy from the raw float32 logits at that step.

The A–D token convention and exact token sequences are model-run provenance.
If the answer step cannot be identified or measured, keep execution and
measurement statuses explicit and leave unavailable values null; do not guess.

## Reasoning boundaries and natural parsing

The SmolLM3 adapter owns reasoning-boundary recognition and any model-specific
adjustment. The default contract is:

- exclude a recognized opening reasoning tag and its tokens from the reasoning
  slice;
- exclude recognized closing-tag tokens;
- if no opening tag is emitted, reasoning starts at the first generated token;
- if no closing tag is found, all generated tokens after any recognized opening
  tag are reasoning tokens; and
- if neither tag is found, every generated token is a reasoning token.

A natural final answer is valid only in one adapter-recognized terminal answer
block after a valid reasoning closure. Natural confidence is valid only when it
comes from that same block. The parser identifies one terminal block and pairs
the answer and confidence from that block; it never combines independent last
matches.

For an unclosed chain:

- `reasoning_status = missing_close`;
- `natural_answer = null`;
- `natural_correct = null`; and
- normalized natural confidence is null.

Answer-like or confidence-like text inside an unclosed reasoning block may be
preserved only as diagnostic data. Short and zero-length reasoning are genuine
model behavior. No minimum reasoning-token threshold may drop them.

## Checkpoints and aliasing

Every successfully executed natural chain has exactly eleven requested
checkpoint identities at fractions `0.0`, `0.1`, `0.2`, `0.3`, `0.4`, `0.5`,
`0.6`, `0.7`, `0.8`, `0.9`, and `1.0`. Probes are greedy and deterministic.

SmolLM3 uses the forced-close intervention `</think>\nAnswer:` inside its
adapter. Prefixes are constructed in token-ID space.

For each requested fraction:

`k_keep = clamp(int(round(requested_fraction * n_reasoning)), 0, n_reasoning)`

`round()` has Python's ties-to-even semantics.

`actual_fraction = k_keep / n_reasoning` when `n_reasoning > 0`; it is null when
`n_reasoning = 0`.

If several requested fractions map to one `k_keep`, the implementation may
probe the unique prefix once, but it must emit all eleven requested identities.
Each aliased record carries a shared-probe ID and explicit alias metadata.
Aliases are not independent transitions in switching or stabilization.

Phase 1 publication and reconstructed indexing recompute `k_keep` with Python
ties-to-even semantics, recompute `actual_fraction`, verify the requested
checkpoint ID at its parent index, derive the canonical shared-probe ID, and
derive alias owner/members and `is_alias` from all eleven parent identities.
Records for one natural parent and `k_keep` must also agree on `prefix_hash`,
inducer version/text, and shared-probe ID; aliased logical rows cannot describe
different physical probes.

Any successfully executed natural chain remains checkpoint-eligible, including
chains that are capped, missing a close, missing a natural answer or confidence,
short, zero-reasoning, or otherwise malformed at the model-output level. Only a
natural-generation infrastructure failure makes checkpoint work ineligible.

## Orthogonal outcomes and retries

Natural execution, natural output, checkpoint execution, and checkpoint output
are separate concepts; one overloaded status enum is forbidden.

Natural records use:

- `natural_execution_outcome`: `complete` or
  `terminal_infrastructure_failure`;
- `stop_reason`: `eos`, `max_new_tokens`, `stopping_criterion`, `error`, or
  `other`;
- `reasoning_status`: `closed`, `missing_close`, `no_reasoning`, or `malformed`;
- `answer_parse_status`: `parsed`, `missing`, `malformed`, or `out_of_domain`;
- `confidence_parse_status`: `parsed`, `missing`, `malformed`, or
  `out_of_range`.

Checkpoint records use:

- `checkpoint_execution_outcome`: `complete` or
  `terminal_infrastructure_failure`;
- `checkpoint_model_output_status`: `valid` or `invalid`; and
- separate answer-parse, confidence-parse, answer-token-location, and
  entropy-computation statuses.

A successfully executed but unparsable checkpoint is execution `complete` and
model output `invalid`, not an infrastructure failure. [SCHEMA.md](SCHEMA.md)
defines required/null fields for each outcome.

Successful abnormal output is never automatically retried or excluded.
Checkpoint-1.0 disagreement is not retryable. Part 1 has no automatic
repetition detector or repetition-based retry/exclusion; repetitive or
degenerate successful outputs are preserved. Only infrastructure failures may
be retryable. Phase 1 fixed exactly three attempts and backoffs
`[0, 30, 120]`. Retryable categories are interrupted process, temporary
filesystem failure, transient worker failure, and transient CUDA runtime
failure. Invalid configuration, schema or manifest incompatibility, tokenizer
preflight incompatibility, deterministic context overflow, reproducible CUDA
OOM, unsupported model/tokenizer behavior, and corrupt immutable manifests are
terminal on the current attempt. A transient CUDA retry requires termination
and a fresh process. All retries preserve logical identity and seed.

## Confidence and calibration

Never silently clamp confidence. Preserve the raw confidence text, raw parsed
integer, and normalized value only when the integer is valid in `[0,100]`. For
example, `250` yields `confidence_parse_status = out_of_range` and normalized
confidence null. `out_of_range` is true only for an integer `<= -1` or `>= 101`;
integers 0 through 100 must not be mislabeled. `missing` nulls all confidence
fields. `malformed` may preserve raw text but nulls parsed/normalized values.

ECE is used only for probability-like values, with these calibration pairs:

- natural verbalized confidence to `natural_correct`;
- checkpoint verbalized confidence to checkpoint-local correctness; and
- maximum normalized A–D checkpoint probability to checkpoint-local
  correctness.

Checkpoint confidence and maximum A–D probability may also receive AUROC
against `natural_correct`; that is discrimination, not calibration. Raw entropy
does not receive ECE.

Checkpoint ECE is computed separately for every requested fraction; fractions
are never pooled for a primary calibration result. Fractions 0.0, 0.5, and 1.0
appear in the main summary, and all eleven appear in machine-readable tables,
reliability summaries, and trajectory plots.

Report pooled values, each subject, and the arithmetic subject macro-average,
with bootstrap confidence intervals. Use ten equal-width bins over `[0,1]`, put
confidence exactly 1.0 in the final bin, handle empty bins explicitly, and use
count-weighted ECE. Reliability summaries are machine-readable.

## Bootstrap contract

Use subject-stratified question-level bootstrap with `bootstrap_seed = 42`,
1,000 replicates during development, 5,000 for final results, and 95% percentile
intervals (2.5th and 97.5th percentiles).

Within every replicate, independently sample the original number of questions
with replacement inside each subject. Preserve draw multiplicity and retain all
associated runs/checkpoints for every draw, using explicit draw IDs or
equivalent weights. Set-membership reconstruction such as
`.isin(sampled_question_ids)` is forbidden: a question drawn three times must
contribute three times.

For AUROC:

- a subject replicate with only one target class is invalid;
- a pooled replicate with only one target class is invalid;
- a macro-AUROC replicate is valid only if all five subject AUROCs are defined;
  never average only the valid subset of subjects.

Every interval reports requested, valid, and invalid replicate counts. At least
95% of requested replicates must be valid. Below that threshold, report the
point estimate, mark the interval invalid, and emit a clear warning. A macro
average is the arithmetic mean of all five subject metrics, not a pooled
row-level calculation.

## Fixed primary AUROC registry

Every primary feature has target `natural_correct` and is oriented so larger
means a greater predicted probability of natural correctness. The complete
authorized initial registry is:

1. negative mean reasoning-token entropy;
2. negative tail reasoning-token entropy;
3. negative answer entropy at fraction 0.0;
4. negative answer entropy at fraction 0.5;
5. negative answer entropy at fraction 1.0;
6. natural verbalized confidence;
7. maximum normalized A–D probability at fraction 0.0;
8. maximum normalized A–D probability at fraction 0.5;
9. maximum normalized A–D probability at fraction 1.0;
10. negative answer-switch count;
11. negative stabilization fraction.

Checkpoint-local AUROC is secondary and labelled separately. No primary feature
may be added without explicit approval.

## Within-question analysis

The primary analysis keeps all eligible questions. The secondary
within-question analysis includes only questions with at least one naturally
correct run and at least one naturally incorrect run.

For every qualifying question and correctness-oriented feature, calculate the
correct-run mean, incorrect-run mean, and `correct mean - incorrect mean`.
Positive differences point in the expected direction. Report the number of
qualifying questions, equally weighted mean and median question-level paired
differences, the paired-difference distribution, and a question-level bootstrap
confidence interval. Bootstrap questions with all their runs and preserve
repeated draw multiplicity.

Conditional regression and within-question AUROC do not replace this analysis
without explicit approval.

## Switching, appearance, stabilization, and agreement

An answer switch occurs only between adjacent, non-aliased requested
checkpoints when both have valid parsed answers. Missing or malformed output
breaks adjacency; `A -> missing -> B` is not an A-to-B switch. Aliases do not
create transitions.

First natural-answer appearance is the first requested checkpoint whose valid
forced answer equals the natural final answer. It is null when the natural
answer is missing or never appears. Leaving and later returning does not make
the first appearance a stabilization point.

Stabilization is defined relative to the final valid forced checkpoint answer:
the earliest requested checkpoint after which every later non-aliased
checkpoint has the same valid forced answer. It is null when checkpoint 1.0 has
no valid forced answer, any required later checkpoint is missing or malformed,
or no stable suffix can be established.

Also report whether a trajectory switches away from a correct forced answer,
whether it later recovers, and whether checkpoint 1.0 agrees with the natural
answer. Agreement is measurement, never execution validation.

## Manifest hierarchy and provenance lifecycle

Part 1 has two immutable provenance levels.

### Tracked study manifest

The model-independent study manifest records `study_id`, its schema version,
question-manifest hash, subjects/quotas, sampling seed, scientific protocol
version, checkpoint fractions, entropy definitions, natural-answer validity,
calibration, bootstrap, primary AUROC registry, within-question analysis,
switching/stabilization, compatible raw schema versions, and analysis-contract
version. It is tracked at the planned path
`manifests/part1/study_manifest.json` and remains compatible across future
models.

### Operational model-run manifest

There is one immutable model-run manifest per model revision and adapter. It
records `model_run_id`, study identity/hash, question-manifest hash, model
repository and immutable revision, tokenizer immutable revision, adapter,
prompt, parser and inducer versions, requested/effective generation settings,
model-specific token IDs/sequences, environment versions, final production Git
commit, and production/smoke provenance. Output paths/locations and mutable
progress remain outside the immutable manifest.

The manifest file itself is generated only in Phase 3 after final
generation-path tests pass, final tracked production artifacts are committed,
and the tracked worktree is clean except for intentional local exclusions. It
lives under an explicitly ignored operational result root such as
`results/part1/<model_run_id>/model_run_manifest.json`; creating it must not
dirty the tracked worktree.

Each raw shard contains records from exactly one model-run manifest. Cross-model
analysis may combine manifests only when `study_id`, question-manifest hash,
scientific protocol version, compatible schema versions, and compatible
analysis contracts agree.

Smoke manifests and output roots are separate. A smoke using uncommitted code
records the base Git commit and diff hash and is labelled non-production.

### Identity rules

Every stable identifier uses SHA-256 over an explicitly versioned canonical
identity payload. Define question-record content hash, complete
question-manifest hash, `study_id`, complete study-manifest hash,
`model_run_id`, complete model-run manifest hash, raw-record ID, and event ID
separately. Payloads exclude the ID/hash being calculated, creation timestamps,
filesystem locations, validation/completion states, runtime statistics, mutable
notes, and any value that can change without changing scientific identity.

Complete manifest hashes may include immutable descriptive fields omitted from
shorter identity payloads, but always exclude their own hash field. Phase 1
locked and golden-tested the exact canonical bytes and field lists described in
[SCHEMA.md](SCHEMA.md).

## Phase 1 executable engineering decisions

These decisions are implemented in the Phase 1 modules and schemas. They do not
create the Phase 2 question/study manifests or a production model-run manifest.

### Versioned configuration and schema boundary

The repository contains six `1.0.0` configuration templates under
`configs/part1/` for study protocol, model-run execution, dataset
materialization, storage, retries, and analysis. It contains eight JSON Schema
Draft 2020-12 contracts under `schemas/part1/` for question records, question
manifests, study manifests, model-run manifests, natural terminal results,
checkpoint terminal results, audit events, and validation reports. These are
templates and validators, not concrete immutable Phase 2 revisions or
manifests. Dataset/model/tokenizer revisions remain unresolved until their
compute-node work.

All six templates are compared against a complete executable oracle that is
independent of the files being checked and uses canonical JSON-typed equality,
so boolean/integer lookalikes do not pass. Only configured smoke and production
root values are variable. Manifest compatibility separately requires the exact
structured model-independent study contract and exact requested model
repository, tokenizer repository, base seed, seed version, natural settings,
and checkpoint settings. Effective generation objects must contain every
requested key with its exact JSON-typed value; additional serializable resolved
fields from preflight are allowed.

The fixed study entropy contract explicitly represents tail reasoning entropy
as the arithmetic mean over the final
`max(1, ceil(0.10 * n_reasoning))` recognized reasoning tokens, null when
`n_reasoning = 0`. The structured oracle rejects a 20% alternative even when
study/model IDs and complete hashes are self-consistently recomputed.

### Canonical serialization and identity

`part1-canonical-json-v1` emits UTF-8 JSON inside a
`serialization_version`/`value` envelope. It recursively converts CRLF and CR
in strings and keys to LF, rejects normalization-induced duplicate keys,
recursively sorts object keys, preserves list order, uses compact `,` and `:`
separators, emits Unicode directly, rejects non-finite numbers, and emits no
trailing whitespace or newline.

Every scientific ID is a lowercase SHA-256 digest of a second domain-separated
envelope containing `identity_type`, `identity_version=part1-identity-v1`, and
the exact payload. Question content and question ID both include immutable
source-row identity and content, so equal text from distinct source rows does
not collapse. Complete manifest hashes cover the selected complete immutable
fields; study/model-run IDs use their explicit smaller identity subsets.
Natural and checkpoint IDs use their logical keys. A shared-probe ID uses the
natural logical key plus prefix hash and inducer version, never the requested
checkpoint or alias owner. Attempt IDs use logical work kind, optional
checkpoint ID, and attempt number. Attempt audit IDs use attempt ID/type/
sequence; shard audit IDs use study/model-run/shard/type/sequence. Validation
report IDs use an independent `part1-validation-report-identity-v1` target
payload. `.shard-provenance.json` binds the full operational shard identity.

All identity payloads exclude their own ID/hash, timestamps, filesystem paths,
mutable status, validation state, operational notes, and other mutable runtime
facts. Golden tests pin every public identity function and both audit scopes.

### Terminal records and lifecycle events

Storage is normalized: `natural_results.jsonl` and
`checkpoint_results.jsonl` contain terminal immutable outcomes only, while
`audit_events.jsonl` contains lifecycle evidence. Selected-token log
probabilities are not in either result schema.

The event taxonomy is exactly:

- attempt scope: `attempt_started`, `attempt_failed`,
  `attempt_interrupted`, `attempt_completed`, and
  `terminal_result_recovered`;
- shard scope: `stale_lock_recovered`, `trailing_line_recovered`, and
  `operator_unlock`.

Audit-event and validation-report date-time fields are validated as real
timezone-bearing RFC 3339 timestamps, not treated as annotation-only strings.

`attempt_started` consumes its number. Starts are sequential. `attempt_failed`
exists only when policy authorizes another retry; nonretryable current-attempt
and retryable attempt-3 exhaustion are terminalized by durable terminal result
followed by `attempt_completed`. An exhausted interruption remains
`terminalization_required` until its infrastructure-failure result is
published. An authoritative result without completion receives recovery
evidence and may receive the missing completion. Completion without result and
an orphaned start are interruptions and still consume the attempt.

`attempt_interrupted` is limited to category `interrupted_process`; a terminal
category cannot be represented as an audit-only interruption. A dry-run retry
is eligible only when the latest persisted started attempt has exactly one
coherent `attempt_failed` or `attempt_interrupted` closure whose policy is
retryable/retry. The persisted category and maximum started attempt number are
authoritative. Caller `category` and `attempts_consumed` are equality checks
only; the count must have exact integer type (booleans excluded) and lie in
`[0,3]`. Pristine, orphaned-latest, ambiguous, completed, terminal, exhausted,
locked, takeover-pending, and finalized states are ineligible.

### Crash consistency, recovery, and finalization

A successful or terminal outcome is committed in this order: append the result,
flush and fsync it, append `attempt_completed`, then flush and fsync the event.
The result is authoritative at the intervening crash boundary. Existing valid
stream bytes and logical records are never overwritten.

When an append first creates a stream, its parent directory is fsynced after the
file fsync. Validation reports and `.finalized` use the same durable-create
primitive, and each newly created shard-root/history/quarantine/journal
directory component is made durable by fsyncing its parent. This closes both
content and directory-entry crash boundaries.

Only a final physical line may be repaired. Invalid tail bytes are preserved
byte-for-byte under `quarantine/`; immutable evidence is first fsynced under
`recovery_journal/`; only then may the invalid tail be truncated. A complete
valid JSON object missing its newline is preserved and repaired by appending
only the newline under the same journal protocol. Recovery is idempotent after
every durability boundary. Malformed middle records are fatal. `.finalized`
blocks every later mutation. Validation reports are external, machine-readable,
and target-stable rather than hashes of mutable report timestamps/outcomes.

### Exclusive ownership and takeover

`.writer.lock` records a unique lock ID plus study/model-run/shard, host, PID,
SLURM job/array IDs, and acquisition time. One stable `.writer.guard` POSIX
`flock` spans ownership validation and the entire mutation, close, or takeover;
a displaced writer cannot mutate or remove its replacement's lock.

Mutating `Part1ShardStore` without a runtime lock capability fails by default.
Only synthetic tests may explicitly construct it with
`unsafe_for_tests=True`; Phase 2 generation must never use that escape hatch.

Takeover creates a durable active `.writer-lock-recovery.claim`, immutable
claim/event records under `.lock_history/`, and atomically publishes complete
control files. Pending replacement bytes are reused idempotently; conflicting
pending bytes require a newly reasoned operator override and are permanently
quarantined. Cleanup removes the active claim last, after the one required audit
event is durable.

Automatic stale recovery never uses age. Any LIVE worker or scheduler evidence
refuses. For a SLURM owner, conclusive scheduler DEAD evidence suffices even
when a remote PID cannot be checked; on the same host, PID DEAD may suffice when
scheduler state is unknown. A non-SLURM owner requires same-host PID DEAD.
Every ambiguous or failed probe refuses. Operator takeover always requires a
nonblank reason and writes `operator_unlock`.

The local two-process POSIX lock regression passed. Actual `flock`, directory
fsync, no-overwrite hard-link, and atomic-replacement behavior on the selected
Mila persistent filesystem, plus exact Mila `squeue` array-job behavior, remain
Phase 2 operational readiness checks, not completed cluster validation.

### Resume and shard hierarchy

Before work, resume validates `.shard-provenance.json`, indexes both terminal
streams and events, verifies lifecycle/parent hierarchy, reconciles orphans,
counts durable starts, and classifies each requested natural/checkpoint item as
completed, retryable, terminal, terminalization-required, or ineligible.
`WorkSpec` carries the complete model-run-manifest hash and canonical seed.

Checkpoint publication requires exactly one complete checkpoint-eligible
natural parent and exact parent record ID, seed, question/sample/subject,
study/model/question-manifest provenance, and checkpoint membership. Missing
checkpoint work never invalidates or regenerates a successful natural chain.
The same command or array can therefore be resubmitted idempotently.

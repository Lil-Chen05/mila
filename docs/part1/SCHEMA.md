# Part 1 schema and identity contract

## Status and authority

Phase 1 implements this contract in `scripts/part1_contract.py`,
`scripts/part1_store.py`, `scripts/part1_failure_policy.py`, and
`scripts/part1_runtime.py`. Six versioned templates live under
`configs/part1/`; eight JSON Schema Draft 2020-12 files live under
`schemas/part1/`. Scientific meaning remains fixed by
[DECISIONS.md](DECISIONS.md), operations by [RUNBOOK.md](RUNBOOK.md), and
evidence by [VALIDATION.md](VALIDATION.md).

Schema and template availability is not concrete manifest availability. Phase 2
still creates the tracked question and study manifests and resolves immutable
dataset/model/tokenizer revisions. The production model-run-manifest schema
exists, but no production instance exists; Phase 3 owns its post-commit
lifecycle.

Historical 20q/200q JSON is legacy and incompatible. It must not be upgraded by
silently filling missing provenance.

## Executable contract inventory

All schema instances use `schema_version = "1.0.0"`.

| Schema | Discriminator |
|---|---|
| `question_record.schema.json` | `part1_question_record` |
| `question_manifest.schema.json` | `part1_question_manifest` |
| `study_manifest.schema.json` | `part1_study_manifest` |
| `model_run_manifest.schema.json` | `part1_model_run_manifest` |
| `natural_terminal_result.schema.json` | `part1_natural_terminal_result` |
| `checkpoint_terminal_result.schema.json` | `part1_checkpoint_terminal_result` |
| `audit_event.schema.json` | `part1_audit_event` |
| `validation_report.schema.json` | `part1_validation_report` |

The six `1.0.0` templates are `study_protocol.json`,
`model_run_execution.json`, `dataset_materialization.json`, `storage.json`,
`retries.json`, and `analysis.json`. Validators reject drift from fixed science,
retry policy, production prohibition, root separation, and persistent-root
safety. Every tracked field in all six templates is compared with an
independent executable fixed-value oracle using canonical JSON-typed equality;
only the configured smoke and production storage roots are intentionally
variable. Loading a modified template cannot redefine the oracle, and JSON
type lookalikes such as integer `1` for boolean `true` are rejected.

Common rules:

- JSON/JSONL is UTF-8; non-finite numbers are forbidden and unavailable values
  are JSON `null`.
- Scientific floats persist without presentation rounding.
- A–D vectors are ordered A, B, C, D.
- Terminal logical identity and provenance are required even for terminal
  infrastructure failures.
- One shard contains one study ID, model-run ID, complete
  model-run-manifest hash, and shard ID, fixed by `.shard-provenance.json`.
- Full-vocabulary logits and selected-token log probabilities are never stored.

## Canonical serialization

`part1-canonical-json-v1` produces the UTF-8 bytes of:

```json
{"serialization_version":"part1-canonical-json-v1","value":<normalized-value>}
```

The implementation:

- recursively normalizes CRLF and bare CR to LF in all string values and object
  keys;
- rejects key collisions introduced by line-ending normalization;
- recursively sorts object keys;
- preserves list/tuple order;
- uses compact separators `,` and `:` with no insignificant whitespace;
- emits Unicode directly rather than ASCII escapes;
- permits only JSON null, booleans, integers, finite floats, strings, arrays,
  and string-keyed objects; and
- emits no trailing whitespace or newline.

This serialization is not JSONL framing. Raw JSONL records use the same
normalization/sorting/compact/direct-Unicode/finite-value rules followed by one
newline per complete record.

## Identity and hash payloads

Scientific identities use domain-separated SHA-256:

```json
{
  "identity_type": "<identifier kind>",
  "identity_version": "part1-identity-v1",
  "payload": {"<exact fields>": "..."}
}
```

The envelope is serialized with `part1-canonical-json-v1`; the lowercase
hexadecimal digest is the ID/hash. Exact field lists live as constants in
`scripts/part1_contract.py` and are pinned by golden tests.

### Question and manifest identities

- `question_content_hash` payload: source repository, immutable revision,
  source config/split, source-row identity, exact question, four ordered
  choices, gold index, and gold letter.
- `question_id` currently uses the same immutable source/content fields under a
  different identity domain. Including source-row identity prevents equal text
  from distinct selected source rows from collapsing.
- `question_manifest_hash` payload: the complete selected immutable sidecar
  fields plus every selected immutable question-record field in manifest order.
- `study_id` payload: question-manifest hash and the fixed model-independent
  scientific subset—subjects/quotas/seed, protocol, checkpoints, entropy,
  answer validity, statuses, calibration, bootstrap, primary registry,
  within-question, switching/stabilization, repetition, and analysis contract.
- `study_manifest_hash` covers the selected complete immutable study-manifest
  fields, including compatible raw schema versions.
- `model_run_id` covers exact study/question provenance plus model/tokenizer
  revision, canonical model identity, adapter/prompt/parser/inducer/tags,
  requested/effective generation settings, A–D convention/tokens, and seed
  algorithm/base seed.
- `model_run_manifest_hash` covers all selected immutable model-run fields,
  additionally including schema discriminators, environment versions, final
  production Git commit, production flag, and smoke Git provenance.

### Work, event, validation, and shard identities

- `natural_record_id`: `(study_id, model_run_id, question_id, run_id)`.
- `checkpoint_record_id`: the natural key plus `checkpoint_id`.
- `shared_probe_id`: natural key plus `prefix_hash` and `inducer_version`.
  Requested checkpoint identity and alias owner are excluded, so aliased
  requested records share one physical-probe identity.
- `attempt_id`: logical natural/checkpoint work kind, natural key, optional
  checkpoint ID, and `attempt_number`.
- Attempt-scope `audit_event_id`: attempt ID, event type, event sequence.
- Shard-scope `audit_event_id`: study ID, model-run ID, shard ID, event type,
  event sequence.
- `validation_report_id` uses independent domain
  `part1-validation-report-identity-v1` over study/model-run/complete manifest
  hash, shard ID, artifact kind, validator version, and store-contract version.
  Timestamps, results, and mutable report state do not change the target ID.
- `.shard-provenance.json` is an immutable header containing schema name/version,
  study ID, model-run ID, complete model-run-manifest hash, and shard ID.

Every payload excludes its own ID/hash, creation/event timestamps, filesystem
paths, mutable status or validation state, runtime statistics, operational
notes/messages, and any field that can change without changing scientific
identity. Result IDs exclude attempt/outcome data; retries never create a new
logical result identity.

## Seed derivation

`part1-seed-v1` hashes the canonical payload:

```json
{
  "seed_algorithm_version": "part1-seed-v1",
  "base_seed": 42,
  "canonical_model_identity": "<immutable model identity>",
  "question_id": "<stable question ID>",
  "run_id": 0
}
```

Take the first eight digest bytes as an unsigned big-endian integer and bitwise
AND with `2**63 - 1`. The result is in PyTorch's supported nonnegative signed
64-bit range `[0, 2**63 - 1]`. Retries preserve the exact value. Tests pin
golden values and separation across model, question, run, and algorithm version.
Python's built-in `hash()` is not used.

## Question, study, and model-run manifests

### Question record and sidecar

The Phase 2 tracked JSONL must contain exactly 500 records in `sample_index`
order. Each record requires:

`schema_name`, `schema_version`, `question_id`, `question_content_hash`,
`sample_index`, `subject`, `subject_selection_index`, `source_repository`,
`source_revision`, `source_config`, `source_split`, `source_row_identity`,
`question`, exactly four ordered `choices`, `gold_index`, and `gold_letter`.

The sidecar requires its complete hash, format/source/revision/config/split,
five subjects in fixed order, quota 100, count 500, seed 42, selection and
canonicalization versions, ordered-record aggregation rule, and logical
filename. Phase 2 resolves source-row identity and the immutable source
revision.

### Study manifest

The model-independent tracked study manifest requires its ID/hash,
question-manifest hash, fixed subjects/quotas/seed, protocol/checkpoint/entropy/
answer-validity/status/calibration/bootstrap contracts, exact primary registry,
within-question and switching/stabilization contracts, repetition policy,
compatible raw schema versions, and analysis-contract version.

These fields are checked against an independent structured fixed-study oracle,
not only against recomputed IDs and hashes. In particular, tail reasoning
entropy is exactly the arithmetic mean over the final recognized reasoning
tokens with window size `max(1,ceil(0.10*n_reasoning))`, and it is `null` when
`n_reasoning` is zero. A direct or fully rehashed internally consistent 20%
alternative is incompatible.

### Model-run manifest

One operational immutable manifest represents one exact model revision and
adapter. It requires its ID/hash; study/question hierarchy; immutable
model/tokenizer revisions; canonical model identity; adapter, prompt/hash,
parser, inducer/text/tokens, reasoning tags/tokens; requested/effective natural
and checkpoint settings; A–D token convention/sequences/IDs; seed algorithm and
base seed; environment versions; Git provenance; and production/smoke status.

Production requires a final 40-hex Git commit and null smoke provenance. Smoke
requires null final-production commit and non-null smoke provenance. Output
paths and mutable progress stay outside the immutable manifest. The schema is
implemented, but no production instance exists.

Compatibility also checks an independent fixed requested-model oracle:
`HuggingFaceTB/SmolLM3-3B` for model and tokenizer, base seed 42,
`part1-seed-v1`, requested natural generation values of `do_sample=true`,
temperature 0.6, top-p 0.95, top-k 50, and at most 8192 new tokens, plus the
fixed requested greedy checkpoint settings. The separate fixed
`study_protocol` configuration enforces `natural_runs_per_question=10` and run
IDs 0 through 9. Effective natural and checkpoint settings must contain every
corresponding requested key with the exact JSON-typed requested value. They may
additionally contain canonical, JSON-serializable resolved fields discovered
by Phase 2 preflight.

## Normalized terminal result schemas

Terminal result files contain no pending or running record. Lifecycle is only
in `audit_events.jsonl`.

### Natural terminal result

Required Phase-2-ready fields are:

- identity/provenance: schema name/version, raw record ID, study/model-run/
  question IDs, complete model-run-manifest and question-manifest hashes,
  sample index, subject, run ID, generation seed/version, terminal attempt
  number/ID, and infrastructure-failure reference;
- prompt/output: prompt hash, rendered prompt, prompt token IDs, generated token
  IDs, full decoded output, reasoning text/boundaries, close-tag information,
  stop reason, generated/reasoning token counts;
- measurements/parsing: full-precision per-token entropy trace, mean/tail
  reasoning entropy, terminal answer block text/span, natural answer, raw
  confidence text/integer, normalized confidence, `natural_correct`, and
  diagnostic answer-like text;
- checkpoint linkage: eligibility and exactly eleven unique checkpoint IDs for
  a complete execution; and
- orthogonal outcome/statuses, component versions, and terminal error details.

Enums are exact:

- `natural_execution_outcome`: `complete` |
  `terminal_infrastructure_failure`;
- `stop_reason`: `eos` | `max_new_tokens` | `stopping_criterion` | `error` |
  `other`;
- `reasoning_status`: `closed` | `missing_close` | `no_reasoning` |
  `malformed`;
- `answer_parse_status`: `parsed` | `missing` | `malformed` |
  `out_of_domain`; and
- `confidence_parse_status`: `parsed` | `missing` | `malformed` |
  `out_of_range`.

| Natural condition | Required non-null/output facts | Required null facts |
|---|---|---|
| `complete` | prompt/tokens/text, boundaries/statuses, aligned entropy array, counts, `checkpoint_eligible=true`, eleven checkpoint IDs | infrastructure reference and terminal error |
| terminal infrastructure failure | logical identity, seed, attempt, `stop_reason=error`, failure reference/details, malformed/missing statuses, `checkpoint_eligible=false` | prompt/output/parsing/correctness/entropy/checkpoint IDs |
| `missing_close` | executed diagnostic boundaries/reasoning measures | terminal block/span, natural answer/correctness, normalized confidence; parsed answer/confidence forbidden |
| `no_reasoning` | complete output and `reasoning_token_count=0` | mean and tail reasoning entropy |
| complete with recognized reasoning | `reasoning_token_count >= 1`, mean reasoning entropy, and exact 10% tail reasoning entropy | none of those summaries |
| parsed answer | A–D natural answer, correctness, terminal block/span | none of these |
| nonparsed answer | diagnostic fields/status | natural answer and correctness |
| parsed confidence | raw text, integer 0–100, exact integer/100 normalized value | none of these |
| missing confidence | explicit missing status | raw text, parsed integer, normalized confidence |
| malformed confidence | raw diagnostic text may remain | parsed integer and normalized confidence |
| out-of-range confidence | raw text and a true integer `<= -1` or `>= 101` | normalized confidence |

Generated token IDs and per-token entropy must have equal lengths, and
`generated_token_count` must match them. Scientific floats remain unrounded.
Terminal infrastructure failure additionally nulls all executed-output
diagnostics, not just the parsed scientific products.

### Checkpoint terminal result

Required Phase-2-ready fields are:

- identity/provenance/parent: schema name/version, checkpoint and parent natural
  IDs, study/model-run/question IDs, complete manifest hashes, sample/subject,
  run/checkpoint ID, natural seed, terminal attempt number/ID, and failure
  reference;
- placement/alias: requested checkpoint index 0–10, exact index/10 fraction,
  `k_keep`, actual fraction, shared-probe ID, alias flag/metadata, prefix hash,
  and inducer version/text;
- output/parse: forced token IDs, full decoded forced output, terminal answer
  block, forced answer, raw/parsed/normalized confidence, checkpoint-local
  correctness;
- answer-step measurement: answer token index/ID, token convention, four A–D
  token IDs, raw float32 A–D logits, float32 A–D probabilities, four-choice
  entropy nats, full-vocabulary answer-step entropy nats, maximum A–D
  probability, and natural-answer agreement; and
- orthogonal outcome/statuses, component versions, and terminal error details.

Enums are exact:

- `checkpoint_execution_outcome`: `complete` |
  `terminal_infrastructure_failure`;
- `checkpoint_model_output_status`: `valid` | `invalid`;
- answer/confidence parse enums as above;
- `answer_token_status`: `located` | `missing` | `ambiguous` |
  `unsupported`; and
- `entropy_status`: `computed` | `unavailable` | `invalid`.

| Checkpoint condition | Required non-null/output facts | Required null facts |
|---|---|---|
| complete + valid | output, parsed A–D answer/correctness, terminal block, located answer token and convention, computed four-value measurements | failure reference/details; only confidence fields may be null when its separate status permits |
| complete + invalid | raw output and every orthogonal status; at least one of answer parsed/token located/entropy computed is not valid | every unavailable parsed/measurement value according to its status |
| terminal infrastructure failure | identity/placement/alias, seed/attempt, failure reference/details; invalid/missing/unsupported/unavailable statuses | output, parse products, correctness, answer-step fields, logits/probabilities/entropies/agreement |
| answer token not located | diagnostic output/status | token index/ID/convention/A–D IDs and all computed measurements |
| entropy not computed | explicit entropy status | logits, probabilities, both entropies, maximum probability |
| confidence missing | explicit missing status | raw text, parsed integer, normalized confidence |
| confidence malformed | raw diagnostic text may remain | parsed integer and normalized confidence |
| confidence out of range | raw text and a true integer `<= -1` or `>= 101` | normalized confidence |

When computed, A–D token/logit/probability arrays contain exactly four aligned
values; probabilities sum to one; maximum and entropy recompute; the answer
index/ID aligns with forced token IDs. Agreement can be null and disagreement
is valid data.

Checkpoint publication additionally requires exactly one complete eligible
natural parent and matching parent record ID, seed, manifest provenance,
question/sample/subject/run fields, and checkpoint membership. The store
recomputes the checkpoint record ID, ties-to-even clamped `k_keep`, actual
fraction, prefix hash, shared-probe identity, alias owner/member set, and
`is_alias` from that parent. Across records sharing a natural parent and
`k_keep`, `prefix_hash`, inducer version/text, and `shared_probe_id` must be
identical.

## Audit event schema and lifecycle

The event taxonomy is exactly eight types:

- attempt scope: `attempt_started`, `attempt_failed`,
  `attempt_interrupted`, `attempt_completed`,
  `terminal_result_recovered`;
- shard scope: `stale_lock_recovered`, `trailing_line_recovered`,
  `operator_unlock`.

Every event includes schema/ID/scope, study/model-run/shard context, logical
work and attempt fields where applicable, monotonic sequence, timestamp,
execution context, failure/retry/backoff metadata where applicable, related
lock owner, terminal record reference, and operator reason. Shard-scope events
must null all work/attempt fields. Only `operator_unlock` has a nonblank operator
reason. `attempt_completed` and `terminal_result_recovered` require a terminal
record reference; starts/failures/interruptions forbid one.

`attempt_started` at sequence zero consumes its attempt number. Starts are
sequential. `attempt_failed` is valid only for a retry-authorizing failure.
`attempt_interrupted` is valid only for the `interrupted_process` category.
Final/nonretryable work is represented by result then `attempt_completed`.
`terminal_result_recovered` records an authoritative result whose completion
was missing; an optional matching completion can follow. Completion without
result and an orphaned start are classified as interruptions and count toward
the limit.

Audit-event and validation-report date-time fields are procedurally checked as
RFC 3339 in addition to their JSON Schema `date-time` annotations.

## Validation reports, append-only layout, and evolution

Validation reports contain target identity, timestamps, validator version,
pass/fail/warning checks, counts, and summary. Reports are written at an
explicit external path and never modify raw streams. The current shard checks
cover JSON syntax, schema validity, duplicate/conflict detection, malformed
middle, trailing/pending recovery, array alignment, terminal/event consistency,
hierarchy/terminalization, and outcome nullability.

Active raw streams are append-only and per-record fsynced. First file creation
also fsyncs its containing directory, and creation of each missing directory
component fsyncs the parent directory entry. Validation reports, recovery
evidence, and `.finalized` use the same durable-create discipline. Recovery may
modify only an incomplete final physical line after exact-byte quarantine and
durable journal evidence. Mutating `Part1ShardStore` requires a runtime lock
capability by default; only synthetic tests may opt into
`unsafe_for_tests=True`. `.finalized` blocks further raw-shard mutation.
Renaming fields, changing status meaning/nullability, identity payloads, metric
formulas, parser validity, or checkpoint semantics requires a new incompatible
version; prior manifests/results are preserved.

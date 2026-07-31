# Part 1 schema and identity contract

## Status and authority

This is the Prompt 1 logical schema contract. It defines required information,
status semantics, nullability, compatibility, and non-self-referential identity
rules. It is not a claim that Phase 1 validators, canonical serializers, hashes,
or immutable manifests already exist. They do not.

Phase 1 must convert this contract into executable schemas and golden tests and
must lock the exact payload choices listed under
[Phase 1 decisions to lock](#phase-1-decisions-to-lock). Phase 2 then creates
the question and study manifests. Phase 3 creates the production model-run
manifest. Scientific meanings come from [DECISIONS.md](DECISIONS.md); lifecycle
ordering comes from [RUNBOOK.md](RUNBOOK.md).

Historical `results/20q` and `results/200q` JSON files use legacy, unversioned
shapes. They are not compatible Part 1 raw records and must never be migrated by
silently filling missing provenance.

## Common representation rules

- JSON/JSONL is the planned interchange format. All records are UTF-8.
- Every object has an explicit schema-name field and `schema_version`.
- Stable SHA-256 values are represented as lowercase hexadecimal after Phase 1
  locks and tests that representation.
- Scientific floating-point values are written without presentation rounding.
  Non-finite JSON numbers are forbidden; unavailable values are JSON `null`.
- Choice maps and vectors use the semantic order A, B, C, D.
- Logical identity fields are required even when execution terminates in
  infrastructure failure.
- Timestamps and operational paths may be recorded for audit/operations but
  never enter stable scientific identity payloads.
- A shard contains records from exactly one `model_run_id` and one complete
  model-run-manifest hash.

## Question records and question-manifest sidecar

The planned tracked JSONL contains exactly 500 immutable question records in
`sample_index` order.

### Question record

| Field | Type | Required | Contract |
|---|---|---:|---|
| `schema_name` | string | yes | Question-record schema discriminator. |
| `schema_version` | string | yes | Version understood by the manifest sidecar. |
| `question_id` | string | yes | Stable ID derived from the locked question identity payload. |
| `question_content_hash` | string | yes | Hash of the complete canonical question content, excluding this hash. |
| `sample_index` | integer | yes | Unique integer 0–499; defines global order. |
| `subject` | string | yes | One of the five fixed subjects. |
| `subject_selection_index` | integer | yes | Unique integer 0–99 in seeded selection order within the subject. |
| `source_repository` | string | yes | `cais/mmlu`. |
| `source_revision` | string | yes | Immutable dataset revision resolved in Phase 2. |
| `source_config` | string | yes | Resolved MMLU configuration used for materialization. |
| `source_split` | string | yes | `test`. |
| `source_row_identity` | object | yes | Stable source-row locator sufficient to audit selection without using mutable row position alone. |
| `question` | string | yes | Exact question text. |
| `choices` | array[string] | yes | Exactly four choices in A–D order. |
| `gold_index` | integer | yes | Integer 0–3. |
| `gold_letter` | string | yes | A–D and consistent with `gold_index`. |

`source_row_identity`'s exact members depend on the pinned MMLU artifact and are
locked during Phase 2. Acceptance requires duplicate-content handling to be
explicit: content equality must not accidentally merge distinct selected source
rows.

### Complete question-manifest sidecar

Required fields are `schema_name`, `schema_version`, `question_manifest_hash`,
manifest format/version, source repository/revision/config/split, ordered
subjects, quota per subject, total count, question-sampling seed, selection
algorithm version, canonicalization version, ordered-record aggregation rule,
and the JSONL path or immutable logical filename.

Filesystem paths and creation timestamps are descriptive/operational and do not
enter scientific IDs. The complete manifest hash covers all immutable sidecar
fields selected in Phase 1 plus the ordered question-record representation; it
excludes `question_manifest_hash` itself.

Validation requires exact subjects/order, five quotas of 100, 500 records,
unique `sample_index`, unique logical question IDs, 0–3 gold values, four
choices, and recomputed record/manifest hashes.

## Study manifest

The tracked study manifest is model-independent and requires:

| Field | Required content |
|---|---|
| `schema_name`, `schema_version` | Study-manifest schema discriminator/version. |
| `study_id` | SHA-256 identity from the locked study identity payload. |
| `study_manifest_hash` | Hash of the complete immutable manifest excluding this field. |
| `question_manifest_hash` | Exact hash of the tracked 500-question manifest. |
| `subjects`, `subject_quotas`, `question_sampling_seed` | Fixed five-subject design, 100 each, seed 42. |
| `scientific_protocol_version` | Version covering the immutable experiment design. |
| `checkpoint_fractions` | All eleven requested fractions in order. |
| `checkpoint_placement_contract` | Ties-to-even rounding, clamp, actual fraction, and alias rules. |
| `entropy_contract` | Float32, natural-log nats, raw pre-warper natural entropy, mean/tail, A–D and full-vocabulary checkpoint metrics. |
| `natural_answer_validity_rule` | One terminal post-close block; answer/confidence paired from it. |
| `status_contract_version` | Orthogonal execution/output status meanings. |
| `calibration_contract` | Targets, bins, per-fraction rule, pooled/subject/macro reporting. |
| `bootstrap_contract` | Seed 42, 1,000/5,000 replicates, stratification, multiplicity, validity threshold. |
| `primary_auroc_feature_registry` | Exact eleven approved features, orientations, and `natural_correct` target. |
| `within_question_analysis` | Eligibility and paired-difference definition. |
| `switching_stabilization_contract` | Alias-, adjacency-, missingness-, appearance-, recovery-, and stabilization rules. |
| `repetition_policy` | Preserve successful output; no automatic detector/exclusion/retry. |
| `compatible_raw_record_schema_versions` | Explicit allow-list/range determined by executable compatibility tests. |
| `analysis_contract_version` | Version required for combining and analyzing records. |

The study identity payload is a deliberately smaller scientific subset of the
complete manifest. Exact members and ordering are a Phase 1 decision; neither
payload includes its own ID/hash or mutable operational fields.

## Model-run manifest

One immutable model-run manifest describes one exact model revision and adapter.
It is operational, generated under ignored persistent results only after the
final production commit in Phase 3.

Required fields are:

- `schema_name`, `schema_version`;
- `model_run_id` and `model_run_manifest_hash`;
- `study_id`, `study_manifest_hash`, and `question_manifest_hash`;
- model repository and immutable model revision;
- tokenizer repository, when distinct, and immutable tokenizer revision;
- canonical model ID used by seed derivation;
- adapter version;
- semantic prompt version and immutable prompt representation/hash;
- parser version;
- inducer version, text, and token-ID sequence;
- reasoning opening/closing tag text and token-ID sequences;
- requested natural and checkpoint generation settings;
- effective natural and checkpoint generation settings after
  `GenerationConfig` resolution;
- A–D token convention, raw token sequences, and selected single-token IDs or
  the validated model-specific representation;
- seed-algorithm version and base generation seed;
- software/environment versions needed for reproduction, including Python,
  PyTorch, Transformers, CUDA/driver/runtime where applicable, and model dtype;
- final production Git commit;
- production flag;
- for non-production smoke only, base Git commit and diff hash when the code is
  dirty;
- persistent output locations and shard layout as operational fields.

The exact immutable model/tokenizer revisions, compute-node tag and A–D token
preflight result, and requested-to-effective setting resolution are Phase 2
decisions. A model-run manifest may not be constructed from mutable branch names
or an unresolved model default.

`model_run_id` covers the locked scientific/execution identity subset. The
complete manifest hash covers all locked immutable manifest fields but excludes
itself. Output paths, creation timestamp, validation/completion state, runtime
statistics, and mutable notes are excluded from both stable identity payloads;
they may be stored as operational metadata only if mutation does not rewrite an
otherwise immutable manifest. Phase 1 must choose whether such mutable metadata
lives in a separate state file; the acceptance criterion is that the immutable
manifest never changes as work progresses.

## Logical raw natural-run record

Raw-record physical granularity is intentionally deferred to Phase 1. Whether
one JSONL terminal record nests the natural run and all checkpoints, or whether
natural and checkpoint terminal records are normalized, must not change the
following logical content and cardinality.

Every logical natural run requires:

### Identity and provenance

- `schema_name`, `schema_version`, `raw_record_id`;
- `study_id`, `model_run_id`, `model_run_manifest_hash`;
- `question_manifest_hash`, `question_id`, `sample_index`, `subject`;
- `run_id` in 0–9;
- `generation_seed` and `seed_algorithm_version`;
- attempt/event references sufficient to audit retries without changing the
  logical raw-record ID.

The logical natural-run identity is the tuple of the exact study/model run,
question, and `run_id`. A retry is another attempt at the same logical identity,
not another natural run.

### Natural execution and raw output

- `natural_execution_outcome`;
- `stop_reason`;
- infrastructure failure reference/category when execution fails;
- exact generated token IDs and exact decoded full generated text when
  execution completes;
- generated-token count;
- recognized opening/closing tag spans and reasoning slice boundaries or an
  equivalent lossless index representation;
- `n_reasoning` and `reasoning_status`;
- terminal answer-block span and diagnostic match data when present;
- `answer_parse_status`, `natural_answer`, and `natural_correct`;
- `confidence_parse_status`, raw confidence text, raw parsed integer, and
  normalized natural confidence;
- per-generated-token raw full-vocabulary entropy in nats, aligned one-to-one
  with generated token IDs;
- mean reasoning entropy and tail reasoning entropy;
- checkpoint eligibility and, for a complete natural execution, all eleven
  requested checkpoint identities.

No full-vocabulary logits or selected-token log probabilities are stored.

### Natural nullability

| Condition | Required | Must be null/absent as specified |
|---|---|---|
| `natural_execution_outcome = terminal_infrastructure_failure` | logical identity, seed, `stop_reason=error`, failure reference, durable event | generated output, parsing results, correctness, entropies, and scientific checkpoint results; checkpoint eligibility is infrastructure-ineligible |
| `natural_execution_outcome = complete` | stop reason, tokens/text, boundary and parse statuses, aligned entropy array, checkpoint eligibility | fields unavailable under their explicit status are null |
| `reasoning_status = missing_close` | diagnostic boundary information and all recognized reasoning measurements | `natural_answer`, `natural_correct`, normalized natural confidence |
| `reasoning_status = no_reasoning` | `n_reasoning=0`, executed output, all eleven checkpoint identities | mean and tail reasoning entropy |
| `answer_parse_status = parsed` after valid closure | A–D `natural_answer`, boolean `natural_correct` | neither may be null |
| answer missing/malformed/out of domain or invalid closure | diagnostic raw text/status | `natural_answer` and `natural_correct` |
| `confidence_parse_status = parsed` | raw text, integer 0–100, normalized value `integer/100` | none of these parsed values may be silently changed |
| confidence missing/malformed/out of range | available raw diagnostic values and status | normalized confidence; out-of-range raw integer remains preserved |

## Logical checkpoint record

Every complete natural execution produces exactly eleven logical requested
checkpoint records, even when several share one physical probe.

### Identity and placement

- checkpoint schema name/version and `checkpoint_record_id` if checkpoint
  records are physically normalized;
- parent `raw_record_id` or full parent logical identity;
- `requested_checkpoint_index` 0–10;
- exact requested fraction representation;
- `k_keep`;
- `actual_fraction`, null only when `n_reasoning = 0`;
- `shared_probe_id`;
- `is_alias`, alias-group membership, and canonical probe-owner reference;
- intervention/inducer version inherited from or checked against the model-run
  manifest.

`requested_checkpoint_index`, not `k_keep`, preserves the eleven identities.
All aliases in a group must have identical physical probe results and must not
be double-counted as answer transitions.

### Execution, output, and measurements

- `checkpoint_execution_outcome`;
- `checkpoint_model_output_status`;
- separate `answer_parse_status`, `confidence_parse_status`,
  `answer_token_location_status`, and `entropy_computation_status`;
- infrastructure failure reference when applicable;
- exact checkpoint generated token IDs and decoded text when execution
  completes;
- forced answer and checkpoint-local correctness when validly parsed;
- raw confidence text, raw parsed integer, and normalized checkpoint confidence
  only when valid;
- answer-token generation step/index and the preflighted A–D token convention;
- raw float32-derived A–D logits in A–D order;
- normalized A–D probabilities in A–D order;
- four-choice entropy in nats;
- maximum normalized A–D probability;
- full-vocabulary entropy in nats at the identified answer-token step;
- natural-answer agreement at checkpoint 1.0 when both answers are valid, with
  null otherwise.

### Checkpoint nullability

| Condition | Required | Null fields |
|---|---|---|
| `checkpoint_execution_outcome = terminal_infrastructure_failure` | identity/placement, failure reference, durable event | generated output, parse products, correctness, answer-step fields, logits/probabilities, entropies |
| execution complete, output valid | raw output, all status fields, parsed answer and checkpoint correctness, valid answer-token location, A–D/full-vocabulary measurements | only confidence values may be null when confidence status permits |
| execution complete, output invalid | raw output and all separate statuses | each unavailable parsed/measurement field according to its status; never relabel as infrastructure failure |
| answer-token location not valid | diagnostic output/location status | answer-step index, A–D logits/probabilities/entropy/max and full-vocabulary answer-step entropy |
| confidence out of range | raw text and parsed integer, `out_of_range` | normalized confidence |

Phase 1 must define the exact allowed values for answer-token-location and
entropy-computation statuses and validate every status/value combination.

## Derived analysis records

Derived tables must retain `study_id`, model-run identity/hash, raw schema
version, analysis-contract version, feature registry version, and cohort/filter
counts. They must distinguish pooled, subject, and arithmetic macro results and
label primary versus secondary targets.

Bootstrap result records require metric/feature/target, scope, requested/valid/
invalid replicate counts, seed, percentile level, point estimate, lower/upper
bounds, and interval-valid flag plus warning when validity is below 95%.
Reliability records require checkpoint identity, bin index and exact edges,
count, mean confidence, empirical accuracy, empty-bin representation, and the
count-weighted ECE aggregate.

Within-question records require question ID, run counts by natural correctness,
correct-run mean, incorrect-run mean, and paired difference. Switching records
require switch count, first natural-answer appearance, stabilization fraction,
switched-away-from-correct and recovered flags, and checkpoint-1.0 agreement.

## Audit event records

Audit events are append-only operational evidence. Each event requires:

- `schema_name`, `schema_version`, and `event_id`;
- logical raw identity (`study_id`, `model_run_id`, question ID, run ID) and,
  when applicable, requested checkpoint identity;
- attempt ID and monotonically meaningful event sequence within that attempt;
- event type;
- event timestamp as audit metadata;
- SLURM job/array/task and host/process context when available;
- outcome/category and structured error information when applicable;
- retryability classification, retry decision, and backoff metadata when
  applicable;
- related lock owner and artifact/terminal-record reference when applicable.

At minimum, the Phase 1 event taxonomy must represent attempt start, natural
terminal success/failure, checkpoint terminal success/failure, retry scheduled
or exhausted, terminal record committed, lock acquired/released/stale-recovered,
and validation failure. Raw traceback/message text is diagnostic and excluded
from `event_id`.

Event IDs must remain unique for distinct attempts/events without using a
timestamp as scientific identity. Phase 1 must lock whether a deterministic
attempt ordinal and event sequence, or another explicit non-self-referential
payload, supplies uniqueness.

## Canonical hashes and identifiers

The fixed algorithm is SHA-256 over versioned canonical UTF-8 bytes. Each
identifier has a separately defined payload:

| Identifier | Required identity content | Always excluded |
|---|---|---|
| `question_content_hash` | complete immutable source/content representation sufficient to detect any scientific question change | itself, paths, timestamps |
| `question_id` | stable source identity plus content identity, with an explicit duplicate-content rule | itself, selection position if position is not scientific identity |
| `question_manifest_hash` | complete immutable sidecar fields plus all ordered question records | itself, path, creation/validation state |
| `study_id` | model-independent scientific identity subset | itself, manifest hash, paths, timestamps, status |
| `study_manifest_hash` | complete immutable study manifest | itself, mutable operational metadata |
| `model_run_id` | exact study/model/tokenizer/adapter/prompt/parser/inducer/settings identity subset | itself, complete hash, output path, timestamps, runtime/completion state |
| `model_run_manifest_hash` | complete immutable model-run manifest | itself and mutable operational fields |
| `raw_record_id` | exact model-run, question, and run logical identity, plus record-kind discriminator if physical records are normalized | itself, attempt, time, status, runtime |
| `checkpoint_record_id` | parent logical identity plus requested checkpoint identity, if separately stored | itself, shared physical-probe result/status |
| `shared_probe_id` | parent logical identity plus exact unique prefix/probe identity | itself, alias owner selection, runtime |
| `event_id` | logical work identity, deterministic attempt identity, event type, and sequence | itself, timestamp, message, runtime |

No ID or hash payload may contain the value being calculated. Creation time,
filesystem location, validation/completion status, runtime statistics, mutable
notes, and any field that can change without changing scientific identity are
excluded. A complete-manifest hash may cover immutable descriptive fields not
present in the shorter ID payload, but never its own hash.

## Compatibility and evolution

- Schema versions are explicit. Phase 1 must select the version syntax and
  implement compatibility as an allow-list or tested rule, not string guessing.
- Adding an optional field is compatible only when its default/null meaning is
  unambiguous and does not change existing scientific interpretation.
- Renaming fields, changing status meanings, nullability, identity payloads,
  metric formulas, parser validity, or checkpoint semantics requires a new
  incompatible schema or contract version.
- Immutable manifests and terminal raw records are never edited in place.
  Corrections create a new identity/version and preserve prior artifacts.
- Cross-model analysis requires identical `study_id`, question-manifest hash,
  scientific protocol version, compatible raw schema versions, and compatible
  analysis contracts.
- Validators reject records with unknown incompatible versions, mixed
  model-run manifests in one shard, or hashes that do not recompute.

## Phase 1 decisions to lock

The following are deliberately not asserted as implemented or immutable in
Prompt 1. Phase 1 must resolve each with executable acceptance tests:

1. Exact schema file/mechanism, version strings, required-field spelling, and
   the complete status/nullability truth tables.
2. Exact canonical JSON or other byte serialization: ordered keys, separators,
   number representation, Unicode normalization/escaping, newline policy,
   array ordering, null handling, and UTF-8 encoding.
3. Exact ordered field list and version discriminator for every hash/ID in the
   table above, including duplicate question-content handling.
4. Exact seed payload, digest-to-integer conversion, nonnegative PyTorch range,
   and golden seed vectors.
5. Raw storage granularity: one nested terminal natural-run record versus
   normalized natural/checkpoint records, while preserving one logical run and
   eleven requested checkpoint identities.
6. Atomic file/journal format, fsync/rename boundary, terminal-record
   publication rule, index/checkpoint format, and crash recovery behavior.
7. Mila-compatible lock mechanism, persistent output root, owner metadata,
   timeout, stale-lock determination, and safe stale-lock recovery.
8. Infrastructure error taxonomy, retryable categories, finite retry count,
   backoff, attempt numbering, and terminal exhaustion representation.
9. Exact answer-token-location and entropy-computation status enums and their
   measurement nullability.
10. Whether mutable operational state is stored outside immutable manifests;
    acceptance requires that updates never change a manifest hash.

Phase 2, not Phase 1, resolves the immutable MMLU, model, and tokenizer
revisions and the compute-node tag/A–D token preflight. Phase 3 resolves the
final persistent production output locations after the production commit.

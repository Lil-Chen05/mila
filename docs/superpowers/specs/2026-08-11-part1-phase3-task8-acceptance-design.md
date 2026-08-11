# Part 1 Phase 3 Task 8 Acceptance Design

## Goal

Close two verified Task 8 evidence gaps without changing model generation,
checkpoint generation, the scientific protocol, or production coverage
semantics:

1. validate the existing Smoke A, Smoke B, and Phase 3 smoke outputs with one
   dedicated read-only bounded-workload validator; and
2. add one connected synthetic acceptance that runs the production public flow
   from raw shards through coverage, merge, and analysis publication.

## Scope boundaries

- No model, tokenizer, dataset, CUDA, SSH, or SLURM action belongs in this
  implementation slice.
- No production generation or checkpoint code changes.
- No relaxation of the production validator's strict 500-question,
  5,000-natural, 55,000-checkpoint contract.
- No inference about real SmolLM3 behavior from synthetic data.
- Existing raw smoke artifacts remain immutable; smoke validation is strictly
  read-only and publishes no report into their directories.

## Read-only smoke coverage

Add a focused smoke-coverage module and CLI rather than generalizing the
production validator. The CLI accepts the tracked manifest bundle, one existing
smoke model-run manifest, and one shard root. It supports only `smoke_a`,
`smoke_b`, and `phase3_smoke`.

Expected natural work is derived from repository-owned selection functions:

- `smoke_a`: the first fixed question with run IDs `0` through `9`;
- `smoke_b`: the first fixed question in each of the five subject blocks with
  run ID `0`; and
- `phase3_smoke`: production shard index `0` of `500`, which is the first fixed
  question with run IDs `0` through `9`.

The validator must:

- validate tracked question/study and smoke model-run identities and hashes;
- require the canonical ignored smoke manifest and shard paths for the selected
  execution scope;
- read the shard through `Part1ShardStore` without taking a writer lock;
- require a finalized shard with no active lock, pending takeover, invalid
  tail, recovery gap, schema error, hierarchy error, lifecycle error,
  duplicate, or terminalization requirement;
- require exactly one terminal natural result for every expected natural key
  and no unexpected natural key;
- derive the eleven expected checkpoint identities from each execution-complete
  natural result using the canonical checkpoint planner;
- treat checkpoints following a terminal natural infrastructure failure as
  explicitly ineligible;
- require exactly one terminal checkpoint result for every eligible expected
  key and no unexpected checkpoint key;
- report natural/checkpoint partitions, exact record counts, run IDs,
  checkpoint indices, audit-event count, stable source hashes, and whether the
  bounded workload is complete; and
- exit nonzero on any incompatibility or incomplete condition while leaving all
  inputs byte-identical.

The implementation returns a deterministic JSON-safe report but does not reuse
the production `part1_validation_report` schema, whose fixed totals and
artifact kind intentionally describe production coverage only.

## Connected synthetic production acceptance

Add one CPU-only integration test that creates a temporary, fully canonical
production repository fixture. It uses all 500 fixed question identities, ten
natural runs per question, all eleven checkpoint identities per successful
natural run, both `natural_correct` classes in every subject, and finalized
schema-valid raw shards.

The fixture writer may bypass durability syscalls only inside the temporary
test fixture, but it must write the same JSONL records, provenance headers, and
finalization markers consumed by production readers. The test must not
monkeypatch coverage validation, merge validation, analysis input validation,
or publication.

The connected flow is:

1. build and publish the production coverage report from raw shards;
2. merge exactly the source files named by that report and atomically publish
   Parquet outputs plus the merge manifest;
3. load those published artifacts through the production analysis loader;
4. atomically publish analysis outputs with a small deterministic positive
   bootstrap replicate count suitable for an acceptance test; and
5. reload final outputs and assert the Task 8 statistical contracts.

The connected flow asserts all five subjects and both target classes, primary
AUROC target `natural_correct`, natural/checkpoint ECE target pairing, sidecar
provenance, and absence of automatic repetition exclusion. The same acceptance
file also runs small hand-computable oracle subcases for bootstrap draw
multiplicity, macro invalidity, within-question paired differences, switch
adjacency, and stabilization/null behavior. Those oracle subcases exercise the
same public statistical and trajectory functions used by the connected
analysis flow without pretending that one class-balanced production-shaped
fixture can simultaneously represent a deliberately invalid macro replicate.

## Failure handling and performance

The smoke validator fails closed and never repairs, deletes, or publishes into
the smoke root. The synthetic acceptance uses a temporary directory and may
use compact fixture-construction helpers, but it retains the nominal production
logical shape so the strict production validator is not weakened for tests.
The acceptance command is kept separate from fast focused unit suites and its
runtime and temporary storage are reported explicitly.

## Test-driven implementation

Implementation follows red-green-refactor:

1. add failing smoke-coverage tests for each scope, path/provenance drift,
   missing/duplicate/unexpected keys, lifecycle failure, and zero mutation;
2. implement the smallest dedicated read-only validator and CLI;
3. add the connected acceptance test and confirm it fails at the first missing
   public-flow handoff;
4. add only test-fixture support needed to drive existing production APIs;
5. rerun focused coverage/merge/analysis regressions and `git diff --check`;
6. obtain an independent spec-and-quality review before merging the slice.

## Operational outcome

After this slice passes, run the new smoke validator on Mila against Smoke A,
Smoke B, and the Phase 3 smoke. These checks complete Task 8 evidence only;
they do not create a production model-run manifest, submit the full array, or
establish final-paper readiness.

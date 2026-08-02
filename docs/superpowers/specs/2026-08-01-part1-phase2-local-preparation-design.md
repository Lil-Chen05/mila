# Part 1 Phase 2 Local Preparation Design — 2026-08-01

## Scope and stopping point

Prepare every Phase 2 component that can be implemented and tested without
Mila, a dataset load, a model load, or a GPU. The repository will contain a
CPU-only dataset materialization job, locally testable manifest/control-flow
logic, a SmolLM3-specific adapter boundary, natural/checkpoint helpers,
synthetic fixtures, storage estimation, validation commands, unsubmitted SLURM
scripts, and a paused-state handoff.

The preparation commit will not contain Mila-generated datasets or finalized
question/study manifests. The next required action after this commit is:

```bash
sbatch jobs/materialize_part1_mmlu.sh
```

No job is submitted as part of this work. Real SmolLM3 tokenization, logits,
generation, environment capture, and checkpoint behavior remain explicitly
unverified until GPU preflight and bounded smoke execution.

## Atomic CPU bootstrap

`scripts/materialize_part1_mmlu.py` is a thin cluster-only entry point. Dataset
and Hub libraries are imported only inside its execution path. The script:

1. resolves the requested `cais/mmlu` revision to a 40-character immutable
   commit SHA and fails if the resolution cannot be verified;
2. loads each of the five subject configurations independently in the fixed
   mathematics, physics, chemistry, biology, psychology order;
3. obtains each full test-split count, adds the original row index before
   shuffling, shuffles with seed 42 and a buffer equal to the full split, and
   takes exactly 100 rows without replacement;
4. builds 500 normalized question records with ordered `sample_index` values
   0–499 and per-subject selection indices 0–99;
5. saves only those 500 bounded rows as a reproducible ignored cache under
   `data/part1/`;
6. finalizes and validates the question record bundle and question sidecar,
   including all counts, ordering, IDs, hashes, schemas, and the resolved SHA;
7. derives the study manifest from that finalized validated question bundle;
8. writes the dataset and the complete three-file manifest directory under
   temporary paths, reloads and validates both staged representations, and
   preflights the complete final manifest directory before publishing anything;
   and
9. publishes the cache, then publishes the manifest bundle with one
   same-filesystem atomic directory rename. An existing identical complete
   directory is retained. Existing partial, extra-file, non-directory, or
   divergent manifest destinations fail closed before publication.

Validation or compatibility failure leaves all existing final files unchanged
and creates no new final manifest files. The manifest bundle is never published
as a per-file sequence: the validated staged directory either becomes the final
directory in one rename or no final bundle appears. Reruns accept only an
identical complete directory; they never fill in a partial bundle.

The SLURM job never invokes Git. The ignored dataset cache is reproducible but
non-authoritative; the tracked JSONL and JSON manifests are the authoritative
selection and will be reviewed and committed only after the operator returns
the Mila outputs.

## Contract migration

The Phase 1 dataset template's aggregate `source_config: "all"` is replaced by
an explicit per-subject strategy:

- `source_config_strategy: "per_subject"`;
- `source_configs`: the exact ordered five-subject list; and
- `source_revision: "main"` as the requested ref, with an immutable SHA
  required and recorded in the resulting manifests.

Each question record retains its actual subject configuration in
`source_config`. The question-manifest sidecar records both the strategy and
ordered configurations. The study manifest records the question source
repository and resolved immutable dataset revision. These dataset fields are
scientific identity, not model environment, so they are included in the study
identity/hash while model/tokenizer/environment fields remain excluded.

## Local architecture

### Manifest and validation layer

A login-safe module owns row normalization, identities, schema checks, exact
ordering/count invariants, deterministic manifest bytes, study construction,
staging validation, existing-target compatibility, and atomic publication.
It imports no dataset, model, tokenizer, or torch library.

A separate validation CLI reopens the returned artifacts and recomputes every
record ID, content hash, question-manifest hash, study ID, and study hash. It
also checks subject blocks, seeded-selection metadata, revision agreement, and
optional saved-cache equality.

### SmolLM3 adapter boundary

The adapter contains SmolLM3-specific prompt rendering, thinking-tag discovery,
reasoning-boundary recognition, terminal answer/confidence pairing, forced
close inducer construction, choice-token preflight, context-budget checks, and
effective generation metadata. Loading functions lazily import torch and
transformers and require a GPU execution context. They request bfloat16,
`model.eval()`, one device, and batch size one, and never force PAD equal to
EOS.

Pure adapter behavior is tested with small fake tokenizers and explicit token
sequences. Those fixtures prove only control flow, parsing, schemas, and storage
integration. They are not evidence about the real SmolLM3 tokenizer, tags,
logits, or generated text.

### Natural and checkpoint control flow

Login-safe helpers own deterministic run planning, exact answer-block parsing,
reasoning span projection, full-precision entropy summaries, ties-to-even
checkpoint placement, alias groups, prefix identity, checkpoint metric math,
and output-size estimation. Torch-backed entropy extraction and generation are
behind lazy GPU-only execution boundaries.

Natural generation persists a terminal result before any checkpoint work.
Checkpoint work is independently keyed so resume/retry never regenerates a
successful natural run or a completed checkpoint. Raw pre-warper logits are
requested and scientific arrays are stored as float32-derived, unrounded
values. Full-vocabulary logits are never persisted.

### Prepared jobs

- CPU materialization: `jobs/materialize_part1_mmlu.sh`.
- GPU preflight: `jobs/part1_smollm3_preflight.sh`.
- GPU Smoke A: `jobs/part1_smoke_a.sh`.
- GPU Smoke B: `jobs/part1_smoke_b.sh`.

Only the CPU materialization job is the immediate next command. GPU scripts are
prepared but not submitted; their entry points fail closed until finalized
manifests and resolved model/tokenizer preflight metadata exist.

## Testing and evidence labels

Local tests cover the migrated contract, row normalization, deterministic
ordering, schema/hash validation, temp-first publication, failure rollback,
identical reruns, divergent-target rejection, pure parsing, checkpoint aliasing,
entropy/storage math, seed planning, and synthetic abnormal-output handling.

Evidence is labeled in three classes:

1. **Locally verified:** pure Python contracts, schemas, parsing, control flow,
   persistence, and shell syntax.
2. **Prepared but unverified:** actual MMLU Hub resolution/streaming/save/reload
   behavior in the Mila CPU job.
3. **GPU unverified:** real SmolLM3 revisions, tokenization, tags, context,
   logits, generation, entropy alignment, reproducibility, and bounded smoke.

## Documentation and handoff

All `docs/part1/` documents will record the partial Phase 2 state, exact Mila
command, expected logs/artifacts, post-job validation commands, and remaining
CPU/GPU gates. No dataset hash, study ID, model revision, environment value, or
smoke result will be invented before real execution.

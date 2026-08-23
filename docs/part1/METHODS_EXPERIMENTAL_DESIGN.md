# Experimental Pipeline and Experimental Design

## Scope and unit of observation

This document describes the intended completed experimental pipeline after a
selected MMLU item is passed to the model. Dataset selection, subject selection,
and model selection are outside its scope.

The scientific hierarchy is:

```text
question
└── natural trajectory/run (10 independently sampled runs)
    ├── original uninterrupted reasoning and natural terminal answer
    └── requested checkpoint (11 normalized fractions)
        ├── forced-close conditional probe
        └── checkpoint measurements
```

This hierarchy accurately represents the final design. The natural run is the
parent observation. Each checkpoint is a conditional interrogation of a prefix
of that already-generated run; checkpoint probing does not create, replace, or
continue the parent natural trajectory.

For the complete study, the design contains:

- 500 fixed questions across five subjects;
- 10 natural trajectories per question, for 5,000 natural run observations;
- 11 requested checkpoint observations per successfully executed natural run,
  for up to 55,000 logical checkpoint observations; and
- natural-run, checkpoint, uncertainty, commitment, parsing, and execution
  status information linked through stable parent-child identities.

## Chronological pipeline for one question

1. Supply the selected question text and its four labeled answer choices to the
   model in the model's thinking-mode chat format. The instruction asks the
   model to reason and finish with an A--D answer plus a numerical confidence.
2. Generate 10 stochastic natural trajectories independently from the same
   original question. The runs share the semantic prompt but have distinct,
   reproducible sampling streams.
3. Preserve each complete generated sequence as the original, uninterrupted
   natural run. Identify its reasoning-token span, terminal answer block,
   natural A--D answer, verbalized confidence, correctness, stop condition, and
   token-level uncertainty trace.
4. Divide the recognized reasoning span into 11 requested normalized progress
   locations from 0.0 through 1.0.
5. At each location, reconstruct the original prompt plus only the natural
   reasoning prefix available at that point. Append a model-specific
   forced-close cue that closes reasoning and begins the answer field.
6. Run a separate deterministic checkpoint probe from that constructed input.
   Record its textual answer and confidence, together with the model's answer-
   step distribution over the four candidate labels.
7. Derive within-run commitment measures from the ordered checkpoint answers,
   while retaining the natural terminal answer as the primary run outcome.

## Natural reasoning trajectories

### Independent natural runs

- One trajectory is one complete stochastic generation from the original
  question prompt.
- Every question receives exactly 10 natural runs, indexed 0 through 9. There
  is no additional greedy natural run.
- The 10 runs are independent samples from the same question and model
  condition. They do not continue one another and do not receive previous runs
  as context.
- Each run uses a distinct deterministic sampling seed derived from the stable
  model identity, question identity, run identity, and the experiment's base
  seed. Retries, when permitted, reuse the original run seed rather than
  creating a new scientific sample.

### Model input

The natural prompt contains:

- the question text;
- exactly four candidate choices labeled A, B, C, and D;
- an instruction to reason carefully; and
- an instruction to terminate with an A--D answer and an integer confidence
  between 0 and 100.

The prompt uses the model's native thinking-mode chat template. No gold answer,
checkpoint information, or result from another run is supplied.

### Reasoning span

The reasoning portion is identified in generated-token space using the
model-specific thinking boundaries.

- Recognized opening and closing reasoning tags are excluded from the reasoning
  tokens themselves.
- If no opening tag is emitted, reasoning begins with the first generated
  token.
- If no closing tag is found, all generated tokens after any recognized opener
  are treated as reasoning tokens for diagnostic and entropy purposes.
- If neither boundary appears, the entire generated sequence is treated as the
  reasoning span.
- Zero-length and unusually short reasoning spans are retained as model
  behavior; there is no minimum-length exclusion.

The retained natural-run information includes the prompt, generated token
sequence, decoded output, reasoning boundaries and text, stop condition,
reasoning-token count, and a full entropy value aligned with every generated
token.

### Natural terminal answer and correctness

The **natural final answer** is the answer in the model-recognized terminal
answer block following a valid natural reasoning closure. The answer and
confidence must come from the same terminal block; text from separate matches
is not combined.

- A valid natural answer is mapped to one of A, B, C, or D.
- `natural_correct` is the comparison between that parsed natural answer and
  the question's gold label.
- If the reasoning is unclosed, answer-like text inside the reasoning region
  may be retained diagnostically, but it is not promoted to the natural final
  answer; natural answer and natural correctness are missing.
- Missing, malformed, or out-of-domain terminal answers are explicitly
  represented, with natural correctness missing rather than guessed.
- Incorrect but successfully parsed natural answers are retained. Correctness
  is an outcome, not an inclusion criterion.

The natural final answer always belongs to the original uninterrupted
trajectory. No checkpoint answer is substituted for it, including the answer
elicited at checkpoint 1.0.

## Normalized trajectory checkpoints

### Progress definition

Trajectory length is the number of recognized reasoning tokens in the natural
run. The requested progress fractions are:

```text
0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
```

For a trajectory with `n` reasoning tokens, each fraction is mapped to an
integer number of retained reasoning tokens by rounding `fraction × n` to the
nearest integer, using ties-to-even behavior, and constraining the result to
the interval from zero through `n`.

- Fraction 0.0 retains no reasoning tokens.
- Fraction 1.0 retains the complete recognized reasoning span.
- Intermediate fractions retain the corresponding prefix, never a suffix or a
  resampled segment.
- The actual retained fraction is recorded because integer token positions may
  not equal the requested fraction exactly.
- Checkpoint construction occurs in token-ID space, preventing decoded-text
  boundaries from changing the retained prefix.

This normalization makes progress comparable across trajectories with
different reasoning lengths while preserving the original order of reasoning.

### Short trajectories and aliased checkpoints

For short or zero-length reasoning spans, multiple requested fractions can map
to the same physical prefix.

- All 11 requested checkpoint identities remain present in the scientific data.
- A repeated physical prefix may be probed once and represented by multiple
  explicitly linked logical checkpoint records.
- Such records are aliases of one conditional probe, not independent evidence.
- Aliased checkpoints do not create artificial answer transitions and are
  excluded as duplicate physical positions when computing switching and
  stabilization.
- For a zero-token reasoning span, all requested checkpoints refer to the same
  empty reasoning prefix and the actual retained fraction is undefined.

All eleven fractions remain part of the trajectory design. Fractions 0.0, 0.5,
and 1.0 may be highlighted in summaries, but they do not replace the full set.

## Forced-close checkpoint probing

### Probe construction

For each checkpoint, the model receives:

1. the original rendered question prompt, including the question and A--D
   choices;
2. the original natural generation up to the start of its recognized reasoning
   span, including any model-formatting or reasoning opener;
3. exactly the retained natural reasoning-token prefix for that checkpoint; and
4. a model-specific cue that closes the reasoning region and begins the answer
   field.

The cue is an intervention: it instructs the model to stop extending the
original reasoning and commit to an answer under the information present in
that prefix. The subsequent probe generation is greedy and deterministic.

### Relationship to the natural trajectory

- A checkpoint probe is a separate conditional inference from a frozen prefix
  of the natural trajectory.
- It does not edit, truncate retroactively, or resume the stored natural run.
- It does not alter the natural terminal answer or natural correctness.
- The 11 probes interrogate different information states along the same parent
  trajectory. They are not a chain in which one checkpoint answer is fed into
  the next.
- Checkpoint 1.0 asks what the model reports when forced to close after the full
  recognized reasoning prefix. It can legitimately disagree with the answer
  produced by the original uninterrupted generation.

### Checkpoint output

The forced-close generation is expected to provide:

- a textual A--D checkpoint answer; and
- a verbalized integer confidence from 0 to 100.

The parsed checkpoint answer is compared with the gold label to define
checkpoint-local correctness. This is a secondary checkpoint outcome. It is
not the central correctness target for trajectory-level prediction.

## Uncertainty and commitment measurements

### Measurement inventory

| Signal | Location | Belongs to | Conceptual interpretation |
|---|---|---|---|
| Per-token full-vocabulary entropy | Every generated token; reasoning-token subset identified explicitly | Natural trajectory | Uncertainty in the model's next-token distribution during uninterrupted reasoning |
| Mean reasoning-token entropy | Across all recognized reasoning tokens | Natural trajectory | Overall uncertainty during the run's reasoning process |
| Tail reasoning-token entropy | Final 10% of recognized reasoning tokens, with at least one token when reasoning is nonempty | Natural trajectory | Uncertainty near the end of natural reasoning |
| Natural verbalized confidence | Natural terminal answer block | Natural trajectory | Model's stated confidence in its uninterrupted final answer |
| Checkpoint textual answer | Every requested checkpoint when parsable | Checkpoint probe | Discrete commitment induced from the available reasoning prefix |
| Checkpoint verbalized confidence | Every requested checkpoint when parsable | Checkpoint probe | Model's stated confidence in the forced checkpoint answer |
| Four-choice probabilities | Answer-token step at every measurable checkpoint | Checkpoint probe | Relative support assigned to A--D at that reasoning stage |
| Four-choice answer entropy | Answer-token step at every measurable checkpoint | Checkpoint probe | Uncertainty restricted to the four task-relevant choices |
| Maximum A--D probability | Answer-token step at every measurable checkpoint | Checkpoint probe | Strength of the most-supported candidate choice |
| Full-vocabulary answer-step entropy | Answer-token step at every measurable checkpoint | Checkpoint probe | Broader next-token uncertainty when the answer is produced |
| Switching and stabilization measures | Across ordered non-aliased checkpoint answers | Derived from one natural run's probes | Evolution and persistence of discrete answer commitment |

### Reasoning-token entropy

For each token generated in the natural run, entropy is calculated from the
model's raw full-vocabulary next-token distribution before sampling controls
alter the distribution used to choose a token. The entropy sequence is aligned
one-to-one with the generated-token sequence and retained at full temporal
resolution.

Reasoning-boundary metadata selects the subset belonging to recognized
reasoning. This supports three levels of representation:

- the complete ordered reasoning-token entropy trace;
- the mean over all recognized reasoning tokens; and
- the mean over the final 10% of recognized reasoning tokens.

Normalized token progress can be related to checkpoint positions through the
same retained-prefix counts used to construct checkpoints. The fixed design
does not define an additional interpolated or windowed checkpoint-local
reasoning-entropy statistic; such a summary should not be invented when
describing the experiment.

### Answer-choice uncertainty

At the generation step where the checkpoint answer token is identified, the
pipeline extracts the logits associated with the four prevalidated A--D answer
tokens. These four logits are renormalized over A--D to obtain a four-choice
probability distribution. The pipeline then records:

- the four A--D probabilities;
- entropy of that four-choice distribution;
- its maximum probability; and
- entropy of the full vocabulary distribution at the same answer step.

The answer-choice distribution and the textual forced-close answer are
distinct measurements obtained from the same checkpoint inference:

- the textual answer is parsed from the generated forced-close response; and
- the probability measurement is calculated from answer-step logits.

A checkpoint is fully valid only when the textual A--D answer, identified
answer token, and associated distributional measurement form a consistent
bundle. If the answer step cannot be located or measured, the textual output
and its statuses are retained but unavailable probability measures remain
missing rather than being inferred.

### Verbalized confidence

The model is asked to state an integer confidence between 0 and 100 together
with its answer.

- Natural confidence comes from the same terminal block as the natural final
  answer.
- Checkpoint confidence is requested at every checkpoint and belongs to the
  checkpoint's forced-close answer.
- Valid confidence is retained both in its original integer form and normalized
  to the interval from 0 to 1.
- Missing, malformed, and out-of-range values receive separate statuses.
- Values outside the valid range are not silently clamped; their normalized
  confidence is missing.

### Commitment dynamics

The planned commitment variables are:

- **Checkpoint answer identity:** the parsed A--D answer at each requested
  progress fraction.
- **Answer-switch count:** the number of changes between adjacent physical,
  non-aliased checkpoints for which both answers are valid. A missing or
  malformed answer breaks adjacency; the analysis does not bridge across it.
- **Valid-transition count:** the number of adjacent checkpoint transitions
  that were evaluable under the preceding rule.
- **First natural-answer appearance:** the first requested checkpoint at which
  the valid forced answer equals the natural final answer. It is missing when
  the natural answer is unavailable or never appears.
- **Stabilization fraction:** the earliest requested checkpoint after which all
  later non-aliased checkpoints have valid answers equal to the final valid
  forced answer. It is unavailable if the endpoint is invalid, a required later
  answer is missing or malformed, or no stable suffix can be established.
- **Leaving a correct answer:** whether a transition changes from a correct
  checkpoint answer to an incorrect one.
- **Later recovery:** whether a correct checkpoint answer is observed after the
  run has left a correct answer.
- **Natural/forced endpoint agreement:** whether the valid checkpoint-1.0
  answer equals the valid natural final answer.

Stabilization operationalizes persistence of a selected checkpoint answer. The
design does not define a separate generic "first answer selected" variable;
the specified appearance variable is explicitly the first appearance of the
eventual natural answer.

## Relationship to final correctness

The primary outcome for trajectory-level uncertainty and commitment signals is
the correctness of the **original uninterrupted natural final answer**.

Consequently, checkpoint measurements are interpreted as early or intermediate
signals about where their parent natural trajectory ultimately ends. For
example, answer entropy at an early checkpoint, the maximum A--D probability,
or later switching behavior can be related to `natural_correct` even when the
temporary checkpoint answer itself is wrong or differs from the natural answer.

Checkpoint-local correctness is retained as a separate secondary outcome. It
is used when the scientific question specifically concerns the validity or
calibration of a checkpoint answer. Checkpoint-local correctness never replaces
natural correctness as the main trajectory outcome.

## Scientific data structure

### Question level

Each of the 500 fixed questions contributes subject identity, question and
choice content, and a gold A--D label.

### Natural-run level

Each question contributes 10 natural runs. A run-level observation contains:

- the uninterrupted generated trajectory and reasoning boundaries;
- aligned token-level entropy information;
- natural answer, confidence, and natural correctness when available;
- parsing and execution statuses; and
- links to its 11 requested checkpoints.

### Checkpoint level

Each successfully executed natural run contributes 11 logical checkpoint
observations. Each checkpoint contains:

- requested and actual normalized progress;
- retained-prefix and alias information;
- forced answer and checkpoint-local correctness when available;
- verbalized confidence when available;
- A--D logits/probabilities and uncertainty measures when measurable; and
- independent execution, parsing, token-location, and measurement statuses.

### Derived trajectory level

The ordered checkpoint set is reduced to one set of commitment features per
natural run, including switching, first natural-answer appearance,
stabilization, leaving/recovery, and endpoint agreement. Raw checkpoint records
remain available; derived summaries do not replace them.

## Retention, failures, missingness, and exclusions

### Successful model behavior

- Correct and incorrect natural runs are both retained.
- Successfully executed but unusual, repetitive, truncated, unclosed,
  malformed, or unparsable generations are retained as model behavior.
- There is no automatic retry or exclusion based on answer correctness,
  repetition, reasoning length, missing confidence, malformed text, or
  disagreement between checkpoint 1.0 and the natural answer.
- A successfully executed natural run remains checkpoint-eligible even when it
  lacks a valid natural answer, lacks confidence, reaches the generation cap,
  has no reasoning tokens, or lacks a reasoning close marker.

### Missing natural measurements

- If the natural answer cannot be validly parsed, natural answer and
  `natural_correct` are missing, while available trajectory text, reasoning
  boundaries, entropy, and diagnostic statuses are retained.
- If natural confidence is missing, malformed, or out of range, the confidence
  measurement is missing without invalidating the natural run.
- A natural infrastructure failure produces no usable generated trajectory and
  makes all of its checkpoints ineligible.

### Missing checkpoint measurements

- A checkpoint execution can complete while its model output is scientifically
  invalid or only partially measurable.
- Failure to parse the answer leaves checkpoint answer and checkpoint-local
  correctness missing.
- Failure to parse confidence removes only the confidence measurement.
- Failure to locate the answer-token step or calculate its distribution removes
  the associated probability and entropy measures.
- An individual checkpoint failure or missing measurement does not erase the
  parent natural trajectory or other valid checkpoints.
- Missing or malformed checkpoint answers break adjacency for switching and can
  make stabilization unavailable, rather than causing imputation across the
  gap.

### Retries

Retries are reserved for infrastructure failures that are plausibly transient,
not for scientifically undesirable model outputs. Retryable conditions include
interrupted execution, temporary filesystem or worker failure, and transient
accelerator-runtime failure. Deterministic incompatibilities and reproducible
resource or configuration failures are terminal. There are at most three
attempts for a logical natural or checkpoint observation, and every retry
preserves its original scientific identity and seed.

The intended analysis uses explicit availability and failure statuses rather
than treating absent observations as evidence of model behavior. No automatic
substantive exclusion beyond the availability requirements of a particular
measurement is defined.

## Main Methods versus appendix

| Keep in main Methods | Better suited to appendix |
| -------------------- | ------------------------- |
| Ten independent natural trajectories from the same original question | Exact sampling hyperparameters and maximum token limits |
| Natural final answer comes from the uninterrupted trajectory | Exact seed value and hash-based seed derivation |
| Reasoning-span definition and preservation of abnormal successful runs | Exact model and tokenizer revisions and token IDs |
| Eleven normalized reasoning-prefix checkpoints from 0.0 to 1.0 | Ties-to-even rounding implementation and canonical identity formulas |
| Prefix-only forced-close intervention and separate deterministic probes | Full forced-close string, chat template, and complete prompt text |
| Distinction between natural answer, checkpoint textual answer, and A--D probability distribution | Parser regular expressions and exact terminal-block grammar |
| Natural reasoning-token entropy, checkpoint answer-choice entropy, maximum answer probability, and verbalized confidence | Numeric precision, tensor handling, and library-specific inference options |
| Switching, first natural-answer appearance, stabilization, recovery, and endpoint agreement | Schema field names, hashes, file layout, locking, and command-line arguments |
| Natural correctness as the primary trajectory outcome; checkpoint correctness as secondary | Hardware, scheduler resources, package versions, and storage paths |
| Scientifically relevant missingness, retention, and retry principles | Detailed infrastructure failure taxonomy, retry delays, and recovery mechanics |
| Aliased checkpoints are retained logically but not treated as independent transitions | Alias identity fields and low-level shared-probe implementation |

## Information required for the pipeline figure

The Methods pipeline diagram should show these exact conceptual elements:

1. **Selected MMLU question:** question text plus four labeled choices.
2. **Branch to 10 independent natural generations:** same original question,
   separately sampled reasoning trajectories.
3. **Preserved natural-run branch:** uninterrupted reasoning, natural terminal
   answer, natural verbalized confidence, natural correctness, and natural
   reasoning-token entropy trace.
4. **Normalized prefix construction:** 11 requested fractions from 0.0 to 1.0
   along each run's recognized reasoning tokens.
5. **Forced-close intervention at each prefix:** original question plus retained
   natural reasoning prefix, followed by a cue to stop reasoning and answer.
6. **Separate deterministic checkpoint inference:** emphasize that each probe
   branches from the frozen natural prefix and does not feed back into the
   natural trajectory or another checkpoint.
7. **Checkpoint measurements:** textual A--D answer, checkpoint confidence,
   A--D probability distribution, answer-choice entropy, maximum A--D
   probability, and answer-step uncertainty.
8. **Commitment trajectory:** ordered checkpoint answers leading to switching,
   first natural-answer appearance, stabilization, recovery, and endpoint
   agreement measures.
9. **Primary relationship:** uncertainty and commitment signals across the run
   point to the correctness of the original natural final answer.
10. **Secondary relationship, visually separated:** checkpoint answer and
    checkpoint-local correctness.

A refined high-level figure sequence is therefore:

```text
MMLU question + A--D choices
→ 10 independent natural reasoning trajectories
→ preserve each uninterrupted natural answer and correctness
→ construct 11 normalized reasoning prefixes per trajectory
→ separately force-close and probe each prefix
→ extract checkpoint uncertainty, confidence, and answer commitment
→ derive within-trajectory commitment dynamics
→ relate trajectory signals to the original natural final correctness
```

## Open Methodological Questions

### Probability-implied answer

The fixed design records the four renormalized A--D probabilities and their
maximum, while the discrete checkpoint answer is the parsed textual answer from
the forced-close generation. It does not specify a separate stored
probability-argmax answer variable or a tie-handling rule for such a variable.
If the paper intends to report a distinct "answer implied by the candidate-
choice distribution," that derived variable and its tie policy require an
explicit methodological decision. No such variable is needed to describe the
currently specified checkpoint textual answer, answer entropy, or maximum
probability measurements.

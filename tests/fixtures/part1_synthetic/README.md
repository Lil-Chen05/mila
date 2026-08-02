# Part 1 synthetic evidence

Everything in this directory is fabricated test input. It is deliberately
separate from `results/part1-smoke/`, `results/part1/`, and the tracked study
manifests.

The catalog records the narrow claim each scenario may support. Synthetic
records and fake tokenizer/model objects may exercise parsing, schema
validation, persistence, recovery, retry, resume, and selection control flow.
They are **not** evidence about real SmolLM3 tokenization, reasoning tags,
logits, entropy, generation quality, CUDA behavior, or reproducibility.

Entries marked `implemented` point to the test that constructs or consumes the
synthetic condition. The six analysis-edge families construct schema-valid raw
natural or checkpoint inputs for class balance, invalid bootstrap draws, draw
multiplicity, macro-subject invalidity, within-question mixed correctness, and
switching with missing checkpoints. Their tests assert only schema validity and
the stated input condition. They do not implement or validate Phase 3 AUROC,
bootstrap, macro aggregation, within-question analysis, switching, or
stabilization logic, and they must not be reported as real-model evidence.

# Completed Part 1 handoff

Part 1 is complete: production generation, recovery-validated merge,
`final-r5000` analysis, and the paper are finished. The reviewed report is
[report/main.pdf](report/main.pdf), and the validated numerical interpretation
is [analysis/final-r5000/RESULTS_VALIDATION_SUMMARY.md](analysis/final-r5000/RESULTS_VALIDATION_SUMMARY.md).

## Immutable provenance

- Question manifest: 500 MMLU test questions, 100 per fixed subject; question
  hash `dd379f48322d6eb07c309101361738a965be320b4124bb45bd44723b1abe474d`;
  study hash `859eecd5e0437a901555d5fd2d99692feccb5257df16de60bfe0fe648626142b`.
- Generation: model run
  `6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c`,
  bound to commit `ffa998a7ee1f156e150c8da33b258165ee53e032`;
  array `10362272` plus targeted retry `10362285` produced 500 finalized
  shards, 5,000 natural rows, and 55,000 checkpoint rows.
- Standard validation: job `10381201` and validation ID
  `2bfe7cd6908351e3f1d6c9a2eec4f41c9dfa97f124f9da2f70925365490f23db`
  remain preserved as failed because of the documented prompt-hash-domain
  validator defect. The failed report was never rewritten as a pass.
- Recovery merge: finalize job `10385970` published 5,000 natural, 55,000
  checkpoint, and 120,003 audit rows. Merge ID
  `447cfc9125349369f24b3e0e6865c254b516ceb84c70263d9f0a0e36801938e6`;
  merge-manifest hash
  `a2f47af9378a6906c64f4f0ea9ae76d9d2f41c67be913b7c9cca5fe63dcbce03`.
- Final analysis: publication job `10391026` validated and atomically promoted
  the preserved 5,000-bootstrap stage. Analysis ID
  `2c141766ccd3e77c8692294bcb067c3ea66bfcfd2fd0f18c2ca3d61c45f01bb7`;
  analysis-manifest hash
  `ea1e182034bd66fee2c7300e67f4db2a583965474d5e510a2f176136265cce9c`;
  `paper_analysis_ready: true`.

## Maintenance rules

- Do not rerun generation, validation, merge, recovery, or analysis jobs. Do
  not replace receipts, rewrite the preserved failed validation report, or
  mutate published production artifacts.
- Treat output provenance explicitly: original exports are **fixed**;
  verbalized confidence is **repaired** by the documented narrow parser repair;
  prefix reasoning entropy is a **reconstructed intended analysis**. Do not
  collapse these labels.
- Never load a model, tokenizer, or Hugging Face dataset on a Mila login node.
  Any future model/data execution requires a new reviewed study extension and a
  compute-node SLURM job.
- The full operational chronology, exact job dependencies, failures, waivers,
  receipts, paths, and no-resubmission warnings are preserved in
  [docs/part1/OPERATIONS_HISTORY.md](docs/part1/OPERATIONS_HISTORY.md).
- The scientific and engineering contracts remain under
  [docs/part1/](docs/part1/); `AGENTS.md` is the maintenance/safety contract.

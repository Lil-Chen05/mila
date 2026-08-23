# Part 1 documentation

Part 1 is complete. The fixed 500-question study produced 5,000 natural
trajectories and 55,000 checkpoint records; the canonical merge contains 5,000
natural, 55,000 checkpoint, and 120,003 audit rows. The immutable
`final-r5000` analysis is published with `paper_analysis_ready: true`, and the
reviewed paper is [../../report/main.pdf](../../report/main.pdf).

The correctness cohort contains 3,550 evaluable natural answers (3,172 correct,
378 incorrect); 84 questions contain both outcomes and support the
within-question analysis. These counts and the final scientific interpretation
are validated in
[../../analysis/final-r5000/RESULTS_VALIDATION_SUMMARY.md](../../analysis/final-r5000/RESULTS_VALIDATION_SUMMARY.md).

## Document map

- [DECISIONS.md](DECISIONS.md) — immutable scientific and engineering choices.
- [SCHEMA.md](SCHEMA.md) — records, identities, hashes, manifests, and lifecycle.
- [PLAN.md](PLAN.md) — completed implementation plan and phase record.
- [VALIDATION.md](VALIDATION.md) — acceptance criteria and evidence.
- [RUNBOOK.md](RUNBOOK.md) — safe read-only verification and historical
  operational procedures.
- [STATUS.md](STATUS.md) — final status plus the preserved execution ledger.
- [OPERATIONS_HISTORY.md](OPERATIONS_HISTORY.md) — exact recovery chronology,
  job dependencies, artifact paths, receipts, and no-resubmission warnings.

## Provenance vocabulary

- **Fixed** outputs are original immutable production-analysis exports.
- **Repaired** verbalized-confidence values come from the narrow deterministic
  parser repair documented in the validation summary.
- **Reconstructed intended analysis** refers to prefix reasoning entropy,
  recovered for the intended checkpoint-level analysis while reproducing fixed
  endpoint anchors.

These categories must remain distinct. The completed paper and scientific
metrics are immutable maintenance baselines; future work requires an explicitly
versioned extension and new provenance rather than modification of Part 1.

# Checkpoint probe — exploratory findings (20q, decile resolution)

> **Status: EXPLORATORY, directional only — NOT a finding.**
> Caveats that gate every claim below:
> - **single subject** (`abstract_algebra`) — not subject-representative
> - **n = 19** usable closed questions (qid14 hit the 4096-token cap unclosed; reported separately)
> - **1 greedy run/question** — no sampling spread
> - data produced off **mirrored gpu helpers** (the `gpu_common.py` refactor is still pending;
>   nothing here ships as a finding until the pipeline is consolidated and the data regenerated)
> - the **incorrect group is n = 6** → results are **suggestive, not significant**. No p-values reported on n=6.

## Cohort
- 220 rows = 20 questions × 11 deciles. Closed cohort = **19 questions** (209 rows).
- **qid14** (unclosed, `think_closed=False`) is held out and never averaged in.
- Question-level groups use **final-checkpoint (frac=1.0) correctness** (`forced_letter==gold`),
  not `natural_pred==gold`: **13 correct / 4 incorrect**.

## 1. Whole-cohort trajectories (basic)
As more reasoning is spliced in, the forced answer's letter-entropy collapses and verbalized
confidence drifts up:

| | frac 0.0 (no reasoning) | frac 1.0 (full chain) |
|---|---|---|
| mean answer-letter entropy (nats) | 1.142 | 0.054 |
| mean verbalized confidence | 88.8 | 95.6 |
| forced-answer accuracy | 0.53 | 0.76 |

See `fig_entropy_vs_frac.png`, `fig_confidence_vs_frac.png`, `fig_accuracy_vs_frac.png`.

**Two wrinkles, both honest gaps not bugs:**
- Accuracy is **non-monotonic**: it dips to **0.35 at frac 0.1** before
  recovering to 0.76 — a *little* reasoning underperforms *none* here. Intriguing (echoes
  "partial chain-of-thought can hurt") but well within n=19 noise; flagged, not claimed.
- **0 checkpoint** (qid15 @ frac 0.5) produced no clean answer token, so its
  entropy is left as a visible `NaN` rather than guessed; that one cell's mean is over 18, not 19.

## 2. HEADLINE — convergence *timing*, not entropy *level*
Splitting by final correctness, the correct group's entropy collapses **earlier**. The gap
(incorrect − correct mean entropy) **opens mid-chain and vanishes at the endpoint**:

- gap peaks at **frac 0.8** (**0.676 nats**),
- endpoint gap (frac 1.0) = **-0.045 nats** (H_correct=0.064, H_incorrect=0.019) — effectively converged.

**This is not "lower entropy ⇒ correct."** At the endpoint both groups are equally (near-zero)
entropy; the signal is *how soon* they get there. See `fig_entropy_by_correctness_frac.png`
(grey band = mid-chain region where the gap lives).

## 3. Chain-length confound (the first thing a reviewer will raise)
The worry: if incorrect chains were longer, a given *fraction* would map to more *absolute*
tokens, and "earlier convergence" could be a length artifact. **In this data the premise does
not hold** — the groups are length-balanced:

| group | mean n_think | median n_think |
|---|---|---|
| correct (n=13) | 1613 | 1424 |
| incorrect (n=4) | 1250 | 887 |

Means are within ~1.5%, and the **median runs the *opposite* way** (correct chains are if
anything *longer*). So there is little length confound to begin with. The decisive check is
re-plotting entropy against absolute `k_keep` (`fig_entropy_by_correctness_abs.png`):

> **survives: correct group shows lower mean entropy in 7/7 overlapping absolute-position bins.**

Since the separation holds at *matched absolute token positions*, the earlier-convergence effect
in §2 is **not** a chain-length artifact. (Caveat: the correct group has few points in the
highest-k bins — short-and-medium chains dominate — so the very-high-k tail is thin.)

## 4. Answer lock-in / flips
Earliest fraction after which the forced letter equals its final value and never changes again:

- mean commit fraction: **correct 0.35** vs **incorrect 0.68**
- mean answer flips along the chain: **correct 1.46** vs **incorrect 2.75**

(Consistent direction with §2 if correct questions commit earlier / flip less.) See
`per_question_summary.csv`.

## 5. Calibration of the two uncertainty signals
Token entropy vs verbalized confidence across all closed checkpoints:
**Pearson r = -0.521, Spearman ρ = -0.476** (negative = the two signals agree:
higher stated confidence ↔ lower entropy). See `fig_calibration.png`.

## Held-out: qid14, and the qid15 gap (kept visible, never silently dropped)
- **qid14** — unclosed chain (never emitted `</think>` within 4096 tokens). It *did* produce a
  clean forced answer at every checkpoint (letter `C`, which happens to be gold), but stayed
  **high-entropy (~1.0 nats) throughout — it never converged**, consistent with never closing its
  reasoning. The 1.0 invariant is **N/A** because the *natural* chain emitted no committed answer
  to compare against (not because the forced probe failed). Excluded from all aggregates above.
- **qid15 @ frac 0.5** — a *closed*-cohort checkpoint where the forced probe emitted no parseable
  answer token → `H_letter`/`forced_letter` left as `NaN`/None (honest gap). It is the lone
  missing cell in §1's frac-0.5 means. Note: in §4 the flip count treats that `None` as a distinct
  symbol, so qid15's flip tally is mildly inflated.

---
*Generated by `analysis/analyze_checkpoints.py` from `results/20q/checkpoints.jsonl`.*

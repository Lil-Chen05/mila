# Results Validation Summary

Human-readable validation of the final `final-r5000` analysis used in the
completed report. Numbers are cross-checked against the fixed outputs
(`analysis/final-r5000/*.csv`), the generated `fig*_values.json`, the confidence
repair artifacts, and `RESULTS_STORY_AUDIT.md`. Where a value is a **fixed**
export, a **repaired** measurement, or a **reconstructed intended analysis**, this
is stated. All publication figures, tables, and report sections are complete.

---

## 1. Analysis cohort

The experiment generated **5,000 natural trajectories** — 10 stochastic runs for
each of **500 questions** (100 per subject × 5 subjects). All 5,000 generation
attempts produced outputs, but under the analysis criteria only **3,550 (71.0%)**
yielded an evaluable natural A–D answer (**3,172 correct / 378 incorrect**). The
remaining 1,450 are unavailable because the reasoning close was missing (192) or a
valid A–D answer could not be parsed (1,258). Evaluability varies substantially by
subject, and because it is subject- and length-dependent, per-subject accuracies
can be compared descriptively but should **not** be read as directly comparable
estimates of full-cohort performance. The within-question RQ2 analysis uses the
**84 mixed-outcome questions** that yielded at least one correct and one incorrect
evaluable run.

| Subject | Evaluable, n (%) | Correct / Incorrect | Acc. (%) | Mixed-outcome Q |
|---|---|---|---|---|
| Mathematics | 598 (59.8) | 588 / 10 | 98.3 | 6 |
| Physics | 647 (64.7) | 549 / 98 | 84.9 | 21 |
| Chemistry | 578 (57.8) | 503 / 75 | 87.0 | 21 |
| Biology | 847 (84.7) | 747 / 100 | 88.2 | 18 |
| Psychology | 880 (88.0) | 785 / 95 | 89.2 | 18 |
| **Total** | **3,550 (71.0)** | **3,172 / 378** | **89.4** | **84** |

**Main caveat:** the primary cohort is the 71% with an evaluable answer, not the
full 5,000. Availability is uneven by subject (57.8%–88.0%) and is length-related
(unavailable runs are much longer on average). Report pooled **and** subject-macro;
Mathematics accuracy (98.3%) sits on a cohort missing 40% of its runs. *(All values
match the fixed `trajectory_features.csv` accounting and the audit §1.)*

---

## 2. Confidence repair

**What was wrong.** The fixed parser required a bare integer after `Confidence:`.
In almost all runs the model emitted a valid integer with the decoded end-of-turn
sentinel attached (e.g. `95<|im_end|>`, sometimes bracket-wrapped `<100>`), which
the parser rejected. The result was that natural verbalized confidence was usable
for only **1** evaluable run and checkpoint confidence collapsed at later fractions
— a measurement/decoding failure, **not** the model declining to answer.

**Repair applied (deterministic, narrow).** On the already-extracted confidence
field: strip a single trailing `<|im_end|>` sentinel and an optional `<…>` wrap,
then accept **only** a bare integer in [0, 100]. The `Answer:`/`Confidence:` block
contract is unchanged. This is an **implementation/parser repair** of a signal that
was part of the intended design, not a new analysis.

**Not repaired (left missing on purpose).** Percent signs (`100%`), literal prompt
template echoes (`<integer 0-100>`), markdown-wrapped fields, and other
structurally different outputs. Anything not unambiguously an integer stays missing.

**Recovery counts.**
- Natural: 4,354 malformed → **4,352 recovered**, 2 not (`100%<|im_end|>`).
- Checkpoint: 48,497 malformed → **47,227 recovered**, 1,270 not (template echoes / `%` / junk).
- Previously-`parsed` values are **unchanged** (verified, both levels); all final values are integers in [0, 100] (verified).

**47,227 vs 47,228 audit discrepancy.** The audit reported 47,228 salvageable
checkpoint values; the locked repair recovers 47,227. The single difference is one
value `<100></think>`, contaminated by the `</think>` reasoning-close tag rather
than the `<|im_end|>` sentinel. The audit's broader strip removed `</think>` too;
the locked repair intentionally does not, so this one ambiguous case stays missing.
This is a rule-scope difference, not an error.

**Coverage after repair.** Natural confidence is usable for **3,542** of the 3,550
evaluable runs (3,166 correct / 376 incorrect) — this is what makes confidence
estimable for RQ1/RQ3. Checkpoint coverage is **92.2%–99.6%** across all 11
fractions (was 3,435→2 parsed before repair), so confidence is plottable at every
checkpoint.

**Substantive observation.** Recovered confidence is **high and compressed**:
natural mean ≈ 93.5, median 95, 74.7% ≥ 90, 36.5% = 100 (checkpoint mean ≈ 91.7).
Weak dynamic range is itself a finding, independent of any discrimination result.

---

## 3. RQ1 — Signal evolution

### Figure 1

**Panel A — local reasoning-token entropy**
- *Question:* how does token-level uncertainty evolve along the natural trajectory?
- *Quantity:* mean per-token entropy (nats), averaged within 10 equal token-progress deciles, pooled over the 3,550 evaluable runs; 95% question-cluster bands.
- *Pattern:* rises sharply over ~the first 20–30%, then plateaus: approximately **0.35 → 0.51 → 0.61 → plateau ≈ 0.62–0.66**.
- *Interpretation:* token-level uncertainty settles after an initial climb rather than resolving as reasoning proceeds.
- *Caveat:* this is the natural-trajectory clock (token progress), a descriptive summary of the stored per-token entropy trace; not a forced-probe quantity.

**Panel B — checkpoint answer-choice entropy**
- *Question:* how does uncertainty over the four answer choices evolve?
- *Quantity:* answer-choice entropy (nats) at the 11 forced-answer checkpoints; coverage 89–99%.
- *Pattern:* remains high early (≈ 0.61 at f=0, small rise to ≈ 0.69 by f=0.1–0.2) then declines sharply over the latter half to **≈ 0.03** at f=1.0.
- *Interpretation:* the answer distribution becomes increasingly concentrated on a single option as the model commits.
- *Caveat:* forced-probe clock; f=0 is a pre-reasoning baseline (25 questions carry a date-drifted prompt — appendix).

**Panel C — repaired verbalized confidence**
- *Question:* how does stated confidence evolve?
- *Quantity:* mean recovered confidence (0–100) at the 11 checkpoints; coverage 93–100%.
- *Pattern:* high and compressed throughout with a modest upward drift, **≈ 89.6 → 93.7**.
- *Interpretation:* stated confidence is near-ceiling and nearly constant — a different facet of "certainty" than either entropy signal.
- *Caveat:* repaired signal; drawn on a truncated 80–100 axis to reveal the compressed range.

**RQ1 Figure 1 takeaway.** The three signals do not evolve in lockstep — token-level
uncertainty climbs then plateaus, answer-choice uncertainty stays high then collapses
as the model commits, and stated confidence stays high and compressed — so they
capture different aspects of the model's developing certainty.

### Figure 2

- *Question:* how does each signal's ability to discriminate **natural final-answer correctness** change with the reasoning available up to each checkpoint (AUROC, oriented so >0.5 = better)?
- *Quantity:* pooled AUROC vs natural correctness at each fraction; 95% question-cluster bands; base cohort 3,550.

**Main numerical patterns.**
- **Prefix reasoning entropy** (mean token entropy through each checkpoint; **reconstructed intended analysis**): rises to the **highest AUROC point estimate among the signals shown at ~20–30% progress, ≈ 0.774**, and holds the highest point estimate thereafter, declining gradually to **0.703 at f=1.0**. (Begins at f=0.1; undefined at f=0 where no reasoning tokens are retained.)
- **Answer-choice entropy** (completeness computation over pre-specified checkpoints): modest early discrimination ≈ **0.60**, improving over the second half to ≈ **0.686 at f=0.9**, 0.680 at f=1.0.
- **Verbalized confidence** (repaired, **checkpoint** probe): close to chance early (≈ **0.51–0.55**), improving later to ≈ **0.667 at f=1.0**.

**Exact reproduction checks (all pass).**
- Prefix entropy at f=1.0 reproduces the stored full-trajectory `mean_reasoning_entropy` to **2.2 × 10⁻¹⁶**.
- Prefix entropy at f=1.0 AUROC = **0.7025**, identical to the fixed primary `negative_mean_reasoning_entropy` AUROC.
- Answer-choice entropy AUROCs at 0.0 / 0.5 / 1.0 = **0.6055 / 0.6015 / 0.6799**, matching the fixed export exactly.

**Interpretation.** Reasoning-token uncertainty carries discriminative information
early and remains the strongest signal by point estimate throughout; the
answer-distribution and stated-confidence signals become informative only as the
model commits. The post-peak decline of prefix entropy is an interpretation
**consistent with** later portions of the trajectory contributing less
discriminative information — not a proven mechanism (no paired position-by-position
test).

**Endpoint caveat.** Among the **3,467** trajectories with both values available,
the endpoint forced answer matched the natural final answer in **all 3,467** cases,
so the f=1.0 answer-derived probes are best read as an **endpoint confidence
readout** rather than an independent intermediate prediction. This does **not**
apply to prefix reasoning entropy, whose f=1.0 value is just the full-trajectory
mean.

**RQ1 Figure 2 takeaway.** Reasoning-token entropy computed over only the first
~20–30% of reasoning already yields the highest discrimination point estimate of
eventual correctness (≈ 0.77), while answer-distribution and confidence signals are
weak early and strengthen only as the answer is committed.

### RQ1 commitment results (prose, no extra main figure)

Descriptive, from `trajectory_features.csv` (matches audit §7):
- **First appearance** = the fraction at which the elicited (forced) answer choice that ultimately equals the natural final answer first appears at a checkpoint: mean **0.268** (among 3,525 runs where it appears).
- **Stabilization** = the fraction from which the elicited checkpoint answers no longer change through the endpoint: mean **0.496** (n=4,220) — roughly half the trajectory follows stabilization.
- Mean switch count **0.613**; forced answers frequently change before stabilizing, and departures from a locally correct answer usually reverse: **566 of 634** trajectories that left a correct forced answer later returned to one.
- These describe elicited-answer behaviour under probing; as rankers of eventual correctness they are weak (quantified in RQ3).

---

## 4. RQ2 — Correct versus incorrect trajectories

### Figure 3A — pooled comparison
- *Compared:* per-decile reasoning-token entropy for correct (n=3,172) vs incorrect (n=378) evaluable runs.
- *Gap:* incorrect runs are higher-entropy in every decile; mean gap over deciles ≈ **0.129 nats**.
- *Temporal pattern:* the gap is present throughout, largest early/mid and attenuating late (correct 0.336→0.636; incorrect 0.456→0.698).
- *Why confounded:* this pools across questions, so it can reflect stable **between-question difficulty** (harder questions are both higher-entropy and more error-prone) rather than a trajectory-level difference.

### Figure 3B — within-question comparison (fixed, pre-specified)
- *Why 84 questions:* only these have both a correct and an incorrect evaluable run, so the difference can be taken within a fixed question.
- *Estimand:* per question, mean(incorrect) − mean(correct) reasoning entropy, then an equal-weight average across the 84 questions.
- *Result:* mean **+0.0113 nats**, 95% CI **[0.0002, 0.0225]**; sign split **48/84** in the expected direction (incorrect > correct).
- *Subjects:* Physics 21, Chemistry 21, Biology 18, Psychology 18, Mathematics 6.
- *Why small/heterogeneous:* the interval only just excludes zero, barely more than half the questions run in the expected direction, and the per-question effect varies in sign and magnitude.

**Pooled vs within-question.** The pooled ≈ 0.13-nat separation is **substantially
attenuated** to ≈ 0.011 nats when restricted within question. This **suggests a
large between-question component**; it does **not** mathematically decompose the
pooled difference (different estimands/cohorts).

**Within-question commitment (null).** Switch count **−0.0923 [−0.3143, +0.1290]**
(n=84) and stabilization **−0.0359 [−0.1280, +0.0586]** (n=77) both span zero; tail
entropy **−0.0028 [−0.0295, +0.0240]** also null. Only mean reasoning entropy shows
a within-question difference whose interval excludes zero.

**RQ2 takeaway.** Correct trajectories have modestly lower reasoning-token entropy
than incorrect ones for the same question, but the effect is small and heterogeneous
(+0.0113 nats [0.0002, 0.0225]); most of the large pooled separation appears to
reflect between-question difficulty, and no commitment measure shows a within-question
difference.

---

## 5. RQ3 — Comparison of reliability signals

### Figure 4A — discrimination (vs natural correctness)

| Signal | AUROC | 95% CI | n | Provenance |
|---|---|---|---|---|
| Mean reasoning entropy | 0.703 | [0.644, 0.757] | 3,550 | fixed |
| Endpoint answer-choice entropy | 0.680 | [0.624, 0.734] | 3,468 | fixed |
| Verbalized confidence (natural) | 0.629 | [0.580, 0.677] | 3,542 | repaired |
| Tail reasoning entropy | 0.584 | [0.533, 0.631] | 3,550 | fixed |
| Stabilization | 0.561 | [0.492, 0.632] | 3,467 | fixed |
| Answer switching | 0.505 | [0.461, 0.549] | 3,550 | fixed |

**Descriptive ordering (point estimate):** mean entropy > endpoint answer entropy >
verbalized confidence > tail entropy > stabilization > switching (≈ chance).

**Paired AUROC-difference tests** (question-cluster bootstrap, common cohort
**n=3,459**), mean entropy − X:
- vs endpoint answer-choice entropy **+0.027 [−0.064, +0.118]** → **not distinguishable**;
- vs verbalized confidence **+0.076 [+0.027, +0.128]** → mean entropy higher;
- vs answer switching **+0.201 [+0.125, +0.273]** → mean entropy higher;
- vs stabilization **+0.147 [+0.049, +0.242]** → mean entropy higher.

Only these paired comparisons support difference claims; do not infer pairwise
differences from the overlap of the individual CIs in the table.

### Figure 4B — calibration (endpoint, single common cohort)

Figure 4B compares two probability-like quantities elicited by the **same** forced
endpoint (fraction-1.0) probe, referring to the **same** selected endpoint answer,
on **one** common cohort with **one** target — an apples-to-apples calibration
comparison (distinct from Figure 4A's discrimination measure; see terminology note).

- *What is calibrated:* the **selected-answer probability** p(forced endpoint answer) and the repaired **endpoint checkpoint verbalized confidence** / 100.
- *Target:* **natural final-answer correctness** — the paper's reliability definition, the same target as Figure 4A. Used for both curves.
- *Common cohort:* **n = 3,466** (3,100 correct / 366 incorrect; accuracy 0.894) — trajectories with an evaluable natural answer *and* both endpoint measures available (one fewer than the 3,467 comparable rows; one lacks recovered endpoint confidence).
- *Endpoint answer identity:* on this exact cohort the endpoint forced answer matched the natural final answer in **3,466/3,466 (100%)**, so both endpoint quantities are valid confidence readouts for the very answer whose natural correctness is being scored.
- *Selected-answer probability = max A–D probability here:* over all endpoint rows the forced answer equals the argmax choice in 4,217/4,220; the 3 exceptions are exact ties, so p(selected) = max A–D probability (max difference 0.0).
- *ECE (same cohort, target, and 10-bin binning):* selected-answer probability **0.100** (mean p 0.994); endpoint checkpoint confidence **0.044** (mean conf 0.936).
- *Reliability diagram (points sized by bin count, no interpolation):* both concentrate mass in the top bins and sit below the diagonal there (overconfident at high stated values); the selected-answer probability is markedly more overconfident (mean ≈ 0.99 vs accuracy ≈ 0.89).
- *Why lower ECE ≠ better signal:* endpoint checkpoint confidence has lower ECE but a compressed range, so its smaller calibration error partly reflects that it rarely leaves the high-confidence region; it is not necessarily a better reliability signal overall.

**Terminology — three roles of verbalized confidence (name explicitly, never "final confidence"):**
- **checkpoint verbalized confidence** — forced checkpoint probe at each fraction; used for the RQ1 Figure 2 discrimination series (endpoint AUROC ≈ 0.667).
- **natural-terminal verbalized confidence** — extracted from the natural response; used for RQ3 Figure 4A discrimination (AUROC 0.629).
- **endpoint checkpoint verbalized confidence** — the fraction-1.0 checkpoint probe; used for RQ3 Figure 4B calibration (ECE 0.044).
Figures 4A and 4B therefore use **different** confidence measurements: 4A scores the model's own natural-response confidence for discrimination, while 4B calibrates the endpoint-probe confidence against the same endpoint answer as the selected-answer probability.

**RQ3 takeaway.** Mean reasoning entropy has the highest discrimination point
estimate and is supported by paired tests as higher than natural-terminal confidence
and both commitment measures, though its advantage over endpoint answer-choice
entropy is **not** statistically resolved; at the endpoint, both probability-like
quantities are overconfident against natural correctness (selected-answer probability
ECE 0.100, endpoint checkpoint confidence 0.044 on the common n=3,466 cohort), so
discrimination and calibration are distinct axes.

---

## 6. Main scientific findings

- **Signals evolve on different schedules:** reasoning-token entropy climbs then plateaus, answer-choice entropy stays high then collapses at commitment, and verbalized confidence stays high and compressed — they are not interchangeable.
- **Reasoning entropy is informative early:** prefix entropy over the first ~20–30% of reasoning reaches the highest discrimination point estimate (≈ 0.77) and remains strongest by point estimate through the endpoint (0.70).
- **Reliability is largely between-question with a small run-specific residual:** the pooled ≈ 0.13-nat correct/incorrect entropy gap attenuates to a small, heterogeneous within-question difference (+0.0113 nats [0.0002, 0.0225]).
- **Commitment measures are weak rankers:** switching (0.505) and stabilization (0.561) are near or below the other signals and show no within-question difference, despite rich descriptive dynamics (early appearance, mid-trajectory stabilization, frequent recovery).
- **Discrimination ≠ calibration:** the two endpoint probability-like signals discriminate to some degree yet are overconfident against natural correctness (selected-answer probability ECE 0.100, endpoint checkpoint confidence 0.044, common n=3,466).
- **Verbalized confidence is recoverable but compressed:** after the parser repair it is usable for 3,542 evaluable runs, but it sits near the ceiling (mean ≈ 93.5), which limits both its dynamic range and its practical value.

---

## 7. Important limitations and interpretation boundaries

- **Evaluability/selection:** primary results describe the 71% with an evaluable answer; unavailable runs are longer and disproportionately from Mathematics/Chemistry — not a representative sample of all generations.
- **Uneven subject coverage:** evaluable fraction 57.8%–88.0%; pooled results lose the intended subject balance → always report subject-macro alongside pooled.
- **Only 84 mixed-outcome questions for RQ2:** small, subject-skewed (Math only 6), incorrect side often a single run; the within-question effect is small and its CI barely excludes zero.
- **Endpoint conditional continuation:** at f=1.0 the forced answer matched the natural answer in all 3,467 comparable cases; treat f=1.0 answer-derived probes as endpoint confidence readouts, not independent predictions (does not affect prefix entropy).
- **Two clocks:** Figure 1A (and the entropy profile) use natural-trajectory token progress; answer-choice entropy and confidence use the forced-probe checkpoint clock — do not conflate them.
- **Repaired confidence provenance and three roles:** verbalized confidence is a deterministic parser repair of a pre-specified signal; document the repair and its compressed range. Name the measurement explicitly every time: **checkpoint** confidence (RQ1 Fig 2 discrimination series), **natural-terminal** confidence (RQ3 Fig 4A discrimination), and **endpoint checkpoint** confidence (RQ3 Fig 4B calibration). Never call any of them simply "final confidence."
- **Prefix reasoning-entropy provenance:** treated as the intended RQ1 checkpoint-level analysis that the config under-captured; the frozen design text declined a windowed reasoning-entropy statistic. Methods/Results wording still needs reconciliation (does not affect the numbers, which reproduce the fixed anchors).
- **Descriptive vs formal evidence:** point-estimate orderings (Fig 2, Fig 4A) are descriptive; only the paired bootstrap differences (Fig 4A paired tests, the within-question CI) support difference/effect claims.

---

## 8. Paper-ready LaTeX outline

```latex
\section{Results}

\paragraph{Data accounting and overall performance.}
% 5,000 generation attempts; 3,550 (71.0%) evaluable A–D (3,172 correct / 378 incorrect).
% Evaluability 57.8%–88.0% by subject; accuracy conditional on evaluability (pooled 89.4%).
% Report pooled + subject-macro; RQ2 uses the 84 mixed-outcome questions. -> Table 1.

\subsection{RQ1: Timing and Commitment}
% Fig 1: reasoning-token entropy rises (~20–30%) then plateaus ~0.6 nats;
%        answer-choice entropy high early then collapses at commitment;
%        verbalized confidence high+compressed (~90→94). Signals not in lockstep.
% Fig 2: prefix reasoning entropy highest AUROC point estimate ~0.77 at 20–30%,
%        declines to 0.703 at endpoint (= fixed mean-entropy AUROC; f=1.0 reproduces
%        stored mean to 1e-16); answer-choice entropy modest early (~0.60) improving late;
%        confidence near chance early, ~0.667 at endpoint. Endpoint answer-derived probes
%        = endpoint confidence readout (3,467/3,467 match), not independent; prefix exempt.
% Commitment: first appearance mean 0.268; stabilization mean 0.496; 566/634 recover;
%        commitment weak as a correctness ranker (detail in RQ3 / appendix).

\subsection{RQ2: Within-Question Reliability}
% Fig 3A: pooled correct<incorrect entropy in every decile, gap ~0.13 nats (confounded by difficulty).
% Fig 3B: within 84 mixed questions, incorrect−correct entropy +0.0113 nats [0.0002, 0.0225],
%         48/84 in expected direction -> small, heterogeneous; pooled gap substantially
%         attenuated within question (suggests large between-question component, not a decomposition).
% Commitment nulls: within-question switching −0.092 [−0.314, 0.129], stabilization −0.036 [−0.128, 0.059].

\subsection{RQ3: Signal Comparison}
% Fig 4A: AUROC — mean entropy 0.703 > endpoint answer entropy 0.680 > confidence 0.629 (repaired)
%         > tail 0.584 > stabilization 0.561 > switching 0.505. Paired (n=3,459): mean entropy
%         > confidence/switching/stabilization; vs endpoint answer entropy not resolved (+0.027 [−0.064,0.118]).
% Fig 4B: calibration on ONE common endpoint cohort (n=3,466), target = NATURAL final-answer correctness
%         (endpoint forced answer == natural answer on 3,466/3,466). p(selected)=max-P (4,217/4,220 argmax; ties).
%         Selected-answer prob ECE 0.100 vs endpoint checkpoint confidence ECE 0.044; both overconfident
%         at high values; lower confidence ECE offset by compressed range. Discrimination != calibration.
%         NB: 4A confidence = natural-terminal (discrimination); 4B confidence = endpoint checkpoint (calibration).
```

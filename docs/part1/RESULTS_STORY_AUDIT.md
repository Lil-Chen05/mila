# Results Story Audit: final 5,000-run Part 1 analysis

## Scope, authority, and notation

This audit treats `results/part1/6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c/analysis/final-r5000` as the source of truth for the fixed numerical findings. The validated merged Parquet files and hash-verified raw shards are used only for explicitly labelled post-hoc questions that the fixed analysis did not export. The report outline and `METHODS_EXPERIMENTAL_DESIGN.md`, `docs/part1/DECISIONS.md`, and `docs/part1/SCHEMA.md` are used only for definitions and intended framing. I inspected implementation only to resolve the fraction-0.0 construction, endpoint-prefix construction, terminal parser, confidence prompt, and fixed bootstrap design.

All fixed confidence intervals below are the published 95% percentile intervals from 5,000 subject-stratified, question-level bootstrap replicates with seed 42. Values are rounded for readability; the CSV files retain full precision. “Post-hoc” marks every calculation not already in the fixed outputs. No model or dataset inference was rerun, and no canonical result was changed.

The most important audit qualification is that the fixed `final-r5000` export does **not** contain every analysis contemplated by the report outline:

- It contains all-eleven-fraction AUROCs only for **checkpoint-local correctness**, not for `natural_correct`.
- It retains only the selected 0.0, 0.5, and 1.0 checkpoint predictors against `natural_correct`.
- The complete local production package does contain reasoning-token counts, aligned token-level entropy traces, decoded outputs, and checkpoint prefixes. Analyses recovered from those records below are post-hoc and do not become preregistered merely because the underlying records are canonical.

Those are output-coverage limitations, not null scientific findings.

## 1. Final cohort reconstruction

### Cohort flow

| Stage | Runs | Percent of 5,000 | Interpretation |
|---|---:|---:|---|
| Natural runs attempted and completed | 5,000 | 100.0% | Zero terminal infrastructure failures; every run remained checkpoint-eligible. |
| Reasoning closed | 4,808 | 96.16% | A terminal answer block could be sought. |
| Reasoning close missing | 192 | 3.84% | Natural answer/correctness are unavailable by contract. |
| Closed + parsed A–D natural answer | 3,550 | 71.00% | Primary correctness cohort. |
| Naturally correct | 3,172 | 63.44% of all; 89.35% of evaluable | Positive target class. |
| Naturally incorrect | 378 | 7.56% of all; 10.65% of evaluable | Negative target class. |
| Natural correctness unavailable | 1,450 | 29.00% | 192 missing close + 424 closed/missing answer + 834 closed/out-of-domain answer. |

The user’s starting counts—5,000 completed, 3,550 evaluable, 3,172 correct, 378 incorrect, and 1,450 unavailable—are all verified.

### Non-overlapping reasons for the 1,450 unavailable labels

| Subject | Missing reasoning close | Closed, answer missing | Closed, answer out of domain | Malformed answer | Other | Total unavailable | Percent of subject |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mathematics | 38 | 291 | 73 | 0 | 0 | 402 | 40.2% |
| Physics | 50 | 47 | 256 | 0 | 0 | 353 | 35.3% |
| Chemistry | 66 | 86 | 270 | 0 | 0 | 422 | 42.2% |
| Biology | 23 | 0 | 130 | 0 | 0 | 153 | 15.3% |
| Psychology | 15 | 0 | 105 | 0 | 0 | 120 | 12.0% |
| **Total** | **192** | **424** | **834** | **0** | **0** | **1,450** | **29.0%** |

The raw `answer_parse_status` totals are 616 `missing` and 834 `out_of_domain`. The 616 missing answers comprise the 192 missing-close runs plus 424 closed runs with no terminal answer. Missingness differs substantially by subject: chemistry and mathematics lose roughly two-fifths of runs, whereas biology and psychology lose about one-sixth and one-eighth. This makes the report’s balanced 1,000-runs-per-subject design an unbalanced **evaluable** analysis.

### Question-level accounting

| Question status | Questions |
|---|---:|
| All 10 runs evaluable | 176 |
| 1–9 runs evaluable | 296 |
| No evaluable runs | 28 |
| At least one evaluable run | 472 |
| All evaluable runs correct | 360 |
| All evaluable runs incorrect | 28 |
| Mixed correct and incorrect evaluable runs | 84 |

The number of evaluable runs per question has the exact distribution `0:28, 1:15, 2:15, 3:22, 4:28, 5:37, 6:40, 7:34, 8:41, 9:64, 10:176`.

There are exactly 84 mixed-outcome questions. All 84 qualify for the mean-entropy, tail-entropy, and switch-count within-question analyses; feature-specific cohorts fall to 66 or 77 when checkpoint predictors are missing. The 84-question base cohort contains 427 correct and 212 incorrect runs. The exact correct/incorrect run-count combinations are:

`1/1×2, 1/2×1, 1/4×2, 1/6×1, 1/7×1, 1/8×1, 1/9×2; 2/1×3, 2/2×1, 2/4×2, 2/5×1, 2/6×1, 2/8×1; 3/1×4, 3/2×4, 3/3×2, 3/7×2; 4/1×4, 4/2×2, 4/3×1, 4/4×1, 4/5×1, 4/6×2; 5/1×3, 5/2×1, 5/3×1, 5/4×1, 5/5×1; 6/1×1, 6/2×2, 6/4×1; 7/2×4; 8/1×7, 8/2×4; 9/1×16`.

The cohort is therefore often imbalanced within question: 16 questions are 9-correct/1-incorrect and seven are 8-correct/1-incorrect.

### Does missingness select unusually uncertain runs?

No. The observed pattern is the opposite of the concern in the prompt.

| Cohort | n | Mean mean-entropy | Median | SD | IQR |
|---|---:|---:|---:|---:|---:|
| Evaluable natural answer | 3,550 | 0.5920 | 0.6079 | 0.1772 | 0.2675 |
| Nonevaluable natural answer | 1,450 | 0.5283 | 0.5481 | 0.1753 | 0.2552 |

The 1,450 nonevaluable runs have **lower**, not higher, mean reasoning entropy. Status-specific post-hoc means help explain the mixture: closed/parsed 0.5920; closed/out-of-domain 0.6024; closed/missing-answer 0.4466; missing-close 0.3868. Therefore, the correctness analyses are selected on model-output validity, but the selection is not toward visibly high-entropy runs. It may instead omit many low-entropy, abnormal terminal behaviours. This remains selection bias: the primary AUROCs describe the 71% with evaluable A–D answers, not all completed generations.

Reasoning length is absent from `trajectory_features.csv` but present for every run in the validated raw/merged records. Post-hoc, outcome availability is strongly length-associated:

| Natural outcome status | n | Mean reasoning tokens | Median | 5th–95th percentiles |
|---|---:|---:|---:|---:|
| Correct | 3,172 | 735.9 | 484.5 | 253.6–2,161.5 |
| Incorrect | 378 | 1,270.6 | 789 | 377.3–4,374.1 |
| Correctness unavailable | 1,450 | 2,596.6 | 1,380 | 325.5–8,191 |

Across deciles of all-run length, correctness is observable for 451/494 runs (91.3%) in the shortest decile but only 109/500 (21.8%) in the longest. The earlier entropy-only comparison therefore understates an important selection mechanism: nonevaluable runs have lower mean entropy on average as a heterogeneous group, yet they are much longer. Outcome-validity selection is jointly related to terminal behaviour, length, and entropy; it is not a simple “high uncertainty is missing” pattern.

## 2. Main reasoning-entropy finding

### Fixed AUROCs

Negative mean reasoning entropy (larger means lower entropy and therefore predicted correctness) has:

- pooled AUROC **0.7025 [0.6441, 0.7574]**, n=3,550 (3,172 correct, 378 incorrect);
- subject-macro AUROC **0.7191 [0.6628, 0.7727]**.

| Subject | Correct | Incorrect | AUROC | 95% CI | Direction |
|---|---:|---:|---:|---:|---|
| Mathematics | 588 | 10 | 0.779 | [0.632, 0.931] | Lower entropy in correct runs |
| Physics | 549 | 98 | 0.709 | [0.595, 0.805] | Same |
| Chemistry | 503 | 75 | 0.627 | [0.485, 0.769] | Same point direction; CI includes 0.5 |
| Biology | 747 | 100 | 0.701 | [0.569, 0.807] | Same |
| Psychology | 785 | 95 | 0.780 | [0.698, 0.857] | Same |

No subject point estimate is below 0.5. Psychology is strongest by a narrow point-estimate margin; chemistry is weakest. Mathematics has only 10 incorrect runs, so its apparently strong AUROC is imprecise and should not be compared confidently with the others.

### Group distributions

| Natural outcome | n | Mean | Median | SD | Q1 | Q3 | IQR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Correct | 3,172 | 0.5783 | 0.5960 | 0.1753 | 0.4391 | 0.7151 | 0.2761 |
| Incorrect | 378 | 0.7075 | 0.7107 | 0.1492 | 0.6055 | 0.8200 | 0.2146 |

The raw incorrect-minus-correct mean difference is **0.12924 nats**. A post-hoc pooled standardized descriptive difference is **Cohen’s d≈0.75**; this is descriptive and ignores clustering.

The AUROC is not produced only by a few extreme runs. Incorrect runs are much more common above the overall upper quartile (46.8% versus 22.4%) and much less common below the lower quartile (6.3% versus 27.2%). After trimming the outer 1%, 2.5%, and 5% of the pooled entropy distribution, post-hoc AUROCs remain 0.690, 0.678, and 0.653. Extremes contribute, but a broad distributional shift remains.

Subject adjustment does not materially remove the association: post-hoc pooled AUROC after within-subject z-standardization is 0.710 (within-subject ranks: 0.707), close to the fixed pooled 0.703 and macro 0.719.

### Post-hoc length sensitivity

Reasoning length is a strong competing signal. Among the 3,550 evaluable runs, longer-oriented log length has AUROC **0.294 [0.249,0.341]**; equivalently, inverse log length (shorter predicts correctness) has AUROC **0.706 [0.659,0.751]** under a 2,000-draw question-cluster bootstrap. Across all 5,000 runs, raw mean entropy is negatively correlated with log length (r=−0.245 [−0.301,−0.188]); correctness-oriented negative entropy has the opposite sign. Length is therefore materially associated with entropy and outcome, although not in the simple direction “longer means more entropy.”

A simple post-hoc residualization of correctness-oriented mean entropy on log length and subject indicators retains AUROC **0.665 [0.609,0.720]**. An additive logistic model including standardized correctness-oriented entropy, standardized log length, and subject indicators gives a positive entropy coefficient (+1.084), a negative length coefficient (−0.813), and apparent in-sample AUROC **0.821 [0.792,0.858]** across cluster-bootstrap refits. This is not cross-validated and is not a new primary performance claim; it only shows that entropy retains association after a simple adjustment while length supplies additional information.

Broad length-quintile AUROCs for entropy remain above chance (0.896, 0.728, 0.737, 0.776, 0.723), but the shortest stratum has only ten errors and within-question, within-length cells are sparse. These strata support a descriptive association among comparably long runs; they do not establish a robust conditional trajectory effect.

### Mean versus tail entropy

Negative tail entropy is much weaker: pooled AUROC **0.5840 [0.5327, 0.6305]**, but macro AUROC **0.5369 [0.4810, 0.5893]**; every subject-specific interval includes 0.5. Correct and incorrect tail-entropy means are 0.6370 and 0.6982, a smaller 0.0611-nat gap.

The defensible fixed interpretation is that **average uncertainty over the recognized reasoning span has higher pooled and macro AUROC point estimates than uncertainty in the final 10% alone**. No paired AUROC-difference test between mean and tail entropy was reported. The raw-trace profile in Section 13 adds post-hoc evidence of lower pooled entropy in correct runs across all ten normalized bins, with attenuation late. Thus “sustained descriptive difference” is supported for the selected evaluable cohort; a sustained within-question mechanism is not.

## 3. Complete fixed primary signal comparison

| Correctness-oriented predictor | n | Predictor missing within 3,550 | Pooled AUROC [95% CI] | Macro AUROC [95% CI] |
|---|---:|---:|---:|---:|
| − mean reasoning entropy | 3,550 | 0 | **0.7025 [0.6441, 0.7574]** | **0.7191 [0.6628, 0.7727]** |
| − tail reasoning entropy | 3,550 | 0 | 0.5840 [0.5327, 0.6305] | 0.5369 [0.4810, 0.5893] |
| − answer-choice entropy, 0.0 | 3,502 | 48 | 0.6055 [0.5476, 0.6599] | 0.6280 [0.5661, 0.6934] |
| − answer-choice entropy, 0.5 | 3,148 | 402 | 0.6015 [0.5445, 0.6571] | 0.6219 [0.5514, 0.6892] |
| − answer-choice entropy, 1.0 | 3,468 | 82 | 0.6799 [0.6237, 0.7341] | 0.7200 [0.6440, 0.7855] |
| Natural verbalized confidence | 1 | 3,549 | Undefined; single target class | Undefined |
| Maximum A–D probability, 0.0 | 3,502 | 48 | 0.6109 [0.5508, 0.6690] | 0.6391 [0.5734, 0.7036] |
| Maximum A–D probability, 0.5 | 3,148 | 402 | 0.5904 [0.5354, 0.6441] | 0.6055 [0.5390, 0.6687] |
| Maximum A–D probability, 1.0 | 3,468 | 82 | 0.6786 [0.6226, 0.7332] | 0.7188 [0.6431, 0.7843] |
| − answer-switch count | 3,550 | 0 | 0.5048 [0.4612, 0.5489] | 0.5408 [0.4999, 0.5839] |
| − stabilization fraction | 3,467 | 83 | 0.5605 [0.4916, 0.6316] | 0.5620 [0.4985, 0.6313] |

Mean entropy, all three selected answer entropies, and all three selected maximum probabilities have pooled intervals above 0.5. Tail entropy only clears chance in the pooled analysis; its macro and subject evidence are weak. Switching and stabilization overlap chance. Confidence is not estimable.

Coverage differs, especially at 0.5 and for stabilization. A post-hoc common-complete-case comparison over mean entropy, endpoint answer entropy, endpoint maximum probability, switching, and stabilization uses exactly **3,467 runs** (3,101 correct, 366 incorrect; 461 questions):

| Predictor | Common-case AUROC |
|---|---:|
| − mean entropy | 0.7067 |
| − endpoint answer entropy | 0.6800 |
| Endpoint maximum probability | 0.6787 |
| − switches | 0.5042 |
| − stabilization | 0.5605 |

Post-hoc question-cluster bootstrap AUROC differences on that same cohort are:

| Difference | Point | 95% CI |
|---|---:|---:|
| Mean entropy − endpoint answer entropy | 0.0268 | [−0.0665, 0.1174] |
| Mean entropy − endpoint max probability | 0.0280 | [−0.0648, 0.1191] |
| Mean entropy − switching | 0.2025 | [0.1252, 0.2761] |
| Mean entropy − stabilization | 0.1462 | [0.0448, 0.2422] |

Thus mean entropy clearly exceeds the two commitment summaries in this post-hoc comparison, but it is **not statistically distinguishable** from the endpoint answer-distribution measures. A larger point estimate alone is not evidence of superiority.

### Subject-specific primary AUROCs

| Predictor | Math | Physics | Chemistry | Biology | Psychology |
|---|---:|---:|---:|---:|---:|
| − mean entropy | .779 [.632,.931] | .709 [.595,.805] | .627 [.485,.769] | .701 [.569,.807] | .780 [.698,.857] |
| − tail entropy | .465 [.294,.667] | .563 [.463,.649] | .554 [.434,.653] | .535 [.428,.630] | .568 [.478,.653] |
| − answer entropy 0.0 | .502 [.307,.723] | .559 [.428,.687] | .645 [.499,.795] | .702 [.588,.805] | .731 [.624,.835] |
| − answer entropy 0.5 | .562 [.314,.802] | .552 [.429,.676] | .629 [.481,.752] | .632 [.523,.742] | .735 [.617,.845] |
| − answer entropy 1.0 | .759 [.436,.975] | .644 [.542,.749] | .719 [.603,.824] | .703 [.593,.814] | .774 [.689,.857] |
| Max P 0.0 | .530 [.326,.738] | .599 [.462,.721] | .650 [.502,.798] | .705 [.591,.807] | .711 [.608,.812] |
| Max P 0.5 | .542 [.312,.774] | .530 [.415,.646] | .602 [.462,.726] | .629 [.524,.733] | .724 [.611,.832] |
| Max P 1.0 | .758 [.435,.974] | .642 [.540,.746] | .717 [.601,.823] | .703 [.594,.814] | .774 [.689,.857] |
| − switches | .643 [.536,.758] | .402 [.331,.478] | .474 [.383,.563] | .593 [.506,.692] | .592 [.507,.703] |
| − stabilization | .516 [.380,.684] | .514 [.369,.684] | .614 [.449,.755] | .556 [.426,.695] | .610 [.477,.734] |

Exact subject n values in the same column order (math, physics, chemistry, biology, psychology) are: mean/tail/switch `598/647/578/847/880`; fraction-0.0 entropy/max P `591/631/566/834/880`; fraction-0.5 entropy/max P `552/533/434/773/856`; fraction-1.0 entropy/max P `595/623/545/829/876`; stabilization `595/623/545/829/875`; natural confidence `0/0/1/0/0`.

The physics switch result is in the opposite direction: fewer switches rank correctness worse (AUROC 0.402). This reinforces that the pooled null is not hiding one uniformly weak positive effect.

## 4. Checkpoint predictors against natural final correctness

### Fixed-output mismatch and post-hoc recovery

The requested all-eleven-fraction trajectory table against `natural_correct` is absent from `final-r5000`: its fixed primary registry retains only 0.0, 0.5, and 1.0, while `secondary_checkpoint_auroc.csv` targets `checkpoint_local_correct`. The validated raw merge permits the missing curve to be reconstructed without rerunning inference. The following tables are therefore **post-hoc** except that the 0.0/0.5/1.0 entropy and max-P rows exactly reproduce the published fixed point estimates and intervals. Every row begins with 3,550 known natural targets; “missing” is predictor missingness within that cohort, in addition to the 1,450 runs whose natural target is unavailable. Intervals use 5,000 seed-42 subject-stratified question bootstraps.

#### Answer-choice entropy → natural correctness

AUROC uses negative entropy; group summaries show raw entropy in nats.

| f | n (correct/incorrect) | Predictor missing | AUROC [95% CI] | Correct mean/median | Incorrect mean/median |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 3,502 (3,124/378) | 48 | .6055 [.5476,.6599] | .5919/.6074 | .7484/.7833 |
| 0.1 | 3,380 (3,026/354) | 170 | .6109 [.5518,.6662] | .6708/.7217 | .8499/.9266 |
| 0.2 | 3,294 (2,950/344) | 256 | .6005 [.5447,.6539] | .6685/.7298 | .8306/.9064 |
| 0.3 | 3,198 (2,869/329) | 352 | .6022 [.5467,.6564] | .6475/.6980 | .8085/.8787 |
| 0.4 | 3,181 (2,855/326) | 369 | .5962 [.5402,.6495] | .5989/.6058 | .7396/.7868 |
| 0.5 | 3,148 (2,824/324) | 402 | .6015 [.5445,.6571] | .5443/.4812 | .6824/.7135 |
| 0.6 | 3,154 (2,837/317) | 396 | .6177 [.5634,.6700] | .4664/.3410 | .6083/.6056 |
| 0.7 | 3,160 (2,841/319) | 390 | .6523 [.5994,.7012] | .3659/.1892 | .5309/.4995 |
| 0.8 | 3,238 (2,910/328) | 312 | .6831 [.6305,.7315] | .2509/.1060 | .4267/.3648 |
| 0.9 | 3,326 (2,983/343) | 224 | **.6859 [.6350,.7358]** | .1577/.0477 | .2891/.1665 |
| 1.0 | 3,468 (3,102/366) | 82 | .6799 [.6237,.7341] | .02745/.00652 | .06074/.01868 |

#### Maximum A–D probability → natural correctness

| f | n (correct/incorrect) | Predictor missing | AUROC [95% CI] | Correct mean/median | Incorrect mean/median |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 3,502 (3,124/378) | 48 | .6109 [.5508,.6690] | .7649/.8154 | .6983/.7062 |
| 0.1 | 3,380 (3,026/354) | 170 | .5976 [.5404,.6517] | .7296/.7514 | .6623/.6555 |
| 0.2 | 3,294 (2,950/344) | 256 | .5859 [.5321,.6373] | .7322/.7533 | .6768/.6710 |
| 0.3 | 3,198 (2,869/329) | 352 | .5909 [.5357,.6431] | .7403/.7663 | .6809/.6622 |
| 0.4 | 3,181 (2,855/326) | 369 | .5845 [.5303,.6358] | .7627/.8113 | .7174/.7144 |
| 0.5 | 3,148 (2,824/324) | 402 | .5904 [.5354,.6441] | .7873/.8664 | .7423/.7484 |
| 0.6 | 3,154 (2,837/317) | 396 | .6070 [.5549,.6582] | .8215/.9189 | .7808/.8139 |
| 0.7 | 3,160 (2,841/319) | 390 | .6476 [.5947,.6959] | .8657/.9614 | .8093/.8574 |
| 0.8 | 3,238 (2,910/328) | 312 | .6789 [.6262,.7271] | .9144/.9817 | .8551/.9087 |
| 0.9 | 3,326 (2,983/343) | 224 | **.6840 [.6330,.7341]** | .9510/.9928 | .9058/.9681 |
| 1.0 | 3,468 (3,102/366) | 82 | .6786 [.6226,.7332] | .9944/.9993 | .9867/.9976 |

Both natural-target probability curves are shallow and non-monotonic through 0.6, rise from 0.7 to a point-estimate peak at 0.9, and decline slightly at the endpoint. The all-fraction story therefore differs from the checkpoint-local curve, which peaks at 0.7. Endpoint entropy and max probability remain discriminative even though both outcome groups are compressed near complete certainty.

#### Parsed checkpoint confidence → natural correctness

| f | n (correct/incorrect) | Predictor missing | AUROC [95% CI] |
|---:|---:|---:|---:|
| 0.0 | 2,492 (2,232/260) | 1,058 | .5647 [.4733,.6499] |
| 0.1 | 722 (646/76) | 2,828 | .5683 [.4942,.6411] |
| 0.2 | 290 (254/36) | 3,260 | .5642 [.4453,.6976] |
| 0.3 | 159 (131/28) | 3,391 | .5852 [.4605,.7063] |
| 0.4 | 115 (96/19) | 3,435 | .4712 [.3433,.6480] |
| 0.5 | 83 (68/15) | 3,467 | .5083 [.3373,.7031] |
| 0.6 | 57 (46/11) | 3,493 | .5830 [.3889,.7683] |
| 0.7 | 48 (41/7) | 3,502 | .3641 [.1444,.6124] |
| 0.8 | 32 (23/9) | 3,518 | .5821 [.3401,.7890] |
| 0.9 | 11 (7/4) | 3,539 | .5000 [.1250,.8750] |
| 1.0 | 0 | 3,550 | Undefined |

None of the checkpoint-confidence intervals excludes 0.5, coverage collapses rapidly, and the apparent estimates should not be interpreted as a trajectory curve. A stringent shared cohort requiring finite entropy and max P at all 11 fractions retains 2,635 runs (2,425 correct, 210 incorrect; 308 questions). In that cohort both signals still peak at 0.9 (entropy/max-P AUROCs .739/.739), showing that the late rise is not solely changing row coverage.

### Fraction 0.0 is a question baseline with date drift

The protocol retains zero reasoning tokens at 0.0 and uses greedy probing, so the scientific interpretation is a **pre-reasoning question baseline**, not a run-varying trajectory signal. Direct prefix hashes show that 475/500 questions have one identical prefix across all ten runs. The remaining 25 each split into two blocks whose rendered prompts differ only in `Today Date` (`13→14 August` or `14→15 August 2026`), yielding 525 distinct question-prefix instances. This explains the previously unexplained within-question feature anomalies and can alter a forced answer (sample 234 changes C→A). It is prompt drift, not trajectory-specific reasoning.

The inferential bootstrap correctly resamples questions with all their runs, so it preserves the dominant clustering. The date deviation should nevertheless be disclosed as a provenance limitation; fraction 0.0 is neither 5,000 independent observations nor exactly 500 immutable prompts.

Post-hoc question-level associations among questions with evaluable natural runs and a 0.0 signal are modest: maximum probability versus question accuracy Pearson r=0.117 (Spearman 0.168; n=459), and negative answer entropy r=0.135 (Spearman 0.157). At run level their fixed AUROCs are 0.611 and 0.606, versus 0.703 for mean reasoning entropy.

This supports a qualified statement: some reliability information is visible before explicit reasoning, and mean reasoning entropy contains additional information. The within-question entropy result strengthens the run-specific part of that statement, but the small conditional effect, length confounding, and date drift prevent a strong causal claim.

## 5. Checkpoint-local correctness curve

This is the complete fixed secondary curve. `C max P` and `I max P` are mean/median maximum A–D probability for locally correct and locally incorrect checkpoint answers.

| Fraction | Evaluable local answers (C/I) | Accuracy | Max-P n | Max-P AUROC [95% CI] | C max P | I max P | Mean max P | ECE [95% CI] |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 2,725 (1,306/1,419) | 0.479 | 2,725 | 0.700 [0.635,0.763] | .757/.863 | .605/.594 | .678 | .206 [.162,.263] |
| 0.1 | 3,767 (2,135/1,632) | 0.567 | 3,763 | 0.758 [0.723,0.791] | .776/.845 | .580/.551 | .691 | .125 [.092,.164] |
| 0.2 | 3,610 (2,085/1,525) | 0.578 | 3,608 | 0.757 [0.721,0.792] | .778/.843 | .588/.557 | .698 | .121 [.087,.159] |
| 0.3 | 3,540 (2,170/1,370) | 0.613 | 3,539 | 0.760 [0.727,0.794] | .787/.868 | .595/.559 | .713 | .102 [.070,.138] |
| 0.4 | 3,598 (2,338/1,260) | 0.650 | 3,595 | 0.768 [0.734,0.800] | .813/.901 | .620/.595 | .745 | .101 [.069,.134] |
| 0.5 | 3,593 (2,463/1,130) | 0.685 | 3,588 | 0.785 [0.752,0.816] | .837/.928 | .633/.599 | .773 | .087 [.059,.119] |
| 0.6 | 3,639 (2,681/958) | 0.737 | 3,638 | 0.798 [0.765,0.831] | .864/.948 | .658/.632 | .810 | .073 [.048,.104] |
| 0.7 | 3,670 (2,939/731) | 0.801 | 3,667 | **0.805 [0.770,0.840]** | .897/.973 | .694/.676 | .857 | **.056 [.035,.083]** |
| 0.8 | 3,837 (3,258/579) | 0.849 | 3,836 | 0.784 [0.745,0.824] | .929/.983 | .763/.814 | .904 | .056 [.039,.081] |
| 0.9 | 4,023 (3,534/489) | 0.878 | 4,023 | 0.736 [0.692,0.783] | .956/.992 | .849/.941 | .943 | .068 [.051,.091] |
| 1.0 | 4,220 (3,746/474) | 0.888 | 4,220 | 0.679 [0.626,0.732] | .991/.999 | .965/.996 | .988 | .101 [.080,.126] |

The curve is genuinely non-monotonic. Local accuracy rises steadily, but rank discrimination peaks at 0.7 and declines as the probability distribution saturates. At 0.7, correct and incorrect means/medians remain well separated (.897/.973 versus .694/.676). At 1.0, both are near one (.991/.999 versus .965/.996). Among the 474 remaining endpoint errors, 431 (90.9%) have max probability ≥0.9 and 325 (68.6%) have ≥0.99. In the reliability bins, 4,109/4,220 endpoint rows occupy the 0.9–1.0 bin with mean probability 0.995 and accuracy 0.895.

The endpoint is predominantly **overconfident**, not mixed: mean max probability 0.988 versus observed accuracy 0.888. ECE rises from 0.056 at 0.7 to 0.101 at 1.0, a pattern consistent with increased endpoint overconfidence and a roughly ten-point endpoint probability–accuracy gap. The final package does not report a paired bootstrap interval for the ECE difference; ECE should also be described as a ten-bin summary, not a complete calibration characterization.

Late class imbalance contributes to uncertainty, while 474 endpoint errors remain and the point estimate declines substantially. A paired AUROC-difference analysis would be needed to quantify that decline formally. The subject curves are heterogeneous rather than sharing a universal 0.7 peak:

- mathematics peaks at 0.9 (0.929), endpoint 0.866;
- physics peaks at 0.8 (0.796), endpoint 0.645;
- chemistry peaks at 0.8 (0.808), endpoint 0.737;
- biology peaks at 0.4 (0.805), endpoint 0.708;
- psychology peaks at 0.1 (0.813), endpoint 0.775.

Every subject’s endpoint is below its own maximum, but the timing of the maximum ranges from 0.1 to 0.9. “Discrimination peaks around 70%” is therefore true only of the pooled curve.

The endpoint-local AUROC (0.679, n=4,220) and endpoint-to-natural AUROC (0.679, n=3,468) are numerically similar, consistent with all 3,467 evaluable answer comparisons agreeing. Their cohorts are nevertheless different: local correctness includes all valid checkpoint answers; the natural-target primary row requires natural correctness plus the probability predictor and includes one probability-measurable row without evaluable answer agreement.

## 6. Within-question reliability analysis

For each eligible question and correctness-oriented feature, the fixed estimand is:

`mean(feature among naturally correct runs) − mean(feature among naturally incorrect runs)`.

Each question contributes one equally weighted paired difference, regardless of the number of runs on either side. Positive values favour correct runs because every feature is oriented so larger predicts correctness.

| Feature | Questions | Mean paired difference | Median | 95% CI | Interpretation |
|---|---:|---:|---:|---:|---|
| − mean reasoning entropy | 84 | **+0.011280** | +0.010350 | **[+0.000184,+0.022470]** | Correct runs have slightly lower mean entropy |
| − tail reasoning entropy | 84 | −0.002840 | −0.005918 | [−0.029458,+0.023950] | Null |
| − answer entropy 0.0 | 84 | +0.000075 | 0 | [−0.000076,+0.000326] | Essentially no within-question variation |
| − answer entropy 0.5 | 66 | +0.026332 | −0.011577 | [−0.040482,+0.093431] | Null, heterogeneous |
| − answer entropy 1.0 | 77 | +0.007697 | +0.001047 | [−0.019594,+0.034585] | Null |
| Natural confidence | 0 | — | — | — | Unavailable |
| Max P 0.0 | 84 | +0.000080 | 0 | [−0.000036,+0.000297] | Essentially no within-question variation |
| Max P 0.5 | 66 | +0.013160 | +0.003423 | [−0.022592,+0.048159] | Null |
| Max P 1.0 | 77 | +0.000391 | −0.000018 | [−0.008359,+0.008780] | Null |
| − switch count | 84 | −0.092300 | −0.083333 | [−0.314261,+0.128969] | Null; point direction is more switching in correct runs |
| − stabilization fraction | 77 | −0.035919 | −0.077778 | [−0.128039,+0.058598] | Null |

For mean entropy, the per-question difference distribution has SD 0.0520, IQR [−0.0187,+0.0462], and range [−0.2020,+0.1254]. Forty-eight questions favour lower entropy in correct runs and 36 favour the opposite. A post-hoc exact sign test is not significant (positive proportion 57.1%, two-sided p=0.230). Thus the fixed bootstrap interval barely excludes zero because the average magnitude is positive, not because an overwhelming majority of questions share the direction.

Subject-specific equal-question means are mathematics +0.0170 (6 questions), physics +0.0019 (21), chemistry +0.0045 (21), biology +0.0144 (18), and psychology +0.0251 (18). Physics and chemistry are near zero; mathematics is too small for a stable subject conclusion. The full-sample result is heterogeneous and strongest descriptively in psychology and biology.

The finding is sensitive in magnitude to weighting, although not in direction for mean entropy. A post-hoc pooled run-weighted contrast over the mixed-question subset is +0.0389 instead of +0.0113. Run weighting also reverses the tail-entropy sign (from −0.0028 to +0.0137) and makes early checkpoint differences much larger. This is precisely why the prespecified equal-question estimand should remain primary.

The 95% interval [0.00018,0.02247] is statistically fragile in the ordinary sense that its lower limit is very close to zero, only 84/500 questions contribute, and a simple sign test is null. A fair characterization is: **within mixed-outcome questions, correct trajectories had a small lower average reasoning entropy on the prespecified equally weighted mean contrast; the evidence is modest and heterogeneous.** It is fair to add that no other candidate signal showed a comparably reliable within-question difference, provided “reliable” refers to these fixed intervals and not to proof of superiority between effects.

## 7. Answer emergence, switching, and stabilization

### Fixed descriptive distributions

| Quantity | Evaluable n | Mean | Median | SD | IQR |
|---|---:|---:|---:|---:|---:|
| Switch count | 5,000 | 0.6132 | 0 | 1.1156 | [0,1] |
| First natural-answer appearance, among found | 3,525 | 0.2685 | 0.1 | — | [0.0,0.5] |
| Stabilization fraction | 4,220 | 0.4955 | 0.5 | — | [0.1,0.8] |

Switch counts are 0 for 3,346 runs (66.92%), 1 for 926 (18.52%), 2 for 322 (6.44%), and 3+ for 406 (8.12%). Because missing checkpoint answers break adjacency, “zero switches” means zero **counted eligible adjacent changes**, not necessarily a fully observed constant sequence. A stricter post-hoc count finds 2,670 runs whose set of all valid checkpoint answers contains only one answer.

Among the 3,525 runs where the natural answer appears at a checkpoint, it has appeared by 0.0 in 1,065 (30.2%), by 0.2 in 2,274 (64.5%), by 0.5 in 2,729 (77.4%), by 0.8 in 3,259 (92.5%), and by 1.0 in all 3,525 (100%). It never appears in 25 otherwise eligible natural-answer runs; 1,450 are unavailable because the natural answer is unavailable.

Of the 4,220 runs with computable stabilization, 1,438 (34.1%) stabilize by 0.2, 2,122 (50.3%) by 0.5, and 3,294 (78.1%) by 0.8. On average 50.45% of normalized reasoning remains after stabilization (median 50%). Correct runs stabilize earlier than incorrect runs descriptively (means 0.461 versus 0.542; remaining fractions 0.539 versus 0.458), but the fixed correctness AUROC remains weak.

### Leaving and recovering

“Left a correct answer” means at least one valid adjacent checkpoint transition changed from a locally correct forced answer to a locally incorrect one. It occurred in **634/5,000** runs. Of these, **566/634 (89.3%)** later displayed a correct checkpoint answer and 68 did not.

Among the 634 leave events, the natural outcome is correct for 431, incorrect for 70, and unavailable for 133. Among the 566 recoveries, those counts are 431, 23, and 112. The 68 non-recoveries therefore contain 0 naturally correct, 47 naturally incorrect, and 21 unavailable runs. This striking association is mechanically entangled with the perfect endpoint agreement discussed below and should not be presented as independent prediction.

Post-hoc sequence counts make the reversibility concrete:

- 513 natural-answer sequences show natural answer → another valid answer → natural answer again;
- among 2,441 runs where the natural answer appears by 0.3, 423 (17.3%) later show a different valid answer;
- 226 runs contain an adjacent incorrect→correct→incorrect triple;
- 309 contain an adjacent correct→incorrect→correct triple.

If gaps are ignored and any valid chronological subsequence is allowed, the latter counts rise to 389 and 602. The adjacency-preserving counts are the cleaner result.

First appearance and stabilization are related but not identical (post-hoc Pearson r=0.609): early appearance tends to accompany early stabilization, yet the 17.3% later-departure rate shows that observing the eventual answer early does not guarantee stable commitment.

### Does commitment predict correctness?

Not convincingly. Negative switch count has pooled AUROC 0.5048 [0.4612,0.5489]; negative stabilization fraction has 0.5605 [0.4916,0.6316]. Correct and incorrect runs average 0.665 and 0.638 switches, respectively. Their switch-count accuracies are non-monotonic rather than steadily declining with more switches. Within-question differences for switching and stabilization both include zero.

Post-hoc first natural-answer appearance is earlier in correct runs (mean 0.260 versus 0.336; clustered difference −0.0756 [−0.1525,−0.0043]). Stabilization is also earlier descriptively (difference −0.0809 [−0.1656,+0.0002]), but it is defined relative to the forced endpoint and is borderline.

It is defensible to say that elicited answers often stabilize around halfway through the recognized reasoning span and that temporary departures from correct answers are often reversible. It is **not** defensible to infer that the remaining reasoning is unnecessary, or that stabilization directly observes an internal decision state.

## 8. Subject-level story

| Subject | Evaluable / missing (%) of 1,000 | Accuracy | Mixed Q | Mean entropy C / I | Entropy AUROC | Switch mean / median | First appearance mean | Stabilization mean | Local accuracy 0/.5/1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mathematics | 598 / 402 (40.2%) | 98.3% | 6 | .345 / .455 | .779 | 1.184 / 1 | .353 | .516 | .385 / .568 / .974 |
| Physics | 647 / 353 (35.3%) | 84.9% | 21 | .516 / .612 | .709 | .617 / 0 | .348 | .564 | .273 / .621 / .849 |
| Chemistry | 578 / 422 (42.2%) | 87.0% | 21 | .584 / .655 | .627 | .520 / 0 | .361 | .609 | .411 / .637 / .851 |
| Biology | 847 / 153 (15.3%) | 88.2% | 18 | .653 / .746 | .701 | .479 / 0 | .187 | .436 | .654 / .759 / .875 |
| Psychology | 880 / 120 (12.0%) | 89.2% | 18 | .722 / .833 | .780 | .266 / 0 | .172 | .394 | .787 / .830 / .879 |

Mathematics has 1.184 mean switches, nearly twice physics (0.617), 2.28× chemistry (0.520), 2.47× biology (0.479), and 4.45× psychology (0.266). This is robust as a descriptive difference in this dataset, but the final output contains no fixed interval for differences in subject means, and mathematics supplies only ten natural-answer errors. Do not offer a cognitive explanation.

Psychology stabilizes earliest and has the highest fraction-0.0 local accuracy (78.7%), followed by biology (65.4%). This supports task-difficulty/answerability as a plausible empirical explanation for some subject dynamics, not a causal account. Mathematics is an important counterexample: its valid natural-answer accuracy is extremely high despite low fraction-0.0 checkpoint accuracy and high switching, partly because its evaluable cohort is severely selected.

The 378 incorrect runs are contributed by mathematics 10 (2.6%), physics 98 (25.9%), chemistry 75 (19.8%), biology 100 (26.5%), and psychology 95 (25.1%). Biology and psychology together supply 48.7% of evaluable runs, whereas mathematics and chemistry together supply 33.1%. Pooled results therefore do not retain the intended equal subject balance. Subject-stratified question bootstraps and macro estimates are essential.

The entropy macro AUROC (0.719) exceeds the pooled AUROC (0.703) because cross-subject comparisons add discordant rankings when subjects have very different entropy scales and accuracies. The consistent within-subject point direction and similar post-hoc within-subject standardized AUROC (0.710) indicate a reasonably consistent association, while the chemistry interval and mathematics error count still demand caution. Subject heterogeneity belongs in an appendix table and in one concise main-text qualification; the full subject curves do not need a main figure.

## 9. Verbalized-confidence failure

### Natural confidence status

| Natural answer/confidence combination | Runs |
|---|---:|
| Valid A–D answer + valid confidence | 1 |
| Valid A–D answer + missing confidence | 6 |
| Valid A–D answer + malformed confidence | 3,543 |
| Valid A–D answer + out-of-range confidence | 0 |
| Unavailable/out-of-domain answer + parsed confidence-like value | 8 |

Across all runs, natural confidence status is parsed 9, missing 637, malformed 4,354, out-of-range 0. Parsed counts by subject are mathematics 0, physics 6, chemistry 3, biology 0, psychology 0. Only one parsed value co-occurs with an evaluable natural A–D answer; it is an incorrect run, so AUROC is undefined.

| Subject | Valid answer + parsed conf. | Valid + missing conf. | Valid + malformed conf. | Unavailable answer + parsed conf. |
|---|---:|---:|---:|---:|
| Mathematics | 0 | 2 | 596 | 0 |
| Physics | 0 | 1 | 646 | 6 |
| Chemistry | 1 | 3 | 574 | 2 |
| Biology | 0 | 0 | 847 | 0 |
| Psychology | 0 | 0 | 880 | 0 |

The natural prompt does request the parser’s exact intended format:

```text
Answer: <A|B|C|D>
Confidence: <integer 0-100>
```

The parser requires a valid reasoning close, selects the last line beginning `Answer:`, and accepts confidence only from an immediately adjacent `Confidence:` line in the same terminal block. A `malformed` confidence means such a line was found but its content was empty or not a bare integer; `missing` means no adjacent confidence line was found. This strict same-block rule is scientifically predefined and prevents combining unrelated text.

The raw shards show that this is predominantly an **EOS-decoding/parser incompatibility**, not a failure to emit a confidence field. Of the 4,354 malformed natural confidences, 3,932 are a bare number followed on the same line by the decoded control token (for example `95<|im_end|>`) and 420 are bracketed forms such as `<100><|im_end|>`; 4,352/4,354 would yield an in-range integer after stripping only the sentinel and optional angle brackets. The five most common raw values are `95<|im_end|>` (1,298), `100<|im_end|>` (1,207), `85<|im_end|>` (864), `<100><|im_end|>` (385), and `90<|im_end|>` (346).

Representative raw cases make the boundary clear:

- malformed but visibly numeric: raw record `7e810542…c472dbd`, `Answer: B\nConfidence: 100<|im_end|>`;
- parsed: `f037a1cd…2478d84`, `Answer: A\nConfidence: 95`;
- missing under the fixed contract despite visible fields: `eeeea055…98ff91`, `**Answer: B**\n**Confidence: 95**<|im_end|>`, whose Markdown labels do not match the terminal `Answer:` rule.

Among the 637 natural `missing` statuses, 452 decoded outputs contain `Confidence:` somewhere and 440 have that label followed by a numeral. Eighteen of the 192 missing-close runs contain such text but are correctly excluded because no valid post-reasoning terminal region exists. Thus the model usually attempted the requested field, while the predefined exact terminal-block and integer-only parser rejected it. A post-hoc salvage parser could recover many existing values without new inference, but it would change the fixed measurement contract and must not replace the canonical confidence result.

### Checkpoint confidence

Parsed confidence status across all 5,000 checkpoint outputs at each fraction is:

| Fraction | Parsed status | Usable with local target | Usable / local-evaluable |
|---:|---:|---:|---:|
| 0.0 | 3,435 | 2,099 | 77.03% |
| 0.1 | 846 | 644 | 17.10% |
| 0.2 | 348 | 278 | 7.70% |
| 0.3 | 186 | 140 | 3.95% |
| 0.4 | 142 | 96 | 2.67% |
| 0.5 | 100 | 68 | 1.89% |
| 0.6 | 68 | 49 | 1.35% |
| 0.7 | 62 | 40 | 1.09% |
| 0.8 | 41 | 28 | 0.73% |
| 0.9 | 22 | 15 | 0.37% |
| 1.0 | 2 | 2 | 0.05% |

Across all 55,000 checkpoints, 5,252 confidence values parse (9.55%), 1,251 are missing, and 48,497 are malformed. Answer parsing remains high at the endpoint (4,220/5,000) while confidence collapses to 2/5,000, so this is not simply loss of checkpoint answers.

The raw checkpoint data identify the same dominant failure: 47,228/48,497 malformed values become in-range integers after removing only the sentinel and optional brackets. At fraction 1.0, 4,976/4,984 malformed fields have that form. This is not primarily a 32-token shortage: both of the two parsed endpoint confidences run to the 32-token limit without EOS, whereas 4,983/4,984 malformed endpoint records end in EOS after a mean of 8.57 generated tokens. A representative malformed endpoint is ` B\nConfidence: 100<|im_end|>` after eight tokens; a parsed endpoint runs past ` B\nConfidence: 85` until the 32-token cap. Confidence becomes sparser late because short, prompt-compliant-looking answer/confidence continuations increasingly terminate immediately, and the decoded EOS sentinel remains attached to the integer. Repairing those stored strings is possible as a labelled post-hoc extraction, but using it as the planned result would silently change the parser contract.

### Why the 200-question pilot differs

The historical pilot reports 150 closed-and-answered, one-greedy-run legacy records, endpoint confidence AUROC 0.651, and confidence near 90. The final design uses ten stochastic natural runs per question, a versioned model-specific terminal-block parser, strict answer/confidence pairing, and different cohort rules. The old pilot is explicitly non-authoritative in `DECISIONS.md`.

The final raw data establish the operational cause of final-pipeline sparsity, but the historical pilot is still not a comparable reanalysis. Its one-greedy-run legacy records stored numeric confidence directly, whereas the final versioned parser operates on decoded terminal text and exposes the attached-sentinel mismatch. The report should say that planned verbalized confidence failed under the final extraction contract because of a parser/decoding mismatch—not that confidence “stopped predicting correctness,” and not that the historical AUROC generalizes to the final cohort.

## 10. Endpoint agreement and artifact risk

An endpoint comparison requires both a valid natural A–D answer and a valid fraction-1.0 forced A–D answer. The exact cross-tab is:

| | Endpoint valid | Endpoint invalid | Total |
|---|---:|---:|---:|
| Natural answer valid | 3,467 | 83 | 3,550 |
| Natural answer invalid | 753 | 697 | 1,450 |
| **Total** | **4,220** | **780** | **5,000** |

Thus 1,533 agreement comparisons are unavailable: 753 because only the natural answer is unavailable, 83 because only the endpoint answer is unavailable, and 697 because both are unavailable. The 83 valid-natural/invalid-endpoint rows are all invalid endpoint answer parses (subject counts: math 3, physics 24, chemistry 33, biology 18, psychology 5).

Every evaluable comparison agrees: **3,467/3,467, 100%; zero disagreements**. Agreement rises post-hoc from 1,065/1,867 at 0.0 to 2,186/2,883 at 0.5, 2,917/3,081 at 0.8, 3,151/3,220 at 0.9, and 3,467/3,467 at 1.0.

The direct raw audit recomputed all 5,000 endpoint prefix hashes exactly from the prompt tokens plus generated tokens up to `reasoning_end_exclusive`: **zero hash mismatches**. For all 4,808 closed naturals, the next excluded token is the `</think>` close token; no complete terminal suffix reappears in its prefix. Every question has ten distinct endpoint prefixes, reflecting its sampled reasoning trajectories. A diverse-answer example (sample 42) has ten distinct prefix hashes and natural answers A, D, and B across runs; each greedy endpoint reproduces its own trajectory’s answer while the natural terminal suffix remains outside the input.

Therefore there is no evidence of direct natural-answer leakage or copying. The 100% agreement is a degenerate but genuine conditional-continuation result: after the full sampled reasoning, greedy forced closure reproduces the corresponding terminal answer. Endpoint predictors are consequently perfect target proxies within the 3,467 comparable rows, not independent elicitations. Of the 83 valid-natural/invalid-endpoint cases, 82 end with no located A–D token and one produces out-of-domain text despite a located token; all are forced-output parse failures.

## 11. Baseline question difficulty versus trajectory information

Fraction 0.0 is best treated as a question-level baseline with a documented prompt-date deviation: it has no retained reasoning tokens, and the inferential bootstrap clusters at question level. Direct prefix hashes show one identical baseline prefix across all ten runs for 475/500 questions. The other 25 questions each have two run blocks differing only in the rendered chat-template date (`13→14 August 2026` or `14→15 August 2026`). The effective scale is therefore **525 question-prefix instances**, not 5,000 independent probes and not exactly 500 immutable inputs.

The baseline’s fixed run-level discrimination is modest (max P 0.6109; negative answer entropy 0.6055), and its post-hoc question-level correlation with the proportion of correct stochastic runs is only r=0.117–0.135. Mean reasoning entropy varies across runs and has AUROC 0.7025. Most importantly, the within-question mean-entropy difference remains positive after the question—and therefore its mostly deterministic 0.0 state—is held fixed.

The date-only drift explains the previously observed fraction-0.0 feature anomalies; in at least one case it changes the forced answer (sample 234, C→A). The evidence still supports: **some reliability is already visible in a predominantly question-level pre-reasoning probe, while average reasoning entropy supplies modest additional run-specific information.** “Additional” here means nonzero conditional association, not an incremental predictive-model comparison or causal benefit of reasoning. The prompt-identity deviation must be disclosed; fraction 0.0 should not be called strictly identical within every question.

## 12. Is the entropy result predominantly between questions?

Yes in variance terms, but not entirely in outcome association.

Post-hoc run-weighted variance decomposition over the 3,550 evaluable rows assigns **92.0%** of mean-entropy variation to between-question differences and 8.0% to within-question differences. Across the 472 questions with at least one evaluable run, question mean entropy correlates negatively with question accuracy: Pearson r=−0.289 and Spearman ρ=−0.306.

| Question outcome class | Questions | Mean question entropy | Median | Mean evaluable-run accuracy |
|---|---:|---:|---:|---:|
| All correct | 360 | 0.5483 | 0.5639 | 1.000 |
| Mixed | 84 | 0.6644 | 0.6735 | 0.653 |
| All incorrect | 28 | 0.7010 | 0.6654 | 0.000 |

The mixed subset is therefore substantially more difficult and higher-entropy than the all-correct majority. It is not representative of all 500 questions; it is the scientifically necessary subset for separating trajectories from stable question differences.

The pooled AUROC is consistent with lower-entropy questions also having higher accuracy. The prespecified within-question contrast shows that between-question association is not the whole pattern, but the residual evidence is much smaller: +0.0113 in correctness-oriented entropy with a lower CI bound of +0.00018. Do not describe the pooled 0.703 AUROC as primarily a run-level trajectory effect.

## 13. Full-trajectory entropy versus temporal entropy shape

The raw records contain aligned per-token entropy and exact reasoning boundaries for all 5,000 runs. A post-hoc profile divided each nonempty reasoning span into ten equal token-progress bins and averaged first within each trajectory, so long traces did not receive greater token weight. The analysis conditions on the 3,550 runs with known natural correctness and uses 5,000 subject-stratified question-cluster bootstraps (audit RNG 20260818).

| Normalized reasoning span | Correct mean | Incorrect mean | Correct − incorrect [95% CI], nats |
|---:|---:|---:|---:|
| 0–10% | .3359 | .4559 | −.1199 [−.1497,−.0921] |
| 10–20% | .4887 | .6930 | −.2043 [−.2502,−.1596] |
| 20–30% | .5884 | .7608 | −.1724 [−.2237,−.1194] |
| 30–40% | .6070 | .7586 | −.1515 [−.2001,−.1022] |
| 40–50% | .6067 | .7617 | −.1550 [−.2078,−.1003] |
| 50–60% | .6170 | .7520 | −.1350 [−.1807,−.0873] |
| 60–70% | .6259 | .7331 | −.1072 [−.1516,−.0628] |
| 70–80% | .6318 | .7431 | −.1113 [−.1530,−.0672] |
| 80–90% | .6485 | .7211 | −.0726 [−.1096,−.0344] |
| 90–100% | .6361 | .6975 | −.0614 [−.0934,−.0272] |

Both profiles rise sharply early. Correct runs then climb gradually to an 80–90% peak before a small final decline; incorrect runs peak around 40–50% and decline overall afterward. Correct runs have lower pooled entropy in every normalized decile, with the largest gap at 10–20% and attenuation toward the endpoint. This supplies direct support for a **sustained descriptive uncertainty difference**, not merely a tail effect. It also explains why the final 10% scalar is weaker: outcome separation is smallest late, whereas the full mean aggregates the much larger early/mid gaps.

The temporal pattern is not uniform enough to infer a universal mechanism. All subjects show negative early/mid differences, but mathematics has only ten incorrect runs and its last bin reverses; psychology approaches zero at 80–90%; biology is near zero there. More importantly, the profile is pooled, post-hoc, and conditioned on strongly length-dependent outcome availability. It does not hold question fixed, and the comparable-length within-question cells are too sparse for a temporal paired analysis. The data support “correct evaluable runs exhibit lower uncertainty throughout normalized reasoning on average,” but not “each successful trajectory reduces uncertainty” or “reasoning causally creates the gap.”

## 14. What the Results can honestly say about commitment

Commitment is descriptively structured but weak as a correctness signal. Switching and stabilization have chance-overlapping pooled AUROCs and null within-question intervals. They are not interchangeable with probabilistic uncertainty:

| Post-hoc pooled correlation | n | Pearson r | Spearman ρ |
|---|---:|---:|---:|
| Mean reasoning entropy vs switch count | 5,000 | −0.202 | −0.238 |
| Mean reasoning entropy vs stabilization fraction | 4,220 | −0.072 | −0.061 |
| Switch count vs stabilization fraction | 4,220 | +0.314 | +0.328 |
| Answer entropy 0.5 vs switch count | 3,928 | +0.470 | +0.504 |
| Answer entropy 1.0 vs switch count | 4,240 | +0.116 | +0.372 |

The negative mean-entropy/switch correlation is largely consistent with subject structure: mathematics has low entropy and high switching, while psychology has high entropy and low switching. It directly shows that lower token uncertainty does not imply stable forced answers.

Using explicit post-hoc thresholds, 281 runs are in the lowest mean-entropy quartile yet have at least two switches; 959 are in the highest entropy quartile yet have zero counted switches; and 116 naturally incorrect runs stabilize by 0.3. These are not rare logical counterexamples.

The conceptual conclusion is supported in a limited sense: commitment describes a different observable aspect of checkpoint behaviour, and stronger commitment does not reliably imply a correct natural answer. Do not strengthen that into “commitment adds independent predictive information”; no multivariable incremental test was fixed, and the univariable evidence is weak.

## 15. Strongest negative and null findings

| Finding | Estimate [95% CI] | n | Why it matters |
|---|---:|---:|---|
| Tail entropy, macro AUROC | 0.5369 [0.4810,0.5893] | 3,550 | The final 10% does not reproduce the mean-entropy result across subjects. |
| Switch count, pooled AUROC | 0.5048 [0.4612,0.5489] | 3,550 | Answer changes are interesting behaviour, not a general reliability ranker. |
| Stabilization, pooled AUROC | 0.5605 [0.4916,0.6316] | 3,467 | Earlier stabilization is not reliably predictive. |
| Within-Q tail entropy | −0.00284 [−0.02946,+0.02395] | 84 Q | No conditional tail effect. |
| Within-Q endpoint max P | +0.00039 [−0.00836,+0.00878] | 77 Q | Endpoint certainty has no detectable run-specific difference within the same questions. |
| Within-Q switches | −0.0923 [−0.3143,+0.1290] | 84 Q | Correct runs do not consistently switch less. |
| Within-Q stabilization | −0.0359 [−0.1280,+0.0586] | 77 Q | Correct runs do not reliably stabilize earlier under the fixed contrast. |
| Natural verbal confidence | Undefined | 1 run | Planned RQ3 signal failed at measurement coverage, not at prediction. |

Main Results should include the tail-entropy contrast, commitment nulls, and confidence coverage failure because each changes the scientific story. Full checkpoint-feature nulls and subject details can move to the appendix. Endpoint answer entropy is **not** a null in pooled AUROC (0.6799 [0.6237,0.7341]); its important qualification is saturation and a null within-question contrast.

## 16. Research Question Mapping

### RQ1

- **Strongest supporting result:** negative mean reasoning entropy AUROC 0.7025 [0.6441,0.7574], with lower mean entropy in correct runs.
- **Important secondary result:** post-hoc natural-target answer metrics rise to about 0.684–0.686 at 0.9, while checkpoint-local max-P discrimination peaks earlier at 0.805 at 0.7 and falls as local accuracy continues upward.
- **Important null:** switching 0.5048 [0.4612,0.5489] and stabilization 0.5605 [0.4916,0.6316].
- **Important limitation:** the all-eleven natural-target curve and normalized entropy profile are post-hoc, not part of `final-r5000`; outcome availability is strongly length-dependent.
- **Safe conclusion:** reliability-related measurements evolve non-monotonically, and correct evaluable runs have lower pooled entropy throughout normalized reasoning, but the temporal profile does not establish a within-question or causal mechanism.

### RQ2

- **Strongest supporting result:** mean-entropy within-question difference +0.01128 [0.00018,0.02247] across 84 questions.
- **Important secondary result:** correct runs show earlier first natural-answer appearance post hoc (−0.0756 [−0.1525,−0.0043]).
- **Important null:** all other fixed paired features, including tail entropy, endpoint probabilities, switching, and stabilization, have intervals spanning zero.
- **Important limitation:** only 84 mixed questions qualify; signs are 48/36, subject effects are heterogeneous, and within-question comparable-length cells are sparse.
- **Safe conclusion:** there is modest evidence of a run-specific mean-entropy difference after holding question fixed, but not a broad conditional separation across all signals.

### RQ3

- **Strongest supporting result:** mean reasoning entropy has the largest fixed pooled point estimate (0.7025) and exceeds commitment measures in post-hoc common-case AUROC differences.
- **Important secondary result:** endpoint answer entropy and max probability remain informative (about 0.679) despite extreme endpoint saturation.
- **Important null:** commitment measures are weak, and tail entropy is not robust at macro/subject level.
- **Important limitation:** natural verbal confidence is analyzable for only one run under the fixed parser; the raw text shows this mainly reflects attached EOS sentinels, so coverage failure is partly measurement-pipeline-specific.
- **Safe conclusion:** the signal families are complementary in behaviour and coverage, but the data establish strongest evidence for mean reasoning entropy, not a statistically proven advantage over endpoint answer-distribution measures.

## 17. Competing scientific stories

### Rank 1 — Reliability is mostly question-level, with modest run-specific entropy information

**Central claim.** Difficulty-related differences dominate mean entropy, but average entropy also separates successful from unsuccessful stochastic trajectories within a limited mixed-question subset.

Supporting findings:

1. Pooled mean-entropy AUROC 0.7025; macro 0.7191; same point direction in all subjects.
2. Approximately 92% of mean-entropy variation is between questions; question entropy and accuracy correlate r=−0.289.
3. Fraction-0.0 question baseline is already modestly predictive (AUROC about 0.61).
4. Mean entropy retains a small within-question difference +0.01128 [0.00018,0.02247].
5. Other within-question signals are null.

Complications: only 84 questions contribute; sign test p=0.230; fraction 0.0 has 25 date-variant questions; outcome availability and entropy both vary with length; no incremental predictive model was prespecified. This story fits the Introduction, Background, and Methods with minimal correction and most honestly integrates RQ1–RQ3.

### Rank 2 — Full-span reasoning uncertainty is more informative than late uncertainty, but not proven superior to endpoint choice certainty

**Central claim.** Reliability is better reflected by entropy aggregated over the reasoning span than by tail entropy, whereas endpoint answer-distribution certainty remains a competitive but saturated signal.

Supporting findings:

1. Mean-entropy AUROC 0.7025 versus tail 0.5840; macro 0.7191 versus 0.5369.
2. Mean entropy is the only fixed within-question feature whose interval excludes zero.
3. Post-hoc, correct runs have lower pooled token entropy in all ten normalized bins, with the largest gap early/mid and attenuation late.
4. Endpoint local probabilities saturate near one, reducing local AUROC and worsening ECE; endpoint natural-target answer metrics still achieve about 0.679.

Complications: common-case AUROC differences versus endpoint entropy and max probability include zero; the temporal profile is post-hoc, pooled, length-selected, and not question-paired. This fits the paper title if phrased as a sustained descriptive association, not proof that trajectory mean statistically beats endpoint certainty or that reasoning causes the gap.

### Rank 3 — Reasoning becomes more committed without becoming uniformly more reliable

**Central claim.** Forced answers emerge and stabilize, accuracy rises, and temporary errors often recover, but the strength and stability of commitment do not reliably rank natural-answer correctness.

Supporting findings:

1. Natural answer first appears at mean fraction 0.268; half of computable answers stabilize by 0.5.
2. Checkpoint-local accuracy rises from 47.9% to 88.8%.
3. Max-probability AUROC peaks at 0.805 before falling to 0.679 as probabilities saturate.
4. 566/634 leave-correct events later recover.
5. Switching and stabilization have chance-overlapping AUROCs.

Complications: commitment is intervention-dependent; the pooled AUROC peak timing is subject-heterogeneous; recovery is entangled with perfect endpoint agreement; this story does not directly explain the strongest primary entropy finding. It is strong secondary Discussion material, not the sole central narrative.

## 18. Recommended minimum Results structure

### 1. Cohort, evaluability, and target performance

- **Scientific job:** define what the primary results represent before reporting an AUROC.
- **Exact findings:** 5,000 completed; 3,550 evaluable; 3,172/3,550 correct; 1,450 unavailable; 84 mixed questions; strong subject variation in evaluability.
- **Display:** compact subject-by-subject cohort table.
- **Appendix:** exact status cross-tabs and all 28 zero-evaluable question IDs.
- **Transition:** the evaluable cohort supports pooled discrimination, but question difficulty and repeated runs require both across- and within-question evidence.

### 2. Reasoning entropy: pooled profile, subject, length, and within-question evidence

- **Scientific job:** establish the strongest reliability association and distinguish question-level from trajectory-level information.
- **Exact findings:** AUROC 0.7025 [0.6441,0.7574]; lower correct-run entropy in all ten post-hoc normalized bins; 92% between-question variation; within-question +0.01128 [0.00018,0.02247]; length-adjusted residualized AUROC 0.665 [0.609,0.720].
- **Display:** one three-panel entropy figure: normalized pooled profiles, compact outcome distributions, and ordered within-question effects.
- **Appendix:** all subject profiles/AUROCs, length sensitivities, trimming checks, variance decomposition details, and all 84 rows.
- **Transition:** mean entropy is informative, but tail entropy and other signals test whether this generalizes to late uncertainty and commitment.

### 3. Reliability signals are not interchangeable

- **Scientific job:** answer RQ3 without treating point-estimate ordering as a significance test.
- **Exact findings:** mean > tail and commitment; endpoint answer metrics about 0.679; common-case mean-versus-endpoint differences include zero; confidence fails coverage.
- **Display:** compact main results table or labeled AUROC dot-and-whisker plot with n/coverage.
- **Appendix:** complete subject matrix and all fixed rows.
- **Transition:** the checkpoint-local curve explains why final confidence strength can rise while discriminative ranking falls.

### 4. Checkpoint natural-target dynamics, local accuracy, and saturation

- **Scientific job:** distinguish predicting the eventual natural answer from knowing the checkpoint-local answer, and distinguish becoming confident from ranking errors.
- **Exact findings:** post-hoc natural-target entropy/max-P AUROCs peak near 0.686/0.684 at 0.9; local accuracy rises 0.479→0.888 while local max-P AUROC peaks 0.805 at 0.7 then falls to 0.679; endpoint ECE is 0.101 and 90.9% of endpoint errors have max P≥0.9.
- **Display:** multi-panel checkpoint figure.
- **Appendix:** subject-specific curves, reliability bins, confidence-coverage curve.
- **Transition:** commitment patterns are descriptively rich but weakly related to natural correctness.

### 5. Commitment as description, not a reliability ranker

- **Scientific job:** answer the commitment portion of RQ1 and prevent overinterpretation.
- **Exact findings:** mean switches 0.613; mean first appearance 0.268; mean stabilization 0.496; 566/634 recovery; switch/stabilization AUROCs overlap 0.5.
- **Display:** no dedicated main figure unless space permits one small cumulative panel inside the checkpoint figure.
- **Appendix:** switch distribution, subject dynamics, transition-pattern counts.

The all-eleven checkpoint-to-`natural_correct` curve is now available as a labelled post-hoc raw-merge analysis. The Results must not present it as a fixed `final-r5000` output or conflate it with the fixed checkpoint-local curve.

## 19. Main figures and tables

### Evaluation of the existing outputs

| Existing display | Decision | Reason and redesign |
|---|---|---|
| `primary_auroc.png` | Regenerate | Numeric “fixed feature order” x-axis has no feature labels, CIs, n, coverage, or chance line. Use a horizontal labeled dot-and-whisker plot, show 95% CIs and n, and mark confidence “not estimable (n=1).” |
| `checkpoint_ece.png` | Regenerate for appendix | It omits intervals and denominators; connecting confidence estimates down to n=2 visually implies a stable curve. Show max-P ECE with CIs; show confidence coverage as a separate panel or suppress estimates below a stated n threshold. |
| `natural_reliability.png` | Omit | It is a one-point n=1 plot and is actively misleading as a calibration curve. Replace with a textual coverage failure. |
| `checkpoint_reliability_main.png` | Appendix after redesign | Current points do not identify fraction/family/cohort clearly. Facet by fraction, label n, and avoid mixing sparse verbal confidence with max probability. |
| `within_question_paired_differences.png` | Regenerate | Numeric feature order is unlabeled; bars show only aggregate means and hide question heterogeneity and intervals. Use an ordered question-effect plot for mean entropy plus a compact feature forest/table. |
| `switching_stabilization.png` | Omit or appendix redesign | The x-axis is unlabeled fixed order and the plot overlays incomparable units. Use named subjects and separate panels or a small table. |

### Candidate-display decisions

**A. Primary reliability-signal comparison.** Use a main-text horizontal AUROC dot-and-whisker plot for a curated set: mean entropy, tail entropy, answer entropy/max P at 0.0 and 1.0, switches, and stabilization. Place n beside each row, use a 0.5 reference line, and visually mark differing cohorts. Put the full 11-feature pooled/macro/subject table in the appendix. Do not plot natural confidence as an estimate; show “not estimable, n=1.” This answers RQ3 and makes coverage visible.

**B. Reasoning entropy and natural correctness.** The strongest display is a three-panel figure: (A) post-hoc normalized entropy profiles for correct n=3,172 and incorrect n=378 with question-cluster bands; (B) compact pooled ECDF/violin with median/IQR; (C) all 84 ordered within-question differences with a zero line, subject colour, and the equally weighted mean/95% CI. This makes the sustained pooled shape visible while immediately showing that the question-held-fixed evidence is much smaller. The caption must label panel A post-hoc, define equal-token-progress binning, and disclose length-dependent target missingness.

**C. Checkpoint dynamics.** Use one carefully separated multi-panel figure: (A) post-hoc natural-target entropy/max-P AUROCs versus fraction; (B) fixed checkpoint-local accuracy and max-P AUROC in separate axes/panels; (C) max-P distributions for locally correct/incorrect checkpoints at 0.5, 0.7, and 1.0, showing saturation. Put ECE in the appendix unless space permits a fourth panel. Distinguish target, cohort, and fixed versus post-hoc status in the legend/caption; do not put accuracy, AUROC, and ECE on one y-axis.

**D. Commitment dynamics.** Keep the full switch/stabilization display in the appendix. If commitment needs main-text visibility, add one compact panel with cumulative natural-answer appearance and cumulative stabilization, both with exact denominators; accompany it with the null AUROCs in text. A standalone main figure would overstate predictive importance.

**E. Within-question comparison.** It deserves main-text placement as panel B of the entropy figure, not a separate redundant figure. The reader must see all question effects, not only the average bar. Use one point per question, ordered from negative to positive, with subject colour and correct/incorrect run counts available in the supplement.

**F. Missingness/evaluability.** Use a compact main table, not a flow diagram. The 29.0% unavailable rate and 12.0%–42.2% subject range materially condition every primary analysis; a table communicates this more precisely than a graphic.

### Ranked main-text set and ordering

1. **Table 1 — “Final natural-run cohort and evaluability by subject.”** Rows: five subjects + total. Columns: attempted, parsed, missing-close, closed/missing, out-of-domain, correct, incorrect, evaluable accuracy, mixed questions. Main takeaway: completion was complete but correctness was only available for 71%, unevenly by subject. RQs: foundation for all. Must be newly created.
2. **Figure 1 — “Reasoning-token entropy across successful and unsuccessful trajectories.”** Three panels: normalized pooled profiles, compact distributions, and ordered within-question effects. Show n, question count, clustered intervals, and sign counts. Main takeaway: a sustained pooled difference coexists with a small heterogeneous question-held-fixed effect. RQs 1–2. Must be newly created.
3. **Figure 2 — “Natural-target and checkpoint-local reliability evolve differently.”** Panels: natural-target probability AUROCs, checkpoint-local accuracy/AUROC, and selected endpoint-saturation distributions. Main takeaway: eventual-answer discrimination rises late, whereas local discrimination peaks earlier even as local accuracy continues rising. RQ1. Must be newly created from fixed plus clearly labelled post-hoc values.
4. **Figure or Table 3 — “Primary reliability-signal discrimination and coverage.”** Prefer a dot-and-whisker figure in main text plus a numeric appendix table. Show curated predictors, fixed 95% CIs, n, and unavailable confidence. Main takeaway: mean entropy is strongest by point estimate; tail and commitment are weak; endpoint probabilities remain competitive. RQ3. Must be regenerated.

### Appendix plan

- full 11-feature pooled/macro/subject AUROC table;
- complete all-eleven natural-target and checkpoint-local numeric tables and subject curves;
- all ECE rows, reliability bins, and confidence coverage by fraction;
- full cohort status and answer/confidence cross-tabs;
- exact ordered 84-question paired-effect table and all paired features;
- switch-count, appearance, stabilization, leave/recovery, and sequence tables;
- subject-level dynamics table;
- common-case and AUROC-difference post-hoc sensitivity checks;
- question-difficulty variance decomposition and trimming checks;
- raw-output confidence examples/salvage counts, endpoint-prefix recomputation, and the fraction-0.0 date-drift audit.

## 20. Required report corrections or qualifications

The Introduction, Background, and Methods do not need structural rewriting, but the following small corrections are scientifically necessary.

1. **Main attachment line 170:** “ensuring that no single subject contributes disproportionately to the overall results” is false for the evaluable cohort. Keep the balanced sampling statement, then add that evaluability varied by subject and report pooled plus macro/subject analyses.
2. **Line 186:** “This answer was compared … to determine whether the trajectory was correct” should become “When a valid post-reasoning A–D terminal answer was parsed, it was compared…”; 1,450/5,000 have no natural correctness label.
3. **Lines 180 and 196:** “obtain … confidence measurements” and “Verbalized confidence was recorded” imply successful measurement. Replace with “attempted to elicit” and “recorded when validly parsed”; final natural coverage is 1/3,550 for discrimination. Explain that decoded EOS sentinels attached to otherwise numeric fields caused most parser rejection rather than saying the model usually omitted confidence.
4. **Results outline line 214:** the all-eleven natural-correctness curve was not part of the fixed analysis and has now been reconstructed post hoc from validated raw/merged records. Label it post-hoc and keep it distinct from the fixed checkpoint-local curve and the fixed 0.0/0.5/1.0 primary rows.
5. **Results outline line 238:** the signal-comparison display must show coverage and measurement failure, not merely discrimination/calibration values.
6. **Fraction 0.0:** identify it as a predominantly question-level pre-reasoning baseline and disclose that 25 questions have two date-variant prompts, producing 525 distinct question-prefix instances. Do not call all ten runs identical for every question.
7. **Endpoint agreement:** report 3,467/3,467 empirical agreement and the direct prefix audit: all 5,000 hashes recomputed, zero terminal suffix leakage. Describe endpoint probing as conditional continuation that perfectly proxies the natural target on comparable rows, not as an independent measure.
8. **Commitment:** retain the existing cautious language that checkpoint answers are intervention-induced observable behaviour. Do not imply switching or stabilization strongly predicts correctness.
9. **Post-hoc temporal/length analyses:** Methods or Results captions must define ten equal token-progress bins, per-run averaging, question-cluster bootstrapping, and simple length adjustment. Do not retroactively call these fixed analyses.
10. **Title/abstract:** “Beyond Endpoint Confidence” is defensible only if “endpoint confidence” includes answer-distribution strength; the final study cannot use canonical natural verbalized confidence for discrimination. Define the term or consider “Beyond Endpoint Certainty.”

No audited entropy, AUROC, switching, stabilization, or bootstrap definition conflicts with the authoritative design documents. The serious mismatches are analysis availability, measurement coverage, and cohort interpretation.

## Audit appendix: ordered within-question mean-entropy effects

Positive values mean lower average reasoning entropy in correct runs. Question IDs are abbreviated; the canonical full IDs and full precision are in `within_question_distribution.csv`.

|#|Subject|Question ID prefix|Correct/incorrect runs|Difference in −mean entropy|
|--:|---|---|---:|---:|
|1|physics|`75e14e9814f4`|5/1|−0.202002|
|2|physics|`99f9cfd4a671`|1/1|−0.127089|
|3|chemistry|`b861e11f31ae`|3/1|−0.068185|
|4|chemistry|`2b45e0010964`|9/1|−0.064902|
|5|biology|`5b418f78c574`|5/4|−0.064521|
|6|chemistry|`7a9638f47391`|4/1|−0.051782|
|7|psychology|`9be0a6addf1e`|5/5|−0.051551|
|8|chemistry|`262e932aee9a`|1/2|−0.045224|
|9|chemistry|`07760bc2b40c`|4/2|−0.041647|
|10|chemistry|`0c4c5ede4007`|2/1|−0.041275|
|11|physics|`2acc5483af0c`|7/2|−0.040031|
|12|physics|`da353d4b6dab`|1/8|−0.032679|
|13|psychology|`f191355e784b`|8/1|−0.031403|
|14|mathematics|`2289d3a2f6e2`|8/2|−0.029898|
|15|psychology|`112ab98f47b2`|8/1|−0.029473|
|16|chemistry|`c2f9964a19f3`|2/2|−0.028016|
|17|biology|`4d08dfc6cc3a`|9/1|−0.027193|
|18|biology|`ca1ba4b96b6b`|4/4|−0.026984|
|19|biology|`468b9a4c6b06`|5/2|−0.021623|
|20|biology|`7eb128074cbb`|9/1|−0.020981|
|21|chemistry|`4bd76769468a`|7/2|−0.018955|
|22|psychology|`351d2f4c22fa`|8/2|−0.018566|
|23|chemistry|`48080e290d4b`|9/1|−0.016830|
|24|chemistry|`5223f24c8eb5`|4/2|−0.016049|
|25|physics|`46decb78eef9`|3/2|−0.015570|
|26|physics|`620f7002bae7`|4/1|−0.014644|
|27|physics|`4c08d246de2d`|3/2|−0.014530|
|28|physics|`4caf55c91628`|3/3|−0.012734|
|29|chemistry|`3a133f01b22a`|5/1|−0.010575|
|30|biology|`d73ff6322eb0`|9/1|−0.009487|
|31|physics|`e624550bea65`|7/2|−0.009081|
|32|biology|`f8ed086535a9`|4/3|−0.006355|
|33|physics|`84a9f9f4e3e7`|5/3|−0.005786|
|34|physics|`81d0bc4fba39`|4/6|−0.004419|
|35|mathematics|`0e147c2121f5`|8/1|−0.002410|
|36|psychology|`bf8c6d73b5ff`|6/2|−0.001313|
|37|biology|`c5585e63533b`|3/7|+0.000894|
|38|biology|`07f001853e62`|9/1|+0.001181|
|39|physics|`7ddae73db3fc`|3/1|+0.001293|
|40|physics|`322df273cbff`|8/1|+0.006771|
|41|mathematics|`c089ea6767cb`|7/2|+0.008633|
|42|biology|`f3c89e6e2d0a`|9/1|+0.009073|
|43|biology|`df001e4b2188`|2/8|+0.011627|
|44|chemistry|`dc696d71a77a`|3/2|+0.012015|
|45|psychology|`9f5cc920801e`|9/1|+0.014896|
|46|psychology|`0402aaa2d965`|9/1|+0.016601|
|47|mathematics|`88c210152c19`|6/2|+0.016874|
|48|psychology|`7aafee347562`|8/1|+0.020164|
|49|chemistry|`4ebeb12b461a`|8/1|+0.023292|
|50|psychology|`0a24b470278d`|9/1|+0.024714|
|51|biology|`01e339efede1`|6/4|+0.028157|
|52|biology|`f92d1b39fe46`|4/1|+0.031415|
|53|chemistry|`ba451ac5eb95`|3/1|+0.031822|
|54|psychology|`4befd78f0144`|2/1|+0.032164|
|55|chemistry|`ca27465a31e2`|6/1|+0.034419|
|56|physics|`dc99db74d7df`|1/1|+0.034491|
|57|psychology|`0a020796a310`|3/2|+0.034522|
|58|psychology|`fd7bac205325`|9/1|+0.035356|
|59|physics|`dd0ce3530dc7`|3/3|+0.036439|
|60|mathematics|`f89110b93c2d`|9/1|+0.036477|
|61|biology|`781805d95b12`|9/1|+0.043772|
|62|chemistry|`a5048523b5ca`|1/4|+0.044323|
|63|biology|`3a289154753c`|1/9|+0.046137|
|64|psychology|`80b7060927c4`|1/7|+0.046220|
|65|psychology|`39876955f00d`|2/6|+0.046382|
|66|chemistry|`f7a596c5c0c2`|3/1|+0.047896|
|67|physics|`c9f1a371ecc6`|9/1|+0.051050|
|68|psychology|`2c76c49933b9`|4/6|+0.052273|
|69|chemistry|`ae4b8bf7c9a4`|1/4|+0.057253|
|70|chemistry|`0d2fe059f3ed`|2/4|+0.059932|
|71|physics|`232dfc710dd4`|5/1|+0.061464|
|72|physics|`23e13775454a`|4/5|+0.063962|
|73|physics|`6188fde1f6c3`|2/4|+0.064939|
|74|biology|`9f284cfaffcb`|2/5|+0.067290|
|75|chemistry|`db2aec759626`|4/1|+0.069780|
|76|biology|`711ab1968803`|9/1|+0.070814|
|77|mathematics|`d9b54e27ae7f`|8/2|+0.072540|
|78|psychology|`847c0d13d6e0`|1/9|+0.074217|
|79|physics|`4b0b3f620eaa`|9/1|+0.086353|
|80|psychology|`ff40cf000e4f`|3/7|+0.090084|
|81|psychology|`3ec3c51e49d4`|8/2|+0.096373|
|82|physics|`ff5b5c32e228`|2/1|+0.111739|
|83|chemistry|`e97def33184e`|8/1|+0.117717|
|84|biology|`7b6654e362e8`|1/6|+0.125441|

## Ten Findings I Should Understand Before Writing

1. All **5,000/5,000** natural runs completed, but only **3,550 (71.0%)** have evaluable natural correctness: **3,172 correct and 378 incorrect**.
2. The **1,450 unavailable** labels are exactly **192 missing reasoning closes, 424 closed/missing answers, and 834 closed/out-of-domain answers**; their median reasoning length is **1,380 tokens**, versus **484.5** for correct and **789** for incorrect runs, and subject missingness ranges from **12.0% to 42.2%**.
3. Negative mean reasoning entropy has pooled AUROC **0.7025 [0.6441,0.7574]** and macro AUROC **0.7191 [0.6628,0.7727]**; incorrect runs average **0.7075** entropy versus **0.5783** for correct runs.
4. The pooled entropy association is predominantly between questions (**92.0%** of variation; question entropy–accuracy **r=−0.289**), yet a smaller question-held-fixed effect remains across **84 mixed questions**: **+0.01128 [0.00018,0.02247]**, with only **48/84** in the expected direction.
5. Post-hoc normalized profiles show lower correct-run entropy in **all ten reasoning deciles** (differences −0.120 to −0.061 nats, every clustered interval excluding zero), attenuating late; after simple length/subject residualization, entropy AUROC remains **0.665 [0.609,0.720]**.
6. Tail entropy is much weaker: pooled AUROC **0.5840 [0.5327,0.6305]**, macro **0.5369 [0.4810,0.5893]**, and within-question difference **−0.00284 [−0.02946,+0.02395]**.
7. Against natural correctness, post-hoc answer-entropy and max-P AUROCs are modest through 0.6, peak at **0.6859/0.6840 at 0.9**, and slip slightly to **0.6799/0.6786 at 1.0**; their shared all-fraction cohort shows the same late peak.
8. Checkpoint-local accuracy rises from **47.9% to 88.8%**, while local max-P AUROC peaks at **0.805 [0.770,0.840] at 0.7** and falls to **0.679 [0.626,0.732]**; **431/474** endpoint errors still have max P≥0.9.
9. Commitment is descriptively rich but weakly predictive: mean switches **0.6132**, mean stabilization **0.4955**, switch AUROC **0.5048 [0.4612,0.5489]**, and stabilization AUROC **0.5605 [0.4916,0.6316]**.
10. Canonical verbalized confidence fails coverage (**1/3,550** natural targets), but raw text shows **4,352/4,354** malformed natural values are salvageable sentinel/bracket forms; this is chiefly a fixed parser/decoding mismatch, not wholesale model omission.

## Five Things I Must Not Overclaim

1. Do not turn the post-hoc pooled temporal profile into a causal or within-question claim; it conditions on the 71% evaluable cohort, whose availability is strongly length-dependent.
2. Do not say mean entropy statistically outperforms endpoint answer entropy or max probability; their post-hoc common-case AUROC-difference intervals include zero.
3. Do not generalize primary AUROCs to all 5,000 runs; they condition on 3,550 evaluable answers, with strong subject- and length-dependent missingness.
4. Do not call stabilization an internal decision point or infer that post-stabilization reasoning is unnecessary; it is a forced-probe behavioural summary and is weakly predictive.
5. Do not call fraction 0.0 exactly one immutable input per question (25 questions have date drift), and do not say the model usually omitted confidence when the raw values mainly contain parser-rejected EOS sentinels.

## Three Most Important Limitations

1. **Outcome selection:** 29.0% of correctness labels are unavailable, unevenly by subject and strongly by reasoning length; the primary cohort is not representative of all completed trajectories.
2. **Measurement/provenance deviations:** canonical verbalized confidence is nearly absent because of an EOS-sentinel parser mismatch, and 25 fraction-0.0 question blocks have date-drifted prompts. Post-hoc salvage cannot silently replace fixed results.
3. **Conditional evidence is limited and heterogeneous:** only 84 mixed-outcome questions contribute; the mean-entropy interval barely excludes zero, signs are 48/36, and question-paired comparable-length cells are sparse.

## Recommended Central Story

The Results and Discussion should center on a qualified two-level finding: average reasoning-token entropy is the strongest fixed reliability signal by pooled point estimate, and post-hoc profiles show a sustained correct/incorrect difference that is largest early/mid and attenuates late; however, most entropy variation is between questions, outcome availability is length-selected, and the prespecified question-held-fixed effect is small. The checkpoint curves then show that natural-target discrimination rises late while checkpoint-local discrimination peaks earlier even as answers become more accurate and saturated. Commitment provides descriptive trajectory structure but little correctness ranking, and canonical verbal confidence mainly exposes a parser/decoding measurement failure. The contribution is the separation of question-level reliability, modest run-specific uncertainty, endpoint saturation, and commitment—not proof that reasoning entropy beats endpoint certainty or causally determines success.

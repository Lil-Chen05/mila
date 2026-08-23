# Outlier gallery — 200q

*1 greedy run/question | grouping = final-checkpoint correctness | truncated (16k) and skipped (no-reasoning) cohorts held out of aggregates*

## Most overconfident wrong answers

Sorted by stated confidence (desc), then answer entropy (asc): the model is wrong, *says* it is sure, and its answer distribution agrees it is sure — the worst calibration failures.

### qid 114 — college_computer_science
**gold B / model C (WRONG)** | conf=100 | H_ans_end=0.002 | H_ans_mid=0.586 | H_think_mean=0.599 | n_think=404

> A program that checks spelling works in the following way. A hash table has been defined in which each entry is a Boolean variable initialized to false. A hash function has been applied to each word in the dictionary, and the appropriate entry in the hash table has been set to true. To check the spelling in a document, the hash function is applied to every word in the document, and the appropriate...

  - A. I only
  - B. II only
  - C. I and II only
  - D. II and III only

### qid 106 — machine_learning
**gold D / model A (WRONG)** | conf=100 | H_ans_end=0.008 | H_ans_mid=1.217 | H_think_mean=0.683 | n_think=432

> Statement 1| L2 regularization of linear models tends to make models more sparse than L1 regularization. Statement 2| Residual connections can be found in ResNets and Transformers.

  - A. True, True
  - B. False, False
  - C. True, False
  - D. False, True

### qid 191 — high_school_macroeconomics
**gold A / model C (WRONG)** | conf=95 | H_ans_end=0.004 | H_ans_mid=1.106 | H_think_mean=0.719 | n_think=718

> Stagflation is the result of

  - A. a leftward shift in the aggregate supply curve.
  - B. a leftward shift in the aggregate demand curve.
  - C. a leftward shift in both the aggregate supply and aggregate demand curves.
  - D. a rightward shift in the aggregate supply curve.

### qid 60 — moral_scenarios
**gold D / model C (WRONG)** | conf=95 | H_ans_end=0.011 | H_ans_mid=0.984 | H_think_mean=0.800 | n_think=267

> For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I make sure my high-watt security light doesn't aim in my neighbor's bedroom window. Scenario 2 | I swung a pillow at my best friend James.

  - A. Wrong, Wrong
  - B. Wrong, Not wrong
  - C. Not wrong, Wrong
  - D. Not wrong, Not wrong

### qid 29 — high_school_chemistry
**gold C / model B (WRONG)** | conf=95 | H_ans_end=0.017 | H_ans_mid=0.067 | H_think_mean=0.644 | n_think=1898

> During the investigation of a chemical reaction by a chemistry student, she was able to determine that the reaction was nonspontaneous at 1 atm and 298 K. However, she learned that cooling the system with dry ice (-78°C) caused the reaction to become spontaneous. Which of the following combinations must apply to this reaction?

  - A. ΔH < 0, ΔS < 0, and ΔG = 0
  - B. ΔH > 0, ΔS < 0, and ΔG > 0
  - C. ΔH < 0, ΔS < 0, and ΔG > 0
  - D. ΔH > 0, ΔS > 0, and ΔG > 0

### qid 113 — moral_scenarios
**gold D / model C (WRONG)** | conf=95 | H_ans_end=0.017 | H_ans_mid=0.917 | H_think_mean=0.793 | n_think=343

> For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | Upon hearing that her son was captured, I assured her not to worry because the authorities were on the move and would resolve the issue. Scenario 2 | I added hot sauce to my friend's dinner.

  - A. Wrong, Wrong
  - B. Wrong, Not wrong
  - C. Not wrong, Wrong
  - D. Not wrong, Not wrong

### qid 45 — professional_psychology
**gold B / model A (WRONG)** | conf=95 | H_ans_end=0.021 | H_ans_mid=0.257 | H_think_mean=0.747 | n_think=377

> You are subpoenaed to appear at a deposition to testify about a former client of yours. During the deposition, the attorney for the opposing party asks you to provide confidential client information. You should:

  - A. refuse to provide the information without a court order.
  - B. assert the privilege unless the client has signed a waiver.
  - C. provide the information if you believe it is pertinent to the case.
  - D. provide the information since privilege is waived in this situation.

### qid 42 — virology
**gold A / model D (WRONG)** | conf=95 | H_ans_end=0.022 | H_ans_mid=0.321 | H_think_mean=0.618 | n_think=295

> The treatment of hepatitis C has been revolutionised most recently by which of the following?

  - A. The use of interferon
  - B. A new vaccine
  - C. Monoclonal antibodies
  - D. Direct-acting Antivirals such as daclatasvir and sofosbuvir

### qid 0 — college_physics
**gold A / model C (WRONG)** | conf=95 | H_ans_end=0.133 | H_ans_mid=0.146 | H_think_mean=0.509 | n_think=828

> Positronium is an atom formed by an electron and a positron (antielectron). It is similar to the hydrogen atom, with the positron replacing the proton. If a positronium atom makes a transition from the state with n=3 to a state with n=1, the energy of the photon emitted in this transition is closest to

  - A. 6.0 e
  - B. 6.8 eV
  - C. 12.2 eV
  - D. 13.6 eV

### qid 187 — human_aging
**gold B / model A (WRONG)** | conf=85 | H_ans_end=0.003 | H_ans_mid=0.683 | H_think_mean=0.591 | n_think=328

> Loss of stomach lining can lead to symptoms that resemble

  - A. Anorexia
  - B. Dementia
  - C. Senescence
  - D. Hepatitis

## Signals disagree: stated certain, distribution uncertain

conf_end >= 90 but highest answer-entropy at the endpoint: verbalized confidence and token entropy point opposite ways.

### qid 4 — prehistory
**gold A / model A (RIGHT)** | conf=90 | H_ans_end=0.536 | H_ans_mid=1.310 | H_think_mean=0.594 | n_think=381

> What was NOT a deciding factor in the development of Mesopotamian civilization?

  - A. the need to concentrate population away from arable lands near rivers
  - B. the need to construct monumental works, such as step-pyramids and temples
  - C. the need to develop a complex social system that would allow the construction of canals
  - D. the use of irrigation to produce enough food

### qid 78 — elementary_mathematics
**gold A / model A (RIGHT)** | conf=95 | H_ans_end=0.212 | H_ans_mid=0.108 | H_think_mean=0.542 | n_think=1709

> Estimate 153 + 44. The sum is between which numbers?

  - A. 100 and 299
  - B. 300 and 499
  - C. 500 and 699
  - D. 700 and 899

### qid 69 — high_school_physics
**gold A / model A (RIGHT)** | conf=95 | H_ans_end=0.205 | H_ans_mid=0.212 | H_think_mean=0.436 | n_think=1574

> How long would it take a car, starting from rest and accelerating uniformly in a straight line at 5 m/s^2, to cover a distance of 200 m ?

  - A. 9.0 s
  - B. 10.5 s
  - C. 12.0 s
  - D. 15.5 s

### qid 0 — college_physics
**gold A / model C (WRONG)** | conf=95 | H_ans_end=0.133 | H_ans_mid=0.146 | H_think_mean=0.509 | n_think=828

> Positronium is an atom formed by an electron and a positron (antielectron). It is similar to the hydrogen atom, with the positron replacing the proton. If a positronium atom makes a transition from the state with n=3 to a state with n=1, the energy of the photon emitted in this transition is closest to

  - A. 6.0 e
  - B. 6.8 eV
  - C. 12.2 eV
  - D. 13.6 eV

### qid 143 — high_school_microeconomics
**gold D / model D (RIGHT)** | conf=95 | H_ans_end=0.111 | H_ans_mid=0.391 | H_think_mean=0.455 | n_think=366

> You are told that the cross-price elasticity between goods X and Y is +2.0. This means that

  - A. goods X and Y are normal goods.
  - B. goods X and Y are inferior goods.
  - C. goods X and Y are complementary goods.
  - D. goods X and Y are substitute goods.

### qid 39 — high_school_mathematics
**gold A / model A (RIGHT)** | conf=95 | H_ans_end=0.098 | H_ans_mid=1.318 | H_think_mean=0.395 | n_think=1754

> If $m$ and $n$ are odd integers, how many terms in the expansion of $(m+n)^6$ are odd?

  - A. 4
  - B. 2
  - C. 1
  - D. 9

### qid 108 — moral_scenarios
**gold A / model A (RIGHT)** | conf=95 | H_ans_end=0.087 | H_ans_mid=1.066 | H_think_mean=0.799 | n_think=287

> For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I tried to spike the lemonade with fentanyl. Scenario 2 | When my son told me he was gay, I rejected him.

  - A. Wrong, Wrong
  - B. Wrong, Not wrong
  - C. Not wrong, Wrong
  - D. Not wrong, Not wrong

### qid 35 — high_school_computer_science
**gold D / model D (RIGHT)** | conf=100 | H_ans_end=0.084 | H_ans_mid=0.015 | H_think_mean=0.472 | n_think=295

> Let l = [1,2,3,4]. What is max(l) in Python3?

  - A. 1
  - B. 2
  - C. 3
  - D. 4

### qid 165 — high_school_psychology
**gold D / model D (RIGHT)** | conf=95 | H_ans_end=0.082 | H_ans_mid=1.182 | H_think_mean=0.602 | n_think=551

> In people, rods, unlike cones,

  - A. are located in the center of the retina.
  - B. synapse with bipolar cells.
  - C. respond more quickly to bright colors.
  - D. have a low absolute threshold for light.

### qid 101 — moral_disputes
**gold C / model C (RIGHT)** | conf=95 | H_ans_end=0.061 | H_ans_mid=0.308 | H_think_mean=0.526 | n_think=346

>  Which of the following is an example of a value-based moral theory?

  - A. consequentialism
  - B. virtue ethics
  - C. both A and B
  - D. neither A nor B

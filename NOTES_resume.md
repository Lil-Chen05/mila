# Resume notes

## State as of this session

The checkpoint experiment is working and has produced a real, paper-worthy signal.
Cluster work is paused at a clean, committed point. Next session starts with a
read-only investigation (no generation), so the gpu_common gate does NOT block it.

## What exists and is verified
- mc_common.py: torch-free helpers; ONE shared markdown-tolerant ANSWER_RE/CONF_RE;
  owns find_answer_token. Login-safe pytest suite green.
- gen_chains.py: original endpoint-generation script. main()-guarded? (CHECK — this is
  part of the retire/refactor question below.)
- checkpoints.py: the live experiment. main()-guarded, self-contained (computes
  correctness/natural_pred/n_think from its OWN regenerated chain, never joins to
  chains.jsonl — the greedy determinism gap makes that join invalid). Mirrors 3 torch
  helpers from gen_chains VERBATIM (token_entropy, split_think, get_letter_token_ids),
  still debt-tagged.
- results/checkpoints_20q.jsonl: 220 rows = 20 questions x 11 deciles, FORCE_CLOSE=True,
  inducer v1. Invariant 19 Y / 0 N / 1 NA (qid14 unclosed). One letters_matched=False
  (qid15 @ 0.5). All abstract_algebra (no-shuffle limitation), 13/19 closed correct.

## The finding (exploratory, suggestive not significant)
- Answer-letter entropy SEPARATES correct from incorrect by CONVERGENCE TIMING, not
  level: correct answers' entropy collapses earlier (mid-chain, ~frac 0.5-0.9);
  incorrect stay high until ~0.9; BOTH collapse to ~0.04 nats at frac 1.0 (endpoint is
  flat — invisible if you only measure the committed answer, which is why checkpoints
  matter).
- Verbalized confidence does NOT separate (85-95 both groups throughout). Token entropy
  is the live signal; verbalized confidence is the dead one here.
- 14/19 questions flip their committed answer at least once across deciles — reasoning
  does real work, not just ratifying a prior.
- CAVEATS: single subject, n=19 usable, n=6 incorrect (no p-values), 1 greedy run/q,
  exploratory off mirrored helpers. CONFOUND to check: incorrect chains are longer, so
  "fraction of chain" != same token position across groups — re-check timing vs ABSOLUTE
  token position (k_keep) on the next run.

## BLOCKER status: gpu_common refactor — UNDER REVIEW
The gate was to prevent drift between two live scripts sharing the 3 mirrored helpers.
BUT checkpoints.py may SUBSUME gen_chains.py (same generation + checkpoints), making
gen_chains retire-able rather than refactor-needed. If only one script is live, no drift
is possible and the gate is moot.

## NEXT ACTION (read-only investigation, no generation, login-safe)
Confirm or refute subsumption before deciding retire-vs-refactor. Check:
1. Does checkpoints output cover every field chains.jsonl has?
2. Does anything actually read chains.jsonl?
3. Is the NATURAL chain's per-token entropy (gen_chains' unique capability; checkpoints
   sets output_logits=False on the natural pass) needed by any planned experiment?
4. Are the 3 helpers byte-identical, and is gen_chains imported anywhere?
Then: RETIRE gen_chains (if fully subsumed + nothing needs its unique output) OR do the
gpu_common.py refactor properly (if a genuine 2nd consumer / natural-trajectory need
exists). Decide with GK-level rigor; the answer to #3 is the deciding factor.

## Still pending before any FINDINGS run (subject-diverse, larger, multi-run)
- Resolve gpu_common gate (retire or refactor — above).
- Fix dataset diversity: data/mmlu_20 is all abstract_algebra (fetch_mmlu.py used
  .take(20) no shuffle). Re-fetch shuffled/diverse; discuss dataset choice with GK.
- Bring the entropy-timing finding + this chart to GK as the "signal + scaling plan."

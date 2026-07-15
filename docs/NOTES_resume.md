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

## RESOLVED: gpu_common gate — gen_chains.py RETIRED
GK confirmed the natural chain's per-token entropy IS needed — but as an ADDITIONAL
signal to correlate with answer-entropy and verbalized confidence, so it must be
measured on the SAME chain the checkpoints probe (greedy determinism gap). That means
it belongs in checkpoints.py's natural pass, which fully subsumes gen_chains.py:
- gen_chains.py, gen_chains_job.sh, results/chains.jsonl removed (nothing imported or
  read them; the 3 torch helpers now live solely in checkpoints.py — no drift possible).
- The gpu_common refactor gate is moot: exactly one live GPU script.

## NEXT ACTION
Add per-token entropy capture to checkpoints.py's natural pass (output_logits=True,
reduce each logit row to an entropy scalar immediately, del out as before). New output:
results/chain_token_entropy_20q.jsonl, one row per qid with the full per-token entropy
array (reasoning + post-think; n_think marks the boundary). Sanity: assert
len(entropies) == len(gen_ids); print mean reasoning entropy per question.

## Still pending before any FINDINGS run (subject-diverse, larger, multi-run)
- Fix dataset diversity: data/mmlu_20 is all abstract_algebra (fetch_mmlu.py used
  .take(20) no shuffle). Re-fetch shuffled/diverse (~100 questions); discuss dataset
  choice with GK.
- GK wants ALL signals collected and correlated (reasoning-token entropy ~ verbalized
  confidence ~ answer entropy), not a choice between them.

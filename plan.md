# Step 5 (smallest version) plan — `checkpoints.py` + `checkpoints_job.sh`

CHECKPOINTS, not early-exit: we never halt generation. We generate ONE full greedy
reasoning chain once, then re-probe the committed answer at points ALONG that same
chain. GPU work → runs as a SLURM job, never on the login node. Debug the smallest
end-to-end version first: **1 MMLU question, the default fractions**, then STOP.

## Scope & non-goals (this step only)
- IN: one question, one full chain, checkpoint probes at `[0.0, 0.25, 0.5, 0.75, 1.0]`,
  a printed trajectory table.
- OUT (do NOT build yet): multiple questions, decile sweep, sampling, the `interval`
  / `step-boundary` strategies, plots, aggregation. The interface is designed so
  those slot in later without rework.

## Files (project naming convention: `<task>.py` + `<task>_job.sh`)
- `checkpoints.py` — the experiment.
- `checkpoints_job.sh` — GPU SLURM job (mirrors `gen_chains_job.sh`).

## Reuse-of-`gen_chains` decision — SURFACED (not silently chosen)
`import gen_chains` is unsafe: that module loads the model and runs the whole
20-question loop at **import time** (module-level code), so importing it would
execute the wrong pipeline. Two ways to honor "reuse the existing generation logic":

- **Option A (recommended for this smallest version):** `checkpoints.py` is
  self-contained — it imports the torch-free helpers from `mc_common`
  (`build_messages`, `parse_answer_confidence`, `find_answer_token`, `is_correct`)
  and **mirrors the three small torch helpers verbatim** from `gen_chains`
  (`token_entropy`, `split_think`, `get_letter_token_ids`), each tagged
  `# mirrored from gen_chains.py — unify into a shared gpu module before scaling`.
  Keeps this step reviewable and leaves the verified `gen_chains.py` untouched.
- **Option B (defer):** first refactor a shared `gpu_common.py` (those 3 helpers +
  a side-effect-free `generate_one_chain()`), repoint `gen_chains.py` at it, re-verify
  its run, *then* build `checkpoints.py`. Cleaner (no duplication) but it's a second
  step that touches already-green code.

Proposal: **Option A now**, with Option B as an explicit gate *before* scaling to
deciles × 20 (so the duplication never reaches the scaled run). Flagging the ~40
mirrored lines as known, time-boxed debt.

## The strategy function (configurable placement, no hard-coded spacing)
```python
def checkpoint_indices(think_token_count,
                       strategy="fraction",
                       fractions=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Return SORTED, DEDUPED think-token prefix lengths to probe.
    v1 supports strategy="fraction": idx = int(round(f * think_token_count)),
    clamped to [0, think_token_count]. "interval"/"step-boundary" raise
    NotImplementedError for now but share this signature so they slot in later.
    """
```
- `fraction=0.0` → keep 0 think tokens (the "blurt", answer with no reasoning).
- `fraction=1.0` → keep all think tokens (full reasoning; drives the invariant below).
- Dedup matters: for short blocks adjacent fractions round to the same index; we
  collapse them and **log** the collapse rather than silently probing a point twice.

## Pipeline (one question, all greedy / deterministic)
1. `load_from_disk("data/mmlu_20")`; pick `QID` (env, default 0).
2. Generate the full chain ONCE exactly like `gen_chains` (SmolLM3-3B, bf16,
   `apply_chat_template(..., enable_thinking=True)`, `do_sample=False`,
   `output_logits=True`). Keep `prompt_ids`, `gen_ids`, and the natural
   `(pred_letter, confidence)` from `parse_answer_confidence(full_text)`.
3. `reasoning_ids, post_ids, think_closed = split_think(gen_ids, tok)`;
   `n_think = len(reasoning_ids)`.
4. Min-length guard (below). If it passes:
   `idxs = checkpoint_indices(n_think, "fraction", FRACTIONS)`.
5. For each checkpoint length `k` in `idxs`: build the forced input in **token-id
   space** (see splicing note), run a **short** forced generation
   (`max_new_tokens≈32`, greedy, `output_logits=True`), then reuse
   `parse_answer_confidence` + `find_answer_token` + `get_letter_token_ids` to read
   `(forced_letter, forced_confidence, answer_letter_entropy, answer_fullvocab_entropy,
   letters_matched)` at the forced answer token.
6. Print the trajectory table; check the 1.0 invariant; (optionally) save a small JSON.

## DESIGN POINT 1 — forcing prompt: close `</think>` vs leave it open
A config flag `FORCE_CLOSE` selects the inducer. Exact strings are version-stamped
and logged at run start:
```
INDUCER_VERSION = "v1"
# built in token-id space, not re-tokenized text:
#   FORCE_CLOSE=True  -> inducer_ids = [close_think_id] + tok.encode("\nAnswer:", add_special_tokens=False)
#   FORCE_CLOSE=False -> inducer_ids =                    tok.encode("\nAnswer:", add_special_tokens=False)
INDUCER_TEXT_CLOSED = "</think>\nAnswer:"
INDUCER_TEXT_OPEN   = "\nAnswer:"
```
Inducer ends at `"Answer:"` (no trailing space) so the model emits the
leading-space letter itself — the same convention the natural chain uses, which
`get_letter_token_ids` already resolves.

**Proposed default: `FORCE_CLOSE = True` (close the think block).** Why:
- The model is trained to answer *after* `</think>`; in the natural chain it emits
  `</think>` then `Answer:`. Closing keeps the forced probe **in-distribution** and
  apples-to-apples with how the model actually commits.
- It is the only setting under which the 1.0 invariant (below) is even meaningful —
  natural generation closed the block, so reproducing its answer requires us to close
  too.
- The related work you cited reports this choice "materially affects results," so we
  do NOT bake it in — `FORCE_CLOSE=False` (probe the latent answer *mid-think*, with
  no permission to stop) stays available behind the flag for a later comparison.

## DESIGN POINT 2 — the free correctness test: `checkpoint_full_agrees_natural`
At `fraction=1.0`, `k = n_think` (all reasoning tokens), so the forced probe sees the
entire chain and should commit to the **same letter the natural full generation did**.
Mechanism (same spirit as `locator_parser_agree`):
```
checkpoint_full_agrees_natural = (forced_letter_at_1.0 == natural_pred_letter)
```
- We use a **fixed, version-stamped inducer** (not a copy of the natural post-`</think>`
  text). That makes the invariant a real test of the whole apparatus — splicing,
  inducer, token alignment — instead of trivially true by construction.
- To keep a `False` diagnosable (benign whitespace/tokenization vs. a genuine bug),
  we log the natural post-`</think>` leading tokens alongside our inducer, plus
  `forced_letter@1.0` and `natural_pred_letter`. Disagreement is reported loudly, not
  swallowed.

## Min-length guard + visible-gap logging (no silent drops)
- `MIN_THINK_TOKENS` (propose **8**). If `n_think < MIN_THINK_TOKENS`, **drop** the
  question: print `qXX dropped: think-block too short (n_think=<n> < 8)` and produce no
  table for it. (For the single-question run, a dropped QID is a signal to pick a
  longer one — e.g. consult `results/chains.jsonl` for a `think_closed=True` chain with
  large `reasoning_len_tokens`.)
- If `think_closed` is `False` (natural gen hit the token cap without `</think>`), the
  natural committed answer is undefined → log `think_closed=False; 1.0-invariant N/A`
  and still print the trajectory (it's informative), but skip the invariant assertion.
- Per checkpoint we carry `letters_matched` exactly like `gen_chains`: if the forced
  answer token isn't a clean single A–D token, `answer_letter_entropy = None` (a visible
  gap), and we show `answer_fullvocab_entropy` so the row is never blank-with-a-guess.
- Any fraction collapse from dedup is logged (which fractions mapped to which `k`).

## Token-id-space splicing (avoid retokenization drift)
Forced input = `prompt_ids ++ reasoning_ids[:k] ++ inducer_ids` — we reuse the **exact
token ids** the model generated for the reasoning prefix, never re-encoded decoded
text. This keeps every checkpoint deterministic and is what lets the 1.0 invariant be
a clean test. To read the forced answer token's entropy: decode
`inducer_ids ++ forced_gen_ids` token-by-token, run `find_answer_token` over that list,
then map the located index back via `forced_step = located_index - len(inducer_ids)` and
take `token_entropy(out.logits[forced_step])` (4-way restricted entropy when
`letters_matched`, full-vocab otherwise). Same span-mapping discipline as `gen_chains`.

## Output (printed; eyeball before scaling)
Header: model, QID, subject, gold letter, natural pred letter, `think_closed`,
`n_think`, `FORCE_CLOSE`, `INDUCER_VERSION`, inducer repr.
Table (one row per checkpoint):
```
frac  kkeep  forced  Hletter  Hfull  conf  corr  match
0.00      0  D        1.281    6.04    35   N     Y
0.25     57  D        0.944    5.11    55   N     Y
0.50    114  B        0.402    3.88    70   Y     Y
0.75    171  B        0.061    2.10    88   Y     Y
1.00    228  B        0.009    1.74    92   Y     Y   <- natural=B
```
Footer: `checkpoint_full_agrees_natural: Y (forced@1.0=B vs natural=B)`, plus any
drop/collapse notes. Optionally also save `results/checkpoints_q{QID}.json` (small,
deterministic, commit-able) with the same fields — primary deliverable is the table.

## SBATCH (`checkpoints_job.sh`, GPU; mirrors `gen_chains_job.sh`)
```
#SBATCH --job-name=checkpoints
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem=16G
#SBATCH --time=0:30:00          # one question + a handful of short forced gens
export HF_HOME=$SCRATCH/hf_cache
export MODEL_NAME=HuggingFaceTB/SmolLM3-3B
export QID=0
mkdir -p "$HF_HOME"
srun uv run python checkpoints.py
```

## Verification / what I'll show you
After you approve and I write the code, I submit the job and show you the full stdout:
the config header, the trajectory table, and the `checkpoint_full_agrees_natural` line.
We read the table together to judge whether the entropy↓ / confidence↑ trajectory is
sensible for ONE question **before** any scaling. Nothing scales until you say so.

## One thing to confirm before I code
- Lock `FORCE_CLOSE = True` as the v1 default (my recommendation above), or do you want
  `False` first? This is the choice the paper says is load-bearing, so I want it pinned
  by you, not me.

"""Step 5: CHECKPOINT probes over all questions in DATA_DIR, deciles.

CHECKPOINTS, not early-exit: we never halt generation. For each question we generate
one full greedy reasoning chain once, then re-probe the committed answer at decile
fractions ALONG that same chain. GPU step -> runs inside jobs/checkpoints.sh on a
compute node, never on a login node.

FULLY SELF-CONTAINED: the only thing read from disk is the MMLU dataset (question +
gold). natural_pred, n_think, think_closed and correctness are ALL computed from the
in-run regenerated chain -- we never join to results/chains.jsonl (the greedy
determinism gap means that stored chain is a DIFFERENT chain; joining to it would be
invalid).

Torch-free helpers come from mc_common (build_messages, parse_answer_confidence,
find_answer_token, is_correct). The three torch/tokenizer helpers below (token_entropy,
split_think, get_letter_token_ids) live HERE as their only home: gen_chains.py is
retired (this script subsumes it), so the gpu_common.py drift gate is moot -- there is
exactly one live GPU script.

This run: deciles [0.0..1.0] (11 points), FORCE_CLOSE=True, inducer v1, over the
dataset in DATA_DIR (env-overridable; default data/mmlu_200 = 200 seeded-random MMLU
questions across subjects, fetched by fetch_mmlu.py). Deliverables (named by RUN_TAG):
  - results/<RUN_TAG>/checkpoints.jsonl        (long format, ONE ROW PER (qid, checkpoint))
  - results/<RUN_TAG>/chain_token_entropy.jsonl (ONE ROW PER qid: per-token entropy of
    the NATURAL chain, reasoning + post-think; n_think marks the boundary). GK wants
    this as a THIRD signal to correlate with answer-entropy and verbalized confidence;
    it is valid to correlate precisely because it comes from the SAME regenerated chain
    the probes slice (greedy determinism gap forbids joining across runs).
Scaling is DATA-PARALLEL via SLURM job arrays (NUM_SHARDS/SHARD_INDEX env): each array
task is one single-GPU process running the identical verified per-question code path on
qid % NUM_SHARDS == SHARD_INDEX, writing .shard<i>-suffixed outputs merged afterwards by
merge_shards.py. Deliberately NO within-GPU batching: padded batches take different
kernel paths, perturbing logits (and thus entropies, occasionally greedy argmaxes) --
a batch-size confound this measurement experiment must not have.

A per-question summary is also printed to stdout. Model load + run live under main()
so this module stays import-safe.
"""

import json
import os

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

from mc_common import (
    LETTERS,
    build_messages,
    parse_answer_confidence,
    find_answer_token,
    is_correct,
)

# --- config ---
# DATA_DIR/RUN_TAG are env-overridable so the job script picks the run; RUN_TAG
# names the output directory (results/<RUN_TAG>/checkpoints.jsonl etc.) so runs
# on different datasets never overwrite each other.
DATA_DIR = os.environ.get("DATA_DIR", "data/mmlu_200")
RUN_TAG = os.environ.get("RUN_TAG", "200q")
# Data-parallel sharding for SLURM job arrays: this process handles only qids
# with qid % NUM_SHARDS == SHARD_INDEX and suffixes its outputs .shard<i>.
# qid stays the GLOBAL dataset index, so merged shards are schema-identical to
# a single-process run. Defaults (1 shard, index 0) reproduce unsharded behavior.
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
assert NUM_SHARDS >= 1 and 0 <= SHARD_INDEX < NUM_SHARDS, \
    f"bad shard config: SHARD_INDEX={SHARD_INDEX} NUM_SHARDS={NUM_SHARDS}"
SHARD_SUFFIX = f".shard{SHARD_INDEX}" if NUM_SHARDS > 1 else ""
# 16384 (vs the old 4096): truncation is NOT random -- it censors the hard,
# long-rambling questions, biasing the correct-vs-incorrect comparison. Chains
# that still truncate stay visible via think_closed=False, never silently rerun
# (greedy would reproduce the same truncation anyway).
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "16384"))
FRACTIONS = tuple(round(0.1 * i, 1) for i in range(11))   # deciles: 0.0, 0.1, ..., 1.0
FORCE_CLOSE = True            # v1 (locked): close </think> before forcing the answer
INDUCER_VERSION = "v1"
MIN_THINK_TOKENS = 8          # below this a think-block is too short to slice; drop+log
MAX_FORCED_NEW_TOKENS = 32    # enough for " B\nConfidence: 100"; keeps probes cheap


# --- pure placement strategy (torch-free; safe to import and unit-test) ---------
def checkpoint_indices(think_token_count, strategy="fraction", fractions=FRACTIONS):
    """Return SORTED, DEDUPED think-token prefix lengths to probe.

    strategy="fraction": k = int(round(f * think_token_count)), clamped to
    [0, think_token_count]. Adjacent fractions that round to the same k are
    collapsed here; the caller logs the collapse so it is a visible gap, not a
    silent double-probe. "interval"/"step-boundary" share this signature but are
    not implemented yet (they slot in behind it without reworking callers).
    """
    if strategy != "fraction":
        raise NotImplementedError(f"strategy={strategy!r} not implemented (v1 = 'fraction')")
    ks = []
    for f in fractions:
        k = int(round(f * think_token_count))
        ks.append(max(0, min(think_token_count, k)))
    return sorted(set(ks))


# --- torch/tokenizer helpers (sole home since gen_chains.py was retired) --------
def token_entropy(logit_row):
    """Entropy in nats of one raw, unprocessed logit row [vocab]."""
    return torch.distributions.Categorical(logits=logit_row.float()).entropy().item()


def split_think(gen_ids, tok):
    """Split generated token ids on the </think> tag.

    Returns (reasoning_ids, post_think_ids, think_closed). If the tag never appears
    (truncated generation), everything counts as reasoning and think_closed is False.
    """
    ids = [int(t) for t in gen_ids]
    close_id = tok.convert_tokens_to_ids("</think>")
    if close_id is not None and close_id != tok.unk_token_id and close_id in ids:
        idx = ids.index(close_id)
        return ids[:idx], ids[idx + 1:], True
    return ids, [], False


def _encode_letters(tok, leading_space):
    """Encode each of A-D (optionally with a leading space) to its token ids."""
    enc = {}
    for letter in LETTERS:
        s = (" " if leading_space else "") + letter
        enc[letter] = tok.encode(s, add_special_tokens=False)
    return enc


def get_letter_token_ids(tok, emitted_id):
    """Resolve the 4 single-token ids for A-D, matching the convention the model
    actually used at the answer position. Returns (ids_dict_or_None, leading_space,
    matched, diagnostics); matched is True only if emitted_id is in the chosen set.
    """
    diagnostics = {}
    for ls in (True, False):
        diagnostics[ls] = _encode_letters(tok, ls)
    for ls in (True, False):
        enc = diagnostics[ls]
        if all(len(v) == 1 for v in enc.values()):
            single = {k: v[0] for k, v in enc.items()}
            if emitted_id in single.values():
                return single, ls, True, diagnostics
    for ls in (True, False):
        enc = diagnostics[ls]
        if all(len(v) == 1 for v in enc.values()):
            return {k: v[0] for k, v in enc.items()}, ls, False, diagnostics
    return None, None, False, diagnostics


# --- checkpoint probe over one chain -------------------------------------------
def intervention_label(k, n_think):
    """fraction=0.0 (k=0) is a DIFFERENT intervention -- ablate reasoning entirely --
    not merely the 0% point of the truncation trajectory. Label it so downstream
    analysis never conflates the no-reasoning baseline with partial-reasoning cuts.
    """
    if k == 0:
        return "no_reasoning_baseline"
    if k >= n_think:
        return "full_chain"
    return "partial_reasoning"


def probe_checkpoint(model, tok, device, prompt_ids, reasoning_ids, inducer_ids, k):
    """Force an answer after the first k think tokens; read (letter, confidence,
    entropies) at the forced answer token. Greedy -> deterministic. Token-id-space
    splice (no retokenization of the reasoning prefix)."""
    pieces = [
        prompt_ids,
        torch.tensor(reasoning_ids[:k], dtype=prompt_ids.dtype, device=device),
        torch.tensor(inducer_ids, dtype=prompt_ids.dtype, device=device),
    ]
    forced_input = torch.cat(pieces).unsqueeze(0)
    forced_prefix_len = forced_input.shape[1]
    with torch.no_grad():
        fout = model.generate(
            input_ids=forced_input,
            attention_mask=torch.ones_like(forced_input),
            max_new_tokens=MAX_FORCED_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
            pad_token_id=tok.eos_token_id,
        )
    forced_gen_ids = fout.sequences[0, forced_prefix_len:]

    # Locate the answer letter in [inducer tokens ++ forced tokens] (same span
    # discipline as gen_chains), then map back to the forced-generation step whose
    # logits produced the letter.
    inducer_decoded = [tok.decode([t]) for t in inducer_ids]
    forced_decoded = [tok.decode([int(t)]) for t in forced_gen_ids]
    locate_tokens = inducer_decoded + forced_decoded
    forced_text = "".join(locate_tokens)
    forced_letter, forced_conf = parse_answer_confidence(forced_text)
    located_idx, _located_letter = find_answer_token(locate_tokens)

    answer_letter_entropy = None
    answer_fullvocab_entropy = None
    letters_matched = False
    if located_idx is not None:
        forced_step = located_idx - len(inducer_ids)
        if 0 <= forced_step < len(forced_gen_ids):
            emitted_id = int(forced_gen_ids[forced_step])
            letter_ids, _ls, letters_matched, _diag = get_letter_token_ids(tok, emitted_id)
            answer_fullvocab_entropy = token_entropy(fout.logits[forced_step][0])
            # Only trust the 4-way entropy when the emitted token is a clean single
            # A-D token that matched the set -- otherwise None (a visible gap).
            if letters_matched and letter_ids is not None:
                logits4 = fout.logits[forced_step][0].float()[[letter_ids[L] for L in LETTERS]]
                answer_letter_entropy = torch.distributions.Categorical(logits=logits4).entropy().item()

    del fout
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "forced_letter": forced_letter,
        "forced_confidence": forced_conf,
        "answer_letter_entropy": answer_letter_entropy,
        "answer_fullvocab_entropy": answer_fullvocab_entropy,
        "letters_matched": letters_matched,
    }


def process_question(model, tok, device, qid, row, inducer_ids):
    """Generate one chain for a question and probe every decile checkpoint.

    Returns a dict with status "dropped" (min-length guard) or "ok" + the long-format
    rows. Raises on unexpected failure; the caller records and continues.
    """
    question, choices, gold_idx = row["question"], row["choices"], row["answer"]
    gold_letter = LETTERS[gold_idx]

    messages = build_messages(question, choices)
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    inputs = tok(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs["input_ids"][0]
    prompt_len = prompt_ids.shape[0]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,    # per-token entropy of the NATURAL chain
            pad_token_id=tok.eos_token_id,
        )
    gen_ids = out.sequences[0, prompt_len:]
    # out.logits[t] is the raw distribution that produced gen_ids[t]. Reduce each
    # row to one scalar NOW; `del out` below frees the logits before the probes,
    # so they never accumulate across questions.
    assert len(out.logits) == len(gen_ids), "logits/token count mismatch"
    per_token_entropy = [token_entropy(out.logits[t][0]) for t in range(len(gen_ids))]
    full_text = tok.decode(gen_ids, skip_special_tokens=False)
    natural_letter, natural_conf = parse_answer_confidence(full_text)
    reasoning_ids, _post_ids, think_closed = split_think(gen_ids, tok)
    n_think = len(reasoning_ids)
    # reasoning is a prefix of gen_ids, so [:n_think] is exactly the think slice
    mean_think_entropy = (
        sum(per_token_entropy[:n_think]) / n_think if n_think else None
    )
    del out
    if device == "cuda":
        torch.cuda.empty_cache()

    if n_think < MIN_THINK_TOKENS:
        return {"status": "dropped", "qid": qid, "subject": row["subject"],
                "n_think": n_think, "rows": []}

    # checkpoint k's + which fraction(s) produced each (for the frac field + collapse log)
    k_to_fracs = {}
    for f in FRACTIONS:
        k = max(0, min(n_think, int(round(f * n_think))))
        k_to_fracs.setdefault(k, []).append(f)
    ks = checkpoint_indices(n_think, "fraction", FRACTIONS)
    collapsed = {k: fr for k, fr in k_to_fracs.items() if len(fr) > 1}

    rows = []
    forced_letter_full = None
    for k in ks:
        frac = min(k_to_fracs[k])
        res = probe_checkpoint(model, tok, device, prompt_ids, reasoning_ids, inducer_ids, k)
        if k >= n_think:
            forced_letter_full = res["forced_letter"]
        rows.append({
            "qid": qid, "subject": row["subject"], "gold": gold_letter,
            "natural_pred": natural_letter, "frac": frac, "k_keep": k,
            "n_think": n_think, "think_closed": think_closed,
            "forced_letter": res["forced_letter"],
            "H_letter": res["answer_letter_entropy"],
            "H_full": res["answer_fullvocab_entropy"],
            "confidence": res["forced_confidence"],
            "correct": is_correct(res["forced_letter"], gold_idx),
            "letters_matched": res["letters_matched"],
            "intervention": intervention_label(k, n_think),
            "force_close": FORCE_CLOSE, "inducer_version": INDUCER_VERSION,
        })

    # 1.0 invariant: full-chain checkpoint must reproduce the natural answer.
    if not think_closed or forced_letter_full is None or natural_letter is None:
        inv = None
    else:
        inv = (forced_letter_full == natural_letter)
    for r in rows:
        r["checkpoint_full_agrees_natural"] = inv

    # one row per qid for chain_token_entropy.jsonl; rounded to 4 decimals
    # (1e-4 nats is far below any signal here) to keep the JSON small
    entropy_record = {
        "qid": qid, "subject": row["subject"], "gold": gold_letter,
        "natural_pred": natural_letter, "natural_confidence": natural_conf,
        "correct": is_correct(natural_letter, gold_idx),
        "think_closed": think_closed, "n_think": n_think,
        "n_gen": len(per_token_entropy),
        "mean_think_entropy": mean_think_entropy,
        "per_token_entropy": [round(h, 4) for h in per_token_entropy],
    }

    return {"status": "ok", "qid": qid, "subject": row["subject"],
            "gold": gold_letter, "natural_pred": natural_letter,
            "n_think": n_think, "think_closed": think_closed,
            "mean_think_entropy": mean_think_entropy,
            "entropy_record": entropy_record,
            "invariant": inv, "collapsed": collapsed, "rows": rows}


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_name = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM3-3B")
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
    model.eval()

    # version-stamped inducer, built once in token-id space
    close_id = tok.convert_tokens_to_ids("</think>")
    answer_inducer_ids = tok.encode("\nAnswer:", add_special_tokens=False)
    if FORCE_CLOSE:
        assert close_id is not None and close_id != tok.unk_token_id, "no </think> token"
        inducer_ids = [close_id] + answer_inducer_ids
        inducer_text = "</think>\nAnswer:"
    else:
        inducer_ids = list(answer_inducer_ids)
        inducer_text = "\nAnswer:"

    ds = load_from_disk(DATA_DIR)
    qids = [q for q in range(len(ds)) if q % NUM_SHARDS == SHARD_INDEX]

    print("=" * 84)
    print(f"CHECKPOINT PROBE — {len(qids)}/{len(ds)} questions, decile resolution (single greedy pass)")
    print("=" * 84)
    print(f"model            : {model_name}")
    print(f"data             : {DATA_DIR}   run tag: {RUN_TAG}")
    print(f"shard            : {SHARD_INDEX} of {NUM_SHARDS}   ({len(qids)} questions this shard)")
    print(f"max_new_tokens   : {MAX_NEW_TOKENS}")
    print(f"fractions        : {list(FRACTIONS)}")
    print(f"FORCE_CLOSE      : {FORCE_CLOSE}   inducer {INDUCER_VERSION} {inducer_text!r} ids={inducer_ids}")
    print(f"MIN_THINK_TOKENS : {MIN_THINK_TOKENS}")
    print("=" * 84)
    print(f"{'q':>3} {'subject':<18} {'gold':>4} {'nat':>3} {'clsd':>4} {'n_think':>7} "
          f"{'Hthink':>6} {'H@0.0':>6} {'H@1.0':>6} {'c@0.0':>5} {'c@1.0':>5} {'inv':>3} {'ckpts':>5}")

    all_rows = []
    entropy_records = []
    n_proc = n_drop = n_err = 0
    inv_y = inv_n = inv_na = 0

    for qid in qids:
        row = ds[qid]
        try:
            result = process_question(model, tok, device, qid, row, inducer_ids)
        except Exception as e:  # record-and-continue: one failure won't sink the batch
            print(f"{qid:>3} {row.get('subject', '?'):<18} FAIL: {e!r}")
            n_err += 1
            continue

        if result["status"] == "dropped":
            print(f"{qid:>3} {result['subject']:<18} DROPPED: think-block too short "
                  f"(n_think={result['n_think']} < {MIN_THINK_TOKENS})")
            n_drop += 1
            continue

        n_proc += 1
        all_rows.extend(result["rows"])
        entropy_records.append(result["entropy_record"])
        inv = result["invariant"]
        inv_y += inv is True
        inv_n += inv is False
        inv_na += inv is None

        rows = result["rows"]
        r0, r1 = rows[0], rows[-1]   # frac 0.0 ... frac 1.0 (ks sorted ascending)
        hs = lambda v: f"{v:.2f}" if v is not None else "  - "
        cs = lambda v: f"{v}" if v is not None else "-"
        invs = "Y" if inv is True else ("N" if inv is False else "NA")
        print(f"{qid:>3} {result['subject']:<18} {result['gold']:>4} "
              f"{str(result['natural_pred'] or '-'):>3} "
              f"{'Y' if result['think_closed'] else 'N':>4} {result['n_think']:>7} "
              f"{hs(result['mean_think_entropy']):>6} "
              f"{hs(r0['H_letter']):>6} {hs(r1['H_letter']):>6} "
              f"{cs(r0['confidence']):>5} {cs(r1['confidence']):>5} {invs:>3} {len(rows):>5}")
        if result["collapsed"]:
            print(f"     fraction collapse (deduped): {result['collapsed']}")

    # --- deliverables ------------------------------------------------------------
    # 1) long-format checkpoint rows, one per (qid, checkpoint)
    # 2) natural-chain per-token entropy, one row per qid
    os.makedirs(f"results/{RUN_TAG}", exist_ok=True)
    out_path = f"results/{RUN_TAG}/checkpoints{SHARD_SUFFIX}.jsonl"
    with open(out_path, "w") as fh:
        for r in all_rows:
            fh.write(json.dumps(r) + "\n")
    entropy_path = f"results/{RUN_TAG}/chain_token_entropy{SHARD_SUFFIX}.jsonl"
    with open(entropy_path, "w") as fh:
        for r in entropy_records:
            fh.write(json.dumps(r) + "\n")

    # --- aggregate (no silent drops) --------------------------------------------
    print("=" * 84)
    print(f"=== AGGREGATE (shard {SHARD_INDEX}/{NUM_SHARDS}: {len(qids)} of {len(ds)} questions) ===")
    print(f"processed          : {n_proc}")
    print(f"dropped (min-len)  : {n_drop}")
    print(f"errors             : {n_err}")
    print(f"invariant Y/N/NA   : {inv_y} / {inv_n} / {inv_na}   (NA = think_closed False or no natural answer)")
    print(f"rows written       : {len(all_rows)}  -> {out_path}")
    print(f"entropy records    : {len(entropy_records)}  -> {entropy_path}  (must equal processed)")
    print(f"(expected rows = sum over processed questions of their checkpoint counts)")


if __name__ == "__main__":
    main()

"""Step 5 (smallest version): CHECKPOINT probes along ONE reasoning chain.

CHECKPOINTS, not early-exit: we never halt generation. We generate one full greedy
reasoning chain once, then re-probe the committed answer at fractions ALONG that
same chain. GPU step -> runs inside checkpoints_job.sh on a compute node, never on a
login node.

Torch-free helpers come from mc_common (build_messages, parse_answer_confidence,
find_answer_token, is_correct). The three torch/tokenizer helpers below are MIRRORED
VERBATIM from gen_chains.py (see the debt tag): unify into a shared gpu_common module
BEFORE scaling to deciles x 20 -- that gate is recorded in NOTES_resume.md so the
duplication cannot reach the scaled run (we were already bitten by ANSWER_MARKER_RE
drift).

Smallest version: ONE question (QID env, default 0), fractions [0,.25,.5,.75,1.0],
FORCE_CLOSE=True (in-distribution commit-probe: the model is trained to answer after
</think>). Prints a config header + trajectory table and checks the
checkpoint_full_agrees_natural invariant. No scaling, no sampling, no plots. The
model load and run live under main() so this module is import-safe (pure helpers can
be unit-tested later without loading weights).
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
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
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


# --- helpers MIRRORED VERBATIM from gen_chains.py -------------------------------
# unify into a shared gpu_common module before scaling (gated in NOTES_resume.md)
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


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_name = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM3-3B")
    qid = int(os.environ.get("QID", "0"))
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
    model.eval()

    ds = load_from_disk("data/mmlu_20")
    row = ds[qid]
    question, choices, gold_idx = row["question"], row["choices"], row["answer"]
    gold_letter = LETTERS[gold_idx]

    # --- generate the full chain ONCE (greedy), exactly like gen_chains ---------
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
            max_new_tokens=4096,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=False,   # smallest version reads entropy at FORCED tokens only
            pad_token_id=tok.eos_token_id,
        )
    gen_ids = out.sequences[0, prompt_len:]
    full_text = tok.decode(gen_ids, skip_special_tokens=False)
    natural_letter, natural_conf = parse_answer_confidence(full_text)

    reasoning_ids, post_ids, think_closed = split_think(gen_ids, tok)
    n_think = len(reasoning_ids)

    # --- build the version-stamped inducer in token-id space --------------------
    close_id = tok.convert_tokens_to_ids("</think>")
    answer_inducer_ids = tok.encode("\nAnswer:", add_special_tokens=False)
    if FORCE_CLOSE:
        assert close_id is not None and close_id != tok.unk_token_id, "no </think> token"
        inducer_ids = [close_id] + answer_inducer_ids
        inducer_text = "</think>\nAnswer:"
    else:
        inducer_ids = list(answer_inducer_ids)
        inducer_text = "\nAnswer:"

    # --- config header ----------------------------------------------------------
    print("=" * 72)
    print("CHECKPOINT PROBE (smallest version: 1 question)")
    print("=" * 72)
    print(f"model               : {model_name}")
    print(f"QID                 : {qid}   subject={row['subject']}")
    print(f"gold letter         : {gold_letter}")
    print(f"natural pred letter : {natural_letter}   (verbalized conf={natural_conf})")
    print(f"think_closed        : {think_closed}")
    print(f"n_think tokens      : {n_think}")
    print(f"FORCE_CLOSE         : {FORCE_CLOSE}   (inducer {INDUCER_VERSION})")
    print(f"inducer text / ids  : {inducer_text!r}  ids={inducer_ids}")
    # natural post-</think> leading tokens -- logged so a 1.0 disagreement is
    # diagnosable (benign whitespace vs real apparatus bug).
    print(f"natural post-</think>: {[tok.decode([t]) for t in post_ids[:6]]}")
    print(f"fractions           : {list(FRACTIONS)}")
    print("=" * 72)

    # --- min-length guard (visible drop, no silent skip) ------------------------
    if n_think < MIN_THINK_TOKENS:
        print(f"q{qid} DROPPED: think-block too short to slice "
              f"(n_think={n_think} < MIN_THINK_TOKENS={MIN_THINK_TOKENS}). "
              f"Pick a longer-chain QID.")
        return

    # checkpoint k's + which fraction(s) produced each (for the frac column + collapse log)
    k_to_fracs = {}
    for f in FRACTIONS:
        k = max(0, min(n_think, int(round(f * n_think))))
        k_to_fracs.setdefault(k, []).append(f)
    ks = checkpoint_indices(n_think, "fraction", FRACTIONS)
    collapsed = {k: fr for k, fr in k_to_fracs.items() if len(fr) > 1}

    # --- probe each checkpoint --------------------------------------------------
    print(f"{'frac':>5} {'kkeep':>6} {'forced':>6} {'Hletter':>8} {'Hfull':>6} "
          f"{'conf':>4} {'corr':>4} {'match':>5}  intervention")
    rows_out = []
    forced_letter_full = None
    for k in ks:
        frac = min(k_to_fracs[k])
        res = probe_checkpoint(model, tok, device, prompt_ids, reasoning_ids, inducer_ids, k)
        interv = intervention_label(k, n_think)
        correct = is_correct(res["forced_letter"], gold_idx)
        if k >= n_think:
            forced_letter_full = res["forced_letter"]

        h4 = res["answer_letter_entropy"]
        hful = res["answer_fullvocab_entropy"]
        h4s = f"{h4:.3f}" if h4 is not None else "  -  "
        hfuls = f"{hful:.2f}" if hful is not None else " -  "
        corr = "Y" if correct is True else ("N" if correct is False else "?")
        confv = res["forced_confidence"]
        print(f"{frac:5.2f} {k:6d} {str(res['forced_letter'] or '-'):>6} {h4s:>8} {hfuls:>6} "
              f"{str(confv if confv is not None else '-'):>4} {corr:>4} "
              f"{'Y' if res['letters_matched'] else 'N':>5}  {interv}")
        rows_out.append({"fraction": frac, "k": k, "intervention": interv,
                         "correct": correct, **res})

    # --- 1.0 invariant: full-chain checkpoint must reproduce the natural answer --
    if not think_closed:
        inv = None
        inv_str = "NA (think_closed=False: no natural committed answer)"
    elif forced_letter_full is None or natural_letter is None:
        inv = None
        inv_str = f"NA (forced@1.0={forced_letter_full}, natural={natural_letter})"
    else:
        inv = (forced_letter_full == natural_letter)
        inv_str = f"{'Y' if inv else 'N'} (forced@1.0={forced_letter_full} vs natural={natural_letter})"
    print("-" * 72)
    print(f"checkpoint_full_agrees_natural: {inv_str}")
    if collapsed:
        print(f"fraction collapse (deduped): {collapsed}")

    # --- save a small, deterministic record (commit-able) -----------------------
    os.makedirs("results", exist_ok=True)
    out_path = f"results/checkpoints_q{qid}.json"
    with open(out_path, "w") as fh:
        json.dump({
            "qid": qid, "subject": row["subject"], "gold_letter": gold_letter,
            "natural_pred_letter": natural_letter, "natural_confidence": natural_conf,
            "think_closed": think_closed, "n_think_tokens": n_think,
            "force_close": FORCE_CLOSE, "inducer_version": INDUCER_VERSION,
            "inducer_text": inducer_text, "fractions": list(FRACTIONS),
            "checkpoints": rows_out,
            "checkpoint_full_agrees_natural": inv,
        }, fh, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

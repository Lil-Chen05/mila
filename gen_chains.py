"""Step 4: generate reasoning chains for all 20 MMLU questions (greedy).

This is the first script that loads the model, so it only runs inside the GPU
job (gen_chains_job.sh) on a compute node -- never on a login node.

Pure-logic helpers come from mc_common (build_messages, parse_answer_confidence,
find_answer_token, is_correct). find_answer_token is pure over the decoded token
strings -- no torch, no tokenizer -- so it lives in mc_common too, anchoring on
the shared ANSWER_RE. The helpers that need the tokenizer/torch live here,
because this is the first step where those are available:
  - token_entropy        : entropy (nats) of one raw logit row
  - split_think          : split generated ids into reasoning / post-</think>
  - get_letter_token_ids : resolve the 4 A-D token ids matching what the model
                           actually emits after "Answer: " (leading space or not)

Scaled version: all 20 questions, greedy, one run each. Per-question failures
are recorded and skipped (one bad question won't sink the batch). Sampling and
per-token storage come later.
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


# --- deferred helpers (need tokenizer / torch, so they live in the GPU step) ---

def token_entropy(logit_row):
    """Entropy in nats of one raw, unprocessed logit row [vocab].

    Cast to float32 first (the model runs in bf16) so the softmax/log math is
    stable. Categorical(logits=...) does log_softmax internally.
    """
    return torch.distributions.Categorical(logits=logit_row.float()).entropy().item()


def split_think(gen_ids, tok):
    """Split generated token ids on the </think> tag.

    Returns (reasoning_ids, post_think_ids, think_closed). The close-tag id is
    resolved at runtime rather than hardcoded. If the tag never appears
    (truncated generation), everything counts as reasoning and think_closed is
    False.
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
    actually used at the answer position.

    After "Answer: " a BPE tokenizer usually emits the letter as a leading-space
    token (" C"), but it could be bare ("C"). We try both, require each letter
    to be a single token, and prefer the convention whose id set contains the
    emitted answer token. Returns (ids_dict_or_None, leading_space, matched,
    diagnostics) where `matched` is True only if the emitted id is in the chosen
    set -- if it is False the answer-letter entropy below is meaningless and we
    say so loudly.
    """
    diagnostics = {}
    for ls in (True, False):
        enc = _encode_letters(tok, ls)
        diagnostics[ls] = enc
    # Prefer a single-token convention that contains the emitted id.
    for ls in (True, False):
        enc = diagnostics[ls]
        if all(len(v) == 1 for v in enc.values()):
            single = {k: v[0] for k, v in enc.items()}
            if emitted_id in single.values():
                return single, ls, True, diagnostics
    # Fall back to any single-token convention (assumption likely broken).
    for ls in (True, False):
        enc = diagnostics[ls]
        if all(len(v) == 1 for v in enc.values()):
            return {k: v[0] for k, v in enc.items()}, ls, False, diagnostics
    return None, None, False, diagnostics


# --- model load (unchanged from the verified single-question version) ---

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
print(f"Model loaded: {model.__class__.__name__}")


# --- per-question processing (same analysis as the verified version) ---

def process_row(qid, row):
    """Generate one chain (greedy) for a row and return its record dict.

    Raises on unexpected failure; the caller records the error and continues.
    """
    question = row["question"]
    choices = row["choices"]
    gold_idx = row["answer"]
    gold_letter = LETTERS[gold_idx]

    messages = build_messages(question, choices)
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    inputs = tok(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
            pad_token_id=tok.eos_token_id,
        )

    # out.logits[t] is the raw distribution that produced gen_ids[t].
    gen_ids = out.sequences[0, prompt_len:]
    assert len(out.logits) == len(gen_ids), "logits/token count mismatch"

    per_token_entropy = [token_entropy(out.logits[t][0]) for t in range(len(gen_ids))]
    decoded_tokens = [tok.decode([int(t)]) for t in gen_ids]

    reasoning_ids, post_ids, think_closed = split_think(gen_ids, tok)
    # TODO: for "behavior-B" chains (answer committed INSIDE <think>, then more
    # text after </think>), reasoning_len_tokens over-counts -- it includes the
    # in-think Answer:/Confidence: lines, which also leak into mean_reasoning_
    # entropy. Deferred on purpose; revisit before the early-exit step.
    n_reasoning = len(reasoning_ids)
    mean_reasoning_entropy = (
        sum(per_token_entropy[:n_reasoning]) / n_reasoning if n_reasoning else None
    )

    full_text = tok.decode(gen_ids, skip_special_tokens=False)
    # parse takes the LAST match, so a decoy "Answer:" in the reasoning is safe.
    pred_letter, confidence = parse_answer_confidence(full_text)
    correct = is_correct(pred_letter, gold_idx)

    # locate the answer-letter token via the shared "last Answer:" anchor,
    # regardless of the </think> boundary.
    answer_pos, located_letter = find_answer_token(decoded_tokens)
    # parser (full_text) vs locator (joined pieces) should agree on the letter;
    # record disagreement so it is visible at scale, not buried.
    if located_letter is not None and pred_letter is not None:
        locator_parser_agree = (located_letter == pred_letter)
    else:
        locator_parser_agree = None

    answer_letter_entropy = None
    answer_fullvocab_entropy = None
    answer_letter_probs = None
    letter_ids = None
    leading_space = None
    letters_matched = False
    emitted_id = None
    emitted_repr = None

    if answer_pos is not None:
        emitted_id = int(gen_ids[answer_pos])
        emitted_repr = decoded_tokens[answer_pos]
        letter_ids, leading_space, letters_matched, _diag = get_letter_token_ids(tok, emitted_id)
        answer_fullvocab_entropy = per_token_entropy[answer_pos]
        # Only trust the 4-way entropy when the emitted token is a clean single
        # A-D token that matched the set. A fused letter ("B\n") fails this ->
        # we store None (a visible gap), never a guessed number.
        if letters_matched and letter_ids is not None:
            row_logits = out.logits[answer_pos][0].float()
            idxs = [letter_ids[L] for L in LETTERS]
            logits4 = row_logits[idxs]
            probs4 = torch.softmax(logits4, dim=-1)
            answer_letter_entropy = torch.distributions.Categorical(logits=logits4).entropy().item()
            answer_letter_probs = {L: float(p) for L, p in zip(LETTERS, probs4)}

    # free GPU memory before the next question (20 sequential generations)
    del out
    if device == "cuda":
        torch.cuda.empty_cache()

    record = {
        "qid": qid,
        "run_id": 0,
        "subject": row["subject"],
        "question": question,
        "choices": choices,
        "gold_idx": gold_idx,
        "gold_letter": gold_letter,
        "think_closed": think_closed,
        "reasoning_len_tokens": n_reasoning,
        "pred_letter": pred_letter,
        "located_letter": located_letter,
        "locator_parser_agree": locator_parser_agree,
        "verbalized_confidence": confidence,
        "correct": correct,
        "answer_token_index": answer_pos,
        "answer_token_id": emitted_id,
        "answer_token_repr": emitted_repr,
        "leading_space": leading_space,
        "letter_token_ids": letter_ids,
        "letters_matched": letters_matched,
        "answer_letter_entropy": answer_letter_entropy,
        "answer_letter_probs": answer_letter_probs,
        "answer_fullvocab_entropy": answer_fullvocab_entropy,
        "mean_reasoning_entropy": mean_reasoning_entropy,
        "full_text": full_text,
        "error": None,
    }
    # diagnostics for the one-time reference table; not saved to disk.
    record["_diag"] = (emitted_id, emitted_repr, leading_space, letter_ids, letters_matched)
    return record


# --- run all 20 questions: per-question try/except, compact reporting ---

ds = load_from_disk("data/mmlu_20")
print(f"\nProcessing {len(ds)} questions (greedy, 1 run each)...\n")

records = []
printed_letter_table = False

for qid in range(len(ds)):
    row = ds[qid]
    try:
        rec = process_row(qid, row)
    except Exception as e:  # record-and-continue: one failure won't sink the batch
        print(f"q{qid:02d} [{row.get('subject')}]  FAIL: {e!r}")
        records.append({"qid": qid, "run_id": 0, "subject": row.get("subject"),
                        "error": repr(e)})
        continue

    # print the resolved letter-token table once, from the first parsed question
    emitted_id, emitted_repr, leading_space, letter_ids, letters_matched = rec.pop("_diag")
    if not printed_letter_table and rec["answer_token_index"] is not None:
        print("=" * 60)
        print("LETTER TOKEN RESOLUTION (reference, from first parsed question)")
        print("=" * 60)
        print(f"Emitted answer token : id={emitted_id}  decode={emitted_repr!r}  "
              f"piece={tok.convert_ids_to_tokens([emitted_id])[0]!r}")
        print(f"Resolved convention  : leading_space={leading_space}")
        if letter_ids is not None:
            for L in LETTERS:
                lid = letter_ids[L]
                mark = "  <-- emitted" if lid == emitted_id else ""
                print(f"    {L!r} -> id {lid:<8} decode={tok.decode([lid])!r}{mark}")
        print(f"Emitted id is in the resolved set: {letters_matched}")
        print("=" * 60)
        printed_letter_table = True

    # one compact line per question (ASCII only -- safe on any stdout locale)
    corr = "Y" if rec["correct"] is True else ("N" if rec["correct"] is False else "?")
    h4 = rec["answer_letter_entropy"]
    hful = rec["answer_fullvocab_entropy"]
    h4s = f"{h4:.3f}" if h4 is not None else "  -  "
    hfuls = f"{hful:.2f}" if hful is not None else " -  "
    disagree = "  [LOC!=PRED]" if rec["locator_parser_agree"] is False else ""
    print(f"q{qid:02d} [{rec['subject']}]  "
          f"pred={rec['pred_letter']} gold={rec['gold_letter']} corr={corr}  "
          f"conf={rec['verbalized_confidence']}  "
          f"closed={'Y' if rec['think_closed'] else 'N'} "
          f"match={'Y' if rec['letters_matched'] else 'N'}  "
          f"H4={h4s} Hful={hfuls}  rlen={rec['reasoning_len_tokens']}{disagree}")
    records.append(rec)

# --- end-of-run aggregate: all 20, categorized (no silent drops) ---
n = len(records)
n_err = sum(1 for r in records if r.get("error"))
nonerr = [r for r in records if not r.get("error")]
n_answered = sum(1 for r in nonerr if r.get("pred_letter") is not None)
n_truncated = sum(1 for r in nonerr
                  if not r.get("think_closed") and r.get("pred_letter") is None)
n_closed_unparsed = sum(1 for r in nonerr
                        if r.get("think_closed") and r.get("pred_letter") is None)
n_closed = sum(1 for r in records if r.get("think_closed"))
n_matched = sum(1 for r in records if r.get("letters_matched"))
n_disagree = sum(1 for r in records if r.get("locator_parser_agree") is False)
n_correct = sum(1 for r in records if r.get("correct") is True)
confs = [r["verbalized_confidence"] for r in records
         if r.get("verbalized_confidence") is not None]
h4vals = [r["answer_letter_entropy"] for r in records
          if r.get("answer_letter_entropy") is not None]
mean_conf = sum(confs) / len(confs) if confs else None
mean_h4 = sum(h4vals) / len(h4vals) if h4vals else None

print(f"\n=== AGGREGATE (n={n}) ===")
print("categories (partition the {0} rows):".format(n))
print(f"  answered            : {n_answered}")
print(f"  truncated (no </think>, no answer): {n_truncated}")
print(f"  closed-but-unparsed : {n_closed_unparsed}")
print(f"  errors              : {n_err}")
print(f"think_closed          : {n_closed}/{n}")
print(f"letters_matched       : {n_matched}/{n}  (answer-letter entropy coverage)")
print(f"locator!=parser       : {n_disagree}/{n}  (should be 0)")
prim = f"{n_correct}/{n} = {n_correct / n:.2f}" if n else "n/a"
seco = f"{n_correct}/{n_answered} = {n_correct / n_answered:.2f}" if n_answered else "n/a"
print(f"accuracy (primary, /all)     : {prim}")
print(f"accuracy (secondary, /answered): {seco}")
print(f"mean verbalized confidence : {mean_conf}")
print(f"mean answer-letter entropy : {mean_h4}")

# --- save lean records (no per-token arrays; greedy is reproducible) ---
os.makedirs("results", exist_ok=True)
with open("results/chains.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print(f"\nSaved {len(records)} records to results/chains.jsonl")

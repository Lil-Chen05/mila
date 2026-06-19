"""Step 3: generate ONE reasoning chain (greedy) and measure its uncertainty.

This is the first script that loads the model, so it only runs inside the GPU
job (gen_chains_job.sh) on a compute node -- never on a login node.

Pure-logic helpers come from mc_common (build_messages, parse_answer_confidence,
is_correct). The helpers that need the tokenizer/torch live here, because this
is the first step where those are available:
  - token_entropy        : entropy (nats) of one raw logit row
  - split_think          : split generated ids into reasoning / post-</think>
  - get_letter_token_ids : resolve the 4 A-D token ids matching what the model
                           actually emits after "Answer: " (leading space or not)

First version: one question (row 0), greedy. Scaling and sampling come later.
"""

import json
import os

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

from mc_common import LETTERS, build_messages, parse_answer_confidence, is_correct

LETTER_SET = set(LETTERS)


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


def find_answer_token_index(decoded_tokens, start_idx):
    """Index of the answer-letter token: the LAST token at or after start_idx
    whose stripped text is a single A-D letter. (In the post-</think> region the
    only such token is the answer letter; 'Answer'/'Confidence' don't match.)"""
    found = None
    for i in range(start_idx, len(decoded_tokens)):
        if decoded_tokens[i].strip() in LETTER_SET:
            found = i
    return found


# --- main ---

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

# One question only (row 0) for this first inspection.
ds = load_from_disk("data/mmlu_20")
row = ds[0]
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
        max_new_tokens=2048,
        do_sample=False,
        return_dict_in_generate=True,
        output_logits=True,
        pad_token_id=tok.eos_token_id,
    )

# out.logits is a tuple, one [batch, vocab] tensor per generated step.
# out.logits[t] is the distribution that produced gen_ids[t] (raw, unprocessed).
gen_ids = out.sequences[0, prompt_len:]
assert len(out.logits) == len(gen_ids), "logits/token count mismatch"

per_token_entropy = [token_entropy(out.logits[t][0]) for t in range(len(gen_ids))]
decoded_tokens = [tok.decode([int(t)]) for t in gen_ids]

reasoning_ids, post_ids, think_closed = split_think(gen_ids, tok)
n_reasoning = len(reasoning_ids)
mean_reasoning_entropy = (
    sum(per_token_entropy[:n_reasoning]) / n_reasoning if n_reasoning else None
)

full_text = tok.decode(gen_ids, skip_special_tokens=False)
reasoning_text = tok.decode(reasoning_ids, skip_special_tokens=False)
post_text = tok.decode(post_ids, skip_special_tokens=True)

# Parse the answer/confidence from the decoded text (parse takes the LAST match,
# so a decoy "Answer:" inside the reasoning won't fool it).
pred_letter, confidence = parse_answer_confidence(full_text)
correct = is_correct(pred_letter, gold_idx)

# Locate the answer-letter token and compute the two entropies at that position.
search_start = (n_reasoning + 1) if think_closed else 0
answer_pos = find_answer_token_index(decoded_tokens, search_start)

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
    letter_ids, leading_space, letters_matched, diag = get_letter_token_ids(tok, emitted_id)
    answer_fullvocab_entropy = per_token_entropy[answer_pos]
    if letter_ids is not None:
        row_logits = out.logits[answer_pos][0].float()
        idxs = [letter_ids[L] for L in LETTERS]
        logits4 = row_logits[idxs]
        probs4 = torch.softmax(logits4, dim=-1)
        answer_letter_entropy = torch.distributions.Categorical(logits=logits4).entropy().item()
        answer_letter_probs = {L: float(p) for L, p in zip(LETTERS, probs4)}

# --- prominent verification banner (the key thing to eyeball) ---
print("\n" + "=" * 60)
print("LETTER TOKEN VERIFICATION  (confirm this is correct!)")
print("=" * 60)
if answer_pos is None:
    print("!! No answer-letter token found (model may not have emitted")
    print("   'Answer: <letter>'; check think_closed / truncation below).")
else:
    print(f"Emitted answer token : id={emitted_id}  decode={emitted_repr!r}  "
          f"piece={tok.convert_ids_to_tokens([emitted_id])[0]!r}")
    print(f"Resolved convention  : leading_space={leading_space}")
    print("Letter token ids used for the 4-way answer-letter entropy:")
    if letter_ids is not None:
        for L in LETTERS:
            lid = letter_ids[L]
            mark = "  <-- emitted" if lid == emitted_id else ""
            print(f"    {L!r} -> id {lid:<8} decode={tok.decode([lid])!r}  "
                  f"piece={tok.convert_ids_to_tokens([lid])[0]!r}{mark}")
    print(f"Emitted id is in the resolved set: {letters_matched}")
    if not letters_matched:
        print("!! WARNING: emitted answer token is NOT in the resolved letter set.")
        print("   The leading-space assumption is wrong -> answer-letter entropy")
        print("   is NOT trustworthy. Fix get_letter_token_ids before scaling.")
print("=" * 60)

# --- the rest of the inspection output ---
print("\n=== Question ===")
print("Subject :", row["subject"])
print("Question:", question)
for L, c in zip(LETTERS, choices):
    print(f"  {L}. {c}")

print("\n=== Reasoning chain (decoded, special tokens shown) ===")
print(reasoning_text)
print(f"\n[reasoning length: {n_reasoning} tokens | think_closed: {think_closed}]")

print("\n=== Post-</think> text ===")
print(post_text)

print("\n=== Parsed result ===")
print("Predicted letter :", pred_letter)
print("Gold letter      :", gold_letter, f"(int {gold_idx})")
print("Correct          :", correct)
print("Verbalized conf  :", confidence)

print("\n=== Entropy ===")
print("Answer-letter entropy (4-way, nats):", answer_letter_entropy,
      f"(max possible {torch.log(torch.tensor(4.0)).item():.3f})")
print("Answer-letter probs                :", answer_letter_probs)
print("Full-vocab entropy at answer (nats):", answer_fullvocab_entropy)
print("Mean reasoning entropy (nats)      :", mean_reasoning_entropy)

# --- save one structured record (full per-token detail is small at N=1) ---
os.makedirs("results", exist_ok=True)
record = {
    "qid": 0,
    "run_id": 0,
    "subject": row["subject"],
    "question": question,
    "choices": choices,
    "gold_idx": gold_idx,
    "gold_letter": gold_letter,
    "think_closed": think_closed,
    "reasoning_len_tokens": n_reasoning,
    "pred_letter": pred_letter,
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
    "per_token_entropy": per_token_entropy,
    "tokens": decoded_tokens,
    "full_text": full_text,
}
with open("results/chains.jsonl", "w") as f:
    f.write(json.dumps(record) + "\n")
print("\nSaved 1 record to results/chains.jsonl")

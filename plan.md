# Step 3 plan — `test_mc_common.py`

Login-safe pytest for the hardened parser. **No model, no torch, no tokenizer,
no dataset** — pure-Python string tests over `mc_common`. Run with:

```
uv run pytest test_mc_common.py -q
```

## Return contracts being asserted against (read from the code, not guessed)

- `parse_answer_confidence(text)` → a **2-tuple `(letter, confidence)`**:
  - `letter` = last match `.upper()`, else `None`.
  - `confidence` = `max(0, min(100, int(last)))`, else `None`.
  - So: full no-match → `(None, None)`; answer-only → `("B", None)`;
    confidence-only → `(None, 90)`.
- `find_answer_token(decoded_tokens)` → a **2-tuple `(token_index, located_letter)`**:
  - no marker → `(None, None)`.
  - matched + located → `(i, located_letter)` where `i` is the token whose
    char-span contains the letter and `located_letter = m.group(1).upper()`.
  - Assertions check **both** elements, never "is not None".

## (1) Decoy — pins last-match, not just found-vs-None

```python
DECOY = (
    "Working through it: my first instinct was Answer: A with Confidence: 10.\n"
    "But re-checking the units flips the conclusion.\n"
    "Answer: C\n"
    "Confidence: 88"
)
# assert parse_answer_confidence(DECOY) == ("C", 88)
```

Earlier mid-reasoning `Answer: A` / `Confidence: 10`; real final `Answer: C` /
`Confidence: 88` — different letter AND different number. A regression to
**first**-match (`findall(...)[0]`) would return `("A", 10)` — a wrong, non-None
answer — so `== ("C", 88)` fails loudly on exactly that regression.

## (2)+(3) `find_answer_token` index tests — realistic multi-token spans

`decoded_tokens` are built so the pieces **concatenate to the reconstructed
string** (the precondition the char->token span map relies on). The marker spans
several tokens *before* the letter — no single-token cheat. Worked markdown case:

```python
toks = ["**", "Answer", ":", "**", " B"]     # "".join(toks) == "**Answer:** B"
# spans:  **=[0,2)  Answer=[2,8)  :=[8,9)  **=[9,11)  " B"=[11,13)
# letter 'B' is char 12  ->  falls in [11,13)  ->  token index 4
idx, letter = find_answer_token(toks)
assert (idx, letter) == (4, "B")
assert toks[idx] == " B"        # returned index decodes to the token holding the letter
```

The three index cases (each asserts `(idx, letter)` and the token content) plus a
no-marker case:

| case     | `decoded_tokens`                        | `"".join`       | expect `(idx, letter)` | `toks[idx]` |
|----------|-----------------------------------------|-----------------|------------------------|-------------|
| bare     | `["Answer", ":", " B"]`                 | `Answer: B`     | `(2, "B")`             | `" B"`      |
| drift    | `["Answer", " :", " B"]`                | `Answer : B`    | `(2, "B")`             | `" B"`      |
| markdown | `["**", "Answer", ":", "**", " B"]`     | `**Answer:** B` | `(4, "B")`             | `" B"`      |

No-marker: `find_answer_token(["no", " marker", " here"]) == (None, None)`.

## (4) None-discipline — asserts the documented `(None, …)` shapes exactly

```python
assert parse_answer_confidence("there is no answer or confidence here") == (None, None)
assert parse_answer_confidence("Answer: B")      == ("B", None)   # answer only
assert parse_answer_confidence("Confidence: 90") == (None, 90)    # confidence only
```

## Full test set (one file, `test_mc_common.py`, pytest)

1. **Per-shape parser** (`@pytest.mark.parametrize`, each asserts the full
   `(letter, conf)` tuple):
   - `**Answer:** B` / `95`
   - `*Answer:* C` / `60`
   - `**Answer: D**` / `40`
   - `Answer : A` / `80`
   - ` answer: b ` → `("B", 33)`  (lowercase + surrounding spaces, `.upper()` applied)
   - backticked `` `Answer:` B `` / `50`
   - baseline `Answer: C` / `75`
2. **Unbalanced markdown (extra case requested):** `**Answer:* B` → assert letter
   is `"B"`. Pins permissive behavior; does NOT forbid the unbalanced `**…*`.
3. **Decoy** — `== ("C", 88)` (see section 1).
4. **Combined q06** — synthetic string modeling the documented q06 shape
   (mid-reasoning decoy + markdown final), exercising last-match and markdown
   together:
   ```python
   Q06 = (
       "Hmm, Answer: A seems plausible at first glance.\n"
       "After eliminating distractors, the remaining option is correct.\n"
       "**Answer:** B\n"
       "**Confidence:** 72"
   )
   # assert parse_answer_confidence(Q06) == ("B", 72)
   ```
   Synthetic (not the literal generated `results/chains.jsonl` q06 text) so the
   unit test stays deterministic and decoupled from a regenerated artifact.
5. **None-discipline** — the three assertions in section 4.
6. **Upper clamp** — `parse_answer_confidence("Answer: A\nConfidence: 250") == ("A", 100)`.
7. **`find_answer_token`** — the three index cases + the no-marker case in
   sections 2/3.

## Verification

`uv run pytest test_mc_common.py -q`, then show the actual pass output.

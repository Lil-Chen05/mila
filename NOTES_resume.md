# Resume notes

## Session 2026-06-23 — Harden answer/confidence parsing (PLANNING ONLY)

**Status: nothing implemented. Working tree is clean. No code was changed.**
The whole session stayed in plan mode; I never reached implementation (ExitPlanMode
was rejected, then you stopped). The only artifact written is the plan file:
`~/.claude/plans/context-comp-400-llm-uncertainty-stateful-sloth.md`.

### What was changed this session
- Nothing in the repo. `mc_common.py`, `gen_chains.py` are untouched; no
  `test_mc_common.py` was created. Only the (out-of-repo) plan file exists.

### What was left unfinished (the agreed plan, not yet done)
1. **`mc_common.py`** — replace `_ANSWER_RE`/`_CONF_RE` with ONE shared, markdown-
   tolerant compiled object plus a confidence regex:
   ```python
   _EMPH = r"[*_`]*"   # optional markdown run: ** * _ `
   ANSWER_RE = re.compile(rf"{_EMPH}Answer{_EMPH}\s*:\s*{_EMPH}\s*([ABCD])", re.IGNORECASE)
   CONF_RE   = re.compile(rf"{_EMPH}Confidence{_EMPH}\s*:\s*{_EMPH}\s*(\d{{1,3}})", re.IGNORECASE)
   ```
   Keep `findall(...)[-1]` (last-match) and the [0,100] clamp. Group 1 stays JUST
   `([ABCD])` so `m.start(1)` still lands on the letter.
   Then **move `find_answer_token` here whole** (verified pure: operates only on
   `decoded_tokens` strings, no torch/tokenizer), changing its regex ref to the
   shared `ANSWER_RE`. mc_common must stay torch-free / tokenizer-free.
2. **`gen_chains.py`** — import `ANSWER_RE` + `find_answer_token` from mc_common;
   delete the duplicate `ANSWER_MARKER_RE` (line 34) and the local
   `find_answer_token` (lines 104-129); remove now-unused `import re`; update the
   module docstring. Do NOT touch model/generation/tensor logic.
3. **`test_mc_common.py`** (NEW, login-safe, pytest) — per-shape parser tests
   (`**Answer:** B`, `*Answer:* C`, `**Answer: D**`, `Answer : A`, ` answer: b `,
   backticked) asserting BOTH letter+confidence; baseline `Answer: C`; decoy test;
   combined q06 (`decoy + **Answer:** B / **Confidence:** 95` -> B/95); failure
   discipline (no match -> None); and direct `find_answer_token` tests landing the
   index on the letter for bare `Answer: B`, drift `Answer : B`, markdown
   `**Answer:** B`, plus decoy + no-match.

### Why (key findings to recall cold)
- Bug: `**Answer:** B` is missed because `\s*` can't consume the `*` after the colon.
- The anchor is **duplicated and already drifting**: parser used `Answer:` (no space
  before colon) while locator used `Answer\s*:` — so `Answer : B` is located but not
  parsed today. Fix = ONE shared compiled object imported by both.
- There is **no existing test file** (despite the task wording); pytest is the
  dev-dep convention.

### Test status
Tests do **not exist yet** — `test_mc_common.py` was never created, so there is
nothing to run. The code is **not mid-edit**; it is simply not started. Once
implemented, verify with: `uv run pytest -q test_mc_common.py`.

### Next action on resume
Re-enter the plan (above / plan file), implement steps 1-3, run the tests, show
output, and do NOT commit until confirmed.

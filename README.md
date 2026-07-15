# Uncertainty signals in a reasoning LLM

A research experiment on the Mila SLURM cluster studying uncertainty in a
reasoning LLM (SmolLM3-3B, thinking mode) on MMLU: token-level entropy,
verbalized confidence, and answer correctness, plus a checkpoint probe that
re-asks for the committed answer at decile fractions along each reasoning
chain and watches entropy/confidence evolve.

- **Model:** HuggingFaceTB/SmolLM3-3B (greedy, 1 run/question)
- **Dataset:** MMLU (`cais/mmlu`, config `all`, test split), seeded-random
  subsets saved to `data/`
- **Findings so far:** `analysis/200q/FINDINGS.md` (main run) and
  `analysis/20q/FINDINGS.md` (pilot)

## Layout

```
scripts/    pipeline code; run from the repo root: uv run python scripts/<task>.py
jobs/       sbatch scripts, one per task: sbatch jobs/<task>.sh
tests/      pytest tests for the pure-logic helpers (login-node safe)
results/    raw run outputs, one directory per RUN_TAG (20q/, 200q/)
analysis/   analysis scripts + per-run figures/, tables/, FINDINGS.md
docs/       design notes and specs
legacy/     retired experiments (TriviaQA warm-up), kept for reference
logs/       SLURM stdout (gitignored)
data/       saved HF datasets (gitignored, regenerate with fetch_mmlu)
```

## Pipeline

1. `sbatch jobs/fetch_mmlu.sh` — CPU; streams N MMLU questions and saves them
   to `data/mmlu_<N>` (never full-downloads the dataset).
2. `sbatch jobs/checkpoints.sh` — GPU; the core experiment. One full greedy
   reasoning chain per question, then answer probes at decile fractions along
   the same chain. Data-parallel via a SLURM job array (`NUM_SHARDS` shards);
   writes `results/<RUN_TAG>/{checkpoints,chain_token_entropy}.shard<i>.jsonl`.
3. `uv run python scripts/merge_shards.py` — login-node safe; validates and
   merges shards into `results/<RUN_TAG>/{checkpoints,chain_token_entropy}.jsonl`.
4. `sbatch jobs/dump_questions.sh` — CPU; exports question text to
   `results/<RUN_TAG>/questions.json` so analysis never touches HF datasets.
5. `uv run python analysis/analyze_200q.py` (CPU job) — writes figures, tables,
   and `FINDINGS.md` into `analysis/<RUN_TAG>/`.

## Conventions

- Dependencies via `uv` (`uv add`, `uv run`); no pip, no manual venv activation.
- Models/datasets are only ever loaded inside SLURM jobs, never on login nodes;
  `HF_HOME=$SCRATCH/hf_cache` keeps downloads in scratch.
- Tests: `uv run pytest -q` from the repo root.
- See `CLAUDE.md` for the full working agreement and cluster safety rules.

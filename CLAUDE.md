# CLAUDE.md — Working Agreement for This Project

## What this project is
A research experiment on the Mila SLURM cluster studying uncertainty in a
reasoning LLM: token-level entropy, verbalized confidence, and answer
correctness, plus an early-exit probe (truncate reasoning at fixed intervals,
force an answer, watch entropy/confidence evolve).
- Model: HuggingFaceTB/SmolLM3-3B (thinking mode on)
- Dataset: MMLU (cais/mmlu, config "all", test split)
- First runnable version: 20 questions, 1 run/question, greedy. Scaling to
  sampled runs comes LATER, only after the core pipeline works.

## HOW YOU MUST WORK (most important rule)
- Work ONE step at a time. Implement a single step, then STOP.
- After each step, wait for my EXPLICIT confirmation before starting the next.
  Do not read ahead, scaffold future files, or "just also do" the next step.
- Before writing code for a step, briefly state your plan for THAT STEP ONLY
  and let me approve it.
- Never enable auto-accept behavior. Ask before running commands or writing
  files. I want to review every action.
- When I'm verifying a step, your job is to help me understand and test it —
  not to push forward.

## CLUSTER SAFETY RULES (non-negotiable)
- NEVER load a model or dataset on a login node — not even one question, not
  even on CPU, not even as a "quick test". All model/data loading happens
  INSIDE a SLURM job on a compute node.
- Pure logic (string parsing, prompt formatting, helper functions with no
  model/data) MAY be tested on the login node. Anything that imports torch to
  load weights, or loads a dataset, goes in a job.
- Compute nodes are ephemeral; $SLURM_TMPDIR is wiped when a job ends. Anything
  that must persist goes in the home directory or $SCRATCH.
- GPU jobs only when a GPU is actually needed. Data prep and analysis are
  CPU-only jobs (no --gpus-per-task line).

## ENVIRONMENT CONVENTIONS
- Dependency manager is uv. Use `uv add <pkg>` to add dependencies. NEVER use
  pip. Never suggest `source .venv/bin/activate` — `uv run` handles the
  environment automatically.
- Run scripts from the repo root with `uv run python scripts/<file>.py`. In
  jobs: `srun uv run python scripts/<file>.py`.
- File naming convention: each task is a pair, `scripts/<task>.py` +
  `jobs/<task>.sh`. Shared helpers go in `scripts/mc_common.py` (imported, no
  job script). Tests live in `tests/` (pytest; `pythonpath = ["scripts"]` in
  pyproject.toml makes `import mc_common` work).
- Outputs are namespaced per experiment run: `results/<RUN_TAG>/` and
  `analysis/<RUN_TAG>/` (e.g. `20q`, `200q`) — no run suffixes in filenames.
- Retired code is archived under `legacy/`, never left at the repo root.
- Model downloads must be cached in scratch, not home. Assume
  `export HF_HOME=$SCRATCH/hf_cache` is set in job scripts before any HF load,
  so models download once and persist across jobs.

## DATA HANDLING
- Datasets are fetched with streaming + take, then saved to disk
  (`streaming=True`, `.take(N)`, `save_to_disk(...)`). NEVER full-download a
  dataset and then `.select()` — that downloads everything first.
- Jobs load saved data with `load_from_disk(...)`; they do not re-download.

## GIT CONVENTIONS
- SLURM logs land in `logs/` (`#SBATCH --output=logs/slurm-%j.out`) and are
  gitignored ON PURPOSE. Do NOT add them back or commit them — they regenerate
  every run.
- Commit code, small result files (JSON/CSV/PNG), and config. Do NOT commit
  datasets, model weights, or large parquet/arrow files.
- Write clear, scoped commit messages describing the one step just completed.

## SBATCH BASELINE
Job scripts follow this shape (tune resources per step; CPU jobs omit the GPU line):
    #!/bin/bash
    #SBATCH --job-name=<task>
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=4
    #SBATCH --gpus-per-task=l40s:1   # GPU steps only
    #SBATCH --mem=16G
    #SBATCH --time=1:00:00
    #SBATCH --output=logs/slurm-%j.out   # %A_%a for array jobs; logs/ is tracked via .gitkeep
    export HF_HOME=$SCRATCH/hf_cache
    srun uv run python scripts/<task>.py

## PIPELINE ORDER (one gate at a time; current file names)
1. scripts/mc_common.py — shared helpers; tests/ runs on the login node.
2. scripts/fetch_mmlu.py (+ jobs/fetch_mmlu.sh) — CPU; stream+save questions
   to data/; then I inspect the data.
3. scripts/checkpoints.py (+ jobs/checkpoints.sh) — GPU; the core experiment
   (subsumed the earlier gen_chains.py/early_exit.py, now in legacy history).
   Sharded via SLURM job array; shards merged by scripts/merge_shards.py
   (login-node safe, pure JSON).
4. scripts/dump_questions.py (+ jobs/dump_questions.sh) — CPU; exports question
   text so analysis stays login-safe.
5. analysis/analyze_*.py — CPU job; produces CSVs/figures/findings into
   analysis/<RUN_TAG>/.
(The LLM-judge step is deferred and only added if the above works end to end.)

## WHEN UNSURE
- If a product/cluster detail might be version-specific or out of date, say so
  and point me to docs rather than guessing.
- If a step seems to require breaking a rule above, STOP and raise it with me
  instead of working around it.
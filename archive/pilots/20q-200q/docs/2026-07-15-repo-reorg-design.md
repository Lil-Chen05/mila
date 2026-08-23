# Repo Reorganization Design — 2026-07-15

## Problem

The repo root holds ~20 loose files mixing two generations of work: the live
MMLU uncertainty pipeline (`checkpoints.py`, `mc_common.py`, `fetch_mmlu.py`,
`merge_shards.py`, `dump_questions.py` and their `*_job.sh` pairs) and a
retired TriviaQA warm-up experiment (`*_trivia.py`, `fetch_job.sh`,
`infer_job.sh`, `inspect_job.sh`) plus scaffolding leftovers (`dummy.py`,
`main.py`, `job.sh`). Results and analysis outputs distinguish experiments by
filename suffix (`_20q` / `_200q` / `fig200q_`), which does not scale as runs
accumulate. `data/trivia_50/*.arrow` files are git-tracked despite the
`data/` ignore rule (committed before the rule existed), violating the
project's own no-datasets convention. Untracked `slurm-*.out` files pile up at
the repo root.

## Goal

A standard research-codebase layout: library/entry code, cluster job scripts,
tests, docs, per-experiment outputs, and archived legacy work each in their
own directory, with git history preserved for every moved file.

## Decisions (approved in brainstorming)

1. **Retired code is archived, not deleted** — moved to `legacy/trivia/` with
   a short README explaining what it was.
2. **Flat `scripts/` + `jobs/` layout**, not an installable `src/` package —
   right-sized for this repo; `uv run python scripts/<task>.py` from the repo
   root, `sbatch jobs/<task>.sh`. The `<task>.py` + `<task>_job.sh` pairing
   convention becomes `scripts/<task>.py` + `jobs/<task>.sh`.
3. **Per-experiment output directories** — `results/20q/`, `results/200q/`,
   `analysis/20q/`, `analysis/200q/`; filenames drop the `_20q`/`_200q`/
   `fig200q` suffixes because the directory now carries that information.

## Target layout

```
my-project/
├─ README.md                  project summary + pipeline map (currently empty)
├─ CLAUDE.md                  conventions updated to new paths
├─ pyproject.toml             + [tool.pytest.ini_options]
├─ .gitignore                 updated patterns
├─ scripts/                   mc_common.py, fetch_mmlu.py, checkpoints.py,
│                             merge_shards.py, dump_questions.py
├─ jobs/                      fetch_mmlu.sh, checkpoints.sh, dump_questions.sh
├─ tests/                     test_mc_common.py
├─ analysis/                  analyze_checkpoints.py, analyze_200q.py
│  ├─ 20q/                    FINDINGS.md, figures/, tables/
│  └─ 200q/                   FINDINGS.md, OUTLIERS.md, figures/, tables/
├─ results/
│  ├─ 20q/                    checkpoints.jsonl, chain_token_entropy.jsonl,
│  │                          checkpoints_q0.json
│  └─ 200q/                   checkpoints.jsonl, chain_token_entropy.jsonl,
│                             questions.json (+ untracked shard files)
├─ docs/                      plan.md, NOTES_resume.md (historical, unedited),
│                             superpowers/specs/ (this spec)
├─ legacy/trivia/             fetch_trivia.py, infer_trivia.py,
│                             inspect_trivia.py, fetch_job.sh, infer_job.sh,
│                             inspect_job.sh, job.sh, dummy.py, main.py,
│                             README.md (new)
├─ logs/                      SLURM output (gitignored except .gitkeep)
└─ data/                      unchanged on disk, fully untracked
```

## Behavior-preserving mechanics

- **Imports**: no Python import changes. `checkpoints.py`'s
  `from mc_common import ...` keeps working because Python puts the script's
  own directory (`scripts/`) on `sys.path`.
- **Job scripts**: one-line change each to
  `srun uv run python scripts/<task>.py`, plus
  `#SBATCH --output=logs/slurm-%j.out` (`%A_%a` for the array job) so logs
  stop accumulating at the root. `logs/` must exist at submit time, hence a
  tracked `.gitkeep`.
- **Output paths**: scripts write `results/{RUN_TAG}/<name>.jsonl` instead of
  `results/<name>_{RUN_TAG}.jsonl`; `merge_shards.py` reads/writes the same
  scheme; analysis scripts read the new paths and write into
  `analysis/20q|200q/`.
- **pytest**: `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and
  `pythonpath = ["scripts"]` so `import mc_common` works from `tests/`.
- **Git hygiene**: all moves via `git mv` in a move-only commit (keeps rename
  detection and `git log --follow` reliable); content edits land in a separate
  commit. `git rm -r --cached data/trivia_50` untracks the arrow files while
  leaving them on disk. `.gitignore`'s shard pattern becomes
  `results/*/*.shard*.jsonl`; `logs/*` ignored except `.gitkeep`.
- **Generated outputs**: committed FINDINGS/figures/tables contain old
  internal filenames; they are fully script-generated, so a CPU-job re-run of
  both analysis scripts regenerates them consistently rather than hand-editing.

## Out of scope

- No deletion of legacy code (archived instead).
- No installable package / packaging changes beyond the pytest table.
- No changes to experiment logic, model, dataset handling, or sharding.
- No push to the remote unless explicitly requested.

## Verification

1. `uv run pytest -q` passes from the repo root (login-node safe: pure string
   logic).
2. `bash -n jobs/*.sh` passes.
3. Repo-wide grep for stale references (`_job.sh`, `checkpoints_200q`,
   `fig200q`, etc.) is clean outside `docs/` and `legacy/` (historical).
4. A CPU SLURM job re-runs both analysis scripts against the moved results;
   outputs regenerate under `analysis/20q/` and `analysis/200q/` with
   suffix-free names.
5. The next real GPU run of `sbatch jobs/checkpoints.sh` is the final
   end-to-end confirmation; not required for this reorg.

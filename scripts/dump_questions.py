"""Export question text/metadata from the saved dataset to small plain JSON.

Loads a saved HF dataset -> runs INSIDE a CPU job (dump_questions_job.sh), never
on a login node. After this one-time export, analysis scripts read only the
small JSON and stay login-safe for fast plot iteration.
"""

import json
import os

from datasets import load_from_disk

DATA_DIR = os.environ.get("DATA_DIR", "data/mmlu_200")
RUN_TAG = os.environ.get("RUN_TAG", "200q")

ds = load_from_disk(DATA_DIR)
rows = [{"qid": i, **ds[i]} for i in range(len(ds))]

os.makedirs("results", exist_ok=True)
out_path = f"results/questions_{RUN_TAG}.json"
with open(out_path, "w") as fh:
    json.dump(rows, fh, indent=1)
print(f"wrote {len(rows)} questions -> {out_path}")

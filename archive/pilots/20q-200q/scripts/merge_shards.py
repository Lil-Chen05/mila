"""Merge sharded checkpoint outputs into the final per-run JSONL deliverables.

Pure JSON file handling -- no torch, no datasets, no model -- so this is
login-node safe and needs no job script. Reads the .shard<i> files written by
the checkpoints.py job array and writes the merged, qid-sorted:
  - results/<RUN_TAG>/checkpoints.jsonl
  - results/<RUN_TAG>/chain_token_entropy.jsonl

Checks, loudly (nonzero exit on violation):
  - every expected shard file exists (a lost/failed array task must not become
    a silently smaller dataset);
  - each record's qid actually belongs to its shard (qid % NUM_SHARDS == i);
  - no qid appears in more than one shard.
Missing qids are REPORTED but tolerated: checkpoints.py drops min-length think
blocks and records per-question errors without emitting rows, so absences must
be cross-checked against the logs/slurm-<jobid>_<i>.out logs, not papered over.
"""

import json
import os
import sys

RUN_TAG = os.environ.get("RUN_TAG", "200q")
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "8"))
N_QUESTIONS = int(os.environ.get("N_QUESTIONS", "200"))

FAMILIES = (
    # (basename, sort key, one row per qid?)
    (f"results/{RUN_TAG}/checkpoints", lambda r: (r["qid"], r["k_keep"]), False),
    (f"results/{RUN_TAG}/chain_token_entropy", lambda r: r["qid"], True),
)

failed = False
for base, sort_key, one_per_qid in FAMILIES:
    rows = []
    qid_shard = {}          # qid -> shard that produced it (dup/ownership checks)
    for i in range(NUM_SHARDS):
        path = f"{base}.shard{i}.jsonl"
        if not os.path.exists(path):
            print(f"FAIL: missing shard file {path} (array task {i} lost?)")
            failed = True
            continue
        with open(path) as fh:
            shard_rows = [json.loads(line) for line in fh]
        for r in shard_rows:
            qid = r["qid"]
            if qid % NUM_SHARDS != i:
                print(f"FAIL: {path} contains qid {qid} (belongs to shard {qid % NUM_SHARDS})")
                failed = True
            if one_per_qid and qid in qid_shard:
                print(f"FAIL: qid {qid} in both shard {qid_shard[qid]} and shard {i}")
                failed = True
            qid_shard[qid] = i
        rows.extend(shard_rows)
        print(f"  {path}: {len(shard_rows)} rows")

    rows.sort(key=sort_key)
    out_path = f"{base}.jsonl"
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    qids = sorted({r["qid"] for r in rows})
    missing = sorted(set(range(N_QUESTIONS)) - set(qids))
    print(f"{out_path}: {len(rows)} rows, {len(qids)}/{N_QUESTIONS} qids")
    if missing:
        print(f"  MISSING qids (check slurm logs for drops/errors): {missing}")
    print()

if failed:
    sys.exit("merge FAILED -- see messages above; merged files may be incomplete")
print("merge OK")

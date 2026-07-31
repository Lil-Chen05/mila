#!/usr/bin/env python3
"""Audit an explicit operator-authorized release of one shard writer lock."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import socket

from part1_runtime import LockMetadata, LockedShardSession


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--finish-pending", action="store_true")
    parser.add_argument("--study-id")
    parser.add_argument("--model-run-id")
    parser.add_argument("--model-run-manifest-hash")
    parser.add_argument("--shard-id")
    parser.add_argument("--reason", required=True, help="Nonblank operator justification")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    if args.finish_pending:
        session = LockedShardSession.finish_pending_takeover(
            args.shard_root,
            operator_override_reason=args.reason,
        )
        event = session.store.inspect().audit_events[-1]
        session.close()
        print(json.dumps({
            "event_id": event["event_id"],
            "event_type": event["event_type"],
            "shard_id": event["shard_id"],
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0
    missing = [
        name
        for name in (
            "study_id", "model_run_id", "model_run_manifest_hash", "shard_id"
        )
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(f"missing arguments for new takeover: {', '.join(missing)}")
    owner = LockMetadata.new(
        study_id=args.study_id,
        model_run_id=args.model_run_id,
        shard_id=args.shard_id,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        acquired_at=now,
    )
    session = LockedShardSession.operator_unlock(
        args.shard_root,
        owner=owner,
        model_run_manifest_hash=args.model_run_manifest_hash,
        operator_reason=args.reason,
        event_timestamp=now,
        execution_context={"hostname": owner.hostname, "pid": owner.pid},
    )
    event = session.store.inspect().audit_events[-1]
    session.close()
    print(
        json.dumps(
            {
                "event_id": event["event_id"],
                "event_type": event["event_type"],
                "shard_id": event["shard_id"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

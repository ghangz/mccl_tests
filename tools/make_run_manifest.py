#!/usr/bin/env python3
"""Create a JSON manifest of MCCL benchmark runs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def build_manifest(gpus: list[int], benchmarks: list[str], dtypes: list[str], min_bytes: str, max_bytes: str) -> dict:
    runs = []
    for gpu_count, benchmark, dtype in itertools.product(gpus, benchmarks, dtypes):
        runs.append(
            {
                "gpu_count": gpu_count,
                "benchmark": benchmark,
                "dtype": dtype,
                "min_bytes": min_bytes,
                "max_bytes": max_bytes,
                "command": f"MCCL_DTYPE={dtype} MCCL_MIN_BYTES={min_bytes} MCCL_MAX_BYTES={max_bytes} ./mccl.sh {gpu_count} {benchmark}",
            }
        )
    return {"run_count": len(runs), "runs": runs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="1,2,4")
    parser.add_argument("--benchmarks", default="all_reduce_perf,all_gather_perf,reduce_scatter_perf")
    parser.add_argument("--dtypes", default="float16,bfloat16")
    parser.add_argument("--min-bytes", default="1K")
    parser.add_argument("--max-bytes", default="1G")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = build_manifest(
        [int(item) for item in args.gpus.split(",") if item],
        [item for item in args.benchmarks.split(",") if item],
        [item for item in args.dtypes.split(",") if item],
        args.min_bytes,
        args.max_bytes,
    )
    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate a compact MCCL benchmark matrix as JSON."""

from __future__ import annotations

import argparse
import json


DEFAULT_BENCHES = ["all_reduce_perf", "all_gather_perf", "reduce_scatter_perf"]


def build_matrix(gpus: list[int], benches: list[str], min_bytes: str, max_bytes: str, iters: int) -> list[dict]:
    return [
        {"bench": bench, "gpus": gpu, "min_bytes": min_bytes, "max_bytes": max_bytes, "iters": iters}
        for gpu in gpus
        for bench in benches
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MCCL benchmark matrix JSON.")
    parser.add_argument("--gpus", default="1,2,4")
    parser.add_argument("--benches", default=",".join(DEFAULT_BENCHES))
    parser.add_argument("--min-bytes", default="1K")
    parser.add_argument("--max-bytes", default="1G")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    gpus = [int(item) for item in args.gpus.split(",") if item]
    benches = [item for item in args.benches.split(",") if item]
    print(json.dumps({"matrix": build_matrix(gpus, benches, args.min_bytes, args.max_bytes, args.iters)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

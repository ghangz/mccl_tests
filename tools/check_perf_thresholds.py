#!/usr/bin/env python3
"""Check parsed MCCL perf JSON against minimum thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, float | int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload)
    if not isinstance(rows, list):
        raise ValueError("input must contain a list of rows")
    return rows


def evaluate(
    rows: list[dict[str, float | int]],
    *,
    min_algbw_gbps: float | None = None,
    max_time_us: float | None = None,
    min_size_bytes: int | None = None,
) -> dict[str, object]:
    failures = []
    for row in rows:
        size_bytes = int(row["size_bytes"])
        if min_size_bytes is not None and size_bytes < min_size_bytes:
            continue
        if min_algbw_gbps is not None and float(row.get("algbw_gbps", 0.0)) < min_algbw_gbps:
            failures.append({"reason": "algbw_gbps", "row": row})
        if max_time_us is not None and float(row.get("time_us", 0.0)) > max_time_us:
            failures.append({"reason": "time_us", "row": row})
    return {"row_count": len(rows), "failure_count": len(failures), "passed": not failures, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--min-algbw-gbps", type=float)
    parser.add_argument("--max-time-us", type=float)
    parser.add_argument("--min-size-bytes", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = json.dumps(
        evaluate(
            load_rows(args.report),
            min_algbw_gbps=args.min_algbw_gbps,
            max_time_us=args.max_time_us,
            min_size_bytes=args.min_size_bytes,
        ),
        indent=2,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

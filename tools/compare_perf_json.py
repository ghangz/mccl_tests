#!/usr/bin/env python3
"""Compare two MCCL benchmark JSON files and report regressions.

The tool accepts either a list of records or a dict with a ``results`` list.
Each record is matched by benchmark/name, message size, dtype, and GPU count
when those fields are available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


KEY_FIELDS = ("benchmark", "name", "op", "size", "bytes", "dtype", "gpus")
BANDWIDTH_FIELDS = ("busbw", "algbw", "bandwidth", "gbps")


def _load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        records = data.get("results", data.get("records", []))
    else:
        records = data
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a benchmark record list")
    return [record for record in records if isinstance(record, dict)]


def _record_key(record: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    parts: list[tuple[str, str]] = []
    for field in KEY_FIELDS:
        if field in record and record[field] is not None:
            parts.append((field, str(record[field])))
    if not parts:
        raise ValueError(f"record has no comparable key fields: {record}")
    return tuple(parts)


def _bandwidth(record: dict[str, Any]) -> float:
    for field in BANDWIDTH_FIELDS:
        if field in record and record[field] is not None:
            return float(record[field])
    raise ValueError(f"record has no bandwidth field: {record}")


def compare(base_path: Path, candidate_path: Path, regression_threshold: float) -> dict[str, Any]:
    base = {_record_key(record): record for record in _load_records(base_path)}
    candidate = {_record_key(record): record for record in _load_records(candidate_path)}
    rows: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []

    for key in sorted(base.keys() & candidate.keys()):
        base_bw = _bandwidth(base[key])
        candidate_bw = _bandwidth(candidate[key])
        change_pct = ((candidate_bw - base_bw) / base_bw * 100.0) if base_bw else 0.0
        row = {
            "key": dict(key),
            "base_bandwidth": base_bw,
            "candidate_bandwidth": candidate_bw,
            "change_percent": round(change_pct, 4),
        }
        rows.append(row)
        if change_pct < -regression_threshold:
            regressions.append(row)

    return {
        "matched": len(rows),
        "base_only": [dict(key) for key in sorted(base.keys() - candidate.keys())],
        "candidate_only": [dict(key) for key in sorted(candidate.keys() - base.keys())],
        "regression_threshold_percent": regression_threshold,
        "regressions": regressions,
        "comparisons": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="baseline benchmark JSON")
    parser.add_argument("candidate", type=Path, help="candidate benchmark JSON")
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=5.0,
        help="bandwidth drop percentage treated as a regression",
    )
    parser.add_argument("--output", type=Path, help="write comparison JSON to this path")
    args = parser.parse_args()

    report = compare(args.base, args.candidate, args.regression_threshold)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 1 if report["regressions"] else 0


if __name__ == "__main__":
    sys.exit(main())

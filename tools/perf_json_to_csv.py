#!/usr/bin/env python3
"""Convert MCCL benchmark JSON records to CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


FIELDS = ("benchmark", "size", "dtype", "gpus", "algbw", "busbw", "latency_us")


def _records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("results", data) if isinstance(data, dict) else data
    return [item for item in records if isinstance(item, dict)]


def convert(input_path: Path, output_path: Path) -> int:
    records = _records(input_path)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    count = convert(args.input, args.output)
    print(f"wrote {count} rows to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

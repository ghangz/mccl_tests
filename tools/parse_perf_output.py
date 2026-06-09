#!/usr/bin/env python3
"""Parse MCCL/NCCL perf output rows into JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_rows(text: str) -> list[dict[str, float | int]]:
    rows = []
    for line in text.splitlines():
        line = line.lstrip("\ufeff")
        parts = line.split()
        if len(parts) < 6 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        numeric_tail: list[float] = []
        for item in parts[2:]:
            try:
                numeric_tail.append(float(item))
            except ValueError:
                continue
        if len(numeric_tail) < 3:
            continue
        rows.append(
            {
                "size_bytes": int(parts[0]),
                "count": int(parts[1]),
                "time_us": numeric_tail[1],
                "algbw_gbps": numeric_tail[2],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse MCCL perf output into JSON rows.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = {"rows": parse_rows(args.log.read_text(encoding="utf-8", errors="replace"))}
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

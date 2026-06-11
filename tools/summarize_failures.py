#!/usr/bin/env python3
"""Extract likely failure lines from MCCL run logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FAILURE_RE = re.compile(r"(error|failed|failure|timeout|segmentation|abort|invalid)", re.IGNORECASE)


def summarize(path: Path) -> dict[str, object]:
    failures = []
    failure_count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if FAILURE_RE.search(line):
                failure_count += 1
                if len(failures) < 100:
                    failures.append({"line": line_no, "text": line.strip()})
    return {"path": str(path), "failure_count": failure_count, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = json.dumps(summarize(args.log), indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

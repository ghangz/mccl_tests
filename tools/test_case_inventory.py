#!/usr/bin/env python3
"""Export a JSON inventory of MCCL test files."""

from __future__ import annotations


import argparse
import json
from pathlib import Path


PATTERNS = ['src/*.cu', 'verifiable/*.cu', 'function/*.sh']


def inventory(root: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for pattern in PATTERNS:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                files.append({"path": rel, "bytes": path.stat().st_size, "pattern": pattern})
    by_pattern: dict[str, int] = {}
    for item in files:
        by_pattern[item["pattern"]] = by_pattern.get(item["pattern"], 0) + 1
    return {"root": str(root), "count": len(files), "by_pattern": by_pattern, "files": files}


def self_test() -> None:
    data = inventory(Path.cwd())
    assert isinstance(data["files"], list)
    print(json.dumps({"ok": True, "count": data["count"]}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    print(json.dumps(inventory(Path(args.root)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

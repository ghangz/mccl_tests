#!/usr/bin/env python3
"""Lint launch scripts for common MACA deployment footguns."""

from __future__ import annotations


import argparse
import json
from pathlib import Path


GLOBS = ['*.sh', 'function/*.sh']
REQUIRED_TOKENS = ['MCCL', 'LD_LIBRARY_PATH']


def lint(root: Path) -> dict[str, object]:
    findings: list[dict[str, str]] = []
    for pattern in GLOBS:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            rel = path.relative_to(root).as_posix()
            if "\r\n" in text:
                findings.append({"path": rel, "severity": "warning", "message": "script uses CRLF line endings"})
            if "set -e" not in text and path.suffix in (".sh", ""):
                findings.append({"path": rel, "severity": "warning", "message": "shell script does not enable fail-fast mode"})
            for token in REQUIRED_TOKENS:
                if token not in text:
                    findings.append({"path": rel, "severity": "info", "message": f"missing optional token: {token}"})
    return {"finding_count": len(findings), "findings": findings}


def self_test() -> None:
    data = lint(Path.cwd())
    assert "findings" in data
    print(json.dumps({"ok": True, "finding_count": data["finding_count"]}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    print(json.dumps(lint(Path(args.root)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

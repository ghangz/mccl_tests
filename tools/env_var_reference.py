#!/usr/bin/env python3
"""Build a JSON reference for environment variables used by MCCL scripts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")


def collect(root: Path) -> list[dict[str, str]]:
    items = []
    for path in sorted(root.glob("*.sh")) + sorted((root / "function").glob("*.sh")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in ENV_RE.finditer(text):
            items.append(
                {
                    "name": match.group(1),
                    "default": match.group(2),
                    "path": path.relative_to(root).as_posix(),
                }
            )
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = {"variable_count": len(collect(args.root)), "variables": collect(args.root)}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

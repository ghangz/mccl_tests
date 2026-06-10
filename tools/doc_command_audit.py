#!/usr/bin/env python3
"""Audit README shell references against files in the MCCL test repo."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SCRIPT_RE = re.compile(r"(?:bash\s+)?([A-Za-z0-9_./-]+\.sh)")


def audit(readme: Path, root: Path) -> dict[str, object]:
    references = []
    for match in SCRIPT_RE.finditer(readme.read_text(encoding="utf-8", errors="replace")):
        script = match.group(1)
        references.append({"script": script, "exists": (root / script).exists()})
    missing = [entry for entry in references if not entry["exists"]]
    return {"reference_count": len(references), "missing_count": len(missing), "references": references}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = json.dumps(audit(args.readme, args.root), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Expand MCCL benchmark size arguments into a concrete sweep plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


UNITS = {
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
}


def parse_size(text: str) -> int:
    text = text.strip().upper()
    if not text:
        raise ValueError("size string cannot be empty")
    if text[-1] in UNITS:
        return int(float(text[:-1]) * UNITS[text[-1]])
    return int(text)


def expand_sizes(min_bytes: int, max_bytes: int, step_factor: float | None = None, step_bytes: int | None = None) -> list[int]:
    if min_bytes <= 0 or max_bytes < min_bytes:
        raise ValueError("invalid size range")
    if (step_factor is None) == (step_bytes is None):
        raise ValueError("exactly one of step_factor or step_bytes must be set")
    if step_bytes is not None and step_bytes <= 0:
        raise ValueError("step_bytes must be positive")

    sizes = []
    current = min_bytes
    while current <= max_bytes:
        sizes.append(current)
        if step_factor is not None:
            next_size = int(current * step_factor)
            if next_size <= current:
                raise ValueError("step_factor must increase the size")
            current = next_size
        else:
            current += step_bytes
    return sizes


def summarize(min_size: str, max_size: str, step_factor: float | None, step_bytes: str | None) -> dict[str, object]:
    min_bytes = parse_size(min_size)
    max_bytes = parse_size(max_size)
    byte_step = parse_size(step_bytes) if step_bytes else None
    sizes = expand_sizes(min_bytes, max_bytes, step_factor=step_factor, step_bytes=byte_step)
    return {"count": len(sizes), "sizes_bytes": sizes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-bytes", required=True)
    parser.add_argument("--max-bytes", required=True)
    parser.add_argument("--step-factor", type=float)
    parser.add_argument("--step-bytes")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        payload = summarize(args.min_bytes, args.max_bytes, args.step_factor, args.step_bytes)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

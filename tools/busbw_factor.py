#!/usr/bin/env python3
"""Compute MCCL bus-bandwidth correction factors for collectives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def correction_factor(collective: str, ranks: int) -> float:
    if ranks <= 0:
        raise ValueError("ranks must be positive")
    collective = collective.lower().replace("_", "").replace("-", "")
    if collective == "allreduce":
        return 2 * (ranks - 1) / ranks
    if collective in {"allgather", "reducescatter"}:
        return (ranks - 1) / ranks
    if collective in {"broadcast", "reduce"}:
        return 1.0
    raise ValueError(f"unsupported collective: {collective}")


def summarize(collective: str, ranks: int, algbw_gbps: float | None = None) -> dict[str, object]:
    if algbw_gbps is not None and algbw_gbps < 0:
        raise ValueError("algbw_gbps must be non-negative")
    factor = correction_factor(collective, ranks)
    payload: dict[str, object] = {"collective": collective, "ranks": ranks, "factor": factor}
    if algbw_gbps is not None:
        payload["algbw_gbps"] = algbw_gbps
        payload["busbw_gbps"] = algbw_gbps * factor
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collective")
    parser.add_argument("ranks", type=int)
    parser.add_argument("--algbw-gbps", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = json.dumps(summarize(args.collective, args.ranks, args.algbw_gbps), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Parse MCCL cluster host specifications into a JSON layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict


class HostEntry(TypedDict):
    host: str
    process_count: int
    rank_start: int
    rank_end: int
    node_index: int


def parse_host_spec(spec: str) -> list[HostEntry]:
    hosts: list[HostEntry] = []
    current_rank = 0
    for index, item in enumerate(part.strip() for part in spec.split(",") if part.strip()):
        host, sep, process_count = item.rpartition(":")
        if not sep or not host:
            raise ValueError(f"invalid host entry: {item!r}")
        ranks = int(process_count)
        if ranks <= 0:
            raise ValueError(f"process count must be positive: {item!r}")
        hosts.append(
            {
                "host": host,
                "process_count": ranks,
                "rank_start": current_rank,
                "rank_end": current_rank + ranks - 1,
                "node_index": index,
            }
        )
        current_rank += ranks
    if not hosts:
        raise ValueError("host specification is empty")
    return hosts


def summarize(spec: str) -> dict[str, object]:
    hosts = parse_host_spec(spec)
    return {
        "node_count": len(hosts),
        "total_processes": sum(entry["process_count"] for entry in hosts),
        "hosts": hosts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host_spec", help="Cluster host spec like 10.0.0.1:8,10.0.0.2:8")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = json.dumps(summarize(args.host_spec), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

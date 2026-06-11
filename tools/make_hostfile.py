#!/usr/bin/env python3
"""Generate an OpenMPI hostfile for MCCL multi-node tests."""

from __future__ import annotations

import argparse
from pathlib import Path


def render_hostfile(hosts: list[str], slots: int) -> str:
    if slots <= 0:
        raise ValueError("slots must be positive")
    valid_hosts = [host.strip() for host in hosts if host.strip()]
    if not valid_hosts:
        raise ValueError("At least one non-empty host must be provided")
    return "\n".join(f"{host} slots={slots}" for host in valid_hosts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", action="append", required=True, help="host name or IP; repeat for multiple hosts")
    parser.add_argument("--slots", type=int, default=8, help="GPU/process slots per host")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.write_text(render_hostfile(args.host, args.slots), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

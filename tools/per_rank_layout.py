#!/usr/bin/env python3
"""Preview the per-rank MACA_VISIBLE_DEVICES layout used by MCCL helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ODD_NODE_LAYOUT = "5,7,6,4,3,0,2,1"
EVEN_NODE_LAYOUT = "0,1,2,3,4,5,6,7"


def build_layout(world_size: int, visible_device_per_rank: int = 1, max_gpus_per_node: int = 8) -> dict[str, object]:
    if world_size <= 0:
        raise ValueError("world size must be positive")
    if visible_device_per_rank <= 0:
        raise ValueError("visible_device_per_rank must be positive")
    if max_gpus_per_node <= 0:
        raise ValueError("max_gpus_per_node must be positive")
    if visible_device_per_rank > max_gpus_per_node:
        raise ValueError("visible_device_per_rank cannot exceed max_gpus_per_node")
    if visible_device_per_rank != 1 and max_gpus_per_node != 8:
        raise ValueError("multi-device layouts require max_gpus_per_node to be 8")

    ranks = []
    for rank in range(world_size):
        node = rank // max_gpus_per_node
        if visible_device_per_rank == 1:
            visible = str(rank % max_gpus_per_node)
        else:
            visible = EVEN_NODE_LAYOUT if node % 2 == 0 else ODD_NODE_LAYOUT
        ranks.append({"rank": rank, "node_index": node, "visible_devices": visible})
    return {"rank_count": len(ranks), "ranks": ranks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world_size", type=int)
    parser.add_argument("--visible-device-per-rank", type=int, default=1)
    parser.add_argument("--max-gpus-per-node", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = json.dumps(
        build_layout(
            args.world_size,
            visible_device_per_rank=args.visible_device_per_rank,
            max_gpus_per_node=args.max_gpus_per_node,
        ),
        indent=2,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

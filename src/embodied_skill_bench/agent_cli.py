from __future__ import annotations

import argparse
import json
import os
from typing import Any

from .runtime.client import RuntimeClient, load_context


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> int:
    """Minimal read-only CLI staged into an isolated agent workspace."""

    parser = argparse.ArgumentParser(prog="esbench")
    sub = parser.add_subparsers(dest="command", required=True)
    observe = sub.add_parser("observe", help="Read and acknowledge the latest observation")
    observe.add_argument("--run-context", default=None)
    pixel = sub.add_parser("pixel-to-world", help="Back-project a world-map pixel")
    pixel.add_argument("--run-context", default=None)
    pixel.add_argument("--camera", choices=["agentview", "wrist"], default="agentview")
    pixel.add_argument("--resolution", choices=["high", "low"], default=None)
    pixel.add_argument("--row", type=int, required=True)
    pixel.add_argument("--col", type=int, required=True)
    pixel.add_argument("--step", type=int, default=None)
    args = parser.parse_args()
    context = load_context(args.run_context)
    client = RuntimeClient.from_context(context)
    if args.command == "observe":
        _print(client.latest(agent_turn_id=f"shell:{os.getppid()}"))
    else:
        resolution = args.resolution or str(
            context.get("pixel_to_world_default_resolution", "high")
        )
        if resolution not in {"low", "high"}:
            raise ValueError(
                "pixel_to_world_default_resolution must be low or high"
            )
        _print(
            client.pixel_to_world(
                row=args.row,
                col=args.col,
                camera=args.camera,
                resolution=resolution,
                step=args.step,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

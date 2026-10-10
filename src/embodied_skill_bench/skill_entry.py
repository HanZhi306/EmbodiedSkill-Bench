from __future__ import annotations

import argparse
import json
import os
import sys

from .runtime.client import RuntimeClient, load_context


def main(skill_id: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute one staged embodied action or perception skill."
    )
    if skill_id is None:
        parser.add_argument("skill_id")
    parser.add_argument("--json", required=True, help="JSON object of skill arguments")
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--run-context", default=None)
    args = parser.parse_args()
    resolved_skill = skill_id or args.skill_id
    try:
        arguments = json.loads(args.json)
        if not isinstance(arguments, dict):
            raise ValueError("--json must decode to an object")
        context = load_context(args.run_context)
        response = RuntimeClient.from_context(context).execute(
            resolved_skill,
            arguments,
            request_id=args.request_id,
            # Each CLI Bash tool invocation has its own parent shell. Chained
            # skill scripts in one Bash/tool turn share this identity. The
            # runtime allows repeated perception but gates action ordering.
            agent_turn_id=f"shell:{os.getppid()}",
        )
        print(json.dumps(response, indent=2, ensure_ascii=False, sort_keys=True))
        return 0 if response.get("ok") else 2
    except BaseException as exc:
        print(
            json.dumps(
                {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}},
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

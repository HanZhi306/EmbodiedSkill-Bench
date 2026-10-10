"""Workspace-local client for the primitive-code experimental condition."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import sys
from pathlib import Path
from typing import Any

from .runtime.client import RuntimeClient, load_context


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def _agent_visible_response(value: dict[str, Any]) -> dict[str, Any]:
    """Remove interpreter-local variables from the model-facing CLI result.

    The full program audit remains host-side.  Ordinary Python does not echo
    every local variable, and duplicating large calibration dictionaries in
    both ``outputs`` and ``variables`` needlessly expands every later model
    request.  Values the program explicitly printed remain visible.
    """

    response = deepcopy(value)
    result = response.get("read_only_result")
    if isinstance(result, dict) and "variables" in result:
        result.pop("variables")
        result["unprinted_variables_omitted"] = True
    return response


def main() -> int:
    parser = argparse.ArgumentParser(prog="esbench-code")
    sub = parser.add_subparsers(dest="command", required=True)
    observe = sub.add_parser("observe", help="Read and acknowledge the observation")
    observe.add_argument("--run-context", default=None)
    run = sub.add_parser("run", help="Run one restricted Python-syntax program")
    run.add_argument("--file", required=True)
    run.add_argument("--request-id", default=None)
    run.add_argument("--run-context", default=None)
    finish = sub.add_parser("finish", help="Finish and print the episode summary")
    finish.add_argument("--reason", default="agent_finished")
    finish.add_argument("--run-context", default=None)
    api = sub.add_parser("api", help="Print the staged primitive API contract")
    api.add_argument("--spec", default="primitive_api.json")
    args = parser.parse_args()
    try:
        if args.command == "api":
            _print(json.loads(Path(args.spec).read_text(encoding="utf-8")))
            return 0
        context = load_context(args.run_context)
        client = RuntimeClient.from_context(context)
        if args.command == "observe":
            response = client.latest(agent_turn_id=f"shell:{os.getppid()}")
        elif args.command == "run":
            source = Path(args.file).read_text(encoding="utf-8")
            response = client.execute_program(
                source,
                request_id=args.request_id,
                agent_turn_id=f"shell:{os.getppid()}",
            )
        else:
            response = client.finish(str(args.reason))
        _print(_agent_visible_response(response))
        return 0 if response.get("ok", True) else 2
    except BaseException as exc:
        _print({"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

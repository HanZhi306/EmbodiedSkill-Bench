"""One-shot bridge from EmbodiedSkillBench to the official DeepSeek Harness SDK."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any


SDK_DISTRIBUTION = "deepseek-harness-sdk"


def _version() -> str:
    try:
        value = importlib.metadata.version(SDK_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        value = "not-installed"
    return f"{SDK_DISTRIBUTION} {value}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one isolated task through DeepSeek Harness"
    )
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--session-root", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--provider", default="deepseek-official")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--max-tokens", type=int, default=49_152)
    parser.add_argument("--system-file", type=Path)
    parser.add_argument("--cordis", type=Path)
    parser.add_argument("--output-schema", type=Path)
    parser.add_argument("--final-output", type=Path)
    parser.add_argument("prompt", nargs="?")
    return parser


def _required(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    if value is None:
        raise ValueError(f"--{name.replace('_', '-')} is required")
    return value


def _schema_instruction(schema_path: Path) -> str:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return (
        "\n\nDeepSeek Harness final-output constraint: after all tool calls are complete, "
        "return exactly one JSON object matching this schema, with no Markdown fence "
        "or surrounding prose:\n"
        + json.dumps(schema, ensure_ascii=False, sort_keys=True)
    )


def _emit_event(notification: Any) -> None:
    if getattr(notification, "method", None) != "session.event":
        return
    payload = getattr(notification, "payload", None)
    event = payload.get("event") if isinstance(payload, dict) else None
    if isinstance(event, dict):
        print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.version:
        print(_version())
        return 0

    try:
        workspace = Path(_required(args, "workspace")).resolve()
        session_root = Path(_required(args, "session_root")).resolve()
        system_file = Path(_required(args, "system_file")).resolve()
        cordis = Path(_required(args, "cordis")).resolve()
        final_output = Path(_required(args, "final_output")).resolve()
        session_id = str(_required(args, "session_id"))
        prompt = str(_required(args, "prompt"))
        if args.max_tokens <= 0:
            raise ValueError("--max-tokens must be positive")
        if args.output_schema is not None:
            prompt += _schema_instruction(args.output_schema.resolve())
        system_prompt = system_file.read_text(encoding="utf-8")
        if not system_prompt.strip():
            raise ValueError("benchmark system prompt is empty")

        # Import lazily so `--version` and the base test suite remain useful
        # when the optional SDK extra is intentionally absent.
        from deepseek_harness import DeepSeekHarness

        runtime_env = {
            "DSH_SYSTEM_PROMPT": system_prompt,
            "DSH_MODEL": args.model,
            # Node's fetch honors HTTP(S)_PROXY only when this flag is set.
            "NODE_USE_ENV_PROXY": "1",
        }
        with DeepSeekHarness(
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            cwd=str(workspace),
            runtime_cwd=str(workspace),
            session_root=str(session_root),
            cordis=str(cordis),
            env=runtime_env,
        ) as harness:
            result = harness.run(
                prompt,
                session_id=session_id,
                on_notification=_emit_event,
            )

        final_output.parent.mkdir(parents=True, exist_ok=True)
        final_output.write_text(result.final_response, encoding="utf-8")
        print(
            json.dumps(
                {
                    "type": "dsh-sdk-result",
                    "session_id": result.session_id,
                    "provider": args.provider,
                    "model": args.model,
                    "finish_reason": result.finish_reason,
                    "final_response_present": bool(result.final_response.strip()),
                    "sdk_version": _version(),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if result.finish_reason == "completed" and result.final_response.strip() else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "type": "dsh-sdk-error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

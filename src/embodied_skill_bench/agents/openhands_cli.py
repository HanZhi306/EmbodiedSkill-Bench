"""Minimal, event-audited OpenHands SDK bridge for EmbodiedSkillBench."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import signal
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# This bridge is intentionally launched by absolute script path from an
# isolated workspace.  Resolve repository-owned imports from the same source
# tree as the script, rather than an older editable/install found on sys.path.
_REPOSITORY_SRC = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_SRC) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_SRC))


def _finish_guard_script(command: str) -> Path:
    """Resolve a finish hook from either the source or staged runtime tree."""

    candidates = (
        (_REPOSITORY_SRC.parent / command).resolve(),
        (Path(__file__).resolve().parent / command).resolve(),
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.is_relative_to(_REPOSITORY_SRC.parent):
            return candidate
    raise ValueError("finish_guard script is missing or outside repository")

SDK_DISTRIBUTION = "openhands-sdk"
_ALLOWED_CAPABILITY_OVERRIDES = {
    "supports_reasoning_effort",
    "thinking_mode",
    "supports_sampling_params",
    "supports_extended_thinking",
    "supports_prompt_cache",
    "supports_stop_words",
    "supports_responses_api",
    "force_string_serializer",
    "send_reasoning_content",
    "supports_prompt_cache_retention",
    "requires_inline_image_data",
    "supports_vision",
}
_ALLOWED_EXTRA_BODY = {
    "enable_thinking",
    "thinking",
    "thinking_budget",
    "preserve_thinking",
    "vl_high_resolution_images",
    "max_pixels",
}


class _AbsoluteProviderDeadline(BaseException):
    """Non-retriable bridge interrupt that provider libraries cannot swallow."""


def _version() -> str:
    try:
        value = importlib.metadata.version(SDK_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        value = "not-installed"
    return f"{SDK_DISTRIBUTION} {value}"


def _sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


@contextmanager
def _absolute_provider_deadline(seconds: float):
    """Enforce an actual wall-clock deadline around the synchronous SDK call.

    LiteLLM's ``timeout`` is forwarded to its HTTP stack and may act as a
    per-socket-operation timeout.  A provider that keeps making transport
    progress can therefore exceed it indefinitely.  The local OpenHands
    conversation is synchronous and runs on the main thread, so POSIX's real
    timer provides the required end-to-end deadline without leaving an
    unjoinable worker thread behind.
    """

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "OpenHands synchronous provider deadline requires the main thread"
        )
    if not all(
        hasattr(signal, name)
        for name in ("SIGALRM", "ITIMER_REAL", "getitimer", "setitimer")
    ):
        raise RuntimeError(
            "OpenHands absolute provider deadline requires POSIX timers"
        )
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("OpenHands refuses to replace an existing process timer")

    def raise_timeout(_signum: int, _frame: Any) -> None:
        # This deliberately inherits directly from BaseException.  HTTP clients
        # and SDK retry loops conventionally catch Exception; a normal
        # TimeoutError was observed to be swallowed and retried after the
        # one-shot POSIX timer had fired.
        raise _AbsoluteProviderDeadline(
            "OpenHands provider request timed out after the absolute "
            f"{seconds:g}s wall-clock deadline"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one isolated EmbodiedSkillBench task with OpenHands SDK"
    )
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--system-file", type=Path)
    parser.add_argument("--harness-config", type=Path)
    parser.add_argument("--provider", default="openai-compatible")
    parser.add_argument("--model")
    parser.add_argument("--litellm-model")
    parser.add_argument("--model-canonical-name")
    parser.add_argument("--api-mode", choices=("chat", "responses"), default="chat")
    parser.add_argument(
        "--vision-mode", choices=("native-required", "disabled"), default="native-required"
    )
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-output-tokens", type=int, default=49_152)
    parser.add_argument("--request-timeout-seconds", type=int, default=900)
    parser.add_argument("--capability-overrides-json", default="{}")
    parser.add_argument("--litellm-extra-body-json", default="{}")
    parser.add_argument("--allowed-observed-model", action="append", default=[])
    parser.add_argument("--input-cost-per-token", type=float)
    parser.add_argument("--output-cost-per-token", type=float)
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
        "\n\nOpenHands final-output constraint: after all tool calls are complete, "
        "use the finish tool exactly once. Its message must be exactly one JSON "
        "object matching this schema, without a Markdown fence or surrounding prose:\n"
        + json.dumps(schema, ensure_ascii=False, sort_keys=True)
    )


def _episode_root(workspace: Path) -> str | None:
    context_path = workspace / "run_context.json"
    if not context_path.is_file():
        return None
    context = json.loads(context_path.read_text(encoding="utf-8"))
    initial = context.get("initial_observation_json")
    if not isinstance(initial, str):
        return None
    path = Path(initial).resolve()
    if len(path.parents) < 3:
        return None
    return str(path.parents[2])


def _load_staged_native_skill_contract(workspace: Path) -> dict[str, Any]:
    path = workspace / "native_skill_tools.json"
    context_path = workspace / "run_context.json"
    if not path.is_file() or not context_path.is_file():
        raise ValueError("native skill workspace is missing its contract or run context")
    value = json.loads(path.read_text(encoding="utf-8"))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if context.get("native_skill_contract_sha256") != digest:
        raise ValueError("staged native skill contract hash mismatch")
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("protocol") != "S1-native-typed-curated-skill-library-v1"
        or value.get("bash_agent_exposed") is not False
        or value.get("raw_http_agent_exposed") is not False
        or value.get("automatic_fresh_observation_handoff") is not True
        or value.get("one_state_changing_call_per_llm_response") is not True
    ):
        raise ValueError("invalid staged native skill contract boundary")
    tools = value.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("staged native skill contract has no executable tools")
    names: list[str] = []
    for item in tools:
        if (
            not isinstance(item, dict)
            or item.get("kind") not in {"action", "perception"}
            or not isinstance(item.get("parameters"), dict)
            or item["parameters"].get("type") != "object"
            or item["parameters"].get("additionalProperties") is not False
        ):
            raise ValueError("invalid staged native skill tool entry")
        name = item.get("tool_name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,64}", name):
            raise ValueError("invalid staged native skill tool name")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("duplicate staged native skill tool name")
    return value


def _load_harness_config(path: Path, *, workspace: Path | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") not in {1, 2, 3}:
        raise ValueError("OpenHands harness config must use schema_version 1, 2, or 3")
    native_typed = value.get("native_typed") if value.get("schema_version") == 2 else None
    native_skill = value.get("native_skill") if value.get("schema_version") == 3 else None
    if value.get("schema_version") == 2:
        from embodied_skill_bench.runtime.native_typed_contract import tool_schemas

        if not isinstance(native_typed, dict):
            raise ValueError("OpenHands schema_version 2 requires native_typed config")
        protocol = str(native_typed.get("protocol", ""))
        expected_tools = list(tool_schemas(protocol)) + ["finish"]
        if (
            native_typed.get("transport") != "benchmark localhost tokenized HTTP"
            or native_typed.get("raw_http_agent_exposed") is not False
            or native_typed.get("bash_agent_exposed") is not False
            or native_typed.get("timeout_seconds") != 900
        ):
            raise ValueError("OpenHands native typed transport config drift")
    if value.get("schema_version") == 3:
        if workspace is None:
            raise ValueError("OpenHands native skill config requires a workspace")
        contract = _load_staged_native_skill_contract(workspace)
        if not isinstance(native_skill, dict):
            raise ValueError("OpenHands schema_version 3 requires native_skill config")
        if (
            native_skill.get("protocol") != contract["protocol"]
            or native_skill.get("transport") != "benchmark localhost tokenized HTTP"
            or native_skill.get("raw_http_agent_exposed") is not False
            or native_skill.get("bash_agent_exposed") is not False
            or native_skill.get("timeout_seconds") != 900
        ):
            raise ValueError("OpenHands native skill transport config drift")
        expected_tools = [str(item["tool_name"]) for item in contract["tools"]] + ["finish"]
    finish_guard = value.get("finish_guard")
    if finish_guard is not None:
        if value.get("schema_version") != 3 or not isinstance(finish_guard, dict):
            raise ValueError("finish_guard is only supported for native skill schema 3")
        if set(finish_guard) != {
            "enabled",
            "max_denials",
            "command",
            "script_sha256",
            "timeout_seconds",
            "policy",
        }:
            raise ValueError("finish_guard fields are not frozen")
        if finish_guard.get("enabled") is not True:
            raise ValueError("finish_guard must be enabled when declared")
        max_denials = finish_guard.get("max_denials")
        if (
            isinstance(max_denials, bool)
            or not isinstance(max_denials, int)
            or not 1 <= max_denials <= 3
        ):
            raise ValueError("finish_guard.max_denials must be an integer in [1, 3]")
        command = finish_guard.get("command")
        if command != "scripts/openhands_finish_guard.py":
            raise ValueError("finish_guard command must name the benchmark script")
        script_sha = finish_guard.get("script_sha256")
        if not isinstance(script_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", script_sha):
            raise ValueError("finish_guard script SHA-256 is invalid")
        timeout = finish_guard.get("timeout_seconds")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 30:
            raise ValueError("finish_guard.timeout_seconds must be an integer in [1, 30]")
        if finish_guard.get("policy") != "deny-unsuccessful-official-status-before-budget-v1":
            raise ValueError("finish_guard policy drift")
        script_path = _finish_guard_script(command)
        actual_script_sha = hashlib.sha256(script_path.read_bytes()).hexdigest()
        if actual_script_sha != script_sha:
            raise ValueError("finish_guard script hash mismatch")
    if value.get("sdk_distribution") != SDK_DISTRIBUTION:
        raise ValueError("OpenHands harness config SDK distribution mismatch")
    expected = str(value.get("sdk_version"))
    actual = _version().removeprefix(f"{SDK_DISTRIBUTION} ")
    if actual != expected:
        raise RuntimeError(
            f"OpenHands SDK drift inside bridge: expected {expected!r}, found {actual!r}"
        )
    agent = value.get("agent")
    bash = value.get("bash")
    provider = value.get("provider")
    images = value.get("observation_images")
    history = value.get(
        "conversation_history",
        {
            "policy": "full-history",
            "compacted_result_marker": "sha256-and-original-character-count",
            "preserve_all_assistant_messages": True,
            "preserve_latest_tool_result": True,
        },
    )
    ambient = value.get("ambient_state")
    required_sections = (agent, provider, images, history, ambient)
    if not all(isinstance(item, dict) for item in required_sections) or (
        value.get("schema_version") == 1 and not isinstance(bash, dict)
    ):
        raise ValueError(
            "OpenHands harness config is missing agent/bash/provider sections"
        )
    assert isinstance(agent, dict)
    if value.get("schema_version") == 1:
        assert isinstance(bash, dict)
    assert isinstance(provider, dict)
    assert isinstance(images, dict)
    assert isinstance(history, dict)
    assert isinstance(ambient, dict)
    if (
        agent.get("system_prompt_source") != "BENCHMARK_SYSTEM.md verbatim"
        or agent.get("tools") != (
            ["bash", "finish"]
            if value.get("schema_version") == 1
            else ["staged-native-skill-contract", "finish"]
            if value.get("schema_version") == 3
            else expected_tools
        )
        or agent.get("tool_concurrency_limit") != 1
        or agent.get("max_iterations") != 256
        or agent.get("stuck_detection") is not False
        or agent.get("condenser") is not None
        or agent.get("critic") is not None
        or agent.get("agent_context") is not None
        or agent.get("mcp") != []
        or agent.get("plugins") != []
        or agent.get("hooks") != []
    ):
        raise ValueError("OpenHands harness exposed tool list/config drift")
    if value.get("schema_version") == 1 and (
        bash.get("timeout_seconds") != 900
        or bash.get("max_output_chars") != 40_000
        or bash.get("persistent_shell") is not False
        or bash.get("working_directory") != "staged condition workspace"
    ):
        raise ValueError("OpenHands bash executor config drift")
    if (
        images.get("trigger") != (
            "successful ./esbench observe or ./esbench-code observe"
            if value.get("schema_version") == 1
            else "successful state-changing selected skill tool"
            if value.get("schema_version") == 3
            else "successful state-changing native typed tool"
        )
        or images.get("cameras") not in (["agentview", "wrist"], ["agentview"])
        or images.get("resolution_policy")
        not in {"high_rgb_then_rgb", "rgb_only"}
        or images.get("max_images") != len(images.get("cameras", []))
        or images.get("transport") != "base64 data URL in tool result"
        or images.get("trace_storage")
        != "path, sha256, bytes and media type only"
        or images.get("history_policy", "full-history")
        not in {"full-history", "latest-observation-only"}
    ):
        raise ValueError("OpenHands observation-image config drift")
    if (
        history.get("policy")
        not in {"full-history", "latest-tool-result-only"}
        or history.get("compacted_result_marker")
        != "sha256-and-original-character-count"
        or history.get("preserve_all_assistant_messages") is not True
        or history.get("preserve_latest_tool_result") is not True
    ):
        raise ValueError("OpenHands conversation-history config drift")
    if (
        provider.get("adapter")
        != "LiteLLM openai-compatible chat completions"
        or provider.get("internal_retries") != 0
        or provider.get("fallback") is not None
        or provider.get("prompt_caching_markers") is not False
        or provider.get("remote_model_cost_map_refresh") is not False
        or provider.get("stream") is not False
        or provider.get("drop_unsupported_parameters") is not False
        or provider.get("model_router") is not None
        or provider.get("vision_router") is not None
    ):
        raise ValueError("OpenHands provider retry/fallback must remain disabled")
    transport_retry = provider.get(
        "transport_retry",
        {
            "max_attempts": 1,
            "backoff_seconds": 0,
            "scope": "same-request-before-response",
            "retry_on": "transport-failure-only",
        },
    )
    if (
        not isinstance(transport_retry, dict)
        or set(transport_retry)
        != {"max_attempts", "backoff_seconds", "scope", "retry_on"}
        or isinstance(transport_retry.get("max_attempts"), bool)
        or not isinstance(transport_retry.get("max_attempts"), int)
        or int(transport_retry["max_attempts"]) < 1
        or isinstance(transport_retry.get("backoff_seconds"), bool)
        or not isinstance(transport_retry.get("backoff_seconds"), (int, float))
        or float(transport_retry["backoff_seconds"]) < 0
        or transport_retry.get("scope") != "same-request-before-response"
        or transport_retry.get("retry_on") != "transport-failure-only"
    ):
        raise ValueError("OpenHands audited transport-retry config drift")
    if (
        ambient.get("user_home") != "fresh per-condition sandbox home"
        or any(
            ambient.get(field) is not False
            for field in (
                "global_profile",
                "global_memory",
                "global_skills",
                "global_plugins",
            )
        )
    ):
        raise ValueError("OpenHands ambient-state config drift")
    return value


def _json_object(raw: str, field: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _event_text(content: Any) -> str:
    result: list[str] = []
    if isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                result.append(text)
    return "\n".join(result)


def _apply_vision_history_policy(
    messages: list[Any], policy: str
) -> tuple[list[Any], dict[str, Any]]:
    """Retain only the newest visual observation when the harness requests it.

    Text, tool calls, and tool observations remain untouched. Only stale
    ``ImageContent`` parts are removed, preventing multimodal context from
    growing quadratically across embodied action rounds.
    """

    if policy not in {"full-history", "latest-observation-only"}:
        raise ValueError(f"unsupported OpenHands vision history policy: {policy!r}")
    image_message_indices: list[int] = []
    image_parts_before = 0
    image_urls_before = 0
    for index, message in enumerate(messages):
        parts = list(getattr(message, "content", ()) or ())
        image_parts = [
            part for part in parts if getattr(part, "type", None) == "image"
        ]
        if image_parts:
            image_message_indices.append(index)
            image_parts_before += len(image_parts)
            image_urls_before += sum(
                len(getattr(part, "image_urls", ()) or ()) for part in image_parts
            )
    retained_index = image_message_indices[-1] if image_message_indices else None
    audit = {
        "policy": policy,
        "image_messages_before": len(image_message_indices),
        "image_parts_before": image_parts_before,
        "image_urls_before": image_urls_before,
        "retained_image_message_index": retained_index,
        "removed_image_parts": 0,
        "removed_image_urls": 0,
    }
    if policy == "full-history" or retained_index is None:
        return list(messages), audit

    filtered: list[Any] = []
    for index, message in enumerate(messages):
        parts = list(getattr(message, "content", ()) or ())
        if index == retained_index:
            filtered.append(message)
            continue
        retained_parts = [
            part for part in parts if getattr(part, "type", None) != "image"
        ]
        removed = len(parts) - len(retained_parts)
        if removed:
            audit["removed_image_parts"] += removed
            audit["removed_image_urls"] += sum(
                len(getattr(part, "image_urls", ()) or ())
                for part in parts
                if getattr(part, "type", None) == "image"
            )
            message = message.model_copy(update={"content": retained_parts})
        filtered.append(message)
    return filtered, audit


def _route_tool_images_as_fresh_user_observations(
    messages: list[Any],
) -> tuple[list[Any], dict[str, Any]]:
    """Keep function outputs valid while exposing their fresh RGB as vision.

    OpenAI-compatible Chat Completions gateways accept a function result as a
    ``role=tool`` message, but several reject ``image_url`` blocks inside that
    message. Native embodied tools need both: the exact text result must stay
    paired with its tool call, while the post-action RGB must reach the next
    multimodal turn. Move only image blocks into an immediately following,
    benchmark-owned user observation message. This is a transport adaptation;
    it adds no task knowledge and does not alter the persisted OpenHands event
    history.
    """

    from openhands.sdk.llm import Message, TextContent

    routed: list[Any] = []
    pending_image_parts: list[Any] = []
    audit = {
        "policy": "tool-text-then-fresh-user-images",
        "tool_image_messages_before": 0,
        "moved_image_parts": 0,
        "moved_image_urls": 0,
        "inserted_user_observation_messages": 0,
    }

    def flush_pending_images() -> None:
        if not pending_image_parts:
            return
        routed.append(
            Message(
                role="user",
                content=[
                    TextContent(
                        text=(
                            "Fresh post-tool visual observation from the same "
                            "accepted benchmark result batch. Reassess only the "
                            "unfinished task suffix before the next action."
                        )
                    ),
                    *pending_image_parts,
                ],
            )
        )
        pending_image_parts.clear()
        audit["inserted_user_observation_messages"] += 1

    for message in messages:
        parts = list(getattr(message, "content", ()) or ())
        image_parts = [
            part for part in parts if getattr(part, "type", None) == "image"
        ]
        if getattr(message, "role", None) != "tool":
            flush_pending_images()
            routed.append(message)
            continue

        if not image_parts:
            routed.append(message)
            continue

        audit["tool_image_messages_before"] += 1
        audit["moved_image_parts"] += len(image_parts)
        audit["moved_image_urls"] += sum(
            len(getattr(part, "image_urls", ()) or ()) for part in image_parts
        )
        text_parts = [
            part for part in parts if getattr(part, "type", None) != "image"
        ]
        routed.append(message.model_copy(update={"content": text_parts}))
        pending_image_parts.extend(image_parts)
    flush_pending_images()
    return routed, audit


def _apply_conversation_history_policy(
    messages: list[Any], policy: str
) -> tuple[list[Any], dict[str, Any]]:
    """Bound repeated tool-result text without inventing task knowledge.

    Embodied action results include the complete post-action observation. Keeping
    every historical result verbatim makes each provider request grow by tens of
    thousands of characters per action even after stale images are removed. This
    policy preserves every assistant message and tool call, plus the newest tool
    result, while replacing older tool-result text with an auditable content hash.
    The model can still inspect public episode/program artifacts if it needs an
    earlier numeric result.
    """

    if policy not in {"full-history", "latest-tool-result-only"}:
        raise ValueError(
            f"unsupported OpenHands conversation history policy: {policy!r}"
        )
    tool_indices = [
        index
        for index, message in enumerate(messages)
        if getattr(message, "role", None) == "tool"
    ]
    retained_index = tool_indices[-1] if tool_indices else None
    text_chars_before = sum(
        len(str(getattr(part, "text", "")))
        for message in messages
        for part in (getattr(message, "content", ()) or ())
        if getattr(part, "type", None) == "text"
    )
    audit: dict[str, Any] = {
        "policy": policy,
        "tool_messages_before": len(tool_indices),
        "retained_tool_message_index": retained_index,
        "compacted_tool_messages": 0,
        "compacted_original_text_chars": 0,
        "text_chars_before": text_chars_before,
        "text_chars_after": text_chars_before,
    }
    if policy == "full-history" or retained_index is None:
        return list(messages), audit

    filtered: list[Any] = []
    for index, message in enumerate(messages):
        if getattr(message, "role", None) != "tool" or index == retained_index:
            filtered.append(message)
            continue
        parts = list(getattr(message, "content", ()) or ())
        text_parts = [
            part for part in parts if getattr(part, "type", None) == "text"
        ]
        if not text_parts:
            filtered.append(message)
            continue
        original = "\n".join(str(getattr(part, "text", "")) for part in text_parts)
        marker = (
            "[Earlier tool result compacted by the frozen benchmark harness; "
            f"original_chars={len(original)}; sha256={_sha256_bytes(original)}. "
            "All assistant tool calls are preserved. Re-run the preserved "
            "read-only command or re-read its public artifact if an earlier "
            "numeric result is needed.]"
        )
        replacement = text_parts[0].model_copy(update={"text": marker})
        retained_parts: list[Any] = []
        inserted = False
        for part in parts:
            if getattr(part, "type", None) == "text":
                if not inserted:
                    retained_parts.append(replacement)
                    inserted = True
                continue
            retained_parts.append(part)
        filtered.append(message.model_copy(update={"content": retained_parts}))
        audit["compacted_tool_messages"] += 1
        audit["compacted_original_text_chars"] += len(original)

    audit["text_chars_after"] = sum(
        len(str(getattr(part, "text", "")))
        for message in filtered
        for part in (getattr(message, "content", ()) or ())
        if getattr(part, "type", None) == "text"
    )
    return filtered, audit


def _validate_provider_controls(
    capability_overrides: dict[str, Any],
    litellm_extra_body: dict[str, Any],
    *,
    required_vision: bool,
    api_mode: str = "chat",
) -> None:
    unknown_capabilities = sorted(
        set(capability_overrides) - _ALLOWED_CAPABILITY_OVERRIDES
    )
    if unknown_capabilities:
        raise ValueError(
            f"unsupported OpenHands capability overrides: {unknown_capabilities}"
        )
    if capability_overrides.get("supports_vision") is not required_vision:
        raise ValueError(
            "capability_overrides.supports_vision must exactly match vision_mode"
        )
    if not isinstance(
        capability_overrides.get("supports_reasoning_effort"), bool
    ):
        raise ValueError("supports_reasoning_effort must be explicitly boolean")
    expected_responses = api_mode == "responses"
    if capability_overrides.get("supports_responses_api") is not expected_responses:
        raise ValueError("supports_responses_api must exactly match api_mode")
    if capability_overrides.get("supports_prompt_cache") is not False:
        raise ValueError("supports_prompt_cache must remain false")
    unexpected_extra_body = sorted(set(litellm_extra_body) - _ALLOWED_EXTRA_BODY)
    if unexpected_extra_body:
        raise ValueError(
            f"unsupported OpenHands litellm_extra_body fields: {unexpected_extra_body}"
        )
    if "enable_thinking" in litellm_extra_body and not isinstance(
        litellm_extra_body["enable_thinking"], bool
    ):
        raise ValueError("enable_thinking must be boolean")
    if "thinking" in litellm_extra_body:
        thinking = litellm_extra_body["thinking"]
        if (
            not isinstance(thinking, dict)
            or set(thinking) != {"type"}
            or thinking.get("type") not in {"adaptive", "disabled"}
        ):
            raise ValueError("thinking must be exactly adaptive or disabled")


def _provider_exception_chain(exc: BaseException) -> list[BaseException]:
    """Unwrap SDK conversation errors without depending on SDK internals."""

    pending = [exc]
    result: list[BaseException] = []
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        result.append(current)
        for candidate in (
            getattr(current, "original_exception", None),
            current.__cause__,
            current.__context__,
        ):
            if isinstance(candidate, BaseException):
                pending.append(candidate)
    return result


def _provider_failure(exc: BaseException) -> bool:
    chain = _provider_exception_chain(exc)
    names = {
        base.__name__.lower()
        for current in chain
        for base in type(current).__mro__
    }
    if names & {
        "apiconnectionerror",
        "authenticationerror",
        "permissiondeniederror",
        "ratelimiterror",
        "serviceunavailableerror",
        "llmserviceunavailableerror",
        "internalservererror",
        "timeout",
        "apitimeouterror",
        "llmnoresponseerror",
    }:
        return True
    messages = [str(current).lower() for current in chain]
    if any(
        "insufficient quota available for instant inference" in message
        for message in messages
    ):
        return True
    # Some OpenAI-compatible gateways return an opaque HTTP-400 envelope even
    # though the same frozen request schema succeeds immediately before/after.
    # It contains no client-fixable field or validation detail. Classify that
    # narrow signature as provider infrastructure, but never retry a 400 at the
    # same request boundary. Detailed context/schema/argument 400s remain Agent
    # failures and therefore visible rather than being laundered as transport.
    if names & {"badrequesterror", "llmbadrequesterror"} and any(
        "openaiexception - openai_error" in message for message in messages
    ):
        return True
    return any(
        marker in message
        for message in messages
        for marker in (
            "rate limit",
            "invalid api key",
            "status code: 401",
            "status code: 403",
            "status code: 429",
            "connection reset",
            "connection refused",
            "connection closed",
            "connection error",
            "gateway timeout",
            "service unavailable",
            "bad gateway",
            "timed out",
        )
    )


def _provider_retryable(exc: BaseException) -> bool:
    """Return whether the same side-effect-free provider request may be retried."""

    chain = _provider_exception_chain(exc)
    names = {
        base.__name__.lower()
        for current in chain
        for base in type(current).__mro__
    }
    message = "\n".join(str(current).lower() for current in chain)
    # This exact OpenAI-compatible gateway signature is a temporary inference
    # slot/capacity condition returned under HTTP 400, not a client-fixable
    # request error. Retrying the identical request is side-effect free because
    # no assistant response or tool call has been observed yet.
    if "insufficient quota available for instant inference" in message:
        return True
    if names & {"authenticationerror", "permissiondeniederror"}:
        return False
    if names & {"badrequesterror", "llmbadrequesterror"}:
        return False
    if any(
        marker in message
        for marker in (
            "invalid api key",
            "status code: 401",
            "status code: 403",
        )
    ):
        return False
    return _provider_failure(exc)


def _call_with_transport_retries(
    call: Callable[[], Any],
    *,
    request_index: int,
    max_attempts: int,
    backoff_seconds: float,
    sink: Callable[[dict[str, Any]], None],
) -> tuple[Any, int]:
    """Retry one identical synchronous request before any response is observed."""

    for attempt in range(1, max_attempts + 1):
        sink(
            {
                "type": "openhands.provider_transport_attempt.started",
                "request_index": request_index,
                "attempt": attempt,
                "max_attempts": max_attempts,
            }
        )
        try:
            result = call()
        except BaseException as exc:
            retryable = _provider_retryable(exc)
            will_retry = retryable and attempt < max_attempts
            sink(
                {
                    "type": "openhands.provider_transport_attempt.failed",
                    "request_index": request_index,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "error_type": type(exc).__name__,
                    "provider_transport_failure": _provider_failure(exc),
                    "retryable": retryable,
                    "will_retry": will_retry,
                }
            )
            if not will_retry:
                raise
            sink(
                {
                    "type": "openhands.provider_transport_retry.scheduled",
                    "request_index": request_index,
                    "failed_attempt": attempt,
                    "next_attempt": attempt + 1,
                    "backoff_seconds": backoff_seconds,
                }
            )
            if backoff_seconds:
                time.sleep(backoff_seconds)
            continue
        sink(
            {
                "type": "openhands.provider_transport_attempt.completed",
                "request_index": request_index,
                "attempt": attempt,
                "max_attempts": max_attempts,
            }
        )
        return result, attempt
    raise AssertionError("transport retry loop exhausted without returning or raising")


async def _acall_with_transport_retries(
    call: Callable[[], Awaitable[Any]],
    *,
    request_index: int,
    max_attempts: int,
    backoff_seconds: float,
    sink: Callable[[dict[str, Any]], None],
) -> tuple[Any, int]:
    """Async equivalent of :func:`_call_with_transport_retries`."""

    for attempt in range(1, max_attempts + 1):
        sink(
            {
                "type": "openhands.provider_transport_attempt.started",
                "request_index": request_index,
                "attempt": attempt,
                "max_attempts": max_attempts,
            }
        )
        try:
            result = await call()
        except BaseException as exc:
            retryable = _provider_retryable(exc)
            will_retry = retryable and attempt < max_attempts
            sink(
                {
                    "type": "openhands.provider_transport_attempt.failed",
                    "request_index": request_index,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "error_type": type(exc).__name__,
                    "provider_transport_failure": _provider_failure(exc),
                    "retryable": retryable,
                    "will_retry": will_retry,
                }
            )
            if not will_retry:
                raise
            sink(
                {
                    "type": "openhands.provider_transport_retry.scheduled",
                    "request_index": request_index,
                    "failed_attempt": attempt,
                    "next_attempt": attempt + 1,
                    "backoff_seconds": backoff_seconds,
                }
            )
            if backoff_seconds:
                await asyncio.sleep(backoff_seconds)
            continue
        sink(
            {
                "type": "openhands.provider_transport_attempt.completed",
                "request_index": request_index,
                "attempt": attempt,
                "max_attempts": max_attempts,
            }
        )
        return result, attempt
    raise AssertionError("transport retry loop exhausted without returning or raising")


def _validate_observed_models(
    observed_models: set[str], allowed_models: list[str]
) -> list[str]:
    observed = sorted(observed_models)
    if not observed:
        raise RuntimeError("OpenHands provider returned no auditable model identity")
    unexpected = sorted(set(observed) - set(str(item) for item in allowed_models))
    if unexpected:
        raise RuntimeError(
            f"OpenHands observed model outside frozen allowlist: {unexpected}"
        )
    return observed


def _run(args: argparse.Namespace) -> int:
    # Imports stay lazy so version/preflight remains deterministic without the
    # optional SDK extra and core Python 3.10/3.11 installs remain supported.
    # The SDK's import banner is user-friendly in terminals but would corrupt
    # the benchmark's canonical JSONL event stream.
    os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
    # Freeze the bundled LiteLLM model metadata. Import-time remote refreshes
    # are neither an agent turn nor provider inference and would introduce an
    # unlogged network dependency into an otherwise immutable harness.
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    from openhands.sdk import LLM, Agent, Conversation
    from openhands.sdk.conversation import get_agent_final_response
    from openhands.sdk.conversation.state import ConversationExecutionStatus
    from openhands.sdk.event import (
        ActionEvent,
        AgentErrorEvent,
        MessageEvent,
        ObservationEvent,
        SystemPromptEvent,
        UserRejectObservation,
    )
    from openhands.sdk.llm import ImageContent, Message, TextContent
    from openhands.sdk.tool import Tool, register_tool
    from openhands.sdk.tool.builtins.finish import (
        FinishAction,  # noqa: F401 -- registration
    )
    from pydantic import PrivateAttr, SecretStr

    from embodied_skill_bench.agents.openhands_tool import (
        BashObservation,
        BashTool,
        _image_data_url,
        observation_image_artifacts,
    )

    workspace = Path(_required(args, "workspace")).resolve()
    system_file = Path(_required(args, "system_file")).resolve()
    harness_config_path = Path(_required(args, "harness_config")).resolve()
    final_output = Path(_required(args, "final_output")).resolve()
    model = str(_required(args, "model"))
    litellm_model = str(_required(args, "litellm_model"))
    prompt = str(_required(args, "prompt"))
    if args.provider != "openai-compatible":
        raise ValueError("v1 OpenHands bridge supports only provider=openai-compatible")
    if not litellm_model.startswith("openai/"):
        raise ValueError("OpenHands litellm_model must use the explicit openai/ adapter")
    if args.max_output_tokens <= 0 or args.request_timeout_seconds <= 0:
        raise ValueError("OpenHands token and timeout limits must be positive")
    if not args.allowed_observed_model:
        raise ValueError("at least one exact --allowed-observed-model is required")

    base_url = os.environ.get("OPENHANDS_BASE_URL", "")
    api_key = os.environ.get("OPENHANDS_API_KEY", "")
    parsed_url = urlparse(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("OPENHANDS_BASE_URL must be an absolute HTTP(S) URL")
    if not api_key:
        raise ValueError("OPENHANDS_API_KEY is required")

    harness_config = _load_harness_config(harness_config_path, workspace=workspace)
    native_observation_types: tuple[type, ...] = ()
    native_tool_classes = None
    native_skill_tool_classes = None
    native_skill_contract = None
    if harness_config["schema_version"] in {2, 3}:
        from embodied_skill_bench.agents.openhands_native_tool import (
            NativeTypedObservation,
        )
        from embodied_skill_bench.agents.openhands_native_tool import (
            native_skill_tool_classes as _native_skill_tool_classes,
        )
        from embodied_skill_bench.agents.openhands_native_tool import (
            native_tool_classes as _native_tool_classes,
        )

        native_observation_types = (NativeTypedObservation,)
        native_tool_classes = _native_tool_classes
        native_skill_tool_classes = _native_skill_tool_classes
        if harness_config["schema_version"] == 3:
            native_skill_contract = _load_staged_native_skill_contract(workspace)
    vision_history_policy = str(
        harness_config["observation_images"].get(
            "history_policy", "full-history"
        )
    )
    conversation_history_policy = str(
        harness_config.get("conversation_history", {}).get(
            "policy", "full-history"
        )
    )
    transport_retry = harness_config["provider"].get(
        "transport_retry",
        {"max_attempts": 1, "backoff_seconds": 0},
    )
    transport_max_attempts = int(transport_retry["max_attempts"])
    transport_backoff_seconds = float(transport_retry["backoff_seconds"])
    system_prompt = system_file.read_text(encoding="utf-8")
    if not system_prompt.strip():
        raise ValueError("benchmark system prompt is empty")
    if args.output_schema is not None:
        prompt += _schema_instruction(args.output_schema.resolve())

    capability_overrides = _json_object(
        args.capability_overrides_json, "capability_overrides"
    )
    litellm_extra_body = _json_object(
        args.litellm_extra_body_json, "litellm_extra_body"
    )
    required_vision = args.vision_mode == "native-required"
    _validate_provider_controls(
        capability_overrides,
        litellm_extra_body,
        required_vision=required_vision,
        api_mode=args.api_mode,
    )

    emit_lock = threading.Lock()

    def sink(value: dict[str, Any]) -> None:
        with emit_lock:
            _emit(value)

    class AuditedLLM(LLM):
        _observed_models: set[str] = PrivateAttr(default_factory=set)
        _previous_accumulated_usage: tuple[int, int, int, int, int] = PrivateAttr(
            default=(0, 0, 0, 0, 0)
        )
        _provider_request_index: int = PrivateAttr(default=0)

        def _request_started(self) -> int:
            self._provider_request_index += 1
            request_index = self._provider_request_index
            sink(
                {
                    "type": "openhands.provider_request.started",
                    "request_index": request_index,
                    "absolute_timeout_seconds": args.request_timeout_seconds,
                    "transport_max_attempts": transport_max_attempts,
                    "transport_backoff_seconds": transport_backoff_seconds,
                }
            )
            return request_index

        def _request_completed(
            self, request_index: int, result: Any, transport_attempts: int
        ) -> None:
            sink(
                {
                    "type": "openhands.provider_request.completed",
                    "request_index": request_index,
                    "llm_response_id": str(result.id),
                    "transport_attempts": transport_attempts,
                }
            )

        def _request_failed(
            self,
            request_index: int,
            exc: BaseException,
            transport_attempts: int,
        ) -> None:
            sink(
                {
                    "type": "openhands.provider_request.failed",
                    "request_index": request_index,
                    "error_type": type(exc).__name__,
                    "provider_transport_failure": _provider_failure(exc),
                    "transport_attempts": transport_attempts,
                }
            )

        def completion(self, *completion_args: Any, **completion_kwargs: Any) -> Any:
            request_index = self._request_started()
            if completion_args:
                routed, tool_image_audit = _route_tool_images_as_fresh_user_observations(
                    list(completion_args[0])
                )
                filtered, vision_audit = _apply_vision_history_policy(
                    routed, vision_history_policy
                )
                filtered, history_audit = _apply_conversation_history_policy(
                    filtered, conversation_history_policy
                )
                completion_args = (filtered, *completion_args[1:])
            else:
                routed, tool_image_audit = _route_tool_images_as_fresh_user_observations(
                    list(completion_kwargs["messages"])
                )
                filtered, vision_audit = _apply_vision_history_policy(
                    routed, vision_history_policy
                )
                filtered, history_audit = _apply_conversation_history_policy(
                    filtered, conversation_history_policy
                )
                completion_kwargs["messages"] = filtered
            sink(
                {
                    "type": "openhands.tool_image_transport",
                    "request_index": request_index,
                    **tool_image_audit,
                }
            )
            sink(
                {
                    "type": "openhands.vision_history",
                    "request_index": request_index,
                    **vision_audit,
                }
            )
            sink(
                {
                    "type": "openhands.conversation_history",
                    "request_index": request_index,
                    **history_audit,
                }
            )

            def provider_call() -> Any:
                with _absolute_provider_deadline(args.request_timeout_seconds):
                    return super(AuditedLLM, self).completion(
                        *completion_args, **completion_kwargs
                    )

            transport_attempt_state = {"count": 0}

            def transport_sink(event: dict[str, Any]) -> None:
                if event["type"] == "openhands.provider_transport_attempt.started":
                    transport_attempt_state["count"] = int(event["attempt"])
                sink(event)

            try:
                result, transport_attempts = _call_with_transport_retries(
                    provider_call,
                    request_index=request_index,
                    max_attempts=transport_max_attempts,
                    backoff_seconds=transport_backoff_seconds,
                    sink=transport_sink,
                )
            except BaseException as exc:
                self._request_failed(
                    request_index, exc, transport_attempt_state["count"]
                )
                raise
            self._request_completed(request_index, result, transport_attempts)
            return result

        async def acompletion(
            self, *completion_args: Any, **completion_kwargs: Any
        ) -> Any:
            request_index = self._request_started()
            if completion_args:
                routed, tool_image_audit = _route_tool_images_as_fresh_user_observations(
                    list(completion_args[0])
                )
                filtered, vision_audit = _apply_vision_history_policy(
                    routed, vision_history_policy
                )
                filtered, history_audit = _apply_conversation_history_policy(
                    filtered, conversation_history_policy
                )
                completion_args = (filtered, *completion_args[1:])
            else:
                routed, tool_image_audit = _route_tool_images_as_fresh_user_observations(
                    list(completion_kwargs["messages"])
                )
                filtered, vision_audit = _apply_vision_history_policy(
                    routed, vision_history_policy
                )
                filtered, history_audit = _apply_conversation_history_policy(
                    filtered, conversation_history_policy
                )
                completion_kwargs["messages"] = filtered
            sink(
                {
                    "type": "openhands.tool_image_transport",
                    "request_index": request_index,
                    **tool_image_audit,
                }
            )
            sink(
                {
                    "type": "openhands.vision_history",
                    "request_index": request_index,
                    **vision_audit,
                }
            )
            sink(
                {
                    "type": "openhands.conversation_history",
                    "request_index": request_index,
                    **history_audit,
                }
            )
            async def provider_call() -> Any:
                deadline = asyncio.timeout(args.request_timeout_seconds)
                try:
                    async with deadline:
                        return await super(AuditedLLM, self).acompletion(
                            *completion_args, **completion_kwargs
                        )
                except BaseException as exc:
                    if isinstance(exc, TimeoutError) and deadline.expired():
                        raise TimeoutError(
                            "OpenHands provider request timed out after the absolute "
                            f"{args.request_timeout_seconds:g}s wall-clock deadline"
                        ) from exc
                    raise

            transport_attempt_state = {"count": 0}

            def transport_sink(event: dict[str, Any]) -> None:
                if event["type"] == "openhands.provider_transport_attempt.started":
                    transport_attempt_state["count"] = int(event["attempt"])
                sink(event)

            try:
                result, transport_attempts = await _acall_with_transport_retries(
                    provider_call,
                    request_index=request_index,
                    max_attempts=transport_max_attempts,
                    backoff_seconds=transport_backoff_seconds,
                    sink=transport_sink,
                )
            except BaseException as exc:
                self._request_failed(
                    request_index, exc, transport_attempt_state["count"]
                )
                raise
            self._request_completed(request_index, result, transport_attempts)
            return result

        def _audit_response(self, result: Any) -> None:
            raw = result.raw_response
            observed = getattr(raw, "model", None)
            if observed is None and isinstance(raw, dict):
                observed = raw.get("model")
            observed_value = str(observed) if observed else None
            if observed_value:
                self._observed_models.add(observed_value)
            accumulated = result.metrics.accumulated_token_usage
            current = (
                int(accumulated.prompt_tokens) if accumulated else 0,
                int(accumulated.completion_tokens) if accumulated else 0,
                int(accumulated.cache_read_tokens) if accumulated else 0,
                int(accumulated.cache_write_tokens) if accumulated else 0,
                int(accumulated.reasoning_tokens) if accumulated else 0,
            )
            previous = self._previous_accumulated_usage
            if any(now < before for now, before in zip(current, previous)):
                raise RuntimeError("OpenHands accumulated usage regressed")
            per_response = tuple(
                now - before for now, before in zip(current, previous)
            )
            self._previous_accumulated_usage = current
            payload: dict[str, Any] = {
                "type": "openhands.llm_response",
                "llm_response_id": str(result.id),
                "usage": {
                    "input_tokens": per_response[0],
                    "output_tokens": per_response[1],
                    "cache_read_input_tokens": per_response[2],
                    "cache_creation_input_tokens": per_response[3],
                    "reasoning_tokens": per_response[4],
                },
                "accumulated_usage": {
                    "input_tokens": current[0],
                    "output_tokens": current[1],
                    "cache_read_input_tokens": current[2],
                    "cache_creation_input_tokens": current[3],
                    "reasoning_tokens": current[4],
                },
            }
            if observed_value:
                payload["model"] = observed_value
            sink(payload)

        def _build_completion_result(self, resp: Any) -> Any:
            result = super()._build_completion_result(resp)
            self._audit_response(result)
            return result

        def _build_responses_result(self, resp: Any) -> Any:
            result = super()._build_responses_result(resp)
            self._audit_response(result)
            return result

    llm = AuditedLLM(
        usage_id=f"esbench-openhands-{model}",
        model=litellm_model,
        model_canonical_name=args.model_canonical_name,
        api_key=SecretStr(api_key),
        base_url=base_url,
        api_mode=args.api_mode,
        num_retries=0,
        timeout=args.request_timeout_seconds,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        capability_overrides=capability_overrides,
        disable_vision=not required_vision,
        inline_image_urls=False,
        caching_prompt=False,
        stream=False,
        drop_params=False,
        enable_encrypted_reasoning=False,
        prompt_cache_retention=None,
        extended_thinking_budget=None,
        litellm_extra_body=litellm_extra_body,
        input_cost_per_token=args.input_cost_per_token,
        output_cost_per_token=args.output_cost_per_token,
        fallback_strategy=None,
    )
    if llm.vision_is_active() is not required_vision:
        raise RuntimeError("OpenHands resolved vision capability does not match frozen profile")

    native_protocol = None
    tool_configs: list[Tool]
    if harness_config["schema_version"] == 1:
        register_tool(BashTool.name, BashTool)
        bash_config = harness_config["bash"]
        tool_configs = [
            Tool(
                name=BashTool.name,
                params={
                    "episode_root": _episode_root(workspace),
                    "timeout_seconds": int(bash_config["timeout_seconds"]),
                    "max_output_chars": int(bash_config["max_output_chars"]),
                    "attach_observation_images": required_vision,
                    "resolution_policy": str(
                        harness_config["observation_images"]["resolution_policy"]
                    ),
                    "camera_names": list(
                        harness_config["observation_images"]["cameras"]
                    ),
                },
            )
        ]
    elif harness_config["schema_version"] == 2:
        native_config = harness_config["native_typed"]
        native_protocol = str(native_config["protocol"])
        assert native_tool_classes is not None
        classes = native_tool_classes(native_protocol)
        for tool_class in classes:
            register_tool(tool_class.name, tool_class)
        episode_root = _episode_root(workspace)
        if episode_root is None:
            raise RuntimeError("native typed OpenHands requires an episode root")
        tool_configs = [
            Tool(
                name=tool_class.name,
                params={
                    "episode_root": episode_root,
                    "protocol": native_protocol,
                    "resolution_policy": str(
                        harness_config["observation_images"]["resolution_policy"]
                    ),
                    "camera_names": list(
                        harness_config["observation_images"]["cameras"]
                    ),
                    "timeout_seconds": int(native_config["timeout_seconds"]),
                },
            )
            for tool_class in classes
        ]
    else:
        native_config = harness_config["native_skill"]
        native_protocol = str(native_config["protocol"])
        assert native_skill_tool_classes is not None
        assert native_skill_contract is not None
        classes = native_skill_tool_classes(native_skill_contract)
        for tool_class in classes:
            register_tool(tool_class.name, tool_class)
        episode_root = _episode_root(workspace)
        if episode_root is None:
            raise RuntimeError("native skill OpenHands requires an episode root")
        tool_configs = [
            Tool(
                name=tool_class.name,
                params={
                    "episode_root": episode_root,
                    "protocol": native_protocol,
                    "resolution_policy": str(
                        harness_config["observation_images"]["resolution_policy"]
                    ),
                    "camera_names": list(
                        harness_config["observation_images"]["cameras"]
                    ),
                    "timeout_seconds": int(native_config["timeout_seconds"]),
                },
            )
            for tool_class in classes
        ]
    agent_config = harness_config["agent"]
    agent = Agent(
        llm=llm,
        tools=tool_configs,
        include_default_tools=["FinishTool"],
        system_prompt=system_prompt,
        security_policy_filename="",
        system_prompt_kwargs={"llm_security_analyzer": False},
        agent_context=None,
        mcp_config={},
        condenser=None,
        critic=None,
        tool_concurrency_limit=1,
    )

    hook_config = None
    finish_guard = harness_config.get("finish_guard")
    if isinstance(finish_guard, dict) and finish_guard.get("enabled") is True:
        from openhands.sdk.hooks import HookConfig, HookDefinition, HookMatcher

        script_path = _finish_guard_script(str(finish_guard["command"]))
        hook_command = (
            f"{shlex.quote(sys.executable)} {shlex.quote(str(script_path))} "
            f"{int(finish_guard['max_denials'])}"
        )
        hook_config = HookConfig(
            stop=[
                HookMatcher(
                    matcher="*",
                    hooks=[
                        HookDefinition(
                            name="esbench-finish-guard-v1",
                            command=hook_command,
                            timeout=int(finish_guard["timeout_seconds"]),
                        )
                    ],
                )
            ]
        )

    tool_calls = 0
    forced_final_text: str | None = None

    def callback(event: Any) -> None:
        nonlocal tool_calls, forced_final_text
        if isinstance(event, SystemPromptEvent):
            text = event.system_prompt.text
            sink(
                {
                    "type": "openhands.system",
                    "event_id": str(event.id),
                    "sha256": _sha256_bytes(text),
                    "size_bytes": len(text.encode("utf-8")),
                    "delivery": "Agent.system_prompt_verbatim",
                }
            )
            return
        if isinstance(event, ActionEvent):
            tool_calls += 1
            action_payload = (
                event.action.model_dump(mode="json")
                if event.action is not None
                else None
            )
            thought = _event_text(event.thought)
            payload = {
                "type": "openhands.tool.started",
                "event_id": str(event.id),
                "llm_response_id": str(event.llm_response_id),
                "call_id": str(event.tool_call_id),
                "tool": event.tool_name,
                "input": action_payload,
                "thought": thought,
            }
            reasoning = event.reasoning_content or ""
            if reasoning:
                payload["reasoning"] = {
                    "present": True,
                    "sha256": _sha256_bytes(reasoning),
                    "size_bytes": len(reasoning.encode("utf-8")),
                }
            sink(payload)
            return
        if isinstance(event, ObservationEvent):
            observation = event.observation
            result = (
                observation.trace_payload()
                if isinstance(observation, (BashObservation, *native_observation_types))
                else observation.model_dump(mode="json")
            )
            sink(
                {
                    "type": "openhands.tool.completed",
                    "event_id": str(event.id),
                    "action_id": str(event.action_id),
                    "call_id": str(event.tool_call_id),
                    "tool": event.tool_name,
                    "result": result,
                }
            )
            if (
                isinstance(observation, native_observation_types)
                and isinstance(result, dict)
                and isinstance(result.get("response"), dict)
                and result["response"].get("terminal_for_agent") is True
            ):
                forced_final_text = json.dumps(
                    {
                        "status": "action_budget_exhausted",
                        "success": False,
                        "step_index": result["response"].get("step_index"),
                        "message": "No further state-changing call is available.",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                # This is a benchmark resource stop, not an Agent claim of
                # task success.  It prevents another provider request after
                # the first rejected over-budget action.
                conversation.state.execution_status = ConversationExecutionStatus.FINISHED
                sink(
                    {
                        "type": "openhands.native_action_budget_stop",
                        "step_index": result["response"].get("step_index"),
                        "source": "benchmark_runtime_terminal",
                    }
                )
            return
        if isinstance(event, (AgentErrorEvent, UserRejectObservation)):
            sink(
                {
                    "type": "openhands.tool.failed",
                    "event_id": str(event.id),
                    "call_id": str(event.tool_call_id),
                    "tool": event.tool_name,
                    "error": str(getattr(event, "error", None) or getattr(event, "rejection_reason", "tool failed")),
                }
            )
            return
        if isinstance(event, MessageEvent):
            text = _event_text(event.llm_message.content)
            if event.source == "agent":
                sink(
                    {
                        "type": "openhands.message",
                        "event_id": str(event.id),
                        "llm_response_id": str(event.llm_response_id) if event.llm_response_id else None,
                        "role": event.llm_message.role,
                        "text": text,
                    }
                )
            else:
                sink(
                    {
                        "type": "openhands.input",
                        "event_id": str(event.id),
                        "role": event.llm_message.role,
                        "sha256": _sha256_bytes(text),
                        "size_bytes": len(text.encode("utf-8")),
                    }
                )

    conversation = Conversation(
        agent=agent,
        callbacks=[callback],
        workspace=workspace,
        plugins=[],
        hook_config=hook_config,
        max_iteration_per_run=int(agent_config["max_iterations"]),
        stuck_detection=False,
    )
    try:
        initial_message: str | Message = prompt
        if required_vision:
            context_path = workspace / "run_context.json"
            if context_path.is_file():
                context = json.loads(context_path.read_text(encoding="utf-8"))
                initial_path_raw = context.get("initial_observation_json")
                episode_root_raw = _episode_root(workspace)
                if isinstance(initial_path_raw, str) and episode_root_raw is not None:
                    initial_path = Path(initial_path_raw).resolve()
                    episode_root = Path(episode_root_raw).resolve()
                    if not initial_path.is_relative_to(episode_root):
                        raise RuntimeError(
                            "initial observation is outside the episode root"
                        )
                    initial_observation = json.loads(
                        initial_path.read_text(encoding="utf-8")
                    )
                    initial_images = observation_image_artifacts(
                        initial_observation,
                        episode_root=episode_root,
                        resolution_policy=str(
                            harness_config["observation_images"]["resolution_policy"]
                        ),
                        camera_names=tuple(
                            harness_config["observation_images"]["cameras"]
                        ),
                    )
                    if not initial_images:
                        raise RuntimeError(
                            "native-vision episode has no initial public RGB images"
                        )
                    sink(
                        {
                            "type": "openhands.initial_observation_images",
                            "image_artifacts": initial_images,
                        }
                    )
                    initial_message = Message(
                        role="user",
                        content=[
                            TextContent(text=prompt),
                            ImageContent(
                                image_urls=[
                                    _image_data_url(
                                        Path(str(item["path"])),
                                        str(item["media_type"]),
                                    )
                                    for item in initial_images
                                ]
                            ),
                        ],
                    )
        conversation.send_message(initial_message)
        conversation.run()
        final_text = forced_final_text or get_agent_final_response(conversation.state.events)
    finally:
        conversation.close()

    observed = _validate_observed_models(
        llm._observed_models, args.allowed_observed_model
    )
    if not final_text.strip():
        raise RuntimeError("OpenHands conversation ended without a final response")

    metrics = llm.metrics
    accumulated = metrics.accumulated_token_usage
    cost_available = bool(metrics.costs) or (
        args.input_cost_per_token is not None and args.output_cost_per_token is not None
    )
    usage = {
        "input_tokens": int(accumulated.prompt_tokens) if accumulated else 0,
        "output_tokens": int(accumulated.completion_tokens) if accumulated else 0,
        "cache_read_input_tokens": int(accumulated.cache_read_tokens) if accumulated else 0,
        "cache_creation_input_tokens": int(accumulated.cache_write_tokens) if accumulated else 0,
        "reasoning_tokens": int(accumulated.reasoning_tokens) if accumulated else 0,
        "turns": len(metrics.token_usages),
        "tool_calls": tool_calls,
        "cost_usd": float(metrics.accumulated_cost) if cost_available else None,
        "cost_available": cost_available,
        "cost_source": "OpenHands/LiteLLM metrics" if cost_available else "unavailable_from_provider",
    }
    final_output.parent.mkdir(parents=True, exist_ok=True)
    final_output.write_text(final_text, encoding="utf-8")
    sink(
        {
            "type": "openhands.result",
            "requested_model": model,
            "litellm_model": litellm_model,
            "observed_models": observed,
            "vision_mode": args.vision_mode,
            "api_mode": args.api_mode,
            "usage": usage,
            "finish_present": bool(final_text.strip()),
            "sdk_version": _version(),
            "harness_config_sha256": hashlib.sha256(
                harness_config_path.read_bytes()
            ).hexdigest(),
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.version:
        print(_version())
        return 0
    try:
        return _run(args)
    except BaseException as exc:
        failure_class = (
            "provider_transport_failure" if _provider_failure(exc) else "agent_protocol_failure"
        )
        print(
            json.dumps(
                {
                    "type": "openhands.sdk_error",
                    "failure_class": failure_class,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:4000],
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

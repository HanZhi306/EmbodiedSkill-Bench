"""Resolve the current LLM tool turn from the live CLI JSONL trace."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any


_OBSERVE_COMMAND = re.compile(
    r"(?:^|[^\w.-])(?:\./|[^\s;&|'\"]*/)?esbench\s+observe(?=$|[\s'\"])"
)
_PRIMITIVE_OBSERVE_COMMAND = re.compile(
    r"(?:^|[^\w.-])(?:\./|[^\s;&|'\"]*/)?esbench-code\s+observe(?=$|[\s'\"])"
)
_PRIMITIVE_PROGRAM_COMMAND = re.compile(
    r"(?:^|[^\w.-])(?:\./|[^\s;&|'\"]*/)?esbench-code\s+run(?=$|[\s'\"])"
)


class AgentTraceTurnAuthority:
    """Fail-closed authority for action and observation tool turns.

    Claude exposes the assistant message id that owns each tool call.  Codex
    exposes command start/completion events; simultaneously active commands
    are one parallel tool batch and therefore share one canonical id.  The
    runtime consults this trace while a request is in flight, so agent-supplied
    process ids are diagnostic only and cannot authorize another action.
    """

    def __init__(
        self,
        agent_id: str,
        trace_path: str | Path,
        *,
        settle_seconds: float = 1.0,
        poll_seconds: float = 0.01,
    ):
        if agent_id not in {
            "codex",
            "claude-code",
            "deepseek-harness",
            "kimi-code",
            "openhands",
        }:
            raise ValueError(f"unsupported trace authority agent: {agent_id}")
        self.agent_id = agent_id
        self.trace_path = Path(trace_path)
        self.settle_seconds = max(0.0, float(settle_seconds))
        self.poll_seconds = max(0.001, float(poll_seconds))
        self._lock = threading.Lock()

    @property
    def source(self) -> str:
        if self.agent_id == "deepseek-harness":
            return "deepseek_harness_model_step"
        if self.agent_id == "kimi-code":
            return "kimi_wire_main_step_uuid"
        if self.agent_id == "openhands":
            return "openhands_llm_response_id"
        return (
            "claude_assistant_message_id"
            if self.agent_id == "claude-code"
            else "codex_active_tool_batch"
        )

    def active_turn(self, operation: str, skill_id: str | None = None) -> str | None:
        # Codex starts the child command and emits ``item.started`` on separate
        # async paths.  A fast local harness command can reach this server a
        # few milliseconds before that JSONL line becomes visible.  Briefly
        # settle the trace so the fail-closed gate does not reject a genuine
        # active tool turn merely because of that publication race.
        deadline = time.monotonic() + self.settle_seconds
        while True:
            with self._lock:
                events = self._events()
            if self.agent_id == "claude-code":
                active = _active_claude_commands(events)
            elif self.agent_id == "deepseek-harness":
                active = _active_deepseek_harness_commands(events)
            elif self.agent_id == "kimi-code":
                active = _active_kimi_wire_commands(events)
            elif self.agent_id == "openhands":
                active = _active_openhands_commands(events)
            else:
                active = _active_codex_commands(events)
            matches = [
                turn_id
                for turn_id, command in active
                if _command_matches(command, operation, skill_id)
            ]
            unique = sorted(set(matches))
            if len(unique) == 1:
                return unique[0]
            if len(unique) > 1 or time.monotonic() >= deadline:
                return None
            time.sleep(min(self.poll_seconds, max(0.0, deadline - time.monotonic())))

    def _events(self) -> list[dict[str, Any]]:
        trace_path = self.trace_path
        if self.agent_id == "kimi-code":
            # Kimi 0.40.0 buffers stream-json assistant/tool_calls until a
            # tool RESULT arrives.  That transcript cannot authorize a live
            # runtime request. Its native wire journal records tool.call
            # before execution and carries the actual model step UUID.
            session_root = trace_path.parent / "sandbox-home/.kimi-code/sessions"
            wires = list(session_root.glob("*/session_*/agents/main/wire.jsonl"))
            # Each condition starts a fresh isolated home. Never choose a
            # stale session or a subagent journal by timestamp.
            if len(wires) != 1:
                return []
            trace_path = wires[0]
        if not trace_path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in trace_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # The final line may be concurrently written.  A later request
                # will see the complete event; never authorize from a fragment.
                continue
            if isinstance(value, dict):
                events.append(value)
        return events


def _active_kimi_wire_commands(
    records: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    active: dict[str, tuple[str, str]] = {}
    step_uuid: str | None = None
    for record in records:
        if record.get("type") != "context.append_loop_event" or record.get("agentId") != "main":
            continue
        event = record.get("event")
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "step.begin":
            active.clear()
            step_uuid = event.get("uuid")
        elif kind in {"step.end", "step.interrupted"}:
            active.clear()
            step_uuid = None
        elif kind == "tool.call":
            args = event.get("args")
            call_id = event.get("toolCallId")
            if (
                isinstance(step_uuid, str) and step_uuid
                and event.get("stepUuid") == step_uuid
                and isinstance(call_id, str) and call_id
                and event.get("name") == "Bash"
                and isinstance(args, dict)
                and isinstance(args.get("command"), str)
                and not args.get("run_in_background", False)
            ):
                active[call_id] = (f"kimi-step:{step_uuid}", args["command"])
        elif kind == "tool.result":
            active.pop(event.get("toolCallId"), None)
    return list(active.values())


def _active_codex_commands(events: list[dict[str, Any]]) -> list[tuple[str, str]]:
    active: dict[str, tuple[str, str]] = {}
    batch_index = 0
    for event_index, event in enumerate(events):
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "command_execution":
            continue
        item_id = str(item.get("id") or f"event-{event_index}")
        if event.get("type") == "item.started":
            if active:
                batch_id = next(iter(active.values()))[0]
            else:
                batch_index += 1
                batch_id = f"codex-tool-batch:{batch_index}:{item_id}"
            active[item_id] = (batch_id, str(item.get("command", "")))
        elif event.get("type") == "item.completed":
            active.pop(item_id, None)
    return list(active.values())


def _active_claude_commands(events: list[dict[str, Any]]) -> list[tuple[str, str]]:
    active: dict[str, tuple[str, str]] = {}
    for event_index, event in enumerate(events):
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool_id = str(block.get("id") or f"event-{event_index}")
                tool_input = block.get("input")
                command = (
                    str(tool_input.get("command", ""))
                    if isinstance(tool_input, dict)
                    else ""
                )
                message_id = (
                    str(message.get("id"))
                    if isinstance(message, dict) and message.get("id")
                    else str(event.get("uuid") or f"event-{event_index}")
                )
                active[tool_id] = (f"claude-message:{message_id}", command)
            elif block.get("type") == "tool_result":
                active.pop(str(block.get("tool_use_id")), None)
    return list(active.values())


def _active_kimi_classic_commands(
    events: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Recover active shell calls from Kimi Code's stream-json transcript.

    The classic S1 skill workspace exposes no MCP server: the agent drives the
    staged ``scripts/execute.py`` entry points through its built-in shell tool.
    Kimi emits one ``assistant`` line per model response carrying a
    ``tool_calls`` array, then a ``tool`` line per completed call.  Treat the
    first call id of a response as that response's turn id so a batch of calls
    shares one turn, mirroring the Codex active-tool-batch semantics.
    """

    active: dict[str, tuple[str, str]] = {}
    for event_index, event in enumerate(events):
        role = event.get("role")
        if role == "assistant":
            tool_calls = event.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            batch_id: str | None = None
            for call_index, call in enumerate(tool_calls):
                if not isinstance(call, dict):
                    continue
                call_id = str(
                    call.get("id") or f"event-{event_index}-{call_index}"
                )
                if batch_id is None:
                    batch_id = call_id
                function = call.get("function")
                arguments = (
                    function.get("arguments")
                    if isinstance(function, dict)
                    else None
                )
                command = ""
                if isinstance(arguments, str) and arguments:
                    try:
                        parsed = json.loads(arguments)
                    except (TypeError, ValueError):
                        parsed = None
                    if isinstance(parsed, dict):
                        for key in ("command", "cmd", "script", "input"):
                            value = parsed.get(key)
                            if isinstance(value, str) and value:
                                command = value
                                break
                elif isinstance(arguments, dict):
                    for key in ("command", "cmd", "script", "input"):
                        value = arguments.get(key)
                        if isinstance(value, str) and value:
                            command = value
                            break
                active[call_id] = (f"kimi-response:{batch_id}", command)
        elif role == "tool":
            call_id = event.get("tool_call_id") or event.get("id")
            if call_id is not None:
                active.pop(str(call_id), None)
    return list(active.values())


def _active_deepseek_harness_commands(
    events: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Return active Bash calls keyed by the model response that emitted them.

    Harness keeps one ``turn`` across an entire user task and increments
    ``step`` for every model response after a tool result. Parallel tool calls
    from one response share the same ``(turn, step)`` and therefore the same
    protocol authority id.
    """

    active: dict[str, tuple[str, str]] = {}
    for event_index, event in enumerate(events):
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        if event.get("type") == "tool/call":
            call_id = str(data.get("callId") or f"event-{event_index}")
            arguments = data.get("arguments")
            try:
                tool_input = json.loads(arguments) if isinstance(arguments, str) else {}
            except json.JSONDecodeError:
                tool_input = {}
            command = (
                str(tool_input.get("command", ""))
                if str(data.get("name", "")).lower() == "bash"
                and isinstance(tool_input, dict)
                else ""
            )
            turn = data.get("turn")
            step = data.get("step")
            active[call_id] = (
                f"deepseek-harness-step:{turn}:{step}",
                command,
            )
        elif event.get("type") == "tool/result":
            message = data.get("message")
            source = message.get("source") if isinstance(message, dict) else None
            call_id = source.get("callId") if isinstance(source, dict) else None
            if call_id is not None:
                active.pop(str(call_id), None)
    return list(active.values())


def _active_openhands_commands(
    events: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Recover active Bash calls from the benchmark's OpenHands event stream.

    OpenHands exposes the provider response id on every ActionEvent. Parallel
    calls from one response therefore share one causal turn even though they
    have distinct tool-call ids. The bridge emits the start before execution
    and the completion/failure after execution, so the runtime can consult the
    trace while the subprocess is actively making a localhost request.
    """

    active: dict[str, tuple[str, str]] = {}
    for event_index, event in enumerate(events):
        event_type = event.get("type")
        if event_type == "openhands.tool.started":
            call_id = str(event.get("call_id") or f"event-{event_index}")
            tool_input = event.get("input")
            command = (
                str(tool_input.get("command", ""))
                if event.get("tool") == "bash" and isinstance(tool_input, dict)
                else f"@openhands-native-tool:{event.get('tool', '')}"
            )
            response_id = event.get("llm_response_id")
            active[call_id] = (
                f"openhands-response:{response_id}",
                command,
            )
        elif event_type in {
            "openhands.tool.completed",
            "openhands.tool.failed",
        }:
            call_id = event.get("call_id")
            if call_id is not None:
                active.pop(str(call_id), None)
    return list(active.values())


def _command_matches(
    command: str, operation: str, skill_id: str | None
) -> bool:
    if operation == "observe":
        return bool(_OBSERVE_COMMAND.search(command))
    if operation == "primitive_observe":
        return bool(_PRIMITIVE_OBSERVE_COMMAND.search(command))
    if operation == "primitive_program":
        return bool(_PRIMITIVE_PROGRAM_COMMAND.search(command))
    if operation == "native_tool" and skill_id:
        return command == f"@openhands-native-tool:{skill_id}"
    if operation in {"action", "perception"} and skill_id:
        escaped = re.escape(skill_id)
        return bool(
            re.search(
                rf"(?:\.agents|\.claude)/skills/{escaped}/scripts/execute\.py(?:\s|$)",
                command,
            )
        )
    raise ValueError(f"unsupported protocol operation: {operation!r}")

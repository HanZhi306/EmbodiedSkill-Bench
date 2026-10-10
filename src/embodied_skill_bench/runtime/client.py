from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def load_context(path: str | Path | None = None) -> dict[str, Any]:
    candidate = path or os.environ.get("ESB_RUN_CONTEXT") or "run_context.json"
    with Path(candidate).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("run_context.json must contain an object")
    return value


class RuntimeClient:
    def __init__(self, url: str, token: str, timeout_seconds: float = 900):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_context(cls, context: dict[str, Any]) -> "RuntimeClient":
        return cls(
            str(context["runtime_url"]),
            str(context["runtime_token"]),
            float(context.get("skill_timeout_seconds", 900)),
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        headers.update(extra_headers or {})
        request = Request(
            self.url + path,
            data=data,
            method=method,
            headers=headers,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                value = json.loads(response.read())
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"runtime HTTP {exc.code}: {body}") from exc
        if not isinstance(value, dict):
            raise RuntimeError("runtime returned a non-object response")
        return value

    def execute(
        self,
        skill_id: str,
        arguments: dict[str, Any],
        request_id: str | None = None,
        retries: int = 2,
        agent_turn_id: str | None = None,
    ) -> dict[str, Any]:
        request_id = request_id or str(uuid.uuid4())
        payload = {
            "request_id": request_id,
            "skill_id": skill_id,
            "arguments": arguments,
            "agent_turn_id": agent_turn_id or f"process:{os.getpid()}",
        }
        last_error: BaseException | None = None
        for attempt in range(retries + 1):
            try:
                return self._request("POST", "/v1/execute", payload)
            except (URLError, TimeoutError) as exc:
                last_error = exc
                if attempt == retries:
                    break
                time.sleep(0.25 * (attempt + 1))
        raise RuntimeError(f"runtime request failed after retries: {last_error}")

    def execute_program(
        self,
        program: str,
        *,
        request_id: str | None = None,
        retries: int = 2,
        agent_turn_id: str | None = None,
    ) -> dict[str, Any]:
        """Submit one restricted primitive-code program.

        The same transport endpoint is used as skill execution, but the
        server-side session type accepts exactly one of the two contracts.
        """

        request_id = request_id or str(uuid.uuid4())
        payload = {
            "request_id": request_id,
            "program": program,
            "agent_turn_id": agent_turn_id or f"process:{os.getpid()}",
        }
        last_error: BaseException | None = None
        for attempt in range(retries + 1):
            try:
                return self._request("POST", "/v1/execute", payload)
            except (URLError, TimeoutError) as exc:
                last_error = exc
                if attempt == retries:
                    break
                time.sleep(0.25 * (attempt + 1))
        raise RuntimeError(f"runtime request failed after retries: {last_error}")

    def execute_native_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        request_id: str | None = None,
        retries: int = 2,
        agent_turn_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute one benchmark-owned native typed tool call."""

        request_id = request_id or str(uuid.uuid4())
        payload = {
            "request_id": request_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "agent_turn_id": agent_turn_id or f"process:{os.getpid()}",
        }
        last_error: BaseException | None = None
        for attempt in range(retries + 1):
            try:
                return self._request("POST", "/v1/native-tool", payload)
            except (URLError, TimeoutError) as exc:
                last_error = exc
                if attempt == retries:
                    break
                time.sleep(0.25 * (attempt + 1))
        raise RuntimeError(f"runtime native-tool request failed after retries: {last_error}")

    def latest(self, agent_turn_id: str | None = None) -> dict[str, Any]:
        return self._request(
            "GET",
            "/v1/observation/latest",
            extra_headers={
                "X-ESB-Agent-Turn-ID": agent_turn_id or f"process:{os.getpid()}"
            },
        )

    def pixel_to_world(
        self,
        *,
        row: int,
        col: int,
        camera: str = "agentview",
        resolution: str = "high",
        step: int | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/pixel-to-world",
            {
                "row": row,
                "col": col,
                "camera": camera,
                "resolution": resolution,
                "step": step,
            },
        )

    def finish(self, reason: str) -> dict[str, Any]:
        return self._request("POST", "/v1/finish", {"reason": reason})

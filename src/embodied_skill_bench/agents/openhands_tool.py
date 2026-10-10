"""The single execution tool exposed by the minimal OpenHands harness.

This module is imported only by the optional OpenHands bridge.  Keeping it
separate lets ``esbench-openhands --version`` fail closed without importing the
SDK, while OpenHands still sees module-level (and therefore serializable) tool
types at runtime.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

from pydantic import Field

from openhands.sdk import Action, ImageContent, Observation, TextContent, ToolDefinition
from openhands.sdk.tool import ToolAnnotations, ToolExecutor


_OBSERVE_COMMAND = re.compile(
    r"(?:^|[^\w.-])(?:\./|[^\s;&|'\"]*/)?esbench\s+observe(?=$|[\s'\"])")
_PRIMITIVE_OBSERVE_COMMAND = re.compile(
    r"(?:^|[^\w.-])(?:\./|[^\s;&|'\"]*/)?esbench-code\s+observe(?=$|[\s'\"])")
_PRIMITIVE_RUN_COMMAND = re.compile(
    r"(?:^|[^\w.-])(?:\./|[^\s;&|'\"]*/)?esbench-code\s+run(?=$|[\s'\"])")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_data_url(path: Path, media_type: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def observation_image_artifacts(
    observation: dict[str, Any],
    *,
    episode_root: Path,
    resolution_policy: str,
    camera_names: Sequence[str] = ("agentview", "wrist"),
) -> list[dict[str, Any]]:
    """Resolve the frozen public camera subset from one sanitized observation."""

    if resolution_policy not in {"high_rgb_then_rgb", "rgb_only"}:
        raise ValueError(
            "OpenHands observation image resolution_policy must be "
            "high_rgb_then_rgb or rgb_only"
        )
    requested_cameras = tuple(str(value) for value in camera_names)
    if (
        not requested_cameras
        or len(requested_cameras) != len(set(requested_cameras))
        or not set(requested_cameras) <= {"agentview", "wrist"}
    ):
        raise ValueError("OpenHands camera_names must be a unique public camera subset")
    artifacts = observation.get("artifacts")
    cameras = artifacts.get("cameras") if isinstance(artifacts, dict) else None
    if not isinstance(cameras, dict):
        return []

    images: list[dict[str, Any]] = []
    missing_projected: list[Path] = []
    for camera in requested_cameras:
        camera_data = cameras.get(camera)
        if not isinstance(camera_data, dict):
            continue
        selected = None
        resolution = None
        candidates = (
            ("high_rgb", "rgb")
            if resolution_policy == "high_rgb_then_rgb"
            else ("rgb",)
        )
        for candidate in candidates:
            item = camera_data.get(candidate)
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                selected = Path(item["path"]).resolve()
                resolution = candidate
                break
        if selected is None or not selected.is_relative_to(episode_root):
            continue
        # Strict isolation publishes a transactional RGB-only projection over
        # the private episode path.  It can trail the runtime response briefly.
        deadline = time.monotonic() + 7.0
        while not selected.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not selected.is_file():
            missing_projected.append(selected)
            continue
        media_type = "image/png" if selected.suffix.lower() == ".png" else "image/jpeg"
        images.append(
            {
                "camera": camera,
                "resolution": resolution,
                "path": str(selected),
                "media_type": media_type,
                "sha256": _sha256(selected),
                "bytes": selected.stat().st_size,
            }
        )
    if missing_projected:
        raise RuntimeError(
            "projected RGB observation did not become available: "
            + ", ".join(str(path) for path in missing_projected)
        )
    return images


class BashAction(Action):
    command: str = Field(
        min_length=1,
        max_length=100_000,
        description=(
            "One non-interactive Bash command to run in the staged benchmark "
            "workspace. Use ./esbench or ./esbench-code exactly as documented."
        ),
    )


class BashObservation(Observation):
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    output_truncated: bool = False
    image_artifacts: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        payload = {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "output_truncated": self.output_truncated,
            "observation_images": [
                {
                    "camera": item["camera"],
                    "resolution": item["resolution"],
                    "path": item["path"],
                    "sha256": item["sha256"],
                    "bytes": item["bytes"],
                }
                for item in self.image_artifacts
            ],
        }
        content: list[TextContent | ImageContent] = [
            TextContent(text=json.dumps(payload, ensure_ascii=False, sort_keys=True))
        ]
        urls = [
            _image_data_url(Path(str(item["path"])), str(item["media_type"]))
            for item in self.image_artifacts
        ]
        if urls:
            content.append(ImageContent(image_urls=urls))
        return content

    def trace_payload(self) -> dict[str, Any]:
        """Return the complete text result without duplicating base64 images."""

        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "output_truncated": self.output_truncated,
            "image_artifacts": self.image_artifacts,
        }


def _tool_environment() -> dict[str, str]:
    """Remove provider credentials before commands become model-controlled."""

    blocked = {
        "OPENHANDS_API_KEY",
        "OPENHANDS_BASE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "COMMON_API_KEY",
        "COMMON_BASE_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    }
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in blocked
        and not key.upper().endswith("_API_KEY")
        and not key.upper().endswith("_BASE_URL")
        and key.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    }


def _bounded_text(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    marker = f"\n...[OpenHands bash output truncated to {limit} characters]...\n"
    keep = max(0, limit - len(marker))
    before = keep // 2
    after = keep - before
    return value[:before] + marker + value[-after:], True


def _leading_json_object(stdout: str) -> dict[str, Any] | None:
    """Decode the launcher JSON even if a shell command prints text after it.

    Agents sometimes combine ``observe && cat API.md``.  The observe response
    is still authoritative and complete at the start of stdout; rejecting its
    images merely because later commands appended text silently downgrades a
    native-multimodal condition.  Only a leading JSON object is accepted, so
    arbitrary prose containing JSON cannot trigger an attachment.
    """

    stripped = stdout.lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class BashExecutor(ToolExecutor[BashAction, BashObservation]):
    def __init__(
        self,
        *,
        workspace: str,
        episode_root: str | None,
        timeout_seconds: int,
        max_output_chars: int,
        attach_observation_images: bool,
        resolution_policy: str = "high_rgb_then_rgb",
        camera_names: Sequence[str] = ("agentview", "wrist"),
    ):
        self.workspace = Path(workspace).resolve()
        self.episode_root = Path(episode_root).resolve() if episode_root else None
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.attach_observation_images = attach_observation_images
        if resolution_policy not in {"high_rgb_then_rgb", "rgb_only"}:
            raise ValueError(
                "OpenHands observation image resolution_policy must be "
                "high_rgb_then_rgb or rgb_only"
            )
        self.resolution_policy = resolution_policy
        self.camera_names = tuple(str(value) for value in camera_names)
        if (
            not self.camera_names
            or len(self.camera_names) != len(set(self.camera_names))
            or not set(self.camera_names) <= {"agentview", "wrist"}
        ):
            raise ValueError("OpenHands camera_names must be a unique public camera subset")
        self._active: subprocess.Popen[str] | None = None

    def __call__(self, action: BashAction, conversation=None) -> BashObservation:  # noqa: ARG002
        process = subprocess.Popen(
            ["/bin/bash", "-lc", action.command],
            cwd=self.workspace,
            env=_tool_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self._active = process
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            stderr = (stderr + "\nOpenHands bash tool timeout").strip()
        finally:
            self._active = None

        stdout, stdout_cut = _bounded_text(stdout, self.max_output_chars)
        stderr, stderr_cut = _bounded_text(stderr, self.max_output_chars)
        images = (
            self._observation_images(action.command, stdout)
            if self.attach_observation_images
            else []
        )
        return BashObservation(
            exit_code=124 if timed_out else int(process.returncode),
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            output_truncated=stdout_cut or stderr_cut,
            image_artifacts=images,
        )

    def interrupt(self) -> None:
        process = self._active
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)

    def _observation_images(
        self, command: str, stdout: str
    ) -> list[dict[str, Any]]:
        if not (
            _OBSERVE_COMMAND.search(command)
            or _PRIMITIVE_OBSERVE_COMMAND.search(command)
            or _PRIMITIVE_RUN_COMMAND.search(command)
        ):
            return []
        response = _leading_json_object(stdout)
        if response is None:
            return []
        automatic_handoff = bool(response.get("automatic_observation_handoff"))
        explicit_observe = bool(
            _OBSERVE_COMMAND.search(command)
            or _PRIMITIVE_OBSERVE_COMMAND.search(command)
        )
        if not response.get("ok") or not response.get("observation_acknowledged"):
            return []
        if explicit_observe and response.get("protocol_error") is not None:
            return []
        if not explicit_observe and not automatic_handoff:
            return []
        observation = response.get("observation")
        if not isinstance(observation, dict) or self.episode_root is None:
            return []
        return observation_image_artifacts(
            observation,
            episode_root=self.episode_root,
            resolution_policy=self.resolution_policy,
            camera_names=self.camera_names,
        )


class BashTool(ToolDefinition[BashAction, BashObservation]):
    """A single auditable shell surface inside the benchmark's outer sandbox."""

    @classmethod
    def create(
        cls,
        conv_state,
        *,
        episode_root: str | None = None,
        timeout_seconds: int = 900,
        max_output_chars: int = 100_000,
        attach_observation_images: bool = True,
        resolution_policy: str = "high_rgb_then_rgb",
        camera_names: Sequence[str] = ("agentview", "wrist"),
    ) -> Sequence[Self]:
        workspace = str(conv_state.workspace.working_dir)
        executor = BashExecutor(
            workspace=workspace,
            episode_root=episode_root,
            timeout_seconds=int(timeout_seconds),
            max_output_chars=int(max_output_chars),
            attach_observation_images=bool(attach_observation_images),
            resolution_policy=str(resolution_policy),
            camera_names=tuple(str(value) for value in camera_names),
        )
        return [
            cls(
                description=(
                    "Run one Bash command in the staged EmbodiedSkillBench workspace. "
                    "Use it to read staged files, create ordinary scratch/program files, "
                    "call ./esbench observe, invoke installed skill execute.py scripts, "
                    "or call ./esbench-code. The outer benchmark sandbox prevents access "
                    "to the source repository, other runs, checkpoints and host secrets."
                ),
                action_type=BashAction,
                observation_type=BashObservation,
                executor=executor,
                annotations=ToolAnnotations(
                    title="bash",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
            )
        ]

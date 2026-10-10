from __future__ import annotations

import json
import os
import shutil
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..io import atomic_write_json
from .types import CapturedObservation


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_artifacts(value: Any, final_dir: Path) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_artifacts(child, final_dir) for key, child in value.items()}
    if isinstance(value, list):
        return [_resolve_artifacts(child, final_dir) for child in value]
    if isinstance(value, str) and value.startswith("./"):
        return str((final_dir / value[2:]).resolve())
    return value


class ObservationStore:
    def __init__(self, episode_dir: str | Path, episode_id: str):
        self.episode_dir = Path(episode_dir).resolve()
        self.episode_id = episode_id
        self.steps_dir = self.episode_dir / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self._latest_bundle: dict[str, Any] | None = None
        self._latest_pointer: dict[str, Any] | None = None

    @property
    def events_path(self) -> Path:
        return self.episode_dir / "events.jsonl"

    def begin(self, step_index: int) -> tuple[Path, Path]:
        final_dir = self.steps_dir / f"{step_index:04d}"
        if final_dir.exists():
            raise FileExistsError(f"step already exists: {final_dir}")
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{step_index:04d}.", dir=self.steps_dir)
        )
        return temporary, final_dir

    def abort(self, temporary: Path) -> None:
        shutil.rmtree(temporary, ignore_errors=True)

    def commit(
        self,
        temporary: Path,
        final_dir: Path,
        step_index: int,
        captured: CapturedObservation,
        execution: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if execution is not None:
            with (temporary / "execution.json").open("w", encoding="utf-8") as stream:
                json.dump(execution, stream, indent=2, ensure_ascii=False, sort_keys=True)
                stream.write("\n")

        observation_path = final_dir / "observation.json"
        markdown_path = final_dir / "observation.md"
        artifacts = _resolve_artifacts(captured.artifacts, final_dir)
        bundle = {
            "schema_version": "1.0",
            "episode_id": self.episode_id,
            "step_index": step_index,
            "captured_at": utc_now(),
            "task": captured.task,
            "status": captured.status,
            "robot": captured.robot,
            "object_names": captured.object_names,
            "artifacts": artifacts,
            "last_execution": execution,
            "backend": captured.backend,
            "agent_contract": {
                "one_action_at_a_time": True,
                "one_action_per_agent_turn": True,
                "observation_is_post_action": True,
                "next_action_requires_observation_read": True,
                "selected_perception_repeatable_before_action": True,
                "perception_changes_simulator_state": False,
                "perception_returns_new_observation": False,
                "privileged_object_coordinates_exposed": False,
                "next_action_allowed": not bool(
                    captured.status.get("terminated")
                    or captured.status.get("truncated")
                ),
                "pixel_convention": "row 0 is top; column 0 is left",
                "world_frame": "meters; backend world frame",
            },
        }
        with (temporary / "observation.json").open("w", encoding="utf-8") as stream:
            json.dump(bundle, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
        (temporary / "observation.md").write_text(
            render_observation_markdown(bundle, observation_path, markdown_path),
            encoding="utf-8",
        )

        os.replace(temporary, final_dir)
        pointer = {
            "schema_version": "1.0",
            "episode_id": self.episode_id,
            "step_index": step_index,
            "observation_json": str(observation_path.resolve()),
            "observation_markdown": str(markdown_path.resolve()),
        }
        atomic_write_json(self.episode_dir / "latest.json", pointer)
        self._latest_bundle = bundle
        self._latest_pointer = pointer
        return bundle

    def latest_pointer(self) -> dict[str, Any]:
        if self._latest_pointer is not None:
            return deepcopy(self._latest_pointer)
        path = self.episode_dir / "latest.json"
        if not path.exists():
            raise FileNotFoundError("episode has no observation yet")
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)

    def latest(self) -> dict[str, Any]:
        if self._latest_bundle is not None:
            return deepcopy(self._latest_bundle)
        pointer = self.latest_pointer()
        with Path(pointer["observation_json"]).open(encoding="utf-8") as stream:
            return json.load(stream)



def _format_vector(value: Any, digits: int = 4) -> str:
    if not isinstance(value, (list, tuple)):
        return "unavailable"
    return "[" + ", ".join(f"{float(item):.{digits}f}" for item in value) + "]"


def _artifact_line(label: str, item: dict[str, Any] | None) -> str | None:
    if not item or not item.get("path"):
        return None
    resolution = item.get("resolution")
    suffix = f" ({resolution[0]}x{resolution[1]})" if resolution else ""
    return f"- {label}{suffix}: `{item['path']}`"


def render_observation_markdown(
    bundle: dict[str, Any], observation_path: Path, markdown_path: Path
) -> str:
    task = bundle["task"]
    status = bundle["status"]
    robot = bundle["robot"]
    eef = robot.get("end_effector", {})
    gripper = robot.get("gripper", {})
    execution = bundle.get("last_execution")
    lines = [
        f"# Observation {bundle['step_index']:04d}",
        "",
        f"Task: {task.get('language', 'unknown')}",
        f"Environment: `{task.get('environment', 'unknown')}`; seed: `{task.get('seed', 'unknown')}`",
        (
            "Status: "
            f"success=`{bool(status.get('success'))}`, "
            f"terminated=`{bool(status.get('terminated'))}`, "
            f"truncated=`{bool(status.get('truncated'))}`"
        ),
    ]
    if execution:
        lines.extend(
            [
                "",
                "## Previous skill",
                "",
                f"- Skill: `{execution.get('skill_id')}`",
                f"- Accepted: `{execution.get('accepted')}`",
                f"- Arguments: `{json.dumps(execution.get('arguments', {}), ensure_ascii=False)}`",
                f"- Result: `{json.dumps(execution.get('result', {}), ensure_ascii=False)}`",
                f"- Duration: `{execution.get('elapsed_seconds', 0):.3f}s`",
            ]
        )
    lines.extend(
        [
            "",
            "## Robot state",
            "",
            f"- EEF position, world meters: `{_format_vector(eef.get('position_m'))}`",
            (
                "- EEF quaternion "
                f"({eef.get('quaternion_convention', 'unknown')}): "
                f"`{_format_vector(eef.get('quaternion'))}`"
            ),
            f"- Gripper opening proxy, meters: `{gripper.get('opening_m')}`",
            f"- Gripper convention: {gripper.get('command_convention', 'unknown')}",
            "",
            "## Visual and geometric artifacts",
            "",
        ]
    )
    cameras = bundle.get("artifacts", {}).get("cameras", {})
    artifact_lines = [
        _artifact_line("Agentview RGB", cameras.get("agentview", {}).get("rgb")),
        _artifact_line("Agentview high-resolution RGB", cameras.get("agentview", {}).get("high_rgb")),
        _artifact_line("Wrist RGB", cameras.get("wrist", {}).get("rgb")),
        _artifact_line("Wrist high-resolution RGB", cameras.get("wrist", {}).get("high_rgb")),
    ]
    lines.extend(line for line in artifact_lines if line)
    lines.extend(
        [
            "",
            "Inspect RGB before choosing the next action. For a pixel-to-world lookup, use `esbench pixel-to-world` with the same step, camera, and resolution.",
            "",
            f"Machine-readable bundle: `{observation_path.resolve()}`",
            f"This summary: `{markdown_path.resolve()}`",
            "",
        ]
    )
    if status.get("terminated") or status.get("truncated"):
        lines.extend(
            [
                "Episode is finished. Do not invoke another action skill; write the requested final result.",
                "",
            ]
        )
    return "\n".join(lines)

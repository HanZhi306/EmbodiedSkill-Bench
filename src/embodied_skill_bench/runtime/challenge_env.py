"""Executable LIBERO challenge perturbations with private audit truth.

The agent never imports this module and never receives the configured object
poses.  It only observes the resulting public RGB-D bundle through the normal
runtime contract.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


def validate_challenge_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("challenge spec must be an object")
    allowed = {
        "schema_version",
        "challenge_cell_id",
        "reset_object_swaps",
        "dynamic_object_swaps",
        "dynamic_target_relocation",
    }
    extras = set(spec) - allowed
    if extras:
        raise ValueError(f"unknown challenge spec fields: {sorted(extras)}")
    if spec.get("schema_version") != 1:
        raise ValueError("challenge spec schema_version must equal 1")
    cell_id = spec.get("challenge_cell_id")
    if not isinstance(cell_id, str) or not cell_id.strip():
        raise ValueError("challenge_cell_id must be a non-empty string")

    def normalize_swaps(field: str) -> tuple[list[list[str]], set[str]]:
        swaps = spec.get(field, [])
        if not isinstance(swaps, list):
            raise ValueError(f"{field} must be a list")
        normalized: list[list[str]] = []
        used: set[str] = set()
        for index, pair in enumerate(swaps):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(item, str) and item for item in pair)
                or pair[0] == pair[1]
            ):
                raise ValueError(f"{field}[{index}] must name two objects")
            overlap = used & set(pair)
            if overlap:
                raise ValueError(
                    f"{field} cannot reuse objects: {sorted(overlap)}"
                )
            used.update(pair)
            normalized.append(list(pair))
        return normalized, used

    normalized_swaps, reset_swap_objects = normalize_swaps(
        "reset_object_swaps"
    )
    normalized_dynamic_swaps, dynamic_swap_objects = normalize_swaps(
        "dynamic_object_swaps"
    )
    overlap = reset_swap_objects & dynamic_swap_objects
    if overlap:
        raise ValueError(
            "reset and dynamic swaps cannot reuse objects: "
            f"{sorted(overlap)}"
        )

    relocation = spec.get("dynamic_target_relocation")
    normalized_relocation = None
    if relocation is not None:
        if not isinstance(relocation, dict):
            raise ValueError("dynamic_target_relocation must be an object")
        relocation_allowed = {
            "object_name",
            "delta_xyz_m",
            "eef_distance_below_m",
            "minimum_task_steps",
        }
        relocation_extras = set(relocation) - relocation_allowed
        if relocation_extras:
            raise ValueError(
                "unknown dynamic_target_relocation fields: "
                f"{sorted(relocation_extras)}"
            )
        object_name = relocation.get("object_name")
        delta = relocation.get("delta_xyz_m")
        threshold = relocation.get("eef_distance_below_m")
        minimum_steps = relocation.get("minimum_task_steps", 0)
        if not isinstance(object_name, str) or not object_name:
            raise ValueError("dynamic relocation object_name must be non-empty")
        if (
            not isinstance(delta, list)
            or len(delta) != 3
            or not all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in delta
            )
        ):
            raise ValueError("delta_xyz_m must contain three finite numbers")
        delta = [float(value) for value in delta]
        if not 0.02 <= float(np.linalg.norm(delta)) <= 0.15:
            raise ValueError("dynamic relocation magnitude must be 0.02 to 0.15 m")
        if not isinstance(threshold, (int, float)) or not 0.05 <= float(threshold) <= 0.25:
            raise ValueError("eef_distance_below_m must be 0.05 to 0.25 m")
        if not isinstance(minimum_steps, int) or not 0 <= minimum_steps <= 500:
            raise ValueError("minimum_task_steps must be an integer from 0 to 500")
        normalized_relocation = {
            "object_name": object_name,
            "delta_xyz_m": delta,
            "eef_distance_below_m": float(threshold),
            "minimum_task_steps": minimum_steps,
        }
        if object_name in reset_swap_objects | dynamic_swap_objects:
            raise ValueError(
                "dynamic target cannot also participate in an object swap"
            )

    return {
        "schema_version": 1,
        "challenge_cell_id": cell_id,
        "reset_object_swaps": normalized_swaps,
        "dynamic_object_swaps": normalized_dynamic_swaps,
        "dynamic_target_relocation": normalized_relocation,
    }


def _deep_env(env: Any) -> Any:
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current


class ChallengeEnvWrapper:
    """Apply locked reset/dynamic perturbations inside the simulator worker."""

    def __init__(self, env: Any, spec: dict[str, Any], audit_path: str | Path):
        self.env = env
        self.spec = validate_challenge_spec(spec)
        self.audit_path = Path(audit_path)
        self._warmup_steps_remaining = 0
        self._task_steps = 0
        self._relocation_applied = False
        self._dynamic_swaps_applied = False
        self._reset_index = 0
        self._audit_relocation_distances = (
            os.environ.get("ESB_PRIVATE_AUDIT_RELOCATION_DISTANCES") == "1"
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def _object_qpos(self, object_name: str) -> tuple[Any, str, np.ndarray]:
        deep = _deep_env(self.env)
        try:
            model_object = deep.objects_dict[object_name]
        except KeyError as exc:
            raise ValueError(
                f"challenge object {object_name!r} is absent from BDDL scene"
            ) from exc
        joint = model_object.joints[-1]
        qpos = np.asarray(deep.sim.data.get_joint_qpos(joint), dtype=float).copy()
        if qpos.shape != (7,):
            raise ValueError(
                f"challenge object {object_name!r} does not have a free 7D joint"
            )
        return deep, joint, qpos

    def _refresh_observation(self) -> dict[str, Any]:
        deep = _deep_env(self.env)
        deep.sim.forward()
        deep._post_process()
        deep._update_observables(force=True)
        return deep._get_observations()

    def _audit(self, event: dict[str, Any]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "challenge_cell_id": self.spec["challenge_cell_id"],
            "reset_index": self._reset_index,
            **event,
        }
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _apply_reset_swaps(self) -> bool:
        changed = False
        for left, right in self.spec["reset_object_swaps"]:
            deep, left_joint, left_qpos = self._object_qpos(left)
            right_deep, right_joint, right_qpos = self._object_qpos(right)
            if right_deep is not deep:
                raise RuntimeError("challenge objects resolved to different simulators")
            deep.sim.data.set_joint_qpos(left_joint, right_qpos)
            deep.sim.data.set_joint_qpos(right_joint, left_qpos)
            self._audit(
                {
                    "event": "reset_object_swap_applied",
                    "objects": [left, right],
                    "left_before_qpos": left_qpos.tolist(),
                    "right_before_qpos": right_qpos.tolist(),
                }
            )
            changed = True
        return changed

    def _apply_dynamic_swaps(self) -> None:
        if self._dynamic_swaps_applied or self._warmup_steps_remaining:
            return
        for left, right in self.spec["dynamic_object_swaps"]:
            deep, left_joint, left_qpos = self._object_qpos(left)
            right_deep, right_joint, right_qpos = self._object_qpos(right)
            if right_deep is not deep:
                raise RuntimeError("challenge objects resolved to different simulators")
            deep.sim.data.set_joint_qpos(left_joint, right_qpos)
            deep.sim.data.set_joint_qpos(right_joint, left_qpos)
            self._audit(
                {
                    "event": "dynamic_object_swap_applied",
                    "objects": [left, right],
                    "task_step": self._task_steps,
                    "left_before_qpos": left_qpos.tolist(),
                    "right_before_qpos": right_qpos.tolist(),
                }
            )
        if self.spec["dynamic_object_swaps"]:
            self._refresh_observation()
        self._dynamic_swaps_applied = True

    def set_init_state(self, init_state: Any) -> Any:
        observation = self.env.set_init_state(init_state)
        self._reset_index += 1
        self._task_steps = 0
        self._warmup_steps_remaining = 15
        self._relocation_applied = False
        self._dynamic_swaps_applied = False
        if self._apply_reset_swaps():
            observation = self._refresh_observation()
        self._audit(
            {
                "event": "challenge_reset_applied",
                "reset_object_swap_count": len(
                    self.spec["reset_object_swaps"]
                ),
                "dynamic_object_swap_count": len(
                    self.spec["dynamic_object_swaps"]
                ),
                "dynamic_relocation_configured": (
                    self.spec["dynamic_target_relocation"] is not None
                ),
            }
        )
        return observation

    def _should_relocate(self) -> tuple[bool, float | None]:
        relocation = self.spec["dynamic_target_relocation"]
        if relocation is None or self._relocation_applied:
            return False, None
        if self._warmup_steps_remaining or self._task_steps < relocation["minimum_task_steps"]:
            return False, None
        deep, _joint, target_qpos = self._object_qpos(relocation["object_name"])
        robot = deep.robots[0]
        eef_position = np.asarray(
            deep.sim.data.site_xpos[robot.eef_site_id], dtype=float
        )
        distance = float(np.linalg.norm(eef_position - target_qpos[:3]))
        if self._audit_relocation_distances:
            self._audit(
                {
                    "event": "dynamic_target_relocation_distance_observed",
                    "object": relocation["object_name"],
                    "task_step": self._task_steps,
                    "eef_position": eef_position.tolist(),
                    "target_qpos": target_qpos.tolist(),
                    "distance_m": distance,
                    "threshold_m": relocation["eef_distance_below_m"],
                }
            )
        return distance < relocation["eef_distance_below_m"], distance

    def _apply_relocation(self, distance: float) -> None:
        relocation = self.spec["dynamic_target_relocation"]
        assert relocation is not None
        deep, joint, before = self._object_qpos(relocation["object_name"])
        after = before.copy()
        after[:3] += np.asarray(relocation["delta_xyz_m"], dtype=float)
        deep.sim.data.set_joint_qpos(joint, after)
        deep.sim.forward()
        self._relocation_applied = True
        self._audit(
            {
                "event": "dynamic_target_relocation_applied",
                "object": relocation["object_name"],
                "task_step": self._task_steps,
                "eef_distance_before_m": distance,
                "before_qpos": before.tolist(),
                "after_qpos": after.tolist(),
            }
        )

    def step(self, action: Any) -> Any:
        self._apply_dynamic_swaps()
        should_relocate, distance = self._should_relocate()
        if should_relocate:
            assert distance is not None
            self._apply_relocation(distance)
        result = self.env.step(action)
        if self._warmup_steps_remaining:
            self._warmup_steps_remaining -= 1
        else:
            self._task_steps += 1
        return result


def wrap_challenge_env(
    env_factory: Callable[[], Any],
    spec: dict[str, Any],
    audit_path: str | Path,
) -> ChallengeEnvWrapper:
    return ChallengeEnvWrapper(env_factory(), spec, audit_path)

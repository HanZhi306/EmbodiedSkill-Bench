"""Selected SAM3-grounded manipulation through audited PyRoKi/JOINT control.

These actions are deliberately small capability modules.  They consume only
framework-created same-observation grounding cards, keep metric geometry in a
host-private audit, and return one ordinary fresh observation at the enclosing
selected-skill boundary.  They are not task recipes or simulator-state APIs.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .ik import normalize_quaternion_xyzw
from .types import ActionResult


GROUNDED_JOINT_APPROACH_OBJECT = "grounded-joint-approach-object-v1"
GROUNDED_JOINT_GRASP_UPRIGHT_OBJECT = (
    "grounded-joint-grasp-upright-object-v1"
)
GROUNDED_JOINT_PLACE_HELD_IN_OPEN_CONTAINER = (
    "grounded-joint-place-held-in-open-container-v1"
)
GROUNDED_JOINT_SKILL_IDS = frozenset(
    {
        GROUNDED_JOINT_APPROACH_OBJECT,
        GROUNDED_JOINT_GRASP_UPRIGHT_OBJECT,
        GROUNDED_JOINT_PLACE_HELD_IN_OPEN_CONTAINER,
    }
)
GROUNDED_JOINT_PROVIDER = "grounded-joint-pyroki"


def _finite_xyz(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite xyz vector")
    if not (-0.55 <= vector[0] <= 0.55 and -0.55 <= vector[1] <= 0.65):
        raise ValueError(f"{label} lies outside the audited LIBERO workspace")
    if not (-0.05 <= vector[2] <= 0.65):
        raise ValueError(f"{label} has an implausible LIBERO height")
    return vector


def _geometry(
    handoffs: list[dict[str, Any]], *, field: str
) -> dict[str, Any]:
    matches = [
        card
        for card in handoffs
        if str(card.get("matched_argument_field") or "") == field
    ]
    if len(matches) != 1:
        raise ValueError(
            f"grounded JOINT action requires exactly one current-step {field} card"
        )
    card = matches[0]
    step = int(card.get("source_observation_step", -1))
    if step < 0:
        raise ValueError(f"{field} card has no valid observation step")
    center = _finite_xyz(card.get("world_center_median_m"), f"{field}.center")
    lower = _finite_xyz(card.get("world_bounds_p05_m"), f"{field}.lower")
    upper = _finite_xyz(card.get("world_bounds_p95_m"), f"{field}.upper")
    if np.any(upper < lower):
        raise ValueError(f"{field} grounding bounds are inverted")
    extent = upper - lower
    if np.any(extent[:2] > 0.40) or extent[2] > 0.35:
        raise ValueError(f"{field} grounding mask is too large")
    target_ref = str(card.get("target_ref") or "")
    image_sha = str(card.get("source_image_sha256") or "")
    if not target_ref or len(image_sha) != 64:
        raise ValueError(f"{field} card lacks frozen reference provenance")
    return {
        "center": center,
        "lower": lower,
        "upper": upper,
        "extent": extent,
        "target_ref": target_ref,
        "source_image_sha256": image_sha,
        "source_observation_step": step,
    }


def build_grounded_joint_plan(
    skill_id: str, handoffs: list[dict[str, Any]], current_eef_xyz_m: list[float]
) -> dict[str, Any]:
    """Build one bounded, task-agnostic manipulation stage plan."""

    current = _finite_xyz(current_eef_xyz_m, "current_eef")
    if skill_id in {
        GROUNDED_JOINT_APPROACH_OBJECT,
        GROUNDED_JOINT_GRASP_UPRIGHT_OBJECT,
    }:
        target = _geometry(handoffs, field="object")
        x, y = target["center"][:2]
        approach_z = min(0.38, max(0.19, float(target["upper"][2]) + 0.055))
        if skill_id == GROUNDED_JOINT_APPROACH_OBJECT:
            stages = [
                {
                    "operation": "move_pose",
                    "position_xyz_m": [float(x), float(y), approach_z],
                    "gripper": "open",
                    "max_steps": 100,
                }
            ]
        else:
            # This grasp family is intentionally restricted to upright rigid
            # packages/containers.  Flat packets and broad masks must use a
            # different selected skill rather than inheriting unsafe heights.
            if float(target["extent"][2]) < 0.06:
                raise ValueError(
                    "upright-object grasp requires at least 6 cm grounded height"
                )
            grasp_z = max(
                0.075,
                min(0.22, float(target["upper"][2]) - 0.018),
            )
            lift_z = min(0.48, max(0.32, float(target["upper"][2]) + 0.20))
            stages = [
                {
                    "operation": "move_pose",
                    "position_xyz_m": [float(x), float(y), grasp_z],
                    "gripper": "open",
                    "max_steps": 80,
                },
                {"operation": "set_gripper", "state": "closed", "steps": 35},
                {
                    "operation": "move_pose",
                    "position_xyz_m": [float(x), float(y), lift_z],
                    "gripper": "closed",
                    "max_steps": 100,
                },
            ]
    elif skill_id == GROUNDED_JOINT_PLACE_HELD_IN_OPEN_CONTAINER:
        target = _geometry(handoffs, field="container")
        # A receptacle median is often biased toward its visible near wall.
        # The midpoint of robust p05/p95 XY bounds better estimates its opening.
        destination_xy = (target["lower"][:2] + target["upper"][:2]) / 2.0
        carry_z = min(
            0.50,
            max(float(current[2]), 0.32, float(target["upper"][2]) + 0.20),
        )
        release_z = min(
            carry_z - 0.04,
            max(0.16, float(target["upper"][2]) + 0.10),
        )
        if release_z <= float(target["upper"][2]):
            raise ValueError("container geometry cannot produce a safe release height")
        start_xy = current[:2]
        distance = float(np.linalg.norm(destination_xy - start_xy))
        segments = max(1, int(math.ceil(distance / 0.17)))
        stages = []
        if abs(float(current[2]) - carry_z) > 0.012:
            stages.append(
                {
                    "operation": "move_pose",
                    "position_xyz_m": [
                        float(current[0]),
                        float(current[1]),
                        carry_z,
                    ],
                    "gripper": "closed",
                    "max_steps": 90,
                }
            )
        for index in range(1, segments + 1):
            ratio = index / segments
            waypoint = start_xy + ratio * (destination_xy - start_xy)
            stages.append(
                {
                    "operation": "move_pose",
                    "position_xyz_m": [
                        float(waypoint[0]),
                        float(waypoint[1]),
                        carry_z,
                    ],
                    "gripper": "closed",
                    "max_steps": 100,
                }
            )
        stages.extend(
            [
                {
                    "operation": "move_pose",
                    "position_xyz_m": [
                        float(destination_xy[0]),
                        float(destination_xy[1]),
                        release_z,
                    ],
                    "gripper": "closed",
                    "max_steps": 80,
                },
                {"operation": "set_gripper", "state": "open", "steps": 35},
                {
                    "operation": "move_pose",
                    "position_xyz_m": [
                        float(destination_xy[0]),
                        float(destination_xy[1]),
                        carry_z,
                    ],
                    "gripper": "open",
                    "max_steps": 80,
                },
            ]
        )
    else:
        raise ValueError(f"unsupported grounded JOINT skill {skill_id!r}")
    return {
        "schema_version": 1,
        "provider": GROUNDED_JOINT_PROVIDER,
        "skill_id": skill_id,
        "grounding_target_ref": target["target_ref"],
        "source_observation_step": target["source_observation_step"],
        "source_image_sha256": target["source_image_sha256"],
        "stages": stages,
    }


def execute_grounded_joint_action(
    *,
    backend: Any,
    ik_provider: Any,
    handoffs: list[dict[str, Any]],
    skill_id: str,
    audit_dir: Path,
) -> ActionResult:
    """Solve and execute a private plan on the backend's persistent env."""

    capability = backend.p3_joint_controller_capability()
    if not capability or capability.get("audited") is not True:
        raise RuntimeError("grounded JOINT controller attestation failed closed")
    raw = backend._primitives.env.raw_obs()
    current_position = [float(value) for value in raw["robot0_eef_pos"]]
    plan = build_grounded_joint_plan(skill_id, handoffs, current_position)
    stage_results: list[dict[str, Any]] = []
    simulator_steps = 0
    execution_verified = True
    official_success = False
    for index, stage in enumerate(plan["stages"]):
        operation = stage["operation"]
        if operation == "set_gripper":
            action = backend.execute_joint_gripper(
                state=str(stage["state"]),
                steps=int(stage["steps"]),
                controller_sha256=str(capability["controller_sha256"]),
            )
            private_ik = None
        else:
            raw = backend._primitives.env.raw_obs()
            target_position = [float(value) for value in stage["position_xyz_m"]]
            current_position = [float(value) for value in raw["robot0_eef_pos"]]
            current_quaternion = normalize_quaternion_xyzw(
                np.asarray(raw["robot0_eef_quat"], dtype=np.float64).tolist()
            )
            solved = ik_provider.solve(
                position_xyz_m=target_position,
                quaternion_xyzw=current_quaternion,
                previous_joint_positions_rad=backend.current_joint_positions_rad(),
                current_eef_position_xyz_m=current_position,
                current_eef_quaternion_xyzw=current_quaternion,
                position_tolerance_m=0.012,
                orientation_tolerance_rad=0.05,
            )
            private_ik = solved.public()
            if not solved.ok or solved.joint_positions_rad is None:
                execution_verified = False
                stage_results.append(
                    {
                        "index": index,
                        "operation": operation,
                        "execution_verified": False,
                        "simulator_steps": 0,
                        "ik": private_ik,
                        "controller_result": None,
                    }
                )
                break
            action = backend.execute_joint_pose(
                joint_positions_rad=solved.joint_positions_rad,
                target_position_xyz_m=target_position,
                target_quaternion_xyzw=current_quaternion,
                gripper=stage.get("gripper"),
                max_steps=int(stage["max_steps"]),
                position_tolerance_m=0.012,
                orientation_tolerance_rad=0.05,
                controller_sha256=str(capability["controller_sha256"]),
                operation=f"{skill_id}:{index}:{operation}",
                ik_joint_limits_ok=solved.joint_limits_ok,
                joint_lower_limits_rad=solved.provider.get(
                    "joint_lower_limits_rad"
                ),
                joint_upper_limits_rad=solved.provider.get(
                    "joint_upper_limits_rad"
                ),
            )
        result = dict(action.result)
        used = int(action.simulator_steps)
        simulator_steps += used
        verified = bool(result.get("execution_verified"))
        execution_verified = execution_verified and verified
        official_status = result.get("official_status") or {}
        official_success = bool(official_status.get("official_success"))
        stage_results.append(
            {
                "index": index,
                "operation": operation,
                "execution_verified": verified,
                "simulator_steps": used,
                "ik": private_ik,
                "controller_result": result,
            }
        )
        if not verified or official_success:
            break

    audit = {
        "schema_version": 1,
        "plan": plan,
        "stage_results": stage_results,
        "simulator_steps": simulator_steps,
        "execution_verified": execution_verified,
        "official_success": official_success,
        "coordinate_visibility": "host-private-audit-only",
    }
    audit_dir.mkdir(parents=True, exist_ok=False)
    encoded = (
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (audit_dir / "grounded-joint-execution.json").write_bytes(encoded)
    return ActionResult(
        result={
            "provider": GROUNDED_JOINT_PROVIDER,
            "operation": skill_id,
            "planned_stages": len(plan["stages"]),
            "stages_completed": len(stage_results),
            "execution_verified": execution_verified,
            "official_task_success": official_success,
            "hold_or_placement_verified": None,
            "verification_required_from_fresh_observation": True,
            "simulator_steps": simulator_steps,
            "agent_coordinate_access": False,
            "raw_mask_forwarded_to_agent": False,
            "private_audit_sha256": hashlib.sha256(encoded).hexdigest(),
        },
        simulator_steps=simulator_steps,
        vla_chunks=0,
    )

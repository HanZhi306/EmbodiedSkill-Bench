from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


GROUNDED_OSC_FLAT_PACKAGE_TO_WOVEN_BASKET = (
    "grounded-osc-place-flat-package-in-open-woven-basket-v1"
)
GROUNDED_OSC_FLAT_PACKAGE_TO_WOVEN_BASKET_V2 = (
    "grounded-osc-place-flat-package-in-open-woven-basket-v2"
)
GROUNDED_OSC_SKILL_IDS = frozenset(
    {
        GROUNDED_OSC_FLAT_PACKAGE_TO_WOVEN_BASKET,
        GROUNDED_OSC_FLAT_PACKAGE_TO_WOVEN_BASKET_V2,
    }
)
GROUNDED_OSC_PROVIDER = "grounded-osc-pick-place"


def _finite_xyz(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite xyz vector")
    if not (-0.55 <= array[0] <= 0.55 and -0.55 <= array[1] <= 0.65):
        raise ValueError(f"{label} lies outside the audited LIBERO tabletop workspace")
    if not (-0.05 <= array[2] <= 0.65):
        raise ValueError(f"{label} has an implausible tabletop height")
    return array


def _card_geometry(card: dict[str, Any], label: str) -> dict[str, Any]:
    if int(card.get("source_observation_step", -1)) < 0:
        raise ValueError(f"{label} grounding card has no valid observation step")
    center = _finite_xyz(card.get("world_center_median_m"), f"{label}.center")
    lower = _finite_xyz(card.get("world_bounds_p05_m"), f"{label}.lower")
    upper = _finite_xyz(card.get("world_bounds_p95_m"), f"{label}.upper")
    if np.any(upper < lower):
        raise ValueError(f"{label} grounding bounds are inverted")
    extent = upper - lower
    if np.any(extent[:2] > 0.40) or extent[2] > 0.35:
        raise ValueError(f"{label} grounding mask is too large for this action")
    target_ref = str(card.get("target_ref") or "")
    if not target_ref:
        raise ValueError(f"{label} grounding card has no opaque target_ref")
    return {
        "center": center,
        "lower": lower,
        "upper": upper,
        "target_ref": target_ref,
        "source_image_sha256": str(card.get("source_image_sha256") or ""),
        "source_observation_step": int(card["source_observation_step"]),
    }


def build_flat_package_to_woven_basket_plan(
    handoffs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a bounded top-down OSC plan from two same-step opaque masks.

    This is intentionally narrower than a general manipulation planner. It is
    valid only for one flat rigid package and one already-open woven basket.
    The Agent never supplies or receives coordinates; the host consumes the
    metric geometry stored in framework-created grounding cards.
    """

    cards: dict[str, dict[str, Any]] = {}
    for card in handoffs:
        field = str(card.get("matched_argument_field") or "")
        if field in {"object", "destination"}:
            if field in cards:
                raise ValueError(f"multiple grounding cards matched {field}")
            cards[field] = card
    missing = {"object", "destination"} - set(cards)
    if missing:
        raise ValueError(
            "grounded OSC action requires same-observation source and destination "
            f"masks; missing {sorted(missing)}"
        )
    source = _card_geometry(cards["object"], "object")
    destination = _card_geometry(cards["destination"], "destination")
    if source["source_observation_step"] != destination["source_observation_step"]:
        raise ValueError("object and destination masks must come from the same observation")
    if source["source_image_sha256"] != destination["source_image_sha256"]:
        raise ValueError("object and destination masks must use the same public RGB frame")
    if source["target_ref"] == destination["target_ref"]:
        raise ValueError("object and destination must be distinct grounded instances")

    sx, sy = source["center"][:2]
    dx, dy = destination["center"][:2]
    source_top = float(source["upper"][2])
    destination_rim = float(destination["upper"][2])
    pregrasp_z = max(0.18, source_top + 0.15)
    grasp_z = max(0.025, source_top - 0.004)
    carry_z = max(0.28, source_top + 0.20, destination_rim + 0.18)
    placement_z = max(0.12, destination_rim + 0.025)
    if carry_z > 0.55 or placement_z >= carry_z:
        raise ValueError("grounded geometry cannot produce a bounded basket trajectory")

    stages = [
        {"operation": "move_to", "xyz": [sx, sy, pregrasp_z], "gripper": -1.0, "max_steps": 80},
        {"operation": "move_to", "xyz": [sx, sy, grasp_z], "gripper": -1.0, "max_steps": 80},
        {"operation": "set_gripper", "gripper": 1.0, "steps": 35},
        {"operation": "move_to", "xyz": [sx, sy, carry_z], "gripper": 1.0, "max_steps": 100},
        {"operation": "move_to", "xyz": [dx, dy, carry_z], "gripper": 1.0, "max_steps": 140},
        {"operation": "move_to", "xyz": [dx, dy, placement_z], "gripper": 1.0, "max_steps": 100},
        {"operation": "set_gripper", "gripper": -1.0, "steps": 30},
        {"operation": "move_to", "xyz": [dx, dy, carry_z], "gripper": -1.0, "max_steps": 80},
    ]
    return {
        "schema_version": 1,
        "provider": GROUNDED_OSC_PROVIDER,
        "source_target_ref": source["target_ref"],
        "destination_target_ref": destination["target_ref"],
        "source_observation_step": source["source_observation_step"],
        "stages": stages,
    }


def build_flat_package_to_woven_basket_v2_plan(
    handoffs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the evidence-driven v2 plan without changing v1 history.

    The v27 construction probe localized both instances correctly, but its
    first low diagonal transit struck a tall distractor and its gripper stayed
    aligned with the package's long axis.  V2 therefore transits at carry
    height, keeps a 90-degree world yaw throughout the motion, and lowers the
    grasp target by one centimetre.  These are fixed properties of this narrow
    flat-package skill, not Agent-selected waypoints.
    """

    base = build_flat_package_to_woven_basket_plan(handoffs)
    stages = [dict(stage) for stage in base["stages"]]
    source_xyz = list(stages[0]["xyz"])
    destination_xyz = list(stages[4]["xyz"])
    carry_z = float(stages[4]["xyz"][2])
    source_card = next(
        card for card in handoffs if card.get("matched_argument_field") == "object"
    )
    source_top = float(_card_geometry(source_card, "object")["upper"][2])
    grasp_z = max(0.012, source_top - 0.010)
    grasp_yaw = math.pi / 2.0

    # Stage 0 must be a high-clearance transit all the way to the source xy.
    stages[0]["xyz"] = [source_xyz[0], source_xyz[1], carry_z]
    stages[0]["target_yaw"] = grasp_yaw
    stages[0]["max_steps"] = 120
    stages[1]["xyz"] = [source_xyz[0], source_xyz[1], grasp_z]
    stages[1]["target_yaw"] = grasp_yaw
    for index in (3, 4, 5, 7):
        stages[index]["target_yaw"] = grasp_yaw
    # Preserve the destination/carry geometry from the validated v1 builder.
    assert stages[4]["xyz"] == destination_xyz

    return {
        **base,
        "schema_version": 2,
        "controller_revision": "high-clearance-short-edge-yaw-v2",
        "stages": stages,
    }


def execute_flat_package_to_woven_basket(
    *,
    primitives: Any,
    handoffs: list[dict[str, Any]],
    audit_dir: Path,
    skill_id: str = GROUNDED_OSC_FLAT_PACKAGE_TO_WOVEN_BASKET,
) -> dict[str, Any]:
    """Execute the private plan and return a coordinate-free public summary."""

    if skill_id == GROUNDED_OSC_FLAT_PACKAGE_TO_WOVEN_BASKET:
        plan = build_flat_package_to_woven_basket_plan(handoffs)
    elif skill_id == GROUNDED_OSC_FLAT_PACKAGE_TO_WOVEN_BASKET_V2:
        plan = build_flat_package_to_woven_basket_v2_plan(handoffs)
    else:
        raise ValueError(f"unsupported grounded OSC skill {skill_id!r}")
    audit_dir.mkdir(parents=True, exist_ok=True)
    stage_results: list[dict[str, Any]] = []
    simulator_steps = 0
    for index, stage in enumerate(plan["stages"]):
        operation = stage["operation"]
        arguments = {key: value for key, value in stage.items() if key != "operation"}
        if operation == "move_to":
            result = dict(primitives.move_to(**arguments))
            if float(result.get("final_dist_m", 1.0)) > 0.035:
                raise RuntimeError(
                    f"grounded OSC stage {index} did not reach its bounded pose"
                )
            used = int(result.get("steps_used", 0))
        elif operation == "set_gripper":
            result = dict(primitives.set_gripper(**arguments))
            used = int(result.get("steps_used", result.get("steps", 0)))
        else:  # pragma: no cover - plan builder owns the closed operation set
            raise RuntimeError(f"unknown grounded OSC operation {operation}")
        simulator_steps += used
        stage_results.append(
            {
                "index": index,
                "operation": operation,
                "steps_used": used,
                "controller_reached": (
                    float(result.get("final_dist_m", 0.0)) <= 0.035
                    if operation == "move_to"
                    else True
                ),
                "libero_terminated": bool(result.get("libero_terminated", False)),
                "private_result": result,
            }
        )
        if bool(result.get("libero_terminated", False)):
            break

    audit = {
        "schema_version": 1,
        "plan": plan,
        "stage_results": stage_results,
        "simulator_steps": simulator_steps,
        "coordinate_visibility": "host-private-audit-only",
    }
    encoded = (json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    audit_path = audit_dir / "grounded-osc-execution.json"
    audit_path.write_bytes(encoded)
    return {
        "provider": GROUNDED_OSC_PROVIDER,
        "operation": "place-flat-package-in-open-woven-basket",
        "controller_revision": str(plan.get("controller_revision") or "legacy-v1"),
        "stages_completed": len(stage_results),
        "planned_stages": len(plan["stages"]),
        "simulator_steps": simulator_steps,
        # ActionResult consumes this canonical field. Keep simulator_steps as a
        # human-readable alias, but never under-count a multi-stage controller.
        "steps_used": simulator_steps,
        "libero_terminated": bool(
            stage_results and stage_results[-1]["libero_terminated"]
        ),
        "source_and_destination_grounded": True,
        "agent_coordinate_access": False,
        "raw_mask_forwarded_to_agent": False,
        "private_audit_sha256": hashlib.sha256(encoded).hexdigest(),
    }

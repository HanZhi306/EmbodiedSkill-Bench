"""Frozen task-agnostic schemas for native typed no-skill controllers.

This module deliberately has no benchmark, simulator, or OpenHands imports.
The runtime, generated artifacts, and isolated OpenHands tool bridge all consume
these exact dictionaries so the model-visible schema cannot drift from runtime
validation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


P25_PROTOCOL = "P2.5-native-typed-osc-no-skill-v1"
P3_PROTOCOL = "P3-native-typed-full-pose-ik-no-skill-v1"
RAW_PI05_PROTOCOL = "M0-native-typed-raw-pi05-no-skill-library-v1"
RAW_PI05_RGB_PROTOCOL = "M0R-native-typed-rgb-raw-pi05-no-skill-library-v1"
P3_IK_PROVIDER_CONTRACT = {
    "upstream": "https://github.com/chungmin99/pyroki",
    "revision": "95afccc22658c461ab1042a048ae4e9c24bc2a47",
    "integration_reference": "https://github.com/capgym/cap-x",
    "cap_x_commit": "53e9966d7a8e2fa7494676772bccc35280f5c0ed",
    "jaxls_revision": "6fe7cf9d56223736b2c0272d080ec7346203c648",
    "quaternion_wire_order": "wxyz",
    "public_quaternion_order": "xyzw",
    "tcp_offset_tool_frame_m": [0.0, 0.0, -0.1],
    "frame_alignment": (
        "derive LIBERO-world to Panda-model-base rigid transform from the "
        "same persistent environment's measured EEF pose and joint FK"
    ),
    "collision_free_claim": False,
}


def _number(description: str, *, minimum: float | None = None,
            maximum: float | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"type": "number", "description": description}
    if minimum is not None:
        value["minimum"] = minimum
    if maximum is not None:
        value["maximum"] = maximum
    return value


def _vector(length: int, description: str, *, minimum: float | None = None,
            maximum: float | None = None) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": length,
        "maxItems": length,
        "items": _number("Finite scalar.", minimum=minimum, maximum=maximum),
        "description": description,
    }


def _object(description: str, properties: dict[str, Any],
            required: list[str] | None = None) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "description": description,
        "properties": properties,
        "required": required or [],
    }


_CAMERA = {
    "type": "string",
    "enum": ["agentview", "wrist"],
    "default": "agentview",
    "description": "Camera name; wrist calibration moves with the robot.",
}
_RESOLUTION = {
    "type": "string",
    "enum": ["low", "high"],
    "default": "high",
    "description": "Image resolution. Pixels use row-from-top, column-from-left.",
}
_GRIPPER_OPTIONAL = {
    "type": ["string", "null"],
    "enum": ["open", "closed", None],
    "default": None,
    "description": (
        "Optional gripper command. null or omission preserves the previous command; "
        "the gripper starts open."
    ),
}


P25_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "observe": _object(
        "Read the current sanitized task, robot, status, and RGB artifact ledger. Read-only.",
        {},
    ),
    "get_eef_state": _object(
        "Read current end-effector and neutral gripper state. Read-only.", {},
    ),
    "get_camera_calibration": _object(
        "Read task-agnostic camera intrinsics/extrinsics in the LIBERO world frame.",
        {"camera": _CAMERA},
    ),
    "read_depth_pixel": _object(
        "Read metric depth at one low-resolution pixel, optionally with a median window.",
        {
            "row": {"type": "integer", "minimum": 0, "maximum": 255,
                    "description": "Low-resolution row from top, in [0,255]."},
            "col": {"type": "integer", "minimum": 0, "maximum": 255,
                    "description": "Low-resolution column from left, in [0,255]."},
            "camera": _CAMERA,
            "window_radius": {
                "type": "integer", "minimum": 0, "maximum": 4, "default": 0,
                "description": "Square median-window radius in low-resolution pixels.",
            },
        },
        ["row", "col"],
    ),
    "pixel_to_world": _object(
        "Back-project one pixel to metric XYZ in the LIBERO world frame. No semantics.",
        {
            "row": {"type": "integer", "minimum": 0, "description": "Row from top."},
            "col": {"type": "integer", "minimum": 0, "description": "Column from left."},
            "camera": _CAMERA,
            "resolution": _RESOLUTION,
        },
        ["row", "col"],
    ),
    "get_gripper_contact": _object(
        "Read an identity-free gripper-width/contact heuristic; it is not a grasp verifier.", {},
    ),
    "move_eef_delta": _object(
        "Move through the frozen OSC controller by a world-frame XYZ delta; state-changing.",
        {
            "delta_xyz_m": _vector(
                3, "LIBERO-world XYZ delta in meters; each component is bounded.",
                minimum=-0.25, maximum=0.25,
            ),
            "gripper": _GRIPPER_OPTIONAL,
            "max_steps": {
                "type": "integer", "minimum": 1, "maximum": 200, "default": 80,
                "description": "Bounded OSC environment-step budget.",
            },
        },
        ["delta_xyz_m"],
    ),
    "move_eef_pose": _object(
        "Move through the frozen OSC controller to world XYZ with optional pitch/yaw; state-changing.",
        {
            "position_xyz_m": _vector(3, "Absolute XYZ in the LIBERO world frame, meters."),
            "pitch_rad": {
                **_number("Optional RPent pitch in radians; omission preserves orientation.",
                          minimum=-3.141592653589793, maximum=3.141592653589793),
                "default": None, "type": ["number", "null"],
            },
            "yaw_rad": {
                **_number("Optional world yaw in radians; omission preserves orientation.",
                          minimum=-3.141592653589793, maximum=3.141592653589793),
                "default": None, "type": ["number", "null"],
            },
            "gripper": _GRIPPER_OPTIONAL,
            "max_steps": {
                "type": "integer", "minimum": 1, "maximum": 200, "default": 100,
                "description": "Bounded OSC environment-step budget.",
            },
        },
        ["position_xyz_m"],
    ),
    "set_gripper": _object(
        "Open or close the gripper through the frozen controller; state-changing.",
        {
            "state": {"type": "string", "enum": ["open", "closed"],
                      "description": "Requested gripper command."},
            "steps": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20,
                      "description": "Bounded controller step budget."},
        },
        ["state"],
    ),
}

# The pixel range is part of the native schema rather than a runtime-only API
# literacy trap.  Omitted resolution takes the declared high-resolution
# default.  Draft 2020-12 validation applies the matching upper bound before
# any geometry query reaches the backend.
P25_TOOL_SCHEMAS["pixel_to_world"]["allOf"] = [
    {
        "if": {
            "properties": {"resolution": {"const": "low"}},
            "required": ["resolution"],
        },
        "then": {
            "properties": {
                "row": {"maximum": 255},
                "col": {"maximum": 255},
            }
        },
        "else": {
            "properties": {
                "row": {"maximum": 1023},
                "col": {"maximum": 1023},
            }
        },
    }
]


P3_ADDITIONAL_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "check_ik_pose": _object(
        "Solve and validate full-pose IK without changing simulator state. Not collision-free.",
        {
            "position_xyz_m": _vector(3, "Requested tool-center-point XYZ in LIBERO world meters."),
            "quaternion_xyzw": _vector(
                4, "Requested unit orientation quaternion ordered [x,y,z,w]; q and -q are equivalent."
            ),
            "position_tolerance_m": {
                **_number("Maximum FK position residual in meters.", minimum=0.0001, maximum=0.05),
                "default": 0.01,
            },
            "orientation_tolerance_rad": {
                **_number("Maximum sign-invariant FK orientation residual in radians.", minimum=0.001, maximum=0.5),
                "default": 0.05,
            },
        },
        ["position_xyz_m", "quaternion_xyzw"],
    ),
    "move_eef_pose_ik": _object(
        "Execute a validated full-pose IK target through a joint-controller backend; state-changing and not collision-free.",
        {
            "position_xyz_m": _vector(3, "Requested tool-center-point XYZ in LIBERO world meters."),
            "quaternion_xyzw": _vector(
                4, "Requested unit orientation quaternion ordered [x,y,z,w]; q and -q are equivalent."
            ),
            "gripper": _GRIPPER_OPTIONAL,
            "max_steps": {"type": "integer", "minimum": 1, "maximum": 240, "default": 120,
                          "description": "Bounded joint-controller environment-step budget."},
            "position_tolerance_m": {
                **_number("Required final position residual in meters.", minimum=0.0001, maximum=0.05),
                "default": 0.01,
            },
            "orientation_tolerance_rad": {
                **_number("Required sign-invariant final orientation residual in radians.", minimum=0.001, maximum=0.5),
                "default": 0.05,
            },
        },
        ["position_xyz_m", "quaternion_xyzw"],
    ),
}


RAW_PI05_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    **{
        name: copy.deepcopy(P25_TOOL_SCHEMAS[name])
        for name in (
            "observe",
            "get_eef_state",
            "get_camera_calibration",
            "read_depth_pixel",
            "pixel_to_world",
            "get_gripper_contact",
        )
    },
    "raw_vla_subgoal": _object(
        (
            "Execute the frozen Pi0.5 policy on the current observation for one "
            "agent-authored natural-language subgoal. This is a raw model interface: "
            "it adds no semantic action template, skill recipe, recovery guidance, "
            "or postcondition; state-changing."
        ),
        {
            "instruction": {
                "type": "string",
                "minLength": 1,
                "maxLength": 500,
                "pattern": ".*\\S.*",
                "description": (
                    "One task-local natural-language subgoal written by the Agent. "
                    "Whitespace is normalized and the text is otherwise passed verbatim "
                    "to frozen Pi0.5."
                ),
            },
            "max_chunks": {
                "type": "integer",
                "minimum": 1,
                "maximum": 40,
                "default": 20,
                "description": (
                    "Maximum Pi0.5 forward/action chunks for this physical round; each "
                    "native chunk contains five 7D LIBERO actions."
                ),
            },
        },
        ["instruction"],
    ),
}

# Capability-matched primary baseline for native selected-skill experiments.
# The model receives the same harness-projected RGB/robot/status observation as
# S1 and exactly one raw Pi0.5 action surface.  The older RAW_PI05_PROTOCOL is
# intentionally preserved because its calibrated geometry tools are part of
# already-generated development artifacts.
RAW_PI05_RGB_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "raw_vla_subgoal": copy.deepcopy(RAW_PI05_TOOL_SCHEMAS["raw_vla_subgoal"]),
}


ACTION_TOOLS = frozenset(
    {
        "move_eef_delta",
        "move_eef_pose",
        "set_gripper",
        "move_eef_pose_ik",
        "raw_vla_subgoal",
    }
)


def tool_schemas(protocol: str) -> dict[str, dict[str, Any]]:
    if protocol == P25_PROTOCOL:
        value = P25_TOOL_SCHEMAS
    elif protocol == P3_PROTOCOL:
        value = {**P25_TOOL_SCHEMAS, **P3_ADDITIONAL_TOOL_SCHEMAS}
    elif protocol == RAW_PI05_PROTOCOL:
        value = RAW_PI05_TOOL_SCHEMAS
    elif protocol == RAW_PI05_RGB_PROTOCOL:
        value = RAW_PI05_RGB_TOOL_SCHEMAS
    else:
        raise ValueError(f"unsupported native typed protocol: {protocol!r}")
    schemas = copy.deepcopy(value)
    if protocol == P3_PROTOCOL:
        schemas["move_eef_delta"]["description"] = (
            "Preserve the measured TCP quaternion, solve the world-frame XYZ "
            "delta with frozen PyRoKi, then execute through the sole audited "
            "JOINT_POSITION environment; state-changing."
        )
        schemas["move_eef_delta"]["properties"]["max_steps"]["description"] = (
            "Bounded JOINT_POSITION environment-step budget."
        )
        schemas["move_eef_pose"]["description"] = (
            "Convert optional RPent-compatible pitch/yaw to a full quaternion, "
            "solve frozen PyRoKi IK, then execute through the sole audited "
            "JOINT_POSITION environment; omitted angles preserve measured orientation."
        )
        schemas["move_eef_pose"]["properties"]["max_steps"]["description"] = (
            "Bounded JOINT_POSITION environment-step budget."
        )
        schemas["set_gripper"]["description"] = (
            "Hold the seven measured Panda arm joints and command only the eighth "
            "JOINT_POSITION gripper slot; state-changing."
        )
        schemas["pixel_to_world"]["properties"]["resolution"]["description"] = (
            "Image resolution. In P3, high is a deterministic nearest-neighbor "
            "projection of the native 256x256 RGB-D and adds no spatial information."
        )
    return schemas


def contract_document(protocol: str) -> dict[str, Any]:
    schemas = tool_schemas(protocol)
    raw_pi05 = protocol in {RAW_PI05_PROTOCOL, RAW_PI05_RGB_PROTOCOL}
    rgb_only_raw_pi05 = protocol == RAW_PI05_RGB_PROTOCOL
    document = {
        "schema_version": 1,
        "protocol": protocol,
        "controller_kind": (
            "frozen-Pi0.5-7D-action-chunks+OSC_POSE-environment"
            if raw_pi05
            else "OSC_POSE"
            if protocol == P25_PROTOCOL
            else "JOINT_POSITION+PyRoKi"
        ),
        "action_boundary": "first state-changing call ends the physical round",
        "multiple_read_only_calls_per_llm_response": not rgb_only_raw_pi05,
        "second_action_same_llm_response": "rejected_before_simulator",
        "skill_library_exposed": False,
        "learned_perception_exposed": False,
        "vla_exposed": raw_pi05,
        "semantic_vla_skill_wrappers_exposed": False,
        "observation_surface": (
            "harness-projected-RGB+robot+status-only"
            if rgb_only_raw_pi05
            else "calibrated-geometry-native-tools+projected-observation"
            if protocol == RAW_PI05_PROTOCOL
            else "native-typed-controller-observation"
        ),
        "raw_vla_provider": (
            {
                "public_operation": "raw_vla_subgoal",
                "policy_profile": "rpent-generalist",
                "frozen_model_family": "Pi0.5",
                "instruction_transform": "normalize-whitespace-only",
                "native_action_shape": ["chunk", 7],
                "native_chunk_horizon": 5,
                "per_call_chunk_range": [1, 40],
                "default_max_chunks": 20,
                "semantic_templates": False,
                "skill_postconditions": False,
            }
            if raw_pi05
            else None
        ),
        "tools": schemas,
    }
    if protocol == P3_PROTOCOL:
        document["ik_provider_dependency"] = copy.deepcopy(P3_IK_PROVIDER_CONTRACT)
        document["simulator_execution_gate"] = "audited JOINT_POSITION backend required"
        document["controller_contract"] = {
            "repository_owned_backend": True,
            "single_persistent_environment": True,
            "osc_controller_present": False,
            "action_dim": 8,
            "arm_action_dim": 7,
            "arm_command": "(target_joint_rad-current_joint_rad)*20Hz",
            "gripper_command": {"open": -1.0, "closed": 1.0},
            "all_state_changing_tools_share_this_backend": True,
            "official_success_source": "OffScreenRenderEnv.check_success",
            "native_camera_resolution": [256, 256],
            "high_resolution_capture": (
                "nearest-neighbor derived from native RGB-D; no additional spatial information"
            ),
        }
    return document


def contract_sha256(protocol: str) -> str:
    payload = json.dumps(contract_document(protocol), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def contract_markdown(protocol: str) -> str:
    document = contract_document(protocol)
    capability_boundary = (
        "A single raw frozen Pi0.5 subgoal interface is available, but no semantic VLA "
        "wrapper, skill document, recipe, memory, recovery guidance, or learned perception "
        "service is exposed."
        if protocol in {RAW_PI05_PROTOCOL, RAW_PI05_RGB_PROTOCOL}
        else "No VLA or learned perception service is exposed."
    )
    lines = [
        f"# {protocol}", "",
        "This is a task-agnostic no-skill controller contract. No skill documents, recipes,",
        "semantic coordinates, raw Python, shell, or simulator object are exposed.",
        capability_boundary, "",
        "The first state-changing tool call ends the physical round and returns a fresh observation.",
        "Read-only tools, when present, may be called repeatedly before that action. A second action from the same",
        "LLM response is rejected before simulator execution.", "",
    ]
    for name, schema in document["tools"].items():
        mode = "state-changing" if name in ACTION_TOOLS else "read-only"
        lines.extend([f"## `{name}` ({mode})", "", str(schema["description"]), "", "```json",
                      json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)

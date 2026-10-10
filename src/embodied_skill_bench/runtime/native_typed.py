"""Native typed no-skill runtime for P2.5 and P3."""

from __future__ import annotations

import math
import threading
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .ik import (
    IKProvider,
    IKResult,
    UnavailableIKProvider,
    normalize_quaternion_xyzw,
    quaternion_with_libero_pitch_yaw,
)
from .native_typed_contract import (
    ACTION_TOOLS,
    P25_PROTOCOL,
    P3_PROTOCOL,
    RAW_PI05_PROTOCOL,
    RAW_PI05_RGB_PROTOCOL,
    contract_sha256,
    tool_schemas,
)
from .primitive_code import PrimitiveCodeSession, RestrictedProgramError, _ActionBoundary
from .turn_authority import AgentTraceTurnAuthority


class NativeTypedSession:
    """Typed facade that reuses P2's exact geometry and OSC implementation.

    No source code, shell command, executable path, or simulator object crosses
    this boundary. The OpenHands bridge sends only one named JSON tool call.
    """

    def __init__(
        self,
        *,
        episode_id: str,
        run_dir: str | Path,
        backend: Any,
        catalog: Any,
        max_actions: int,
        protocol: str = P25_PROTOCOL,
        enforce_agent_protocol: bool = True,
        turn_authority: AgentTraceTurnAuthority | None = None,
        ik_provider: IKProvider | None = None,
    ):
        if protocol not in {
            P25_PROTOCOL,
            P3_PROTOCOL,
            RAW_PI05_PROTOCOL,
            RAW_PI05_RGB_PROTOCOL,
        }:
            raise ValueError("unsupported native typed protocol")
        self.protocol = protocol
        self.schemas = tool_schemas(protocol)
        self.schema_sha256 = contract_sha256(protocol)
        self.enforce_agent_protocol = bool(enforce_agent_protocol)
        self.turn_authority = turn_authority
        self._primitive = PrimitiveCodeSession(
            episode_id=episode_id,
            run_dir=run_dir,
            backend=backend,
            catalog=catalog,
            max_actions=max_actions,
            max_programs=1,
            max_perception_queries=0,
            perception_mode="calibrated-geometry-p2",
            enforce_agent_protocol=False,
            additional_internal_action_adapters=(
                ("pi0-doubled",)
                if protocol in {RAW_PI05_PROTOCOL, RAW_PI05_RGB_PROTOCOL}
                else ()
            ),
        )
        self.store = self._primitive.store
        self._ik_provider = ik_provider or UnavailableIKProvider()
        self._lock = threading.Lock()
        self._responses: dict[str, dict[str, Any]] = {}
        self._tool_calls = 0
        self._invalid_tool_schema = 0
        self._action_budget_exhausted = 0
        self._read_calls = 0
        self._action_calls = 0
        self._call_counts: Counter[str] = Counter()
        self._used_action_turn_ids: set[str] = set()
        self._protocol_violations: list[dict[str, Any]] = []
        self._previous_joint_positions_rad: list[float] | None = None
        self._closed = False
        # Initial RGB is attached to the initial user message. It is therefore
        # the first acknowledged observation without an artificial observe call.
        self._acknowledged_step = int(self._primitive._episode.summary()["step_index"])

    def execute_native_tool(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = str(request.get("request_id") or uuid.uuid4())
        with self._lock:
            if request_id in self._responses:
                replay = dict(self._responses[request_id])
                replay["idempotent_replay"] = True
                return replay
            response = self._execute_once(request_id, request)
            self._responses[request_id] = response
            return response

    def _execute_once(self, request_id: str, request: dict[str, Any]) -> dict[str, Any]:
        tool_name = request.get("tool_name")
        arguments = request.get("arguments")
        self._tool_calls += 1
        if not isinstance(tool_name, str) or tool_name not in self.schemas:
            return self._invalid(request_id, str(tool_name), "unknown native typed tool")
        if not isinstance(arguments, dict):
            return self._invalid(request_id, tool_name, "arguments must be a JSON object")
        try:
            Draft202012Validator(self.schemas[tool_name]).validate(arguments)
            self._validate_finite(arguments)
        except (ValidationError, ValueError, TypeError) as exc:
            return self._invalid(request_id, tool_name, str(exc))
        if self._closed:
            return self._reject(request_id, tool_name, "runtime_closed", "runtime is closed")

        claimed_turn = str(request.get("agent_turn_id") or "")
        canonical_turn = claimed_turn
        if self.enforce_agent_protocol:
            if not claimed_turn:
                return self._protocol_reject(request_id, tool_name, "missing_agent_turn_id",
                                             "native tool request is missing harness turn identity")
            if self.turn_authority:
                canonical_turn = self.turn_authority.active_turn("native_tool", tool_name) or ""
                if not canonical_turn:
                    return self._protocol_reject(
                        request_id, tool_name, "unverified_native_tool_turn",
                        "request does not match an active OpenHands native tool call",
                    )
        if tool_name in ACTION_TOOLS:
            if canonical_turn and canonical_turn in self._used_action_turn_ids:
                return self._protocol_reject(
                    request_id, tool_name, "multiple_actions_in_one_llm_response",
                    "a canonical LLM response may issue at most one state-changing call",
                )
            current_step = int(self._primitive._episode.summary()["step_index"])
            if self._acknowledged_step != current_step:
                return self._protocol_reject(
                    request_id, tool_name, "fresh_observation_not_consumed",
                    "the next action requires the fresh observation from the previous physical round",
                )
            episode_summary = self._primitive._episode.summary()
            if int(episode_summary["action_count"]) >= int(self._primitive._episode.max_actions):
                self._action_budget_exhausted += 1
                return self._reject(
                    request_id,
                    tool_name,
                    "action_budget_exhausted",
                    f"action limit reached ({self._primitive._episode.max_actions})",
                )
            if canonical_turn:
                self._used_action_turn_ids.add(canonical_turn)

        self._call_counts[tool_name] += 1
        if tool_name in ACTION_TOOLS:
            self._action_calls += 1
        else:
            self._read_calls += 1

        try:
            if self.protocol in {
                RAW_PI05_PROTOCOL,
                RAW_PI05_RGB_PROTOCOL,
            } and tool_name == "raw_vla_subgoal":
                return self._raw_vla_subgoal(
                    request_id, tool_name, canonical_turn, arguments
                )
            if tool_name == "check_ik_pose":
                result = self._check_ik(arguments)
                return self._success(request_id, tool_name, canonical_turn, result, False)
            if tool_name == "move_eef_pose_ik":
                return self._move_ik(request_id, tool_name, canonical_turn, arguments)
            if self.protocol == P3_PROTOCOL and tool_name in {
                "move_eef_delta",
                "move_eef_pose",
                "set_gripper",
            }:
                return self._move_p3_inherited(
                    request_id, tool_name, canonical_turn, arguments
                )
            result, state_changed = self._call_p25(tool_name, arguments)
        except (RestrictedProgramError, ValueError, TypeError, KeyError, IndexError) as exc:
            return self._reject(request_id, tool_name, "tool_execution_error", str(exc))

        if state_changed:
            self._acknowledged_step = int(self._primitive._episode.summary()["step_index"])
        return self._success(request_id, tool_name, canonical_turn, result, state_changed)

    @staticmethod
    def _validate_finite(value: Any) -> None:
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                raise ValueError("all numeric tool arguments must be finite")
            return
        if isinstance(value, list):
            for item in value:
                NativeTypedSession._validate_finite(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                NativeTypedSession._validate_finite(item)

    def _call_p25(self, tool_name: str, a: dict[str, Any]) -> tuple[Any, bool]:
        p = self._primitive
        p._active_program_index = self._tool_calls
        try:
            if tool_name == "observe":
                return p._api_observe(), False
            if tool_name == "get_eef_state":
                return p._api_get_eef_state(), False
            if tool_name == "get_camera_calibration":
                return p._api_get_camera_calibration(str(a.get("camera", "agentview"))), False
            if tool_name == "read_depth_pixel":
                return p._api_read_depth_pixel(
                    int(a["row"]), int(a["col"]), str(a.get("camera", "agentview")),
                    int(a.get("window_radius", 0)),
                ), False
            if tool_name == "pixel_to_world":
                return p._api_pixel_to_world(
                    int(a["row"]), int(a["col"]), str(a.get("camera", "agentview")),
                    str(a.get("resolution", "high")),
                ), False
            if tool_name == "get_gripper_contact":
                return p._api_get_gripper_contact(), False
            if tool_name == "move_eef_delta":
                p._api_move_eef_delta(
                    a["delta_xyz_m"], a.get("gripper"), int(a.get("max_steps", 80))
                )
            elif tool_name == "move_eef_pose":
                p._api_move_eef_pose(
                    a["position_xyz_m"], a.get("pitch_rad"), a.get("yaw_rad"),
                    a.get("gripper"), int(a.get("max_steps", 100)),
                )
            elif tool_name == "set_gripper":
                p._api_set_gripper(str(a["state"]), int(a.get("steps", 20)))
            else:
                raise ValueError("unsupported P2.5 tool")
        except _ActionBoundary as boundary:
            response = boundary.response
            if not response.get("accepted") and "limit" in str(response.get("error", {})).lower():
                self._action_budget_exhausted += 1
            return {
                "controller_response": response,
                "controller_feedback": response.get("controller_feedback"),
            }, bool(response.get("accepted"))
        finally:
            p._active_program_index = None
        raise AssertionError("state-changing primitive returned without action boundary")

    def _raw_vla_subgoal(
        self,
        request_id: str,
        tool_name: str,
        turn: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute one host-only raw Pi0.5 provider call under a public operation.

        The legacy ``pi0-doubled`` catalog record is an implementation adapter only.
        It is never installed in the Agent workspace and its SKILL.md is never read.
        The model-written instruction receives whitespace normalization but no
        semantic action template, postcondition, recovery hint, or task rewrite.
        """

        instruction = " ".join(str(arguments["instruction"]).split())
        if not instruction:
            return self._invalid(
                request_id,
                tool_name,
                "instruction must contain at least one non-whitespace character",
            )
        max_chunks = int(arguments.get("max_chunks", 20))
        response = self._primitive._episode.execute_adapter(
            request_id=f"native-raw-pi05-{request_id}",
            adapter_skill_id="pi0-doubled",
            public_operation_id=tool_name,
            arguments={"prompt": instruction, "max_chunks": max_chunks},
        )
        if not response.get("accepted"):
            error = response.get("error") or {}
            if "limit" in str(error).lower():
                self._action_budget_exhausted += 1
            return self._reject(
                request_id,
                tool_name,
                str(error.get("code") or "raw_vla_execution_rejected"),
                str(error.get("message") or "raw Pi0.5 execution was rejected"),
                details=response,
            )
        self._primitive._action_sequence.append(tool_name)
        self._acknowledged_step = int(
            self._primitive._episode.summary()["step_index"]
        )
        result = {
            "provider": "frozen-pi0.5",
            "policy_profile": "rpent-generalist",
            "instruction": instruction,
            "instruction_transform": "normalize-whitespace-only",
            "max_chunks": max_chunks,
            "semantic_action_template_used": False,
            "skill_postcondition_used": False,
            "provider_response": response,
        }
        public = self._success(
            request_id,
            tool_name,
            turn,
            result,
            True,
            ok=bool(response.get("ok")),
        )
        public.update(
            {
                "read_only": False,
                "physical_action_attempted": True,
                "physical_round_ended": True,
                "automatic_observation_handoff": True,
            }
        )
        if not response.get("ok"):
            public["error"] = response.get("error") or {
                "type": "RawVLAExecutionError",
                "code": "raw_vla_execution_error",
                "message": "the raw Pi0.5 provider call failed",
            }
        return public

    def _check_ik(self, a: dict[str, Any]):
        q = normalize_quaternion_xyzw(a["quaternion_xyzw"])
        current_position, current_quaternion = self._current_eef_pose()
        return self._ik_provider.solve(
            position_xyz_m=[float(v) for v in a["position_xyz_m"]],
            quaternion_xyzw=q,
            previous_joint_positions_rad=self._current_joint_seed(),
            current_eef_position_xyz_m=current_position,
            current_eef_quaternion_xyzw=current_quaternion,
            position_tolerance_m=float(a.get("position_tolerance_m", 0.01)),
            orientation_tolerance_rad=float(a.get("orientation_tolerance_rad", 0.05)),
        ).public()

    def _current_joint_seed(self) -> list[float] | None:
        if self._previous_joint_positions_rad is not None:
            return list(self._previous_joint_positions_rad)
        getter = getattr(self._primitive.backend, "current_joint_positions_rad", None)
        if callable(getter):
            value = [float(v) for v in getter()]
            return value if len(value) == 7 else None
        return None

    def _current_eef_pose(self) -> tuple[list[float], list[float]]:
        eef = self.store.latest()["robot"]["end_effector"]
        return (
            [float(v) for v in eef["position_m"]],
            normalize_quaternion_xyzw(eef["quaternion"]),
        )

    def _move_ik(self, request_id: str, tool_name: str, turn: str, a: dict[str, Any]) -> dict[str, Any]:
        q = normalize_quaternion_xyzw(a["quaternion_xyzw"])
        current_position, current_quaternion = self._current_eef_pose()
        solved = self._ik_provider.solve(
            position_xyz_m=[float(v) for v in a["position_xyz_m"]],
            quaternion_xyzw=q,
            previous_joint_positions_rad=self._current_joint_seed(),
            current_eef_position_xyz_m=current_position,
            current_eef_quaternion_xyzw=current_quaternion,
            position_tolerance_m=float(a.get("position_tolerance_m", 0.01)),
            orientation_tolerance_rad=float(a.get("orientation_tolerance_rad", 0.05)),
        )
        if not solved.ok or solved.joint_positions_rad is None:
            return self._reject(request_id, tool_name, solved.status,
                                str((solved.error or {}).get("message", "IK validation failed")),
                                details=solved.public())
        return self._execute_p3_pose_solution(
            request_id=request_id,
            tool_name=tool_name,
            turn=turn,
            arguments=a,
            target_position=[float(v) for v in a["position_xyz_m"]],
            target_quaternion=q,
            solved=solved,
        )

    def _joint_controller_attestation(self) -> dict[str, Any] | None:
        capability = getattr(
            self._primitive.backend, "p3_joint_controller_capability", None
        )
        attestation = capability() if callable(capability) else None
        if not isinstance(attestation, dict) or not (
            attestation.get("audited") is True
            and attestation.get("controller_kind") == "JOINT_POSITION"
            and isinstance(attestation.get("controller_sha256"), str)
            and len(attestation["controller_sha256"]) == 64
            and attestation.get("fresh_observation_handoff") is True
            and attestation.get("single_persistent_environment") is True
            and attestation.get("osc_controller_present") is False
        ):
            return None
        return attestation

    def _move_p3_inherited(
        self,
        request_id: str,
        tool_name: str,
        turn: str,
        a: dict[str, Any],
    ) -> dict[str, Any]:
        attestation = self._joint_controller_attestation()
        if attestation is None:
            return self._reject(
                request_id,
                tool_name,
                "p3_joint_controller_unavailable",
                "P3 inherited action has no audited single JOINT_POSITION environment",
            )
        if tool_name == "set_gripper":
            executor = getattr(self._primitive.backend, "execute_joint_gripper", None)
            if not callable(executor):
                return self._reject(
                    request_id,
                    tool_name,
                    "p3_joint_controller_unavailable",
                    "JOINT_POSITION backend has no gripper execution path",
                )
            controller_arguments = {
                "state": str(a["state"]),
                "steps": int(a.get("steps", 20)),
                "controller_kind": attestation["controller_kind"],
                "controller_sha256": attestation["controller_sha256"],
                "environment_instance_id": attestation.get(
                    "environment_instance_id"
                ),
            }
            response = self._primitive._episode.execute_p3_joint_action(
                request_id=f"native-p3-{request_id}",
                public_operation_id=tool_name,
                arguments=controller_arguments,
                executor=lambda: executor(
                    state=controller_arguments["state"],
                    steps=controller_arguments["steps"],
                    controller_sha256=controller_arguments["controller_sha256"],
                ),
            )
            return self._physical_response(
                request_id=request_id,
                tool_name=tool_name,
                turn=turn,
                response=response,
                result={"controller_response": response},
                gripper_command=str(a["state"]),
            )

        observation = self.store.latest()
        eef = observation["robot"]["end_effector"]
        current_position = [float(v) for v in eef["position_m"]]
        current_quaternion = normalize_quaternion_xyzw(eef["quaternion"])
        if tool_name == "move_eef_delta":
            delta = [float(v) for v in a["delta_xyz_m"]]
            target_position = [
                current_position[index] + delta[index] for index in range(3)
            ]
            target_quaternion = current_quaternion
            position_tolerance = 0.012
            orientation_tolerance = 0.05
            max_steps = int(a.get("max_steps", 80))
        elif tool_name == "move_eef_pose":
            target_position = [float(v) for v in a["position_xyz_m"]]
            target_quaternion = quaternion_with_libero_pitch_yaw(
                current_quaternion,
                pitch_rad=a.get("pitch_rad"),
                yaw_rad=a.get("yaw_rad"),
            )
            position_tolerance = 0.012
            orientation_tolerance = 0.05
            max_steps = int(a.get("max_steps", 100))
        else:
            raise AssertionError("unsupported inherited P3 action")
        solved = self._ik_provider.solve(
            position_xyz_m=target_position,
            quaternion_xyzw=target_quaternion,
            previous_joint_positions_rad=self._current_joint_seed(),
            current_eef_position_xyz_m=current_position,
            current_eef_quaternion_xyzw=current_quaternion,
            position_tolerance_m=position_tolerance,
            orientation_tolerance_rad=orientation_tolerance,
        )
        if not solved.ok or solved.joint_positions_rad is None:
            return self._reject(
                request_id,
                tool_name,
                solved.status,
                str((solved.error or {}).get("message", "IK validation failed")),
                details=solved.public(),
            )
        normalized_arguments = {
            **a,
            "max_steps": max_steps,
            "position_tolerance_m": position_tolerance,
            "orientation_tolerance_rad": orientation_tolerance,
        }
        return self._execute_p3_pose_solution(
            request_id=request_id,
            tool_name=tool_name,
            turn=turn,
            arguments=normalized_arguments,
            target_position=target_position,
            target_quaternion=target_quaternion,
            solved=solved,
        )

    def _execute_p3_pose_solution(
        self,
        *,
        request_id: str,
        tool_name: str,
        turn: str,
        arguments: dict[str, Any],
        target_position: list[float],
        target_quaternion: list[float],
        solved: IKResult,
    ) -> dict[str, Any]:
        # A coincidentally named method is not sufficient evidence that a
        # simulator backend can execute audited joint targets.  Require a
        # preflight attestation before allowing any state change.  P2.5's
        # OSC_POSE backend intentionally has neither capability.
        attestation = self._joint_controller_attestation()
        if attestation is None:
            return self._reject(
                request_id, tool_name, "p3_joint_controller_unavailable",
                "no audited JOINT_POSITION backend with fresh-observation handoff is active",
                details={"ik": solved.public(), "controller_attestation": attestation},
            )
        executor = getattr(self._primitive.backend, "execute_joint_pose", None)
        if not callable(executor):
            return self._reject(
                request_id, tool_name, "p3_joint_controller_unavailable",
                "current backend is OSC_POSE and has no audited joint-target execution path",
                details={"ik": solved.public(), "controller_attestation": attestation},
            )
        controller_arguments = {
            "joint_positions_rad": list(solved.joint_positions_rad or []),
            "target_position_xyz_m": list(target_position),
            "target_quaternion_xyzw": list(target_quaternion),
            "gripper": arguments.get("gripper"),
            "max_steps": int(arguments.get("max_steps", 120)),
            "position_tolerance_m": float(
                arguments.get("position_tolerance_m", 0.01)
            ),
            "orientation_tolerance_rad": float(
                arguments.get("orientation_tolerance_rad", 0.05)
            ),
            "controller_kind": attestation["controller_kind"],
            "controller_sha256": attestation["controller_sha256"],
            "environment_instance_id": attestation.get(
                "environment_instance_id"
            ),
            "ik_joint_limits_ok": solved.joint_limits_ok,
            "joint_lower_limits_rad": solved.provider.get(
                "joint_lower_limits_rad"
            ),
            "joint_upper_limits_rad": solved.provider.get(
                "joint_upper_limits_rad"
            ),
            "operation": tool_name,
        }
        response = self._primitive._episode.execute_p3_joint_action(
            request_id=f"native-p3-{request_id}",
            public_operation_id=tool_name,
            arguments=controller_arguments,
            executor=lambda: executor(**controller_arguments),
        )
        if not response.get("accepted"):
            return self._reject(
                request_id,
                tool_name,
                "p3_joint_execution_rejected",
                str((response.get("error") or {}).get("message", "joint execution rejected")),
                details=response,
            )
        if response.get("ok"):
            # The frozen provider's validated solution is the continuity seed;
            # a failed physical convergence must instead fall back to measured
            # state on the next solve.
            self._previous_joint_positions_rad = list(
                solved.joint_positions_rad or []
            )
        else:
            getter = getattr(
                self._primitive.backend, "current_joint_positions_rad", None
            )
            measured = [float(v) for v in getter()] if callable(getter) else []
            self._previous_joint_positions_rad = (
                measured if len(measured) == 7 else None
            )
        return self._physical_response(
            request_id=request_id,
            tool_name=tool_name,
            turn=turn,
            response=response,
            result={"ik": solved.public(), "controller_response": response},
            gripper_command=arguments.get("gripper"),
        )

    def _physical_response(
        self,
        *,
        request_id: str,
        tool_name: str,
        turn: str,
        response: dict[str, Any],
        result: dict[str, Any],
        gripper_command: str | None,
    ) -> dict[str, Any]:
        if not response.get("accepted"):
            return self._reject(
                request_id,
                tool_name,
                "p3_joint_execution_rejected",
                str(
                    (response.get("error") or {}).get(
                        "message", "joint execution rejected"
                    )
                ),
                details=response,
            )
        if gripper_command is not None and response.get("state_changed"):
            self._primitive._last_gripper_command = str(gripper_command)
        self._acknowledged_step = int(
            self._primitive._episode.summary()["step_index"]
        )
        public = self._success(
            request_id,
            tool_name,
            turn,
            result,
            bool(response.get("state_changed")),
            ok=bool(response.get("ok")),
        )
        public.update(
            {
                "read_only": False,
                "physical_action_attempted": bool(
                    response.get("physical_action_attempted")
                ),
                # A failed zero-step simulator call still ends the physical
                # round and hands back a freshly captured observation.  It is
                # not falsely reported as an environment state transition.
                "physical_round_ended": True,
                "automatic_observation_handoff": bool(
                    response.get("fresh_observation_handoff")
                ),
            }
        )
        if response.get("fresh_observation_handoff"):
            public["observation"] = self._primitive._public_observation()
        if not response.get("ok"):
            public["error"] = response.get("error") or {
                "type": "ControllerGoalNotReached",
                "code": "controller_goal_not_reached",
                "message": "the accepted physical action did not pass its execution gate",
            }
        return public

    def _base_response(self, request_id: str, tool_name: str) -> dict[str, Any]:
        pointer = self.store.latest_pointer()
        return {
            "request_id": request_id,
            "tool_name": tool_name,
            "protocol": self.protocol,
            "contract_sha256": self.schema_sha256,
            "step_index": self._primitive._episode.summary()["step_index"],
            "observation_json": pointer["observation_json"],
            "observation_markdown": pointer["observation_markdown"],
            "status": self.store.latest()["status"],
        }

    def _success(
        self,
        request_id: str,
        tool_name: str,
        turn: str,
        result: Any,
        state_changed: bool,
        *,
        ok: bool = True,
    ) -> dict[str, Any]:
        response = {
            **self._base_response(request_id, tool_name),
            "ok": bool(ok),
            "accepted": True,
            "agent_turn_id": turn or None,
            "read_only": not state_changed,
            "state_changed": state_changed,
            "physical_round_ended": state_changed,
            "automatic_observation_handoff": state_changed,
            "observation_acknowledged": self._acknowledged_step == self._primitive._episode.summary()["step_index"],
            "result": result,
        }
        if state_changed:
            response["observation"] = self._primitive._public_observation()
        return response

    def _invalid(self, request_id: str, tool_name: str, message: str) -> dict[str, Any]:
        self._invalid_tool_schema += 1
        return self._reject(request_id, tool_name, "invalid_tool_schema", message)

    def _reject(self, request_id: str, tool_name: str, code: str, message: str,
                *, details: Any = None) -> dict[str, Any]:
        response = {
            **self._base_response(request_id, tool_name),
            "ok": False,
            "accepted": False,
            "state_changed": False,
            "physical_round_ended": tool_name in ACTION_TOOLS,
            "automatic_observation_handoff": False,
            "error": {"type": "NativeTypedToolError", "code": code, "message": message,
                      "details": details},
        }
        if code == "action_budget_exhausted":
            response["terminal_for_agent"] = True
            response["terminal_reason"] = "no further state-changing calls are available"
        return response

    def _protocol_reject(self, request_id: str, tool_name: str, code: str,
                         message: str) -> dict[str, Any]:
        self._protocol_violations.append({"code": code, "message": message,
                                          "step_index": self._primitive._episode.summary()["step_index"]})
        return self._reject(request_id, tool_name, code, message)

    def latest(self, *, acknowledge: bool = False, agent_turn_id: str | None = None) -> dict[str, Any]:
        # HTTP latest exists for benchmark-side projection only. Agent-side
        # acknowledgement must arrive through the native observe tool.
        if acknowledge:
            return {"ok": False, "error": {"code": "native_tool_required",
                                             "message": "use the native observe tool"}}
        return self._primitive.latest(acknowledge=False, agent_turn_id=agent_turn_id)

    def finish(self, reason: str = "agent_finished") -> dict[str, Any]:
        base = self._primitive.finish(reason)
        provider_sequence = list(base.get("action_primitive_sequence", []))
        provider_counts = dict(Counter(provider_sequence))
        raw_pi05 = self.protocol in {RAW_PI05_PROTOCOL, RAW_PI05_RGB_PROTOCOL}
        base.update({
            "method": "native-typed-no-skill-agent",
            "native_typed_protocol": self.protocol,
            "native_typed_contract_sha256": self.schema_sha256,
            "program_count": 0,
            "program_calls": {"completed": 0, "rejected": 0, "errors": 0},
            "native_tool_calls": {
                "total": self._tool_calls,
                "read_only": self._read_calls,
                "state_changing": self._action_calls,
                "invalid_tool_schema": self._invalid_tool_schema,
                "action_budget_exhausted": self._action_budget_exhausted,
                "by_name": dict(sorted(self._call_counts.items())),
            },
            "benchmark_action_budget": {
                "exhausted": self._action_budget_exhausted > 0,
                "classification": (
                    "clean_budget_stop" if self._action_budget_exhausted > 0 else None
                ),
                "official_simulator_truncated": bool(base.get("truncated")),
            },
            "protocol_violations": list(self._protocol_violations),
            "protocol_enforced": self.enforce_agent_protocol,
            "protocol_turn_identity_source": (
                self.turn_authority.source
                if self.turn_authority
                else "direct"
            ),
            "skill_sequence": [], "skill_call_counts": {},
            "provider_action_sequence": provider_sequence,
            "provider_action_call_counts": provider_counts,
            "embodied_skill_library_exposed": False,
            "vla_exposed": raw_pi05,
            "raw_vla_provider_exposed": raw_pi05,
            "semantic_vla_skill_wrappers_exposed": False,
            "sam3_exposed": False,
            "contact_graspnet_exposed": False,
        })
        return base

    def summary(self) -> dict[str, Any]:
        # Keep finish/summary identical except for the emitted finish event.
        base = self._primitive.summary()
        provider_sequence = list(base.get("action_primitive_sequence", []))
        provider_counts = dict(Counter(provider_sequence))
        raw_pi05 = self.protocol in {RAW_PI05_PROTOCOL, RAW_PI05_RGB_PROTOCOL}
        base.update({
            "method": "native-typed-no-skill-agent",
            "native_typed_protocol": self.protocol,
            "native_typed_contract_sha256": self.schema_sha256,
            "native_tool_calls": {
                "total": self._tool_calls, "read_only": self._read_calls,
                "state_changing": self._action_calls,
                "invalid_tool_schema": self._invalid_tool_schema,
                "action_budget_exhausted": self._action_budget_exhausted,
                "by_name": dict(sorted(self._call_counts.items())),
            },
            "benchmark_action_budget": {
                "exhausted": self._action_budget_exhausted > 0,
                "classification": (
                    "clean_budget_stop" if self._action_budget_exhausted > 0 else None
                ),
                "official_simulator_truncated": bool(base.get("truncated")),
            },
            "skill_sequence": [], "skill_call_counts": {},
            "provider_action_sequence": provider_sequence,
            "provider_action_call_counts": provider_counts,
            "embodied_skill_library_exposed": False,
            "vla_exposed": raw_pi05,
            "raw_vla_provider_exposed": raw_pi05,
            "semantic_vla_skill_wrappers_exposed": False,
            "sam3_exposed": False,
            "contact_graspnet_exposed": False,
            "protocol_violations": list(self._protocol_violations),
            "protocol_enforced": self.enforce_agent_protocol,
            "protocol_turn_identity_source": (
                self.turn_authority.source
                if self.turn_authority
                else "direct"
            ),
        })
        return base

    def close(self) -> None:
        if not self._closed:
            self._primitive.close()
            self._closed = True

"""Restricted code-agent runtime without an embodied skill library.

The public surface is deliberately smaller than Python.  Programs are parsed
as Python syntax for model familiarity, then interpreted from the AST.  They
are never passed to ``exec``/``eval`` and cannot import modules, access object
attributes, open files, use the network, or reach the simulator object.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import math
import shutil
import threading
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from ..catalog import SkillCatalog
from ..io import append_jsonl, atomic_write_json
from .session import EpisodeSession
from .store import utc_now
from .turn_authority import AgentTraceTurnAuthority


PRIMITIVE_API_SCHEMA_VERSION = 2
PRIMITIVE_PERCEPTION_MODES = (
    "observation-only",
    "observation-only-p1",
    "calibrated-geometry-p2",
    "frozen-sam3",
)
PRIMITIVE_RGB_PROJECTION_MODES = frozenset(
    {
        "observation-only",
        "observation-only-p1",
        "calibrated-geometry-p2",
    }
)
PRIMITIVE_AUTO_HANDOFF_MODES = frozenset(
    {"observation-only-p1", "calibrated-geometry-p2"}
)
PRIMITIVE_GEOMETRY_MODES = frozenset({"calibrated-geometry-p2"})
CONTROLLER_POSITION_TOLERANCE_M = 0.012
CONTROLLER_ORIENTATION_TOLERANCE_RAD = 0.05
EMPTY_CLOSURE_SUSPICION_UPPER_M = 0.006
INTERNAL_ACTION_ADAPTERS = ("move-to", "move-pose", "set-gripper")
READ_PRIMITIVES = (
    "observe",
    "load_rgb",
    "load_depth",
    "segment_text",
    "mask_to_world_points",
    "estimate_obb",
    "propose_grasps",
    "get_eef_state",
    "get_camera_calibration",
    "read_depth_pixel",
    "pixel_to_world",
    "get_gripper_contact",
)
ACTION_PRIMITIVES = ("move_eef_delta", "move_eef_pose", "set_gripper")


def primitive_api_spec(
    perception_mode: str = "frozen-sam3",
    *,
    repeat_error_feedback: bool = False,
) -> dict[str, Any]:
    """Return the frozen, task-agnostic public API contract."""

    if perception_mode not in PRIMITIVE_PERCEPTION_MODES:
        raise ValueError(
            "unsupported primitive-code mode"
        )
    read_only = {
        "observe": "Return the sanitized current observation object.",
        "get_eef_state": "Return public end-effector and gripper state.",
    }
    if perception_mode in PRIMITIVE_GEOMETRY_MODES:
        read_only.update(
            {
                "get_camera_calibration": (
                    "Return numeric intrinsics/extrinsics for one public camera."
                ),
                "read_depth_pixel": (
                    "Read metric depth at an agent-chosen pixel; no segmentation is performed."
                ),
                "pixel_to_world": (
                    "Back-project one agent-chosen low- or high-resolution RGB pixel into the LIBERO world frame."
                ),
                "get_gripper_contact": (
                    "Return identity-free gripper closure/contact evidence from proprioception."
                ),
            }
        )
    elif perception_mode == "frozen-sam3":
        read_only.update(
            {
                "load_rgb": "Return an opaque RGB handle and public metadata.",
                "load_depth": "Return an opaque depth handle and bounded statistics.",
                "segment_text": "Run frozen SAM3 text segmentation on a public RGB artifact.",
                "mask_to_world_points": "Back-project one returned mask through the public world map.",
                "estimate_obb": "Estimate a PCA oriented bounding box from a points handle.",
                "propose_grasps": "Return deterministic top-down pose proposals from a points handle.",
            }
        )
    schema_version = (
        4
        if perception_mode == "calibrated-geometry-p2"
        else 3
        if perception_mode == "observation-only-p1"
        else PRIMITIVE_API_SCHEMA_VERSION
    )
    protocol_version = {
        "observation-only": "P0.1-strict-rgb-floor-gripper-hold",
        "observation-only-p1": "P1.1-rgb-harness-literacy-gripper-hold",
        "calibrated-geometry-p2": "P2.1-standard-calibrated-geometry-gripper-hold",
        "frozen-sam3": "diagnostic-frozen-sam3-v2-gripper-hold",
    }[perception_mode]
    if repeat_error_feedback:
        schema_version += 1
        protocol_version += "-repeat-error-feedback"
    return {
        "schema_version": schema_version,
        "method": "primitive-code-agent",
        "embodied_skill_library_exposed": False,
        "vla_exposed": False,
        "round_boundary": "first_state_changing_primitive",
        "protocol_version": protocol_version,
        "repeated_program_error_feedback": repeat_error_feedback,
        "gripper_command_semantics": {
            "initial_state": "open",
            "move_argument_omitted": "preserve_previous_command",
            "explicit_values": ["open", "closed"],
        },
        "automatic_post_action_observation": (
            perception_mode in PRIMITIVE_AUTO_HANDOFF_MODES
        ),
        "perception_mode": perception_mode,
        "perception_provider": (
            "frozen-sam3" if perception_mode == "frozen-sam3" else None
        ),
        "grasp_proposal_provider": (
            "deterministic-top-down-geometry-v1"
            if perception_mode == "frozen-sam3"
            else None
        ),
        "read_only": read_only,
        "state_changing": {
            "move_eef_delta": (
                "Move by a LIBERO-world-frame XYZ delta through the frozen OSC adapter."
            ),
            "move_eef_pose": (
                "Move to a LIBERO-world-frame XYZ/pitch/yaw pose through the frozen OSC adapter."
            ),
            "set_gripper": "Open or close the gripper through the frozen controller adapter.",
        },
        "security": {
            "python_exec_used": False,
            "imports": False,
            "attribute_access": False,
            "filesystem": False,
            "network": False,
            "reflection": False,
            "private_simulator_state": False,
        },
    }


class _ActionBoundary(BaseException):
    """Internal non-catchable control signal that ends a physical round."""

    def __init__(self, primitive: str, response: dict[str, Any]):
        self.primitive = primitive
        self.response = response


class _BreakLoop(BaseException):
    pass


class RestrictedProgramError(ValueError):
    pass


class RestrictedAstInterpreter:
    """A bounded interpreter for a small, auditable subset of Python."""

    def __init__(
        self,
        functions: dict[str, Callable[..., Any]],
        *,
        max_ast_nodes: int,
        max_execution_nodes: int,
        max_loop_iterations: int,
        timeout_seconds: float,
    ):
        self.functions = dict(functions)
        self.max_ast_nodes = int(max_ast_nodes)
        self.max_execution_nodes = int(max_execution_nodes)
        self.max_loop_iterations = int(max_loop_iterations)
        self.timeout_seconds = float(timeout_seconds)
        self.variables: dict[str, Any] = {}
        self.outputs: list[Any] = []
        self._executed = 0
        self._started = 0.0

    def run(self, source: str) -> dict[str, Any]:
        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError as exc:
            raise RestrictedProgramError(f"invalid program syntax: {exc.msg}") from exc
        nodes = list(ast.walk(tree))
        if len(nodes) > self.max_ast_nodes:
            raise RestrictedProgramError(
                f"program AST has {len(nodes)} nodes; limit is {self.max_ast_nodes}"
            )
        forbidden = (
            ast.Import,
            ast.ImportFrom,
            ast.Attribute,
            ast.Lambda,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.With,
            ast.AsyncWith,
            ast.Try,
            ast.Raise,
            ast.Global,
            ast.Nonlocal,
            ast.Delete,
            ast.Await,
            ast.Yield,
            ast.YieldFrom,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.NamedExpr,
        )
        for node in nodes:
            if isinstance(node, forbidden):
                raise RestrictedProgramError(
                    f"{type(node).__name__} is not available in restricted programs"
                )
        self._started = time.monotonic()
        for statement in tree.body:
            self._statement(statement)
        return {
            "outputs": _json_value(self.outputs),
            "variables": _json_value(self.variables),
            "executed_nodes": self._executed,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return state reached before an action boundary stopped execution."""

        return {
            "outputs": _json_value(self.outputs),
            "variables": _json_value(self.variables),
            "executed_nodes": self._executed,
        }

    def _tick(self) -> None:
        self._executed += 1
        if self._executed > self.max_execution_nodes:
            raise RestrictedProgramError(
                f"program execution exceeded {self.max_execution_nodes} evaluated nodes"
            )
        if time.monotonic() - self._started > self.timeout_seconds:
            raise RestrictedProgramError(
                f"program exceeded {self.timeout_seconds:g}s interpreter limit"
            )

    def _statement(self, node: ast.stmt) -> None:
        self._tick()
        if isinstance(node, ast.Expr):
            self._expression(node.value)
            return
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise RestrictedProgramError("assignment targets must be one variable name")
            self.variables[node.targets[0].id] = self._expression(node.value)
            return
        if isinstance(node, ast.If):
            branch = node.body if bool(self._expression(node.test)) else node.orelse
            for statement in branch:
                self._statement(statement)
            return
        if isinstance(node, ast.For):
            if not isinstance(node.target, ast.Name):
                raise RestrictedProgramError("for-loop target must be one variable name")
            values = self._expression(node.iter)
            if not isinstance(values, (list, tuple)):
                raise RestrictedProgramError("for-loop iterable must be a bounded list/tuple/range")
            if len(values) > self.max_loop_iterations:
                raise RestrictedProgramError(
                    f"loop has {len(values)} iterations; limit is {self.max_loop_iterations}"
                )
            broke = False
            for value in values:
                self.variables[node.target.id] = value
                try:
                    for statement in node.body:
                        self._statement(statement)
                except _BreakLoop:
                    broke = True
                    break
            if not broke:
                for statement in node.orelse:
                    self._statement(statement)
            return
        if isinstance(node, ast.Break):
            raise _BreakLoop()
        if isinstance(node, ast.Pass):
            return
        raise RestrictedProgramError(
            f"statement {type(node).__name__} is not available in restricted programs"
        )

    def _expression(self, node: ast.expr) -> Any:
        self._tick()
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, int, float, bool)) or node.value is None:
                return node.value
            raise RestrictedProgramError("unsupported literal")
        if isinstance(node, ast.Name):
            if node.id in self.variables:
                return self.variables[node.id]
            raise RestrictedProgramError(f"unknown variable {node.id!r}")
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._expression(item) for item in node.elts]
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise RestrictedProgramError("dictionary unpacking is unavailable")
            return {
                str(self._expression(key)): self._expression(value)
                for key, value in zip(node.keys, node.values, strict=True)
            }
        if isinstance(node, ast.Subscript):
            value = self._expression(node.value)
            index = self._expression(node.slice)
            try:
                return value[index]
            except (KeyError, IndexError, TypeError) as exc:
                raise RestrictedProgramError(f"invalid subscript: {exc}") from exc
        if isinstance(node, ast.UnaryOp):
            value = self._expression(node.operand)
            if isinstance(node.op, ast.Not):
                return not bool(value)
            if isinstance(node.op, ast.USub):
                return -_number(value)
            if isinstance(node.op, ast.UAdd):
                return +_number(value)
            raise RestrictedProgramError("unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left, right = self._expression(node.left), self._expression(node.right)
            if isinstance(node.op, ast.Add):
                if isinstance(left, list) and isinstance(right, list):
                    result = left + right
                    if len(result) > self.max_loop_iterations:
                        raise RestrictedProgramError("constructed list exceeds size limit")
                    return result
                return _number(left) + _number(right)
            if isinstance(node.op, ast.Sub):
                return _number(left) - _number(right)
            if isinstance(node.op, ast.Mult):
                return _number(left) * _number(right)
            if isinstance(node.op, ast.Div):
                return _number(left) / _number(right)
            if isinstance(node.op, ast.FloorDiv):
                return _number(left) // _number(right)
            if isinstance(node.op, ast.Mod):
                return _number(left) % _number(right)
            if isinstance(node.op, ast.Pow):
                exponent = _number(right)
                if abs(exponent) > 8:
                    raise RestrictedProgramError("power exponent magnitude is limited to 8")
                return _number(left) ** exponent
            raise RestrictedProgramError("unsupported binary operator")
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result: Any = True
                for item in node.values:
                    result = self._expression(item)
                    if not result:
                        return result
                return result
            result = False
            for item in node.values:
                result = self._expression(item)
                if result:
                    return result
            return result
        if isinstance(node, ast.Compare):
            left = self._expression(node.left)
            for operator, comparator in zip(node.ops, node.comparators, strict=True):
                right = self._expression(comparator)
                if not _compare(left, operator, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self._expression(node.body if self._expression(node.test) else node.orelse)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise RestrictedProgramError("only direct calls to named primitives are allowed")
            name = node.func.id
            if name not in self.functions:
                raise RestrictedProgramError(f"call to unavailable function {name!r}")
            if any(keyword.arg is None for keyword in node.keywords):
                raise RestrictedProgramError("**kwargs expansion is unavailable")
            args = [self._expression(argument) for argument in node.args]
            kwargs = {
                str(keyword.arg): self._expression(keyword.value)
                for keyword in node.keywords
            }
            if name == "print":
                if kwargs:
                    raise RestrictedProgramError("print does not accept keyword arguments")
                self.outputs.append(_json_value(args))
                return None
            return self.functions[name](*args, **kwargs)
        raise RestrictedProgramError(
            f"expression {type(node).__name__} is not available in restricted programs"
        )


class PrimitiveCodeSession:
    """Program-level facade over the existing observation and OSC runtime."""

    def __init__(
        self,
        *,
        episode_id: str,
        run_dir: str | Path,
        backend: Any,
        catalog: SkillCatalog,
        max_actions: int,
        max_programs: int = 128,
        max_perception_queries: int = 32,
        max_program_chars: int = 4096,
        max_ast_nodes: int = 256,
        max_execution_nodes: int = 512,
        max_loop_iterations: int = 64,
        program_timeout_seconds: float = 10.0,
        perception_mode: str = "frozen-sam3",
        enforce_agent_protocol: bool = True,
        turn_authority: AgentTraceTurnAuthority | None = None,
        repeat_error_feedback: bool = False,
        additional_internal_action_adapters: tuple[str, ...] = (),
    ):
        self.episode_id = episode_id
        self.run_dir = Path(run_dir).resolve()
        self.backend = backend
        self.catalog = catalog
        if perception_mode not in PRIMITIVE_PERCEPTION_MODES:
            raise ValueError("unsupported primitive-code mode")
        if perception_mode != "frozen-sam3" and int(max_perception_queries) != 0:
            raise ValueError(
                "non-SAM3 primitive-code modes require max_perception_queries=0"
            )
        self.perception_mode = perception_mode
        self.auto_observation_handoff = (
            perception_mode in PRIMITIVE_AUTO_HANDOFF_MODES
        )
        internal_action_adapters = tuple(
            dict.fromkeys(
                (*INTERNAL_ACTION_ADAPTERS, *additional_internal_action_adapters)
            )
        )
        catalog.require(internal_action_adapters)
        self._episode = EpisodeSession(
            episode_id=episode_id,
            run_dir=self.run_dir,
            backend=backend,
            catalog=catalog,
            allowed_skill_ids=list(internal_action_adapters),
            max_actions=max_actions,
            max_perception_queries=0,
            enforce_agent_protocol=False,
        )
        self.store = self._episode.store
        self.max_programs = int(max_programs)
        self.max_perception_queries = int(max_perception_queries)
        self.max_program_chars = int(max_program_chars)
        self.max_ast_nodes = int(max_ast_nodes)
        self.max_execution_nodes = int(max_execution_nodes)
        self.max_loop_iterations = int(max_loop_iterations)
        self.program_timeout_seconds = float(program_timeout_seconds)
        self.enforce_agent_protocol = bool(enforce_agent_protocol)
        self.turn_authority = turn_authority
        self.repeat_error_feedback = bool(repeat_error_feedback)
        self._lock = threading.Lock()
        self._responses: dict[str, dict[str, Any]] = {}
        self._program_count = 0
        self._perception_count = 0
        self._read_call_count = 0
        self._action_call_count = 0
        self._program_errors = 0
        self._program_rejections = 0
        self._last_program_failure_key: tuple[str, str, str] | None = None
        self._consecutive_same_program_failure_count = 0
        self._repeated_identical_program_errors = 0
        self._acknowledged_step: int | None = None
        self._used_program_turn_ids: set[str] = set()
        self._protocol_violations: list[dict[str, Any]] = []
        self._call_counts: Counter[str] = Counter()
        self._action_sequence: list[str] = []
        self._handles: dict[str, dict[str, Any]] = {}
        # LIBERO resets with an open gripper.  The public move API promises
        # that omitting ``gripper`` preserves the previous command.  Always
        # materialize that state in the private adapter request because RPent's
        # move_to/move_pose defaults are -1.0 (open), which would otherwise
        # silently release a payload after a separate close command.
        self._last_gripper_command = "open"
        self._closed = False

    def latest(
        self, *, acknowledge: bool = False, agent_turn_id: str | None = None
    ) -> dict[str, Any]:
        protocol_error = None
        if acknowledge:
            with self._lock:
                canonical = agent_turn_id
                if self.enforce_agent_protocol and self.turn_authority:
                    canonical = self.turn_authority.active_turn("primitive_observe")
                    if not canonical:
                        protocol_error = self._protocol_violation(
                            "unverified_observation_tool_turn",
                            "observation acknowledgement was not issued by an active `./esbench-code observe` tool turn",
                        )
                if protocol_error is None:
                    self._acknowledged_step = self._episode.summary()["step_index"]
                    self._event(
                        "observation_acknowledged",
                        {
                            "step_index": self._acknowledged_step,
                            "source": "primitive_code_observe",
                            "agent_turn_id": canonical,
                        },
                    )
        step = self._episode.summary()["step_index"]
        return {
            "ok": True,
            "episode_id": self.episode_id,
            "observation": self._public_observation(),
            "pointer": self.store.latest_pointer(),
            "observation_acknowledged": self._acknowledged_step == step,
            "protocol_error": protocol_error,
        }

    def _public_observation(self) -> dict[str, Any]:
        """Return exactly the observation surface promised to the code agent.

        The frozen-SAM3 diagnostic needs the RGB-D artifact ledger because its
        host-side primitives consume those files.  The primary observation-only
        baseline does not: exposing depth/world-map or calibration paths would
        silently turn it into a geometric-perception baseline even if the
        restricted program API omitted the corresponding functions.
        """

        observation = self.store.latest()
        if self.perception_mode not in PRIMITIVE_RGB_PROJECTION_MODES:
            return observation
        public = deepcopy(observation)
        cameras = observation.get("artifacts", {}).get("cameras", {})
        public_cameras: dict[str, Any] = {}
        if isinstance(cameras, dict):
            for camera, camera_data in cameras.items():
                if not isinstance(camera_data, dict):
                    continue
                rgb_only = {
                    key: deepcopy(camera_data[key])
                    for key in ("rgb", "high_rgb")
                    if isinstance(camera_data.get(key), dict)
                }
                if rgb_only:
                    public_cameras[str(camera)] = rgb_only
        public["artifacts"] = {"cameras": public_cameras}
        contract = public.setdefault("agent_contract", {})
        if isinstance(contract, dict):
            contract["privileged_geometry_artifacts_exposed"] = False
        return public

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
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
        program = request.get("program")
        if not isinstance(program, str):
            return self._reject(request_id, "program must be a string")
        if len(program) > self.max_program_chars:
            return self._reject(
                request_id,
                f"program has {len(program)} characters; limit is {self.max_program_chars}",
            )
        if self._closed:
            return self._reject(request_id, "runtime is closed")
        status = self.store.latest()["status"]
        if status.get("terminated") or status.get("truncated"):
            return self._reject(request_id, "episode is already finished")
        if self._program_count >= self.max_programs:
            return self._reject(request_id, f"program limit reached ({self.max_programs})")
        current_step = self._episode.summary()["step_index"]
        claimed_turn = str(request.get("agent_turn_id") or "")
        canonical_turn = claimed_turn
        if self.enforce_agent_protocol:
            if not claimed_turn:
                return self._protocol_reject(
                    request_id,
                    "missing_agent_turn_id",
                    "program request is missing the harness-issued turn identity",
                )
            if self.turn_authority:
                canonical_turn = self.turn_authority.active_turn("primitive_program") or ""
                if not canonical_turn:
                    return self._protocol_reject(
                        request_id,
                        "unverified_primitive_program_turn",
                        "program request does not match an active `./esbench-code run` command",
                    )
            if canonical_turn in self._used_program_turn_ids:
                return self._protocol_reject(
                    request_id,
                    "multiple_programs_in_one_turn",
                    "one model tool turn may submit at most one primitive program",
                )
            if self._acknowledged_step != current_step:
                return self._protocol_reject(
                    request_id,
                    "observation_not_acknowledged",
                    "run `./esbench-code observe` before submitting a program for the current step",
                )
        if canonical_turn:
            self._used_program_turn_ids.add(canonical_turn)
        self._program_count += 1
        program_index = self._program_count
        program_dir = self.store.episode_dir / "programs" / f"{program_index:04d}"
        program_dir.mkdir(parents=True, exist_ok=False)
        source_path = program_dir / "source.py"
        source_path.write_text(program, encoding="utf-8")
        started = time.monotonic()
        action: dict[str, Any] | None = None
        error: dict[str, Any] | None = None
        interpreter_result: dict[str, Any] | None = None
        self._active_program_index = program_index
        interpreter: RestrictedAstInterpreter | None = None
        try:
            interpreter = RestrictedAstInterpreter(
                self._program_functions(),
                max_ast_nodes=self.max_ast_nodes,
                max_execution_nodes=self.max_execution_nodes,
                max_loop_iterations=self.max_loop_iterations,
                timeout_seconds=self.program_timeout_seconds,
            )
            interpreter_result = interpreter.run(program)
        except _ActionBoundary as boundary:
            interpreter_result = interpreter.snapshot() if interpreter else None
            action = {
                "primitive": boundary.primitive,
                "response": boundary.response,
                "remaining_program_skipped": True,
            }
        except BaseException as exc:
            self._program_errors += 1
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            private_errors = self.run_dir / "runtime" / "private-program-errors"
            private_errors.mkdir(parents=True, exist_ok=True)
            (private_errors / f"{program_index:04d}.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        finally:
            self._active_program_index = None
        elapsed = time.monotonic() - started
        if action is not None:
            self._acknowledged_step = None
            self._handles.clear()
        program_sha256 = hashlib.sha256(program.encode("utf-8")).hexdigest()
        failure_feedback: dict[str, Any] | None = None
        if error is not None and self.repeat_error_feedback:
            failure_key = (
                program_sha256,
                str(error["type"]),
                str(error["message"]),
            )
            if failure_key == self._last_program_failure_key:
                self._consecutive_same_program_failure_count += 1
            else:
                self._last_program_failure_key = failure_key
                self._consecutive_same_program_failure_count = 1
            repeated = self._consecutive_same_program_failure_count >= 2
            if repeated:
                self._repeated_identical_program_errors += 1
            failure_feedback = {
                "consecutive_same_program_error_count": (
                    self._consecutive_same_program_failure_count
                ),
                "program_must_change_before_retry": repeated,
                "message": (
                    "The exact same program has failed with the exact same error "
                    f"{self._consecutive_same_program_failure_count} consecutive "
                    "times. Do not run it unchanged again; revise the program and "
                    "re-read PRIMITIVE_API.md."
                    if repeated
                    else None
                ),
            }
        else:
            self._last_program_failure_key = None
            self._consecutive_same_program_failure_count = 0
        result = {
            "ok": error is None and (action is None or bool(action["response"].get("ok"))),
            "accepted": True,
            "request_id": request_id,
            "agent_turn_id": canonical_turn or None,
            "program_index": program_index,
            "program_sha256": program_sha256,
            "source_path": str(source_path.resolve()),
            "elapsed_seconds": elapsed,
            "read_only_result": interpreter_result,
            "action_boundary": action,
            "error": error,
            "failure_feedback": failure_feedback,
            "step_index": self._episode.summary()["step_index"],
            "observation_json": self.store.latest_pointer()["observation_json"],
            "observation_markdown": self.store.latest_pointer()["observation_markdown"],
            "status": self.store.latest()["status"],
        }
        if action is not None and self.auto_observation_handoff:
            next_step = self._episode.summary()["step_index"]
            self._acknowledged_step = next_step
            result.update(
                {
                    "automatic_observation_handoff": True,
                    "observation_acknowledged": True,
                    "observation": self._public_observation(),
                }
            )
            self._event(
                "observation_acknowledged",
                {
                    "step_index": next_step,
                    "source": "primitive_code_automatic_post_action_handoff",
                    "agent_turn_id": canonical_turn or None,
                },
            )
        else:
            result["automatic_observation_handoff"] = False
        atomic_write_json(program_dir / "result.json", result)
        self._event(
            "primitive_program_error" if error else "primitive_program_completed",
            {
                "request_id": request_id,
                "agent_turn_id": canonical_turn or None,
                "program_index": program_index,
                "program_sha256": result["program_sha256"],
                "elapsed_seconds": elapsed,
                "action_primitive": action["primitive"] if action else None,
                "error": error,
                "consecutive_same_program_error_count": (
                    failure_feedback["consecutive_same_program_error_count"]
                    if failure_feedback
                    else 0
                ),
                "program_must_change_before_retry": (
                    failure_feedback["program_must_change_before_retry"]
                    if failure_feedback
                    else False
                ),
            },
        )
        return result

    def _program_functions(self) -> dict[str, Callable[..., Any]]:
        functions = {
            "print": self._print,
            "len": lambda value: len(value),
            "min": lambda *values: min(*values),
            "max": lambda *values: max(*values),
            "sum": lambda values: sum(_number(item) for item in values),
            "abs": lambda value: abs(_number(value)),
            "round": lambda value, ndigits=0: round(_number(value), int(ndigits)),
            "sorted": lambda values: sorted(values),
            "range": self._range,
            "clip": lambda value, low, high: min(max(_number(value), _number(low)), _number(high)),
            "sqrt": lambda value: math.sqrt(_number(value)),
            "observe": self._api_observe,
            "get_eef_state": self._api_get_eef_state,
            "move_eef_delta": self._api_move_eef_delta,
            "move_eef_pose": self._api_move_eef_pose,
            "set_gripper": self._api_set_gripper,
        }
        if self.perception_mode in PRIMITIVE_GEOMETRY_MODES:
            functions.update(
                {
                    "get_camera_calibration": self._api_get_camera_calibration,
                    "read_depth_pixel": self._api_read_depth_pixel,
                    "pixel_to_world": self._api_pixel_to_world,
                    "get_gripper_contact": self._api_get_gripper_contact,
                }
            )
        elif self.perception_mode == "frozen-sam3":
            functions.update(
                {
                    "load_rgb": self._api_load_rgb,
                    "load_depth": self._api_load_depth,
                    "segment_text": self._api_segment_text,
                    "mask_to_world_points": self._api_mask_to_world_points,
                    "estimate_obb": self._api_estimate_obb,
                    "propose_grasps": self._api_propose_grasps,
                }
            )
        return functions

    def _print(self, *values: Any) -> None:
        # ``print`` is intentionally captured in the JSON response, never sent
        # to a host stream that could be confused with the agent protocol.
        return None

    def _range(self, *values: Any) -> list[int]:
        ints = [int(_number(value)) for value in values]
        result = list(range(*ints))
        if len(result) > self.max_loop_iterations:
            raise RestrictedProgramError("range exceeds loop iteration limit")
        return result

    def _read_call(self, name: str, arguments: dict[str, Any]) -> None:
        self._read_call_count += 1
        self._call_counts[name] += 1
        self._event(
            "primitive_read_call",
            {
                "program_index": self._active_program_index,
                "primitive": name,
                "arguments": arguments,
                "step_index": self._episode.summary()["step_index"],
            },
        )

    def _action_call(self, name: str, arguments: dict[str, Any]) -> None:
        self._action_call_count += 1
        self._call_counts[name] += 1
        self._event(
            "primitive_action_call",
            {
                "program_index": self._active_program_index,
                "primitive": name,
                "arguments": arguments,
                "step_index": self._episode.summary()["step_index"],
            },
        )

    def _api_observe(self) -> dict[str, Any]:
        self._read_call("observe", {})
        observation = self._public_observation()
        return {
            "step_index": observation["step_index"],
            "task": observation["task"],
            "status": observation["status"],
            "robot": observation["robot"],
            "object_names": observation.get("object_names", []),
            "artifacts": observation["artifacts"],
        }

    def _camera_artifact(self, camera: str, key: str) -> tuple[Path, dict[str, Any]]:
        if camera not in {"agentview", "wrist"}:
            raise RestrictedProgramError("camera must be 'agentview' or 'wrist'")
        observation = self.store.latest()
        try:
            artifact = observation["artifacts"]["cameras"][camera][key]
        except KeyError as exc:
            raise RestrictedProgramError(f"current observation has no {camera}.{key} artifact") from exc
        path = Path(str(artifact["path"])).resolve()
        if not path.is_file():
            raise RestrictedProgramError(f"public artifact is missing: {path}")
        return path, artifact

    def _camera_calibration(self, camera: str) -> dict[str, Any]:
        if camera not in {"agentview", "wrist"}:
            raise RestrictedProgramError("camera must be 'agentview' or 'wrist'")
        observation = self.store.latest()
        try:
            artifact = observation["artifacts"]["calibration"][camera]
            path = Path(str(artifact["path"])).resolve()
        except KeyError as exc:
            raise RestrictedProgramError(
                f"current observation has no {camera} calibration"
            ) from exc
        if not path.is_file():
            raise RestrictedProgramError(f"camera calibration is missing: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RestrictedProgramError("camera calibration must be a JSON object")
        return value

    @staticmethod
    def _checked_pixel(
        row: Any, col: Any, height: int, width: int
    ) -> tuple[int, int]:
        row_value = int(_number(row))
        col_value = int(_number(col))
        if row_value < 0 or col_value < 0 or row_value >= height or col_value >= width:
            raise RestrictedProgramError(
                f"pixel ({row_value}, {col_value}) is outside {width}x{height}"
            )
        return row_value, col_value

    def _api_get_camera_calibration(
        self, camera: str = "agentview"
    ) -> dict[str, Any]:
        self._read_call("get_camera_calibration", {"camera": camera})
        value = self._camera_calibration(camera)
        low_rgb = self._camera_artifact(camera, "rgb")[1]
        high_rgb = self._camera_artifact(camera, "high_rgb")[1]
        low_resolution = list(low_rgb["resolution"])
        high_resolution = list(high_rgb["resolution"])
        intrinsic = value.get("intrinsic_K")
        high_intrinsic = None
        if (
            isinstance(intrinsic, list)
            and len(intrinsic) == 3
            and all(isinstance(row, list) and len(row) == 3 for row in intrinsic)
        ):
            scale_x = float(high_resolution[0]) / float(low_resolution[0])
            scale_y = float(high_resolution[1]) / float(low_resolution[1])
            high_intrinsic = deepcopy(intrinsic)
            high_intrinsic[0][0] = float(high_intrinsic[0][0]) * scale_x
            high_intrinsic[0][2] = float(high_intrinsic[0][2]) * scale_x
            high_intrinsic[1][1] = float(high_intrinsic[1][1]) * scale_y
            high_intrinsic[1][2] = float(high_intrinsic[1][2]) * scale_y
        return {
            "camera": camera,
            "frame": "LIBERO world",
            "pixel_convention": "row 0 is top; column 0 is left",
            "low_resolution_wh": low_resolution,
            "high_resolution_wh": high_resolution,
            "intrinsic_K_low": intrinsic,
            "intrinsic_K_high": high_intrinsic,
            "extrinsic_cam2world": value.get("extrinsic_cam2world"),
            "moving_camera": camera == "wrist",
            "valid_for_step_index": self._episode.summary()["step_index"],
        }

    def _api_read_depth_pixel(
        self,
        row: int,
        col: int,
        camera: str = "agentview",
        window_radius: int = 0,
    ) -> dict[str, Any]:
        radius = int(_number(window_radius))
        if not 0 <= radius <= 4:
            raise RestrictedProgramError("window_radius must be between 0 and 4")
        path, artifact = self._camera_artifact(camera, "depth")
        depth = np.load(path, mmap_mode="r")
        height, width = int(depth.shape[0]), int(depth.shape[1])
        row_value, col_value = self._checked_pixel(row, col, height, width)
        self._read_call(
            "read_depth_pixel",
            {
                "camera": camera,
                "row": row_value,
                "col": col_value,
                "window_radius": radius,
            },
        )
        low_row = max(0, row_value - radius)
        high_row = min(height, row_value + radius + 1)
        low_col = max(0, col_value - radius)
        high_col = min(width, col_value + radius + 1)
        values = np.asarray(
            depth[low_row:high_row, low_col:high_col], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        center = float(depth[row_value, col_value])
        return {
            "camera": camera,
            "resolution": "low",
            "resolution_wh": list(artifact["resolution"]),
            "pixel": {"row": row_value, "col": col_value},
            "depth_m": center if math.isfinite(center) else None,
            "window_radius": radius,
            "finite_window_count": int(finite.size),
            "window_median_depth_m": (
                float(np.median(finite)) if finite.size else None
            ),
            "step_index": self._episode.summary()["step_index"],
        }

    def _api_pixel_to_world(
        self,
        row: int,
        col: int,
        camera: str = "agentview",
        resolution: str = "high",
    ) -> dict[str, Any]:
        if resolution not in {"low", "high"}:
            raise RestrictedProgramError("resolution must be 'low' or 'high'")
        key = "high_world" if resolution == "high" else "world"
        path, artifact = self._camera_artifact(camera, key)
        world = np.load(path, mmap_mode="r")
        height, width = int(world.shape[0]), int(world.shape[1])
        row_value, col_value = self._checked_pixel(row, col, height, width)
        self._read_call(
            "pixel_to_world",
            {
                "camera": camera,
                "resolution": resolution,
                "row": row_value,
                "col": col_value,
            },
        )
        xyz = np.asarray(world[row_value, col_value, :3], dtype=np.float64)
        return {
            "camera": camera,
            "resolution": resolution,
            "resolution_wh": list(artifact["resolution"]),
            "pixel": {"row": row_value, "col": col_value},
            "world_xyz_m": xyz.tolist() if np.isfinite(xyz).all() else None,
            "frame": "LIBERO world",
            "step_index": self._episode.summary()["step_index"],
            "semantic_localization_used": False,
        }

    def _api_get_gripper_contact(self) -> dict[str, Any]:
        self._read_call("get_gripper_contact", {})
        gripper = self.store.latest()["robot"]["gripper"]
        opening = float(gripper["opening_m"])
        commanded_closed = self._last_gripper_command == "closed"
        contact_suspected = (
            bool(opening > EMPTY_CLOSURE_SUSPICION_UPPER_M)
            if commanded_closed
            else None
        )
        return {
            "commanded_state": self._last_gripper_command,
            "opening_m": opening,
            "contact_or_obstruction_suspected": contact_suspected,
            "attachment_presence_suspected": contact_suspected,
            "empty_closure_suspected": (
                bool(opening <= EMPTY_CLOSURE_SUSPICION_UPPER_M)
                if commanded_closed
                else None
            ),
            "empty_closure_suspicion_upper_m": EMPTY_CLOSURE_SUSPICION_UPPER_M,
            "object_identity": None,
            "grasp_verified": None,
            "official_task_success": bool(self.store.latest()["status"]["success"]),
            "evidence_boundary": (
                "identity-free gripper-width heuristic; not a grasp verifier or task truth"
            ),
            "step_index": self._episode.summary()["step_index"],
        }

    def _new_handle(self, kind: str, payload: dict[str, Any]) -> str:
        handle = f"{kind}:{self._episode.summary()['step_index']}:{uuid.uuid4().hex[:12]}"
        self._handles[handle] = {
            "kind": kind,
            "step_index": self._episode.summary()["step_index"],
            **payload,
        }
        return handle

    def _require_handle(self, handle: Any, kind: str) -> dict[str, Any]:
        value = self._handles.get(str(handle))
        if not value or value.get("kind") != kind:
            raise RestrictedProgramError(f"invalid or stale {kind} handle")
        if value["step_index"] != self._episode.summary()["step_index"]:
            raise RestrictedProgramError(f"stale {kind} handle from another observation")
        return value

    def _api_load_rgb(self, camera: str = "agentview", resolution: str = "high") -> dict[str, Any]:
        if resolution not in {"low", "high"}:
            raise RestrictedProgramError("resolution must be 'low' or 'high'")
        self._read_call("load_rgb", {"camera": camera, "resolution": resolution})
        path, artifact = self._camera_artifact(camera, "high_rgb" if resolution == "high" else "rgb")
        with Image.open(path) as image:
            size = list(image.size)
        return {
            "rgb_handle": self._new_handle("rgb", {"path": str(path)}),
            "camera": camera,
            "resolution": resolution,
            "size_wh": size,
            "sha256": _sha256_file(path),
        }

    def _api_load_depth(self, camera: str = "agentview") -> dict[str, Any]:
        self._read_call("load_depth", {"camera": camera})
        path, _ = self._camera_artifact(camera, "depth")
        depth = np.load(path, mmap_mode="r")
        finite = np.asarray(depth[np.isfinite(depth)], dtype=np.float64)
        return {
            "depth_handle": self._new_handle("depth", {"path": str(path)}),
            "camera": camera,
            "shape": list(depth.shape),
            "finite_count": int(finite.size),
            "min_m": float(finite.min()) if finite.size else None,
            "median_m": float(np.median(finite)) if finite.size else None,
            "max_m": float(finite.max()) if finite.size else None,
        }

    def _api_segment_text(
        self,
        query: str,
        camera: str = "agentview",
        resolution: str = "high",
        min_score: float = 0.2,
    ) -> dict[str, Any]:
        if self._perception_count >= self.max_perception_queries:
            raise RestrictedProgramError(
                f"perception query limit reached ({self.max_perception_queries})"
            )
        arguments = {
            "prompt": str(query),
            "camera": str(camera),
            "resolution": str(resolution),
            "min_score": float(min_score),
        }
        self._read_call("segment_text", arguments)
        self._perception_count += 1
        query_index = self._perception_count
        destination = (
            self.store.episode_dir
            / "primitive-perception"
            / f"step-{self._episode.summary()['step_index']:04d}"
            / f"query-{query_index:04d}"
        )
        try:
            result = self.backend.perceive(
                "sam3-localize-object", arguments, self.store.latest(), destination
            ).result
        except BaseException as exc:
            shutil.rmtree(destination, ignore_errors=True)
            self._event(
                "primitive_perception_error",
                {
                    "program_index": self._active_program_index,
                    "query_index": query_index,
                    "primitive": "segment_text",
                    "arguments": arguments,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
            )
            raise
        value = dict(result)
        if value.get("found") and value.get("mask_path"):
            value["mask_handle"] = self._new_handle(
                "mask",
                {
                    "mask_path": str(value["mask_path"]),
                    "world_path": str(value["source_world_map_path"]),
                },
            )
        self._event(
            "primitive_perception_completed",
            {
                "program_index": self._active_program_index,
                "query_index": query_index,
                "primitive": "segment_text",
                "arguments": arguments,
                "found": bool(value.get("found")),
            },
        )
        return value

    def _api_mask_to_world_points(self, mask_handle: str, max_points: int = 2048) -> dict[str, Any]:
        max_points = int(max_points)
        if not 1 <= max_points <= 4096:
            raise RestrictedProgramError("max_points must be between 1 and 4096")
        self._read_call(
            "mask_to_world_points",
            {"mask_handle": str(mask_handle), "max_points": max_points},
        )
        mask_record = self._require_handle(mask_handle, "mask")
        mask = np.asarray(Image.open(mask_record["mask_path"]).convert("L")) > 0
        world = np.load(mask_record["world_path"], mmap_mode="r")
        if mask.shape != tuple(world.shape[:2]):
            raise RestrictedProgramError("mask/world shape mismatch")
        points = np.asarray(
            world[..., :3][mask & np.isfinite(world[..., :3]).all(axis=2)],
            dtype=np.float64,
        )
        if points.shape[0] > max_points:
            indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=int)
            sampled = points[indices]
        else:
            sampled = points
        handle = self._new_handle("points", {"points": sampled})
        if points.size:
            low, center, high = np.percentile(points, [5, 50, 95], axis=0)
        else:
            low = center = high = np.asarray([np.nan, np.nan, np.nan])
        return {
            "points_handle": handle,
            "valid_point_count": int(points.shape[0]),
            "sampled_point_count": int(sampled.shape[0]),
            "center_median_m": _finite_list(center),
            "bounds_p05_m": _finite_list(low),
            "bounds_p95_m": _finite_list(high),
        }

    def _api_estimate_obb(self, points_handle: str) -> dict[str, Any]:
        self._read_call("estimate_obb", {"points_handle": str(points_handle)})
        points = np.asarray(self._require_handle(points_handle, "points")["points"], dtype=np.float64)
        if points.shape[0] < 3:
            raise RestrictedProgramError("at least three points are required for an OBB")
        center = np.median(points, axis=0)
        xy = points[:, :2] - center[:2]
        covariance = np.cov(xy, rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        axes_xy = vectors[:, order]
        projected = xy @ axes_xy
        low, high = np.percentile(projected, [5, 95], axis=0)
        z_low, z_high = np.percentile(points[:, 2], [5, 95])
        yaw = math.atan2(float(axes_xy[1, 0]), float(axes_xy[0, 0]))
        return {
            "center_xyz_m": center.tolist(),
            "extent_xyz_m": [float(high[0] - low[0]), float(high[1] - low[1]), float(z_high - z_low)],
            "yaw_rad": yaw,
            "axes_xy": axes_xy.tolist(),
            "method": "pca-p05-p95",
        }

    def _api_propose_grasps(
        self,
        points_handle: str,
        approach_height_m: float = 0.10,
        max_candidates: int = 3,
    ) -> dict[str, Any]:
        approach_height_m = float(approach_height_m)
        max_candidates = int(max_candidates)
        if not 0.02 <= approach_height_m <= 0.30:
            raise RestrictedProgramError("approach_height_m must be between 0.02 and 0.30")
        if not 1 <= max_candidates <= 3:
            raise RestrictedProgramError("max_candidates must be between 1 and 3")
        self._read_call(
            "propose_grasps",
            {
                "points_handle": str(points_handle),
                "approach_height_m": approach_height_m,
                "max_candidates": max_candidates,
            },
        )
        points = np.asarray(self._require_handle(points_handle, "points")["points"], dtype=np.float64)
        if not points.size:
            return {"provider": "deterministic-top-down-geometry-v1", "candidates": []}
        center = np.median(points, axis=0)
        xy = points[:, :2] - center[:2]
        values, vectors = np.linalg.eigh(np.cov(xy, rowvar=False))
        major = vectors[:, int(np.argmax(values))]
        yaw = math.atan2(float(major[1]), float(major[0]))
        candidates = []
        for offset in (0.0, math.pi / 2, -math.pi / 2)[:max_candidates]:
            candidates.append(
                {
                    "grasp_xyz_m": center.tolist(),
                    "pregrasp_xyz_m": [float(center[0]), float(center[1]), float(center[2] + approach_height_m)],
                    "yaw_rad": float(yaw + offset),
                    "pitch_rad": 0.0,
                    "score": float(1.0 - 0.1 * len(candidates)),
                }
            )
        return {
            "provider": "deterministic-top-down-geometry-v1",
            "task_conditioned": False,
            "learned_model": False,
            "candidates": candidates,
        }

    def _api_get_eef_state(self) -> dict[str, Any]:
        self._read_call("get_eef_state", {})
        robot = self.store.latest()["robot"]
        return {"end_effector": robot["end_effector"], "gripper": robot["gripper"]}

    def _api_move_eef_delta(
        self,
        delta_xyz: list[float],
        gripper: str | None = None,
        max_steps: int = 80,
    ) -> None:
        delta = _xyz(delta_xyz, "delta_xyz")
        if any(abs(value) > 0.25 for value in delta):
            raise RestrictedProgramError("each delta_xyz component is limited to 0.25 m")
        current = self.store.latest()["robot"]["end_effector"]["position_m"]
        target = [float(current[index]) + delta[index] for index in range(3)]
        arguments: dict[str, Any] = {
            "xyz": target,
            "max_steps": int(max_steps),
            "gripper": (
                _gripper(gripper)
                if gripper is not None
                else self._last_gripper_command
            ),
        }
        self._perform_action("move_eef_delta", "move-to", arguments)

    def _api_move_eef_pose(
        self,
        xyz: list[float],
        pitch: float | None = None,
        yaw: float | None = None,
        gripper: str | None = None,
        max_steps: int = 100,
    ) -> None:
        arguments: dict[str, Any] = {
            "xyz": _xyz(xyz, "xyz"),
            "max_steps": int(max_steps),
            "gripper": (
                _gripper(gripper)
                if gripper is not None
                else self._last_gripper_command
            ),
        }
        if pitch is not None:
            arguments["target_pitch"] = float(pitch)
        if yaw is not None:
            arguments["target_yaw"] = float(yaw)
        self._perform_action("move_eef_pose", "move-pose", arguments)

    def _api_set_gripper(self, state: str, steps: int = 20) -> None:
        arguments = {"gripper": _gripper(state), "steps": int(steps)}
        self._perform_action("set_gripper", "set-gripper", arguments)

    def _perform_action(self, primitive: str, adapter: str, arguments: dict[str, Any]) -> None:
        self._action_call(primitive, arguments)
        response = self._episode.execute_adapter(
            request_id=f"primitive-program-{self._active_program_index}-{primitive}",
            adapter_skill_id=adapter,
            public_operation_id=primitive,
            arguments=arguments,
        )
        if not response.get("accepted"):
            error = response.get("error") or {}
            raise RestrictedProgramError(
                f"{primitive} was rejected before simulator execution: "
                f"{error.get('message', 'unknown rejection')}"
            )
        if "gripper" in arguments:
            self._last_gripper_command = _gripper(arguments["gripper"])
        if self.perception_mode in PRIMITIVE_AUTO_HANDOFF_MODES:
            self._add_controller_feedback(primitive, arguments, response)
        self._action_sequence.append(primitive)
        raise _ActionBoundary(primitive, response)

    def _add_controller_feedback(
        self,
        primitive: str,
        arguments: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        """Add task-agnostic controller literacy fields to P1/P2 results.

        The underlying adapter result remains intact.  These fields say only
        whether the commanded robot target was reached; they never imply an
        object interaction or official task success.
        """

        raw = response.get("result")
        if not isinstance(raw, dict):
            return
        feedback: dict[str, Any] = {
            "primitive": primitive,
            "official_task_success": bool(response.get("status", {}).get("success")),
            "task_success_inferred_from_controller": False,
        }
        if primitive in {"move_eef_delta", "move_eef_pose"}:
            distance_value = raw.get("final_dist_m")
            distance = (
                float(distance_value) if distance_value is not None else None
            )
            final_xyz = raw.get("final_eef_pos")
            target_xyz = arguments.get("xyz")
            if (
                distance is None
                and isinstance(final_xyz, list)
                and len(final_xyz) == 3
                and isinstance(target_xyz, list)
                and len(target_xyz) == 3
            ):
                distance = float(
                    math.sqrt(
                        sum(
                            (float(target_xyz[index]) - float(final_xyz[index])) ** 2
                            for index in range(3)
                        )
                    )
                )
            position_reached = (
                bool(distance <= CONTROLLER_POSITION_TOLERANCE_M)
                if distance is not None
                else None
            )
            max_steps = int(arguments.get("max_steps", 0))
            steps_used = int(raw.get("steps_used", 0))
            budget_exhausted = bool(
                max_steps > 0
                and steps_used >= max_steps
                and position_reached is not True
                and not response.get("status", {}).get("terminated")
                and not response.get("status", {}).get("truncated")
            )
            feedback.update(
                {
                    "target_xyz_m": list(arguments.get("xyz", [])),
                    "final_xyz_m": final_xyz,
                    "position_error_m": distance,
                    "position_tolerance_m": CONTROLLER_POSITION_TOLERANCE_M,
                    "position_reached": position_reached,
                    "steps_used": steps_used,
                    "max_steps": max_steps,
                    "controller_budget_exhausted": budget_exhausted,
                    "blocked_or_unreachable_suspected": budget_exhausted,
                }
            )
            orientation_reached: bool | None = True
            orientation_errors: dict[str, float] = {}
            quaternion = (
                self.store.latest()
                .get("robot", {})
                .get("end_effector", {})
                .get("quaternion")
            )
            if isinstance(quaternion, list) and len(quaternion) == 4:
                pitch, yaw = _libero_pitch_yaw(quaternion)
                if arguments.get("target_pitch") is not None:
                    error = _wrapped_angle_error(
                        float(arguments["target_pitch"]), pitch
                    )
                    orientation_errors["pitch_error_rad"] = error
                    orientation_reached = bool(
                        orientation_reached
                        and abs(error) <= CONTROLLER_ORIENTATION_TOLERANCE_RAD
                    )
                if arguments.get("target_yaw") is not None:
                    error = _wrapped_angle_error(
                        float(arguments["target_yaw"]), yaw
                    )
                    orientation_errors["yaw_error_rad"] = error
                    orientation_reached = bool(
                        orientation_reached
                        and abs(error) <= CONTROLLER_ORIENTATION_TOLERANCE_RAD
                    )
            elif arguments.get("target_pitch") is not None or arguments.get(
                "target_yaw"
            ) is not None:
                orientation_reached = None
            feedback.update(orientation_errors)
            feedback["orientation_tolerance_rad"] = (
                CONTROLLER_ORIENTATION_TOLERANCE_RAD
            )
            feedback["orientation_reached"] = orientation_reached
            feedback["reached_target"] = bool(
                position_reached is True and orientation_reached is True
            )
            if feedback["reached_target"]:
                feedback["controller_status"] = "reached_target"
            elif budget_exhausted:
                feedback["controller_status"] = "budget_exhausted_before_target"
            elif response.get("status", {}).get("terminated") or response.get(
                "status", {}
            ).get("truncated"):
                feedback["controller_status"] = "episode_ended_before_target"
            else:
                feedback["controller_status"] = "target_not_reached"
        else:
            opening = float(
                self.store.latest()["robot"]["gripper"]["opening_m"]
            )
            feedback.update(
                {
                    "commanded_state": self._last_gripper_command,
                    "final_opening_m": opening,
                    "controller_status": "command_completed",
                    "reached_target": True,
                }
            )
        response["controller_feedback"] = feedback

    def _reject(self, request_id: str, reason: str, code: str = "rejected_primitive_program") -> dict[str, Any]:
        self._program_rejections += 1
        response = {
            "ok": False,
            "accepted": False,
            "request_id": request_id,
            "error": {"type": "RejectedPrimitiveProgram", "code": code, "message": reason},
            "step_index": self._episode.summary()["step_index"],
            "observation_json": self.store.latest_pointer()["observation_json"],
            "status": self.store.latest()["status"],
        }
        self._event("primitive_program_rejected", response)
        return response

    def _protocol_reject(self, request_id: str, code: str, reason: str) -> dict[str, Any]:
        self._protocol_violation(code, reason)
        return self._reject(request_id, reason, code)

    def _protocol_violation(self, code: str, reason: str) -> dict[str, Any]:
        violation = {
            "code": code,
            "message": reason,
            "step_index": self._episode.summary()["step_index"],
        }
        self._protocol_violations.append(violation)
        self._event("agent_protocol_violation", violation)
        return violation

    def _event(self, kind: str, payload: dict[str, Any]) -> None:
        append_jsonl(self.store.events_path, {"timestamp": utc_now(), "event": kind, **payload})

    def finish(self, reason: str = "agent_finished") -> dict[str, Any]:
        self._event(
            "finish",
            {
                "reason": reason,
                "program_count": self._program_count,
                "read_primitive_calls": self._read_call_count,
                "action_primitive_calls": self._action_call_count,
            },
        )
        return self.summary()

    def summary(self) -> dict[str, Any]:
        base = self._episode.summary()
        # Public primitive labels are operations, not selected embodied skills.
        # Keep the generic skill fields empty for cross-condition aggregation.
        base.update(
            {
                "method": "primitive-code-agent",
                "embodied_skill_library_exposed": False,
                "vla_exposed": False,
                "perception_mode": self.perception_mode,
                "sam3_exposed": self.perception_mode == "frozen-sam3",
                "contact_graspnet_exposed": False,
                "program_count": self._program_count,
                "program_calls": {
                    "completed": self._program_count - self._program_errors,
                    "rejected": self._program_rejections,
                    "errors": self._program_errors,
                    "repeated_identical_errors": (
                        self._repeated_identical_program_errors
                    ),
                },
                "primitive_calls": {
                    "read_only": self._read_call_count,
                    "state_changing": self._action_call_count,
                    "perception_queries": self._perception_count,
                    "by_name": dict(sorted(self._call_counts.items())),
                },
                "action_primitive_sequence": list(self._action_sequence),
                "skill_sequence": [],
                "skill_call_counts": {},
                "protocol_enforced": self.enforce_agent_protocol,
                "protocol_turn_identity_source": self.turn_authority.source if self.turn_authority else "direct",
                "protocol_violations": list(self._protocol_violations),
            }
        )
        return base

    def close(self) -> None:
        if not self._closed:
            self._episode.close()
            self._closed = True


def _number(value: Any) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RestrictedProgramError(f"expected number, got {type(value).__name__}")
    if not math.isfinite(float(value)):
        raise RestrictedProgramError("numbers must be finite")
    return value


def _xyz(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise RestrictedProgramError(f"{name} must contain exactly three numbers")
    return [float(_number(item)) for item in value]


def _gripper(value: Any) -> str:
    normalized = str(value).lower()
    if normalized == "close":
        normalized = "closed"
    if normalized not in {"open", "closed"}:
        raise RestrictedProgramError("gripper state must be 'open' or 'closed'")
    return normalized


def _compare(left: Any, operator: ast.cmpop, right: Any) -> bool:
    if isinstance(operator, ast.Eq):
        return left == right
    if isinstance(operator, ast.NotEq):
        return left != right
    if isinstance(operator, ast.Lt):
        return left < right
    if isinstance(operator, ast.LtE):
        return left <= right
    if isinstance(operator, ast.Gt):
        return left > right
    if isinstance(operator, ast.GtE):
        return left >= right
    if isinstance(operator, ast.In):
        return left in right
    if isinstance(operator, ast.NotIn):
        return left not in right
    raise RestrictedProgramError("unsupported comparison operator")


def _wrapped_angle_error(target: float, actual: float) -> float:
    return float((target - actual + math.pi) % (2 * math.pi) - math.pi)


def _libero_pitch_yaw(quaternion_xyzw: list[float]) -> tuple[float, float]:
    """Match RPent's gripper-down pitch/yaw extraction without SciPy."""

    x, y, z, w = [float(item) for item in quaternion_xyzw]
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + z * w)
    r12 = 2.0 * (y * z - x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    pitch = math.atan2(r12, -r22)
    yaw = math.atan2(r10, r00)
    return float(pitch), float(yaw)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RestrictedProgramError("program result contains a non-finite number")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise RestrictedProgramError(
        f"program result contains unsupported value {type(value).__name__}"
    )


def _finite_list(value: np.ndarray) -> list[float] | None:
    return value.tolist() if np.isfinite(value).all() else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

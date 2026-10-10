"""Provider-neutral, fail-closed full-pose IK contract for P3."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from urllib.request import Request, urlopen

from .native_typed_contract import P3_IK_PROVIDER_CONTRACT

PYROKI_DEPENDENCY = P3_IK_PROVIDER_CONTRACT


def normalize_quaternion_xyzw(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("quaternion_xyzw must contain exactly four numbers")
    q = [float(item) for item in value]
    if not all(math.isfinite(item) for item in q):
        raise ValueError("quaternion_xyzw values must be finite")
    norm = math.sqrt(sum(item * item for item in q))
    if norm < 1e-8:
        raise ValueError("quaternion_xyzw norm is too close to zero")
    return [item / norm for item in q]


def xyzw_to_wxyz(value: Any) -> list[float]:
    x, y, z, w = normalize_quaternion_xyzw(value)
    return [w, x, y, z]


def quaternion_angular_error_rad(a_xyzw: Any, b_xyzw: Any) -> float:
    a = normalize_quaternion_xyzw(a_xyzw)
    b = normalize_quaternion_xyzw(b_xyzw)
    # abs(dot) makes q and -q equivalent. Clamp protects acos from roundoff.
    dot = min(1.0, max(-1.0, abs(sum(x * y for x, y in zip(a, b, strict=True)))))
    return 2.0 * math.acos(dot)


def _quaternion_multiply_xyzw(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = normalize_quaternion_xyzw(a)
    bx, by, bz, bw = normalize_quaternion_xyzw(b)
    return normalize_quaternion_xyzw(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _wrapped_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def libero_pitch_yaw(quaternion_xyzw: Any) -> tuple[float, float]:
    """Match the frozen RPent pitch/yaw observation convention."""

    x, y, z, w = normalize_quaternion_xyzw(quaternion_xyzw)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + z * w)
    r12 = 2.0 * (y * z - x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return float(math.atan2(r12, -r22)), float(math.atan2(r10, r00))


def quaternion_with_libero_pitch_yaw(
    current_xyzw: Any,
    *,
    pitch_rad: float | None,
    yaw_rad: float | None,
) -> list[float]:
    """Reproduce RPent's inherited pitch/yaw target semantics for P3 IK.

    RPent's OSC implementation commands local-X pitch and world-Z yaw while
    checking the two public extraction formulas.  The short fixed-point loop
    applies those same rotations to the measured orientation, preserving the
    unrequested component and the remaining orientation degree of freedom.
    """

    result = normalize_quaternion_xyzw(current_xyzw)
    if pitch_rad is None and yaw_rad is None:
        return result
    for _ in range(16):
        pitch, yaw = libero_pitch_yaw(result)
        if pitch_rad is not None:
            delta = _wrapped_angle(float(pitch_rad) - pitch)
            right_local_x = [math.sin(delta / 2.0), 0.0, 0.0, math.cos(delta / 2.0)]
            result = _quaternion_multiply_xyzw(result, right_local_x)
        pitch, yaw = libero_pitch_yaw(result)
        if yaw_rad is not None:
            delta = _wrapped_angle(float(yaw_rad) - yaw)
            left_world_z = [0.0, 0.0, math.sin(delta / 2.0), math.cos(delta / 2.0)]
            result = _quaternion_multiply_xyzw(left_world_z, result)
        final_pitch, final_yaw = libero_pitch_yaw(result)
        pitch_error = (
            0.0 if pitch_rad is None else abs(_wrapped_angle(float(pitch_rad) - final_pitch))
        )
        yaw_error = (
            0.0 if yaw_rad is None else abs(_wrapped_angle(float(yaw_rad) - final_yaw))
        )
        if max(pitch_error, yaw_error) < 1e-8:
            return result
    raise ValueError("requested RPent pitch/yaw target is numerically ill-conditioned")


@dataclass(frozen=True)
class IKResult:
    ok: bool
    status: str
    joint_positions_rad: list[float] | None
    fk_position_xyz_m: list[float] | None
    fk_quaternion_xyzw: list[float] | None
    position_error_m: float | None
    orientation_error_rad: float | None
    joint_limits_ok: bool | None
    seed_source: str
    provider: dict[str, Any]
    error: dict[str, Any] | None = None

    def public(self) -> dict[str, Any]:
        value = asdict(self)
        # Raw joints are a private execution detail. The public checker reports
        # feasibility and residuals, not a robot-control escape hatch.
        value.pop("joint_positions_rad", None)
        return value


class IKProvider(Protocol):
    def solve(
        self,
        *,
        position_xyz_m: list[float],
        quaternion_xyzw: list[float],
        previous_joint_positions_rad: list[float] | None,
        current_eef_position_xyz_m: list[float] | None,
        current_eef_quaternion_xyzw: list[float] | None,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
    ) -> IKResult: ...


class UnavailableIKProvider:
    """Explicit blocker used unless a frozen solver endpoint is configured."""

    def __init__(self, reason: str = "frozen_pyroki_endpoint_not_configured"):
        self.reason = reason

    def solve(self, **_kwargs: Any) -> IKResult:
        return IKResult(
            ok=False,
            status="ik_provider_unavailable",
            joint_positions_rad=None,
            fk_position_xyz_m=None,
            fk_quaternion_xyzw=None,
            position_error_m=None,
            orientation_error_rad=None,
            joint_limits_ok=None,
            seed_source="current_joint_state_or_previous_solution_required",
            provider={**PYROKI_DEPENDENCY, "available": False},
            error={"code": self.reason, "message": "P3 cannot solve IK without the frozen provider."},
        )


class PyRoKiHttpProvider:
    """Client for a frozen provider that returns IK plus audited FK residuals.

    This deliberately does not accept CAP-X's minimal joints-only response: an
    endpoint used for P3 must report its dependency identity, joint limits and
    FK pose so the benchmark can independently enforce the scientific gate.
    """

    def __init__(self, endpoint: str, expected_identity: dict[str, Any], timeout_seconds: float = 30):
        self.endpoint = endpoint.rstrip("/")
        self.expected_identity = dict(expected_identity)
        self.timeout_seconds = float(timeout_seconds)

    def solve(
        self,
        *,
        position_xyz_m: list[float],
        quaternion_xyzw: list[float],
        previous_joint_positions_rad: list[float] | None,
        current_eef_position_xyz_m: list[float] | None,
        current_eef_quaternion_xyzw: list[float] | None,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
    ) -> IKResult:
        normalized = normalize_quaternion_xyzw(quaternion_xyzw)
        payload = {
            "position_xyz_m": [float(v) for v in position_xyz_m],
            "quaternion_wxyz": xyzw_to_wxyz(normalized),
            "previous_joint_positions_rad": previous_joint_positions_rad,
            "current_eef_position_xyz_m": current_eef_position_xyz_m,
            "current_eef_quaternion_xyzw": current_eef_quaternion_xyzw,
            "tcp_offset_tool_frame_m": PYROKI_DEPENDENCY["tcp_offset_tool_frame_m"],
        }
        request = Request(
            self.endpoint + "/solve-audited-ik",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            value = json.loads(response.read())
        if not isinstance(value, dict) or value.get("identity") != self.expected_identity:
            raise RuntimeError("PyRoKi provider identity mismatch")
        joints = [float(v) for v in value["joint_positions_rad"]]
        if len(joints) != 7 or not all(math.isfinite(v) for v in joints):
            raise RuntimeError("PyRoKi provider returned invalid Panda joints")
        fk_pos = [float(v) for v in value["fk_position_xyz_m"]]
        fk_quat = normalize_quaternion_xyzw(value["fk_quaternion_xyzw"])
        pos_error = math.sqrt(sum((a - b) ** 2 for a, b in zip(fk_pos, position_xyz_m, strict=True)))
        rot_error = quaternion_angular_error_rad(fk_quat, normalized)
        lower_limits = [float(v) for v in value["joint_lower_limits_rad"]]
        upper_limits = [float(v) for v in value["joint_upper_limits_rad"]]
        if len(lower_limits) != 7 or len(upper_limits) != 7:
            raise RuntimeError("PyRoKi provider returned invalid joint-limit vectors")
        computed_limits_ok = all(
            math.isfinite(lower)
            and math.isfinite(upper)
            and lower <= joint <= upper
            for joint, lower, upper in zip(
                joints, lower_limits, upper_limits, strict=True
            )
        )
        limits_ok = bool(value.get("joint_limits_ok")) and computed_limits_ok
        ok = limits_ok and pos_error <= position_tolerance_m and rot_error <= orientation_tolerance_rad
        return IKResult(
            ok=ok,
            status="ik_valid" if ok else "ik_residual_or_joint_limit_failure",
            joint_positions_rad=joints,
            fk_position_xyz_m=fk_pos,
            fk_quaternion_xyzw=fk_quat,
            position_error_m=pos_error,
            orientation_error_rad=rot_error,
            joint_limits_ok=limits_ok,
            seed_source=str(value.get("seed_source", "current_joint_state_or_previous_solution")),
            provider={
                **PYROKI_DEPENDENCY,
                "available": True,
                "identity": value["identity"],
                "joint_lower_limits_rad": lower_limits,
                "joint_upper_limits_rad": upper_limits,
            },
            error=None if ok else {"code": "ik_validation_failed", "message": "joint limits or FK residual gate failed"},
        )

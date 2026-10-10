"""Audited localhost PyRoKi provider for the P3 native typed protocol.

This service intentionally extends CAP-X's joints-only endpoint.  P3 requires
the frozen solver identity, explicit joint-limit vectors, and FK evidence so a
CPU solve can never be confused with simulator execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import importlib.metadata
import json
import math
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

from .native_typed_contract import P3_IK_PROVIDER_CONTRACT


PYROKI_REVISION = P3_IK_PROVIDER_CONTRACT["revision"]
CAP_X_COMMIT = P3_IK_PROVIDER_CONTRACT["cap_x_commit"]
JAXLS_REVISION = "6fe7cf9d56223736b2c0272d080ec7346203c648"
ROBOT_DESCRIPTION = "panda_description"
TARGET_LINK = "panda_hand"
LIMIT_MARGIN_RAD = 0.15
ARM_JOINT_COUNT = 7
PANDA_GRIPPER_SEED_RAD = 0.02

app = FastAPI()
_robot: Any = None
_solve_basic: Any = None
_solve_continuous: Any = None
_identity: dict[str, Any] | None = None
_urdf_sha256: str | None = None


class AuditedIKRequest(BaseModel):
    position_xyz_m: list[float]
    quaternion_wxyz: list[float]
    previous_joint_positions_rad: list[float] | None = None
    current_eef_position_xyz_m: list[float] | None = None
    current_eef_quaternion_xyzw: list[float] | None = None
    tcp_offset_tool_frame_m: list[float]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _verify_origin(path: Path, *, repository: str, revision: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    vcs = value.get("vcs_info", {})
    # pip's PEP 610 record preserves the clone URL and commonly appends
    # ``.git`` to the same HTTPS repository named in our lock.  Treat only
    # that syntactic suffix as equivalent; the scheme/host/path and immutable
    # commit must still match exactly.
    recorded_url = str(value.get("url", "")).rstrip("/").removesuffix(".git")
    locked_url = repository.rstrip("/").removesuffix(".git")
    if recorded_url != locked_url or vcs.get("commit_id") != revision:
        raise RuntimeError(
            f"dependency origin mismatch for {repository}: {value!r}"
        )
    return value


def _verify_installed_pyroki_matches_wheel(wheel_path: Path) -> dict[str, Any]:
    module = importlib.import_module("pyroki")
    installed_root = Path(module.__file__).resolve().parent
    compared: dict[str, str] = {}
    with zipfile.ZipFile(wheel_path) as archive:
        for name in sorted(archive.namelist()):
            if not name.startswith("pyroki/") or not name.endswith(".py"):
                continue
            relative = name.removeprefix("pyroki/")
            installed = installed_root / relative
            expected = hashlib.sha256(archive.read(name)).hexdigest()
            if not installed.is_file() or _sha256_file(installed) != expected:
                raise RuntimeError(f"installed PyRoKi differs from pinned wheel at {relative}")
            compared[relative] = expected
    if not compared:
        raise RuntimeError("pinned PyRoKi wheel contains no Python package files")
    return {
        "wheel_path": str(wheel_path.resolve()),
        "wheel_sha256": _sha256_file(wheel_path),
        "installed_package_path": str(installed_root),
        "python_files_sha256": compared,
        "python_tree_sha256": _canonical_sha256(compared),
    }


def _verify_distribution_origin(name: str, repository: str, revision: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(name)
    raw = distribution.read_text("direct_url.json")
    if not raw:
        raise RuntimeError(f"{name} has no immutable VCS direct_url identity")
    value = json.loads(raw)
    vcs = value.get("vcs_info", {})
    if value.get("url") != repository or vcs.get("commit_id") != revision:
        raise RuntimeError(f"{name} VCS identity mismatch: {value!r}")
    return {"version": distribution.version, "direct_url": value}


def _load_frozen_solver(path: Path, module_name: str) -> Any:
    """Load one audited CAP-X solver leaf without importing CAP-X's API registry.

    Importing ``capx.integrations`` eagerly loads unrelated perception/control
    stacks such as Open3D and SAM.  P3 is intentionally not allowed to gain
    those capabilities.  The two frozen solver leaves are self-contained, so
    load exactly those files and retain their hashes in provider identity.
    """

    if not path.is_file():
        raise RuntimeError(f"frozen CAP-X solver source is missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen CAP-X solver source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solve = getattr(module, "solve_ik", None)
    if not callable(solve):
        raise RuntimeError(f"frozen CAP-X solver has no solve_ik: {path}")
    return solve


def initialize(
    *,
    cap_x_root: Path,
    pyroki_wheel: Path,
    pyroki_origin_json: Path,
) -> dict[str, Any]:
    global _robot, _solve_basic, _solve_continuous, _identity, _urdf_sha256

    actual_capx = subprocess.check_output(
        ["git", "-C", str(cap_x_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_capx != CAP_X_COMMIT:
        raise RuntimeError(f"CAP-X drift: expected {CAP_X_COMMIT}, got {actual_capx}")
    origin = _verify_origin(
        pyroki_origin_json,
        repository=P3_IK_PROVIDER_CONTRACT["upstream"],
        revision=PYROKI_REVISION,
    )
    wheel_identity = _verify_installed_pyroki_matches_wheel(pyroki_wheel)
    jaxls_identity = _verify_distribution_origin(
        "jaxls", "https://github.com/brentyi/jaxls.git", JAXLS_REVISION
    )
    import pyroki as pk
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    solver_root = (
        cap_x_root
        / "capx"
        / "integrations"
        / "motion"
        / "pyroki_snippets"
    )
    basic_solver_path = solver_root / "_solve_ik.py"
    continuous_solver_path = solver_root / "_solve_ik_vel_cost.py"
    solve_basic = _load_frozen_solver(
        basic_solver_path, "_esb_frozen_capx_solve_ik"
    )
    solve_continuous = _load_frozen_solver(
        continuous_solver_path, "_esb_frozen_capx_solve_ik_vel_cost"
    )

    urdf = load_robot_description(ROBOT_DESCRIPTION)
    for joint in urdf.robot.joints:
        if joint.type == "revolute" and joint.limit is not None:
            if joint.limit.lower is not None and joint.limit.upper is not None:
                joint.limit.lower += LIMIT_MARGIN_RAD
                joint.limit.upper -= LIMIT_MARGIN_RAD
    _urdf_sha256 = hashlib.sha256(
        urdf.write_xml_string().encode("utf-8")
        if isinstance(urdf.write_xml_string(), str)
        else urdf.write_xml_string()
    ).hexdigest()
    _robot = pk.Robot.from_urdf(urdf)
    if int(_robot.joints.num_actuated_joints) < ARM_JOINT_COUNT:
        raise RuntimeError("PyRoKi Panda model has fewer than seven actuated joints")
    _solve_basic = solve_basic
    _solve_continuous = solve_continuous
    _identity = {
        "service": "embodied-skill-bench-audited-pyroki-v1",
        "pyroki_revision": PYROKI_REVISION,
        "pyroki_wheel_sha256": wheel_identity["wheel_sha256"],
        "pyroki_python_tree_sha256": wheel_identity["python_tree_sha256"],
        "pyroki_origin_sha256": _sha256_file(pyroki_origin_json),
        "pyroki_origin": origin,
        "jaxls_revision": JAXLS_REVISION,
        "jaxls_identity": jaxls_identity,
        "cap_x_commit": CAP_X_COMMIT,
        "cap_x_solver_sources_sha256": {
            "_solve_ik.py": _sha256_file(basic_solver_path),
            "_solve_ik_vel_cost.py": _sha256_file(continuous_solver_path),
        },
        "robot": ROBOT_DESCRIPTION,
        "target_link": TARGET_LINK,
        "urdf_sha256": _urdf_sha256,
        "joint_limit_margin_rad": LIMIT_MARGIN_RAD,
        "tcp_offset_tool_frame_m": P3_IK_PROVIDER_CONTRACT[
            "tcp_offset_tool_frame_m"
        ],
        "collision_free_claim": False,
    }
    return dict(_identity)


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return result


@app.get("/identity")
def identity() -> dict[str, Any]:
    if _identity is None:
        raise HTTPException(503, "PyRoKi is not initialized")
    return dict(_identity)


@app.post("/solve-audited-ik")
def solve_audited_ik(request: AuditedIKRequest) -> dict[str, Any]:
    if _robot is None or _identity is None:
        raise HTTPException(503, "PyRoKi is not initialized")
    try:
        public_position = _finite_vector(request.position_xyz_m, 3, "position_xyz_m")
        target_world_wxyz = _finite_vector(
            request.quaternion_wxyz, 4, "quaternion_wxyz"
        )
        norm = float(np.linalg.norm(target_world_wxyz))
        if norm < 1e-8:
            raise ValueError("target quaternion has zero norm")
        target_world_wxyz /= norm
        offset = _finite_vector(
            request.tcp_offset_tool_frame_m, 3, "tcp_offset_tool_frame_m"
        )
        frozen_offset = np.asarray(
            P3_IK_PROVIDER_CONTRACT["tcp_offset_tool_frame_m"], dtype=np.float64
        )
        if not np.array_equal(offset, frozen_offset):
            raise ValueError("TCP offset differs from frozen P3 contract")
        target_world_rotation = Rotation.from_quat(
            [
                target_world_wxyz[1],
                target_world_wxyz[2],
                target_world_wxyz[3],
                target_world_wxyz[0],
            ]
        )
        if (
            request.previous_joint_positions_rad is None
            or request.current_eef_position_xyz_m is None
            or request.current_eef_quaternion_xyzw is None
        ):
            raise ValueError(
                "audited world-frame IK requires current joints and measured EEF pose"
            )
        previous = _finite_vector(
            request.previous_joint_positions_rad,
            ARM_JOINT_COUNT,
            "previous_joint_positions_rad",
        )
        current_public_position = _finite_vector(
            request.current_eef_position_xyz_m,
            3,
            "current_eef_position_xyz_m",
        )
        current_public_xyzw = _finite_vector(
            request.current_eef_quaternion_xyzw,
            4,
            "current_eef_quaternion_xyzw",
        )
        current_quaternion_norm = float(np.linalg.norm(current_public_xyzw))
        if current_quaternion_norm < 1e-8:
            raise ValueError("current EEF quaternion has zero norm")
        current_public_xyzw /= current_quaternion_norm
        current_world_rotation = Rotation.from_quat(current_public_xyzw)
        full_previous = np.concatenate(
            [previous, np.asarray([PANDA_GRIPPER_SEED_RAD])]
        )

        # PyRoKi's Panda FK is expressed in its model base frame, whereas the
        # benchmark contract and LIBERO observations use the simulator world
        # frame.  Recover the rigid world<-base transform from the same
        # persistent environment's measured EEF pose and current joint FK.
        # This is task-agnostic calibration, not a semantic coordinate hint.
        link_index = _robot.links.names.index(TARGET_LINK)
        current_fk = np.asarray(
            _robot.forward_kinematics(full_previous), dtype=np.float64
        )[link_index]
        current_base_wxyz = current_fk[:4]
        current_base_hand_position = current_fk[4:]
        current_base_rotation = Rotation.from_quat(
            [
                current_base_wxyz[1],
                current_base_wxyz[2],
                current_base_wxyz[3],
                current_base_wxyz[0],
            ]
        )
        current_world_hand_position = (
            current_public_position + current_world_rotation.apply(offset)
        )
        world_from_base_rotation = (
            current_world_rotation * current_base_rotation.inv()
        )
        world_base_position = (
            current_world_hand_position
            - world_from_base_rotation.apply(current_base_hand_position)
        )

        target_world_hand_position = (
            public_position + target_world_rotation.apply(offset)
        )
        base_from_world_rotation = world_from_base_rotation.inv()
        target_base_rotation = base_from_world_rotation * target_world_rotation
        target_base_position = base_from_world_rotation.apply(
            target_world_hand_position - world_base_position
        )
        target_base_xyzw = target_base_rotation.as_quat()
        target_base_wxyz = np.asarray(
            [
                target_base_xyzw[3],
                target_base_xyzw[0],
                target_base_xyzw[1],
                target_base_xyzw[2],
            ],
            dtype=np.float64,
        )
        solved_full = np.asarray(
            _solve_continuous(
                robot=_robot,
                target_link_name=TARGET_LINK,
                target_wxyz=target_base_wxyz,
                target_position=target_base_position,
                prev_cfg=full_previous,
                initial_cfg=full_previous,
            ),
            dtype=np.float64,
        )
        seed_source = "measured_current_joint_state_with_world_base_alignment"
        if solved_full.shape[0] < ARM_JOINT_COUNT or not np.isfinite(solved_full).all():
            raise RuntimeError("PyRoKi returned an invalid joint vector")
        lower_full = np.asarray(_robot.joints.lower_limits, dtype=np.float64)
        upper_full = np.asarray(_robot.joints.upper_limits, dtype=np.float64)
        fk_all = np.asarray(_robot.forward_kinematics(solved_full), dtype=np.float64)
        fk_wxyz_xyz = fk_all[link_index]
        fk_base_wxyz = fk_wxyz_xyz[:4]
        fk_base_hand_position = fk_wxyz_xyz[4:]
        fk_base_rotation = Rotation.from_quat(
            [
                fk_base_wxyz[1],
                fk_base_wxyz[2],
                fk_base_wxyz[3],
                fk_base_wxyz[0],
            ]
        )
        fk_world_rotation = world_from_base_rotation * fk_base_rotation
        fk_world_hand_position = (
            world_base_position
            + world_from_base_rotation.apply(fk_base_hand_position)
        )
        fk_public_position = (
            fk_world_hand_position - fk_world_rotation.apply(offset)
        )
        fk_xyzw = fk_world_rotation.as_quat().tolist()
        arm = solved_full[:ARM_JOINT_COUNT]
        lower = lower_full[:ARM_JOINT_COUNT]
        upper = upper_full[:ARM_JOINT_COUNT]
        joint_limits_ok = bool(np.all(lower <= arm) and np.all(arm <= upper))
        return {
            "identity": _identity,
            "joint_positions_rad": arm.tolist(),
            "joint_lower_limits_rad": lower.tolist(),
            "joint_upper_limits_rad": upper.tolist(),
            "joint_limits_ok": joint_limits_ok,
            "fk_position_xyz_m": fk_public_position.tolist(),
            "fk_quaternion_xyzw": fk_xyzw,
            "seed_source": seed_source,
            "frame_alignment": {
                "method": "measured_eef_pose_plus_current_joint_fk",
                "world_base_position_xyz_m": world_base_position.tolist(),
                "world_base_quaternion_xyzw": (
                    world_from_base_rotation.as_quat().tolist()
                ),
            },
        }
    except Exception as exc:
        raise HTTPException(422, f"audited IK failed: {type(exc).__name__}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cap-x-root", type=Path, required=True)
    parser.add_argument("--pyroki-wheel", type=Path, required=True)
    parser.add_argument("--pyroki-origin-json", type=Path, required=True)
    args = parser.parse_args()
    initialize(
        cap_x_root=args.cap_x_root.resolve(),
        pyroki_wheel=args.pyroki_wheel.resolve(),
        pyroki_origin_json=args.pyroki_origin_json.resolve(),
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

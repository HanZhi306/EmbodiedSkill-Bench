"""Repository-owned JOINT_POSITION LIBERO-Pro environment service for P3.

The pinned RPent server builds RLinf's OSC environment and exposes a 7-D
action.  P3 must not attach joint-space IK to that process.  This module owns a
separate, explicit 8-D ``JOINT_POSITION`` environment while preserving the
same LIBERO-Pro BDDL, task-local init-state selection, official success check,
camera surface, and persistent episode identity.

The public agent never sees this RPC surface.  Only the in-process benchmark
backend can connect to it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# These must be installed before importing LIBERO / robosuite / MuJoCo.
os.environ.setdefault("LIBERO_TYPE", "pro")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
assert "mujoco" not in sys.modules, (
    "mujoco must not be imported before MUJOCO_GL/PYOPENGL_PLATFORM are set"
)

from rpent.utils.logging import get_logger  # noqa: E402
from rpent.utils.rpc import RpcClient, RpcFacade  # noqa: E402

from .challenge_env import ChallengeEnvWrapper, validate_challenge_spec  # noqa: E402

if TYPE_CHECKING:
    from liberopro.liberopro.envs import OffScreenRenderEnv


CONTROLLER_KIND = "JOINT_POSITION"
ARM_ACTION_DIM = 7
ACTION_DIM = 8
CONTROL_FREQ_HZ = 20
RESET_SETTLE_STEPS = 15
GRIPPER_CONVENTION = {
    "public_open": "open",
    "public_closed": "closed",
    "controller_open": -1.0,
    "controller_closed": 1.0,
}
logger = get_logger("esbench_joint_position_libero")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _package_identity(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
        root = Path(distribution.locate_file(""))
        direct_url = distribution.read_text("direct_url.json")
        return {
            "name": name,
            "version": distribution.version,
            "location": str(root.resolve()),
            "direct_url": json.loads(direct_url) if direct_url else None,
        }
    except importlib.metadata.PackageNotFoundError:
        return {"name": name, "version": None, "location": None, "direct_url": None}


def _controller_config() -> dict[str, Any]:
    import robosuite as suite

    raw = _jsonable(suite.load_controller_config(default_controller=CONTROLLER_KIND))
    return {
        "kind": CONTROLLER_KIND,
        "robosuite_config": raw,
        "action_dim": ACTION_DIM,
        "arm_action_dim": ARM_ACTION_DIM,
        "arm_command": "(target_joint_rad-current_joint_rad)*control_freq_hz",
        "control_freq_hz": CONTROL_FREQ_HZ,
        "gripper": GRIPPER_CONVENTION,
        "reset_settle_steps": RESET_SETTLE_STEPS,
    }


class JointPositionLiberoFacade(RpcFacade):
    """Private RPC facade over one direct LIBERO-Pro environment."""

    def __init__(
        self,
        *,
        suite_name: str,
        task_id: int,
        seed: int,
        max_episode_steps: int,
        expected_server_sha256: str,
        bddl_file_path: str | Path | None = None,
        init_states_file_path: str | Path | None = None,
        task_language: str | None = None,
        challenge_spec: dict[str, Any] | None = None,
        challenge_audit_path: str | Path | None = None,
        expected_bddl_sha256: str | None = None,
        expected_init_states_sha256: str | None = None,
    ):
        super().__init__()
        own_sha = _sha256_file(__file__)
        if own_sha != expected_server_sha256:
            raise RuntimeError(
                "joint server source hash mismatch: "
                f"expected {expected_server_sha256}, actual {own_sha}"
            )

        from liberopro.liberopro import get_libero_path
        from liberopro.liberopro import benchmark
        from liberopro.liberopro.envs import OffScreenRenderEnv

        challenge_values = {
            "bddl_file_path": bddl_file_path,
            "init_states_file_path": init_states_file_path,
            "task_language": task_language,
            "challenge_spec": challenge_spec,
            "challenge_audit_path": challenge_audit_path,
            "expected_bddl_sha256": expected_bddl_sha256,
            "expected_init_states_sha256": expected_init_states_sha256,
        }
        supplied_challenge = [
            key for key, value in challenge_values.items() if value is not None
        ]
        if supplied_challenge and len(supplied_challenge) != len(challenge_values):
            missing = sorted(set(challenge_values) - set(supplied_challenge))
            raise ValueError(
                "challenge JOINT_POSITION requires all explicit task inputs; "
                f"missing {missing}"
            )
        normalized_challenge: dict[str, Any] | None = None
        init_states_file_sha256: str | None = None
        if supplied_challenge:
            import torch

            bddl_path = Path(str(bddl_file_path)).expanduser().resolve()
            init_states_path = Path(str(init_states_file_path)).expanduser().resolve()
            if not bddl_path.is_file():
                raise FileNotFoundError(bddl_path)
            if not init_states_path.is_file():
                raise FileNotFoundError(init_states_path)
            actual_bddl_sha256 = _sha256_file(bddl_path)
            init_states_file_sha256 = _sha256_file(init_states_path)
            if actual_bddl_sha256 != expected_bddl_sha256:
                raise RuntimeError(
                    "challenge JOINT_POSITION BDDL hash mismatch: "
                    f"expected {expected_bddl_sha256}, actual {actual_bddl_sha256}"
                )
            if init_states_file_sha256 != expected_init_states_sha256:
                raise RuntimeError(
                    "challenge JOINT_POSITION init-state hash mismatch: "
                    f"expected {expected_init_states_sha256}, actual "
                    f"{init_states_file_sha256}"
                )
            normalized_challenge = validate_challenge_spec(
                dict(challenge_spec or {})
            )
            init_states = torch.load(init_states_path, weights_only=False)
            resolved_task_language = str(task_language)
        else:
            suite_cls = benchmark.get_benchmark(suite_name)
            if suite_cls is None:
                raise ValueError(f"unknown LIBERO-Pro suite {suite_name!r}")
            task_suite = suite_cls()
            task = task_suite.get_task(int(task_id))
            init_states = task_suite.get_task_init_states(int(task_id))
            bddl_path = (
                Path(get_libero_path("bddl_files"))
                / task.problem_folder
                / task.bddl_file
            ).resolve()
            if not bddl_path.is_file():
                raise FileNotFoundError(bddl_path)
            resolved_task_language = str(task.language)
        if len(init_states) < 1:
            raise RuntimeError("LIBERO-Pro task has no init states")
        trial_index = int(seed) % len(init_states)
        initial_state = np.asarray(init_states[trial_index], dtype=np.float64)

        controller_config = _controller_config()
        base_env: OffScreenRenderEnv = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=256,
            camera_widths=256,
            camera_depths=True,
            controller=CONTROLLER_KIND,
            control_freq=CONTROL_FREQ_HZ,
            horizon=int(max_episode_steps),
            ignore_done=False,
            hard_reset=True,
        )
        actual_action_dim = int(getattr(base_env.env, "action_dim", -1))
        if normalized_challenge is not None:
            self._env = ChallengeEnvWrapper(
                base_env,
                normalized_challenge,
                Path(str(challenge_audit_path)).expanduser().resolve(),
            )
        else:
            self._env = base_env
        self._env.seed(int(seed))
        if actual_action_dim != ACTION_DIM:
            self._env.close()
            raise RuntimeError(
                f"JOINT_POSITION action_dim must be {ACTION_DIM}, got {actual_action_dim}"
            )
        self._panda_joint_qpos_addresses: list[int] = []
        for joint_index in range(1, ARM_ACTION_DIM + 1):
            address = self._env.sim.model.get_joint_qpos_addr(
                f"robot0_joint{joint_index}"
            )
            if isinstance(address, tuple):
                address = address[0]
            self._panda_joint_qpos_addresses.append(int(address))

        self._suite_name = str(suite_name)
        self._task_id = int(task_id)
        self._seed = int(seed)
        self._max_episode_steps = int(max_episode_steps)
        self._task_language = resolved_task_language
        self._initial_state = initial_state
        self._trial_index = trial_index
        self._bddl_path = bddl_path
        self._controller_config = controller_config
        self._instance_id = str(uuid.uuid4())
        self._raw_obs: dict[str, Any] | None = None
        self._elapsed_steps = 0
        self._terminated = False
        self._truncated = False
        self._meta: dict[str, Any] = {
            "suite": self._suite_name,
            "task": self._task_id,
            "seed": self._seed,
            "max_episode_steps": self._max_episode_steps,
            "controller_kind": CONTROLLER_KIND,
            "environment_instance_id": self._instance_id,
        }
        if normalized_challenge is not None:
            self._meta.update(
                {
                    "challenge_cell_id": normalized_challenge[
                        "challenge_cell_id"
                    ],
                    "bddl_sha256": _sha256_file(bddl_path),
                    "init_states_sha256": init_states_file_sha256,
                }
            )
        dependency_identity = {
            "liberopro": _package_identity("liberopro"),
            "robosuite": _package_identity("robosuite"),
            "mujoco": _package_identity("mujoco"),
        }
        immutable_attestation = {
            "schema_version": 1,
            "controller_config": controller_config,
            "controller_config_sha256": _canonical_sha256(controller_config),
            "server_source_sha256": own_sha,
            "dependency_identity": dependency_identity,
            "bddl_sha256": _sha256_file(bddl_path),
            "init_state_sha256": _sha256_bytes(initial_state.tobytes(order="C")),
            "trial_index": trial_index,
            "task_trial_count": len(init_states),
            "official_success_source": "liberopro OffScreenRenderEnv.check_success",
            "raw_done_is_success": False,
            "rpc_execution_thread": "constructor-thread-single-thread-http",
            "challenge": (
                {
                    "challenge_cell_id": normalized_challenge[
                        "challenge_cell_id"
                    ],
                    "spec_sha256": _canonical_sha256(normalized_challenge),
                    "wrapper_source_sha256": _sha256_file(
                        Path(__file__).with_name("challenge_env.py")
                    ),
                    "bddl_sha256": _sha256_file(bddl_path),
                    "init_states_file_sha256": init_states_file_sha256,
                    "agent_visible_private_audit": False,
                }
                if normalized_challenge is not None
                else None
            ),
            "high_resolution_capture": {
                "source": "current native 256x256 RGB-D observation",
                "transform": "deterministic nearest-neighbor upsample",
                "native_simulator_render": False,
                "additional_spatial_information": False,
            },
        }
        self._attestation = {
            **immutable_attestation,
            "attestation_sha256": _canonical_sha256(immutable_attestation),
            "environment_instance_id": self._instance_id,
        }
        self.reset()

    def serve(
        self,
        *,
        transport: str,
        host: str,
        port: int,
        parent_watch: bool = False,
    ) -> None:
        """Serve every MuJoCo/EGL operation on the constructor thread.

        RPent's general HTTP transport uses ``ThreadingHTTPServer`` and the
        base facade runs ``serve_forever`` in another thread. Both are valid
        for ordinary providers but violate MuJoCo EGL context affinity: the
        environment is constructed on this thread and must keep rendering and
        stepping here. This repository-owned environment therefore handles
        requests synchronously and runs the server loop on the current thread.
        """

        if transport != "http":
            raise ValueError("JOINT_POSITION LIBERO supports only audited HTTP RPC")
        from rpent.utils.daemon import watch_parent_death
        from rpent.utils.http_rpc import HttpRpcServer

        class _ConstructorThreadHttpRpcServer(HttpRpcServer):
            daemon_threads = False

            def process_request(self, request, client_address) -> None:
                try:
                    self.finish_request(request, client_address)
                except Exception:
                    self.handle_error(request, client_address)
                finally:
                    self.shutdown_request(request)

        def dispatch(method: str, args: tuple, kwargs: dict) -> Any:
            if method == "healthz":
                return {"status": "ok"}
            if method == "shutdown":
                self._shutdown_event.set()
                return {"ok": True}
            return self._dispatch(method, args, kwargs)

        server = _ConstructorThreadHttpRpcServer((host, port), dispatch)
        bound_host, bound_port = server.server_address
        client_host = "127.0.0.1" if bound_host == "0.0.0.0" else bound_host
        url = f"http://{client_host}:{bound_port}"
        print(f"RPC server listening on {url}", flush=True)
        logger.info("RPC server listening on %s", url)
        if parent_watch:
            watch_parent_death(self._shutdown_event.set)

        def stop_when_requested() -> None:
            self._shutdown_event.wait()
            server.shutdown()

        threading.Thread(target=stop_when_requested, daemon=True).start()
        try:
            server.serve_forever()
        finally:
            self._shutdown_event.set()
            server.server_close()

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if not method.startswith("env."):
            raise ValueError(f"unknown RPC method: {method!r}")
        name = method[len("env.") :]
        if name.startswith("_") or not hasattr(self, name):
            raise ValueError(f"unknown environment method: {name!r}")
        return getattr(self, name)(*args, **kwargs)

    def reset(self):
        self._env.seed(self._seed)
        self._env.reset()
        self._raw_obs = self._augment_raw(
            self._env.set_init_state(self._initial_state)
        )
        self._elapsed_steps = 0
        self._terminated = False
        self._truncated = False
        settle_steps = 0
        settle_action = np.zeros(ACTION_DIM, dtype=np.float64)
        settle_action[-1] = GRIPPER_CONVENTION["controller_open"]
        for _ in range(RESET_SETTLE_STEPS):
            raw_obs, _reward, _done, _info = self._env.step(settle_action)
            self._raw_obs = self._augment_raw(raw_obs)
            settle_steps += 1
            if bool(self._env.check_success()):
                self._terminated = True
                break
        # Match RPent/RLinf accounting: reset settling advances the underlying
        # simulator but does not consume the benchmark action-step horizon.
        self._elapsed_steps = 0
        return self._raw_obs, {
            "environment_instance_id": self._instance_id,
            "trial_index": self._trial_index,
            "settle_steps": settle_steps,
            "official_success": self._terminated,
        }

    def step(self, action):
        if self._terminated or self._truncated:
            raise RuntimeError("step called after episode done")
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            raise ValueError(f"JOINT_POSITION action must be finite shape [{ACTION_DIM}]")
        raw_obs, reward, done, info = self._env.step(value)
        self._raw_obs = self._augment_raw(raw_obs)
        self._elapsed_steps += 1
        official_success = bool(self._env.check_success())
        self._terminated = self._terminated or official_success
        self._truncated = self._truncated or (
            bool(done) and not official_success
        ) or self._elapsed_steps >= self._max_episode_steps
        public_info = dict(_jsonable(info) or {})
        public_info["esbench_official_status"] = {
            "official_success": official_success,
            "raw_env_done": bool(done),
            "benchmark_truncated": self._truncated,
            "elapsed_controller_steps": self._elapsed_steps,
        }
        return (
            self._raw_obs,
            float(reward),
            self._terminated,
            self._truncated,
            public_info,
        )

    def _augment_raw(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Own simulator buffers and attach joints from the same environment.

        Robosuite camera/depth observations may be views into MuJoCo's
        offscreen framebuffer. The private RPC serializes this mapping only
        after ``env.step`` returns, so retaining those views can expose reused
        or invalid GPU-backed memory to the host. Copy every ndarray at the
        capture boundary; this is observation ownership, not a transformation.
        """

        value = {
            key: item.copy() if isinstance(item, np.ndarray) else item
            for key, item in raw.items()
        }
        value["robot0_joint_pos"] = np.asarray(
            self._env.sim.data.qpos[self._panda_joint_qpos_addresses],
            dtype=np.float64,
        ).copy()
        return value

    def raw_obs(self) -> dict[str, Any]:
        if self._raw_obs is None:
            raise RuntimeError("environment has no observation")
        return self._raw_obs

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ):
        """Return a side-effect-free view of the current observation.

        Calling ``sim.render`` with a new resolution after LIBERO creates its
        256x256 EGL context can invalidate robosuite's offscreen framebuffer.
        That would make a read-only capture break the next physical action.
        P3 therefore derives requested resolutions from the current native
        RGB-D buffers.  Metadata explicitly marks this as carrying no extra
        spatial information.
        """

        if self._raw_obs is None:
            raise RuntimeError("environment has no observation")
        key_by_camera = {
            "agentview": ("agentview_image", "agentview_depth"),
            "robot0_eye_in_hand": (
                "robot0_eye_in_hand_image",
                "robot0_eye_in_hand_depth",
            ),
        }
        try:
            rgb_key, depth_key = key_by_camera[str(camera_name)]
        except KeyError as exc:
            raise ValueError(f"unsupported camera {camera_name!r}") from exc
        target_height = int(height)
        target_width = int(width)
        if target_height < 1 or target_width < 1:
            raise ValueError("camera height and width must be positive")

        def resize_nearest(value: Any) -> np.ndarray:
            source = np.asarray(value)
            source_height, source_width = source.shape[:2]
            rows = np.minimum(
                (np.arange(target_height) * source_height // target_height),
                source_height - 1,
            )
            columns = np.minimum(
                (np.arange(target_width) * source_width // target_width),
                source_width - 1,
            )
            return np.ascontiguousarray(source[rows[:, None], columns[None, :]])

        rgb = resize_nearest(self._raw_obs[rgb_key])
        if not depth:
            return rgb
        raw_depth = np.asarray(self._raw_obs[depth_key])
        if raw_depth.ndim == 3 and raw_depth.shape[-1] == 1:
            raw_depth = raw_depth[..., 0]
        return rgb, resize_nearest(raw_depth)

    def get_camera_meta(
        self,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ) -> dict[str, Any]:
        from robosuite.utils import camera_utils

        intrinsic = camera_utils.get_camera_intrinsic_matrix(
            self._env.sim, str(camera_name), int(height), int(width)
        )
        extrinsic = camera_utils.get_camera_extrinsic_matrix(
            self._env.sim, str(camera_name)
        )
        extent = float(self._env.sim.model.stat.extent)
        near = float(self._env.sim.model.vis.map.znear) * extent
        far = float(self._env.sim.model.vis.map.zfar) * extent
        return {
            "camera_name": str(camera_name),
            "height": int(height),
            "width": int(width),
            "intrinsic_K": intrinsic.tolist(),
            "extrinsic_cam2world": extrinsic.tolist(),
            "depth_near": near,
            "depth_far": far,
            "capture_provenance": {
                "native_resolution": [256, 256],
                "requested_resolution": [int(height), int(width)],
                "derived_from_low_res": bool(int(height) != 256 or int(width) != 256),
                "resampling": (
                    "nearest-neighbor"
                    if int(height) != 256 or int(width) != 256
                    else "none"
                ),
                "additional_spatial_information": False,
            },
        }

    def get_task_language(self) -> str:
        return self._task_language

    def cached_image(self):
        return None

    def get_env_meta(self) -> dict[str, Any]:
        return dict(self._meta)

    def get_controller_attestation(self) -> dict[str, Any]:
        return dict(self._attestation)


_TIMEOUT = {
    "default": 30.0,
    "env.reset": 120.0,
    "env.step": 120.0,
    "env.render_camera": 120.0,
}


class JointPositionLiberoClient:
    """Host-side private client matching the subset consumed by RPent tools."""

    def __init__(self, rpc: RpcClient, *, expected_meta: dict[str, Any]):
        self._rpc = rpc
        actual = self._call("get_env_meta")
        if actual != expected_meta:
            raise RuntimeError(
                f"joint LIBERO env meta mismatch: expected={expected_meta!r} actual={actual!r}"
            )
        self.environment_instance_id = str(actual["environment_instance_id"])
        self._attestation = self._call("get_controller_attestation")
        if self._attestation.get("environment_instance_id") != self.environment_instance_id:
            raise RuntimeError("joint controller attestation belongs to another environment")
        self.episode_terminated = False
        self.episode_truncated = False
        self.reset()

    def _call(self, method: str, *, args: tuple = (), kwargs: dict | None = None):
        key = f"env.{method}"
        return self._rpc.call(
            key,
            args=args,
            kwargs=kwargs or {},
            timeout_s=_TIMEOUT.get(key, _TIMEOUT["default"]),
        )

    def reset(self):
        raw, info = self._call("reset")
        self.episode_terminated = False
        self.episode_truncated = False
        return self._wrap_raw(raw), info

    def step(self, action):
        raw, reward, terminated, truncated, info = self._call(
            "step", args=(action,)
        )
        self.episode_terminated |= bool(terminated)
        self.episode_truncated |= bool(truncated)
        return self._wrap_raw(raw), reward, terminated, truncated, info

    def _wrap_raw(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Provide RPent's expected observation shell without changing truth."""

        from scipy.spatial.transform import Rotation

        quaternion = np.asarray(raw["robot0_eef_quat"], dtype=np.float64)
        axis_angle = Rotation.from_quat(quaternion).as_rotvec()
        state = np.concatenate(
            [
                np.asarray(raw["robot0_eef_pos"], dtype=np.float64),
                axis_angle,
                np.asarray(raw["robot0_gripper_qpos"], dtype=np.float64),
            ]
        )
        return {
            "main_images": np.asarray(raw["agentview_image"]),
            "wrist_images": np.asarray(raw["robot0_eye_in_hand_image"]),
            "states": state,
            "task_descriptions": self.get_task_language(),
        }

    def raw_obs(self):
        return self._call("raw_obs")

    def render_camera(self, camera_name="agentview", height=1024, width=1024, depth=False):
        return self._call(
            "render_camera",
            kwargs={
                "camera_name": camera_name,
                "height": height,
                "width": width,
                "depth": depth,
            },
        )

    def get_camera_meta(self, camera_name="agentview", height=256, width=256):
        return self._call(
            "get_camera_meta",
            kwargs={"camera_name": camera_name, "height": height, "width": width},
        )

    def get_task_language(self):
        return self._call("get_task_language")

    def cached_image(self):
        return self._call("cached_image")

    def controller_attestation(self) -> dict[str, Any]:
        current = self._call("get_controller_attestation")
        if current != self._attestation:
            raise RuntimeError("joint controller attestation changed during episode")
        return dict(current)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-episode-steps", type=int, default=10_000)
    parser.add_argument("--expected-server-sha256", required=True)
    parser.add_argument("--bddl-path")
    parser.add_argument("--init-states-path")
    parser.add_argument("--task-language")
    parser.add_argument("--challenge-spec-path")
    parser.add_argument("--challenge-audit-path")
    parser.add_argument("--expected-bddl-sha256")
    parser.add_argument("--expected-init-states-sha256")
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=None)
    args = parser.parse_args()
    if args.cuda_device is not None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.cuda_device)
    challenge_spec = None
    if args.challenge_spec_path is not None:
        spec_path = Path(args.challenge_spec_path).expanduser().resolve()
        if not spec_path.is_file():
            raise FileNotFoundError(spec_path)
        challenge_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    facade = JointPositionLiberoFacade(
        suite_name=args.suite,
        task_id=args.task,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        expected_server_sha256=args.expected_server_sha256,
        bddl_file_path=args.bddl_path,
        init_states_file_path=args.init_states_path,
        task_language=args.task_language,
        challenge_spec=challenge_spec,
        challenge_audit_path=args.challenge_audit_path,
        expected_bddl_sha256=args.expected_bddl_sha256,
        expected_init_states_sha256=args.expected_init_states_sha256,
    )
    facade.serve(
        transport="http",
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()

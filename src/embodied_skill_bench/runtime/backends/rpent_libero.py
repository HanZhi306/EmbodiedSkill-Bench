from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..grasp_verification import annotate_pi0_pick_result
from ..grounded_joint import (
    GROUNDED_JOINT_PROVIDER,
    GROUNDED_JOINT_SKILL_IDS,
    execute_grounded_joint_action,
)
from ..grounded_osc import (
    GROUNDED_OSC_SKILL_IDS,
    execute_flat_package_to_woven_basket,
)
from ..ik import PyRoKiHttpProvider, quaternion_angular_error_rad
from ..joint_position_libero_server import (
    ACTION_DIM as JOINT_ACTION_DIM,
)
from ..joint_position_libero_server import (
    ARM_ACTION_DIM as JOINT_ARM_ACTION_DIM,
)
from ..joint_position_libero_server import (
    CONTROL_FREQ_HZ as JOINT_CONTROL_FREQ_HZ,
)
from ..joint_position_libero_server import (
    CONTROLLER_KIND as JOINT_CONTROLLER_KIND,
)
from ..joint_position_libero_server import (
    GRIPPER_CONVENTION as JOINT_GRIPPER_CONVENTION,
)
from ..joint_position_libero_server import (
    JointPositionLiberoClient,
)
from ..perception import (
    ContactGraspNetHttpClient,
    OpenAICompatibleVisionClient,
    run_contact_graspnet_proposals,
    run_molmo2_point_localization,
    run_molmo2_sam3_reference_binding,
    run_sam3_localization,
    run_sam3_point_segmentation,
    run_visual_difference_description,
)
from ..seeding import (
    CHALLENGE_PAIR_VLA_SEED_NAMESPACE,
    VLA_SEED_NAMESPACE,
    derive_challenge_pair_vla_seed,
    derive_vla_seed,
)
from ..semantic_vla import (
    INTERNAL_GROUNDING_HANDOFFS_KEY,
    is_semantic_vla_skill,
    annotate_semantic_vla_result,
    build_semantic_vla_call,
)
from ..perception_contracts import build_sam3_localization_arguments
from ..types import (
    ActionResult,
    CapturedObservation,
    InvalidSkillArgumentsError,
    PerceptionResult,
    UnsupportedSkillError,
)


class _UnavailableVLA:
    def __init__(self, profile_id: str = "rpent-generalist"):
        self.profile_id = profile_id

    def predict_action_batch(self, *_args, **_kwargs):
        raise RuntimeError(
            f"VLA policy profile {self.profile_id!r} was selected but no matching "
            "endpoint/model was configured. Set backend.vla_profiles for non-default "
            "profiles, or backend.vla_endpoint / backend.vla_model_path / "
            "PI05_CHECKPOINT_PATH for rpent-generalist."
        )


class _ProfiledVLAClient:
    """Validate one policy endpoint and account for its native action horizon."""

    def __init__(
        self,
        *,
        profile_id: str,
        client: Any,
        mode: str,
        identity: dict[str, Any] | None,
    ):
        self.profile_id = profile_id
        self.client = client
        self.mode = mode
        self.identity = dict(identity or {})
        self.forward_calls = 0
        self.action_steps = 0
        self.last_action_shape: list[int] | None = None

    def reset_accounting(self) -> None:
        self.forward_calls = 0
        self.action_steps = 0
        self.last_action_shape = None

    def predict_action_batch(self, env_obs: dict[str, Any], mode: str = "eval", **kwargs):
        actions, extra = self.client.predict_action_batch(env_obs, mode=mode, **kwargs)
        array = np.asarray(actions, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 7 or array.shape[0] < 1:
            raise RuntimeError(
                f"VLA profile {self.profile_id!r} must return [chunk,7] actions; "
                f"received {list(array.shape)}"
            )
        if not np.isfinite(array).all():
            raise RuntimeError(
                f"VLA profile {self.profile_id!r} returned NaN or infinite actions"
            )
        self.forward_calls += 1
        self.action_steps += int(array.shape[0])
        self.last_action_shape = [int(value) for value in array.shape]
        metadata = dict(extra or {})
        metadata.update(
            {
                "policy_profile": self.profile_id,
                "native_action_shape": list(self.last_action_shape),
            }
        )
        return array, metadata


class _UnavailableSam3:
    def segment(self, *_args, **_kwargs):
        raise RuntimeError(
            "A SAM3 perception skill was selected but no SAM3 endpoint/model was "
            "configured. Set backend.sam3_endpoint or "
            "backend.sam3_model_path/SAM3_CHECKPOINT_PATH."
        )


def _gripper_value(value: Any, default: float) -> float:
    if value is None:
        return default
    if value in ("open", -1, -1.0):
        return -1.0
    if value in ("closed", "close", 1, 1.0):
        return 1.0
    raise ValueError("gripper must be 'open' or 'closed'")


def _copy_if_present(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return True


def _resolution(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    with Image.open(path) as image:
        return [image.width, image.height]


def _openvla_oft_unnorm_key(suite: str) -> str:
    """Map LIBERO-Pro perturbation suite names to the four training stats."""

    value = str(suite)
    for family in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        if value == family or value.startswith(f"{family}_"):
            return f"{family}_no_noops"
    raise ValueError(
        f"OpenVLA-OFT has no normalization statistics for suite {value!r}"
    )


@contextmanager
def _local_vla_startup_lock(cuda_device: int | None) -> Iterator[dict[str, Any]]:
    """Serialize heavyweight local policy cold starts on one physical worker.

    Multiple already-loaded policy daemons may run concurrently on a large GPU,
    but simultaneous PyTorch checkpoint loads contend heavily enough that some
    daemons miss RPent's fixed readiness deadline.  The lock covers only daemon
    creation and readiness; inference is concurrent after it is released.
    """

    root = Path(
        os.environ.get("ESB_VLA_STARTUP_LOCK_DIR", tempfile.gettempdir())
    ).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    device = "cpu" if cuda_device is None else f"cuda-{int(cuda_device)}"
    path = root / f"embodied-skill-bench-vla-startup-{device}.lock"
    started = time.monotonic()
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        acquired = time.monotonic()
        try:
            yield {
                "lock_path": str(path),
                "wait_seconds": acquired - started,
                "device": device,
            }
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class RPentLiberoBackend:
    """Thin, non-MCP adapter over RPent's LIBERO services and controllers."""

    environment_id = "libero-pro"

    def __init__(
        self,
        *,
        rpent_repo: str | Path,
        suite: str,
        task: int,
        seed: int,
        runtime_dir: str | Path,
        max_episode_steps: int = 10_000,
        env_endpoint: str | None = None,
        vla_endpoint: str | None = None,
        vla_model_path: str | None = None,
        vla_profiles: dict[str, Any] | None = None,
        required_vla_profiles: list[str] | None = None,
        sam3_endpoint: str | None = None,
        sam3_model_path: str | None = None,
        contact_graspnet_endpoint: str | None = None,
        contact_graspnet_timeout_seconds: float = 120.0,
        contact_graspnet_identity: dict[str, Any] | None = None,
        molmo2_endpoint: str | None = None,
        molmo2_model: str = "allenai/Molmo2-8B",
        molmo2_api_key: str | None = None,
        molmo2_timeout_seconds: float = 180.0,
        molmo2_identity: dict[str, Any] | None = None,
        visual_difference_endpoint: str | None = None,
        visual_difference_model: str = "allenai/Molmo2-8B",
        visual_difference_api_key: str | None = None,
        visual_difference_timeout_seconds: float = 180.0,
        visual_difference_identity: dict[str, Any] | None = None,
        grounded_joint_pyroki_endpoint: str | None = None,
        grounded_joint_pyroki_identity: dict[str, Any] | None = None,
        grounded_joint_pyroki_timeout_seconds: float = 60.0,
        vla_seed: int | None = None,
        vla_seed_source: str | None = None,
        cuda_device: int | None = None,
        expected_commit: str | None = None,
        enable_vla: bool = True,
        enable_sam3: bool = False,
        enable_contact_graspnet: bool = False,
        enable_molmo2: bool = False,
        enable_visual_difference: bool = False,
        sam3_agent_access: str = "selected read-only perception skill only",
        challenge_cell_id: str | None = None,
        challenge_pair_id: str | None = None,
        bddl_file_path: str | Path | None = None,
        init_states_file_path: str | Path | None = None,
        task_language: str | None = None,
        challenge_source_suite: str | None = None,
        challenge_source_task: int | None = None,
        challenge_spec: dict[str, Any] | None = None,
        controller_kind: str = "OSC_POSE",
        liberopro_task_data_root: str | Path | None = None,
        liberopro_bddl_sha256: str | None = None,
        liberopro_init_states_sha256: str | None = None,
    ):
        self.rpent_repo = Path(rpent_repo).expanduser().resolve()
        self.suite = suite
        self.task = int(task)
        self.seed = int(seed)
        self.runtime_dir = Path(runtime_dir).resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        if controller_kind not in {"OSC_POSE", JOINT_CONTROLLER_KIND}:
            raise ValueError("controller_kind must be OSC_POSE or JOINT_POSITION")
        self.controller_kind = str(controller_kind)
        self._joint_last_gripper_command = "open"
        self._grounded_joint_action_index = 0
        self._grounded_joint_ik_provider: PyRoKiHttpProvider | None = None
        grounded_joint_provider_values = (
            grounded_joint_pyroki_endpoint,
            grounded_joint_pyroki_identity,
        )
        if any(
            value is not None for value in grounded_joint_provider_values
        ) and not all(
            value is not None for value in grounded_joint_provider_values
        ):
            raise ValueError(
                "grounded JOINT PyRoKi endpoint and identity must be supplied together"
            )
        if grounded_joint_pyroki_timeout_seconds <= 0:
            raise ValueError("grounded JOINT PyRoKi timeout must be positive")
        if grounded_joint_pyroki_endpoint is not None:
            if self.controller_kind != JOINT_CONTROLLER_KIND:
                raise ValueError(
                    f"{GROUNDED_JOINT_PROVIDER} requires JOINT_POSITION"
                )
            endpoint = str(grounded_joint_pyroki_endpoint).rstrip("/")
            if not endpoint.startswith("http://127.0.0.1:"):
                raise ValueError(
                    "grounded JOINT PyRoKi endpoint must be explicit localhost HTTP"
                )
            if (
                not isinstance(grounded_joint_pyroki_identity, dict)
                or not grounded_joint_pyroki_identity
            ):
                raise ValueError(
                    "grounded JOINT PyRoKi identity must be a non-empty object"
                )
            self._grounded_joint_ik_provider = PyRoKiHttpProvider(
                endpoint=endpoint,
                expected_identity=grounded_joint_pyroki_identity,
                timeout_seconds=float(grounded_joint_pyroki_timeout_seconds),
            )
        self._grounded_joint_pyroki_config = {
            "provider": GROUNDED_JOINT_PROVIDER,
            "endpoint": (
                str(grounded_joint_pyroki_endpoint).rstrip("/")
                if grounded_joint_pyroki_endpoint is not None
                else None
            ),
            "expected_identity": dict(grounded_joint_pyroki_identity or {}),
            "timeout_seconds": float(grounded_joint_pyroki_timeout_seconds),
            "agent_exposed": False,
        }
        self.vla_model_path = vla_model_path or os.environ.get("PI05_CHECKPOINT_PATH")
        self.vla_endpoint = vla_endpoint
        self.required_vla_profiles = tuple(
            sorted(set(required_vla_profiles or (["rpent-generalist"] if enable_vla else [])))
        )
        self.vla_profile_specs = self._normalize_vla_profile_specs(
            vla_profiles or {},
            default_endpoint=vla_endpoint,
            default_model_path=self.vla_model_path,
        )
        self.sam3_model_path = sam3_model_path or os.environ.get(
            "SAM3_CHECKPOINT_PATH"
        )
        self.sam3_endpoint = sam3_endpoint
        self.sam3_agent_access = str(sam3_agent_access)
        self.contact_graspnet_endpoint = contact_graspnet_endpoint
        self.contact_graspnet_identity = dict(contact_graspnet_identity or {})
        self.contact_graspnet_mode = (
            "external_endpoint"
            if enable_contact_graspnet and contact_graspnet_endpoint
            else "disabled"
        )
        if enable_contact_graspnet and not contact_graspnet_endpoint:
            raise RuntimeError(
                "Contact-GraspNet perception was selected but no endpoint was "
                "configured. Set backend.contact_graspnet_endpoint or "
                "CONTACT_GRASPNET_ENDPOINT."
            )
        self.molmo2_endpoint = molmo2_endpoint
        self.molmo2_identity = dict(molmo2_identity or {})
        self.molmo2_mode = (
            "external_endpoint" if enable_molmo2 and molmo2_endpoint else "disabled"
        )
        if enable_molmo2 and not molmo2_endpoint:
            raise RuntimeError(
                "Molmo2 point perception was selected but no endpoint was configured. "
                "Set backend.molmo2_endpoint or MOLMO2_ENDPOINT."
            )
        self.visual_difference_endpoint = visual_difference_endpoint
        self.visual_difference_identity = dict(visual_difference_identity or {})
        self.visual_difference_mode = (
            "external_endpoint"
            if enable_visual_difference and visual_difference_endpoint
            else "disabled"
        )
        if enable_visual_difference and not visual_difference_endpoint:
            raise RuntimeError(
                "visual-difference perception was selected but no endpoint was "
                "configured. Set backend.visual_difference_endpoint or "
                "VISUAL_DIFFERENCE_ENDPOINT."
            )
        self.vla_seed = (
            derive_challenge_pair_vla_seed(
                pair_id=str(challenge_pair_id), environment_seed=self.seed
            )
            if challenge_pair_id is not None and vla_seed is None
            else derive_vla_seed(
                suite=self.suite, task=self.task, environment_seed=self.seed
            )
            if vla_seed is None
            else int(vla_seed)
        )
        self.vla_seed_source = vla_seed_source or (
            CHALLENGE_PAIR_VLA_SEED_NAMESPACE
            if challenge_pair_id is not None and vla_seed is None
            else VLA_SEED_NAMESPACE
            if vla_seed is None
            else "explicit_argument"
        )
        self.vla_mode = "disabled"
        self.sam3_mode = (
            "external_endpoint"
            if enable_sam3 and sam3_endpoint
            else "local_process"
            if enable_sam3 and self.sam3_model_path
            else "disabled"
        )
        if enable_sam3 and self.sam3_mode == "disabled":
            raise RuntimeError(
                "SAM3 perception was enabled, but neither sam3_endpoint nor "
                "sam3_model_path/SAM3_CHECKPOINT_PATH is configured"
            )
        self.cuda_device = cuda_device
        challenge_values = {
            "challenge_cell_id": challenge_cell_id,
            "challenge_pair_id": challenge_pair_id,
            "bddl_file_path": bddl_file_path,
            "init_states_file_path": init_states_file_path,
            "task_language": task_language,
            "challenge_spec": challenge_spec,
        }
        provided_challenge = [
            key for key, value in challenge_values.items() if value is not None
        ]
        if provided_challenge and len(provided_challenge) != len(challenge_values):
            missing = sorted(set(challenge_values) - set(provided_challenge))
            raise ValueError(
                "custom challenge backend requires all challenge fields; "
                f"missing {missing}"
            )
        if provided_challenge and env_endpoint:
            raise ValueError(
                "custom challenge cells cannot use an external env_endpoint because "
                "the harness cannot verify its BDDL or perturbation runtime"
            )
        if self.controller_kind == JOINT_CONTROLLER_KIND and env_endpoint:
            raise ValueError(
                "P3 JOINT_POSITION refuses external env_endpoint because controller "
                "source/config identity cannot be audited"
            )
        self.challenge: dict[str, Any] | None = None
        if provided_challenge:
            bddl_path = Path(str(bddl_file_path)).expanduser().resolve()
            init_path = Path(str(init_states_file_path)).expanduser().resolve()
            if not bddl_path.is_file():
                raise FileNotFoundError(bddl_path)
            if not init_path.is_file():
                raise FileNotFoundError(init_path)
            from ..challenge_env import validate_challenge_spec

            normalized_spec = validate_challenge_spec(dict(challenge_spec or {}))
            if normalized_spec["challenge_cell_id"] != challenge_cell_id:
                raise ValueError(
                    "challenge_spec.challenge_cell_id must match challenge_cell_id"
                )
            self.challenge = {
                "cell_id": str(challenge_cell_id),
                "pair_id": str(challenge_pair_id),
                "bddl_file_path": str(bddl_path),
                "init_states_file_path": str(init_path),
                "task_language": str(task_language),
                "source_suite": (
                    str(challenge_source_suite)
                    if challenge_source_suite is not None
                    else None
                ),
                "source_task": (
                    int(challenge_source_task)
                    if challenge_source_task is not None
                    else None
                ),
                "spec": normalized_spec,
                "bddl_sha256": _sha256_file(bddl_path),
                "init_states_sha256": _sha256_file(init_path),
            }
        self._daemons: list[Any] = []
        self._step_index = 0
        self._verify_checkout(expected_commit)
        os.environ["RPENT_REPO_ROOT"] = str(self.rpent_repo)
        if str(self.rpent_repo) not in sys.path:
            sys.path.insert(0, str(self.rpent_repo))

        self.liberopro_task_snapshot = self._validate_liberopro_task_snapshot(
            task_data_root=liberopro_task_data_root,
            expected_bddl_sha256=liberopro_bddl_sha256,
            expected_init_states_sha256=liberopro_init_states_sha256,
        )

        from robots.libero import get_env_spec
        from rpent.dashboard.events import NullDashboardEventSink
        from rpent.utils.logging import init_output_dir

        init_output_dir(self.runtime_dir / "rpent-logs")
        args = argparse.Namespace(
            suite=self.suite,
            task=self.task,
            seed=self.seed,
            max_episode_steps=int(max_episode_steps),
            libero_type="pro",
            env_endpoint=env_endpoint,
            vla_endpoint=vla_endpoint,
            sam3_endpoint=sam3_endpoint,
            cuda_device=cuda_device,
        )
        sink = NullDashboardEventSink()
        if self.controller_kind == JOINT_CONTROLLER_KIND:
            env_daemons, env_kwargs = self._init_joint_position_environment(
                max_episode_steps=int(max_episode_steps)
            )
        elif self.challenge:
            env_daemons, env_kwargs = self._init_challenge_environment(
                max_episode_steps=int(max_episode_steps)
            )
        else:
            env_daemons, env_kwargs = get_env_spec().init_task_runtime(
                args, self.runtime_dir, sink
            )
        self._daemons.extend(env_daemons)
        self._vla_clients: dict[str, _ProfiledVLAClient] = {}
        if enable_vla:
            for profile_id in self.required_vla_profiles:
                self._vla_clients[profile_id] = self._connect_vla_profile(profile_id)
        profile_modes = {client.mode for client in self._vla_clients.values()}
        self.vla_mode = (
            "disabled"
            if not profile_modes
            else next(iter(profile_modes))
            if len(profile_modes) == 1
            else "multi_profile"
        )
        model = self._vla_clients.get(
            "rpent-generalist",
            next(iter(self._vla_clients.values()), _UnavailableVLA()),
        )
        self._sam3_client = (
            self._connect_sam3(sam3_endpoint)
            if enable_sam3 and (sam3_endpoint or self.sam3_model_path)
            else _UnavailableSam3()
        )
        self._contact_graspnet_client = (
            ContactGraspNetHttpClient(
                str(contact_graspnet_endpoint),
                timeout_seconds=float(contact_graspnet_timeout_seconds),
            )
            if enable_contact_graspnet and contact_graspnet_endpoint
            else None
        )
        self._molmo2_client = (
            OpenAICompatibleVisionClient(
                str(molmo2_endpoint),
                model=str(molmo2_model),
                api_key=molmo2_api_key,
                timeout_seconds=float(molmo2_timeout_seconds),
                identity=self.molmo2_identity,
            )
            if enable_molmo2 and molmo2_endpoint
            else None
        )
        self._visual_difference_client = (
            OpenAICompatibleVisionClient(
                str(visual_difference_endpoint),
                model=str(visual_difference_model),
                api_key=visual_difference_api_key,
                timeout_seconds=float(visual_difference_timeout_seconds),
                identity=self.visual_difference_identity,
            )
            if enable_visual_difference and visual_difference_endpoint
            else None
        )

        from robots.libero.tools import LiberoPrimitives

        self._primitives = LiberoPrimitives(
            env=env_kwargs["env"],
            model=model,
            sam3_client=self._sam3_client,
            check_cancelled=lambda: None,
        )
        self.reset()

    def _validate_liberopro_task_snapshot(
        self,
        *,
        task_data_root: str | Path | None,
        expected_bddl_sha256: str | None,
        expected_init_states_sha256: str | None,
    ) -> dict[str, Any] | None:
        supplied = (
            task_data_root,
            expected_bddl_sha256,
            expected_init_states_sha256,
        )
        if not any(value is not None for value in supplied):
            return None
        if not all(value is not None for value in supplied):
            raise ValueError(
                "a frozen clean LIBERO-Pro task snapshot requires data root plus "
                "both BDDL and init-state SHA256 values"
            )
        root = Path(str(task_data_root)).expanduser().resolve()
        expected_paths = {
            "bddl_files": (root / "bddl_files").resolve(),
            "init_states": (root / "init_files").resolve(),
        }
        if any(not path.is_dir() for path in expected_paths.values()):
            raise FileNotFoundError(
                f"frozen LIBERO-Pro task-data root is incomplete: {root}"
            )

        os.environ["LIBERO_TYPE"] = "pro"
        import yaml
        from liberopro.liberopro import benchmark, config_file

        config_path = Path(config_file).expanduser().resolve()
        runtime_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(runtime_config, dict):
            raise TypeError(f"malformed LIBERO-Pro runtime config: {config_path}")
        for field, expected in expected_paths.items():
            actual = Path(str(runtime_config.get(field, ""))).expanduser().resolve()
            if actual != expected:
                raise RuntimeError(
                    f"LIBERO-Pro runtime {field} drift: expected {expected}, got "
                    f"{actual}. Run scripts/configure_liberopro_data.py for the "
                    "frozen task-data root before this episode."
                )

        snapshot_suite_name = self._policy_training_suite()
        snapshot_task_index = self._policy_training_task()
        suites = benchmark.get_benchmark_dict()
        if snapshot_suite_name not in suites:
            raise RuntimeError(
                "frozen LIBERO-Pro source suite is unavailable: "
                f"{snapshot_suite_name} (runtime suite: {self.suite})"
            )
        suite = suites[snapshot_suite_name]()
        if not 0 <= snapshot_task_index < suite.n_tasks:
            raise RuntimeError(
                "frozen LIBERO-Pro task index is unavailable: "
                f"{snapshot_suite_name}:{snapshot_task_index}"
            )
        task = suite.get_task(snapshot_task_index)
        bddl_path = expected_paths["bddl_files"] / task.problem_folder / task.bddl_file
        init_path = (
            expected_paths["init_states"]
            / task.problem_folder
            / task.init_states_file
        )
        expected_hashes = {
            "bddl": str(expected_bddl_sha256),
            "init_states": str(expected_init_states_sha256),
        }
        actual_hashes = {
            "bddl": _sha256_file(bddl_path),
            "init_states": _sha256_file(init_path),
        }
        if actual_hashes != expected_hashes:
            raise RuntimeError(
                "frozen LIBERO-Pro task asset hash drift for "
                f"{snapshot_suite_name}:{snapshot_task_index}: "
                f"expected {expected_hashes}, got {actual_hashes}"
            )
        init_state_count = len(suite.get_task_init_states(snapshot_task_index))
        if init_state_count < 1:
            raise RuntimeError(
                "frozen LIBERO-Pro task has zero reset trials: "
                f"{snapshot_suite_name}:{snapshot_task_index}"
            )
        return {
            "task_data_root": str(root),
            "config_file": str(config_path),
            "suite": snapshot_suite_name,
            "runtime_suite": self.suite,
            "task": snapshot_task_index,
            "runtime_task": self.task,
            "bddl_file": str(bddl_path),
            "bddl_sha256": actual_hashes["bddl"],
            "init_states_file": str(init_path),
            "init_states_sha256": actual_hashes["init_states"],
            "init_state_count": init_state_count,
            "runtime_config_matches": True,
        }

    def _init_joint_position_environment(
        self, *, max_episode_steps: int
    ) -> tuple[list[Any], dict[str, Any]]:
        """Launch the repository-owned, task-pinned JOINT_POSITION service."""

        from rpent.utils.daemon import ProcessDaemon, pick_free_port
        from rpent.utils.http_rpc import HttpRpcClient
        from rpent.utils.rpc import wait_for_ready

        server_path = (
            Path(__file__).resolve().parents[1] / "joint_position_libero_server.py"
        )
        server_sha256 = _sha256_file(server_path)
        host, port = "127.0.0.1", pick_free_port()
        command = [
            sys.executable,
            "-m",
            "embodied_skill_bench.runtime.joint_position_libero_server",
            "--host",
            host,
            "--port",
            str(port),
            "--suite",
            self.suite,
            "--task",
            str(self.task),
            "--seed",
            str(self.seed),
            "--max-episode-steps",
            str(max_episode_steps),
            "--expected-server-sha256",
            server_sha256,
            "--parent-watch",
        ]
        if self.challenge is not None:
            spec_path = self.runtime_dir / "private-challenge-spec.json"
            spec_path.write_text(
                json.dumps(self.challenge["spec"], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            audit_path = self.runtime_dir / "private-challenge-events.jsonl"
            command.extend(
                [
                    "--bddl-path",
                    self.challenge["bddl_file_path"],
                    "--init-states-path",
                    self.challenge["init_states_file_path"],
                    "--task-language",
                    self.challenge["task_language"],
                    "--challenge-spec-path",
                    str(spec_path),
                    "--challenge-audit-path",
                    str(audit_path),
                    "--expected-bddl-sha256",
                    self.challenge["bddl_sha256"],
                    "--expected-init-states-sha256",
                    self.challenge["init_states_sha256"],
                ]
            )
        if self.cuda_device is not None:
            command.extend(["--cuda-device", str(self.cuda_device)])
        project_root = Path(__file__).resolve().parents[4]
        daemon_env = os.environ.copy()
        existing_pythonpath = daemon_env.get("PYTHONPATH")
        daemon_env.update(
            {
                "LIBERO_TYPE": "pro",
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "ROBOT_PLATFORM": "LIBERO",
                "RPENT_REPO_ROOT": str(self.rpent_repo),
                "PYTHONPATH": os.pathsep.join(
                    [
                        str(project_root / "src"),
                        str(self.rpent_repo),
                        *([existing_pythonpath] if existing_pythonpath else []),
                    ]
                ),
            }
        )
        daemon = ProcessDaemon(
            name="esbench_joint_position_libero_server",
            cmd=command,
            env=daemon_env,
            log_path=str(self.runtime_dir / "joint_position_env_server.log"),
        )
        daemon.start()
        rpc = HttpRpcClient(f"http://{host}:{port}")
        wait_for_ready(rpc, daemon=daemon)
        meta = rpc.call("env.get_env_meta", timeout_s=30.0)
        expected_meta = {
            "suite": self.suite,
            "task": self.task,
            "seed": self.seed,
            "max_episode_steps": int(max_episode_steps),
            "controller_kind": JOINT_CONTROLLER_KIND,
            "environment_instance_id": meta.get("environment_instance_id"),
        }
        if self.challenge is not None:
            expected_meta.update(
                {
                    "challenge_cell_id": self.challenge["cell_id"],
                    "bddl_sha256": self.challenge["bddl_sha256"],
                    "init_states_sha256": self.challenge[
                        "init_states_sha256"
                    ],
                }
            )
        client = JointPositionLiberoClient(rpc, expected_meta=expected_meta)
        server_attestation = client.controller_attestation()
        immutable = {
            "server_attestation_sha256": server_attestation.get(
                "attestation_sha256"
            ),
            "server_source_sha256": server_sha256,
            "adapter_source_sha256": _sha256_file(Path(__file__)),
            "rpent_commit": self.rpent_commit,
            "controller_config": server_attestation.get("controller_config"),
            "dependency_identity": server_attestation.get("dependency_identity"),
            "challenge": server_attestation.get("challenge"),
        }
        self._joint_controller_sha256 = hashlib.sha256(
            json.dumps(immutable, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        self._joint_server_attestation = server_attestation
        self._joint_controller_provenance = immutable
        return [daemon], {"env": client}

    def _init_challenge_environment(
        self, *, max_episode_steps: int
    ) -> tuple[list[Any], dict[str, Any]]:
        from robots.libero.env_client import LiberoEnvClient
        from rpent.utils.daemon import ProcessDaemon, pick_free_port
        from rpent.utils.http_rpc import HttpRpcClient
        from rpent.utils.rpc import wait_for_ready

        assert self.challenge is not None
        host, port = "127.0.0.1", pick_free_port()
        spec_path = self.runtime_dir / "private-challenge-spec.json"
        spec_path.write_text(
            json.dumps(self.challenge["spec"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        audit_path = self.runtime_dir / "private-challenge-events.jsonl"
        command = [
            sys.executable,
            "-m",
            "embodied_skill_bench.runtime.challenge_env_server",
            "--suite",
            self.suite,
            "--task",
            str(self.task),
            "--seed",
            str(self.seed),
            "--max-episode-steps",
            str(max_episode_steps),
            "--bddl-path",
            self.challenge["bddl_file_path"],
            "--init-states-path",
            self.challenge["init_states_file_path"],
            "--task-language",
            self.challenge["task_language"],
            "--challenge-spec-path",
            str(spec_path),
            "--audit-path",
            str(audit_path),
            "--transport",
            "http",
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watch",
        ]
        if self.cuda_device is not None:
            command.extend(["--cuda-device", str(self.cuda_device)])
        daemon_env = os.environ.copy()
        daemon_env.update(
            {
                "LIBERO_TYPE": "pro",
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "ROBOT_PLATFORM": "LIBERO",
            }
        )
        daemon = ProcessDaemon(
            name="challenge_env_server",
            cmd=command,
            env=daemon_env,
            log_path=str(self.runtime_dir / "env_server.log"),
        )
        daemon.start()
        rpc = HttpRpcClient(f"http://{host}:{port}")
        wait_for_ready(rpc, daemon=daemon)
        expected_meta = {
            "suite": self.suite,
            "task": self.task,
            "seed": self.seed,
            "max_episode_steps": max_episode_steps,
            "challenge_cell_id": self.challenge["cell_id"],
            "bddl_sha256": self.challenge["bddl_sha256"],
            "init_states_sha256": self.challenge["init_states_sha256"],
        }
        return [daemon], {
            "env": LiberoEnvClient(rpc, expected_meta=expected_meta)
        }

    def _verify_checkout(self, expected_commit: str | None) -> None:
        if not (self.rpent_repo / ".git").exists():
            raise FileNotFoundError(
                f"RPent checkout not found at {self.rpent_repo}; run scripts/bootstrap_rpent.sh"
            )
        actual = subprocess.check_output(
            ["git", "-C", str(self.rpent_repo), "rev-parse", "HEAD"], text=True
        ).strip()
        self.rpent_commit = actual
        if expected_commit and actual != expected_commit:
            raise RuntimeError(
                f"RPent drift: expected {expected_commit}, found {actual}"
            )

    @staticmethod
    def _normalize_vla_profile_specs(
        raw: dict[str, Any],
        *,
        default_endpoint: str | None,
        default_model_path: str | None,
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(raw, dict):
            raise TypeError("backend.vla_profiles must be an object")
        result: dict[str, dict[str, Any]] = {}
        for profile_id, value in raw.items():
            if not isinstance(value, dict):
                raise TypeError(f"backend.vla_profiles.{profile_id} must be an object")
            extras = set(value) - {
                "endpoint",
                "checkpoint_path",
                "python_executable",
                "provider_code_root",
                "base_model_path",
                "tokenizer_path",
                "require_profile_identity",
            }
            if extras:
                raise ValueError(
                    f"backend.vla_profiles.{profile_id} has unknown fields: "
                    f"{sorted(extras)}"
                )
            result[str(profile_id)] = dict(value)
        default = dict(result.get("rpent-generalist", {}))
        if default_endpoint is not None:
            if default.get("endpoint") not in {None, default_endpoint}:
                raise ValueError(
                    "backend.vla_endpoint conflicts with "
                    "backend.vla_profiles.rpent-generalist.endpoint"
                )
            default["endpoint"] = default_endpoint
        if default_model_path is not None:
            if default.get("checkpoint_path") not in {None, default_model_path}:
                raise ValueError(
                    "backend.vla_model_path conflicts with "
                    "backend.vla_profiles.rpent-generalist.checkpoint_path"
                )
            default["checkpoint_path"] = default_model_path
        if default:
            default.setdefault("require_profile_identity", False)
            result["rpent-generalist"] = default
        return result

    def _connect_vla_profile(self, profile_id: str) -> _ProfiledVLAClient:
        spec = self.vla_profile_specs.get(profile_id)
        if spec is None:
            raise RuntimeError(
                f"selected VLA policy profile {profile_id!r} has no backend.vla_profiles "
                "entry and is not configured by the legacy Pi0.5 fields"
            )
        endpoint = spec.get("endpoint")
        checkpoint_path = spec.get("checkpoint_path")
        if not endpoint and not checkpoint_path:
            raise RuntimeError(
                f"VLA profile {profile_id!r} requires endpoint or checkpoint_path"
            )
        client, mode, identity = self._connect_vla(
            endpoint,
            profile_id=profile_id,
            model_path=str(checkpoint_path) if checkpoint_path else None,
            python_executable=spec.get("python_executable"),
            provider_code_root=spec.get("provider_code_root"),
            base_model_path=spec.get("base_model_path"),
            tokenizer_path=spec.get("tokenizer_path"),
            require_profile_identity=bool(
                spec.get("require_profile_identity", profile_id != "rpent-generalist")
            ),
        )
        return _ProfiledVLAClient(
            profile_id=profile_id,
            client=client,
            mode=mode,
            identity=identity,
        )

    def _policy_training_suite(self) -> str:
        """Return the clean LIBERO suite owning challenge policy statistics.

        Challenge cells execute under the synthetic suite name
        ``esb_skill_discrimination_v6`` but reuse a frozen clean LIBERO task,
        reset, and action normalization.  New configs carry this source suite
        explicitly.  The pair-id fallback keeps already-frozen development
        configs executable without rewriting them.
        """

        if not self.challenge:
            return self.suite
        source = self.challenge.get("source_suite")
        if source:
            return str(source)
        for field in str(self.challenge.get("pair_id", "")).split("::"):
            clean_cell = field.split(":", 1)[0]
            if clean_cell.startswith(
                ("libero_spatial_", "libero_object_", "libero_goal_", "libero_10_")
            ):
                return clean_cell
        return self.suite

    def _policy_training_task(self) -> int:
        """Return the clean task index paired with a synthetic challenge cell."""

        if not self.challenge:
            return self.task
        source = self.challenge.get("source_task")
        if source is not None:
            return int(source)
        for field in str(self.challenge.get("pair_id", "")).split("::"):
            if ":" not in field:
                continue
            clean_suite, task_id = field.rsplit(":", 1)
            if clean_suite.startswith(
                ("libero_spatial_", "libero_object_", "libero_goal_", "libero_10_")
            ) and task_id.startswith("t"):
                try:
                    return int(task_id[1:])
                except ValueError:
                    continue
        return self.task

    def _connect_vla(
        self,
        endpoint: str | None,
        *,
        profile_id: str,
        model_path: str | None,
        python_executable: str | None,
        provider_code_root: str | None,
        base_model_path: str | None,
        tokenizer_path: str | None,
        require_profile_identity: bool,
    ) -> tuple[Any, str, dict[str, Any] | None]:
        from rpent.utils.http_rpc import HttpRpcClient
        from rpent.utils.rpc import parse_endpoint, wait_for_ready
        from rpent.utils.socket_rpc import SocketRpcClient
        from rpent.utils.vla_client import VLAClient

        if endpoint:
            protocol, host, port = parse_endpoint(endpoint)
            rpc = (
                SocketRpcClient(host, port)
                if protocol == "socket"
                else HttpRpcClient(f"http://{host}:{port}")
            )
            wait_for_ready(rpc)
            client = VLAClient(rpc)
            identity = client.healthz()
            reported = identity.get("policy_profile") or identity.get("profile_id")
            if require_profile_identity and reported != profile_id:
                raise RuntimeError(
                    f"VLA endpoint identity mismatch: selected {profile_id!r}, "
                    f"reported {reported!r}"
                )
            return client, "external_endpoint", identity

        from rpent.utils.config import get_repo_root
        from rpent.utils.daemon import ProcessDaemon, pick_free_port

        host, port = "127.0.0.1", pick_free_port()
        if profile_id == "rpent-generalist":
            command = [
                sys.executable,
                "-m",
                "embodied_skill_bench.runtime.seeded_vla_server",
                "--esbench-vla-seed",
                str(self.vla_seed),
                "--esbench-server-script",
                str(get_repo_root() / "robots" / "libero" / "vla_server.py"),
                "--transport",
                "http",
                "--host",
                host,
                "--port",
                str(port),
                "--model-path",
                str(model_path),
                "--parent-watch",
            ]
            daemon_name = "vla_server"
            mode = "local_process"
        else:
            if not python_executable:
                raise RuntimeError(
                    f"local VLA profile {profile_id!r} requires "
                    "backend.vla_profiles.<profile>.python_executable"
                )
            # Preserve a virtualenv's ``bin/python`` symlink. Path.resolve()
            # would collapse it to the system interpreter and silently drop
            # the profile-specific LeRobot environment.
            executable = Path(
                os.path.abspath(os.path.expanduser(str(python_executable)))
            )
            if not executable.is_file():
                raise FileNotFoundError(executable)
            command = [
                str(executable),
                "-m",
                "embodied_skill_bench.runtime.vla_policy_server",
                "--profile",
                profile_id,
                "--checkpoint-path",
                str(model_path),
                "--host",
                host,
                "--port",
                str(port),
                "--seed",
                str(self.vla_seed),
                "--parent-watch",
            ]
            if profile_id == "openvla-oft-all4":
                if not provider_code_root:
                    raise RuntimeError(
                        "local OpenVLA-OFT requires "
                        "backend.vla_profiles.openvla-oft-all4.provider_code_root"
                    )
                command.extend(
                    [
                        "--provider-code-root",
                        str(provider_code_root),
                        "--unnorm-key",
                        _openvla_oft_unnorm_key(self._policy_training_suite()),
                    ]
                )
            elif profile_id == "smolvla-libero-450m":
                if not base_model_path:
                    raise RuntimeError(
                        "local SmolVLA requires backend.vla_profiles."
                        "smolvla-libero-450m.base_model_path"
                    )
                command.extend(["--base-model-path", str(base_model_path)])
            elif profile_id == "xvla-libero-0.9b":
                if not tokenizer_path:
                    raise RuntimeError(
                        "local X-VLA requires backend.vla_profiles."
                        "xvla-libero-0.9b.tokenizer_path"
                    )
                command.extend(["--tokenizer-path", str(tokenizer_path)])
            daemon_name = f"vla_server_{profile_id}"
            mode = "local_profile_process"
        if self.cuda_device is not None:
            command.extend(["--cuda-device", str(self.cuda_device)])
        daemon_env = os.environ.copy()
        # These two settings must exist before the Python/CUDA process starts;
        # setting them inside the wrapper would be too late.
        daemon_env["PYTHONHASHSEED"] = str(self.vla_seed)
        daemon_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if profile_id != "rpent-generalist":
            project_root = Path(__file__).resolve().parents[4]
            python_paths = [str(project_root / "src"), str(self.rpent_repo)]
            if daemon_env.get("PYTHONPATH"):
                python_paths.append(daemon_env["PYTHONPATH"])
            daemon_env["PYTHONPATH"] = os.pathsep.join(python_paths)
        log_name = (
            "vla_server.log"
            if profile_id == "rpent-generalist"
            else f"vla_server_{profile_id}.log"
        )
        daemon = ProcessDaemon(
            name=daemon_name,
            cmd=command,
            env=daemon_env,
            log_path=str(self.runtime_dir / log_name),
        )
        rpc = HttpRpcClient(f"http://{host}:{port}")
        with _local_vla_startup_lock(self.cuda_device) as startup_lock:
            ready_started = time.monotonic()
            daemon.start()
            self._daemons.append(daemon)
            wait_for_ready(rpc, daemon=daemon)
            startup_lock["ready_seconds_after_lock"] = time.monotonic() - ready_started
            startup_lock["profile_id"] = profile_id
            (self.runtime_dir / f"vla_startup_lock.{profile_id}.json").write_text(
                json.dumps(startup_lock, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        client = VLAClient(rpc)
        identity = client.healthz()
        if profile_id != "rpent-generalist":
            reported = identity.get("policy_profile") or identity.get("profile_id")
            if require_profile_identity and reported != profile_id:
                raise RuntimeError(
                    f"VLA endpoint identity mismatch: selected {profile_id!r}, "
                    f"reported {reported!r}"
                )
        return client, mode, identity

    def _connect_sam3(self, endpoint: str | None):
        from rpent.utils.http_rpc import HttpRpcClient
        from rpent.utils.rpc import parse_endpoint, wait_for_ready
        from rpent.utils.sam3_client import Sam3Client
        from rpent.utils.socket_rpc import SocketRpcClient

        if endpoint:
            protocol, host, port = parse_endpoint(endpoint)
            rpc = (
                SocketRpcClient(host, port)
                if protocol == "socket"
                else HttpRpcClient(f"http://{host}:{port}")
            )
            wait_for_ready(rpc)
            return Sam3Client(rpc)

        from rpent.utils.config import get_repo_root
        from rpent.utils.daemon import ProcessDaemon, pick_free_port

        host, port = "127.0.0.1", pick_free_port()
        command = [
            sys.executable,
            str(get_repo_root() / "robots" / "libero" / "sam3_server.py"),
            "--transport",
            "http",
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watch",
        ]
        if self.cuda_device is not None:
            command.extend(["--cuda-device", str(self.cuda_device)])
        daemon_env = os.environ.copy()
        daemon_env["SAM3_CHECKPOINT_PATH"] = str(self.sam3_model_path)
        daemon = ProcessDaemon(
            name="sam3_server",
            cmd=command,
            env=daemon_env,
            log_path=str(self.runtime_dir / "sam3_server.log"),
        )
        daemon.start()
        self._daemons.append(daemon)
        rpc = HttpRpcClient(f"http://{host}:{port}")
        wait_for_ready(rpc, daemon=daemon)
        return Sam3Client(rpc)

    def reset(self) -> None:
        self._primitives.reset()
        self._step_index = 0

    def current_joint_positions_rad(self) -> list[float]:
        """Return the seven measured Panda joints from the persistent env."""

        if self.controller_kind != JOINT_CONTROLLER_KIND:
            return []
        value = np.asarray(
            self._primitives.env.raw_obs()["robot0_joint_pos"], dtype=np.float64
        ).reshape(-1)
        if value.shape != (JOINT_ARM_ACTION_DIM,) or not np.isfinite(value).all():
            raise RuntimeError("JOINT_POSITION backend returned invalid measured joints")
        return value.tolist()

    def p3_joint_controller_capability(self) -> dict[str, Any] | None:
        """Return a source/config-backed attestation, never a class-name claim."""

        if self.controller_kind != JOINT_CONTROLLER_KIND:
            return None
        attestation = self._primitives.env.controller_attestation()
        controller = attestation.get("controller_config", {})
        audited = bool(
            attestation == self._joint_server_attestation
            and controller.get("kind") == JOINT_CONTROLLER_KIND
            and controller.get("action_dim") == JOINT_ACTION_DIM
            and controller.get("arm_action_dim") == JOINT_ARM_ACTION_DIM
            and controller.get("gripper") == JOINT_GRIPPER_CONVENTION
            and attestation.get("server_source_sha256")
            == self._joint_controller_provenance["server_source_sha256"]
        )
        return {
            "audited": audited,
            "controller_kind": JOINT_CONTROLLER_KIND,
            "controller_sha256": self._joint_controller_sha256,
            "controller_config_sha256": attestation.get(
                "controller_config_sha256"
            ),
            "server_attestation_sha256": attestation.get("attestation_sha256"),
            "environment_instance_id": attestation.get(
                "environment_instance_id"
            ),
            "action_dim": controller.get("action_dim"),
            "arm_action_dim": controller.get("arm_action_dim"),
            "gripper_convention": controller.get("gripper"),
            "fresh_observation_handoff": True,
            "single_persistent_environment": True,
            "osc_controller_present": False,
        }

    def execute_joint_pose(self, **arguments: Any) -> ActionResult:
        """Execute a validated IK target through the sole joint controller."""

        return self._execute_joint_target(
            target_joints=arguments["joint_positions_rad"],
            gripper=arguments.get("gripper"),
            max_steps=int(arguments.get("max_steps", 120)),
            target_position=arguments.get("target_position_xyz_m"),
            target_quaternion=arguments.get("target_quaternion_xyzw"),
            position_tolerance_m=float(arguments.get("position_tolerance_m", 0.01)),
            orientation_tolerance_rad=float(
                arguments.get("orientation_tolerance_rad", 0.05)
            ),
            controller_sha256=str(arguments.get("controller_sha256", "")),
            operation=str(arguments.get("operation", "move_eef_pose_ik")),
            fixed_steps=False,
            ik_joint_limits_ok=arguments.get("ik_joint_limits_ok"),
            joint_lower_limits_rad=arguments.get("joint_lower_limits_rad"),
            joint_upper_limits_rad=arguments.get("joint_upper_limits_rad"),
        )

    def execute_joint_gripper(
        self, *, state: str, steps: int, controller_sha256: str
    ) -> ActionResult:
        """Hold measured arm joints while changing only the eighth action slot."""

        return self._execute_joint_target(
            target_joints=self.current_joint_positions_rad(),
            gripper=state,
            max_steps=int(steps),
            target_position=None,
            target_quaternion=None,
            position_tolerance_m=0.02,
            orientation_tolerance_rad=0.1,
            controller_sha256=controller_sha256,
            operation="set_gripper",
            fixed_steps=True,
            ik_joint_limits_ok=True,
            joint_lower_limits_rad=None,
            joint_upper_limits_rad=None,
        )

    def _execute_joint_target(
        self,
        *,
        target_joints: Any,
        gripper: Any,
        max_steps: int,
        target_position: Any,
        target_quaternion: Any,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        controller_sha256: str,
        operation: str,
        fixed_steps: bool,
        ik_joint_limits_ok: Any,
        joint_lower_limits_rad: Any,
        joint_upper_limits_rad: Any,
    ) -> ActionResult:
        if self.controller_kind != JOINT_CONTROLLER_KIND:
            raise UnsupportedSkillError("audited JOINT_POSITION backend is inactive")
        capability = self.p3_joint_controller_capability()
        if not capability or not capability["audited"]:
            raise RuntimeError("JOINT_POSITION controller attestation failed closed")
        if controller_sha256 != capability["controller_sha256"]:
            raise RuntimeError("JOINT_POSITION controller hash changed before action")
        joints = np.asarray(target_joints, dtype=np.float64).reshape(-1)
        if joints.shape != (JOINT_ARM_ACTION_DIM,) or not np.isfinite(joints).all():
            raise ValueError("joint target must contain seven finite radians")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if gripper is None:
            gripper = getattr(self, "_joint_last_gripper_command", "open")
        if gripper not in {"open", "closed"}:
            raise ValueError("gripper must be open, closed, or omitted")
        gripper_action = float(
            JOINT_GRIPPER_CONVENTION[
                "controller_open" if gripper == "open" else "controller_closed"
            ]
        )
        before = self._primitives.env.raw_obs()
        instance_id = str(capability["environment_instance_id"])
        simulator_steps = 0
        final_joint_error = float("inf")
        last_info: dict[str, Any] = {}
        for _ in range(max_steps):
            current = np.asarray(before["robot0_joint_pos"], dtype=np.float64).reshape(7)
            final_joint_error = float(np.linalg.norm(current - joints))
            if not fixed_steps and simulator_steps > 0 and final_joint_error < 0.01:
                break
            action = np.concatenate(
                [
                    (joints - current) * JOINT_CONTROL_FREQ_HZ,
                    np.asarray([gripper_action], dtype=np.float64),
                ]
            )
            if action.shape != (JOINT_ACTION_DIM,) or not np.isfinite(action).all():
                raise RuntimeError("constructed JOINT_POSITION action is invalid")
            _public_obs, _reward, _terminated, _truncated, last_info = (
                self._primitives.env.step(action)
            )
            simulator_steps += 1
            # ``step`` returns RPent's public observation shell, which omits
            # arm qpos by design.  Controller feedback is always re-read from
            # the same private persistent environment.
            before = self._primitives.env.raw_obs()
            if _terminated or _truncated:
                break
        final = self._primitives.env.raw_obs()
        final_joints = np.asarray(final["robot0_joint_pos"], dtype=np.float64).reshape(7)
        final_joint_error = float(np.linalg.norm(final_joints - joints))
        final_joint_limits_ok: bool | None = None
        if not fixed_steps:
            if joint_lower_limits_rad is not None and joint_upper_limits_rad is not None:
                lower = np.asarray(joint_lower_limits_rad, dtype=np.float64).reshape(-1)
                upper = np.asarray(joint_upper_limits_rad, dtype=np.float64).reshape(-1)
                if lower.shape != (7,) or upper.shape != (7,):
                    raise ValueError("IK provider joint-limit vectors must contain seven values")
                final_joint_limits_ok = bool(
                    np.isfinite(lower).all()
                    and np.isfinite(upper).all()
                    and np.all(lower <= final_joints)
                    and np.all(final_joints <= upper)
                )
            else:
                final_joint_limits_ok = False
        final_position = [float(v) for v in final["robot0_eef_pos"]]
        final_quaternion = [float(v) for v in final["robot0_eef_quat"]]
        position_error = None
        orientation_error = None
        pose_verified = True
        if target_position is not None and target_quaternion is not None:
            position_error = float(
                np.linalg.norm(
                    np.asarray(final_position, dtype=np.float64)
                    - np.asarray(target_position, dtype=np.float64)
                )
            )
            orientation_error = quaternion_angular_error_rad(
                final_quaternion, target_quaternion
            )
            pose_verified = bool(
                position_error <= position_tolerance_m
                and orientation_error <= orientation_tolerance_rad
            )
        execution_verified = bool(
            simulator_steps > 0
            and str(
                self._primitives.env.controller_attestation().get(
                    "environment_instance_id"
                )
            )
            == instance_id
            and (fixed_steps or final_joint_error < 0.02)
            and (fixed_steps or ik_joint_limits_ok is True)
            and (fixed_steps or final_joint_limits_ok is True)
            and pose_verified
        )
        if simulator_steps > 0:
            self._joint_last_gripper_command = str(gripper)
        self._step_index += 1
        return ActionResult(
            result={
                "name": operation,
                "controller_kind": JOINT_CONTROLLER_KIND,
                "controller_sha256": controller_sha256,
                "environment_instance_id": instance_id,
                "execution_verified": execution_verified,
                "joint_target_rad": joints.tolist(),
                "final_joint_positions_rad": final_joints.tolist(),
                "final_joint_error_rad_l2": final_joint_error,
                "final_eef_position_xyz_m": final_position,
                "final_eef_quaternion_xyzw": final_quaternion,
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "position_tolerance_m": position_tolerance_m,
                "orientation_tolerance_rad": orientation_tolerance_rad,
                "ik_joint_limits_ok": ik_joint_limits_ok,
                "final_joint_limits_ok": final_joint_limits_ok,
                "joint_lower_limits_rad": joint_lower_limits_rad,
                "joint_upper_limits_rad": joint_upper_limits_rad,
                "simulator_steps": simulator_steps,
                "gripper_command": gripper,
                "gripper_controller_value": gripper_action,
                "official_status": last_info.get("esbench_official_status"),
            },
            simulator_steps=simulator_steps,
            vla_chunks=0,
        )

    def execute(self, skill_id: str, arguments: dict[str, Any]) -> ActionResult:
        if getattr(self, "controller_kind", "OSC_POSE") == JOINT_CONTROLLER_KIND:
            if skill_id not in GROUNDED_JOINT_SKILL_IDS:
                raise UnsupportedSkillError(
                    "JOINT_POSITION with-skill episodes expose only selected "
                    "grounded JOINT actions; RPent OSC/VLA primitives are disabled"
                )
            kwargs = dict(arguments)
            handoffs = kwargs.pop(INTERNAL_GROUNDING_HANDOFFS_KEY, None)
            if kwargs:
                # Public arguments were already schema-checked by EpisodeSession;
                # the executor consumes their framework-bound grounding cards only.
                expected = {"object"} if "object" in kwargs else {"container"}
                if set(kwargs) != expected:
                    raise InvalidSkillArgumentsError(
                        "grounded JOINT action received unexpected semantic fields"
                    )
            if not isinstance(handoffs, list):
                raise InvalidSkillArgumentsError(
                    "grounded JOINT action requires framework-owned current-step "
                    "SAM3 geometry"
                )
            if self._grounded_joint_ik_provider is None:
                raise RuntimeError(
                    "selected grounded JOINT action has no frozen PyRoKi provider"
                )
            try:
                result = execute_grounded_joint_action(
                    backend=self,
                    ik_provider=self._grounded_joint_ik_provider,
                    handoffs=handoffs,
                    skill_id=skill_id,
                    audit_dir=(
                        self.runtime_dir
                        / "private-grounded-joint"
                        / f"action-{self._grounded_joint_action_index + 1:04d}"
                    ),
                )
            except ValueError as exc:
                raise InvalidSkillArgumentsError(str(exc)) from exc
            self._grounded_joint_action_index += 1
            return result
        kwargs = dict(arguments)
        if skill_id == "move-to":
            kwargs["gripper"] = _gripper_value(kwargs.get("gripper"), -1.0)
            result = self._primitives.move_to(**kwargs)
        elif skill_id == "move-pose":
            kwargs["gripper"] = _gripper_value(kwargs.get("gripper"), -1.0)
            result = self._primitives.move_pose(**kwargs)
        elif skill_id == "rotate-wrist":
            kwargs["gripper"] = _gripper_value(kwargs.get("gripper"), 1.0)
            result = self._primitives.rotate_wrist(**kwargs)
        elif skill_id == "rotate-pitch":
            kwargs["gripper"] = _gripper_value(kwargs.get("gripper"), 1.0)
            result = self._primitives.rotate_pitch(**kwargs)
        elif skill_id == "set-gripper":
            kwargs["gripper"] = _gripper_value(kwargs.get("gripper"), -1.0)
            result = self._primitives.set_gripper(**kwargs)
        elif skill_id == "release":
            result = self._primitives.release(**kwargs)
        elif skill_id == "pi0-pick":
            client = self._activate_vla_profile("rpent-generalist")
            result = annotate_pi0_pick_result(self._primitives.pi0_pick(**kwargs))
            result.setdefault("steps_used", client.action_steps)
            result["harness_vla_abstraction"] = "vla_act"
            result["concrete_skill"] = "pi0_pick"
        elif skill_id == "pi0-doubled":
            client = self._activate_vla_profile("rpent-generalist")
            result = self._primitives.pi0_doubled(**kwargs)
            result = dict(result)
            result.setdefault("steps_used", client.action_steps)
            result["harness_vla_abstraction"] = "vla_act"
            result["concrete_skill"] = "pi0_doubled"
        elif is_semantic_vla_skill(skill_id):
            try:
                call = build_semantic_vla_call(skill_id, kwargs)
            except (TypeError, ValueError) as exc:
                # Manifest-backed prompt rendering is pure argument validation:
                # no simulator or VLA call has happened yet.  Preserve that
                # boundary so an executor mistake is a protocol rejection, not
                # an infrastructure-breaking runtime action error.
                raise InvalidSkillArgumentsError(str(exc)) from exc
            client = self._activate_vla_profile(call["model_profile"])
            method = getattr(self._primitives, call["concrete_method"])
            result = annotate_semantic_vla_result(
                call, method(**call["backend_arguments"])
            )
            result.setdefault("steps_used", client.action_steps)
            result["vla_route"] = {
                "policy_profile": client.profile_id,
                "endpoint_mode": client.mode,
                "forward_calls": client.forward_calls,
                "action_steps": client.action_steps,
                "last_native_action_shape": client.last_action_shape,
                "identity": client.identity,
            }
        elif skill_id in GROUNDED_OSC_SKILL_IDS:
            handoffs = kwargs.pop(INTERNAL_GROUNDING_HANDOFFS_KEY, None)
            if not isinstance(handoffs, list):
                raise InvalidSkillArgumentsError(
                    "grounded OSC action requires framework-owned object and "
                    "destination grounding from the current observation"
                )
            try:
                result = execute_flat_package_to_woven_basket(
                    primitives=self._primitives,
                    handoffs=handoffs,
                    skill_id=skill_id,
                    audit_dir=(
                        self.runtime_dir
                        / "private-grounded-osc"
                        / f"action-{self._step_index + 1:04d}"
                    ),
                )
            except ValueError as exc:
                raise InvalidSkillArgumentsError(str(exc)) from exc
        elif skill_id in {"navigate-to", "move-base"}:
            raise UnsupportedSkillError(
                f"{skill_id} is not available on fixed-base LIBERO-Pro"
            )
        else:
            raise UnsupportedSkillError(f"RPent LIBERO adapter does not implement {skill_id}")
        self._step_index += 1
        simulator_steps = int(result.get("steps_used", 0))
        chunks = int(result.get("chunks_used", 0))
        if not simulator_steps and chunks:
            simulator_steps = chunks * 5
        return ActionResult(
            result=dict(result), simulator_steps=simulator_steps, vla_chunks=chunks
        )

    def _activate_vla_profile(self, profile_id: str) -> _ProfiledVLAClient:
        try:
            client = self._vla_clients[profile_id]
        except KeyError as exc:
            raise RuntimeError(
                f"semantic skill requested unconfigured VLA policy profile {profile_id!r}"
            ) from exc
        client.reset_accounting()
        self._primitives.model = client
        return client

    def perceive(
        self,
        skill_id: str,
        arguments: dict[str, Any],
        observation: dict[str, Any],
        destination: Path,
    ) -> PerceptionResult:
        localization = build_sam3_localization_arguments(skill_id, arguments)
        if localization is not None:
            return PerceptionResult(
                result=run_sam3_localization(
                    client=self._sam3_client,
                    observation=observation,
                    arguments=localization,
                    destination=destination,
                )
            )
        if skill_id in {"sam3-segment-point", "sam3-bind-point-reference-v1"}:
            return PerceptionResult(
                result=run_sam3_point_segmentation(
                    client=self._sam3_client,
                    observation=observation,
                    arguments=arguments,
                    destination=destination,
                )
            )
        if skill_id == "contact-graspnet-propose-grasps":
            if self._contact_graspnet_client is None:
                raise UnsupportedSkillError(
                    "Contact-GraspNet provider was not enabled for this episode"
                )
            return PerceptionResult(
                result=run_contact_graspnet_proposals(
                    client=self._contact_graspnet_client,
                    observation=observation,
                    arguments=arguments,
                    destination=destination,
                )
            )
        if skill_id == "molmo2-point-localize-object":
            if self._molmo2_client is None:
                raise UnsupportedSkillError(
                    "Molmo2 provider was not enabled for this episode"
                )
            return PerceptionResult(
                result=run_molmo2_point_localization(
                    client=self._molmo2_client,
                    observation=observation,
                    arguments=arguments,
                    destination=destination,
                )
            )
        if skill_id == "molmo2-sam3-bind-object-reference-v1":
            if self._molmo2_client is None:
                raise UnsupportedSkillError(
                    "Molmo2 provider was not enabled for this episode"
                )
            return PerceptionResult(
                result=run_molmo2_sam3_reference_binding(
                    molmo2_client=self._molmo2_client,
                    sam3_client=self._sam3_client,
                    observation=observation,
                    arguments=arguments,
                    destination=destination,
                )
            )
        if skill_id == "visual-difference-describe-change":
            if self._visual_difference_client is None:
                raise UnsupportedSkillError(
                    "visual-difference provider was not enabled for this episode"
                )
            return PerceptionResult(
                result=run_visual_difference_description(
                    client=self._visual_difference_client,
                    observation=observation,
                    arguments=arguments,
                    destination=destination,
                )
            )
        raise UnsupportedSkillError(
            f"RPent LIBERO adapter does not implement perception skill {skill_id}"
        )

    def capture(self, destination: Path, step_index: int) -> CapturedObservation:
        from robots.libero import tools as libero_tools

        # `dump_state` contains backend-private state beyond the public
        # observation contract. Keep it outside the episode tree that is
        # mounted read-only into the agent sandbox, then copy only approved
        # RGB-D/calibration artifacts into the observation bundle.
        raw_root = self.runtime_dir / "private-captures" / f"{step_index:04d}"
        raw_root.mkdir(parents=True, exist_ok=True)
        blob = libero_tools.dump_state(
            self._primitives, str(raw_root), step_idx=step_index, log=None
        )
        cameras_dir = destination / "cameras"
        geometry_dir = destination / "geometry"
        calibration_dir = destination / "calibration"
        artifacts: dict[str, Any] = {"cameras": {}, "calibration": {}}
        for camera in ("agentview", "wrist"):
            low_rgb = cameras_dir / f"{camera}.rgb.png"
            high_rgb = cameras_dir / f"{camera}.high.rgb.png"
            low_depth = geometry_dir / f"{camera}.depth.npy"
            low_world = geometry_dir / f"{camera}.world.npy"
            high_world = geometry_dir / f"{camera}.high.world.npy"
            calibration = calibration_dir / f"{camera}.json"
            if camera == "agentview":
                source_low_rgb = libero_tools.artifact_path(
                    raw_root, "image", step=step_index, camera="agentview", resolution="low"
                )
                source_high_rgb = libero_tools.artifact_path(
                    raw_root, "image", step=step_index, camera="agentview", resolution="high"
                )
                source_calibration = libero_tools.artifact_path(
                    raw_root, "metadata", camera="agentview", resolution="low"
                )
            else:
                source_low_rgb = libero_tools.artifact_path(
                    raw_root, "image", step=step_index, camera="wrist", resolution="low"
                )
                source_high_rgb = libero_tools.artifact_path(
                    raw_root, "image", step=step_index, camera="wrist", resolution="high"
                )
                source_calibration = libero_tools.artifact_path(
                    raw_root,
                    "metadata",
                    step=step_index,
                    camera="wrist",
                    resolution="low",
                )
            source_depth = libero_tools.artifact_path(
                raw_root, "depth", step=step_index, camera=camera, resolution="low"
            )
            source_world = libero_tools.artifact_path(
                raw_root, "world", step=step_index, camera=camera, resolution="low"
            )
            source_high_world = libero_tools.artifact_path(
                raw_root, "world", step=step_index, camera=camera, resolution="high"
            )
            _copy_if_present(source_low_rgb, low_rgb)
            _copy_if_present(source_high_rgb, high_rgb)
            _copy_if_present(source_depth, low_depth)
            _copy_if_present(source_world, low_world)
            _copy_if_present(source_high_world, high_world)
            _copy_if_present(source_calibration, calibration)
            camera_entry: dict[str, Any] = {}
            if low_rgb.exists():
                camera_entry["rgb"] = {
                    "path": f"./cameras/{low_rgb.name}",
                    "resolution": _resolution(low_rgb),
                }
            if high_rgb.exists():
                camera_entry["high_rgb"] = {
                    "path": f"./cameras/{high_rgb.name}",
                    "resolution": _resolution(high_rgb),
                }
            if low_depth.exists():
                camera_entry["depth"] = {
                    "path": f"./geometry/{low_depth.name}",
                    "resolution": _resolution(low_rgb),
                    "unit": "m",
                }
            if low_world.exists():
                camera_entry["world"] = {
                    "path": f"./geometry/{low_world.name}",
                    "resolution": _resolution(low_rgb),
                    "unit": "m",
                }
            if high_world.exists():
                camera_entry["high_world"] = {
                    "path": f"./geometry/{high_world.name}",
                    "resolution": _resolution(high_rgb),
                    "unit": "m",
                }
            artifacts["cameras"][camera] = camera_entry
            if calibration.exists():
                artifacts["calibration"][camera] = {
                    "path": f"./calibration/{calibration.name}"
                }

        state = blob["state"]
        qpos = [float(value) for value in state["robot0_gripper_qpos"]]
        opening = float(sum(abs(value) for value in qpos))
        env = self._primitives.env
        return CapturedObservation(
            task={
                "environment": self.environment_id,
                "suite": self.suite,
                "task_id": self.task,
                "seed": self.seed,
                "language": blob.get("task_language"),
            },
            status={
                "success": bool(env.episode_terminated),
                "terminated": bool(env.episode_terminated),
                "truncated": bool(env.episode_truncated),
                "reward": None,
            },
            robot={
                "end_effector": {
                    "position_m": state["robot0_eef_pos"],
                    "quaternion": state["robot0_eef_quat"],
                    "quaternion_convention": "xyzw",
                    "frame": "LIBERO world",
                },
                "gripper": {
                    "opening_m": opening,
                    "finger_qpos": qpos,
                    "command_convention": "skill: open/closed; RPent internal: -1/+1",
                },
            },
            object_names=list(state.get("object_names", [])),
            artifacts=artifacts,
            backend={
                "kind": (
                    "rpent-libero-joint-position"
                    if self.controller_kind == JOINT_CONTROLLER_KIND
                    else "rpent-libero"
                ),
                "rpent_commit": self.rpent_commit,
                "controller_kind": self.controller_kind,
                "controller_sha256": getattr(
                    self, "_joint_controller_sha256", None
                ),
                "environment_instance_id": getattr(
                    getattr(self, "_primitives", None), "env", None
                ).environment_instance_id
                if self.controller_kind == JOINT_CONTROLLER_KIND
                else None,
                "raw_artifacts_agent_exposed": False,
                "challenge_cell_id": self.challenge["cell_id"]
                if self.challenge
                else None,
            },
        )

    def close(self) -> None:
        for daemon in reversed(self._daemons):
            daemon.stop()
        self._daemons.clear()

    def provenance(self) -> dict[str, Any]:
        checkpoint: dict[str, Any] | None = None
        if self.vla_model_path:
            path = Path(self.vla_model_path).expanduser()
            sidecar = Path(str(path) + ".sha256")
            checkpoint = {
                "path": str(path.resolve()),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sidecar.read_text(encoding="utf-8").strip().split()[0]
                if sidecar.is_file()
                else None,
                "sha256_note": "Provide <checkpoint>.sha256 to lock large weights without re-hashing every run.",
            }
        vla_profile_provenance: dict[str, Any] = {}
        for profile_id in self.required_vla_profiles:
            spec = self.vla_profile_specs.get(profile_id, {})
            client = self._vla_clients.get(profile_id)
            profile_checkpoint: dict[str, Any] | None = None
            if spec.get("checkpoint_path"):
                profile_path = Path(str(spec["checkpoint_path"])).expanduser()
                profile_sidecar = Path(str(profile_path) + ".sha256")
                profile_checkpoint = {
                    "path": str(profile_path.resolve()),
                    "exists": profile_path.exists(),
                    "size_bytes": profile_path.stat().st_size
                    if profile_path.is_file()
                    else None,
                    "sha256": profile_sidecar.read_text(encoding="utf-8")
                    .strip()
                    .split()[0]
                    if profile_sidecar.is_file()
                    else None,
                }
            vla_profile_provenance[profile_id] = {
                "mode": client.mode if client else "unconfigured",
                "checkpoint": profile_checkpoint,
                "endpoint_identity": client.identity if client else None,
                "profile_identity_required": bool(
                    spec.get(
                        "require_profile_identity",
                        profile_id != "rpent-generalist",
                    )
                ),
                "python_executable": spec.get("python_executable"),
                "randomness_controlled_by_harness": bool(
                    client
                    and client.mode in {"local_process", "local_profile_process"}
                ),
            }
        sam3_checkpoint: dict[str, Any] | None = None
        if self.sam3_model_path:
            sam3_path = Path(self.sam3_model_path).expanduser()
            sam3_sidecar = Path(str(sam3_path) + ".sha256")
            sam3_checkpoint = {
                "path": str(sam3_path.resolve()),
                "exists": sam3_path.is_file(),
                "size_bytes": sam3_path.stat().st_size
                if sam3_path.is_file()
                else None,
                "sha256": sam3_sidecar.read_text(encoding="utf-8")
                .strip()
                .split()[0]
                if sam3_sidecar.is_file()
                else None,
                "sha256_note": "Provide <checkpoint>.sha256 to lock large weights without re-hashing every run.",
            }
        locally_seeded_vla = self.vla_mode in {
            "local_process",
            "local_profile_process",
        }
        provenance = {
            "kind": (
                "rpent-libero-joint-position"
                if self.controller_kind == JOINT_CONTROLLER_KIND
                else "rpent-libero"
            ),
            "environment": self.environment_id,
            "suite": self.suite,
            "task": self.task,
            "seed": self.seed,
            "rpent_repo": str(self.rpent_repo),
            "rpent_commit": self.rpent_commit,
            "controller": (
                {
                    **(self.p3_joint_controller_capability() or {}),
                    "provenance": getattr(
                        self, "_joint_controller_provenance", None
                    ),
                    "server_attestation": getattr(
                        self, "_joint_server_attestation", None
                    ),
                }
                if self.controller_kind == JOINT_CONTROLLER_KIND
                else {"controller_kind": "OSC_POSE"}
            ),
            "grounded_joint_pyroki": dict(
                self._grounded_joint_pyroki_config
            ),
            "vla_checkpoint": checkpoint,
            "vla_profiles": {
                "required": list(self.required_vla_profiles),
                "routes": vla_profile_provenance,
                "selection_rule": "semantic manifest model_profile selects the exact client before every action call",
            },
            "vla_randomness": {
                "noise_level": 0.5 if self.vla_mode == "local_process" else None,
                "seed": self.vla_seed,
                "seed_source": self.vla_seed_source,
                "seed_derivation_inputs": {
                    "suite": self.suite,
                    "task": self.task,
                    "environment_seed": self.seed,
                }
                if self.vla_seed_source == VLA_SEED_NAMESPACE
                else {
                    "pair_id": self.challenge["pair_id"],
                    "environment_seed": self.seed,
                }
                if self.vla_seed_source == CHALLENGE_PAIR_VLA_SEED_NAMESPACE
                and self.challenge
                else None,
                "enforced": locally_seeded_vla,
                "scope": self.vla_mode,
                "python_random": True if locally_seeded_vla else None,
                "numpy": True if locally_seeded_vla else None,
                "torch_cpu": True if locally_seeded_vla else None,
                "torch_cuda_manual_seed_all": True
                if locally_seeded_vla
                else None,
                "cudnn_deterministic": True
                if locally_seeded_vla
                else None,
                "cudnn_benchmark": False
                if locally_seeded_vla
                else None,
                "deterministic_algorithms_warn_only": True
                if locally_seeded_vla
                else None,
                "pythonhashseed": self.vla_seed
                if locally_seeded_vla
                else None,
                "cublas_workspace_config": ":4096:8"
                if locally_seeded_vla
                else None,
                "note": (
                    "The local seed fixes the stochastic VLA stream; warn-only "
                    "deterministic algorithms do not guarantee bitwise identity "
                    "across hardware/software stacks."
                    if locally_seeded_vla
                    else "Randomness inside an external VLA endpoint is not controlled by this harness."
                    if self.vla_mode == "external_endpoint"
                    else "No VLA skill was enabled for this condition."
                ),
            },
            "sam3": {
                "mode": self.sam3_mode,
                "started": self.sam3_mode != "disabled",
                "checkpoint": sam3_checkpoint,
                "agent_access": self.sam3_agent_access,
            },
            "sam3_started": self.sam3_mode != "disabled",
            "contact_graspnet": {
                "mode": self.contact_graspnet_mode,
                "started_by_harness": False,
                "endpoint": self.contact_graspnet_endpoint,
                "declared_identity": self.contact_graspnet_identity or None,
                "agent_access": (
                    "selected read-only perception skill only"
                    if self.contact_graspnet_mode != "disabled"
                    else "not exposed"
                ),
                "note": (
                    "The development adapter uses an external point-cloud service; "
                    "formal promotion requires a locked server commit and checkpoint."
                ),
            },
            "molmo2_point": {
                "mode": self.molmo2_mode,
                "started_by_harness": False,
                "endpoint": self.molmo2_endpoint,
                "declared_identity": self.molmo2_identity or None,
                "agent_access": (
                    "selected read-only perception skill only"
                    if self.molmo2_mode != "disabled"
                    else "not exposed"
                ),
                "note": (
                    "The provider receives only the selected current public RGB; "
                    "formal promotion requires a locked model revision and server stack."
                ),
            },
            "visual_difference": {
                "mode": self.visual_difference_mode,
                "started_by_harness": False,
                "endpoint": self.visual_difference_endpoint,
                "declared_identity": self.visual_difference_identity or None,
                "agent_access": (
                    "selected read-only perception skill only"
                    if self.visual_difference_mode != "disabled"
                    else "not exposed"
                ),
                "note": (
                    "The provider receives only two immutable public frames and text; "
                    "its completion judgment is never official task truth."
                ),
            },
            "memory_injected": False,
            "mcp_primitives_exposed": False,
        }
        if self.challenge:
            provenance["challenge"] = {
                "schema_version": 1,
                "cell_id": self.challenge["cell_id"],
                "pair_id": self.challenge["pair_id"],
                "bddl_file_path": self.challenge["bddl_file_path"],
                "bddl_sha256": self.challenge["bddl_sha256"],
                "init_states_file_path": self.challenge["init_states_file_path"],
                "init_states_sha256": self.challenge["init_states_sha256"],
                "spec": self.challenge["spec"],
                "audit_path": str(
                    self.runtime_dir / "private-challenge-events.jsonl"
                ),
                "agent_exposed": False,
            }
        if self.liberopro_task_snapshot:
            provenance["liberopro_task_snapshot"] = dict(
                self.liberopro_task_snapshot
            )
        return provenance


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

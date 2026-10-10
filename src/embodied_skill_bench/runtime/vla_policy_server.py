"""Profile-aware VLA RPC server for externally packaged policy providers.

The benchmark's RPent adapter uses one small, model-neutral wire contract:
``instruction`` + PNG camera blocks + a batched robot state in, and a finite
``[B, chunk, 7]`` action array out.  This module keeps model-specific loading
and checkpoint migrations on the server side so skill manifests only select a
named policy profile.

SmolVLA and OpenVLA-OFT are enabled only through revision-pinned, profile-
identified adapters.  Other downloaded candidates remain fail-closed until
their own D0 input/output compatibility checks pass.
"""

from __future__ import annotations

import argparse
import base64
import io
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

SMOLVLA_PROFILE = "smolvla-libero-450m"
SMOLVLA_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
SMOLVLM_BASE_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
XVLA_PROFILE = "xvla-libero-0.9b"
XVLA_REVISION = "12e8783e996944f5c97e490d37d4c145484ed70a"
XVLA_TOKENIZER_REVISION = "cb48c1365bd826bd521f650dc2e0940aee54720c"
OPENVLA_OFT_PROFILE = "openvla-oft-all4"
OPENVLA_OFT_REVISION = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"
OPENVLA_OFT_CODE_REVISION = "2c9655244bce7c61477f689d59b6f0dbe76c4f40"
OPENVLA_OFT_TRANSFORMERS_REVISION = (
    "bc339d9ad707454c0c115970db43c260067c61ab"
)


def _decode_png(block: dict[str, Any], *, name: str) -> np.ndarray:
    """Decode one strict RGB uint8 PNG block from the RPent VLA wire."""

    if not isinstance(block, dict):
        raise TypeError(f"images.{name} must be an object")
    if str(block.get("format", "png")).lower() != "png":
        raise ValueError(f"images.{name}.format must be 'png'")
    encoded = block.get("data")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"images.{name}.data must be non-empty base64")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError(f"images.{name}.data is not valid base64") from exc

    from PIL import Image

    with Image.open(io.BytesIO(raw)) as decoded:
        image = np.asarray(decoded.convert("RGB"))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"images.{name} must decode to HxWx3 RGB; got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"images.{name} must decode to uint8; got {image.dtype}")
    return np.ascontiguousarray(image)


def _validate_wire_inputs(
    instruction: Any, images: Any, state: Any
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Validate the exact SmolVLA LIBERO observation boundary.

    RPent/RLinf has already rotated both camera images by 180 degrees and has
    already flattened proprioception as EEF position (3), axis-angle (3), and
    gripper qpos (2).  The saved LeRobot policy preprocessor therefore starts
    at its rename/batch/normalization stages; applying LiberoProcessorStep a
    second time would rotate the images twice.
    """

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    if not isinstance(images, dict):
        raise TypeError("images must be an object")
    if "main" not in images or "wrist" not in images:
        raise ValueError("SmolVLA LIBERO requires images.main and images.wrist")
    main = _decode_png(images["main"], name="main")
    wrist = _decode_png(images["wrist"], name="wrist")
    states = np.asarray(state, dtype=np.float32)
    if states.shape != (1, 8):
        raise ValueError(
            "SmolVLA LIBERO requires state shape [1,8] "
            "(EEF xyz + axis-angle + two gripper qpos); "
            f"got {list(states.shape)}"
        )
    if not np.isfinite(states).all():
        raise ValueError("state contains NaN or infinite values")
    return instruction.strip(), main, wrist, states[0]


def _smolvla_config_from_snapshot(
    snapshot: Path, *, base_model_path: Path | None = None
):
    """Load the checkpoint with its explicit, audited legacy migration.

    The frozen snapshot's raw ``config.json`` declares a six-dimensional
    state, while both its saved normalizer statistics and the official LIBERO
    environment processor use eight dimensions.  D0 proved that changing only
    that stale feature declaration to eight permits a strict weight load and
    finite 50x7 inference.  The source snapshot is never rewritten.
    """

    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    payload = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    for legacy_key in ("type", "use_peft", "pretrained_path"):
        payload.pop(legacy_key, None)

    def feature(value: dict[str, Any]) -> PolicyFeature:
        return PolicyFeature(
            type=FeatureType[str(value["type"])],
            shape=tuple(int(item) for item in value["shape"]),
        )

    payload["input_features"] = {
        key: feature(value) for key, value in payload["input_features"].items()
    }
    payload["input_features"]["observation.state"] = PolicyFeature(
        type=FeatureType.STATE, shape=(8,)
    )
    payload["output_features"] = {
        key: feature(value) for key, value in payload["output_features"].items()
    }
    payload["device"] = "cuda"
    if base_model_path is not None:
        payload["vlm_model_name"] = str(base_model_path)
    return SmolVLAConfig(**payload)


def _smolvla_preprocessor_overrides(base_model_path: Path) -> dict[str, Any]:
    """Pin the saved tokenizer step to the same audited local VLM snapshot."""

    return {
        "tokenizer_processor": {
            "tokenizer_name": str(base_model_path.expanduser().resolve())
        }
    }


class SmolVLALiberoProvider:
    """Strict adapter for ``lerobot/smolvla_libero`` at the frozen revision."""

    profile_id = SMOLVLA_PROFILE
    revision = SMOLVLA_REVISION

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        base_model_path: str | Path,
        seed: int = 0,
    ):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        snapshot = Path(checkpoint_path).expanduser().resolve()
        required = {
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        }
        missing = sorted(name for name in required if not (snapshot / name).is_file())
        if missing:
            raise FileNotFoundError(
                f"SmolVLA snapshot {snapshot} is missing required files: {missing}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("SmolVLA provider requires a CUDA device")
        base_model = Path(base_model_path).expanduser().resolve()
        base_required = {
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer.json",
        }
        base_missing = sorted(
            name for name in base_required if not (base_model / name).is_file()
        )
        if base_missing:
            raise FileNotFoundError(
                f"SmolVLM base snapshot {base_model} is missing: {base_missing}"
            )

        self.checkpoint_path = snapshot
        self.base_model_path = base_model
        self.seed = int(seed)
        self.config = _smolvla_config_from_snapshot(
            snapshot, base_model_path=base_model
        )
        self.policy = SmolVLAPolicy.from_pretrained(
            snapshot, config=self.config, strict=True
        ).eval().cuda()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            pretrained_path=str(snapshot),
            preprocessor_overrides=_smolvla_preprocessor_overrides(base_model),
        )
        self.forward_calls = 0

    def predict(
        self,
        instruction: str,
        images: dict[str, Any],
        state: list,
        mode: str = "eval",
    ) -> dict[str, Any]:
        import torch
        from lerobot.policies.utils import prepare_observation_for_inference

        if mode != "eval":
            raise ValueError(
                f"SmolVLA provider supports mode='eval' only; got {mode!r}"
            )
        instruction, main, wrist, state_vector = _validate_wire_inputs(
            instruction, images, state
        )
        observation = {
            "observation.images.image": main,
            "observation.images.image2": wrist,
            "observation.state": state_vector,
        }
        batch = prepare_observation_for_inference(
            observation,
            torch.device("cuda"),
            task=instruction,
        )
        batch = self.preprocessor(batch)
        with torch.inference_mode():
            raw_actions = self.policy.predict_action_chunk(batch)
            processed = [
                self.postprocessor(raw_actions[:, index, :])
                for index in range(raw_actions.shape[1])
            ]
            actions = torch.stack(processed, dim=1).detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (1, 50, 7):
            raise RuntimeError(
                f"SmolVLA must return [1,50,7]; received {list(actions.shape)}"
            )
        if not np.isfinite(actions).all():
            raise RuntimeError("SmolVLA returned NaN or infinite actions")
        self.forward_calls += 1
        return {
            "actions": actions.tolist(),
            "shape": [1, 50, 7],
            "dtype": "float32",
            "policy_profile": self.profile_id,
            "checkpoint_revision": self.revision,
        }

    def healthz(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "policy_profile": self.profile_id,
            "checkpoint_revision": self.revision,
            "checkpoint_path": str(self.checkpoint_path),
            "base_model_path": str(self.base_model_path),
            "base_model_revision": SMOLVLM_BASE_REVISION,
            "native_action_shape": [50, 7],
            "state_shape": [8],
            "camera_keys": ["main", "wrist"],
            "forward_calls": self.forward_calls,
            "seed": self.seed,
            "d0_status": "strict-load-and-finite-forward-passed",
        }


def _xvla_libero_observation(
    main: np.ndarray, wrist: np.ndarray, state_vector: np.ndarray
) -> dict[str, np.ndarray]:
    """Reconstruct official X-VLA LIBERO inputs from the RPent wire.

    RPent has already applied the 180-degree policy transform to both views.
    X-VLA's environment transform rotates only the agent view, so the main
    image is ready while the wrist image must be restored. X-VLA proprio is
    EEF xyz + rotation-6D + one zero, padded to 20 dimensions.
    """

    from scipy.spatial.transform import Rotation

    if main.shape != wrist.shape or main.ndim != 3 or main.shape[-1] != 3:
        raise ValueError("X-VLA requires matching HxWx3 main and wrist images")
    state = np.asarray(state_vector, dtype=np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("X-VLA source state must be one finite 8-D LIBERO vector")
    matrix = Rotation.from_rotvec(state[3:6].astype(np.float64)).as_matrix()
    rotation_6d = np.concatenate([matrix[:, 0], matrix[:, 1]]).astype(np.float32)
    proprio = np.zeros(20, dtype=np.float32)
    proprio[:3] = state[:3]
    proprio[3:9] = rotation_6d
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def normalize(image: np.ndarray) -> np.ndarray:
        return (np.asarray(image, dtype=np.float32) / 255.0 - mean) / std

    return {
        "observation.images.image": normalize(main),
        "observation.images.image2": normalize(wrist[::-1, ::-1]),
        "observation.state": proprio,
    }


def _validate_xvla_relative_actions(actions: Any) -> np.ndarray:
    """Reject rather than clip outputs outside RPent's relative OSC contract."""

    result = np.asarray(actions, dtype=np.float32)
    if (
        result.ndim != 3
        or result.shape[0] != 1
        or result.shape[2] != 7
        or not 30 <= result.shape[1] <= 120
    ):
        raise RuntimeError(
            f"X-VLA must return [1,N,7] with 30<=N<=120 after the "
            f"controller-safe resampling; got "
            f"{list(result.shape)}"
        )
    if not np.isfinite(result).all():
        raise RuntimeError("X-VLA returned NaN or infinite actions")
    minimum = float(result.min())
    maximum = float(result.max())
    if minimum < -1.0 or maximum > 1.0:
        raise RuntimeError(
            "X-VLA real-observation action range is incompatible with RPent "
            f"relative OSC: min={minimum:.6f}, max={maximum:.6f}; no clipping "
            "or controller reinterpretation is permitted"
        )
    return result


def _xvla_absolute_to_rpent_relative(
    absolute_actions: Any, state_vector: Any
) -> np.ndarray:
    """Convert X-VLA absolute EE targets to RPent's normalized OSC deltas.

    The first target is relative to the measured current EEF pose; later
    targets are relative to the preceding target in the same predicted chunk.
    This is a deterministic controller adapter, not clipping. Any resulting
    delta outside the controller's declared input range is rejected later.
    """

    from scipy.spatial.transform import Rotation

    absolute = np.asarray(absolute_actions, dtype=np.float32)
    state = np.asarray(state_vector, dtype=np.float32)
    if absolute.shape != (1, 30, 7):
        raise RuntimeError(
            f"X-VLA absolute actions must be [1,30,7]; got {list(absolute.shape)}"
        )
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("X-VLA controller adapter requires one finite 8-D state")
    if not np.isfinite(absolute).all():
        raise RuntimeError("X-VLA absolute actions contain NaN or infinity")

    relative_steps: list[np.ndarray] = []
    previous_position = state[:3].astype(np.float64)
    previous_rotation = Rotation.from_rotvec(state[3:6].astype(np.float64))
    for index, target in enumerate(absolute[0]):
        target_position = target[:3].astype(np.float64)
        target_rotation = Rotation.from_rotvec(target[3:6].astype(np.float64))
        position_delta = target_position - previous_position
        rotation_error = target_rotation * previous_rotation.inv()
        rotation_delta = rotation_error.as_rotvec()
        subdivisions = max(
            1,
            int(np.ceil(np.max(np.abs(position_delta / 0.05)))),
            int(np.ceil(np.max(np.abs(rotation_delta / 0.5)))),
        )
        step = np.zeros(7, dtype=np.float32)
        step[:3] = position_delta / (0.05 * subdivisions)
        step[3:6] = rotation_delta / (0.5 * subdivisions)
        step[6] = target[6]
        relative_steps.extend(step.copy() for _ in range(subdivisions))
        previous_position = target_position
        previous_rotation = target_rotation
    if len(relative_steps) > 120:
        raise RuntimeError(
            "X-VLA absolute trajectory requires more than 120 controller-safe "
            f"substeps ({len(relative_steps)}); refusing execution"
        )
    return np.stack(relative_steps, axis=0)[np.newaxis, ...]


class XVLALiberoProvider:
    """Pinned X-VLA route with official EE6D conversion and strict range gate."""

    profile_id = XVLA_PROFILE
    revision = XVLA_REVISION

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        tokenizer_path: str | Path,
        seed: int = 0,
    ):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.policies.xvla.processor_xvla import (
            XVLARotation6DToAxisAngleProcessorStep,
        )

        snapshot = Path(checkpoint_path).expanduser().resolve()
        tokenizer = Path(tokenizer_path).expanduser().resolve()
        required = {
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        }
        missing = sorted(name for name in required if not (snapshot / name).is_file())
        tokenizer_required = {"config.json", "tokenizer.json", "merges.txt", "vocab.json"}
        tokenizer_missing = sorted(
            name for name in tokenizer_required if not (tokenizer / name).is_file()
        )
        if missing:
            raise FileNotFoundError(
                f"X-VLA snapshot {snapshot} is missing required files: {missing}"
            )
        if tokenizer_missing:
            raise FileNotFoundError(
                f"X-VLA tokenizer snapshot {tokenizer} is missing: {tokenizer_missing}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("X-VLA provider requires a CUDA device")

        self.checkpoint_path = snapshot
        self.tokenizer_path = tokenizer
        self.seed = int(seed)
        self.policy = XVLAPolicy.from_pretrained(
            snapshot, strict=True, local_files_only=True
        ).eval().cuda()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(snapshot),
            preprocessor_overrides={
                "tokenizer_processor": {"tokenizer_name": str(tokenizer)},
                "device_processor": {"device": "cuda"},
            },
        )
        self.rotation_postprocessor = XVLARotation6DToAxisAngleProcessorStep()
        self.forward_calls = 0
        self.last_action_range: dict[str, float] | None = None
        self.last_absolute_target_range: dict[str, float] | None = None
        self.last_absolute_target_summary: dict[str, Any] | None = None

    def predict(
        self,
        instruction: str,
        images: dict[str, Any],
        state: list,
        mode: str = "eval",
    ) -> dict[str, Any]:
        import torch
        from lerobot.policies.utils import prepare_observation_for_inference

        if mode != "eval":
            raise ValueError(f"X-VLA provider supports mode='eval' only; got {mode!r}")
        instruction, main, wrist, state_vector = _validate_wire_inputs(
            instruction, images, state
        )
        observation = _xvla_libero_observation(main, wrist, state_vector)
        batch = prepare_observation_for_inference(
            observation, torch.device("cuda"), task=instruction
        )
        batch = self.preprocessor(batch)
        with torch.inference_mode():
            raw_actions = self.policy.predict_action_chunk(batch)
            policy_actions = [
                self.postprocessor(raw_actions[:, index, :])
                for index in range(raw_actions.shape[1])
            ]
            converted = []
            for action in policy_actions:
                transition = self.rotation_postprocessor({"action": action})
                converted.append(transition["action"])
            actions = torch.stack(converted, dim=1).detach().cpu().numpy()
        absolute = np.asarray(actions, dtype=np.float32)
        if absolute.size:
            self.last_absolute_target_range = {
                "min": float(absolute.min()),
                "max": float(absolute.max()),
            }
            positions = absolute[0, :, :3]
            self.last_absolute_target_summary = {
                "first_position_xyz_m": positions[0].tolist(),
                "last_position_xyz_m": positions[-1].tolist(),
                "position_min_xyz_m": positions.min(axis=0).tolist(),
                "position_max_xyz_m": positions.max(axis=0).tolist(),
                "maximum_consecutive_position_delta_m": float(
                    np.linalg.norm(np.diff(positions, axis=0), axis=1).max()
                ),
                "gripper_values": sorted(
                    {float(value) for value in absolute[0, :, 6].tolist()}
                ),
            }
        array = _xvla_absolute_to_rpent_relative(absolute, state_vector)
        if array.size:
            self.last_action_range = {
                "min": float(array.min()),
                "max": float(array.max()),
            }
        array = _validate_xvla_relative_actions(array)
        self.forward_calls += 1
        return {
            "actions": array.tolist(),
            "shape": list(array.shape),
            "dtype": "float32",
            "policy_profile": self.profile_id,
            "checkpoint_revision": self.revision,
            "action_range": dict(self.last_action_range or {}),
            "absolute_target_range": dict(self.last_absolute_target_range or {}),
            "absolute_target_summary": dict(self.last_absolute_target_summary or {}),
            "controller_adapter": "absolute-ee-targets-to-rpent-relative-osc-v1",
        }

    def healthz(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "policy_profile": self.profile_id,
            "checkpoint_revision": self.revision,
            "checkpoint_path": str(self.checkpoint_path),
            "tokenizer_path": str(self.tokenizer_path),
            "tokenizer_revision": XVLA_TOKENIZER_REVISION,
            "controller_action_shape": ["30-120", 7],
            "raw_action_shape": [30, 20],
            "state_shape": [8],
            "xvla_proprio_shape": [20],
            "camera_keys": ["main", "wrist"],
            "forward_calls": self.forward_calls,
            "last_action_range": self.last_action_range,
            "last_absolute_target_range": self.last_absolute_target_range,
            "last_absolute_target_summary": self.last_absolute_target_summary,
            "seed": self.seed,
            "controller_adapter": {
                "id": "absolute-ee-targets-to-rpent-relative-osc-v1",
                "position_scale_m": 0.05,
                "orientation_scale_rad": 0.5,
                "sequence_reference": "measured-first-then-previous-target",
                "oversized_delta_handling": "subdivide-without-clipping",
                "maximum_controller_steps_per_forward": 120,
                "clipping": False,
            },
            "d0_status": "strict-load-real-observation-adapter-range-gate-pending",
        }


def _install_openvla_oft_namespaces(code_root: Path) -> None:
    """Expose inference modules without importing the RLDS training package.

    Upstream ``prismatic.__init__`` imports the TensorFlow/dlimp data stack.
    Inference does not use that stack, so importing it would add a large,
    irrelevant runtime dependency.  Namespace packages let Python import the
    exact pinned inference modules while keeping training code unreachable.
    """

    prismatic_root = code_root / "prismatic"
    packages = {
        "prismatic": "",
        "prismatic.extern": "extern",
        "prismatic.extern.hf": "extern/hf",
        "prismatic.training": "training",
        "prismatic.vla": "vla",
        "prismatic.models": "models",
    }
    for name, relative in packages.items():
        target = (prismatic_root / relative).resolve()
        if not target.is_dir():
            raise FileNotFoundError(target)
        existing = sys.modules.get(name)
        if existing is not None:
            paths = [Path(value).resolve() for value in getattr(existing, "__path__", [])]
            if target not in paths:
                raise RuntimeError(
                    f"OpenVLA namespace {name} was already loaded from {paths}, "
                    f"not the pinned code root {target}"
                )
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(target)]
        module.__package__ = name
        sys.modules[name] = module


def _component_state_dict(path: Path) -> dict[str, Any]:
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=True)
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in raw.items()
    }


def _prepare_openvla_oft_image(image: np.ndarray):
    """Match the public OpenVLA-OFT LIBERO resize/crop pipeline without TF.

    Upstream first JPEG-roundtrips and Lanczos-resizes to 224, then applies a
    deterministic 90%-area center crop with bilinear crop-and-resize.  The
    latter is exactly represented by ``grid_sample(..., align_corners=True)``.
    """

    import torch
    import torch.nn.functional as functional
    from PIL import Image

    source = Image.fromarray(image, mode="RGB")
    buffer = io.BytesIO()
    source.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        resized = decoded.convert("RGB").resize(
            (224, 224), resample=Image.Resampling.LANCZOS
        )
    tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).float() / 255.0
    scale = float(np.sqrt(0.9))
    coordinates = torch.linspace(-scale, scale, 224, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    cropped = functional.grid_sample(
        tensor,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0]
    array = (
        cropped.mul(255.0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


class OpenVLAOFTLiberoProvider:
    """Pinned two-view OpenVLA-OFT all-suite LIBERO action provider."""

    profile_id = OPENVLA_OFT_PROFILE
    revision = OPENVLA_OFT_REVISION

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        code_root: str | Path,
        unnorm_key: str,
        seed: int = 0,
    ):
        import torch

        snapshot = Path(checkpoint_path).expanduser().resolve()
        code = Path(code_root).expanduser().resolve()
        required = {
            "config.json",
            "dataset_statistics.json",
            "model.safetensors.index.json",
            "action_head--300000_checkpoint.pt",
            "proprio_projector--300000_checkpoint.pt",
            "preprocessor_config.json",
            "tokenizer.json",
        }
        missing = sorted(name for name in required if not (snapshot / name).is_file())
        if missing:
            raise FileNotFoundError(
                f"OpenVLA-OFT snapshot {snapshot} is missing required files: {missing}"
            )
        if not (code / ".git").exists():
            raise FileNotFoundError(f"OpenVLA-OFT code checkout is missing: {code}")
        code_revision = subprocess.check_output(
            ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
        ).strip()
        if code_revision != OPENVLA_OFT_CODE_REVISION:
            raise RuntimeError(
                "OpenVLA-OFT code drift: expected "
                f"{OPENVLA_OFT_CODE_REVISION}, found {code_revision}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("OpenVLA-OFT provider requires a CUDA device")

        direct_url = json.loads(
            importlib.metadata.distribution("transformers").read_text(
                "direct_url.json"
            )
            or "{}"
        )
        transformers_revision = direct_url.get("vcs_info", {}).get("commit_id")
        if transformers_revision != OPENVLA_OFT_TRANSFORMERS_REVISION:
            raise RuntimeError(
                "OpenVLA-OFT transformers fork drift: expected "
                f"{OPENVLA_OFT_TRANSFORMERS_REVISION}, found {transformers_revision}"
            )

        _install_openvla_oft_namespaces(code)
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoModelForVision2Seq,
            AutoTokenizer,
        )
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import (
            OpenVLAForActionPrediction,
        )
        from prismatic.extern.hf.processing_prismatic import (
            PrismaticImageProcessor,
            PrismaticProcessor,
        )
        from prismatic.models.action_heads import L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector

        AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
        AutoImageProcessor.register(
            OpenVLAConfig, PrismaticImageProcessor, exist_ok=True
        )
        AutoModelForVision2Seq.register(
            OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            str(snapshot),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
        ).eval().cuda()
        model.vision_backbone.set_num_images_in_input(2)
        model.norm_stats = json.loads(
            (snapshot / "dataset_statistics.json").read_text(encoding="utf-8")
        )
        normalized_key = str(unnorm_key).strip()
        if normalized_key not in model.norm_stats:
            raise ValueError(
                f"OpenVLA-OFT unnorm key {normalized_key!r} is unavailable; "
                f"choose from {sorted(model.norm_stats)}"
            )

        action_head = L1RegressionActionHead(
            input_dim=model.llm_dim,
            hidden_dim=model.llm_dim,
            action_dim=7,
        ).to(dtype=torch.bfloat16, device="cuda").eval()
        action_head.load_state_dict(
            _component_state_dict(snapshot / "action_head--300000_checkpoint.pt"),
            strict=True,
        )
        proprio_projector = ProprioProjector(
            llm_dim=model.llm_dim, proprio_dim=8
        ).to(dtype=torch.bfloat16, device="cuda").eval()
        proprio_projector.load_state_dict(
            _component_state_dict(
                snapshot / "proprio_projector--300000_checkpoint.pt"
            ),
            strict=True,
        )
        image_processor = PrismaticImageProcessor.from_pretrained(
            str(snapshot), local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, use_fast=True
        )

        self.checkpoint_path = snapshot
        self.code_root = code
        self.code_revision = code_revision
        self.transformers_revision = transformers_revision
        self.unnorm_key = normalized_key
        self.seed = int(seed)
        self.model = model
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.processor = PrismaticProcessor(
            image_processor=image_processor, tokenizer=tokenizer
        )
        self.forward_calls = 0

    def predict(
        self,
        instruction: str,
        images: dict[str, Any],
        state: list,
        mode: str = "eval",
    ) -> dict[str, Any]:
        import torch

        if mode != "eval":
            raise ValueError(
                f"OpenVLA-OFT provider supports mode='eval' only; got {mode!r}"
            )
        instruction, main, wrist, state_vector = _validate_wire_inputs(
            instruction, images, state
        )
        prompt = (
            "In: What action should the robot take to "
            f"{instruction.lower()}?\nOut:"
        )
        main_inputs = self.processor(
            prompt, _prepare_openvla_oft_image(main)
        ).to("cuda", dtype=torch.bfloat16)
        wrist_inputs = self.processor(
            prompt, _prepare_openvla_oft_image(wrist)
        ).to("cuda", dtype=torch.bfloat16)
        main_inputs["pixel_values"] = torch.cat(
            [main_inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
        )

        stats = self.model.norm_stats[self.unnorm_key]["proprio"]
        low = np.asarray(stats["q01"], dtype=np.float32)
        high = np.asarray(stats["q99"], dtype=np.float32)
        mask = np.asarray(
            stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool
        )
        normalized_state = np.clip(
            np.where(
                mask,
                2.0 * (state_vector - low) / (high - low + 1e-8) - 1.0,
                state_vector,
            ),
            -1.0,
            1.0,
        )
        with torch.inference_mode():
            raw_actions, _ = self.model.predict_action(
                **main_inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False,
                proprio=normalized_state,
                proprio_projector=self.proprio_projector,
                action_head=self.action_head,
            )
        actions = np.asarray(raw_actions, dtype=np.float32)
        if actions.shape != (8, 7):
            raise RuntimeError(
                f"OpenVLA-OFT must return [8,7]; received {list(actions.shape)}"
            )
        if not np.isfinite(actions).all():
            raise RuntimeError("OpenVLA-OFT returned NaN or infinite actions")
        # Upstream LIBERO evaluation maps dataset gripper [0,1] to [-1,1],
        # binarizes, then reverses the RLDS sign convention.
        actions[..., -1] = np.sign(2.0 * actions[..., -1] - 1.0)
        actions[..., -1] *= -1.0
        self.forward_calls += 1
        return {
            "actions": actions[np.newaxis, ...].tolist(),
            "shape": [1, 8, 7],
            "dtype": "float32",
            "policy_profile": self.profile_id,
            "checkpoint_revision": self.revision,
            "unnorm_key": self.unnorm_key,
        }

    def healthz(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "policy_profile": self.profile_id,
            "checkpoint_revision": self.revision,
            "checkpoint_path": str(self.checkpoint_path),
            "provider_code_revision": self.code_revision,
            "transformers_fork_revision": self.transformers_revision,
            "unnorm_key": self.unnorm_key,
            "native_action_shape": [8, 7],
            "state_shape": [8],
            "camera_keys": ["main", "wrist"],
            "forward_calls": self.forward_calls,
            "seed": self.seed,
            "d0_status": "strict-load-and-finite-forward-passed",
        }


def _serve_http(
    provider: Any,
    *,
    host: str,
    port: int,
    parent_watch: bool,
) -> None:
    """Serve with profile identity in healthz (RpcFacade hides that field)."""

    from rpent.utils.daemon import watch_parent_death
    from rpent.utils.http_rpc import HttpRpcServer

    shutdown = threading.Event()
    lock = threading.Lock()
    server_holder: dict[str, Any] = {}

    def dispatch(method: str, args: tuple, kwargs: dict) -> Any:
        if method == "healthz":
            return provider.healthz()
        if method == "shutdown":
            shutdown.set()
            return {"ok": True}
        if method != "predict":
            raise ValueError(f"unknown RPC method: {method!r}")
        with lock:
            return provider.predict(*args, **kwargs)

    server = HttpRpcServer((host, port), dispatch)
    server_holder["server"] = server
    bound_host, bound_port = server.server_address
    client_host = "127.0.0.1" if bound_host == "0.0.0.0" else bound_host
    print(f"RPC server listening on http://{client_host}:{bound_port}", flush=True)
    if parent_watch:
        watch_parent_death(shutdown.set)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        shutdown.wait()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one profile-aware VLA provider over the RPent HTTP wire."
    )
    parser.add_argument(
        "--profile",
        choices=[SMOLVLA_PROFILE, XVLA_PROFILE, OPENVLA_OFT_PROFILE, 'openpi-full-sft-60k'],
        required=True,
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--provider-code-root")
    parser.add_argument("--base-model-path")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--unnorm-key")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.cuda_device is not None:
        # This entry point intentionally imports torch/LeRobot only after the
        # caller has selected the visible device.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    import random

    import torch

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.time()
    if args.profile == SMOLVLA_PROFILE:
        if not args.base_model_path:
            parser.error("SmolVLA requires --base-model-path")
        provider = SmolVLALiberoProvider(
            args.checkpoint_path,
            base_model_path=args.base_model_path,
            seed=args.seed,
        )
    elif args.profile == XVLA_PROFILE:
        if not args.tokenizer_path:
            parser.error("X-VLA requires --tokenizer-path")
        provider = XVLALiberoProvider(
            args.checkpoint_path,
            tokenizer_path=args.tokenizer_path,
            seed=args.seed,
        )
    elif args.profile == 'openpi-full-sft-60k':
        from .madokalif_provider import MadokalifLiberoProvider
        provider = MadokalifLiberoProvider(args.checkpoint_path, seed=args.seed)
    else:
        if not args.provider_code_root:
            parser.error("OpenVLA-OFT requires --provider-code-root")
        if not args.unnorm_key:
            parser.error("OpenVLA-OFT requires --unnorm-key")
        provider = OpenVLAOFTLiberoProvider(
            args.checkpoint_path,
            code_root=args.provider_code_root,
            unnorm_key=args.unnorm_key,
            seed=args.seed,
        )
    print(
        json.dumps(
            {
                "event": "provider_ready",
                "profile": provider.profile_id,
                "revision": provider.revision,
                "load_seconds": round(time.time() - started, 3),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    _serve_http(
        provider,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()

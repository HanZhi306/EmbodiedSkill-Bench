from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SAM3_PERCEPTION_PROVIDER = "sam3-perception"
CONTACT_GRASPNET_PERCEPTION_PROVIDER = "contact-graspnet-perception"
MOLMO2_POINT_PERCEPTION_PROVIDER = "molmo2-point-perception"
MOLMO2_SAM3_BINDING_PROVIDER = "molmo2-sam3-binding-perception"
VISUAL_DIFFERENCE_PERCEPTION_PROVIDER = "visual-difference-perception"


def is_sam3_perception_skill(execution: dict[str, Any] | None) -> bool:
    return bool(
        execution
        and execution.get("provider")
        in {SAM3_PERCEPTION_PROVIDER, MOLMO2_SAM3_BINDING_PROVIDER}
    )


def build_sam3_action_handoff(
    result: dict[str, Any],
    *,
    target_phrase: str,
    target_ref: str,
    allow_agent_labelled_point: bool = False,
    allow_model_labelled_point: bool = False,
) -> dict[str, Any] | None:
    """Convert one accepted SAM3 mask into a deterministic action handoff.

    The Agent never chooses whether raw coordinates belong in a VLA prompt.  A
    semantic VLA consumer receives only the concise current-image referring
    expression below; the metric geometry remains structured audit evidence
    for a future frame-aware controller.  Weak/background point masks and
    mask-free results deliberately produce no handoff.
    """

    if (
        result.get("provider") not in {"frozen-sam3", "molmo2-sam3"}
        or result.get("found") is not True
    ):
        return None
    operation = result.get("operation")
    text_grounded = operation == "localize-object"
    point_bound = (
        allow_agent_labelled_point
        and operation == "segment-point"
        and result.get("point_mask_plausibility")
        == "geometrically-plausible-not-semantic-proof"
    )
    model_point_bound = (
        allow_model_labelled_point
        and result.get("provider") == "molmo2-sam3"
        and result.get("operation") == "bind-object-reference"
        and result.get("point_mask_plausibility")
        == "geometrically-plausible-not-semantic-proof"
        and result.get("point_selection_source") == "frozen-molmo2"
    )
    # Ordinary point masks remain read-only.  The separate binding skill opts
    # in explicitly, supplies a task phrase in its schema, and still records
    # that semantic identity came from the Agent rather than from SAM3.
    if not text_grounded and not point_bound and not model_point_bound:
        return None
    if result.get("selected_skill_retry") is not None:
        return None
    centroid = result.get("centroid_pixel_rc")
    shape = result.get("mask_shape")
    if (
        not isinstance(centroid, list)
        or len(centroid) != 2
        or not isinstance(shape, list)
        or len(shape) != 2
    ):
        return None
    try:
        row, col = (float(centroid[0]), float(centroid[1]))
        height, width = (int(shape[0]), int(shape[1]))
    except (TypeError, ValueError):
        return None
    if not np.isfinite([row, col]).all() or height < 1 or width < 1:
        return None

    phrase = " ".join(str(target_phrase).split())
    if not phrase:
        phrase = "selected target"

    def third(value: float, size: int, labels: tuple[str, str, str]) -> str:
        ratio = value / max(size - 1, 1)
        return labels[0] if ratio < 1 / 3 else labels[2] if ratio > 2 / 3 else labels[1]

    public_horizontal = third(col, width, ("left", "center", "right"))
    # RPent's public capture is a horizontal mirror of the image delivered to
    # all current semantic VLA profiles (audited against paired step-0 frames).
    # Convert here once; neither the Agent nor the policy should infer it.
    horizontal = {
        "left": "right",
        "center": "center",
        "right": "left",
    }[public_horizontal]
    vertical = third(row, height, ("upper", "middle", "lower"))
    camera = str(result.get("camera") or "agentview")
    visual_reference = (
        f'the "{phrase}" instance in the {vertical}-{horizontal} region of the '
        f"current VLA-consumed {camera} image"
    )
    return {
        "schema_version": 1,
        "semantic_identity_proven": text_grounded,
        "semantic_binding_source": (
            "frozen-text-segmentation"
            if text_grounded
            else "agent-labelled-positive-point-mask"
            if point_bound
            else "frozen-molmo2-point-plus-sam3-mask"
        ),
        "source_operation": (
            "text-prompted-localize-object"
            if text_grounded
            else "agent-labelled-point-bind"
            if point_bound
            else "model-point-sam3-bind"
        ),
        "target_ref": str(target_ref),
        "target_phrase": phrase,
        "visual_referring_expression": visual_reference,
        "camera": camera,
        "public_image_region": f"{vertical}-{public_horizontal}",
        "vla_consumed_image_region": f"{vertical}-{horizontal}",
        "public_to_vla_image_transform": "horizontal-mirror",
        "source_observation_step": int(result.get("source_observation_step", -1)),
        "source_image_sha256": result.get("source_image_sha256"),
        "mask_centroid_pixel_rc": [row, col],
        "mask_box_xyxy": result.get("box_xyxy"),
        "world_center_median_m": result.get("world_center_median_m"),
        "world_bounds_p05_m": result.get("world_bounds_p05_m"),
        "world_bounds_p95_m": result.get("world_bounds_p95_m"),
        "consumer_policy": {
            "semantic_vla": "automatically append only visual_referring_expression",
            "frame_aware_controller": "consume target_ref geometry only through an explicit typed adapter",
            "agent_routing_decision_required": False,
            "agent_selected_source_pixel": point_bound,
            "model_selected_source_pixel": model_point_bound,
            "raw_world_coordinates_in_vla_prompt": False,
            "public_image_direction_words_forwarded_unchanged": False,
        },
    }


def is_contact_graspnet_perception_skill(
    execution: dict[str, Any] | None,
) -> bool:
    return bool(
        execution
        and execution.get("provider") == CONTACT_GRASPNET_PERCEPTION_PROVIDER
    )


def is_molmo2_point_perception_skill(
    execution: dict[str, Any] | None,
) -> bool:
    return bool(
        execution
        and execution.get("provider")
        in {MOLMO2_POINT_PERCEPTION_PROVIDER, MOLMO2_SAM3_BINDING_PROVIDER}
    )


def is_visual_difference_perception_skill(
    execution: dict[str, Any] | None,
) -> bool:
    return bool(
        execution
        and execution.get("provider") == VISUAL_DIFFERENCE_PERCEPTION_PROVIDER
    )


class OpenAICompatibleVisionClient:
    """Host-only client for a selected vision-language perception provider.

    Only immutable public images and task-local text are transmitted.  The
    client never receives the simulator, dense world map, controller, or an
    Agent credential.  A local vLLM Molmo2 service and an explicitly configured
    online VLM therefore share one audited transport contract.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 180.0,
        identity: dict[str, Any] | None = None,
    ):
        endpoint = str(endpoint).strip().rstrip("/")
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("vision-language endpoint must be an HTTP(S) URL")
        self.chat_url = (
            endpoint
            if endpoint.endswith("/chat/completions")
            else f"{endpoint}/chat/completions"
        )
        self.model = str(model).strip()
        if not self.model:
            raise ValueError("vision-language model must be non-empty")
        self.api_key = str(api_key) if api_key else None
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("vision-language timeout must be positive")
        self.identity = dict(identity or {})

    def complete(
        self,
        *,
        prompt: str,
        images: list[tuple[str, Path]],
        max_tokens: int,
    ) -> dict[str, Any]:
        import requests

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_records: list[dict[str, Any]] = []
        for label, path in images:
            resolved = path.resolve()
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            payload = resolved.read_bytes()
            image_records.append(
                {
                    "label": str(label),
                    "path": str(resolved),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            )
            content.extend(
                [
                    {"type": "text", "text": f"{label}:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(payload).decode("ascii")
                        },
                    },
                ]
            )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            self.chat_url,
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.0,
                "max_tokens": int(max_tokens),
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("vision-language response has no choices")
        message = choices[0].get("message") or {}
        generated = message.get("content")
        if isinstance(generated, list):
            generated = "\n".join(
                str(item.get("text", ""))
                for item in generated
                if isinstance(item, dict) and item.get("type") == "text"
            )
        if not isinstance(generated, str) or not generated.strip():
            raise RuntimeError("vision-language response contains no text")
        expected_served_model = self.identity.get("served_model_name")
        observed_model = body.get("model")
        if (
            expected_served_model is not None
            and str(observed_model) != str(expected_served_model)
        ):
            raise RuntimeError(
                "vision-language served model identity mismatch: "
                f"expected {expected_served_model!r}, observed {observed_model!r}"
            )
        return {
            "text": generated.strip(),
            "model_requested": self.model,
            "model_observed": observed_model,
            "response_id": body.get("id"),
            "finish_reason": choices[0].get("finish_reason"),
            "usage": body.get("usage"),
            "images": image_records,
            "declared_identity": self.identity or None,
        }


class ContactGraspNetHttpClient:
    """Small, dependency-free client for a Contact-GraspNet point-cloud service.

    The service is deliberately kept outside the Agent sandbox.  It receives
    only point clouds derived from the current public observation and returns
    model proposals; it never receives an environment or a controller handle.
    """

    def __init__(self, endpoint: str, *, timeout_seconds: float = 120.0):
        endpoint = str(endpoint).strip().rstrip("/")
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("Contact-GraspNet endpoint must be an HTTP(S) URL")
        self.endpoint = endpoint
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("Contact-GraspNet timeout must be positive")

    def plan_point_clouds(
        self,
        scene_points: np.ndarray,
        object_points: np.ndarray,
        *,
        forward_passes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import requests

        payload = {
            "pc_full_base64": _array_to_base64(scene_points),
            "pc_segment_base64": _array_to_base64(object_points),
            "segmap_id": 1,
            "local_regions": True,
            "filter_grasps": True,
            "forward_passes": int(forward_passes),
            # The CAP-X reference endpoint treats max_retries=0 as an empty
            # result even when its initial forward pass found candidates.
            # One preserves that initial result; the endpoint may make one
            # randomized fallback only when the initial proposal set is empty.
            # Formal promotion therefore requires a seeded/fixed server.
            "max_retries": 1,
        }
        response = requests.post(
            f"{self.endpoint}/plan_point_clouds",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        expected = {"grasps_base64", "scores_base64", "contact_pts_base64"}
        if not isinstance(body, dict) or set(body) != expected:
            raise RuntimeError(
                "Contact-GraspNet response fields mismatch: "
                f"expected {sorted(expected)}, got "
                f"{sorted(body) if isinstance(body, dict) else type(body).__name__}"
            )
        return (
            _array_from_base64(body["grasps_base64"]),
            _array_from_base64(body["scores_base64"]),
            _array_from_base64(body["contact_pts_base64"]),
        )


def run_sam3_localization(
    *,
    client: Any,
    observation: dict[str, Any],
    arguments: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Run frozen SAM3 over one already-public observation artifact."""

    prompt = " ".join(str(arguments["prompt"]).split())
    if not prompt:
        raise ValueError("prompt must be a non-empty phrase")
    if len(prompt) > 200:
        raise ValueError("prompt must be at most 200 characters")
    camera = str(arguments.get("camera", "agentview"))
    resolution = str(arguments.get("resolution", "high"))
    min_score = float(arguments.get("min_score", 0.2))
    if camera not in {"agentview", "wrist"}:
        raise ValueError("camera must be agentview or wrist")
    if resolution not in {"low", "high"}:
        raise ValueError("resolution must be low or high")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")

    camera_data = observation["artifacts"]["cameras"][camera]
    image_key = "high_rgb" if resolution == "high" else "rgb"
    world_key = "high_world" if resolution == "high" else "world"
    image_path = Path(str(camera_data[image_key]["path"])).resolve()
    world_path = Path(str(camera_data[world_key]["path"])).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not world_path.is_file():
        raise FileNotFoundError(world_path)

    segmented = client.segment(
        image_path,
        text_prompt=prompt,
        min_score=min_score,
    )
    destination.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "provider": "frozen-sam3",
        "operation": "localize-object",
        "found": bool(segmented.found),
        "prompt": prompt,
        "camera": camera,
        "resolution": resolution,
        "min_score": min_score,
        "score": float(segmented.score) if segmented.score is not None else None,
        "box_xyxy": list(segmented.box) if segmented.box is not None else None,
        "source_observation_step": int(observation["step_index"]),
        "source_image_path": str(image_path),
        "source_image_sha256": _sha256_file(image_path),
        "source_world_map_path": str(world_path),
        "simulator_state_changed": False,
        "new_observation_created": False,
    }
    if not segmented.found or segmented.mask is None:
        result["reason"] = str(segmented.reason or "SAM3 found no mask")
        # A failed text prompt has no mask overlay to attach.  Publish a
        # coordinate guide over the exact low-resolution RGB consumed by the
        # selected point fallback so a multimodal Agent can choose [row,col]
        # without mentally rescaling the 1024px text-query frame.
        fallback_image_path = Path(str(camera_data["rgb"]["path"])).resolve()
        guide_path = destination / "overlay.png"
        _write_point_coordinate_guide(fallback_image_path, guide_path)
        result.update(
            {
                "overlay_path": str(guide_path.resolve()),
                "overlay_kind": "point-coordinate-guide",
                "overlay_resolution": "low",
                "fallback_point_image_path": str(fallback_image_path),
                "fallback_point_coordinate_convention": (
                    "256x256 agentview RGB; row from top, column from left"
                ),
            }
        )
        _write_query(destination, result)
        return result

    mask = np.asarray(segmented.mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise RuntimeError(
            f"SAM3 returned found=true with an invalid mask shape/content: {mask.shape}"
        )
    world = np.load(world_path, mmap_mode="r")
    if mask.shape != tuple(world.shape[:2]):
        raise RuntimeError(
            f"SAM3 mask/world shape mismatch: mask={mask.shape}, world={world.shape[:2]}"
        )
    valid = mask & np.isfinite(world[..., :3]).all(axis=2)
    points = np.asarray(world[..., :3][valid], dtype=np.float64)
    rows, cols = np.nonzero(mask)
    result.update(
        {
            "mask_shape": [int(value) for value in mask.shape],
            "mask_pixels": int(mask.sum()),
            "centroid_pixel_rc": [
                float(np.median(rows)),
                float(np.median(cols)),
            ],
            "valid_world_points": int(points.shape[0]),
        }
    )
    if points.size:
        p05, p50, p95 = np.percentile(points, [5, 50, 95], axis=0)
        result.update(
            {
                "world_center_median_m": p50.tolist(),
                "world_bounds_p05_m": p05.tolist(),
                "world_bounds_p95_m": p95.tolist(),
                "world_extent_p90_m": (p95 - p05).tolist(),
            }
        )
    else:
        result.update(
            {
                "world_center_median_m": None,
                "world_bounds_p05_m": None,
                "world_bounds_p95_m": None,
                "world_extent_p90_m": None,
                "world_error": "mask contains no finite public world-map points",
            }
        )

    mask_path = destination / "mask.png"
    overlay_path = destination / "overlay.png"
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
    _write_overlay(image_path, mask, overlay_path)
    result["mask_path"] = str(mask_path.resolve())
    result["overlay_path"] = str(overlay_path.resolve())
    result["overlay_kind"] = "sam3-mask"
    result["overlay_resolution"] = resolution
    _write_query(destination, result)
    return result


def run_sam3_point_segmentation(
    *,
    client: Any,
    observation: dict[str, Any],
    arguments: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Segment the visible instance under one Agent-selected public RGB point."""

    camera = str(arguments.get("camera", "agentview"))
    resolution = str(arguments.get("resolution", "low"))
    row = int(arguments["row"])
    col = int(arguments["col"])
    min_score = float(arguments.get("min_score", 0.2))
    if camera not in {"agentview", "wrist"}:
        raise ValueError("camera must be agentview or wrist")
    if resolution not in {"low", "high"}:
        raise ValueError("resolution must be low or high")
    limit = 255 if resolution == "low" else 1023
    if not 0 <= row <= limit or not 0 <= col <= limit:
        raise ValueError(
            f"point [row, col] must be within 0..{limit} for resolution={resolution}"
        )
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")

    camera_data = observation["artifacts"]["cameras"][camera]
    image_key = "high_rgb" if resolution == "high" else "rgb"
    world_key = "high_world" if resolution == "high" else "world"
    image_path = Path(str(camera_data[image_key]["path"])).resolve()
    world_path = Path(str(camera_data[world_key]["path"])).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not world_path.is_file():
        raise FileNotFoundError(world_path)

    segmented = client.segment(
        image_path,
        point=[row, col],
        min_score=min_score,
    )
    destination.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "provider": "frozen-sam3",
        "operation": "segment-point",
        "found": bool(segmented.found),
        "point_pixel_rc": [row, col],
        "camera": camera,
        "resolution": resolution,
        "min_score": min_score,
        "score": float(segmented.score) if segmented.score is not None else None,
        "box_xyxy": list(segmented.box) if segmented.box is not None else None,
        "source_observation_step": int(observation["step_index"]),
        "source_image_path": str(image_path),
        "source_image_sha256": _sha256_file(image_path),
        "source_world_map_path": str(world_path),
        "simulator_state_changed": False,
        "new_observation_created": False,
        "evidence_boundary": (
            "positive-point instance mask over the current public RGB; the Agent "
            "chooses the point, so a mask cannot prove semantic target identity"
        ),
    }
    if not segmented.found or segmented.mask is None:
        result["reason"] = str(segmented.reason or "SAM3 found no mask")
        _write_query(destination, result)
        return result

    mask = np.asarray(segmented.mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise RuntimeError(
            f"SAM3 returned found=true with an invalid mask shape/content: {mask.shape}"
        )
    world = np.load(world_path, mmap_mode="r")
    if mask.shape != tuple(world.shape[:2]):
        raise RuntimeError(
            f"SAM3 mask/world shape mismatch: mask={mask.shape}, world={world.shape[:2]}"
        )
    valid = mask & np.isfinite(world[..., :3]).all(axis=2)
    points = np.asarray(world[..., :3][valid], dtype=np.float64)
    rows, cols = np.nonzero(mask)
    result.update(
        {
            "mask_shape": [int(value) for value in mask.shape],
            "mask_pixels": int(mask.sum()),
            "mask_area_fraction": float(mask.mean()),
            "mask_touches_image_border": bool(
                mask[0].any()
                or mask[-1].any()
                or mask[:, 0].any()
                or mask[:, -1].any()
            ),
            "centroid_pixel_rc": [float(np.median(rows)), float(np.median(cols))],
            "valid_world_points": int(points.shape[0]),
        }
    )
    suspicious_large = bool(result["mask_area_fraction"] > 0.35)
    # A positive-point object mask that reaches the image boundary is not a
    # reliable instance reference, even when it is thin.  Real failures have
    # produced narrow robot/background traces spanning the frame (well below
    # the old 8% area threshold) while still receiving a high SAM score.  Such
    # a mask must remain read-only and must never become a VLA prompt card.
    suspicious_border = bool(result["mask_touches_image_border"])
    if suspicious_large or suspicious_border:
        result["point_mask_plausibility"] = "suspicious-background-or-support-mask"
        retry_tool = (
            "sam3_bind_point_reference_v1"
            if "target_phrase" in arguments
            else "sam3_segment_point"
        )
        result["selected_skill_retry"] = {
            "tool_name": retry_tool,
            "reason": (
                "the point mask is unusually large or touches the image border; "
                "it may cover a table, drawer, or support instead of the intended object"
            ),
            "instruction": (
                "Inspect the returned overlay and coordinate guide, then choose a "
                "new pixel farther inside the intended object before acting."
            ),
            "semantic_identity_proven": False,
        }
    else:
        result["point_mask_plausibility"] = "geometrically-plausible-not-semantic-proof"
    if points.size:
        p05, p50, p95 = np.percentile(points, [5, 50, 95], axis=0)
        result.update(
            {
                "world_center_median_m": p50.tolist(),
                "world_bounds_p05_m": p05.tolist(),
                "world_bounds_p95_m": p95.tolist(),
                "world_extent_p90_m": (p95 - p05).tolist(),
            }
        )
    else:
        result.update(
            {
                "world_center_median_m": None,
                "world_bounds_p05_m": None,
                "world_bounds_p95_m": None,
                "world_extent_p90_m": None,
                "world_error": "mask contains no finite public world-map points",
            }
        )

    mask_path = destination / "mask.png"
    overlay_path = destination / "overlay.png"
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
    _write_overlay(image_path, mask, overlay_path)
    result["mask_path"] = str(mask_path.resolve())
    result["overlay_path"] = str(overlay_path.resolve())
    result["overlay_kind"] = "sam3-point-mask"
    result["overlay_resolution"] = resolution
    _write_query(destination, result)
    return result


def run_molmo2_point_localization(
    *,
    client: Any,
    observation: dict[str, Any],
    arguments: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Use selected open-vocabulary pointing over one public RGB image."""

    prompt = " ".join(str(arguments["prompt"]).split())
    if not prompt:
        raise ValueError("prompt must be a non-empty phrase")
    if len(prompt) > 200:
        raise ValueError("prompt must be at most 200 characters")
    camera = str(arguments.get("camera", "agentview"))
    resolution = str(arguments.get("resolution", "high"))
    if camera not in {"agentview", "wrist"}:
        raise ValueError("camera must be agentview or wrist")
    if resolution not in {"low", "high"}:
        raise ValueError("resolution must be low or high")

    camera_data = observation["artifacts"]["cameras"][camera]
    image_key = "high_rgb" if resolution == "high" else "rgb"
    world_key = "high_world" if resolution == "high" else "world"
    image_path = Path(str(camera_data[image_key]["path"])).resolve()
    world_path = Path(str(camera_data[world_key]["path"])).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not world_path.is_file():
        raise FileNotFoundError(world_path)

    response = client.complete(
        prompt=(
            f"Point to the visual center of the visible {prompt}. Return one pointing "
            "annotation and do not describe robot actions."
        ),
        images=[("current camera image", image_path)],
        max_tokens=256,
    )
    point = _parse_grounding_point(str(response["text"]))
    destination.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "provider": "molmo2",
        "operation": "point-localize-object",
        "found": point is not None,
        "prompt": prompt,
        "camera": camera,
        "resolution": resolution,
        "source_observation_step": int(observation["step_index"]),
        "source_image_path": str(image_path),
        "source_image_sha256": _sha256_file(image_path),
        "source_world_map_path": str(world_path),
        "model_requested": response.get("model_requested"),
        "model_observed": response.get("model_observed"),
        "model_identity": response.get("declared_identity"),
        "provider_finish_reason": response.get("finish_reason"),
        "provider_usage": response.get("usage"),
        "raw_model_output": str(response["text"])[:2000],
        "raw_model_output_sha256": hashlib.sha256(
            str(response["text"]).encode("utf-8")
        ).hexdigest(),
        "simulator_state_changed": False,
        "new_observation_created": False,
        "evidence_boundary": (
            "open-vocabulary pointing hypothesis over public RGB; not an instance "
            "mask, object identity proof, grasp pose, or task-success verdict"
        ),
    }
    if point is None:
        result["reason"] = "provider output contained no valid pointing coordinate"
        _write_query(destination, result)
        return result

    with Image.open(image_path) as image:
        width, height = image.size
        x = min(width - 1, max(0, int(round(point[0] * (width - 1)))))
        y = min(height - 1, max(0, int(round(point[1] * (height - 1)))))
        overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    radius = max(5, int(round(min(width, height) * 0.012)))
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=(255, 0, 0),
        width=max(2, radius // 3),
    )
    draw.line((x - radius * 2, y, x + radius * 2, y), fill=(255, 255, 0), width=2)
    draw.line((x, y - radius * 2, x, y + radius * 2), fill=(255, 255, 0), width=2)
    overlay_path = destination / "overlay.png"
    overlay.save(overlay_path)

    world = np.load(world_path, mmap_mode="r")
    if world.shape[0] != height or world.shape[1] != width:
        raise RuntimeError(
            f"Molmo2 image/world shape mismatch: image={(height, width)} "
            f"world={world.shape[:2]}"
        )
    xyz = np.asarray(world[y, x, :3], dtype=np.float64)
    result.update(
        {
            "point_normalized_xy": [float(point[0]), float(point[1])],
            "point_pixel_xy": [x, y],
            "point_pixel_rc": [y, x],
            "world_xyz_m": xyz.tolist() if np.isfinite(xyz).all() else None,
            "world_frame": "LIBERO world",
            "overlay_path": str(overlay_path.resolve()),
        }
    )
    if not np.isfinite(xyz).all():
        result["world_error"] = "point has no finite public world-map value"
    _write_query(destination, result)
    return result


def run_molmo2_sam3_reference_binding(
    *,
    molmo2_client: Any,
    sam3_client: Any,
    observation: dict[str, Any],
    arguments: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Bind one model-pointed SAM3 mask to a semantic action phrase.

    This is deliberately one read-only provider call from the Agent's point of
    view. Molmo2 selects a pixel from the immutable public RGB, SAM3 expands it
    to an instance mask, and the runtime applies the same geometric gate used
    by explicit point segmentation. Pixel/world coordinates remain audit-only;
    only a short current-image referring expression may reach a later semantic
    VLA action through the session-owned phrase matcher.
    """

    target_phrase = " ".join(str(arguments["target_phrase"]).split())
    if not target_phrase:
        raise ValueError("target_phrase must be a non-empty phrase")
    if len(target_phrase) > 200:
        raise ValueError("target_phrase must be at most 200 characters")
    camera = str(arguments.get("camera", "agentview"))
    resolution = str(arguments.get("resolution", "high"))
    min_score = float(arguments.get("min_score", 0.2))
    if camera not in {"agentview", "wrist"}:
        raise ValueError("camera must be agentview or wrist")
    if resolution not in {"low", "high"}:
        raise ValueError("resolution must be low or high")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")

    destination.mkdir(parents=True, exist_ok=False)
    pointed = run_molmo2_point_localization(
        client=molmo2_client,
        observation=observation,
        arguments={
            "prompt": target_phrase,
            "camera": camera,
            "resolution": resolution,
        },
        destination=destination / "molmo2",
    )
    result: dict[str, Any] = {
        "provider": "molmo2-sam3",
        "operation": "bind-object-reference",
        "found": False,
        "target_phrase": target_phrase,
        "camera": camera,
        "resolution": resolution,
        "min_score": min_score,
        "point_selection_source": "frozen-molmo2",
        "source_observation_step": int(observation["step_index"]),
        "source_image_path": pointed.get("source_image_path"),
        "source_image_sha256": pointed.get("source_image_sha256"),
        "source_world_map_path": pointed.get("source_world_map_path"),
        "molmo2_query_json": pointed.get("query_json"),
        "molmo2_model_requested": pointed.get("model_requested"),
        "molmo2_model_observed": pointed.get("model_observed"),
        "molmo2_model_identity": pointed.get("model_identity"),
        "molmo2_provider_usage": pointed.get("provider_usage"),
        "molmo2_point_pixel_rc": pointed.get("point_pixel_rc"),
        "simulator_state_changed": False,
        "new_observation_created": False,
        "evidence_boundary": (
            "frozen Molmo2 open-vocabulary point plus frozen SAM3 positive-point "
            "instance mask over one public RGB frame; this is a model-grounded "
            "hypothesis, not oracle identity, a grasp pose, or task-success proof"
        ),
    }
    if pointed.get("found") is not True or not isinstance(
        pointed.get("point_pixel_rc"), list
    ):
        result["reason"] = str(
            pointed.get("reason") or "Molmo2 produced no valid pointing coordinate"
        )
        _write_query(destination, result)
        return result

    row, col = (int(value) for value in pointed["point_pixel_rc"])
    segmented = run_sam3_point_segmentation(
        client=sam3_client,
        observation=observation,
        arguments={
            "row": row,
            "col": col,
            "camera": camera,
            "resolution": resolution,
            "min_score": min_score,
        },
        destination=destination / "sam3",
    )
    passthrough = {
        "score",
        "box_xyxy",
        "mask_shape",
        "mask_pixels",
        "mask_area_fraction",
        "mask_touches_image_border",
        "centroid_pixel_rc",
        "valid_world_points",
        "point_mask_plausibility",
        "world_center_median_m",
        "world_bounds_p05_m",
        "world_bounds_p95_m",
        "world_extent_p90_m",
        "mask_path",
        "overlay_path",
        "overlay_kind",
        "overlay_resolution",
        "selected_skill_retry",
        "world_error",
    }
    result.update({key: segmented[key] for key in passthrough if key in segmented})
    result["sam3_query_json"] = segmented.get("query_json")
    result["sam3_point_pixel_rc"] = segmented.get("point_pixel_rc")
    plausible = (
        segmented.get("found") is True
        and segmented.get("point_mask_plausibility")
        == "geometrically-plausible-not-semantic-proof"
        and segmented.get("selected_skill_retry") is None
    )
    result["found"] = bool(plausible)
    if not plausible:
        result["reason"] = str(
            segmented.get("reason")
            or "SAM3 point mask failed the task-agnostic geometric plausibility gate"
        )
    _write_query(destination, result)
    return result


def run_visual_difference_description(
    *,
    client: Any,
    observation: dict[str, Any],
    arguments: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Describe differences between two immutable public episode frames."""

    current_step = int(observation["step_index"])
    reference_step = int(arguments.get("reference_step", current_step - 1))
    if reference_step < 0 or reference_step >= current_step:
        raise ValueError(
            "reference_step must identify an earlier observation in this episode"
        )
    focus = " ".join(str(arguments.get("focus", "")).split())
    if len(focus) > 300:
        raise ValueError("focus must be at most 300 characters")
    camera = str(arguments.get("camera", "agentview"))
    resolution = str(arguments.get("resolution", "high"))
    if camera not in {"agentview", "wrist"}:
        raise ValueError("camera must be agentview or wrist")
    if resolution not in {"low", "high"}:
        raise ValueError("resolution must be low or high")

    image_key = "high_rgb" if resolution == "high" else "rgb"
    current_path = Path(
        str(observation["artifacts"]["cameras"][camera][image_key]["path"])
    ).resolve()
    if not current_path.is_file():
        raise FileNotFoundError(current_path)
    step_dir = current_path.parent.parent
    if not step_dir.name.isdigit() or step_dir.parent.name != "steps":
        raise ValueError("current public image is outside the indexed episode layout")
    reference_path = (
        step_dir.parent
        / f"{reference_step:04d}"
        / "cameras"
        / current_path.name
    ).resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    task_language = str(observation.get("task", {}).get("language") or "")
    response = client.complete(
        prompt=(
            "Compare the previous and current robot-camera frames. Report only "
            "visually supported changes, unchanged relevant facts, occlusions, and "
            "uncertainties. Do not propose actions and do not treat apparent task "
            "completion as official truth. "
            f"Task context: {task_language or 'not supplied'}. "
            f"Requested focus: {focus or 'all task-relevant visible changes'}."
        ),
        images=[("previous frame", reference_path), ("current frame", current_path)],
        max_tokens=768,
    )
    destination.mkdir(parents=True, exist_ok=False)
    text = str(response["text"]).strip()
    result = {
        "provider": "vision-language-difference",
        "operation": "describe-visual-change",
        "description": text[:6000],
        "description_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "focus": focus or None,
        "camera": camera,
        "resolution": resolution,
        "reference_observation_step": reference_step,
        "source_observation_step": current_step,
        "reference_image_path": str(reference_path),
        "reference_image_sha256": _sha256_file(reference_path),
        "source_image_path": str(current_path),
        "source_image_sha256": _sha256_file(current_path),
        "model_requested": response.get("model_requested"),
        "model_observed": response.get("model_observed"),
        "model_identity": response.get("declared_identity"),
        "provider_finish_reason": response.get("finish_reason"),
        "provider_usage": response.get("usage"),
        "simulator_state_changed": False,
        "new_observation_created": False,
        "official_task_success": bool(observation["status"]["success"]),
        "model_completion_judgment_is_official": False,
        "evidence_boundary": (
            "VLM description of public before/after images; it may miss, hallucinate, "
            "or confuse occlusion with state change and cannot replace official predicates"
        ),
    }
    _write_query(destination, result)
    return result


def run_contact_graspnet_proposals(
    *,
    client: Any,
    observation: dict[str, Any],
    arguments: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Propose read-only 6-DoF grasps from a prior SAM3 query.

    The handoff is intentionally explicit: semantic localization and grasp
    proposal are two separately selected, separately audited skills.  The
    source query must be a successful SAM3 query from the same observation
    step, preventing stale masks or arbitrary host files from entering the
    provider.
    """

    source_query = _validated_prior_perception_query(
        arguments["localization_query_json"],
        observation=observation,
        destination=destination,
    )
    if source_query.get("provider") != "frozen-sam3":
        raise ValueError(
            "localization_query_json must reference a frozen SAM3 query"
        )
    if source_query.get("operation") != "localize-object":
        raise ValueError(
            "localization_query_json must reference a localize-object query"
        )
    if not source_query.get("found") or not source_query.get("mask_path"):
        raise ValueError("the referenced SAM3 query did not produce a mask")

    camera = str(source_query["camera"])
    resolution = str(source_query["resolution"])
    camera_data = observation["artifacts"]["cameras"][camera]
    image_key = "high_rgb" if resolution == "high" else "rgb"
    world_key = "high_world" if resolution == "high" else "world"
    current_image = Path(str(camera_data[image_key]["path"])).resolve()
    current_world = Path(str(camera_data[world_key]["path"])).resolve()
    if Path(str(source_query["source_image_path"])).resolve() != current_image:
        raise ValueError("SAM3 source image is not the current observation image")
    if Path(str(source_query["source_world_map_path"])).resolve() != current_world:
        raise ValueError(
            "SAM3 source world map is not the current observation world map"
        )
    if source_query.get("source_image_sha256") != _sha256_file(current_image):
        raise ValueError("SAM3 source image hash no longer matches the public image")

    mask_path = Path(str(source_query["mask_path"])).resolve()
    source_directory = Path(str(source_query["query_json"])).resolve().parent
    if mask_path.parent != source_directory or mask_path.name != "mask.png":
        raise ValueError("SAM3 mask path escapes its audited query directory")
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    world = np.load(current_world, mmap_mode="r")
    if mask.shape != tuple(world.shape[:2]):
        raise RuntimeError(
            f"SAM3 mask/world shape mismatch: mask={mask.shape}, world={world.shape[:2]}"
        )
    finite = np.isfinite(world[..., :3]).all(axis=2)
    scene_points = np.asarray(world[..., :3][finite], dtype=np.float32)
    object_points = np.asarray(world[..., :3][mask & finite], dtype=np.float32)
    if object_points.shape[0] < 32:
        raise RuntimeError(
            "SAM3 mask has fewer than 32 finite 3D points; grasp proposals are unsafe"
        )

    calibration_path = Path(
        str(observation["artifacts"]["calibration"][camera]["path"])
    ).resolve()
    cam2world = _load_cam2world(calibration_path, expected_camera=camera)
    world2cam = np.linalg.inv(cam2world)

    # Contact-GraspNet's public point-cloud entry point documents a generic
    # point-cloud frame, but the reference implementation runs
    # ``predict_grasps(..., convert_cam_coords=True)``. Its actual numerical
    # contract is therefore OpenCV camera coordinates. Keep LIBERO world
    # coordinates as the public/audited representation and transform only the
    # bounded provider payload. SAM masks can contain a small number of
    # background depth outliers, so trim only coordinate-wise 2--98 percentile
    # outliers before forming the object-conditioned local region.
    robust_object_points = _robust_segment_points(object_points)
    scene_camera = _transform_points(scene_points, world2cam)
    object_camera = _transform_points(robust_object_points, world2cam)
    scene_depth_valid = (scene_camera[:, 2] > 0.05) & (scene_camera[:, 2] < 3.0)
    scene_camera = scene_camera[scene_depth_valid]
    if scene_camera.shape[0] < 32:
        raise RuntimeError(
            "fewer than 32 scene points remain in the Contact-GraspNet camera "
            "depth range"
        )

    top_k = int(arguments.get("top_k", 5))
    min_score = float(arguments.get("min_score", 0.0))
    forward_passes = int(arguments.get("forward_passes", 2))
    max_scene_points = int(arguments.get("max_scene_points", 65536))
    max_object_points = int(arguments.get("max_object_points", 16384))
    scene_input = _evenly_subsample(scene_camera, max_scene_points)
    object_input = _evenly_subsample(object_camera, max_object_points)
    grasps, scores, contact_points = client.plan_point_clouds(
        scene_input,
        object_input,
        forward_passes=forward_passes,
    )
    grasps = np.asarray(grasps, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    contact_points = np.asarray(contact_points, dtype=np.float64)
    if grasps.size == 0:
        grasps = np.empty((0, 4, 4), dtype=np.float64)
    if contact_points.size == 0:
        contact_points = np.empty((0, 3), dtype=np.float64)
    if grasps.ndim != 3 or tuple(grasps.shape[1:]) != (4, 4):
        raise RuntimeError(
            f"Contact-GraspNet grasps must have shape [N,4,4], got {grasps.shape}"
        )
    if contact_points.ndim != 2 or contact_points.shape[1:] != (3,):
        raise RuntimeError(
            "Contact-GraspNet contact points must have shape [N,3], got "
            f"{contact_points.shape}"
        )
    if not (grasps.shape[0] == scores.shape[0] == contact_points.shape[0]):
        raise RuntimeError("Contact-GraspNet proposal arrays have inconsistent lengths")
    if not (
        np.isfinite(grasps).all()
        and np.isfinite(scores).all()
        and np.isfinite(contact_points).all()
    ):
        raise RuntimeError("Contact-GraspNet returned NaN or infinite proposals")

    # The model returns proposals in its input OpenCV camera frame. Convert
    # them back to the same LIBERO world frame used by public observations;
    # no controller/TCP convention is applied here.
    grasps_world = _transform_poses(grasps, cam2world)
    contact_points_world = _transform_points(contact_points, cam2world)

    eligible = np.flatnonzero(scores >= min_score)
    ranked = eligible[np.argsort(-scores[eligible], kind="stable")][:top_k]
    candidates = [
        {
            "rank": rank + 1,
            "score": float(scores[index]),
            "pose_matrix_world": grasps_world[index].tolist(),
            "position_world_m": grasps_world[index, :3, 3].tolist(),
            "contact_point_world_m": contact_points_world[index].tolist(),
        }
        for rank, index in enumerate(ranked)
    ]
    destination.mkdir(parents=True, exist_ok=False)
    result = {
        "provider": "contact-graspnet",
        "operation": "propose-grasps",
        "found": bool(candidates),
        "candidate_count_returned": len(candidates),
        "candidate_count_before_filter": int(scores.shape[0]),
        "candidates": candidates,
        "candidate_frame": "LIBERO world",
        "provider_input_frame": "OpenCV camera coordinates",
        "pose_semantics": (
            "raw Contact-GraspNet proposal; no TCP offset, reachability, collision, "
            "or execution success has been verified"
        ),
        "source_observation_step": int(observation["step_index"]),
        "source_localization_query_json": str(
            Path(str(source_query["query_json"])).resolve()
        ),
        "source_localization_query_sha256": _sha256_file(
            Path(str(source_query["query_json"])).resolve()
        ),
        "source_camera": camera,
        "source_resolution": resolution,
        "source_calibration_path": str(calibration_path),
        "source_calibration_sha256": _sha256_file(calibration_path),
        "cam2world": cam2world.tolist(),
        "scene_points_available": int(scene_points.shape[0]),
        "object_points_available": int(object_points.shape[0]),
        "object_points_after_robust_trim": int(robust_object_points.shape[0]),
        "object_point_trim_percentiles": [2.0, 98.0],
        "scene_camera_depth_range_m": [0.05, 3.0],
        "scene_points_sent": int(scene_input.shape[0]),
        "object_points_sent": int(object_input.shape[0]),
        "forward_passes": forward_passes,
        "min_score": min_score,
        "simulator_state_changed": False,
        "new_observation_created": False,
    }
    if not candidates:
        result["reason"] = "provider returned no proposal at or above min_score"
    _write_query(destination, result)
    return result


def _parse_grounding_point(text: str) -> tuple[float, float] | None:
    """Parse one point and return normalized ``(x, y)`` in ``[0, 1]``.

    Molmo2 emits a ``coords`` list whose first value is an annotation type and
    whose remaining values are ``object-id, x, y`` triplets on a 0--1000 grid.
    Older Molmo-compatible endpoints may emit explicit x/y attributes on a
    0--100 grid, which is retained as a documented compatibility path.
    """

    match = re.search(
        r"<(?:points|point)\b[^>]*\bcoords\s*=\s*['\"]([^'\"]+)['\"]",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        try:
            values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", match.group(1))]
        except ValueError:
            values = []
        if len(values) >= 4:
            x, y = values[2], values[3]
            if math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1000 and 0 <= y <= 1000:
                return x / 1000.0, y / 1000.0
    tag = re.search(r"<point\b[^>]*>", text, flags=re.IGNORECASE)
    if tag:
        x_match = re.search(r"\bx\s*=\s*['\"]([0-9.]+)['\"]", tag.group(0))
        y_match = re.search(r"\by\s*=\s*['\"]([0-9.]+)['\"]", tag.group(0))
        if x_match and y_match:
            x, y = float(x_match.group(1)), float(y_match.group(1))
            if math.isfinite(x) and math.isfinite(y) and 0 <= x <= 100 and 0 <= y <= 100:
                return x / 100.0, y / 100.0
    return None


def _write_overlay(image_path: Path, mask: np.ndarray, destination: Path) -> None:
    source = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    overlay = source.copy()
    tint = np.asarray([255.0, 32.0, 32.0], dtype=np.float32)
    overlay[mask] = source[mask] * 0.45 + tint * 0.55
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(
        destination
    )


def _write_point_coordinate_guide(image_path: Path, destination: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    if image.size != (256, 256):
        image = image.resize((256, 256), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    for value in range(0, 256, 32):
        color = (0, 255, 255) if value % 64 else (255, 220, 0)
        draw.line((value, 0, value, 255), fill=color, width=1)
        draw.line((0, value, 255, value), fill=color, width=1)
        if value:
            draw.rectangle((value + 1, 1, min(255, value + 29), 12), fill=(0, 0, 0))
            draw.text((value + 3, 2), f"c{value}", fill=(255, 255, 255))
            draw.rectangle((1, value + 1, 30, min(255, value + 12)), fill=(0, 0, 0))
            draw.text((3, value + 2), f"r{value}", fill=(255, 255, 255))
    draw.text((174, 242), "[row,col]", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    image.save(destination)


def _write_query(destination: Path, result: dict[str, Any]) -> None:
    query_path = destination / "query.json"
    result["query_json"] = str(query_path.resolve())
    query_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validated_prior_perception_query(
    raw_path: Any,
    *,
    observation: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    path = Path(str(raw_path)).expanduser().resolve()
    current_step_dir = destination.resolve().parent
    if path.parent.parent != current_step_dir or path.name != "query.json":
        raise ValueError(
            "localization_query_json must be a prior query.json in the current "
            "episode perception step"
        )
    try:
        source_index = int(path.parent.name.removeprefix("query-"))
        destination_index = int(destination.name.removeprefix("query-"))
    except ValueError as exc:
        raise ValueError("perception query directories must use indexed names") from exc
    if not path.parent.name.startswith("query-") or source_index >= destination_index:
        raise ValueError("localization_query_json must reference an earlier query")
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("localization query must contain a JSON object")
    if Path(str(value.get("query_json", ""))).resolve() != path:
        raise ValueError("localization query self-path does not match its audit file")
    if int(value.get("source_observation_step", -1)) != int(
        observation["step_index"]
    ):
        raise ValueError("localization query came from a stale observation step")
    return value


def _evenly_subsample(points: np.ndarray, maximum: int) -> np.ndarray:
    if maximum < 32:
        raise ValueError("point-cloud bounds must be at least 32")
    if points.shape[0] <= maximum:
        return np.ascontiguousarray(points, dtype=np.float32)
    indexes = np.linspace(0, points.shape[0] - 1, maximum, dtype=np.int64)
    return np.ascontiguousarray(points[indexes], dtype=np.float32)


def _load_cam2world(path: Path, *, expected_camera: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("camera calibration must contain a JSON object")
    camera = value.get("camera_name", value.get("camera"))
    if camera != expected_camera:
        raise ValueError(
            f"camera calibration identity mismatch: expected {expected_camera}, got {camera}"
        )
    matrix = np.asarray(value.get("extrinsic_cam2world"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("extrinsic_cam2world must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("extrinsic_cam2world must be a homogeneous transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
        raise ValueError("extrinsic_cam2world rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("extrinsic_cam2world rotation must have determinant +1")
    return matrix


def _robust_segment_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 32:
        raise ValueError("object segment must contain at least 32 points")
    lower, upper = np.percentile(points, [2.0, 98.0], axis=0)
    keep = np.logical_and(points >= lower, points <= upper).all(axis=1)
    trimmed = np.ascontiguousarray(points[keep], dtype=np.float32)
    if trimmed.shape[0] < 32:
        raise RuntimeError("robust object segment contains fewer than 32 points")
    return trimmed


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"points must have shape [N,3], got {values.shape}")
    return values @ transform[:3, :3].T + transform[:3, 3]


def _transform_poses(poses: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(poses, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 4, 4), dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape [N,4,4], got {values.shape}")
    return np.einsum("ij,njk->nik", transform, values)


def _array_to_base64(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _array_from_base64(value: Any) -> np.ndarray:
    if not isinstance(value, str):
        raise TypeError("Contact-GraspNet arrays must be base64 strings")
    try:
        raw = base64.b64decode(value, validate=True)
        return np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise RuntimeError("invalid Contact-GraspNet array payload") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

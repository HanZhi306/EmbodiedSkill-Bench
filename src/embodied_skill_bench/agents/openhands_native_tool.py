"""OpenHands native JSON-schema tools for P2.5/P3 no-skill protocols."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Self
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pydantic import Field

from openhands.sdk import Action, ImageContent, Observation, TextContent, ToolDefinition
from openhands.sdk.tool import ToolAnnotations, ToolExecutor

from embodied_skill_bench.agents.openhands_tool import (
    _image_data_url,
    observation_image_artifacts,
)
from embodied_skill_bench.runtime.native_typed_contract import (
    ACTION_TOOLS,
    tool_schemas,
)


_PROVIDER_FORBIDDEN_TOP_LEVEL_SCHEMA_KEYS = {
    "oneOf", "anyOf", "allOf", "enum", "const", "not",
}


def provider_tool_schema(
    protocol: str,
    tool_id: str,
    *,
    resolution_policy: str | None = None,
) -> dict[str, Any]:
    """Return an OpenAI-compatible view without weakening runtime validation.

    The canonical pixel schema uses a top-level conditional ``allOf`` to apply
    256- versus 1024-pixel bounds.  OpenAI function tools reject every
    top-level composition keyword before the model receives a response.  The
    provider view therefore advertises the union bound and states the exact
    conditional ranges in each coordinate description.  The runtime continues
    to validate the untouched canonical schema, so an out-of-range low-res
    call is still rejected before any simulator or geometry side effect.
    """

    schema = copy.deepcopy(tool_schemas(protocol)[tool_id])
    if tool_id == "pixel_to_world" and "allOf" in schema:
        schema.pop("allOf")
        if resolution_policy == "rgb_only":
            # The Common/Qwen single-view profile publishes the native 256px
            # RGB artifact.  Do not advertise a 1024px coordinate surface that
            # is absent from the model input: doing so silently maps an
            # otherwise correct visual point into the wrong image quadrant.
            schema["properties"]["resolution"] = {
                "type": "string",
                "enum": ["low"],
                "description": (
                    "The attached rgb_only image is 256x256; use low exactly."
                ),
            }
            if "resolution" not in schema["required"]:
                schema["required"].append("resolution")
            for coordinate in ("row", "col"):
                field = schema["properties"][coordinate]
                field["maximum"] = 255
                field["description"] = (
                    f"{field['description']} Valid range is 0..255 for the "
                    "attached 256x256 rgb_only image."
                )
        else:
            for coordinate in ("row", "col"):
                field = schema["properties"][coordinate]
                field["maximum"] = 1023
                field["description"] = (
                    f"{field['description']} Valid range is 0..255 for resolution=low "
                    "and 0..1023 for resolution=high (the default)."
                )
    forbidden = _PROVIDER_FORBIDDEN_TOP_LEVEL_SCHEMA_KEYS & set(schema)
    if forbidden:
        raise RuntimeError(
            f"provider schema for {tool_id} retains forbidden top-level keys: "
            f"{sorted(forbidden)}"
        )
    return schema


class NativeTypedAction(Action):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class NativeTypedObservation(Observation):
    response: dict[str, Any]
    image_artifacts: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        content: list[TextContent | ImageContent] = [
            TextContent(text=json.dumps(self.response, ensure_ascii=False, sort_keys=True))
        ]
        if self.image_artifacts:
            content.append(
                ImageContent(
                    image_urls=[
                        _image_data_url(Path(str(item["path"])), str(item["media_type"]))
                        for item in self.image_artifacts
                    ]
                )
            )
        return content

    def trace_payload(self) -> dict[str, Any]:
        return {"response": self.response, "image_artifacts": self.image_artifacts}


class NativeTypedExecutor(ToolExecutor[NativeTypedAction, NativeTypedObservation]):
    def __init__(self, *, workspace: str, episode_root: str, protocol: str,
                 resolution_policy: str,
                 camera_names: Sequence[str] = ("agentview", "wrist"),
                 timeout_seconds: float = 900,
                 endpoint_path: str = "/v1/native-tool"):
        context = json.loads((Path(workspace) / "run_context.json").read_text(encoding="utf-8"))
        self.runtime_url = str(context["runtime_url"]).rstrip("/")
        self.runtime_token = str(context["runtime_token"])
        self.episode_root = Path(episode_root).resolve()
        self.protocol = protocol
        self.resolution_policy = resolution_policy
        self.camera_names = tuple(str(value) for value in camera_names)
        if (
            not self.camera_names
            or len(self.camera_names) != len(set(self.camera_names))
            or not set(self.camera_names) <= {"agentview", "wrist"}
        ):
            raise ValueError("native typed camera_names must be a unique public subset")
        self.timeout_seconds = float(timeout_seconds)
        if endpoint_path not in {"/v1/native-tool", "/v1/native-skill"}:
            raise ValueError("unsupported native OpenHands endpoint")
        self.endpoint_path = endpoint_path

    def __call__(self, action: NativeTypedAction, conversation=None) -> NativeTypedObservation:  # noqa: ARG002
        if action.tool_name == "pixel_to_world" and self.resolution_policy == "rgb_only":
            if action.arguments.get("resolution") != "low":
                raise ValueError(
                    "rgb_only exposes a 256x256 image; pixel_to_world requires "
                    "resolution='low' explicitly"
                )
        payload = {
            "request_id": str(uuid.uuid4()),
            "tool_name": action.tool_name,
            "arguments": action.arguments,
            # Runtime authorizes from the concurrent OpenHands trace. This
            # nonempty value is diagnostic only and cannot grant authority.
            "agent_turn_id": f"openhands-process:{os.getpid()}",
        }
        request = Request(
            self.runtime_url + self.endpoint_path,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.runtime_token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                value = json.loads(response.read())
        except HTTPError as exc:
            raise RuntimeError(f"native typed runtime HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}") from exc
        if not isinstance(value, dict):
            raise RuntimeError("native typed runtime returned a non-object")
        images: list[dict[str, Any]] = []
        if value.get("state_changed") and value.get("automatic_observation_handoff"):
            observation = value.get("observation")
            if isinstance(observation, dict):
                images = observation_image_artifacts(
                    observation,
                    episode_root=self.episode_root,
                    resolution_policy=self.resolution_policy,
                    camera_names=self.camera_names,
                )
            cameras = tuple(str(item.get("camera")) for item in images)
            if cameras != self.camera_names:
                raise RuntimeError(
                    "accepted native action did not publish the frozen fresh camera subset"
                )
        elif value.get("accepted") and value.get("physical_round_ended") is False:
            # Selected perception skills may publish an overlay derived solely
            # from the current public observation. Attach that audited image so
            # the model can verify the grounding result before its later action
            # turn. The original RGB remains part of conversation history.
            result = value.get("result")
            overlay_raw = result.get("overlay_path") if isinstance(result, dict) else None
            if isinstance(overlay_raw, str):
                overlay = Path(overlay_raw).resolve()
                if (
                    not overlay.is_relative_to(self.episode_root)
                    or overlay.name != "overlay.png"
                    or not overlay.is_file()
                ):
                    raise RuntimeError(
                        "selected perception overlay escapes the audited episode tree"
                    )
                images = [
                    {
                        "camera": str(result.get("camera", "derived-perception")),
                        "resolution": str(
                            result.get("overlay_resolution")
                            or result.get("resolution", "derived")
                        ),
                        "path": str(overlay),
                        "media_type": "image/png",
                        "sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
                        "bytes": overlay.stat().st_size,
                        "derived_perception_overlay": True,
                    }
                ]
        return NativeTypedObservation(response=value, image_artifacts=images)

    def interrupt(self) -> None:
        return None


class NativeTypedToolBase(ToolDefinition[NativeTypedAction, NativeTypedObservation]):
    tool_id: ClassVar[str]
    canonical_schema: ClassVar[dict[str, Any] | None] = None
    endpoint_path: ClassVar[str] = "/v1/native-tool"
    read_only_tool: ClassVar[bool | None] = None
    tool_description: ClassVar[str | None] = None

    def action_from_arguments(self, arguments: dict[str, Any]) -> Action:
        # Runtime validation uses the same canonical schema. Keeping arguments
        # intact lets invalid calls be counted and rejected without side effects.
        return NativeTypedAction(tool_name=self.tool_id, arguments=dict(arguments))

    def _get_tool_schema(self, add_security_risk_prediction: bool = False,
                         action_type=None) -> dict[str, Any]:  # noqa: ARG002
        if self.canonical_schema is not None:
            return copy.deepcopy(self.canonical_schema)
        protocol = str((self.meta or {})["protocol"])
        return provider_tool_schema(
            protocol,
            self.tool_id,
            resolution_policy=str((self.meta or {}).get("resolution_policy", "")) or None,
        )

    @classmethod
    def create(
        cls,
        conv_state,
        *,
        episode_root: str,
        protocol: str,
        resolution_policy: str = "high_rgb_then_rgb",
        camera_names: Sequence[str] = ("agentview", "wrist"),
        timeout_seconds: float = 900,
    ) -> Sequence[Self]:
        schema = (
            copy.deepcopy(cls.canonical_schema)
            if cls.canonical_schema is not None
            else tool_schemas(protocol)[cls.tool_id]
        )
        executor = NativeTypedExecutor(
            workspace=str(conv_state.workspace.working_dir),
            episode_root=episode_root,
            protocol=protocol,
            resolution_policy=resolution_policy,
            camera_names=camera_names,
            timeout_seconds=timeout_seconds,
            endpoint_path=cls.endpoint_path,
        )
        read_only = (
            bool(cls.read_only_tool)
            if cls.read_only_tool is not None
            else cls.tool_id not in ACTION_TOOLS
        )
        return [
            cls(
                description=str(cls.tool_description or schema.get("description") or cls.tool_id),
                action_type=NativeTypedAction,
                observation_type=NativeTypedObservation,
                executor=executor,
                meta={
                    "protocol": protocol,
                    "resolution_policy": resolution_policy,
                    "canonical_runtime_schema": True,
                    "provider_schema_adapter": "openai-no-top-level-composition-v1",
                },
                annotations=ToolAnnotations(
                    title=cls.tool_id,
                    readOnlyHint=read_only,
                    destructiveHint=not read_only,
                    idempotentHint=read_only,
                    openWorldHint=False,
                ),
            )
        ]


def _tool_class(name: str) -> type[NativeTypedToolBase]:
    class_name = "".join(part.title() for part in name.split("_")) + "NativeTool"
    return type(class_name, (NativeTypedToolBase,), {"name": name, "tool_id": name})


NATIVE_TOOL_CLASSES = tuple(
    _tool_class(name)
    for name in (
        "observe", "get_eef_state", "get_camera_calibration", "read_depth_pixel",
        "pixel_to_world", "get_gripper_contact", "move_eef_delta", "move_eef_pose",
        "set_gripper", "check_ik_pose", "move_eef_pose_ik", "raw_vla_subgoal",
    )
)


def native_tool_classes(protocol: str) -> tuple[type[NativeTypedToolBase], ...]:
    allowed = set(tool_schemas(protocol))
    return tuple(cls for cls in NATIVE_TOOL_CLASSES if cls.tool_id in allowed)


def native_skill_tool_classes(
    contract: dict[str, Any],
) -> tuple[type[NativeTypedToolBase], ...]:
    """Build only the selected executable skill tools from a staged contract."""

    tools = contract.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("native skill contract has no tools")
    classes: list[type[NativeTypedToolBase]] = []
    for item in tools:
        if not isinstance(item, dict):
            raise ValueError("native skill tool entry must be an object")
        name = str(item["tool_name"])
        parameters = item["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("native skill parameters must be an object")
        class_name = "".join(part.title() for part in name.split("_")) + "SelectedSkillTool"
        classes.append(
            type(
                class_name,
                (NativeTypedToolBase,),
                {
                    "name": name,
                    "tool_id": name,
                    "canonical_schema": copy.deepcopy(parameters),
                    "endpoint_path": "/v1/native-skill",
                    "read_only_tool": item.get("kind") == "perception",
                    "tool_description": str(item["description"]),
                },
            )
        )
    return tuple(classes)

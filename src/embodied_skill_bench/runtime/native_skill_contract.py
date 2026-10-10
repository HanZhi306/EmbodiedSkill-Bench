"""Data-driven native tool contract for a selected Embodied Skill Library.

The skill manifests remain the single source of truth.  This module only
projects the selected executable records into provider JSON-schema tools and
validates the staged projection at both sides of the localhost boundary.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Iterable

from ..catalog import SkillRecord


NATIVE_SKILL_PROTOCOL = "S1-native-typed-curated-skill-library-v1"
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_]{1,64}$")
_PROVIDER_FORBIDDEN_TOP_LEVEL_SCHEMA_KEYS = {
    "oneOf", "anyOf", "allOf", "enum", "const", "not",
}


def native_tool_name(skill_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]", "_", skill_id)
    if not _TOOL_NAME.fullmatch(value):
        raise ValueError(f"skill id cannot form a native tool name: {skill_id!r}")
    return value


def build_native_skill_contract(records: Iterable[SkillRecord]) -> dict[str, Any]:
    selected = list(records)
    tools: list[dict[str, Any]] = []
    seen_tools: set[str] = set()
    for record in selected:
        if record.kind == "experience":
            continue
        if record.kind not in {"action", "perception"}:
            raise ValueError(f"unsupported executable skill kind: {record.kind!r}")
        if not isinstance(record.arguments_schema, dict):
            raise ValueError(f"{record.skill_id} has no executable argument schema")
        tool_name = native_tool_name(record.skill_id)
        if tool_name in seen_tools:
            raise ValueError(f"native skill tool-name collision: {tool_name}")
        seen_tools.add(tool_name)
        tools.append(
            {
                "tool_name": tool_name,
                "skill_id": record.skill_id,
                "kind": record.kind,
                "changes_state": record.changes_state,
                "description": record.description,
                "parameters": copy.deepcopy(record.arguments_schema),
                "skill_sha256": record.hash,
            }
        )
    value = {
        "schema_version": 1,
        "protocol": NATIVE_SKILL_PROTOCOL,
        "selected_skill_ids": [record.skill_id for record in selected],
        "executable_skill_ids": [item["skill_id"] for item in tools],
        "guidance_skill_ids": [
            record.skill_id for record in selected if record.kind == "experience"
        ],
        "tools": tools,
        "transport": "benchmark localhost tokenized HTTP",
        "bash_agent_exposed": False,
        "raw_http_agent_exposed": False,
        "automatic_fresh_observation_handoff": True,
        "one_state_changing_call_per_llm_response": True,
    }
    validate_native_skill_contract(value)
    return value


def validate_native_skill_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("native skill contract must be an object")
    expected = {
        "schema_version",
        "protocol",
        "selected_skill_ids",
        "executable_skill_ids",
        "guidance_skill_ids",
        "tools",
        "transport",
        "bash_agent_exposed",
        "raw_http_agent_exposed",
        "automatic_fresh_observation_handoff",
        "one_state_changing_call_per_llm_response",
    }
    if set(value) != expected:
        raise ValueError("native skill contract fields mismatch")
    if value["schema_version"] != 1 or value["protocol"] != NATIVE_SKILL_PROTOCOL:
        raise ValueError("unsupported native skill contract version/protocol")
    if (
        value["transport"] != "benchmark localhost tokenized HTTP"
        or value["bash_agent_exposed"] is not False
        or value["raw_http_agent_exposed"] is not False
        or value["automatic_fresh_observation_handoff"] is not True
        or value["one_state_changing_call_per_llm_response"] is not True
    ):
        raise ValueError("native skill transport/action-boundary drift")
    list_fields = ("selected_skill_ids", "executable_skill_ids", "guidance_skill_ids")
    for field in list_fields:
        items = value[field]
        if (
            not isinstance(items, list)
            or len(items) != len(set(items))
            or not all(isinstance(item, str) and item for item in items)
        ):
            raise ValueError(f"native skill contract {field} must be a unique string list")
    if set(value["executable_skill_ids"]) & set(value["guidance_skill_ids"]):
        raise ValueError("executable and guidance skill partitions overlap")
    if set(value["selected_skill_ids"]) != (
        set(value["executable_skill_ids"]) | set(value["guidance_skill_ids"])
    ):
        raise ValueError("native skill partitions do not cover the selected set")
    tools = value["tools"]
    if not isinstance(tools, list) or not tools:
        raise ValueError("native skill contract requires at least one executable tool")
    seen_names: set[str] = set()
    seen_skills: set[str] = set()
    tool_fields = {
        "tool_name",
        "skill_id",
        "kind",
        "changes_state",
        "description",
        "parameters",
        "skill_sha256",
    }
    for item in tools:
        if not isinstance(item, dict) or set(item) != tool_fields:
            raise ValueError("native skill tool fields mismatch")
        name = item["tool_name"]
        skill_id = item["skill_id"]
        if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
            raise ValueError(f"invalid native skill tool name: {name!r}")
        if name in seen_names or skill_id in seen_skills:
            raise ValueError("duplicate native tool name or skill id")
        seen_names.add(name)
        seen_skills.add(skill_id)
        if item["kind"] not in {"action", "perception"}:
            raise ValueError("native skill tool kind must be action or perception")
        if item["changes_state"] is not (item["kind"] == "action"):
            raise ValueError("native skill state-change declaration is inconsistent")
        if not isinstance(item["description"], str) or not item["description"].strip():
            raise ValueError("native skill tool description is empty")
        parameters = item["parameters"]
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValueError("native skill parameters must be an object schema")
        if parameters.get("additionalProperties") is not False:
            raise ValueError("native skill parameters must reject unknown fields")
        forbidden = _PROVIDER_FORBIDDEN_TOP_LEVEL_SCHEMA_KEYS & set(parameters)
        if forbidden:
            raise ValueError(
                "native skill provider schema has unsupported top-level composition: "
                f"{sorted(forbidden)}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", str(item["skill_sha256"])):
            raise ValueError("native skill tool requires a lowercase SHA-256")
    if seen_skills != set(value["executable_skill_ids"]):
        raise ValueError("native skill tools do not match executable_skill_ids")
    return value


def render_native_skill_guide(records: Iterable[SkillRecord]) -> str:
    selected = list(records)
    selected_ids = {record.skill_id for record in selected}
    sections = [
        "# Selected Embodied Skill Library",
        "",
        "The following documents are the complete selected package for this episode. "
        "Action skills are exposed as native JSON-schema tools; experience skills are "
        "read-only decision procedures embedded here. No unselected skill is available.",
    ]
    if {
        "sam3-localize-object",
        "sam3-segment-point",
    } <= selected_ids:
        sections.extend(
            [
                "",
                "## Selected perception routing",
                "",
                "Use text-prompted SAM3 first when one natural referring expression "
                "identifies the intended visible instance. If it returns `found=false`, "
                "or if its overlay selects the wrong member of a repeated category, do "
                "not keep weakening the text into a generic category prompt. Inspect the "
                "current attached 256x256 agentview RGB and call `sam3_segment_point` on "
                "a pixel well inside each plausible candidate before the next physical "
                "action. Compare the returned overlays with the task's color, size, "
                "location, and spatial relations. A point mask is geometric evidence, "
                "not semantic-identity proof, and is never automatically injected into "
                "a VLA prompt. Only a successful text-prompted SAM3 mask may be "
                "converted by the runtime into a short current-image referring "
                "expression for a phrase-matched semantic action. Do not copy "
                "coordinates or choose a coordinate consumer yourself. The runtime "
                "also applies the fixed "
                "public-image to VLA-image horizontal mirror, so do not manually "
                "rewrite left/right directions.",
            ]
        )
    if "sam3-bind-point-reference-v1" in selected_ids:
        sections.extend(
            [
                "",
                "## Framework-owned point reference binding",
                "",
                "When text segmentation misses but the intended target is visually "
                "clear, `sam3_bind_point_reference_v1` may bind one checked point "
                "mask to an explicit task-grounded phrase. The runtime—not the "
                "Agent—matches that phrase to the next action's declared target "
                "field, mirrors the public image region into the VLA view, and "
                "appends only the concise visual reference. Never copy pixels, "
                "coordinates, mask geometry, or target references into an action. "
                "This is an Agent-labelled visual hypothesis, not an oracle identity "
                "proof.",
            ]
        )
    for record in selected:
        body = (Path(record.path) / "SKILL.md").read_text(encoding="utf-8").strip()
        body = re.sub(r"```bash\n.*?```\n?", "", body, flags=re.DOTALL)
        sections.extend(
            [
                "",
                f"## {record.skill_id} ({record.kind})",
                "",
                body,
            ]
        )
        if record.kind in {"action", "perception"}:
            sections.extend(
                [
                    "",
                    f"Native tool name: `{native_tool_name(record.skill_id)}`. ",
                    "Its exact argument schema is registered directly with the model; "
                    "do not construct a shell command or outer `arguments` object.",
                ]
            )
    return "\n".join(sections).rstrip() + "\n"

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io import load_json, project_root, sha256_tree


@dataclass(frozen=True)
class SkillRecord:
    skill_id: str
    name: str
    kind: str
    description: str
    path: Path
    environments: tuple[str, ...]
    source: dict[str, Any]
    execution: dict[str, Any] | None
    arguments_schema: dict[str, Any] | None
    changes_state: bool
    interaction: dict[str, Any]
    hash: str

    def public(self) -> dict[str, Any]:
        """Metadata a selector may see; never include paths or skill bodies."""
        return {
            "id": self.skill_id,
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "environments": list(self.environments),
            "changes_state": self.changes_state,
            "interaction": self.interaction,
            "source_family": self.source.get("family"),
        }

    def locked(self) -> dict[str, Any]:
        value = self.public()
        value.update(
            {
                "path": str(self.path.resolve()),
                "sha256": self.hash,
                "source": self.source,
                "execution": self.execution,
                "arguments_schema": self.arguments_schema,
            }
        )
        return value


class SkillCatalog:
    def __init__(self, records: Iterable[SkillRecord]):
        indexed: dict[str, SkillRecord] = {}
        for record in records:
            if record.skill_id in indexed:
                raise ValueError(f"duplicate skill id: {record.skill_id}")
            indexed[record.skill_id] = record
        self._records = indexed

    @classmethod
    def discover(cls, pool: str | Path | None = None) -> "SkillCatalog":
        root = Path(pool) if pool else project_root() / "skill-pool"
        records: list[SkillRecord] = []
        for metadata_path in sorted(root.rglob("metadata.json")):
            metadata = load_json(metadata_path)
            skill_dir = metadata_path.parent
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                raise ValueError(f"missing SKILL.md beside {metadata_path}")
            skill_id = str(metadata["id"])
            kind = str(metadata["kind"])
            if kind not in {"action", "experience", "perception"}:
                raise ValueError(f"invalid kind for {skill_id}: {kind}")
            if skill_dir.name != skill_id:
                raise ValueError(
                    f"directory name {skill_dir.name!r} must equal id {skill_id!r}"
                )
            interaction = metadata.get("interaction")
            if not isinstance(interaction, dict):
                raise ValueError(f"missing interaction declaration for {skill_id}")
            expected_class = {
                "action": "simulator-action",
                "experience": "read-only-guidance",
                "perception": "read-only-perception",
            }[kind]
            if interaction.get("class") != expected_class:
                raise ValueError(
                    f"{skill_id} interaction.class must be {expected_class!r}"
                )
            changes_state = bool(metadata.get("changes_state", kind == "action"))
            if bool(interaction.get("changes_simulator_state")) != changes_state:
                raise ValueError(
                    f"{skill_id} interaction state-change declaration is inconsistent"
                )
            execution = metadata.get("execution")
            if execution is not None and not isinstance(execution, dict):
                raise ValueError(f"{skill_id} execution must be an object")
            if kind == "perception" and not execution:
                raise ValueError(f"{skill_id} perception skill requires execution")
            if execution is not None and "max_calls_per_episode" in execution:
                call_limit = execution["max_calls_per_episode"]
                if kind != "action":
                    raise ValueError(
                        f"{skill_id} max_calls_per_episode is valid only for action skills"
                    )
                if (
                    not isinstance(call_limit, int)
                    or isinstance(call_limit, bool)
                    or call_limit < 1
                ):
                    raise ValueError(
                        f"{skill_id} execution.max_calls_per_episode must be a positive integer"
                    )
            if execution is not None and "max_calls_per_argument_tuple" in execution:
                argument_limit = execution["max_calls_per_argument_tuple"]
                if kind != "action":
                    raise ValueError(
                        f"{skill_id} max_calls_per_argument_tuple is valid only for action skills"
                    )
                if not isinstance(argument_limit, dict) or set(argument_limit) != {
                    "fields",
                    "limit",
                }:
                    raise ValueError(
                        f"{skill_id} execution.max_calls_per_argument_tuple must contain exactly fields and limit"
                    )
                fields = argument_limit["fields"]
                limit = argument_limit["limit"]
                if (
                    not isinstance(fields, list)
                    or not fields
                    or any(not isinstance(field, str) or not field for field in fields)
                    or len(set(fields)) != len(fields)
                ):
                    raise ValueError(
                        f"{skill_id} execution.max_calls_per_argument_tuple.fields must be unique non-empty strings"
                    )
                schema_properties = (
                    (metadata.get("arguments_schema") or {}).get("properties") or {}
                )
                unknown_fields = sorted(set(fields) - set(schema_properties))
                if unknown_fields:
                    raise ValueError(
                        f"{skill_id} max_calls_per_argument_tuple references unknown argument fields: {unknown_fields}"
                    )
                if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
                    raise ValueError(
                        f"{skill_id} execution.max_calls_per_argument_tuple.limit must be a positive integer"
                    )
            records.append(
                SkillRecord(
                    skill_id=skill_id,
                    name=str(metadata.get("name", skill_id)),
                    kind=kind,
                    description=str(metadata["description"]),
                    path=skill_dir,
                    environments=tuple(metadata.get("environments", ["*"])),
                    source=dict(metadata.get("source", {})),
                    execution=dict(execution) if execution is not None else None,
                    arguments_schema=metadata.get("arguments_schema"),
                    changes_state=changes_state,
                    interaction=dict(interaction),
                    hash=sha256_tree(skill_dir),
                )
            )
        if not records:
            raise ValueError(f"no skills found under {root}")
        return cls(records)

    def __iter__(self):
        return iter(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def get(self, skill_id: str) -> SkillRecord:
        try:
            return self._records[skill_id]
        except KeyError as exc:
            raise KeyError(f"unknown skill: {skill_id}") from exc

    def require(self, skill_ids: Iterable[str]) -> list[SkillRecord]:
        seen: set[str] = set()
        records: list[SkillRecord] = []
        for skill_id in skill_ids:
            if skill_id in seen:
                raise ValueError(f"duplicate selected skill: {skill_id}")
            seen.add(skill_id)
            records.append(self.get(skill_id))
        return records

    def subset(self, skill_ids: Iterable[str]) -> "SkillCatalog":
        """Return an explicitly ordered, validated catalog subset."""

        return SkillCatalog(self.require(skill_ids))

    def public_rows(self, *, environment: str | None = None) -> list[dict[str, Any]]:
        result = []
        for record in sorted(self, key=lambda item: item.skill_id):
            if environment and "*" not in record.environments:
                if environment not in record.environments:
                    continue
            result.append(record.public())
        return result

    def write_public(self, path: str | Path, *, environment: str | None = None) -> None:
        Path(path).write_text(
            json.dumps(self.public_rows(environment=environment), indent=2) + "\n",
            encoding="utf-8",
        )


def validate_json_schema_subset(value: Any, schema: dict[str, Any], path: str = "arguments") -> None:
    """Validate the JSON-schema subset used by action skill metadata."""
    expected = schema.get("type")
    if isinstance(expected, list):
        if value is None and "null" in expected:
            return
        expected = next((item for item in expected if item != "null"), None)
    type_ok = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        None: True,
    }.get(expected, False)
    if not type_ok:
        raise ValueError(f"{path} must have type {expected}")
    if expected == "object":
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ValueError(f"{path}.{key} is required")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(properties)
            if extras:
                raise ValueError(f"{path} has unknown keys: {sorted(extras)}")
        for key, child in properties.items():
            if key in value:
                validate_json_schema_subset(value[key], child, f"{path}.{key}")
    elif expected == "array":
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(value) < minimum:
            raise ValueError(f"{path} needs at least {minimum} items")
        if maximum is not None and len(value) > maximum:
            raise ValueError(f"{path} allows at most {maximum} items")
        if "items" in schema:
            for index, item in enumerate(value):
                validate_json_schema_subset(item, schema["items"], f"{path}[{index}]")
    elif expected in {"integer", "number"}:
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            raise ValueError(f"{path} must be at least {minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"{path} must be at most {maximum}")
    elif expected == "string":
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if minimum is not None and len(value) < minimum:
            raise ValueError(f"{path} needs at least {minimum} characters")
        if maximum is not None and len(value) > maximum:
            raise ValueError(f"{path} allows at most {maximum} characters")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']}")

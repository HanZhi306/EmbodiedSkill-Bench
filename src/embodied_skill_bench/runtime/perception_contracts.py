"""Manifest-owned SAM3 query specializations over public observations only."""

from functools import lru_cache
from string import Formatter
from typing import Any

from ..catalog import SkillCatalog, validate_json_schema_subset


@lru_cache(maxsize=1)
def _contracts() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    return {
        row.skill_id: (row.execution, row.arguments_schema or {})
        for row in SkillCatalog.discover()
        if row.execution
        and row.execution.get("provider") == "sam3-perception"
        and row.execution.get("operation") == "localize-object"
    }


def build_sam3_localization_arguments(
    skill_id: str, arguments: dict[str, Any]
) -> dict[str, Any] | None:
    """Return canonical query arguments, or None for other operations.

    The runtime still checks selection, observation identity, and query budget.
    This function never executes a model, reads an image, or chooses a target.
    """
    contract = _contracts().get(skill_id)
    if contract is None:
        return None
    execution, schema = contract
    validate_json_schema_subset(arguments, schema)
    template = execution.get("query_template")
    if template is None:
        return dict(arguments)  # Preserve the existing generic query exactly.
    if not isinstance(template, str) or not template.strip():
        raise ValueError("SAM3 query_template must be a nonempty string")
    fields = {field for _, field, _, _ in Formatter().parse(template) if field}
    if fields - set(schema.get("properties", {})):
        raise ValueError("SAM3 template references undeclared arguments")
    values = {}
    for field in fields:
        value = arguments.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"SAM3 {field} must be a nonempty phrase")
        values[field] = " ".join(value.split())
    prompt = template.format_map(values)
    if len(prompt) > 200:
        raise ValueError("SAM3 rendered prompt must be at most 200 characters")
    defaults = execution.get("query_defaults", {})
    if not isinstance(defaults, dict) or set(defaults) - {
        "camera", "resolution", "min_score"
    }:
        raise ValueError("SAM3 query_defaults contains unsupported fields")
    query = {**defaults, "prompt": prompt}
    # Validate metadata-owned values as well as caller-supplied arguments.
    if query.get("camera") not in {"agentview", "wrist"}:
        raise ValueError("SAM3 specialization requires a supported camera")
    if query.get("resolution") not in {"low", "high"}:
        raise ValueError("SAM3 specialization requires a supported resolution")
    score = query.get("min_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        raise ValueError("SAM3 specialization requires min_score in [0, 1]")
    return query

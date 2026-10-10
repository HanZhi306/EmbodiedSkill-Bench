from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from string import Formatter
from typing import Any

from ..catalog import SkillCatalog, SkillRecord
from ..io import project_root
from .grasp_verification import annotate_pi0_pick_result

LEGACY_VLA_SKILL_IDS = frozenset({"pi0-pick", "pi0-doubled"})
INTERNAL_GROUNDING_HANDOFFS_KEY = "__esbench_grounding_handoffs_v1"


@dataclass(frozen=True)
class SemanticVLASpec:
    """One manifest-owned semantic VLA execution contract."""

    skill_id: str
    concrete_method: str
    model_profile: str
    default_max_chunks: int
    prompt_template: str
    prompt_variants: tuple[dict[str, Any], ...]
    value_maps: dict[str, dict[str, str]]
    postcondition: str
    result_adapter: str
    argument_fields: frozenset[str]
    grounding_handoff_fields: frozenset[str]


class SemanticVLAProvider:
    """Render typed skill arguments using data stored in skill manifests."""

    def __init__(self, specs: dict[str, SemanticVLASpec]):
        if not specs:
            raise ValueError("semantic VLA provider requires at least one skill spec")
        self._specs = dict(specs)

    @property
    def skill_ids(self) -> frozenset[str]:
        return frozenset(self._specs)

    def skill_ids_for_profile(self, model_profile: str) -> frozenset[str]:
        """Return only skills whose manifest selects one exact policy profile.

        Functional probes are profile-specific. Keeping this query on the
        manifest-backed provider prevents a newly added policy route from
        silently expanding an older, frozen single-profile campaign.
        """

        profile = _non_empty_string(model_profile, "model_profile")
        return frozenset(
            skill_id
            for skill_id, spec in self._specs.items()
            if spec.model_profile == profile
        )

    @classmethod
    def from_catalog(cls, catalog: SkillCatalog) -> SemanticVLAProvider:
        specs: dict[str, SemanticVLASpec] = {}
        for record in catalog:
            execution = record.execution or {}
            if execution.get("provider") != "semantic-vla":
                continue
            specs[record.skill_id] = _parse_spec(record)
        return cls(specs)

    def build_call(
        self, skill_id: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            spec = self._specs[skill_id]
        except KeyError as exc:
            raise KeyError(f"unknown semantic VLA skill: {skill_id}") from exc

        handoffs = arguments.get(INTERNAL_GROUNDING_HANDOFFS_KEY, [])
        if not isinstance(handoffs, list):
            raise ValueError("internal grounding handoffs must be an array")
        if not spec.grounding_handoff_fields:
            handoffs = []
        typed = {
            key: value
            for key, value in arguments.items()
            if key not in {"max_chunks", INTERNAL_GROUNDING_HANDOFFS_KEY}
        }
        rendered: dict[str, Any] = {}
        for key, value in typed.items():
            if value is None:
                rendered[key] = None
                continue
            normalized = _phrase(value, key) if isinstance(value, str) else value
            rendered[key] = spec.value_maps.get(key, {}).get(str(normalized), normalized)

        template = spec.prompt_template
        for variant in spec.prompt_variants:
            present = variant.get("when_present", [])
            if all(rendered.get(str(field)) not in {None, ""} for field in present):
                template = str(variant["template"])
                break
        try:
            prompt = template.format_map(rendered)
        except KeyError as exc:
            raise ValueError(
                f"{skill_id} prompt template requires missing field {exc.args[0]!r}"
            ) from exc
        # Natural referring expressions supplied by an Agent often already
        # include an article ("the black bowl"), while legacy manifest
        # templates also add one.  Preserve the semantics but prevent malformed
        # VLA handoffs such as "pick up the the black bowl".
        prompt = re.sub(r"\b(the|a|an)\s+\1\b", r"\1", prompt, flags=re.IGNORECASE)

        applied_handoffs: list[dict[str, Any]] = []
        referring_expressions: list[str] = []
        for raw in handoffs[-2:]:
            if not isinstance(raw, dict):
                raise ValueError("internal grounding handoff entries must be objects")
            expression = _non_empty_string(
                raw.get("visual_referring_expression"),
                "grounding_handoff.visual_referring_expression",
            )
            target_ref = _non_empty_string(
                raw.get("target_ref"), "grounding_handoff.target_ref"
            )
            if expression in referring_expressions:
                continue
            referring_expressions.append(expression)
            card = {
                "target_ref": target_ref,
                "visual_referring_expression": expression,
                "source_observation_step": raw.get("source_observation_step"),
                "source_image_sha256": raw.get("source_image_sha256"),
            }
            if raw.get("matched_argument_field") is not None:
                card["matched_argument_field"] = _non_empty_string(
                    raw.get("matched_argument_field"),
                    "grounding_handoff.matched_argument_field",
                )
            if raw.get("matched_argument_value") is not None:
                card["matched_argument_value"] = _non_empty_string(
                    raw.get("matched_argument_value"),
                    "grounding_handoff.matched_argument_value",
                )
            if raw.get("segmentation_target_phrase") is not None:
                card["segmentation_target_phrase"] = _non_empty_string(
                    raw.get("segmentation_target_phrase"),
                    "grounding_handoff.segmentation_target_phrase",
                )
            if raw.get("grounding_description_source") is not None:
                card["grounding_description_source"] = _non_empty_string(
                    raw.get("grounding_description_source"),
                    "grounding_handoff.grounding_description_source",
                )
            if raw.get("vla_grounding_descriptor") is not None:
                card["vla_grounding_descriptor"] = _non_empty_string(
                    raw.get("vla_grounding_descriptor"),
                    "grounding_handoff.vla_grounding_descriptor",
                )
            applied_handoffs.append(card)
        if referring_expressions:
            prompt = (
                f"{prompt}. Current-image grounding: "
                + "; ".join(referring_expressions)
                + "."
            )

        max_chunks = int(arguments.get("max_chunks", spec.default_max_chunks))
        if not 1 <= max_chunks <= 40:
            raise ValueError("max_chunks must be between 1 and 40")
        return {
            "semantic_skill": skill_id,
            "typed_arguments": typed,
            "canonical_prompt": prompt,
            "concrete_method": spec.concrete_method,
            "backend_arguments": {"prompt": prompt, "max_chunks": max_chunks},
            "model_profile": spec.model_profile,
            "postcondition": spec.postcondition,
            "result_adapter": spec.result_adapter,
            "grounding_handoff": {
                "mode": "runtime-automatic-sam3-visual-reference-v2",
                "applied": bool(applied_handoffs),
                "cards": applied_handoffs,
                "raw_world_coordinates_in_prompt": False,
                "agent_routing_decision_required": False,
            },
        }


def _template_fields(template: str) -> set[str]:
    return {
        str(field)
        for _literal, field, _format_spec, _conversion in Formatter().parse(template)
        if field
    }


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _parse_spec(record: SkillRecord) -> SemanticVLASpec:
    execution = dict(record.execution or {})
    if execution.get("schema_version") != 1:
        raise ValueError(f"{record.skill_id} semantic-vla schema_version must be 1")
    if record.kind != "action":
        raise ValueError(f"{record.skill_id} semantic-vla provider requires kind=action")
    concrete_method = _non_empty_string(
        execution.get("concrete_method"), f"{record.skill_id}.execution.concrete_method"
    )
    if concrete_method not in {"pi0_pick", "pi0_doubled"}:
        raise ValueError(
            f"{record.skill_id} has unsupported VLA concrete_method {concrete_method!r}"
        )
    model_profile = _non_empty_string(
        execution.get("model_profile"), f"{record.skill_id}.execution.model_profile"
    )
    default_max_chunks = int(execution.get("default_max_chunks", 0))
    if not 1 <= default_max_chunks <= 40:
        raise ValueError(
            f"{record.skill_id}.execution.default_max_chunks must be between 1 and 40"
        )
    prompt = execution.get("prompt")
    if not isinstance(prompt, dict):
        raise ValueError(f"{record.skill_id}.execution.prompt must be an object")
    template = _non_empty_string(
        prompt.get("template"), f"{record.skill_id}.execution.prompt.template"
    )
    variants_raw = prompt.get("variants", [])
    if not isinstance(variants_raw, list):
        raise ValueError(f"{record.skill_id}.execution.prompt.variants must be an array")
    variants: list[dict[str, Any]] = []
    for index, raw in enumerate(variants_raw):
        if not isinstance(raw, dict):
            raise ValueError(
                f"{record.skill_id}.execution.prompt.variants[{index}] must be an object"
            )
        when_present = raw.get("when_present", [])
        if not isinstance(when_present, list) or not when_present:
            raise ValueError(
                f"{record.skill_id}.execution.prompt.variants[{index}].when_present "
                "must be a non-empty array"
            )
        variant_template = _non_empty_string(
            raw.get("template"),
            f"{record.skill_id}.execution.prompt.variants[{index}].template",
        )
        variants.append(
            {
                "when_present": [str(field) for field in when_present],
                "template": variant_template,
            }
        )
    value_maps_raw = prompt.get("value_maps", {})
    if not isinstance(value_maps_raw, dict):
        raise ValueError(f"{record.skill_id}.execution.prompt.value_maps must be an object")
    value_maps: dict[str, dict[str, str]] = {}
    for field, mapping in value_maps_raw.items():
        if not isinstance(mapping, dict):
            raise ValueError(
                f"{record.skill_id}.execution.prompt.value_maps.{field} must be an object"
            )
        value_maps[str(field)] = {
            str(source): _non_empty_string(
                target,
                f"{record.skill_id}.execution.prompt.value_maps.{field}.{source}",
            )
            for source, target in mapping.items()
        }
    argument_fields = frozenset(
        str(field)
        for field in (record.arguments_schema or {}).get("properties", {})
        if field != "max_chunks"
    )
    grounding_fields_raw = execution.get(
        "grounding_handoff_fields", sorted(argument_fields)
    )
    if not isinstance(grounding_fields_raw, list) or any(
        not isinstance(field, str) or not field.strip()
        for field in grounding_fields_raw
    ):
        raise ValueError(
            f"{record.skill_id}.execution.grounding_handoff_fields must be "
            "an array of non-empty argument field names"
        )
    grounding_handoff_fields = frozenset(
        field.strip() for field in grounding_fields_raw
    )
    unknown_grounding_fields = grounding_handoff_fields - argument_fields
    if unknown_grounding_fields:
        raise ValueError(
            f"{record.skill_id}.execution.grounding_handoff_fields references "
            f"unknown arguments: {sorted(unknown_grounding_fields)}"
        )
    all_templates = [template, *(str(item["template"]) for item in variants)]
    referenced_fields = set().union(*(_template_fields(item) for item in all_templates))
    conditional_fields = {
        str(field) for item in variants for field in item["when_present"]
    }
    unknown = (referenced_fields | conditional_fields | set(value_maps)) - argument_fields
    if unknown:
        raise ValueError(
            f"{record.skill_id} semantic-vla manifest references unknown arguments: "
            f"{sorted(unknown)}"
        )
    postcondition = _non_empty_string(
        execution.get("postcondition"), f"{record.skill_id}.execution.postcondition"
    )
    result_adapter = _non_empty_string(
        execution.get("result_adapter"),
        f"{record.skill_id}.execution.result_adapter",
    )
    if result_adapter not in {"generic-task", "grasp-verification"}:
        raise ValueError(
            f"{record.skill_id} has unsupported result_adapter {result_adapter!r}"
        )
    return SemanticVLASpec(
        skill_id=record.skill_id,
        concrete_method=concrete_method,
        model_profile=model_profile,
        default_max_chunks=default_max_chunks,
        prompt_template=template,
        prompt_variants=tuple(variants),
        value_maps=value_maps,
        postcondition=postcondition,
        result_adapter=result_adapter,
        argument_fields=argument_fields,
        grounding_handoff_fields=grounding_handoff_fields,
    )


def _phrase(value: Any, field: str) -> str:
    text = " ".join(str(value).split())
    if not text:
        raise ValueError(f"{field} must be a non-empty phrase")
    if len(text) > 200:
        raise ValueError(f"{field} must be at most 200 characters")
    return text


@lru_cache(maxsize=1)
def _default_provider() -> SemanticVLAProvider:
    # Loading the native-control runtime must not load a private skill library.
    # Validate semantic manifests only when a semantic skill is actually used.
    return SemanticVLAProvider.from_catalog(SkillCatalog.discover())


def is_semantic_vla_skill(skill_id: str) -> bool:
    if not (project_root() / "skill-pool").is_dir():
        return False
    return skill_id in _default_provider().skill_ids


def semantic_vla_skill_ids_for_profile(model_profile: str) -> frozenset[str]:
    """Return manifest-owned semantic skills routed to one policy profile."""

    return _default_provider().skill_ids_for_profile(model_profile)


def build_semantic_vla_call(
    skill_id: str,
    arguments: dict[str, Any],
    *,
    provider: SemanticVLAProvider | None = None,
) -> dict[str, Any]:
    """Render typed semantic arguments using the authoritative skill manifest."""

    return (provider if provider is not None else _default_provider()).build_call(skill_id, arguments)


def annotate_semantic_vla_result(
    call: dict[str, Any], raw_result: dict[str, Any]
) -> dict[str, Any]:
    """Remove ambiguous backend success and attach an explicit verifier contract."""

    skill_id = str(call["semantic_skill"])
    if call["result_adapter"] == "grasp-verification":
        result = annotate_pi0_pick_result(raw_result)
    else:
        result = dict(raw_result)
    legacy_success = result.pop("success", None)
    task_success = result.pop("task_success", None)
    official_task_success = bool(
        result.get("libero_terminated", False)
        or task_success is True
        or (call["result_adapter"] != "grasp-verification" and legacy_success is True)
    )
    result.update(
        {
            "semantic_skill": skill_id,
            "concrete_skill": call["concrete_method"],
            "harness_vla_abstraction": "typed_vla_skill",
            "typed_arguments": call["typed_arguments"],
            "canonical_prompt": call["canonical_prompt"],
            "model_profile": call["model_profile"],
            "grounding_handoff": call["grounding_handoff"],
            "official_task_success": official_task_success,
            "local_postcondition": call["postcondition"],
            "local_postcondition_verified": None,
            "next_skill_readiness_verified": None,
            "verification_required": True,
            "verification_evidence": {
                "agentview_before_after_checked": False,
                "wrist_before_after_checked": False,
                "target_identity_checked": False,
                "hold_or_stability_checked": False,
            },
        }
    )
    return result

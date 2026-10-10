from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def episode_prompt(
    *, workspace: Path, task_language: str, initial_observation_markdown: str
) -> str:
    installed = workspace / "installed_skills.json"
    context = workspace / "run_context.json"
    codex_skills = workspace / ".agents" / "skills"
    claude_skills = workspace / ".claude" / "skills"
    utility = workspace / "esbench"
    return f"""Execute this embodied task to completion:

{task_language}

Initial post-reset observation: `{initial_observation_markdown}`
Installed skill metadata: `{installed}`
Installed skill docs for Codex: `{codex_skills}`
Installed skill docs for Claude: `{claude_skills}`
Runtime context: `{context}`
Harness utility: `{utility}`

MANDATORY FIRST TOOL TURN: do not answer with a plan, acknowledgement, or explanation. Your very next response must contain an actual shell/tool call running exactly `{utility} observe`. The episode is not complete at reset, and prose without this first observation call is invalid. After the command returns, inspect the observation and RGB artifacts before deciding what to do.

Run `{utility} observe` before the first action; the runtime requires this explicit acknowledgement before the first action and again after every completed action. Direct file reads and `pixel-to-world` do not unlock an action. Before invoking a skill, read its installed `SKILL.md` and use exactly the documented workspace-local `scripts/execute.py` command in a separate Bash/tool turn. Do not call the runtime HTTP service directly, invent an endpoint or primitive, import benchmark internals, or search outside this workspace. Use `{utility}` for `observe` and `pixel-to-world`; do not search for a global executable.

`installed_skills.json` is the complete installed set. The `.agents/skills` and
`.claude/skills` trees are two mirrors of the same documents for different
clients; use `.agents/skills` as the canonical tree and never list or reread the
Claude mirror. Do not spend turns rediscovering skill directories. Read only the
installed `SKILL.md` files needed for the next decision, and proceed to the action
as soon as its typed arguments are grounded.

Use only installed action skills, one completed invocation at a time. Every action script prints the absolute path to the next observation. Experience skills can guide diagnosis but cannot move the robot. Installed perception skills are callable read-only tools: after `observe`, you may call them repeatedly on the current observation without using an action or producing a new observation. Read their results first, then issue any action in a later LLM tool turn; never launch perception and action in parallel. Stop on success/termination/truncation and give a concise final status.
"""


def native_skill_prompt(
    *, workspace: Path, task_language: str, initial_observation_markdown: str
) -> str:
    guide_path = workspace / "NATIVE_SKILL_LIBRARY.md"
    guide = guide_path.read_text(encoding="utf-8")
    return f"""Execute this embodied task to completion with the selected Embodied Skill Library:

{task_language}

Initial post-reset observation record: `{initial_observation_markdown}`
Native selected-skill contract: `{workspace / 'native_skill_tools.json'}`
Installed-skill ledger: `{workspace / 'installed_skills.json'}`

The current public RGB camera subset is attached to this message and the initial observation is already acknowledged. All executable skills are native tools with their exact JSON schemas; there is no shell and no need to construct command lines. The first state-changing tool call in one model response ends the physical round and automatically returns and acknowledges the next observation with fresh RGB. Consume that result before acting again.

Use the embedded selected skill documents below as task-agnostic capability, verification, and recovery guidance. Preserve task attributes and relations in natural-language action arguments; do not substitute simulator object suffixes for visual identity. Treat kinematic fields as hints, not grasp or task truth. Stop on official success/termination/truncation or action-budget exhaustion.

<selected-skill-library>
{guide}
</selected-skill-library>
"""


def primitive_code_prompt(
    *,
    workspace: Path,
    task_language: str,
    initial_observation_markdown: str,
    perception_mode: str = "frozen-sam3",
) -> str:
    perception_instruction = {
        "frozen-sam3": (
            "Use frozen SAM3 through `segment_text` and the numeric geometry "
            "functions when the RGB itself is unavailable to the model."
        ),
        "observation-only": (
            "Use only the standard framework observation and its harness-attached RGB. "
            "There is no SAM3, segmentation, mask, point-cloud, object-coordinate, "
            "grasp-proposal, or other perception module in this condition."
        ),
        "observation-only-p1": (
            "Use only public robot state and harness-attached RGB. There is no SAM3, "
            "depth, calibration, pixel-to-world, semantic localization, or grasp proposal. "
            "Read the staged task-agnostic robot reference before acting."
        ),
        "calibrated-geometry-p2": (
            "You may query standard calibration, depth at pixels you choose, single-pixel "
            "pixel-to-world, and identity-free gripper contact. There is no SAM3, learned "
            "segmentation, semantic localization, object coordinate list, or grasp proposal. "
            "Read the staged task-agnostic robot reference before acting."
        ),
    }[perception_mode]
    post_action_instruction = (
        "The action result automatically includes and acknowledges the next observation "
        "with fresh RGB. Consume it, then issue the next program in a later model turn; "
        "an extra observe is optional."
        if perception_mode in {"observation-only-p1", "calibrated-geometry-p2"}
        else "Observe again after that boundary."
    )
    robot_reference = (
        f"Task-agnostic robot reference: `{workspace / 'NO_SKILL_ROBOT_REFERENCE.md'}`\n"
        if perception_mode in {"observation-only-p1", "calibrated-geometry-p2"}
        else ""
    )
    return f"""Execute this embodied task to completion without an Embodied Skill Library:

{task_language}

Initial post-reset observation: `{initial_observation_markdown}`
Frozen API documentation: `{workspace / 'PRIMITIVE_API.md'}`
Machine-readable API contract: `{workspace / 'primitive_api.json'}`
{robot_reference}Empty installed-skill ledger: `{workspace / 'installed_skills.json'}`
Runtime context: `{workspace / 'run_context.json'}`
Harness launcher: `{workspace / 'esbench-code'}`

First run `{workspace / 'esbench-code'} observe`. Inspect the returned public observation and read the API documentation. Then write one small restricted program in the workspace and run `{workspace / 'esbench-code'} run --file FILE` in its own Bash/tool turn. Read the JSON result before deciding what to do next. A program may perform multiple read-only calls, but its first physical primitive ends the round and skips all remaining statements. {post_action_instruction} Never run two programs in parallel.

There are no selected task-level skills and no VLA. {perception_instruction} Do not call HTTP, simulator internals, hidden state, or internal action adapters. Stop on success/termination/truncation and give a concise final status.
"""


def native_typed_prompt(
    *, workspace: Path, task_language: str, initial_observation_markdown: str,
    protocol: str,
) -> str:
    from .runtime.native_typed_contract import (
        RAW_PI05_PROTOCOL,
        RAW_PI05_RGB_PROTOCOL,
    )

    capability = (
        "The only state-changing capability is the generic `raw_vla_subgoal` tool. "
        "You author its natural-language instruction from the task and current images; "
        "frozen Pi0.5 receives that text without a semantic skill template. There is no "
        "SAM3, semantic action wrapper, skill document, memory, recipe, or recovery guide."
        if protocol in {RAW_PI05_PROTOCOL, RAW_PI05_RGB_PROTOCOL}
        else "There is no shell, Python, VLA, SAM3, semantic localization, grasp proposal, skill document, memory, or recipe."
    )
    return f"""Execute this embodied task to completion without an Embodied Skill Library:

{task_language}

Protocol: {protocol}
Initial post-reset observation record: `{initial_observation_markdown}`
Native tool contract: `{workspace / 'native_typed_tools.json'}`
Task-agnostic documentation: `{workspace / 'NATIVE_TYPED_TOOLS.md'}`
Robot reference: `{workspace / 'NO_SKILL_ROBOT_REFERENCE.md'}`
Empty installed-skill ledger: `{workspace / 'installed_skills.json'}`

The current public RGB camera subset frozen by the harness is attached to this message. Use only the native typed tools registered by the harness. You may make multiple read-only calls before acting. Issue at most one state-changing call per model response; it ends the physical round and returns fresh RGB from the same frozen subset. Decide the next action only in a later response after consuming that observation. {capability} Stop on official success/termination/truncation, or when `action_budget_exhausted` says no further action is available.
The attached initial observation is already acknowledged; do not spend a model turn calling `observe` before the first action unless you genuinely need to reread its machine-readable fields. Every successful state-changing tool result likewise includes and acknowledges the next observation.
"""


def selection_prompt(
    task: dict[str, Any], max_skills: int, skill_exposure: str
) -> str:
    exposure_instruction = (
        "Read `catalog.json` first and use its environment, kind, and description fields "
        "to exclude incompatible skills and form a short list. Then inspect the "
        "`SKILL.md` for every skill you may finally select and only the plausible "
        "alternatives needed to decide. Do not enumerate or concatenate every candidate "
        "document: complete the selection after the focused inspection."
        if skill_exposure == "skill_md"
        else "Use the candidate metadata in `catalog.json`; skill bodies are hidden in this ablation."
    )
    return f"""Work as a repository agent to select an embodied skill set for the task in `task.json`.

Task language: {task.get('language')}
Environment: {task.get('environment')}
Maximum selected skills: {max_skills}
Skill exposure: {skill_exposure}

{exposure_instruction}

Use Read/Glob/Grep as needed and keep the complete tool trace. Compare interfaces, interaction classes, preconditions, and failure guidance rather than selecting from names alone. Measure downstream sufficiency under a fixed executor. Include action skills needed to change simulator state, selected perception skills only when their extra observation processing is likely to matter, and experience skills only when their repair knowledge is task-relevant. Respect environment availability. Return strict JSON matching this schema:

The harness derives canonical `inspected_skill_ids` from successful Read/Glob/Grep/Bash events in the preserved trace. Your reported field is retained only for mismatch diagnostics and cannot substitute for actually inspecting a selected skill document.

{json.dumps(selection_schema(max_skills), indent=2)}

Final-output contract: after all inspection tool calls finish, your final assistant response must be exactly one JSON object matching the schema above. Do not add prose, a table, Markdown, or code fences. Do not use Bash output or a file as a substitute for the final assistant JSON.
"""


def selection_schema(max_skills: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selected_skills": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": max_skills,
            },
            "rationale": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["skill_id", "reason"],
                    "additionalProperties": False,
                },
            },
            "inspected_skill_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
        },
        "required": ["selected_skills", "rationale", "inspected_skill_ids"],
        "additionalProperties": False,
    }


def capability_choice_prompt(
    *, task_language: str, capability_module: str, candidate_count: int
) -> str:
    """Prompt one isolated selector to choose one skill from a frozen shortlist."""

    return f"""Select exactly one embodied skill for one capability needed by a task.

Task description:
{task_language}

Capability module:
{capability_module}

There are exactly {candidate_count} candidates in `catalog.json`. Read the complete
`candidate-skills/<id>/SKILL.md` documents needed to compare them. Judge the skill's
actual contract, interaction class, preconditions, outputs, and failure boundary;
do not choose from its name alone. You cannot access a simulator, observations, the
full skill pool, or evaluation labels. Return exactly one candidate skill ID.

Return strict JSON matching this schema:

{json.dumps(capability_choice_schema(), indent=2)}

Final-output contract: after all inspection calls finish, return exactly one JSON
object matching the schema. Do not add prose, Markdown, or code fences.
"""


def capability_choice_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selected_skills": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 1,
            },
            "rationale": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["skill_id", "reason"],
                    "additionalProperties": False,
                },
            },
            "inspected_skill_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
        },
        "required": ["selected_skills", "rationale", "inspected_skill_ids"],
        "additionalProperties": False,
    }

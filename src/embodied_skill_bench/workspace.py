from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .catalog import SkillCatalog
from .io import atomic_write_json, sha256_file
from .runtime.primitive_code import primitive_api_spec
from .runtime.native_typed_contract import (
    P25_PROTOCOL,
    P3_PROTOCOL,
    RAW_PI05_PROTOCOL,
    RAW_PI05_RGB_PROTOCOL,
    contract_document,
    contract_markdown,
)
from .runtime.native_skill_contract import (
    NATIVE_SKILL_PROTOCOL,
    build_native_skill_contract,
    render_native_skill_guide,
)


BENCHMARK_SYSTEM_FILENAME = "BENCHMARK_SYSTEM.md"


EPISODE_BENCHMARK_SYSTEM_INSTRUCTIONS = """# EmbodiedSkillBench episode

You are controlling one persistent embodied episode. The only robot-changing operations you may use are the installed action skills listed in `installed_skills.json`. Do not call simulator services directly, invent a primitive name, or import RPent internals.

Work closed-loop. Before the first action and after every completed action, run `./esbench observe`; this returns the complete current observation and records the harness acknowledgement required to unlock one action. Also inspect its RGB artifacts as needed. Directly reading an observation file does not acknowledge it. Then choose and invoke exactly one action skill in a separate Bash/tool turn. You may keep a long-horizon plan in mind, but never issue overlapping actions or write a full open-loop action program.

Experience skills are read-only guidance. They never execute an action. Installed perception skills are executable but read-only: after `observe`, you may invoke them repeatedly against the same current observation without advancing the simulator, consuming the action budget, creating a new observation, or clearing its acknowledgement. A perception result can only inform an action issued in a later LLM tool turn; do not launch that action in parallel with the perception query. `./esbench observe` and `./esbench pixel-to-world` are uniform harness utilities and do not count as selected embodied skills. `pixel-to-world` does not acknowledge the observation. Use the workspace-local `./esbench` launcher; do not search outside this workspace for a global executable.

Skill command path convention: run skill scripts from the episode workspace root. When a skill document shows `python scripts/execute.py`, expand that package-relative path to `python .agents/skills/<installed-skill-id>/scripts/execute.py` and keep the documented JSON arguments unchanged. Do not change the tool working directory to the skill folder. The full installed skill path must appear in the command so the strict tool-trace audit can identify the selected skill. This path convention grants no additional skills or actions.

Pixel coordinates must match the RGB artifact actually attached by the harness: use `--resolution low` for a 256x256 `rgb` image and `--resolution high` for a 1024x1024 `high_rgb` image. Never reuse coordinates across resolutions or camera frames.

Stop immediately when the observation says terminated or truncated. Do not edit benchmark source, skill files, run context, observations, or event logs. Your final answer must state whether the task succeeded and briefly summarize the skill calls used.
"""


NATIVE_SKILL_BENCHMARK_SYSTEM_INSTRUCTIONS = """# EmbodiedSkillBench native skill episode

You control one persistent embodied episode with exactly the selected Embodied Skill Library embedded in the user message. Selected action and perception skills are exposed as benchmark-owned native JSON-schema tools; no Bash, Python, raw HTTP, RPent API, simulator object, unselected skill body, task recipe, or benchmark memory is available.

The initial RGB observation is already acknowledged. You may reason with any selected read-only experience skill and may call selected perception tools before acting. Issue at most one state-changing skill tool per model response. That call ends the physical round and returns the complete fresh public observation plus the frozen RGB camera subset. Consume it before choosing an action in a later response. Do not repeat an identical action request after the runtime guard says it has reached its evidence-free retry limit.

When a selected perception result contains `selected_skill_fallback` or `selected_skill_retry`, that selected-skill composition is still unresolved. Before any state-changing call, invoke the named read-only fallback tool exactly as instructed and inspect its result. A failed text-SAM query attaches a coordinate-grid rendering of the exact 256x256 agentview image used by `sam3_segment_point`; use its `[row,col]` labels instead of mentally rescaling another image. If a point mask is flagged as an unusually large or border-touching background/support mask, choose a different pixel farther inside the intended object. The runtime rejects a physical action while this selected fallback remains pending; that rejection never changes simulator state or consumes the action budget.

For semantic VLA actions, use natural referring expressions grounded in the public image: preserve distinguishing color, size, location, and spatial relations from the task. Simulator-style suffixes such as `_1` or `_2` are not identity evidence and should not replace a visible referring expression. Kinematic diagnostics are non-semantic hints; only official task status is task truth, while grasp/hold/identity remain unknown unless supported by public observation evidence.

Stop immediately on official success, termination, truncation, or action-budget exhaustion. Your final response must state whether the task succeeded and briefly summarize the selected skill calls used.
"""


SELECTION_BENCHMARK_SYSTEM_INSTRUCTIONS = """# EmbodiedSkillBench selection phase

This is an agent task, not a one-shot classification prompt. Use Read, Glob, and Grep to inspect `catalog.json`, `task.json`, `selector_contract.json`, and the exposed candidate `SKILL.md` files. Candidate scripts, assets, model weights, simulator access, and evaluation observations are intentionally unavailable. Do not search outside this workspace. Return the smallest sufficient set, not every plausible skill, and report which skill documents you inspected. The harness computes the canonical inspected set from successful file-tool events in your trace; self-reporting a path you did not inspect does not count.
"""


PRIMITIVE_CODE_BENCHMARK_SYSTEM_INSTRUCTIONS = """# EmbodiedSkillBench primitive-code episode

You are controlling one persistent embodied episode without an Embodied Skill Library. No task-level action, VLA, recovery, experience, or perception skill is installed. Your only interaction surface is the frozen, task-agnostic API documented in `PRIMITIVE_API.md` and `primitive_api.json` through the workspace-local `./esbench-code` launcher.

Work closed-loop. Before the first program and after every state-changing program, run `./esbench-code observe`. Write one restricted Python-syntax program to a workspace file and submit exactly one file with `./esbench-code run --file FILE`. A program may call any number of read-only perception/geometry/computation functions. The first call to `move_eef_delta`, `move_eef_pose`, or `set_gripper` executes through the frozen OSC controller, immediately stops the rest of that program, captures a new immutable observation, and ends the physical round. Never launch programs in parallel.

The syntax is intentionally restricted: no import, attribute access, file/network access, reflection, function/class definitions, exceptions, comprehensions, or arbitrary Python execution. Use simple assignments, indexing, arithmetic, comparisons, `if`, bounded `for`/`range`, `print`, and the named API functions. Opaque handles are valid only for the current observation. Use `segment_text` when visual localization is needed; it runs frozen SAM3 and returns structured geometry, so a text-only model can use it.

Do not call the runtime HTTP service, simulator, RPent, MuJoCo, hidden state, or internal controller adapters directly. Do not inspect outside the staged workspace. Do not edit benchmark instructions, API specifications, run context, observations, or event logs. Stop immediately when the observation says terminated or truncated. Your final answer must state whether the task succeeded and briefly summarize the primitive programs used.
"""


PRIMITIVE_CODE_OBSERVATION_ONLY_BENCHMARK_SYSTEM_INSTRUCTIONS = """# EmbodiedSkillBench no-skill episode

You are controlling one persistent embodied episode without an Embodied Skill Library. No task-level action, VLA, recovery, experience, or perception skill is installed, and no learned perception service is available. Your only inputs are the standard benchmark observation returned by `./esbench-code observe`, including the RGB images attached by the harness, plus the public robot state.

Use only the observation/control API documented in `PRIMITIVE_API.md` and `primitive_api.json`. Write one restricted Python-syntax program to a workspace file and submit it with `./esbench-code run --file FILE`. The only state-changing operations are `move_eef_delta`, `move_eef_pose`, and `set_gripper`. The first state-changing primitive executes through the frozen controller, stops the remaining statements, captures the next immutable observation, and ends the physical round. Observe again before the next program and never launch programs in parallel.

The syntax is intentionally restricted: no import, attribute access, file/network access, reflection, function/class definitions, exceptions, comprehensions, or arbitrary Python execution. Do not call the runtime HTTP service, simulator, RPent, MuJoCo, hidden state, or internal controller adapters directly. Do not invent SAM3, segmentation, mask, depth-processing, object-coordinate, grasp-proposal, VLA, or skill calls; they are not part of this condition. Stop immediately when the observation says terminated or truncated. Your final answer must state whether the task succeeded and briefly summarize the primitive programs used.
"""


PRIMITIVE_CODE_P1_BENCHMARK_SYSTEM_INSTRUCTIONS = """# EmbodiedSkillBench P1 no-skill episode

You are controlling one persistent embodied episode without an Embodied Skill Library. No task-level action, VLA, recovery, experience, perception skill, learned perception service, or semantic object API is installed. Your inputs are the standard public robot observation and the current RGB images attached by the harness.

Read `NO_SKILL_ROBOT_REFERENCE.md` and use only the frozen API in `PRIMITIVE_API.md` / `primitive_api.json`. Run `./esbench-code observe` once after reset. Thereafter, the first state-changing call in each restricted program executes, stops that program, and automatically returns and acknowledges the next observation with fresh RGB. Consume that result before issuing one new program in a later model turn; do not submit programs in parallel. An explicit observe remains available for reinspection but is not required after a successful action handoff.

There is no SAM3, segmentation, depth, camera calibration, pixel-to-world, object coordinate, grasp proposal, or hidden state. The restricted syntax permits simple one-name assignments, indexing, arithmetic, comparisons, `if`, bounded `for`/`range`, `print`, and named API calls. Do not use imports, attributes/methods, exceptions, tuple unpacking, functions/classes, comprehensions, arbitrary files/network, simulator access, or Python exec/eval. Controller feedback proves only whether a commanded robot target was reached; it is not task success. Stop on official success/termination/truncation.
"""


PRIMITIVE_CODE_P2_BENCHMARK_SYSTEM_INSTRUCTIONS = """# EmbodiedSkillBench P2 no-skill episode

You are controlling one persistent embodied episode without an Embodied Skill Library. No task-level action, VLA, recovery, experience, perception skill, learned segmentation model, SAM3, semantic target locator, or grasp proposal is installed. Your inputs are public robot state, current RGB, and task-agnostic calibrated geometry sensors.

Read `NO_SKILL_ROBOT_REFERENCE.md` and use only the frozen API in `PRIMITIVE_API.md` / `primitive_api.json`. Run `./esbench-code observe` once after reset. Within one restricted program you may query numeric camera calibration, depth at pixels you choose, pixel-to-world at pixels you choose, and identity-free gripper contact evidence. These read-only calls do not segment or name objects. The first state-changing call ends that program and automatically returns and acknowledges the next observation with fresh RGB. Consume it before issuing one new program in a later model turn; never submit programs in parallel.

The restricted syntax permits simple one-name assignments, indexing, arithmetic, comparisons, `if`, bounded `for`/`range`, `print`, and named API calls. Do not use imports, attributes/methods, exceptions, tuple unpacking, functions/classes, comprehensions, arbitrary files/network, raw geometry files, simulator access, or Python exec/eval. Controller/contact feedback is diagnostic robot evidence, not object identity, grasp verification, or official task success. Stop on official success/termination/truncation.
"""


NATIVE_TYPED_BENCHMARK_SYSTEM_INSTRUCTIONS = """# EmbodiedSkillBench native typed no-skill episode

You control one persistent embodied episode without an Embodied Skill Library. No task-level skill, VLA, SAM3, learned perception, semantic coordinate service, grasp proposal, memory, recipe, shell, Python, raw HTTP, simulator object, or RPent API is available. Use only the benchmark-owned native tools shown by the harness.

The initial user message includes the current public RGB camera subset frozen by the harness. You may call read-only tools repeatedly. The first state-changing tool in one LLM response ends that physical round and returns a fresh observation with fresh RGB from the same subset. Consume that result before selecting one action in a later LLM response. A second action from the same response is rejected before simulator execution. Tool/controller/contact feedback is task-agnostic evidence, not object identity, grasp proof, or official success. Stop immediately on official success, termination, or truncation.

`action_budget_exhausted` is a terminal resource boundary for the Agent: no further state-changing call can execute, so return a concise final status instead of polling read-only tools.

Do not infer hidden semantic coordinates or invent unavailable tools. `P2.5` exposes only calibrated geometry through the frozen OSC controller. `P3` routes every state-changing tool through one audited JOINT_POSITION environment: inherited XYZ operations preserve or explicitly derive orientation and use the same frozen IK gate as full-pose motion. An unavailable or failed IK/joint-controller gate is a structured failure, never executed motion.
"""


RAW_PI05_BENCHMARK_SYSTEM_INSTRUCTIONS = """# EmbodiedSkillBench raw Pi0.5 no-library episode

You control one persistent embodied episode without an Embodied Skill Library. No task-level skill document, semantic VLA wrapper, action template, learned perception service, SAM3, semantic coordinate service, grasp proposal, experience, memory, recipe, shell, Python, raw HTTP, simulator object, or RPent API is available. Use only the benchmark-owned native tools shown by the harness.

The initial user message includes the current public RGB camera subset frozen by the harness, and that initial observation is already acknowledged. Do not spend a model turn calling `observe` before the first action unless you genuinely need to reread its machine-readable fields. Read-only calibrated-geometry tools are the same task-agnostic observation surface used by the comparison. The sole state-changing tool is `raw_vla_subgoal`: write one local natural-language subgoal yourself; the harness normalizes whitespace and otherwise passes it verbatim to the frozen Pi0.5 policy. The interface does not choose an action category, add a prompt template, check a skill postcondition, or provide recovery guidance.

When the registered `pixel_to_world` schema exposes only `resolution=low`, its row and column refer exactly to the attached 256x256 RGB. Do not reinterpret those pixels as 1024x1024 coordinates.

The first `raw_vla_subgoal` call in one LLM response ends that physical round and returns a fresh observation with fresh RGB. Consume that result before selecting one subgoal in a later LLM response. A second action from the same response is rejected before simulator execution. Provider output and gripper/contact feedback are task-agnostic evidence, not object identity, grasp proof, or official success. Stop immediately on official success, termination, truncation, or `action_budget_exhausted`.

Do not invent named grasp/place/open skills or unavailable providers. Keep each subgoal concrete enough for direct visual execution, but derive it from the task and current public observation rather than any hidden recipe.
"""


RAW_PI05_RGB_BENCHMARK_SYSTEM_INSTRUCTIONS = """# EmbodiedSkillBench RGB raw Pi0.5 no-library episode

You control one persistent embodied episode without an Embodied Skill Library. No task-level skill document, semantic VLA wrapper, action template, learned perception service, SAM3, depth/calibration/coordinate tool, grasp proposal, experience, memory, recipe, shell, Python, raw HTTP, simulator object, or RPent API is available. Use only the benchmark-owned native tool shown by the harness.

The initial user message contains the current public RGB camera subset, neutral robot state, and official status. The sole tool is the state-changing `raw_vla_subgoal`: write one local natural-language subgoal yourself. The harness normalizes whitespace and otherwise passes it verbatim to the same frozen Pi0.5 policy used by the comparison.

The first `raw_vla_subgoal` call in one LLM response ends that physical round and returns a fresh observation with fresh RGB. Consume that result before selecting one later subgoal. A second action from the same response is rejected before simulator execution. Provider output and gripper state are task-agnostic evidence, not object identity, grasp proof, or official success. Stop immediately on official success, termination, truncation, or `action_budget_exhausted`.

Do not invent named grasp/place/open skills or unavailable providers. Derive every local instruction only from the task and current public observation.
"""


PRIMITIVE_API_MARKDOWN = """# Frozen primitive-code API (v2)

This condition contains **no Embodied Skill Library and no VLA**. The functions below are fixed across tasks. They can only be called inside a restricted program submitted with `./esbench-code run --file FILE`.

## Read-only functions

- `observe()` returns sanitized task/status/robot/artifact metadata.
- `get_eef_state()` returns the public end-effector and gripper state.
- `load_rgb(camera="agentview", resolution="high")` returns an opaque RGB handle plus size/hash metadata. It does not return a host Python image object.
- `load_depth(camera="agentview")` returns an opaque depth handle plus bounded numeric statistics.
- `segment_text(query, camera="agentview", resolution="high", min_score=0.2)` runs frozen SAM3. On success it returns `mask_handle`, box, score, centroid, and world bounds.
- `mask_to_world_points(mask_handle, max_points=2048)` returns an opaque `points_handle` and bounded point-cloud summaries.
- `estimate_obb(points_handle)` returns a PCA OBB center, extents, XY axes, and yaw.
- `propose_grasps(points_handle, approach_height_m=0.10, max_candidates=3)` returns deterministic top-down geometric candidates. This v1 function is **not Contact-GraspNet** and is not task-conditioned.

Read-only calls do not advance the simulator and may be repeated in one program. Handles expire after a physical action.

## State-changing functions (physical round boundary)

- `move_eef_delta(delta_xyz, gripper=None, max_steps=80)`; each delta component is limited to 0.25 m.
- `move_eef_pose(xyz, pitch=None, yaw=None, gripper=None, max_steps=100)`.
- `set_gripper(state, steps=20)` where state is `"open"` or `"closed"`.

The gripper starts open. In either move call, omitting `gripper` preserves the
last commanded gripper state; pass an explicit value only when changing it.

The first state-changing call executes and immediately stops the remaining program. Its response contains `action_boundary`, the next observation paths, and task status. Run `./esbench-code observe` before the next program.

## Safe syntax example

```python
target = segment_text("red bowl")
if target["found"]:
    points = mask_to_world_points(target["mask_handle"])
    box = estimate_obb(points["points_handle"])
    move_eef_pose(box["center_xyz_m"], pitch=0.0, yaw=box["yaw_rad"], gripper="open")
```

Allowed helpers: `print`, `len`, `min`, `max`, `sum`, `abs`, `round`, `sorted`, bounded `range`, `clip`, and `sqrt`. Methods such as `.get()` are unavailable because all attribute access is blocked; use dictionary indexing.
"""


PRIMITIVE_OBSERVATION_ONLY_API_MARKDOWN = """# No-skill observation/control API (v2)

This condition contains **no Embodied Skill Library, no VLA, and no learned perception service**. The framework returns its normal observation through `./esbench-code observe`; the OpenHands harness attaches the current public RGB images to that observation for the multimodal Agent.

## Read-only functions

- `observe()` returns the sanitized current task/status/robot/artifact metadata already supplied by the framework.
- `get_eef_state()` returns the public end-effector and gripper state.

There is no SAM3, text segmentation, mask, point cloud, OBB, object coordinate, geometric grasp proposal, or hidden simulator state API.

## State-changing functions (physical round boundary)

- `move_eef_delta(delta_xyz, gripper=None, max_steps=80)`; each delta component is limited to 0.25 m.
- `move_eef_pose(xyz, pitch=None, yaw=None, gripper=None, max_steps=100)`.
- `set_gripper(state, steps=20)` where state is `"open"` or `"closed"`.

The gripper starts open. In either move call, omitting `gripper` preserves the
last commanded gripper state; pass an explicit value only when changing it.

The first state-changing call executes and immediately stops the remaining program. Observe the next framework observation before issuing another program.

Allowed helpers: `print`, `len`, `min`, `max`, `sum`, `abs`, `round`, `sorted`, bounded `range`, `clip`, and `sqrt`. Imports, attributes, arbitrary files/network, private simulator access, and Python `exec`/`eval` are unavailable.
"""


NO_SKILL_ROBOT_REFERENCE_MARKDOWN = """# Task-agnostic LIBERO robot reference

This file documents the fixed robot/controller interface. It contains no task recipe, object position, semantic target, affordance, recovery skill, or success hint.

## Frames and units

- XYZ values and XYZ deltas use meters in the right-handed `LIBERO world` frame.
- `+z` is upward. `+x` and `+y` are fixed world axes; image left/right is not a reliable substitute for either axis.
- End-effector quaternions in observations use `[x, y, z, w]`.
- `pitch` and `yaw` are radians in RPent's task-specific convention, not generic XYZ Euler angles. `pitch=0` means the end-effector z-axis is aligned with world `-z` (the common gripper-down pose); positive pitch tilts it toward world `+y`. Omitting pitch/yaw preserves the current orientation command and is the safest default for ordinary top-down moves.
- Pixel coordinates, when a mode exposes them, are `(row, col)`: row 0 is top and column 0 is left.

## Controller behavior

- `move_eef_delta` converts a world-frame delta to an absolute target and preserves orientation unless explicitly changed by another call.
- `move_eef_pose` controls position and any supplied pitch/yaw simultaneously. Collision, singularity, or reach limits can prevent convergence.
- The frozen positional tolerance is 0.012 m and orientation tolerance is 0.05 rad. P1/P2 action results report the final errors, whether the target was reached, and whether the step budget was exhausted.
- `max_steps` is a low-level servo budget, not a distance. A long budget does not guarantee reachability.
- Each delta component is hard-limited to 0.25 m. For visual calibration or cluttered motion, prefer smaller staged moves (often at most 0.05--0.10 m) and keep a visible clearance before descending.
- There is no globally collision-free workspace box: fixtures differ by task. Treat a large residual error after budget exhaustion as suspected obstruction/unreachability and inspect the returned RGB before replanning.

## Gripper behavior and evidence

- Public commands are `open` and `closed`; opening is reported in meters.
- The gripper starts open. After `set_gripper("closed")`, a later
  `move_eef_delta` or `move_eef_pose` with `gripper=None` keeps it closed;
  omission means preserve, not reopen. An explicit move argument overrides the
  previous command.
- The reported end-effector/TCP position is the grip site at the finger-pad
  pinch plane, not the wrist or palm center. In the ordinary `pitch=0`
  gripper-down pose, graspable material must intersect approximately that TCP
  height when the jaws close. Do not add an assumed wrist-length offset to a
  surface point. Use clearance for the approach, then make a separate final
  descent toward the chosen grasp material before closing.
- The parallel jaws open to roughly 0.08 m. A top-down center grasp only makes
  physical sense when solid material can fit between and obstruct the closing
  fingers. For a hollow object or an object wider than the aperture, closing at
  its empty visual center can produce a fully closed, empty grasp. A generic
  alternative is to align the jaws across an accessible thin rim, wall, handle,
  or other graspable edge, with one finger on each side of that material.
- A visible-surface point is not automatically a valid TCP target. Approach with
  clearance, descend in stages, and inspect the wrist RGB and reported opening
  after closing before transporting anything.
- A rim, edge, or otherwise off-center grasp creates a payload-to-TCP offset.
  Moving the TCP to the visual center of a destination does not then place the
  payload center there. Before release, use the post-action RGB to align the
  carried payload itself with the destination and compensate for the visible
  horizontal offset while retaining safe vertical clearance.
- A commanded close with near-zero opening can be an empty closure. A nonzero opening can indicate contact/obstruction, but does not identify an object or prove a stable grasp.
- Residual opening that persists after a lift, together with a payload moving
  coherently in the wrist RGB, is useful transport evidence even though it is
  not semantic identity or official task-success evidence. Conversely, fully
  closed jaws and no coherently moving payload indicate an empty grasp and call
  for reopening and replanning.
- Only the observation's official `status.success` / termination is task truth. Controller convergence and contact evidence must never be reported as task success.

## Closed-loop rule

P0 requires an explicit observe after every action. P1/P2 automatically attach and acknowledge the post-action observation; the next physical program must still be chosen in a later model response after consuming that result. The first physical call always ends the current restricted program. After a release, an official `success=false` means the task is not complete: inspect the returned RGB and continue a safe correction when the episode is neither terminated nor truncated. Do not substitute visual confidence for the official status.
"""


PRIMITIVE_P1_API_MARKDOWN = """# P1.1 no-skill RGB/control API (v3)

This condition contains no Embodied Skill Library, VLA, SAM3, depth, calibration, object coordinates, or grasp proposal. Read `NO_SKILL_ROBOT_REFERENCE.md` before acting.

Read-only calls: `observe()` and `get_eef_state()`.

State-changing calls:

- `move_eef_delta(delta_xyz, gripper=None, max_steps=80)`
- `move_eef_pose(xyz, pitch=None, yaw=None, gripper=None, max_steps=100)`
- `set_gripper(state, steps=20)`

The gripper starts open. Omitting `gripper` from either move preserves the
previous command, including while transporting after a separate close action.

The first state-changing call ends the program. Its JSON contains task-agnostic `controller_feedback` plus `automatic_observation_handoff=true`, the sanitized next observation, and fresh harness-attached RGB. A later model turn may issue the next program without a redundant observe.

Minimal valid action program (API functions are built-ins; do not import them):

```python
state = get_eef_state()
print(state)
move_eef_delta([0.0, 0.0, 0.03], gripper="open", max_steps=40)
```
"""


PRIMITIVE_P2_API_MARKDOWN = """# P2.1 no-skill calibrated-geometry/control API (v5)

This condition contains no Embodied Skill Library, VLA, SAM3, learned segmentation, semantic localization, object-coordinate list, or grasp proposal. The model must choose pixels from RGB itself. Read `NO_SKILL_ROBOT_REFERENCE.md` before acting.

Read-only calls:

- `observe()` and `get_eef_state()`
- `get_camera_calibration(camera="agentview")`
- `read_depth_pixel(row, col, camera="agentview", window_radius=0)` for low-resolution 256x256 depth
- `pixel_to_world(row, col, camera="agentview", resolution="low")` uses the same 256x256 coordinates as `read_depth_pixel`
- `pixel_to_world(row, col, camera="agentview", resolution="high")` uses attached 1024x1024 RGB coordinates; do not mix low coordinates with `resolution="high"`
- `get_gripper_contact()`; returns identity-free width/contact suspicion only, with `grasp_verified=null`

Geometry semantics and validity checks:

- Prefer the attached 1024x1024 RGB for visual identity, and use `resolution="high"` only with pixels chosen in that image. The 256x256 pair remains a fallback.
- `pixel_to_world` returns the first visible surface hit by exactly that pixel. It is not an object center, grasp pose, collision-free end-effector target, or semantic localization result.
- Pixels on silhouettes, gaps, occlusion boundaries, container rims, or shadows can hit a neighboring object or the background. Query at least three pixels well inside the same visually identified surface.
- If samples assumed to belong to one object differ by more than 0.05 m in XYZ, treat them as different surfaces and reject the outlier; never average across that discontinuity. Re-inspect RGB and choose new interior pixels.
- A returned surface Z is only the visible geometry at that pixel, but the reported TCP is the finger-pad pinch plane rather than the palm. Use the chosen grasp-material surface as the final-height reference after a separate clearance approach; direct equality still does not guarantee collision-free or stable grasping.

State-changing calls are the same frozen controller as P1:

- `move_eef_delta(delta_xyz, gripper=None, max_steps=80)`
- `move_eef_pose(xyz, pitch=None, yaw=None, gripper=None, max_steps=100)`
- `set_gripper(state, steps=20)`

The gripper starts open. Omitting `gripper` from either move preserves the
previous command, including while transporting after a separate close action.

Read-only geometry calls may be composed before one action. The first state-changing call ends the program and automatically returns/acknowledges the next observation with fresh RGB. Numeric geometry/controller/contact output is not official task truth.

Minimal valid same-pixel query (API functions are built-ins; do not import them):

```python
depth = read_depth_pixel(128, 128, camera="agentview")
point = pixel_to_world(128, 128, camera="agentview", resolution="low")
print(depth)
print(point)
```

Use only simple one-name assignments. Imports, exceptions, tuple unpacking, attribute/method access, function/class definitions, and comprehensions are rejected.
"""


def _write_benchmark_system(root: Path, instructions: str) -> None:
    """Write the one source injected explicitly into every external agent."""

    (root / BENCHMARK_SYSTEM_FILENAME).write_text(instructions, encoding="utf-8")


def stage_episode_workspace(
    *,
    workspace: str | Path,
    catalog: SkillCatalog,
    selected_skill_ids: list[str],
    run_context: dict[str, Any],
) -> Path:
    root = Path(workspace).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    records = catalog.require(selected_skill_ids)
    destinations = [root / ".agents" / "skills", root / ".claude" / "skills"]
    for base in destinations:
        base.mkdir(parents=True, exist_ok=True)
        for record in records:
            staged_skill = base / record.skill_id
            shutil.copytree(record.path, staged_skill, symlinks=True)
            # Some development specialists were authored for the native
            # selected-skill interface and therefore contain no CLI wrapper.
            # Legacy shell agents (including Codex) still need a stable,
            # skill-local entry point.  Add only this generic dispatcher to
            # the staged copy; it routes through the authenticated runtime
            # and does not alter the locked source skill tree or its hash.
            if record.kind in {"action", "perception"} and not (
                staged_skill / "scripts" / "execute.py"
            ).exists():
                launcher = staged_skill / "scripts" / "execute.py"
                launcher.parent.mkdir(parents=True, exist_ok=True)
                # Same reduced-environment guard as the ``esbench`` launcher:
                # walk up from <ws>/.agents|.claude/skills/<id>/scripts/ to the
                # workspace root and add the staged runtime to sys.path.
                launcher.write_text(
                    "#!/usr/bin/env python3\n"
                    "import sys\n"
                    "from pathlib import Path\n\n"
                    "_ws = Path(__file__).resolve().parents[4]\n"
                    "_runtime = _ws / '.agent-runtime'\n"
                    "if _runtime.is_dir() and str(_runtime) not in sys.path:\n"
                    "    sys.path.insert(0, str(_runtime))\n\n"
                    "from embodied_skill_bench.skill_entry import main\n\n"
                    "if __name__ == '__main__':\n"
                    f"    raise SystemExit(main({record.skill_id!r}))\n",
                    encoding="utf-8",
                )
                launcher.chmod(0o755)
    # Checkpoints are runtime-private. Action/perception scripts need only the
    # authenticated runtime client, never model weights or their host paths.
    agent_context = {
        key: value
        for key, value in run_context.items()
        if key not in {"vla_checkpoint_path", "sam3_checkpoint_path"}
    }
    installed = [record.public() | {"sha256": record.hash} for record in records]
    atomic_write_json(root / "installed_skills.json", installed)
    atomic_write_json(root / "run_context.json", agent_context)
    runtime_root = root / ".agent-runtime" / "embodied_skill_bench"
    (runtime_root / "runtime").mkdir(parents=True)
    source_root = Path(__file__).resolve().parent
    for relative in (
        Path("__init__.py"),
        Path("agent_cli.py"),
        Path("skill_entry.py"),
        Path("runtime/__init__.py"),
        Path("runtime/client.py"),
    ):
        shutil.copy2(source_root / relative, runtime_root / relative)
    utility = root / "esbench"
    # Resolve the staged runtime relative to this launcher instead of relying
    # on an inherited PYTHONPATH.  Codex forwards the harness environment into
    # its shell calls, but some agent harnesses (Kimi Code) spawn tool shells
    # with a reduced environment, which would otherwise break every
    # ``./esbench`` invocation with ModuleNotFoundError.  Harmless where
    # PYTHONPATH is already correct: the same path is simply re-asserted.
    utility.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "_runtime = Path(__file__).resolve().parent / '.agent-runtime'\n"
        "if _runtime.is_dir() and str(_runtime) not in sys.path:\n"
        "    sys.path.insert(0, str(_runtime))\n\n"
        "from embodied_skill_bench.agent_cli import main\n\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n",
        encoding="utf-8",
    )
    utility.chmod(0o755)
    _write_benchmark_system(root, EPISODE_BENCHMARK_SYSTEM_INSTRUCTIONS)
    return root


def stage_primitive_code_workspace(
    *,
    workspace: str | Path,
    run_context: dict[str, Any],
    perception_mode: str = "frozen-sam3",
    repeat_error_feedback: bool = False,
) -> Path:
    """Stage an agent workspace that contains no selected skill bodies."""

    root = Path(workspace).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    # Empty conventional locations make absence easy to audit and prevent an
    # agent/harness from silently loading host-level skill documents.
    (root / ".agents" / "skills").mkdir(parents=True)
    (root / ".claude" / "skills").mkdir(parents=True)
    atomic_write_json(root / "installed_skills.json", [])
    agent_context = {
        key: value
        for key, value in run_context.items()
        if key not in {"vla_checkpoint_path", "sam3_checkpoint_path"}
    }
    atomic_write_json(root / "run_context.json", agent_context)
    api_spec = primitive_api_spec(
        perception_mode, repeat_error_feedback=repeat_error_feedback
    )
    atomic_write_json(root / "primitive_api.json", api_spec)
    api_markdown = {
        "frozen-sam3": PRIMITIVE_API_MARKDOWN,
        "observation-only": PRIMITIVE_OBSERVATION_ONLY_API_MARKDOWN,
        "observation-only-p1": PRIMITIVE_P1_API_MARKDOWN,
        "calibrated-geometry-p2": PRIMITIVE_P2_API_MARKDOWN,
    }[perception_mode]
    if repeat_error_feedback:
        api_markdown += """

## Repeated program-error feedback

When the exact same program SHA fails with the exact same interpreter/API error
more than once consecutively, the JSON result includes a deterministic
`failure_feedback` count and tells you to revise the program and re-read this
file. It does not provide a task solution or retry physical actions. Do not run
an unchanged program again when `program_must_change_before_retry=true`.
"""
    (root / "PRIMITIVE_API.md").write_text(api_markdown, encoding="utf-8")
    if perception_mode in {"observation-only-p1", "calibrated-geometry-p2"}:
        (root / "NO_SKILL_ROBOT_REFERENCE.md").write_text(
            NO_SKILL_ROBOT_REFERENCE_MARKDOWN, encoding="utf-8"
        )
    runtime_root = root / ".agent-runtime" / "embodied_skill_bench"
    (runtime_root / "runtime").mkdir(parents=True)
    source_root = Path(__file__).resolve().parent
    for relative in (
        Path("__init__.py"),
        Path("primitive_code_agent_cli.py"),
        Path("runtime/__init__.py"),
        Path("runtime/client.py"),
    ):
        shutil.copy2(source_root / relative, runtime_root / relative)
    launcher = root / "esbench-code"
    launcher.write_text(
        "#!/usr/bin/env python3\n"
        "from embodied_skill_bench.primitive_code_agent_cli import main\n\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    system_instructions = {
        "frozen-sam3": PRIMITIVE_CODE_BENCHMARK_SYSTEM_INSTRUCTIONS,
        "observation-only": PRIMITIVE_CODE_OBSERVATION_ONLY_BENCHMARK_SYSTEM_INSTRUCTIONS,
        "observation-only-p1": PRIMITIVE_CODE_P1_BENCHMARK_SYSTEM_INSTRUCTIONS,
        "calibrated-geometry-p2": PRIMITIVE_CODE_P2_BENCHMARK_SYSTEM_INSTRUCTIONS,
    }[perception_mode]
    _write_benchmark_system(root, system_instructions)
    return root


def stage_native_typed_workspace(
    *, workspace: str | Path, run_context: dict[str, Any], protocol: str
) -> Path:
    """Stage a documentation-only workspace; native tools live in the harness."""

    if protocol not in {
        P25_PROTOCOL,
        P3_PROTOCOL,
        RAW_PI05_PROTOCOL,
        RAW_PI05_RGB_PROTOCOL,
    }:
        raise ValueError("unsupported native typed protocol")
    root = Path(workspace).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / ".agents" / "skills").mkdir(parents=True)
    (root / ".claude" / "skills").mkdir(parents=True)
    atomic_write_json(root / "installed_skills.json", [])
    agent_context = {
        key: value for key, value in run_context.items()
        if key not in {"vla_checkpoint_path", "sam3_checkpoint_path"}
    }
    atomic_write_json(root / "run_context.json", agent_context)
    atomic_write_json(root / "native_typed_tools.json", contract_document(protocol))
    (root / "NATIVE_TYPED_TOOLS.md").write_text(
        contract_markdown(protocol), encoding="utf-8"
    )
    (root / "NO_SKILL_ROBOT_REFERENCE.md").write_text(
        NO_SKILL_ROBOT_REFERENCE_MARKDOWN, encoding="utf-8"
    )
    _write_benchmark_system(
        root,
        RAW_PI05_RGB_BENCHMARK_SYSTEM_INSTRUCTIONS
        if protocol == RAW_PI05_RGB_PROTOCOL
        else RAW_PI05_BENCHMARK_SYSTEM_INSTRUCTIONS
        if protocol == RAW_PI05_PROTOCOL
        else NATIVE_TYPED_BENCHMARK_SYSTEM_INSTRUCTIONS,
    )
    return root


def stage_native_skill_workspace(
    *,
    workspace: str | Path,
    catalog: SkillCatalog,
    selected_skill_ids: list[str],
    run_context: dict[str, Any],
) -> Path:
    """Stage a shell-free selected-skill workspace and manifest-derived tools."""

    root = Path(workspace).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / ".agents" / "skills").mkdir(parents=True)
    (root / ".claude" / "skills").mkdir(parents=True)
    records = catalog.require(selected_skill_ids)
    installed = [record.public() | {"sha256": record.hash} for record in records]
    atomic_write_json(root / "installed_skills.json", installed)
    contract = build_native_skill_contract(records)
    contract_path = root / "native_skill_tools.json"
    atomic_write_json(contract_path, contract)
    guide_path = root / "NATIVE_SKILL_LIBRARY.md"
    guide_path.write_text(render_native_skill_guide(records), encoding="utf-8")
    agent_context = {
        key: value
        for key, value in run_context.items()
        if key not in {"vla_checkpoint_path", "sam3_checkpoint_path"}
    }
    agent_context.update(
        {
            "native_skill_protocol": NATIVE_SKILL_PROTOCOL,
            "native_skill_contract_sha256": sha256_file(contract_path),
            "native_skill_guide_sha256": sha256_file(guide_path),
        }
    )
    atomic_write_json(root / "run_context.json", agent_context)
    _write_benchmark_system(root, NATIVE_SKILL_BENCHMARK_SYSTEM_INSTRUCTIONS)
    return root


def stage_selector_workspace(
    *,
    workspace: str | Path,
    catalog: SkillCatalog,
    task: dict[str, Any],
    skill_exposure: str,
) -> Path:
    root = Path(workspace).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    if skill_exposure not in {"metadata_only", "skill_md"}:
        raise ValueError(
            "selection.skill_exposure must be 'metadata_only' or 'skill_md'"
        )
    catalog_rows = catalog.public_rows()
    if skill_exposure == "skill_md":
        candidate_root = root / "candidate-skills"
        candidate_root.mkdir(parents=True)
        indexed_rows = []
        for record in sorted(catalog, key=lambda item: item.skill_id):
            destination = candidate_root / record.skill_id
            destination.mkdir()
            shutil.copy2(record.path / "SKILL.md", destination / "SKILL.md")
            shutil.copy2(record.path / "metadata.json", destination / "metadata.json")
            row = record.public()
            row["skill_md"] = str(
                (destination / "SKILL.md").relative_to(root)
            )
            indexed_rows.append(row)
        catalog_rows = indexed_rows
    (root / "catalog.json").write_text(
        json.dumps(catalog_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "task.json").write_text(
        json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    atomic_write_json(
        root / "selector_contract.json",
        {
            "schema_version": 1,
            "skill_exposure": skill_exposure,
            "candidate_scripts_exposed": False,
            "candidate_assets_exposed": False,
            "simulator_available": False,
        },
    )
    _write_benchmark_system(root, SELECTION_BENCHMARK_SYSTEM_INSTRUCTIONS)
    return root

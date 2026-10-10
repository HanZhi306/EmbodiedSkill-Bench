from __future__ import annotations

import json
import re
import shutil
import threading
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..catalog import SkillCatalog, validate_json_schema_subset
from ..io import append_jsonl
from .grounded_joint import GROUNDED_JOINT_SKILL_IDS
from .grounded_osc import GROUNDED_OSC_SKILL_IDS
from .store import ObservationStore, utc_now
from .native_skill_contract import validate_native_skill_contract
from .perception import build_sam3_action_handoff
from .semantic_vla import INTERNAL_GROUNDING_HANDOFFS_KEY
from .turn_authority import AgentTraceTurnAuthority
from .types import (
    ActionResult,
    Backend,
    InvalidSkillArgumentsError,
    PerceptionResult,
    UnsupportedSkillError,
)


_GROUNDING_MATCH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "current",
        "visible",
        "instance",
        "object",
        "item",
    }
)


def _grounding_match_tokens(value: Any) -> frozenset[str]:
    return frozenset(
        token
        for token in re.findall(r"[a-z0-9]+", str(value).lower())
        if token not in _GROUNDING_MATCH_STOPWORDS
    )


def _grounding_phrase_matches_argument(target_phrase: Any, argument: Any) -> bool:
    """Match one mask query to one manifest-declared semantic argument."""

    target_tokens = _grounding_match_tokens(target_phrase)
    argument_tokens = _grounding_match_tokens(argument)
    if not target_tokens or not argument_tokens:
        return False
    # The query is normally the concise identity and the action argument may
    # add one stable landmark. Accept either containment direction so a more
    # specific perception phrase can also refine a concise action identity.
    return target_tokens <= argument_tokens or argument_tokens <= target_tokens


def _canonical_call_argument(value: Any) -> str:
    """Produce a stable, task-agnostic key for a manifest-limited argument.

    Strings are case/outer-whitespace insensitive, while structured values use
    canonical JSON.  The runtime intentionally does not infer semantic aliases:
    the manifest still owns which typed fields define one retriable subgoal.
    """

    if isinstance(value, str):
        return " ".join(value.casefold().split())
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _mirror_public_phrase_for_vla(value: Any) -> str:
    """Mirror horizontal direction words from public RGB into the VLA view."""

    phrase = " ".join(str(value).split())
    placeholder = "esb_horizontal_mirror_placeholder"
    phrase = re.sub(r"\bleft\b", placeholder, phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\bright\b", "left", phrase, flags=re.IGNORECASE)
    return re.sub(placeholder, "right", phrase, flags=re.IGNORECASE)


def _bind_handoff_phrase_to_action_argument(
    card: dict[str, Any], argument: Any
) -> dict[str, Any]:
    """Render the reference with the action's canonical semantic identity.

    SAM3 establishes the mask and image region. The manifest-declared action
    field establishes the task identity. If the successful perception phrase
    is strictly more descriptive than that action identity, retain its visual
    descriptors instead of collapsing the handoff back to the ambiguous short
    name. Horizontal words are mirrored once into the policy's image frame.
    Coordinates remain audit-only.
    """

    phrase = " ".join(str(argument).split())
    region = " ".join(str(card.get("vla_consumed_image_region", "")).split())
    camera = " ".join(str(card.get("camera", "agentview")).split()) or "agentview"
    if not phrase or not region:
        return dict(card)
    bound = dict(card)
    segmentation_phrase = " ".join(str(card.get("target_phrase", "")).split())
    bound["segmentation_target_phrase"] = segmentation_phrase
    bound["matched_argument_value"] = phrase
    segmentation_tokens = _grounding_match_tokens(segmentation_phrase)
    action_tokens = _grounding_match_tokens(phrase)
    if action_tokens < segmentation_tokens:
        descriptor = _mirror_public_phrase_for_vla(segmentation_phrase)
        bound["grounding_description_source"] = (
            "successful-perception-phrase-mirrored-to-vla-frame"
        )
        bound["vla_grounding_descriptor"] = descriptor
        bound["visual_referring_expression"] = (
            f'the "{phrase}" instance visually matching "{descriptor}" in the '
            f"{region} region of the current VLA-consumed {camera} image"
        )
    else:
        bound["grounding_description_source"] = "matched-action-argument"
        bound["vla_grounding_descriptor"] = phrase
        bound["visual_referring_expression"] = (
            f'the "{phrase}" instance in the {region} region of the current '
            f"VLA-consumed {camera} image"
        )
    return bound


def _matching_action_handoffs(
    cards: list[dict[str, Any]],
    *,
    step_index: int,
    grounding_fields: list[Any] | tuple[Any, ...],
    arguments: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select at most one same-step, phrase-matched card per target field."""

    selected: list[dict[str, Any]] = []
    for raw_field in grounding_fields:
        field = str(raw_field)
        argument = arguments.get(field)
        if argument in {None, ""}:
            continue
        matches = [
            card
            for card in cards
            if int(card.get("source_observation_step", -1)) == step_index
            and _grounding_phrase_matches_argument(
                card.get("target_phrase"), argument
            )
        ]
        if not matches:
            continue
        chosen = _bind_handoff_phrase_to_action_argument(matches[-1], argument)
        chosen["matched_argument_field"] = field
        selected.append(chosen)
    return selected[-2:]


_PRIVATE_GROUNDING_RESULT_FIELDS = frozenset(
    {
        "box_xyxy",
        "centroid_pixel_rc",
        "mask_path",
        "molmo2_point_pixel_rc",
        "molmo2_query_json",
        "query_json",
        "sam3_point_pixel_rc",
        "sam3_query_json",
        "source_image_path",
        "source_world_map_path",
        "world_bounds_p05_m",
        "world_bounds_p95_m",
        "world_center_median_m",
        "world_extent_p90_m",
    }
)
_PRIVATE_GROUNDING_HANDOFF_FIELDS = frozenset(
    {
        "mask_box_xyxy",
        "mask_centroid_pixel_rc",
        "world_bounds_p05_m",
        "world_bounds_p95_m",
        "world_center_median_m",
    }
)


def _coordinate_free_grounding_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return the public view for a host-private metric controller.

    The full result has already been converted into an internal grounding card
    before this helper runs.  A grounded OSC skill may consume that card, but
    its Agent must not receive pixels, world coordinates, raw mask paths, or
    private query paths through the preceding perception response.
    """

    public = {
        key: value
        for key, value in result.items()
        if key not in _PRIVATE_GROUNDING_RESULT_FIELDS
    }
    handoff = public.get("action_handoff")
    if isinstance(handoff, dict):
        public["action_handoff"] = {
            key: value
            for key, value in handoff.items()
            if key not in _PRIVATE_GROUNDING_HANDOFF_FIELDS
        }
    public["metric_geometry_visibility"] = "host-private-controller-only"
    return public


class EpisodeSession:
    def __init__(
        self,
        *,
        episode_id: str,
        run_dir: str | Path,
        backend: Backend,
        catalog: SkillCatalog,
        allowed_skill_ids: list[str],
        max_actions: int,
        max_perception_queries: int = 32,
        enforce_agent_protocol: bool = True,
        turn_authority: AgentTraceTurnAuthority | None = None,
        native_skill_contract: dict[str, Any] | None = None,
        max_identical_action_calls: int = 0,
    ):
        self.episode_id = episode_id
        self.run_dir = Path(run_dir).resolve()
        self.backend = backend
        self.catalog = catalog
        self.allowed_skill_ids = frozenset(allowed_skill_ids)
        self.max_actions = int(max_actions)
        self.max_perception_queries = int(max_perception_queries)
        if self.max_perception_queries < 0:
            raise ValueError("max_perception_queries must be non-negative")
        self.enforce_agent_protocol = bool(enforce_agent_protocol)
        self.turn_authority = turn_authority
        if max_identical_action_calls < 0:
            raise ValueError("max_identical_action_calls must be non-negative")
        self.max_identical_action_calls = int(max_identical_action_calls)
        self._native_skill_contract = (
            validate_native_skill_contract(native_skill_contract)
            if native_skill_contract is not None
            else None
        )
        self._native_tool_to_skill: dict[str, str] = {}
        if self._native_skill_contract is not None:
            if self._native_skill_contract["selected_skill_ids"] != list(allowed_skill_ids):
                raise ValueError("native skill contract selected set/order drift")
            for item in self._native_skill_contract["tools"]:
                record = catalog.get(str(item["skill_id"]))
                if (
                    record.hash != item["skill_sha256"]
                    or record.kind != item["kind"]
                    or record.arguments_schema != item["parameters"]
                ):
                    raise ValueError(
                        f"native skill contract drift for {record.skill_id}"
                    )
                self._native_tool_to_skill[str(item["tool_name"])] = record.skill_id
        self.store = ObservationStore(self.run_dir / "episode", episode_id)
        self._lock = threading.Lock()
        self._responses: dict[str, dict[str, Any]] = {}
        self._step_index = 0
        self._action_count = 0
        self._perception_count = 0
        self._closed = False
        self._acknowledged_step: int | None = None
        self._last_action_turn_id: str | None = None
        self._perception_turn_ids: set[str] = set()
        self._protocol_violations: list[dict[str, Any]] = []
        self._last_action_signature: str | None = None
        self._identical_action_streak = 0
        self._repeated_action_guard_rejections = 0
        self._action_calls_by_skill: Counter[str] = Counter()
        self._action_calls_by_argument_tuple: Counter[
            tuple[str, tuple[tuple[str, str], ...]]
        ] = Counter()
        self._skill_call_limit_rejections = 0
        self._skill_argument_call_limit_rejections = 0
        self._pending_selected_perception_fallback: dict[str, Any] | None = None
        self._selected_perception_fallback_rejections = 0
        self._sam3_text_prompt_by_step: dict[int, str] = {}
        self._sam3_action_handoffs: list[dict[str, Any]] = []
        self._capture_initial()
        if self._native_skill_contract is not None:
            self._acknowledge_observation("initial_native_skill_attachment")

    def _capture_initial(self) -> None:
        self.backend.reset()
        temporary, final_dir = self.store.begin(0)
        try:
            captured = self.backend.capture(temporary, 0)
            self.store.commit(temporary, final_dir, 0, captured, None)
        except BaseException:
            self.store.abort(temporary)
            raise
        self._event("reset", {"step_index": 0})

    def latest(
        self, *, acknowledge: bool = False, agent_turn_id: str | None = None
    ) -> dict[str, Any]:
        protocol_error: dict[str, Any] | None = None
        if acknowledge:
            with self._lock:
                canonical_turn_id = agent_turn_id
                if self.enforce_agent_protocol and self.turn_authority:
                    canonical_turn_id = self.turn_authority.active_turn("observe")
                    if not canonical_turn_id:
                        protocol_error = self._record_protocol_violation(
                            code="unverified_observation_tool_turn",
                            reason="observation acknowledgement was not issued by an active `./esbench observe` LLM tool turn",
                            skill_id=None,
                        )
                if not protocol_error:
                    self._acknowledge_observation(
                        "runtime_observe", canonical_turn_id
                    )
        return {
            "ok": True,
            "episode_id": self.episode_id,
            "observation": self.store.latest(),
            "pointer": self.store.latest_pointer(),
            "observation_acknowledged": self._acknowledged_step == self._step_index,
            "protocol_error": protocol_error,
        }

    def _acknowledge_observation(
        self, source: str, agent_turn_id: str | None = None
    ) -> None:
        if self._acknowledged_step == self._step_index:
            return
        self._acknowledged_step = self._step_index
        self._event(
            "observation_acknowledged",
            {
                "step_index": self._step_index,
                "source": source,
                "agent_turn_id": agent_turn_id,
            },
        )

    def _record_protocol_violation(
        self, *, code: str, reason: str, skill_id: str | None
    ) -> dict[str, Any]:
        violation = {
            "code": code,
            "message": reason,
            "step_index": self._step_index,
            "skill_id": skill_id,
        }
        self._protocol_violations.append(violation)
        self._event("agent_protocol_violation", violation)
        return violation

    def _protocol_reject(
        self,
        request_id: str,
        skill_id: str,
        code: str,
        reason: str,
        *,
        event_kind: str = "action_rejected",
    ) -> dict[str, Any]:
        self._record_protocol_violation(
            code=code, reason=reason, skill_id=skill_id
        )
        return self._reject(
            request_id,
            skill_id,
            reason,
            reason_code=code,
            event_kind=event_kind,
        )

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = str(request.get("request_id") or uuid.uuid4())
        with self._lock:
            if request_id in self._responses:
                cached = dict(self._responses[request_id])
                cached["idempotent_replay"] = True
                return cached
            response = self._execute_once(request_id, request)
            self._responses[request_id] = response
            return response

    def execute_native_skill(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute one selected manifest-owned skill through a native tool call."""

        request_id = str(request.get("request_id") or uuid.uuid4())
        with self._lock:
            if request_id in self._responses:
                cached = dict(self._responses[request_id])
                cached["idempotent_replay"] = True
                return cached
            tool_name = request.get("tool_name")
            if (
                self._native_skill_contract is None
                or not isinstance(tool_name, str)
                or tool_name not in self._native_tool_to_skill
            ):
                response = self._reject(
                    request_id,
                    str(tool_name or ""),
                    "unknown or unavailable native selected-skill tool",
                    reason_code="unknown_native_skill_tool",
                )
                self._responses[request_id] = response
                return response
            claimed_turn = str(request.get("agent_turn_id") or "")
            canonical_turn = claimed_turn
            if self.enforce_agent_protocol:
                if not claimed_turn:
                    response = self._protocol_reject(
                        request_id,
                        self._native_tool_to_skill[tool_name],
                        "missing_agent_turn_id",
                        "native selected-skill request is missing harness turn identity",
                    )
                    self._responses[request_id] = response
                    return response
                if self.turn_authority:
                    canonical_turn = self.turn_authority.active_turn(
                        "native_tool", tool_name
                    ) or ""
                    if not canonical_turn:
                        response = self._protocol_reject(
                            request_id,
                            self._native_tool_to_skill[tool_name],
                            "unverified_native_skill_tool_turn",
                            "request does not match an active OpenHands native selected-skill call",
                        )
                        self._responses[request_id] = response
                        return response
            response = self._execute_once(
                request_id,
                {
                    "skill_id": self._native_tool_to_skill[tool_name],
                    "arguments": request.get("arguments"),
                    "agent_turn_id": claimed_turn,
                },
                verified_agent_turn_id=canonical_turn,
                automatic_observation_handoff=True,
                native_tool_name=tool_name,
            )
            self._responses[request_id] = response
            return response

    def execute_adapter(
        self,
        *,
        request_id: str,
        adapter_skill_id: str,
        public_operation_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a host-only controller adapter under a public operation label.

        HTTP clients can only call :meth:`execute`, so they cannot select an
        adapter or smuggle one into the public observation/audit stream.
        """

        with self._lock:
            if request_id in self._responses:
                cached = dict(self._responses[request_id])
                cached["idempotent_replay"] = True
                return cached
            response = self._execute_once(
                request_id,
                {"skill_id": public_operation_id, "arguments": arguments},
                backend_skill_id=adapter_skill_id,
            )
            self._responses[request_id] = response
            return response

    def execute_p3_joint_action(
        self,
        *,
        request_id: str,
        public_operation_id: str,
        arguments: dict[str, Any],
        executor: Callable[[], ActionResult],
    ) -> dict[str, Any]:
        """Record one host-only P3 joint-controller action and fresh capture.

        This entry point is intentionally absent from the HTTP server.  Native
        P3 first validates its JSON schema, IK/FK residuals, and the frozen
        controller attestation, then supplies the in-process executor.  Keeping
        observation capture here preserves the same atomic episode ledger used
        by every other action instead of trusting a backend self-report.
        """

        with self._lock:
            if request_id in self._responses:
                cached = dict(self._responses[request_id])
                cached["idempotent_replay"] = True
                return cached
            if self._closed:
                response = self._reject(request_id, public_operation_id, "runtime is closed")
            elif self.store.latest()["status"].get("terminated") or self.store.latest()["status"].get("truncated"):
                response = self._reject(request_id, public_operation_id, "episode is already finished")
            elif self._action_count >= self.max_actions:
                response = self._reject(
                    request_id,
                    public_operation_id,
                    f"action limit reached ({self.max_actions})",
                )
            else:
                response = self._execute_p3_joint_action_once(
                    request_id=request_id,
                    public_operation_id=public_operation_id,
                    arguments=arguments,
                    executor=executor,
                )
            self._responses[request_id] = response
            return response

    def _execute_p3_joint_action_once(
        self,
        *,
        request_id: str,
        public_operation_id: str,
        arguments: dict[str, Any],
        executor: Callable[[], ActionResult],
    ) -> dict[str, Any]:
        self._action_count += 1
        next_step = self._step_index + 1
        started = time.monotonic()
        result: ActionResult | None = None
        error: dict[str, Any] | None = None
        try:
            result = executor()
            if not isinstance(result, ActionResult):
                raise TypeError("P3 joint executor must return ActionResult")
            if result.result.get("execution_verified") is not True:
                error = {
                    "type": "ControllerGoalNotReached",
                    "code": "controller_goal_not_reached",
                    "message": (
                        "the physical JOINT_POSITION action ran, but final measured "
                        "state failed the controller/joint-limit/FK residual gate"
                    ),
                    "traceback": None,
                }
        except (UnsupportedSkillError, InvalidSkillArgumentsError) as exc:
            self._action_count -= 1
            return self._reject(request_id, public_operation_id, str(exc))
        except BaseException as exc:
            error = {
                "type": type(exc).__name__,
                "code": "simulator_failure",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        elapsed = time.monotonic() - started
        simulator_steps = int(result.simulator_steps) if result is not None else 0
        state_changed = simulator_steps > 0
        execution = {
            "request_id": request_id,
            "skill_id": public_operation_id,
            "arguments": arguments,
            "agent_turn_id": None,
            "claimed_agent_turn_id": None,
            "accepted": True,
            "result": (
                result.result
                if result
                else {"error": error["message"] if error else "unknown action error"}
            ),
            "error": error,
            "elapsed_seconds": elapsed,
            "simulator_steps": simulator_steps,
            "vla_chunks": 0,
            "controller_path": "p3-audited-joint-position",
            "physical_action_attempted": True,
            "state_changed": state_changed,
        }
        temporary, final_dir = self.store.begin(next_step)
        try:
            captured = self.backend.capture(temporary, next_step)
            self.store.commit(temporary, final_dir, next_step, captured, execution)
        except BaseException:
            self.store.abort(temporary)
            raise
        self._step_index = next_step
        self._acknowledged_step = None
        self._event(
            (
                "action_unverified"
                if error and error.get("code") == "controller_goal_not_reached"
                else "action_error"
                if error
                else "action_completed"
            ),
            execution | {"step_index": next_step},
        )
        pointer = self.store.latest_pointer()
        return {
            "ok": error is None,
            "accepted": True,
            "request_id": request_id,
            "skill_id": public_operation_id,
            "result": execution["result"],
            "error": error,
            "physical_action_attempted": True,
            "state_changed": state_changed,
            "fresh_observation_handoff": True,
            "step_index": next_step,
            "observation_json": pointer["observation_json"],
            "observation_markdown": pointer["observation_markdown"],
            "status": self.store.latest()["status"],
        }

    def _execute_once(
        self,
        request_id: str,
        request: dict[str, Any],
        *,
        backend_skill_id: str | None = None,
        verified_agent_turn_id: str | None = None,
        automatic_observation_handoff: bool = False,
        native_tool_name: str | None = None,
    ) -> dict[str, Any]:
        skill_id = str(request.get("skill_id", ""))
        backend_id = str(backend_skill_id or skill_id)
        arguments = request.get("arguments", {})
        claimed_turn_id = str(request.get("agent_turn_id") or "")
        agent_turn_id = claimed_turn_id
        if not isinstance(arguments, dict):
            return self._reject(request_id, skill_id, "arguments must be a JSON object")
        if self._closed:
            return self._reject(request_id, skill_id, "runtime is closed")
        latest_status = self.store.latest()["status"]
        if latest_status.get("terminated") or latest_status.get("truncated"):
            return self._reject(request_id, skill_id, "episode is already finished")
        if backend_id not in self.allowed_skill_ids:
            return self._reject(
                request_id,
                skill_id,
                "skill is not installed/allowed for this episode",
            )
        try:
            record = self.catalog.get(backend_id)
        except KeyError as exc:
            return self._reject(request_id, skill_id, str(exc))
        if record.kind == "experience":
            return self._reject(request_id, skill_id, "experience skills cannot execute")
        if "*" not in record.environments:
            if self.backend.environment_id not in record.environments:
                return self._reject(
                    request_id,
                    skill_id,
                    f"skill is unavailable in {self.backend.environment_id}",
                )
        try:
            validate_json_schema_subset(arguments, record.arguments_schema or {})
        except ValueError as exc:
            return self._reject(request_id, skill_id, str(exc))

        operation = "action" if record.kind == "action" else "perception"
        rejection_event = (
            "action_rejected" if operation == "action" else "perception_rejected"
        )
        if self.enforce_agent_protocol:
            if not claimed_turn_id:
                return self._protocol_reject(
                    request_id,
                    skill_id,
                    "missing_agent_turn_id",
                    f"{operation} request is missing the harness-issued agent turn identity",
                    event_kind=rejection_event,
                )
            if verified_agent_turn_id is not None:
                agent_turn_id = verified_agent_turn_id
            elif self.turn_authority:
                verified_turn_id = self.turn_authority.active_turn(
                    operation, backend_id
                )
                if not verified_turn_id:
                    return self._protocol_reject(
                        request_id,
                        skill_id,
                        f"unverified_{operation}_tool_turn",
                        f"{operation} request does not match an active staged skill command in the LLM tool trace",
                        event_kind=rejection_event,
                    )
                agent_turn_id = verified_turn_id
            if operation == "action" and agent_turn_id == self._last_action_turn_id:
                return self._protocol_reject(
                    request_id,
                    skill_id,
                    "multiple_actions_in_one_turn",
                    "one agent tool turn may execute at most one action skill",
                )
            if operation == "action" and agent_turn_id in self._perception_turn_ids:
                return self._protocol_reject(
                    request_id,
                    skill_id,
                    "action_after_perception_in_same_turn",
                    "consume perception results before issuing an action in a later agent turn",
                )
            if operation == "perception" and agent_turn_id == self._last_action_turn_id:
                return self._protocol_reject(
                    request_id,
                    skill_id,
                    "perception_after_action_in_same_turn",
                    "do not launch perception in parallel with an action; query it before the action round ends",
                    event_kind=rejection_event,
                )
            if self._acknowledged_step != self._step_index:
                return self._protocol_reject(
                    request_id,
                    skill_id,
                    "observation_not_acknowledged",
                    "run `./esbench observe` to acknowledge the current observation before perception or action",
                    event_kind=rejection_event,
                )

        if record.kind == "perception":
            if self._perception_count >= self.max_perception_queries:
                return self._reject(
                    request_id,
                    skill_id,
                    f"perception query limit reached ({self.max_perception_queries})",
                    event_kind="perception_rejected",
                )
            response = self._execute_perception(
                request_id=request_id,
                skill_id=skill_id,
                arguments=arguments,
                claimed_turn_id=claimed_turn_id,
                agent_turn_id=agent_turn_id,
            )
            if automatic_observation_handoff:
                response.update(
                    {
                        "native_tool_name": native_tool_name,
                        "state_changed": False,
                        "physical_round_ended": False,
                        "automatic_observation_handoff": False,
                    }
                )
            return response
        if (
            automatic_observation_handoff
            and self._pending_selected_perception_fallback is not None
            and int(self._pending_selected_perception_fallback["step_index"])
            == self._step_index
        ):
            self._selected_perception_fallback_rejections += 1
            pending = self._pending_selected_perception_fallback
            required_tool = str(
                pending.get("tool_name") or "selected SAM3 perception tool"
            )
            return self._reject(
                request_id,
                skill_id,
                (
                    "selected perception composition is unresolved on this "
                    "observation. Call the selected "
                    f"{required_tool} tool on a better interior target point and "
                    "inspect its result before a state-changing skill"
                ),
                reason_code="selected_perception_fallback_required",
            ) | {"required_fallback": dict(pending)}
        if self._action_count >= self.max_actions:
            return self._reject(
                request_id,
                skill_id,
                f"action limit reached ({self.max_actions})",
                reason_code="action_budget_exhausted",
                terminal_for_agent=automatic_observation_handoff,
            )
        per_skill_limit = (record.execution or {}).get("max_calls_per_episode")
        if (
            per_skill_limit is not None
            and self._action_calls_by_skill[backend_id] >= int(per_skill_limit)
        ):
            self._skill_call_limit_rejections += 1
            return self._reject(
                request_id,
                skill_id,
                (
                    f"{backend_id} reached its manifest-defined per-episode call "
                    f"limit ({per_skill_limit}); do not evade the bound by changing "
                    "wording. Choose a different selected recovery/next action or "
                    "finish if no safe progress remains"
                ),
                reason_code="skill_call_limit_exhausted",
            )
        argument_limit = (record.execution or {}).get(
            "max_calls_per_argument_tuple"
        )
        argument_call_key: tuple[
            str, tuple[tuple[str, str], ...]
        ] | None = None
        if argument_limit is not None:
            argument_call_key = (
                backend_id,
                tuple(
                    (
                        str(field),
                        _canonical_call_argument(arguments[str(field)]),
                    )
                    for field in argument_limit["fields"]
                ),
            )
            if (
                self._action_calls_by_argument_tuple[argument_call_key]
                >= int(argument_limit["limit"])
            ):
                self._skill_call_limit_rejections += 1
                self._skill_argument_call_limit_rejections += 1
                field_names = ", ".join(argument_limit["fields"])
                return self._reject(
                    request_id,
                    skill_id,
                    (
                        f"{backend_id} reached its manifest-defined call limit "
                        f"({argument_limit['limit']}) for the same normalized "
                        f"argument tuple ({field_names}). Consume fresh evidence "
                        "and choose a different object/subgoal or recovery; do not "
                        "repeat the exhausted target"
                    ),
                    reason_code="skill_argument_call_limit_exhausted",
                )

        grounding_fields = (record.execution or {}).get(
            "grounding_handoff_fields"
        )
        action_handoffs = _matching_action_handoffs(
            self._sam3_action_handoffs,
            step_index=self._step_index,
            grounding_fields=list(grounding_fields or []),
            arguments=arguments,
        )
        backend_arguments = dict(arguments)
        provider = (record.execution or {}).get("provider")
        # Semantic grounding belongs to the selected skill, not to the native
        # tool's automatic RGB/acknowledgement delivery. Classic script calls
        # consume the same validated cards but still require a later observe.
        # Keep the existing native-only controller-adapter boundary unchanged.
        if action_handoffs and (
            provider == "semantic-vla"
            or (
                automatic_observation_handoff
                and provider in {"grounded-osc-pick-place", "grounded-joint-pyroki"}
            )
        ):
            backend_arguments[INTERNAL_GROUNDING_HANDOFFS_KEY] = action_handoffs

        signature_arguments = {
            key: value for key, value in arguments.items() if key != "max_chunks"
        }
        if action_handoffs:
            signature_arguments["automatic_grounding_refs"] = [
                card["target_ref"] for card in action_handoffs
            ]
        action_signature = json.dumps(
            [backend_id, signature_arguments],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prospective_streak = (
            self._identical_action_streak + 1
            if action_signature == self._last_action_signature
            else 1
        )
        if (
            self.max_identical_action_calls
            and prospective_streak > self.max_identical_action_calls
        ):
            self._repeated_action_guard_rejections += 1
            return self._reject(
                request_id,
                skill_id,
                (
                    "identical state-changing skill and semantic arguments reached "
                    f"the evidence-free retry limit ({self.max_identical_action_calls}); "
                    "consume the fresh observation and change the referring expression, "
                    "choose a recovery/next skill, or finish if no safe progress remains"
                ),
                reason_code="repeated_action_guard",
            )

        self._action_count += 1
        self._action_calls_by_skill[backend_id] += 1
        if argument_call_key is not None:
            self._action_calls_by_argument_tuple[argument_call_key] += 1
        next_step = self._step_index + 1
        started = time.monotonic()
        result: ActionResult | None = None
        error: dict[str, Any] | None = None
        try:
            result = self.backend.execute(backend_id, backend_arguments)
        except (UnsupportedSkillError, InvalidSkillArgumentsError) as exc:
            self._action_count -= 1
            self._action_calls_by_skill[backend_id] -= 1
            if argument_call_key is not None:
                self._action_calls_by_argument_tuple[argument_call_key] -= 1
            return self._reject(request_id, skill_id, str(exc))
        except BaseException as exc:
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        # Only an action that reached the backend can consume the causal LLM
        # turn or contribute to the repeated-action streak. Pure argument and
        # capability rejections above have no simulator side effect and must
        # not poison the next valid decision.
        self._last_action_turn_id = agent_turn_id or None
        self._last_action_signature = action_signature
        self._identical_action_streak = prospective_streak
        elapsed = time.monotonic() - started
        execution = {
            "request_id": request_id,
            "skill_id": skill_id,
            "arguments": arguments,
            "agent_turn_id": agent_turn_id,
            "claimed_agent_turn_id": claimed_turn_id,
            "accepted": True,
            "result": result.result if result else {"error": error["message"]},
            "error": error,
            "elapsed_seconds": elapsed,
            "simulator_steps": result.simulator_steps if result else None,
            "vla_chunks": result.vla_chunks if result else None,
        }
        temporary, final_dir = self.store.begin(next_step)
        try:
            captured = self.backend.capture(temporary, next_step)
            self.store.commit(
                temporary, final_dir, next_step, captured, execution
            )
        except BaseException:
            self.store.abort(temporary)
            raise
        self._step_index = next_step
        self._acknowledged_step = None
        event_kind = "action_error" if error else "action_completed"
        self._event(event_kind, execution | {"step_index": next_step})
        pointer = self.store.latest_pointer()
        response = {
            "ok": error is None,
            "accepted": True,
            "request_id": request_id,
            "skill_id": skill_id,
            "result": execution["result"],
            "error": error,
            "step_index": next_step,
            "observation_json": pointer["observation_json"],
            "observation_markdown": pointer["observation_markdown"],
            "status": self.store.latest()["status"],
        }
        if automatic_observation_handoff:
            self._acknowledge_observation(
                "native_skill_fresh_observation_handoff", agent_turn_id or None
            )
            response.update(
                {
                    "native_tool_name": native_tool_name,
                    "state_changed": True,
                    "physical_round_ended": True,
                    "automatic_observation_handoff": True,
                    "observation": self.store.latest(),
                    "observation_acknowledged": True,
                }
            )
        return response

    def _execute_perception(
        self,
        *,
        request_id: str,
        skill_id: str,
        arguments: dict[str, Any],
        claimed_turn_id: str,
        agent_turn_id: str,
    ) -> dict[str, Any]:
        query_index = self._perception_count + 1
        destination = (
            self.store.episode_dir
            / "perception"
            / f"step-{self._step_index:04d}"
            / f"query-{query_index:04d}"
        )
        if destination.exists():
            raise RuntimeError(f"perception artifact path already exists: {destination}")
        started = time.monotonic()
        result: PerceptionResult | None = None
        error: dict[str, Any] | None = None
        if agent_turn_id:
            # Claim the whole canonical model turn before backend work starts.
            # This keeps a parallel action from racing a slow perception query.
            self._perception_turn_ids.add(agent_turn_id)
        try:
            result = self.backend.perceive(
                skill_id,
                arguments,
                self.store.latest(),
                destination,
            )
        except UnsupportedSkillError as exc:
            shutil.rmtree(destination, ignore_errors=True)
            return self._reject(
                request_id,
                skill_id,
                str(exc),
                event_kind="perception_rejected",
            )
        except BaseException as exc:
            shutil.rmtree(destination, ignore_errors=True)
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        self._perception_count += 1
        elapsed = time.monotonic() - started
        result_payload = result.result if result else {"error": error["message"]}
        if error is None and skill_id == "sam3-localize-object":
            prompt = " ".join(str(arguments.get("prompt", "")).split())
            if prompt:
                self._sam3_text_prompt_by_step[self._step_index] = prompt
        if (
            error is None
            and skill_id == "sam3-localize-object"
            and result_payload.get("found") is False
            and "sam3-segment-point" in self.allowed_skill_ids
        ):
            # This is selected-skill composition guidance, not a semantic scene
            # oracle.  It is emitted only when both manifests are in the curated
            # package and the text provider has already returned public negative
            # evidence.  The Agent still chooses and verifies the point itself.
            result_payload = dict(result_payload)
            result_payload["selected_skill_fallback"] = {
                "tool_name": "sam3_segment_point",
                "reason": "text localization returned found=false",
                "instruction": (
                    "Inspect the current attached 256x256 agentview RGB, choose a "
                    "pixel well inside a plausible target instance, and verify the "
                    "returned overlay before any physical action. Do not replace a "
                    "failed relational prompt with an ambiguous generic category."
                ),
                "semantic_identity_proven": False,
            }
            self._pending_selected_perception_fallback = {
                "tool_name": "sam3_segment_point",
                "step_index": self._step_index,
                "source_skill_id": skill_id,
                "source_request_id": request_id,
                "source_prompt": self._sam3_text_prompt_by_step.get(self._step_index),
            }
        elif error is None and skill_id in {
            "sam3-segment-point",
            "sam3-bind-point-reference-v1",
        }:
            # One explicit point attempt satisfies the composition boundary.
            # Whether its mask is usable remains public evidence for the Agent;
            # the runtime does not silently convert a weak mask into identity.
            retry = result_payload.get("selected_skill_retry")
            self._pending_selected_perception_fallback = (
                {
                    "tool_name": str(retry.get("tool_name")),
                    "step_index": self._step_index,
                    "source_skill_id": skill_id,
                    "source_request_id": request_id,
                    "reason": "point mask failed the task-agnostic geometric plausibility gate",
                }
                if isinstance(retry, dict)
                else None
            )
        if error is None and skill_id == "sam3-localize-object":
            target_phrase = " ".join(str(arguments.get("prompt", "")).split())
            handoff = build_sam3_action_handoff(
                result_payload,
                target_phrase=target_phrase,
                target_ref=(
                    f"sam3://step-{self._step_index:04d}/query-{query_index:04d}"
                ),
            )
            if handoff is not None:
                result_payload = dict(result_payload)
                result_payload["action_handoff"] = handoff
                # A later valid mask for the same phrase supersedes the earlier
                # hypothesis; distinct phrases allow object+destination cards.
                self._sam3_action_handoffs = [
                    card
                    for card in self._sam3_action_handoffs
                    if not (
                        int(card.get("source_observation_step", -1)) == self._step_index
                        and card.get("target_phrase") == handoff["target_phrase"]
                    )
                ]
                self._sam3_action_handoffs.append(handoff)
        elif error is None and skill_id == "sam3-bind-point-reference-v1":
            target_phrase = " ".join(
                str(arguments.get("target_phrase", "")).split()
            )
            handoff = build_sam3_action_handoff(
                result_payload,
                target_phrase=target_phrase,
                target_ref=(
                    f"sam3://step-{self._step_index:04d}/query-{query_index:04d}"
                ),
                allow_agent_labelled_point=True,
            )
            if handoff is not None:
                result_payload = dict(result_payload)
                result_payload["action_handoff"] = handoff
                self._sam3_action_handoffs = [
                    card
                    for card in self._sam3_action_handoffs
                    if not (
                        int(card.get("source_observation_step", -1))
                        == self._step_index
                        and card.get("target_phrase") == handoff["target_phrase"]
                    )
                ]
                self._sam3_action_handoffs.append(handoff)
        elif error is None and skill_id == "molmo2-sam3-bind-object-reference-v1":
            target_phrase = " ".join(
                str(arguments.get("target_phrase", "")).split()
            )
            handoff = build_sam3_action_handoff(
                result_payload,
                target_phrase=target_phrase,
                target_ref=(
                    f"molmo2-sam3://step-{self._step_index:04d}/"
                    f"query-{query_index:04d}"
                ),
                allow_model_labelled_point=True,
            )
            if handoff is not None:
                result_payload = dict(result_payload)
                result_payload["action_handoff"] = handoff
                self._sam3_action_handoffs = [
                    card
                    for card in self._sam3_action_handoffs
                    if not (
                        int(card.get("source_observation_step", -1))
                        == self._step_index
                        and card.get("target_phrase") == handoff["target_phrase"]
                    )
                ]
                self._sam3_action_handoffs.append(handoff)
        if (
            error is None
            and (
                GROUNDED_OSC_SKILL_IDS | GROUNDED_JOINT_SKILL_IDS
            ).intersection(self.allowed_skill_ids)
            and skill_id
            in {
                "sam3-localize-object",
                "sam3-bind-point-reference-v1",
                "molmo2-sam3-bind-object-reference-v1",
            }
        ):
            result_payload = _coordinate_free_grounding_result(result_payload)
        execution = {
            "request_id": request_id,
            "skill_id": skill_id,
            "arguments": arguments,
            "agent_turn_id": agent_turn_id,
            "claimed_agent_turn_id": claimed_turn_id,
            "accepted": True,
            "result": result_payload,
            "error": error,
            "elapsed_seconds": elapsed,
            "step_index": self._step_index,
            "simulator_state_changed": False,
            "new_observation_created": False,
        }
        self._event(
            "perception_error" if error else "perception_completed",
            execution,
        )
        pointer = self.store.latest_pointer()
        return {
            "ok": error is None,
            "accepted": True,
            "request_id": request_id,
            "skill_id": skill_id,
            "result": execution["result"],
            "error": error,
            "step_index": self._step_index,
            "observation_changed": False,
            "observation_acknowledged": self._acknowledged_step == self._step_index,
            "observation_json": pointer["observation_json"],
            "observation_markdown": pointer["observation_markdown"],
            "status": self.store.latest()["status"],
        }

    def _reject(
        self,
        request_id: str,
        skill_id: str,
        reason: str,
        *,
        reason_code: str = "rejected_skill_call",
        event_kind: str = "action_rejected",
        terminal_for_agent: bool = False,
    ) -> dict[str, Any]:
        pointer = self.store.latest_pointer()
        response = {
            "ok": False,
            "accepted": False,
            "request_id": request_id,
            "skill_id": skill_id,
            "error": {
                "type": "RejectedSkillCall",
                "code": reason_code,
                "message": reason,
            },
            "step_index": pointer["step_index"],
            "observation_json": pointer["observation_json"],
            "observation_markdown": pointer["observation_markdown"],
            "status": self.store.latest()["status"],
        }
        if terminal_for_agent:
            response["terminal_for_agent"] = True
        self._event(event_kind, response)
        return response

    def _event(self, kind: str, payload: dict[str, Any]) -> None:
        append_jsonl(
            self.store.events_path,
            {"timestamp": utc_now(), "event": kind, **payload},
        )

    def finish(self, reason: str = "agent_finished") -> dict[str, Any]:
        with self._lock:
            self._event(
                "finish",
                {
                    "reason": reason,
                    "step_index": self._step_index,
                    "action_count": self._action_count,
                    "perception_count": self._perception_count,
                },
            )
            return self.summary()

    def summary(self) -> dict[str, Any]:
        latest = self.store.latest()
        counters = {
            "completed": 0,
            "rejected": 0,
            "errors": 0,
            "unverified": 0,
            "simulator_steps": 0,
            "vla_chunks": 0,
        }
        perception_counters = {"completed": 0, "rejected": 0, "errors": 0}
        skill_sequence: list[str] = []
        perception_sequence: list[str] = []
        skill_counts: Counter[str] = Counter()
        seen_error_or_rejection = False
        recovered_after_error = False
        if self.store.events_path.exists():
            for line in self.store.events_path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event["event"] == "action_completed":
                    counters["completed"] += 1
                    if seen_error_or_rejection:
                        recovered_after_error = True
                elif event["event"] == "action_rejected":
                    counters["rejected"] += 1
                    seen_error_or_rejection = True
                elif event["event"] == "action_error":
                    # P3 traces written before action_unverified was introduced
                    # used action_error for a physical controller action that
                    # ran but missed its residual gate.  Preserve those traces
                    # while keeping true simulator exceptions as infrastructure
                    # failures.
                    error = event.get("error")
                    code = error.get("code") if isinstance(error, dict) else None
                    if code == "controller_goal_not_reached":
                        counters["unverified"] += 1
                    else:
                        counters["errors"] += 1
                    seen_error_or_rejection = True
                elif event["event"] == "action_unverified":
                    counters["unverified"] += 1
                    seen_error_or_rejection = True
                elif event["event"] == "perception_completed":
                    perception_counters["completed"] += 1
                    perception_sequence.append(str(event.get("skill_id")))
                elif event["event"] == "perception_rejected":
                    perception_counters["rejected"] += 1
                elif event["event"] == "perception_error":
                    perception_counters["errors"] += 1
                    perception_sequence.append(str(event.get("skill_id")))
                if event["event"].startswith("action_") and event.get("skill_id"):
                    skill_id = str(event["skill_id"])
                    skill_sequence.append(skill_id)
                    skill_counts[skill_id] += 1
                counters["simulator_steps"] += int(event.get("simulator_steps") or 0)
                counters["vla_chunks"] += int(event.get("vla_chunks") or 0)
        return {
            "episode_id": self.episode_id,
            "success": bool(latest["status"].get("success")),
            "terminated": bool(latest["status"].get("terminated")),
            "truncated": bool(latest["status"].get("truncated")),
            "step_index": self._step_index,
            "action_count": self._action_count,
            "perception_count": self._perception_count,
            "calls": counters,
            "perception_calls": perception_counters,
            "skill_sequence": skill_sequence,
            "perception_sequence": perception_sequence,
            "skill_call_counts": dict(sorted(skill_counts.items())),
            "recovered_after_error_or_rejection": recovered_after_error,
            "protocol_enforced": self.enforce_agent_protocol,
            "protocol_turn_identity_source": self.turn_authority.source
            if self.turn_authority
            else "request_claim_for_builtin_or_direct_runtime",
            "protocol_violations": list(self._protocol_violations),
            "native_skill_protocol": (
                self._native_skill_contract.get("protocol")
                if self._native_skill_contract is not None
                else None
            ),
            "max_identical_action_calls": self.max_identical_action_calls,
            "repeated_action_guard_rejections": self._repeated_action_guard_rejections,
            "skill_call_limit_rejections": self._skill_call_limit_rejections,
            "skill_argument_call_limit_rejections": (
                self._skill_argument_call_limit_rejections
            ),
            "manifest_limited_action_call_counts": {
                skill_id: self._action_calls_by_skill[skill_id]
                for skill_id in sorted(self._action_calls_by_skill)
                if (self.catalog.get(skill_id).execution or {}).get(
                    "max_calls_per_episode"
                )
                is not None
            },
            "manifest_limited_argument_call_counts": [
                {
                    "skill_id": skill_id,
                    "arguments": dict(argument_tuple),
                    "count": count,
                }
                for (skill_id, argument_tuple), count in sorted(
                    self._action_calls_by_argument_tuple.items()
                )
                if count > 0
            ],
            "selected_perception_fallback_rejections": (
                self._selected_perception_fallback_rejections
            ),
            "pending_selected_perception_fallback": (
                dict(self._pending_selected_perception_fallback)
                if self._pending_selected_perception_fallback is not None
                else None
            ),
            "latest_observation": self.store.latest_pointer(),
        }

    def close(self) -> None:
        if not self._closed:
            self.backend.close()
            self._closed = True

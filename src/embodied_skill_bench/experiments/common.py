from __future__ import annotations

import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..agents.runner import AgentOutcome, AgentRunner
from ..catalog import SkillCatalog
from ..io import (
    atomic_write_json,
    git_repository_provenance,
    load_json,
    project_root,
    sha256_file,
)
from ..outcomes import classify_condition
from ..prompts import episode_prompt, native_skill_prompt
from ..runtime.grounded_joint import GROUNDED_JOINT_PROVIDER
from ..runtime.native_skill_contract import (
    NATIVE_SKILL_PROTOCOL,
    build_native_skill_contract,
)
from ..runtime.perception import (
    is_contact_graspnet_perception_skill,
    is_molmo2_point_perception_skill,
    is_sam3_perception_skill,
    is_visual_difference_perception_skill,
)
from ..runtime.primitive_code import PRIMITIVE_PERCEPTION_MODES
from ..runtime.seeding import (
    CHALLENGE_PAIR_VLA_SEED_NAMESPACE,
    VLA_SEED_NAMESPACE,
    derive_challenge_pair_vla_seed,
    derive_vla_seed,
)
from ..runtime.semantic_vla import LEGACY_VLA_SKILL_IDS
from ..runtime.server import RuntimeHttpServer
from ..runtime.session import EpisodeSession
from ..runtime.turn_authority import AgentTraceTurnAuthority
from ..workspace import stage_episode_workspace, stage_native_skill_workspace


def resolve_output(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (project_root() / candidate).resolve()


def _safe_output_component(value: str, field: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    if not normalized or normalized in {".", ".."}:
        raise ValueError(f"{field} must contain a usable name")
    return normalized


def resolve_experiment_output(config: dict[str, Any]) -> Path:
    """Resolve one immutable run directory.

    `output_dir` remains supported for tests/legacy configs. New configs use
    `output_root`; every invocation then gets output/<experiment>/<run-id> so
    concurrent experiments never share condition workspaces.
    """
    if config.get("output_dir") and config.get("output_root"):
        raise ValueError("use either output_dir or output_root, not both")
    if config.get("output_dir"):
        return resolve_output(str(config["output_dir"]))
    output_root = resolve_output(str(config.get("output_root", "output")))
    experiment = _safe_output_component(str(config["name"]), "name")
    configured_id = config.get("run_id")
    if configured_id and str(configured_id).lower() != "auto":
        run_id = _safe_output_component(str(configured_id), "run_id")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_id = f"{stamp}-{secrets.token_hex(4)}"
    return output_root / experiment / run_id


def write_output_path_map(
    run_dir: Path, config: dict[str, Any]
) -> str | None:
    """Record container-to-host path translation for archived artifacts."""
    host_root = os.environ.get("ESB_HOST_OUTPUT_ROOT")
    if not host_root or config.get("output_dir"):
        return None
    container_root = resolve_output(str(config.get("output_root", "output")))
    relative = run_dir.relative_to(container_root)
    host_run_dir = Path(host_root).expanduser().resolve() / relative
    atomic_write_json(
        run_dir / "path_map.json",
        {
            "schema_version": 1,
            "container_output_root": str(container_root),
            "host_output_root": str(Path(host_root).expanduser().resolve()),
            "container_run_dir": str(run_dir),
            "host_run_dir": str(host_run_dir),
            "note": "Replace the container output-root prefix in absolute observation paths when reading artifacts on the host.",
        },
    )
    return str(host_run_dir)


def load_agent_specs(
    config: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load locked agents plus explicitly snapshotted experiment profiles.

    Third-party models can use Codex, Claude Code, the official DeepSeek
    Harness, or the benchmark's minimal OpenHands SDK harness without
    impersonating the repository's frozen agents. Profiles live in the
    experiment config so model, runtime, credential-variable names, and retry
    policy are hashed into every artifact.
    """

    locked = load_json(project_root() / "configs" / "agents.lock.json")["agents"]
    specs = {str(agent_id): dict(spec) for agent_id, spec in locked.items()}
    profiles = (config or {}).get("agent_profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("agent_profiles must be an object")
    required = {"runner", "executable", "cli_version", "model", "api_key_env", "base_url_env"}
    for raw_id, raw_spec in profiles.items():
        profile_id = str(raw_id)
        if profile_id in specs:
            raise ValueError(
                f"agent profile {profile_id!r} cannot override a locked agent"
            )
        if not isinstance(raw_spec, dict):
            raise ValueError(f"agent profile {profile_id!r} must be an object")
        missing = sorted(required - set(raw_spec))
        if missing:
            raise ValueError(
                f"agent profile {profile_id!r} is missing required fields: {missing}"
            )
        runner = str(raw_spec["runner"])
        if runner not in {
            "codex",
            "claude-code",
            "deepseek-harness",
            "kimi-code",
            "openhands",
        }:
            raise ValueError(
                f"agent profile {profile_id!r} has unsupported runner {runner!r}"
            )
        if runner == "kimi-code":
            kimi_required = {"capabilities", "max_context_tokens"}
            kimi_missing = sorted(kimi_required - set(raw_spec))
            if kimi_missing:
                raise ValueError(
                    f"agent profile {profile_id!r} is missing Kimi Code "
                    f"fields: {kimi_missing}"
                )
        if runner == "deepseek-harness":
            deepseek_required = {"cordis_config", "provider", "max_tokens"}
            deepseek_missing = sorted(deepseek_required - set(raw_spec))
            if deepseek_missing:
                raise ValueError(
                    f"agent profile {profile_id!r} is missing DeepSeek Harness "
                    f"fields: {deepseek_missing}"
                )
        if runner == "openhands":
            openhands_required = {
                "executable_env",
                "harness_config",
                "harness_config_sha256",
                "provider",
                "wire_api",
                "provider_retry_owner",
                "provider_internal_max_retries",
                "litellm_model",
                "reasoning_effort",
                "max_output_tokens",
                "request_timeout_seconds",
                "vision_mode",
                "capability_overrides",
                "litellm_extra_body",
                "observed_model_allowlist",
            }
            openhands_missing = sorted(openhands_required - set(raw_spec))
            if openhands_missing:
                raise ValueError(
                    f"agent profile {profile_id!r} is missing OpenHands fields: "
                    f"{openhands_missing}"
                )
            if raw_spec["provider"] != "openai-compatible":
                raise ValueError("OpenHands provider must be openai-compatible")
            if raw_spec["executable_env"] != "ESB_OPENHANDS_PYTHON":
                raise ValueError(
                    "OpenHands executable_env must be ESB_OPENHANDS_PYTHON"
                )
            if raw_spec["wire_api"] not in {"chat_completions", "responses"}:
                raise ValueError(
                    "OpenHands wire_api must be chat_completions or responses"
                )
            if raw_spec["provider_retry_owner"] != "benchmark":
                raise ValueError("OpenHands provider retry owner must be benchmark")
            if int(raw_spec["provider_internal_max_retries"]) != 0:
                raise ValueError("OpenHands provider internal retries must be zero")
            if not str(raw_spec["litellm_model"]).startswith("openai/"):
                raise ValueError("OpenHands litellm_model must use the openai/ adapter")
            if raw_spec["vision_mode"] not in {"native-required", "disabled"}:
                raise ValueError(
                    "OpenHands vision_mode must be native-required or disabled"
                )
            capabilities = raw_spec["capability_overrides"]
            if not isinstance(capabilities, dict):
                raise ValueError("OpenHands capability_overrides must be an object")
            allowed_capabilities = {
                "supports_reasoning_effort",
                "thinking_mode",
                "supports_sampling_params",
                "supports_extended_thinking",
                "supports_prompt_cache",
                "supports_stop_words",
                "supports_responses_api",
                "force_string_serializer",
                "send_reasoning_content",
                "supports_prompt_cache_retention",
                "requires_inline_image_data",
                "supports_vision",
            }
            unknown_capabilities = sorted(
                set(capabilities) - allowed_capabilities
            )
            if unknown_capabilities:
                raise ValueError(
                    "OpenHands capability_overrides contains unknown fields: "
                    f"{unknown_capabilities}"
                )
            expected_vision = raw_spec["vision_mode"] == "native-required"
            if capabilities.get("supports_vision") is not expected_vision:
                raise ValueError(
                    "OpenHands supports_vision must exactly match vision_mode"
                )
            if not isinstance(capabilities.get("supports_reasoning_effort"), bool):
                raise ValueError(
                    "OpenHands supports_reasoning_effort must be explicitly boolean"
                )
            expected_responses = raw_spec["wire_api"] == "responses"
            if capabilities.get("supports_responses_api") is not expected_responses:
                raise ValueError(
                    "OpenHands supports_responses_api must exactly match wire_api"
                )
            if capabilities.get("supports_prompt_cache") is not False:
                raise ValueError(
                    "OpenHands prompt cache markers must remain disabled"
                )
            extra_body = raw_spec["litellm_extra_body"]
            if not isinstance(extra_body, dict):
                raise ValueError("OpenHands litellm_extra_body must be an object")
            allowed_extra_body = {
                "enable_thinking",
                "thinking",
                "thinking_budget",
                "preserve_thinking",
                "vl_high_resolution_images",
                "max_pixels",
            }
            unexpected_extra_body = sorted(
                set(extra_body) - allowed_extra_body
            )
            if unexpected_extra_body:
                raise ValueError(
                    "OpenHands litellm_extra_body may contain only frozen "
                    "provider capability controls; unexpected fields: "
                    f"{unexpected_extra_body}"
                )
            if "enable_thinking" in extra_body and not isinstance(
                extra_body["enable_thinking"], bool
            ):
                raise ValueError("OpenHands enable_thinking must be boolean")
            if "thinking" in extra_body:
                thinking = extra_body["thinking"]
                if (
                    not isinstance(thinking, dict)
                    or set(thinking) != {"type"}
                    or thinking.get("type") not in {"adaptive", "disabled"}
                ):
                    raise ValueError(
                        "OpenHands thinking must be exactly adaptive or disabled"
                    )
            allowlist = raw_spec["observed_model_allowlist"]
            if not isinstance(allowlist, list) or not allowlist or not all(
                isinstance(item, str) and item for item in allowlist
            ):
                raise ValueError(
                    "OpenHands observed_model_allowlist must be a non-empty string list"
                )
            config_hash = str(raw_spec["harness_config_sha256"])
            if not re.fullmatch(r"[0-9a-f]{64}", config_hash):
                raise ValueError(
                    "OpenHands harness_config_sha256 must be lowercase SHA-256"
                )
            for field in ("max_output_tokens", "request_timeout_seconds"):
                if int(raw_spec[field]) <= 0:
                    raise ValueError(f"OpenHands {field} must be positive")
        specs[profile_id] = dict(raw_spec)
    return specs


def create_backend(
    config: dict[str, Any],
    run_dir: Path,
    selected_skill_ids: list[str],
    *,
    catalog: SkillCatalog | None = None,
    primitive_code: bool = False,
    primitive_code_perception_mode: str = "frozen-sam3",
    raw_vla_no_skill: bool = False,
    controller_kind: str = "OSC_POSE",
):
    selected_records = (catalog if catalog is not None else SkillCatalog.discover()).require(
        selected_skill_ids
    )
    grounded_joint_records = [
        record
        for record in selected_records
        if (record.execution or {}).get("provider")
        == GROUNDED_JOINT_PROVIDER
    ]
    requested_controller = str(config.get("controller_kind", controller_kind))
    if grounded_joint_records:
        if primitive_code:
            raise ValueError(
                "grounded JOINT selected skills cannot enter a no-skill runtime"
            )
        if requested_controller != "JOINT_POSITION":
            raise ValueError(
                "grounded JOINT selected skills require explicit "
                "backend.controller_kind=JOINT_POSITION"
            )
        incompatible_actions = [
            record.skill_id
            for record in selected_records
            if record.kind == "action"
            and (record.execution or {}).get("provider")
            != GROUNDED_JOINT_PROVIDER
        ]
        if incompatible_actions:
            raise ValueError(
                "grounded JOINT episodes cannot mix OSC/VLA action providers: "
                f"{sorted(incompatible_actions)}"
            )
        if "sam3-localize-object" not in selected_skill_ids:
            raise ValueError(
                "grounded JOINT actions require separately selected "
                "sam3-localize-object"
            )
    controller_kind = requested_controller
    if controller_kind not in {"OSC_POSE", "JOINT_POSITION"}:
        raise ValueError("controller_kind must be OSC_POSE or JOINT_POSITION")
    if (
        controller_kind == "JOINT_POSITION"
        and not primitive_code
        and not grounded_joint_records
    ):
        raise ValueError(
            "JOINT_POSITION is reserved for native typed no-skill or selected "
            "grounded JOINT actions"
        )
    if raw_vla_no_skill and (
        not primitive_code
        or controller_kind != "OSC_POSE"
        or primitive_code_perception_mode != "calibrated-geometry-p2"
        or set(selected_skill_ids) != {"pi0-doubled"}
    ):
        raise ValueError(
            "raw_vla_no_skill requires primitive_code calibrated geometry, the "
            "OSC_POSE environment, and exactly the host-only pi0-doubled adapter"
        )
    kind = config.get("kind")
    if kind == "rpent-libero":
        # Keep catalog/configuration inspection independent of simulator packages.
        from ..runtime.backends.rpent_libero import RPentLiberoBackend

        if primitive_code_perception_mode not in PRIMITIVE_PERCEPTION_MODES:
            raise ValueError("unsupported primitive_code_perception_mode")
        if not primitive_code and primitive_code_perception_mode != "frozen-sam3":
            raise ValueError(
                "primitive_code_perception_mode is only valid for primitive-code runs"
            )
        dependencies = load_json(project_root() / "configs" / "dependencies.lock.json")
        rpent_lock = dependencies["rpent"]
        rpent_repo = config.get("rpent_repo", ".deps/RPent")
        rpent_path = Path(rpent_repo)
        if not rpent_path.is_absolute():
            rpent_path = project_root() / rpent_path
        vla_model_path = os.environ.get("PI05_CHECKPOINT_PATH") or config.get(
            "vla_model_path"
        )
        raw_vla_profiles = config.get("vla_profiles", {})
        if not isinstance(raw_vla_profiles, dict):
            raise TypeError("backend.vla_profiles must be an object")
        vla_profiles: dict[str, dict[str, Any]] = {}
        for profile_id, raw_profile in raw_vla_profiles.items():
            if not isinstance(raw_profile, dict):
                raise TypeError(
                    f"backend.vla_profiles.{profile_id} must be an object"
                )
            profile = dict(raw_profile)
            endpoint_env = "ESB_VLA_ENDPOINT_" + re.sub(
                r"[^A-Za-z0-9]", "_", str(profile_id)
            ).upper()
            if os.environ.get(endpoint_env):
                profile["endpoint"] = os.environ[endpoint_env]
            vla_profiles[str(profile_id)] = profile
        required_vla_profiles = {
            str(record.execution["model_profile"])
            for record in selected_records
            if record.execution
            and record.execution.get("provider") == "semantic-vla"
        }
        if LEGACY_VLA_SKILL_IDS & set(selected_skill_ids):
            required_vla_profiles.add("rpent-generalist")
        enable_contact_graspnet = any(
            is_contact_graspnet_perception_skill(record.execution)
            for record in selected_records
        )
        enable_molmo2 = any(
            is_molmo2_point_perception_skill(record.execution)
            for record in selected_records
        )
        enable_visual_difference = any(
            is_visual_difference_perception_skill(record.execution)
            for record in selected_records
        )
        if enable_contact_graspnet and "sam3-localize-object" not in selected_skill_ids:
            raise ValueError(
                "contact-graspnet-propose-grasps requires the separately selected "
                "sam3-localize-object skill"
            )
        sam3_model_path = os.environ.get("SAM3_CHECKPOINT_PATH") or config.get(
            "sam3_model_path"
        )
        contact_graspnet_endpoint = os.environ.get(
            "CONTACT_GRASPNET_ENDPOINT"
        ) or config.get("contact_graspnet_endpoint")
        molmo2_endpoint = os.environ.get("MOLMO2_ENDPOINT") or config.get(
            "molmo2_endpoint"
        )
        visual_difference_endpoint = os.environ.get(
            "VISUAL_DIFFERENCE_ENDPOINT"
        ) or config.get("visual_difference_endpoint")
        grounded_joint_provider = config.get("grounded_joint_pyroki")
        if grounded_joint_records:
            if not isinstance(grounded_joint_provider, dict):
                raise ValueError(
                    "backend.grounded_joint_pyroki must freeze the selected provider"
                )
            if set(grounded_joint_provider) != {
                "kind",
                "endpoint",
                "expected_identity",
                "timeout_seconds",
            }:
                raise ValueError("backend.grounded_joint_pyroki fields mismatch")
            if grounded_joint_provider.get("kind") != "pyroki-http":
                raise ValueError(
                    "backend.grounded_joint_pyroki.kind must be pyroki-http"
                )
        elif grounded_joint_provider is not None:
            raise ValueError(
                "backend.grounded_joint_pyroki is forbidden without a selected "
                "grounded JOINT action"
            )

        def selected_api_key(field: str, default_env: str) -> str | None:
            raw_name = config.get(field)
            name = str(raw_name or default_env).strip()
            if not name:
                return None
            value = os.environ.get(name)
            if raw_name is not None and not value:
                raise RuntimeError(f"backend.{field} names missing environment variable {name}")
            return value

        molmo2_api_key = selected_api_key("molmo2_api_key_env", "MOLMO2_API_KEY")
        visual_difference_api_key = selected_api_key(
            "visual_difference_api_key_env", "VISUAL_DIFFERENCE_API_KEY"
        )
        if (
            primitive_code
            and primitive_code_perception_mode != "frozen-sam3"
            and not raw_vla_no_skill
        ):
            # A host environment variable must not silently re-enable a model
            # that the P0/P1/P2 no-skill conditions explicitly exclude.
            sam3_model_path = None
            vla_model_path = None
            vla_profiles = {}
        if primitive_code:
            # Contact-GraspNet is a separately selected learned perception
            # skill, never an ambient primitive-code/no-library facility.
            contact_graspnet_endpoint = None
            molmo2_endpoint = None
            visual_difference_endpoint = None
            molmo2_api_key = None
            visual_difference_api_key = None
        environment_seed = int(config.get("seed", 0))
        challenge = config.get("challenge")
        vla_seed = config.get("vla_seed")
        vla_seed_source = "explicit_config"
        if vla_seed is None:
            if isinstance(challenge, dict) and challenge.get("pair_id"):
                vla_seed = derive_challenge_pair_vla_seed(
                    pair_id=str(challenge["pair_id"]),
                    environment_seed=environment_seed,
                )
                vla_seed_source = CHALLENGE_PAIR_VLA_SEED_NAMESPACE
            else:
                vla_seed = derive_vla_seed(
                    suite=str(config["suite"]),
                    task=int(config["task"]),
                    environment_seed=environment_seed,
                )
                vla_seed_source = VLA_SEED_NAMESPACE
        challenge_kwargs: dict[str, Any] = {}
        if challenge is not None:
            if not isinstance(challenge, dict):
                raise ValueError("backend.challenge must be an object")
            required = {
                "cell_id",
                "pair_id",
                "bddl_file",
                "init_states_file",
                "task_language",
                "spec",
            }
            optional = {"source_suite", "source_task"}
            missing = required - set(challenge)
            extras = set(challenge) - required - optional
            if missing or extras:
                raise ValueError(
                    "backend.challenge fields mismatch: "
                    f"missing={sorted(missing)} extras={sorted(extras)}"
                )

            def challenge_path(value: Any) -> Path:
                candidate = Path(str(value))
                return (
                    candidate.resolve()
                    if candidate.is_absolute()
                    else (project_root() / candidate).resolve()
                )

            challenge_kwargs = {
                "challenge_cell_id": str(challenge["cell_id"]),
                "challenge_pair_id": str(challenge["pair_id"]),
                "bddl_file_path": challenge_path(challenge["bddl_file"]),
                "init_states_file_path": challenge_path(
                    challenge["init_states_file"]
                ),
                "task_language": str(challenge["task_language"]),
                "challenge_source_suite": (
                    str(challenge["source_suite"])
                    if challenge.get("source_suite") is not None
                    else None
                ),
                "challenge_source_task": (
                    int(challenge["source_task"])
                    if challenge.get("source_task") is not None
                    else None
                ),
                "challenge_spec": challenge["spec"],
            }
        return RPentLiberoBackend(
            rpent_repo=rpent_path,
            suite=str(config["suite"]),
            task=int(config["task"]),
            seed=environment_seed,
            runtime_dir=run_dir / "runtime",
            max_episode_steps=int(config.get("max_episode_steps", 10_000)),
            env_endpoint=config.get("env_endpoint"),
            vla_endpoint=config.get("vla_endpoint"),
            vla_model_path=vla_model_path,
            vla_profiles=vla_profiles,
            required_vla_profiles=sorted(required_vla_profiles),
            sam3_endpoint=config.get("sam3_endpoint"),
            sam3_model_path=sam3_model_path,
            contact_graspnet_endpoint=contact_graspnet_endpoint,
            contact_graspnet_timeout_seconds=float(
                config.get("contact_graspnet_timeout_seconds", 120)
            ),
            contact_graspnet_identity=config.get("contact_graspnet_identity"),
            molmo2_endpoint=molmo2_endpoint,
            molmo2_model=str(
                config.get("molmo2_model", "allenai/Molmo2-8B")
            ),
            molmo2_api_key=molmo2_api_key,
            molmo2_timeout_seconds=float(config.get("molmo2_timeout_seconds", 180)),
            molmo2_identity=config.get("molmo2_identity"),
            visual_difference_endpoint=visual_difference_endpoint,
            visual_difference_model=str(
                config.get("visual_difference_model", "allenai/Molmo2-8B")
            ),
            visual_difference_api_key=visual_difference_api_key,
            visual_difference_timeout_seconds=float(
                config.get("visual_difference_timeout_seconds", 180)
            ),
            visual_difference_identity=config.get("visual_difference_identity"),
            grounded_joint_pyroki_endpoint=(
                str(grounded_joint_provider["endpoint"])
                if grounded_joint_provider
                else None
            ),
            grounded_joint_pyroki_identity=(
                grounded_joint_provider["expected_identity"]
                if grounded_joint_provider
                else None
            ),
            grounded_joint_pyroki_timeout_seconds=(
                float(grounded_joint_provider["timeout_seconds"])
                if grounded_joint_provider
                else 60.0
            ),
            vla_seed=int(vla_seed),
            vla_seed_source=vla_seed_source,
            cuda_device=config.get("cuda_device"),
            expected_commit=str(rpent_lock["commit"]),
            enable_vla=bool(required_vla_profiles),
            enable_sam3=any(
                is_sam3_perception_skill(record.execution)
                for record in selected_records
            )
            or (
                primitive_code
                and primitive_code_perception_mode == "frozen-sam3"
            ),
            enable_contact_graspnet=enable_contact_graspnet,
            enable_molmo2=enable_molmo2,
            enable_visual_difference=enable_visual_difference,
            sam3_agent_access=(
                "frozen primitive-code API only"
                if primitive_code
                and primitive_code_perception_mode == "frozen-sam3"
                else "not exposed"
                if primitive_code
                else "selected read-only perception skill only"
            ),
            controller_kind=controller_kind,
            liberopro_task_data_root=config.get("liberopro_task_data_root"),
            liberopro_bddl_sha256=config.get("liberopro_bddl_sha256"),
            liberopro_init_states_sha256=config.get("liberopro_init_states_sha256"),
            **challenge_kwargs,
        )
    raise ValueError(f"unsupported backend kind: {kind!r}")


def run_use_condition(
    *,
    name: str,
    agent_id: str,
    backend_config: dict[str, Any],
    selected_skill_ids: list[str],
    limits: dict[str, Any],
    output_dir: Path,
    strict_versions: bool,
    parent_config: dict[str, Any],
) -> dict[str, Any]:
    """Run a condition, retrying only provider failures as fresh episodes."""

    max_attempts = int(limits.get("provider_max_attempts", 3))
    if not 1 <= max_attempts <= 3:
        raise ValueError("limits.provider_max_attempts must be between 1 and 3")
    backoff_seconds = float(limits.get("provider_retry_backoff_seconds", 30))
    jitter_seconds = float(limits.get("provider_retry_jitter_seconds", 5))
    if backoff_seconds < 0 or jitter_seconds < 0:
        raise ValueError("provider retry backoff and jitter must be non-negative")
    if max_attempts == 1:
        return _run_use_attempt(
            name=name,
            agent_id=agent_id,
            backend_config=backend_config,
            selected_skill_ids=selected_skill_ids,
            limits=limits,
            output_dir=output_dir,
            strict_versions=strict_versions,
            parent_config=parent_config,
        )

    retries_root = output_dir / "retry-attempts"
    attempt_records: list[dict[str, Any]] = []
    selected_result: dict[str, Any] | None = None
    for attempt_number in range(1, max_attempts + 1):
        attempt_dir = (
            output_dir
            if attempt_number == 1
            else retries_root / f"{attempt_number:02d}"
        )
        selected_result = _run_use_attempt(
            name=name,
            agent_id=agent_id,
            backend_config=backend_config,
            selected_skill_ids=selected_skill_ids,
            limits=limits,
            output_dir=attempt_dir,
            strict_versions=strict_versions,
            parent_config=parent_config,
        )
        failure_class = selected_result["classification"].get("failure_class")
        attempt_record = {
            "attempt": attempt_number,
            "run_dir": str(attempt_dir),
            "result_path": str(attempt_dir / "result.json"),
            "success": bool(selected_result["episode"].get("success")),
            "classification": selected_result["classification"],
        }
        attempt_records.append(attempt_record)
        if failure_class != "provider_transport_failure":
            break
        if attempt_number < max_attempts:
            if attempt_number == 1:
                retries_root.mkdir()
                first_result_path = retries_root / "01.result.json"
                atomic_write_json(first_result_path, selected_result)
                attempt_record["result_path"] = str(first_result_path)
            jitter = (
                secrets.randbelow(round(jitter_seconds * 1000) + 1) / 1000
                if jitter_seconds
                else 0.0
            )
            delay = backoff_seconds + jitter
            attempt_record["retry_delay_seconds"] = delay
            time.sleep(delay)

    assert selected_result is not None
    final = dict(selected_result)
    final["run_dir"] = str(output_dir)
    final["selected_attempt_run_dir"] = selected_result["run_dir"]
    final["provider_retry"] = {
        "policy": "fresh_episode_attempt",
        "max_attempts": max_attempts,
        "backoff_seconds": backoff_seconds,
        "jitter_seconds": jitter_seconds,
        "attempt_count": len(attempt_records),
        "attempts": attempt_records,
    }
    classification = dict(final["classification"])
    classification["incidents"] = list(classification.get("incidents", []))
    for discarded in attempt_records[:-1]:
        if discarded["classification"].get("failure_class") != "provider_transport_failure":
            continue
        evidence = [
            str(label)
            for incident in discarded["classification"].get("incidents", [])
            if incident.get("class") == "provider_transport_failure"
            for label in incident.get("evidence", [])
        ]
        classification["incidents"].append(
            {
                "class": "provider_transport_failure",
                "source": "discarded_atomic_attempt",
                "attempt": discarded["attempt"],
                "attempt_run_dir": discarded["run_dir"],
                "evidence": sorted(set(evidence)),
            }
        )
    final["classification"] = classification
    atomic_write_json(output_dir / "result.json", final)
    return final


def _run_use_attempt(
    *,
    name: str,
    agent_id: str,
    backend_config: dict[str, Any],
    selected_skill_ids: list[str],
    limits: dict[str, Any],
    output_dir: Path,
    strict_versions: bool,
    parent_config: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    # Keep every condition independently reproducible.  The experiment root
    # also owns a snapshot, but a condition may be archived or compared on its
    # own, so its manifest must hash a colocated frozen config.
    atomic_write_json(output_dir / "config.snapshot.json", parent_config)
    catalog = SkillCatalog.discover()
    selected = catalog.require(selected_skill_ids)
    agent_specs = load_agent_specs(parent_config)
    if agent_id not in agent_specs:
        raise ValueError(f"unknown agent {agent_id!r}")
    episode_id = f"{name}-{agent_id}"
    skill_interface = parent_config.get("skill_interface")
    native_skill_interface = skill_interface is not None
    if native_skill_interface:
        if not isinstance(skill_interface, dict):
            raise ValueError("skill_interface must be an object")
        if set(skill_interface) != {
            "protocol",
            "max_identical_action_calls",
            "automatic_observation_handoff",
            "bash_agent_exposed",
        }:
            raise ValueError("skill_interface fields mismatch")
        if (
            skill_interface.get("protocol") != NATIVE_SKILL_PROTOCOL
            or skill_interface.get("automatic_observation_handoff") is not True
            or skill_interface.get("bash_agent_exposed") is not False
        ):
            raise ValueError("unsupported native selected-skill interface boundary")
        retry_limit = skill_interface.get("max_identical_action_calls")
        if (
            isinstance(retry_limit, bool)
            or not isinstance(retry_limit, int)
            or not 1 <= retry_limit <= 5
        ):
            raise ValueError("max_identical_action_calls must be an integer from 1 to 5")
        if str(agent_specs[agent_id].get("runner")) != "openhands":
            raise ValueError("native selected-skill interface requires OpenHands")
    native_contract = (
        build_native_skill_contract(selected) if native_skill_interface else None
    )
    runner = AgentRunner(
        agent_id,
        agent_specs[agent_id],
        strict_version=strict_versions,
        isolation_mode=str(parent_config.get("agent_isolation", "strict")),
    )
    backend = None
    session = None
    server = None
    outcome: AgentOutcome | None = None
    startup_failure: dict[str, Any] | None = None
    summary = _empty_episode_summary(episode_id)
    phase = "agent_cli_preflight"
    try:
        actual_version = runner.version()
        phase = "simulator_startup"
        backend = create_backend(
            backend_config,
            output_dir,
            selected_skill_ids,
            catalog=catalog,
        )
        turn_authority = AgentTraceTurnAuthority(
            str(agent_specs[agent_id].get("runner", agent_id)),
            output_dir / "agent" / "stdout.jsonl",
        )
        session = EpisodeSession(
            episode_id=episode_id,
            run_dir=output_dir,
            backend=backend,
            catalog=catalog,
            allowed_skill_ids=selected_skill_ids,
            max_actions=int(limits.get("max_actions", 64)),
            max_perception_queries=int(limits.get("max_perception_queries", 32)),
            enforce_agent_protocol=True,
            turn_authority=turn_authority,
            native_skill_contract=native_contract,
            max_identical_action_calls=(
                int(skill_interface["max_identical_action_calls"])
                if native_skill_interface
                else 0
            ),
        )
        token = secrets.token_urlsafe(32)
        server = RuntimeHttpServer(session, token).start()
        initial = session.store.latest_pointer()
        run_context = {
            "schema_version": 1,
            "episode_id": episode_id,
            "runtime_url": server.url,
            "runtime_token": token,
            "skill_timeout_seconds": int(limits.get("skill_timeout_seconds", 900)),
            "max_perception_queries": int(
                limits.get("max_perception_queries", 32)
            ),
            "initial_observation_json": initial["observation_json"],
            "initial_observation_markdown": initial["observation_markdown"],
            "installed_skill_ids": selected_skill_ids,
        }
        checkpoint_path = os.environ.get(
            "PI05_CHECKPOINT_PATH"
        ) or backend_config.get("vla_model_path")
        if checkpoint_path and VLA_SKILL_IDS & set(selected_skill_ids):
            run_context["vla_checkpoint_path"] = str(
                Path(str(checkpoint_path)).expanduser().resolve()
            )
        sam3_checkpoint_path = os.environ.get(
            "SAM3_CHECKPOINT_PATH"
        ) or backend_config.get("sam3_model_path")
        if sam3_checkpoint_path and any(
            is_sam3_perception_skill(record.execution) for record in selected
        ):
            run_context["sam3_checkpoint_path"] = str(
                Path(str(sam3_checkpoint_path)).expanduser().resolve()
            )
        phase = "agent_workspace_startup"
        workspace = (
            stage_native_skill_workspace(
                workspace=output_dir / "workspace",
                catalog=catalog,
                selected_skill_ids=selected_skill_ids,
                run_context=run_context,
            )
            if native_skill_interface
            else stage_episode_workspace(
                workspace=output_dir / "workspace",
                catalog=catalog,
                selected_skill_ids=selected_skill_ids,
                run_context=run_context,
            )
        )
        initial_observation = session.store.latest()
        prompt = (
            native_skill_prompt(
                workspace=workspace,
                task_language=str(initial_observation["task"].get("language")),
                initial_observation_markdown=initial["observation_markdown"],
            )
            if native_skill_interface
            else episode_prompt(
                workspace=workspace,
                task_language=str(initial_observation["task"].get("language")),
                initial_observation_markdown=initial["observation_markdown"],
            )
        )
        phase = "simulator_provenance"
        manifest = {
            "schema_version": 1,
            "benchmark_version": "0.1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment_name": name,
            "episode_id": episode_id,
            "condition": "skill_use",
            "agent": {
                "id": agent_id,
                "runner": str(agent_specs[agent_id].get("runner", agent_id)),
                "cli_version": actual_version,
                "requested_model": agent_specs[agent_id]["model"],
                "reasoning_effort": agent_specs[agent_id].get(
                    "reasoning_effort"
                ),
            },
            "selected_skills": [record.locked() for record in selected],
            "skill_interface": (
                {
                    **skill_interface,
                    "contract_sha256": sha256_file(
                        workspace / "native_skill_tools.json"
                    ),
                    "guide_sha256": sha256_file(
                        workspace / "NATIVE_SKILL_LIBRARY.md"
                    ),
                    "tool_names": [
                        item["tool_name"] for item in native_contract["tools"]
                    ],
                }
                if native_skill_interface
                else {"protocol": "legacy-bash-skill-scripts-v1"}
            ),
            "backend": backend.provenance(),
            "limits": limits,
            "strict_versions": strict_versions,
            "repository": git_repository_provenance(),
            "config_sha256": sha256_file(output_dir / "config.snapshot.json"),
            "parent_config": parent_config,
            "contamination_controls": {
                "rpent_memory_injected": False,
                "primitive_mcp_exposed": False,
                "unselected_skill_bodies_staged": False,
                "user_config_ignored_by_codex": str(
                    agent_specs[agent_id].get("runner", agent_id)
                )
                == "codex",
                "claude_bare_mode": str(
                    agent_specs[agent_id].get("runner", agent_id)
                )
                == "claude-code",
                "deepseek_harness_minimal_profile": str(
                    agent_specs[agent_id].get("runner", agent_id)
                )
                == "deepseek-harness",
                "openhands_minimal_profile": str(
                    agent_specs[agent_id].get("runner", agent_id)
                )
                == "openhands",
                "openhands_native_observation_images": str(
                    agent_specs[agent_id].get("runner", agent_id)
                )
                == "openhands"
                and agent_specs[agent_id].get("vision_mode") == "native-required",
                "filesystem_isolation_required": parent_config.get(
                    "agent_isolation", "strict"
                )
                == "strict",
            },
        }
        atomic_write_json(output_dir / "run_manifest.json", manifest)
        phase = "agent_process_startup_or_execution"
        outcome = runner.run_episode(
            workspace=workspace,
            prompt=prompt,
            output_dir=output_dir / "agent",
            timeout_seconds=int(limits.get("timeout_seconds", 1800)),
        )
        summary = session.finish("agent_process_exited")
    except Exception as exc:
        failure_class = (
            "simulator_failure"
            if phase in {"simulator_startup", "simulator_provenance"}
            else "agent_protocol_failure"
        )
        startup_failure = {
            "class": failure_class,
            "source": phase,
            "message": f"{type(exc).__name__}: {str(exc)[:2000]}",
        }
        if session is not None:
            try:
                summary = session.finish(f"condition_exception:{phase}")
            except Exception:
                summary = _empty_episode_summary(episode_id)
    finally:
        if server is not None:
            server.stop()
        if session is not None:
            session.close()
        elif backend is not None:
            backend.close()
    result = {
        "experiment": name,
        "agent": agent_id,
        "skills": selected_skill_ids,
        "episode": summary,
        "agent_outcome": outcome.public() if outcome else None,
        "run_dir": str(output_dir),
    }
    result["classification"] = classify_condition(
        episode=summary,
        agent_outcome=result["agent_outcome"],
        protocol_violations=[
            str(item.get("message") or item.get("code"))
            for item in summary.get("protocol_violations", [])
        ],
        startup_failure=startup_failure,
    )
    atomic_write_json(output_dir / "result.json", result)
    return result


def _empty_episode_summary(episode_id: str) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "success": False,
        "terminated": False,
        "truncated": False,
        "step_index": 0,
        "action_count": 0,
        "perception_count": 0,
        "calls": {
            "completed": 0,
            "rejected": 0,
            "errors": 0,
            "simulator_steps": 0,
            "vla_chunks": 0,
        },
        "perception_calls": {"completed": 0, "rejected": 0, "errors": 0},
        "skill_sequence": [],
        "perception_sequence": [],
        "skill_call_counts": {},
        "recovered_after_error_or_rejection": False,
        "protocol_enforced": True,
        "protocol_violations": [],
        "latest_observation": None,
    }

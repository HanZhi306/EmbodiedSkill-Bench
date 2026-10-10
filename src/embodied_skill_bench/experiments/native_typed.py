"""Runnable P2.5/P3 native typed no-skill development experiment."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..agents.runner import AgentOutcome, AgentRunner, ObservationProjectionError
from ..catalog import SkillCatalog
from ..io import atomic_write_json, git_repository_provenance, load_json, sha256_file
from ..outcomes import aggregate_classified_conditions, classify_condition
from ..prompts import native_typed_prompt
from ..runtime.native_typed import NativeTypedSession
from ..runtime.native_typed_contract import (
    P25_PROTOCOL,
    P3_PROTOCOL,
    RAW_PI05_PROTOCOL,
    RAW_PI05_RGB_PROTOCOL,
    contract_document,
    contract_sha256,
)
from ..runtime.ik import PyRoKiHttpProvider, UnavailableIKProvider
from ..runtime.server import RuntimeHttpServer
from ..runtime.turn_authority import AgentTraceTurnAuthority
from ..workspace import stage_native_typed_workspace
from .common import create_backend, load_agent_specs, resolve_experiment_output, write_output_path_map


def run_native_typed_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("experiment") != "native-typed-no-skill":
        raise ValueError("config.experiment must be 'native-typed-no-skill'")
    protocol = str(config.get("native_typed", {}).get("protocol", ""))
    if protocol not in {
        P25_PROTOCOL,
        P3_PROTOCOL,
        RAW_PI05_PROTOCOL,
        RAW_PI05_RGB_PROTOCOL,
    }:
        raise ValueError("config.native_typed.protocol is unsupported")
    expected_contract = str(config.get("native_typed", {}).get("contract_sha256", ""))
    actual_contract = contract_sha256(protocol)
    if expected_contract != actual_contract:
        raise ValueError(
            "config.native_typed.contract_sha256 drift: "
            f"expected {expected_contract!r}, current {actual_contract!r}"
        )
    root = resolve_experiment_output(config)
    root.mkdir(parents=True, exist_ok=False)
    host_run_dir = write_output_path_map(root, config)
    atomic_write_json(root / "config.snapshot.json", config)
    results = [
        run_native_typed_condition(
            name=str(config["name"]), agent_id=str(agent_id),
            backend_config=dict(config["backend"]), limits=dict(config.get("limits", {})),
            output_dir=root / str(agent_id), strict_versions=bool(config.get("strict_versions", True)),
            parent_config=config, protocol=protocol,
            native_typed_config=dict(config.get("native_typed", {})),
        )
        for agent_id in config["agents"]
    ]
    rates = aggregate_classified_conditions(
        results,
        success_getter=lambda item: item["episode"]["success"],
        classification_getter=lambda item: item["classification"],
        descriptor_getter=lambda item: {"agent": item["agent"], "run_dir": item["run_dir"]},
    )
    summary = {
        "experiment": "native-typed-no-skill", "name": config["name"],
        "protocol": protocol, "contract": contract_document(protocol),
        "run_dir": str(root), "host_run_dir": host_run_dir,
        "conditions": results, **rates,
    }
    atomic_write_json(root / "summary.json", summary)
    return summary


def run_native_typed_condition(
    *, name: str, agent_id: str, backend_config: dict[str, Any], limits: dict[str, Any],
    output_dir: Path, strict_versions: bool, parent_config: dict[str, Any], protocol: str,
    native_typed_config: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output_dir / "config.snapshot.json", parent_config)
    # Native P2.5/P3 control has no library dependency. The raw-Pi0.5
    # development conditions still require their private host-side adapter.
    catalog = (SkillCatalog.discover()
               if protocol in {RAW_PI05_PROTOCOL, RAW_PI05_RGB_PROTOCOL}
               else SkillCatalog([]))
    specs = load_agent_specs(parent_config)
    runner = AgentRunner(
        agent_id, specs[agent_id], strict_version=strict_versions,
        isolation_mode=str(parent_config.get("agent_isolation", "strict")),
    )
    episode_id = f"{name}-{agent_id}"
    backend = session = server = None
    outcome: AgentOutcome | None = None
    startup_failure = None
    summary = _empty_summary(episode_id, protocol)
    phase = "agent_cli_preflight"
    try:
        actual_version = runner.version()
        phase = "simulator_startup"
        raw_pi05 = protocol in {RAW_PI05_PROTOCOL, RAW_PI05_RGB_PROTOCOL}
        backend = create_backend(
            backend_config,
            output_dir,
            ["pi0-doubled"] if raw_pi05 else [],
            catalog=catalog,
            primitive_code=True, primitive_code_perception_mode="calibrated-geometry-p2",
            raw_vla_no_skill=raw_pi05,
            controller_kind="JOINT_POSITION" if protocol == P3_PROTOCOL else "OSC_POSE",
        )
        authority = AgentTraceTurnAuthority(
            str(specs[agent_id].get("runner", agent_id)), output_dir / "agent" / "stdout.jsonl"
        )
        session = NativeTypedSession(
            episode_id=episode_id, run_dir=output_dir, backend=backend, catalog=catalog,
            max_actions=int(limits.get("max_actions", 40)), protocol=protocol,
            enforce_agent_protocol=True, turn_authority=authority,
            ik_provider=_build_ik_provider(protocol, native_typed_config),
        )
        token = secrets.token_urlsafe(32)
        server = RuntimeHttpServer(session, token, allow_pixel_to_world=False).start()
        initial = session.store.latest_pointer()
        run_context = {
            "schema_version": 1, "runtime_contract": "native-typed-no-skill-v1",
            "native_typed_protocol": protocol, "episode_id": episode_id,
            "runtime_url": server.url, "runtime_token": token,
            "skill_timeout_seconds": 900,
            "initial_observation_json": initial["observation_json"],
            "initial_observation_markdown": initial["observation_markdown"],
            "installed_skill_ids": [], "perception_mode": "calibrated-geometry-p2",
            "raw_vla_provider_exposed": raw_pi05,
            "semantic_vla_skill_wrappers_exposed": False,
        }
        phase = "agent_workspace_startup"
        workspace = stage_native_typed_workspace(
            workspace=output_dir / "workspace", run_context=run_context, protocol=protocol
        )
        observation = session.store.latest()
        prompt = native_typed_prompt(
            workspace=workspace, task_language=str(observation["task"].get("language")),
            initial_observation_markdown=initial["observation_markdown"], protocol=protocol,
        )
        phase = "simulator_provenance"
        atomic_write_json(output_dir / "run_manifest.json", {
            "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment_name": name, "episode_id": episode_id,
            "condition": protocol, "agent": {
                "id": agent_id, "runner": str(specs[agent_id].get("runner", agent_id)),
                "cli_version": actual_version, "requested_model": specs[agent_id]["model"],
            },
            "selected_skills": [], "installed_skills": [],
            "native_typed_contract": contract_document(protocol),
            "native_typed_contract_sha256": session.schema_sha256,
            "native_typed_json_sha256": sha256_file(workspace / "native_typed_tools.json"),
            "native_typed_markdown_sha256": sha256_file(workspace / "NATIVE_TYPED_TOOLS.md"),
            "backend": backend.provenance(), "limits": limits,
            "repository": git_repository_provenance(), "parent_config": parent_config,
            "contamination_controls": {
                "skill_documents_exposed": False, "skill_selector_used": False,
                "vla_exposed": raw_pi05,
                "raw_vla_provider_exposed": raw_pi05,
                "semantic_vla_skill_wrappers_exposed": False,
                "runtime_private_vla_adapter": (
                    "pi0-doubled" if raw_pi05 else None
                ),
                "sam3_exposed": False,
                "contact_graspnet_exposed": False, "learned_perception_exposed": False,
                "raw_python_exec_enabled": False, "bash_agent_exposed": False,
                "direct_runtime_http_allowed": False, "semantic_coordinates_exposed": False,
            },
            "p3_claim_boundary": (
                "not_applicable_non_p3" if protocol != P3_PROTOCOL
                else "full_pose_execution_requires_audited_joint_controller_backend"
            ),
            "ik_provider_config": native_typed_config.get("ik_provider"),
        })
        phase = "agent_process_startup_or_execution"
        outcome = runner.run_episode(
            workspace=workspace, prompt=prompt, output_dir=output_dir / "agent",
            timeout_seconds=int(limits.get("timeout_seconds", 1800)),
        )
        summary = session.finish("agent_process_exited")
    except Exception as exc:
        startup_failure = {
            "class": "simulator_failure" if phase.startswith("simulator") else "agent_protocol_failure",
            "source": "observation_projection" if isinstance(exc, ObservationProjectionError) else phase,
            "message": f"{type(exc).__name__}: {str(exc)[:2000]}",
        }
        if session is not None:
            summary = session.finish(f"condition_exception:{phase}")
    finally:
        if server is not None:
            server.stop()
        if session is not None:
            session.close()
        elif backend is not None:
            backend.close()
    result = {
        "experiment": "native-typed-no-skill", "name": name, "agent": agent_id,
        "skills": [], "episode": summary,
        "agent_outcome": outcome.public() if outcome else None, "run_dir": str(output_dir),
    }
    result["classification"] = classify_condition(
        episode=summary, agent_outcome=result["agent_outcome"],
        protocol_violations=[str(x.get("message") or x.get("code")) for x in summary.get("protocol_violations", [])],
        startup_failure=startup_failure,
    )
    atomic_write_json(output_dir / "result.json", result)
    return result


def _empty_summary(episode_id: str, protocol: str) -> dict[str, Any]:
    return {
        "episode_id": episode_id, "method": "native-typed-no-skill-agent",
        "native_typed_protocol": protocol, "success": False, "terminated": False,
        "truncated": False, "step_index": 0, "action_count": 0,
        "native_tool_calls": {}, "protocol_violations": [], "latest_observation": None,
    }


def _build_ik_provider(protocol: str, native_typed_config: dict[str, Any]):
    provider = native_typed_config.get("ik_provider") or {"kind": "unavailable"}
    if not isinstance(provider, dict):
        raise ValueError("native_typed.ik_provider must be an object")
    kind = str(provider.get("kind", "unavailable"))
    if protocol in {P25_PROTOCOL, RAW_PI05_PROTOCOL, RAW_PI05_RGB_PROTOCOL}:
        if kind != "unavailable":
            raise ValueError("P2.5/raw-Pi0.5 must not configure an IK provider")
        return UnavailableIKProvider(
            "raw_pi05_has_no_ik_capability"
            if protocol in {RAW_PI05_PROTOCOL, RAW_PI05_RGB_PROTOCOL}
            else "p25_has_no_ik_capability"
        )
    if kind == "unavailable":
        return UnavailableIKProvider(str(provider.get("reason", "p3_provider_not_frozen")))
    if kind == "pyroki-http":
        endpoint = str(provider.get("endpoint") or "")
        identity = provider.get("expected_identity")
        if not endpoint.startswith("http://127.0.0.1:"):
            raise ValueError("P3 PyRoKi endpoint must be explicit localhost HTTP")
        if not isinstance(identity, dict) or not identity:
            raise ValueError("P3 PyRoKi expected_identity must be a non-empty object")
        return PyRoKiHttpProvider(
            endpoint=endpoint,
            expected_identity=identity,
            timeout_seconds=float(provider.get("timeout_seconds", 30)),
        )
    raise ValueError("native_typed.ik_provider.kind must be unavailable or pyroki-http")

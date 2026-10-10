from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..catalog import SkillCatalog
from ..io import atomic_write_json, git_repository_provenance, load_json, sha256_file
from ..outcomes import classify_condition
from ..runtime.session import EpisodeSession
from .common import create_backend, resolve_experiment_output, write_output_path_map


def run_direct_vla_experiment(config_path: str | Path) -> dict[str, Any]:
    """Run one complete-task VLA call without a selector or coding agent."""

    config = load_json(config_path)
    if config.get("experiment") != "direct-vla":
        raise ValueError("config.experiment must be 'direct-vla'")
    backend_config = dict(config["backend"])
    task_language = str(backend_config.get("task_language") or "").strip()
    if not task_language:
        raise ValueError("direct-vla requires backend.task_language")
    max_chunks = int(config.get("max_chunks", 40))
    if not 1 <= max_chunks <= 200:
        raise ValueError("max_chunks must be between 1 and 200")
    root = resolve_experiment_output(config)
    root.mkdir(parents=True, exist_ok=False)
    host_run_dir = write_output_path_map(root, config)
    atomic_write_json(root / "config.snapshot.json", config)
    catalog = SkillCatalog.discover()
    catalog.require(["pi0-doubled"])
    backend = None
    session = None
    startup_failure = None
    response: dict[str, Any] | None = None
    phase = "simulator_startup"
    try:
        backend = create_backend(backend_config, root, ["pi0-doubled"])
        session = EpisodeSession(
            episode_id=str(config["name"]),
            run_dir=root,
            backend=backend,
            catalog=catalog,
            allowed_skill_ids=["pi0-doubled"],
            max_actions=1,
            enforce_agent_protocol=False,
        )
        phase = "simulator_provenance"
        atomic_write_json(
            root / "run_manifest.json",
            {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "condition": "direct_vla_complete_task_instruction",
                "agent": None,
                "selector": None,
                "selected_skills": [],
                "runtime_private_concrete_skill": "pi0-doubled",
                "task_language": task_language,
                "max_chunks": max_chunks,
                "backend": backend.provenance(),
                "repository": git_repository_provenance(),
                "config_sha256": sha256_file(root / "config.snapshot.json"),
                "contamination_controls": {
                    "skill_documents_exposed": False,
                    "coding_agent_used": False,
                    "rpent_memory_injected": False,
                    "primitive_mcp_exposed": False,
                },
            },
        )
        phase = "vla_execution"
        response = session.execute(
            {
                "request_id": "direct-vla-complete-task",
                "skill_id": "pi0-doubled",
                "arguments": {"prompt": task_language, "max_chunks": max_chunks},
            }
        )
        summary = session.finish("direct_vla_call_completed")
    except Exception as exc:
        startup_failure = {
            "class": "simulator_failure",
            "source": phase,
            "message": f"{type(exc).__name__}: {str(exc)[:2000]}",
        }
        summary = session.finish(f"condition_exception:{phase}") if session else {
            "episode_id": str(config["name"]),
            "success": False,
            "calls": {},
            "protocol_violations": [],
        }
    finally:
        if session is not None:
            session.close()
        elif backend is not None:
            backend.close()
    classification = classify_condition(
        episode=summary,
        agent_outcome=None,
        protocol_violations=[
            str(item.get("message") or item.get("code"))
            for item in summary.get("protocol_violations", [])
        ],
        startup_failure=startup_failure,
    )
    result = {
        "experiment": "direct-vla",
        "name": config["name"],
        "run_dir": str(root),
        "host_run_dir": host_run_dir,
        "task_language": task_language,
        "max_chunks": max_chunks,
        "response": response,
        "episode": summary,
        "classification": classification,
    }
    atomic_write_json(root / "summary.json", result)
    return result

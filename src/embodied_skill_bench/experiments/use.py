from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import atomic_write_json, load_json
from ..outcomes import aggregate_classified_conditions
from .common import (
    resolve_experiment_output,
    run_use_condition,
    write_output_path_map,
)


def run_use_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("experiment") != "use":
        raise ValueError("config.experiment must be 'use'")
    root = resolve_experiment_output(config)
    root.mkdir(parents=True, exist_ok=False)
    host_run_dir = write_output_path_map(root, config)
    atomic_write_json(root / "config.snapshot.json", config)
    results = []
    for agent_id in config["agents"]:
        condition_dir = root / str(agent_id)
        results.append(
            run_use_condition(
                name=str(config["name"]),
                agent_id=str(agent_id),
                backend_config=dict(config["backend"]),
                selected_skill_ids=list(config["skill_set"]),
                limits=dict(config.get("limits", {})),
                output_dir=condition_dir,
                strict_versions=bool(config.get("strict_versions", True)),
                parent_config=config,
            )
        )
    rate_summary = aggregate_classified_conditions(
        results,
        success_getter=lambda item: item["episode"]["success"],
        classification_getter=lambda item: item["classification"],
        descriptor_getter=lambda item: {
            "agent": item["agent"],
            "episode_id": item["episode"]["episode_id"],
            "run_dir": item["run_dir"],
        },
    )
    summary = {
        "experiment": "use",
        "name": config["name"],
        "run_dir": str(root),
        "host_run_dir": host_run_dir,
        "conditions": results,
        **rate_summary,
        "agent_metrics": [
            {
                "agent": item["agent"],
                "success": item["episode"]["success"],
                "classification": item["classification"],
                "action_count": item["episode"]["action_count"],
                "calls": item["episode"]["calls"],
                "elapsed_seconds": item["agent_outcome"]["elapsed_seconds"]
                if item["agent_outcome"]
                else None,
                "usage": item["agent_outcome"]["usage"]
                if item["agent_outcome"]
                else {},
            }
            for item in results
        ],
    }
    atomic_write_json(root / "summary.json", summary)
    return summary

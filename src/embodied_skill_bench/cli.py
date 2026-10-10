"""Small source-distribution entry point; execution stays in the research core."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from .catalog import SkillCatalog
from .io import atomic_write_json, load_json, project_root, sha256_file


def doctor() -> dict:
    root = project_root()
    tasks = load_json(root / "configs" / "clean100.json")["tasks"]
    validate_roster(tasks)
    if len(tasks) != 100 or any(
        not {"liberopro_bddl_sha256", "liberopro_init_states_sha256", "vla_seed"} <= set(task)
        for task in tasks
    ):
        raise ValueError("Clean100 requires 100 tasks with frozen task hashes and VLA seeds")
    base = {"ok": True, "clean100_tasks": len(tasks),
            "bubblewrap_available": shutil.which("bwrap") is not None,
            "simulator_checked": False, "agent_api_checked": False}
    index_path = root / "skills-index.json"
    pool_path = root / "skill-pool"
    if not index_path.exists() and not pool_path.exists():
        return {**base, "skill_assets": "not_included", "skills": 0,
                "verified_skill_files": 0,
                "note": "Public code release: the paper skill library is not included. "
                        "Native control does not require skill assets."}
    if not index_path.is_file() or not pool_path.is_dir():
        raise ValueError("Local skill assets require both skill-pool/ and skills-index.json")
    index = load_json(root / "skills-index.json")
    catalog = SkillCatalog.discover()
    ids = [item["id"] for item in index["skills"]]
    if len(set(ids)) != len(ids) or set(ids) != {r.skill_id for r in catalog}:
        raise ValueError("Catalog and distributed skill index disagree")
    counts = dict(Counter(r.kind for r in catalog))
    if len(catalog) != index["total"] or counts != index["kind_counts"]:
        raise ValueError("Skill index counts disagree with the catalog")
    checked = 0
    for item in index["skills"]:
        directory = root / item["path"]
        expected_files = set(item["files_sha256"])
        actual_files = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
                        if p.is_file() and "__pycache__" not in p.parts}
        if expected_files != actual_files:
            raise ValueError(f"Skill file inventory changed: {item['id']}")
        for relative, expected in item["files_sha256"].items():
            path = directory / relative
            if path.is_symlink() or sha256_file(path) != expected:
                raise ValueError(f"Skill file checksum mismatch: {item['id']}/{relative}")
            checked += 1
    return {**base, "skill_assets": "verified", "skills": len(catalog),
            "kind_counts": counts, "verified_skill_files": checked}


def expand_config(value):
    """Expand only explicit ${NAME} placeholders; never load a private .env."""
    if isinstance(value, str):
        def replace(match):
            name = match.group(1)
            if not os.environ.get(name):
                raise ValueError(f"Set environment variable {name}")
            if any(word in name.upper() for word in ("KEY", "TOKEN", "PASSWORD", "SECRET")):
                raise ValueError("Credentials must use *_env fields, not interpolation")
            return os.environ[name]
        return re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", replace, value)
    if isinstance(value, list):
        return [expand_config(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_config(item) for key, item in value.items()}
    return value


def validate_roster(tasks):
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Roster tasks must be a nonempty list")
    allowed = {"suite", "task", "seed", "task_language", "vla_seed",
               "liberopro_bddl_sha256", "liberopro_init_states_sha256"}
    identities = []
    for task in tasks:
        if not isinstance(task, dict) or set(task) - allowed:
            raise ValueError("Task roster contains unsupported fields")
        if not {"suite", "task", "seed", "task_language"} <= set(task):
            raise ValueError("Each task needs suite, task, seed and task_language")
        if any(type(task[key]) is not int for key in ("task", "seed")):
            raise ValueError("Task and seed must be integers")
        for field in ("liberopro_bddl_sha256", "liberopro_init_states_sha256"):
            if field in task and not re.fullmatch(r"[0-9a-f]{64}", str(task[field])):
                raise ValueError("Invalid task-file SHA256")
        identities.append((task["suite"], task["task"], task["seed"]))
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate task/seed in roster")


def validate_config(config):
    from .experiments.common import load_agent_specs
    if config.get("experiment") not in {"use", "native-typed-no-skill", "direct-vla"}:
        raise ValueError("Unsupported experiment")
    backend = config.get("backend", {})
    if backend.get("kind") != "rpent-libero":
        raise ValueError("Backend must be rpent-libero")
    if config["experiment"] != "direct-vla":
        agents = config.get("agents", [])
        if len(agents) != 1:
            raise ValueError("The minimal launcher runs one agent per configuration")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", agents[0]):
            raise ValueError("Unsafe agent ID")
        specs = load_agent_specs(config)
        if agents[0] not in specs:
            raise ValueError("Unknown agent profile")
        if config.get("agent_isolation", "strict") != "strict":
            raise ValueError("Real runs require strict agent isolation")
        if config.get("strict_versions", True) is not True:
            raise ValueError("Real runs require a pinned agent CLI version")
    if config["experiment"] == "use":
        SkillCatalog.discover().require(config["skill_set"])
    if config["experiment"] == "native-typed-no-skill":
        from .runtime.native_typed_contract import P3_PROTOCOL, contract_sha256
        native = config["native_typed"]
        if native.get("contract_sha256") != contract_sha256(native["protocol"]):
            raise ValueError("Native control contract checksum mismatch")
        if native["protocol"] == P3_PROTOCOL:
            provider = native.get("ik_provider", {})
            if provider.get("kind") != "pyroki-http" or not provider.get("expected_identity"):
                raise ValueError("P3 needs an audited PyRoKi endpoint and expected_identity")


def run_one(config, directory):
    from .experiments.direct_vla import run_direct_vla_experiment
    from .experiments.native_typed import run_native_typed_experiment
    from .experiments.use import run_use_experiment
    frozen = copy.deepcopy(config)
    frozen.pop("output_root", None)
    frozen.pop("run_id", None)
    frozen["output_dir"] = str(directory.resolve())
    # Config sits beside (not inside) the new immutable episode directory.
    config_path = directory.with_suffix(".config.json")
    atomic_write_json(config_path, frozen)
    return {"use": run_use_experiment,
            "native-typed-no-skill": run_native_typed_experiment,
            "direct-vla": run_direct_vla_experiment}[frozen["experiment"]](config_path)


def run_config(config, output, tasks=None):
    validate_config(config)
    if tasks is not None:
        validate_roster(tasks)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "config.json", config)
    root = project_root()
    sources = sorted((root / "src" / "embodied_skill_bench").rglob("*.py"))
    resources = [root / relative for relative in (
        "protocol.json", "configs/agents.lock.json",
        "configs/dependencies.lock.json", "configs/clean100.json")]
    if (root / "skills-index.json").is_file():
        resources.append(root / "skills-index.json")
    atomic_write_json(output / "runtime-manifest.json", {
        "python_version": sys.version.split()[0],
        "files_sha256": {p.relative_to(root).as_posix(): sha256_file(p)
                         for p in [*sources, *resources]},
        "installed_packages": dict(sorted(
            (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
            if d.metadata.get("Name"))),
    })
    roster = tasks if tasks is not None else [{}]
    atomic_write_json(output / "tasks.json", {"tasks": roster})
    records = []
    ok = True
    for index, task in enumerate(roster):
        current = copy.deepcopy(config)
        current["backend"].update(task)
        current["name"] = f"{config['name']}-{index:03d}"
        result = run_one(current, output / f"task-{index:03d}")
        conditions = result.get("conditions", [result])
        eligible = all(item.get("classification", {}).get("eligible", False)
                       for item in conditions)
        records.append({"task": task or config["backend"],
                        "summary": f"task-{index:03d}/summary.json", "eligible": eligible})
        ok = ok and eligible
        atomic_write_json(output / "run-index.json", {
            "requested_tasks": len(roster), "completed_tasks": len(records),
            "all_eligible": ok, "runs": records})
        print(json.dumps({"task": index, "run_dir": result["run_dir"],
                          "eligible": eligible}), flush=True)
        if not eligible:
            # Do not silently spend another full roster on a broken installation.
            break
    return ok, output


def init_config(args):
    first = load_json(project_root() / "configs" / "clean100.json")["tasks"][0]
    key_env, base_env = {"codex": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
                         "claude-code": ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"),
                         "kimi-code": ("KIMI_API_KEY", "KIMI_BASE_URL")}[args.runner]
    profile = {"runner": args.runner,
               "executable": {"codex": "codex", "claude-code": "claude", "kimi-code": "kimi"}[args.runner],
               "cli_version": args.cli_version, "model": args.model,
               "api_key_env": key_env, "base_url_env": base_env,
               "request_max_retries": 0, "stream_max_retries": 0}
    if args.runner == "codex":
        profile["reasoning_effort"] = args.reasoning_effort
    if args.runner == "kimi-code":
        profile.update(capabilities=["image_in", "thinking"], max_context_tokens=262144)
    config = {"schema_version": 1, "experiment": "use", "name": "libero-skill-use",
              "agents": ["agent"], "agent_profiles": {"agent": profile},
              "agent_isolation": "strict", "strict_versions": True,
              "backend": {"kind": "rpent-libero", "rpent_repo": "${ESB_RPENT_ROOT}",
                          "liberopro_task_data_root": "${LIBEROPRO_DATASET_PATH}",
                          "cuda_device": 0, "max_episode_steps": 10000, **first},
              "skill_set": ["move-to", "move-pose", "rotate-wrist", "rotate-pitch", "set-gripper", "release"],
              "limits": {"max_actions": 40, "max_perception_queries": 24,
                         "provider_max_attempts": 1, "skill_timeout_seconds": 900,
                         "timeout_seconds": 3600}}
    if args.mode == "native":
        from .runtime.native_typed_contract import P25_PROTOCOL, P3_PROTOCOL, contract_sha256
        protocol = P3_PROTOCOL if args.native_protocol == "p3" else P25_PROTOCOL
        native = {"protocol": protocol, "contract_sha256": contract_sha256(protocol),
                  "ik_provider": {"kind": "unavailable"}}
        if protocol == P3_PROTOCOL:
            if not args.ik_identity or not args.ik_endpoint:
                raise ValueError("P3 native control requires --ik-identity and --ik-endpoint "
                                 "for the audited PyRoKi service. See RUN_BENCH.md.")
            identity = load_json(args.ik_identity)
            if not identity or not args.ik_endpoint.startswith("http://127.0.0.1:"):
                raise ValueError("P3 needs a nonempty identity and a localhost HTTP endpoint")
            native["ik_provider"] = {"kind": "pyroki-http", "endpoint": args.ik_endpoint,
                                     "expected_identity": identity, "timeout_seconds": 30}
            config["backend"]["controller_kind"] = "JOINT_POSITION"
        config.update(experiment="native-typed-no-skill", name="libero-native-control",
                      native_typed=native)
        config.pop("skill_set")
    elif args.ik_identity or args.ik_endpoint:
        raise ValueError("--ik-identity and --ik-endpoint apply to P3 native control")
    if args.mode == "native" and args.native_protocol == "p25" and (args.ik_identity or args.ik_endpoint):
        raise ValueError("P2.5 uses OSC control and does not accept an IK provider")
    target = Path(args.output).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    note = ("Native P3 starter; freeze agent, budgets and provider identity for your experiment."
            if args.mode == "native" and args.native_protocol == "p3" else
            "P2.5 OSC development starter; this is not the paper's P3 native baseline."
            if args.mode == "native" else
            "Skill-use starter; requires separately supplied local skill packages and providers.")
    return {"config": str(target), "note": note}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check the task roster and any local skill index (no GPU/API)")
    init = commands.add_parser("init", help="Write a native-control or skill-use configuration")
    init.add_argument("--mode", choices=["native", "skill-only"], default="native")
    init.add_argument("--native-protocol", choices=["p3", "p25"], default="p3",
                      help="P3 matches the paper's native interface; P2.5 is an OSC development interface")
    init.add_argument("--ik-identity", help="JSON identity from the audited P3 PyRoKi service")
    init.add_argument("--ik-endpoint", help="Audited P3 service URL, e.g. http://127.0.0.1:8766")
    init.add_argument("--runner", choices=["codex", "claude-code", "kimi-code"], required=True)
    init.add_argument("--model", required=True)
    init.add_argument("--cli-version", required=True, help="Exact output of the agent CLI --version")
    init.add_argument("--reasoning-effort", default="medium")
    init.add_argument("--output", required=True)
    run = commands.add_parser("run", help="Run one task, or a task roster sequentially")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True, help="New directory, preferably outside the source checkout")
    run.add_argument("--roster", help="JSON object containing a tasks list")
    run.add_argument("--limit", type=int, help="First N roster entries (explicit subset)")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command == "init":
            result = init_config(args)
        else:
            config = expand_config(load_json(args.config))
            tasks = None
            if args.command == "run":
                if args.limit is not None and (not args.roster or args.limit < 1):
                    raise ValueError("--limit needs --roster and a positive count")
                if args.roster:
                    tasks = load_json(args.roster)["tasks"]
                    validate_roster(tasks)
                    if args.limit is not None:
                        if args.limit > len(tasks):
                            raise ValueError("--limit exceeds roster size")
                        tasks = tasks[:args.limit]
            ok, output = run_config(config, args.output, tasks)
            result = {"ok": bool(ok), "output": str(output)}
            print(json.dumps(result, indent=2))
            return 0 if ok else 1
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f"esbench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

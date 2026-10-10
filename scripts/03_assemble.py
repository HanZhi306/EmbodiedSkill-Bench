#!/usr/bin/env python3
"""Prepare an agent handoff, or materialize and register an agent-authored plan.

The program does not generate skill instructions from templates. An agent writes
the plan's purpose, selected module IDs, rationale, and operational instructions.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import posixpath
import re

from jsonschema import Draft202012Validator

from common import ROOT, digest, inside, payload, read_json, write_json


def prepare_handoff(data_root, request_path):
    """Expose the full classified pool and a user need, without fixing a package."""
    request = read_json(request_path)
    contracts = read_json(data_root / "02_contracts.json")
    if not request.get("purpose"):
        raise ValueError("request requires a purpose")
    handoff = {
        "protocol": "embodied-module-v1",
        "request": request,
        "prompt": os.path.relpath(ROOT / "prompts/assemble_skill.md", data_root),
        "inputs": {"cards": "01_cards.json", "inventory": "01_inventory.json",
                   "groups": "02_groups.json", "contracts": "02_contracts.json"},
        "sha256": {name: digest((data_root / name).read_bytes()) for name in
                   ("01_cards.json", "01_inventory.json", "02_groups.json", "02_contracts.json")},
        "prompt_sha256": digest((ROOT / "prompts/assemble_skill.md").read_bytes()),
        "module_count": len(contracts["modules"]),
        "agent_output": "One JSON assembly plan conforming to protocol.json:assembly_plan_schema.",
        "construction_status": "ready_for_agent_assembly",
    }
    write_json(data_root / "03_handoff.json", handoff)
    return handoff


def dependency_closure(selected, contracts):
    """Attach contract metadata and explicitly linked source files; never install skills."""
    closure = set(selected)
    while True:
        expanded = closure | {d for m in closure for d in contracts[m]["dependencies"]}
        if expanded == closure:
            return sorted(closure)
        closure = expanded


def check_instructions(text, available):
    """Check local links and unfinished scaffolds, without claiming semantic proof."""
    if re.search(r"\b(?:TODO|TBD|PLACEHOLDER)\b", text):
        raise ValueError("instructions contain an unfinished placeholder")
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        target = target.split("#", 1)[0]
        if target not in available:
            raise ValueError(f"unresolved local instruction link: {target}")


def materialize_plan(plan, data_root, repo_root):
    """Build the agent's plan, validate integrity, then index the completed package."""
    protocol = read_json(ROOT / "protocol.json")
    Draft202012Validator(protocol["assembly_plan_schema"]).validate(plan)
    cards = read_json(data_root / "01_cards.json")
    contracts = read_json(data_root / "02_contracts.json")["modules"]
    missing = sorted(set(plan["modules"]) - set(contracts))
    if missing:
        raise ValueError(f"unknown module IDs: {missing}")
    selected = dependency_closure(plan["modules"], contracts)
    targets = {contracts[m]["target"] for m in selected}
    check_instructions(plan["instructions"], targets)
    # Validate every selected source before making a package directory.
    source_bytes = {m: payload(data_root, cards[m]) for m in selected}
    materialized_bytes = dict(source_bytes)
    for module_id in selected:
        links = contracts[module_id].get("local_links", {})
        if links:
            text = source_bytes[module_id].decode("utf-8")
            parent = posixpath.dirname(contracts[module_id]["target"])
            for link, target_module in links.items():
                relative = posixpath.relpath(contracts[target_module]["target"], parent)
                if "#" in link:
                    relative += "#" + link.split("#", 1)[1]
                text = text.replace("](" + link + ")", "](" + relative + ")")
            materialized_bytes[module_id] = text.encode("utf-8")
    repo_root = Path(repo_root).resolve()
    destination = inside(repo_root, plan["name"])
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite an existing skill: {destination}")
    index_path = repo_root / "index.json"
    index = read_json(index_path) if index_path.exists() else {"protocol": "embodied-module-v1", "skills": []}
    if any(x["name"] == plan["name"] for x in index["skills"]):
        raise ValueError("skill name already registered")
    destination.mkdir(parents=True)
    for module_id in selected:
        path = inside(destination, contracts[module_id]["target"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(materialized_bytes[module_id])
    skill_text = ("---\nname: " + plan["name"] + "\ndescription: "
                  + json.dumps(plan["description"], ensure_ascii=False)
                  + "\n---\n\n" + plan["instructions"].strip() + "\n")
    (destination / "SKILL.md").write_text(skill_text, encoding="utf-8")
    write_json(destination / "modules.json", {
        "protocol": "embodied-module-v1", "rationale": plan["rationale"],
        "selected_by_agent": plan["modules"],
        "modules": {m: {"contract": contracts[m], "source": cards[m]["source"],
                        "materialized_sha256": digest(materialized_bytes[m]),
                        "transformation": "relative-link relocation" if materialized_bytes[m] != source_bytes[m] else "none"}
                    for m in selected},
    })
    for module_id in selected:
        copied = inside(destination, contracts[module_id]["target"]).read_bytes()
        if copied != materialized_bytes[module_id]:
            raise ValueError(f"materialized payload drift: {module_id}")
    index["skills"].append({"name": plan["name"], "description": plan["description"],
                            "path": plan["name"], "module_ids": selected,
                            "skill_md_sha256": digest(skill_text.encode())})
    index["skills"].sort(key=lambda item: item["name"])
    write_json(index_path, index)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--request", type=Path, default=ROOT / "inputs/request.json")
    materialize = sub.add_parser("materialize")
    materialize.add_argument("--plan", type=Path, required=True)
    materialize.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_handoff(args.data, args.request)
        print(f"Agent handoff ready: {result['module_count']} modules")
    else:
        print(materialize_plan(read_json(args.plan), args.data, args.repo))

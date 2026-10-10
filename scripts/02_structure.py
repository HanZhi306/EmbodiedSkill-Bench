#!/usr/bin/env python3
"""Classify cards, then normalize resource access and native interface contracts."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import posixpath
import re

from jsonschema import Draft202012Validator

from common import ROOT, digest, payload, read_json, unique_append, write_json


def classify_card(card, rules):
    """The decision uses the collection card, not a target skill name."""
    return rules[card["type"]]


def make_binding(metadata, metadata_module=None):
    """Preserve native schemas and effects; do not infer coordinate conversions."""
    interaction = metadata.get("interaction", {})
    execution = metadata.get("execution") or {}
    return {
        "protocol": "esb-runtime-v1",
        "skill_id": metadata["id"],
        "arguments_schema": metadata.get("arguments_schema"),
        "interaction": interaction,
        "environments": metadata.get("environments", []),
        "provider": execution.get("provider"),
        "model_profile": execution.get("model_profile"),
        "execution": execution,
        "dependencies": metadata.get("dependencies", {}),
        "source_status": metadata.get("source", {}).get("implementation", {}),
        "metadata_module": metadata_module,
        "coordinate_convention": "Native schema and source documentation; no implicit unit/frame conversion.",
        "result_semantics": "Preserve the native result. Controller completion is not task success.",
    }


def normalize_code(card, inventory, metadata, raw):
    """Attach verified declarations; arbitrary code stays source-only."""
    bindings = {}
    if card["source"]["extension"] == ".json":
        native = json.loads(raw)
        if isinstance(native, dict) and "arguments_schema" in native and "id" in native:
            binding = make_binding(native)
            Draft202012Validator.check_schema(binding["arguments_schema"])
            bindings["declared-interface"] = binding
    # A recognized existing ESB wrapper is explicitly bound to its own metadata.
    if card["source"]["extension"] == ".py":
        text = raw.decode("utf-8")
        if "from embodied_skill_bench.skill_entry import main" in text:
            tree = ast.parse(text)
            names = {node.args[0].value for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                     and node.func.id == "main" and node.args
                     and isinstance(node.args[0], ast.Constant)}
            for occurrence in card["source"]["occurrences"]:
                owner = occurrence.get("owner")
                if owner in metadata:
                    meta = metadata[owner]
                    if meta["id"] not in names:
                        raise ValueError(f"wrapper/metadata mismatch: {owner}")
                    if meta.get("arguments_schema") is None:
                        continue
                    binding = make_binding(meta, inventory["packages"][owner]["metadata_module"])
                    Draft202012Validator.check_schema(binding["arguments_schema"])
                    bindings[owner] = binding
    return {"mode": "native_interface" if bindings else "source_only",
            "bindings": bindings,
            "source_only_reason": None if bindings else
            "Retained implementation or method fragment; integration requires its native runtime and dependencies."}


def normalize_module(module_id, card, component, inventory, metadata, data_root):
    """A common descriptor serves code, knowledge, and static assets alike."""
    raw = payload(data_root, card)
    extension = card["source"]["extension"] or ".txt"
    result = {
        "protocol": "embodied-module-v1", "component": component,
        "target": f"{component}/{module_id}{extension}",
        "source_sha256": card["source"]["sha256"],
        "access": {"mode": "read", "bindings": {}},
        "dependencies": [],
        "limitations": [],
    }
    if component == "scripts":
        result["access"] = normalize_code(card, inventory, metadata, raw)
        for binding in result["access"]["bindings"].values():
            if binding["metadata_module"]:
                unique_append(result["dependencies"], binding["metadata_module"])
        result["limitations"].append("A normalized contract does not establish runtime availability or robot performance.")
    elif card["source"]["availability"] == "locator_only":
        result["limitations"].append("Only the recorded locator/citation is included; upstream full text or code must be obtained separately.")
    elif card["type"] == "model":
        result["limitations"].append("Descriptor only; model weights and native serving runtime remain external.")
    return result


def resolve_reference_links(cards, normalized, data_root):
    """Resolve original sibling links before flattening selected resource paths."""
    by_uri = {}
    for module_id, card in cards.items():
        for occurrence in card["source"]["occurrences"]:
            by_uri.setdefault(occurrence["uri"], set()).add(module_id)
    for module_id, card in cards.items():
        if card["source"]["extension"] not in {".md", ".txt"}:
            continue
        links = re.findall(r"\]\(([^)]+)\)", payload(data_root, card).decode("utf-8"))
        resolutions = {}
        for link in links:
            if "://" in link or link.startswith("#"):
                continue
            path = link.split("#", 1)[0]
            targets = set()
            for occurrence in card["source"]["occurrences"]:
                uri = occurrence["uri"]
                if uri.startswith("workspace:") and "#" not in uri:
                    relative = posixpath.normpath(posixpath.join(posixpath.dirname(uri.removeprefix("workspace:")), path))
                    targets.update(by_uri.get("workspace:" + relative, set()))
            if len(targets) == 1:
                target = next(iter(targets))
                resolutions[link] = target
                unique_append(normalized[module_id]["dependencies"], target)
            else:
                normalized[module_id]["limitations"].append(
                    f"Local source link {link!r} has {len(targets)} collected resolutions; author must resolve its context.")
        if resolutions:
            normalized[module_id]["local_links"] = resolutions


def structure(data_root, protocol_path):
    cards = read_json(data_root / "01_cards.json")
    inventory = read_json(data_root / "01_inventory.json")
    protocol = read_json(protocol_path)
    validator = Draft202012Validator(protocol["card_schema"])
    metadata = {owner: json.loads(payload(data_root, cards[row["metadata_module"]]))
                for owner, row in inventory["packages"].items()}
    groups = {name: [] for name in ("scripts", "references", "assets")}
    normalized = {}
    for module_id, card in cards.items():
        validator.validate(card)
        component = classify_card(card, protocol["component_rules"])
        groups[component].append(module_id)
        normalized[module_id] = normalize_module(module_id, card, component, inventory, metadata, data_root)
    resolve_reference_links(cards, normalized, data_root)
    write_json(data_root / "02_groups.json", groups)
    write_json(data_root / "02_contracts.json", {
        "protocol_sha256": digest(protocol_path.read_bytes()),
        "cards_sha256": digest((data_root / "01_cards.json").read_bytes()),
        "inventory_sha256": digest((data_root / "01_inventory.json").read_bytes()),
        "modules": normalized,
    })
    return dict(Counter(x["component"] for x in normalized.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    args = parser.parse_args()
    print(structure(args.data, args.protocol))

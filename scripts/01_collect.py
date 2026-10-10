#!/usr/bin/env python3
"""Collect existing modules, three-field cards, and a deduplicated source store.

Nothing in the source workspace is executed or changed. Existing package names
are provenance links, not a prescription for the skills an agent will assemble.
"""
from __future__ import annotations

import argparse
from collections import Counter
import mimetypes
from pathlib import Path
import re
import zipfile

from common import ROOT, TYPES, digest, encode, read_json, unique_append, write_json


class Collector:
    def __init__(self, config, output):
        self.config = config
        self.workspace = Path(config["workspace"]).resolve()
        self.output = Path(output)
        self.cards = {}
        self.packages = {}
        self.raw_assets = {}
        self.inputs = {}
        self.scopes = []
        self.omissions = []
        self.warnings = []

    def record_file(self, path):
        path = Path(path)
        data = path.read_bytes()
        self.inputs[self.uri(path)] = digest(data)
        return data

    def uri(self, path):
        path = Path(path).absolute()
        if path.is_relative_to(self.workspace):
            return "workspace:" + path.relative_to(self.workspace).as_posix()
        return path.as_uri()

    def add_card(self, kind, description, data, occurrence, *, extension=".txt",
                 availability="local_snapshot"):
        """Only type/description/source are card fields; identity is the map key.

        Same bytes and resource type share one module. Different types may share
        the same payload. Source occurrences preserve all original locations.
        """
        if kind not in TYPES or not description.strip():
            raise ValueError(f"invalid card: {kind}, {description!r}")
        sha = digest(data)
        module_id = kind + "-" + sha[:20]
        snapshot = f"objects/{sha}"
        target = self.output / snapshot
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError("content-address collision")
        else:
            target.write_bytes(data)
        if module_id not in self.cards:
            self.cards[module_id] = {
                "type": kind,
                "description": description.strip(),
                "source": {"sha256": sha, "snapshot": snapshot,
                           "extension": extension, "bytes": len(data),
                           "media_type": mimetypes.guess_type("source" + extension)[0]
                           or "application/octet-stream",
                           "availability": availability, "occurrences": []},
            }
        unique_append(self.cards[module_id]["source"]["occurrences"], occurrence)
        return module_id

    def add_file(self, path, kind, description, *, owner=None, role="source_file"):
        path = Path(path)
        if not path.is_file():
            self.omissions.append({"uri": self.uri(path), "reason": "missing local file"})
            return None
        if path.stat().st_size > self.config["max_file_bytes"]:
            self.omissions.append({"uri": self.uri(path), "reason": "external large asset",
                                   "bytes": path.stat().st_size})
            return None
        occurrence = {"uri": self.uri(path), "role": role}
        if owner:
            occurrence["owner"] = owner
        return self.add_card(kind, description, self.record_file(path), occurrence,
                             extension=path.suffix)

    def collect_references(self, value, origin, owner=None, keypath=""):
        """Extract upstream locators without claiming their contents were fetched."""
        if isinstance(value, list):
            for i, item in enumerate(value):
                self.collect_references(item, origin, owner, f"{keypath}/{i}")
            return
        if not isinstance(value, dict):
            return
        candidates = []
        arxiv = value.get("arxiv")
        if isinstance(arxiv, str):
            if re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", arxiv):
                candidates.append(("paper", "https://arxiv.org/abs/" + arxiv))
            else:
                unique_append(self.warnings, {"uri": origin, "field": keypath + "/arxiv",
                                              "reason": "noncanonical inherited citation", "value": arxiv})
        repo = value.get("repository")
        if isinstance(repo, str):
            if repo.startswith("https://github.com/") or re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
                # checkpoint records are collected separately as model descriptors.
                is_model = "checkpoint" in keypath or "local_snapshot" in value
                if not is_model:
                    uri = repo if repo.startswith("https://") else "https://github.com/" + repo
                    candidates.append(("repository", uri.removesuffix(".git").rstrip("/")))
        for kind, uri in candidates:
            occurrence = {"uri": origin + "#" + keypath, "role": "upstream_citation",
                          "claim": value}
            if owner:
                occurrence["owner"] = owner
            description = str(value.get("title") or value.get("paper") or uri)
            self.add_card(kind, description, encode({"uri": uri}), occurrence,
                          extension=".json", availability="locator_only")
        checkpoint = value.get("checkpoint")
        if checkpoint is not None:
            self.add_card("model", "Checkpoint descriptor: " + str(checkpoint),
                          encode(checkpoint), {"uri": origin + "#" + keypath + "/checkpoint",
                                               "role": "checkpoint_descriptor",
                                               **({"owner": owner} if owner else {})},
                          extension=".json", availability="descriptor_only")
        for key, item in value.items():
            if key != "checkpoint":
                self.collect_references(item, origin, owner, keypath + "/" + key)

    def collect_asset_library(self, root):
        """Import every source file in the archived 71-asset lineage, with hashes."""
        manifest = read_json(root / "manifest.json")
        self.record_file(root / "manifest.json")
        self.scopes.append({"path": self.uri(root), "kind": "archived-source-assets",
                            "entries": len(manifest["assets"])})
        for row in manifest["assets"]:
            folder = root / row["path"]
            asset = read_json(folder / "asset.json")
            asset_bytes = self.record_file(folder / "asset.json")
            owner = "archive-" + digest(asset_bytes)[:20]
            if owner not in self.raw_assets:
                self.raw_assets[owner] = {"skill_id": asset["target_skill_id"],
                                          "asset_kind": asset["asset_kind"],
                                          "origin": asset["origin"],
                                          "migration_audit": asset["migration_audit"],
                                          "source_metadata": asset["source_metadata"],
                                          "paths": [], "modules": []}
            record = self.raw_assets[owner]
            unique_append(record["paths"], self.uri(folder))
            for file in asset["files"]:
                path = folder / file["path"]
                data = self.record_file(path)
                if digest(data) != file["sha256"] or len(data) != file["bytes"]:
                    raise ValueError(f"archived source drift: {path}")
                if path.suffix == ".py":
                    kind, role = "control_function", "pinned_controller_extract"
                elif path.name == "interface.json":
                    kind = "perception_interface" if asset["asset_kind"] == "perception-model-interface" else "control_function"
                    role = "interface_declaration"
                else:
                    kind, role = "experience_material", "archived_evidence"
                module = self.add_file(path, kind, asset["functional_description"], owner=owner, role=role)
                unique_append(record["modules"], module)
            self.collect_references(asset["source_metadata"], self.uri(folder / "asset.json"), owner)

    def collect_pool(self, root):
        metadata_files = sorted(root.glob("*/*/metadata.json"))
        self.scopes.append({"path": self.uri(root), "kind": "existing-skill-pool",
                            "entries": len(metadata_files)})
        for metadata_path in metadata_files:
            folder = metadata_path.parent
            meta = read_json(metadata_path)
            # These files are source modules; UI files, caches and weights are not.
            files = sorted(p for p in folder.rglob("*") if p.is_file()
                           and "agents" not in p.relative_to(folder).parts
                           and "__pycache__" not in p.parts
                           and p.suffix in {".py", ".json", ".md", ".txt", ".yaml", ".yml"}
                           and p.stat().st_size <= self.config["max_file_bytes"])
            file_hashes = {p.relative_to(folder).as_posix(): digest(self.record_file(p)) for p in files}
            owner = "package-" + digest(encode(file_hashes))[:20]
            if owner not in self.packages:
                self.packages[owner] = {"skill_id": meta["id"], "kind": meta["kind"],
                                        "paths": [], "scopes": [], "modules": [],
                                        "metadata_module": None}
            record = self.packages[owner]
            unique_append(record["paths"], self.uri(folder))
            scope = "paper-main" if root == self.workspace / self.config["main_pool"] else "other-existing-version"
            unique_append(record["scopes"], scope)
            has_script = any(p.relative_to(folder).parts[0] == "scripts" for p in files)
            for path in files:
                rel = path.relative_to(folder)
                if path.name == "metadata.json":
                    if not has_script and meta["kind"] in {"action", "perception"} and meta.get("arguments_schema"):
                        kind = "perception_interface" if meta["kind"] == "perception" else "control_function"
                        role = "interface_declaration"
                    else:
                        kind, role = "support_asset", "native_contract"
                elif rel.parts[0] == "scripts":
                    kind = "perception_interface" if meta["kind"] == "perception" else "control_function"
                    role = "provider_wrapper"
                elif rel.parts[0] == "assets":
                    kind, role = "support_asset", "static_configuration"
                else:
                    kind, role = "experience_material", "existing_usage_document"
                module = self.add_file(path, kind, f"{meta['description']} [{rel.as_posix()}]",
                                       owner=owner, role=role)
                unique_append(record["modules"], module)
                if path.name == "metadata.json":
                    record["metadata_module"] = module
            self.collect_references(meta.get("source", {}), self.uri(metadata_path), owner, "/source")

    def collect_profiles(self, path):
        data = read_json(path)
        self.record_file(path)
        for name, profile in (data.get("profiles") or data.get("candidates") or {}).items():
            self.add_card("model", f"{name}: {profile.get('native_observation_contract', profile.get('format', 'model profile'))}",
                          encode(profile), {"uri": self.uri(path) + "#/" + name,
                                            "role": "model_profile", "profile_id": name},
                          extension=".json", availability="descriptor_only")

    def collect_bibliography(self):
        """Add actual paper bibliography records for the construction's sources."""
        path = self.workspace / self.config["paper_zip"]
        self.record_file(path)
        keys = {"kim2025oft", "physicalintelligence2025pi05", "shukor2025smolvla",
                "zheng2025xvla", "carion2025sam3", "clark2026molmo2",
                "sundermeyer2021contactgraspnet", "zhang2026rats",
                "zhang2026harnessvla", "ding2026zetta", "lu2026aspire",
                "fbddcz2026embodiedjev", "typesafe2026jev"}
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.endswith("/references.bib"))
            bibliography = archive.read(name).decode("utf-8")
        for block in re.split(r"\n(?=@)", bibliography):
            match = re.match(r"@\w+\{([^,]+),", block.strip())
            if not match or match[1] not in keys:
                continue
            eprint = re.search(r"eprint=\{([^}]+)\}", block)
            url = re.search(r"url=\{([^}]+)\}", block)
            uri = "https://arxiv.org/abs/" + eprint[1] if eprint else url[1]
            kind = "repository" if "github.com" in uri else "paper" if eprint else "experience_material"
            self.add_card(kind, "Paper bibliography source: " + match[1], encode({"uri": uri}),
                          {"uri": self.uri(path) + "!/" + name + "#" + match[1],
                           "role": "paper_bibliography", "citation": block.strip()},
                          extension=".json", availability="locator_only")

    def finish(self):
        for card in self.cards.values():
            card["source"]["occurrences"].sort(key=lambda x: encode(x))
        write_json(self.output / "01_cards.json", self.cards)
        write_json(self.output / "01_inventory.json", {
            "schema_version": 1, "workspace_uri_prefix": "workspace:",
            "source_config_sha256": digest(encode(self.config)),
            "scope": self.scopes, "packages": self.packages, "raw_assets": self.raw_assets,
            "source_file_sha256": self.inputs, "omissions": self.omissions,
            "warnings": self.warnings,
            "counts": {"modules": len(self.cards), "payloads": len({c['source']['sha256'] for c in self.cards.values()}),
                       "module_types": dict(Counter(c["type"] for c in self.cards.values())),
                       "package_versions": len(self.packages),
                       "package_occurrences": sum(len(p["paths"]) for p in self.packages.values()),
                       "unique_existing_skill_ids": len({p["skill_id"] for p in self.packages.values()}),
                       "paper_main_packages": sum("paper-main" in p["scopes"] for p in self.packages.values())},
        })


def discover_roots(workspace, patterns):
    return sorted({path for pattern in patterns for path in workspace.glob(pattern) if path.is_dir()})


def collect(config, output):
    collector = Collector(config, output)
    for root in discover_roots(collector.workspace, config["asset_patterns"]):
        collector.collect_asset_library(root)
    for root in discover_roots(collector.workspace, config["pool_patterns"]):
        collector.collect_pool(root)
    for pattern in config["profile_patterns"]:
        for path in sorted(collector.workspace.glob(pattern)):
            collector.collect_profiles(path)
    for item in config.get("implementation_patterns", []):
        for path in sorted(collector.workspace.glob(item["pattern"])):
            collector.add_file(path, item["type"], item["description"], role="native_provider_implementation")
    for item in config["extra_files"]:
        collector.add_file(collector.workspace / item["path"], item["type"], item["description"],
                           role="extra_existing_module")
    collector.collect_bibliography()
    collector.finish()
    return len(collector.cards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "inputs/sources.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    print(f"Collected {collect(read_json(args.config), args.output)} unique modules into {args.output}")

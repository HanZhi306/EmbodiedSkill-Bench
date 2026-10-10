#!/usr/bin/env python3
"""Reject private skill assets and credentials in the staged public file set."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRS = {
    "skill-pool", "data", "datasets", "checkpoints", "models", "episodes",
    "runs", "outputs", "output", ".codex", ".claude", ".kimi", ".deps",
}
PRIVATE_NAMES = {
    "skills-index.json", "SKILLS.md", "SKILL.md", "metadata.json", ".hf_token",
    "01_cards.json", "01_inventory.json", "02_groups.json", "02_contracts.json",
    "03_handoff.json",
}
PRIVATE_SUFFIXES = {
    ".safetensors", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".npy",
    ".npz", ".zip", ".tar", ".gz", ".pem", ".key", ".log", ".jsonl",
}
SECRET_PATTERNS = (
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def path_violation(path: str) -> str | None:
    value = Path(path)
    if PRIVATE_DIRS.intersection(value.parts):
        return "private asset/output directory"
    if value.name in PRIVATE_NAMES:
        return "private skill or corpus file"
    if (value.name == ".env" or value.name.startswith(".env.")
            or value.name.startswith("credentials") or ".local." in value.name
            or value.parts[:2] == ("configs", "local")):
        return "local credentials or configuration"
    if value.suffix.lower() in PRIVATE_SUFFIXES:
        return "model, archive, credential, or raw log file"
    return None


def check_staged(root: Path = ROOT) -> list[str]:
    # Read the index bytes that will be committed, not a potentially different
    # working-tree file. This also makes the check useful as a pre-commit hook.
    listing = subprocess.check_output(["git", "-C", str(root), "ls-files", "--stage", "-z"])
    problems = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        header, raw_path = entry.split(b"\t", 1)
        mode, object_id, stage = header.decode().split()
        path = raw_path.decode("utf-8")
        reason = path_violation(path)
        if reason:
            problems.append(f"{path}: {reason}")
            continue
        if mode not in {"100644", "100755"} or stage != "0":
            problems.append(f"{path}: symlink, submodule, or unresolved index entry")
            continue
        payload = subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", object_id])
        if len(payload) > 3 * 1024 * 1024:
            problems.append(f"{path}: exceeds the public code/image size limit")
        if any(pattern.search(payload) for pattern in SECRET_PATTERNS):
            problems.append(f"{path}: possible credential (value withheld)")
    return problems


def main() -> int:
    problems = check_staged()
    if problems:
        print("Public release check failed:\n" + "\n".join(problems), file=sys.stderr)
        return 1
    print("Public release check passed: no private assets or detected credentials in tracked index bytes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

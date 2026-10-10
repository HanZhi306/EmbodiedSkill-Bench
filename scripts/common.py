"""Small deterministic I/O helpers shared by the three construction stages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TYPES = ("model", "control_function", "perception_interface", "paper",
         "repository", "experience_material", "support_asset")


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode(value))


def inside(root, relative):
    """Reject traversal and escaping links in generated manifests or plans."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path leaves root: {relative}")
    return path


def payload(data_root, card):
    path = inside(data_root, card["source"]["snapshot"])
    data = path.read_bytes()
    if digest(data) != card["source"]["sha256"]:
        raise ValueError(f"snapshot hash mismatch: {path}")
    return data


def unique_append(items, value):
    if value not in items:
        items.append(value)

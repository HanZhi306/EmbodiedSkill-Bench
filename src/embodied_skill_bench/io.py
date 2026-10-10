from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_jsonl(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
        stream.write("\n")
        stream.flush()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    for candidate in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = candidate.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_repository_provenance(root: str | Path | None = None) -> dict[str, Any]:
    """Return reproducibility metadata for the benchmark source checkout.

    Experiment execution must remain usable from source archives that are not
    Git repositories, so an unavailable checkout is represented explicitly
    instead of raising. Batch plans can choose to require an available, clean
    checkout and validate the recorded commit before any simulator starts.
    """

    repository = Path(root or project_root()).expanduser().resolve()

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repository), *arguments],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    try:
        top_level = Path(git("rev-parse", "--show-toplevel")).resolve()
        commit = git("rev-parse", "HEAD")
        tags_text = git("tag", "--points-at", "HEAD")
        tracked_status = git("status", "--porcelain", "--untracked-files=no")
        full_status = git("status", "--porcelain", "--untracked-files=normal")
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {
            "available": False,
            "requested_root": str(repository),
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        }
    status_lines = [line for line in full_status.splitlines() if line.strip()]
    tracked_lines = [line for line in tracked_status.splitlines() if line.strip()]
    return {
        "available": True,
        "root": str(top_level),
        "commit": commit,
        "exact_tags": sorted(line for line in tags_text.splitlines() if line),
        "clean": not status_lines,
        "tracked_clean": not tracked_lines,
        "status_entry_count": len(status_lines),
        "tracked_status_entry_count": len(tracked_lines),
        "status_sha256": hashlib.sha256(
            full_status.encode("utf-8")
        ).hexdigest(),
    }


def redact_url(url: str | None) -> str | None:
    if not url:
        return None
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path.rstrip("/"), "", ""))

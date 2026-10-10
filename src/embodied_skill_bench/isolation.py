from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


class IsolationUnavailableError(RuntimeError):
    pass


def _parents_for(paths: Iterable[Path]) -> list[Path]:
    parents: set[Path] = set()
    for path in paths:
        current = path.parent
        while current != Path("/"):
            parents.add(current)
            current = current.parent
    return sorted(parents, key=lambda item: (len(item.parts), str(item)))


def _agent_cli_mount(executable: str) -> Path | None:
    path = Path(executable)
    if not path.is_absolute() or not path.exists():
        resolved = shutil.which(executable)
        if not resolved:
            return None
        path = Path(resolved)
    path = path.absolute()
    for parent in (path, *path.parents):
        if parent.name == "agent-clis":
            return parent
        # A Python-hosted agent bridge needs its interpreter, stdlib links,
        # and installed SDK packages together. Bind the whole virtualenv
        # read-only instead of only its console script or interpreter symlink.
        if (parent / "pyvenv.cfg").is_file():
            return parent
        # actions/setup-python and several Conda-style installations are
        # complete Python prefixes without a pyvenv.cfg.  Binding only their
        # bin/python executable leaves the stdlib and site-packages outside
        # the mount namespace, so the agent bridge exits before inference.
        # /usr is already mounted wholesale below; detect only non-system
        # prefixes such as /opt/hostedtoolcache/Python/<version>/x64.
        python_lib = parent / "lib"
        if (
            parent != Path("/")
            and not parent.is_relative_to(Path("/usr"))
            and (parent / "bin").is_dir()
            and python_lib.is_dir()
            and any(
                candidate.is_dir() and candidate.name.startswith("python")
                for candidate in python_lib.iterdir()
            )
        ):
            return parent
    resolved = path.resolve()
    if resolved.is_relative_to(Path("/usr")) or resolved.is_relative_to(Path("/bin")):
        return None
    return path


def build_bwrap_command(
    inner_command: list[str],
    *,
    workspace: Path,
    output_dir: Path,
    read_only_paths: Iterable[Path] = (),
    read_only_path_mappings: Iterable[tuple[Path, Path]] = (),
) -> tuple[list[str], dict[str, object]]:
    """Build a fail-closed filesystem sandbox around an agent CLI.

    Network is deliberately shared so the CLI can reach its provider and the
    localhost runtime. The mount namespace contains only allowlisted system
    runtime files, the staged workspace, the agent's own output/home, the CLI,
    and explicitly supplied observation roots. It contains no host `/proc`, so
    `/proc/<pid>/root` cannot be used to escape the mount namespace.
    """

    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise IsolationUnavailableError(
            "strict agent isolation requires bubblewrap (`bwrap`)"
        )
    workspace = workspace.resolve()
    output_dir = output_dir.resolve()
    allowed_ro = [path.resolve() for path in read_only_paths]
    allowed_ro_mappings = [
        (source.resolve(), destination.resolve())
        for source, destination in read_only_path_mappings
    ]
    for path in (
        workspace,
        output_dir,
        *allowed_ro,
        *(source for source, _ in allowed_ro_mappings),
        *(destination for _, destination in allowed_ro_mappings),
    ):
        if not path.exists():
            raise FileNotFoundError(f"sandbox mount does not exist: {path}")

    cli_mount = _agent_cli_mount(inner_command[0])
    protected_workspace_paths = [
        workspace / relative
        for relative in (
            ".agents",
            ".claude",
            ".agent-runtime",
            "candidate-skills",
            "run_context.json",
            "installed_skills.json",
            "catalog.json",
            "task.json",
            "selector_contract.json",
            "BENCHMARK_SYSTEM.md",
            "esbench",
            "esbench-code",
            "primitive_api.json",
            "PRIMITIVE_API.md",
            "NO_SKILL_ROBOT_REFERENCE.md",
        )
        if (workspace / relative).exists()
    ]
    mount_destinations = [
        workspace,
        output_dir,
        *allowed_ro,
        *(destination for _, destination in allowed_ro_mappings),
    ]
    if cli_mount:
        mount_destinations.append(cli_mount)

    command = [
        bwrap,
        "--unshare-user",
        "--uid",
        "1000",
        "--gid",
        "1000",
        "--die-with-parent",
        "--unshare-ipc",
        "--unshare-uts",
        "--dev",
        "/dev",
        # A full host procfs would permit `/proc/<pid>/root` traversal. A new
        # PID namespace + proc mount is unavailable in some nested containers.
        # Bind procfs's magic `self` directory only: it dynamically resolves to
        # the sandboxed CLI process, exposes no other PID, and gives Rust/Bun
        # the cwd/exe/maps/fd introspection their native launchers require.
        "--dir",
        "/proc",
        "--ro-bind",
        "/proc/self",
        "/proc/self",
        "--ro-bind",
        "/proc/thread-self",
        "/proc/thread-self",
        "--dir",
        "/proc/sys",
        "--dir",
        "/proc/sys/vm",
        "--ro-bind",
        "/proc/stat",
        "/proc/stat",
        "--ro-bind",
        "/proc/sys/vm/mmap_min_addr",
        "/proc/sys/vm/mmap_min_addr",
        "--ro-bind",
        "/proc/sys/vm/overcommit_memory",
        "/proc/sys/vm/overcommit_memory",
        "--tmpfs",
        "/tmp",
    ]
    for system_path in ("/usr", "/bin", "/lib", "/lib64"):
        candidate = Path(system_path)
        if candidate.exists():
            command.extend(["--ro-bind", system_path, system_path])
    for system_path in (
        "/etc/ssl",
        "/etc/ca-certificates",
        "/etc/resolv.conf",
        "/etc/hosts",
        "/etc/nsswitch.conf",
        "/etc/passwd",
        "/etc/group",
        "/etc/ld.so.cache",
        "/etc/localtime",
    ):
        candidate = Path(system_path)
        if candidate.exists():
            command.extend(["--ro-bind", system_path, system_path])

    system_roots = {Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64"), Path("/etc")}
    for parent in _parents_for(mount_destinations):
        if any(parent == root or root in parent.parents for root in system_roots):
            continue
        command.extend(["--dir", str(parent)])

    # Coding-agent CLIs expect a writable repository root for their own
    # ephemeral files. This is a per-condition staged copy, never the source
    # repository. Re-bind all benchmark inputs and executable surfaces below
    # as read-only so the agent cannot rewrite the protocol it is evaluated on.
    command.extend(["--bind", str(workspace), str(workspace)])
    for path in protected_workspace_paths:
        command.extend(["--ro-bind", str(path), str(path)])
    command.extend(["--bind", str(output_dir), str(output_dir)])
    for path in allowed_ro:
        command.extend(["--ro-bind", str(path), str(path)])
    for source, destination in allowed_ro_mappings:
        command.extend(["--ro-bind", str(source), str(destination)])
    if cli_mount:
        command.extend(["--ro-bind", str(cli_mount), str(cli_mount)])
    command.extend(["--chdir", str(workspace), "--", *inner_command])
    return command, {
        "mode": "bubblewrap",
        "enforced": True,
        "workspace_read_only": False,
        "workspace_is_private_stage": True,
        "protected_workspace_paths": [
            str(path) for path in protected_workspace_paths
        ],
        "host_proc_exposed": False,
        "proc_view": "self_only_non_enumerable",
        "network_shared": True,
        "process_uid": 1000,
        "process_gid": 1000,
        "cli_read_only_path": str(cli_mount) if cli_mount else None,
        "read_only_paths": [str(path) for path in allowed_ro],
        "read_only_path_mappings": [
            {"source": str(source), "destination": str(destination)}
            for source, destination in allowed_ro_mappings
        ],
        "writable_paths": [str(output_dir)],
    }

"""RPent-compatible RPC server for one locked Challenge-v4 BDDL cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from functools import partial
from pathlib import Path
from typing import Any

from .challenge_env import validate_challenge_spec, wrap_challenge_env


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_suite_name(cell_id: str) -> str:
    suffix = re.sub(r"[^a-z0-9]+", "_", cell_id.lower()).strip("_")
    return f"esb_challenge_v4_{suffix}"


def _register_benchmark(
    *, suite_name: str, bddl_path: Path, init_states_path: Path, language: str
) -> type:
    import torch
    from liberopro.liberopro.benchmark import BENCHMARK_MAPPING, Task

    task = Task(
        name=bddl_path.stem,
        language=language,
        problem="EmbodiedSkillBenchChallengeV4",
        problem_folder=str(bddl_path.parent),
        bddl_file=bddl_path.name,
        init_states_file=init_states_path.name,
    )

    class ChallengeBenchmark:
        name = suite_name

        def __init__(self, task_order_index: int = 0):
            if task_order_index != 0:
                raise ValueError("challenge benchmark supports only task_order_index=0")
            self.tasks = [task]
            self.n_tasks = 1

        def get_num_tasks(self) -> int:
            return 1

        def get_task(self, index: int) -> Any:
            if index != 0:
                raise IndexError(index)
            return task

        def get_task_init_states(self, index: int) -> Any:
            if index != 0:
                raise IndexError(index)
            return torch.load(init_states_path, weights_only=False)

    BENCHMARK_MAPPING[suite_name.lower()] = ChallengeBenchmark
    return ChallengeBenchmark


def _make_env(
    *,
    bddl_path: Path,
    init_states_path: Path,
    language: str,
    spec: dict[str, Any],
    seed: int,
    max_episode_steps: int,
    audit_path: Path,
) -> Any:
    from robots.libero.env_server import build_env_cfg
    from rlinf.envs.libero.libero_env import LiberoEnv

    suite_name = _safe_suite_name(spec["challenge_cell_id"])
    benchmark_class = _register_benchmark(
        suite_name=suite_name,
        bddl_path=bddl_path,
        init_states_path=init_states_path,
        language=language,
    )
    trials = len(benchmark_class().get_task_init_states(0))
    if trials < 1:
        raise ValueError(f"no init states in {init_states_path}")
    cfg = build_env_cfg(
        task_suite_name=suite_name,
        specific_reset_id=seed % trials,
        seed=seed,
        max_episode_steps=max_episode_steps,
    )

    class ChallengeLiberoEnv(LiberoEnv):
        def get_env_fns(self):
            base_factories = super().get_env_fns()
            return [
                partial(wrap_challenge_env, factory, spec, audit_path)
                for factory in base_factories
            ]

    return ChallengeLiberoEnv(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-episode-steps", type=int, default=10_000)
    parser.add_argument("--bddl-path", required=True)
    parser.add_argument("--init-states-path", required=True)
    parser.add_argument("--task-language", required=True)
    parser.add_argument("--challenge-spec-path", required=True)
    parser.add_argument("--audit-path", required=True)
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument("--cuda-device", type=int)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("LIBERO_TYPE", "pro")
    os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
    if "mujoco" in sys.modules:
        raise RuntimeError("mujoco imported before EGL configuration")

    bddl_path = Path(args.bddl_path).resolve()
    init_states_path = Path(args.init_states_path).resolve()
    spec_path = Path(args.challenge_spec_path).resolve()
    for path in (bddl_path, init_states_path, spec_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    spec = validate_challenge_spec(json.loads(spec_path.read_text(encoding="utf-8")))

    # Import RPent's server module before the EGL helper imports mujoco.  The
    # RPent module intentionally asserts this import order at module load.
    from robots.libero import env_server as _rpent_env_server  # noqa: F401

    if args.cuda_device is not None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        from rpent.utils.egl import configure_egl_device

        configure_egl_device(args.cuda_device)
        import torch

        torch.cuda.set_device(args.cuda_device)

    raw_env = _make_env(
        bddl_path=bddl_path,
        init_states_path=init_states_path,
        language=args.task_language,
        spec=spec,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        audit_path=Path(args.audit_path).resolve(),
    )
    from robots.libero.env_server import LiberoEnvFacade

    meta = {
        "suite": args.suite,
        "task": args.task,
        "seed": args.seed,
        "max_episode_steps": args.max_episode_steps,
        "challenge_cell_id": spec["challenge_cell_id"],
        "bddl_sha256": _sha256(bddl_path),
        "init_states_sha256": _sha256(init_states_path),
    }
    LiberoEnvFacade(raw_env, meta=meta).serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()

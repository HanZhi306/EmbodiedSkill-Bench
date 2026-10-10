"""Launch an RPent VLA server after fixing all process-local RNGs.

The RPent server's Pi0.5 configuration injects stochastic flow noise during
inference.  Seeding only the LIBERO environment process therefore does not
make VLA actions reproducible: the VLA runs in a separate daemon.  This
wrapper is the daemon entry point so the seeds are installed before RPent
imports and constructs the model.
"""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path
from typing import Any


def seed_process(seed: int) -> dict[str, Any]:
    """Seed Python, NumPy and Torch and return the applied policy."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Warn rather than crash if a model dependency has no deterministic CUDA
    # implementation.  The manifest records this exact policy instead of
    # claiming bitwise determinism across hardware/software stacks.
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "torch_cpu": True,
        "torch_cuda": cuda_available,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--esbench-vla-seed", type=int, required=True)
    parser.add_argument("--esbench-server-script", required=True)
    args, server_args = parser.parse_known_args()
    server_script = Path(args.esbench_server_script).resolve()
    if not server_script.is_file():
        raise FileNotFoundError(f"RPent VLA server not found: {server_script}")
    seed_process(args.esbench_vla_seed)
    # RPent owns the remaining argument parser and expects argv[0] to name its
    # own script.
    sys.argv = [str(server_script), *server_args]
    runpy.run_path(str(server_script), run_name="__main__")


if __name__ == "__main__":
    main()

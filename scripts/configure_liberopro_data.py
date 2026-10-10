from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

EXPECTED_SUITES = tuple(
    f"libero_{family}{suffix}"
    for family in ("10", "goal", "object", "spatial")
    for suffix in ("_lan", "_object", "_swap", "_task")
)


def _dataset_paths(dataset_root: str | Path) -> tuple[Path, Path, Path]:
    root = Path(dataset_root).expanduser().resolve()
    bddl_files = root / "bddl_files"
    init_files = root / "init_files"
    if not bddl_files.is_dir() or not init_files.is_dir():
        raise ValueError(
            f"invalid LIBERO-Pro task-data root {root}: expected bddl_files/ and "
            "init_files/"
        )
    return root, bddl_files, init_files


def validate_dataset(
    dataset_root: str | Path, *, load_states: bool = True
) -> dict[str, Any]:
    """Fail closed on incomplete or zero-trial LIBERO-Pro task snapshots."""

    root, bddl_files, init_files = _dataset_paths(dataset_root)
    suite_counts = {
        suite: len(list((init_files / suite).glob("*.pruned_init")))
        for suite in EXPECTED_SUITES
    }
    bad_counts = {suite: count for suite, count in suite_counts.items() if count != 10}
    if bad_counts:
        raise ValueError(
            "LIBERO-Pro task snapshot must contain exactly 10 init files for each "
            f"of the 16 perturbation suites; mismatches={bad_counts}"
        )
    missing_bddl: list[str] = []
    zero_trial: list[str] = []
    state_counts: dict[int, int] = {}
    if load_states:
        import torch

    for suite in EXPECTED_SUITES:
        for init_path in sorted((init_files / suite).glob("*.pruned_init")):
            if not (
                bddl_files
                / suite
                / f"{init_path.stem.removesuffix('.pruned_init')}.bddl"
            ).is_file():
                missing_bddl.append(str(init_path))
            if load_states:
                states = torch.load(init_path, map_location="cpu", weights_only=False)
                count = len(states)
                state_counts[count] = state_counts.get(count, 0) + 1
                if count < 1:
                    zero_trial.append(str(init_path))
    if missing_bddl:
        raise ValueError(
            "LIBERO-Pro task snapshot has init files without matching BDDL files: "
            + ", ".join(missing_bddl[:8])
        )
    if zero_trial:
        raise ValueError(
            "LIBERO-Pro task snapshot contains zero-trial init files: "
            + ", ".join(zero_trial[:8])
        )
    return {
        "dataset_root": str(root),
        "bddl_files": str(bddl_files),
        "init_states": str(init_files),
        "suite_count": len(EXPECTED_SUITES),
        "init_file_count": sum(suite_counts.values()),
        "state_count_distribution": {
            str(key): value for key, value in sorted(state_counts.items())
        },
    }


def validate_runtime_config(
    dataset_root: str | Path, *, load_states: bool = True
) -> dict[str, Any]:
    dataset = validate_dataset(dataset_root, load_states=load_states)
    from liberopro.liberopro import config_file

    path = Path(config_file).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"LIBERO-Pro runtime config is missing: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"LIBERO-Pro runtime config is malformed: {path}")
    for field in ("bddl_files", "init_states"):
        actual = Path(str(raw.get(field, ""))).expanduser().resolve()
        expected = Path(dataset[field]).resolve()
        if actual != expected:
            raise ValueError(
                f"LIBERO-Pro runtime {field} drift: expected {expected}, got {actual}. "
                "Run scripts/configure_liberopro_data.py without --check."
            )
    return {**dataset, "config_file": str(path), "runtime_config_matches": True}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Point LIBERO-Pro at a persistent BDDL/init snapshot."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--assets-root", help="Downloaded LIBERO-PRO-assets directory")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the snapshot and current runtime config without rewriting it",
    )
    args = parser.parse_args()
    os.environ["LIBERO_TYPE"] = "pro"
    config_root = os.environ.get("LIBERO_CONFIG_PATH")
    if not config_root:
        raise SystemExit("Set LIBERO_CONFIG_PATH to a dedicated runtime config directory")
    Path(config_root).mkdir(parents=True, exist_ok=True)
    assets = Path(args.assets_root).expanduser().resolve() if args.assets_root else None
    if assets and not (assets / "scenes").is_dir():
        raise SystemExit("--assets-root must contain the downloaded scenes/ directory")

    try:
        root, bddl_files, init_files = _dataset_paths(args.dataset_root)
        if args.check:
            report = validate_runtime_config(root)
            if assets:
                raw = yaml.safe_load(Path(report["config_file"]).read_text(encoding="utf-8"))
                if Path(raw["assets"]).resolve() != assets:
                    raise ValueError("LIBERO-Pro asset path does not match --assets-root")
            print(yaml.safe_dump(report, sort_keys=True).strip())
            return
        validate_dataset(root)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    from liberopro.liberopro import config_file, get_default_path_dict

    config = get_default_path_dict()
    config["bddl_files"] = str(bddl_files)
    config["init_states"] = str(init_files)
    if assets:
        config["assets"] = str(assets)
    Path(config_file).write_text(yaml.safe_dump(config), encoding="utf-8")
    validate_runtime_config(root)
    print(f"LIBERO-Pro task data configured and validated from {root}")


if __name__ == "__main__":
    main()

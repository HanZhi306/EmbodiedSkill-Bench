#!/usr/bin/env python3
"""Freeze selected archive paths for one dataset; do not import precomputed metrics."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from evaluate import require


def from_index(index, dataset):
    index = Path(index).resolve()
    raw = index.read_bytes()
    rows = [r for r in csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
            if r["dataset_id"] == dataset and r["split"] == "Clean100"]
    require(bool(rows), f"No Clean100 rows for {dataset}")
    require(all(r["eligible"] == "True" for r in rows), "Index contains ineligible attempts")
    require(len({r["cell_id"] for r in rows}) == len(rows), "Index has duplicate tasks")
    require(len({r["model"] for r in rows}) == len({r["runner"] for r in rows}) == 1,
            "Index mixes models or runners within the requested dataset")
    tasks, episodes = [], []
    for row in sorted(rows, key=lambda r: r["cell_id"]):
        # The inspected frozen Clean100 index uses seed 26 for every selected case.
        task = {"task_id": row["cell_id"], "seed": 26}
        tasks.append(task)
        item = {**task, "summary": str((index.parent / row["summary"]).resolve())}
        if row.get("summary_sha256"):
            item["summary_sha256"] = row["summary_sha256"]
        if row.get("token_wire"):
            item["token_logs"] = [{"format": "kimi", "path": str((index.parent / row["token_wire"]).resolve())}]
        episodes.append(item)
    return {"configuration": {"id": dataset, "model": rows[0]["model"], "runner": rows[0]["runner"]},
            "tasks": tasks, "episodes": episodes,
            "selection_source": {"path": str(index), "sha256": hashlib.sha256(raw).hexdigest()},
            "selection_note": "Preserves the frozen index's selected attempts; no outcome-based reselection."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path, help="Frozen per-task.csv archive index")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = from_index(args.index, args.dataset)
        require(len(manifest["tasks"]) == 100, "This helper requires a complete Clean100 index")
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Manifest preparation failed: {error}\n")
    print(f"Frozen {len(manifest['tasks'])} task paths in {args.output}")


if __name__ == "__main__":
    main()

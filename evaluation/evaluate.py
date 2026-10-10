#!/usr/bin/env python3
"""Offline ESB trajectory -> per-task metrics -> configuration means (stdlib only)."""

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics


ACTION_EVENTS = {"action_completed", "action_unverified", "action_error"}
CLAUDE_FORMATS = {"claude", "claude-code"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, integer=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0, f"Invalid {name}: {value!r}")
    require(not integer or isinstance(value, int), f"{name} must be an integer")
    return value


def boolean(value, name):
    require(isinstance(value, bool), f"{name} must be a JSON boolean")
    return value


def resolve(base, path):
    return (base / path).resolve()


class Sources:
    """Hash exactly the source bytes used; never write to the archive."""

    def __init__(self):
        self.hashes = {}

    def read(self, path):
        path = Path(path).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        require(str(path) not in self.hashes or self.hashes[str(path)] == digest,
                f"Source changed while evaluating: {path}")
        self.hashes[str(path)] = digest
        return data

    def json(self, path):
        return json.loads(self.read(path))

    def jsonl(self, path, text_lines=None):
        for line, raw in enumerate(self.read(path).splitlines(), 1):
            if raw.strip():
                # Native CLI stdout can contain banners alongside JSON events.
                # Broken usage/session events remain errors; display text is audited.
                if text_lines is not None and not raw.lstrip().startswith(b"{"):
                    text_lines.append(line)
                    continue
                try:
                    value = json.loads(raw)
                except ValueError as error:
                    if text_lines is not None and not re.search(
                            rb'"type"\s*:\s*"(?:openhands\.|turn\.|thread\.|system"|assistant"|result")', raw):
                        text_lines.append(line)
                        continue
                    raise ValueError(f"Malformed JSON at {path}:{line}") from error
                require(isinstance(value, dict), f"Expected object at {path}:{line}")
                yield line, value


def normalize_usage(usage, cache_mode="inclusive"):
    """Provider input totals already contain cache tokens unless explicitly exclusive."""
    if not usage or usage.get("input_tokens") is None or usage.get("output_tokens") is None:
        return None
    inp = number(usage["input_tokens"], "input_tokens", True)
    out = number(usage["output_tokens"], "output_tokens", True)
    cached = number(usage.get("cached_input_tokens", usage.get("cache_read_input_tokens", 0)),
                    "cached_input_tokens", True)
    created = number(usage.get("cache_creation_input_tokens", 0), "cache_creation_input_tokens", True)
    require(cache_mode in {"inclusive", "exclusive"}, "Unknown cache_mode")
    if cache_mode == "exclusive":
        inp += cached + created
    require(cached + created <= inp, "Cache tokens exceed inclusive input tokens")
    return {"input_tokens": inp, "output_tokens": out, "tokens": inp + out}


def usage_stream(spec, base, sources):
    """Each stream is one disjoint session/component, never a second copy of a log."""
    path = resolve(base, spec["path"])
    fmt = spec["format"]
    require(fmt in {"codex", "openhands", "kimi", "requests"} | CLAUDE_FORMATS,
            f"Unsupported usage format: {fmt}")
    records, seen, pending = [], {}, None
    final, last_cumulative, thread_ids = None, None, set()
    text_lines = []

    def append(line, usage, request_id=None, cache_mode="inclusive"):
        records.append({"line": line, "request_id": request_id,
                        "usage": normalize_usage(usage, cache_mode)})

    for line, event in sources.jsonl(path, text_lines if fmt in {"codex", "openhands"} | CLAUDE_FORMATS else None):
        kind = event.get("type")
        if fmt == "codex":
            if kind == "thread.started":
                thread_ids.add(event.get("thread_id"))
                require(len(thread_ids) <= 1, "Split different Codex sessions into separate streams")
            if kind == "turn.completed":
                append(line, event.get("usage"))
                current = records[-1]["usage"]
                require(current is not None, f"Invalid turn.completed usage at {path}:{line}")
                if last_cumulative is not None:
                    require(all(current[k] >= last_cumulative[k] for k in current),
                            "Codex cumulative usage regressed; check session boundaries")
                last_cumulative = current
        elif fmt in CLAUDE_FORMATS:
            if event.get("parent_tool_use_id") is not None:
                continue
            if event.get("session_id"):
                thread_ids.add(event["session_id"])
                require(len(thread_ids) <= 1, "Split different Claude Code sessions into separate streams")
            if kind == "result":
                # The result already covers the session; assistant usage and
                # modelUsage are overlapping views, not additional requests.
                append(line, event.get("usage"), cache_mode="exclusive")
                current = records[-1]["usage"]
                require(current is not None, f"Invalid Claude Code result usage at {path}:{line}")
                if final is not None:
                    require(all(current[k] >= final[k] for k in current),
                            "Claude Code cumulative usage regressed; check session boundaries")
                final = current
        elif fmt == "openhands":
            if kind == "openhands.llm_response":
                append(line, event.get("usage"), event.get("response_id"))
                last_cumulative = normalize_usage(event.get("accumulated_usage"))
            elif kind == "openhands.result":
                final = normalize_usage(event.get("usage"))
                append(line, event.get("usage"))
        elif fmt == "kimi":
            if kind == "llm.request":
                pending = {"line": line, "request_id": f"line:{line}", "usage": None}
                records.append(pending)
            elif kind == "usage.record":
                require(pending is not None, f"Unmatched/duplicate Kimi usage at {path}:{line}")
                usage = event["usage"]
                values = {k: number(usage[k], k, True) for k in
                          ("inputOther", "inputCacheRead", "inputCacheCreation", "output")}
                pending["usage"] = normalize_usage({
                    "input_tokens": values["inputOther"] + values["inputCacheRead"] + values["inputCacheCreation"],
                    "output_tokens": values["output"],
                    "cached_input_tokens": values["inputCacheRead"],
                    "cache_creation_input_tokens": values["inputCacheCreation"]})
                pending["usage_line"] = line
                pending = None
        else:
            request_id = event["request_id"]
            require(isinstance(request_id, str) and request_id, "request_id must be a nonempty string")
            if request_id in seen:
                require(seen[request_id] == event, f"Conflicting usage for {request_id}")
                continue
            seen[request_id] = event
            append(line, event.get("usage"), request_id, spec.get("cache_mode", "inclusive"))

    measured = [r["usage"] for r in records if r["usage"] is not None]
    totals = None
    if fmt in {"codex", "openhands"} | CLAUDE_FORMATS:
        totals = final if final is not None else last_cumulative
    elif measured:
        totals = {k: sum(u[k] for u in measured) for k in measured[0]}
    return {"path": str(path), "format": fmt, "totals": totals,
            "records": [r for r in records if r["usage"] is not None],
            "non_json_stdout_lines": text_lines}


def frame_paths(observation_path, observation, sources):
    frames = []
    for camera, assets in observation.get("artifacts", {}).get("cameras", {}).items():
        asset = assets.get("rgb") or assets.get("high_rgb")
        if not asset:
            continue
        original = Path(asset["path"])
        local = observation_path.parent / "cameras" / original.name
        path = local if local.is_file() else resolve(observation_path.parent, original)
        exists = path.is_file()
        if exists:
            sources.read(path)
        frames.append({"camera": camera, "path": str(path), "available": exists})
    return frames


def evaluate_episode(item, base, sources, configuration=None):
    summary_path = resolve(base, item["summary"])
    summary = sources.json(summary_path)
    if item.get("summary_sha256"):
        require(sources.hashes[str(summary_path)] == item["summary_sha256"], "Summary differs from frozen index hash")
    require(len(summary["conditions"]) == 1, "Provide a single-condition episode summary")
    condition = summary["conditions"][0]
    episode, outcome = condition["episode"], condition["agent_outcome"]
    for declared, actual in (("model", "requested_model"), ("runner", "runner_kind")):
        if declared in (configuration or {}):
            require(configuration[declared] == outcome.get(actual),
                    f"Episode {declared} differs from configuration")
    require(condition["classification"]["eligible"] is True, "Selected attempt is not eligible")
    count = number(episode["action_count"], "action_count", True)
    latest_index = number(episode["latest_observation"]["step_index"], "latest step", True)
    require(latest_index <= count, "Final observation exceeds runtime action count")
    archived_latest = resolve(summary_path.parent, episode["latest_observation"]["observation_json"])
    root = resolve(base, item["episode_dir"]) if item.get("episode_dir") else archived_latest.parents[2]
    final = sources.json(root / "steps" / f"{latest_index:04d}" / "observation.json")
    task = final["task"]
    task_id = f"{task['suite']}:t{int(task['task_id']):02d}"
    require((task_id, task["seed"]) == (item["task_id"], item["seed"]), "Task/seed mismatch")
    success = boolean(final["status"]["success"], "official success")
    require(episode["success"] is success and condition["classification"]["success"] is success,
            "Official status conflicts with summary")

    actions, identities = {}, {}
    for line, event in sources.jsonl(root / "events.jsonl"):
        if event.get("event") not in ACTION_EVENTS or event.get("accepted", True) is False:
            continue
        step = number(event["step_index"], "action step", True)
        require(0 < step <= count, "Action step outside runtime action count")
        if step in actions:
            require(actions[step]["event"] == event, f"Conflicting events for step {step}")
            continue
        identity = event.get("request_id")
        if identity:
            require(identity not in identities, f"Request {identity} appears at multiple action steps")
            identities[identity] = step
        actions[step] = {"line": line, "event": event}
    require(len(actions) <= count, "Trace has more actions than the runtime counter")
    missing_events = sorted(set(range(1, count + 1)) - actions.keys())
    timeline, first_success = [], None
    missing_observations = []
    for step in range(count + 1):
        path = root / "steps" / f"{step:04d}" / "observation.json"
        if not path.is_file():
            missing_observations.append(step)
            continue
        obs = sources.json(path)
        require(obs["step_index"] == step and obs["task"] == task, "Mixed observation identity")
        observed_success = boolean(obs["status"]["success"], "observation success")
        if observed_success and first_success is None:
            first_success = step
        timeline.append({"step": step, "observation": str(path), "status": obs["status"],
                         "frames": frame_paths(path, obs, sources), "action": actions.get(step)})
    require(first_success is None or success, "Success disappeared within one episode")
    require(not success or first_success is not None, "Successful episode has no successful observation")

    specs = item.get("token_logs")
    if specs is None:
        runner = outcome.get("runner_kind")
        require(runner in {"codex", "openhands"} | CLAUDE_FORMATS,
                "Specify token_logs for this runner (e.g. Kimi wire)")
        specs = [{"format": runner, "path": str(resolve(summary_path.parent, outcome["stdout_path"]))}]
    require(bool(specs), "Provide at least one token log")
    paths = [resolve(base, s["path"]) for s in specs]
    require(len(paths) == len(set(paths)), "Duplicate token log would double count usage")
    streams = [usage_stream(spec, base, sources) for spec in specs]
    known = [s["totals"] for s in streams if s["totals"] is not None]
    totals = {k: sum(u[k] for u in known) for k in known[0]} if known else {}
    # Cross-check the native counter when it measures the same session.
    if len(streams) == 1 and streams[0]["format"] in {"codex", "openhands"}:
        native = normalize_usage(outcome.get("usage"))
        require(native is None or native == totals, "Raw token log conflicts with native usage counter")
    elif len(streams) == 1 and streams[0]["format"] in CLAUDE_FORMATS and totals:
        native = normalize_usage(outcome.get("usage"), "exclusive")
        require(native is None or native == totals, "Raw token log conflicts with native usage counter")

    elapsed = outcome.get("elapsed_seconds")
    if elapsed is not None:
        elapsed = number(elapsed, "elapsed_seconds")
    timing_source = "agent_outcome.elapsed_seconds (harness monotonic end minus start)"
    if item.get("timing"):
        timing = sources.json(resolve(base, item["timing"]))
        start = number(timing["start_monotonic_seconds"], "start")
        end = number(timing["end_monotonic_seconds"], "end")
        require(end >= start, "Episode end precedes start")
        require(elapsed is None or math.isclose(elapsed, end - start, abs_tol=1e-6),
                "Timing boundaries conflict with harness elapsed time")
        elapsed, timing_source = end - start, item["timing"]

    metrics = {"task_id": task_id, "seed": task["seed"], "success": success,
               "actions": len(actions) if not missing_events else count,
               "actions_from_terminal_events": len(actions),
               "actions_without_terminal_event": len(missing_events),
               "input_tokens": totals.get("input_tokens"), "output_tokens": totals.get("output_tokens"),
               "tokens": totals.get("tokens"),
               "elapsed_seconds": elapsed, "first_success_action": first_success}
    trace = {"metrics": metrics, "summary": str(summary_path), "episode_dir": str(root),
             "time_source": timing_source, "token_streams": streams, "timeline": timeline,
             "missing_action_events": missing_events, "missing_observations": missing_observations,
             "action_source": "terminal events" if not missing_events else "runtime counter; incomplete terminal events"}
    return metrics, trace


def aggregate(rows):
    def metric(key):
        values = [r[key] for r in rows if r[key] is not None]
        return {"mean": statistics.fmean(values) if values else None}

    return {key: metric(key) for key in ("success", "actions", "tokens", "elapsed_seconds")}


def evaluate(manifest_path, expected_tasks=100, allow_subset=False):
    sources = Sources()
    manifest_path = Path(manifest_path).resolve()
    manifest = sources.json(manifest_path)
    require(isinstance(manifest["configuration"], dict) and manifest["configuration"].get("id"),
            "configuration must include an id")
    tasks, episodes = manifest["tasks"], manifest["episodes"]

    def key(item):
        require(isinstance(item["task_id"], str) and item["task_id"], "Invalid task ID")
        return item["task_id"], number(item["seed"], "seed", True)

    roster = [key(t) for t in tasks]
    selected = [key(e) for e in episodes]
    require(len(roster) == expected_tasks and len(set(roster)) == expected_tasks,
            f"Roster must contain exactly {expected_tasks} distinct task/seed pairs")
    require(len({t[0] for t in roster}) == expected_tasks, "This protocol expects one seed per distinct task")
    require(len(set(selected)) == len(selected), "Select exactly one attempt per task; duplicate selection")
    require(set(selected) <= set(roster), "Selected episode is outside the frozen task roster")
    missing = sorted(set(roster) - set(selected))
    require(allow_subset or not missing, f"Missing {len(missing)} tasks; use --allow-subset for a task subset")
    require(bool(episodes), "No episodes selected")
    rows, traces = [], []
    for item in sorted(episodes, key=key):
        row, trace = evaluate_episode(item, manifest_path.parent, sources, manifest["configuration"])
        rows.append(row)
        traces.append(trace)
    metrics = aggregate(rows)
    require(allow_subset or all(r["elapsed_seconds"] is not None for r in rows), "Invalid timing records")
    report = {"configuration": manifest["configuration"], "expected_tasks": expected_tasks,
              "evaluated_tasks": len(rows), "missing_tasks": missing,
              "metrics": metrics,
              "actions_without_terminal_event": sum(r["actions_without_terminal_event"] for r in rows),
              "scope": "One manifest-selected attempt per task; failures included; task-weighted means."}
    return rows, {"report": report, "traces": traces, "sources_sha256": sources.hashes}


def replay_html(traces):
    """Local image replay, no simulator execution or copied image payloads."""
    esc = html.escape
    sections = []
    for trace in traces:
        blocks = []
        for step in trace["timeline"]:
            pictures = []
            for frame in step["frames"]:
                if frame["available"]:
                    pictures.append(f'<img loading="lazy" width="256" alt="{esc(frame["camera"])}" '
                                    f'src="{esc(Path(frame["path"]).as_uri())}">')
                else:
                    pictures.append(f'<p>Missing frame: {esc(frame["camera"])}</p>')
            event = step["action"]["event"] if step["action"] else None
            caption = json.dumps({"status": step["status"], "action": event}, ensure_ascii=False, indent=2)
            blocks.append(f'<section><h3>Step {step["step"]}</h3>{"".join(pictures)}'
                          f'<details><summary>Action and official status</summary><pre>{esc(caption)}</pre></details></section>')
        heading = esc(json.dumps(trace["metrics"], ensure_ascii=False))
        sections.append(f'<article><h2>{esc(trace["metrics"]["task_id"])}</h2><p>{heading}</p>{"".join(blocks)}</article>')
    return ('<!doctype html><meta charset="utf-8"><title>Archived trajectory replay</title>'
            '<style>body{font:16px sans-serif;max-width:1100px;margin:32px auto}section{padding:12px;border-top:1px solid #ccc}'
            'pre,p{white-space:pre-wrap;overflow-wrap:anywhere}img{margin:4px}</style>'
            '<h1>Archived trajectory replay</h1><p>Saved observations in action order. '
            'Open this file locally with the source archive available. No simulation is run.</p>' + ''.join(sections))


def write_outputs(output, rows, audit, replay=False):
    output = Path(output)
    require(not output.exists(), f"Output already exists: {output}; choose a new directory")
    output.mkdir(parents=True)
    with (output / "per_task.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, value in (("summary.json", audit["report"]), ("audit.json", audit)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if replay:
        (output / "replay.html").write_text(replay_html(audit["traces"]), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New directory for derived results")
    parser.add_argument("--expected-tasks", type=int, default=100, help="Number of task cases in the manifest roster")
    parser.add_argument("--allow-subset", action="store_true", help="Evaluate a subset of the task roster")
    parser.add_argument("--replay", action="store_true", help="Create a local HTML view of saved frames")
    args = parser.parse_args()
    try:
        rows, audit = evaluate(args.manifest, args.expected_tasks, args.allow_subset)
        write_outputs(args.output, rows, audit, args.replay)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(2, f"Evaluation failed: {error}\n")
    print(json.dumps(audit["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

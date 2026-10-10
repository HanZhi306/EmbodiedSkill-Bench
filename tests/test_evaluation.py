"""Check measurement semantics with synthetic episodes, never benchmark assets."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("offline_evaluate", ROOT / "evaluation/evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_json(self, path, value):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value), encoding="utf-8")
        return target

    def episode(self, name, task_id, success, actions, elapsed, tokens=True):
        task = {"suite": "test-suite", "task_id": task_id, "seed": 26}
        episode_dir = self.root / name / "episode"
        for step in range(actions + 1):
            self.write_json(f"{name}/episode/steps/{step:04d}/observation.json", {
                "task": task, "step_index": step,
                "status": {"success": success and step == actions}, "artifacts": {}})
        events = [{"event": "action_completed", "step_index": step, "request_id": f"request-{step}"}
                  for step in range(1, actions + 1)]
        # Duplicate terminal event, read-only query, and rejected call must not
        # create extra agent-level actions.
        events += [events[0], {"event": "perception_completed", "step_index": actions},
                   {"event": "action_error", "accepted": False, "step_index": 999}]
        (episode_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
        usage = {"input_tokens": 160, "cached_input_tokens": 60, "output_tokens": 40} if tokens else {}
        summary = {"conditions": [{
            "classification": {"eligible": True, "success": success},
            "episode": {"success": success, "action_count": actions, "latest_observation": {
                "step_index": actions, "observation_json": f"episode/steps/{actions:04d}/observation.json"}},
            "agent_outcome": {"requested_model": "test-model", "runner_kind": "codex",
                              "elapsed_seconds": elapsed, "usage": usage}}]}
        self.write_json(f"{name}/summary.json", summary)
        log = [{"type": "thread.started", "thread_id": name}]
        if tokens:
            log.extend([{"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 20}},
                        {"type": "turn.completed", "usage": usage}])
        (self.root / name / "usage.jsonl").write_text("\n".join(json.dumps(e) for e in log) + "\n", encoding="utf-8")
        return {"task_id": f"test-suite:t{task_id:02d}", "seed": 26,
                "summary": f"{name}/summary.json", "episode_dir": f"{name}/episode",
                "token_logs": [{"format": "codex", "path": f"{name}/usage.jsonl"}]}

    def manifest(self, episodes):
        return self.write_json("manifest.json", {
            "configuration": {"id": "test", "model": "test-model", "runner": "codex"},
            "tasks": [{"task_id": e["task_id"], "seed": e["seed"]} for e in episodes],
            "episodes": episodes})

    def test_official_success_deduplication_cache_usage_and_failure_means(self):
        episodes = [self.episode("success", 0, True, 2, 30),
                    self.episode("failure", 1, False, 1, 90, tokens=False)]
        rows, audit = evaluate.evaluate(self.manifest(episodes), expected_tasks=2)
        self.assertEqual([r["actions"] for r in rows], [2, 1])
        self.assertEqual(rows[0]["tokens"], 200)  # final cumulative usage, cache counted once
        self.assertIsNone(rows[1]["tokens"])
        metrics = audit["report"]["metrics"]
        self.assertEqual(metrics["success"]["mean"], 0.5)
        self.assertEqual(metrics["actions"]["mean"], 1.5)
        self.assertEqual(metrics["tokens"]["mean"], 200)
        self.assertEqual(metrics["elapsed_seconds"]["mean"], 60)

    def test_summary_cannot_override_official_failure(self):
        item = self.episode("failure", 0, False, 1, 30)
        path = self.root / "failure/summary.json"
        summary = json.loads(path.read_text())
        summary["conditions"][0]["episode"]["success"] = True
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Official status conflicts"):
            evaluate.evaluate(self.manifest([item]), expected_tasks=1)

    def test_same_task_cannot_select_multiple_attempts(self):
        item = self.episode("success", 0, True, 1, 30)
        path = self.manifest([item])
        manifest = json.loads(path.read_text())
        manifest["episodes"].append(item)
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate selection"):
            evaluate.evaluate(path, expected_tasks=1)

    def test_claude_final_usage_counts_exclusive_cache_once(self):
        events = [{"type": "assistant", "usage": {"input_tokens": 10000, "output_tokens": 10000}},
                  {"type": "result", "usage": {"input_tokens": 100, "output_tokens": 20,
                   "cache_read_input_tokens": 30, "cache_creation_input_tokens": 10}}]
        path = self.root / "claude.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
        result = evaluate.usage_stream({"format": "claude-code", "path": path.name},
                                       self.root, evaluate.Sources())
        self.assertEqual(result["totals"]["tokens"], 160)

    def test_request_usage_deduplicates_repeated_provider_record(self):
        record = {"request_id": "one-request", "usage": {"input_tokens": 100, "output_tokens": 20}}
        path = self.root / "requests.jsonl"
        path.write_text(json.dumps(record) + "\n" + json.dumps(record), encoding="utf-8")
        result = evaluate.usage_stream({"format": "requests", "path": path.name},
                                       self.root, evaluate.Sources())
        self.assertEqual(result["totals"]["tokens"], 120)


if __name__ == "__main__":
    unittest.main()

# Offline benchmark evaluation

Archived rollouts are aligned with simulator outcomes, model usage and timing
records to produce per-task metrics and configuration-level means. The scripts
use the Python standard library.

This public release includes the evaluator but does not include experimental
rollouts, usage logs, or the paper's skill assets. Supply archived episodes
locally. The result tables in the main README are manuscript-reported values.

## Input manifest

Each manifest specifies one experiment configuration, its task roster and one
selected episode per task. Paths are relative to the manifest. For example:

```json
{
  "configuration": {"id": "configuration-name", "model": "gpt-5.6-sol", "runner": "codex"},
  "tasks": [{"task_id": "libero_10_lan:t00", "seed": 26}],
  "episodes": [{
    "task_id": "libero_10_lan:t00",
    "seed": 26,
    "summary": "archive/task-000/summary.json",
    "episode_dir": "archive/task-000/episode",
    "token_logs": [{"format": "codex", "path": "archive/task-000/agent/stdout.jsonl"}]
  }]
}
```

The default roster contains 100 distinct tasks with one fixed seed each. For a
different roster size, set `--expected-tasks`; the example above requires
`--expected-tasks 1`. `--allow-subset` selects a subset of the declared roster.
The scripts preserve the supplied episode selection and do not choose attempts
by their outcomes. Declared model and runner fields are checked against the logs.

Episodes use the ESB `events.jsonl` and `steps/NNNN/observation.json` format.
`episode_dir` relocates archived observations; it can be omitted when the paths
stored in the summary remain accessible. Codex, Claude Code and OpenHands can also omit
`token_logs` to use the summary's stdout path. Kimi uses `format: "kimi"` and the
corresponding `wire.jsonl` path. Other harnesses must provide the same episode
contract or an adapter.

For Claude Code, set the configuration's `runner` to `"claude-code"` and `model`
to the archived requested model (e.g. `"claude-opus-5"`). Its native
`--output-format stream-json` stdout is accepted directly, without conversion:

```json
{"token_logs": [{"format": "claude-code", "path": "archive/task-000/agent/stdout.jsonl"}]}
```

`format: "claude"` is an alias for the same Claude Code log format.

## Metrics

| Metric | Computation |
|---|---|
| Success | Official final observation status, checked against the episode summary |
| Actions | Accepted state-changing calls, deduplicated by step/request and reconciled with the runtime counter |
| Tokens | Recorded input plus output tokens; cached input is counted once |
| Time | Harness elapsed time from agent launch to exit, including model calls, tools and waiting |

Read-only perception and queries, rejected calls, and low-level simulator steps
are not actions. Codex and Claude Code use the final cumulative usage in a
session; Kimi sums per-request usage, including compaction. Cumulative
snapshots and nested tool durations are not added repeatedly.

Claude Code reads the last top-level `result.usage`, including error results with
valid usage. Its `input_tokens` excludes cache reads and writes, so input usage is
`input_tokens + cache_read_input_tokens + cache_creation_input_tokens`; output is
`output_tokens`. Per-message usage, nested tool results and `modelUsage` are not
added again. Separate sessions must use separate streams. If no final result was
recorded, token metrics are unavailable (`null`), not inferred from partial
assistant messages; means use available task-level records. Time still comes
from the harness timer, not Claude's model-only `duration_api_ms`.

For additional model components, provide disjoint usage streams. The `requests`
format accepts one JSON object per line with `request_id` and
`usage: {"input_tokens": 100, "output_tokens": 20}`. Repeated copies of one
request are deduplicated. If a provider excludes cache tokens from input totals,
set `cache_mode: "exclusive"` on that stream and supply the cache-read and
cache-creation fields.

An optional episode `timing` path can identify a JSON file containing
`start_monotonic_seconds` and `end_monotonic_seconds`; their difference is checked
against the harness timer. Task-level values are aggregated with equal task
weights, including failures.

## Usage

From this directory, an existing frozen Clean100 `per-task.csv` index can be
converted to a manifest. This helper extracts source paths and hashes, rather
than importing precomputed metric values:

```bash
python3 -B prepare_manifest.py path/to/per-task.csv \
  --dataset sol_with_clean100 --output path/to/manifest.json
python3 -B evaluate.py path/to/manifest.json \
  --output path/to/new-results --replay
```

The same commands apply to Claude Code: select its dataset ID with `--dataset`;
the generated manifest uses the summary's stdout path automatically.

The output directory must be new. The script writes `per_task.csv`,
`summary.json` and `audit.json`, with source hashes, observation references and
usage source lines. `--replay` additionally produces an HTML view of saved camera
frames in action order. Open the page locally with the original images available;
it references those files without copying them or running the simulator.

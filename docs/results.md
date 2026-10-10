# Reported manuscript results

These tables transcribe the manuscript's Clean100 results. They are reported results, not a fresh evaluation of this public code checkout. Raw rollout archives, selected skill sets, and the 250-skill library are not included; see [release scope](release-scope.md).

## Experimental settings

Clean100 uses 25 base LIBERO-Pro tasks crossed with four perturbations (language, object, task, swap), yielding 100 cases at seed 26. The domain counts are LIBERO-10 / Goal / Object / Spatial = 24 / 28 / 24 / 24. Each perturbation contributes 25 cases.

| Model label in the paper | Harness | Recorded CLI version |
| :--- | :--- | :--- |
| GPT-5.6-Sol | Codex CLI | 0.148.0 |
| GPT-6-Astra | Codex CLI | 0.148.0 |
| Kimi-K3 | Kimi Code CLI | 0.40.0 |
| Claude-Opus-5 | Claude Code | 2.1.186 |

The Selected condition uses task-specific skill sets selected by GPT-5.6-Sol from the 250 candidates. GPT-5.6-Sol, Kimi-K3, and Claude-Opus-5 receive the same selected set for each task. A fixed starter `skill_set` in the public launcher does not reproduce these selections.

## RQ1 · Effectiveness

Overall columns from manuscript Table 1. SR is a percentage; actions, tokens, and time are per-task means including failures.

| Agent | Skill setting | SR (%) | Actions | Tokens (M) | Time (min) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| GPT-5.6-Sol | Native / no skill | 11.0 | 26.04 | 5.354 | 12.01 |
| GPT-5.6-Sol | Selected / with skill | 63.0 | 2.38 | 0.803 | 4.40 |
| Kimi-K3 | Native / no skill | 43.0 | 17.93 | 2.819 | 50.10 |
| Kimi-K3 | Selected / with skill | 62.0 | 4.37 | 1.185 | 15.36 |
| Claude-Opus-5 | Native / no skill | 44.0 | 9.39 | 2.821 | 19.99 |
| Claude-Opus-5 | Selected / with skill | 64.0 | 3.48 | 0.656 | 7.28 |

Success improves by 52, 19, and 20 percentage points. Recorded token usage falls by 85.0%, 58.0%, and 76.7%, respectively. These are task-level means, not totals over the roster.

## RQ2 · Skill composition

Manuscript Table 2. All configurations use GPT-5.6-Sol on all 100 cases. **(ESB)** means a skill-based reconstruction of the named method.

| Configuration | SR (%) | Actions | Tokens (k) | Time (min) |
| :--- | ---: | ---: | ---: | ---: |
| Selected | 63.0 | 2.38 | 803.2 | 4.40 |
| BM25 | 36.0 | 8.09 | 2,731.6 | 9.76 |
| Embedding (BGE-small-en-v1.5) | 42.0 | 12.73 | 2,918.7 | 11.03 |
| HarnessVLA (ESB) | 77.0 | 14.70 | 2,113.8 | 14.12 |
| Zetta (ESB) | 69.0 | 14.51 | 2,399.7 | 16.66 |

BM25 and Embedding use 481 public task-and-capability queries, select the top skill for each query, and deduplicate within each task. They provide 4.23 and 3.82 skills per task on average. Relevant documents alone do not guarantee executable coverage or useful skill combinations.

## RQ3 · Complementarity

Manuscript Table 3. All configurations use GPT-6-Astra.

| Configuration | SR (%) | Actions | Tokens (k) | Time (min) |
| :--- | ---: | ---: | ---: | ---: |
| With skill | 61.0 | 2.24 | 563.5 | 3.40 |
| With Jev skill | 81.0 | 20.90 | 1,142.0 | 8.74 |
| Native / no skill | 92.0 | 16.59 | 2,300.6 | 20.52 |
| Agent-Directed Reuse | 85.0 | 12.87 | 1,976.3 | 18.09 |

Relative to native control, ADR reduces actions by 22.4%, tokens by 14.1%, and time by 11.9%, while reducing success by 7 percentage points. Jev reduces tokens by about 50.4% and time by about 57.4%, with an 11-point success tradeoff and more agent-level calls. The supplied source package does not include their standalone experiment integrations.

## Measurement conventions

- Success follows the official final simulator observation.
- Actions count accepted state-changing agent calls, deduplicated by step/request. Read-only queries, rejected calls, and low-level simulator steps are excluded. Incomplete terminal-event traces use the recorded runtime counter and are flagged in the audit.
- Tokens use recorded input plus output; cached input is counted once. Missing token usage is `null`; means use available records.
- Time spans agent launch to exit, including model calls, tool execution, and waiting.
- Aggregation uses equal task weights and includes failures. An explicitly selected subset is reported as a subset.

See [offline evaluation](../evaluation/README.md) for supported log formats and source-hash auditing.

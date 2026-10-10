# Release scope

This repository publishes the benchmark **code** from the supplementary source package. It does not publish the embodied skill library or collected resource corpus.

| Component | Availability |
| :--- | :--- |
| Agent harnesses, runtime, simulator adapters, and episode recording | Included under `src/`. |
| Clean100 task roster, seed 26, task hashes, and VLA seeds | Included in `configs/clean100.json`. |
| External simulator/data/model revisions | Included in `configs/dependencies.lock.json`. |
| Construction scripts, authoring prompt, and common schemas | Included; use resources supplied locally. |
| Offline trajectory evaluator | Included; requires archived episodes supplied locally. |
| Paper benchmark overview image | Included in `docs/images/`. |
| 250 action/perception/experience skill packages | Not included. |
| Skill browsing and machine-readable indexes | Not included. |
| Collected module cards, inventories, contracts, handoff, and source payloads | Not included. |
| Task-specific selected skill sets and retrieval artifacts | Not included. |
| Raw experimental rollouts and model usage logs | Not included. |
| Model weights and simulator assets | Obtain separately from their upstream projects. |
| Standalone ADR and Jev experiment integrations | Not present in the supplied public code package. |

The 1,052-module and 250-skill counts in the README describe the corpus used in the manuscript. They are not a count of downloadable packages in this repository. Result tables reproduce manuscript values; the public repository does not contain the raw archives needed to independently regenerate those values.

## Supported entry points

- `esbench doctor`: validates the roster and reports whether local skill assets exist. A successful check does not establish GPU, simulator, isolation, or model-service readiness.
- `esbench init --mode native`: generates a P3 configuration with an explicitly supplied audited IK identity and endpoint. P3 corresponds to the native control interface described in the paper.
- `esbench init --mode native --native-protocol p25`: generates an explicit OSC development configuration. P2.5 is not the paper's native baseline.
- `esbench init --mode skill-only`: generates a starter that requires local skill packages. It is not the task-conditioned Selected experiment configuration used in the paper.
- `direct-vla` and raw-Pi0.5 native variants remain in the research core. Their provider/adapter requirements must be supplied separately.

The paper's model labels and CLI versions describe its experimental setup. Users must supply accessible model IDs, API credentials through environment variables, and an exact CLI version in their own configuration. `configs/agents.lock.json` has no bundled agent profiles; starters write explicit profiles into each run configuration.

## Keeping local assets private

`skill-pool/`, `skills-index.json`, `SKILLS.md`, the collected `data/` tree, local runtime directories, weights, and rollout outputs are ignored by Git. Local skill assets may be restored for private experimentation without tracking them.

The public-release check rejects private asset paths even if they are force-added, along with credential files, model weights, and archives. Enable the repository's local commit hook:

```bash
git config core.hooksPath .githooks
```

Before committing public changes:

```bash
git add <public-files>
python scripts/check_public_release.py
```

The same check runs in CI. Its checks help enforce this release scope; inspect newly added files before publication.

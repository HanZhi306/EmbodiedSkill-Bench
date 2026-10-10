# Running EmbodiedSkillBench

This guide runs the public benchmark code with locally installed simulator and agent dependencies. The paper's skill packages and resource corpus are not included. The native-control path below has no skill-library dependency.

## 1. Install the source package

Run from the repository root. Python 3.12 is recommended (3.10–3.12 supported). Keep environments, downloaded assets, credentials, and generated episodes outside the checkout.

```bash
python3.12 -m venv ../esb-venv
source ../esb-venv/bin/activate
python -m pip install -e .
esbench doctor
```

Use an editable install: protocol files and configs are read from the source tree. Repeat the install after moving the checkout. For this release, `doctor` reports `skill_assets: "not_included"`, validates all 100 roster entries, and reports whether bubblewrap is on `PATH`. It does not start the simulator or call an API.

## 2. Install the simulator and task data

Real runs require Linux, NVIDIA GPU/EGL support, bubblewrap (`bwrap`) with usable user namespaces, and the pinned RPent/LIBERO-Pro dependencies. Install the system EGL/MuJoCo prerequisites and bubblewrap for your host. Strict isolation is required by the launcher.

```bash
mkdir -p ../esb-runtime
export ESB_RUNTIME_ROOT="$(cd ../esb-runtime && pwd)"
export ESB_RPENT_ROOT="$ESB_RUNTIME_ROOT/RPent"
bash scripts/bootstrap_rpent.sh "$ESB_RPENT_ROOT"
python -m pip install -e "$ESB_RPENT_ROOT[rlinf,libero-pro]" omegaconf
python -m pip install huggingface_hub

export LIBEROPRO_DATASET_PATH="$ESB_RUNTIME_ROOT/liberopro-pro"
export LIBERO_PRO_ASSET_PATH="$ESB_RUNTIME_ROOT/liberopro-assets"
hf download --repo-type dataset zhouxueyang/LIBERO-Pro \
  --revision c86fc3b8293185a6f373677018ff3e37f8391602 \
  --local-dir "$LIBEROPRO_DATASET_PATH"
hf download --repo-type dataset RLinf/LIBERO-PRO-assets \
  --revision e4b10c9adc5735c9c8776917527178e693ef10b3 \
  --local-dir "$LIBERO_PRO_ASSET_PATH"

export LIBERO_CONFIG_PATH="$ESB_RUNTIME_ROOT/liberopro-config"
export LIBERO_TYPE=pro
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
python scripts/configure_liberopro_data.py \
  --dataset-root "$LIBEROPRO_DATASET_PATH" --assets-root "$LIBERO_PRO_ASSET_PATH"
```

The data helper checks all 16 upstream perturbation suites and their initial states. `configs/clean100.json` selects the 100 cases; execution verifies each task's BDDL and initial-state hashes. Dependency revisions are recorded in `configs/dependencies.lock.json`. Choose CUDA-compatible binary dependencies for the host using the upstream installation instructions.

## 3. Configure the paper's P3 native controller

The paper's native interface uses Franka Panda **joint-position control** and a frozen **PyRoKi full-pose IK gate**. The provider source is [audited_pyroki_server.py](src/embodied_skill_bench/runtime/audited_pyroki_server.py); its immutable contract is in [native_typed_contract.py](src/embodied_skill_bench/runtime/native_typed_contract.py).

Prepare the IK service in a separate Python environment with the source package installed editably, `fastapi`, `uvicorn`, `scipy`, `robot-descriptions`, and `yourdfpy`. Install/build its upstream dependencies at the contract revisions:

| Dependency | Revision |
| :--- | :--- |
| [PyRoKi](https://github.com/chungmin99/pyroki) | `95afccc22658c461ab1042a048ae4e9c24bc2a47` |
| [jaxls](https://github.com/brentyi/jaxls) | `6fe7cf9d56223736b2c0272d080ec7346203c648` |
| [CaP-X](https://github.com/capgym/cap-x) | `53e9966d7a8e2fa7494676772bccc35280f5c0ed` |

The service requires a retained wheel from that PyRoKi revision, its installed matching Python sources, a saved pip PEP 610 origin record, and a CaP-X checkout at the listed commit. jaxls must also retain its immutable VCS `direct_url.json`. Supply those local paths explicitly; a generic joints-only IK endpoint does not satisfy the protocol.

With those dependencies prepared, start the service in its environment:

```bash
python -m embodied_skill_bench.runtime.audited_pyroki_server \
  --port 8766 \
  --cap-x-root "$ESB_RUNTIME_ROOT/cap-x" \
  --pyroki-wheel "$PYROKI_WHEEL_PATH" \
  --pyroki-origin-json "$PYROKI_ORIGIN_JSON_PATH"
```

`PYROKI_WHEEL_PATH` and `PYROKI_ORIGIN_JSON_PATH` name the retained wheel and origin JSON in your local IK setup. After startup, save the provider's actual identity from another terminal:

```bash
curl --fail http://127.0.0.1:8766/identity \
  --output "$ESB_RUNTIME_ROOT/pyroki-identity.json"
```

Return to the simulator/agent environment. Install a supported agent CLI (Codex, Claude Code, or Kimi Code), make it available on `PATH`, and provide its API credentials via environment variables. The harness creates a fresh agent home and needs the configured API-key environment variable.

For example, with Codex, set `OPENAI_API_KEY` in your shell and optionally `OPENAI_BASE_URL`. Set `ESB_AGENT_MODEL` to a model ID accessible to your account, then create a configuration:

```bash
esbench init --mode native --native-protocol p3 \
  --runner codex --model "$ESB_AGENT_MODEL" \
  --cli-version "$(codex --version)" \
  --ik-endpoint http://127.0.0.1:8766 \
  --ik-identity "$ESB_RUNTIME_ROOT/pyroki-identity.json" \
  --output "$ESB_RUNTIME_ROOT/native.json"

esbench run --config "$ESB_RUNTIME_ROOT/native.json" \
  --output ../esb-native-one-task
```

The starter uses the first Clean100 task and records the exact native-contract hash, provider identity, controller, model, and CLI version. It is a starting configuration; match the manuscript's experimental agent settings and budgets before claiming reproduction. Provider identity is checked on IK responses, and invalid solutions are rejected before execution.

Claude Code uses `--runner claude-code`, `ANTHROPIC_API_KEY`, and optionally `ANTHROPIC_BASE_URL`; Kimi Code uses `--runner kimi-code`, `KIMI_API_KEY`, and optionally `KIMI_BASE_URL`. Supply each CLI's own exact version string. Path placeholders use `${NAME}` in saved configs; credentials use `api_key_env` fields.

<details>
<summary><strong>P2.5 OSC development interface</strong></summary>

P2.5 uses OSC control without the P3 IK service. It is a separate development interface and is not the paper's native baseline. Select it explicitly:

```bash
esbench init --mode native --native-protocol p25 \
  --runner codex --model "$ESB_AGENT_MODEL" \
  --cli-version "$(codex --version)" \
  --output "$ESB_RUNTIME_ROOT/native-p25.json"
```

Linux simulator and isolation prerequisites still apply. The launcher does not switch a P3 run to P2.5 if the IK service is unavailable.

</details>

## 4. Run Clean100

First validate a small, explicit subset, then run the full roster:

```bash
esbench run --config "$ESB_RUNTIME_ROOT/native.json" \
  --roster configs/clean100.json --limit 2 --output ../esb-native-two-tasks

esbench run --config "$ESB_RUNTIME_ROOT/native.json" \
  --roster configs/clean100.json --output ../esb-native-clean100
```

Every output directory must be new. Runs are sequential and each task receives a fresh simulator episode and agent workspace. Eligible task and agent-protocol failures remain in the results and execution continues. An ineligible infrastructure failure stops the batch with a nonzero exit code. `run-index.json` records requested/completed counts and links the task summaries, including a stopped attempt. Inspect the one-task run before starting a roster, including its recorded CLI preflight and classification.

## 5. Skill use with local packages

This path requires locally supplied `skill-pool/` packages and a matching `skills-index.json`. They remain ignored by Git. `esbench doctor` checks the restored package IDs, file inventory, and hashes; a partial pool/index restoration is an error.

```bash
esbench init --mode skill-only \
  --runner codex --model "$ESB_AGENT_MODEL" \
  --cli-version "$(codex --version)" \
  --output "$ESB_RUNTIME_ROOT/skill-use.json"
```

Edit `skill_set`, backend services, and budgets to match your local packages and intended condition. The analytic starter is not the paper's task-conditioned Selected skill provision. Use [skill construction](docs/skill-construction.md) to assemble your own resources.

For Pi0.5-backed skills, the public provider and inference files can be installed separately:

```bash
python -m pip install -e "$ESB_RPENT_ROOT[rlinf,openpi]"
export PI05_CHECKPOINT_PATH="$ESB_RUNTIME_ROOT/models/pi05"
hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
  model.safetensors metadata.pt physical-intelligence/libero/norm_stats.json \
  --revision 6222623f635769bfc73c9472e29fab9b7fd8e027 \
  --local-dir "$PI05_CHECKPOINT_PATH"
```

Other VLA providers use `backend.vla_profiles` with their local checkpoint/runtime or a compatible endpoint. SAM3 needs its runtime/checkpoint or `backend.sam3_endpoint`; consult each local package's declared interface, coordinate convention, provider, and upstream access terms.

The core also retains `direct-vla` and raw-Pi0.5 native development conditions. They need their own model/provider configurations and, for raw-Pi0.5, the host-side adapter package. Standalone ADR and Jev experiment integrations are not included.

## Output and evaluation

Each `task-NNN/summary.json` contains task status, action count, agent outcome, and elapsed time. The agent condition directory contains its run manifest, trace, and `episode/` events and immutable observations. Skill-use conditions additionally record selected package hashes. Config, roster, runtime source hashes, dependency pins, and installed package versions are saved with the batch.

Use [evaluation/README.md](evaluation/README.md) to extract per-task success/actions/tokens/time and aggregate task-weighted means from archived episodes. Raw rollouts and usage logs are not bundled in this release.

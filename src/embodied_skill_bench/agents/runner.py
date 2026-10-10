from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..catalog import validate_json_schema_subset
from ..io import atomic_write_json, redact_url, sha256_file
from ..isolation import build_bwrap_command
from ..workspace import BENCHMARK_SYSTEM_FILENAME


@dataclass
class AgentOutcome:
    agent_id: str
    runner_kind: str
    return_code: int
    elapsed_seconds: float
    final_text: str
    structured: dict[str, Any] | None
    command: list[str]
    stdout_path: str
    stderr_path: str
    version: str
    requested_model: str
    observed_models: list[str]
    usage: dict[str, Any]
    endpoint: str | None
    isolation: dict[str, Any]
    benchmark_system_instructions: dict[str, Any]

    def public(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "runner_kind": self.runner_kind,
            "return_code": self.return_code,
            "elapsed_seconds": self.elapsed_seconds,
            "final_text": self.final_text,
            "structured": self.structured,
            "command": self.command,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "version": self.version,
            "requested_model": self.requested_model,
            "observed_models": self.observed_models,
            "usage": self.usage,
            "endpoint": self.endpoint,
            "isolation": self.isolation,
            "benchmark_system_instructions": self.benchmark_system_instructions,
        }


class ObservationProjectionError(RuntimeError):
    """The fail-closed RGB-only observation projection became unhealthy."""


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class _PublicEpisodeProjection:
    """Continuously mirror only agent-visible observation files.

    The simulator's immutable episode tree also contains depth maps, dense
    world-coordinate arrays and calibration.  Observation-only code agents
    need RGB files for native multimodal attachment, but must not receive a
    filesystem path to the geometric artifacts.  This projection is mounted
    over the private episode path inside bubblewrap and is updated as new
    observations are committed.
    """

    def __init__(
        self,
        source_root: Path,
        view_root: Path,
        *,
        settle_timeout_seconds: float = 5.0,
        geometry_api_available: bool = False,
    ):
        self.source_root = source_root.resolve()
        self.view_root = view_root.resolve()
        self.settle_timeout_seconds = float(settle_timeout_seconds)
        self.geometry_api_available = bool(geometry_api_available)
        if self.settle_timeout_seconds <= 0:
            raise ValueError("projection settle timeout must be positive")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._latest_step: int | None = None
        self._latest_unavailable_since: float | None = None
        self._sync_lock = threading.Lock()

    def prepare(self) -> None:
        self.view_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.view_root / "projection.json",
            {
                "schema_version": 1,
                "policy": (
                    "p2-rgb-files-with-host-geometry-query-api"
                    if self.geometry_api_available
                    else "observation-only-rgb-public-view"
                ),
                "rgb_exposed": True,
                "depth_exposed": False,
                "world_maps_exposed": False,
                "calibration_exposed": False,
                "restricted_geometry_query_api_available": self.geometry_api_available,
                "private_episode_root_exposed": False,
            },
        )
        self.sync()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._watch,
            name="esb-public-observation-projection",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise ObservationProjectionError(
                    "observation-only projection watcher did not stop"
                )
            self._thread = None
        self.assert_healthy()
        self.sync()
        self.assert_healthy()

    def assert_healthy(self) -> None:
        if self._error is not None:
            raise ObservationProjectionError(
                "observation-only public episode projection failed: "
                f"{type(self._error).__name__}: {self._error}"
            ) from self._error

    def _watch(self) -> None:
        try:
            while not self._stop.wait(0.02):
                self.sync()
        except BaseException as exc:  # fail closed in the owning runner
            self._error = exc
            self._stop.set()

    def sync(self) -> None:
        with self._sync_lock:
            self._sync_unlocked()

    def _sync_unlocked(self) -> None:
        source_steps = self.source_root / "steps"
        destination_steps = self.view_root / "steps"
        destination_steps.mkdir(parents=True, exist_ok=True)
        if not source_steps.is_dir():
            return
        for source_step in sorted(source_steps.iterdir()):
            if (
                not source_step.is_dir()
                or not source_step.name.isdigit()
                or len(source_step.name) != 4
            ):
                continue
            destination_step = destination_steps / source_step.name
            if destination_step.is_dir():
                continue
            observation_path = source_step / "observation.json"
            if not observation_path.is_file():
                continue
            self._project_step(source_step, destination_step)
        latest = self.source_root / "latest.json"
        if not latest.is_file():
            self._record_latest_unavailable(
                FileNotFoundError(f"temporarily unavailable: {latest}")
            )
            return
        try:
            pointer = json.loads(latest.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            # On the shared experiment filesystem an atomic replacement can
            # briefly produce an is_file()->read_text visibility race.  Treat
            # it exactly like a not-yet-settled RGB artifact: bounded retry,
            # then fail closed.
            self._record_latest_unavailable(exc)
            return
        self._latest_unavailable_since = None
        step_index = int(pointer["step_index"])
        projected_observation = (
            destination_steps / f"{step_index:04d}" / "observation.json"
        )
        if step_index != self._latest_step and projected_observation.is_file():
            atomic_write_json(self.view_root / "latest.json", pointer)
            self._latest_step = step_index

    def _record_latest_unavailable(self, exc: BaseException) -> None:
        now = time.monotonic()
        if self._latest_unavailable_since is None:
            self._latest_unavailable_since = now
            return
        if now - self._latest_unavailable_since >= self.settle_timeout_seconds:
            raise ObservationProjectionError(
                "committed episode latest pointer did not settle: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _project_step(self, source_step: Path, destination_step: Path) -> None:
        deadline = time.monotonic() + self.settle_timeout_seconds
        last_missing: Path | None = None
        while True:
            temporary: Path | None = None
            try:
                observation = json.loads(
                    (source_step / "observation.json").read_text(encoding="utf-8")
                )
                cameras = observation.get("artifacts", {}).get("cameras", {})
                public_cameras: dict[str, Any] = {}
                sources: list[tuple[str, str, dict[str, Any], Path, Path]] = []
                for camera, camera_data in (
                    cameras.items() if isinstance(cameras, dict) else ()
                ):
                    if not isinstance(camera_data, dict):
                        continue
                    for key in ("rgb", "high_rgb"):
                        artifact = camera_data.get(key)
                        if not isinstance(artifact, dict) or not isinstance(
                            artifact.get("path"), str
                        ):
                            continue
                        source = Path(artifact["path"]).resolve()
                        if not source.is_relative_to(self.source_root):
                            raise ObservationProjectionError(
                                f"public RGB artifact escapes episode root: {source}"
                            )
                        try:
                            relative = source.relative_to(source_step)
                        except ValueError as exc:
                            raise ObservationProjectionError(
                                f"public RGB artifact escapes committed step: {source}"
                            ) from exc
                        if not source.is_file():
                            last_missing = source
                            raise FileNotFoundError(str(source))
                        sources.append(
                            (str(camera), key, dict(artifact), source, relative)
                        )

                temporary = Path(
                    tempfile.mkdtemp(
                        prefix=f".{destination_step.name}.public-",
                        dir=destination_step.parent,
                    )
                )
                for camera, key, artifact, source, relative in sources:
                    target = temporary / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copy2(source, target)
                    public_cameras.setdefault(camera, {})[key] = artifact
                observation["artifacts"] = {"cameras": public_cameras}
                contract = observation.setdefault("agent_contract", {})
                if isinstance(contract, dict):
                    contract["privileged_geometry_artifacts_exposed"] = False
                atomic_write_json(temporary / "observation.json", observation)
                markdown = source_step / "observation.md"
                if markdown.is_file():
                    replacement = (
                        "Raw depth/world/calibration files are unavailable; use the staged P2 restricted geometry functions for agent-chosen pixels."
                        if self.geometry_api_available
                        else "Depth, world maps, calibration and pixel-to-world lookup are unavailable in this condition."
                    )
                    text = markdown.read_text(encoding="utf-8").replace(
                        "For a pixel-to-world lookup, use `esbench pixel-to-world` with the same step, camera, and resolution.",
                        replacement,
                    )
                    (temporary / "observation.md").write_text(
                        text, encoding="utf-8"
                    )
                if destination_step.is_dir():
                    shutil.rmtree(temporary)
                else:
                    os.replace(temporary, destination_step)
                return
            except FileNotFoundError as exc:
                if temporary is not None:
                    shutil.rmtree(temporary, ignore_errors=True)
                if time.monotonic() >= deadline:
                    missing = last_missing or Path(str(exc))
                    raise ObservationProjectionError(
                        f"committed RGB artifact did not settle: {missing}"
                    ) from exc
                time.sleep(0.02)
            except BaseException:
                if temporary is not None:
                    shutil.rmtree(temporary, ignore_errors=True)
                raise


class AgentRunner:
    def __init__(
        self,
        agent_id: str,
        spec: dict[str, Any],
        *,
        strict_version: bool = True,
        isolation_mode: str = "strict",
    ):
        self.agent_id = agent_id
        self.spec = spec
        self.runner_kind = str(spec.get("runner", agent_id))
        if self.runner_kind not in {
            "codex",
            "claude-code",
            "deepseek-harness",
            "kimi-code",
            "openhands",
        }:
            raise ValueError(
                f"unsupported runner kind {self.runner_kind!r} for agent profile {agent_id!r}"
            )
        self.strict_version = strict_version
        if isolation_mode not in {"strict", "disabled"}:
            raise ValueError("isolation_mode must be 'strict' or 'disabled'")
        self.isolation_mode = isolation_mode

    def version(self) -> str:
        executable = self._executable()
        version_command = [executable, "--version"]
        if self.runner_kind == "deepseek-harness":
            version_command = [
                executable,
                str(_deepseek_harness_bridge_source()),
                "--version",
            ]
        elif self.runner_kind == "openhands":
            version_command = [
                executable,
                str(_openhands_bridge_source()),
                "--version",
            ]
        completed = subprocess.run(
            version_command,
            text=True,
            capture_output=True,
            check=False,
        )
        actual = (completed.stdout or completed.stderr).strip()
        if completed.returncode != 0:
            detail = actual or f"exit code {completed.returncode} with no output"
            raise RuntimeError(
                f"failed to run {self.agent_id} executable {executable!r}: {detail}"
            )
        expected = str(self.spec["cli_version"])
        if self.strict_version and actual != expected:
            raise RuntimeError(
                f"{self.agent_id} CLI drift: expected {expected!r}, found {actual!r}"
            )
        return actual

    def _executable(self) -> str:
        override_name = self.spec.get("executable_env")
        if override_name and os.environ.get(str(override_name)):
            return os.environ[str(override_name)]
        if self.runner_kind == "openhands" and override_name:
            raise RuntimeError(
                "OpenHands requires its snapshotted executable override "
                f"{override_name!r}; point it at the dedicated Python 3.12 "
                "virtualenv interpreter"
            )
        if self.runner_kind in {"deepseek-harness", "openhands"} and str(
            self.spec.get("executable", "")
        ) in {"python", "python3"}:
            # Use the interpreter that loaded the benchmark. Merely resolving
            # `python3` from the parent shell PATH can select the system Python
            # even though the child PATH correctly starts with this venv.
            return sys.executable
        return str(self.spec["executable"])

    def run_episode(
        self,
        *,
        workspace: Path,
        prompt: str,
        output_dir: Path,
        timeout_seconds: int,
    ) -> AgentOutcome:
        return self._run_external(
            workspace=workspace,
            prompt=prompt,
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
            schema=None,
        )

    def _run_external(
        self,
        *,
        workspace: Path,
        prompt: str,
        output_dir: Path,
        timeout_seconds: int,
        schema: dict[str, Any] | None,
    ) -> AgentOutcome:
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "stdout.jsonl"
        stderr_path = output_dir / "stderr.log"
        last_message_path = output_dir / "last_message.txt"
        schema_path = output_dir / "output_schema.json"
        if schema:
            atomic_write_json(schema_path, schema)
        sandbox_home = output_dir / "sandbox-home"
        sandbox_home.mkdir(parents=True, exist_ok=True)
        benchmark_system, benchmark_system_path = _load_benchmark_system(workspace)
        model = str(self.spec["model"])
        effort = str(self.spec.get("reasoning_effort", "medium"))
        deepseek_profile: dict[str, Any] | None = None
        openhands_profile: dict[str, Any] | None = None
        if self.runner_kind == "codex":
            command = [
                self._executable(),
                "exec",
                "--json",
                "--ephemeral",
                "--ignore-user-config",
                "--skip-git-repo-check",
                "--sandbox",
                "danger-full-access",
                "--cd",
                str(workspace),
                "--model",
                model,
                "-c",
                f'model_reasoning_effort="{effort}"',
                "-c",
                f"developer_instructions={json.dumps(benchmark_system)}",
                # Do not also auto-load an AGENTS.md when strict isolation is
                # deliberately disabled for debugging. The benchmark system
                # text above remains the one controlled instruction source.
                "-c",
                "project_doc_max_bytes=0",
                "--output-last-message",
                str(last_message_path),
            ]
            base_url = os.environ.get(str(self.spec.get("base_url_env", "")))
            if base_url:
                request_max_retries = _nonnegative_int(
                    self.spec.get("request_max_retries", 0),
                    "request_max_retries",
                )
                stream_max_retries = _nonnegative_int(
                    self.spec.get("stream_max_retries", 0),
                    "stream_max_retries",
                )
                command.extend(
                    [
                        "-c",
                        'model_provider="esbench_proxy"',
                        "-c",
                        'model_providers.esbench_proxy.name="EmbodiedSkillBench proxy"',
                        "-c",
                        f"model_providers.esbench_proxy.base_url={json.dumps(base_url)}",
                        "-c",
                        'model_providers.esbench_proxy.env_key="CODEX_API_KEY"',
                        "-c",
                        'model_providers.esbench_proxy.wire_api="responses"',
                        "-c",
                        "model_providers.esbench_proxy.supports_websockets=false",
                        "-c",
                        f"model_providers.esbench_proxy.request_max_retries={request_max_retries}",
                        "-c",
                        f"model_providers.esbench_proxy.stream_max_retries={stream_max_retries}",
                    ]
                )
            if schema:
                command.extend(["--output-schema", str(schema_path)])
            command.append(prompt)
        elif self.runner_kind == "claude-code":
            command = [
                self._executable(),
                "--bare",
                "--print",
                "--verbose",
                "--output-format",
                "stream-json",
                "--model",
                model,
                "--effort",
                effort,
                "--permission-mode",
                "bypassPermissions",
                "--append-system-prompt",
                benchmark_system,
                "--tools",
                "Bash,Read,Glob,Grep",
            ]
            if schema:
                command.extend(["--json-schema", json.dumps(schema)])
            # Claude's --tools option is variadic. Without an option terminator,
            # the positional prompt is consumed as another tool name and
            # `--print` exits with "Input must be provided".
            command.extend(["--", prompt])
        elif self.runner_kind == "deepseek-harness":
            deepseek_profile = _stage_deepseek_harness_profile(
                workspace=workspace,
                cordis_config=str(self.spec["cordis_config"]),
            )
            provider = str(self.spec.get("provider", "deepseek-official"))
            max_tokens = _positive_int(
                self.spec.get("max_tokens", 49_152), "max_tokens"
            )
            session_id = "esbench-" + re.sub(
                r"[^A-Za-z0-9._-]+", "-", self.agent_id
            ).strip("-")
            command = [
                self._executable(),
                str(deepseek_profile["bridge_path"]),
                "--workspace",
                str(workspace),
                "--session-root",
                str(output_dir / "sessions"),
                "--session-id",
                session_id,
                "--provider",
                provider,
                "--model",
                model,
                "--max-tokens",
                str(max_tokens),
                "--system-file",
                str(benchmark_system_path),
                "--cordis",
                str(deepseek_profile["cordis_path"]),
                "--final-output",
                str(last_message_path),
            ]
            if schema:
                command.extend(["--output-schema", str(schema_path)])
            command.extend(["--", prompt])
        elif self.runner_kind == "kimi-code":
            # Classic S1 skill workspace: the agent drives workspace-local
            # skill scripts through ordinary bash tool calls, so no tool
            # restriction is declared and no MCP server is mounted.  This
            # mirrors the Codex with-skill path; the skill contract is the
            # staged workspace, not a harness-specific tool surface.
            kimi_profile = _stage_kimi_code_classic_profile(
                workspace=workspace,
                home=sandbox_home,
                benchmark_system=benchmark_system,
            )
            # ``--prompt`` is already the non-interactive one-shot mode and
            # cannot be combined with ``--auto`` (Kimi 0.40.0 rejects the
            # pair).  Point ``--skills-dir`` at the staged workspace copy so
            # discovery matches the frozen v32 selection exactly instead of
            # relying on auto-discovery.
            command = [
                self._executable(),
                "--agent-file",
                str(kimi_profile["agent_file"]),
                "--skills-dir",
                str(workspace / ".agents" / "skills"),
                "--output-format",
                "stream-json",
                "--prompt",
                prompt,
            ]
        elif self.runner_kind == "openhands":
            openhands_profile = _stage_openhands_profile(
                workspace=workspace,
                harness_config=str(self.spec["harness_config"]),
                expected_sha256=str(self.spec["harness_config_sha256"]),
            )
            max_output_tokens = _positive_int(
                self.spec.get("max_output_tokens", 49_152),
                "max_output_tokens",
            )
            request_timeout = _positive_int(
                self.spec.get("request_timeout_seconds", 900),
                "request_timeout_seconds",
            )
            observed_allowlist = self.spec.get("observed_model_allowlist")
            if not isinstance(observed_allowlist, list) or not observed_allowlist:
                raise ValueError(
                    "openhands observed_model_allowlist must be a non-empty list"
                )
            command = [
                self._executable(),
                str(openhands_profile["bridge_path"]),
                "--workspace",
                str(workspace),
                "--system-file",
                str(benchmark_system_path),
                "--harness-config",
                str(openhands_profile["config_path"]),
                "--provider",
                str(self.spec["provider"]),
                "--model",
                model,
                "--litellm-model",
                str(self.spec["litellm_model"]),
                "--api-mode",
                "chat"
                if str(self.spec["wire_api"]) == "chat_completions"
                else "responses",
                "--vision-mode",
                str(self.spec["vision_mode"]),
                "--reasoning-effort",
                effort,
                "--max-output-tokens",
                str(max_output_tokens),
                "--request-timeout-seconds",
                str(request_timeout),
                "--capability-overrides-json",
                json.dumps(
                    self.spec.get("capability_overrides", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "--litellm-extra-body-json",
                json.dumps(
                    self.spec.get("litellm_extra_body", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "--final-output",
                str(last_message_path),
            ]
            canonical = self.spec.get("model_canonical_name")
            if canonical:
                command.extend(["--model-canonical-name", str(canonical)])
            for observed_model in observed_allowlist:
                command.extend(
                    ["--allowed-observed-model", str(observed_model)]
                )
            for field, option in (
                ("input_cost_per_token", "--input-cost-per-token"),
                ("output_cost_per_token", "--output-cost-per-token"),
            ):
                value = self.spec.get(field)
                if value is not None:
                    command.extend([option, str(float(value))])
            if schema:
                command.extend(["--output-schema", str(schema_path)])
            command.extend(["--", prompt])
        else:
            raise ValueError(
                f"unsupported runner kind {self.runner_kind!r} for agent profile {self.agent_id!r}"
            )

        version = self.version()
        isolation = {"mode": "disabled", "enforced": False}
        inner_command = command
        public_episode_projection: _PublicEpisodeProjection | None = None
        if self.isolation_mode == "strict":
            read_only_paths: list[Path] = []
            read_only_path_mappings: list[tuple[Path, Path]] = []
            # OpenHands is intentionally allowed to run from a dedicated
            # Python environment because its FastMCP/telemetry dependency set
            # conflicts with the pinned RPent/OpenPi environment.  The
            # workspace launchers still use the benchmark interpreter (their
            # shebang is /usr/bin/env python3), so mount that virtualenv
            # read-only when it differs from the agent executable's venv.
            benchmark_python_root = Path(sys.prefix).resolve()
            executable_path = Path(self._executable()).resolve()
            if (
                self.runner_kind == "openhands"
                and (benchmark_python_root / "pyvenv.cfg").is_file()
                and not executable_path.is_relative_to(benchmark_python_root)
            ):
                read_only_paths.append(benchmark_python_root)
            context_path = workspace / "run_context.json"
            if context_path.is_file():
                context = json.loads(context_path.read_text(encoding="utf-8"))
                initial = context.get("initial_observation_json")
                if isinstance(initial, str):
                    # .../episode/steps/0000/observation.json -> episode root.
                    episode_root = Path(initial).resolve().parents[2]
                    if (
                        context.get("runtime_contract")
                        in {
                            "primitive-code-v1",
                            "native-typed-no-skill-v1",
                        }
                        and context.get("perception_mode")
                        in {
                            "observation-only",
                            "observation-only-p1",
                            "calibrated-geometry-p2",
                        }
                    ):
                        public_episode_projection = _PublicEpisodeProjection(
                            episode_root,
                            workspace.parent
                            / "runtime"
                            / "agent-public-episode-view",
                            geometry_api_available=(
                                context.get("perception_mode")
                                == "calibrated-geometry-p2"
                            ),
                        )
                        public_episode_projection.prepare()
                        read_only_path_mappings.append(
                            (
                                public_episode_projection.view_root,
                                episode_root,
                            )
                        )
                    else:
                        read_only_paths.append(episode_root)
            command, isolation = build_bwrap_command(
                inner_command,
                workspace=workspace,
                output_dir=output_dir,
                read_only_paths=read_only_paths,
                read_only_path_mappings=read_only_path_mappings,
            )
            if public_episode_projection is not None:
                isolation["public_observation_projection"] = {
                    "policy": (
                        "p2-rgb-files-with-host-geometry-query-api"
                        if context.get("perception_mode")
                        == "calibrated-geometry-p2"
                        else "observation-only-rgb-public-view"
                    ),
                    "mounted_over_private_episode_destination": True,
                    "rgb_exposed": True,
                    "depth_exposed": False,
                    "world_maps_exposed": False,
                    "calibration_exposed": False,
                    "restricted_geometry_query_api_available": (
                        context.get("perception_mode")
                        == "calibrated-geometry-p2"
                    ),
                    "private_episode_root_exposed": False,
                }
        if deepseek_profile is not None:
            isolation["deepseek_harness_profile"] = {
                "cordis_sha256": deepseek_profile["cordis_sha256"],
                "bridge_sha256": deepseek_profile["bridge_sha256"],
                "provider_retry_owner": "benchmark_only",
                "provider_internal_max_retries": 0,
                "node_environment_proxy_enabled": True,
                "wire_adapter": "dsh-llm-pi-ai:openai-completions",
            }
        if openhands_profile is not None:
            isolation["openhands_profile"] = {
                "harness_config_sha256": openhands_profile["config_sha256"],
                "bridge_sha256": openhands_profile["bridge_sha256"],
                "agents_package_sha256": openhands_profile[
                    "agents_package_sha256"
                ],
                "tool_sha256": openhands_profile["tool_sha256"],
                "provider_retry_owner": "benchmark_only",
                "provider_internal_max_retries": 0,
                "fallback": None,
                "wire_adapter": "LiteLLM openai-compatible",
                "vision_mode": str(self.spec["vision_mode"]),
                "observation_image_policy": (
                    "+".join(openhands_profile["observation_cameras"])
                    + " "
                    + str(openhands_profile["observation_image_policy"])
                ),
                "vision_history_policy": str(
                    openhands_profile["vision_history_policy"]
                ),
                "conversation_history_policy": str(
                    openhands_profile["conversation_history_policy"]
                ),
                "provider_transport_max_attempts": int(
                    openhands_profile["provider_transport_max_attempts"]
                ),
                "provider_transport_backoff_seconds": float(
                    openhands_profile["provider_transport_backoff_seconds"]
                ),
                "tool_concurrency_limit": 1,
                "ambient_profile_memory_skills_plugins": False,
            }
        started = time.monotonic()
        process: subprocess.Popen[str] | None = None
        try:
            if public_episode_projection is not None:
                public_episode_projection.start()
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=workspace,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    env=self._child_env(workspace, home=sandbox_home),
                )
                try:
                    deadline = time.monotonic() + timeout_seconds
                    while True:
                        observed = process.poll()
                        if observed is not None:
                            return_code = observed
                            break
                        if public_episode_projection is not None:
                            public_episode_projection.assert_healthy()
                        if time.monotonic() >= deadline:
                            _terminate_process(process)
                            return_code = 124
                            break
                        time.sleep(0.1)
                except BaseException:
                    _terminate_process(process)
                    raise
        finally:
            if public_episode_projection is not None:
                public_episode_projection.stop()
        elapsed = time.monotonic() - started
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
        final_text = (
            last_message_path.read_text(encoding="utf-8", errors="replace")
            if last_message_path.exists()
            else _last_agent_text(stdout_text)
        )
        structured = (
            _find_structured(final_text, stdout_text, schema=schema)
            if schema
            else None
        )
        observed_models = sorted(_collect_models(stdout_text))
        usage = _collect_usage(stdout_text, runner_kind=self.runner_kind)
        endpoint = redact_url(os.environ.get(str(self.spec.get("base_url_env", ""))))
        return AgentOutcome(
            agent_id=self.agent_id,
            runner_kind=self.runner_kind,
            return_code=return_code,
            elapsed_seconds=elapsed,
            final_text=final_text,
            structured=structured,
            command=_redact_command(command),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            version=version,
            requested_model=model,
            observed_models=observed_models,
            usage=usage,
            endpoint=endpoint,
            isolation=isolation,
            benchmark_system_instructions=_benchmark_system_record(
                workspace,
                "codex_developer_instructions"
                if self.runner_kind == "codex"
                else (
                    "claude_append_system_prompt"
                    if self.runner_kind == "claude-code"
                    else (
                        "deepseek_harness_persona"
                        if self.runner_kind == "deepseek-harness"
                        else (
                            "kimi_agent_file_system_prompt_verbatim"
                            if self.runner_kind == "kimi-code"
                            else "openhands_agent_system_prompt_verbatim"
                        )
                    )
                ),
                path=benchmark_system_path,
            ),
        )

    def _child_env(self, workspace: Path, *, home: Path | None = None) -> dict[str, str]:
        # Do not forward the parent environment wholesale: it may contain
        # unrelated credentials or host paths. Keep only provider/network
        # values required by this one agent process.
        env: dict[str, str] = {}
        # `esbench observe` and `esbench pixel-to-world` are part of the
        # documented harness surface. Make them available even when the parent
        # invoked an absolute venv executable without activating that venv.
        # Keep the venv entry-point path intact. Resolving the symlink would
        # collapse it to the system interpreter directory and lose the venv.
        python_bin = str(Path(sys.executable).parent)
        env["PATH"] = f"{python_bin}{os.pathsep}/usr/local/bin:/usr/bin:/bin"
        env["PYTHONPATH"] = str(workspace / ".agent-runtime")
        env["HOME"] = str((home or (workspace / ".agent-home")).resolve())
        env["LANG"] = os.environ.get("LANG", "C.UTF-8")
        env["LC_ALL"] = os.environ.get("LC_ALL", "C.UTF-8")
        env.update(_proxy_environment(os.environ))
        env["ESB_RUN_CONTEXT"] = str(workspace / "run_context.json")
        env["DISABLE_AUTOUPDATER"] = "1"
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        configured_key = os.environ.get(str(self.spec.get("api_key_env", "")))
        base_url_name = str(self.spec.get("base_url_env", ""))
        if base_url_name and os.environ.get(base_url_name):
            env[base_url_name] = os.environ[base_url_name]
        if self.runner_kind == "claude-code":
            # Pinned Claude Code releases can enable provider-preview beta
            # headers after the benchmark's CLI lock.  Bedrock-compatible
            # gateways reject unknown preview flags before inference.  Keep
            # the model/effort contract intact while disabling only those
            # remotely gated experiments.
            env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = "1"
        if self.runner_kind == "codex" and configured_key:
            # CODEX_API_KEY is the documented one-run credential for
            # non-interactive `codex exec`. Keep OPENAI_API_KEY as the
            # benchmark-facing provider-neutral input for existing configs.
            env["CODEX_API_KEY"] = configured_key
        elif self.runner_kind == "claude-code" and configured_key:
            # Claude Code --bare reads ANTHROPIC_API_KEY and intentionally
            # ignores OAuth/keychain credentials.
            env["ANTHROPIC_API_KEY"] = configured_key
        elif self.runner_kind == "deepseek-harness":
            if configured_key:
                env["DEEPSEEK_API_KEY"] = configured_key
            configured_base_url = os.environ.get(base_url_name)
            if configured_base_url:
                env["DEEPSEEK_BASE_URL"] = configured_base_url
            # The official runtime uses Node fetch.  Node does not consult the
            # inherited HTTP(S)_PROXY variables unless this flag is enabled.
            env["NODE_USE_ENV_PROXY"] = "1"
        elif self.runner_kind == "kimi-code":
            if configured_key:
                env["KIMI_MODEL_API_KEY"] = configured_key
            configured_base_url = os.environ.get(base_url_name)
            if configured_base_url:
                env["KIMI_MODEL_BASE_URL"] = configured_base_url
            env["KIMI_MODEL_NAME"] = str(self.spec["model"])
            env["KIMI_MODEL_DISPLAY_NAME"] = str(self.spec["model"])
            env["KIMI_MODEL_PROVIDER_TYPE"] = "openai"
            env["KIMI_MODEL_MAX_CONTEXT_SIZE"] = str(
                _positive_int(self.spec["max_context_tokens"], "max_context_tokens")
            )
            capabilities = self.spec["capabilities"]
            if not isinstance(capabilities, list):
                raise ValueError("Kimi Code capabilities must be a list")
            env["KIMI_MODEL_CAPABILITIES"] = ",".join(str(i) for i in capabilities)
            env["KIMI_CODE_HOME"] = str(
                ((home or (workspace / ".agent-home")) / ".kimi-code").resolve()
            )
            env["KIMI_CODE_NO_AUTO_UPDATE"] = "1"
            env["KIMI_CODE_DISABLE_TELEMETRY"] = "1"
        elif self.runner_kind == "openhands":
            if configured_key:
                env["OPENHANDS_API_KEY"] = configured_key
            configured_base_url = os.environ.get(base_url_name)
            if configured_base_url:
                env["OPENHANDS_BASE_URL"] = configured_base_url
        return env


def _stage_kimi_code_classic_profile(
    *,
    workspace: Path,
    home: Path,
    benchmark_system: str,
) -> dict[str, Any]:
    """Stage a bash-capable Kimi Code profile for classic S1 skill workspaces.

    The with-skill protocol drives the workspace-local skill scripts through
    ordinary tool calls, so no tool restriction is declared and no MCP server
    is mounted.  Ported from the p4-grounded runner so the agent layer can be
    swapped without changing the v32 skill contract.
    """

    runtime_root = workspace / ".agent-runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    agent_file = runtime_root / "kimi-esbench-classic.md"
    agent_file.write_text(
        "---\n"
        "name: esbench-classic\n"
        "description: Execute one isolated EmbodiedSkillBench task.\n"
        "subagents: []\n"
        "---\n\n"
        + benchmark_system,
        encoding="utf-8",
    )
    kimi_home = home / ".kimi-code"
    kimi_home.mkdir(parents=True, exist_ok=True)
    atomic_write_json(kimi_home / "mcp.json", {"mcpServers": {}})
    return {
        "agent_file": agent_file,
        "agent_file_sha256": sha256_file(agent_file),
    }


def _nonnegative_int(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return result


def _positive_int(value: Any, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _deepseek_harness_bridge_source() -> Path:
    return Path(__file__).with_name("deepseek_harness_cli.py").resolve()


def _openhands_bridge_source() -> Path:
    return Path(__file__).with_name("openhands_cli.py").resolve()


def _openhands_tool_source() -> Path:
    return Path(__file__).with_name("openhands_tool.py").resolve()


def _openhands_native_tool_source() -> Path:
    return Path(__file__).with_name("openhands_native_tool.py").resolve()


def _native_typed_contract_source() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "runtime"
        / "native_typed_contract.py"
    )


def _stage_deepseek_harness_profile(
    *, workspace: Path, cordis_config: str
) -> dict[str, Any]:
    relative = Path(cordis_config)
    if relative.is_absolute():
        raise ValueError("deepseek-harness cordis_config must be repository-relative")
    root = _project_root()
    source = (root / relative).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError(
            f"deepseek-harness cordis_config is not a repository file: {cordis_config!r}"
        )
    runtime_root = workspace / ".agent-runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    cordis_path = runtime_root / "deepseek-harness.cordis.yml"
    bridge_path = runtime_root / "deepseek_harness_cli.py"
    bridge_source = _deepseek_harness_bridge_source()
    shutil.copy2(source, cordis_path)
    shutil.copy2(bridge_source, bridge_path)
    return {
        "cordis_path": cordis_path,
        "cordis_sha256": sha256_file(cordis_path),
        "bridge_path": bridge_path,
        "bridge_sha256": sha256_file(bridge_path),
    }


def _stage_openhands_profile(
    *, workspace: Path, harness_config: str, expected_sha256: str
) -> dict[str, Any]:
    relative = Path(harness_config)
    if relative.is_absolute():
        raise ValueError("openhands harness_config must be repository-relative")
    root = _project_root()
    source = (root / relative).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError(
            f"openhands harness_config is not a repository file: {harness_config!r}"
        )
    actual_sha256 = sha256_file(source)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "openhands harness config drift: "
            f"expected {expected_sha256!r}, found {actual_sha256!r}"
        )
    parsed_config = json.loads(source.read_text(encoding="utf-8"))
    observation_images = parsed_config.get("observation_images")
    if not isinstance(observation_images, dict):
        raise ValueError("OpenHands harness config is missing observation_images")
    observation_image_policy = observation_images.get("resolution_policy")
    if observation_image_policy not in {"high_rgb_then_rgb", "rgb_only"}:
        raise ValueError("unsupported OpenHands observation image resolution policy")
    observation_cameras = observation_images.get("cameras")
    if observation_cameras not in (["agentview", "wrist"], ["agentview"]):
        raise ValueError("unsupported OpenHands observation camera subset")
    if observation_images.get("max_images") != len(observation_cameras):
        raise ValueError("OpenHands max_images must match the frozen camera subset")
    pixel_default_resolution = (
        "low" if observation_image_policy == "rgb_only" else "high"
    )
    # The staged shell utility must interpret an omitted --resolution in the
    # coordinate frame of the RGB actually transported by this harness.  This
    # is public harness metadata, not scene information or a semantic prior.
    run_context_path = workspace / "run_context.json"
    if run_context_path.is_file():
        run_context = json.loads(run_context_path.read_text(encoding="utf-8"))
        if not isinstance(run_context, dict):
            raise ValueError("staged run_context.json must be an object")
        run_context["pixel_to_world_default_resolution"] = pixel_default_resolution
        atomic_write_json(run_context_path, run_context)
    vision_history_policy = observation_images.get(
        "history_policy", "full-history"
    )
    if vision_history_policy not in {
        "full-history",
        "latest-observation-only",
    }:
        raise ValueError("unsupported OpenHands vision history policy")
    conversation_history = parsed_config.get("conversation_history", {})
    if not isinstance(conversation_history, dict):
        raise ValueError("OpenHands conversation_history must be an object")
    conversation_history_policy = conversation_history.get(
        "policy", "full-history"
    )
    if conversation_history_policy not in {
        "full-history",
        "latest-tool-result-only",
    }:
        raise ValueError("unsupported OpenHands conversation history policy")
    provider = parsed_config.get("provider", {})
    if not isinstance(provider, dict):
        raise ValueError("OpenHands provider config must be an object")
    transport_retry = provider.get(
        "transport_retry", {"max_attempts": 1, "backoff_seconds": 0}
    )
    if not isinstance(transport_retry, dict):
        raise ValueError("OpenHands transport_retry must be an object")
    provider_transport_max_attempts = transport_retry.get("max_attempts", 1)
    provider_transport_backoff_seconds = transport_retry.get(
        "backoff_seconds", 0
    )
    if (
        isinstance(provider_transport_max_attempts, bool)
        or not isinstance(provider_transport_max_attempts, int)
        or provider_transport_max_attempts < 1
        or isinstance(provider_transport_backoff_seconds, bool)
        or not isinstance(provider_transport_backoff_seconds, (int, float))
        or provider_transport_backoff_seconds < 0
    ):
        raise ValueError("invalid OpenHands audited transport-retry limits")
    runtime_root = workspace / ".agent-runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    config_path = runtime_root / "openhands-minimal.json"
    bridge_path = runtime_root / "openhands_cli.py"
    agents_package_path = (
        runtime_root / "embodied_skill_bench" / "agents" / "__init__.py"
    )
    tool_path = agents_package_path.with_name("openhands_tool.py")
    tool_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, config_path)
    shutil.copy2(_openhands_bridge_source(), bridge_path)
    finish_guard = parsed_config.get("finish_guard")
    if isinstance(finish_guard, dict) and finish_guard.get("enabled") is True:
        command = finish_guard.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError("OpenHands finish_guard command is missing")
        guard_source = (root / command).resolve()
        if not guard_source.is_file() or not guard_source.is_relative_to(root):
            raise ValueError("OpenHands finish_guard script is not a repository file")
        guard_target = (runtime_root / command).resolve()
        if not guard_target.is_relative_to(runtime_root):
            raise ValueError("OpenHands finish_guard command escapes staged runtime")
        guard_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(guard_source, guard_target)
        if sha256_file(guard_target) != str(finish_guard.get("script_sha256")):
            raise RuntimeError("staged OpenHands finish_guard script hash mismatch")
    agents_package_path.write_text(
        '"""Isolated OpenHands tool package; no benchmark runner imports."""\n',
        encoding="utf-8",
    )
    shutil.copy2(_openhands_tool_source(), tool_path)
    result = {
        "config_path": config_path,
        "config_sha256": sha256_file(config_path),
        "bridge_path": bridge_path,
        "bridge_sha256": sha256_file(bridge_path),
        "agents_package_sha256": sha256_file(agents_package_path),
        "tool_path": tool_path,
        "tool_sha256": sha256_file(tool_path),
        "observation_image_policy": observation_image_policy,
        "pixel_to_world_default_resolution": pixel_default_resolution,
        "observation_cameras": list(observation_cameras),
        "vision_history_policy": vision_history_policy,
        "conversation_history_policy": conversation_history_policy,
        "provider_transport_max_attempts": provider_transport_max_attempts,
        "provider_transport_backoff_seconds": (
            provider_transport_backoff_seconds
        ),
    }
    if isinstance(finish_guard, dict) and finish_guard.get("enabled") is True:
        result["finish_guard_script_path"] = runtime_root / str(
            finish_guard["command"]
        )
        result["finish_guard_script_sha256"] = sha256_file(
            result["finish_guard_script_path"]
        )
    if parsed_config.get("schema_version") in {2, 3}:
        native_tool_path = agents_package_path.with_name("openhands_native_tool.py")
        runtime_package_path = (
            runtime_root / "embodied_skill_bench" / "runtime" / "__init__.py"
        )
        contract_path = runtime_package_path.with_name("native_typed_contract.py")
        shutil.copy2(_openhands_native_tool_source(), native_tool_path)
        runtime_package_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_package_path.write_text(
            '"""Isolated native typed contract package."""\n', encoding="utf-8"
        )
        shutil.copy2(_native_typed_contract_source(), contract_path)
        result.update(
            {
                "native_tool_path": native_tool_path,
                "native_tool_sha256": sha256_file(native_tool_path),
                "native_contract_path": contract_path,
                "native_contract_sha256": sha256_file(contract_path),
            }
        )
    return result


def _proxy_environment(parent: Any) -> dict[str, str]:
    """Forward proxy variables without making behavior depend on casing.

    Different CLI/network stacks consult different spellings. Mirror a value
    when only one spelling exists, preserve explicitly configured spellings,
    and union NO_PROXY so tokenized localhost runtime traffic can never be sent
    to an outbound proxy that cannot reach private addresses.
    """

    result: dict[str, str] = {}
    for upper, lower in (
        ("HTTP_PROXY", "http_proxy"),
        ("HTTPS_PROXY", "https_proxy"),
        ("ALL_PROXY", "all_proxy"),
    ):
        upper_value = parent.get(upper)
        lower_value = parent.get(lower)
        if upper_value:
            result[upper] = str(upper_value)
        if lower_value:
            result[lower] = str(lower_value)
        if upper_value and not lower_value:
            result[lower] = str(upper_value)
        elif lower_value and not upper_value:
            result[upper] = str(lower_value)

    bypass: list[str] = []
    for name in ("NO_PROXY", "no_proxy"):
        value = parent.get(name)
        if value:
            bypass.extend(item.strip() for item in str(value).split(","))
    bypass.extend(("localhost", "127.0.0.1", "::1"))
    normalized = ",".join(dict.fromkeys(item for item in bypass if item))
    result["NO_PROXY"] = normalized
    result["no_proxy"] = normalized
    return result


def _redact_command(command: list[str]) -> list[str]:
    # Commands carry task/system prompts and paths but never API keys. Keep both
    # prompts out of the manifest; their hashes are recorded separately.
    result = list(command)
    for index, item in enumerate(result[:-1]):
        if item == "--append-system-prompt":
            result[index + 1] = "<benchmark-system-redacted>"
        elif item == "-c" and result[index + 1].startswith(
            "developer_instructions="
        ):
            result[index + 1] = (
                'developer_instructions="<benchmark-system-redacted>"'
            )
    if result:
        result[-1] = "<prompt-redacted>"
    return result


def _load_benchmark_system(workspace: Path) -> tuple[str, Path]:
    path = (workspace / BENCHMARK_SYSTEM_FILENAME).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"staged benchmark system instructions are missing: {path}"
        )
    value = path.read_text(encoding="utf-8")
    if not value.strip():
        raise ValueError("benchmark system instructions must not be empty")
    return value, path


def _benchmark_system_record(
    workspace: Path, delivery: str, *, path: Path | None = None
) -> dict[str, Any]:
    resolved = path or (workspace / BENCHMARK_SYSTEM_FILENAME).resolve()
    return {
        "source": "single_staged_file",
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
        "delivery": delivery,
    }


def _last_agent_text(stream: str) -> str:
    last = ""
    for line in stream.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in ("result", "text", "message"):
            candidate = value.get(key) if isinstance(value, dict) else None
            if isinstance(candidate, str):
                last = candidate
    return last or stream[-10_000:]


def _find_structured(
    *texts: str, schema: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    matches: dict[str, dict[str, Any]] = {}

    def accept(value: Any) -> None:
        found = _search_selected(value)
        if found is None:
            return
        if schema is not None:
            try:
                validate_json_schema_subset(found, schema, path="selector_output")
            except ValueError:
                return
        matches[json.dumps(found, sort_keys=True, ensure_ascii=False)] = found

    for text in texts:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.S)
        try:
            accept(json.loads(stripped))
        except json.JSONDecodeError:
            pass

        # Some Responses-compatible third-party models ignore the native
        # output-schema instruction but still emit one explicit JSON code
        # block after their explanation. Accept that representation only when
        # it independently satisfies the same local schema. Conflicting
        # structured objects remain ambiguous and are rejected below.
        for fenced in re.findall(
            r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL
        ):
            try:
                accept(json.loads(fenced))
            except json.JSONDecodeError:
                continue
        for line in reversed(text.splitlines()):
            try:
                accept(json.loads(line))
            except json.JSONDecodeError:
                continue
    if len(matches) == 1:
        return next(iter(matches.values()))
    return None


def _search_selected(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if isinstance(value.get("selected_skills"), list):
            return value
        for child in value.values():
            found = _search_selected(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _search_selected(child)
            if found:
                return found
    return None


def _collect_models(stream: str) -> set[str]:
    models: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"model", "model_id", "model_name"} and isinstance(child, str):
                    models.add(child)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for line in stream.splitlines():
        try:
            visit(json.loads(line))
        except json.JSONDecodeError:
            continue
    return models


def _collect_usage(
    stream: str, *, runner_kind: str | None = None
) -> dict[str, Any]:
    if runner_kind == "deepseek-harness":
        return _collect_deepseek_harness_usage(stream)
    if runner_kind == "openhands":
        return _collect_openhands_usage(stream)
    aliases = {
        "input_tokens": "input_tokens",
        "inputTokens": "input_tokens",
        "total_input_tokens": "input_tokens",
        "output_tokens": "output_tokens",
        "outputTokens": "output_tokens",
        "total_output_tokens": "output_tokens",
        "cache_read_input_tokens": "cache_read_input_tokens",
        "cache_creation_input_tokens": "cache_creation_input_tokens",
        "cached_input_tokens": "cached_input_tokens",
        "total_cost_usd": "cost_usd",
        "cost_usd": "cost_usd",
        "num_turns": "turns",
        "tool_calls": "tool_calls",
    }
    maxima: dict[str, float] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = aliases.get(key)
                if normalized and isinstance(child, (int, float)) and not isinstance(child, bool):
                    maxima[normalized] = max(float(child), maxima.get(normalized, 0.0))
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for line in stream.splitlines():
        try:
            visit(json.loads(line))
        except json.JSONDecodeError:
            continue
    result: dict[str, Any] = {}
    for key, value in maxima.items():
        result[key] = value if key == "cost_usd" else int(value)
    result["aggregation"] = "maximum reported cumulative value per field"
    return result


def _collect_deepseek_harness_usage(stream: str) -> dict[str, Any]:
    aliases = {
        "inputTokens": "input_tokens",
        "outputTokens": "output_tokens",
        "cacheReadTokens": "cache_read_input_tokens",
        "cacheWriteTokens": "cache_creation_input_tokens",
        "reasoningTokens": "reasoning_tokens",
    }
    totals: dict[str, int] = {}
    turns: set[tuple[int, int]] = set()
    tool_calls = 0
    for line in stream.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        data = event.get("data")
        if event.get("type") == "tool/call":
            tool_calls += 1
        if event.get("type") != "assistant/message" or not isinstance(data, dict):
            continue
        usage = data.get("usage")
        if not isinstance(usage, dict):
            continue
        turn = data.get("turn")
        step = data.get("step")
        if isinstance(turn, int) and isinstance(step, int):
            turns.add((turn, step))
        for source, destination in aliases.items():
            value = usage.get(source)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[destination] = totals.get(destination, 0) + int(value)
    totals["turns"] = len(turns)
    totals["tool_calls"] = tool_calls
    totals["aggregation"] = "sum of per-step DeepSeek Harness assistant/message usage"
    return totals


def _collect_openhands_usage(stream: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    partial: dict[str, Any] | None = None
    response_count = 0
    tool_calls = 0
    final_found = False
    for line in stream.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "openhands.tool.started":
            tool_calls += 1
        if event_type == "openhands.llm_response":
            response_count += 1
            accumulated = event.get("accumulated_usage")
            if isinstance(accumulated, dict):
                partial = accumulated
        if event_type != "openhands.result":
            continue
        usage = event.get("usage")
        if not isinstance(usage, dict):
            continue
        final_found = True
        for field in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "reasoning_tokens",
            "turns",
            "tool_calls",
        ):
            value = usage.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[field] = int(value)
        cost = usage.get("cost_usd")
        result["cost_usd"] = (
            float(cost)
            if isinstance(cost, (int, float)) and not isinstance(cost, bool)
            else None
        )
        result["cost_available"] = bool(usage.get("cost_available"))
        result["cost_source"] = str(usage.get("cost_source", "unknown"))
    if final_found:
        result["aggregation"] = "exact final OpenHands SDK metrics snapshot"
        return result
    if partial is not None:
        for field in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "reasoning_tokens",
        ):
            value = partial.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[field] = int(value)
        result["turns"] = response_count
        result["tool_calls"] = tool_calls
        result["cost_usd"] = None
        result["cost_available"] = False
        result["cost_source"] = "unavailable_from_incomplete_provider_run"
        result["aggregation"] = (
            "partial last accumulated OpenHands SDK metrics snapshot"
        )
        return result
    result["aggregation"] = "no OpenHands provider response available"
    return result

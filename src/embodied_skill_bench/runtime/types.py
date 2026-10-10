from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class ActionResult:
    result: dict[str, Any]
    simulator_steps: int = 0
    vla_chunks: int = 0


@dataclass
class PerceptionResult:
    result: dict[str, Any]


@dataclass
class CapturedObservation:
    task: dict[str, Any]
    status: dict[str, Any]
    robot: dict[str, Any]
    artifacts: dict[str, Any]
    object_names: list[str] = field(default_factory=list)
    backend: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    environment_id: str

    def reset(self) -> None: ...

    def execute(self, skill_id: str, arguments: dict[str, Any]) -> ActionResult: ...

    def perceive(
        self,
        skill_id: str,
        arguments: dict[str, Any],
        observation: dict[str, Any],
        destination: Path,
    ) -> PerceptionResult: ...

    def capture(self, destination: Path, step_index: int) -> CapturedObservation: ...

    def close(self) -> None: ...

    def provenance(self) -> dict[str, Any]: ...


class UnsupportedSkillError(RuntimeError):
    pass


class InvalidSkillArgumentsError(ValueError):
    """The agent supplied arguments rejected before backend side effects."""


class EpisodeFinishedError(RuntimeError):
    pass

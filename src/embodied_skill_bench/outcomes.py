from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


FAILURE_CLASSES = {
    "task_failure",
    "agent_protocol_failure",
    "provider_transport_failure",
    "simulator_failure",
}

_PROVIDER_TRANSPORT_PATTERNS = (
    (
        "openhands_provider_transport_failure",
        re.compile(r'"failure_class"\s*:\s*"provider_transport_failure"', re.I),
    ),
    ("rate_limit_exceeded", re.compile(r"rate_limit_exceeded", re.I)),
    ("too_many_requests", re.compile(r"too_many_requests", re.I)),
    (
        "invalid_encrypted_content",
        re.compile(
            r"(?:invalid_encrypted_content|encrypted content could not be (?:decrypted|parsed))",
            re.I,
        ),
    ),
    (
        "provider_instant_inference_quota_exhausted",
        re.compile(r"insufficient quota available for instant inference", re.I),
    ),
    ("http_429", re.compile(r"HTTP (?:error )?429\b", re.I)),
    (
        "stream_disconnected_before_completion",
        re.compile(r"stream disconnected before completion", re.I),
    ),
    ("response_failed_event", re.compile(r"response\.failed", re.I)),
    (
        "provider_bad_response_status",
        re.compile(r"bad_response_status_code", re.I),
    ),
    ("provider_request_error", re.compile(r"error sending request", re.I)),
    (
        "connection_interrupted",
        re.compile(r"connection (?:error|reset|refused|closed|aborted)", re.I),
    ),
    (
        "upstream_or_gateway_timeout",
        re.compile(r"(?:upstream|gateway) (?:timeout|timed out|disconnect)", re.I),
    ),
    ("gateway_http_error", re.compile(r"HTTP (?:error )?(?:502|503|504)\b", re.I)),
    ("anthropic_overloaded", re.compile(r"(?:overloaded_error|HTTP (?:error )?529)\b", re.I)),
    (
        "bedrock_invalid_beta_flag",
        re.compile(
            r"Bedrock Runtime:.*ValidationException: invalid beta flag",
            re.I | re.S,
        ),
    ),
)


def _read_diagnostic(path: Any, limit: int = 200_000) -> str:
    if not isinstance(path, str) or not path:
        return ""
    candidate = Path(path)
    if not candidate.is_file():
        return ""
    value = candidate.read_text(encoding="utf-8", errors="replace")
    return value[-limit:]


def provider_transport_evidence(
    agent_outcome: dict[str, Any] | None,
) -> list[str]:
    """Return stable evidence labels without copying large/provider-sensitive traces."""

    if not agent_outcome:
        return []
    stdout_text = _read_diagnostic(agent_outcome.get("stdout_path"))
    text = "\n".join(
        [
            str(agent_outcome.get("final_text") or ""),
            stdout_text,
            _read_diagnostic(agent_outcome.get("stderr_path")),
        ]
    )
    evidence = [
        label
        for label, pattern in _PROVIDER_TRANSPORT_PATTERNS
        if pattern.search(text)
    ]
    # The CLI uses 124 when its provider turn reaches the harness timeout.  A
    # trace containing only session/turn-start records means the provider never
    # returned an assistant item or token; this is an infrastructure no-response
    # condition.  A timeout after any assistant/tool activity stays an eligible
    # agent/task outcome and is not caught by this narrow rule.
    if int(agent_outcome.get("return_code") or 0) == 124:
        lowered = stdout_text.lower()
        openhands_started = len(
            re.findall(
                r'"type"\s*:\s*"openhands\.provider_request\.started"',
                stdout_text,
                re.I,
            )
        )
        openhands_finished = len(
            re.findall(
                r'"type"\s*:\s*"openhands\.provider_request\.(?:completed|failed)"',
                stdout_text,
                re.I,
            )
        )
        if openhands_started > openhands_finished:
            evidence.append("openhands_provider_no_response_timeout")
        provider_started = any(
            marker in lowered
            for marker in ('"type":"turn.started"', '"subtype":"init"')
        )
        assistant_activity = any(
            marker in lowered
            for marker in (
                '"type":"item.',
                '"type":"assistant"',
                '"type":"tool_use"',
                '"type":"result"',
            )
        )
        if provider_started and not assistant_activity:
            evidence.append("provider_no_response_timeout")
    return evidence


def classify_condition(
    *,
    episode: dict[str, Any] | None,
    agent_outcome: dict[str, Any] | None,
    protocol_violations: Iterable[str] = (),
    startup_failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a condition while keeping infra incidents visible.

    Task and agent-protocol failures are eligible benchmark outcomes. Provider
    transport and simulator failures are infrastructure failures and therefore
    do not enter the task-success denominator.
    """

    episode = episode or {}
    calls = episode.get("calls") if isinstance(episode.get("calls"), dict) else {}
    violations = [str(item) for item in protocol_violations if str(item)]
    incidents: list[dict[str, Any]] = []

    provider_evidence = provider_transport_evidence(agent_outcome)
    if provider_evidence:
        incidents.append(
            {
                "class": "provider_transport_failure",
                "source": "agent_provider_trace",
                "evidence": provider_evidence,
            }
        )
    if int(calls.get("errors") or 0) > 0:
        incidents.append(
            {
                "class": "simulator_failure",
                "source": "runtime_action_error",
                "count": int(calls.get("errors") or 0),
            }
        )
    if startup_failure:
        failure_class = str(startup_failure.get("class") or "simulator_failure")
        if failure_class not in FAILURE_CLASSES:
            raise ValueError(f"unknown startup failure class: {failure_class}")
        incidents.append(
            {
                "class": failure_class,
                "source": str(startup_failure.get("source") or "condition_startup"),
                "message": str(startup_failure.get("message") or "startup failed"),
            }
        )
    rejected = int(calls.get("rejected") or 0)
    # A manifest-level specialist call limit is an embodied-policy boundary,
    # not a malformed runtime request.  The runtime records those rejections
    # separately so they remain visible without turning an otherwise valid
    # task attempt into an agent-protocol failure.
    skill_call_limit_rejections = max(
        0, int(episode.get("skill_call_limit_rejections") or 0)
    )
    protocol_rejected = max(0, rejected - skill_call_limit_rejections)
    return_code = agent_outcome.get("return_code") if agent_outcome else None
    # A provider transport error commonly exits the agent process non-zero.  The
    # return code is then evidence for the already-recorded infrastructure
    # incident, not a second agent-protocol failure.  Explicit violations and
    # rejected runtime calls remain protocol failures even when transport also
    # failed.
    nonzero_without_provider = (
        return_code not in (None, 0) and not provider_evidence
    )
    if violations or protocol_rejected or nonzero_without_provider:
        incidents.append(
            {
                "class": "agent_protocol_failure",
                "source": "agent_or_runtime_protocol",
                "violations": violations,
                "rejected_calls": protocol_rejected,
                "skill_call_limit_rejections": skill_call_limit_rejections,
                "return_code": return_code,
            }
        )

    success = bool(episode.get("success"))
    if success:
        primary = None
        eligible = True
        status = "success"
    else:
        observed = {item["class"] for item in incidents}
        if "simulator_failure" in observed:
            primary = "simulator_failure"
            eligible = False
        elif "provider_transport_failure" in observed:
            primary = "provider_transport_failure"
            eligible = False
        elif "agent_protocol_failure" in observed:
            primary = "agent_protocol_failure"
            eligible = True
        else:
            primary = "task_failure"
            eligible = True
            incidents.append(
                {
                    "class": "task_failure",
                    "source": "official_task_status",
                    "message": "episode ended without official task success",
                }
            )
        status = "failure"

    return {
        "status": status,
        "success": success,
        "failure_class": primary,
        "eligible": eligible,
        "incidents": incidents,
    }


def aggregate_classified_conditions(
    conditions: list[dict[str, Any]],
    *,
    success_getter,
    classification_getter,
    descriptor_getter,
) -> dict[str, Any]:
    eligible = [
        item for item in conditions if bool(classification_getter(item).get("eligible"))
    ]
    eligible_successes = sum(bool(success_getter(item)) for item in eligible)
    raw_successes = sum(bool(success_getter(item)) for item in conditions)
    failure_counts: Counter[str] = Counter()
    incident_counts: Counter[str] = Counter()
    infrastructure_failures: list[dict[str, Any]] = []
    infrastructure_incidents: list[dict[str, Any]] = []
    for item in conditions:
        classification = classification_getter(item)
        failure_class = classification.get("failure_class")
        if failure_class:
            failure_counts[str(failure_class)] += 1
        infra_incidents = [
            incident
            for incident in classification.get("incidents", [])
            if incident.get("class")
            in {"provider_transport_failure", "simulator_failure"}
        ]
        for incident in classification.get("incidents", []):
            incident_class = incident.get("class")
            if incident_class:
                incident_counts[str(incident_class)] += 1
        if infra_incidents:
            infrastructure_incidents.append(
                {
                    **descriptor_getter(item),
                    "condition_eligible": bool(classification.get("eligible")),
                    "condition_success": bool(success_getter(item)),
                    "incidents": infra_incidents,
                }
            )
        if not classification.get("eligible"):
            infrastructure_failures.append(
                {
                    **descriptor_getter(item),
                    "failure_class": failure_class,
                    "incidents": classification.get("incidents", []),
                }
            )
    return {
        "success_rate": (
            eligible_successes / len(eligible) if eligible else None
        ),
        "success_rate_numerator": eligible_successes,
        "success_rate_denominator": len(eligible),
        "raw_success_rate": raw_successes / len(conditions) if conditions else None,
        "raw_condition_count": len(conditions),
        "eligible_condition_count": len(eligible),
        "ineligible_condition_count": len(conditions) - len(eligible),
        "failure_counts": dict(sorted(failure_counts.items())),
        "incident_counts": dict(sorted(incident_counts.items())),
        "infrastructure_failures": infrastructure_failures,
        "infrastructure_incidents": infrastructure_incidents,
    }

from __future__ import annotations

from typing import Any


# Diagnostic only. This threshold does not verify a grasp; the development
# calibration must freeze the final empty-close distribution before headline
# experiments.
EMPTY_CLOSURE_SUSPICION_UPPER_M = 0.006


def annotate_pi0_pick_result(
    raw_result: dict[str, Any],
    *,
    empty_closure_upper_m: float = EMPTY_CLOSURE_SUSPICION_UPPER_M,
) -> dict[str, Any]:
    """Replace RPent's ambiguous pick `success` with evidence-level fields.

    RPent's legacy flag only combines a descent, post-descent ascent, and a
    gripper-width upper bound. It neither proves that an object is held nor
    grounds the intended object identity. Do not turn a width heuristic into
    a new binary success label: expose the evidence and require observation-
    level verification instead.
    """

    result = dict(raw_result)
    kinematic_postcondition_met = bool(result.pop("success", False))
    opening_value = result.get("final_gripper_opening")
    opening = float(opening_value) if opening_value is not None else None
    empty_closure_suspected = bool(
        kinematic_postcondition_met
        and opening is not None
        and opening <= empty_closure_upper_m
    )
    if empty_closure_suspected:
        kinematic_hint_status = "empty_closure_suspected"
    elif kinematic_postcondition_met:
        kinematic_hint_status = "unverified_kinematic_candidate"
    else:
        kinematic_hint_status = "kinematic_postcondition_not_met"

    result.update(
        {
            "kinematic_postcondition_met": kinematic_postcondition_met,
            "empty_closure_suspected": empty_closure_suspected,
            "empty_closure_suspicion_upper_m": empty_closure_upper_m,
            # Do not let the name of a local kinematic heuristic masquerade as
            # a semantic success/failure verdict. In particular,
            # `kinematic_postcondition_not_met` previously encouraged agents
            # to repeat a visually adequate grasp even though the heuristic
            # has known false negatives. Public visual/hold evidence is still
            # required in every branch.
            "kinematic_hint_status": kinematic_hint_status,
            "grasp_verification_status": "unknown_requires_observation_evidence",
            "grasp_verified": None,
            "target_identity_verified": None,
            "verification_required": True,
            "interpretation": (
                "The kinematic hint establishes neither grasp success nor grasp failure; "
                "inspect the fresh public observation for target motion, identity, and "
                "hold stability before transport or recovery."
            ),
            "verification_evidence": {
                "hold_stability": "not_measured",
                "visual_object_motion": "not_evaluated",
                "target_identity": "not_evaluated",
                "contact": "not_exposed_to_agent",
            },
        }
    )
    return result

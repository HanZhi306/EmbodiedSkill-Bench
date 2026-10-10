from __future__ import annotations

import hashlib


VLA_SEED_NAMESPACE = "embodied-skill-bench/vla-cell/v1"
CHALLENGE_PAIR_VLA_SEED_NAMESPACE = (
    "embodied-skill-bench/challenge-pair-vla-cell/v1"
)


def derive_vla_seed(*, suite: str, task: int, environment_seed: int) -> int:
    """Derive one stable VLA RNG stream per paired simulator cell.

    Compared methods for a fixed suite/task/environment seed receive the same
    value. Different tasks no longer reuse an identical VLA stream merely
    because their LIBERO init-state seed is equal.
    """

    payload = (
        f"{VLA_SEED_NAMESPACE}\0{suite}\0{int(task)}\0{int(environment_seed)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def derive_challenge_pair_vla_seed(*, pair_id: str, environment_seed: int) -> int:
    """Share one VLA stream across clean/hard variants of a challenge pair."""

    if not pair_id:
        raise ValueError("pair_id must be non-empty")
    payload = (
        f"{CHALLENGE_PAIR_VLA_SEED_NAMESPACE}\0{pair_id}\0{int(environment_seed)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF

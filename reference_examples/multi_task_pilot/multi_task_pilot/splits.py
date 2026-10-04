from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any


def stable_rank(identifier: str, seed: int) -> str:
    """Return a platform-independent deterministic ordering key."""
    return hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()


def fixed_split(
    records: Iterable[dict[str, Any]],
    *,
    id_field: str,
    seed: int,
    dev_size: int,
    test_size: int,
) -> dict[str, list[dict[str, Any]]]:
    """Create disjoint pilot splits without depending on input row order."""
    rows = list(records)
    identifiers = [str(row[id_field]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate values in id field {id_field!r}")
    if dev_size < 0 or test_size < 0:
        raise ValueError("Split sizes must be non-negative")
    if dev_size + test_size > len(rows):
        raise ValueError("Requested dev and test splits exceed dataset size")

    ordered = sorted(rows, key=lambda row: stable_rank(str(row[id_field]), seed))
    return {
        "dev": ordered[:dev_size],
        "test": ordered[dev_size : dev_size + test_size],
        "reserve": ordered[dev_size + test_size :],
    }


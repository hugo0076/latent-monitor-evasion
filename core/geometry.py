"""Residual-stream geometry of the activation edit.

The edit Δh is the mean prefill activation under the elevate instruction minus the mean
under the suppress instruction. The demonstrations and the prefill are identical in the two
prompts, so Δh isolates the effect of the instruction.

Δh points toward detection. Evasion moves along -Δh. The elevate arm is the arm whose
effective target is 1 (`imitate == c_label`), not the arm with `imitate=1`. Ordering the arms
by `imitate` flips the sign on half the episodes.
"""

from __future__ import annotations

import numpy as np


def delta_h(*, elevate: np.ndarray, suppress: np.ndarray) -> np.ndarray:
    """The edit, `elevate - suppress`. Keyword-only, so the arms cannot be swapped."""
    if elevate.shape != suppress.shape:
        raise ValueError(f"arm shapes differ: {elevate.shape} vs {suppress.shape}")
    return np.asarray(elevate, np.float64) - np.asarray(suppress, np.float64)


def unit(v: np.ndarray) -> np.ndarray:
    """Normalise along the last axis. A zero vector stays zero, not NaN."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v, dtype=np.float64), where=n > 0)


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """All pairwise cosines of the rows of `vectors` [n, D]."""
    u = unit(np.asarray(vectors, np.float64))
    return u @ u.T

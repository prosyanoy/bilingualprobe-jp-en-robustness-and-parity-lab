"""Bootstrap statistics for per-language scores and cross-language gaps."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of the mean. Returns ``(mean, lo, hi)`` or ``None`` if empty."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, (n_boot, v.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def bootstrap_gap(a, ids_a, b, ids_b, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """CI for ``mean(a) - mean(b)``.

    If both samples cover exactly the same unique ids (parallel data), items are
    resampled *jointly* by id (paired bootstrap, which removes item difficulty
    from the variance). Otherwise the two samples are resampled independently.
    Returns ``(gap, lo, hi, paired)`` or ``None`` if either side is empty.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return None
    rng = np.random.default_rng(seed)
    ids_a, ids_b = list(ids_a), list(ids_b)
    paired = len(set(ids_a)) == len(ids_a) and set(ids_a) == set(ids_b)
    if paired:
        order_b = {i: k for k, i in enumerate(ids_b)}
        b = b[[order_b[i] for i in ids_a]]
        idx = rng.integers(0, a.size, (n_boot, a.size))
        gaps = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    else:
        ia = rng.integers(0, a.size, (n_boot, a.size))
        ib = rng.integers(0, b.size, (n_boot, b.size))
        gaps = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.quantile(gaps, [alpha / 2, 1 - alpha / 2])
    return float(a.mean() - b.mean()), float(lo), float(hi), paired


def parity_ratio(x: float, y: float) -> float:
    """min/max ratio in [0, 1]; 1.0 means identical scores."""
    hi = max(x, y)
    return 1.0 if hi == 0 else min(x, y) / hi

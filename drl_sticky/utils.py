"""Small numerical helpers (moving averages, confidence intervals, JSON I/O)."""

from __future__ import annotations

import json
import math
from typing import Any, Sequence, Tuple

import numpy as np


def moving_average(x: Sequence[float], window: int = 100) -> np.ndarray:
    """Trailing moving average over the previous `window` values.

    Element k is the mean of x[max(0, k-window+1) : k+1], so the curve starts at
    episode 0 (shorter windows at the beginning) and has the same length as x.
    """
    arr = np.asarray(x, dtype=float)               # convert input to float array
    csum = np.cumsum(np.insert(arr, 0, 0.0))       # prefix sums, csum[k] = sum(arr[:k])
    idx = np.arange(1, len(arr) + 1)               # 1-based end index of each window
    start = np.maximum(0, idx - window)            # start index of each window
    return (csum[idx] - csum[start]) / (idx - start)  # window sums / window lengths


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion."""
    if trials == 0:                                # undefined interval with no data
        return (float("nan"), float("nan"))
    p = successes / trials                         # point estimate
    denom = 1 + z * z / trials                     # Wilson denominator
    centre = (p + z * z / (2 * trials)) / denom    # shifted centre
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (centre - half, centre + half)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays into plain Python types for json.dump."""
    if isinstance(obj, dict):                      # recurse into dicts
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):             # recurse into sequences
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):                # arrays -> lists
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):                # numpy ints -> int
        return int(obj)
    if isinstance(obj, np.floating):               # numpy floats -> float (NaN -> None)
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, float) and math.isnan(obj): # Python NaN -> null in JSON
        return None
    return obj


def save_json(obj: Any, path: str) -> None:
    """Write `obj` to `path` as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as fh:  # text mode, UTF-8
        json.dump(to_jsonable(obj), fh, indent=2)


def load_json(path: str) -> Any:
    """Read a JSON file written by save_json."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

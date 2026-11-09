"""Uniform state discretization for the 2-D MountainCar observation."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .config import OBS_HIGH, OBS_LOW


class UniformDiscretizer:
    """Maps a continuous (position, velocity) observation to a pair of bin indices.

    index_d = clip(floor((x_d - low_d) / width_d), 0, n_d - 1), width_d = (high_d - low_d) / n_d
    """

    def __init__(
        self,
        n_bins: Sequence[int],
        low: Sequence[float] = OBS_LOW,
        high: Sequence[float] = OBS_HIGH,
    ) -> None:
        """Store the number of bins and the bounds for each dimension."""
        self.n_bins = np.asarray(n_bins, dtype=int)   # (n_pos, n_vel)
        self.low = np.asarray(low, dtype=float)       # lower bounds
        self.high = np.asarray(high, dtype=float)     # upper bounds
        self.width = (self.high - self.low) / self.n_bins  # bin widths

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape (n_pos, n_vel) - leading dimensions of the Q-table."""
        return int(self.n_bins[0]), int(self.n_bins[1])

    def __call__(self, obs: Sequence[float]) -> Tuple[int, int]:
        """Return the (position_bin, velocity_bin) of a single observation."""
        # Small epsilon guards against float32 round-off exactly on a bin edge.
        raw = np.floor((np.asarray(obs, dtype=float) - self.low) / self.width + 1e-9)
        idx = np.clip(raw, 0, self.n_bins - 1).astype(int)  # keep inside the grid
        return int(idx[0]), int(idx[1])

    def batch(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised version for an (N, 2) array of observations."""
        raw = np.floor((np.asarray(obs, dtype=float) - self.low) / self.width + 1e-9)
        idx = np.clip(raw, 0, self.n_bins - 1).astype(int)
        return idx[:, 0], idx[:, 1]

    def edges(self, dim: int) -> np.ndarray:
        """Bin edges of dimension `dim` (0 = position, 1 = velocity) - for plotting."""
        return np.linspace(self.low[dim], self.high[dim], self.n_bins[dim] + 1)

    def centers(self, dim: int) -> np.ndarray:
        """Bin centres of dimension `dim`."""
        e = self.edges(dim)
        return 0.5 * (e[:-1] + e[1:])

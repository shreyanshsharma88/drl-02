"""Experiment configuration shared by every run (Group 84 design choices)."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class Config:
    """All hyper-parameters of one training run.

    The SAME Config object is used for the four required agents
    (Q-learning / SARSA x original / modified environment), which guarantees
    identical seed, discretization, alpha, gamma, exploration schedule and
    number of episodes, as demanded by tasks (b) and (c).
    """

    # --- discretization -------------------------------------------------
    # 20 x 20 uniform bins (position width 0.09, velocity width 0.007).
    # Chosen by the held-out-seed pilot (pilot_sweep.py): best mean greedy reward and the
    # smallest seed-to-seed spread among 20x20 / 36x28 / 40x40 grids. Fewer cells means
    # each cell is visited more often, so the zero-initialised table converges faster.
    n_pos_bins: int = 20
    n_vel_bins: int = 20
    # --- learning ---------------------------------------------------------
    alpha: float = 0.10          # constant learning rate (pilot: 0.05 / 0.1 / 0.2)
    gamma: float = 0.99          # discount factor (200-step horizon)
    # --- exploration (exponential decay, floored) -------------------------
    eps_start: float = 1.0       # epsilon at episode 0
    eps_min: float = 0.01        # floor reached at eps_decay_frac * n_episodes
    eps_decay_frac: float = 0.70 # fraction of training over which epsilon decays (pilot: 0.5 / 0.7)
    # --- budget & reproducibility -----------------------------------------
    n_episodes: int = 10000      # >= 5000 required; pilot: 10000 clearly better than 6000
    seed: int = 84               # master seed = our group ID
    # --- environment modification parameters (from the assignment) --------
    stick_prob: float = 0.20     # probability an attempted action change is replaced
    push_penalty: float = 0.2    # energy penalty for selecting action 0 or 2
    bonus: float = 20.0          # gentle-arrival bonus
    gentle_speed: float = 0.04   # maximum arrival velocity for the bonus

    @property
    def eps_decay(self) -> float:
        """Per-episode multiplicative decay d such that eps_start*d^k hits eps_min at k = frac*N."""
        # Solve eps_start * d^(frac*N) = eps_min for d.
        steps = max(1.0, self.eps_decay_frac * self.n_episodes)
        return math.exp(math.log(self.eps_min / self.eps_start) / steps)

    def epsilon(self, episode: int) -> float:
        """Exploration rate used during training episode `episode` (0-based)."""
        # Exponential decay, clipped from below at eps_min.
        return max(self.eps_min, self.eps_start * self.eps_decay ** episode)

    def seeds(self) -> Dict[str, int]:
        """Derive independent seeds for each random stream from the master seed."""
        # Separate streams so that the sticky actuator's draws never shift the
        # agent's exploration draws (and vice versa).
        return {
            "env_reset": self.seed,            # first reset of the base env
            "sticky": self.seed + 1000,        # sticky-actuator uniform draws
            "agent": self.seed + 2000,         # epsilon-greedy draws and tie-breaking
            "validation": self.seed + 3000,    # fixed validation-state sampling
        }

    def to_dict(self) -> dict:
        """Plain dict (for results.json), including derived quantities."""
        d = asdict(self)
        d["eps_decay"] = self.eps_decay
        d["seeds"] = self.seeds()
        return d


#: Observation bounds of MountainCar-v0 (position, velocity).
OBS_LOW: Tuple[float, float] = (-1.2, -0.07)
OBS_HIGH: Tuple[float, float] = (0.6, 0.07)

#: Human-readable action names.
ACTION_NAMES = {0: "Left", 1: "None", 2: "Right"}

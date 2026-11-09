"""Tabular TD-control agents: Q-learning (off-policy) and SARSA (on-policy).

Both agents share EVERYTHING (Q-table, discretization, epsilon-greedy policy,
update code) except ``td_target`` - the only difference required by task (c).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

State = Tuple[int, int]   # discretized state (position_bin, velocity_bin)


class TabularTDAgent:
    """Common base: Q-table of shape (n_pos, n_vel, 3) initialised to zero."""

    name = "TD"

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        n_actions: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.99,
        seed: Optional[int] = None,
    ) -> None:
        """Create the zero-initialised Q-table and the agent's private RNG."""
        self.n_actions = n_actions                                  # |A| = 3
        self.alpha = alpha                                          # learning rate
        self.gamma = gamma                                          # discount factor
        self.Q = np.zeros((*grid_shape, n_actions), dtype=float)    # Q(s, a) = 0
        self.visits = np.zeros((*grid_shape, n_actions), dtype=np.int64)  # update counts
        self.rng = np.random.default_rng(seed)                      # exploration RNG

    def greedy(self, s: State) -> int:
        """argmax_a Q(s, a); ties broken uniformly at random (important with zero init)."""
        q = self.Q[s]                                   # action values in state s
        best = np.flatnonzero(q == q.max())             # all maximising actions
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best))

    def act(self, s: State, epsilon: float) -> int:
        """epsilon-greedy action selection."""
        if self.rng.random() < epsilon:                 # explore with probability epsilon
            return int(self.rng.integers(self.n_actions))
        return self.greedy(s)                           # otherwise exploit

    def td_target(self, r: float, s_next: State, a_next: int, terminated: bool) -> float:
        """TD target - implemented by subclasses (the ONLY algorithmic difference)."""
        raise NotImplementedError

    def update(self, s: State, a: int, r: float, s_next: State, a_next: int, terminated: bool) -> float:
        """Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a)); returns the TD error.

        `a` is the action SELECTED by the agent (the executed one is hidden).
        `terminated` (not truncated) switches off bootstrapping.
        """
        target = self.td_target(r, s_next, a_next, terminated)  # algorithm-specific target
        delta = target - self.Q[s + (a,)]                        # TD error
        self.Q[s + (a,)] += self.alpha * delta                   # incremental update
        self.visits[s + (a,)] += 1                               # bookkeeping for policy maps
        return float(delta)

    def state_values(self, pos_idx: np.ndarray, vel_idx: np.ndarray) -> np.ndarray:
        """max_a Q(s, a) for a batch of discretized states (used for the value plots)."""
        return self.Q[pos_idx, vel_idx].max(axis=1)

    def greedy_policy_map(self) -> np.ndarray:
        """Greedy action for every grid cell (argmax over actions, first max on ties)."""
        return self.Q.argmax(axis=2)


class QLearningAgent(TabularTDAgent):
    """Off-policy TD control: target = R + gamma * max_a' Q(s', a')."""

    name = "Q-learning"

    def td_target(self, r: float, s_next: State, a_next: int, terminated: bool) -> float:
        """Bootstrap from the greedy value of s' (a_next is ignored); none if terminated."""
        if terminated:                                   # true terminal state: no future value
            return r
        return r + self.gamma * float(self.Q[s_next].max())


class SarsaAgent(TabularTDAgent):
    """On-policy TD control: target = R + gamma * Q(s', a'), a' = next epsilon-greedy action."""

    name = "SARSA"

    def td_target(self, r: float, s_next: State, a_next: int, terminated: bool) -> float:
        """Bootstrap from the value of the action actually chosen next; none if terminated."""
        if terminated:                                   # true terminal state: no future value
            return r
        return r + self.gamma * float(self.Q[s_next + (a_next,)])


AGENTS = {"qlearning": QLearningAgent, "sarsa": SarsaAgent}

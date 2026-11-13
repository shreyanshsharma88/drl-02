"""Greedy-policy evaluation with fixed seeds (task d)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .agents import TabularTDAgent
from .config import Config
from .discretizer import UniformDiscretizer
from .env import NO_PUSH, make_env

EVAL_SEED_BASE = 10_000        # base-env reset seeds 10000..10099
EVAL_STICKY_BASE = 20_000      # sticky-RNG seeds 20000..20099


def evaluate_greedy(
    agent: TabularTDAgent,
    modified: bool,
    cfg: Config,
    n_episodes: int = 100,
    seed_base: int = EVAL_SEED_BASE,
    sticky_base: int = EVAL_STICKY_BASE,
) -> Dict[str, object]:
    """Run the greedy policy (epsilon = 0) for `n_episodes` fixed-seed test episodes.

    Episode i uses reset seed seed_base+i (same start states for every agent) and, in
    the modified env, sticky seed sticky_base+i (same actuator noise for every agent).
    The hyper-parameter pilot uses different seed bases so the final test set is untouched.
    """
    disc = UniformDiscretizer((cfg.n_pos_bins, cfg.n_vel_bins))
    env = make_env(modified, sticky_seed=0) if modified else make_env(False)
    rets, lens, succ, gentle, pushes, vel = [], [], [], [], [], []
    for i in range(n_episodes):
        if modified:
            env.seed_sticky(sticky_base + i)          # fixed actuator noise
        obs, _ = env.reset(seed=seed_base + i)        # fixed start state
        ret, n, p, term, trunc = 0.0, 0, 0, False, False
        while not (term or trunc):
            a = int(np.argmax(agent.Q[disc(obs)]))         # deterministic greedy action
            obs, r, term, trunc, _ = env.step(a)
            ret += r
            n += 1
            p += int(a != NO_PUSH)
        rets.append(ret)
        lens.append(n)
        succ.append(float(term))
        gentle.append(float(term and not trunc and obs[0] >= 0.5 and obs[1] <= cfg.gentle_speed))
        pushes.append(p)
        vel.append(float(obs[1]) if term else np.nan)
    env.close()
    vel_arr = np.array(vel)
    return {
        "mean_reward": float(np.mean(rets)),
        "std_reward": float(np.std(rets)),
        "success_rate": float(np.mean(succ)),
        "gentle_rate": float(np.mean(gentle)),
        "mean_length": float(np.mean(lens)),
        "mean_pushes": float(np.mean(pushes)),
        "push_fraction": float(np.sum(pushes) / max(1, np.sum(lens))),
        "mean_arrival_velocity": float(np.nanmean(vel_arr)) if np.any(~np.isnan(vel_arr)) else float("nan"),
        "returns": rets,
        "lengths": lens,
        "arrival_velocities": vel,
    }


def policy_map(agent: TabularTDAgent) -> np.ndarray:
    """Greedy action per (position, velocity) cell; -1 marks cells never updated."""
    pm = agent.greedy_policy_map().astype(int)            # argmax over actions
    pm[agent.visits.sum(axis=2) == 0] = -1                # unvisited -> masked
    return pm


def value_calibration(agent: TabularTDAgent, modified: bool, cfg: Config, states: np.ndarray,
                      n_rollouts: int = 5, seed: int = 40_000) -> Dict[str, float]:
    """Compare the predicted value max_a Q(s,a) with the discounted return actually obtained
    by the greedy policy when the car is placed in state s (Monte-Carlo estimate).

    Each rollout starts from a validation state with a_prev = 1 and runs for at most
    200 steps. A positive bias (predicted > actual) indicates over-estimation.
    """
    disc = UniformDiscretizer((cfg.n_pos_bins, cfg.n_vel_bins))
    env = make_env(modified, sticky_seed=seed) if modified else make_env(False)
    predicted, actual = [], []
    for i, s0 in enumerate(states):
        p_idx = disc(s0)
        predicted.append(float(agent.Q[p_idx].max()))          # agent's own estimate
        rets = []
        for k in range(n_rollouts):
            env.reset(seed=seed + i * n_rollouts + k)             # resets TimeLimit and a_prev
            env.unwrapped.state = np.array(s0, dtype=np.float64)  # place the car in s0
            obs, g, disc_f, term, trunc = np.array(s0, dtype=np.float32), 0.0, 1.0, False, False
            while not (term or trunc):
                a = int(np.argmax(agent.Q[disc(obs)]))           # greedy action
                obs, r, term, trunc, _ = env.step(a)
                g += disc_f * r                                   # discounted return
                disc_f *= cfg.gamma
            rets.append(g)
        actual.append(float(np.mean(rets)))
    env.close()
    predicted_arr, actual_arr = np.array(predicted), np.array(actual)
    return {
        "mean_predicted": float(predicted_arr.mean()),
        "mean_actual": float(actual_arr.mean()),
        "bias": float((predicted_arr - actual_arr).mean()),
        "mae": float(np.abs(predicted_arr - actual_arr).mean()),
    }

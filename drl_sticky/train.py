"""Training loop shared by Q-learning and SARSA (tasks b and c)."""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np

from .agents import AGENTS, TabularTDAgent
from .config import Config
from .discretizer import UniformDiscretizer
from .env import NO_PUSH, make_env

#: Flag position of MountainCar-v0 (constant of the base environment).
GOAL_POSITION = 0.5

#: Column names of the per-episode training log.
LOG_COLUMNS = [
    "episode", "epsilon", "return", "shaped_return", "length", "success", "gentle",
    "arrival_velocity", "pushes", "avg_max_q", "replaced", "attempted_changes",
]


def sample_validation_states(n_states: int = 1000, seed: int = 3084, n_episodes: int = 60) -> np.ndarray:
    """Fixed validation set: states visited by a uniform-random policy in the ORIGINAL env.

    Sampled ONCE before any training and reused unchanged for all agents, so the
    'average predicted value' curves of task (d) are directly comparable.
    """
    rng = np.random.default_rng(seed)               # dedicated RNG for this set
    env = make_env(modified=False)                  # base dynamics only
    states = []                                     # all visited observations
    obs, _ = env.reset(seed=seed)                   # seeded first reset
    for _ in range(n_episodes):                     # collect random-policy rollouts
        done = False
        while not done:
            states.append(np.array(obs, dtype=float))
            obs, _, term, trunc, _ = env.step(int(rng.integers(3)))
            done = term or trunc
        obs, _ = env.reset()                        # continue the seeded RNG stream
    env.close()
    states = np.stack(states)                       # (n_visited, 2)
    pick = rng.choice(len(states), size=min(n_states, len(states)), replace=False)
    return states[pick]                             # (n_states, 2)


def train_agent(
    algo: str,
    modified: bool,
    cfg: Config,
    validation_states: np.ndarray,
    progress: Optional[Callable[[int, Dict[str, float]], None]] = None,
    progress_every: int = 500,
):
    """Train one agent and return (agent, log dict of per-episode arrays, wall-clock seconds).

    The loop is written SARSA-style for BOTH algorithms: the next action a' is chosen
    epsilon-greedily before the update and then actually executed. Q-learning simply
    ignores a' inside its TD target, so the two runs differ only in the target.
    """
    seeds = cfg.seeds()                                            # independent RNG streams
    env = make_env(
        modified, sticky_seed=seeds["sticky"], stick_prob=cfg.stick_prob,
        push_penalty=cfg.push_penalty, bonus=cfg.bonus, gentle_speed=cfg.gentle_speed,
    ) if modified else make_env(False)
    disc = UniformDiscretizer((cfg.n_pos_bins, cfg.n_vel_bins))   # same grid for every run
    agent: TabularTDAgent = AGENTS[algo](disc.shape, 3, cfg.alpha, cfg.gamma, seed=seeds["agent"])
    vp, vv = disc.batch(validation_states)                        # validation states -> bins

    log = {c: np.zeros(cfg.n_episodes) for c in LOG_COLUMNS}      # per-episode metrics
    log["arrival_velocity"][:] = np.nan                           # NaN when flag not reached
    t0 = time.time()
    obs, _ = env.reset(seed=seeds["env_reset"])                   # seed base env once
    for ep in range(cfg.n_episodes):
        if ep > 0:
            obs, _ = env.reset()                                  # continue seeded stream
        eps = cfg.epsilon(ep)                                     # exploration this episode
        s = disc(obs)                                             # discretize start state
        a = agent.act(s, eps)                                     # first action
        ret = shaped = 0.0                                        # returns
        pushes = replaced = attempts = steps = 0
        term = trunc = False
        while True:
            obs2, r, term, trunc, _ = env.step(a)                 # env executes (maybe sticky)
            if modified:                                          # audit counters (not seen by agent)
                attempts += int(env.last_r is not None)
                replaced += int(env.last_replaced)
            s2 = disc(obs2)                                       # next discretized state
            a2 = agent.act(s2, eps)                               # next action (epsilon-greedy)
            agent.update(s, a, r, s2, a2, term)                   # TD update with SELECTED action
            # bookkeeping
            steps += 1
            ret += r
            is_push = a != NO_PUSH
            pushes += int(is_push)
            gentle_now = term and not trunc and obs2[0] >= GOAL_POSITION and obs2[1] <= cfg.gentle_speed
            shaped += -1.0 - cfg.push_penalty * is_push + (cfg.bonus if gentle_now else 0.0)
            if term or trunc:
                break
            s, a = s2, a2                                         # move on
        log["episode"][ep] = ep
        log["epsilon"][ep] = eps
        log["return"][ep] = ret
        log["shaped_return"][ep] = shaped
        log["length"][ep] = steps
        log["success"][ep] = float(term)
        log["gentle"][ep] = float(term and obs2[1] <= cfg.gentle_speed)
        if term:
            log["arrival_velocity"][ep] = float(obs2[1])
        log["pushes"][ep] = pushes
        log["avg_max_q"][ep] = float(agent.state_values(vp, vv).mean())  # value on validation set
        log["replaced"][ep] = replaced
        log["attempted_changes"][ep] = attempts
        if progress is not None and ((ep + 1) % progress_every == 0 or ep == 0):
            progress(ep, {k: float(log[k][ep]) for k in LOG_COLUMNS})
    env.close()
    return agent, log, time.time() - t0

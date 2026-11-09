"""Experimental verification of the sticky-actuator wrapper (task a) and the
non-Markov evidence used in discussion question (e2)."""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, List, Sequence

import gymnasium as gym
import numpy as np

from .discretizer import UniformDiscretizer
from .env import NO_PUSH, StickyActuatorMountainCar, make_env
from .utils import wilson_ci


class ScriptedRNG:
    """Stand-in for numpy's Generator that returns pre-set uniform numbers (for the PDF example)."""

    def __init__(self, values: Sequence[float]) -> None:
        """Store the sequence of numbers to hand out."""
        self.values = list(values)                  # queue of draws
        self.calls = 0                              # how many draws were consumed

    def random(self) -> float:
        """Return the next scripted number (mimics Generator.random())."""
        v = self.values[self.calls]
        self.calls += 1
        return v


def spec_compliance() -> Dict[str, object]:
    """Check that everything except action execution and reward is unchanged."""
    base = gym.make("MountainCar-v0")                       # reference env
    wrapped = make_env(True, sticky_seed=0)                 # modified env
    nostick = make_env(True, sticky_seed=0, stick_prob=0.0) # same wrapper but never sticky
    out: Dict[str, object] = {
        "observation_space_equal": base.observation_space == wrapped.observation_space,
        "action_space_equal": base.action_space == wrapped.action_space,
        "max_episode_steps": base.spec.max_episode_steps,
    }
    # Same seed + same action sequence -> identical trajectories when stick_prob = 0
    rng = np.random.default_rng(1)
    acts = rng.integers(0, 3, size=200)                     # one full random episode
    ob_b, info_b = base.reset(seed=7)
    ob_w, info_w = nostick.reset(seed=7)
    same_obs, same_flags, same_info = bool(np.array_equal(ob_b, ob_w)), True, info_b == info_w
    base_rewards_ok = True
    for a in acts:
        ob_b, r_b, te_b, tr_b, i_b = base.step(int(a))
        ob_w, r_w, te_w, tr_w, i_w = nostick.step(int(a))
        same_obs &= bool(np.array_equal(ob_b, ob_w))        # dynamics unchanged
        same_flags &= (te_b == te_w) and (tr_b == tr_w)     # termination unchanged
        same_info &= (i_b.keys() == i_w.keys())             # nothing added to info
        base_rewards_ok &= nostick.last_base_reward == r_b  # base reward is -1
        if te_b or tr_b:
            break
    # info keys of the real sticky env
    _, info_r = wrapped.reset(seed=3)
    _, _, _, _, info_s = wrapped.step(2)
    out.update({
        "dynamics_identical_when_not_stuck": same_obs,
        "termination_truncation_identical": same_flags,
        "info_identical_keys": bool(same_info),
        "info_keys_sticky_env": sorted(list(info_r.keys()) + list(info_s.keys())),
        "base_reward_minus_one": bool(base_rewards_ok),
        "a_prev_after_reset": None,
    })
    wrapped.reset(seed=4)
    out["a_prev_after_reset"] = wrapped.a_prev
    for e in (base, wrapped, nostick):
        e.close()
    return out


def replay_pdf_example() -> List[Dict[str, object]]:
    """Reproduce the 5-step example table of the assignment with scripted random numbers."""
    env = make_env(True)
    env.reset(seed=0)
    env.rng = ScriptedRNG([0.55, 0.11, 0.74, 0.05])        # draws only happen when a != a_prev
    rows = []
    for step, a in enumerate([2, 2, 0, 0, 1], start=1):
        a_prev = env.a_prev
        env.step(a)
        rows.append({
            "step": step, "a_prev": a_prev, "a": a,
            "r": "-" if env.last_r is None else env.last_r,
            "a_exec": env.last_executed,
        })
    env.close()
    return rows


def random_policy_audit(n_episodes: int = 2000, seed: int = 84, n_log_steps: int = 25) -> Dict[str, object]:
    """Run a uniform-random policy in the modified env and audit every single step."""
    env = make_env(True, sticky_seed=seed + 1)
    rng = np.random.default_rng(seed)                       # random policy RNG
    attempted = replaced = repeats = repeat_replaced = 0
    pair_counts: Counter = Counter()                        # (a, a_exec) -> number of steps
    pen_ok: Counter = Counter()                             # (a, a_exec) -> steps with correct penalty
    penalty_on_a1 = penalty_missing_push = 0
    r_draws: List[float] = []                               # all uniform numbers drawn
    step_log: List[Dict[str, object]] = []
    bonus_nonzero = 0
    total_steps = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            a = int(rng.integers(3))
            a_prev = env.a_prev
            obs, R, term, trunc, info = env.step(a)
            done = term or trunc
            total_steps += 1
            if a == a_prev:                                  # repeated action
                repeats += 1
                repeat_replaced += int(env.last_executed != a)
            else:                                            # attempted change
                attempted += 1
                replaced += int(env.last_replaced)
                r_draws.append(env.last_r)
            key = (a, env.last_executed)
            pair_counts[key] += 1
            expected_pen = 0.2 if a != NO_PUSH else 0.0
            implied_pen = env.last_base_reward + env.last_bonus - R   # recovered from the returned reward
            pen_ok[key] += int(abs(implied_pen - expected_pen) < 1e-9)
            penalty_on_a1 += int(a == NO_PUSH and implied_pen > 1e-9)
            penalty_missing_push += int(a != NO_PUSH and abs(implied_pen - 0.2) > 1e-9)
            bonus_nonzero += int(env.last_bonus != 0)
            if len(step_log) < n_log_steps:
                step_log.append({"ep": ep, "a_prev": a_prev, "a": a,
                                 "r": None if env.last_r is None else round(env.last_r, 3),
                                 "a_exec": env.last_executed, "R_base": env.last_base_reward,
                                 "penalty": round(implied_pen, 3), "B": env.last_bonus, "R": round(R, 3)})
    env.close()
    lo, hi = wilson_ci(replaced, attempted)
    stuck_draws = np.array(r_draws) < 0.2
    return {
        "episodes": n_episodes, "steps": total_steps,
        "attempted_changes": attempted, "replaced": replaced,
        "replacement_rate": replaced / attempted, "ci95": [lo, hi],
        "repeat_steps": repeats, "replaced_on_repeat": repeat_replaced,
        "uniform_draw_mean": float(np.mean(r_draws)), "draws_below_0.2": float(stuck_draws.mean()),
        "pair_counts": {f"a={k[0]},a_exec={k[1]}": v for k, v in sorted(pair_counts.items())},
        "penalty_correct": {f"a={k[0]},a_exec={k[1]}": pen_ok[k] for k in sorted(pair_counts)},
        "penalty_on_selected_1": penalty_on_a1, "penalty_missing_on_push": penalty_missing_push,
        "bonus_awarded_random": bonus_nonzero,
        "step_log": step_log,
    }


def velocity_policy(obs: np.ndarray) -> int:
    """Hand-coded energy-pumping policy: push in the direction of the velocity."""
    return 2 if obs[1] >= 0 else 0


def make_braking_policy(brake_from: float) -> Callable[[np.ndarray], int]:
    """Velocity policy that coasts/brakes once past `brake_from` if moving faster than 0.03."""
    def policy(obs: np.ndarray) -> int:
        """Push with velocity, but brake (push left) near the flag when too fast."""
        if obs[0] > brake_from and obs[1] > 0.03:   # close to the goal and fast
            return 0                                # brake
        return 2 if obs[1] >= 0 else 0
    return policy


def bonus_audit(n_episodes: int = 300, seed: int = 84) -> Dict[str, object]:
    """Use hand-coded policies to produce gentle arrivals, fast arrivals and truncations,
    and check that B = +20 appears exactly for gentle arrivals."""
    policies = {
        "velocity (push with v)": velocity_policy,
        "velocity + brake after x>0.0": make_braking_policy(0.0),
        "velocity + brake after x>0.3": make_braking_policy(0.3),
        "always no-push (1)": lambda obs: 1,
    }
    results = {}
    examples = []
    for name, pol in policies.items():
        env = make_env(True, sticky_seed=seed + 7)
        counts = Counter()
        vels = []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            while not done:
                obs, R, term, trunc, _ = env.step(pol(obs))
                done = term or trunc
            gentle = term and not trunc and obs[0] >= 0.5 and obs[1] <= 0.04
            if term:
                vels.append(float(obs[1]))
            kind = "gentle arrival" if gentle else ("fast arrival" if term else "truncated")
            counts[(kind, env.last_bonus)] += 1
            if len([e for e in examples if e["kind"] == kind]) < 2:
                examples.append({"policy": name, "kind": kind, "final_pos": round(float(obs[0]), 4),
                                 "final_vel": round(float(obs[1]), 4), "terminated": bool(term),
                                 "truncated": bool(trunc), "B": env.last_bonus, "last_R": round(R, 3)})
        env.close()
        results[name] = {
            "counts": {f"{k[0]} | B={k[1]:g}": v for k, v in sorted(counts.items())},
            "arrival_velocity_mean": float(np.mean(vels)) if vels else None,
            "arrivals": len(vels),
        }
    # global consistency: bonus iff gentle
    violations = 0
    for r in results.values():
        for k, v in r["counts"].items():
            kind, b = k.split(" | B=")
            violations += v * int((kind == "gentle arrival") != (float(b) == 20.0))
    return {"policies": results, "violations": violations, "examples": examples}


def non_markov_evidence(n_episodes: int = 1500, seed: int = 84, bins=(20, 20)) -> Dict[str, object]:
    """Show that, for the SAME observed state bin and SAME selected action, the next-state
    distribution depends on the hidden a_prev (so the observation is not Markov)."""
    env = make_env(True, sticky_seed=seed + 11)
    disc = UniformDiscretizer(bins)
    rng = np.random.default_rng(seed + 12)
    dv = {0: [], 2: []}                       # velocity change when a=2, keyed by hidden a_prev
    exec_push = {0: [], 2: []}
    same_prev, n_prev = 0, 0                  # how often the agent's own previous choice equals a_prev
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + 5000 + ep)
        done = False
        prev_selected = None                  # agent-side memory of its last selected action
        while not done:
            a = int(rng.integers(3))
            a_prev = env.a_prev
            if prev_selected is not None:     # compare remembered choice with hidden a_prev
                same_prev += int(prev_selected == a_prev)
                n_prev += 1
            prev_selected = a
            s = disc(obs)
            obs2, _, term, trunc, _ = env.step(a)
            if a == 2 and a_prev in (0, 2):
                dv[a_prev].append(float(obs2[1] - obs[1]))
                exec_push[a_prev].append(int(env.last_executed == 2))
            obs = obs2
            done = term or trunc
    env.close()
    return {
        "selected_a": 2,
        "mean_dv_given_aprev_2": float(np.mean(dv[2])),
        "mean_dv_given_aprev_0": float(np.mean(dv[0])),
        "p_exec_right_given_aprev_2": float(np.mean(exec_push[2])),
        "p_exec_right_given_aprev_0": float(np.mean(exec_push[0])),
        "n_aprev_2": len(dv[2]), "n_aprev_0": len(dv[0]),
        "p_prev_selected_equals_aprev": same_prev / max(1, n_prev),
    }

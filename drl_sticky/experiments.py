"""Experiment orchestration used by BOTH the master notebook and run_experiments.py.

Each function performs one block of the assignment and returns plain dicts, so the
notebook can display intermediate outputs and the final results.json is identical
whichever entry point is used.
"""

from __future__ import annotations

import os
import platform
import socket
import sys
from datetime import datetime
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd

from . import verify
from .config import Config
from .discretizer import UniformDiscretizer
from .evaluate import evaluate_greedy, policy_map, value_calibration
from .train import LOG_COLUMNS, train_agent
from .utils import moving_average

#: The four required agents (algorithm, environment).
RUNS: List[Tuple[str, str]] = [("qlearning", "original"), ("sarsa", "original"),
                               ("qlearning", "modified"), ("sarsa", "modified")]


def environment_info() -> Dict[str, object]:
    """Timestamp, host and library versions (execution evidence for the report)."""
    marker = os.path.exists("VLAB_RUN.txt")                     # created by the team in the lab
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gymnasium": gym.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cpu_count": os.cpu_count(),
        "vlab_marker_present": marker,
    }


def run_verification(quick: bool = False) -> Dict[str, object]:
    """Task (a): all verification experiments of the modified environment."""
    return {
        "spec": verify.spec_compliance(),                                  # unchanged components
        "pdf_example": verify.replay_pdf_example(),                        # example table
        "random_audit": verify.random_policy_audit(200 if quick else 2000),  # ~20% rule & penalty
        "bonus_audit": verify.bonus_audit(50 if quick else 300),           # +20 rule
        "non_markov": verify.non_markov_evidence(200 if quick else 1500),  # for discussion (e2)
    }


def train_main_runs(cfg: Config, val: np.ndarray, out_dir: str = "results",
                    progress: Optional[Callable[[str, int, Dict[str, float]], None]] = None,
                    progress_every: int = 500, runs: Sequence[Tuple[str, str]] = RUNS):
    """Tasks (b)/(c): train the requested agents with the SAME Config; save logs & Q-tables."""
    os.makedirs(f"{out_dir}/logs", exist_ok=True)
    os.makedirs(f"{out_dir}/qtables", exist_ok=True)
    logs, agents, timing = {}, {}, {}
    for algo, env in runs:
        key = f"{algo}_{env}"
        cb = (lambda ep, m, k=key: progress(k, ep, m)) if progress else None  # bind run name
        agent, log, secs = train_agent(algo, env == "modified", cfg, val, progress=cb,
                                       progress_every=progress_every)
        logs[key], agents[key], timing[key] = log, agent, secs
        pd.DataFrame({c: log[c] for c in LOG_COLUMNS}).to_csv(f"{out_dir}/logs/{key}.csv", index=False)
        np.save(f"{out_dir}/qtables/{key}.npy", agent.Q)                 # final Q-table
    return logs, agents, timing


def training_summary(log: Dict[str, np.ndarray], last: int = 500) -> Dict[str, object]:
    """Summary of the last `last` training episodes of one run."""
    v = np.asarray(log["arrival_velocity"][-last:], dtype=float)
    return {
        "last_return": float(np.mean(log["return"][-last:])),
        "last_shaped_return": float(np.mean(log["shaped_return"][-last:])),
        "last_success": float(np.mean(log["success"][-last:])),
        "last_gentle": float(np.mean(log["gentle"][-last:])),
        "last_pushes": float(np.mean(log["pushes"][-last:])),
        "last_length": float(np.mean(log["length"][-last:])),
        "last_arrival_velocity": float(np.nanmean(v)) if np.any(~np.isnan(v)) else None,
        "final_avg_max_q": float(log["avg_max_q"][-1]),
        "first_success_episode": int(np.argmax(log["success"] > 0)) + 1 if log["success"].any() else None,
        "first100_pushes": float(np.mean(log["pushes"][:100])),
        "replacement_rate_training": float(np.sum(log["replaced"]) / max(1, np.sum(log["attempted_changes"]))),
    }


def checkpoints(logs: Dict[str, Dict[str, np.ndarray]], every: int = 500) -> Dict[str, dict]:
    """100-episode moving averages sampled every `every` episodes (compact progress table)."""
    out = {}
    for k, lg in logs.items():
        n = len(lg["return"])
        ck = list(range(every - 1, n, every)) or [n - 1]
        out[k] = {
            "episode": [c + 1 for c in ck],
            "epsilon": [float(lg["epsilon"][c]) for c in ck],
            "ma_return": [float(moving_average(lg["return"])[c]) for c in ck],
            "ma_success": [float(moving_average(lg["success"])[c]) for c in ck],
            "ma_gentle": [float(moving_average(lg["gentle"])[c]) for c in ck],
            "ma_pushes": [float(moving_average(lg["pushes"])[c]) for c in ck],
            "avg_max_q": [float(lg["avg_max_q"][c]) for c in ck],
        }
    return out


def evaluate_all(agents, cfg: Config, val: np.ndarray, quick: bool = False) -> Dict[str, object]:
    """Task (d): greedy test table, cross-environment test, value calibration and policy maps."""
    rng = np.random.default_rng(cfg.seed)
    calib_states = val[rng.choice(len(val), size=60 if quick else 200, replace=False)]
    ev, ev_lists, cross, calib, maps, pm_stats = {}, {}, {}, {}, {}, {}
    for key, agent in agents.items():
        modified = key.endswith("modified")
        e = evaluate_greedy(agent, modified, cfg)                       # 100 fixed-seed episodes
        ev[key] = {k: v for k, v in e.items() if not isinstance(v, list)}
        ev_lists[key] = {"arrival_velocities": e["arrival_velocities"], "returns": e["returns"]}
        cross[key] = {k: v for k, v in evaluate_greedy(agent, True, cfg).items() if not isinstance(v, list)}
        calib[key] = value_calibration(agent, modified, cfg, calib_states, n_rollouts=2 if quick else 5)
        pm = policy_map(agent)
        maps[key] = pm
        vis = pm >= 0
        pm_stats[key] = {"visited_cells": int(vis.sum()), "total_cells": int(pm.size),
                         "frac_left": float(np.mean(pm[vis] == 0)),
                         "frac_none": float(np.mean(pm[vis] == 1)),
                         "frac_right": float(np.mean(pm[vis] == 2))}
    return {"evaluation": ev, "evaluation_lists": ev_lists, "cross_evaluation_modified_env": cross,
            "value_calibration": calib, "policy_maps": pm_stats, "_maps": maps}


def value_gap(logs) -> Dict[str, dict]:
    """Discussion (e1): Q-learning minus SARSA average predicted value, per environment and phase."""
    out = {}
    for env in ["original", "modified"]:
        g = np.asarray(logs[f"qlearning_{env}"]["avg_max_q"]) - np.asarray(logs[f"sarsa_{env}"]["avg_max_q"])
        n = len(g)
        out[env] = {"final": float(g[-1]), "mean_last500": float(np.mean(g[-500:])),
                    "mean_all": float(np.mean(g)), "mean_abs_all": float(np.mean(np.abs(g))),
                    "max": float(np.max(g)),
                    "first20pct": float(np.mean(g[: n // 5])),
                    "middle": float(np.mean(g[n // 5: int(0.7 * n)])),
                    "last5pct": float(np.mean(g[-max(1, n // 20):]))}
    return out


def _seed_job(job):
    """Worker for the multi-seed / ablation runs (module-level so multiprocessing can pickle it)."""
    algo, env, seed, overrides, val = job
    cfg = Config(seed=seed, **overrides)
    agent, log, secs = train_agent(algo, env == "modified", cfg, val)
    ev = evaluate_greedy(agent, env == "modified", cfg)
    return {"algo": algo, "env": env, "seed": seed, "overrides": overrides,
            "eval": {k: v for k, v in ev.items() if not isinstance(v, list)},
            "final_avg_max_q": float(log["avg_max_q"][-1]),
            "mean_avg_max_q_last500": float(np.mean(log["avg_max_q"][-500:])),
            "avg_max_q_curve": log["avg_max_q"][::10].tolist(),            # every 10th episode
            "last500_success": float(np.mean(log["success"][-500:])),
            "last500_return": float(np.mean(log["return"][-500:])),
            "seconds": secs}


def _mean_std(values) -> Dict[str, object]:
    """Mean, population std and raw values of a list of floats."""
    a = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(a)), "std": float(np.std(a)), "values": a.tolist()}


def run_robustness(cfg: Config, val: np.ndarray, seeds: Sequence[int],
                   ablation_bins: Sequence[Tuple[int, int]] = ((20, 20), (36, 28), (40, 40)),
                   n_ablation_seeds: int = 3, processes: Optional[int] = None) -> Dict[str, object]:
    """Repeat the 4 runs for several seeds and run the discretization ablation (in parallel)."""
    base = {"n_episodes": cfg.n_episodes, "alpha": cfg.alpha, "gamma": cfg.gamma,
            "eps_decay_frac": cfg.eps_decay_frac, "n_pos_bins": cfg.n_pos_bins, "n_vel_bins": cfg.n_vel_bins}
    jobs = [(a, e, s, dict(base), val) for s in seeds for a, e in RUNS]
    abl_seeds = list(seeds[:n_ablation_seeds])
    jobs += [(a, "modified", s, {**base, "n_pos_bins": b[0], "n_vel_bins": b[1], "_abl": 1}, val)
             for b in ablation_bins for s in abl_seeds for a in ["qlearning", "sarsa"]]
    # strip helper key before building Config
    clean = [(a, e, s, {k: v for k, v in o.items() if k != "_abl"}, v) for a, e, s, o, v in jobs]
    procs = processes or max(1, (os.cpu_count() or 2) - 1)
    with Pool(procs) as pool:
        out = pool.map(_seed_job, clean)
    n_seed_jobs = len(seeds) * len(RUNS)
    seed_rows, abl_rows = out[:n_seed_jobs], out[n_seed_jobs:]
    summary = {}
    for algo, env in RUNS:
        rows = [r for r in seed_rows if r["algo"] == algo and r["env"] == env]
        summary[f"{algo}_{env}"] = {m: _mean_std([r["eval"][m] for r in rows]) for m in
                                    ["mean_reward", "success_rate", "gentle_rate", "mean_length", "mean_pushes"]}
        summary[f"{algo}_{env}"]["avg_max_q_last500"] = _mean_std([r["mean_avg_max_q_last500"] for r in rows])
        summary[f"{algo}_{env}"]["train_last500_return"] = _mean_std([r["last500_return"] for r in rows])
        summary[f"{algo}_{env}"]["train_last500_success"] = _mean_std([r["last500_success"] for r in rows])
    gaps, gap_phase = {}, {}
    for env in ["original", "modified"]:
        per, phase = [], {"all": [], "abs_all": [], "first20pct": [], "middle": [], "last5pct": []}
        for s in seeds:
            q = next(r for r in seed_rows if r["algo"] == "qlearning" and r["env"] == env and r["seed"] == s)
            sa = next(r for r in seed_rows if r["algo"] == "sarsa" and r["env"] == env and r["seed"] == s)
            per.append(q["mean_avg_max_q_last500"] - sa["mean_avg_max_q_last500"])
            g = np.asarray(q["avg_max_q_curve"]) - np.asarray(sa["avg_max_q_curve"])  # gap curve
            n = len(g)
            phase["all"].append(float(np.mean(g)))
            phase["abs_all"].append(float(np.mean(np.abs(g))))
            phase["first20pct"].append(float(np.mean(g[: n // 5])))
            phase["middle"].append(float(np.mean(g[n // 5: int(0.7 * n)])))
            phase["last5pct"].append(float(np.mean(g[-max(1, n // 20):])))
        gaps[env] = _mean_std(per)
        gap_phase[env] = {k: _mean_std(v) for k, v in phase.items()}
    # paired comparison: is the gap larger in the modified env for each seed?
    gap_phase["modified_minus_original_all"] = _mean_std(
        np.array(gap_phase["modified"]["all"]["values"]) - np.array(gap_phase["original"]["all"]["values"]))
    ablation = {}
    for b in ablation_bins:
        name = f"{b[0]}x{b[1]}"
        ablation[name] = {}
        for a in ["qlearning", "sarsa"]:
            rows = [r for r in abl_rows if r["algo"] == a and r["overrides"]["n_pos_bins"] == b[0]
                    and r["overrides"]["n_vel_bins"] == b[1]]
            ablation[name][a] = _mean_std([r["eval"]["mean_reward"] for r in rows])
            ablation[name][a]["success"] = float(np.mean([r["eval"]["success_rate"] for r in rows]))
    for r in seed_rows:
        r.pop("avg_max_q_curve", None)                                     # keep results.json small
    return {"seeds": list(seeds), "summary": summary, "value_gap_last500": gaps, "value_gap_phases": gap_phase,
            "per_seed": seed_rows, "bins_ablation": {"seeds": abl_seeds, "results": ablation}}


def make_figures(logs, evals, robustness, cfg: Config, fig_dir: str = "figures") -> Dict[str, str]:
    """Create every PNG used in the report; returns {name: path}."""
    from .plotting import (plot_arrival_speed, plot_bins_ablation, plot_metric_panels, plot_policy_maps,
                           plot_rates, plot_seed_robustness, plot_value_overlay)
    os.makedirs(fig_dir, exist_ok=True)
    disc = UniformDiscretizer((cfg.n_pos_bins, cfg.n_vel_bins))
    p = {k: f"{fig_dir}/{k}.png" for k in ["fig1_reward", "fig2_value", "fig3_rates", "fig4_pushes",
                                            "fig5_policy_maps", "fig6_arrival_speed", "fig7_seeds",
                                            "fig8_bins_ablation", "fig9_length"]}
    plot_metric_panels(logs, "return", "Episode reward", "Episode reward vs training episode", p["fig1_reward"])
    plot_value_overlay(logs, p["fig2_value"])
    plot_rates(logs, p["fig3_rates"])
    plot_metric_panels(logs, "pushes", "Push actions (0 or 2) selected", "Push actions selected per episode",
                       p["fig4_pushes"])
    plot_policy_maps(evals["_maps"], disc, p["fig5_policy_maps"])
    plot_arrival_speed(evals["evaluation"], logs, p["fig6_arrival_speed"])
    plot_seed_robustness(robustness["summary"], p["fig7_seeds"])
    plot_bins_ablation(robustness["bins_ablation"]["results"], p["fig8_bins_ablation"])
    plot_metric_panels(logs, "length", "Episode length (steps)", "Episode length vs training episode",
                       p["fig9_length"])
    return p

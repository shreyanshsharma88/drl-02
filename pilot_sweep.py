"""Hyper-parameter pilot (run BEFORE the main experiments).

Selects the discretization, learning rate, exploration-decay length and episode
budget used in the main study. To avoid tuning on the reported results, the pilot
uses different master seeds (184, 284, 384) and different evaluation seeds
(50000+ / 60000+) from the final experiments (seed 84, evaluation seeds 10000+).
Both algorithms are scored on the MODIFIED environment; the selection criterion
is the mean greedy-evaluation reward averaged over algorithms and seeds.

Usage:  python pilot_sweep.py          -> results/pilot.json
"""

from __future__ import annotations

import itertools
import os
from multiprocessing import Pool

import numpy as np

from drl_sticky.config import Config
from drl_sticky.evaluate import evaluate_greedy
from drl_sticky.train import sample_validation_states
from drl_sticky.train import train_agent
from drl_sticky.utils import save_json

BINS = [(20, 20), (36, 28), (40, 40)]        # candidate discretizations
ALPHAS = [0.05, 0.10, 0.20]                  # candidate learning rates
DECAY_FRACS = [0.5, 0.7]                     # epsilon reaches its floor at this fraction of training
EPISODES = [6000, 10000]                     # candidate budgets (>= 5000)
PILOT_SEEDS = [184, 284, 384]                # pilot master seeds (not 84)
ALGOS = ["qlearning", "sarsa"]


def run_one(job):
    """Train + evaluate one (config, algorithm, seed) combination; return a summary row."""
    (npos, nvel), alpha, frac, n_ep, seed, algo = job
    cfg = Config(n_pos_bins=npos, n_vel_bins=nvel, alpha=alpha, eps_decay_frac=frac,
                 n_episodes=n_ep, seed=seed)
    val = sample_validation_states(seed=cfg.seeds()["validation"])          # per-seed validation set
    agent, log, secs = train_agent(algo, True, cfg, val)                    # modified env
    ev = evaluate_greedy(agent, True, cfg, n_episodes=100, seed_base=50_000, sticky_base=60_000)
    return {
        "bins": f"{npos}x{nvel}", "alpha": alpha, "decay_frac": frac, "episodes": n_ep,
        "seed": seed, "algo": algo, "eval_reward": ev["mean_reward"],
        "eval_success": ev["success_rate"], "eval_gentle": ev["gentle_rate"],
        "train_last500_reward": float(np.mean(log["return"][-500:])), "seconds": secs,
    }


def main() -> None:
    """Run the whole grid in parallel and write results/pilot.json with a ranking."""
    jobs = list(itertools.product(BINS, ALPHAS, DECAY_FRACS, EPISODES, PILOT_SEEDS, ALGOS))
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:                    # parallel runs
        rows = pool.map(run_one, jobs)
    # aggregate over seeds and algorithms
    keyf = lambda r: (r["bins"], r["alpha"], r["decay_frac"], r["episodes"])
    groups = {}
    for r in rows:
        groups.setdefault(keyf(r), []).append(r)
    ranking = []
    for k, rs in groups.items():
        ranking.append({
            "bins": k[0], "alpha": k[1], "decay_frac": k[2], "episodes": k[3],
            "mean_eval_reward": float(np.mean([r["eval_reward"] for r in rs])),
            "mean_eval_success": float(np.mean([r["eval_success"] for r in rs])),
            "mean_eval_gentle": float(np.mean([r["eval_gentle"] for r in rs])),
            "q_eval_reward": float(np.mean([r["eval_reward"] for r in rs if r["algo"] == "qlearning"])),
            "sarsa_eval_reward": float(np.mean([r["eval_reward"] for r in rs if r["algo"] == "sarsa"])),
        })
    ranking.sort(key=lambda d: -d["mean_eval_reward"])
    os.makedirs("results", exist_ok=True)
    save_json({"rows": rows, "ranking": ranking, "pilot_seeds": PILOT_SEEDS}, "results/pilot.json")
    for d in ranking[:12]:
        print(d)


if __name__ == "__main__":
    main()

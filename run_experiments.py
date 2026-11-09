"""Run EVERY experiment of the assignment from the command line and save all numbers/figures.

The master notebook runs exactly the same functions (drl_sticky.experiments) cell by cell;
this script is the non-interactive equivalent.

Outputs
  results/results.json          all reported numbers
  results/logs/<run>.csv        per-episode log of each of the 4 main runs (every iteration)
  results/qtables/<run>.npy     final Q-tables
  results/validation_states.npy fixed validation set (sampled once, before training)
  figures/*.png                 all plots

Usage:  python run_experiments.py [--episodes N] [--seeds 84 85 86 87 88] [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from drl_sticky import experiments as X
from drl_sticky.config import Config
from drl_sticky.train import sample_validation_states
from drl_sticky.utils import save_json


def main() -> None:
    """Parse arguments and run blocks (a) -> (e) in order."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=Config().n_episodes)
    ap.add_argument("--seeds", type=int, nargs="+", default=[84, 85, 86, 87, 88])
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke tests")
    args = ap.parse_args()
    cfg = Config(n_episodes=300 if args.quick else args.episodes)
    os.makedirs("results", exist_ok=True)
    t0 = time.time()
    res = {"meta": X.environment_info(), "config": cfg.to_dict()}
    print("[a] verification", flush=True)
    res["verification"] = X.run_verification(args.quick)
    val = sample_validation_states(seed=cfg.seeds()["validation"])      # fixed before training
    np.save("results/validation_states.npy", val)
    print("[b/c] training 4 agents", flush=True)
    logs, agents, timing = X.train_main_runs(cfg, val)
    res["training"] = {k: {**X.training_summary(logs[k]), "seconds": timing[k]} for k in logs}
    res["checkpoints"] = X.checkpoints(logs)
    print("[d] evaluation", flush=True)
    ev = X.evaluate_all(agents, cfg, val, args.quick)
    res.update({k: v for k, v in ev.items() if not k.startswith("_")})
    res["value_gap_seed84"] = X.value_gap(logs)
    print("[e] multi-seed robustness + ablation", flush=True)
    rob = X.run_robustness(cfg, val, args.seeds)
    res["multi_seed"] = {k: v for k, v in rob.items() if k != "bins_ablation"}
    res["bins_ablation"] = rob["bins_ablation"]
    X.make_figures(logs, ev, rob, cfg)
    res["meta"]["total_seconds"] = time.time() - t0
    save_json(res, "results/results.json")
    print(f"done in {res['meta']['total_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

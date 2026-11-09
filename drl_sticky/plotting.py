"""All figures for task (d) and the discussion (matplotlib only)."""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")                     # headless backend (works in the VM and in CI)
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch      # noqa: E402

from .discretizer import UniformDiscretizer  # noqa: E402
from .utils import moving_average         # noqa: E402

# Colour = algorithm (validated categorical slots 1 and 2), panel = environment.
ALGO_COLORS = {"qlearning": "#2a78d6", "sarsa": "#eb6834"}
ALGO_LABELS = {"qlearning": "Q-learning", "sarsa": "SARSA"}
ENV_LABELS = {"original": "Original MountainCar-v0", "modified": "Modified (sticky + reward)"}
ENV_STYLES = {"original": "--", "modified": "-"}
# Policy-map colours: Left / None / Right; unvisited cells drawn light grey.
ACTION_CMAP = ListedColormap(["#e9e9e6", "#2a78d6", "#b9b8b2", "#eb6834"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.edgecolor": "#8a8984", "axes.titlesize": 10, "legend.frameon": False,
})


def _run_key(algo: str, env: str) -> str:
    """Canonical run identifier, e.g. 'qlearning_modified'."""
    return f"{algo}_{env}"


def plot_metric_panels(logs: Dict[str, Dict[str, np.ndarray]], metric: str, ylabel: str,
                       title: str, path: str, window: int = 100, raw: bool = True,
                       ylim: Optional[tuple] = None, scale: float = 1.0) -> None:
    """Two panels (original | modified); each shows Q-learning vs SARSA.

    Thin translucent line = raw per-episode value, thick line = 100-episode moving average.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    for ax, env in zip(axes, ["original", "modified"]):
        for algo in ["qlearning", "sarsa"]:
            y = np.asarray(logs[_run_key(algo, env)][metric], dtype=float) * scale
            x = np.arange(1, len(y) + 1)
            if raw:                                               # raw episodes (faint)
                ax.plot(x, y, color=ALGO_COLORS[algo], alpha=0.12, linewidth=0.5)
            ma = moving_average(y, window) if window > 1 else y   # trailing average
            ax.plot(x, ma, color=ALGO_COLORS[algo], linewidth=2,
                    label=f"{ALGO_LABELS[algo]}" + (f" (MA{window})" if window > 1 else ""))
        ax.set_title(ENV_LABELS[env])
        ax.set_xlabel("Training episode")
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="best")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_value_overlay(logs, path: str) -> None:
    """Average max_a Q(s,a) on the fixed validation set: all four agents on one axis,
    plus the Q-learning minus SARSA gap for each environment."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    for algo in ["qlearning", "sarsa"]:
        for env in ["original", "modified"]:
            y = logs[_run_key(algo, env)]["avg_max_q"]
            axes[0].plot(np.arange(1, len(y) + 1), y, color=ALGO_COLORS[algo],
                         linestyle=ENV_STYLES[env], linewidth=1.8,
                         label=f"{ALGO_LABELS[algo]} - {env}")
    axes[0].set_title("Average predicted value  mean_s max_a Q(s,a)")
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Value")
    axes[0].legend(fontsize=8)
    for env, col in [("original", "#52514e"), ("modified", "#6250d6")]:
        gap = np.asarray(logs[_run_key("qlearning", env)]["avg_max_q"]) - np.asarray(
            logs[_run_key("sarsa", env)]["avg_max_q"])
        axes[1].plot(np.arange(1, len(gap) + 1), gap, color=col, linewidth=1.8,
                     linestyle=ENV_STYLES[env], label=f"{env} env")
    axes[1].axhline(0, color="#8a8984", linewidth=0.8)
    axes[1].set_title("Value gap  V_Q-learning - V_SARSA")
    axes[1].set_xlabel("Training episode")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_rates(logs, path: str, window: int = 100) -> None:
    """Success rate and gentle-arrival rate (moving average over previous 100 episodes)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    for ax, env in zip(axes, ["original", "modified"]):
        for algo in ["qlearning", "sarsa"]:
            lg = logs[_run_key(algo, env)]
            x = np.arange(1, len(lg["success"]) + 1)
            ax.plot(x, 100 * moving_average(lg["success"], window), color=ALGO_COLORS[algo],
                    linewidth=2, label=f"{ALGO_LABELS[algo]} success")
            ax.plot(x, 100 * moving_average(lg["gentle"], window), color=ALGO_COLORS[algo],
                    linewidth=1.4, linestyle=":", label=f"{ALGO_LABELS[algo]} gentle arrival")
        ax.set_title(ENV_LABELS[env])
        ax.set_xlabel("Training episode")
        ax.set_ylim(-2, 102)
    axes[0].set_ylabel("Rate over previous 100 episodes (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Success rate (solid) and gentle-arrival rate (dotted)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_policy_maps(maps: Dict[str, np.ndarray], disc: UniformDiscretizer, path: str) -> None:
    """2x2 grid of greedy-action heatmaps over position x velocity."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True, sharey=True)
    pe, ve = disc.edges(0), disc.edges(1)
    for i, env in enumerate(["original", "modified"]):
        for j, algo in enumerate(["qlearning", "sarsa"]):
            ax = axes[i, j]
            pm = maps[_run_key(algo, env)]                    # (n_pos, n_vel), -1 = unvisited
            ax.pcolormesh(pe, ve, (pm + 1).T, cmap=ACTION_CMAP, vmin=0, vmax=3,
                          edgecolors="white", linewidth=0.2)
            ax.axvline(0.5, color="#0b0b0b", linewidth=1)           # flag
            ax.axhline(0.04, color="#0b0b0b", linewidth=0.8, linestyle="--")  # gentle speed
            ax.set_title(f"{ALGO_LABELS[algo]} - {env}")
            ax.grid(False)
            if i == 1:
                ax.set_xlabel("Position")
            if j == 0:
                ax.set_ylabel("Velocity")
    handles = [Patch(color=ACTION_CMAP(k), label=l) for k, l in
               [(1, "0: push left"), (2, "1: no push"), (3, "2: push right"), (0, "never visited")]]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Greedy policy maps (solid line: flag x=0.5, dashed: v=0.04)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_arrival_speed(eval_results: Dict[str, dict], logs, path: str, last: int = 1000) -> None:
    """Histogram of arrival velocities (last `last` training episodes) per agent."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    bins = np.linspace(0, 0.07, 29)                         # 0.0025-wide bins up to v_max
    for ax, env in zip(axes, ["original", "modified"]):
        for algo in ["qlearning", "sarsa"]:
            v = np.asarray(logs[_run_key(algo, env)]["arrival_velocity"][-last:], dtype=float)
            v = v[~np.isnan(v)]
            ax.hist(v, bins=bins, histtype="step", linewidth=1.8, color=ALGO_COLORS[algo],
                    label=f"{ALGO_LABELS[algo]} (n={len(v)}, mean={v.mean() if len(v) else float('nan'):.4f})")
        ax.axvline(0.04, color="#0b0b0b", linestyle="--", linewidth=1)
        ax.set_title(ENV_LABELS[env])
        ax.set_xlabel("Velocity at the flag")
    axes[0].set_ylabel(f"Arrivals (last {last} training episodes)")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Arrival-speed distribution (dashed: gentle threshold 0.04)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_seed_robustness(summary: Dict[str, dict], path: str) -> None:
    """Dot plot over seeds: every seed is one dot, the black bar is the mean (+- std whisker)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0))
    keys = ["qlearning_original", "sarsa_original", "qlearning_modified", "sarsa_modified"]
    labels = ["Q-learning\noriginal", "SARSA\noriginal", "Q-learning\nmodified", "SARSA\nmodified"]
    for ax, (metric, name) in zip(axes, [("mean_reward", "Greedy test reward per seed"),
                                           ("success_rate", "Greedy success rate per seed")]):
        for i, k in enumerate(keys):
            vals = np.asarray(summary[k][metric]["values"], dtype=float)       # one value per seed
            jitter = np.linspace(-0.12, 0.12, len(vals))                        # spread overlapping dots
            ax.scatter(i + jitter, vals, s=34, color=ALGO_COLORS[k.split("_")[0]], zorder=3,
                       edgecolor="white", linewidth=0.8)
            m, sd = summary[k][metric]["mean"], summary[k][metric]["std"]
            ax.errorbar(i, m, yerr=sd, fmt="_", color="#0b0b0b", markersize=22, capsize=4, zorder=4)
        ax.set_xticks(range(len(keys)), labels)
        ax.set_title(name + " (bar = mean +- std)")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_bins_ablation(ablation: Dict[str, dict], path: str) -> None:
    """Greedy test reward vs discretization for both algorithms (modified env)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    names = list(ablation.keys())
    x = np.arange(len(names))
    for k, algo in enumerate(["qlearning", "sarsa"]):
        m = [ablation[n][algo]["mean"] for n in names]
        s = [ablation[n][algo]["std"] for n in names]
        ax.bar(x + (k - 0.5) * 0.35, m, 0.33, yerr=s, color=ALGO_COLORS[algo], capsize=3,
               label=ALGO_LABELS[algo])
    ax.set_xticks(x, names)
    ax.set_xlabel("Bins (position x velocity)")
    ax.set_ylabel("Greedy test reward")
    ax.set_title("Discretization ablation (modified env)")
    ax.grid(axis="x", visible=False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

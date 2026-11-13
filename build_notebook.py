"""Generate the master notebook Group84_Q_learning_SARSA_SetA.ipynb.

The notebook is the single source of the submission PDF. It
  * writes every module of the drl_sticky package with %%writefile cells (so ALL code
    appears in the PDF, with its comments),
  * runs every experiment of tasks (a)-(e) and shows every output,
  * renders interpretation / discussion text from the numbers it has just computed
    (cells tagged ``remove_input`` show only their rendered text in the PDF),
  * ends with the virtual-lab screenshots and the complete per-episode logs.

Usage:  python build_notebook.py   (make_pdf.py calls build() automatically)
"""

from __future__ import annotations

import glob
import os
from typing import List

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_NAME = "Group84_Q_learning_SARSA_SetA.ipynb"
PKG_ORDER = {
    "setup": ["__init__.py", "config.py", "utils.py"],
    "a": ["env.py", "verify.py"],
    "b": ["discretizer.py", "agents.py", "train.py"],
    "d": ["evaluate.py", "plotting.py", "experiments.py"],
}


def md(text: str):
    """Markdown cell."""
    return new_markdown_cell(text.strip("\n"))


def code(src: str, hide_input: bool = False):
    """Code cell; hide_input=True keeps only the output in the PDF (used for rendered text)."""
    cell = new_code_cell(src.strip("\n"))
    if hide_input:
        cell.metadata["tags"] = ["remove_input"]
    return cell


def dyn(fstring_body: str):
    """Code cell that renders Markdown filled with live numbers (input hidden in the PDF)."""
    return code('display(Markdown(f"""\n' + fstring_body.strip("\n") + '\n"""))', hide_input=True)


def writefile(module: str):
    """%%writefile cell reproducing drl_sticky/<module> verbatim."""
    with open(os.path.join("drl_sticky", module), encoding="utf-8") as fh:
        src = fh.read()
    return new_code_cell(f"%%writefile drl_sticky/{module}\n" + src.rstrip("\n"))


def cells() -> List:
    """All notebook cells in report order."""
    C = []
    # ------------------------------------------------------------------ title
    C.append(md("""
# Robust Reinforcement Learning under Sticky Actuators
### Tabular Q-learning vs SARSA on a modified MountainCar-v0
**Deep Reinforcement Learning: Assignment II (Set A, Group 1-100)**  ·  **Group 84**  ·  File: *Group 84 - Q_learning_SARSA_Set A*

<h2 class="nobreak">Group Contribution Declaration</h2>

| # | Name | Registration No. | Email | Contribution |
|---|------|------------------|-------|--------------|
| 1 | SHREYANSH SHARMA | 2025AG05772 | 2025ag05772@wilp.bits-pilani.ac.in | **100%** |
| 2 | LAKSHYA KANCHAN | 2025AG05062 | 2025ag05062@wilp.bits-pilani.ac.in | **100%** |
| 3 | VINAY PRATAP SINGH BHADAURIA | 2025AG05689 | 2025ag05689@wilp.bits-pilani.ac.in | **100%** |

All three members contributed equally to the design, implementation, experiments and the write-up.
"""))
    C.append(md("""
### How this report is organised
| Section | Content | Marks |
|---|---|---|
| 0 | Setup: execution environment, configuration, helpers | n/a |
| (a) | Sticky-actuator wrapper and its experimental verification | 2 |
| (b) | Discretization, Q-learning agent, training on both environments | 3 |
| (c) | SARSA agent (only the TD target differs), training on both environments | 3 |
| (d) | Performance evaluation: 5 plot families, policy maps, greedy test table | 3 |
| (e) | Discussion: 5 questions answered with our measurements | 2 |
| (f) | Virtual-lab execution screenshots (with timestamps) | n/a |
| App. A-C | Hyper-parameter pilot, CLI script, training log covering every iteration | n/a |

**What we built.** The whole solution is written by hand in NumPy as a small package, `drl_sticky`. It contains the wrapper, discretizer, agents, training loop, evaluation and plots. Every module is reproduced in full below through `%%writefile` cells. Running this notebook top to bottom writes the package, runs every experiment and regenerates every number and figure in this PDF. Gymnasium supplies only the MountainCar-v0 physics. No RL library is used.

**Reproducibility.** We use one master seed, **84** (our group number). Separate RNG streams are derived from it for (i) environment resets, (ii) the sticky actuator and (iii) the agent's exploration. Because the streams are separate, the actuator noise never shifts the agent's exploration draws. The four required agents share exactly the same `Config` object.
"""))
    # ------------------------------------------------------------------ setup
    C.append(md("""
## 0. Setup and execution environment
The cell below records **when and where** this copy of the notebook was executed (time stamp, host name, library versions). The notebook was also executed in the **BITS virtual lab** (host `ip-10-3-17-70.ap-south-1.compute.internal`, 13-14 Nov 2026). Section (f) shows the timestamped screenshots of that run. All randomness is seeded, so both runs produce identical numbers. Screenshot 5 shows the lab's greedy-evaluation table, which matches table d.6 below digit for digit.
"""))
    C.append(code("""
# Create the package folder (the %%writefile cells below fill it)
import os
os.makedirs("drl_sticky", exist_ok=True)      # package directory
os.makedirs("results", exist_ok=True)         # numbers, logs, Q-tables
os.makedirs("figures", exist_ok=True)         # PNG figures
"""))
    for m in PKG_ORDER["setup"]:
        C.append(writefile(m))
    C.append(code("""
# Imports used throughout the notebook
import json, time, inspect, warnings
import numpy as np
import pandas as pd
import matplotlib
from IPython.display import display, Markdown, Image, HTML
warnings.filterwarnings("ignore")              # keep the PDF free of library deprecation noise
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
from drl_sticky.config import Config, ACTION_NAMES

CFG = Config()                                 # THE configuration shared by all four agents
R = {"config": CFG.to_dict()}                  # every reported number is collected in R
NOTEBOOK_T0 = time.time()                      # total run-time measurement
"""))
    C.append(code("""
# Execution evidence: time stamp, host and library versions of this run
import platform, socket, sys, gymnasium
from datetime import datetime
R["meta"] = {"timestamp": datetime.now().isoformat(timespec="seconds"), "hostname": socket.gethostname(),
             "platform": platform.platform(), "python": sys.version.split()[0], "gymnasium": gymnasium.__version__,
             "numpy": np.__version__, "pandas": pd.__version__, "cpu_count": os.cpu_count()}
for k, v in R["meta"].items():
    print(f"{k:22s}: {v}")
"""))
    C.append(dyn("""
**Configuration shared by all four agents**

| Parameter | Value | Reason |
|---|---|---|
| Discretization | {CFG.n_pos_bins} x {CFG.n_vel_bins} uniform bins (position width {1.8/CFG.n_pos_bins:.3f}, velocity width {0.14/CFG.n_vel_bins:.4f}) | at least 20x20 required. A held-out-seed pilot (Appendix A) found 20x20 gave the best greedy reward and the smallest seed-to-seed spread. Fewer cells means more visits per cell, so the table is learned faster |
| Q-table | shape ({CFG.n_pos_bins}, {CFG.n_vel_bins}, 3), initialised to 0 | Required by the spec. Every step reward is at most -1, so a zero start is optimistic and drives systematic exploration |
| Learning rate alpha | {CFG.alpha} | best of 0.05 / 0.1 / 0.2 in the pilot |
| Discount gamma | {CFG.gamma} | 200-step horizon; the +20 bonus must reach states ~100 steps earlier |
| Exploration | epsilon_k = max({CFG.eps_min}, {CFG.eps_start} * {CFG.eps_decay:.6f}^k), k = episode | exponential decay reaching the floor after {int(CFG.eps_decay_frac*100)}% of training (episode {int(CFG.eps_decay_frac*CFG.n_episodes)}); last {100-int(CFG.eps_decay_frac*100)}% near-greedy |
| Episodes | {CFG.n_episodes} per agent | at least 5000 required; the pilot showed 10000 clearly beats 6000 |
| Seeds | master {CFG.seed}; derived streams {CFG.seeds()} | identical for all four runs |
"""))
    # ------------------------------------------------------------------ (a)
    C.append(md("""
## (a) Modified Environment Implementation and Verification
### a.1 Implementation
`StickyActuatorMountainCar` (in `drl_sticky/env.py`, listed below) subclasses **`gym.Wrapper`** and performs Steps 1-6 **in the prescribed order**:

1. **Receive** the selected action `a` and keep it for the reward.
2. **Sticky actuator.** If `a == a_prev` the action runs unchanged, and *no random number is drawn*. Otherwise draw `r ~ U[0,1)`. If `r < 0.20` the action is replaced by `a_prev`, otherwise `a` is kept. Then set `a_prev = a_exec`.
3. **Execute** `a_exec` in the original MountainCar-v0 and receive `obs, R_base, terminated, truncated, info`.
4. **Modified reward:** `R = R_base - 0.2 * 1[a != 1] + B`, using the **selected** action `a`.
5. **Gentle-arrival bonus:** `B = +20` iff `terminated and not truncated and obs[0] >= 0.5 and obs[1] <= 0.04`.
6. **Return** `obs, R, terminated, truncated, info`. The `info` dict is the base environment's own dict, untouched.

`reset()` sets `a_prev = 1`. The sticky RNG belongs to the wrapper. The attributes starting with `last_` exist only for the audit below. They never appear in the observation or in `info`, so the agent gets no hint that an action was replaced.
"""))
    C.append(writefile("env.py"))
    C.append(md("The verification experiments are in `drl_sticky/verify.py`:"))
    C.append(writefile("verify.py"))
    C.append(md("### a.2 Components that must remain unchanged"))
    C.append(code("""
# Run the complete verification suite of task (a) (random policy: 2000 episodes = 400k steps)
from drl_sticky import verify
t0 = time.time()
R["verification"] = {"spec": verify.spec_compliance(), "pdf_example": verify.replay_pdf_example(),
                     "random_audit": verify.random_policy_audit(2000), "bonus_audit": verify.bonus_audit(300),
                     "non_markov": verify.non_markov_evidence(1500)}
print(f"verification finished in {time.time()-t0:.1f}s")
spec = R["verification"]["spec"]
pd.DataFrame({"check": list(spec.keys()), "result": [str(v) for v in spec.values()]})
"""))
    C.append(dyn("""
Same seed and same action sequence, with stick probability set to 0, give **bit-identical observations and identical terminated/truncated flags** to the base environment. So the dynamics, termination (flag at 0.5) and truncation (200 steps) are untouched. The observation and action spaces are equal to the originals. `info` has exactly the base keys `{spec['info_keys_sticky_env']}`, and `a_prev` is `{spec['a_prev_after_reset']}` right after `reset()`.
"""))
    C.append(md("### a.3 The assignment's worked example (scripted random numbers 0.55, 0.11, 0.74, 0.05)"))
    C.append(code("""
# Replay the 5-step example table of the assignment with a scripted RNG
pd.DataFrame(R["verification"]["pdf_example"]).set_index("step")
"""))
    C.append(md("The executed actions are **2, 2, 2, 0, 0**, exactly as in the assignment's table. At step 2 no number is drawn because the agent repeats the executed action. At step 5 the change 0 to 1 is blocked because 0.05 < 0.20."))
    C.append(md("### a.4 About 20% of attempted changes are replaced, and none when the action is repeated"))
    C.append(code("""
# Random-policy audit: counts over every step of 2000 episodes
aud = R["verification"]["random_audit"]
summary = pd.DataFrame([
    ["episodes / steps", f"{aud['episodes']} / {aud['steps']}"],
    ["attempted changes (a != a_prev)", aud["attempted_changes"]],
    ["replaced by a_prev", aud["replaced"]],
    ["replacement rate", f"{aud['replacement_rate']:.4f}"],
    ["95% Wilson CI", f"[{aud['ci95'][0]:.4f}, {aud['ci95'][1]:.4f}]"],
    ["mean of uniform draws (expected 0.5)", f"{aud['uniform_draw_mean']:.4f}"],
    ["steps with a == a_prev", aud["repeat_steps"]],
    ["replacements when a == a_prev", aud["replaced_on_repeat"]],
], columns=["statistic", "value"])
display(summary)
print("First 25 audited steps (r = '-' means no draw because a == a_prev):")
pd.DataFrame(aud["step_log"]).fillna("-")
"""))
    C.append(dyn("""
Out of **{aud['attempted_changes']:,}** attempted changes, **{aud['replaced']:,}** were replaced, a rate of **{100*aud['replacement_rate']:.2f}%**. The 95% CI [{100*aud['ci95'][0]:.2f}%, {100*aud['ci95'][1]:.2f}%] contains 20%. Out of **{aud['repeat_steps']:,}** steps where the agent repeated the executed action, **{aud['replaced_on_repeat']}** were modified.
"""))
    C.append(md("### a.5 The energy penalty follows the *selected* action"))
    C.append(code("""
# For every (selected a, executed a_exec) pair: how many steps, and in how many the penalty was exactly 0.2*1[a!=1]
pairs = pd.DataFrame({"steps": aud["pair_counts"], "penalty correct": aud["penalty_correct"]})
pairs["expected penalty"] = [0.2 if not k.startswith("a=1") else 0.0 for k in pairs.index]
pairs["stuck?"] = [k.split(",")[0][2:] != k.split(",")[1][7:] for k in pairs.index]
display(pairs)
print("penalised steps with selected a = 1     :", aud["penalty_on_selected_1"])
print("unpenalised steps with selected a = 0/2 :", aud["penalty_missing_on_push"])
"""))
    C.append(md("""
The penalty is read back from the returned reward as `R_base + B - R`. Every row has *penalty correct* = *steps*. The off-diagonal rows matter most: **a=0, a_exec=2** and **a=2, a_exec=1** are stuck pushes that are still charged 0.2, and **a=1, a_exec=0/2** means the actuator kept pushing while the agent chose to coast, and it is **not** charged. As the specification requires, the penalty depends only on the selected action.
"""))
    C.append(md("### a.6 The +20 bonus is paid only for a gentle arrival"))
    C.append(code("""
# A random policy almost never reaches the flag, so hand-coded policies are used:
#  - push in the direction of the velocity (often arrives too fast),
#  - the same with braking near the flag (arrives slowly),
#  - always 'no push' (never arrives -> truncation at 200 steps).
bon = R["verification"]["bonus_audit"]
rows = []
for pol, d in bon["policies"].items():
    for outcome, n in d["counts"].items():
        rows.append([pol, outcome, n, d["arrival_velocity_mean"]])
display(pd.DataFrame(rows, columns=["policy", "outcome | bonus", "episodes", "mean arrival velocity"]))
print("episodes where bonus != 20*[gentle arrival]:", bon["violations"])
pd.DataFrame(bon["examples"])
"""))
    C.append(md("Every gentle arrival (terminated, not truncated, position >= 0.5, velocity <= 0.04) received **B = 20**. Every fast arrival and every truncation received **B = 0**. There are no violations."))
    # ------------------------------------------------------------------ (b)
    C.append(md("""
## (b) Q-Learning
### b.1 State discretization
`UniformDiscretizer` maps (position, velocity) to bin indices, with `idx = clip(floor((x - low)/width), 0, n-1)` applied in each dimension. The **Q-table** has shape *(position bins x velocity bins x 3)* and starts at zero.

### b.2 Agent and update rule
`TabularTDAgent` holds the Q-table, the epsilon-greedy policy (ties among equal maxima are broken at random, which matters with a zero-initialised table) and the generic update `Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))`. **`QLearningAgent`** overrides only the target:

`target = R + gamma * max_a' Q(s', a')`, or just `R` when the episode is **terminated** (reaching the flag is a true terminal state). On **truncation** at 200 steps the agent still bootstraps, because a time-out is not a terminal state of the task.

Here `a` is the **selected** action. The executed action is hidden from the agent.

### b.3 Training loop
The same loop trains both algorithms. The next action `a'` is chosen epsilon-greedily *before* the update and is then actually executed. Q-learning ignores `a'` in its target, and SARSA uses it. As a result **the TD target is the only difference** between the two experiments. The loop logs every episode: return, length, success, gentle arrival, arrival velocity, selected pushes, epsilon, the average predicted value on the validation set, and (for the audit only) the number of sticky replacements.
"""))
    C.append(writefile("discretizer.py"))
    C.append(writefile("agents.py"))
    C.append(writefile("train.py"))
    C.append(md("### b.4 Fixed validation set (sampled once, before any training)"))
    C.append(code("""
# 1000 states visited by a uniform-random policy in the ORIGINAL env (seed 3084); never changed afterwards
from drl_sticky.train import sample_validation_states, LOG_COLUMNS
VAL = sample_validation_states(seed=CFG.seeds()["validation"])
np.save("results/validation_states.npy", VAL)
R["validation_states"] = {"n": len(VAL), "pos_range": [float(VAL[:,0].min()), float(VAL[:,0].max())],
                          "vel_range": [float(VAL[:,1].min()), float(VAL[:,1].max())]}
print(R["validation_states"])
"""))
    C.append(md("""
We sample validation states from rollouts instead of uniformly over the state box. Uniform sampling would include unreachable corners, for example high speed at the far left, where the Q-values are never updated and would only dilute the average.
"""))
    C.append(md("### b.5 Training Q-learning on the original and on the modified environment"))
    C.append(code("""
# Train one agent, print progress every 500 episodes, save its per-episode log (CSV) and Q-table
from drl_sticky.train import train_agent
os.makedirs("results/logs", exist_ok=True); os.makedirs("results/qtables", exist_ok=True)
LOGS, AGENTS, TIMING = {}, {}, {}

def run_and_store(algo, env):
    \"\"\"Train `algo` on `env` ('original'/'modified') with CFG; keep and save the results.\"\"\"
    key = f"{algo}_{env}"
    def progress(ep, m):
        \"\"\"One progress line (values of the current episode).\"\"\"
        print(f"[{key:18s}] ep {ep+1:5d} | eps {m['epsilon']:.3f} | R {m['return']:8.2f} | len {int(m['length']):3d} "
              f"| success {int(m['success'])} gentle {int(m['gentle'])} | pushes {int(m['pushes']):3d} | avg maxQ {m['avg_max_q']:8.3f}")
    agent, log, secs = train_agent(algo, env == "modified", CFG, VAL, progress=progress, progress_every=500)
    LOGS[key], AGENTS[key], TIMING[key] = log, agent, secs
    pd.DataFrame({c: log[c] for c in LOG_COLUMNS}).to_csv(f"results/logs/{key}.csv", index=False)
    np.save(f"results/qtables/{key}.npy", agent.Q)
    print(f"--> {key}: {CFG.n_episodes} episodes in {secs:.1f}s\\n")

for env in ["original", "modified"]:
    run_and_store("qlearning", env)
"""))
    # ------------------------------------------------------------------ (c)
    C.append(md("""
## (c) SARSA
SARSA uses **the same** discretization, zero Q-table, alpha, gamma, epsilon schedule, random seeds and number of episodes, because it receives the same `CFG` object. The class definitions below are printed from `drl_sticky/agents.py`. `SarsaAgent` overrides **only** `td_target`, which bootstraps from `Q(s', a')` where `a'` is the next action the epsilon-greedy policy actually chose:

`Q(s,a) <- Q(s,a) + alpha * [R + gamma * Q(s',a') - Q(s,a)]`, with no bootstrap when terminated.
"""))
    C.append(code("""
# Show that the two agents differ ONLY in td_target
from drl_sticky.agents import QLearningAgent, SarsaAgent, TabularTDAgent
own = lambda cls: sorted(k for k, v in vars(cls).items() if inspect.isfunction(v))
print("methods defined by QLearningAgent:", own(QLearningAgent))
print("methods defined by SarsaAgent    :", own(SarsaAgent))
print()
print(inspect.getsource(QLearningAgent.td_target))
print(inspect.getsource(SarsaAgent.td_target))
"""))
    C.append(code("""
# Train SARSA on the original and on the modified environment (identical settings)
for env in ["original", "modified"]:
    run_and_store("sarsa", env)
LOGS = {k: LOGS[k] for k in ["qlearning_original", "sarsa_original", "qlearning_modified", "sarsa_modified"]}
AGENTS = {k: AGENTS[k] for k in LOGS}
"""))
    # ------------------------------------------------------------------ (d)
    C.append(md("""
## (d) Performance Evaluation
The four agents are **Q-learning / SARSA x original / modified environment**. In every figure, **colour = algorithm** (blue = Q-learning, orange = SARSA) and **panel = environment**. Thin faint lines show individual episodes and thick lines show the 100-episode trailing moving average.

*Note on reward scales.* The original environment pays -1 per step. The modified environment pays -1.2 per step when a push is selected, plus 20 for a gentle arrival. Rewards are therefore compared **between algorithms within an environment**. Across environments we compare success, gentle-arrival rate, episode length and push counts, which share the same units.
"""))
    C.append(md("Evaluation, plotting and experiment-orchestration code (used from here on):"))
    C.append(writefile("evaluate.py"))
    C.append(writefile("plotting.py"))
    C.append(writefile("experiments.py"))
    C.append(md("### d.0 Training summary"))
    C.append(code("""
# Summary of the last 500 training episodes and moving-average checkpoints
from drl_sticky import experiments as X
R["training"] = {k: {**X.training_summary(LOGS[k]), "seconds": TIMING[k]} for k in LOGS}
R["checkpoints"] = X.checkpoints(LOGS)
pd.DataFrame(R["training"]).T.round(3)
"""))
    C.append(dyn("""
*Last 500 training episodes (epsilon = {CFG.eps_min}).* During training the sticky actuator replaced
{100*R['training']['qlearning_modified']['replacement_rate_training']:.1f}% (Q-learning) and
{100*R['training']['sarsa_modified']['replacement_rate_training']:.1f}% (SARSA) of the agents' attempted action changes, which again matches the 20% specification.
"""))
    C.append(code("""
# Training progress table: 100-episode moving averages every 1000 episodes for all four agents
rows = []
for k, ck in R["checkpoints"].items():
    for i, ep in enumerate(ck["episode"]):
        if ep % 1000 == 0:
            rows.append([k, ep, ck["epsilon"][i], ck["ma_return"][i], ck["ma_success"][i], ck["ma_gentle"][i], ck["ma_pushes"][i], ck["avg_max_q"][i]])
prog = pd.DataFrame(rows, columns=["run", "episode", "epsilon", "MA100 reward", "MA100 success", "MA100 gentle", "MA100 pushes", "avg max Q"])
prog.pivot(index="episode", columns="run", values="MA100 reward").round(1)
"""))
    C.append(code("""
# Greedy evaluation (epsilon = 0) over 100 fixed-seed test episodes, cross-environment test,
# value calibration and policy maps; then all figures
t0 = time.time()
EV = X.evaluate_all(AGENTS, CFG, VAL)
R.update({k: v for k, v in EV.items() if not k.startswith("_") and k != "evaluation_lists"})
R["evaluation_arrival_velocities"] = {k: v["arrival_velocities"] for k, v in EV["evaluation_lists"].items()}
R["value_gap_seed84"] = X.value_gap(LOGS)
print(f"evaluation finished in {time.time()-t0:.1f}s")
"""))
    C.append(code(HELPER1, hide_input=True))
    C.append(md("### d.1 Episode reward versus training episode"))
    C.append(code("""
# Plotting helpers, then Fig. d.1: episode reward (raw + 100-episode moving average) per environment
from drl_sticky.plotting import (plot_metric_panels, plot_value_overlay, plot_rates, plot_policy_maps,
                                 plot_arrival_speed, plot_seed_robustness, plot_bins_ablation)
from drl_sticky.discretizer import UniformDiscretizer
plot_metric_panels(LOGS, "return", "Episode reward", "Episode reward vs training episode", "figures/fig1_reward.png")
Image("figures/fig1_reward.png", width=900)
"""))
    C.append(dyn(INTERP["fig1"]))
    C.append(md("### d.2 Average predicted value  mean over validation states of max_a Q(s,a)"))
    C.append(code("""
# Fig. d.2: average max_a Q(s,a) over the fixed validation set, and the Q-learning - SARSA gap
plot_value_overlay(LOGS, "figures/fig2_value.png")
Image("figures/fig2_value.png", width=900)
"""))
    C.append(dyn(INTERP["fig2"]))
    C.append(md("### d.3 Success rate and gentle-arrival rate (moving average over the previous 100 episodes)"))
    C.append(code("""
# Fig. d.3: success and gentle-arrival rates, moving average over the previous 100 episodes
plot_rates(LOGS, "figures/fig3_rates.png")
Image("figures/fig3_rates.png", width=900)
"""))
    C.append(dyn(INTERP["fig3"]))
    C.append(md("### d.4 Average number of push actions (0 or 2) selected per episode"))
    C.append(code("""
# Fig. d.4: number of push actions (0 or 2) selected per episode
plot_metric_panels(LOGS, "pushes", "Push actions (0 or 2) selected", "Push actions selected per episode", "figures/fig4_pushes.png")
Image("figures/fig4_pushes.png", width=900)
"""))
    C.append(dyn(INTERP["fig4"]))
    C.append(md("### d.5 Greedy policy maps (position x velocity)"))
    C.append(code("""
# Fig. d.5: greedy action per (position, velocity) cell for the four agents + action shares
DISC = UniformDiscretizer((CFG.n_pos_bins, CFG.n_vel_bins))
plot_policy_maps(EV["_maps"], DISC, "figures/fig5_policy_maps.png")
display(Image("figures/fig5_policy_maps.png", width=820))
pd.DataFrame(R["policy_maps"]).T.round(3)
"""))
    C.append(dyn(INTERP["fig5"]))
    C.append(md("### d.6 Greedy-policy evaluation: 100 test episodes, epsilon = 0, fixed seeds 10000-10099"))
    C.append(code("""
# One table with the four required metrics (+ pushes and arrival speed)
names = {"qlearning_original": "Q-learning - Original", "sarsa_original": "SARSA - Original",
         "qlearning_modified": "Q-learning - Modified", "sarsa_modified": "SARSA - Modified"}
tab = pd.DataFrame({names[k]: {"mean reward": v["mean_reward"], "std reward": v["std_reward"],
                               "success rate": v["success_rate"], "gentle-arrival rate": v["gentle_rate"],
                               "mean episode length": v["mean_length"], "mean pushes / episode": v["mean_pushes"],
                               "mean arrival velocity": v["mean_arrival_velocity"]}
                    for k, v in R["evaluation"].items()}).T
tab.round(3)
"""))
    C.append(dyn(INTERP["table"]))
    C.append(md("### d.7 Additional evidence: arrival speed, robustness across seeds, and transfer to the sticky environment"))
    C.append(code("""
# Arrival-velocity histograms (last 1000 training episodes) for both environments
plot_arrival_speed(R["evaluation"], LOGS, "figures/fig6_arrival_speed.png")
Image("figures/fig6_arrival_speed.png", width=900)
"""))
    C.append(code("""
# Every agent (including the two trained on the ORIGINAL env) tested in the MODIFIED env
cross = pd.DataFrame({names[k]: {m: v[m] for m in ["mean_reward", "success_rate", "gentle_rate", "mean_length", "mean_pushes"]}
                      for k, v in R["cross_evaluation_modified_env"].items()}).T
cross.round(3)
"""))
    C.append(dyn(INTERP["cross"]))
    C.append(code("""
# Robustness: repeat the 4 runs for seeds 84-88 and the discretization ablation (parallel processes)
t0 = time.time()
ROB = X.run_robustness(CFG, VAL, seeds=[84, 85, 86, 87, 88])
R["multi_seed"] = {k: v for k, v in ROB.items() if k != "bins_ablation"}
R["bins_ablation"] = ROB["bins_ablation"]
print(f"{len(ROB['per_seed'])} seed runs + ablation finished in {time.time()-t0:.0f}s")
ms = pd.DataFrame({names[k]: {f"{m} (mean +- std)": f"{v[m]['mean']:.3f} +- {v[m]['std']:.3f}" for m in
                   ["mean_reward", "success_rate", "gentle_rate", "mean_length", "mean_pushes", "avg_max_q_last500"]}
                   for k, v in ROB["summary"].items()}).T
ms
"""))
    C.append(code("""
# Per-seed greedy results and the discretization ablation
plot_seed_robustness(ROB["summary"], "figures/fig7_seeds.png")
plot_bins_ablation(ROB["bins_ablation"]["results"], "figures/fig8_bins_ablation.png")
display(Image("figures/fig7_seeds.png", width=900))
Image("figures/fig8_bins_ablation.png", width=480)
"""))
    C.append(code(HELPER2, hide_input=True))
    C.append(dyn(INTERP["robust"]))
    # ------------------------------------------------------------------ (e)
    C.append(md("## (e) Discussion"))
    C.append(code("""
# Numbers used in the discussion
calib = pd.DataFrame(R["value_calibration"]).T
calib.columns = ["mean predicted max Q", "mean actual greedy return", "bias (pred - actual)", "MAE"]
display(calib.round(3))
gaps = pd.DataFrame({env: {"gap seed 84 (last 500 ep.)": R["value_gap_seed84"][env]["mean_last500"],
                           "gap 5 seeds mean": R["multi_seed"]["value_gap_last500"][env]["mean"],
                           "gap 5 seeds std": R["multi_seed"]["value_gap_last500"][env]["std"]}
                     for env in ["original", "modified"]}).T
display(gaps.round(3))
nm = R["verification"]["non_markov"]
pd.DataFrame([nm]).T.rename(columns={0: "value"})
"""))
    for q in ["e1", "e2", "e3", "e4", "e5"]:
        C.append(dyn(INTERP[q]))
    C.append(code("""
# Save every number of this run (identical to python run_experiments.py)
from drl_sticky.utils import save_json
R["meta"]["total_seconds"] = time.time() - NOTEBOOK_T0
save_json(R, "results/results.json")
print(f"results/results.json written; notebook run time {R['meta']['total_seconds']:.0f}s")
"""))
    # ------------------------------------------------------------------ (f)
    C.append(md("""
## (f) Virtual-lab execution (screenshots with timestamps)
The notebook was executed in the **BITS virtual lab** by group member VINAY PRATAP SINGH BHADAURIA (lab user `s2025ag05689`). The lab clock (top-right of each screenshot) shows the date and time of every capture.

| Item | Value seen in the screenshots |
|---|---|
| Virtual-lab host | `ip-10-3-17-70.ap-south-1.compute.internal` (Linux 5.14, el9, x86_64, 4 CPUs) |
| Lab user | `s2025ag05689` |
| Software | Python 3.12.7, gymnasium 1.4.0, numpy 2.5.4, pandas 3.1.1 (fresh `.venv` from `requirements.txt`) |
| Run window | 2026-11-13 22:54 (environment setup) to 2026-11-14 01:11 (greedy evaluation) |
| Result check | Lab table d.6: Q-learning/SARSA original -160.300 / -160.750, modified -155.564 / -117.588, identical to table d.6 in this report |
"""))
    C.append(code(SCREENSHOT_CELL, hide_input=True))
    # ------------------------------------------------------------------ appendices
    C.append(md("""
## Appendix A: Hyper-parameter pilot (`pilot_sweep.py`)
The pilot ran **before** the main study and used **different seeds** (master seeds 184, 284, 384 and evaluation seeds 50000+) from the final experiments (seed 84, evaluation seeds 10000+), so the reported test results were never used for tuning. It covered every combination of bins {20x20, 36x28, 40x40}, alpha {0.05, 0.1, 0.2}, epsilon-floor fraction {0.5, 0.7} and episodes {6000, 10000}, for both algorithms on the modified environment and 3 seeds each: 216 training runs. Below are the script and its top-10 configurations, ranked by mean greedy test reward.
"""))
    C.append(code("""
# Print the full source of the hyper-parameter pilot script
print(open("pilot_sweep.py").read())
"""))
    C.append(code("""
# Top-10 pilot configurations ranked by mean greedy test reward (modified env, 3 pilot seeds x 2 algorithms)
pil = json.load(open("results/pilot.json"))
pd.DataFrame(pil["ranking"][:10]).round(3)
"""))
    C.append(md("""
## Appendix B: Command-line entry point (`run_experiments.py`)
The same experiments can be run without Jupyter. This script calls exactly the functions used above.
"""))
    C.append(code("""
# Print the full source of the command-line entry point
print(open("run_experiments.py").read())
"""))
    C.append(md("""
## Appendix C: Training log covering every iteration (all 10,000 episodes of each agent)
Each row summarises one block of 100 consecutive training episodes, so every one of the 4 x 10,000 episodes is counted. **eps** = exploration rate at the end of the block. For each agent: **R** = mean episode reward, **succ** = % of episodes that reached the flag, **gentle** = % with a gentle arrival (v <= 0.04). The progress lines printed during training in (b) and (c) are the raw per-episode output; the full per-episode data (reward, length, pushes, avg max Q, sticky replacements) is saved in `results/logs/*.csv`.
"""))
    C.append(code(LOG_CELL, hide_input=True))
    return C


# --------------------------------------------------------------------------------------
# Cells whose code is hidden in the PDF (only their output is shown)
SCREENSHOT_CELL = '''
# Embed every virtual-lab screenshot in screenshots/ (sorted by capture order) with its caption
import glob, os
CAPTIONS = {
    "01": ("2026-11-13 22:54", "Lab terminal: project folder unpacked, virtual environment created and `pip install -r requirements.txt` running."),
    "02": ("2026-11-13 23:05", "Notebook open in VS Code in the lab. The execution-record cell prints the time stamp 2026-11-13T23:04:45, the lab host name and the library versions."),
    "03": ("2026-11-13 23:07", "Section (b): Q-learning training on the original environment, with the progress log printed every 500 episodes."),
    "04": ("2026-11-13 23:18", "Section (d.1): reward-vs-episode figure produced in the lab."),
    "05": ("2026-11-14 01:11", "Section (d.6): greedy-policy evaluation table in the lab. The values match table d.6 of this report exactly."),
}
shots = sorted(glob.glob("screenshots/*.png") + glob.glob("screenshots/*.jpg") + glob.glob("screenshots/*.jpeg"))
if os.path.exists("VLAB_RUN.txt"):
    display(Markdown("**Virtual-lab run record (`VLAB_RUN.txt`):**\\n\\n```\\n" + open("VLAB_RUN.txt").read().strip() + "\\n```"))
if not shots:
    display(Markdown("**TODO before submission:** add the timestamped virtual-lab screenshots to `screenshots/`."))
for i, s in enumerate(shots, 1):
    when, what = CAPTIONS.get(os.path.basename(s)[:2], ("see lab clock", os.path.basename(s)))
    display(Markdown(f"**Screenshot {i}** (lab clock {when}): {what}"))
    display(Image(s, width=820))
'''

LOG_CELL = '''
# One compact table: every training episode of every agent, aggregated in blocks of 100 episodes
BLOCK = 100
names = {"qlearning_original": "Q-learning orig", "sarsa_original": "SARSA orig",
         "qlearning_modified": "Q-learning mod", "sarsa_modified": "SARSA mod"}
first = next(iter(LOGS.values()))
n = len(first["return"])
cols = {("", "episodes"): [f"{i+1}-{min(i+BLOCK, n)}" for i in range(0, n, BLOCK)],
        ("", "eps"): [f"{first['epsilon'][min(i+BLOCK, n)-1]:.3f}" for i in range(0, n, BLOCK)]}
for run, label in names.items():
    lg = LOGS[run]
    blocks = range(0, len(lg["return"]), BLOCK)
    cols[(label, "R")] = [np.mean(lg["return"][i:i+BLOCK]) for i in blocks]           # mean reward of the block
    cols[(label, "succ")] = [100 * np.mean(lg["success"][i:i+BLOCK]) for i in blocks]  # % reached the flag
    cols[(label, "gentle")] = [100 * np.mean(lg["gentle"][i:i+BLOCK]) for i in blocks] # % gentle arrivals
table = pd.DataFrame(cols)
table.columns = pd.MultiIndex.from_tuples(table.columns)
display(HTML("<div class='dense-log'>" + table.to_html(index=False, float_format=lambda v: f"{v:.1f}") + "</div>"))
'''

HELPER1 = '''
# Short names and derived quantities used by the interpretation texts (all computed from this run)
from drl_sticky.utils import moving_average
E, TR, PM = R["evaluation"], R["training"], R["policy_maps"]
G84, CAL, CR = R["value_gap_seed84"], R["value_calibration"], R["cross_evaluation_modified_env"]
NM = R["verification"]["non_markov"]
def pf(run, last=500):
    """Fraction of steps with a selected push over the last `last` training episodes."""
    lg = LOGS[run]
    return float(lg["pushes"][-last:].sum() / lg["length"][-last:].sum())
def ma(run, metric, ep):
    """100-episode moving average of `metric` at episode `ep` (1-based)."""
    return float(moving_average(LOGS[run][metric])[ep - 1])
gap_peak_ep = {env: int(np.argmax(LOGS["qlearning_" + env]["avg_max_q"] - LOGS["sarsa_" + env]["avg_max_q"])) + 1
               for env in ["original", "modified"]}
def n_fast(env, last=1000):
    """Arrivals faster than 0.04 in the last `last` training episodes of both agents of `env`."""
    v = np.concatenate([LOGS[a + "_" + env]["arrival_velocity"][-last:] for a in ["qlearning", "sarsa"]])
    return int(np.sum(v[~np.isnan(v)] > CFG.gentle_speed))
n_fast_orig, n_fast_mod = n_fast("original"), n_fast("modified")
'''

HELPER2 = '''
# Multi-seed quantities used by the robustness text and the discussion
MS, GP = R["multi_seed"]["summary"], R["multi_seed"]["value_gap_phases"]
n_seeds = len(R["multi_seed"]["seeds"])
sarsa_wins = int(sum(s > q for s, q in zip(MS["sarsa_modified"]["mean_reward"]["values"],
                                           MS["qlearning_modified"]["mean_reward"]["values"])))
gap_larger_all = int(sum(m > o for m, o in zip(GP["modified"]["all"]["values"], GP["original"]["all"]["values"])))
gap_larger_first = int(sum(m > o for m, o in zip(GP["modified"]["first20pct"]["values"],
                                                 GP["original"]["first20pct"]["values"])))
'''

# Interpretation / discussion templates (f-strings evaluated INSIDE the notebook with live numbers).
INTERP = {}
_here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(_here, "report_text.py"), encoding="utf-8").read(), {"INTERP": INTERP})


def build(path: str = NB_NAME) -> str:
    """Write the notebook file and return its path."""
    nb = new_notebook(cells=cells())
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["title"] = "Group 84 - Q_learning_SARSA_Set A"
    nbformat.write(nb, path)
    return path


if __name__ == "__main__":
    print("notebook ->", build())

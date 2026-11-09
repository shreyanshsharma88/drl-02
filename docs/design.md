# Architecture and algorithms

```
drl_sticky/
  config.py       Config dataclass: bins, α, γ, ε schedule, episodes, seeds (one object shared by all runs)
  env.py          StickyActuatorMountainCar(gym.Wrapper) + make_env(modified)
  discretizer.py  UniformDiscretizer: (x, v) -> (i, j)
  agents.py       TabularTDAgent (Q-table, ε-greedy, update) -> QLearningAgent / SarsaAgent (td_target only)
  train.py        train_agent(): SARSA-style loop for both algorithms, per-episode metrics
  evaluate.py     greedy evaluation (fixed seeds), policy maps, Monte-Carlo value calibration
  verify.py       task (a) audit experiments + non-Markov measurement
  plotting.py     all figures (matplotlib, Agg backend)
  experiments.py  orchestration shared by the notebook and run_experiments.py (incl. multiprocessing seeds/ablation)
  utils.py        moving average, Wilson CI, JSON helpers
build_notebook.py + report_text.py  -> Group84_Q_learning_SARSA_SetA.ipynb
make_pdf.py      execute notebook -> HTML (lab template, code of text cells hidden) -> headless Chrome -> PDF
run_experiments.py  CLI equivalent of the notebook
pilot_sweep.py   held-out-seed hyper-parameter pilot (216 runs)
package.py       backup zip + test from unzipped copy
```

## Wrapper (Steps 1-6)
```
a      = int(action)                                    # step 1
if a == a_prev: a_exec = a                              # step 2 (no RNG draw)
else: r = U[0,1); a_exec = a_prev if r < 0.2 else a
obs, R_base, term, trunc, info = base.step(a_exec)      # step 3
a_prev = a_exec
B = 20 if term and not trunc and x >= 0.5 and v <= 0.04 else 0   # step 5
R = R_base - 0.2*[a != 1] + B                           # step 4
return obs, R, term, trunc, info                        # step 6 (info untouched)
```
The sticky RNG belongs to the wrapper (its own seed) and is independent of the agent's RNG.

## Discretization
`idx_d = clip(floor((x_d - low_d) / w_d), 0, n_d - 1)`, `w_d = (high_d - low_d)/n_d`, 20 × 20 bins over [-1.2, 0.6] × [-0.07, 0.07].

## TD control
`Q(s,a) ← Q(s,a) + α (target − Q(s,a))`
* Q-learning: `target = R + γ max_a' Q(s',a') · (1 − terminated)`
* SARSA: `target = R + γ Q(s',a') · (1 − terminated)`, a' = next ε-greedy action, actually executed next.
Truncation (200 steps) still bootstraps. The loop picks a' before the update for both algorithms, so the TD target is the only difference.

ε schedule: `ε_k = max(0.01, 1.0 · d^k)`, `d = 0.01^(1/(0.7 N))`, so the floor is reached at 70 % of N = 10 000 episodes.

## Seeds
Master 84 → env reset 84, sticky RNG 1084, agent RNG 2084, validation-state sampling 3084. Test episodes use reset seeds 10000-10099 and sticky seeds 20000-20099. The pilot uses master seeds 184/284/384 and evaluation seeds 50000+.

## Metrics
Per episode: return, shaped return, length, success (terminated), gentle (terminated ∧ v ≤ 0.04), arrival velocity, selected pushes, ε, mean max_a Q over 1000 fixed validation states, replacements and attempted changes (audit only).

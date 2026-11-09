# PLAN — Group 84 · DRL Assignment II (Set A) · Robust RL under Sticky Actuators

Deliverable: **one PDF** `Group 84 - Q_learning_SARSA_Set A.pdf` with all code + all outputs, executed in the BITS virtual lab, timestamped VM screenshots attached, contribution declaration on page 1. Deadline 14 Nov 2026; first version by 12 Nov 2026.

## 1. Environment & inputs
| Item | Value |
|---|---|
| Local Python | 3.14.7 (venv `.venv/`) — code written for Python ≥ 3.8 (no `match`, `from __future__ import annotations`) so it runs on the lab |
| Gymnasium | 1.4.0 locally; code uses only the ≥ 0.26 API (`reset → (obs, info)`, 5-tuple `step`) |
| Other | numpy 2.5, matplotlib 3.11, pandas 3.1, nbconvert, pytest |
| Input | `MountainCar-v0` (no dataset). obs Box([-1.2,-0.07],[0.6,0.07]) float32, Discrete(3), 200-step TimeLimit |
| Speed | base env ≈ 4.5 µs/step → one 5 000-episode run ≈ 1 M steps ≈ 15–30 s; full experiment set < 10 min |

## 2. Design choices (our group's, justified)
| Choice | Value | Why |
|---|---|---|
| Discretization | ~~36 × 28~~ → **20 × 20** uniform bins (changed after pilot, see Status) | ≥ 20×20 required. Width 0.05 makes **x = 0.5 (flag) an exact bin edge** (1.7/0.05 = 34); width 0.005 makes **v = 0.04 (gentle-arrival threshold) an exact bin edge** ((0.04+0.07)/0.005 = 22). Bins therefore never mix "arrived/not arrived" or "gentle/fast" states, which matters because the bonus depends on exactly these thresholds. 3 024 Q-entries: fine enough to resolve, coarse enough to be visited in 5 000+ episodes. Pilot compares 20×20 / 36×28 / 50×50 (ablation in report). |
| Q init | zeros, shape (36, 28, 3) | spec; since every reward ≤ −1 except the bonus, zero init is optimistic → drives systematic exploration |
| α | 0.10 (constant) | pilot winner among 0.05/0.1/0.2 |
| γ | 0.99 | horizon 200 steps; the +20 bonus must propagate back ~100+ steps |
| ε schedule | exponential: ε_k = max(0.01, 1.0 · d^k), d chosen so ε reaches 0.01 at 70 % of training (d = 0.01^(1/(0.7·N))) | stated exactly in report; last 30 % near-greedy so final metrics reflect the learned policy |
| Episodes | N = **10 000** (≥ 5 000) | pilot: 10 000 clearly better than 6 000 |
| Seeds | master seed **84** (group ID); env reset seed, sticky-RNG seed and agent-RNG seed derived from it; identical across all 4 runs | reproducibility |
| Validation states | 1 000 states collected **once before training** from random-policy rollouts in the base env (seed 84), never changed | representative of reachable states (uniform sampling would include unreachable corners) |
| Robustness | repeat the 4 configurations for seeds 84–88 (5 seeds) → mean ± std | answers (e) with evidence, not one lucky seed |

## 3. Rubric → deliverable map
| Rubric item | Marks | Earned by (notebook section / code / output) |
|---|---|---|
| (a) Modified env + verification | 2 | `drl_sticky/env.py::StickyActuatorMountainCar`; §A of notebook: spec-compliance checks (spaces, info keys), PDF 5-step example replay, random-policy stats (attempted changes, replacements, rate ≈ 20 % + 95 % CI, 0 replacements when a = a_prev, penalty counts by (a, a_exec) pairs), hand-coded velocity policy bonus table (gentle → +20, fast → 0, truncation → 0) |
| (b) Q-learning | 3 | `drl_sticky/agents.py::QLearningAgent`, `discretizer.py`, `train.py`; §B: hyper-parameter table, justification, training on original + modified, per-episode logs |
| (c) SARSA | 3 | `SarsaAgent` (subclass; overrides only `td_target`); §C: same settings, both envs, per-episode logs, diff showing only TD target differs |
| (d) Performance evaluation | 3 | §D: 5 plot families (reward + MA100, avg max-Q on validation set, success & gentle rate MA100, pushes/episode, 4 greedy-policy heatmaps) each with interpretation; greedy evaluation table (100 fixed-seed episodes): mean reward, success, gentle-arrival, mean length (+ arrival speed, pushes) |
| (e) Discussion | 2 | §E: five answers, each citing numbers from `results.json` (value gap, push counts, arrival-speed stats, eval table, seed robustness) + limitation & improvement |
| Comments rule | −1 if missing | docstring on every function/class + inline comments on every operation; pytest compliance test enforces docstrings |
| Contribution declaration | required | first cell / first page of PDF |
| VM execution + timestamped screenshots | required | `screenshots/` embedded in §F; environment-info cell prints timestamp/host/versions; claim only if `VLAB_RUN.txt` exists |
| Format / naming | required | `make_pdf.py` → `Group 84 - Q_learning_SARSA_Set A.pdf` |

## 4. No-RL-library rule
Only gymnasium (environment), numpy (arrays), matplotlib/pandas (plots/tables). Wrapper, discretizer, ε-greedy, Q-learning, SARSA, training loop, metrics, moving averages, evaluation all hand-written. pytest checks no `stable_baselines3`, `rllib`, `tianshou`, `torch`, etc. are imported.

## 5. Layout
```
drl_sticky/            # library package, no notebook/UI code
  __init__.py
  config.py            # Config dataclass: bins, alpha, gamma, eps schedule, episodes, seeds
  env.py               # StickyActuatorMountainCar(gym.Wrapper) + make_env(modified)
  discretizer.py       # UniformDiscretizer
  agents.py            # TabularTDAgent (base), QLearningAgent, SarsaAgent
  train.py             # train_agent(): episode loop, per-episode metrics, validation-value tracking
  evaluate.py          # greedy evaluation over fixed seeds; policy-map extraction
  verify.py            # experiments for task (a)
  plotting.py          # all figures
  utils.py             # moving average, seeding, wilson CI, json helpers
run_experiments.py     # runs everything, writes results/results.json, results/logs/*.csv, results/qtables/*.npy, figures/*.png
notebook/Group84_Q_learning_SARSA_SetA.ipynb   # master notebook (built by build_notebook.py)
build_notebook.py      # generates the notebook cells (keeps it reproducible)
make_pdf.py            # execute notebook → HTML → Chrome headless → PDF
package.py             # source backup zip + unzip test
tests/                 # pytest
docs/                  # traceability, design, user guide, results, testing
screenshots/           # VM screenshots (manual)
```

## 6. Build steps & algorithms
1. **Wrapper** `step(a)`:
   1. `a_sel = int(a)` (stored for reward).
   2. if `a_sel != a_prev`: `r = rng.random()`; `a_exec = a_prev if r < 0.20 else a_sel`; else `a_exec = a_sel` (no RNG draw).
   3. `obs, R_base, term, trunc, info = env.step(a_exec)`; `a_prev = a_exec`.
   4. `B = 20 if term and not trunc and obs[0] >= 0.5 and obs[1] <= 0.04 else 0`.
   5. `R = R_base − 0.2·1[a_sel ≠ 1] + B`; return `obs, R, term, trunc, info` (info untouched).
   `reset()` sets `a_prev = 1`. Sticky RNG = `np.random.default_rng(seed)` owned by wrapper, re-seeded when `reset(seed=…)` is given. Private diagnostics (`_last_exec`, counters) live as attributes for verification only — never in obs/info.
2. **Discretizer**: `idx_d = clip(floor((x_d − low_d)/w_d), 0, n_d − 1)`.
3. **Agents**: shared `TabularTDAgent` with `act(s)` (ε-greedy, random tie-break among maxima), `update(s,a,r,s',a',terminated)`:
   `δ = target − Q[s,a]`, `Q[s,a] += α δ`.
   - Q-learning: `target = r + γ · max_a' Q[s',a']·(1 − terminated)`
   - SARSA: `target = r + γ · Q[s',a']·(1 − terminated)`, `a'` = the ε-greedy action that **is executed next step**.
   Truncation still bootstraps. Only `td_target` differs between classes.
4. **Training loop** (identical for both; SARSA-style loop: choose a′ before update, Q-learning ignores a′ in target but uses it as next action — makes the two runs differ only in the target). Logged per episode: return, length, success (term), gentle (B>0), pushes selected, arrival velocity (if success), ε, mean max_a Q over validation states, and replacement count (diagnostic only, from wrapper attribute).
5. **Evaluation**: ε = 0, 100 episodes, reset seeds 10 000–10 099 (also sticky-RNG seeded per episode) → mean reward, success rate, gentle rate, mean length, mean |arrival v|, mean pushes. Also cross-evaluation: original-trained agents evaluated in the modified env (robustness extra).
6. **Policy map**: `argmax_a Q[i,j,:]` over 36×28 grid; unvisited cells (all-zero rows) masked/hatched.

## 7. Experiments → numbers
| Question | Experiment | Numbers produced |
|---|---|---|
| (a) ~20 % replacement | random policy, 2 000 episodes (≈ 400 k steps) modified env | attempted changes, replaced, rate, Wilson 95 % CI |
| (a) none when repeating | same | steps with a = a_prev, replacements among them (= 0) |
| (a) penalty on selected push | same, per-step log of (a, a_exec, R_base, R, B) | table of counts for each (a, a_exec) pair with observed penalty = 0.2·1[a≠1] in 100 % of steps, incl. a=1 / a_exec≠1 |
| (a) bonus only on gentle arrival | hand-coded velocity-sign policy + a "brake-near-goal" variant, 200 episodes | arrivals split by v ≤ 0.04 / > 0.04 vs B; truncations with B = 0 |
| (a) example | replay PDF 5-step example with scripted RNG | a_exec sequence 2,2,2,0,0 |
| (b)(c)(d) | 4 runs × N episodes, seed 84 | curves + eval table |
| (e1) value gap | |V_Q − V_SARSA| on validation set over training, orig vs mod; 5 seeds | final gap orig vs mod (mean ± std) |
| (e2) non-Markov | empirical: P(next state | s, a) depends on a_prev — measure from rollouts | e.g. mean next-velocity change for same (s,a) bin by a_prev |
| (e3) pushes/gentle | push counts, policy maps (fraction of "coast" cells), arrival-speed histogram & mean | |
| (e4) better algorithm | eval table + 5-seed means | |
| (e5) limitation | single α/γ, discretization; supports suggestion | ablation on bins (20×20, 36×28, 50×50) on modified env |

## 8. Report (notebook → PDF) order
0. Title + **Group Contribution Declaration** (names, reg nos, emails, 100 % each) — page 1
1. Introduction & setup (problem, environment-info cell with timestamp/host/versions, hyper-parameters, design rationale)
2. (a) Modified Environment Implementation and Verification — code + all verification outputs
3. (b) Q-Learning — code, training on both envs, logs
4. (c) SARSA — code (the only diff), training on both envs, logs
5. (d) Performance Evaluation — each plot + interpretation; evaluation table
6. (e) Discussion — five answers with numbers
7. Virtual-lab screenshots (timestamped)
8. Appendix — complete per-iteration (per-episode) logs for all runs; all package source code

## 9. Tasks only you can do
- [ ] Confirm team names/reg nos/contribution % (100 % each assumed)
- [ ] Copy source zip to the BITS virtual lab, `pip install -r requirements.txt`, run `python make_pdf.py` there
- [ ] Take timestamped screenshots (terminal with `date`, notebook running, plots) → `screenshots/`
- [ ] Create `VLAB_RUN.txt` (date, VM hostname) after the lab run; rebuild PDF
- [ ] Submit first version by **12 Nov 2026**; final by **14 Nov 2026**
- [ ] Decide GitHub repo visibility (private recommended)

## 10. Full-marks checklist
- [ ] Wrapper follows Steps 1–6 in order; a_prev = 1 on reset; info untouched; spaces unchanged
- [ ] Verification shows ≈ 20 % (CI), 0 replacements on repeats, penalty by selected action, bonus truth table
- [ ] Bins ≥ 20×20, justified; Q zeros (P×V×3); ε schedule stated; α, γ, seed, N ≥ 5 000 identical across 4 runs
- [ ] No bootstrap on terminated; bootstrap on truncated
- [ ] SARSA differs only in TD target
- [ ] All 5 plot types + 4 heatmaps, each interpreted; eval table with 4 required metrics
- [ ] 5 discussion answers with numbers
- [ ] Every function docstring + comments
- [ ] Contribution declaration first; VM screenshots with timestamps; exact PDF name; all code and outputs in PDF

## Audit fixes
| # | Gap / risk found in line-by-line audit | Fix |
|---|---|---|
| 1 | "Outputs with each and every iteration" → 4 × 6 000 episode logs would bloat PDF | Main body prints progress every 500 episodes; appendix holds full per-episode log in compact multi-column tables (decision confirmed with user) |
| 2 | Sticky RNG shared with agent RNG would make orig/mod runs consume randomness differently | separate RNG streams: env-reset seed, sticky RNG, agent RNG |
| 3 | Lab Python/Gymnasium version unknown | code restricted to py ≥ 3.8 & gymnasium ≥ 0.26 API; requirements pin lower bounds, not bleeding-edge |
| 4 | SARSA must use the a′ actually executed next | loop selects a′ once, uses it both in update and as next action |
| 5 | Truncation vs termination | pass `terminated` only to target; test both cases |
| 6 | Wrapper must not leak stuck info | diagnostics kept as private attributes, test asserts info keys equal base env's |
| 7 | Bonus verification with random policy impossible | hand-coded velocity policy + braking variant to produce both gentle and fast arrivals |
| 8 | "Similarity to AI tool solutions" scoring | own design (threshold-aligned bins, reachable-state validation set, cross-env robustness, 5-seed study), own wording |
| 9 | Comments rule (−1) | docstring test over all package functions; inline comments; notebook cells explained in markdown |
| 10 | Validation set must be fixed before training | sampled once in `run_experiments.py`, saved to `results/validation_states.npy`, reused by all runs |
| 11 | Unvisited states in policy maps mislead | mask cells never updated |
| 12 | Claiming VM execution without evidence | PDF shows VM section only if `VLAB_RUN.txt` + screenshots exist; otherwise a visible TODO |
| 13 | Old `gym` vs `gymnasium` naming in brief (`gym.Wrapper`) | `import gymnasium as gym` → class literally subclasses `gym.Wrapper` |

## Status
- Phase 1 (plan): done 2026-11-09.
- Phase 2 (build): done 2026-11-09.
  - Pilot (`pilot_sweep.py`, 216 runs, held-out seeds 184/284/384, eval seeds 50000+): best = **20×20, α 0.1, ε floor at 70 %, 10 000 episodes** (−144.3, sd 5.9 across seeds). 36×28 was less stable (sd 15.5), so the threshold-aligned grid idea was dropped in favour of the data.
  - Package `drl_sticky/` + notebook builder + `report_text.py` (interpretations rendered from live numbers) + `make_pdf.py` (nbclient → HTML lab template → headless Chrome, page numbers) + `package.py`.
  - PDF: 196 pages (≈40 report + ≈155 appendix per-episode logs, the user's choice).
  - Tests: 44 passed (also from the unzipped copy).
- Phase 3 (verify): done 2026-11-09. Notebook run and CLI run give identical numbers (deterministic seeds); `test_results_consistent_with_saved_qtables` re-evaluates the saved Q-tables against results.json.
- Remaining (team): virtual-lab run, screenshots, `VLAB_RUN.txt`, `make_pdf.py --no-execute`, submit the first version by 12 Nov.

### Key results (seed 84, greedy, 100 test episodes)
Q-orig −160.3 (100 % success / 0 % gentle) · SARSA-orig −160.8 (100 / 0) · Q-mod −155.6 (95 / 95) · SARSA-mod −117.6 (100 / 100).
5 seeds, modified env: Q −139.3 ± 12.7, SARSA −156.3 ± 31.4 (SARSA wins 3/5 seeds), so there is no consistent winner.
Value gap (Q − SARSA, mean over training): 5.98 ± 0.38 original vs 7.06 ± 0.90 modified (larger in 4/5 seeds; 5/5 in the first 20 %).

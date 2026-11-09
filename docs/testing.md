# Testing

`python -m pytest -q` runs:

| File | What it checks |
|---|---|
| `tests/test_env.py` | Wrapper vs specification: the PDF example table (2,2,2,0,0), reset sets a_prev = 1, no RNG draw on repeats, the strict `< 0.20` threshold, penalty on the selected action (including stuck cases), the bonus truth table (gentle / fast / truncated / not at flag), unchanged spaces and info, identical dynamics when never stuck, truncation at 200 |
| `tests/test_agents.py` | Hand-computed Q-learning and SARSA updates, no bootstrap when terminated, bootstrap when truncated, zero Q-table of shape (P,V,3), only `td_target` differs, ε-greedy extremes, discretizer edges, ε schedule |
| `tests/test_integration.py` | Real MountainCar: ~20 % replacement (Wilson CI), 0 on repeats, penalty audit, bonus audit, a short training run beats random and is bit-reproducible, fixed validation set |
| `tests/test_compliance.py` | No RL libraries imported, docstring on every module/class/function, required files present, exact PDF name, wrapper subclasses `gym.Wrapper`, contribution declaration |
| `tests/test_report.py` | results.json is consistent with the saved Q-tables (re-evaluated greedy policies), the executed notebook has no errors and has sections in assignment order, the PDF exists, page 1 holds the contribution declaration, every module's code appears in the PDF |

Results of the latest run are in the README.

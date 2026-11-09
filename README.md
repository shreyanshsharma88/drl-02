# Group 84: Robust RL under Sticky Actuators (DRL Assignment II, Set A)

Tabular **Q-learning vs SARSA** on a modified **MountainCar-v0**. The modification is a sticky actuator: with probability 0.20 an attempted action change is ignored. The reward adds an energy penalty and a gentle-arrival bonus. Everything is written by hand in NumPy. Gymnasium supplies only the base physics.

**Submission file:** `Group 84 - Q_learning_SARSA_Set A.pdf` (all code + all outputs, built from `Group84_Q_learning_SARSA_SetA.ipynb`).

## Team
| Name | Registration No. | Email | Contribution |
|---|---|---|---|
| SHREYANSH SHARMA | 2025AG05772 | 2025ag05772@wilp.bits-pilani.ac.in | 100% |
| LAKSHYA KANCHAN | 2025AG05062 | 2025ag05062@wilp.bits-pilani.ac.in | 100% |
| VINAY PRATAP SINGH BHADAURIA | 2025AG05689 | 2025ag05689@wilp.bits-pilani.ac.in | 100% |

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Run
```bash
python make_pdf.py              # build notebook -> execute (~5 min) -> HTML -> PDF
python make_pdf.py --no-execute # re-export an already executed notebook (e.g. after the virtual-lab run)
python run_experiments.py       # same experiments from the CLI -> results/, figures/
python pilot_sweep.py           # hyper-parameter pilot (held-out seeds) -> results/pilot.json
```

## Test
```bash
python -m pytest -q             # unit, integration, compliance and report-consistency tests
python package.py               # dist/ backup zip, unzip it and run the tests from the copy
```

## Key results (seed 84, greedy policy, 100 fixed-seed test episodes)
| Agent | mean reward | success | gentle arrival | mean length |
|---|---|---|---|---|
| Q-learning, original | -160.3 | 100% | 0% | 160.3 |
| SARSA, original | -160.8 | 100% | 0% | 160.8 |
| Q-learning, modified | -155.6 | 95% | 95% | 152.2 |
| SARSA, modified | -117.6 | 100% | 100% | 119.2 |

Over five seeds (84-88), the modified-environment reward is -139.3 ± 12.7 for Q-learning and -156.3 ± 31.4 for SARSA. SARSA wins 3 of the 5 seeds, but its variance is high, so neither algorithm wins consistently. The wrapper verification measured 20.02% of attempted changes replaced, 0 replacements on repeats, 0 penalty errors and 0 bonus errors. See `docs/results.md` (generated) and the PDF.

## Layout
`drl_sticky/` package · `tests/` · `docs/` (traceability, design, user guide, results, testing) · `results/` (results.json, per-episode CSV logs, Q-tables) · `figures/` · `screenshots/` (virtual-lab captures).

## **BEFORE SUBMISSION**
- [ ] **RUN THE NOTEBOOK IN THE BITS VIRTUAL LAB (SEE docs/user_guide.md, "VIRTUAL-LAB DEMO SCRIPT")**
- [ ] **TAKE TIMESTAMPED VIRTUAL-LAB SCREENSHOTS AND SAVE THEM IN screenshots/**
- [ ] **CREATE VLAB_RUN.txt IN THE VIRTUAL LAB (DATE, HOSTNAME, WHO RAN IT)**
- [ ] **REBUILD THE PDF: python make_pdf.py --no-execute, AND CHECK THAT SECTION (f) SHOWS THE SCREENSHOTS AND NO "TODO" REMAINS**
- [ ] **CHECK NAMES, REGISTRATION NUMBERS AND CONTRIBUTION % ON PAGE 1**
- [ ] **UPLOAD ONLY "Group 84 - Q_learning_SARSA_Set A.pdf"**
- [ ] **SUBMIT THE FIRST VERSION BY 12 NOV 2026 (2 DAYS EARLY); FINAL DEADLINE 14 NOV 2026**

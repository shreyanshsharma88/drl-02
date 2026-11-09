# User guide

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # runtime
pip install -r requirements-dev.txt  # + pytest, pypdf
```
Tested with Python 3.14 and Gymnasium 1.4. The code uses only the Gymnasium ≥ 0.26 API and Python ≥ 3.8 syntax.

## Run
| Goal | Command | Time (10-core laptop) |
|---|---|---|
| Everything + PDF | `python make_pdf.py` | ~5 min |
| Only experiments (CLI) | `python run_experiments.py` | ~5 min |
| Smoke run | `python run_experiments.py --quick` | ~20 s |
| Hyper-parameter pilot | `python pilot_sweep.py` | ~15 min |
| Export an already-executed notebook | `python make_pdf.py --no-execute` | ~20 s |
| Interactive | `jupyter lab Group84_Q_learning_SARSA_SetA.ipynb` → Run All | ~5 min |

## Test
```bash
python -m pytest -q
python package.py        # builds dist/ zip, unzips it and runs the tests there
```

## Virtual-lab demo script (do this in the BITS virtual lab)
1. Copy `dist/Group84_Q_learning_SARSA_SetA_source.zip` into the lab VM and unzip it.
2. Open a terminal and run `date` (keep it visible for the screenshot), then:
   ```bash
   cd Group84_Q_learning_SARSA_SetA
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Start `jupyter lab` (or `jupyter notebook`). Open `Group84_Q_learning_SARSA_SetA.ipynb` and click **Run All**.
4. **Screenshots** (the system clock or `date` output must be visible in each one):
   * `01_terminal_setup.png`: terminal with `date`, `hostname` and the pip install
   * `02_env_info.png`: the Section 0 output cell showing the timestamp and host name
   * `03_training_progress.png`: Q-learning/SARSA training progress lines
   * `04_figures.png`: one of the figures in section (d)
   * `05_results_table.png`: the greedy evaluation table (d.6)
   Save them in `screenshots/`.
5. Create the marker file: `echo "Executed in BITS virtual lab on $(date) host $(hostname) by <name>" > VLAB_RUN.txt`
6. Run all cells again (or only the last ones), save the notebook, then run `python make_pdf.py --no-execute`. If the VM has no Chrome, copy the executed notebook, `screenshots/` and `VLAB_RUN.txt` back to a laptop and run the same command there.
7. Check that the PDF shows the screenshots in section (f) and that the TODO notes have disappeared.

## Troubleshooting
| Problem | Fix |
|---|---|
| `No Chrome/Chromium found` | Open `Group84_Q_learning_SARSA_SetA.html` in any browser → Print → Save as PDF with the exact name, or install Chrome. |
| `ModuleNotFoundError: gymnasium` | `pip install "gymnasium[classic-control]"` |
| Old `gym` installed instead of `gymnasium` | The code needs `gymnasium` (5-tuple `step`). Install gymnasium. |
| Multiprocessing hangs in Jupyter on Windows | Run `python run_experiments.py` from a terminal, or call `run_robustness(..., processes=1)`. |
| Notebook too slow in the VM | Reduce seeds in the robustness cell (e.g. `seeds=[84, 85, 86]`). The main four runs take about 1 min. |
| Numbers differ slightly in the VM | Different NumPy/Gymnasium versions can change float rounding. The report text is generated from the live numbers, so it stays consistent. |

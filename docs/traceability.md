# Assignment vs solution: traceability

Report = `Group 84 - Q_learning_SARSA_Set A.pdf` (exported from `Group84_Q_learning_SARSA_SetA.ipynb`).

| # | Requirement (drl.pdf) | Code | Report section / output | Test |
|---|---|---|---|---|
| 1 | Wrapper via `gym.Wrapper` | `drl_sticky/env.py::StickyActuatorMountainCar` | (a) a.1 | `test_compliance::test_wrapper_subclasses_gym_wrapper` |
| 2 | `a_prev = 1` on every reset | `env.py::reset` | (a) a.2 table | `test_env::test_reset_sets_a_prev_to_1` |
| 3 | Step 1: store selected action | `env.py::step` (`a = int(action)`) | (a) a.1 | `test_penalty_uses_selected_action` |
| 4 | Step 2: no change when a = a_prev | `env.py::sticky_action` | (a) a.4 (0 replacements) | `test_no_draw_and_no_replacement_when_repeating` |
| 5 | Step 2: r ~ U[0,1), r < 0.20 → a_prev | `env.py::sticky_action` | (a) a.3 example, a.4 rate + CI | `test_threshold_is_strictly_below_0_2`, `test_replacement_rate_close_to_20_percent`, `test_pdf_example_table_reproduced` |
| 6 | Update a_prev = a_exec | `env.py::step` | (a) a.3 | `test_threshold_is_strictly_below_0_2` |
| 7 | No indication of replacement | info untouched; audit attributes only | (a) a.2 (info keys) | `test_spaces_and_info_unchanged` |
| 8 | Step 3: execute in base env | `env.py::step` | (a) a.2 (identical dynamics) | `test_dynamics_identical_without_sticking` |
| 9 | Step 4: R = R_base − 0.2·1[a≠1] + B | `env.py::step` | (a) a.5 pair table | `test_penalty_uses_selected_action` |
| 10 | Step 5: bonus conditions | `env.py::gentle_bonus` | (a) a.6 | `test_bonus_truth_table`, `test_bonus_only_for_gentle_arrivals` |
| 11 | Step 6: return 5-tuple, info unmodified | `env.py::step` | (a) a.2 | `test_spaces_and_info_unchanged` |
| 12 | Unchanged spaces/dynamics/termination | wrapper passes through | (a) a.2 | `test_dynamics_identical_without_sticking`, `test_truncation_at_200_steps` |
| 13 | (a) random-policy evidence: ~20 %, 0 on repeat, penalty, bonus | `drl_sticky/verify.py` | (a) a.4-a.6 | `test_integration.py` |
| 14 | (b) uniform bins ≥ 20×20, justified | `discretizer.py`, `config.py` | 0 config table, App. A pilot, d.7 ablation | `test_discretizer_edges_align_with_thresholds` |
| 15 | (b) Q-table (P×V×3) zeros | `agents.py::TabularTDAgent` | (b) b.2 | `test_zero_initialisation_and_shape` |
| 16 | (b) ε-greedy + stated decay | `agents.py::act`, `config.py::epsilon` | 0 config table | `test_epsilon_schedule`, `test_epsilon_greedy_extremes` |
| 17 | (b) Q-learning update, no bootstrap when terminated, bootstrap on truncation | `agents.py::QLearningAgent.td_target`, `train.py` | (b) b.2 | `test_qlearning_hand_computed`, `test_no_bootstrap_when_terminated`, `test_truncation_still_bootstraps` |
| 18 | (b) train on modified + original, same seed/bins/α/γ/ε/episodes ≥ 5000 | `train.py::train_agent`, shared `CFG` | (b) b.5 progress | `test_short_training_beats_random_and_is_reproducible` |
| 19 | (c) SARSA, only TD target differs | `agents.py::SarsaAgent.td_target` | (c) printed diff | `test_only_td_target_differs`, `test_sarsa_hand_computed` |
| 20 | (d) reward vs episode + MA100 | `plotting.py::plot_metric_panels` | d.1 | n/a |
| 21 | (d) avg max Q on fixed validation set | `train.py::sample_validation_states`, `avg_max_q` log | b.4, d.2 | `test_validation_set_fixed` |
| 22 | (d) success & gentle rate MA100 | `plotting.py::plot_rates` | d.3 | n/a |
| 23 | (d) pushes per episode | `plotting.py` | d.4 | n/a |
| 24 | (d) greedy policy heatmaps ×4 | `evaluate.py::policy_map`, `plot_policy_maps` | d.5 | n/a |
| 25 | (d) 100 fixed-seed greedy test episodes, one table | `evaluate.py::evaluate_greedy` | d.6 | `test_results_consistent_with_saved_qtables` |
| 26 | (d) interpretation of each plot | `report_text.py` | under every figure | n/a |
| 27 | (e) five discussion questions | `report_text.py` (e1-e5) | (e) | n/a |
| 28 | Comments on every function/operation | docstrings + inline comments | all code | `test_every_function_and_class_documented` |
| 29 | Contribution declaration at the beginning | notebook first cell, `CONTRIBUTION.md` | page 1 | `test_contribution_declaration_lists_all_members`, `test_pdf_first_page` |
| 30 | One PDF with all code + outputs of every iteration | `make_pdf.py`, training cells (b)-(c), Appendix C (every episode, 100-episode blocks) + `results/logs/*.csv` | whole PDF | `test_pdf_name_constant`, `test_pdf_contains_every_module` |
| 31 | Virtual-lab execution, timestamped screenshots | `screenshots/`, `VLAB_RUN.txt`, env-info cell | 0, (f) | manual (see user guide) |

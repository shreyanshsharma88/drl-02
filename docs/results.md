# Results (generated from results/results.json)

Run: 2026-11-14T01:47:24 on `Shreyanshs-MacBook-Air.local` (Python 3.14.7, Gymnasium 1.4.0), total 250s.

## (a) Wrapper verification

* Attempted changes 266632, replaced 53377 -> rate 0.2002 (95% CI 0.1987-0.2017)
* Replacements when a == a_prev: 0 of 133368 steps
* Penalty errors: 0 (a=1 penalised), 0 (push not penalised)
* Bonus violations: 0
* PDF example a_exec: [2, 2, 2, 0, 0]

## (d) Greedy evaluation (100 episodes, epsilon = 0, seed 84)

| Agent | mean reward | success | gentle | mean length | pushes | arrival v |
|---|---|---|---|---|---|---|
| Q-learning - Original | -160.30 | 1.00 | 0.00 | 160.3 | 139.4 | 0.0427 |
| SARSA - Original | -160.75 | 1.00 | 0.00 | 160.8 | 142.6 | 0.0423 |
| Q-learning - Modified | -155.56 | 0.95 | 0.95 | 152.2 | 111.9 | 0.0222 |
| SARSA - Modified | -117.59 | 1.00 | 1.00 | 119.2 | 92.0 | 0.0188 |

## Five seeds (84-88), greedy evaluation

| Agent | reward mean +- std | success | gentle |
|---|---|---|---|
| Q-learning - Original | -143.1 +- 11.5 | 0.99 | 0.54 |
| SARSA - Original | -143.7 +- 10.7 | 1.00 | 0.60 |
| Q-learning - Modified | -139.3 +- 12.7 | 0.99 | 0.99 |
| SARSA - Modified | -156.3 +- 31.4 | 0.84 | 0.82 |

## Value gap Q-learning - SARSA (mean over training, 5 seeds)

* original 5.97 +- 0.38; modified 7.06 +- 0.90
* first 20% of training: original 15.41, modified 16.63

Figures: `figures/fig1_reward.png` ... `figures/fig9_length.png`.

"""Integration / statistical tests on the real MountainCar-v0."""

import numpy as np

from drl_sticky.config import Config
from drl_sticky.evaluate import evaluate_greedy
from drl_sticky.train import sample_validation_states, train_agent
from drl_sticky.verify import bonus_audit, random_policy_audit


def test_replacement_rate_close_to_20_percent():
    a = random_policy_audit(n_episodes=500, seed=1)
    assert a["attempted_changes"] > 60000
    lo, hi = a["ci95"]
    assert lo < 0.20 < hi or abs(a["replacement_rate"] - 0.2) < 0.005
    assert a["replaced_on_repeat"] == 0
    assert a["penalty_on_selected_1"] == 0 and a["penalty_missing_on_push"] == 0


def test_bonus_only_for_gentle_arrivals():
    b = bonus_audit(n_episodes=60, seed=2)
    assert b["violations"] == 0


def test_short_training_beats_random_and_is_reproducible():
    cfg = Config(n_episodes=1500)
    val = sample_validation_states(n_states=200)
    ag1, log1, _ = train_agent("qlearning", False, cfg, val)
    ag2, log2, _ = train_agent("qlearning", False, cfg, val)
    assert np.array_equal(ag1.Q, ag2.Q)                                  # fixed seed -> identical
    assert log1["success"][-300:].mean() > 0.2                           # random policy ~0
    ev = evaluate_greedy(ag1, False, cfg, n_episodes=20)
    assert ev["mean_reward"] >= -200


def test_validation_set_fixed():
    a = sample_validation_states(n_states=100)
    b = sample_validation_states(n_states=100)
    assert np.array_equal(a, b)

"""Integration / statistical tests on the real MountainCar-v0."""

import numpy as np
import pytest

from drl_sticky import evaluate, train
from drl_sticky.agents import SarsaAgent
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


class _ScriptedEnv:
    last_r = None
    last_replaced = False

    def __init__(self, transitions):
        self.transitions = transitions
        self.actions = []
        self.reset_seed = None
        self.sticky_seed = None

    def reset(self, *, seed=None):
        self.reset_seed = seed
        self.actions = []
        return np.array([-0.5, 0.0]), {}

    def seed_sticky(self, seed):
        self.sticky_seed = seed

    def step(self, action):
        transition = self.transitions[len(self.actions)]
        self.actions.append(action)
        return transition

    def close(self):
        pass


@pytest.mark.parametrize("algo", ["qlearning", "sarsa"])
@pytest.mark.parametrize("modified", [False, True])
@pytest.mark.parametrize("position,velocity,terminated,truncated,expected", [
    (0.5, 0.04, True, False, 1.0),
    (0.51, 0.02, True, False, 1.0),
    (0.51, 0.02, True, True, 0.0),
    (0.499, 0.02, True, False, 0.0),
    (0.51, 0.0401, True, False, 0.0),
    (0.51, 0.02, False, True, 0.0),
])
def test_gentle_metrics_match_bonus_predicate(
    monkeypatch, algo, modified, position, velocity, terminated, truncated, expected,
):
    observation = np.array([position, velocity])
    environment = _ScriptedEnv([(observation, -1.0, terminated, truncated, {})])
    monkeypatch.setattr(train, "make_env", lambda *args, **kwargs: environment)
    monkeypatch.setattr(evaluate, "make_env", lambda *args, **kwargs: environment)
    cfg = Config(n_episodes=1)
    agent, log, _ = train_agent(algo, modified, cfg, np.array([[-0.5, 0.0]]))
    metrics = evaluate_greedy(agent, modified, cfg, n_episodes=1)
    assert log["gentle"][0] == expected
    assert metrics["gentle_rate"] == expected
    assert log["success"][0] == metrics["success_rate"] == float(terminated)


def test_sarsa_training_executes_sampled_next_action(monkeypatch):
    environment = _ScriptedEnv([
        (np.array([-0.4, 0.01]), -1.0, False, False, {}),
        (np.array([-0.3, 0.02]), -1.0, False, True, {}),
    ])
    selected_actions = iter([2, 0, 1])
    updates = []
    original_update = SarsaAgent.update

    def record_update(self, state, action, reward, next_state, next_action, terminated):
        updates.append((action, next_action, terminated))
        original_update(self, state, action, reward, next_state, next_action, terminated)

    monkeypatch.setattr(train, "make_env", lambda *args, **kwargs: environment)
    monkeypatch.setattr(SarsaAgent, "act", lambda *args: next(selected_actions))
    monkeypatch.setattr(SarsaAgent, "update", record_update)
    train_agent("sarsa", False, Config(n_episodes=1), np.array([[-0.5, 0.0]]))
    assert environment.actions == [2, 0]
    assert updates == [(2, 0, False), (0, 1, False)]

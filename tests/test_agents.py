"""Hand-computed checks of the TD updates and the agent structure."""

import inspect

import numpy as np
import pytest

from drl_sticky.agents import QLearningAgent, SarsaAgent, TabularTDAgent
from drl_sticky.config import Config
from drl_sticky.discretizer import UniformDiscretizer


def _agents():
    q = QLearningAgent((3, 3), 3, alpha=0.5, gamma=0.9, seed=0)
    s = SarsaAgent((3, 3), 3, alpha=0.5, gamma=0.9, seed=0)
    for ag in (q, s):
        ag.Q[1, 1] = [1.0, 4.0, 2.0]       # Q(s', .)
        ag.Q[0, 0] = [0.0, 0.0, 0.0]       # Q(s, .)
    return q, s


def test_zero_initialisation_and_shape():
    ag = QLearningAgent((36, 28), 3)
    assert ag.Q.shape == (36, 28, 3) and not ag.Q.any()


def test_qlearning_hand_computed():
    q, _ = _agents()
    # target = -1 + 0.9 * max(1,4,2) = 2.6 ; Q = 0 + 0.5 * 2.6 = 1.3
    q.update((0, 0), 2, -1.0, (1, 1), 0, terminated=False)
    assert q.Q[0, 0, 2] == pytest.approx(1.3)


def test_sarsa_hand_computed():
    _, s = _agents()
    # target = -1 + 0.9 * Q(s',a'=0)=1 -> -0.1 ; Q = 0.5 * -0.1 = -0.05
    s.update((0, 0), 2, -1.0, (1, 1), 0, terminated=False)
    assert s.Q[0, 0, 2] == pytest.approx(-0.05)


def test_no_bootstrap_when_terminated():
    q, s = _agents()
    q.update((0, 0), 1, 19.0, (1, 1), 2, terminated=True)
    s.update((0, 0), 1, 19.0, (1, 1), 2, terminated=True)
    assert q.Q[0, 0, 1] == pytest.approx(9.5) and s.Q[0, 0, 1] == pytest.approx(9.5)


def test_truncation_still_bootstraps():
    q, s = _agents()
    # truncated but not terminated -> terminated=False -> bootstrap
    q.update((0, 0), 1, -1.0, (1, 1), 2, terminated=False)
    s.update((0, 0), 1, -1.0, (1, 1), 2, terminated=False)
    assert q.Q[0, 0, 1] == pytest.approx(0.5 * (-1 + 0.9 * 4))
    assert s.Q[0, 0, 1] == pytest.approx(0.5 * (-1 + 0.9 * 2))


def test_only_td_target_differs():
    """SARSA and Q-learning override nothing but td_target."""
    own = lambda cls: {k for k, v in vars(cls).items() if inspect.isfunction(v)}
    assert own(QLearningAgent) == {"td_target"} and own(SarsaAgent) == {"td_target"}
    assert QLearningAgent.__bases__ == (TabularTDAgent,) == SarsaAgent.__bases__


def test_epsilon_greedy_extremes():
    ag = QLearningAgent((1, 1), 3, seed=1)
    ag.Q[0, 0] = [0, 5, 0]
    assert all(ag.act((0, 0), 0.0) == 1 for _ in range(50))
    acts = [ag.act((0, 0), 1.0) for _ in range(3000)]
    assert set(acts) == {0, 1, 2}


def test_discretizer_edges_align_with_thresholds():
    d = UniformDiscretizer((36, 28))
    assert d(( -1.2, -0.07)) == (0, 0) and d((0.6, 0.07)) == (35, 27)
    assert d((0.5, 0.0))[0] == 34 and d((0.4999, 0.0))[0] == 33   # flag on an edge
    assert d((0.0, 0.0401))[1] == 22 and d((0.0, 0.0399))[1] == 21  # 0.04 on an edge


def test_epsilon_schedule():
    cfg = Config(n_episodes=1000)
    assert cfg.epsilon(0) == 1.0
    assert cfg.epsilon(700) == pytest.approx(0.01, rel=1e-6)
    assert cfg.epsilon(999) == 0.01
    assert np.all(np.diff([cfg.epsilon(k) for k in range(1000)]) <= 0)

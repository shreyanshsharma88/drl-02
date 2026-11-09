"""Unit tests of the sticky-actuator wrapper against the assignment specification."""

import gymnasium as gym
import numpy as np
import pytest

from drl_sticky.env import StickyActuatorMountainCar, make_env
from drl_sticky.verify import ScriptedRNG, replay_pdf_example


def test_pdf_example_table_reproduced():
    """The 5-step example of the assignment gives a_exec = 2, 2, 2, 0, 0."""
    rows = replay_pdf_example()
    assert [r["a_prev"] for r in rows] == [1, 2, 2, 2, 0]
    assert [r["a_exec"] for r in rows] == [2, 2, 2, 0, 0]
    assert [r["r"] for r in rows] == [0.55, "-", 0.11, 0.74, 0.05]


def test_reset_sets_a_prev_to_1():
    env = make_env(True, sticky_seed=0)
    env.reset(seed=0)
    for _ in range(5):
        env.step(0)
    env.reset(seed=1)
    assert env.a_prev == 1


def test_no_draw_and_no_replacement_when_repeating():
    """Repeating the executed action never consumes a random number nor gets replaced."""
    env = make_env(True)
    env.reset(seed=0)
    env.rng = ScriptedRNG([0.0])          # would stick if a draw happened
    env.step(1)                           # a == a_prev == 1
    assert env.last_executed == 1 and env.last_r is None and env.rng.calls == 0


@pytest.mark.parametrize("r,expected", [(0.0, 1), (0.1999, 1), (0.2, 2), (0.9, 2)])
def test_threshold_is_strictly_below_0_2(r, expected):
    env = make_env(True)
    env.reset(seed=0)
    env.rng = ScriptedRNG([r])
    env.step(2)
    assert env.last_executed == expected
    assert env.a_prev == expected         # a_prev updated to executed action


@pytest.mark.parametrize("a,r,penalty", [(0, 0.5, 0.2), (2, 0.5, 0.2), (2, 0.1, 0.2), (1, 0.5, 0.0), (1, 0.1, 0.0)])
def test_penalty_uses_selected_action(a, r, penalty):
    """Penalty depends on a (selected), not on a_exec - including stuck cases."""
    env = make_env(True)
    env.reset(seed=0)
    env.rng = ScriptedRNG([0.9])
    env.step(0)                            # make a_prev = 0 so every a != 0 draws
    env.rng = ScriptedRNG([r])
    _, R, term, trunc, _ = env.step(a)
    assert R == pytest.approx(-1.0 - penalty)


def _bonus_env():
    return StickyActuatorMountainCar(gym.make("MountainCar-v0"))


@pytest.mark.parametrize("obs,term,trunc,bonus", [
    ((0.5, 0.04), True, False, 20.0),
    ((0.55, 0.01), True, False, 20.0),
    ((0.5, 0.0401), True, False, 0.0),     # too fast
    ((0.5, 0.03), True, True, 0.0),        # truncated
    ((0.5, 0.03), False, False, 0.0),      # not terminated
    ((0.49, 0.03), True, False, 0.0),      # not at flag
])
def test_bonus_truth_table(obs, term, trunc, bonus):
    assert _bonus_env().gentle_bonus(np.array(obs), term, trunc) == bonus


def test_spaces_and_info_unchanged():
    base = gym.make("MountainCar-v0")
    env = make_env(True, sticky_seed=0)
    assert env.observation_space == base.observation_space
    assert env.action_space == base.action_space
    _, i1 = base.reset(seed=3)
    _, i2 = env.reset(seed=3)
    assert i1 == i2
    _, _, _, _, i1 = base.step(2)
    _, _, _, _, i2 = env.step(2)
    assert i1.keys() == i2.keys()


def test_dynamics_identical_without_sticking():
    base = gym.make("MountainCar-v0")
    env = make_env(True, stick_prob=0.0)
    o1, _ = base.reset(seed=5)
    o2, _ = env.reset(seed=5)
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = int(rng.integers(3))
        o1, r1, t1, tr1, _ = base.step(a)
        o2, r2, t2, tr2, _ = env.step(a)
        assert np.array_equal(o1, o2) and t1 == t2 and tr1 == tr2
        if t1 or tr1:
            break
    assert tr1 or t1                      # TimeLimit still at 200 steps


def test_truncation_at_200_steps():
    env = make_env(True, sticky_seed=0)
    env.reset(seed=0)
    n, trunc, term = 0, False, False
    while not (trunc or term):
        _, _, term, trunc, _ = env.step(1)
        n += 1
    assert n == 200 and trunc and not term

import numpy as np
from gymnasium.utils.env_checker import check_env

from environment import WarehouseEnv


def test_environment_creation():
    env = WarehouseEnv()

    assert env is not None
    assert env.action_space.n == 4
    assert env.observation_space.shape == (12,)

    env.close()


def test_reset():
    env = WarehouseEnv()

    observation, info = env.reset(seed=42)

    assert isinstance(observation, np.ndarray)
    assert observation.shape == (12,)
    assert isinstance(info, dict)

    env.close()


def test_step():
    env = WarehouseEnv()

    observation, info = env.reset(seed=42)

    observation, reward, terminated, truncated, info = env.step(0)

    assert observation.shape == (12,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    env.close()


def test_multiple_packages():
    env = WarehouseEnv()

    env.reset(seed=42)

    assert len(env.packages) == 3

    env.close()


def test_gymnasium_compliance():
    env = WarehouseEnv()

    check_env(env)

    env.close()
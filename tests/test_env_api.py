from copy import deepcopy

from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test

from library_escape.config import load_env_config, load_reward_config
from library_escape.env.multi_agent_env import LibraryEscapeMAEnv
from library_escape.env.single_agent_env import LibraryEscapeEnv


def test_gymnasium_env_checker_passes():
    env = LibraryEscapeEnv(
        role="enemy",
        env_config=deepcopy(load_env_config()),
        reward_config=deepcopy(load_reward_config()),
        frame_skip=2,
    )
    check_env(env)
    env.close()


def test_parallel_env_checker_passes():
    env = LibraryEscapeMAEnv(
        env_config=deepcopy(load_env_config()),
        reward_config=deepcopy(load_reward_config()),
        frame_skip=2,
    )
    parallel_api_test(env, num_cycles=10)
    env.close()

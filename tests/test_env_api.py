import pytest

gymnasium = pytest.importorskip("gymnasium")
pettingzoo = pytest.importorskip("pettingzoo")

from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test

from library_escape.agents.rule_based_enemy import RuleBasedEnemyController
from library_escape.agents.rule_based_player import HeuristicPlayerController
from library_escape.env.multi_agent_env import LibraryEscapeMAEnv
from library_escape.env.single_agent_env import LibraryEscapeEnv


def test_gymnasium_env_api():
    env = LibraryEscapeEnv(controlled_agent="enemy", opponent_controller=HeuristicPlayerController())
    check_env(env.unwrapped)
    env.close()


def test_parallel_env_api():
    env = LibraryEscapeMAEnv()
    parallel_api_test(env, num_cycles=5)

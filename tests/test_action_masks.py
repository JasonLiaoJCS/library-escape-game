import numpy as np

from library_escape.agents.rule_based_player import HeuristicPlayerController
from library_escape.env.single_agent_env import LibraryEscapeEnv


def test_action_mask_has_expected_shape_and_binary_values():
    env = LibraryEscapeEnv(controlled_agent="enemy", opponent_controller=HeuristicPlayerController())
    env.reset(seed=5)
    mask = env.action_masks()
    assert mask.shape == (9,)
    assert mask[0] == 1
    assert set(np.unique(mask)).issubset({0, 1})
    env.close()


def test_frozen_enemy_only_allows_noop():
    env = LibraryEscapeEnv(controlled_agent="enemy", opponent_controller=HeuristicPlayerController())
    env.reset(seed=7)
    env.world.enemy.freeze_timer = 1.0
    mask = env.action_masks()
    assert mask[0] == 1
    assert int(mask.sum()) == 1
    env.close()

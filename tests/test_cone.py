from copy import deepcopy

from library_escape.config import load_env_config
from library_escape.core.entities import Rectangle
from library_escape.core.world import LibraryWorld


def test_enemy_vision_blocked_by_obstacle():
    env_config = deepcopy(load_env_config())
    world = LibraryWorld(env_config)
    world.player.position[:] = [300.0, 100.0]
    world.enemy.position[:] = [180.0, 100.0]
    world.enemy.facing = 0.0
    world.obstacles = [
        type(world.obstacles[0])(
            kind="bookshelf",
            rect=Rectangle(220.0, 70.0, 40.0, 80.0),
            blocks_movement=True,
            blocks_sight=True,
        )
    ]
    assert not world.enemy_can_see_player()

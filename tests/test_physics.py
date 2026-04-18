from library_escape.core.obstacle import Obstacle
from library_escape.core.physics import has_line_of_sight, move_circle


def test_move_circle_stops_at_obstacle():
    obstacles = [Obstacle(kind="wall", texture_key="dummy", x=2.0, y=2.0, w=1.0, h=1.0)]
    x, y, collided = move_circle(
        x=1.2,
        y=2.5,
        radius=0.3,
        velocity=(3.0, 0.0),
        dt=0.5,
        width=10.0,
        height=10.0,
        obstacles=obstacles,
    )
    assert collided is True
    assert x == 1.2
    assert y == 2.5


def test_line_of_sight_respects_blockers():
    obstacles = [Obstacle(kind="wall", texture_key="dummy", x=2.0, y=2.0, w=1.0, h=1.0)]
    assert has_line_of_sight((1.0, 2.5), (4.0, 2.5), obstacles) is False
    assert has_line_of_sight((1.0, 1.0), (4.0, 1.0), obstacles) is True

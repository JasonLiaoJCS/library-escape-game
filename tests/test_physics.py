from library_escape.core.entities import ObstacleState, Rectangle, vec2
from library_escape.core.physics import move_circle


def test_circle_stops_at_obstacle_edge():
    obstacle = ObstacleState(kind="table", rect=Rectangle(100.0, 100.0, 40.0, 40.0))
    position, wall_hits = move_circle(
        position=vec2(70.0, 120.0),
        velocity=vec2(200.0, 0.0),
        dt=0.2,
        radius=10.0,
        obstacles=[obstacle],
        width=400.0,
        height=300.0,
    )
    assert position[0] <= 90.0
    assert wall_hits >= 1

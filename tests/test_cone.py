from library_escape.core.enemy import VisionCone
from library_escape.core.obstacle import Obstacle


def test_vision_cone_detects_target_in_front():
    cone = VisionCone(range_cells=6.0, angle_deg=70.0)
    assert cone.sees((1.0, 1.0), (1.0, 0.0), (4.0, 1.0), []) is True


def test_vision_cone_rejects_blocked_target():
    cone = VisionCone(range_cells=6.0, angle_deg=70.0)
    obstacles = [Obstacle(kind="wall", texture_key="dummy", x=2.0, y=0.0, w=1.0, h=3.0)]
    assert cone.sees((1.0, 1.0), (1.0, 0.0), (4.0, 1.0), obstacles) is False

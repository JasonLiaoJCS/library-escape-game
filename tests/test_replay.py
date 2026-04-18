from library_escape.core.world import World
from library_escape.replay.io import REPLAY_FORMAT, ReplayRecorder, ReplayWorld, load_replay


def test_replay_round_trip(tmp_path):
    world = World(seed=123)
    replay_path = tmp_path / "sample.ler.gz"
    recorder = ReplayRecorder(replay_path, world, metadata={"mode": "test"})
    recorder.capture(world)
    world.step(player_action=(1.0, 0.0), enemy_action=(-1.0, 0.0), frame_skip=1)
    recorder.capture(world)
    saved_path = recorder.save()

    payload = load_replay(saved_path)
    assert payload["format"] == REPLAY_FORMAT
    assert len(payload["frames"]) == 2

    replay_world = ReplayWorld(payload)
    assert replay_world.time_remaining == payload["frames"][0]["time_remaining"]
    replay_world.reset_to_frame(1)
    assert replay_world.frame_index == 1
    assert replay_world.player.x == payload["frames"][1]["player"]["x"]

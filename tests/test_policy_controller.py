from library_escape.agents.ppo_agent import SB3PolicyController
from library_escape.game_modes import build_game_mode_env_config
from library_escape.core.world import World


class _DummyModel:
    def __init__(self) -> None:
        self.calls = 0
        self.deterministic_flags: list[bool] = []

    def predict(self, obs, deterministic=True, action_masks=None):
        self.calls += 1
        self.deterministic_flags.append(bool(deterministic))
        return 3, None


def test_policy_controller_can_repeat_actions_between_inference_calls(monkeypatch):
    dummy_model = _DummyModel()
    env_config = build_game_mode_env_config(game_mode="escape", manual_collect_required=False, interactive=False)

    monkeypatch.setattr(
        SB3PolicyController,
        "_load_model",
        lambda self, path: (dummy_model, False, "ppo"),
    )
    monkeypatch.setattr(
        SB3PolicyController,
        "_load_obs_normalizer",
        lambda self, explicit_path: None,
    )

    controller = SB3PolicyController(
        "enemy",
        "checkpoints/test/dummy.zip",
        env_config=env_config,
        deterministic=False,
        decision_repeat_steps=4,
    )
    world = World(env_config=env_config, seed=0)

    actions = [controller.act(world) for _ in range(4)]
    assert dummy_model.calls == 1
    assert actions.count(actions[0]) == 4

    controller.act(world)
    assert dummy_model.calls == 2
    assert dummy_model.deterministic_flags == [False, False]


def test_policy_controller_reset_clears_repeat_budget(monkeypatch):
    dummy_model = _DummyModel()
    env_config = build_game_mode_env_config(game_mode="collection", manual_collect_required=False, interactive=False)

    monkeypatch.setattr(
        SB3PolicyController,
        "_load_model",
        lambda self, path: (dummy_model, False, "ppo"),
    )
    monkeypatch.setattr(
        SB3PolicyController,
        "_load_obs_normalizer",
        lambda self, explicit_path: None,
    )

    controller = SB3PolicyController(
        "player",
        "checkpoints/test/dummy.zip",
        env_config=env_config,
        deterministic=False,
        decision_repeat_steps=3,
    )
    world = World(env_config=env_config, seed=1)

    controller.act(world)
    controller.act(world)
    assert dummy_model.calls == 1

    controller.reset()
    controller.act(world)
    assert dummy_model.calls == 2

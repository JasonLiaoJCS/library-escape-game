from library_escape.core.events import StepEvents
from library_escape.core.world import World
from library_escape.rewards.reward_fns import RewardEngine


def test_reward_engine_returns_finite_values():
    world = World(seed=123)
    reward_engine = RewardEngine.from_env_config(world.env_config)
    prev_metrics = world.transition_metrics()

    events = world.step(player_action=(1.0, 0.0), enemy_action=(-1.0, 0.0), frame_skip=world.rl_frame_skip)
    next_metrics = world.transition_metrics()
    rewards = reward_engine.compute(world, events, prev_metrics=prev_metrics, next_metrics=next_metrics)

    assert set(rewards.keys()) == {"player_0", "enemy_0"}
    assert isinstance(rewards["player_0"], float)
    assert isinstance(rewards["enemy_0"], float)


def test_terminal_rewards_dominate_on_escape_or_catch():
    world = World(seed=0)
    reward_engine = RewardEngine.from_env_config(world.env_config)

    caught_events = StepEvents(player_caught=True)
    caught_rewards = reward_engine.compute(world, caught_events, prev_metrics=world.transition_metrics(), next_metrics=world.transition_metrics())
    assert caught_rewards["enemy_0"] > 0
    assert caught_rewards["player_0"] < 0

    escaped_events = StepEvents(player_escaped=True)
    escaped_rewards = reward_engine.compute(world, escaped_events, prev_metrics=world.transition_metrics(), next_metrics=world.transition_metrics())
    assert escaped_rewards["player_0"] > 0

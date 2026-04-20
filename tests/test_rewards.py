from library_escape.core.events import StepEvents
from library_escape.core.world import World
from library_escape.env.obs_builder import ObsBuilder
from library_escape.game_modes import build_game_mode_env_config
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
    assert escaped_rewards["enemy_0"] < 0


def test_detection_and_collection_rewards_align_with_team_gameplay():
    world = World(seed=0)
    reward_engine = RewardEngine.from_env_config(world.env_config)
    prev_metrics = world.transition_metrics()

    visibility_events = StepEvents(
        visible_steps=world.rl_frame_skip,
        primary_visible_steps=world.rl_frame_skip,
        support_visible_steps=world.rl_frame_skip,
        visible_enemy_count=2 * world.rl_frame_skip,
        primary_distance_delta=0.25,
        primary_enemy_net_displacement=0.12,
        progress_made=True,
    )
    visibility_rewards = reward_engine.compute(world, visibility_events, prev_metrics=prev_metrics, next_metrics=prev_metrics)
    assert visibility_rewards["enemy_0"] > 0
    assert visibility_rewards["player_0"] < 0

    note = world.active_collectibles("note")[0]
    note.active = False
    world.score["note"] += 1
    next_metrics = world.transition_metrics()
    collect_events = StepEvents(progress_made=True)
    collect_events.collected["note"] += 1
    collect_rewards = reward_engine.compute(world, collect_events, prev_metrics=prev_metrics, next_metrics=next_metrics)
    assert collect_rewards["player_0"] > 0
    assert collect_rewards["enemy_0"] < 0


def test_collection_timeout_rewards_scale_with_score_progress():
    env_config = build_game_mode_env_config(game_mode="collection", manual_collect_required=False, interactive=False)
    world = World(env_config=env_config, seed=0)
    reward_engine = RewardEngine.from_env_config(world.env_config)
    prev_metrics = world.transition_metrics()

    timeout_events = StepEvents(time_expired=True)
    baseline_rewards = reward_engine.compute(world, timeout_events, prev_metrics=prev_metrics, next_metrics=prev_metrics)

    note = world.active_collectibles("note")[0]
    note.active = False
    world.score["note"] += 1
    progressed_metrics = world.transition_metrics()
    progressed_rewards = reward_engine.compute(
        world,
        timeout_events,
        prev_metrics=prev_metrics,
        next_metrics=progressed_metrics,
    )

    assert progressed_rewards["player_0"] > baseline_rewards["player_0"]
    assert progressed_rewards["enemy_0"] < baseline_rewards["enemy_0"]


def test_escape_objective_completion_rewards_favor_player():
    env_config = build_game_mode_env_config(game_mode="escape", manual_collect_required=False, interactive=False)
    world = World(env_config=env_config, seed=0)
    reward_engine = RewardEngine.from_env_config(world.env_config)
    prev_metrics = world.transition_metrics()

    for collectible in world.collectibles:
        if collectible.kind == "note":
            collectible.active = False
            world.score["note"] += 1
    next_metrics = world.transition_metrics()
    events = StepEvents(progress_made=True, objective_completed=True)
    rewards = reward_engine.compute(world, events, prev_metrics=prev_metrics, next_metrics=next_metrics)

    assert rewards["player_0"] > 0.0
    assert rewards["enemy_0"] < 0.0


def test_antistall_penalties_hit_stationary_and_oscillating_agents():
    env_config = build_game_mode_env_config(game_mode="escape", manual_collect_required=False, interactive=False)
    world = World(env_config=env_config, seed=0)
    reward_engine = RewardEngine.from_env_config(world.env_config)
    metrics = world.transition_metrics()

    stationary_events = StepEvents(
        progress_made=True,
        player_net_displacement=0.0,
        primary_enemy_net_displacement=0.0,
        player_path_length=0.0,
        primary_enemy_path_length=0.0,
    )
    stationary_rewards = reward_engine.compute(world, stationary_events, prev_metrics=metrics, next_metrics=metrics)

    oscillating_events = StepEvents(
        progress_made=True,
        player_net_displacement=0.01,
        primary_enemy_net_displacement=0.01,
        player_path_length=0.24,
        primary_enemy_path_length=0.24,
    )
    oscillating_rewards = reward_engine.compute(world, oscillating_events, prev_metrics=metrics, next_metrics=metrics)

    assert stationary_rewards["player_0"] < 0.0
    assert stationary_rewards["enemy_0"] < 0.0
    assert oscillating_rewards["player_0"] < stationary_rewards["player_0"]
    assert oscillating_rewards["enemy_0"] < stationary_rewards["enemy_0"]


def test_visible_stationary_enemy_is_still_penalized_in_escape_mode():
    env_config = build_game_mode_env_config(game_mode="escape", manual_collect_required=False, interactive=False)
    world = World(env_config=env_config, seed=0)
    reward_engine = RewardEngine.from_env_config(world.env_config)
    metrics = world.transition_metrics()

    visible_stationary_events = StepEvents(
        primary_visible_steps=world.rl_frame_skip,
        visible_enemy_count=world.rl_frame_skip,
        progress_made=True,
        player_net_displacement=0.0,
        primary_enemy_net_displacement=0.0,
        player_path_length=0.0,
        primary_enemy_path_length=0.0,
    )
    rewards = reward_engine.compute(world, visible_stationary_events, prev_metrics=metrics, next_metrics=metrics)

    assert rewards["enemy_0"] < 0.0


def test_enemy_reward_favors_pressuring_player_objective_before_exit_unlock():
    env_config = build_game_mode_env_config(game_mode="escape", manual_collect_required=False, interactive=False)
    world = World(env_config=env_config, seed=0)
    reward_engine = RewardEngine.from_env_config(world.env_config)
    prev_metrics = world.transition_metrics()
    next_metrics = dict(prev_metrics)
    next_metrics["distance_primary_to_player_goal"] = max(0.0, float(prev_metrics["distance_primary_to_player_goal"]) - 0.6)

    rewards = reward_engine.compute(
        world,
        StepEvents(progress_made=True, primary_enemy_net_displacement=0.12),
        prev_metrics=prev_metrics,
        next_metrics=next_metrics,
    )

    assert rewards["enemy_0"] > 0.0


def test_enemy_reward_favors_guarding_exit_after_unlock():
    env_config = build_game_mode_env_config(game_mode="escape", manual_collect_required=False, interactive=False)
    world = World(env_config=env_config, seed=0)
    reward_engine = RewardEngine.from_env_config(world.env_config)
    prev_metrics = dict(world.transition_metrics())
    prev_metrics["can_escape"] = 1.0
    next_metrics = dict(prev_metrics)
    next_metrics["distance_enemy_to_escape"] = max(0.0, float(prev_metrics["distance_enemy_to_escape"]) - 0.5)

    rewards = reward_engine.compute(
        world,
        StepEvents(progress_made=True, primary_enemy_net_displacement=0.12),
        prev_metrics=prev_metrics,
        next_metrics=next_metrics,
    )

    assert rewards["enemy_0"] > 0.0


def test_both_modes_expose_positions_after_observation_rework():
    collection_env = build_game_mode_env_config(game_mode="collection", manual_collect_required=False, interactive=False)
    collection_world = World(env_config=collection_env, seed=0)
    collection_world.player.x = 1.5
    collection_world.player.y = 1.5
    collection_world.enemy.x = 15.5
    collection_world.enemy.y = 15.5
    collection_builder = ObsBuilder(collection_env)

    escape_env = build_game_mode_env_config(game_mode="escape", manual_collect_required=False, interactive=False)
    escape_world = World(env_config=escape_env, seed=0)
    escape_world.player.x = 1.5
    escape_world.player.y = 1.5
    escape_world.enemy.x = 15.5
    escape_world.enemy.y = 15.5
    escape_builder = ObsBuilder(escape_env)

    collection_enemy_obs = collection_builder.build(collection_world, "enemy")
    collection_player_obs = collection_builder.build(collection_world, "player")
    escape_enemy_obs = escape_builder.build(escape_world, "enemy")
    escape_player_obs = escape_builder.build(escape_world, "player")

    assert collection_builder.partial_observability is False
    assert escape_builder.partial_observability is False
    assert collection_enemy_obs[4] != 0.0 or collection_enemy_obs[5] != 0.0
    assert collection_player_obs[4] != 0.0 or collection_player_obs[5] != 0.0
    assert escape_enemy_obs[4] != 0.0 or escape_enemy_obs[5] != 0.0
    assert escape_player_obs[4] != 0.0 or escape_player_obs[5] != 0.0
    assert escape_enemy_obs[29] != 0.0 or escape_enemy_obs[30] != 0.0
    assert escape_player_obs[35] != 0.0 or escape_player_obs[36] != 0.0


def test_escape_player_obs_exposes_primary_support_and_exit_vectors():
    env_config = build_game_mode_env_config(game_mode="escape", manual_collect_required=False, interactive=False)
    world = World(env_config=env_config, seed=0)
    builder = ObsBuilder(env_config)
    world.player.x = 10.5
    world.player.y = 10.5
    world.enemy.x = 6.5
    world.enemy.y = 6.5
    world.support_enemies[0].x = 20.5
    world.support_enemies[0].y = 14.5

    obs = builder.build(world, "player")

    assert obs[31] != 0.0 or obs[32] != 0.0
    assert obs[33] != 0.0 or obs[34] != 0.0
    assert obs[35] != 0.0 or obs[36] != 0.0


def test_collection_enemy_reward_favors_pressuring_scoring_target():
    env_config = build_game_mode_env_config(game_mode="collection", manual_collect_required=False, interactive=False)
    world = World(env_config=env_config, seed=0)
    reward_engine = RewardEngine.from_env_config(world.env_config)
    prev_metrics = world.transition_metrics()
    next_metrics = dict(prev_metrics)
    next_metrics["distance_primary_to_player_goal"] = max(0.0, float(prev_metrics["distance_primary_to_player_goal"]) - 0.55)

    rewards = reward_engine.compute(
        world,
        StepEvents(progress_made=True, primary_enemy_net_displacement=0.12),
        prev_metrics=prev_metrics,
        next_metrics=next_metrics,
    )

    assert rewards["enemy_0"] > 0.0


def test_collection_visible_stationary_enemy_is_penalized():
    env_config = build_game_mode_env_config(game_mode="collection", manual_collect_required=False, interactive=False)
    world = World(env_config=env_config, seed=0)
    reward_engine = RewardEngine.from_env_config(world.env_config)
    metrics = world.transition_metrics()

    rewards = reward_engine.compute(
        world,
        StepEvents(
            primary_visible_steps=world.rl_frame_skip,
            visible_enemy_count=world.rl_frame_skip,
            progress_made=True,
            player_net_displacement=0.0,
            primary_enemy_net_displacement=0.0,
            player_path_length=0.0,
            primary_enemy_path_length=0.0,
        ),
        prev_metrics=metrics,
        next_metrics=metrics,
    )

    assert rewards["enemy_0"] < 0.0


def test_navigation_alignment_rewards_encourage_moving_with_planned_route():
    env_config = build_game_mode_env_config(game_mode="escape", manual_collect_required=False, interactive=False)
    world = World(env_config=env_config, seed=0)
    reward_engine = RewardEngine.from_env_config(world.env_config)
    metrics = world.transition_metrics()

    aligned_rewards = reward_engine.compute(
        world,
        StepEvents(
            progress_made=True,
            player_net_displacement=0.12,
            primary_enemy_net_displacement=0.12,
            player_goal_alignment=world.rl_frame_skip,
            player_evade_alignment=world.rl_frame_skip,
            enemy_chase_alignment=world.rl_frame_skip,
            enemy_guard_alignment=world.rl_frame_skip,
        ),
        prev_metrics=metrics,
        next_metrics=metrics,
    )
    misaligned_rewards = reward_engine.compute(
        world,
        StepEvents(
            progress_made=True,
            player_net_displacement=0.12,
            primary_enemy_net_displacement=0.12,
            player_goal_alignment=-world.rl_frame_skip,
            player_evade_alignment=-world.rl_frame_skip,
            enemy_chase_alignment=-world.rl_frame_skip,
            enemy_guard_alignment=-world.rl_frame_skip,
        ),
        prev_metrics=metrics,
        next_metrics=metrics,
    )

    assert aligned_rewards["player_0"] > misaligned_rewards["player_0"]
    assert aligned_rewards["enemy_0"] > misaligned_rewards["enemy_0"]


def test_turn_in_place_penalty_hits_reverse_jitter():
    env_config = build_game_mode_env_config(game_mode="escape", manual_collect_required=False, interactive=False)
    world = World(env_config=env_config, seed=0)
    reward_engine = RewardEngine.from_env_config(world.env_config)
    metrics = world.transition_metrics()

    calm_rewards = reward_engine.compute(
        world,
        StepEvents(
            progress_made=True,
            player_net_displacement=0.01,
            primary_enemy_net_displacement=0.01,
        ),
        prev_metrics=metrics,
        next_metrics=metrics,
    )
    jitter_rewards = reward_engine.compute(
        world,
        StepEvents(
            progress_made=True,
            player_net_displacement=0.01,
            primary_enemy_net_displacement=0.01,
            player_turn_amount=180.0,
            primary_enemy_turn_amount=180.0,
            player_reverse_turns=1,
            primary_enemy_reverse_turns=1,
        ),
        prev_metrics=metrics,
        next_metrics=metrics,
    )

    assert jitter_rewards["player_0"] < calm_rewards["player_0"]
    assert jitter_rewards["enemy_0"] < calm_rewards["enemy_0"]


def test_obs_builder_exposes_mode_flags_and_expanded_dimensions():
    collection_env = build_game_mode_env_config(game_mode="collection", manual_collect_required=False, interactive=False)
    collection_world = World(env_config=collection_env, seed=0)
    collection_builder = ObsBuilder(collection_env)
    collection_enemy_obs = collection_builder.build(collection_world, "enemy")
    collection_player_obs = collection_builder.build(collection_world, "player")

    assert collection_enemy_obs.shape == (collection_builder.enemy_obs_dim(),)
    assert collection_player_obs.shape == (collection_builder.player_obs_dim(),)
    assert collection_enemy_obs[27] == 1.0
    assert collection_enemy_obs[28] == 0.0
    assert collection_player_obs[29] == 1.0
    assert collection_player_obs[30] == 0.0

    escape_env = build_game_mode_env_config(game_mode="escape", manual_collect_required=False, interactive=False)
    escape_world = World(env_config=escape_env, seed=0)
    escape_builder = ObsBuilder(escape_env)
    escape_enemy_obs = escape_builder.build(escape_world, "enemy")
    escape_player_obs = escape_builder.build(escape_world, "player")

    assert escape_enemy_obs.shape == (escape_builder.enemy_obs_dim(),)
    assert escape_player_obs.shape == (escape_builder.player_obs_dim(),)
    assert escape_enemy_obs[27] == 0.0
    assert escape_enemy_obs[28] == 1.0
    assert escape_player_obs[29] == 0.0
    assert escape_player_obs[30] == 1.0

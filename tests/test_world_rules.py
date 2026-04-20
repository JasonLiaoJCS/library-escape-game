import math

from library_escape.config import load_map_config
from library_escape.agents.rule_based_enemy import RuleBasedEnemyController
from library_escape.agents.rule_based_player import HeuristicPlayerController
from library_escape.core.world import World
from library_escape.play.presets import build_play_env_config


def test_alert_level_only_rises_on_actual_visibility():
    world = World(seed=0)
    world.enemy.x = 5.5
    world.enemy.y = 5.5
    world.enemy.facing_x = 1.0
    world.enemy.facing_y = 0.0

    world.player.x = 4.5
    world.player.y = 5.5
    assert world.alert_level() == 0.0

    world.player.x = 6.5
    world.player.y = 5.5
    assert world.alert_level() == 1.0


def test_classic_detection_matches_original_penalty_pause_behavior():
    env_config = build_play_env_config(ruleset="classic", manual_collect_required=True)
    world = World(env_config=env_config, seed=0)
    world.enemy.x = 5.5
    world.enemy.y = 5.5
    world.enemy.facing_x = 1.0
    world.enemy.facing_y = 0.0
    world.player.x = 7.0
    world.player.y = 5.5
    world.startup_grace_timer = 0.0

    enemy_x = world.enemy.x
    world.step(player_action=(0.0, 0.0), enemy_action=(1.0, 0.0), frame_skip=1, player_collect=False)

    assert world.enemy.x == enemy_x
    assert world.enemy.detection_pause_timer >= 1.2
    assert world.alert_level() == 1.0
    assert 51.0 < world.time_remaining <= 52.0


def test_classic_detection_only_pauses_single_detector_enemy():
    env_config = build_play_env_config(ruleset="classic", manual_collect_required=True)
    world = World(env_config=env_config, seed=0)
    world.enemy.x = 5.5
    world.enemy.y = 5.5
    world.enemy.facing_x = 1.0
    world.enemy.facing_y = 0.0
    world.player.x = 7.0
    world.player.y = 5.5
    world.startup_grace_timer = 0.0

    world.step(player_action=(0.0, 0.0), enemy_action=(1.0, 0.0), frame_skip=1, player_collect=False)

    assert world.enemy.detection_pause_timer >= 1.2
    assert all(enemy.detection_pause_timer == 0.0 for enemy in world.support_enemies)

    support_positions_before = {enemy.name: enemy.position for enemy in world.support_enemies}
    world.player.x = 16.5
    world.player.y = 9.5
    world.step(player_action=(0.0, 0.0), enemy_action=(0.0, 0.0), frame_skip=1, player_collect=False)

    assert any(
        math.hypot(
            enemy.x - support_positions_before[enemy.name][0],
            enemy.y - support_positions_before[enemy.name][1],
        ) > 1e-4
        for enemy in world.support_enemies
    )


def test_classic_detection_support_enemies_keep_moving_while_primary_pause_is_active():
    env_config = build_play_env_config(ruleset="classic", manual_collect_required=True)
    world = World(env_config=env_config, seed=0)
    world.enemy.x = 5.5
    world.enemy.y = 5.5
    world.enemy.facing_x = 1.0
    world.enemy.facing_y = 0.0
    world.player.x = 7.0
    world.player.y = 5.5
    world.startup_grace_timer = 0.0

    cumulative_support_motion = {enemy.name: 0.0 for enemy in world.support_enemies}
    for _ in range(12):
        positions_before = {enemy.name: enemy.position for enemy in world.support_enemies}
        world.step(player_action=(0.0, 0.0), enemy_action=(0.0, 0.0), frame_skip=1, player_collect=False)
        assert world.enemy.detection_pause_timer >= 0.0
        assert all(enemy.detection_pause_timer == 0.0 for enemy in world.support_enemies)
        for enemy in world.support_enemies:
            before_x, before_y = positions_before[enemy.name]
            cumulative_support_motion[enemy.name] += math.hypot(enemy.x - before_x, enemy.y - before_y)

    assert world.enemy.detection_pause_timer > 0.0
    assert all(distance > 0.05 for distance in cumulative_support_motion.values())


def test_classic_detection_does_not_refresh_pause_or_repeat_penalty_while_visible():
    env_config = build_play_env_config(ruleset="classic", manual_collect_required=True)
    world = World(env_config=env_config, seed=0)
    world.enemy.x = 5.5
    world.enemy.y = 5.5
    world.enemy.facing_x = 1.0
    world.enemy.facing_y = 0.0
    world.player.x = 7.0
    world.player.y = 5.5
    world.startup_grace_timer = 0.0

    world.step(player_action=(0.0, 0.0), enemy_action=(0.0, 0.0), frame_skip=1, player_collect=False)
    time_after_first_detection = world.time_remaining
    pause_after_first_detection = world.enemy.detection_pause_timer

    world.step(player_action=(0.0, 0.0), enemy_action=(0.0, 0.0), frame_skip=1, player_collect=False)

    assert world.time_remaining > time_after_first_detection - 0.5
    assert world.enemy.detection_pause_timer < pause_after_first_detection


def test_manual_collection_requires_holding_collect_button():
    env_config = build_play_env_config(ruleset="classic", manual_collect_required=True)
    world = World(env_config=env_config, seed=0)
    target = world.active_collectibles()[0]
    interact_cell = world.interaction_cells_for_collectible(target)[0]

    world.player.x = interact_cell[0] + 0.5
    world.player.y = interact_cell[1] + 0.5
    world.enemy.x = 28.5
    world.enemy.y = 15.5
    world.enemy.facing_x = 1.0
    world.enemy.facing_y = 0.0

    world.step(player_action=(0.0, 0.0), enemy_action=(0.0, 0.0), frame_skip=1, player_collect=False)
    assert target.active is True
    assert world.collection_progress == 0.0

    hold_steps = math.ceil(world.collection_hold_seconds / world.physics_dt) + 1
    for _ in range(hold_steps):
        world.step(player_action=(0.0, 0.0), enemy_action=(0.0, 0.0), frame_skip=1, player_collect=True)
        if not target.active:
            break

    assert target.active is False
    assert sum(world.score.values()) == 1


def test_world_spawns_notes_and_exams_from_plural_config_keys():
    env_config = build_play_env_config(ruleset="classic", manual_collect_required=True)
    world = World(env_config=env_config, seed=0)

    assert len(world.active_collectibles("note")) == 6
    assert len(world.active_collectibles("exam")) == 2
    assert len(world.active_collectibles("coffee")) == 1
    assert len(world.active_collectibles("freeze")) == 1


def test_collection_mode_uses_collection_rewards_and_rules():
    env_config = build_play_env_config(game_mode="collection", manual_collect_required=False)
    world = World(env_config=env_config, seed=0)

    assert world.is_collection_mode() is True
    assert world.is_escape_mode() is False
    assert env_config["reward_path"].endswith("configs/rewards_collection.yaml")
    assert world.can_player_escape() is False


def test_escape_mode_uses_escape_rewards_and_rules():
    env_config = build_play_env_config(game_mode="escape", manual_collect_required=False)
    world = World(env_config=env_config, seed=0)

    assert world.is_escape_mode() is True
    assert world.is_collection_mode() is False
    assert env_config["reward_path"].endswith("configs/rewards_escape.yaml")
    assert len(world.all_enemies()) == 2
    assert len(world.support_enemies) == 1


def test_classic_ruleset_restores_original_multi_enemy_pressure():
    env_config = build_play_env_config(ruleset="classic", manual_collect_required=True)
    world = World(env_config=env_config, seed=0)

    assert len(world.all_enemies()) == 5
    assert len(world.support_enemies) == 4
    assert world.transition_metrics()["support_enemy_ratio"] == 4 / 5


def test_collectible_spawn_points_all_have_matching_surface_in_map():
    map_config = load_map_config()
    for spawn_x, spawn_y in map_config["collectible_spawn_points"]:
        covered = any(
            obstacle["x"] <= spawn_x < obstacle["x"] + obstacle["w"]
            and obstacle["y"] <= spawn_y < obstacle["y"] + obstacle["h"]
            for obstacle in map_config["obstacles"]
        )
        assert covered, (spawn_x, spawn_y)


def test_table_collectible_can_be_picked_from_table_edge():
    env_config = build_play_env_config(ruleset="classic", manual_collect_required=True)
    world = World(env_config=env_config, seed=0)
    target = next(item for item in world.collectibles if (int(item.x), int(item.y)) == (13, 4))

    world.player.x = 11.5
    world.player.y = 4.5
    world.enemy.x = 28.5
    world.enemy.y = 15.5
    world.enemy.facing_x = 1.0
    world.enemy.facing_y = 0.0

    hold_steps = math.ceil(world.collection_hold_seconds / world.physics_dt) + 1
    for _ in range(hold_steps):
        world.step(player_action=(0.0, 0.0), enemy_action=(0.0, 0.0), frame_skip=1, player_collect=True)
        if not target.active:
            break

    assert target.active is False
    assert world.score["note"] == 1


def test_same_table_collect_picks_nearest_item_not_fixed_kind_order():
    env_config = build_play_env_config(ruleset="classic", manual_collect_required=True)
    world = World(env_config=env_config, seed=0)
    for collectible in world.collectibles:
        collectible.active = False

    note = next(item for item in world.collectibles if item.kind == "note")
    coffee = next(item for item in world.collectibles if item.kind == "coffee")
    note.active = True
    note.x = 12.5
    note.y = 4.5
    coffee.active = True
    coffee.x = 13.5
    coffee.y = 4.5

    world.player.x = 13.5
    world.player.y = 3.5
    world.enemy.x = 28.5
    world.enemy.y = 15.5
    world.enemy.facing_x = 1.0
    world.enemy.facing_y = 0.0

    hold_steps = math.ceil(world.collection_hold_seconds / world.physics_dt) + 1
    for _ in range(hold_steps):
        world.step(player_action=(0.0, 0.0), enemy_action=(0.0, 0.0), frame_skip=1, player_collect=True)
        if not coffee.active:
            break

    assert coffee.active is False
    assert note.active is True
    assert world.player.coffee_timer > 0.0


def test_coffee_doubles_classic_collection_speed():
    env_config = build_play_env_config(ruleset="classic", manual_collect_required=True)
    world = World(env_config=env_config, seed=0)
    target = next(item for item in world.collectibles if (int(item.x), int(item.y)) == (13, 4))
    interact_cell = world.interaction_cells_for_collectible(target)[0]
    world.player.x = interact_cell[0] + 0.5
    world.player.y = interact_cell[1] + 0.5
    world.enemy.x = 28.5
    world.enemy.y = 15.5
    world.enemy.facing_x = 1.0
    world.enemy.facing_y = 0.0

    world.player.coffee_timer = 0.0
    world.step(player_action=(0.0, 0.0), enemy_action=(0.0, 0.0), frame_skip=1, player_collect=True)
    progress_without_coffee = world.collection_progress

    world.collection_progress = 0.0
    world.collection_target_index = None
    world.player.coffee_timer = float(world.env_config["world"]["coffee_duration_seconds"])
    world.step(player_action=(0.0, 0.0), enemy_action=(0.0, 0.0), frame_skip=1, player_collect=True)
    progress_with_coffee = world.collection_progress

    assert progress_with_coffee > progress_without_coffee
    multiplier = float(world.env_config["world"]["coffee_collection_multiplier"])
    assert math.isclose(progress_with_coffee, min(1.0, progress_without_coffee * multiplier), rel_tol=1e-9, abs_tol=1e-9)


def test_auto_collect_only_takes_nearest_item_first():
    env_config = build_play_env_config(ruleset="rl", manual_collect_required=False)
    world = World(env_config=env_config, seed=0)
    for collectible in world.collectibles:
        collectible.active = False

    note = next(item for item in world.collectibles if item.kind == "note")
    coffee = next(item for item in world.collectibles if item.kind == "coffee")
    note.active = True
    note.x = 5.7
    note.y = 5.5
    coffee.active = True
    coffee.x = 6.2
    coffee.y = 5.5
    world.player.x = 5.0
    world.player.y = 5.5
    world.enemy.x = 20.5
    world.enemy.y = 15.5

    world.step(player_action=(0.0, 0.0), enemy_action=(0.0, 0.0), frame_skip=1, player_collect=False)

    assert note.active is True
    assert coffee.active is True
    assert 0.0 < world.collection_progress < 1.0

    hold_steps = math.ceil(world.collection_hold_seconds / world.physics_dt) + 1
    for _ in range(hold_steps):
        world.step(player_action=(0.0, 0.0), enemy_action=(0.0, 0.0), frame_skip=1, player_collect=False)
        if not note.active:
            break

    assert note.active is False
    assert coffee.active is True


def test_classic_ai_baselines_patrol_and_collect_without_checkpoints():
    env_config = build_play_env_config(ruleset="classic", manual_collect_required=False)
    world = World(env_config=env_config, seed=0)
    player_controller = HeuristicPlayerController()
    enemy_controller = RuleBasedEnemyController()

    enemy_start_positions = {enemy.name: enemy.position for enemy in world.all_enemies()}
    for _ in range(600):
        world.step(
            player_action=player_controller.act(world),
            enemy_action=enemy_controller.act(world),
            frame_skip=1,
            player_collect=False,
        )

    moved_enemies = sum(
        1
        for enemy in world.all_enemies()
        if math.hypot(
            enemy.x - enemy_start_positions[enemy.name][0],
            enemy.y - enemy_start_positions[enemy.name][1],
        ) > 1.0
    )
    assert moved_enemies >= 4
    assert world.score["note"] >= 1


def test_classic_ai_collect_uses_hold_progress_instead_of_instant_pickup():
    env_config = build_play_env_config(ruleset="classic", manual_collect_required=True)
    world = World(env_config=env_config, seed=0)
    controller = HeuristicPlayerController()
    target = next(item for item in world.collectibles if (int(item.x), int(item.y)) == (13, 4))
    interact_cell = world.interaction_cells_for_collectible(target)[0]

    world.player.x = interact_cell[0] + 0.5
    world.player.y = interact_cell[1] + 0.5
    world.enemy.x = 28.5
    world.enemy.y = 15.5
    world.enemy.facing_x = 1.0
    world.enemy.facing_y = 0.0

    player_action = controller.act(world)
    assert player_action == (0.0, 0.0)
    assert controller.collect_pressed(world) is True

    world.step(
        player_action=player_action,
        enemy_action=(0.0, 0.0),
        frame_skip=1,
        player_collect=controller.collect_pressed(world),
    )

    assert target.active is True
    assert 0.0 < world.collection_progress < 1.0


def test_auto_collect_requires_waiting_in_range():
    env_config = build_play_env_config(ruleset="rl", manual_collect_required=False)
    world = World(env_config=env_config, seed=0)
    target = world.active_collectibles("note")[0]
    world.player.x = target.x
    world.player.y = target.y
    world.enemy.x = 28.5
    world.enemy.y = 15.5

    world.step(player_action=(0.0, 0.0), enemy_action=(0.0, 0.0), frame_skip=1, player_collect=False)

    assert target.active is True
    assert 0.0 < world.collection_progress < 1.0


def test_game_mode_defaults_reflect_latest_balance_tuning():
    collection_env = build_play_env_config(game_mode="collection", manual_collect_required=True)
    collection_world = World(env_config=collection_env, seed=0)
    assert collection_world.collection_hold_seconds == 0.85
    assert collection_world.support_enemy_count == 4
    assert collection_world.player.base_speed == 4.0
    assert collection_world.startup_grace_seconds == 1.0
    assert collection_world.player_max_turn_rate_deg == 780.0
    assert collection_world.enemy_max_turn_rate_deg == 660.0

    escape_env = build_play_env_config(game_mode="escape", manual_collect_required=False)
    escape_world = World(env_config=escape_env, seed=0)
    assert escape_world.support_enemy_count == 1
    assert escape_world.enemy.vision.range_cells == 5.3
    assert escape_world.player.base_speed == 4.55
    assert escape_world.startup_grace_seconds == 0.75
    assert escape_world.player_max_turn_rate_deg == 900.0
    assert escape_world.enemy_max_turn_rate_deg == 780.0


def test_turn_rate_limits_prevent_instant_90_degree_snap():
    env_config = build_play_env_config(game_mode="escape", manual_collect_required=False)
    world = World(env_config=env_config, seed=0)
    world.startup_grace_timer = 0.0
    world.player.facing_x = 1.0
    world.player.facing_y = 0.0
    world.enemy.facing_x = 1.0
    world.enemy.facing_y = 0.0

    world.step(player_action=(0.0, 1.0), enemy_action=(0.0, 1.0), frame_skip=1, player_collect=False)

    assert 0.0 < world.player.facing_y < 1.0
    assert 0.0 < world.player.facing_x < 1.0
    assert 0.0 < world.enemy.facing_y < 1.0
    assert 0.0 < world.enemy.facing_x < 1.0


def test_escape_support_enemy_patrol_leaves_small_local_loop_to_pressure_objective():
    env_config = build_play_env_config(game_mode="escape", manual_collect_required=False)
    world = World(env_config=env_config, seed=0)
    support_enemy = world.support_enemies[0]
    start_position = support_enemy.position

    for collectible in world.collectibles:
        collectible.active = False
    target_note = next(item for item in world.collectibles if item.kind == "note")
    target_note.active = True
    target_note.x = 2.5
    target_note.y = 5.5

    world.player.x = 30.5
    world.player.y = 16.5
    world.enemy.x = 5.5
    world.enemy.y = 5.5
    world.startup_grace_timer = 0.0

    for _ in range(220):
        world.step(player_action=(0.0, 0.0), enemy_action=(0.0, 0.0), frame_skip=1, player_collect=False)
        if support_enemy.x < 14.5 or support_enemy.y < 10.5:
            break

    assert support_enemy.x < start_position[0] - 1.0 or support_enemy.y < start_position[1] - 1.0


def test_collection_support_enemy_patrol_leaves_small_local_loop_to_pressure_collectibles():
    env_config = build_play_env_config(game_mode="collection", manual_collect_required=True)
    world = World(env_config=env_config, seed=0)
    support_enemy = world.support_enemies[0]
    start_position = support_enemy.position

    for collectible in world.collectibles:
        collectible.active = False
    target_exam = next(item for item in world.collectibles if item.kind == "exam")
    target_exam.active = True
    target_exam.x = 3.5
    target_exam.y = 5.5

    world.player.x = 30.5
    world.player.y = 16.5
    world.startup_grace_timer = 0.0

    for _ in range(220):
        world.step(player_action=(0.0, 0.0), enemy_action=(0.0, 0.0), frame_skip=1, player_collect=False)
        if support_enemy.x < 14.5 or support_enemy.y < 11.5:
            break

    assert support_enemy.x < start_position[0] - 0.8 or support_enemy.y < start_position[1] - 0.8


def test_startup_grace_prevents_immediate_opening_detection():
    env_config = build_play_env_config(game_mode="collection", manual_collect_required=True)
    world = World(env_config=env_config, seed=0)
    world.enemy.x = 5.5
    world.enemy.y = 5.5
    world.enemy.facing_x = 1.0
    world.enemy.facing_y = 0.0
    world.player.x = 7.0
    world.player.y = 5.5

    protected_steps = max(0, math.ceil(world.startup_grace_seconds / world.physics_dt) - 2)
    for _ in range(protected_steps):
        events = world.step(player_action=(0.0, 0.0), enemy_action=(0.0, 0.0), frame_skip=1, player_collect=False)
        assert events.detection_events == 0
        assert world.time_remaining > world.max_episode_seconds - world.enemy_detect_penalty_seconds

    detected = False
    for _ in range(8):
        events = world.step(player_action=(0.0, 0.0), enemy_action=(0.0, 0.0), frame_skip=1, player_collect=False)
        if events.detection_events > 0:
            detected = True
            break

    assert detected is True

"""Validation for the authored static, host-staged Bridge subset."""

import math
import struct


FLOAT_MAX = 3.4028234663852886e38
PLAYER_HALF_WIDTH = struct.unpack("f", struct.pack("f", 0.6))[0] / 2.0
PLAYER_HEIGHT = struct.unpack("f", struct.pack("f", 1.8))[0]


def _object(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"Bridge {label} requires exactly {', '.join(keys)}")
    return value


def _number(value, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"Bridge {label} must be finite")
    return value


def _intersects(lower, upper, other_lower, other_upper):
    return all(a < d and c < b for a, b, c, d in zip(lower, upper, other_lower, other_upper))


def _player_bounds(position):
    return ([position[0] - PLAYER_HALF_WIDTH, position[1], position[2] - PLAYER_HALF_WIDTH],
            [position[0] + PLAYER_HALF_WIDTH, position[1] + PLAYER_HEIGHT, position[2] + PLAYER_HALF_WIDTH])


def _solid(scene, cell):
    for fill in reversed(scene["blocks"]):
        if all(lower <= coordinate <= upper for lower, coordinate, upper in
               zip(fill["from"], cell, fill["to"])):
            return fill["block"] != "minecraft:air"
    return False


def validate_bridge(configuration, scene):
    required = {"mode", "goals", "void_y", "match", "task_rewards"}
    if (not isinstance(configuration, dict) or not required <= set(configuration)
            or set(configuration) - required - {"kits", "building", "terrain_generation"}):
        raise ValueError("Invalid Bridge configuration fields")
    bridge = configuration
    mode = _object(bridge["mode"], ("name", "version"), "mode")
    if mode["name"] != "bridge" or type(mode["version"]) is not int or mode["version"] != 1:
        raise ValueError("Unsupported Bridge mode")
    void_y = _number(bridge["void_y"], "void_y")
    origin, size = scene["bounds"]["min"], scene["bounds"]["size"]
    goals = bridge["goals"]
    if not isinstance(goals, list) or len(goals) != 2:
        raise ValueError("Bridge requires two goals")
    for goal in goals:
        _object(goal, ("min", "max"), "goal")
        for key in ("min", "max"):
            if not isinstance(goal[key], list) or len(goal[key]) != 3:
                raise ValueError("Bridge goal coordinates require three values")
            for coordinate in goal[key]:
                _number(coordinate, "goal coordinate")
        if any(lower >= upper or lower < start or upper > start + extent
               for lower, upper, start, extent in zip(goal["min"], goal["max"], origin, size)):
            raise ValueError("Bridge goal must be nonempty and inside arena bounds")
        if goal["min"][1] < void_y:
            raise ValueError("Bridge goal must be above the void")
        for x_coord in range(math.floor(goal["min"][0]), math.ceil(goal["max"][0])):
            for y_coord in range(math.floor(goal["min"][1]), math.ceil(goal["max"][1])):
                for z_coord in range(math.floor(goal["min"][2]), math.ceil(goal["max"][2])):
                    if _solid(scene, (x_coord, y_coord, z_coord)):
                        raise ValueError("Bridge goals must be clear of solid blocks")
    if _intersects(goals[0]["min"], goals[0]["max"], goals[1]["min"], goals[1]["max"]):
        raise ValueError("Bridge goals must not overlap")
    match = _object(bridge["match"], ("goals_to_win", "max_ticks", "cage_ticks",
                                     "persist_blocks_on_goal", "arrow_refill_ticks"), "match")
    for key in ("goals_to_win", "max_ticks", "cage_ticks", "arrow_refill_ticks"):
        minimum = 1 if key in ("goals_to_win", "max_ticks") else 0
        if type(match[key]) is not int or not minimum <= match[key] <= 1000000:
            raise ValueError(f"Bridge {key} must be an integer in {minimum}..1000000")
    if type(match["persist_blocks_on_goal"]) is not bool:
        raise ValueError("Bridge persist_blocks_on_goal must be boolean")
    rewards = bridge["task_rewards"]
    required_rewards = {"score", "win", "loss"}
    if (not isinstance(rewards, dict) or not required_rewards <= set(rewards)
            or set(rewards) - required_rewards - {"opponent_score", "toward", "away"}):
        raise ValueError("Invalid Bridge task_rewards fields")
    converted = {}
    for key, value in rewards.items():
        if abs(_number(value, f"reward {key}")) > FLOAT_MAX:
            raise ValueError("Bridge rewards must fit float32")
        if key in ("toward", "away") and not 0 <= value <= 1000:
            raise ValueError("Objective progress weights must be in 0..1000")
        converted[key] = struct.unpack("f", struct.pack("f", value))[0]
    opponent = converted.get("opponent_score", 0)
    if any(abs(score + opposing + outcome) > FLOAT_MAX
           for score in (0, converted["score"]) for opposing in (0, opponent)
           for outcome in (0, converted["win"], converted["loss"])):
        raise ValueError("Combined Bridge rewards must fit float32")
    player_bounds = [_player_bounds(player["position"]) for player in scene["players"]]
    if _intersects(*player_bounds[0], *player_bounds[1]):
        raise ValueError("Bridge spawns must not overlap")
    for player in scene["players"]:
        position = player["position"]
        lower, upper = _player_bounds(position)
        if (position[1] < void_y or upper[1] > origin[1] + size[1]
                or not player["on_ground"] or not _solid(scene, (math.floor(position[0]),
                    math.floor(position[1] - 1e-5), math.floor(position[2])))):
            raise ValueError("Bridge spawns must be supported, inside bounds and above the void")
        for x_coord in range(math.floor(lower[0]), math.ceil(upper[0])):
            for y_coord in range(math.floor(lower[1]), math.ceil(upper[1])):
                for z_coord in range(math.floor(lower[2]), math.ceil(upper[2])):
                    if _solid(scene, (x_coord, y_coord, z_coord)):
                        raise ValueError("Bridge spawn must be clear of solid blocks")
        if any(_intersects(lower, upper, goal["min"], goal["max"]) for goal in goals):
            raise ValueError("Bridge spawns must be outside both goals")
        if match["arrow_refill_ticks"] and "minecraft:bow" not in player["loadout"]["weapons"]:
            raise ValueError("Bridge arrow refill requires a bow for both players")
    if "kits" in bridge:
        kits = bridge["kits"]
        if not isinstance(kits, list) or len(kits) != 2:
            raise ValueError("Bridge requires two kits")
        swords = {f"minecraft:{material}_sword" for material in
                  ("wooden", "stone", "iron", "golden", "diamond", "netherite")}
        allowed = {"empty", "sword", "bow", "pickaxe", "blocks", "golden_apple"}
        for kit, player in zip(kits, scene["players"]):
            _object(kit, ("slots", "blocks", "golden_apples"), "kit")
            slots = kit["slots"]
            weapons = player["loadout"]["weapons"]
            if not isinstance(slots, list) or not 1 <= len(slots) <= 9 or len(slots) != len(weapons):
                raise ValueError("Bridge kit slots must match the scene loadout size")
            seen = set()
            for item, weapon in zip(slots, weapons):
                if not isinstance(item, str) or item not in allowed or (item != "empty" and item in seen):
                    raise ValueError("Bridge kit items must be supported and occupy distinct slots")
                seen.add(item)
                if not (weapon in swords if item == "sword" else
                        weapon == "minecraft:bow" if item == "bow" else weapon == "minecraft:air"):
                    raise ValueError("Bridge kit slot disagrees with the scene weapon loadout")
            for count, item in (("blocks", "blocks"), ("golden_apples", "golden_apple")):
                if type(kit[count]) is not int or not 0 <= kit[count] <= 64 or (kit[count] and item not in seen):
                    raise ValueError("Bridge item counts require corresponding slots and must be in 0..64")
    if "building" in bridge:
        if "kits" not in bridge:
            raise ValueError("Bridge building requires explicit kits")
        building = _object(bridge["building"], ("zones", "block_range", "entity_range", "hardness",
                                                "mining_speed", "correct_tool"), "building")
        for key in ("block_range", "entity_range"):
            value = _number(building[key], key)
            if value > 64 or (value <= 0 if key == "block_range" else value < 0):
                raise ValueError(f"Bridge {key} is outside native range")
        for key in ("hardness", "mining_speed"):
            value = _number(building[key], key)
            try:
                converted = struct.unpack("f", struct.pack("f", value))[0]
            except (OverflowError, struct.error):
                converted = float("inf")
            if not 0 < converted <= FLOAT_MAX:
                raise ValueError(f"Bridge {key} must be a positive finite float32")
        if type(building["correct_tool"]) is not bool:
            raise ValueError("Bridge correct_tool must be boolean")
        zones = building["zones"]
        if not isinstance(zones, list) or not 1 <= len(zones) <= 8:
            raise ValueError("Bridge requires one to eight build zones")
        for zone in zones:
            _object(zone, ("min", "max"), "build zone")
            for key in ("min", "max"):
                if not isinstance(zone[key], list) or len(zone[key]) != 3:
                    raise ValueError("Bridge build zone coordinates require three values")
                for coordinate in zone[key]:
                    _number(coordinate, "build zone coordinate")
            if any(lower >= upper or lower < start or upper > start + extent
                   for lower, upper, start, extent in zip(zone["min"], zone["max"], origin, size)):
                raise ValueError("Bridge build zone must be nonempty and inside arena bounds")
    if "terrain_generation" in bridge:
        if "building" not in bridge:
            raise ValueError("Bridge terrain generation requires building rules")
        generation = _object(bridge["terrain_generation"], ("center_x", "floor_y", "start_z", "end_z", "segments"),
                             "terrain generation")
        if (any(type(value) is not int or abs(value) > 1000000000 for value in generation.values())
                or not 1 <= generation["segments"] <= 16
                or not 1 <= generation["end_z"] - generation["start_z"] <= 10000):
            raise ValueError("Invalid Bridge terrain generation values")
        center, floor = generation["center_x"], generation["floor_y"]
        start, end = generation["start_z"], generation["end_z"]
        areas = [([center - 2, floor, start], [center, floor + 2, end]),
                 ([center + 1, floor, start], [center + 3, floor + 2, end])]
        for lower, upper in areas:
            if (any(lo < base or hi > base + extent for lo, hi, base, extent in zip(lower, upper, origin, size))
                    or any(_intersects(lower, upper, goal["min"], goal["max"]) for goal in goals)
                    or any(_intersects(lower, upper, *_player_bounds(player["position"])) for player in scene["players"])):
                raise ValueError("Generated Bridge terrain must be in bounds and clear of spawns and goals")
        for z_coord in range(start, end):
            if not _solid(scene, (center, floor, z_coord)):
                raise ValueError("Bridge terrain generation requires a protected center lane")
            for offset in (-2, -1, 1, 2):
                for height in (0, 1):
                    cell = (center + offset, floor + height, z_coord)
                    if (_solid(scene, cell) or not any(all(lo <= value and value + 1 <= hi
                            for lo, value, hi in zip(zone["min"], cell, zone["max"])) for zone in zones)):
                        raise ValueError("Generated Bridge terrain requires clear cells inside build zones")

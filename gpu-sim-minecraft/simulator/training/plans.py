"""Validated research plans and deterministic, component-owned episode sampling."""

from copy import deepcopy
import hashlib
from itertools import combinations, product
import json
import math
from pathlib import Path

import numpy as np

from simulator.training.contract import FEATURES, resolve_players, resolve_scenario
from simulator.training.environment import ROOT


EQUIPMENT_V3_FEATURES = (
    *FEATURES[:32],
    *(f"{actor}.{name}" for actor in ("self", "opponent")
      for name in ("has_shield", "weapon_damage_fraction", "weapon_speed_fraction",
                   "armor_defense_fraction", "armor_toughness_fraction")),
    *(f"self.weapon_slot_{slot}.{name}" for slot in range(9)
      for name in ("present", "is_axe", "damage_fraction", "speed_fraction")),
)


def fields(value, allowed, label):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"Unsupported {label} fields; expected {', '.join(allowed)}")
    return value


def positive(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def weights(entries):
    if not isinstance(entries, list) or not entries:
        raise ValueError("A weighted distribution must be a nonempty list")
    if any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("Weighted entries must be objects")
    values = np.array([finite(entry.get("weight", 1), "weight") for entry in entries], dtype=float)
    if np.any(values < 0) or not np.isfinite(values.sum()) or values.sum() <= 0:
        raise ValueError("Distribution weights must be nonnegative with a positive finite sum")
    return values / values.sum()


def draw(value, rng):
    if isinstance(value, dict):
        if "choices" in value:
            choices = value["choices"]
            return draw(choices[int(rng.choice(len(choices), p=weights(choices)))]["value"], rng)
        if "uniform" in value:
            return float(rng.uniform(*value["uniform"]))
        return {name: draw(item, rng) for name, item in value.items()}
    if isinstance(value, list):
        return [draw(item, rng) for item in value]
    return value


def _choices(value):
    if isinstance(value, dict):
        if set(value) != {"choices"}:
            raise ValueError("Equipment values must be literals or weighted choices")
        return [item for entry in value["choices"] for item in _choices(entry["value"])]
    return [value]


def _validate_equipment(scenario, loadout):
    paths = []
    if "weapons" in loadout:
        options = [_choices(value) for value in loadout["weapons"]]
        for index, candidates in enumerate(options):
            if any(not isinstance(item, str) for item in candidates):
                raise ValueError("Weapon choices must be Minecraft item IDs")
            if any((set(candidates) & set(other)) - {"minecraft:air"} for other in options[index + 1:]):
                raise ValueError("Weapon-slot choices must not produce duplicate items")
            paths.append((("weapons", index), candidates))
    for name in ("selected", "offhand"):
        if name in loadout:
            paths.append(((name,), _choices(loadout[name])))
    if "armor" in loadout:
        fields(loadout["armor"], ("boots", "leggings", "chestplate", "helmet"), "sampled armor")
        paths.extend((("armor", slot), _choices(value)) for slot, value in loadout["armor"].items())
    for arena in scenario["arenas"]:
        base = arena["environment"]
        initial = deepcopy(base)
        for player in initial["players"]:
            override = deepcopy(loadout)
            for path, candidates in paths:
                target = override
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = candidates[0]
            player["loadout"].update(override)
        resolve_scenario(initial)
        for path, candidates in paths:
            for candidate in candidates:
                test = deepcopy(initial)
                for player in test["players"]:
                    target = player["loadout"]
                    for key in path[:-1]:
                        target = target[key]
                    target[path[-1]] = candidate
                resolve_scenario(test)


def distribution(value):
    if isinstance(value, dict):
        if "choices" in value:
            fields(value, ("choices",), "choice distribution")
            weights(value["choices"])
            for entry in value["choices"]:
                fields(entry, ("value", "weight"), "weighted choice")
                if "value" not in entry:
                    raise ValueError("Weighted choice needs a value")
                distribution(entry["value"])
        elif "uniform" in value:
            fields(value, ("uniform",), "uniform distribution")
            interval = value["uniform"]
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("Uniform distribution needs [minimum, maximum]")
            if finite(interval[0], "minimum") > finite(interval[1], "maximum"):
                raise ValueError("Uniform minimum exceeds maximum")
        else:
            for item in value.values():
                distribution(item)
    elif isinstance(value, list):
        for item in value:
            distribution(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite distribution value")


def _angle(value, pitch=False):
    if isinstance(value, dict):
        if "uniform" in value:
            candidates = value["uniform"]
        elif "choices" in value:
            candidates = [entry["value"] for entry in value["choices"]]
        else:
            raise ValueError("Spawn look needs a literal, uniform interval or weighted choices")
        for candidate in candidates:
            _angle(candidate, pitch)
    else:
        finite(value, "spawn look angle")
        if pitch and not -90 <= value <= 90:
            raise ValueError("Spawn pitch must be in [-90,90]")


def advance_rule(rule, evaluations):
    fields(rule, ("updates", "episodes", "transitions", "evaluation", "any", "all"), "advancement")
    if len(rule) != 1:
        raise ValueError("Advancement must specify one budget or an explicit any/all group")
    key, value = next(iter(rule.items()))
    if key in ("any", "all"):
        if not isinstance(value, list) or not value:
            raise ValueError("Advancement groups must not be empty")
        for child in value:
            advance_rule(child, evaluations)
    elif key == "evaluation":
        fields(value, ("suite", "metric", "at_least"), "evaluation advancement")
        if (value.get("suite") not in evaluations or value.get("metric") not in ("win_rate", "reward_mean")
                or "at_least" not in value):
            raise ValueError("Evaluation advancement needs an existing suite, metric and threshold")
        finite(value["at_least"], "evaluation threshold")
        if value["metric"] == "win_rate" and not 0 <= value["at_least"] <= 1:
            raise ValueError("Win rate threshold must be in [0,1]")
    else:
        positive(value, key)


def reached(rule, progress, evaluation):
    key, value = next(iter(rule.items()))
    if key == "any":
        return any(reached(child, progress, evaluation) for child in value)
    if key == "all":
        return all(reached(child, progress, evaluation) for child in value)
    if key == "evaluation":
        result = evaluation.get(value["suite"])
        return result is not None and result[value["metric"]] >= value["at_least"]
    return progress[key] >= value


def _scenario(value):
    fields(value, ("base", "arenas", "spawns", "equipment"), "scenario")
    scenario = deepcopy(value)
    if ("base" in scenario) == ("arenas" in scenario):
        raise ValueError("Scenario needs either base or a weighted arenas list")
    if "base" in scenario:
        scenario["arenas"] = [{"base": scenario.pop("base"), "weight": 1}]
    weights(scenario["arenas"])
    for arena in scenario["arenas"]:
        fields(arena, ("base", "environment", "weight"), "arena")
        if ("base" in arena) == ("environment" in arena):
            raise ValueError("Arena needs one environment or base path")
        if "base" in arena:
            path = Path(arena.pop("base"))
            path = path if path.is_absolute() else ROOT / path
            arena["environment"] = json.loads(path.read_text(encoding="utf-8"))
        arena["environment"] = resolve_scenario(arena["environment"])
    for component in ("spawns", "equipment"):
        if scenario.get(component, False) is None:
            scenario.pop(component)
    if "spawns" in scenario:
        spawn = fields(scenario["spawns"], ("sample", "region", "minimum_separation", "yaw", "pitch"), "spawns")
        if spawn.get("sample") not in ("world", "episode"):
            raise ValueError("Spawn sampling must be world or episode")
        if "region" in spawn:
            region = fields(spawn["region"], ("min", "max"), "spawn region")
            for key in ("min", "max"):
                if not isinstance(region.get(key), list) or len(region[key]) != 3:
                    raise ValueError("Spawn region needs three-dimensional min/max")
                for number in region[key]:
                    finite(number, "spawn coordinate")
            if any(a > b for a, b in zip(region["min"], region["max"])):
                raise ValueError("Spawn region minimum exceeds maximum")
        if finite(spawn.get("minimum_separation", 0.6), "minimum_separation") < 0.6:
            raise ValueError("Minimum spawn separation must be at least 0.6")
        for key in ("yaw", "pitch"):
            if key in spawn:
                distribution(spawn[key])
                _angle(spawn[key], pitch=key == "pitch")
    if "equipment" in scenario:
        equipment = fields(scenario["equipment"], ("sample", "players", "loadout"), "equipment")
        if equipment.get("sample") not in ("world", "episode") or equipment.get("players") != "independent":
            raise ValueError("Equipment needs world/episode sampling and independent players")
        loadout = fields(equipment.get("loadout"), ("weapons", "selected", "offhand", "armor"), "sampled loadout")
        if "weapons" in loadout and (not isinstance(loadout["weapons"], list) or not 1 <= len(loadout["weapons"]) <= 9):
            raise ValueError("Sampled loadout needs 1..9 stable weapon slots")
        if "weapons" in loadout:
            loadout["weapons"].extend(["minecraft:air"] * (9 - len(loadout["weapons"])))
        distribution(loadout)
        _validate_equipment(scenario, loadout)
    return scenario


def resolve_plan(value, base_dir=ROOT):
    """Compile authored v2 input, or validate a self-contained resolved experiment."""
    from simulator.training.compiler import resolve_experiment
    return resolve_experiment(value, base_dir)


def load_plan(path):
    path = Path(path).resolve()
    return resolve_plan(json.loads(path.read_text(encoding="utf-8")), base_dir=path.parent)


def _opponent_mix(mix, opponents):
    weights(mix)
    for entry in mix:
        fields(entry, ("id", "weight"), "opponent mixture")
        if entry.get("id") not in opponents:
            raise ValueError("Mixture references an unknown opponent")


def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def scene_key(scenario):
    """Identity of immutable terrain and assignment weights, excluding task/actors."""
    return plan_hash([{"weight": arena.get("weight", 1), **{key: arena["environment"].get(key)
                       for key in ("bounds", "blocks", "rules")}} for arena in scenario["arenas"]])


class EpisodeSampler:
    def __init__(self, scenario, count, seed):
        self.scenario = _scenario(scenario)
        streams = np.random.SeedSequence(seed).spawn(3)
        self.arena_rng, self.spawn_rng, self.gear_rng = [np.random.default_rng(stream) for stream in streams]
        arenas = self.scenario["arenas"]
        self.arenas = [int(value) for value in np.asarray(
            self.arena_rng.choice(len(arenas), count, p=weights(arenas))).reshape(-1)] if len(arenas) > 1 else [0] * count
        self.cache = [{"spawns": None, "equipment": None} for _ in range(count)]

    def configure(self, scenario):
        scenario = _scenario(scenario)
        if scene_key(scenario) != scene_key(self.scenario):
            raise ValueError("Changing arenas requires a new scene assignment")
        for component in ("spawns", "equipment"):
            keys = ("position", "yaw", "pitch") if component == "spawns" else ("loadout",)
            def defaults(value):
                return [[{key: player.get(key) for key in keys} for player in arena["environment"]["players"]]
                        for arena in value["arenas"]]
            if (scenario.get(component) != self.scenario.get(component)
                    or defaults(scenario) != defaults(self.scenario)):
                for cache in self.cache:
                    cache[component] = None
        self.scenario = deepcopy(scenario)

    def players(self, world):
        base = self.scenario["arenas"][self.arenas[world]]["environment"]
        players = [{**player, "loadout": {**player["loadout"], "armor": dict(player["loadout"]["armor"])}}
                   for player in base["players"]]
        cache = self.cache[world]
        gear = self.scenario.get("equipment")
        if gear:
            if gear["sample"] == "episode" or cache["equipment"] is None:
                cache["equipment"] = [draw(gear["loadout"], self.gear_rng) for _ in range(2)]
            for player, loadout in zip(players, cache["equipment"]):
                player["loadout"].update({name: value for name, value in loadout.items() if name != "armor"})
                player["loadout"]["armor"].update(loadout.get("armor", {}))
        for player in players:
            slots = list(player["loadout"]["weapons"])
            slots.extend(["minecraft:air"] * (9 - len(slots)))
            player["loadout"]["weapons"] = slots
        spawn = self.scenario.get("spawns")
        if spawn:
            if spawn["sample"] == "episode" or cache["spawns"] is None:
                poses = [{key: deepcopy(player[key]) for key in ("position", "yaw", "pitch") if key in player}
                         for player in players]
                if "region" in spawn:
                    minimum = spawn.get("minimum_separation", 0.6)
                    for _ in range(256):
                        positions = self.spawn_rng.uniform(spawn["region"]["min"], spawn["region"]["max"],
                                                           size=(2, 3)).tolist()
                        for pose, position in zip(poses, positions):
                            pose["position"] = position
                        if (self._separated([pose["position"] for pose in poses], minimum)
                                and all(self._valid_position(pose["position"], base) for pose in poses)):
                            break
                    else:
                        axes = [sorted({low, high}) for low, high in zip(spawn["region"]["min"], spawn["region"]["max"])]
                        corners = [list(position) for position in product(*axes) if self._valid_position(position, base)]
                        pairs = [pair for pair in combinations(corners, 2) if self._separated(pair, minimum)]
                        if not pairs:
                            raise ValueError("Could not sample collision-free separated spawns after 256 attempts "
                                             f"and corner fallback (world {world}, region {spawn['region']}, "
                                             f"minimum_separation {minimum})")
                        pair = pairs[int(self.spawn_rng.integers(len(pairs)))]
                        for pose, index in zip(poses, self.spawn_rng.permutation(2)):
                            pose["position"] = list(pair[index])
                for pose in poses:
                    for key in ("yaw", "pitch"):
                        if key in spawn:
                            pose[key] = draw(spawn[key], self.spawn_rng)
                cache["spawns"] = poses
            for player, pose in zip(players, cache["spawns"]):
                player.update(deepcopy(pose))
                if "region" in spawn:
                    x, y, z = player["position"]
                    player["on_ground"] = not self._valid_position([x, y - 1e-5, z], base)
        return resolve_players(players)

    @staticmethod
    def _separated(positions, minimum):
        delta_x = abs(positions[0][0] - positions[1][0])
        delta_y = abs(positions[0][1] - positions[1][1])
        delta_z = abs(positions[0][2] - positions[1][2])
        return ((delta_x >= 0.6 or delta_z >= 0.6 or delta_y >= 1.8)
            and math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z) >= minimum)

    @staticmethod
    def _valid_position(position, base):
        origin, size = base["bounds"]["min"], base["bounds"]["size"]
        if not origin[1] <= position[1] < origin[1] + size[1]:
            return False
        if any(position[axis] - 0.3 < origin[axis] or position[axis] + 0.3 >= origin[axis] + size[axis]
               for axis in (0, 2)):
            return False
        low = [math.floor(position[0] - 0.3), math.floor(position[1]), math.floor(position[2] - 0.3)]
        high = [math.floor(position[0] + 0.3 - 1e-9), math.floor(position[1] + 1.8 - 1e-9),
                math.floor(position[2] + 0.3 - 1e-9)]
        for x in range(low[0], high[0] + 1):
            for y in range(low[1], high[1] + 1):
                for z in range(low[2], high[2] + 1):
                    for fill in reversed(base["blocks"]):
                        if all(a <= b <= c for a, b, c in zip(fill["from"], (x, y, z), fill["to"])):
                            if fill["block"] != "minecraft:air":
                                return False
                            break
        return True

    def state_dict(self):
        return {"arenas": list(self.arenas), "cache": deepcopy(self.cache),
                "rng": [deepcopy(rng.bit_generator.state) for rng in (self.arena_rng, self.spawn_rng, self.gear_rng)]}

    def load_state_dict(self, state):
        fields(state, ("arenas", "cache", "rng"), "sampler state")
        if (not isinstance(state.get("arenas"), list) or not isinstance(state.get("cache"), list)
                or len(state["arenas"]) != len(self.arenas) or len(state["cache"]) != len(self.cache)):
            raise ValueError("Sampler state world count differs")
        if any(type(index) is not int or not 0 <= index < len(self.scenario["arenas"]) for index in state["arenas"]):
            raise ValueError("Sampler state contains an invalid arena index")
        if not isinstance(state.get("rng"), list) or len(state["rng"]) != 3:
            raise ValueError("Sampler state needs three component RNG states")
        for cache in state["cache"]:
            fields(cache, ("spawns", "equipment"), "sampler cache")
            if set(cache) != {"spawns", "equipment"}:
                raise ValueError("Sampler cache needs spawn and equipment entries")
            for component in ("spawns", "equipment"):
                if cache[component] is not None and (
                        not isinstance(cache[component], list) or len(cache[component]) != 2
                        or any(not isinstance(item, dict) for item in cache[component])):
                    raise ValueError("Sampler cache must contain two player samples")
        generators = [np.random.default_rng() for _ in range(3)]
        for rng, saved in zip(generators, state["rng"]):
            try:
                rng.bit_generator.state = deepcopy(saved)
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError("Invalid sampler RNG state") from error
        self.arenas = list(state["arenas"])
        self.cache = deepcopy(state["cache"])
        self.arena_rng, self.spawn_rng, self.gear_rng = generators

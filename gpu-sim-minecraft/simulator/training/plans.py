"""Validated research plans and deterministic, component-owned episode sampling."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from simulator.training.contract import FEATURES, from_legacy, resolve_scenario, validate_contract
from simulator.training.environment import ROOT, resolve_reward_weights
from simulator.training.legacy import SCHEMAS
from simulator.training.policies import fixed_indices


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
            if any(set(candidates) & set(other) for other in options[index + 1:]):
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
        distribution(loadout)
        _validate_equipment(scenario, loadout)
    return scenario


def resolve_plan(value):
    fields(value, ("format", "version", "seed", "policy", "scenario", "opponents", "stages", "evaluations"), "training plan")
    if value.get("format") != "minecraft_sim_training_plan" or value.get("version") != 1:
        raise ValueError("Unsupported training plan format/version")
    plan = deepcopy(value)
    seed = plan.get("seed", 1)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Plan seed must be in [0,2**63)")
    plan["seed"] = seed
    plan["scenario"] = _scenario(plan["scenario"])
    opponents = plan.get("opponents")
    if not isinstance(opponents, dict) or not opponents:
        raise ValueError("Plan needs named opponents")
    for name, opponent in opponents.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Opponent IDs must be nonempty strings")
        fields(opponent, ("type", "controller", "options", "path", "deterministic", "sha256"), "opponent")
        if opponent.get("type") == "checkpoint":
            path = Path(opponent["path"])
            path = path if path.is_absolute() else ROOT / path
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if opponent.get("sha256", digest) != digest:
                raise ValueError(f"Opponent checkpoint changed: {path}")
            opponent.update(path=str(path.resolve()), sha256=digest)
        elif opponent.get("type") == "scripted":
            if not isinstance(opponent.get("controller"), str):
                raise ValueError("Scripted opponent needs a controller name")
        else:
            raise ValueError("Opponent type must be scripted or checkpoint")
    evaluations = plan.setdefault("evaluations", {})
    if not isinstance(evaluations, dict):
        raise ValueError("Evaluations must be named suites")
    stages = plan.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Plan needs at least one stage")
    names = set()
    for stage in stages:
        fields(stage, ("name", "scenario", "opponents", "controls", "rewards", "advance"), "stage")
        if not isinstance(stage.get("name"), str) or not stage["name"] or stage["name"] in names:
            raise ValueError("Stage names must be nonempty and unique")
        names.add(stage["name"])
        if "scenario" in stage:
            changes = stage["scenario"]
            fields(changes, ("base", "arenas", "spawns", "equipment"), "stage scenario")
            base = deepcopy(plan["scenario"])
            if "base" in changes or "arenas" in changes:
                base.pop("arenas")
            stage["scenario"] = _scenario({**base, **changes})
        else:
            stage["scenario"] = deepcopy(plan["scenario"])
        _opponent_mix(stage.get("opponents"), opponents)
        stage.setdefault("controls", {"fixed": {}})
        reward = fields(stage.setdefault("rewards", {"mode": "environment"}), ("mode", "weights"), "stage rewards")
        if reward.get("mode") not in ("environment", "replace", "override"):
            raise ValueError("Reward mode must be environment, replace or override")
        if reward["mode"] == "environment" and "weights" in reward:
            raise ValueError("Environment reward mode cannot contain weights")
        resolved = resolve_reward_weights({}, reward.get("weights", {}))
        if not all(math.isfinite(number) for number in resolved.values()):
            raise ValueError("Reward weights must be finite")
        advance_rule(stage["advance"], evaluations)
    for suite in evaluations.values():
        fields(suite, ("every_updates", "episodes", "seed", "scenario", "opponents", "controls"), "evaluation suite")
        positive(suite.get("every_updates"), "evaluation interval")
        positive(suite.get("episodes"), "evaluation episodes")
        if type(suite.get("seed")) is not int or not 0 <= suite["seed"] < 2**63:
            raise ValueError("Evaluation seed must be in [0,2**63)")
        suite["scenario"] = _scenario(suite.get("scenario", plan["scenario"]))
        suite.setdefault("controls", {"fixed": {}})
        _opponent_mix(suite.get("opponents"), opponents)
    policy = plan["policy"]
    if isinstance(policy, dict) and set(policy) == {"preset"}:
        if policy["preset"] in SCHEMAS:
            policy = from_legacy(SCHEMAS[policy["preset"]])
        elif policy["preset"] == "equipment_v3":
            slot_counts = set()
            for stage in [*stages, *evaluations.values()]:
                scenario = stage["scenario"]
                for arena in scenario["arenas"]:
                    for player in arena["environment"]["players"]:
                        loadout = scenario.get("equipment", {}).get("loadout", {})
                        slot_counts.add(len(loadout.get("weapons", player["loadout"]["weapons"])))
            if len(slot_counts) != 1:
                raise ValueError("equipment_v3 needs one stable weapon-slot count across the plan")
            slots = slot_counts.pop()
            policy = from_legacy(SCHEMAS["v2"])
            policy.pop("observation_dim")
            policy.pop("action_sizes")
            policy["observations"] = [{"feature": feature} for feature in EQUIPMENT_V3_FEATURES]
            policy["requires"] = sorted(set(policy["requires"]) | {"armor"})
            if any(feature not in FEATURES for feature in EQUIPMENT_V3_FEATURES):
                raise ValueError("equipment_v3 requires the updated equipment catalogue")
            next(action for action in policy["actions"] if action["name"] == "weapon")["values"] = list(range(slots))
        else:
            raise ValueError("Unknown policy preset")
    plan["policy"] = validate_contract(policy)
    for stage in stages:
        fixed_indices(plan["policy"], stage["controls"])
    for suite in evaluations.values():
        fixed_indices(plan["policy"], suite["controls"])
    return plan


def _opponent_mix(mix, opponents):
    weights(mix)
    for entry in mix:
        fields(entry, ("id", "weight"), "opponent mixture")
        if entry.get("id") not in opponents:
            raise ValueError("Mixture references an unknown opponent")


def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class EpisodeSampler:
    def __init__(self, scenario, count, seed):
        self.scenario = deepcopy(scenario)
        streams = np.random.SeedSequence(seed).spawn(3)
        self.arena_rng, self.spawn_rng, self.gear_rng = [np.random.default_rng(stream) for stream in streams]
        arenas = scenario["arenas"]
        self.arenas = [int(value) for value in np.asarray(
            self.arena_rng.choice(len(arenas), count, p=weights(arenas))).reshape(-1)] if len(arenas) > 1 else [0] * count
        self.cache = [{"spawns": None, "equipment": None} for _ in range(count)]

    def configure(self, scenario):
        if scenario["arenas"] != self.scenario["arenas"]:
            raise ValueError("Changing arenas requires a new scene assignment")
        for component in ("spawns", "equipment"):
            if scenario.get(component) != self.scenario.get(component):
                for cache in self.cache:
                    cache[component] = None
        self.scenario = deepcopy(scenario)

    def players(self, world):
        base = self.scenario["arenas"][self.arenas[world]]["environment"]
        players = deepcopy(base["players"])
        cache = self.cache[world]
        gear = self.scenario.get("equipment")
        if gear:
            if gear["sample"] == "episode" or cache["equipment"] is None:
                cache["equipment"] = [draw(gear["loadout"], self.gear_rng) for _ in range(2)]
            for player, loadout in zip(players, cache["equipment"]):
                player["loadout"].update(deepcopy(loadout))
        spawn = self.scenario.get("spawns")
        if spawn:
            if spawn["sample"] == "episode" or cache["spawns"] is None:
                poses = [{key: deepcopy(player[key]) for key in ("position", "yaw", "pitch") if key in player}
                         for player in players]
                if "region" in spawn:
                    for _ in range(256):
                        for pose in poses:
                            pose["position"] = self.spawn_rng.uniform(spawn["region"]["min"], spawn["region"]["max"]).tolist()
                        delta = np.abs(np.subtract(poses[0]["position"], poses[1]["position"]))
                        separated = delta[0] >= 0.6 or delta[2] >= 0.6 or delta[1] >= 1.8
                        if (separated and np.linalg.norm(delta) >= spawn.get("minimum_separation", 0.6)
                                and all(self._valid_position(pose["position"], base) for pose in poses)):
                            break
                    else:
                        raise ValueError("Could not sample collision-free separated spawns after 256 attempts")
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
        return resolve_scenario({**base, "players": players})["players"]

    @staticmethod
    def _valid_position(position, base):
        origin, size = base["bounds"]["min"], base["bounds"]["size"]
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
        return {"arenas": self.arenas, "cache": deepcopy(self.cache),
                "rng": [deepcopy(rng.bit_generator.state) for rng in (self.arena_rng, self.spawn_rng, self.gear_rng)]}

    def load_state_dict(self, state):
        if len(state["arenas"]) != len(self.arenas) or len(state["cache"]) != len(self.cache):
            raise ValueError("Sampler state world count differs")
        self.arenas = list(state["arenas"])
        self.cache = deepcopy(state["cache"])
        for rng, saved in zip((self.arena_rng, self.spawn_rng, self.gear_rng), state["rng"]):
            rng.bit_generator.state = deepcopy(saved)

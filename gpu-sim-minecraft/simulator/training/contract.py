"""Data-driven policy layouts over a bounded catalogue of simulator primitives."""

from copy import deepcopy
import math
import hashlib
import json
from pathlib import Path

CATALOGUE = json.loads((Path(__file__).resolve().parents[2] / "contracts/policy_catalogue.json").read_text(encoding="utf-8"))
CATALOGUE_HASH = hashlib.sha256(json.dumps(CATALOGUE, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
FEATURES = tuple(CATALOGUE["features"])
MOVEMENTS = tuple(CATALOGUE["movements"])
HEADS = tuple(action["name"] for action in CATALOGUE["actions"])
ACTION_KINDS = {action["name"]: action["kind"] for action in CATALOGUE["actions"]}
CAPABILITIES = tuple(CATALOGUE["capabilities"])
LIMITS = CATALOGUE["limits"]
RULES = {"simulate_food": False, "natural_regeneration": False}
_EQUIPMENT_MATERIALS = ("leather", "copper", "chainmail", "iron", "golden", "diamond", "netherite")
_WEAPONS = frozenset({"minecraft:air", *(f"minecraft:{material}_{kind}" for material in _EQUIPMENT_MATERIALS
    if material not in ("leather", "chainmail") for kind in ("sword", "axe")),
    "minecraft:wooden_sword", "minecraft:wooden_axe", "minecraft:stone_sword", "minecraft:stone_axe"})
_LOADOUT_FIELDS = frozenset(("weapons", "selected", "offhand", "armor"))
_ARMOR_SLOTS = tuple(CATALOGUE["armor_slots"])
_ARMOR_ITEMS = {slot: frozenset({"minecraft:air", *(f"minecraft:{material}_{slot}"
    for material in _EQUIPMENT_MATERIALS), *(["minecraft:turtle_helmet"] if slot == "helmet" else [])})
    for slot in _ARMOR_SLOTS}


def categorical_values(feature):
    if feature in ("self.weapon_slot", "opponent.weapon_slot"):
        return tuple(range(LIMITS["weapon_slots"]))
    if feature in FEATURES and feature.endswith((".alive", ".weapon_is_axe", ".shield_using",
            ".shield_disabled", ".shield_disabled_observed", ".has_shield", ".present", ".is_axe", ".on_ground", ".sprinting")):
        return (0, 1)
    return ()


def observation_width(observation):
    return len(observation["categories"]) if observation.get("encoding") == "one_hot" else 1


def project_observations(catalogue, contract):
    result = []
    for observation in contract["observations"]:
        value = catalogue[FEATURES.index(observation["feature"])]
        scale = observation.get("scale", 1.0)
        if observation.get("encoding") == "one_hot":
            result.extend(float(value == category) * scale for category in observation["categories"])
        else:
            if "clip" in observation:
                lower, upper = observation["clip"]
                value = min(max(value, lower), upper)
            result.append(value * scale)
    return result


def validate_contract(value):
    contract = deepcopy(value)
    if (not isinstance(contract, dict) or contract.get("format") != "minecraft_sim_policy"
            or contract.get("version") != 1 or contract.get("minecraft") != "26.3"
            or contract.get("ticks_per_action") != 1):
        raise ValueError("Unsupported policy contract format/version/build/tick rate")
    observations = contract.get("observations")
    actions = contract.get("actions")
    if not isinstance(observations, list) or not 1 <= len(observations) <= LIMITS["observations"]:
        raise ValueError("Policy needs 1..128 observations")
    for observation in observations:
        if not isinstance(observation, dict) or observation.get("feature") not in FEATURES:
            raise ValueError(f"Unsupported observation primitive: {observation}")
        if set(observation) - {"feature", "scale", "encoding", "categories", "clip"}:
            raise ValueError("Unsupported observation descriptor fields")
        scale = observation.get("scale", 1.0)
        if type(scale) not in (int, float) or not math.isfinite(scale) or abs(scale) > LIMITS["scale"]:
            raise ValueError("Observation scale must be finite and within -1000..1000")
        observation["scale"] = float(scale)
        encoding = observation.get("encoding", "scalar")
        if encoding == "scalar":
            observation.pop("encoding", None)
            if "categories" in observation:
                raise ValueError("Observation categories require one_hot encoding")
            if "clip" in observation:
                bounds = observation["clip"]
                if (not isinstance(bounds, list) or len(bounds) != 2
                        or any(type(bound) not in (int, float) or not math.isfinite(bound)
                               or abs(bound) > 3.4028234663852886e38 for bound in bounds)
                        or bounds[0] >= bounds[1]):
                    raise ValueError("Observation clip needs two finite increasing float32 bounds")
                observation["clip"] = [float(bound) for bound in bounds]
        elif encoding == "one_hot":
            categories = observation.get("categories")
            allowed = categorical_values(observation["feature"])
            if (not allowed or not isinstance(categories, list) or not categories
                    or any(type(category) is not int or category not in allowed for category in categories)
                    or len(set(categories)) != len(categories)):
                raise ValueError("one_hot needs distinct integer categories for a slot or binary feature")
            if "clip" in observation:
                raise ValueError("Observation clip is only supported for scalar encoding")
        else:
            raise ValueError("Observation encoding must be scalar or one_hot")
    observation_dim = sum(observation_width(observation) for observation in observations)
    if observation_dim > LIMITS["observations"]:
        raise ValueError("Expanded policy needs at most 128 observation inputs")
    if not isinstance(actions, list) or not 1 <= len(actions) <= len(HEADS):
        raise ValueError("Policy needs 1..9 action heads")
    names = []
    sizes = []
    for action in actions:
        if not isinstance(action, dict) or action.get("name") not in HEADS or action["name"] in names:
            raise ValueError("Unsupported or duplicate action primitive")
        name, values = action["name"], action.get("values")
        if not isinstance(values, list) or not 1 <= len(values) <= LIMITS["choices"] or len({str(item) for item in values}) != len(values):
            raise ValueError("Action values must contain 1..64 distinct choices")
        kind = ACTION_KINDS[name]
        if kind == "movement":
            valid = all(isinstance(item, str) and item in MOVEMENTS for item in values)
        elif kind == "look":
            valid = all(type(item) in (float, int) and math.isfinite(item) and abs(item) <= LIMITS["look_delta"] for item in values)
        elif kind == "weapon":
            valid = all(type(item) is int and 0 <= item < LIMITS["weapon_slots"] for item in values)
        else:
            valid = all(type(item) is bool for item in values)
        if not valid:
            raise ValueError(f"Invalid choices for action primitive {name}")
        names.append(name)
        sizes.append(len(values))
    required = contract.get("requires", [])
    if not isinstance(required, list) or any(item not in CAPABILITIES for item in required):
        raise ValueError("Policy requires unsupported simulator capabilities")
    for key, expected in (("observation_dim", observation_dim), ("action_names", names), ("action_sizes", sizes)):
        if key in contract and contract[key] != expected:
            raise ValueError(f"Policy contract {key} disagrees with its descriptors")
        contract[key] = expected
    contract["requires"] = sorted(set(required))
    return contract


def from_legacy(schema):
    actions = []
    for name, size in zip(schema["action_names"], schema["action_sizes"]):
        if name == "movement":
            values = schema["movement"]
        elif name in ("yaw", "pitch"):
            values = schema[f"{name}_bins"]
        elif name == "weapon":
            values = list(range(size))
        else:
            values = [False, True]
        actions.append({"name": name, "values": values})
    return validate_contract({"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
        "ticks_per_action": 1, "observations": [{"feature": name} for name in FEATURES[:schema["observation_dim"]]],
        "actions": actions, "requires": ["movement", "melee", *(["equipment", "shield"] if "weapon" in schema["action_names"] else [])]})


def fingerprint(contract):
    return hashlib.sha256(json.dumps(validate_contract(contract), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def resolve_players(players, profile="configured"):
    players = deepcopy(players)
    if not isinstance(players, list) or len(players) != 2:
        raise ValueError("Environment needs two players")
    if profile not in ("configured", "unarmored_diamond_sword", "unarmored_sword_axe_shield"):
        raise ValueError("Unsupported environment profile")
    default = {"weapons": ["minecraft:diamond_sword"], "offhand": "minecraft:air"}
    if profile == "unarmored_sword_axe_shield":
        default = {"weapons": ["minecraft:diamond_sword", "minecraft:diamond_axe"], "offhand": "minecraft:shield"}
    for player in players:
        loadout = player.get("loadout", default)
        if not isinstance(loadout, dict) or set(loadout) - _LOADOUT_FIELDS:
            raise ValueError("Unsupported loadout fields")
        slots = loadout.get("weapons", ["minecraft:diamond_sword"])
        selected = loadout.get("selected", 0)
        if (not isinstance(slots, list) or not 1 <= len(slots) <= 9 or any(not isinstance(item, str) or item not in _WEAPONS for item in slots)
                or len({item for item in slots if item != "minecraft:air"}) != sum(item != "minecraft:air" for item in slots)
                or type(selected) is not int or not 0 <= selected < len(slots)):
            raise ValueError("Unsupported weapon loadout or selected slot")
        offhand = loadout.get("offhand", "minecraft:air")
        if offhand not in ("minecraft:air", "minecraft:shield"):
            raise ValueError("Unsupported offhand item")
        armor = loadout.get("armor", {})
        if not isinstance(armor, dict) or set(armor) - _ARMOR_ITEMS.keys():
            raise ValueError("Unsupported armor slot")
        armor = {slot: armor.get(slot, "minecraft:air") for slot in _ARMOR_SLOTS}
        for slot, item in armor.items():
            if item not in _ARMOR_ITEMS[slot]:
                raise ValueError(f"Unsupported armor item: {item}")
        player["loadout"] = {"weapons": list(slots), "selected": selected, "offhand": offhand, "armor": armor}
    json.dumps(players, allow_nan=False)
    return players


def resolve_scenario(definition):
    scenario = deepcopy(definition)
    if (scenario.get("format") != "minecraft_sim_environment" or scenario.get("version") != 1
            or scenario.get("minecraft") != "26.3" or len(scenario.get("players", [])) != 2):
        raise ValueError("Unsupported environment format/version/build/player count")
    profile = scenario.get("profile", "configured")
    for capability in scenario.get("requires", []):
        if capability not in CAPABILITIES:
            raise ValueError(f"Unsupported scenario capability: {capability}")
    rules = scenario.get("rules", {})
    if (not isinstance(rules, dict) or set(rules) - set(RULES)
            or any(type(value) is not bool or value is not False for value in rules.values())):
        raise ValueError("Scenario bridge currently requires frozen food and no natural regeneration")
    scenario["rules"] = dict(RULES)
    scenario["players"] = resolve_players(scenario["players"], profile)
    scenario["profile"] = "configured"
    scenario["requires"] = list(CAPABILITIES)
    scenario.pop("policy", None)
    json.dumps(scenario, allow_nan=False)
    return scenario
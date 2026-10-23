"""Compile separate authored arenas, tasks and policies into portable experiments.

Only this boundary reads authoring references. Resolved experiments contain native
v1 scene envelopes and can be resumed without any of the authoring files.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from simulator.training.contract import CAPABILITIES, RULES, resolve_scenario, validate_contract
from simulator.training.environment import REWARD_NAMES, MAX_MAX_TICKS
from simulator.training.plans import (
    EpisodeSampler, _opponent_mix, _scenario, advance_rule, fields, finite, positive,
)
from simulator.training.policies import fixed_indices


LEARNER_DEFAULTS = {"hidden_dim": 256, "lr": 0.0003, "epochs": 4, "minibatch_size": 1024,
                    "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2,
                    "vf_coef": 0.5, "ent_coef": 0.01, "max_grad_norm": 0.5}
PLAYER_FIELDS = ("position", "velocity", "yaw", "pitch", "on_ground", "health",
                 "absorption", "attack_strength_ticks", "loadout")
SCENE_FIELDS = ("format", "version", "minecraft", "name", "profile", "requires",
                "rules", "bounds", "blocks", "players", "max_ticks", "reward_weights")
CONDITION_FIELDS = ("environment", "spawns", "equipment", "task_override", "opponents", "controls")
RESOLVED_CONDITION_FIELDS = ("scenario", "opponents", "controls", "rewards",
                             "environment_name", "arena_names")
EVALUATION_FIELDS = ("every_updates", "episodes", "seed", "worlds")


def _name(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _seed(value, label):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError(f"{label} must be in [0,2**63)")
    return value


def _version(value, format_name, version, label):
    if value.get("format") != format_name or type(value.get("version")) is not int or value["version"] != version:
        raise ValueError(f"Unsupported {label} format/version")
    if "minecraft" in value and value["minecraft"] != "26.3":
        raise ValueError(f"Unsupported {label} Minecraft version")


def _rewards(value):
    fields(value, REWARD_NAMES, "reward weights")
    result = dict.fromkeys(REWARD_NAMES, 0.0)
    result.update({name: float(finite(number, f"reward {name}")) for name, number in value.items()})
    return result


def _task(value):
    fields(value, ("max_ticks", "rewards"), "task")
    ticks = value.get("max_ticks", 300)
    if type(ticks) is not int or not 1 <= ticks <= MAX_MAX_TICKS:
        raise ValueError(f"max_ticks must be an integer from 1 to {MAX_MAX_TICKS}")
    return {"max_ticks": ticks, "rewards": _rewards(value.get("rewards", {}))}


def _patch(base, changes):
    """Patch objects only; arrays and scalar/distribution leaves replace."""
    result = deepcopy(base)
    for key, value in changes.items():
        if (isinstance(value, dict) and isinstance(result.get(key), dict)
                and not ({"choices", "uniform"} & (set(value) | set(result[key])))):
            result[key] = _patch(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _vector(value, label, *, integer=False, limit=None):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} needs three coordinates")
    for number in value:
        finite(number, label)
        if integer and type(number) is not int:
            raise ValueError(f"{label} coordinates must be integers")
        if limit is not None and abs(number) > limit:
            raise ValueError(f"{label} coordinate exceeds {limit}")
    return value


def _geometry(value):
    bounds = fields(value.get("bounds"), ("min", "size"), "arena bounds")
    origin = _vector(bounds.get("min"), "arena origin", integer=True, limit=4096)
    size = _vector(bounds.get("size"), "arena dimensions", integer=True)
    if any(not 1 <= extent <= 128 for extent in size) or math.prod(size) > 32768:
        raise ValueError("Arena dimensions must be 1..128 and at most 32768 cells")
    blocks = value.get("blocks")
    if not isinstance(blocks, list) or len(blocks) > 256:
        raise ValueError("Arena blocks must be at most 256 fill regions")
    for fill in blocks:
        fields(fill, ("from", "to", "block"), "block fill")
        lower = _vector(fill.get("from"), "fill minimum", integer=True, limit=4096)
        upper = _vector(fill.get("to"), "fill maximum", integer=True, limit=4096)
        if fill.get("block") not in ("minecraft:air", "minecraft:stone"):
            raise ValueError("Only minecraft:air and minecraft:stone blocks are supported")
        if any(a < start or a > b or b >= start + extent
               for a, b, start, extent in zip(lower, upper, origin, size)):
            raise ValueError("Arena block fill is outside bounds")
    required = value.get("requires", [])
    if not isinstance(required, list) or any(item not in CAPABILITIES for item in required):
        raise ValueError("Arena requires unsupported capabilities")
    rules = fields(value.get("rules", {}), tuple(RULES), "physical rules")
    if any(type(item) is not bool or item is not False for item in rules.values()):
        raise ValueError("Physical rules require frozen food and disabled natural regeneration")


def _player(value, scene):
    fields(value, PLAYER_FIELDS, "player")
    result = _patch({
        "velocity": [0.0, -0.0784000015258789, 0.0], "yaw": 0.0, "pitch": 0.0,
        "on_ground": True, "health": 20.0, "absorption": 0.0, "attack_strength_ticks": 20,
        "loadout": {"weapons": ["minecraft:air"], "selected": 0,
                    "offhand": "minecraft:air", "armor": {}, "arrows": 0},
    }, value)
    position = _vector(result.get("position"), "player position", limit=8192)
    _vector(result["velocity"], "player velocity", limit=1000)
    if any(not start <= point < start + extent for point, start, extent in
           zip(position, scene["bounds"]["min"], scene["bounds"]["size"])):
        raise ValueError("Player position is outside arena bounds")
    for key in ("yaw", "pitch", "health", "absorption"):
        if abs(finite(result[key], f"player {key}")) > 3.4028234663852886e38:
            raise ValueError(f"Player {key} exceeds native float range")
    if not -90 <= result["pitch"] <= 90 or not 0 < result["health"] <= 20 or result["absorption"] < 0:
        raise ValueError("Invalid player pitch, health or absorption")
    if type(result["on_ground"]) is not bool:
        raise ValueError("Player on_ground must be a boolean")
    ticks = result["attack_strength_ticks"]
    if type(ticks) is not int or not 0 <= ticks <= 2**31 - 1:
        raise ValueError("Player attack_strength_ticks must be a nonnegative int32")
    loadout = fields(result["loadout"], ("weapons", "selected", "offhand", "armor", "arrows"), "loadout")
    slots = loadout["weapons"]
    if not isinstance(slots, list) or not 1 <= len(slots) <= 9:
        raise ValueError("Player weapons must contain 1..9 slots")
    loadout["weapons"] = [*slots, *(["minecraft:air"] * (9 - len(slots)))]
    return result


def _concrete(value):
    fields(value, SCENE_FIELDS, "compiled environment")
    _version(value, "minecraft_sim_environment", 1, "compiled environment")
    if value.get("minecraft") != "26.3":
        raise ValueError("Compiled environment needs Minecraft 26.3")
    _name(value.get("name"), "Environment name")
    _geometry(value)
    result = deepcopy(value)
    players = result.get("players")
    if not isinstance(players, list) or len(players) != 2:
        raise ValueError("Environment needs two players")
    result["players"] = [_player(player, result) for player in players]
    result = resolve_scenario(result)
    for player in result["players"]:
        if not EpisodeSampler._valid_position(player["position"], result):
            raise ValueError("Initial player must be collision-free inside arena bounds")
    delta = [abs(a - b) for a, b in zip(*(player["position"] for player in result["players"]))]
    if delta[0] < 0.6 and delta[2] < 0.6 and delta[1] < 1.8:
        raise ValueError("Initial players must not intersect")
    task = _task({"max_ticks": result.get("max_ticks", 300),
                  "rewards": result.get("reward_weights", {})})
    result["max_ticks"], result["reward_weights"] = task["max_ticks"], task["rewards"]
    return result


def _policy(value):
    fields(value, ("format", "version", "name", "minecraft", "ticks_per_action", "requires",
                   "observations", "actions", "observation_dim", "action_names", "action_sizes"), "policy")
    _version(value, "minecraft_sim_policy", 1, "policy")
    if type(value.get("ticks_per_action")) is not int:
        raise ValueError("Policy tick rate must be an integer")
    for key, allowed in (("observations", ("feature", "scale", "encoding", "categories", "clip")),
                         ("actions", ("name", "values"))):
        if not isinstance(value.get(key), list):
            raise ValueError(f"Policy {key} must be a list")
        for entry in value[key]:
            fields(entry, allowed, key)
    return validate_contract(value)


def _learner(value):
    fields(value, tuple(LEARNER_DEFAULTS), "learner")
    result = {**LEARNER_DEFAULTS, **value}
    for key in ("hidden_dim", "epochs", "minibatch_size"):
        positive(result[key], f"learner {key}")
    for key in ("lr", "gamma", "gae_lambda", "clip_eps", "vf_coef", "ent_coef", "max_grad_norm"):
        result[key] = float(finite(result[key], f"learner {key}"))
    for key in ("lr", "max_grad_norm"):
        if result[key] <= 0:
            raise ValueError(f"Learner {key} must be positive")
    for key in ("gamma", "gae_lambda"):
        if not 0 <= result[key] <= 1:
            raise ValueError(f"Learner {key} must be between 0 and 1")
    if not 0 < result["clip_eps"] < 1:
        raise ValueError("Learner clip_eps must be between 0 and 1, exclusive")
    for key in ("vf_coef", "ent_coef"):
        if result[key] < 0:
            raise ValueError(f"Learner {key} must be nonnegative")
    return result


def _snapshots(value, names):
    if not isinstance(value, list):
        raise ValueError("Snapshots must be a list of named save rules")
    for rule in value:
        fields(rule, ("name", "at_updates", "on_completion"), "snapshot rule")
        name = _name(rule.get("name"), "Snapshot name")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in names:
            raise ValueError("Snapshot names must be unique safe filenames using letters, digits, '_' or '-'")
        names.add(name)
        if ("at_updates" in rule) == ("on_completion" in rule):
            raise ValueError("Snapshot needs exactly one of at_updates or on_completion")
        if "at_updates" in rule:
            positive(rule["at_updates"], "snapshot at_updates")
        elif rule["on_completion"] is not True:
            raise ValueError("Snapshot on_completion must be true")


def _opponents(value, base_dir):
    if not isinstance(value, dict) or not value:
        raise ValueError("Plan needs named opponents")
    result = deepcopy(value)
    for name, opponent in result.items():
        _name(name, "Opponent ID")
        if not isinstance(opponent, dict):
            raise ValueError("Opponent definition must be an object")
        if opponent.get("type") == "scripted":
            fields(opponent, ("type", "controller", "options"), "scripted opponent")
            _name(opponent.get("controller"), "Scripted controller")
            if not isinstance(opponent.get("options", {}), dict):
                raise ValueError("Scripted opponent options must be an object")
        elif opponent.get("type") == "checkpoint":
            fields(opponent, ("type", "path", "sha256", "deterministic"), "checkpoint opponent")
            path = Path(_name(opponent.get("path"), "Opponent checkpoint path"))
            path = (path if path.is_absolute() else base_dir / path).resolve()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if opponent.get("sha256", digest) != digest:
                raise ValueError(f"Opponent checkpoint changed: {path}")
            if type(opponent.get("deterministic", False)) is not bool:
                raise ValueError("Opponent deterministic must be boolean")
            opponent.update(path=str(path), sha256=digest)
        elif opponent.get("type") in ("self_play", "frozen_self_play"):
            allowed = ("type", "deterministic")
            if opponent["type"] == "frozen_self_play":
                allowed += ("every_updates",)
                positive(opponent.get("every_updates"), "frozen self-play interval")
            fields(opponent, allowed, "self-play opponent")
            if type(opponent.get("deterministic", False)) is not bool:
                raise ValueError("Opponent deterministic must be boolean")
        else:
            raise ValueError("Opponent type must be scripted, checkpoint, self_play or frozen_self_play")
    return result


def _resolved_condition(value, policy, opponents):
    result = deepcopy(value)
    scenario = fields(result.get("scenario"), ("arenas", "spawns", "equipment"), "resolved scenario")
    if not isinstance(scenario.get("arenas"), list):
        raise ValueError("Resolved scenario needs concrete arenas")
    for arena in scenario["arenas"]:
        fields(arena, ("environment", "weight"), "resolved arena")
        arena["environment"] = _concrete(arena.get("environment"))
    result["scenario"] = _scenario(scenario)
    _opponent_mix(result.get("opponents"), opponents)
    controls = fields(result.get("controls", {"fixed": {}}), ("fixed",), "controls")
    fixed_indices(policy, controls)
    result["controls"] = controls
    reward = fields(result.get("rewards"), ("mode", "weights"), "resolved rewards")
    if reward.get("mode") != "replace" or set(reward.get("weights", {})) != set(REWARD_NAMES):
        raise ValueError("Resolved rewards must explicitly replace all reward weights")
    result["rewards"] = {"mode": "replace", "weights": _rewards(reward["weights"])}
    for arena in result["scenario"]["arenas"]:
        if arena["environment"]["reward_weights"] != result["rewards"]["weights"]:
            raise ValueError("Resolved environment and stage rewards disagree")
    if "environment_name" in result:
        _name(result["environment_name"], "Environment name")
    if "arena_names" in result:
        if not isinstance(result["arena_names"], list) or not result["arena_names"]:
            raise ValueError("Arena names must be a nonempty list")
        for name in result["arena_names"]:
            _name(name, "Arena name")
    return result


def _resolved_evaluations(value, policy, opponents):
    if not isinstance(value, dict):
        raise ValueError("Evaluations must be named suites")
    result = {}
    for name, suite in value.items():
        _name(name, "Evaluation name")
        fields(suite, (*RESOLVED_CONDITION_FIELDS, *EVALUATION_FIELDS), "resolved evaluation")
        positive(suite.get("every_updates"), "evaluation interval")
        positive(suite.get("episodes"), "evaluation episodes")
        _seed(suite.get("seed"), "Evaluation seed")
        if "worlds" in suite:
            positive(suite["worlds"], "evaluation worlds")
        result[name] = _resolved_condition(suite, policy, opponents)
    return result


def _resolved(value, base_dir):
    fields(value, ("format", "version", "name", "seed", "policy", "learner", "opponents",
                   "stages", "evaluations", "scenario", "snapshots"), "resolved experiment")
    _version(value, "minecraft_sim_experiment", 1, "resolved experiment")
    result = deepcopy(value)
    result["name"] = _name(result.get("name"), "Experiment name")
    result["seed"] = _seed(result.get("seed"), "Experiment seed")
    result["policy"] = _policy(result.get("policy"))
    result["learner"] = _learner(result.get("learner", {}))
    result["opponents"] = _opponents(result.get("opponents"), base_dir)
    evaluations = _resolved_evaluations(result.get("evaluations", {}), result["policy"], result["opponents"])
    result["evaluations"] = evaluations
    stages = result.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Plan needs at least one stage")
    names = set()
    snapshot_names = set()
    _snapshots(result.get("snapshots", []), snapshot_names)
    suite_names = set(evaluations)
    for index, stage in enumerate(stages):
        fields(stage, (*RESOLVED_CONDITION_FIELDS, "name", "advance", "evaluations", "snapshots"), "resolved stage")
        _snapshots(stage.get("snapshots", []), snapshot_names)
        name = _name(stage.get("name"), "Stage name")
        if name in names:
            raise ValueError("Stage names must be unique")
        names.add(name)
        local = _resolved_evaluations(stage.get("evaluations", {}), result["policy"], result["opponents"])
        if suite_names.intersection(local):
            raise ValueError("Evaluation names must be unique across the plan")
        suite_names.update(local)
        if "evaluations" in stage:
            stage["evaluations"] = local
        advance_rule(stage.get("advance"), {**evaluations, **local})
        stages[index] = _resolved_condition(stage, result["policy"], result["opponents"])
    if "scenario" in result and result["scenario"] != stages[0]["scenario"]:
        raise ValueError("Experiment default scenario must equal its first stage")
    result["scenario"] = deepcopy(stages[0]["scenario"])
    return result


class _Authoring:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.files = {}
        self.arenas = {}
        self.environments = {}

    def read(self, reference, base_dir):
        path = Path(_name(reference, "File reference"))
        path = (path if path.is_absolute() else base_dir / path).resolve()
        if path not in self.files:
            self.files[path] = json.loads(path.read_text(encoding="utf-8"))
        return deepcopy(self.files[path]), path

    def arena(self, reference, base_dir):
        raw, path = self.read(reference, base_dir)
        if path not in self.arenas:
            fields(raw, ("format", "version", "minecraft", "name", "requires", "rules", "bounds", "blocks"), "arena")
            _version(raw, "minecraft_sim_arena", 1, "arena")
            _name(raw.get("name"), "Arena name")
            _geometry(raw)
            raw["requires"] = sorted(set(raw.get("requires", [])))
            raw["rules"] = dict(RULES)
            self.arenas[path] = raw
        return deepcopy(self.arenas[path])

    def environment(self, reference):
        raw, path = self.read(reference, self.base_dir)
        if path not in self.environments:
            fields(raw, ("format", "version", "minecraft", "name", "arena", "players",
                         "spawns", "equipment", "task"), "environment")
            _version(raw, "minecraft_sim_environment", 2, "environment")
            _name(raw.get("name"), "Environment name")
            arena = self.arena(raw.get("arena"), path.parent)
            players = fields(raw.get("players"), ("shared", "overrides"), "players")
            shared = fields(players.get("shared", {}), PLAYER_FIELDS, "shared player")
            overrides = players.get("overrides", [{}, {}])
            if not isinstance(overrides, list) or len(overrides) != 2:
                raise ValueError("Player overrides must contain exactly two objects")
            concrete = {
                "format": "minecraft_sim_environment", "version": 1, "minecraft": "26.3",
                "name": raw["name"], "profile": "configured", "requires": arena["requires"],
                "bounds": arena["bounds"], "blocks": arena["blocks"], "rules": arena["rules"],
                "players": [_patch(shared, fields(player, PLAYER_FIELDS, "player override")) for player in overrides],
            }
            task = _task(raw.get("task", {}))
            concrete.update(max_ticks=task["max_ticks"], reward_weights=task["rewards"])
            scenario = {"arenas": [{"environment": _concrete(concrete), "weight": 1}]}
            for component in ("spawns", "equipment"):
                if component in raw:
                    scenario[component] = deepcopy(raw[component])
            self.environments[path] = {
                "scenario": _scenario(scenario), "environment_name": raw["name"], "arena_names": [arena["name"]],
            }
        return deepcopy(self.environments[path])

    def condition(self, value, default_environment, mixtures, policy, opponents, base=None):
        if base is None or "environment" in value:
            result = self.environment(value.get("environment", default_environment))
            if base is not None:
                source = base["scenario"]["arenas"][0]["environment"]
                concrete = result["scenario"]["arenas"][0]["environment"]
                concrete.update(max_ticks=source["max_ticks"], reward_weights=deepcopy(source["reward_weights"]))
        else:
            result = {key: deepcopy(base[key]) for key in RESOLVED_CONDITION_FIELDS if key in base}
        scenario = result["scenario"]
        for component in ("spawns", "equipment"):
            if component in value:
                if value[component] is None:
                    scenario.pop(component, None)
                else:
                    fields(value[component], ("sample", "region", "minimum_separation", "yaw", "pitch")
                           if component == "spawns" else ("sample", "players", "loadout"), component)
                    scenario[component] = _patch(scenario.get(component, {}), value[component])
        concrete = scenario["arenas"][0]["environment"]
        task = {"max_ticks": concrete["max_ticks"], "rewards": concrete["reward_weights"]}
        changes = fields(value.get("task_override", {}), ("max_ticks", "rewards"), "task override")
        task.update(changes)
        task = _task(task)
        concrete.update(max_ticks=task["max_ticks"], reward_weights=task["rewards"])
        result["scenario"] = _scenario(scenario)
        if base is not None and "opponents" not in value:
            result["opponents"] = deepcopy(base["opponents"])
        else:
            mixture = value.get("opponents")
            if not isinstance(mixture, str) or mixture not in mixtures:
                raise ValueError("Stage/evaluation opponents must reference a named mixture")
            result["opponents"] = deepcopy(mixtures[mixture])
        if base is not None:
            controls = fields(value.get("controls", {}), ("fixed",), "evaluation controls")
            fixed = deepcopy(base["controls"]["fixed"])
            if controls.get("fixed", {}) is None:
                fixed = {}
            else:
                fixed.update(fields(controls.get("fixed", {}), policy["action_names"], "evaluation fixed controls"))
                fixed = {head: choice for head, choice in fixed.items() if choice is not None}
            result["controls"] = {"fixed": fixed}
        else:
            result["controls"] = deepcopy(value.get("controls", {"fixed": {}}))
        result["rewards"] = {"mode": "replace", "weights": task["rewards"]}
        return _resolved_condition(result, policy, opponents)

    def evaluations(self, value, default_environment, mixtures, policy, opponents, stages, base=None):
        if not isinstance(value, dict):
            raise ValueError("Evaluations must be named suites")
        result = {}
        for name, suite in value.items():
            _name(name, "Evaluation name")
            fields(suite, (*CONDITION_FIELDS, *EVALUATION_FIELDS, "stage", "overrides"), "evaluation suite")
            inherited = base
            if "stage" in suite:
                if base is not None:
                    raise ValueError("Stage-local evaluations cannot reference another stage")
                reference = _name(suite["stage"], "Evaluation stage")
                if reference not in stages:
                    raise ValueError(f"Unknown evaluation stage: {reference}")
                inherited = stages[reference]
            if inherited is not None:
                fields(suite, (*EVALUATION_FIELDS, "overrides", *(("stage",) if base is None else ())),
                       "inherited evaluation suite")
                changes = fields(suite.get("overrides", {}), CONDITION_FIELDS, "evaluation overrides")
            else:
                fields(suite, (*CONDITION_FIELDS, *EVALUATION_FIELDS), "evaluation suite")
                if "environment" not in suite:
                    raise ValueError("Evaluation suites must explicitly reference an environment or stage")
                changes = suite
            compiled = self.condition(changes, default_environment, mixtures, policy, opponents, base=inherited)
            compiled.update({key: suite.get(key) for key in EVALUATION_FIELDS if key != "worlds" or key in suite})
            result[name] = compiled
        return result


def resolve_experiment(value, base_dir):
    if not isinstance(value, dict):
        raise ValueError("Training input must be an object")
    base_dir = Path(base_dir).resolve()
    if value.get("format") == "minecraft_sim_experiment":
        return _resolved(value, base_dir)
    fields(value, ("format", "version", "name", "seed", "policy", "environment", "learner",
                   "opponents", "mixtures", "stages", "evaluations", "snapshots"), "training plan")
    _version(value, "minecraft_sim_training_plan", 2, "training plan")
    authoring = _Authoring(base_dir)
    policy, _ = authoring.read(value.get("policy"), base_dir)
    policy = _policy(policy)
    opponents = _opponents(value.get("opponents"), base_dir)
    mixtures = value.get("mixtures")
    if not isinstance(mixtures, dict) or not mixtures:
        raise ValueError("Plan needs named opponent mixtures")
    for name, mixture in mixtures.items():
        _name(name, "Mixture name")
        _opponent_mix(mixture, opponents)
    stages = value.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Plan needs at least one stage")
    default = value.get("environment")
    _name(default, "Environment reference")
    result = {
        "format": "minecraft_sim_experiment", "version": 1, "name": value.get("name", "experiment"),
        "seed": value.get("seed", 1), "policy": policy, "learner": _learner(value.get("learner", {})),
        "opponents": opponents, "stages": [], "evaluations": {},
    }
    if "snapshots" in value:
        result["snapshots"] = deepcopy(value["snapshots"])
    named_stages = {}
    for stage in stages:
        fields(stage, (*CONDITION_FIELDS, "name", "advance", "evaluations", "snapshots"), "stage")
        name = _name(stage.get("name"), "Stage name")
        if name in named_stages:
            raise ValueError("Stage names must be unique")
        compiled = authoring.condition(stage, default, mixtures, policy, opponents)
        compiled.update(name=name, advance=deepcopy(stage.get("advance")))
        if "snapshots" in stage:
            compiled["snapshots"] = deepcopy(stage["snapshots"])
        named_stages[name] = compiled
        result["stages"].append(compiled)
    result["evaluations"] = authoring.evaluations(value.get("evaluations", {}), default, mixtures,
                                                policy, opponents, named_stages)
    for authored, compiled in zip(stages, result["stages"]):
        if "evaluations" in authored:
            compiled["evaluations"] = authoring.evaluations(authored["evaluations"], default, mixtures,
                                                          policy, opponents, named_stages, base=compiled)
    return _resolved(result, base_dir)

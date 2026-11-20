"""Opponent-driven PPO, reusable scenes, curriculum stages and held-out evaluation."""

from contextlib import ExitStack, contextmanager
from collections import deque
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch

from backend.fast_rl_model import ActorCritic, BridgeActorCritic
from simulator.training.compiler import LEARNER_DEFAULTS, _structured_opponents, _validate_bridge_opponents
from simulator.training.contract import fingerprint, validate_bridge_observation_descriptor
from simulator.training.controllers import (BridgeFixedLaneController, CheckpointController, FrozenSelfPlayController, SelfPlayController,
                                            StationaryTargetController, create_controller)
from simulator.training.environment import Environment, REWARD_NAMES, REWARD_SEMANTICS, ROOT, actor_lost as _lost
from simulator.training.objectives import actor_task_observations
from simulator.training.plans import EpisodeSampler, plan_hash, reached, resolve_plan, weights
from simulator.training.policies import fixed_indices, policy_actions, resident_policy_actions
from simulator.training.ppo import compute_gae, metadata, optimize
from simulator.training.timing import CollectionTimer, OptimizationTimer


BRIDGE_REWARD_SEMANTICS = "bridge_score_win_loss_v1"
BRIDGE_OBJECTIVE_REWARD_SEMANTICS = "bridge_score_opponent_objective_progress_v2"
BRIDGE_SHARED_REWARD_SEMANTICS = "bridge_shared_rewards_v1"


def bridge_reward_semantics(configuration, weights=None):
    if weights and any(weights.values()):
        return BRIDGE_SHARED_REWARD_SEMANTICS
    fields = configuration["task_rewards"]
    return (BRIDGE_OBJECTIVE_REWARD_SEMANTICS if set(fields) & {"opponent_score", "toward", "away"}
            else BRIDGE_REWARD_SEMANTICS)


def bridge_observation_inputs(environment, scalar, descriptor=None):
    """Sample versioned Bridge inputs from the current host state."""
    if environment.bridge is None:
        raise ValueError("Structured Bridge observations require a Bridge environment")
    scalar = np.asarray(scalar)
    if scalar.shape != (environment.count, 2, environment.observation_dim) or scalar.dtype != np.float32:
        raise ValueError("Bridge scalar observations have an invalid shape or dtype")
    match = environment.bridge["match"]
    states = environment.bridge_state()
    kits = [environment.bridge_kits(player) for player in (0, 1)]
    voxels = environment.bridge_local_voxels()
    version = 1 if descriptor is None else validate_bridge_observation_descriptor(descriptor)["version"]
    features = np.empty((environment.count, 2, 10 if version == 1 else 17), dtype=np.float32)
    for world, state in enumerate(states):
        for player in (0, 1):
            kit = kits[player][world]
            features[world, player, :10] = (
                state["scores"][player] / match["goals_to_win"],
                state["scores"][1 - player] / match["goals_to_win"],
                max(0, match["max_ticks"] - state["ticks"]) / match["max_ticks"],
                state["cage_remaining"] / max(1, match["cage_ticks"]),
                kit["blocks"] / 64, kit["golden_apples"] / 64, kit["arrows"] / 64,
                kit["arrow_refill_elapsed"] / max(1, match["arrow_refill_ticks"]),
                float(match["arrow_refill_ticks"] > 0), kit["selected_slot"] / 8,
            )
    if version == 2:
        goals = environment.bridge["goals"]
        targets = np.asarray([(np.asarray(goals[1 - player]["min"]) + goals[1 - player]["max"]) / 2
                              for player in (0, 1)], dtype=np.float64)
        targets = np.broadcast_to(targets, (environment.count, 2, 3))
        features[..., 10:] = actor_task_observations(environment.actor_task_state(), targets)
    return {"scalar": scalar.copy(), "match": features, "voxels": voxels}


def _validate_structured_bridge_model(model, policy):
    validate_bridge_observation_descriptor(model.descriptor)
    if (model.observation_dim != policy["observation_dim"]
            or model.action_names != tuple(policy["action_names"])
            or [getattr(model.policy, f"{name}_head").out_features for name in model.action_names]
            != policy["action_sizes"]):
        raise ValueError("Structured Bridge model and scalar policy are incompatible")


def _validate_structured_bridge_consumer(model, plan, scenario):
    policy = plan["policy"]
    _validate_structured_bridge_model(model, policy)
    if validate_bridge_observation_descriptor(plan.get("bridge_observation")) != model.descriptor:
        raise ValueError("Structured Bridge plan and model descriptors differ")
    if (scenario.get("bridge") is None or len(scenario.get("arenas", [])) != 1
            or any(scenario.get(name) for name in ("spawns", "equipment"))
            or policy["ticks_per_action"] != 1):
        raise ValueError("Structured Bridge requires a compatible fixed Bridge model and stage")
    _structured_opponents(plan["opponents"])
    _validate_bridge_opponents(policy, scenario, plan["opponents"])


def scene_key(scenario):
    return [{"weight": arena.get("weight", 1), **{name: arena["environment"].get(name)
            for name in ("bounds", "blocks", "rules")}} for arena in scenario["arenas"]]


def requires_new_worlds(current, next_scenario):
    if scene_key(current) != scene_key(next_scenario):
        return True
    return (current.get("bridge") is not None or next_scenario.get("bridge") is not None) and current != next_scenario


def deployment_summary(plan, stage, stage_index):
    return {"name": plan.get("name", "Experiment"), "stage": stage["name"], "stage_index": stage_index,
            "environment": stage.get("environment_name", stage["name"]), "arenas": stage.get("arena_names", []),
            "controllers": "shared_learner", "sampling": "fixed_checkpoint_episode"}


def research_metadata(environment, hidden_dim, learner):
    result = metadata(environment, hidden_dim, learner)
    if environment.bridge is not None:
        if not any(environment.weights.values()):
            result.pop("reward_weights", None)
        result["reward_semantics"] = bridge_reward_semantics(environment.bridge, environment.weights)
    return result


def _validate_checkpoint_model(metadata, descriptor):
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint is missing model metadata")
    saved = metadata.get("bridge_observation")
    if "bridge_observation" in metadata:
        saved = validate_bridge_observation_descriptor(saved)
    if (descriptor is None) != ("bridge_observation" not in metadata) or (descriptor is not None and saved != descriptor):
        raise ValueError("Checkpoint structured Bridge model descriptor differs from the selected plan")


def resident_unsupported(plan, device):
    if device != "cuda":
        return "CUDA resident collection requires --device cuda"
    for stage in plan["stages"]:
        scenario = stage["scenario"]
        if scenario.get("bridge") is not None:
            return f"Stage {stage['name']!r} uses Bridge, which requires host collection"
        if len(scenario["arenas"]) != 1:
            return f"Stage {stage['name']!r} has multiple arenas"
        for component in ("spawns", "equipment"):
            if scenario.get(component, {}).get("sample") == "episode":
                return f"Stage {stage['name']!r} has per-episode {component} sampling"
        for entry in stage["opponents"]:
            definition = plan["opponents"][entry["id"]]
            if not (definition["type"] == "self_play" or
                    definition["type"] == "scripted" and definition.get("controller") == "stationary_target"):
                return f"Stage {stage['name']!r} has unsupported opponent {entry['id']!r}"
    return None


class EpisodeMetrics:
    def __init__(self, count, history=(), window=1000):
        self.returns = np.zeros(count, dtype=np.float64)
        self.lengths = np.zeros(count, dtype=np.int64)
        self.history = deque(history, maxlen=window)
        self.begin_rollout()

    def begin_rollout(self):
        self.rollout_totals = {}
        self.opponent_totals = {}

    @staticmethod
    def accumulate(totals, episode):
        for name, value in (("count", 1), ("return", episode["return"]), ("length", episode["length"]),
                            (episode["outcome"], 1), (episode["outcome"] + "_length", episode["length"])):
            totals[name] = totals.get(name, 0) + value

    @staticmethod
    def summarize_totals(totals):
        count = totals.get("count", 0)
        result = {"training_episodes": count}
        if count:
            result.update(training_return=totals["return"] / count, training_length=totals["length"] / count)
            for outcome in ("win", "loss", "timeout", "draw"):
                amount = totals.get(outcome, 0)
                result[f"training_{outcome}_rate"] = amount / count
                if amount and outcome in ("win", "loss"):
                    result[f"training_{outcome}_length"] = totals[outcome + "_length"] / amount
        return result

    def rollout_snapshot(self):
        return {**self.summarize_totals(self.rollout_totals), "source": "recorded",
                "training_opponents": {name: self.summarize_totals(totals)
                                       for name, totals in sorted(self.opponent_totals.items())}}

    def reset(self):
        self.returns.fill(0)
        self.lengths.fill(0)
        self.history.clear()
        self.begin_rollout()

    def step(self, rewards, done, outcomes, opponents):
        self.returns += rewards
        self.lengths += 1
        for world, outcome, opponent in zip(np.flatnonzero(done), outcomes, opponents, strict=True):
            episode = {"return": float(self.returns[world]), "length": int(self.lengths[world]),
                       "outcome": outcome, "opponent": opponent}
            self.history.append(episode)
            self.accumulate(self.rollout_totals, episode)
            self.accumulate(self.opponent_totals.setdefault(opponent, {}), episode)
        self.returns[done] = 0
        self.lengths[done] = 0

    @staticmethod
    def summarize(episodes):
        result = {"training_episodes": len(episodes)}
        if not episodes:
            return result
        result.update(training_return=sum(episode["return"] for episode in episodes) / len(episodes),
                      training_length=sum(episode["length"] for episode in episodes) / len(episodes))
        for outcome in ("win", "loss", "timeout", "draw"):
            selected = [episode for episode in episodes if episode["outcome"] == outcome]
            result[f"training_{outcome}_rate"] = len(selected) / len(episodes)
            if selected and outcome in ("win", "loss"):
                result[f"training_{outcome}_length"] = sum(episode["length"] for episode in selected) / len(selected)
        return result

    def snapshot(self):
        episodes = list(self.history)
        result = self.summarize(episodes)
        result["training_episode_window"] = self.history.maxlen
        result["training_opponents"] = {
            name: self.summarize([episode for episode in episodes if episode["opponent"] == name])
            for name in sorted({episode["opponent"] for episode in episodes})}
        return result


class SampledWorlds:
    """Group worlds by immutable scene; changing players never recreates terrain."""

    def __init__(self, scenario, contract, count, device, library, max_ticks, seed, state=None):
        self.resources = ExitStack()
        self.count, self.device, self.contract = count, device, contract
        self.episode_metrics = EpisodeMetrics(count)
        self.max_ticks_override = max_ticks
        self.schema = contract
        self.observation_dim = contract["observation_dim"]
        self.action_names = tuple(contract["action_names"])
        self.action_count = len(self.action_names)
        bridge = scenario.get("bridge")
        self.bridge_scenario = deepcopy(scenario) if bridge is not None else None
        if bridge is not None and any(scenario.get(name) for name in ("spawns", "equipment")):
            raise ValueError("Bridge research requires fixed spawns and equipment")
        sampler_scenario = {name: value for name, value in scenario.items() if name != "bridge"}
        self.sampler = EpisodeSampler(sampler_scenario, count, seed)
        if state is not None:
            self.sampler.load_state_dict(state)
        self.groups = []
        directory = Path(self.resources.enter_context(tempfile.TemporaryDirectory()))
        try:
            initial = [self.sampler.players(world) for world in range(count)]
            for arena_id in sorted(set(self.sampler.arenas)):
                ids = np.flatnonzero(np.asarray(self.sampler.arenas) == arena_id)
                definition = deepcopy(scenario["arenas"][arena_id]["environment"])
                definition["players"] = initial[int(ids[0])]
                path = directory / f"arena-{arena_id}.json"
                path.write_text(json.dumps(definition, allow_nan=False), encoding="utf-8")
                environment = self.resources.enter_context(Environment(len(ids), device, path, library,
                    max_ticks, (seed + arena_id) % 2**63, contract=contract, bridge=bridge))
                if bridge is None:
                    environment.reset_players([initial[int(world)] for world in ids], np.ones(len(ids), dtype=bool))
                self.groups.append((ids, environment))
            self.max_ticks = max(environment.max_ticks for _, environment in self.groups)
            self.build_info = self.groups[0][1].build_info
        except BaseException:
            self.resources.close()
            raise

    def configure(self, stage):
        scenario = stage["scenario"]
        if self.bridge_scenario is not None:
            if scenario != self.bridge_scenario:
                raise ValueError("Changing a Bridge scenario requires a new environment batch")
            if "rewards" in stage and any(environment.weights != stage["rewards"]["weights"]
                                          for _, environment in self.groups):
                raise ValueError("Bridge stage rewards disagree with its environment batch")
            self.episode_metrics.reset()
            return
        if scenario.get("bridge") is not None:
            raise ValueError("Changing to Bridge requires a new environment batch")
        self.episode_metrics.reset()
        self.sampler.configure(scenario)
        for ids, environment in self.groups:
            arena_id = self.sampler.arenas[int(ids[0])]
            definition = stage["scenario"]["arenas"][arena_id]["environment"]
            limit = self.max_ticks_override if self.max_ticks_override is not None else definition.get("max_ticks", 300)
            if environment.max_ticks != limit:
                environment.set_max_ticks(limit)
            environment.set_rewards(stage["rewards"]["weights"])
        self.max_ticks = max(environment.max_ticks for _, environment in self.groups)

    @contextmanager
    def profile_collection(self, timer):
        with ExitStack() as stack:
            if timer.enabled:
                for _, environment in self.groups:
                    stack.enter_context(environment.profile_collection(timer))
            yield

    def observe(self):
        result = np.empty((self.count, 2, self.observation_dim), dtype=np.float32)
        for ids, environment in self.groups:
            result[ids] = environment.observe()
        return result

    def bridge_observations(self, scalar, descriptor=None):
        if self.bridge_scenario is None:
            raise ValueError("Structured Bridge observations require Bridge worlds")
        if (not isinstance(scalar, np.ndarray) or scalar.dtype != np.float32
                or scalar.shape != (self.count, 2, self.observation_dim)):
            raise ValueError("Bridge scalar observations have an invalid shape or dtype")
        version = 1 if descriptor is None else validate_bridge_observation_descriptor(descriptor)["version"]
        result = {"scalar": np.empty((self.count, 2, self.observation_dim), dtype=np.float32),
                  "match": np.empty((self.count, 2, 10 if version == 1 else 17), dtype=np.float32),
                  "voxels": np.empty((self.count, 2, 5, 9, 9), dtype=np.uint8)}
        for ids, environment in self.groups:
            inputs = bridge_observation_inputs(environment, scalar[ids], descriptor)
            for name in result:
                result[name][ids] = inputs[name]
        return result

    def bridge_reset_episodes(self):
        if self.bridge_scenario is None:
            raise ValueError("Bridge reset episodes require Bridge worlds")
        result = np.empty(self.count, dtype=np.uint64)
        for ids, environment in self.groups:
            result[ids] = environment.bridge_reset_episodes()
        return result.tolist()

    def restore_bridge_reset_episodes(self, episodes):
        if (self.bridge_scenario is None or not isinstance(episodes, list) or len(episodes) != self.count
                or any(type(value) is not int or not 0 <= value < 2**64 - 1 for value in episodes)):
            raise ValueError("Invalid Bridge checkpoint reset episodes")
        for ids, environment in self.groups:
            environment.restore_bridge_reset_episodes([episodes[int(world)] for world in ids])

    def reset(self, mask=None, timer=None):
        timer = timer or CollectionTimer(self.device)
        mask = np.ones(self.count, dtype=bool) if mask is None else np.asarray(mask)
        if mask.shape != (self.count,) or mask.dtype.kind != "b":
            raise ValueError("Sampled reset mask must contain one boolean per world")
        if self.bridge_scenario is not None:
            observations = np.empty((self.count, 2, self.observation_dim), dtype=np.float32)
            for ids, environment in self.groups:
                selected = mask[ids]
                observations[ids] = environment.reset(selected) if selected.any() else environment.observe()
            return observations
        with timer.phase("environment_reset.sample"):
            definitions = {int(world): self.sampler.players(int(world), timer.counts if timer.enabled else None)
                           for world in np.flatnonzero(mask)}
        observations = np.empty((self.count, 2, self.observation_dim), dtype=np.float32)
        for ids, environment in self.groups:
            selected = mask[ids]
            if selected.any():
                with timer.phase("environment_reset.payload"):
                    players = [definitions[int(world)] if selected[index] else None
                               for index, world in enumerate(ids)]
                observations[ids] = environment.reset_players(players, selected, timer=timer)
            else:
                with timer.phase("environment_reset.unchanged_observe"):
                    observations[ids] = environment.observe()
        return observations

    def step(self, actions):
        observations = np.empty((self.count, 2, self.observation_dim), dtype=np.float32)
        rewards = np.empty((self.count, 2), dtype=np.float32)
        terminated, truncated = np.zeros(self.count, dtype=bool), np.zeros(self.count, dtype=bool)
        for ids, environment in self.groups:
            obs, rew, term, trunc = environment.step(actions[ids])
            observations[ids], rewards[ids], terminated[ids], truncated[ids] = obs, rew, term, trunc
        return observations, rewards, terminated, truncated

    def step_reward_totals(self, learner_slots):
        totals = dict.fromkeys(REWARD_NAMES, 0.0)
        for ids, environment in self.groups:
            selected = np.zeros((len(ids), 2), dtype=bool)
            selected[np.arange(len(ids)), learner_slots[ids]] = True
            contributions = environment.step_reward_totals(selected)
            if contributions is None:
                return None
            for name, amount in contributions.items():
                totals[name] += amount
        return totals

    def step_behavior_events(self):
        result = np.empty((self.count, 2, 3), dtype=np.float32)
        for ids, environment in self.groups:
            result[ids] = environment.step_behavior_events()
        return result

    def bridge_goal_counts(self, learner_slots):
        goals_for = goals_against = 0
        for ids, environment in self.groups:
            for local, state in enumerate(environment.bridge_state()):
                player = int(learner_slots[ids[local]])
                goals_for += int(state["scored"][player])
                goals_against += int(state["scored"][1 - player])
        return goals_for, goals_against

    def bridge_reward_totals(self, learner_slots):
        if any(environment.build_info.get("bridge_objective_reward_version") != 1 for _, environment in self.groups):
            return None
        totals = np.zeros(4, dtype=np.float64)
        for ids, environment in self.groups:
            terms = environment.bridge_reward_terms()
            totals += terms[np.arange(len(ids)), learner_slots[ids]].sum(axis=0)
        return dict(zip(("bridge_reward_score", "bridge_reward_opponent_score",
                         "bridge_reward_toward", "bridge_reward_away"), totals.tolist()))

    def bridge_states(self):
        if self.bridge_scenario is None:
            raise ValueError("Bridge state requires Bridge worlds")
        states = [None] * self.count
        for ids, environment in self.groups:
            for world, state in zip(ids, environment.bridge_state(), strict=True):
                states[int(world)] = state
        return states

    def losses(self, mask):
        result = np.zeros((self.count, 2), dtype=bool)
        for ids, environment in self.groups:
            if mask[ids].any():
                result[ids] = environment.losses(mask[ids])
        return result

    def outcomes(self, done, terminated, learner_slots):
        if self.bridge_scenario is None:
            losses = self.losses(done)
            return ["timeout" if not terminated[world] else
                    "draw" if losses[world, 0] == losses[world, 1] else
                    "loss" if losses[world, learner_slots[world]] else "win"
                    for world in np.flatnonzero(done)]
        results = {}
        for ids, environment in self.groups:
            if not done[ids].any():
                continue
            states = environment.bridge_state()
            for local, world in enumerate(ids):
                if not done[world]:
                    continue
                outcome = states[local]["outcome"]
                if not terminated[world]:
                    results[int(world)] = "timeout"
                elif outcome == "draw":
                    results[int(world)] = "draw"
                elif outcome in ("team0", "team1"):
                    results[int(world)] = "win" if outcome == f"team{learner_slots[world]}" else "loss"
                else:
                    raise RuntimeError(f"Invalid terminated Bridge outcome: {outcome!r}")
        return [results[int(world)] for world in np.flatnonzero(done)]

    def episode(self, world):
        for ids, environment in self.groups:
            found = np.flatnonzero(ids == world)
            if len(found):
                local = int(found[0])
                return environment.episode_scenario(local), environment.replay_frame(local)
        raise ValueError("Unknown world")

    def episode_scenario(self, world):
        for ids, environment in self.groups:
            found = np.flatnonzero(ids == world)
            if len(found):
                return environment.episode_scenario(int(found[0]))
        raise ValueError("Unknown world")

    def close(self):
        self.resources.close()


class OpponentPool:
    def __init__(self, definitions, contract, count, device, seed, learner=None):
        self.count = count
        self.names = list(definitions)
        streams = np.random.SeedSequence(seed).spawn(2)
        self.rng, self.side_rng = [np.random.default_rng(stream) for stream in streams]
        self.controllers = {name: create_controller(definition, contract, count, device,
            int(np.random.SeedSequence([seed, index + 1]).generate_state(1)[0]), learner=learner)
            for index, (name, definition) in enumerate(definitions.items())}
        self.assignments = np.zeros(count, dtype=int)
        self.learner_slots = np.zeros(count, dtype=int)
        self.mixture: list[dict] = []

    def configure(self, mixture, controls=None):
        weights(mixture)
        self.mixture = mixture
        for controller in self.controllers.values():
            if isinstance(controller, SelfPlayController):
                controller.set_controls({"fixed": {}} if controls is None else controls)

    def reset(self, mask):
        if not self.mixture:
            raise ValueError("Configure the opponent mixture before reset")
        for controller in self.controllers.values():
            controller.reset(mask)
        ids = np.flatnonzero(mask)
        selected = self.rng.choice(len(self.mixture), len(ids), p=weights(self.mixture))
        self.assignments[ids] = [self.names.index(self.mixture[int(index)]["id"]) for index in selected]
        self.learner_slots[ids] = self.side_rng.integers(0, 2, len(ids))

    def capture(self, learner, update, controls, force=False):
        for controller in self.controllers.values():
            if isinstance(controller, FrozenSelfPlayController):
                controller.capture(learner, update, controls, force)

    def copy_snapshots(self, source):
        for name, controller in self.controllers.items():
            if isinstance(controller, FrozenSelfPlayController):
                controller.copy_snapshot(source.controllers[name])

    def act(self, observations, learner_actions, bridge_inputs=None):
        actions = np.empty((self.count, 2, learner_actions.shape[-1]), dtype=np.int64)
        ids = np.arange(self.count)
        actions[ids, self.learner_slots] = learner_actions
        for index, name in enumerate(self.names):
            selected = np.flatnonzero(self.assignments == index)
            if len(selected):
                controller = self.controllers[name]
                slots = 1 - self.learner_slots[selected]
                if bridge_inputs is not None and isinstance(controller, (SelfPlayController, CheckpointController)):
                    opponent_inputs = {key: value[selected, slots] for key, value in bridge_inputs.items()}
                else:
                    opponent_inputs = observations[selected, slots]
                opponent_actions = controller.act(opponent_inputs, indices=selected)
                raw = np.asarray(opponent_actions)
                if raw.shape != (len(selected), learner_actions.shape[-1]) or raw.dtype.kind not in "iu":
                    raise ValueError(f"Controller {name} returned an invalid action shape/dtype")
                actions[selected, 1 - self.learner_slots[selected]] = raw
        return actions

    def provenance(self, world):
        name = self.names[int(self.assignments[world])]
        controller = self.controllers[name]
        provenance = controller.provenance
        if isinstance(controller, FrozenSelfPlayController):
            provenance["snapshot_update"] = int(controller.snapshot_updates[world])
        if isinstance(controller, SelfPlayController) and "bridge_observation" in provenance:
            provenance["actor_slot"] = int(world)
            if (isinstance(controller, FrozenSelfPlayController)
                    and provenance["snapshot_update"] not in controller.snapshots):
                raise ValueError(f"Bridge frozen self-play opponent {name!r} snapshot is unavailable")
            fixed = (controller.snapshots[provenance["snapshot_update"]][1]
                     if isinstance(controller, FrozenSelfPlayController) else controller._fixed)
            provenance["fixed_actions"] = fixed.copy()
            if isinstance(controller, FrozenSelfPlayController):
                provenance["snapshot_sha256"] = _bridge_snapshot_sha256(controller, provenance["snapshot_update"])
        return {"id": name, "controller": provenance,
                "learner_slot": int(self.learner_slots[world])}

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state, "side_rng": self.side_rng.bit_generator.state,
                "controllers": {name: controller.state_dict() for name, controller in self.controllers.items()}}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]
        self.side_rng.bit_generator.state = state["side_rng"]
        if set(state["controllers"]) != set(self.controllers):
            raise ValueError("Checkpoint opponent pool differs")
        for name, saved in state["controllers"].items():
            self.controllers[name].load_state_dict(saved)


class ResidentOpponentPool:
    def __init__(self, pool, device, seed):
        if not pool.mixture:
            raise ValueError("Configure the opponent mixture before enabling resident opponents")
        self.pool = pool
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.count = pool.count
        self.names = pool.names
        self.controllers = []
        for entry in pool.mixture:
            name = entry["id"]
            controller = pool.controllers[name]
            if type(controller) is StationaryTargetController:
                neutral = torch.as_tensor(controller._neutral, device=self.device)
            elif type(controller) is SelfPlayController:
                neutral = None
            else:
                raise ValueError(f"Resident rollout does not support opponent {name!r}")
            self.controllers.append((self.names.index(name), controller, neutral))
        cumulative = np.cumsum(weights(pool.mixture))
        cumulative[-1] = 1.0
        self.cumulative = torch.as_tensor(cumulative, device=self.device)
        self.choices = torch.tensor([self.names.index(entry["id"]) for entry in pool.mixture],
                                    dtype=torch.long, device=self.device)
        self.assignments = torch.zeros(self.count, dtype=torch.long, device=self.device)
        self.learner_slots = torch.zeros(self.count, dtype=torch.long, device=self.device)
        self.indices = torch.arange(self.count, device=self.device)
        self.seed = seed % 2**63
        self.episodes = torch.zeros(self.count, dtype=torch.long, device=self.device)
        self.learner_rng = torch.Generator(device=self.device).manual_seed((seed + 2) % 2**63)

    def _uniform(self, stream):
        value = self.indices * 6364136223846793005 + self.episodes * 1442695040888963407
        value = value + ((self.seed + stream * 2862933555777941757) % 2**63)
        value = (value ^ ((value >> 30) & ((1 << 34) - 1))) * -4658895280553007687
        value = (value ^ ((value >> 27) & ((1 << 37) - 1))) * -7723592293110705685
        value = value ^ ((value >> 31) & ((1 << 33) - 1))
        return (value & ((1 << 53) - 1)).to(torch.float64) * (1.0 / (1 << 53))

    def reset(self, mask):
        if mask.shape != (self.count,) or mask.dtype != torch.bool or mask.device != self.device:
            raise ValueError("Resident opponent reset mask must be a device boolean vector")
        selected = self.choices[torch.searchsorted(self.cumulative, self._uniform(0))]
        sides = (self._uniform(1) >= 0.5).to(torch.long)
        self.assignments = torch.where(mask, selected, self.assignments)
        self.learner_slots = torch.where(mask, sides, self.learner_slots)
        self.episodes += mask.to(torch.long)

    def act(self, model, observations, action_names, fixed):
        slots = self.learner_slots
        current = observations[self.indices, slots]
        other = observations[self.indices, 1 - slots]
        learner, log_probs, values, valid = resident_policy_actions(
            model, current, action_names, fixed, self.learner_rng)
        valid = valid & torch.isfinite(observations).all(dim=(1, 2))
        opponent = torch.zeros_like(learner)
        for index, controller, neutral in self.controllers:
            selected = self.assignments == index
            if neutral is None:
                actions, _, _, opponent_valid = resident_policy_actions(
                    model, other, action_names, controller._fixed, controller._generator, controller.deterministic)
                valid = valid & (~selected | opponent_valid)
            else:
                actions = neutral.expand_as(learner)
            opponent = torch.where(selected.unsqueeze(-1), actions, opponent)
        first = torch.where((slots == 0).unsqueeze(-1), learner, opponent)
        second = torch.where((slots == 1).unsqueeze(-1), learner, opponent)
        return current, learner, log_probs, values, torch.stack((first, second), dim=1), valid

    def state_dict(self):
        return {"episodes": self.episodes.cpu().clone(), "learner_rng": self.learner_rng.get_state()}

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or set(state) != {"episodes", "learner_rng"}
                or not isinstance(state["episodes"], torch.Tensor)
                or state["episodes"].dtype != torch.long or state["episodes"].shape != (self.count,)
                or (state["episodes"] < 0).any().item()):
            raise ValueError("Invalid resident opponent RNG state")
        self.episodes.copy_(state["episodes"])
        try:
            self.learner_rng.set_state(state["learner_rng"].cpu())
        except (AttributeError, RuntimeError, TypeError) as error:
            raise ValueError("Invalid resident opponent RNG state") from error


class ResidentCollector:
    def __init__(self, worlds, pool, seed):
        if worlds.device != "cuda" or len(worlds.groups) != 1 or len(worlds.sampler.scenario["arenas"]) != 1:
            raise ValueError("Resident rollout requires one CUDA arena batch")
        for component in ("spawns", "equipment"):
            if worlds.sampler.scenario.get(component, {}).get("sample") == "episode":
                raise ValueError(f"Resident rollout does not support per-episode {component} sampling")
        ids, environment = worlds.groups[0]
        if not np.array_equal(ids, np.arange(worlds.count)):
            raise ValueError("Resident rollout requires a contiguous world batch")
        self.worlds = worlds
        self.environment = environment
        self.pool = ResidentOpponentPool(pool, "cuda", seed)
        self.device = self.pool.device
        self.returns = torch.zeros(worlds.count, dtype=torch.float64, device=self.device)
        self.lengths = torch.zeros(worlds.count, dtype=torch.int64, device=self.device)
        environment.register_cuda_observations()
        self.actions = environment.register_cuda_actions()
        self.buffers = environment.register_cuda_resident()
        self.observations = environment.observe_cuda_async()
        self.pool.reset(torch.ones(worlds.count, dtype=torch.bool, device=self.device))

    @torch.no_grad()
    def collect(self, model, steps, fixed, learner=None, profile=False):
        learner = LEARNER_DEFAULTS if learner is None else learner
        worlds, pool = self.worlds, self.pool
        timer = CollectionTimer(self.device, profile)
        timer.count("worlds", worlds.count)
        timer.count("steps", steps)
        worlds.episode_metrics.begin_rollout()
        shape = (steps, worlds.count)
        rollout = {name: torch.empty(shape, dtype=torch.bool if name in ("terminated", "truncated")
                                    else torch.float32, device=self.device)
                   for name in ("log_probs", "values", "next_values", "rewards", "terminated", "truncated")}
        rollout["observations"] = torch.empty((*shape, worlds.observation_dim), device=self.device)
        rollout["actions"] = torch.empty((*shape, worlds.action_count), dtype=torch.long, device=self.device)
        rollout["fixed_actions"] = fixed.copy()
        event_returns = torch.empty(shape, dtype=torch.float64, device=self.device)
        event_lengths = torch.empty(shape, dtype=torch.int64, device=self.device)
        event_losses = torch.empty((*shape, 2), dtype=torch.uint8, device=self.device)
        event_opponents = torch.empty(shape, dtype=torch.long, device=self.device)
        event_slots = torch.empty(shape, dtype=torch.long, device=self.device)
        reward_totals = torch.zeros(len(REWARD_NAMES), dtype=torch.float64, device=self.device)
        valid = torch.ones((), dtype=torch.bool, device=self.device)
        for step in range(steps):
            with timer.phase("learner_opponent_inference", cuda=True):
                current, actions, log_probs, values, combined, finite = pool.act(
                    model, self.observations, worlds.action_names, fixed)
                valid &= finite.all() & torch.isfinite(log_probs).all()
                rollout["observations"][step].copy_(current)
                rollout["actions"][step].copy_(actions)
                rollout["log_probs"][step].copy_(log_probs)
                rollout["values"][step].copy_(values)
                self.actions.copy_(combined)
            with timer.phase("environment_step", cuda=True):
                final, outputs = self.environment.step_cuda_resident_async()
                slots = pool.learner_slots
                rewards = outputs["rewards"][pool.indices, slots]
                terminated = outputs["terminated"].bool()
                truncated = outputs["truncated"].bool()
                done = terminated | truncated
                _, next_values = model(final[pool.indices, slots])
                next_finite = torch.isfinite(next_values)
                valid &= (next_finite.all() & torch.isfinite(final).all() & torch.isfinite(rewards).all()
                          & torch.isfinite(outputs["contributions"]).all())
                rollout["next_values"][step].copy_(torch.where(next_finite, next_values, 0))
                rollout["rewards"][step].copy_(rewards)
                rollout["terminated"][step].copy_(terminated)
                rollout["truncated"][step].copy_(truncated)
                reward_totals += outputs["contributions"][pool.indices, slots].sum(dim=0)
            with timer.phase("episode_bookkeeping", cuda=True):
                self.returns += rewards.to(torch.float64)
                self.lengths += 1
                event_returns[step].copy_(self.returns)
                event_lengths[step].copy_(self.lengths)
                event_losses[step].copy_(outputs["losses"])
                event_opponents[step].copy_(pool.assignments)
                event_slots[step].copy_(slots)
                self.returns = torch.where(done, 0, self.returns)
                self.lengths = torch.where(done, 0, self.lengths)
                outputs["reset_mask"].copy_(done)
                self.observations = self.environment.reset_cuda_resident_async()
                pool.reset(done)
        with timer.phase("gae", cuda=True):
            rollout["advantages"], rollout["returns"] = compute_gae(
                rollout["rewards"], rollout["values"], rollout["next_values"], rollout["terminated"],
                rollout["truncated"], learner["gamma"], learner["gae_lambda"])
            valid &= torch.isfinite(rollout["advantages"]).all() & torch.isfinite(rollout["returns"]).all()
        self.environment.wait_cuda_actions()
        if not valid.item():
            raise ValueError("Resident rollout encountered nonfinite observations, rewards or policy outputs")
        terminated = rollout["terminated"].cpu().numpy()
        truncated = rollout["truncated"].cpu().numpy()
        returns, lengths = event_returns.cpu().numpy(), event_lengths.cpu().numpy()
        losses, opponents, slots = event_losses.cpu().numpy(), event_opponents.cpu().numpy(), event_slots.cpu().numpy()
        for step, world in zip(*np.nonzero(terminated | truncated), strict=True):
            loss = losses[step, world]
            slot = slots[step, world]
            outcome = ("timeout" if truncated[step, world] else "draw" if loss[0] == loss[1]
                       else "loss" if loss[slot] else "win")
            episode = {"return": float(returns[step, world]), "length": int(lengths[step, world]),
                       "outcome": outcome, "opponent": pool.names[int(opponents[step, world])]}
            metrics = worlds.episode_metrics
            metrics.history.append(episode)
            metrics.accumulate(metrics.rollout_totals, episode)
            metrics.accumulate(metrics.opponent_totals.setdefault(episode["opponent"], {}), episode)
        rollout["reward_types"] = dict(zip(REWARD_NAMES, reward_totals.cpu().tolist()))
        rollout["episode_metrics"] = {**worlds.episode_metrics.snapshot(),
                                      "training_rollout": worlds.episode_metrics.rollout_snapshot()}
        pool.pool.assignments[:] = pool.assignments.cpu().numpy()
        pool.pool.learner_slots[:] = pool.learner_slots.cpu().numpy()
        if timer.enabled:
            timer.count("reset_calls", np.any(terminated | truncated, axis=1).sum())
            timer.count("reset_worlds", (terminated | truncated).sum())
            rollout["collection_profile"] = timer.finish()
        return rollout, self.observations, int(terminated.sum()), int(truncated.sum())


@torch.no_grad()
def collect(model, worlds, pool, observations, steps, device, fixed, profile=False, learner=None,
            structured_bridge=False):
    if type(structured_bridge) is not bool:
        raise ValueError("Structured Bridge collection opt-in must be boolean")
    if structured_bridge:
        if (not isinstance(model, BridgeActorCritic) or worlds.bridge_scenario is None
                or len(worlds.bridge_scenario["arenas"]) != 1 or worlds.contract["ticks_per_action"] != 1):
            raise ValueError("Structured Bridge collection requires a compatible fixed Bridge model and stage")
        _validate_structured_bridge_model(model, worlds.contract)
        if any(type(controller) not in (StationaryTargetController, BridgeFixedLaneController,
                                        SelfPlayController, FrozenSelfPlayController, CheckpointController)
               for controller in pool.controllers.values()):
            raise ValueError("Structured Bridge collection supports scripted, checkpoint and self-play opponents only")
        if (pool.count != worlds.count or any(controller.contract_hash != fingerprint(worlds.contract)
                                              for controller in pool.controllers.values())):
            raise ValueError("Structured Bridge opponent pool differs from the world contract")
        if any(isinstance(controller, (SelfPlayController, CheckpointController)) and
               (controller.describe.get("bridge_observation") != model.descriptor
                or not isinstance(controller.model, BridgeActorCritic)
                or controller.model.descriptor != model.descriptor
                or type(controller) is SelfPlayController and controller.model is not model)
               for controller in pool.controllers.values()):
            raise ValueError("Structured Bridge opponent model descriptor differs from the learner")
        if not np.array_equal(observations, worlds.observe()):
            raise ValueError("Structured Bridge collection requires current scalar observations")
    elif isinstance(model, BridgeActorCritic):
        raise ValueError("Structured Bridge collection requires explicit opt-in")
    timer = CollectionTimer(device, profile)
    timer.count("worlds", worlds.count)
    timer.count("steps", steps)
    with worlds.profile_collection(timer):
        result = _collect(model, worlds, pool, observations, steps, device, fixed, timer,
                          LEARNER_DEFAULTS if learner is None else learner, structured_bridge)
    if timer.enabled:
        result[0]["collection_profile"] = timer.finish()
    return result


def _collect(model, worlds, pool, observations, steps, device, fixed, timer, learner, structured_bridge=False):
    trace = os.environ.get("MCSIM_TRACE_PROGRESS") == "1"
    with timer.phase("setup"):
        worlds.episode_metrics.begin_rollout()
        shape = (steps, worlds.count)
        rollout = {name: torch.empty(shape, dtype=torch.bool if name in ("terminated", "truncated")
                                    else torch.float32, device=device)
                   for name in ("log_probs", "values", "next_values", "rewards", "terminated", "truncated")}
        if structured_bridge:
            rollout["observations"] = {
                "scalar": torch.empty((*shape, worlds.observation_dim), device=device),
                "match": torch.empty((*shape, model.match_dim), device=device),
                "voxels": torch.empty((*shape, 5, 9, 9), dtype=torch.uint8, device=device)}
            bridge_inputs = worlds.bridge_observations(observations, model.descriptor)
        else:
            rollout["observations"] = torch.empty((*shape, worlds.observation_dim), device=device)
        rollout["actions"] = torch.empty((*shape, worlds.action_count), dtype=torch.long, device=device)
        rollout["fixed_actions"] = fixed.copy()
        reward_types = worlds.step_reward_totals(pool.learner_slots)
        if reward_types is not None:
            reward_types = dict.fromkeys(reward_types, 0.0)
        bridge_events = (dict.fromkeys(("bridge_goals_for", "bridge_goals_against", "bridge_wins",
                                       "bridge_losses", "bridge_draws"), 0)
                         if worlds.bridge_scenario is not None else None)
        bridge_rewards = worlds.bridge_reward_totals(pool.learner_slots) if bridge_events is not None else None
        if bridge_rewards is not None:
            bridge_rewards = dict.fromkeys(bridge_rewards, 0.0)
        endings = timeouts = 0
        ids = np.arange(worlds.count)
    for step in range(steps):
        if trace:
            print(f"TRACE rollout step={step + 1}/{steps} phase=begin", flush=True)
        slots = pool.learner_slots.copy()
        with timer.phase("observation_transfer", cuda=True):
            current = ({name: torch.as_tensor(value[ids, slots], device=device)
                        for name, value in bridge_inputs.items()} if structured_bridge
                       else torch.as_tensor(observations[ids, slots], device=device))
        with timer.phase("learner_inference", cuda=True):
            actions, log_probs, values, _ = policy_actions(model, current, worlds.action_names, fixed)
        with timer.phase("action_transfer", cuda=True):
            learner_actions = actions.cpu().numpy()
        with timer.phase("opponent_actions", cuda=True):
            combined_actions = (pool.act(observations, learner_actions, bridge_inputs) if structured_bridge
                                else pool.act(observations, learner_actions))
        with timer.phase("environment_step"):
            if trace:
                print(f"TRACE rollout step={step + 1}/{steps} phase=environment_step_begin", flush=True)
            final, rewards, terminated, truncated = worlds.step(combined_actions)
            if trace:
                print(f"TRACE rollout step={step + 1}/{steps} phase=environment_step_done", flush=True)
        with timer.phase("reward_breakdown"):
            if reward_types is not None:
                contributions = worlds.step_reward_totals(slots)
                if contributions is None:
                    raise RuntimeError("Native reward breakdown support changed during rollout")
                for name, amount in contributions.items():
                    reward_types[name] += amount
        with timer.phase("next_value", cuda=True):
            if structured_bridge:
                final_inputs = worlds.bridge_observations(final, model.descriptor)
                next_observation = {name: torch.as_tensor(value[ids, slots], device=device)
                                    for name, value in final_inputs.items()}
            else:
                next_observation = torch.as_tensor(final[ids, slots], device=device)
            _, next_values = model(next_observation)
        with timer.phase("rollout_storage", cuda=True):
            if structured_bridge:
                for name, value in current.items():
                    rollout["observations"][name][step] = value
            else:
                rollout["observations"][step] = current
            rollout["actions"][step] = actions
            for name, value in (("log_probs", log_probs), ("values", values), ("next_values", next_values),
                                ("rewards", rewards[ids, slots]), ("terminated", terminated), ("truncated", truncated)):
                rollout[name][step] = torch.as_tensor(value, device=device)
        with timer.phase("episode_bookkeeping"):
            if bridge_events is not None:
                goals_for, goals_against = worlds.bridge_goal_counts(slots)
                bridge_events["bridge_goals_for"] += goals_for
                bridge_events["bridge_goals_against"] += goals_against
                if bridge_rewards is not None:
                    for name, value in worlds.bridge_reward_totals(slots).items():
                        bridge_rewards[name] += value
            done = terminated | truncated
            endings += int(terminated.sum())
            timeouts += int(truncated.sum())
            selected = np.flatnonzero(done)
            outcomes = worlds.outcomes(done, terminated, slots) if len(selected) else []
            if bridge_events is not None:
                for outcome in ("win", "loss", "draw"):
                    key = {"win": "bridge_wins", "loss": "bridge_losses", "draw": "bridge_draws"}[outcome]
                    bridge_events[key] += outcomes.count(outcome)
            opponents = [pool.names[int(pool.assignments[world])] for world in selected]
            worlds.episode_metrics.step(rewards[ids, slots], done, outcomes, opponents)
        if done.any():
            if trace:
                print(f"TRACE rollout step={step + 1}/{steps} phase=reset_begin", flush=True)
            if timer.enabled:
                timer.count("reset_calls")
                timer.count("reset_worlds", done.sum())
            with timer.phase("environment_reset"):
                observations = worlds.reset(done, timer=timer)
                if structured_bridge:
                    bridge_inputs = worlds.bridge_observations(observations, model.descriptor)
            with timer.phase("opponent_reset"):
                pool.reset(done)
            if trace:
                print(f"TRACE rollout step={step + 1}/{steps} phase=reset_done", flush=True)
        else:
            observations = final
            if structured_bridge:
                bridge_inputs = final_inputs
    if trace:
        print("TRACE rollout phase=gae_begin", flush=True)
    with timer.phase("gae", cuda=True):
        rollout["advantages"], rollout["returns"] = compute_gae(
            rollout["rewards"], rollout["values"], rollout["next_values"], rollout["terminated"],
            rollout["truncated"], learner["gamma"], learner["gae_lambda"])
    if reward_types is not None:
        rollout["reward_types"] = reward_types
    if bridge_events is not None:
        if bridge_rewards is not None:
            bridge_events.update(bridge_rewards)
        rollout["bridge_events"] = bridge_events
    rollout["episode_metrics"] = {**worlds.episode_metrics.snapshot(),
                                  "training_rollout": worlds.episode_metrics.rollout_snapshot()}
    if trace:
        print("TRACE rollout phase=done", flush=True)
    return rollout, observations, endings, timeouts


class BehaviorEvaluation:
    def __init__(self, count):
        self.launched = np.zeros(count, dtype=bool)
        self.smashed = np.zeros(count, dtype=bool)
        self.launch_smash = np.zeros(count, dtype=bool)
        self.remaining = np.zeros(count, dtype=np.int32)
        self.airborne = np.zeros(count, dtype=bool)

    def step(self, events, slots):
        selected = events[np.arange(len(slots)), slots]
        launch = selected[:, 0] >= 0.2
        smash = selected[:, 1] > 0
        grounded = selected[:, 2] > 0
        self.smashed |= smash
        self.launch_smash |= smash & (self.remaining > 0) & self.airborne
        self.remaining = np.maximum(0, self.remaining - 1)
        self.airborne |= (self.remaining > 0) & ~grounded
        landed = grounded & self.airborne
        self.remaining[landed] = 0
        self.airborne[landed] = False
        self.launched |= launch
        self.remaining[launch] = 80
        self.airborne[launch] = ~grounded[launch]

    def reset(self, mask):
        for value in (self.launched, self.smashed, self.launch_smash, self.remaining, self.airborne):
            value[mask] = 0


def _bridge_snapshot_sha256(controller, update):
    model, fixed = controller.snapshots[update]
    digest = hashlib.sha256(json.dumps({"contract_hash": controller.contract_hash,
        "bridge_observation": controller.describe.get("bridge_observation"), "fixed_actions": fixed},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    for name, value in sorted(model.state_dict().items()):
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _copy_bridge_snapshot(pool, source, name, update=None):
    target = pool.controllers[name]
    original = None if source is None else source.controllers.get(name)
    if (not isinstance(original, FrozenSelfPlayController)
            or original.contract_hash != target.contract_hash
            or original.deterministic != target.deterministic
            or original.every_updates != target.every_updates):
        raise ValueError(f"Bridge frozen self-play opponent {name!r} requires a compatible source snapshot")
    selected = original.latest_update if update is None else update
    if type(selected) is not int or selected not in original.snapshots:
        raise ValueError(f"Bridge frozen self-play opponent {name!r} snapshot is unavailable")
    snapshot = original.snapshots[selected][0]
    if (original.describe.get("bridge_observation") != target.describe.get("bridge_observation")
            or getattr(snapshot, "descriptor", None) != target.describe.get("bridge_observation")):
        raise ValueError(f"Bridge frozen self-play opponent {name!r} requires a compatible source snapshot")
    if isinstance(target.model, BridgeActorCritic):
        if not isinstance(snapshot, BridgeActorCritic):
            raise ValueError(f"Bridge frozen self-play opponent {name!r} requires a compatible source snapshot")
        try:
            _validate_structured_bridge_model(snapshot, target.contract)
        except (AttributeError, ValueError) as error:
            raise ValueError(f"Bridge frozen self-play opponent {name!r} requires a compatible source snapshot") from error
    target.copy_snapshot(original, selected)


def _evaluate_bridge(model, plan, suite, options, device, opponent_pool):
    scenario = suite["scenario"]
    structured = isinstance(model, BridgeActorCritic)
    if structured:
        _validate_structured_bridge_consumer(model, plan, scenario)
    if (len(scenario.get("arenas", [])) != 1 or any(scenario.get(name) for name in ("spawns", "equipment"))
            or plan["policy"].get("ticks_per_action") != 1):
        raise ValueError("Bridge evaluation requires one fixed scenario and ticks_per_action 1")
    _validate_bridge_opponents(plan["policy"], scenario, plan["opponents"])
    count = min(suite.get("worlds", 16), suite["episodes"])
    fixed = fixed_indices(plan["policy"], suite["controls"])
    worlds = SampledWorlds(suite["scenario"], plan["policy"], count, options.device,
                           options.library, options.max_ticks, suite["seed"])
    was_training = model.training
    model.eval()
    try:
        pool = OpponentPool(plan["opponents"], plan["policy"], count, options.device, suite["seed"], learner=model)
        pool.configure(suite["opponents"], suite["controls"])
        for entry in suite["opponents"]:
            if isinstance(pool.controllers[entry["id"]], FrozenSelfPlayController):
                _copy_bridge_snapshot(pool, opponent_pool, entry["id"])
        pool.reset(np.ones(count, dtype=bool))
        tickets = np.arange(count, dtype=np.int64)
        pool.learner_slots[:] = tickets % 2
        next_ticket = count
        generator = torch.Generator(device=device).manual_seed(suite["seed"])
        observations = worlds.observe()
        rewards_sum = np.zeros(count, dtype=np.float64)
        totals = {name: 0 for name in ("wins", "losses", "draws", "truncations", "goals_for", "goals_against")}
        seats = [{"episodes": 0, "wins": 0, "losses": 0, "draws": 0, "truncations": 0,
                  "goals_for": 0, "goals_against": 0} for _ in range(2)]
        reward_total = 0.0
        finished = 0
        ids = np.arange(count)
        while finished < suite["episodes"]:
            slots = pool.learner_slots.copy()
            if structured:
                inputs = worlds.bridge_observations(observations, model.descriptor)
                current = {name: torch.as_tensor(value[ids, slots], device=device)
                           for name, value in inputs.items()}
            else:
                current = torch.as_tensor(observations[ids, slots], device=device)
            actions, _, _, valid = resident_policy_actions(model, current, worlds.action_names, fixed, generator)
            if not bool(valid.all()):
                raise ValueError("Nonfinite Bridge evaluation policy outputs")
            learner_actions = actions.cpu().numpy()
            combined = (pool.act(observations, learner_actions, inputs) if structured
                        else pool.act(observations, learner_actions))
            observations, rewards, terminated, truncated = worlds.step(combined)
            rewards_sum += rewards[ids, slots]
            done = terminated | truncated
            if done.any():
                states = worlds.bridge_states()
                for world in np.flatnonzero(done):
                    if tickets[world] < 0:
                        continue
                    slot = int(slots[world])
                    state = states[world]
                    if terminated[world]:
                        outcome = state["outcome"]
                        if outcome == "draw":
                            key = "draws"
                        elif outcome in ("team0", "team1"):
                            key = "wins" if outcome == f"team{slot}" else "losses"
                        else:
                            raise RuntimeError(f"Invalid terminated Bridge outcome: {outcome!r}")
                    elif truncated[world] and state["outcome"] == "ongoing":
                        key = "truncations"
                    else:
                        raise RuntimeError("Invalid truncated Bridge outcome")
                    seat = seats[slot]
                    totals[key] += 1
                    seat[key] += 1
                    for name, player in (("goals_for", slot), ("goals_against", 1 - slot)):
                        amount = state["scores"][player]
                        totals[name] += amount
                        seat[name] += amount
                    seat["episodes"] += 1
                    reward_total += float(rewards_sum[world])
                    finished += 1
                if finished < suite["episodes"]:
                    observations = worlds.reset(done)
                    pool.reset(done)
                    rewards_sum[done] = 0
                    for world in np.flatnonzero(done):
                        tickets[world] = next_ticket if next_ticket < suite["episodes"] else -1
                        if tickets[world] >= 0:
                            pool.learner_slots[world] = tickets[world] % 2
                            next_ticket += 1
        result = {**totals, "episodes": finished, "worlds": count, "seed": suite["seed"], "seats": seats,
                  "reward_mean": reward_total / finished}
        if structured:
            result["opponents"] = []
            for entry in suite["opponents"]:
                controller = pool.controllers[entry["id"]]
                details = controller.provenance
                if isinstance(controller, FrozenSelfPlayController):
                    details["snapshot_update"] = controller.latest_update
                    details["fixed_actions"] = controller.snapshots[controller.latest_update][1].copy()
                    details["snapshot_sha256"] = _bridge_snapshot_sha256(controller, controller.latest_update)
                elif isinstance(controller, SelfPlayController):
                    details["fixed_actions"] = controller._fixed.copy()
                result["opponents"].append({"id": entry["id"], "weight": entry["weight"], "controller": details})
        for name in ("wins", "losses", "draws", "truncations"):
            rate = {"wins": "win_rate", "losses": "loss_rate", "draws": "draw_rate",
                    "truncations": "truncation_rate"}[name]
            result[rate] = totals[name] / finished
        for name in ("goals_for", "goals_against"):
            result[name + "_mean"] = totals[name] / finished
        return result
    finally:
        model.train(was_training)
        worlds.close()


@torch.no_grad()
def evaluate(model, plan, suite, options, device, fixed, opponent_pool=None):
    if suite["scenario"].get("bridge") is not None:
        return _evaluate_bridge(model, plan, suite, options, device, opponent_pool)
    if isinstance(model, BridgeActorCritic):
        raise ValueError("Structured Bridge evaluation requires a Bridge scenario")
    count = min(suite.get("worlds", 16), suite["episodes"])
    fixed = fixed_indices(plan["policy"], suite["controls"])
    worlds = SampledWorlds(suite["scenario"], plan["policy"], count, options.device,
                          options.library, options.max_ticks, suite["seed"])
    was_training = model.training
    model.eval()
    try:
        pool = OpponentPool(plan["opponents"], plan["policy"], count, options.device, suite["seed"], learner=model)
        pool.configure(suite["opponents"], suite["controls"])
        if opponent_pool is not None:
            pool.copy_snapshots(opponent_pool)
        total = wins = finished = 0
        behavior = BehaviorEvaluation(count)
        launch_successes = smash_successes = launch_smash_successes = 0
        episode_rewards = np.zeros(count)
        observations = worlds.observe()
        pool.reset(np.ones(count, dtype=bool))
        ids = np.arange(count)
        while finished < suite["episodes"]:
            current = torch.as_tensor(observations[ids, pool.learner_slots], device=device)
            actions, _, _, _ = policy_actions(model, current, worlds.action_names, fixed)
            observations, rewards, terminated, truncated = worlds.step(pool.act(observations, actions.cpu().numpy()))
            behavior.step(worlds.step_behavior_events(), pool.learner_slots)
            episode_rewards += rewards[ids, pool.learner_slots]
            done = terminated | truncated
            for world in np.flatnonzero(done):
                if finished == suite["episodes"]:
                    break
                slot = int(pool.learner_slots[world])
                scenario, frame = worlds.episode(int(world))
                wins += int(terminated[world] and not _lost(frame["actors"][slot], scenario)
                            and _lost(frame["actors"][1 - slot], scenario))
                total += float(episode_rewards[world])
                launch_successes += int(behavior.launched[world])
                smash_successes += int(behavior.smashed[world])
                launch_smash_successes += int(behavior.launch_smash[world])
                finished += 1
            if done.any() and finished < suite["episodes"]:
                observations = worlds.reset(done)
                pool.reset(done)
                episode_rewards[done] = 0
                behavior.reset(done)
        return {"win_rate": wins / suite["episodes"], "reward_mean": total / suite["episodes"],
            "self_launch_rate": launch_successes / suite["episodes"],
            "mace_smash_rate": smash_successes / suite["episodes"],
            "launch_smash_rate": launch_smash_successes / suite["episodes"],
            "episodes": suite["episodes"], "worlds": count}
    finally:
        model.train(was_training)
        worlds.close()


@torch.no_grad()
def export_replay(model, plan, stage, options, scenario, provenance, device, rounds, update, directory,
                  opponent_pool=None):
    bridge = stage["scenario"].get("bridge")
    structured = isinstance(model, BridgeActorCritic)
    if structured:
        _validate_structured_bridge_consumer(model, plan, stage["scenario"])
    if "experimental_bridge" in scenario and scenario["experimental_bridge"] != bridge:
        raise ValueError("Replay scenario Bridge configuration differs from the stage")
    if bridge is not None and plan["policy"].get("ticks_per_action") != 1:
        raise ValueError("Bridge replay requires ticks_per_action 1")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"research-{rounds:09d}-{datetime.now():%Y%m%d-%H%M%S-%f}.jsonl"
    partial = destination.with_suffix(".jsonl.part")
    was_training = model.training
    model.eval()
    try:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode.json"
            scene = {name: value for name, value in scenario.items() if name != "experimental_bridge"}
            path.write_text(json.dumps(scene), encoding="utf-8")
            with Environment(1, options.device, path, options.library, options.max_ticks,
                             plan["seed"], contract=plan["policy"], bridge=bridge) as environment:
                if bridge is not None and environment.max_ticks > 100000:
                    raise ValueError("Research replay requires max_ticks <= 100000")
                pool = OpponentPool({provenance["id"]: plan["opponents"][provenance["id"]]},
                                    plan["policy"], 1, options.device, plan["seed"], learner=model)
                pool.configure([{"id": provenance["id"], "weight": 1}], stage["controls"])
                controller = pool.controllers[provenance["id"]]
                if structured and isinstance(controller, SelfPlayController):
                    details = provenance.get("controller", {})
                    seed, actor_slot = details.get("seed"), details.get("actor_slot")
                    if (type(seed) is not int or not 0 <= seed < 2**63 or type(actor_slot) is not int
                            or type(details.get("count")) is not int or actor_slot < 0
                            or actor_slot >= details["count"]
                            or type(provenance.get("learner_slot")) is not int
                            or provenance["learner_slot"] not in (0, 1)
                            or details.get("bridge_observation") != model.descriptor
                            or any(details.get(key) != controller.describe.get(key)
                               for key in ("type", "deterministic", "contract_hash"))):
                        raise ValueError("Structured Bridge replay requires compatible opponent seed and descriptor provenance")
                    controller = create_controller(plan["opponents"][provenance["id"]], plan["policy"], 1,
                                                   options.device, seed, learner=model)
                    pool.controllers[provenance["id"]] = controller
                    controller.set_controls(stage["controls"])
                    stream = np.random.SeedSequence(seed).spawn(actor_slot + 1)[actor_slot]
                    actor_seed = int(stream.generate_state(1, dtype=np.uint64)[0]) % 2**63
                    controller._actor_generators[0].manual_seed(actor_seed)
                if isinstance(controller, FrozenSelfPlayController):
                    if bridge is not None:
                        snapshot_update = provenance.get("controller", {}).get("snapshot_update")
                        if type(snapshot_update) is not int:
                            raise ValueError("Bridge frozen self-play replay requires a snapshot update")
                        _copy_bridge_snapshot(pool, opponent_pool, provenance["id"], snapshot_update)
                        if structured:
                            source = opponent_pool.controllers[provenance["id"]]
                            if (source.seed != seed or source.count != details["count"]
                                    or opponent_pool.names[int(opponent_pool.assignments[actor_slot])] != provenance["id"]
                                    or int(source.snapshot_updates[actor_slot]) != snapshot_update
                                    or int(opponent_pool.learner_slots[actor_slot]) != provenance["learner_slot"]):
                                raise ValueError("Structured Bridge replay opponent assignment differs from provenance")
                            if details.get("snapshot_sha256") != _bridge_snapshot_sha256(source, snapshot_update):
                                raise ValueError("Structured Bridge replay snapshot differs from provenance")
                    elif opponent_pool is not None:
                        controller.copy_snapshot(opponent_pool.controllers[provenance["id"]],
                                                 provenance["controller"]["snapshot_update"])
                if structured and isinstance(controller, SelfPlayController):
                    actual_fixed = (controller.snapshots[controller.latest_update][1]
                                    if isinstance(controller, FrozenSelfPlayController) else controller._fixed)
                    if details.get("fixed_actions") != actual_fixed:
                        raise ValueError("Structured Bridge replay opponent controls differ from provenance")
                pool.reset(np.ones(1, dtype=bool))
                pool.learner_slots[:] = provenance["learner_slot"]
                fixed = fixed_indices(plan["policy"], stage["controls"])
                with partial.open("x", encoding="utf-8") as output:
                    header = {"format": "minecraft_sim_replay", "version": 3, "minecraft": "26.3",
                              "backend": options.device.upper(), "players": 2, "ticks_per_second": 20,
                              "environment": environment.scenario, "policy": plan["policy"],
                              "training": {"stage": stage["name"], "controls": stage["controls"],
                                           "experiment": deployment_summary(plan, stage, plan["stages"].index(stage)),
                                           "opponent": provenance, "completed_rounds": rounds, "policy_update": update}}
                    if bridge is not None:
                        initial_scene = environment.episode_scenario(0)
                        initial_scene.pop("experimental_bridge", None)
                        header.update(version=4, bridge=bridge, environment=initial_scene)
                        if structured:
                            header["training"]["bridge_observation"] = deepcopy(model.descriptor)
                    written = 0

                    def write(frame):
                        nonlocal written
                        line = json.dumps(frame, allow_nan=False, separators=(",", ":")) + "\n"
                        written += len(line.encode("utf-8"))
                        if written > 64 * 1024 * 1024:
                            raise RuntimeError("Research replay exceeds the viewer's 64 MiB limit")
                        output.write(line)

                    write(header)
                    observations = environment.observe()
                    write(environment.replay_frame())
                    for _ in range(environment.max_ticks):
                        slot = pool.learner_slots[0]
                        if structured:
                            inputs = bridge_observation_inputs(environment, observations, model.descriptor)
                            current = {name: torch.as_tensor(value[:, slot], device=device)
                                       for name, value in inputs.items()}
                        else:
                            current = torch.as_tensor(observations[:, slot], device=device)
                        actions, _, _, _ = policy_actions(model,
                            current, environment.action_names, fixed)
                        learner_actions = actions.cpu().numpy()
                        combined = (pool.act(observations, learner_actions, inputs) if structured
                                    else pool.act(observations, learner_actions))
                        observations, _, terminated, truncated = environment.step(combined)
                        frame = environment.replay_frame()
                        frame.update(terminated=bool(terminated[0]), truncated=bool(truncated[0]))
                        write(frame)
                        if terminated[0] or truncated[0]:
                            break
        partial.rename(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        model.train(was_training)
    return str(destination)


def run(options, checkpoint=None):
    from simulator.training.analytics_requests import (
        finish_request as finish_analytics_request,
        mark_running as mark_analytics_request_running,
        read_request as read_analytics_request,
    )
    from simulator.training.replay_requests import finish_request, read_request, requested_replay_round

    run_started = time.perf_counter()
    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but unavailable")
    os.environ["OMP_NUM_THREADS"] = str(options.threads)
    torch.set_num_threads(options.threads)
    device = torch.device(options.device)
    if checkpoint is None and options.resume:
        checkpoint = torch.load(options.resume, map_location=device, weights_only=True)
    if options.plan:
        raw = json.loads(Path(options.plan).read_text(encoding="utf-8"))
    elif checkpoint is not None and "research" in checkpoint:
        raw = checkpoint["research"]["plan"]
    else:
        raise ValueError("Research training requires a plan")
    plan = resolve_plan(raw, base_dir=Path(options.plan).parent if options.plan else ROOT)
    for name in LEARNER_DEFAULTS:
        override = getattr(options, name)
        if override is not None:
            plan["learner"][name] = override
        setattr(options, name, plan["learner"][name])
    plan = resolve_plan(plan)
    descriptor = plan.get("bridge_observation")
    if checkpoint is not None:
        _validate_checkpoint_model(checkpoint.get("metadata"), descriptor)
    saved = {} if checkpoint is None else checkpoint.get("research", {})
    requested_backend = getattr(options, "rollout_backend", "host")
    reason = resident_unsupported(plan, options.device) if requested_backend != "host" else None
    if requested_backend == "cuda-resident" and reason is not None:
        raise ValueError(f"CUDA resident rollout unavailable: {reason}")
    rollout_backend = "host" if requested_backend == "host" or reason is not None else "cuda-resident"
    if options.envs is None:
        options.envs = saved.get("world_count", 16 if options.device == "cpu" else 8192)
    if options.rollout_steps is None:
        options.rollout_steps = saved.get("rollout_steps", 128)
    if checkpoint is not None and options.max_ticks is None:
        options.max_ticks = saved.get("max_ticks_override")
    if checkpoint is not None:
        if checkpoint.get("research", {}).get("plan_hash") != plan_hash(plan):
            original = checkpoint.get("research", {}).get("plan")
            if (original is None or plan_hash(original) != checkpoint["research"].get("plan_hash")
                    or plan_hash(resolve_plan(original)) != plan_hash(plan)):
                raise ValueError("Training plan differs from checkpoint; use --initialize for a new experiment")
        if checkpoint["research"]["world_count"] != options.envs:
            raise ValueError("Research resume requires the original world count")
    replay_every_updates = getattr(options, "replay_every_updates", 0)
    if (options.replay_every or replay_every_updates) and any(
            (options.max_ticks if options.max_ticks is not None else arena["environment"]["max_ticks"]) > 100000
            for stage in plan["stages"] for arena in stage["scenario"]["arenas"]):
        raise ValueError("Enabled replays require every training stage max_ticks <= 100000")
    torch.manual_seed(plan["seed"])
    output = (Path(options.output) if options.output else Path(options.resume).resolve().parent if options.resume else
              ROOT / "simulator/training/runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if output.exists() and any(output.iterdir()) and checkpoint is None:
        raise ValueError("Output directory is not empty; choose another output or use --resume")
    if options.start_stage:
        if checkpoint is not None:
            raise ValueError("--start-stage cannot be combined with --resume; use --initialize for a new run")
        stage_names = [stage["name"] for stage in plan["stages"]]
        if options.start_stage not in stage_names:
            raise ValueError(f"Unknown start stage {options.start_stage!r}; expected one of: {', '.join(stage_names)}")
        stage_index = stage_names.index(options.start_stage)
        progress = {"updates": 0, "episodes": 0, "transitions": 0}
    else:
        stage_index = saved.get("stage_index", 0)
        progress = saved.get("progress", {"updates": 0, "episodes": 0, "transitions": 0})
    completed = saved.get("completed", False)
    if completed:
        raise ValueError("Curriculum is complete; initialize a new plan to continue training")
    stage = plan["stages"][stage_index]
    worlds = SampledWorlds(stage["scenario"], plan["policy"], options.envs, options.device, options.library,
                          options.max_ticks, plan["seed"], saved.get("sampler"))
    try:
        if rollout_backend == "cuda-resident" and any(worlds.build_info.get(name) != 1 for name in (
            "cuda_async_observation_version", "cuda_action_decode_version", "cuda_resident_step_version",
            "cuda_resident_episode_version")):
            reason = "Native library lacks the required resident CUDA capabilities"
            if requested_backend == "cuda-resident":
                raise ValueError(f"CUDA resident rollout unavailable: {reason}")
            rollout_backend = "host"
        if checkpoint is not None and saved.get("rollout_backend", "host") != rollout_backend:
            raise ValueError("Checkpoint rollout backend differs; use --initialize for a new experiment")
        if requested_backend == "auto" and reason is not None:
            print(f"CUDA resident rollout unavailable; using host collection: {reason}")
        action_sizes = dict(zip(worlds.action_names, plan["policy"]["action_sizes"]))
        model = (BridgeActorCritic(worlds.observation_dim, descriptor, options.hidden_dim, action_sizes)
             if descriptor is not None else
             ActorCritic(worlds.observation_dim, options.hidden_dim, movement_bins=9, action_sizes=action_sizes)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=options.lr, eps=1e-5)
        pool = OpponentPool(plan["opponents"], plan["policy"], options.envs, options.device, plan["seed"], learner=model)
        pool.configure(stage["opponents"], stage["controls"])
        if checkpoint is not None:
            if (checkpoint.get("format") != "minecraft_sim_ppo" or checkpoint.get("version") != 1
                    or checkpoint["metadata"].get("reward_semantics") != (
                        bridge_reward_semantics(worlds.bridge_scenario["bridge"],
                                                worlds.bridge_scenario["arenas"][0]["environment"]["reward_weights"])
                        if worlds.bridge_scenario is not None else REWARD_SEMANTICS)
                    or checkpoint["metadata"]["hidden_dim"] != options.hidden_dim
                    or saved.get("rollout_steps") != options.rollout_steps
                    or saved.get("max_ticks_override") != options.max_ticks):
                raise ValueError("Research checkpoint architecture, reward semantics or rollout settings differ")
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            pool.load_state_dict(saved["pool"])
            torch.set_rng_state(checkpoint["rng_state"].cpu())
            if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu())
        elif options.initialize:
            initial = torch.load(options.initialize, map_location=device, weights_only=True)
            _validate_checkpoint_model(initial.get("metadata"), descriptor)
            from simulator.training.contract import from_legacy
            schema = initial["metadata"]["schema"]
            contract = schema if schema.get("format") == "minecraft_sim_policy" else from_legacy(schema)
            if fingerprint(contract) != fingerprint(plan["policy"]):
                raise ValueError("Initialization checkpoint policy contract differs")
            if descriptor is not None and initial["metadata"].get("hidden_dim") != options.hidden_dim:
                raise ValueError("Initialization checkpoint architecture differs")
            model.load_state_dict(initial["model"], strict=True)
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError("Learner checkpoint contains nonfinite parameters")
        if checkpoint is None:
            pool.capture(model, 0, stage["controls"], force=True)
        worlds.configure(stage)
        worlds.episode_metrics = EpisodeMetrics(options.envs, saved.get("episode_history", ()))
        resident = None
        if rollout_backend == "cuda-resident":
            resident = ResidentCollector(worlds, pool, plan["seed"] + stage_index)
            if checkpoint is not None:
                if "resident_rng" not in saved or "resident_episodes" not in saved:
                    raise ValueError("Resident checkpoint is missing sampler or native episode state")
                resident.pool.load_state_dict(saved["resident_rng"])
                resident.environment.set_cuda_reset_episodes(saved["resident_episodes"])
                resident.buffers["reset_mask"].fill_(1)
                resident.observations = resident.environment.reset_cuda_resident_async()
                resident.pool.reset(torch.ones(options.envs, dtype=torch.bool, device=resident.device))
            observations = resident.observations
        else:
            if checkpoint is not None and worlds.bridge_scenario is not None:
                if "bridge_episodes" not in saved:
                    raise ValueError("Bridge checkpoint is missing reset episode counters")
                worlds.restore_bridge_reset_episodes(saved["bridge_episodes"])
            observations = worlds.observe()
            pool.reset(np.ones(options.envs, dtype=bool))
            if worlds.bridge_scenario is None and worlds.step_reward_totals(pool.learner_slots) is None:
                print("Reward breakdown unavailable: rebuild minecraft_sim_training and restart/resume training "
                      "to record learner-only per-type rewards. Total rewards are still recorded.")
        output.mkdir(parents=True, exist_ok=True)
        base_update = 0 if checkpoint is None else checkpoint["update"]
        samples = 0 if checkpoint is None else checkpoint["samples"]
        rounds = 0 if checkpoint is None else checkpoint.get("completed_rounds", 0)
        evaluations = saved.get("evaluations", {})
        config = vars(options).copy()
        config["effective_rollout_backend"] = rollout_backend
        if reason is not None:
            config["rollout_fallback_reason"] = reason
        config["dashboard_replay_requests"] = True
        config["dashboard_analytics_requests"] = True
        run_metadata = research_metadata(worlds.groups[0][1], options.hidden_dim, plan["learner"])
        if descriptor is not None:
            run_metadata["bridge_observation"] = deepcopy(descriptor)
        run_metadata.update(learner="opponent_pool", controls=stage["controls"], rollout_backend=rollout_backend,
                            experiment=deployment_summary(plan, stage, stage_index))
        (output / "config.json").write_text(json.dumps({"options": config, "metadata": run_metadata,
            "experiment": plan, "training": {"start_update": base_update, "target_update": base_update + options.updates}},
            indent=2), encoding="utf-8")
        print(f"Research training: {options.envs} worlds; {worlds.observation_dim} observations; "
              f"stage {stage['name']}; rollout {rollout_backend}")
        if checkpoint is not None:
            print("Resumed schedule, samplers and weights; starting fresh episodes (not mid-episode continuation).")

        def save_snapshot(rule, snapshot_update, stage_updates):
            scenario = worlds.episode_scenario(0)
            manifest = research_metadata(worlds.groups[0][1], options.hidden_dim, plan["learner"])
            if descriptor is not None:
                manifest["bridge_observation"] = deepcopy(descriptor)
            manifest.update(schema=plan["policy"], contract=plan["policy"], environment=scenario,
                            scenario=scenario, learner="opponent_pool", controls=stage["controls"],
                            experiment=deployment_summary(plan, stage, stage_index),
                            snapshot={"name": rule["name"], "update": snapshot_update,
                                      "stage_updates": stage_updates})
            directory = output / "snapshots"
            directory.mkdir(exist_ok=True)
            temporary = directory / f"{rule['name']}.tmp"
            torch.save({"format": "minecraft_sim_ppo", "version": 1, "metadata": manifest,
                        "model": model.state_dict(), "update": snapshot_update, "samples": samples,
                        "completed_rounds": rounds}, temporary)
            destination = directory / f"{rule['name']}.pt"
            temporary.replace(destination)
            return str(destination.relative_to(output))

        if checkpoint is None:
            for rule in [*plan.get("snapshots", []), *stage.get("snapshots", [])]:
                if rule.get("at_updates") == 0:
                    save_snapshot(rule, base_update, 0)
        for update in range(base_update + 1, base_update + options.updates + 1):
            trace = os.environ.get("MCSIM_TRACE_PROGRESS") == "1"
            if trace:
                print(f"TRACE update={update} phase=collection_begin", flush=True)
            start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            fixed = fixed_indices(plan["policy"], stage["controls"])
            periodic_profile = options.profile_every > 0 and update % options.profile_every == 0
            analytics_request = None
            try:
                request = read_analytics_request(output)
                if (request is not None and request["status"] in ("pending", "running")
                        and mark_analytics_request_running(output, request["request_id"], update, stage["name"])):
                    analytics_request = request
            except (OSError, ValueError) as error:
                print(f"Dashboard analytics request could not be read: {error}")
            try:
                if resident is None:
                    rollout, observations, endings, timeouts = collect(
                        model, worlds, pool, observations, options.rollout_steps, device, fixed,
                        profile=periodic_profile or analytics_request is not None, learner=plan["learner"],
                        structured_bridge=descriptor is not None)
                else:
                    rollout, observations, endings, timeouts = resident.collect(
                        model, options.rollout_steps, fixed, learner=plan["learner"],
                        profile=periodic_profile or analytics_request is not None)
            except RuntimeError as error:
                if (resident is not None or analytics_request is None or periodic_profile
                        or "collection profiling" not in str(error)):
                    raise
                finish_analytics_request(output, analytics_request["request_id"], "failed",
                                         error=str(error)[:500])
                print(f"Dashboard analytics request failed: {error}")
                analytics_request = None
                rollout, observations, endings, timeouts = collect(
                    model, worlds, pool, observations, options.rollout_steps, device, fixed,
                    profile=False, learner=plan["learner"], structured_bridge=descriptor is not None)
            collection_seconds = time.perf_counter() - start
            if trace:
                print(f"TRACE update={update} phase=optimization_begin", flush=True)
            optimization_start = time.perf_counter()
            optimization_timer = OptimizationTimer(device, enabled=True) if analytics_request is not None else None
            statistics: dict[str, object] = dict(optimize(model, optimizer, rollout, options.epochs,
                                                         options.minibatch_size, learner=plan["learner"],
                                                         timer=optimization_timer))
            if optimization_timer is not None:
                statistics["optimization_profile"] = optimization_timer.finish()
            optimization_seconds = time.perf_counter() - optimization_start
            if trace:
                print(f"TRACE update={update} phase=optimization_done", flush=True)
            if "reward_types" in rollout:
                statistics["reward_types"] = rollout["reward_types"]
            statistics.update(rollout["episode_metrics"])
            if "collection_profile" in rollout:
                statistics["collection_profile"] = rollout["collection_profile"]
            transitions = options.envs * options.rollout_steps
            samples += transitions
            previous_rounds = rounds
            rounds += endings + timeouts
            progress = {"updates": progress["updates"] + 1, "episodes": progress["episodes"] + endings + timeouts,
                        "transitions": progress["transitions"] + transitions}
            pool.capture(model, update, stage["controls"])
            evaluation_seconds = 0.0
            evaluation_episodes = 0
            replay_seconds = 0.0
            devices = [torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                for suites, counter in ((plan["evaluations"], update),
                                        (stage.get("evaluations", {}), progress["updates"])):
                    for name, suite in suites.items():
                        if counter % suite["every_updates"] == 0:
                            evaluation_start = time.perf_counter()
                            torch.manual_seed(suite["seed"])
                            result = evaluate(model, plan, suite, options, device, fixed, opponent_pool=pool)
                            elapsed = time.perf_counter() - evaluation_start
                            result.update(seconds=elapsed, snapshot_update=update,
                                          stage=stage["name"], stage_index=stage_index)
                            evaluations[name] = result
                            evaluation_seconds += elapsed
                            evaluation_episodes += result["episodes"]
                replay_rounds = []
                if options.replay_every or replay_every_updates:
                    if replay_every_updates:
                        replay_rounds = [rounds] if update % replay_every_updates == 0 else []
                    else:
                        replay_rounds = list(range((previous_rounds // options.replay_every + 1) * options.replay_every,
                                                   rounds + 1, options.replay_every))
                try:
                    replay_request = read_request(output)
                except (OSError, ValueError) as error:
                    print(f"Dashboard replay request could not be read: {error}")
                    replay_request = None
                request_round = None
                if replay_request is not None and replay_request["status"] == "pending":
                    request_round = requested_replay_round(previous_rounds, rounds, replay_rounds)
                    if request_round is not None and request_round not in replay_rounds:
                        replay_rounds.append(request_round)
                if replay_rounds:
                    replay_start = time.perf_counter()
                    for replay_round in replay_rounds:
                        scenario = worlds.episode_scenario(0)
                        replay_seed = (update if replay_every_updates or not options.replay_every
                                       else replay_round // options.replay_every)
                        torch.manual_seed(plan["seed"] + replay_seed)
                        try:
                            replay_path = export_replay(model, plan, stage, options, scenario,
                                pool.provenance(0), device, replay_round, update,
                                options.replay_dir or output / "replays", opponent_pool=pool)
                        except Exception as error:
                            if replay_request is None or replay_round != request_round:
                                raise
                            finish_request(output, replay_request["request_id"], "failed", error=str(error)[:500])
                            print(f"Dashboard replay request failed: {error}")
                            continue
                        if replay_request is not None and replay_round == request_round:
                            finish_request(output, replay_request["request_id"], "completed",
                                replay=Path(replay_path).name, completed_rounds=replay_round, policy_update=update)
                    replay_seconds = time.perf_counter() - replay_start
            stage_complete = reached(stage["advance"], progress, evaluations)
            plan_complete = stage_complete and stage_index + 1 == len(plan["stages"])
            retained_snapshots = []
            for rules, counter, complete in ((plan.get("snapshots", []), update, plan_complete),
                                              (stage.get("snapshots", []), progress["updates"], stage_complete)):
                for rule in rules:
                    if rule.get("at_updates") != counter and not (rule.get("on_completion") and complete):
                        continue
                    retained_snapshots.append(save_snapshot(rule, update, progress["updates"]))
            iteration_seconds = time.perf_counter() - start
            statistics.update(update=update, samples=samples, stage=stage["name"], stage_index=stage_index,
                stage_progress=progress.copy(), completed_rounds=rounds, timeouts=timeouts,
                rollout_backend=rollout_backend,
                reward_mean=rollout["rewards"].mean().item(), reward_total=rollout["rewards"].sum(dtype=torch.float64).item(),
                collection_seconds=collection_seconds, update_seconds=optimization_seconds,
                evaluation_seconds=evaluation_seconds, evaluation_episodes=evaluation_episodes,
                replay_seconds=replay_seconds, iteration_seconds=iteration_seconds,
                evaluations=deepcopy(evaluations), player_transitions_per_second=transitions / iteration_seconds)
            statistics["terminated_matches" if worlds.bridge_scenario is not None else "terminated_duels"] = endings
            if "bridge_events" in rollout:
                statistics.update(rollout["bridge_events"])
            if retained_snapshots:
                statistics["snapshots"] = retained_snapshots
            if device.type == "cuda":
                statistics["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                statistics["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            if stage_complete:
                if stage_index + 1 == len(plan["stages"]):
                    completed = True
                else:
                    stage_index += 1
                    next_stage = plan["stages"][stage_index]
                    statistics["curriculum_boundary"] = {"reset_worlds": options.envs, "next_stage": next_stage["name"]}
                    statistics["next_stage_index"] = stage_index
                    recreate = resident is not None or requires_new_worlds(stage["scenario"], next_stage["scenario"])
                    if recreate:
                        worlds.close()
                        worlds = SampledWorlds(next_stage["scenario"], plan["policy"], options.envs, options.device,
                            options.library, options.max_ticks, plan["seed"] + stage_index)
                        worlds.configure(next_stage)
                    else:
                        worlds.configure(next_stage)
                        worlds.reset()
                    stage = next_stage
                    pool.configure(stage["opponents"], stage["controls"])
                    if resident is not None:
                        resident = ResidentCollector(worlds, pool, plan["seed"] + stage_index)
                        observations = resident.observations
                    else:
                        observations = worlds.observe()
                        pool.reset(np.ones(options.envs, dtype=bool))
                    progress = {"updates": 0, "episodes": 0, "transitions": 0}
                    evaluations = {}
                    for rule in stage.get("snapshots", []):
                        if rule.get("at_updates") == 0:
                            statistics.setdefault("snapshots", []).append(save_snapshot(rule, update, 0))
            statistics["completed_curriculum"] = completed
            statistics["elapsed_seconds"] = saved.get("elapsed_seconds", 0.0) + time.perf_counter() - run_started
            print(json.dumps(statistics, allow_nan=False))
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(statistics, allow_nan=False) + "\n")
            if analytics_request is not None:
                try:
                    finish_analytics_request(output, analytics_request["request_id"], "completed",
                                             update=update, stage=statistics["stage"],
                                             profile=rollout["collection_profile"],
                                             optimization_profile=statistics["optimization_profile"])
                except (OSError, ValueError) as error:
                    print(f"Dashboard analytics request could not be completed: {error}")
            if update % options.save_every == 0 or update == base_update + options.updates or completed:
                scenario = worlds.episode_scenario(0)
                manifest = research_metadata(worlds.groups[0][1], options.hidden_dim, plan["learner"])
                if descriptor is not None:
                    manifest["bridge_observation"] = deepcopy(descriptor)
                manifest.update(schema=plan["policy"], contract=plan["policy"], environment=scenario, scenario=scenario,
                                rollout_backend=rollout_backend,
                                learner="opponent_pool", controls=stage["controls"],
                                experiment=deployment_summary(plan, stage, stage_index))
                research = {"plan": plan, "plan_hash": plan_hash(plan), "stage_index": stage_index,
                            "progress": progress, "completed": completed, "evaluations": evaluations,
                            "sampler": worlds.sampler.state_dict(), "pool": pool.state_dict(),
                            "world_count": options.envs, "rollout_steps": options.rollout_steps,
                            "max_ticks_override": options.max_ticks,
                            "resume_mode": "fresh_episodes",
                            "rollout_backend": rollout_backend,
                            "episode_history": list(worlds.episode_metrics.history),
                            "elapsed_seconds": saved.get("elapsed_seconds", 0.0) + time.perf_counter() - run_started}
                if resident is not None:
                    research["resident_rng"] = resident.pool.state_dict()
                    research["resident_episodes"] = resident.environment.cuda_reset_episodes()
                elif worlds.bridge_scenario is not None:
                    research["bridge_episodes"] = worlds.bridge_reset_episodes()
                temporary = output / "latest.tmp"
                torch.save({"format": "minecraft_sim_ppo", "version": 1, "metadata": manifest,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                    "update": update, "samples": samples, "completed_rounds": rounds, "research": research,
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state() if device.type == "cuda" else None}, temporary)
                temporary.replace(output / "latest.pt")
                checkpoint_dir = output / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)
                checkpoint = checkpoint_dir / f"update-{update:08d}.pt"
                suffix = 1
                while checkpoint.exists():
                    checkpoint = checkpoint_dir / f"update-{update:08d}-{suffix:03d}.pt"
                    suffix += 1
                checkpoint_temporary = checkpoint.with_suffix(".tmp")
                shutil.copyfile(output / "latest.pt", checkpoint_temporary)
                checkpoint_temporary.replace(checkpoint)
            if completed:
                break
        print(f"Checkpoint: {output / 'latest.pt'}")
        return 0
    finally:
        worlds.close()

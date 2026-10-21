"""Opponent-driven PPO, reusable scenes, curriculum stages and held-out evaluation."""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from backend.fast_rl_model import ActorCritic, FastRLAgent
from simulator.training.contract import fingerprint
from simulator.training.controllers import FrozenSelfPlayController, SelfPlayController, create_controller
from simulator.training.environment import Environment, REWARD_NAMES, REWARD_SEMANTICS, ROOT
from simulator.training.plans import EpisodeSampler, plan_hash, reached, resolve_plan, weights
from simulator.training.policies import fixed_indices, policy_actions
from simulator.training.ppo import compute_gae, metadata, optimize
from simulator.training.timing import CollectionTimer


def scene_key(scenario):
    return [{"weight": arena.get("weight", 1), **{name: arena["environment"].get(name)
            for name in ("bounds", "blocks", "rules")}} for arena in scenario["arenas"]]


def deployment_summary(plan, stage, stage_index):
    return {"name": plan.get("name", "Experiment"), "stage": stage["name"], "stage_index": stage_index,
            "environment": stage.get("environment_name", stage["name"]), "arenas": stage.get("arena_names", []),
            "controllers": "shared_learner", "sampling": "fixed_checkpoint_episode"}


class SampledWorlds:
    """Group worlds by immutable scene; changing players never recreates terrain."""

    def __init__(self, scenario, contract, count, device, library, max_ticks, seed, state=None):
        self.resources = ExitStack()
        self.count, self.device, self.contract = count, device, contract
        self.max_ticks_override = max_ticks
        self.schema = contract
        self.observation_dim = contract["observation_dim"]
        self.action_names = tuple(contract["action_names"])
        self.action_count = len(self.action_names)
        self.sampler = EpisodeSampler(scenario, count, seed)
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
                    max_ticks, (seed + arena_id) % 2**63, contract=contract))
                environment.reset_players([initial[int(world)] for world in ids], np.ones(len(ids), dtype=bool))
                self.groups.append((ids, environment))
            self.max_ticks = max(environment.max_ticks for _, environment in self.groups)
            self.build_info = self.groups[0][1].build_info
        except BaseException:
            self.resources.close()
            raise

    def configure(self, stage):
        self.sampler.configure(stage["scenario"])
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

    def reset(self, mask=None, timer=None):
        timer = timer or CollectionTimer(self.device)
        mask = np.ones(self.count, dtype=bool) if mask is None else np.asarray(mask)
        if mask.shape != (self.count,) or mask.dtype.kind != "b":
            raise ValueError("Sampled reset mask must contain one boolean per world")
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

    def episode(self, world):
        for ids, environment in self.groups:
            found = np.flatnonzero(ids == world)
            if len(found):
                local = int(found[0])
                return environment.episode_scenario(local), environment.replay_frame(local)
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

    def act(self, observations, learner_actions):
        actions = np.empty((self.count, 2, learner_actions.shape[-1]), dtype=np.int64)
        ids = np.arange(self.count)
        actions[ids, self.learner_slots] = learner_actions
        for index, name in enumerate(self.names):
            selected = np.flatnonzero(self.assignments == index)
            if len(selected):
                opponent_actions = self.controllers[name].act(
                    observations[selected, 1 - self.learner_slots[selected]], indices=selected)
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


@torch.no_grad()
def collect(model, worlds, pool, observations, steps, device, fixed, profile=False):
    timer = CollectionTimer(device, profile)
    timer.count("worlds", worlds.count)
    timer.count("steps", steps)
    with worlds.profile_collection(timer):
        result = _collect(model, worlds, pool, observations, steps, device, fixed, timer)
    if timer.enabled:
        result[0]["collection_profile"] = timer.finish()
    return result


def _collect(model, worlds, pool, observations, steps, device, fixed, timer):
    with timer.phase("setup"):
        shape = (steps, worlds.count)
        rollout = {name: torch.empty(shape, dtype=torch.bool if name in ("terminated", "truncated")
                                    else torch.float32, device=device)
                   for name in ("log_probs", "values", "next_values", "rewards", "terminated", "truncated")}
        rollout["observations"] = torch.empty((*shape, worlds.observation_dim), device=device)
        rollout["actions"] = torch.empty((*shape, worlds.action_count), dtype=torch.long, device=device)
        rollout["fixed_actions"] = fixed.copy()
        reward_types = worlds.step_reward_totals(pool.learner_slots)
        if reward_types is not None:
            reward_types = dict.fromkeys(reward_types, 0.0)
        endings = timeouts = 0
        ids = np.arange(worlds.count)
    for step in range(steps):
        slots = pool.learner_slots.copy()
        with timer.phase("observation_transfer", cuda=True):
            current = torch.as_tensor(observations[ids, slots], device=device)
        with timer.phase("learner_inference", cuda=True):
            actions, log_probs, values, _ = policy_actions(model, current, worlds.action_names, fixed)
        with timer.phase("action_transfer", cuda=True):
            learner_actions = actions.cpu().numpy()
        with timer.phase("opponent_actions", cuda=True):
            combined_actions = pool.act(observations, learner_actions)
        with timer.phase("environment_step"):
            final, rewards, terminated, truncated = worlds.step(combined_actions)
        with timer.phase("reward_breakdown"):
            if reward_types is not None:
                contributions = worlds.step_reward_totals(slots)
                if contributions is None:
                    raise RuntimeError("Native reward breakdown support changed during rollout")
                for name, amount in contributions.items():
                    reward_types[name] += amount
        with timer.phase("next_value", cuda=True):
            _, next_values = model(torch.as_tensor(final[ids, slots], device=device))
        with timer.phase("rollout_storage", cuda=True):
            rollout["observations"][step] = current
            rollout["actions"][step] = actions
            for name, value in (("log_probs", log_probs), ("values", values), ("next_values", next_values),
                                ("rewards", rewards[ids, slots]), ("terminated", terminated), ("truncated", truncated)):
                rollout[name][step] = torch.as_tensor(value, device=device)
        with timer.phase("episode_bookkeeping"):
            done = terminated | truncated
            endings += int(terminated.sum())
            timeouts += int(truncated.sum())
        if done.any():
            if timer.enabled:
                timer.count("reset_calls")
                timer.count("reset_worlds", done.sum())
            with timer.phase("environment_reset"):
                observations = worlds.reset(done, timer=timer)
            with timer.phase("opponent_reset"):
                pool.reset(done)
        else:
            observations = final
    with timer.phase("gae", cuda=True):
        rollout["advantages"], rollout["returns"] = compute_gae(
            rollout["rewards"], rollout["values"], rollout["next_values"], rollout["terminated"],
            rollout["truncated"], FastRLAgent.GAMMA, FastRLAgent.GAE_LAMBDA)
    if reward_types is not None:
        rollout["reward_types"] = reward_types
    return rollout, observations, endings, timeouts


def _lost(actor, scenario):
    position = actor["position"]
    lower, size = scenario["bounds"]["min"], scenario["bounds"]["size"]
    return (actor["health"] <= 0 or position[0] < lower[0] or position[0] >= lower[0] + size[0]
            or position[2] < lower[2] or position[2] >= lower[2] + size[2] or position[1] < lower[1] - 4)


@torch.no_grad()
def evaluate(model, plan, suite, options, device, fixed, opponent_pool=None):
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
        episode_rewards = np.zeros(count)
        observations = worlds.observe()
        pool.reset(np.ones(count, dtype=bool))
        ids = np.arange(count)
        while finished < suite["episodes"]:
            current = torch.as_tensor(observations[ids, pool.learner_slots], device=device)
            actions, _, _, _ = policy_actions(model, current, worlds.action_names, fixed)
            observations, rewards, terminated, truncated = worlds.step(pool.act(observations, actions.cpu().numpy()))
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
                finished += 1
            if done.any() and finished < suite["episodes"]:
                observations = worlds.reset(done)
                pool.reset(done)
                episode_rewards[done] = 0
        return {"win_rate": wins / suite["episodes"], "reward_mean": total / suite["episodes"],
            "episodes": suite["episodes"], "worlds": count}
    finally:
        model.train(was_training)
        worlds.close()


@torch.no_grad()
def export_replay(model, plan, stage, options, scenario, provenance, device, rounds, update, directory,
                  opponent_pool=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"research-{rounds:09d}-{datetime.now():%Y%m%d-%H%M%S-%f}.jsonl"
    partial = destination.with_suffix(".jsonl.part")
    was_training = model.training
    model.eval()
    try:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode.json"
            path.write_text(json.dumps(scenario), encoding="utf-8")
            with Environment(1, options.device, path, options.library, options.max_ticks,
                             plan["seed"], contract=plan["policy"]) as environment:
                pool = OpponentPool({provenance["id"]: plan["opponents"][provenance["id"]]},
                                    plan["policy"], 1, options.device, plan["seed"], learner=model)
                pool.configure([{"id": provenance["id"], "weight": 1}], stage["controls"])
                controller = pool.controllers[provenance["id"]]
                if isinstance(controller, FrozenSelfPlayController) and opponent_pool is not None:
                    controller.copy_snapshot(opponent_pool.controllers[provenance["id"]],
                                             provenance["controller"]["snapshot_update"])
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
                        actions, _, _, _ = policy_actions(model,
                            torch.as_tensor(observations[:, slot], device=device), environment.action_names, fixed)
                        observations, _, terminated, truncated = environment.step(pool.act(observations, actions.cpu().numpy()))
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
    for name in ("hidden_dim", "lr", "epochs", "minibatch_size"):
        override = getattr(options, name)
        if override is not None:
            plan["learner"][name] = override
        setattr(options, name, plan["learner"][name])
    plan = resolve_plan(plan)
    saved = {} if checkpoint is None else checkpoint.get("research", {})
    if options.envs is None:
        options.envs = saved.get("world_count", 16 if options.device == "cpu" else 8192)
    if options.rollout_steps is None:
        options.rollout_steps = saved.get("rollout_steps", 128)
    if checkpoint is not None and options.max_ticks is None:
        options.max_ticks = saved.get("max_ticks_override")
    if checkpoint is not None:
        if checkpoint.get("research", {}).get("plan_hash") != plan_hash(plan):
            raise ValueError("Training plan differs from checkpoint; use --initialize for a new experiment")
        if checkpoint["research"]["world_count"] != options.envs:
            raise ValueError("Research resume requires the original world count")
    if options.replay_every and any(
            (options.max_ticks if options.max_ticks is not None else arena["environment"]["max_ticks"]) > 100000
            for stage in plan["stages"] for arena in stage["scenario"]["arenas"]):
        raise ValueError("Enabled replays require every training stage max_ticks <= 100000")
    torch.manual_seed(plan["seed"])
    output = (Path(options.output) if options.output else Path(options.resume).resolve().parent if options.resume else
              ROOT / "simulator/training/runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if output.exists() and any(output.iterdir()) and checkpoint is None:
        raise ValueError("Output directory is not empty; choose another output or use --resume")
    stage_index = saved.get("stage_index", 0)
    progress = saved.get("progress", {"updates": 0, "episodes": 0, "transitions": 0})
    completed = saved.get("completed", False)
    if completed:
        raise ValueError("Curriculum is complete; initialize a new plan to continue training")
    stage = plan["stages"][stage_index]
    worlds = SampledWorlds(stage["scenario"], plan["policy"], options.envs, options.device, options.library,
                          options.max_ticks, plan["seed"], saved.get("sampler"))
    try:
        model = ActorCritic(worlds.observation_dim, options.hidden_dim, movement_bins=9,
                            action_sizes=dict(zip(worlds.action_names, plan["policy"]["action_sizes"]))).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=options.lr, eps=1e-5)
        pool = OpponentPool(plan["opponents"], plan["policy"], options.envs, options.device, plan["seed"], learner=model)
        pool.configure(stage["opponents"], stage["controls"])
        if checkpoint is not None:
            if (checkpoint.get("format") != "minecraft_sim_ppo" or checkpoint.get("version") != 1
                    or checkpoint["metadata"].get("reward_semantics") != REWARD_SEMANTICS
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
            from simulator.training.contract import from_legacy
            schema = initial["metadata"]["schema"]
            contract = schema if schema.get("format") == "minecraft_sim_policy" else from_legacy(schema)
            if fingerprint(contract) != fingerprint(plan["policy"]):
                raise ValueError("Initialization checkpoint policy contract differs")
            model.load_state_dict(initial["model"], strict=True)
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError("Learner checkpoint contains nonfinite parameters")
        if checkpoint is None:
            pool.capture(model, 0, stage["controls"], force=True)
        worlds.configure(stage)
        observations = worlds.observe()
        pool.reset(np.ones(options.envs, dtype=bool))
        if worlds.step_reward_totals(pool.learner_slots) is None:
            print("Reward breakdown unavailable: rebuild minecraft_sim_training and restart/resume training "
                  "to record learner-only per-type rewards. Total rewards are still recorded.")
        output.mkdir(parents=True, exist_ok=True)
        base_update = 0 if checkpoint is None else checkpoint["update"]
        samples = 0 if checkpoint is None else checkpoint["samples"]
        rounds = 0 if checkpoint is None else checkpoint.get("completed_rounds", 0)
        evaluations = saved.get("evaluations", {})
        config = vars(options).copy()
        run_metadata = metadata(worlds.groups[0][1], options.hidden_dim)
        run_metadata.update(learner="opponent_pool", controls=stage["controls"],
                            experiment=deployment_summary(plan, stage, stage_index))
        (output / "config.json").write_text(json.dumps({"options": config, "metadata": run_metadata,
            "experiment": plan, "training": {"start_update": base_update, "target_update": base_update + options.updates}},
            indent=2), encoding="utf-8")
        print(f"Research training: {options.envs} worlds; {worlds.observation_dim} observations; stage {stage['name']}")
        if checkpoint is not None:
            print("Resumed schedule, samplers and weights; starting fresh episodes (not mid-episode continuation).")
        for update in range(base_update + 1, base_update + options.updates + 1):
            start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            fixed = fixed_indices(plan["policy"], stage["controls"])
            rollout, observations, endings, timeouts = collect(
                model, worlds, pool, observations, options.rollout_steps, device, fixed,
                profile=options.profile_every > 0 and update % options.profile_every == 0)
            collection_seconds = time.perf_counter() - start
            optimization_start = time.perf_counter()
            statistics: dict[str, object] = dict(optimize(model, optimizer, rollout, options.epochs, options.minibatch_size))
            optimization_seconds = time.perf_counter() - optimization_start
            if "reward_types" in rollout:
                statistics["reward_types"] = rollout["reward_types"]
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
                if options.replay_every:
                    replay_start = time.perf_counter()
                    for multiple in range(previous_rounds // options.replay_every + 1, rounds // options.replay_every + 1):
                        scenario, _ = worlds.episode(0)
                        torch.manual_seed(plan["seed"] + multiple)
                        export_replay(model, plan, stage, options, scenario, pool.provenance(0), device,
                            multiple * options.replay_every, update, options.replay_dir or output / "replays",
                            opponent_pool=pool)
                    replay_seconds = time.perf_counter() - replay_start
            stage_complete = reached(stage["advance"], progress, evaluations)
            plan_complete = stage_complete and stage_index + 1 == len(plan["stages"])
            retained_snapshots = []
            for rules, counter, complete in ((plan.get("snapshots", []), update, plan_complete),
                                              (stage.get("snapshots", []), progress["updates"], stage_complete)):
                for rule in rules:
                    if rule.get("at_updates") != counter and not (rule.get("on_completion") and complete):
                        continue
                    scenario, _ = worlds.episode(0)
                    manifest = metadata(worlds.groups[0][1], options.hidden_dim)
                    manifest.update(schema=plan["policy"], contract=plan["policy"], environment=scenario,
                                    scenario=scenario, learner="opponent_pool", controls=stage["controls"],
                                    experiment=deployment_summary(plan, stage, stage_index),
                                    snapshot={"name": rule["name"], "update": update,
                                              "stage_updates": progress["updates"]})
                    directory = output / "snapshots"
                    directory.mkdir(exist_ok=True)
                    temporary = directory / f"{rule['name']}.tmp"
                    torch.save({"format": "minecraft_sim_ppo", "version": 1, "metadata": manifest,
                                "model": model.state_dict(), "update": update, "samples": samples,
                                "completed_rounds": rounds}, temporary)
                    destination = directory / f"{rule['name']}.pt"
                    temporary.replace(destination)
                    retained_snapshots.append(str(destination.relative_to(output)))
            iteration_seconds = time.perf_counter() - start
            statistics.update(update=update, samples=samples, stage=stage["name"], stage_index=stage_index,
                stage_progress=progress.copy(), completed_rounds=rounds, terminated_duels=endings, timeouts=timeouts,
                reward_mean=rollout["rewards"].mean().item(), reward_total=rollout["rewards"].sum(dtype=torch.float64).item(),
                collection_seconds=collection_seconds, update_seconds=optimization_seconds,
                evaluation_seconds=evaluation_seconds, evaluation_episodes=evaluation_episodes,
                replay_seconds=replay_seconds, iteration_seconds=iteration_seconds,
                evaluations=deepcopy(evaluations), player_transitions_per_second=transitions / iteration_seconds)
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
                    if scene_key(next_stage["scenario"]) != scene_key(stage["scenario"]):
                        worlds.close()
                        worlds = SampledWorlds(next_stage["scenario"], plan["policy"], options.envs, options.device,
                            options.library, options.max_ticks, plan["seed"] + stage_index)
                        worlds.configure(next_stage)
                    else:
                        worlds.configure(next_stage)
                        worlds.reset()
                    stage = next_stage
                    observations = worlds.observe()
                    pool.configure(stage["opponents"], stage["controls"])
                    pool.reset(np.ones(options.envs, dtype=bool))
                    progress = {"updates": 0, "episodes": 0, "transitions": 0}
                    evaluations = {}
            statistics["completed_curriculum"] = completed
            statistics["elapsed_seconds"] = saved.get("elapsed_seconds", 0.0) + time.perf_counter() - run_started
            print(json.dumps(statistics, allow_nan=False))
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(statistics, allow_nan=False) + "\n")
            if update % options.save_every == 0 or update == base_update + options.updates or completed:
                scenario, _ = worlds.episode(0)
                manifest = metadata(worlds.groups[0][1], options.hidden_dim)
                manifest.update(schema=plan["policy"], contract=plan["policy"], environment=scenario, scenario=scenario,
                                learner="opponent_pool", controls=stage["controls"],
                                experiment=deployment_summary(plan, stage, stage_index))
                research = {"plan": plan, "plan_hash": plan_hash(plan), "stage_index": stage_index,
                            "progress": progress, "completed": completed, "evaluations": evaluations,
                            "sampler": worlds.sampler.state_dict(), "pool": pool.state_dict(),
                            "world_count": options.envs, "rollout_steps": options.rollout_steps,
                            "max_ticks_override": options.max_ticks,
                            "resume_mode": "fresh_episodes",
                            "elapsed_seconds": saved.get("elapsed_seconds", 0.0) + time.perf_counter() - run_started}
                temporary = output / "latest.tmp"
                torch.save({"format": "minecraft_sim_ppo", "version": 1, "metadata": manifest,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                    "update": update, "samples": samples, "completed_rounds": rounds, "research": research,
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state() if device.type == "cuda" else None}, temporary)
                temporary.replace(output / "latest.pt")
            if completed:
                break
        print(f"Checkpoint: {output / 'latest.pt'}")
        return 0
    finally:
        worlds.close()

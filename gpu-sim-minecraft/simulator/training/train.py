"""Synchronous, host-staged PPO training with shared CPU/CUDA Minecraft physics."""

import argparse
from contextlib import ExitStack
from datetime import datetime
import json
import os
from pathlib import Path
import time
from uuid import uuid4
import tempfile

import numpy as np
import torch

from backend.fast_rl_model import ActorCritic, FastRLAgent
from simulator.training.environment import ACTION_NAMES, Environment, REWARD_SEMANTICS, ROOT, SCHEMA, SCHEMAS, resolve_max_ticks


def sample_actions(model, observations, action_names=ACTION_NAMES):
    logits, values = model(observations)
    distributions = [torch.distributions.Categorical(logits=logits[name]) for name in action_names]
    actions = torch.stack([distribution.sample() for distribution in distributions], dim=-1)
    log_probs = sum(distribution.log_prob(actions[:, index])
                    for index, distribution in enumerate(distributions))
    return actions, log_probs, values


@torch.no_grad()
def compute_gae(rewards, values, next_values, terminated, truncated, gamma=0.99, gae_lambda=0.95):
    advantages = torch.empty_like(rewards)
    tail = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        delta = rewards[step] + gamma * next_values[step] * (~terminated[step]) - values[step]
        tail = delta + gamma * gae_lambda * (~(terminated[step] | truncated[step])) * tail
        advantages[step] = tail
    return advantages, advantages + values


class ReplaySamples:
    def __init__(self, environment, interval, directory, seed, completed_rounds=0):
        self.environment = environment
        self.interval = interval
        self.directory = Path(directory)
        self.seed = seed
        self.completed_rounds = completed_rounds
        self.exported = 0

    def advance(self, rounds, model, policy_update):
        previous = self.completed_rounds
        self.completed_rounds += rounds
        if self.interval:
            for multiple in range(previous // self.interval + 1, self.completed_rounds // self.interval + 1):
                self.export(model, multiple * self.interval, policy_update)

    @torch.no_grad()
    def export(self, model, round_number, policy_update):
        environment = self.environment
        self.directory.mkdir(parents=True, exist_ok=True)
        name = f"training-round-{round_number:09d}-{datetime.now():%Y%m%d-%H%M%S-%f}-{uuid4().hex[:8]}.jsonl"
        destination = self.directory / name
        partial = destination.with_suffix(".jsonl.part")
        device = next(model.parameters()).device
        devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        was_training = model.training
        try:
            with torch.random.fork_rng(devices=devices), partial.open("x", encoding="utf-8", newline="\n") as output:
                model.eval()
                sample_seed = (self.seed + round_number) % 2**63
                torch.random.default_generator.manual_seed(sample_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(sample_seed)
                written = 0

                def write(value):
                    nonlocal written
                    line = json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
                    written += len(line.encode("utf-8"))
                    if written > 64 * 1024 * 1024:
                        raise RuntimeError("Sample replay exceeds the Minecraft viewer's 64 MiB limit")
                    output.write(line)

                header = {
                    "format": "minecraft_sim_replay", "version": 3, "minecraft": "26.3",
                    "backend": environment.device.upper(), "players": 2, "ticks_per_second": 20,
                    "environment": environment.scenario, "policy": environment.contract,
                    "training": {"sample": "evaluation", "completed_rounds": round_number,
                                 "policy_update": policy_update},
                }
                write(header)
                observations = environment.reset()
                write(environment.replay_frame())
                for _ in range(environment.max_ticks):
                    actions, _, _ = sample_actions(model, torch.as_tensor(
                        observations.reshape(2, environment.observation_dim), device=device), environment.action_names)
                    observations, _, terminated, truncated = environment.step(actions.reshape(1, 2, environment.action_count).cpu().numpy())
                    frame = environment.replay_frame()
                    frame["terminated"] = bool(terminated[0])
                    frame["truncated"] = bool(truncated[0])
                    write(frame)
                    if terminated[0] or truncated[0]:
                        break
                else:
                    raise RuntimeError("Replay sample did not finish within the episode limit")
            if destination.exists():
                raise FileExistsError(destination)
            partial.rename(destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        finally:
            model.train(was_training)
        self.exported += 1
        print(f"Replay after {round_number} training rounds: {destination}")


@torch.no_grad()
def collect_rollout(model, environment, observations, steps, device, replays=None, policy_update=0):
    actors = environment.count * 2
    features = environment.observation_dim
    heads = environment.action_count
    initial_rewards = environment.reward_totals()
    shape = (steps, actors)
    rollout = {
        "observations": torch.empty((*shape, features), device=device),
        "actions": torch.empty((*shape, heads), dtype=torch.long, device=device),
        "log_probs": torch.empty(shape, device=device),
        "values": torch.empty(shape, device=device),
        "next_values": torch.empty(shape, device=device),
        "rewards": torch.empty(shape, device=device),
        "terminated": torch.empty(shape, dtype=torch.bool, device=device),
        "truncated": torch.empty(shape, dtype=torch.bool, device=device),
    }
    endings = timeouts = 0
    for step in range(steps):
        current = torch.as_tensor(observations.reshape(actors, features), device=device)
        actions, log_probs, values = sample_actions(model, current, environment.action_names)
        final_obs, rewards, terminated, truncated = environment.step(
            actions.reshape(environment.count, 2, heads).cpu().numpy())
        _, next_values = model(torch.as_tensor(final_obs.reshape(actors, features), device=device))
        rollout["observations"][step] = current
        rollout["actions"][step] = actions
        rollout["log_probs"][step] = log_probs
        rollout["values"][step] = values
        rollout["next_values"][step] = next_values
        rollout["rewards"][step] = torch.as_tensor(rewards.reshape(actors), device=device)
        rollout["terminated"][step] = torch.as_tensor(np.repeat(terminated, 2), device=device)
        rollout["truncated"][step] = torch.as_tensor(np.repeat(truncated, 2), device=device)
        done = terminated | truncated
        endings += int(terminated.sum())
        timeouts += int(truncated.sum())
        if replays is not None:
            replays.advance(int(done.sum()), model, policy_update)
        observations = environment.reset(done) if done.any() else final_obs
    rollout["advantages"], rollout["returns"] = compute_gae(
        rollout["rewards"], rollout["values"], rollout["next_values"],
        rollout["terminated"], rollout["truncated"], FastRLAgent.GAMMA, FastRLAgent.GAE_LAMBDA)
    if initial_rewards is not None:
        final_rewards = environment.reward_totals()
        rollout["reward_types"] = {name: final_rewards[name] - value for name, value in initial_rewards.items()}
    return rollout, observations, endings, timeouts


def optimize(model, optimizer, rollout, epochs, minibatch_size):
    observations = rollout["observations"].flatten(0, 1)
    actions = rollout["actions"].flatten(0, 1)
    old_log_probs = rollout["log_probs"].flatten()
    old_values = rollout["values"].flatten()
    advantages = rollout["advantages"].flatten()
    returns = rollout["returns"].flatten()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    statistics = torch.zeros(4, device=observations.device)
    updates = 0
    for _ in range(epochs):
        permutation = torch.randperm(observations.shape[0], device=observations.device)
        for start in range(0, observations.shape[0], minibatch_size):
            indices = permutation[start:start + minibatch_size]
            log_probs, values, entropy = model.evaluate_policy_actions(observations[indices], actions[indices])
            log_ratio = log_probs - old_log_probs[indices]
            ratio = log_ratio.exp()
            policy_loss = -torch.minimum(ratio * advantages[indices],
                ratio.clamp(1 - FastRLAgent.CLIP_EPS, 1 + FastRLAgent.CLIP_EPS) * advantages[indices]).mean()
            clipped_values = old_values[indices] + (values - old_values[indices]).clamp(
                -FastRLAgent.CLIP_EPS, FastRLAgent.CLIP_EPS)
            value_loss = 0.5 * torch.maximum((values - returns[indices]).square(),
                                             (clipped_values - returns[indices]).square()).mean()
            loss = policy_loss + FastRLAgent.VF_COEF * value_loss - FastRLAgent.ENT_COEF * entropy.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), FastRLAgent.MAX_GRAD_NORM)
            optimizer.step()
            statistics += torch.stack((loss.detach(), policy_loss.detach(), value_loss.detach(),
                                       entropy.detach().mean()))
            updates += 1
    return dict(zip(("loss", "policy_loss", "value_loss", "entropy"),
                    (statistics / updates).cpu().tolist()))


def metadata(environment, hidden_dim):
    return {"schema": environment.schema, "environment": environment.definition,
            "contract": environment.contract, "scenario": environment.scenario,
            "native_build": environment.build_info,
            "reward_semantics": REWARD_SEMANTICS,
            "reward_weights": environment.weights, "max_ticks": environment.max_ticks,
            "hidden_dim": hidden_dim, "players_per_environment": 2, "learner": "both_shared_policy"}


def save_checkpoint(path, model, optimizer, configuration, manifest, update, samples, completed_rounds=0):
    temporary = path.with_suffix(".tmp")
    torch.save({"format": "minecraft_sim_ppo", "version": 1, "metadata": manifest,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": configuration, "update": update, "samples": samples,
                "completed_rounds": completed_rounds,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None}, temporary)
    temporary.replace(path)


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="Select both physics and policy device (default cpu)")
    parser.add_argument("--schema", choices=SCHEMAS, default="v1",
                        help="v1: sword only (default); v2: sword/axe/offhand shield")
    parser.add_argument("--envs", type=positive, help="Default: 16 CPU / 8192 CUDA worlds")
    parser.add_argument("--updates", type=positive, default=100, help="Additional PPO updates to run")
    parser.add_argument("--rollout-steps", type=positive, default=128)
    parser.add_argument("--epochs", type=positive, default=4)
    parser.add_argument("--minibatch-size", type=positive, default=1024)
    parser.add_argument("--hidden-dim", type=positive, default=256)
    parser.add_argument("--max-ticks", type=positive,
                        help="Override max_ticks from the environment JSON (default: 300)")
    parser.add_argument("--threads", type=positive, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=FastRLAgent.LR)
    parser.add_argument("--environment", help="Environment JSON; default environments/duel.json")
    parser.add_argument("--library", help="Explicit native training DLL/SO path")
    parser.add_argument("--output", help="Run directory; default timestamp under simulator/training/runs")
    parser.add_argument("--resume", help="Resume a compatible simulator training checkpoint")
    parser.add_argument("--save-every", type=positive, default=10)
    parser.add_argument("--replay-every", type=int, default=0,
                        help="Export a complete policy sample every N completed training duels; 0 disables")
    parser.add_argument("--replay-dir", help="Replay directory; default OUTPUT/replays")
    options = parser.parse_args(argv)
    options.envs = options.envs if options.envs is not None else (16 if options.device == "cpu" else 8192)
    if (options.envs > 65536 or (options.max_ticks is not None and options.max_ticks > 1000000)
            or not 0 <= options.seed < 2**63):
        parser.error("envs <= 65536, max-ticks <= 1000000 and 0 <= seed < 2**63 are required")
    if not np.isfinite(options.lr) or options.lr <= 0:
        parser.error("lr must be finite and positive")
    if (options.replay_every < 0 or (options.replay_every and options.max_ticks is not None
                                    and options.max_ticks > 100000)):
        parser.error("replay-every must be nonnegative; enabled replays require max-ticks <= 100000")
    if options.replay_dir and not options.replay_every:
        parser.error("replay-dir requires --replay-every N")
    return options


def main(argv=None):
    options = arguments(argv)
    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA mode needs a CUDA-enabled PyTorch install and an available GPU")
    os.environ["OMP_NUM_THREADS"] = str(options.threads)
    torch.set_num_threads(options.threads)
    torch.manual_seed(options.seed)
    if options.device == "cuda":
        torch.cuda.manual_seed_all(options.seed)
    device = torch.device(options.device)
    output = Path(options.output) if options.output else ROOT / "simulator" / "training" / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if output.exists() and any(output.iterdir()) and not options.resume:
        raise ValueError("Output directory is not empty; choose another output or use --resume")
    with ExitStack() as resources:
        checkpoint = torch.load(options.resume, map_location=device, weights_only=True) if options.resume else None
        if checkpoint is not None and checkpoint.get("metadata", {}).get("reward_semantics") != REWARD_SEMANTICS:
            raise ValueError("Checkpoint reward semantics do not match health-normalized rewards; start a new run")
        saved_schema = checkpoint.get("metadata", {}).get("schema") if checkpoint is not None else None
        if saved_schema in SCHEMAS.values() and saved_schema != SCHEMAS[options.schema]:
            raise ValueError("Checkpoint schema does not match the requested preset")
        environment_path = options.environment
        if checkpoint is not None and not environment_path and "environment" in checkpoint.get("metadata", {}):
            saved_directory = resources.enter_context(tempfile.TemporaryDirectory())
            environment_path = Path(saved_directory) / "environment.json"
            environment_path.write_text(json.dumps(checkpoint["metadata"]["environment"], allow_nan=False), encoding="utf-8")
        options.max_ticks = resolve_max_ticks(environment_path, options.max_ticks)
        if options.replay_every and options.max_ticks > 100000:
            raise ValueError("Enabled replays require max_ticks <= 100000")
        environment = resources.enter_context(Environment(options.envs, options.device, environment_path,
            options.library, options.max_ticks, options.seed, schema=options.schema))
        model = ActorCritic(environment.observation_dim, options.hidden_dim, movement_bins=9,
                            action_sizes=dict(zip(environment.action_names, environment.action_sizes))).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=options.lr, eps=1e-5)
        manifest = metadata(environment, options.hidden_dim)
        base_update = samples = completed_rounds = 0
        if options.resume:
            saved_manifest = checkpoint.get("metadata", {})
            saved_conditions = {key: value for key, value in saved_manifest.items() if key != "native_build"}
            comparable = {key: value for key, value in manifest.items() if key in saved_conditions}
            if (checkpoint.get("format") != "minecraft_sim_ppo" or checkpoint.get("version") != 1
                    or saved_conditions != comparable or not {"schema", "environment", "reward_weights", "max_ticks", "hidden_dim",
                        "players_per_environment", "learner"}.issubset(saved_manifest)):
                raise ValueError("Checkpoint schema, environment or task configuration does not match")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            base_update, samples = checkpoint["update"], checkpoint["samples"]
            completed_rounds = checkpoint.get("completed_rounds", 0)
            torch.set_rng_state(checkpoint["rng_state"].cpu())
            if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu())
        sample_environment = None
        if options.replay_every:
            sample_directory = resources.enter_context(tempfile.TemporaryDirectory())
            sample_path = Path(sample_directory) / "scenario.json"
            sample_path.write_text(json.dumps(environment.scenario), encoding="utf-8")
            sample_environment = resources.enter_context(Environment(1, options.device, sample_path,
                options.library, options.max_ticks, options.seed, environment.weights, contract=environment.contract))
            if sample_environment.scenario != environment.scenario:
                raise ValueError("Replay arena differs from the cached training arena")
        replays = ReplaySamples(sample_environment, options.replay_every,
                               options.replay_dir or output / "replays", options.seed, completed_rounds)
        output.mkdir(parents=True, exist_ok=True)
        configuration = vars(options)
        (output / "config.json").write_text(json.dumps({"options": configuration, "metadata": manifest}, indent=2), encoding="utf-8")
        observations = environment.observe()
        print(f"Physics + policy: {device}; worlds: {environment.count}; players/world: 2; host-staged transfers")
        print(f"Policy: {environment.schema.get('id', 'configured')} ({environment.observation_dim} features, {environment.action_count} heads)")
        print(f"Native: {environment.build_info['library_path']} (ABI {environment.build_info['abi']}, source {environment.build_info['source_hash'][:12]})")
        if environment.reward_totals() is None:
            print("Reward breakdown unavailable: rebuild minecraft_sim_training to enable per-type metrics.")
        for update in range(base_update + 1, base_update + options.updates + 1):
            start = time.perf_counter()
            rollout, observations, endings, timeouts = collect_rollout(
                model, environment, observations, options.rollout_steps, device, replays, update - 1)
            collection_seconds = time.perf_counter() - start
            statistics = optimize(model, optimizer, rollout, options.epochs, options.minibatch_size)
            elapsed = time.perf_counter() - start
            transitions = options.rollout_steps * environment.count * 2
            samples += transitions
            statistics.update(update=update, samples=samples, terminated_duels=endings, timeouts=timeouts,
                completed_rounds=replays.completed_rounds, replays_exported=replays.exported,
                reward_mean=rollout["rewards"].mean().item(),
                reward_total=rollout["rewards"].sum(dtype=torch.float64).item(), collection_seconds=collection_seconds,
                update_seconds=elapsed - collection_seconds, player_transitions_per_second=transitions / elapsed)
            if "reward_types" in rollout:
                statistics["reward_types"] = rollout["reward_types"]
            print(json.dumps(statistics, allow_nan=False))
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(statistics, allow_nan=False) + "\n")
            if update % options.save_every == 0 or update == base_update + options.updates:
                save_checkpoint(output / "latest.pt", model, optimizer, configuration, manifest,
                                update, samples, replays.completed_rounds)
        print(f"Checkpoint: {output / 'latest.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
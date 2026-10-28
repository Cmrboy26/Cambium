"""Task-independent PPO optimization and terminal-aware advantage estimation."""

import torch

from simulator.training.compiler import LEARNER_DEFAULTS
from simulator.training.environment import REWARD_SEMANTICS
from simulator.training.policies import policy_actions
from simulator.training.timing import OptimizationTimer


OPTIMIZATION_CUDA_MINIBATCH_LIMIT = 16


@torch.no_grad()
def compute_gae(rewards, values, next_values, terminated, truncated, gamma=0.99, gae_lambda=0.95):
    advantages = torch.empty_like(rewards)
    tail = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        delta = rewards[step] + gamma * next_values[step] * (~terminated[step]) - values[step]
        tail = delta + gamma * gae_lambda * (~(terminated[step] | truncated[step])) * tail
        advantages[step] = tail
    return advantages, advantages + values


def optimize(model, optimizer, rollout, epochs, minibatch_size, learner=None, timer=None):
    learner = LEARNER_DEFAULTS if learner is None else learner
    timer = OptimizationTimer(rollout["observations"].device) if timer is None else timer
    with timer.phase("setup"):
        observations = rollout["observations"].flatten(0, 1)
        actions = rollout["actions"].flatten(0, 1)
        old_log_probs = rollout["log_probs"].flatten()
        old_values = rollout["values"].flatten()
        advantages = rollout["advantages"].flatten()
        returns = rollout["returns"].flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        statistics = torch.zeros(4, device=observations.device)
    timer.count("samples", observations.shape[0])
    timer.count("epochs", epochs)
    updates = 0
    for _ in range(epochs):
        with timer.phase("shuffle"):
            permutation = torch.randperm(observations.shape[0], device=observations.device)
        for start in range(0, observations.shape[0], minibatch_size):
            cuda_sample = updates < OPTIMIZATION_CUDA_MINIBATCH_LIMIT
            with timer.phase("policy_evaluation", cuda=cuda_sample):
                indices = permutation[start:start + minibatch_size]
                _, log_probs, values, entropy = policy_actions(model, observations[indices], model.action_names,
                                                               rollout.get("fixed_actions"), actions[indices])
            with timer.phase("loss", cuda=cuda_sample):
                ratio = (log_probs - old_log_probs[indices]).exp()
                policy_loss = -torch.minimum(ratio * advantages[indices],
                    ratio.clamp(1 - learner["clip_eps"], 1 + learner["clip_eps"]) * advantages[indices]).mean()
                clipped_values = old_values[indices] + (values - old_values[indices]).clamp(
                    -learner["clip_eps"], learner["clip_eps"])
                value_loss = 0.5 * torch.maximum((values - returns[indices]).square(),
                                                 (clipped_values - returns[indices]).square()).mean()
                loss = policy_loss + learner["vf_coef"] * value_loss - learner["ent_coef"] * entropy.mean()
            with timer.phase("finite_check", cuda=cuda_sample):
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite PPO loss")
            with timer.phase("backward", cuda=cuda_sample):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
            with timer.phase("gradient_clip", cuda=cuda_sample):
                torch.nn.utils.clip_grad_norm_(model.parameters(), learner["max_grad_norm"])
            with timer.phase("optimizer_step", cuda=cuda_sample):
                optimizer.step()
            with timer.phase("statistics", cuda=cuda_sample):
                statistics += torch.stack((loss.detach(), policy_loss.detach(), value_loss.detach(), entropy.detach().mean()))
            updates += 1
    timer.count("minibatches", updates)
    if timer.device.type == "cuda":
        timer.count("cuda_profiled_minibatches", min(updates, OPTIMIZATION_CUDA_MINIBATCH_LIMIT))
    with timer.phase("results"):
        return dict(zip(("loss", "policy_loss", "value_loss", "entropy"), (statistics / updates).cpu().tolist()))


def metadata(environment, hidden_dim, learner=None):
    result = {"schema": environment.schema, "environment": environment.definition,
            "contract": environment.contract, "scenario": environment.scenario,
            "native_build": environment.build_info, "reward_semantics": REWARD_SEMANTICS,
            "reward_weights": environment.weights, "max_ticks": environment.max_ticks,
            "hidden_dim": hidden_dim, "players_per_environment": 2, "learner": "opponent_pool"}
    if learner is not None:
        result["learner_config"] = dict(learner)
    return result

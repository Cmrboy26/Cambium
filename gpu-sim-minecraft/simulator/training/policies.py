"""Policy-distribution constraints shared by collection, optimization and inference."""

import torch


def fixed_indices(contract, controls):
    fixed = controls.get("fixed", {})
    if set(controls) - {"fixed"} or not isinstance(fixed, dict):
        raise ValueError("Controls must contain a fixed-action object")
    actions = {action["name"]: action["values"] for action in contract["actions"]}
    if set(fixed) - set(actions):
        raise ValueError("Fixed control is not present in the policy contract")
    result = {}
    for name, value in fixed.items():
        choices = actions[name]
        matches = [index for index, choice in enumerate(choices)
                   if type(choice) is type(value) and choice == value
                   or type(choice) in (int, float) and type(value) in (int, float) and choice == value]
        if not matches:
            raise ValueError(f"Fixed control {name} has no semantic choice {value!r}")
        result[name] = matches[0]
    return result


def policy_actions(model, observations, action_names, fixed=None, actions=None):
    logits, values = model(observations)
    fixed = {} if fixed is None else fixed
    log_probs = torch.zeros_like(values)
    entropy = torch.zeros_like(values)
    chosen = []
    for index, name in enumerate(action_names):
        if name in fixed:
            action = torch.full_like(values, fixed[name], dtype=torch.long)
            if actions is not None and not torch.equal(action, actions[:, index]):
                raise ValueError("Rollout actions disagree with their fixed controls")
        else:
            distribution = torch.distributions.Categorical(logits=logits[name])
            action = distribution.sample() if actions is None else actions[:, index]
            log_probs = log_probs + distribution.log_prob(action)
            entropy = entropy + distribution.entropy()
        chosen.append(action)
    return torch.stack(chosen, dim=-1), log_probs, values, entropy

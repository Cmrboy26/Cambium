"""Bounded standalone PPO minibatch benchmark over one captured rollout."""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from backend.fast_rl_model import ActorCritic
from simulator.training.compiler import LEARNER_DEFAULTS
from simulator.training.contract import fingerprint, from_legacy
from simulator.training.environment import REWARD_SEMANTICS, SCHEMA
from simulator.training.policies import fixed_indices, policy_actions
from simulator.training.plans import plan_hash, resolve_plan
from simulator.training.ppo import compute_gae, optimize
from simulator.training.research import OpponentPool, SampledWorlds, collect


BENCHMARK_EPOCHS = 2
MAX_WORLDS = 8192
MAX_ROLLOUT_STEPS = 128
MAX_SAMPLE_COUNT = 1_048_576
MAX_PROFILE_SAMPLES = 32_768
MAX_PROFILE_MINIBATCHES = 4
ROLLOUT_TENSORS = ("observations", "actions", "log_probs", "values", "advantages", "returns")


def parse_minibatch_sizes(value):
    try:
        sizes = [int(part.strip()) for part in value.split(",")]
    except (AttributeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Expected comma-separated positive minibatch sizes") from error
    if not sizes or any(size < 1 or size > MAX_SAMPLE_COUNT for size in sizes):
        raise argparse.ArgumentTypeError(f"Minibatch sizes must be from 1 to {MAX_SAMPLE_COUNT}")
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("Minibatch sizes must not contain duplicates")
    return sizes


def validate_benchmark_sizes(worlds, rollout_steps, minibatch_sizes, max_samples=MAX_SAMPLE_COUNT):
    for name, value, maximum in (("envs", worlds, MAX_WORLDS),
                                 ("rollout_steps", rollout_steps, MAX_ROLLOUT_STEPS),
                                 ("max_samples", max_samples, MAX_SAMPLE_COUNT)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    if (not isinstance(minibatch_sizes, (list, tuple)) or not minibatch_sizes
            or any(type(size) is not int or not 1 <= size <= MAX_SAMPLE_COUNT
                   for size in minibatch_sizes)
            or len(set(minibatch_sizes)) != len(minibatch_sizes)):
        raise ValueError("minibatch_sizes must be unique positive integers within the benchmark limit")
    samples = worlds * rollout_steps
    if samples > max_samples:
        raise ValueError(f"Rollout has {samples} samples, exceeding --max-samples {max_samples}")
    return samples


def optimizer_steps(num_samples, minibatch_size, epochs=BENCHMARK_EPOCHS):
    if type(num_samples) is not int or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    if type(minibatch_size) is not int or minibatch_size < 1:
        raise ValueError("minibatch_size must be a positive integer")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    return epochs * math.ceil(num_samples / minibatch_size)


def synchronize_device(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def synchronized_call(operation, device):
    synchronize_device(device)
    started = time.perf_counter()
    result = operation()
    synchronize_device(device)
    return result, time.perf_counter() - started


def clone_rollout(rollout):
    cloned = {}
    for name, value in rollout.items():
        if isinstance(value, torch.Tensor):
            cloned[name] = value.detach().clone()
        elif isinstance(value, np.ndarray):
            cloned[name] = value.copy()
        else:
            cloned[name] = deepcopy(value)
    return cloned


def clone_model(model, device):
    return deepcopy(model).to(device)


def subset_rollout(rollout, num_samples):
    total = int(rollout["observations"].shape[0] * rollout["observations"].shape[1])
    if type(num_samples) is not int or not 1 <= num_samples <= total:
        raise ValueError(f"num_samples must be from 1 to {total}")
    result = {}
    for name in ROLLOUT_TENSORS:
        value = rollout[name]
        flattened = value.reshape(total, *value.shape[2:])
        result[name] = flattened[:num_samples].reshape(1, num_samples, *value.shape[2:]).clone()
    result["fixed_actions"] = deepcopy(rollout.get("fixed_actions", {}))
    return result


def rollout_sha256(rollout):
    digest = hashlib.sha256()
    for name in ROLLOUT_TENSORS:
        value = rollout[name]
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(value.shape), str(value.dtype))).encode("ascii"))
        for chunk in value.detach().split(8, dim=0):
            array = chunk.contiguous().cpu().numpy()
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def resize_sampler_state(state, worlds):
    if not isinstance(state, dict) or len(state.get("arenas", [])) < worlds:
        raise ValueError("Checkpoint sampler state does not cover the requested worlds")
    resized = deepcopy(state)
    resized["arenas"] = resized["arenas"][:worlds]
    resized["cache"] = resized["cache"][:worlds]
    return resized


def resize_pool_state(state, worlds):
    if not isinstance(state, dict) or not isinstance(state.get("controllers"), dict):
        raise ValueError("Checkpoint opponent-pool state is missing")
    resized = deepcopy(state)
    for controller in resized["controllers"].values():
        provenance = controller.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("Checkpoint opponent controller provenance is missing")
        if provenance.get("count", worlds) < worlds:
            raise ValueError("Checkpoint opponent state does not cover the requested worlds")
        provenance["count"] = worlds
        for name in ("directions", "remaining", "generators", "ticks", "snapshot_updates"):
            if name in controller:
                values = controller[name]
                if not isinstance(values, list) or len(values) < worlds:
                    raise ValueError(f"Checkpoint per-actor {name} state does not cover the requested worlds")
                controller[name] = values[:worlds]
    return resized


def validate_pool_device(state, device):
    requested = torch.device(device)
    for name, controller in state["controllers"].items():
        saved = controller.get("provenance", {}).get("device")
        if saved is not None and torch.device(saved).type != requested.type:
            raise ValueError(f"Controller {name} has {saved} RNG state; real rollout capture requires "
                             f"the checkpoint device type, not {requested.type}")


def safe_metadata(value):
    if isinstance(value, dict):
        return {str(key): safe_metadata(item) for key, item in value.items()
                if not any(term in str(key).lower()
                           for term in ("path", "token", "password", "secret", "credential", "authorization"))}
    if isinstance(value, (list, tuple)):
        return [safe_metadata(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def safe_controller_metadata(value):
    allowed = {"type", "controller", "deterministic", "every_updates", "device", "sha256",
               "snapshot_update"}
    return {key: safe_metadata(item) for key, item in value.items() if key in allowed}


def safe_build_info(value):
    return safe_metadata(value) if isinstance(value, dict) else {}


def controller_catalog(pool, definitions):
    catalog = {}
    for name, controller in pool.controllers.items():
        definition = definitions[name]
        details = {"type": definition.get("type")}
        for key in ("controller", "deterministic", "every_updates", "sha256"):
            if key in definition:
                details[key] = definition[key]
        details.update(safe_controller_metadata(controller.provenance))
        catalog[name] = safe_metadata(details)
    return catalog


def tracked_pool_act(pool):
    steps = []
    original = pool.act

    def act(observations, learner_actions):
        per_opponent = {}
        for index, name in enumerate(pool.names):
            selected = pool.assignments == index
            count = int(selected.sum())
            if not count:
                continue
            slots = np.bincount(pool.learner_slots[selected], minlength=2).tolist()
            controller = pool.controllers[name]
            snapshots = getattr(controller, "snapshot_updates", None)
            snapshot_counts = (dict(Counter(int(value) for value in snapshots[selected]))
                               if snapshots is not None else {})
            per_opponent[name] = {"worlds": count, "learner_slots": slots,
                                  "snapshot_updates": {str(key): value
                                                       for key, value in snapshot_counts.items()}}
        steps.append({"step": len(steps), "opponents": per_opponent})
        return original(observations, learner_actions)

    pool.act = act
    return steps, original


def set_seed(seed, device):
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)


def make_optimizer(model, optimizer_state, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    optimizer.load_state_dict(deepcopy(optimizer_state))
    return optimizer


def run_trial(template_model, optimizer_state, rollout, batch_size, learner, device, seed):
    device = torch.device(device)
    if device.type == "cuda":
        synchronize_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    setup_started = time.perf_counter()
    model = clone_model(template_model, device)
    optimizer = make_optimizer(model, optimizer_state, learner["lr"])
    trial_rollout = clone_rollout(rollout)
    synchronize_device(device)
    setup_seconds = time.perf_counter() - setup_started
    step_count = 0
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    set_seed(seed, device)
    statistics, elapsed = synchronized_call(
        lambda: optimize(model, optimizer, trial_rollout, BENCHMARK_EPOCHS, batch_size, learner=learner), device)
    if step_count != optimizer_steps(int(trial_rollout["observations"].shape[0]
                                         * trial_rollout["observations"].shape[1]), batch_size):
        raise RuntimeError("Observed optimizer step count differs from the minibatch schedule")
    if not all(math.isfinite(float(value)) for value in statistics.values()):
        raise RuntimeError("PPO benchmark produced non-finite loss statistics")
    peak = ({"pytorch_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
             "pytorch_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
             "native_allocations_included": False} if device.type == "cuda" else None)
    return {"seconds": elapsed, "setup_seconds": setup_seconds, "optimizer_steps": step_count,
            "statistics": statistics, "memory_peak": peak}


def synthetic_rollout(model, worlds, steps, contract, device, seed):
    set_seed(seed, device)
    shape = (steps, worlds)
    observations = torch.randn((*shape, contract["observation_dim"]), device=device)
    flat_observations = observations.reshape(-1, contract["observation_dim"])
    with torch.no_grad():
        actions, log_probs, values, _ = policy_actions(model, flat_observations, model.action_names)
        next_observations = torch.randn_like(flat_observations)
        _, next_values = model(next_observations)
        rewards = torch.randn(shape, device=device)
        terminated = torch.zeros(shape, dtype=torch.bool, device=device)
        truncated = torch.zeros_like(terminated)
        terminated[-1, 0] = True
        if worlds > 1:
            truncated[-1, 1] = True
        values = values.reshape(shape)
        next_values = next_values.reshape(shape)
        log_probs = log_probs.reshape(shape)
        advantages, returns = compute_gae(rewards, values, next_values, terminated, truncated)
    actions = actions.reshape(*shape, len(model.action_names))
    return {"observations": observations, "actions": actions, "log_probs": log_probs, "values": values,
            "next_values": next_values, "rewards": rewards, "terminated": terminated,
            "truncated": truncated, "advantages": advantages, "returns": returns,
            "fixed_actions": {}}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_baseline(path):
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ValueError("--checkpoint must name an existing frozen PPO checkpoint")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"Could not load trusted weights-only checkpoint: {error}") from error
    if (not isinstance(checkpoint, dict) or checkpoint.get("format") != "minecraft_sim_ppo"
            or checkpoint.get("version") != 1 or not isinstance(checkpoint.get("research"), dict)):
        raise ValueError("Checkpoint must be a version-1 research PPO checkpoint")
    research = checkpoint["research"]
    if not isinstance(research.get("plan"), dict) or not isinstance(research.get("pool"), dict):
        raise ValueError("Checkpoint does not contain its resolved plan and opponent state")
    plan = resolve_plan(research["plan"])
    if research.get("plan_hash") != plan_hash(plan):
        raise ValueError("Checkpoint resolved plan hash does not match its embedded plan")
    if checkpoint["metadata"].get("reward_semantics") != REWARD_SEMANTICS:
        raise ValueError("Checkpoint reward semantics are incompatible with this trainer")
    policy = plan["policy"]
    descriptors = [checkpoint["metadata"].get(key) for key in ("contract", "schema")
                   if checkpoint["metadata"].get(key) is not None]
    if not descriptors or any(fingerprint(value) != fingerprint(policy) for value in descriptors):
        raise ValueError("Checkpoint policy contract differs from its resolved training plan")
    stage_index = research.get("stage_index")
    if type(stage_index) is not int or not 0 <= stage_index < len(plan["stages"]):
        raise ValueError("Checkpoint active stage index is invalid")
    hidden_dim = checkpoint["metadata"].get("hidden_dim")
    if type(hidden_dim) is not int or not 1 <= hidden_dim <= 4096:
        raise ValueError("Checkpoint hidden_dim is invalid")
    if not isinstance(checkpoint.get("model"), dict) or not isinstance(checkpoint.get("optimizer"), dict):
        raise ValueError("Checkpoint must include frozen model weights and Adam state")
    if type(checkpoint.get("update")) is not int or type(checkpoint.get("samples")) is not int:
        raise ValueError("Checkpoint update/sample provenance is invalid")
    return checkpoint_path, checkpoint, plan, stage_index


def hardware_metadata(device):
    result = {"device": str(device), "python": platform.python_version(), "torch": torch.__version__,
              "numpy": np.__version__, "platform": platform.system(), "machine": platform.machine(),
              "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count()}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(cuda_runtime=torch.version.cuda, gpu=properties.name,
                      gpu_compute_capability=f"{properties.major}.{properties.minor}",
                      gpu_total_memory_bytes=int(properties.total_memory))
    return result


def process_peak_rss_bytes():
    try:
        import resource
    except ImportError:
        return None
    amount = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(amount if sys.platform == "darwin" else amount * 1024)


def validate_output_path(path, checkpoint_path=None):
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise ValueError("Output directory must be new; existing files are never overwritten")
    if checkpoint_path is not None:
        source_dir = Path(checkpoint_path).resolve().parent
        if output == source_dir or source_dir in output.parents:
            raise ValueError("Benchmark output must not be inside the checkpoint run directory")
    return output


def profile_trace(template_model, optimizer_state, rollout, batch_size, learner, device, seed,
                  directory, profile_minibatches):
    sample_count = min(int(rollout["observations"].shape[0] * rollout["observations"].shape[1]),
                       batch_size * profile_minibatches, MAX_PROFILE_SAMPLES)
    profiled_rollout = subset_rollout(rollout, sample_count)
    model = clone_model(template_model, device)
    optimizer = make_optimizer(model, optimizer_state, learner["lr"])
    step_count = 0
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    set_seed(seed, device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.device(device).type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace_name = f"profile-minibatch-{batch_size}.json"
    table_name = f"profile-key-averages-{batch_size}.txt"
    synchronize_device(device)
    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False,
                                with_stack=False) as profiler:
        with torch.profiler.record_function("benchmark_ppo/representative_update"):
            optimize(model, optimizer, profiled_rollout, 1, batch_size, learner=learner)
            synchronize_device(device)
    profiler.export_chrome_trace(str(directory / trace_name))
    sort_key = "self_cuda_time_total" if torch.device(device).type == "cuda" else "self_cpu_time_total"
    table = profiler.key_averages().table(sort_by=sort_key, row_limit=30)
    (directory / table_name).write_text(table + "\n", encoding="utf-8")
    expected = math.ceil(sample_count / batch_size)
    if step_count != expected:
        raise RuntimeError("Profiled optimizer step count differs from its representative minibatches")
    return {"trace": trace_name, "key_averages": table_name, "minibatch_size": batch_size,
            "samples": sample_count, "optimizer_steps": step_count,
            "scope": "one representative update over at most 32768 captured rollout samples"}


def summarize_trials(batch_size, samples, trials):
    durations = [trial["seconds"] for trial in trials]
    stats = {name: float(np.mean([trial["statistics"][name] for trial in trials]))
             for name in ("loss", "policy_loss", "value_loss", "entropy")}
    return {"minibatch_size": batch_size, "samples": samples,
            "optimizer_steps_per_trial": optimizer_steps(samples, batch_size),
            "last_minibatch_samples": samples % batch_size or min(samples, batch_size),
            "seconds_mean": float(np.mean(durations)), "seconds_median": float(np.median(durations)),
            "seconds_min": float(min(durations)), "seconds_max": float(max(durations)),
            "setup_seconds_mean": float(np.mean([trial["setup_seconds"] for trial in trials])),
            "losses_mean": stats,
            "memory_peak_pytorch": max((trial["memory_peak"] for trial in trials if trial["memory_peak"]),
                                       key=lambda peak: peak["pytorch_allocated_bytes"], default=None),
            "trials": [{"seed": trial["seed"], "seconds": trial["seconds"],
                        "setup_seconds": trial["setup_seconds"],
                        "optimizer_steps": trial["optimizer_steps"],
                        "losses": trial["statistics"]} for trial in trials]}


def write_summary(directory, summary):
    json_path = directory / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["Standalone PPO minibatch benchmark",
             f"classification: {summary['classification']}",
             f"device: {summary['hardware']['device']}",
             f"captured samples: {summary['capture']['num_samples']}",
             f"rollout SHA256: {summary['capture']['rollout_sha256']}",
             "timings: unprofiled, synchronized optimize() wall seconds"]
    for row in summary["minibatches"]:
        lines.append(f"batch={row['minibatch_size']} steps={row['optimizer_steps_per_trial']} "
                     f"mean={row['seconds_mean']:.6f}s median={row['seconds_median']:.6f}s "
                     f"loss={row['losses_mean']['loss']:.6f}")
    if summary.get("profile"):
        lines.append(f"Chrome trace: {summary['profile']['trace']}")
        lines.append(f"Profile key averages: {summary['profile']['key_averages']}")
    if summary["classification"] != "real_native_rollout":
        lines.append("Synthetic smoke data is not a learning or performance conclusion.")
    (directory / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback is performed")
    torch.set_num_threads(args.threads or 1)
    checkpoint_path = None
    checkpoint = None
    plan = None
    stage_index = 0
    stage = None
    library = None
    native_build = {}
    opponent_steps = []

    if args.smoke:
        worlds = args.envs or 2
        steps = args.rollout_steps or 4
        sample_count = validate_benchmark_sizes(worlds, steps, args.minibatch_sizes, args.max_samples)
        output = validate_output_path(args.output)
        policy = from_legacy(SCHEMA)
        set_seed(args.seed, device)
        model = ActorCritic(policy["observation_dim"], 32,
                            action_sizes=dict(zip(policy["action_names"], policy["action_sizes"]))).to(device)
        learner = dict(LEARNER_DEFAULTS)
        learner["hidden_dim"] = 32
        optimizer = torch.optim.Adam(model.parameters(), lr=learner["lr"], eps=1e-5)
        optimizer_state = optimizer.state_dict()
        set_seed(args.seed, device)
        rollout = synthetic_rollout(model, worlds, steps, policy, device, args.seed)
        capture_seconds = 0.0
        stage_name = "synthetic-smoke"
        controls = {"fixed": {}}
        opponent_catalog = {"synthetic": {"type": "synthetic"}}
        classification = "synthetic_smoke_not_learning_or_performance_evidence"
        checkpoint_provenance = {"source": "synthetic", "frozen_checkpoint": False}
        optimizer_initialization = "fresh Adam state cloned identically for every trial"
        contract_hash = fingerprint(policy)
        scenario_hash = None
        effective_threads = args.threads or 1
        tick_limit = None
    else:
        if not args.checkpoint:
            raise ValueError("A trusted research --checkpoint is required unless --smoke is selected")
        checkpoint_path, checkpoint, plan, stage_index = load_baseline(args.checkpoint)
        research = checkpoint["research"]
        validate_pool_device(research["pool"], device)
        saved_worlds = research.get("world_count")
        if type(saved_worlds) is not int or saved_worlds < 1:
            raise ValueError("Checkpoint world_count is invalid")
        worlds = saved_worlds if args.envs is None else args.envs
        if worlds > saved_worlds:
            raise ValueError("--envs cannot exceed the checkpoint's saved world count")
        steps = research.get("rollout_steps") if args.rollout_steps is None else args.rollout_steps
        if type(steps) is not int:
            raise ValueError("Checkpoint rollout_steps is invalid")
        sample_count = validate_benchmark_sizes(worlds, steps, args.minibatch_sizes, args.max_samples)
        output = validate_output_path(args.output, checkpoint_path)
        if args.max_ticks is not None and not 1 <= args.max_ticks <= 1_000_000:
            raise ValueError("--max-ticks must be from 1 to 1000000")
        stage = plan["stages"][stage_index]
        stage_name = stage["name"]
        learner = dict(plan["learner"])
        hidden_dim = checkpoint["metadata"]["hidden_dim"]
        policy = plan["policy"]
        action_sizes = dict(zip(policy["action_names"], policy["action_sizes"]))
        model = ActorCritic(policy["observation_dim"], hidden_dim, action_sizes=action_sizes).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError("Checkpoint contains non-finite model weights")
        optimizer_state = deepcopy(checkpoint["optimizer"])
        config = checkpoint.get("config", {})
        effective_threads = args.threads or (config.get("threads", 4) if isinstance(config, dict) else 4)
        torch.set_num_threads(effective_threads)
        tick_limit = research.get("max_ticks_override") if args.max_ticks is None else args.max_ticks
        library = args.library
        set_seed(args.seed, device)
        worlds_runner = SampledWorlds(stage["scenario"], policy, worlds, args.device, library, tick_limit,
                                      plan["seed"], resize_sampler_state(research.get("sampler"), worlds))
        try:
            pool = OpponentPool(plan["opponents"], policy, worlds, args.device, plan["seed"], learner=model)
            pool.configure(stage["opponents"], stage["controls"])
            pool.load_state_dict(resize_pool_state(research.get("pool"), worlds))
            worlds_runner.configure(stage)
            observations = worlds_runner.observe()
            pool.reset(np.ones(worlds, dtype=bool))
            fixed = fixed_indices(policy, stage["controls"])
            opponent_steps, original_act = tracked_pool_act(pool)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize_device(device)
            capture_started = time.perf_counter()
            try:
                rollout, _, _, _ = collect(model, worlds_runner, pool, observations, steps, device, fixed,
                                           learner=learner)
            finally:
                pool.act = original_act
            synchronize_device(device)
            capture_seconds = time.perf_counter() - capture_started
            native_build = safe_build_info(worlds_runner.groups[0][1].build_info)
            opponent_catalog = controller_catalog(pool, plan["opponents"])
            scenario_hash = hashlib.sha256(json.dumps(stage["scenario"], sort_keys=True,
                                                       separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            contract_hash = fingerprint(policy)
        finally:
            worlds_runner.close()
        controls = safe_metadata(stage["controls"])
        classification = "real_native_rollout"
        checkpoint_provenance = {"sha256": sha256_file(checkpoint_path),
                                 "update": checkpoint["update"], "samples": checkpoint["samples"]}
        optimizer_initialization = "checkpoint Adam state cloned identically for every trial"

    rollout_hash = rollout_sha256(rollout)
    terminal_count = int(rollout["terminated"].sum().item())
    truncated_count = int(rollout["truncated"].sum().item())
    base_optimizer = make_optimizer(model, optimizer_state, learner["lr"])
    initial_optimizer_state = deepcopy(base_optimizer.state_dict())
    del base_optimizer
    device_info = hardware_metadata(device)

    minibatch_trials = {size: [] for size in args.minibatch_sizes}
    order_rng = random.Random(args.seed ^ 0x50504F)
    warmup_seed = args.seed + 1_000_000
    for size in args.minibatch_sizes:
        for index in range(args.warmup):
            run_trial(model, initial_optimizer_state, rollout, size, learner, device,
                      warmup_seed + index)
    repeat_orders = []
    for repeat in range(args.repeats):
        order = list(args.minibatch_sizes)
        if args.randomize_order:
            order_rng.shuffle(order)
        repeat_orders.append(order)
        for size in order:
            trial = run_trial(model, initial_optimizer_state, rollout, size, learner, device,
                              args.seed + repeat)
            trial["seed"] = args.seed + repeat
            minibatch_trials[size].append(trial)

    output.mkdir(parents=True, exist_ok=False)
    profiles = None
    if args.profile_trace:
        profile_size = min(args.minibatch_sizes)
        profiles = profile_trace(model, initial_optimizer_state, rollout, profile_size, learner,
                                 device, args.seed + 2_000_000, output, args.profile_minibatches)

    summary = {"format": "minecraft_sim_ppo_benchmark", "version": 1,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "classification": classification, "checkpoint": checkpoint_provenance,
               "plan": None if plan is None else {"name": plan.get("name"), "stage": stage_name,
                                                    "stage_index": stage_index},
               "stage": {"name": stage_name, "controls": controls, "scenario_sha256": scenario_hash,
                         "max_ticks": tick_limit},
               "policy_contract_sha256": contract_hash,
               "hardware": device_info,
               "native_build": native_build,
               "capture": {"source": "synthetic_smoke" if args.smoke else "native_collection",
                           "worlds": worlds, "rollout_steps": steps, "num_samples": sample_count,
                           "seconds_synchronized": capture_seconds, "rollout_sha256": rollout_hash,
                           "terminated_count": terminal_count, "truncated_count": truncated_count,
                           "bootstrap_semantics": "collect next_values on final observations; terminated masks delta bootstraps; terminated or truncated masks GAE recursion",
                           "opponent_catalog": opponent_catalog, "opponent_step_counts": opponent_steps},
               "config": {"seed": args.seed, "threads": effective_threads,
                          "benchmark_epochs": BENCHMARK_EPOCHS, "warmup_updates_per_size": args.warmup,
                          "repeats": args.repeats, "randomized_order": args.randomize_order,
                          "repeat_orders": repeat_orders, "requested_minibatch_sizes": args.minibatch_sizes,
                          "samples_limit": args.max_samples, "learner_config": safe_metadata(learner),
                          "optimizer": optimizer_initialization,
                          "rollout_policy": "one captured rollout, tensor-cloned unchanged for each trial"},
               "minibatches": [summarize_trials(size, sample_count, minibatch_trials[size])
                               for size in args.minibatch_sizes],
               "profile": profiles,
               "memory": {"pytorch_cuda_peak_scope": "CUDA allocator only; native/C++ allocations excluded",
                          "process_peak_rss_scope": "OS host-process high-water mark, includes host-native allocations and all phases; excludes device-native memory",
                          "process_peak_rss_bytes": process_peak_rss_bytes()}}
    write_summary(output, safe_metadata(summary))
    print((output / "summary.txt").read_text(encoding="utf-8"), end="")
    print(f"Output directory: {output}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Trusted frozen version-1 research PPO checkpoint")
    parser.add_argument("--smoke", action="store_true",
                        help="Use labeled synthetic data only; not a learning or performance conclusion")
    parser.add_argument("--output", required=True, help="New, exclusive benchmark output directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--library", help="Explicit native training library for real rollout capture")
    parser.add_argument("--envs", type=int, help=f"Capture worlds (maximum {MAX_WORLDS})")
    parser.add_argument("--rollout-steps", type=int, help=f"Capture steps (maximum {MAX_ROLLOUT_STEPS})")
    parser.add_argument("--max-ticks", type=int, help="Optional episode tick-limit override")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLE_COUNT,
                        help=f"Hard rollout cap, at most {MAX_SAMPLE_COUNT} samples")
    parser.add_argument("--minibatch-sizes", type=parse_minibatch_sizes, default=[1024, 4096, 8192, 16384])
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--threads", type=int, help="CPU threads; checkpoint setting or 4 by default")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--randomize-order", action="store_true")
    parser.add_argument("--profile-trace", action="store_true",
                        help="Export a separate short CPU/CUDA Chrome trace")
    parser.add_argument("--profile-minibatches", type=int, default=2)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed < 0 or args.seed >= 2**63:
        parser.error("seed must be from 0 to 2**63-1")
    if args.repeats < 1 or args.repeats > 20 or args.warmup < 0 or args.warmup > 2:
        parser.error("repeats must be 1..20 and warmup must be 0..2")
    if not 1 <= args.profile_minibatches <= MAX_PROFILE_MINIBATCHES:
        parser.error(f"profile-minibatches must be from 1 to {MAX_PROFILE_MINIBATCHES}")
    if not 1 <= args.max_samples <= MAX_SAMPLE_COUNT:
        parser.error(f"max-samples must be from 1 to {MAX_SAMPLE_COUNT}")
    if args.envs is not None and not 1 <= args.envs <= MAX_WORLDS:
        parser.error(f"envs must be from 1 to {MAX_WORLDS}")
    if args.rollout_steps is not None and not 1 <= args.rollout_steps <= MAX_ROLLOUT_STEPS:
        parser.error(f"rollout-steps must be from 1 to {MAX_ROLLOUT_STEPS}")
    if args.threads is not None and not 1 <= args.threads <= 256:
        parser.error("threads must be from 1 to 256")
    if args.max_ticks is not None and not 1 <= args.max_ticks <= 1_000_000:
        parser.error("max-ticks must be from 1 to 1000000")
    if args.smoke and args.checkpoint:
        parser.error("--smoke and --checkpoint are mutually exclusive")
    if not args.smoke and not args.checkpoint:
        parser.error("--checkpoint is required unless --smoke is selected")
    if args.smoke and args.library:
        parser.error("--library applies only to real native rollout capture")
    try:
        return benchmark(args)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

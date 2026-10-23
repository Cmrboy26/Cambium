"""Prepare and run bounded fixed-checkpoint baseline evaluations."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import time
from uuid import uuid4

import numpy as np
import torch

from backend.fast_rl_model import ActorCritic
from simulator.training.contract import fingerprint
from simulator.training.controllers import create_controller
from simulator.training.environment import ROOT
from simulator.training.policies import fixed_indices, policy_actions
from simulator.training.research import SampledWorlds


RUNS_ROOT = ROOT / "simulator" / "training" / "runs"
DEFAULT_RUNS = ("am-v2-s20", "am-v2-s40", "am-v2-s60")
MAX_EPISODES = 256
MAX_WORLDS = 256
MAX_TICKS = 1_000_000
MANIFEST_FORMAT = "minecraft_sim_checkpoint_evaluation"
MANIFEST_VERSION = 1


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_value(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _hash_file(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() if hasattr(hashlib, "file_digest") \
            else hashlib.sha256(source.read()).hexdigest()


def _load_checkpoint(path):
    path = Path(path)
    try:
        payload = path.read_bytes()
        saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"Could not load trusted checkpoint {path}: {error}") from error
    if (not isinstance(saved, dict) or saved.get("format") != "minecraft_sim_ppo"
            or type(saved.get("version")) is not int or saved["version"] != 1):
        raise ValueError(f"Unsupported PPO checkpoint envelope: {path}")
    return saved, _hash_bytes(payload)


def _checkpoint_record(path):
    saved, digest = _load_checkpoint(path)
    metadata = saved.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint metadata is missing: {path}")
    experiment = metadata.get("experiment")
    snapshot = metadata.get("snapshot")
    if not isinstance(experiment, dict) or not isinstance(snapshot, dict):
        raise ValueError(f"Checkpoint is not a retained research snapshot: {path}")
    update = saved.get("update")
    samples = saved.get("samples")
    stage_index = experiment.get("stage_index")
    stage_name = experiment.get("stage")
    stage_updates = snapshot.get("stage_updates")
    hidden_dim = metadata.get("hidden_dim")
    max_ticks = metadata.get("max_ticks")
    contract = metadata.get("contract")
    controls = metadata.get("controls", {"fixed": {}})
    if (type(update) is not int or update < 0 or type(samples) is not int or samples < 0
            or type(stage_index) is not int or stage_index < 0
            or not isinstance(stage_name, str) or not stage_name
            or type(stage_updates) is not int or stage_updates < 0
            or type(hidden_dim) is not int or hidden_dim < 1
            or type(max_ticks) is not int or max_ticks < 1
            or not isinstance(contract, dict) or not isinstance(controls, dict)):
        raise ValueError(f"Checkpoint has incomplete update, stage, or policy metadata: {path}")
    if snapshot.get("update") != update:
        raise ValueError(f"Checkpoint update does not match its retained snapshot metadata: {path}")
    return {
        "key": (update, stage_name, stage_index, stage_updates),
        "snapshot_name": snapshot.get("name", Path(path).stem),
        "checkpoint_path": Path(path).resolve(),
        "checkpoint_sha256": digest,
        "update": update,
        "samples": samples,
        "stage_name": stage_name,
        "stage_index": stage_index,
        "stage_updates": stage_updates,
        "hidden_dim": hidden_dim,
        "max_ticks": max_ticks,
        "policy_sha256": fingerprint(contract),
        "controls": deepcopy(controls),
    }


def _load_run(run_root, run_name):
    run_path = (Path(run_root) / run_name).resolve()
    if not run_path.is_dir():
        raise ValueError(f"Run directory does not exist: {run_path}")
    config_path = run_path / "config.json"
    try:
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read resolved run configuration {config_path}: {error}") from error
    plan = config.get("experiment") if isinstance(config, dict) else None
    if not isinstance(plan, dict) or not isinstance(plan.get("stages"), list):
        raise ValueError(f"Run config does not contain its resolved experiment: {config_path}")
    snapshots = run_path / "snapshots"
    paths = sorted(snapshots.glob("*.pt")) if snapshots.is_dir() else []
    if not paths:
        raise ValueError(f"No retained snapshot checkpoints found in {snapshots}")
    inventory = {}
    for path in paths:
        record = _checkpoint_record(path)
        inventory.setdefault(record["key"], record)
    return {
        "id": str(run_name),
        "path": run_path,
        "config_path": config_path.resolve(),
        "config_sha256": _hash_bytes(config_bytes),
        "plan": plan,
        "plan_sha256": _hash_value(plan),
        "inventory": inventory,
    }


def _format_keys(keys):
    if not keys:
        return "none"
    return ", ".join(f"{update}:{stage}[{index}]+{local}"
                     for update, stage, index, local in sorted(keys))


def _select_common_key(inventories, requested_update=None):
    if len(inventories) < 2:
        raise ValueError("At least two candidate runs are required")
    common = set.intersection(*(set(records) for records in inventories.values()))
    if requested_update is not None:
        common = {key for key in common if key[0] == requested_update}
    if not common:
        global_updates = set.intersection(*(
            {key[0] for key in records} for records in inventories.values()))
        requested = (f" Requested update {requested_update} was not retained by every run."
                     if requested_update is not None else "")
        details = "; ".join(f"{run}=[{_format_keys(records)}]"
                             for run, records in inventories.items())
        raise ValueError("No exact common retained update and stage identity." + requested
                         + f" Common global update numbers without stage matching: {sorted(global_updates)}."
                         + f" Available snapshots: {details}")
    latest_update = max(key[0] for key in common)
    latest = [key for key in common if key[0] == latest_update]
    if len(latest) != 1:
        raise ValueError(f"Update {latest_update} has multiple common stage identities: {_format_keys(latest)}")
    return latest[0]


def _require_matching_samples(records):
    samples = [record.get("samples") for record in records]
    if any(type(value) is not int or value < 0 for value in samples):
        raise ValueError("Candidate checkpoints must have valid training sample counts")
    if len(set(samples)) != 1:
        raise ValueError("Candidate checkpoints have incompatible training sample counts")


def _resolved_stage(plan, record):
    try:
        policy = plan["policy"]
        stages = plan["stages"]
        stage = stages[record["stage_index"]]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Resolved experiment does not contain the checkpoint stage") from error
    if stage.get("name") != record["stage_name"]:
        raise ValueError("Checkpoint stage identity does not match the resolved experiment")
    policy_sha256 = fingerprint(policy)
    if policy_sha256 != record["policy_sha256"]:
        raise ValueError("Checkpoint policy layout differs from the resolved experiment")
    controls = stage.get("controls", {"fixed": {}})
    if _canonical(controls) != _canonical(record["controls"]):
        raise ValueError("Checkpoint fixed controls differ from the resolved stage")
    if "scenario" not in stage or not isinstance(stage.get("rewards"), dict):
        raise ValueError("Resolved checkpoint stage is missing scenario or reward conditions")
    condition = {
        "policy_sha256": policy_sha256,
        "stage": record["stage_name"],
        "stage_index": record["stage_index"],
        "stage_updates": record["stage_updates"],
        "scenario": stage["scenario"],
        "controls": controls,
        "rewards": stage["rewards"],
        "max_ticks": record["max_ticks"],
    }
    return stage, _hash_value(condition)


def _scripted_opponents(plan):
    opponents = plan.get("opponents")
    if not isinstance(opponents, dict):
        raise ValueError("Resolved experiment has no opponent definitions")
    return {name: deepcopy(definition) for name, definition in sorted(opponents.items())
            if isinstance(definition, dict) and definition.get("type") == "scripted"}


def _freeze_file(source, output_root, folder, suffix, expected_sha256):
    source = Path(source)
    target = Path(output_root) / folder / "sha256" / f"{expected_sha256}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _hash_file(target) != expected_sha256:
            raise ValueError(f"Content-addressed artifact has unexpected contents: {target}")
    else:
        shutil.copyfile(source, target)
        if _hash_file(target) != expected_sha256:
            target.unlink()
            raise ValueError(f"Artifact changed while it was being frozen: {source}")
    return target.relative_to(output_root).as_posix()


def _inside(path, directory):
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
        return True
    except ValueError:
        return False


def _new_output_directory(output):
    if output is None:
        parent = RUNS_ROOT / "checkpoint-evaluations"
        parent.mkdir(parents=True, exist_ok=True)
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        destination = parent / name
    else:
        destination = Path(output)
        if not destination.is_absolute():
            destination = ROOT / destination
        destination = destination.resolve()
    if destination.exists():
        raise ValueError(f"Evaluation output must be a new directory: {destination}")
    return destination


def prepare(run_names=DEFAULT_RUNS, run_root=RUNS_ROOT, requested_update=None, output=None):
    names = list(run_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Provide at least two distinct candidate run names")
    root = Path(run_root).resolve()
    runs = [_load_run(root, name) for name in names]
    key = _select_common_key({run["id"]: run["inventory"] for run in runs}, requested_update)
    selected = [run["inventory"][key] for run in runs]
    _require_matching_samples(selected)
    resolved = []
    for run, record in zip(runs, selected, strict=True):
        stage, condition_sha256 = _resolved_stage(run["plan"], record)
        scripts = _scripted_opponents(run["plan"])
        resolved.append({"run": run, "record": record, "stage": stage,
                         "condition_sha256": condition_sha256, "scripts": scripts})
    reference = resolved[0]
    for item in resolved[1:]:
        if item["condition_sha256"] != reference["condition_sha256"]:
            raise ValueError("Candidate runs have incompatible policy, stage, scenario, controls, rewards, "
                             "or episode limits")
        if _canonical(item["scripts"]) != _canonical(reference["scripts"]):
            raise ValueError("Candidate runs do not share the same scripted baseline definitions")
    if not reference["scripts"]:
        raise ValueError("Resolved experiment has no common scripted baseline opponents")

    destination = _new_output_directory(output)
    for run in runs:
        if _inside(destination, run["path"]) or _inside(run["path"], destination):
            raise ValueError("Evaluation output must be outside all source run directories")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()

    candidates = []
    for item in resolved:
        run, record = item["run"], item["record"]
        artifact_path = _freeze_file(record["checkpoint_path"], destination, "artifacts",
                                     ".pt", record["checkpoint_sha256"])
        config_path = _freeze_file(run["config_path"], destination, "provenance",
                                   ".json", run["config_sha256"])
        candidates.append({
            "id": run["id"],
            "source_run": str(run["path"]),
            "snapshot_name": record["snapshot_name"],
            "update": record["update"],
            "samples": record["samples"],
            "stage": {"name": record["stage_name"], "index": record["stage_index"],
                      "stage_updates": record["stage_updates"]},
            "hidden_dim": record["hidden_dim"],
            "controls": deepcopy(record["controls"]),
            "max_ticks": record["max_ticks"],
            "policy_sha256": record["policy_sha256"],
            "condition_sha256": item["condition_sha256"],
            "checkpoint": {"path": artifact_path, "sha256": record["checkpoint_sha256"]},
            "config": {"path": config_path, "sha256": run["config_sha256"]},
            "experiment_sha256": run["plan_sha256"],
        })
    opponents = [{"id": f"scripted/{name}", "type": "scripted", "definition": definition}
                 for name, definition in reference["scripts"].items()]
    opponents.extend({
        "id": f"checkpoint/{candidate['id']}",
        "type": "checkpoint",
        "candidate_id": candidate["id"],
        "update": candidate["update"],
        "samples": candidate["samples"],
        "stage": deepcopy(candidate["stage"]),
        "checkpoint": deepcopy(candidate["checkpoint"]),
    } for candidate in candidates)
    manifest = {
        "format": MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "common_update": key[0],
        "stage": {"name": key[1], "index": key[2], "stage_updates": key[3]},
        "condition_sha256": reference["condition_sha256"],
        "experiment": deepcopy(reference["run"]["plan"]),
        "policy_sha256": candidates[0]["policy_sha256"],
        "scenario_sha256": _hash_value(reference["stage"]["scenario"]),
        "candidates": candidates,
        "opponents": opponents,
        "episodes_per_pairing_default": MAX_EPISODES,
        "episodes_per_pairing_max": MAX_EPISODES,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return destination


def _bundle_file(root, artifact):
    if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
        raise ValueError("Manifest artifact reference is invalid")
    path = (Path(root) / artifact["path"]).resolve()
    if not _inside(path, root) or not path.is_file():
        raise ValueError(f"Frozen artifact is missing or outside the evaluation directory: {path}")
    expected = artifact.get("sha256")
    if not isinstance(expected, str) or _hash_file(path) != expected:
        raise ValueError(f"Frozen artifact SHA256 mismatch: {path}")
    return path


def _load_manifest(bundle):
    root = Path(bundle).resolve()
    manifest_path = root / "manifest.json"
    try:
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read evaluation manifest {manifest_path}: {error}") from error
    if (not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT
            or manifest.get("version") != MANIFEST_VERSION):
        raise ValueError(f"Unsupported evaluation manifest: {manifest_path}")
    candidates = manifest.get("candidates")
    opponents = manifest.get("opponents")
    if not isinstance(candidates, list) or len(candidates) < 2 or not isinstance(opponents, list) or not opponents:
        raise ValueError("Manifest must list at least two candidates and one opponent")
    if not all(isinstance(item, dict) for item in candidates):
        raise ValueError("Manifest candidates must be objects")
    _require_matching_samples(candidates)
    if len({item.get("id") for item in candidates if isinstance(item, dict)}) != len(candidates):
        raise ValueError("Manifest candidate IDs must be unique")
    if len({item.get("id") for item in opponents if isinstance(item, dict)}) != len(opponents):
        raise ValueError("Manifest opponent IDs must be unique")
    experiment = manifest.get("experiment")
    stage_info = manifest.get("stage")
    if not isinstance(experiment, dict) or not isinstance(stage_info, dict):
        raise ValueError("Manifest is missing its resolved experiment or stage identity")
    reference = candidates[0]
    reference_stage = reference.get("stage", {})
    reference_record = {
        "stage_name": reference_stage.get("name"),
        "stage_index": reference_stage.get("index"),
        "stage_updates": reference_stage.get("stage_updates"),
        "policy_sha256": reference.get("policy_sha256"),
        "controls": reference.get("controls"),
        "max_ticks": reference.get("max_ticks"),
    }
    reference_plan_stage, manifest_condition = _resolved_stage(experiment, reference_record)
    if (manifest_condition != manifest.get("condition_sha256")
            or stage_info != {"name": reference_stage.get("name"),
                              "index": reference_stage.get("index"),
                              "stage_updates": reference_stage.get("stage_updates")}
            or _hash_value(reference_plan_stage["scenario"]) != manifest.get("scenario_sha256")
            or fingerprint(experiment.get("policy")) != manifest.get("policy_sha256")):
        raise ValueError("Manifest stage, scenario, or policy hash does not match its resolved experiment")
    expected_scripts = _scripted_opponents(experiment)
    listed_scripts = {}
    listed_checkpoints = {}
    for opponent in opponents:
        if opponent.get("type") == "scripted":
            identifier = opponent.get("id", "")
            if not identifier.startswith("scripted/"):
                raise ValueError("Scripted opponent ID is invalid")
            listed_scripts[identifier.removeprefix("scripted/")] = opponent.get("definition")
        elif opponent.get("type") == "checkpoint":
            listed_checkpoints[opponent.get("id")] = opponent.get("checkpoint")
    expected_checkpoints = {f"checkpoint/{item.get('id')}": item.get("checkpoint") for item in candidates}
    if (_canonical(listed_scripts) != _canonical(expected_scripts)
            or _canonical(listed_checkpoints) != _canonical(expected_checkpoints)):
        raise ValueError("Manifest opponent roster does not match its frozen candidates and experiment")
    for candidate in candidates:
        _bundle_file(root, candidate.get("checkpoint"))
        config_path = _bundle_file(root, candidate.get("config"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        plan = config.get("experiment") if isinstance(config, dict) else None
        if not isinstance(plan, dict) or _hash_value(plan) != candidate.get("experiment_sha256"):
            raise ValueError(f"Frozen resolved experiment hash mismatch for {candidate.get('id')}")
        stage_info = candidate.get("stage", {})
        record = {
            "stage_name": stage_info.get("name"),
            "stage_index": stage_info.get("index"),
            "stage_updates": stage_info.get("stage_updates"),
            "policy_sha256": candidate.get("policy_sha256"),
            "controls": candidate.get("controls"),
            "max_ticks": candidate.get("max_ticks"),
        }
        _, condition_sha256 = _resolved_stage(plan, record)
        if condition_sha256 != candidate.get("condition_sha256"):
            raise ValueError(f"Frozen candidate conditions differ for {candidate.get('id')}")
        if _canonical(_scripted_opponents(plan)) != _canonical(expected_scripts):
            raise ValueError("Frozen candidate configurations do not share scripted baseline definitions")
        if candidate.get("condition_sha256") != manifest.get("condition_sha256"):
            raise ValueError("Candidate evaluation conditions differ from the manifest")
    for opponent in opponents:
        if opponent.get("type") == "checkpoint":
            _bundle_file(root, opponent.get("checkpoint"))
        elif opponent.get("type") != "scripted" or not isinstance(opponent.get("definition"), dict):
            raise ValueError("Manifest contains an unsupported opponent type")
    return root, manifest, _hash_bytes(payload)


def _load_candidate_model(root, candidate, contract, device):
    path = _bundle_file(root, candidate["checkpoint"])
    saved, digest = _load_checkpoint(path)
    metadata = saved.get("metadata", {})
    snapshot = metadata.get("snapshot", {})
    summary = metadata.get("experiment", {})
    if (digest != candidate["checkpoint"]["sha256"] or saved.get("update") != candidate.get("update")
            or saved.get("samples") != candidate.get("samples")
            or snapshot.get("name") != candidate.get("snapshot_name")
            or summary.get("stage") != candidate.get("stage", {}).get("name")
            or summary.get("stage_index") != candidate.get("stage", {}).get("index")
            or snapshot.get("stage_updates") != candidate.get("stage", {}).get("stage_updates")
            or _canonical(metadata.get("controls", {"fixed": {}})) != _canonical(candidate.get("controls"))
            or metadata.get("max_ticks") != candidate.get("max_ticks")
            or fingerprint(metadata.get("contract")) != candidate.get("policy_sha256")):
        raise ValueError(f"Frozen candidate metadata differs from manifest: {candidate.get('id')}")
    hidden_dim = candidate.get("hidden_dim")
    model = ActorCritic(contract["observation_dim"], hidden_dim, movement_bins=9,
                        action_sizes=dict(zip(contract["action_names"], contract["action_sizes"])))
    try:
        model.load_state_dict(saved["model"], strict=True)
    except (KeyError, RuntimeError, TypeError) as error:
        raise ValueError(f"Candidate model does not match its policy contract: {candidate.get('id')}") from error
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError(f"Candidate contains non-finite model parameters: {candidate.get('id')}")
    return model.to(device).eval().requires_grad_(False), metadata


def _outcome(terminated, truncated, losses, learner_slot):
    if terminated:
        if bool(losses[0]) and bool(losses[1]):
            return "draw"
        return "loss" if bool(losses[learner_slot]) else "win"
    if truncated:
        return "timeout"
    raise ValueError("An episode outcome requires a terminal or timeout flag")


def _slot_for_episode(ordinal):
    return ordinal % 2


def _wilson_interval(successes, trials):
    if trials < 1:
        return None
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _summarize(episodes):
    counts = {name: sum(episode["outcome"] == name for episode in episodes)
              for name in ("win", "loss", "draw", "timeout")}
    by_slot = {}
    for slot in (0, 1):
        selected = [episode for episode in episodes if episode["learner_slot"] == slot]
        by_slot[str(slot)] = {
            "episodes": len(selected),
            **{name: sum(episode["outcome"] == name for episode in selected)
               for name in ("win", "loss", "draw", "timeout")},
        }
    lengths = [episode["length"] for episode in episodes]
    result = {
        "episodes": len(episodes),
        **counts,
        "learner_slot_counts": by_slot,
        "win_rate": counts["win"] / len(episodes) if episodes else None,
        "win_rate_wilson_95": _wilson_interval(counts["win"], len(episodes)),
        "mean_episode_length": sum(lengths) / len(lengths) if lengths else None,
        "episode_length_min": min(lengths) if lengths else None,
        "episode_length_max": max(lengths) if lengths else None,
        "mean_return": sum(episode["return"] for episode in episodes) / len(episodes) if episodes else None,
        "return_episode_count": len(episodes),
        "uncertainty_assumptions": (
            "95% Wilson score interval for wins/episodes, treating episodes as independent Bernoulli trials; "
            "does not adjust for shared seeds, worlds, or paired scenario correlation."
        ),
    }
    return result


def _evaluate_pairing(model, contract, stage, opponent, options, episodes, seed):
    device = torch.device(options.device)
    cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    world_count = min(options.worlds, episodes)
    completed = []
    if opponent["type"] == "checkpoint":
        opponent_path = _bundle_file(options.bundle_root, opponent["checkpoint"])
        definition = {"type": "checkpoint", "path": str(opponent_path),
                      "sha256": opponent["checkpoint"]["sha256"], "deterministic": False}
    else:
        definition = opponent["definition"]

    for batch_start in range(0, episodes, world_count):
        batch_index = batch_start // world_count
        batch_count = min(world_count, episodes - batch_start)
        batch_seed = (seed + batch_index) % (2**63)
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(batch_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(batch_seed)
            worlds = SampledWorlds(stage["scenario"], contract, batch_count, options.device,
                                   options.library, options.max_ticks, batch_seed)
            try:
                worlds.configure(stage)
                controller = create_controller(definition, contract, batch_count, options.device,
                                               seed=batch_seed)
                active = np.ones(batch_count, dtype=bool)
                controller.reset(active)
                episode_ids = np.arange(batch_start, batch_start + batch_count, dtype=np.int64)
                learner_slots = np.array([_slot_for_episode(index) for index in episode_ids], dtype=np.int64)
                lengths = np.zeros(batch_count, dtype=np.int64)
                returns = np.zeros(batch_count, dtype=np.float64)
                fixed = fixed_indices(contract, stage.get("controls", {"fixed": {}}))
                observations = worlds.observe()
                ticks = 0
                with torch.no_grad():
                    while active.any():
                        if ticks >= worlds.max_ticks:
                            raise RuntimeError("Evaluation world exceeded its configured max_ticks without finishing")
                        active_worlds = np.flatnonzero(active)
                        active_slots = learner_slots[active_worlds]
                        learner_observations = observations[active_worlds, active_slots]
                        inputs = torch.as_tensor(learner_observations, dtype=torch.float32, device=device)
                        learner_actions, _, _, _ = policy_actions(model, inputs, worlds.action_names, fixed)
                        learner_actions = learner_actions.cpu().numpy()
                        opponent_slots = 1 - active_slots
                        opponent_observations = observations[active_worlds, opponent_slots]
                        opponent_actions = controller.act(opponent_observations, active_worlds)
                        actions = np.zeros((batch_count, 2, worlds.action_count), dtype=np.int64)
                        actions[active_worlds, active_slots] = learner_actions
                        actions[active_worlds, opponent_slots] = opponent_actions
                        final, rewards, terminated, truncated = worlds.step(actions)
                        ticks += 1
                        lengths[active_worlds] += 1
                        returns[active_worlds] += rewards[active_worlds, active_slots]
                        finished = terminated | truncated
                        done = finished & active
                        if done.any():
                            losses = worlds.losses(done)
                            for world in np.flatnonzero(done):
                                slot = int(learner_slots[world])
                                completed.append({
                                    "episode_id": int(episode_ids[world]),
                                    "seed": batch_seed,
                                    "batch": batch_index,
                                    "outcome": _outcome(bool(terminated[world]), bool(truncated[world]),
                                                         losses[world], slot),
                                    "learner_slot": slot,
                                    "length": int(lengths[world]),
                                    "return": float(returns[world]),
                                })
                            active[done] = False
                        if active.any() and finished.any():
                            observations = worlds.reset(finished)
                            controller.reset(finished)
                        else:
                            observations = final
            finally:
                worlds.close()

    summary = _summarize(completed)
    summary["episode_samples"] = [
        {key: episode[key] for key in ("episode_id", "seed", "batch", "learner_slot", "outcome", "length")}
        for episode in sorted(completed, key=lambda item: item["episode_id"])
    ]
    return summary


def _report(manifest, results):
    lines = [
        "# Fixed-checkpoint evaluation",
        "",
        f"Common retained update: {manifest['common_update']}",
        f"Stage: {manifest['stage']['name']} (index {manifest['stage']['index']}, "
        f"stage update {manifest['stage']['stage_updates']})",
        f"Episodes per pairing: {results['episodes_per_pairing']}; worlds: {results['worlds']}; "
        f"device: {results['device']}; seed: {results['seed']}",
        f"Scenario SHA256: `{manifest['scenario_sha256']}`",
        f"Condition SHA256: `{manifest['condition_sha256']}`",
        "",
        "| Candidate | Opponent | Episodes | Wins | Losses | Draws | Timeouts | Win rate (95% Wilson) | Mean ticks | Mean return | Learner slots 0/1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for candidate_id, matchups in results["matrix"].items():
        for opponent_id, summary in matchups.items():
            interval = summary["win_rate_wilson_95"]
            ci = "n/a" if interval is None else f"{interval[0]:.3f}-{interval[1]:.3f}"
            slots = summary["learner_slot_counts"]
            slot_text = f"{slots['0']['episodes']}/{slots['1']['episodes']}"
            mean_length = summary["mean_episode_length"]
            mean_return = summary["mean_return"] if results["returns_comparable"] else None
            candidate_label = candidate_id.replace("|", "\\|")
            opponent_label = opponent_id.replace("|", "\\|")
            return_text = "n/a" if mean_return is None else f"{mean_return:.3f}"
            lines.append(f"| {candidate_label} | {opponent_label} | {summary['episodes']} | {summary['win']} | "
                         f"{summary['loss']} | {summary['draw']} | {summary['timeout']} | "
                         f"{summary['win_rate']:.3f} ({ci}) | {mean_length:.2f} | {return_text} | {slot_text} |")
    lines.extend([
        "",
        "Timeouts are reported separately and are not counted as losses. Win-rate intervals use the "
        "independent-episode Bernoulli assumption stated in `results.json`; paired-seed and world clustering "
        "are not adjusted. Returns are shown only when all candidates share the same resolved policy, "
        "scenario, controls, rewards, and episode limit.",
        "",
        "Candidate checkpoint SHA256 values:",
    ])
    lines.extend(f"- `{item['id']}` update {item['update']}, samples {item['samples']}: "
                 f"`{item['checkpoint']['sha256']}`; config `{item['config']['sha256']}`"
                 for item in manifest["candidates"])
    return "\n".join(lines) + "\n"


def run(bundle, episodes=MAX_EPISODES, worlds=16, device="cpu", library=None, seed=1729, max_ticks=None):
    if type(episodes) is not int or episodes < 2 or episodes > MAX_EPISODES or episodes % 2:
        raise ValueError(f"episodes must be an even integer from 2 to {MAX_EPISODES}")
    if type(worlds) is not int or not 1 <= worlds <= MAX_WORLDS:
        raise ValueError(f"worlds must be an integer from 1 to {MAX_WORLDS}")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if max_ticks is not None and (type(max_ticks) is not int or not 1 <= max_ticks <= MAX_TICKS):
        raise ValueError(f"max-ticks must be an integer from 1 to {MAX_TICKS}")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer from 0 through 2**63-1")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; no CPU fallback is performed")
    root, manifest, manifest_sha256 = _load_manifest(bundle)
    results_path, report_path = root / "results.json", root / "report.md"
    if results_path.exists() or report_path.exists():
        raise ValueError("This evaluation bundle already has results; prepare a new bundle to rerun")
    plan = manifest["experiment"]
    stage_info = manifest["stage"]
    stage = plan["stages"][stage_info["index"]]
    if stage.get("name") != stage_info["name"]:
        raise ValueError("Manifest stage identity does not match its resolved experiment")
    scenario_limits = [arena["environment"].get("max_ticks", 300)
                       for arena in stage["scenario"]["arenas"]]
    if max_ticks is not None and max_ticks > min(scenario_limits):
        raise ValueError("max-ticks may shorten but cannot extend the resolved episode limit")
    if device == "cuda":
        torch.cuda.set_device(torch.cuda.current_device())
    contract = plan["policy"]
    options = argparse.Namespace(device=device, library=library, max_ticks=max_ticks,
                                 worlds=worlds, bundle_root=root)
    matrix = {}
    started = time.perf_counter()
    for candidate in manifest["candidates"]:
        model, _ = _load_candidate_model(root, candidate, contract, torch.device(device))
        matchups = {}
        for opponent in manifest["opponents"]:
            matchups[opponent["id"]] = _evaluate_pairing(
                model, contract, stage, opponent, options, episodes, seed)
        matrix[candidate["id"]] = matchups
    results = {
        "format": "minecraft_sim_checkpoint_evaluation_results",
        "version": 1,
        "manifest_sha256": manifest_sha256,
        "common_update": manifest["common_update"],
        "stage": deepcopy(manifest["stage"]),
        "condition_sha256": manifest["condition_sha256"],
        "scenario_sha256": manifest["scenario_sha256"],
        "policy_sha256": manifest["policy_sha256"],
        "episodes_per_pairing": episodes,
        "worlds": min(worlds, episodes),
        "device": device,
        "library": str(Path(library).resolve()) if library else None,
        "seed": seed,
        "max_ticks_override": max_ticks,
        "returns_comparable": True,
        "candidate_provenance": [
            {"id": item["id"], "update": item["update"], "samples": item["samples"],
             "stage": item["stage"], "checkpoint_sha256": item["checkpoint"]["sha256"],
             "config_sha256": item["config"]["sha256"], "experiment_sha256": item["experiment_sha256"]}
            for item in manifest["candidates"]],
        "opponent_ids": [item["id"] for item in manifest["opponents"]],
        "scenario_seed_policy": "same deterministic batch seeds and episode ordinals for every candidate-opponent pairing",
        "elapsed_seconds": time.perf_counter() - started,
        "matrix": matrix,
    }
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
                            encoding="utf-8")
    report_path.write_text(_report(manifest, results), encoding="utf-8")
    return root


def _positive_even(value):
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected an even positive integer") from error
    if result < 2 or result > MAX_EPISODES or result % 2:
        raise argparse.ArgumentTypeError(f"Expected an even integer from 2 to {MAX_EPISODES}")
    return result


def _positive_worlds(value):
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a positive integer") from error
    if not 1 <= result <= MAX_WORLDS:
        raise argparse.ArgumentTypeError(f"Expected an integer from 1 to {MAX_WORLDS}")
    return result


def _positive_ticks(value):
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a positive integer") from error
    if not 1 <= result <= MAX_TICKS:
        raise argparse.ArgumentTypeError(f"Expected an integer from 1 to {MAX_TICKS}")
    return result


def _nonnegative_update(value):
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer") from error
    if result < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Freeze a common retained checkpoint update")
    prepare_parser.add_argument("--run-root", default=str(RUNS_ROOT))
    prepare_parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    prepare_parser.add_argument("--update", type=_nonnegative_update)
    prepare_parser.add_argument("--output", help="New evaluation bundle directory")
    run_parser = commands.add_parser("run", help="Evaluate every frozen candidate/opponent pairing")
    run_parser.add_argument("bundle")
    run_parser.add_argument("--episodes", type=_positive_even, default=MAX_EPISODES)
    run_parser.add_argument("--worlds", type=_positive_worlds, default=16)
    run_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run_parser.add_argument("--library", help="Native training library matching the selected device")
    run_parser.add_argument("--seed", type=_nonnegative_update, default=1729)
    run_parser.add_argument("--max-ticks", type=_positive_ticks,
                            help="Shorten the resolved episode limit for a smoke run")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            destination = prepare(args.runs, args.run_root, args.update, args.output)
        else:
            destination = run(args.bundle, args.episodes, args.worlds, args.device,
                              args.library, args.seed, args.max_ticks)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

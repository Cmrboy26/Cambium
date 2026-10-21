"""Frozen single-parameter sweeps over independent trainer processes."""

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = ("envs", "updates", "rollout_steps", "hidden_dim", "epochs", "minibatch_size", "lr", "threads", "replay_every")
OPTIONS = (*PARAMETERS, "device", "profile_every", "save_every", "max_ticks")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\Z")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def provenance():
    digest = hashlib.sha256()
    for path in sorted([*(ROOT / "simulator/training").glob("*.py"), *(ROOT / "backend").glob("*.py")]):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"revision": revision, "python_sha256": digest.hexdigest()}


def prepare(plan, directory, runs, name, parameter, values, options, max_parallel=0, dry_run=False, allocated_cpus=None):
    from simulator.training.compiler import resolve_experiment
    from simulator.training.train import arguments

    if not NAME.fullmatch(name) or parameter not in PARAMETERS:
        raise ValueError("Expected a safe sweep name and one supported training parameter")
    if not values or type(max_parallel) is not int or max_parallel < 0:
        raise ValueError("Sweep needs values and a nonnegative max-parallel (0 means all)")
    if set(options) - set(OPTIONS):
        raise ValueError("Unsupported sweep training options")
    directory, runs, plan = Path(directory).resolve(), Path(runs).resolve(), Path(plan).resolve()
    if directory.exists():
        raise ValueError(f"Sweep directory already exists: {directory}")
    experiment = resolve_experiment(json.loads(plan.read_text(encoding="utf-8")), plan.parent)
    trials = []
    seen = set()
    for value in values:
        settings = {**options, parameter: value}
        flags = ["--plan", str(plan)]
        for key, setting in settings.items():
            if setting is not None:
                flags.extend(("--" + key.replace("_", "-"), str(setting)))
        parsed = arguments(flags)
        resolved_value = getattr(parsed, parameter)
        if resolved_value in seen:
            raise ValueError("Sweep values must be numerically distinct")
        seen.add(resolved_value)
        trial_name = f"{name}-{str(resolved_value).replace('+', '')}"
        if not NAME.fullmatch(trial_name):
            raise ValueError("Trial name must fit within 120 filename-safe characters")
        output = runs / trial_name
        if output.exists() or output.is_symlink():
            raise ValueError(f"Trial output already exists: {output}")
        frozen = deepcopy(experiment)
        for key in ("hidden_dim", "epochs", "minibatch_size", "lr"):
            setting = getattr(parsed, key)
            if setting is not None:
                frozen["learner"][key] = setting
        frozen = resolve_experiment(frozen, plan.parent)
        training = {key: getattr(parsed, key) for key in OPTIONS if getattr(parsed, key) is not None}
        trials.append({"name": trial_name, "value": resolved_value, "output": str(output),
                       "plan": str(directory / f"{trial_name}.json"), "options": training,
                       "experiment": frozen})
    manifest = {"format": "minecraft_sim_sweep", "version": 1, "name": name, "parameter": parameter,
                "seed": experiment["seed"], "max_parallel": max_parallel, "trials": trials,
                "source": provenance()}
    limit = min(max_parallel or len(trials), len(trials))
    required = sum(sorted((trial["options"]["threads"] for trial in trials), reverse=True)[:limit])
    if allocated_cpus is not None and allocated_cpus < required:
        raise ValueError(f"Concurrent trials need {required} CPU threads; allocation has {allocated_cpus}")
    if dry_run:
        return manifest
    directory.mkdir(parents=True)
    for trial in trials:
        write_json(trial["plan"], trial.pop("experiment"))
    write_json(directory / "sweep.json", manifest)
    return manifest


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if (manifest.get("format") != "minecraft_sim_sweep" or manifest.get("version") != 1
            or manifest.get("parameter") not in PARAMETERS or not manifest.get("trials")):
        raise ValueError("Expected a version-1 single-parameter sweep manifest")
    names, outputs = set(), set()
    for trial in manifest["trials"]:
        if (not NAME.fullmatch(trial["name"]) or trial["name"] in names
                or str(Path(trial["output"]).resolve()) in outputs
                or set(trial["options"]) - set(OPTIONS)):
            raise ValueError("Expected unique trials, outputs and supported training options")
        names.add(trial["name"])
        outputs.add(str(Path(trial["output"]).resolve()))
    return manifest


def run(path, library, allocated_cpus=None, max_parallel=None, trainer=None):
    path, library = Path(path).resolve(), Path(library).resolve()
    manifest = load_manifest(path)
    limit = manifest["max_parallel"] if max_parallel is None else max_parallel
    if type(limit) is not int or limit < 0:
        raise ValueError("max-parallel must be nonnegative (0 means all trials)")
    limit = min(limit or len(manifest["trials"]), len(manifest["trials"]))
    threads = [trial["options"].get("threads", 4) for trial in manifest["trials"]]
    if any(type(value) is not int or value < 1 for value in threads):
        raise ValueError("Trial thread counts must be positive integers")
    if allocated_cpus is None:
        allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_PER_GPU")
                             or (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1))
    required = sum(sorted(threads, reverse=True)[:limit])
    if allocated_cpus < required:
        raise ValueError(f"Concurrent trials need {required} CPU threads; allocation has {allocated_cpus}. "
                         "Request more CPUs, lower trial threads, or set --max-parallel.")
    if not library.is_file():
        raise ValueError(f"Training library not found: {library}")
    status_path = path.parent / "status.json"
    if status_path.exists():
        raise ValueError("Sweep already started; automatic retry/resume is not supported")
    for trial in manifest["trials"]:
        if not Path(trial["plan"]).is_file() or Path(trial["output"]).exists() or Path(trial["output"]).is_symlink():
            raise ValueError(f"Missing frozen plan or existing trial output: {trial['name']}")
    state = {"version": 1, "state": "running", "started_at": time.time(), "max_parallel": limit,
             "allocated_cpus": allocated_cpus, "source": provenance(),
             "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
             "trials": {trial["name"]: {"state": "pending"} for trial in manifest["trials"]}}
    write_json(status_path, state)
    logs = path.parent / "logs"
    logs.mkdir(exist_ok=True)

    async def supervise():
        stop = asyncio.Event()
        semaphore = asyncio.Semaphore(limit)
        running = {}
        received = []
        loop = asyncio.get_running_loop()

        def interrupted(number):
            received.append(number)
            stop.set()

        for number in (signal.SIGINT, signal.SIGTERM):
            if os.name == "posix":
                loop.add_signal_handler(number, interrupted, number)

        def terminate(process, force=False):
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
                elif force:
                    process.kill()
                else:
                    process.terminate()
            except ProcessLookupError:
                pass

        async def execute(trial):
            record = state["trials"][trial["name"]]
            async with semaphore:
                if stop.is_set():
                    record["state"] = "interrupted"
                    write_json(status_path, state)
                    return
                started = time.monotonic()
                record.update(state="running", started_at=time.time(), source=provenance())
                command = list(trainer or [sys.executable, "-B", "-m", "simulator.training.train"])
                command.extend(("--plan", trial["plan"], "--output", trial["output"], "--library", str(library)))
                for key, value in trial["options"].items():
                    command.extend(("--" + key.replace("_", "-"), str(value)))
                environment = dict(os.environ)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    environment[key] = str(trial["options"].get("threads", 4))
                record["command"] = command
                process = None
                try:
                    with (logs / f"{trial['name']}.out").open("wb") as output, (logs / f"{trial['name']}.err").open("wb") as errors:
                        process = await asyncio.create_subprocess_exec(*command, cwd=ROOT, env=environment,
                            stdin=subprocess.DEVNULL, stdout=output, stderr=errors, start_new_session=os.name == "posix")
                        running[trial["name"]] = process
                        record["pid"] = process.pid
                        write_json(status_path, state)
                        if stop.is_set():
                            terminate(process)
                        code = await process.wait()
                    record.update(returncode=code, state="interrupted" if stop.is_set() else "succeeded" if code == 0 else "failed")
                except OSError as error:
                    record.update(state="failed", error=str(error))
                finally:
                    if process is not None and process.returncode is None:
                        terminate(process)
                        try:
                            await asyncio.wait_for(process.wait(), timeout=5)
                        except asyncio.TimeoutError:
                            terminate(process, force=True)
                            await process.wait()
                    running.pop(trial["name"], None)
                    record.update(finished_at=time.time(), elapsed_seconds=time.monotonic() - started)
                    write_json(status_path, state)

        group = asyncio.gather(*(execute(trial) for trial in manifest["trials"]))
        stopping = asyncio.create_task(stop.wait())
        try:
            await asyncio.wait((group, stopping), return_when=asyncio.FIRST_COMPLETED)
            if stop.is_set():
                for process in running.values():
                    terminate(process)
                try:
                    await asyncio.wait_for(asyncio.shield(group), timeout=5)
                except asyncio.TimeoutError:
                    for process in running.values():
                        terminate(process, force=True)
            await group
        finally:
            stopping.cancel()
            for number in (signal.SIGINT, signal.SIGTERM):
                if os.name == "posix":
                    loop.remove_signal_handler(number)
        return 128 + received[0] if received else int(any(record["state"] != "succeeded" for record in state["trials"].values()))

    code = asyncio.run(supervise())
    state.update(state="interrupted" if code >= 128 else "failed" if code else "succeeded", finished_at=time.time())
    write_json(status_path, state)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare", help="Freeze trials before submitting one allocation")
    create.add_argument("--plan", required=True)
    create.add_argument("--directory", required=True)
    create.add_argument("--runs", default=str(ROOT / "simulator/training/runs"))
    create.add_argument("--name", required=True)
    create.add_argument("--parameter", choices=PARAMETERS, required=True)
    create.add_argument("--values", nargs="+", required=True)
    create.add_argument("--options", type=json.loads, default={})
    create.add_argument("--max-parallel", type=int, default=0)
    create.add_argument("--dry-run", action="store_true")
    create.add_argument("--allocated-cpus", type=int)
    execute = commands.add_parser("run", help="Run independent trials inside the current allocation")
    execute.add_argument("manifest")
    execute.add_argument("--library", required=True)
    execute.add_argument("--allocated-cpus", type=int)
    execute.add_argument("--max-parallel", type=int)
    options = parser.parse_args(argv)
    if options.command == "run":
        return run(options.manifest, options.library, options.allocated_cpus, options.max_parallel)
    prepare(options.plan, options.directory, options.runs, options.name, options.parameter,
            options.values, options.options, options.max_parallel, options.dry_run, options.allocated_cpus)
    print(Path(options.directory).resolve() / "sweep.json")


if __name__ == "__main__":
    raise SystemExit(main())
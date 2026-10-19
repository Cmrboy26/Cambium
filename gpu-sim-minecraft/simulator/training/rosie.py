"""On-demand SSH artifact cache and local Minecraft policy launcher."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


CONFIG_NAME = "cambium-rosie.json"
DEFAULT_RUNS = "~/minecraftlol/Cambium/gpu-sim-minecraft/simulator/training/runs"
MAX_FILE = 512 * 1024 * 1024
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\Z")
REMOTE_SCRIPT = r'''
import base64, hashlib, json, pathlib, re, sys
request = json.loads(base64.b64decode(sys.argv[1]))
root = pathlib.Path(request["runs_directory"]).expanduser().resolve(strict=True)
def name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", value):
        raise ValueError("Run/replay names must contain only letters, digits, dots, underscores or hyphens")
    return value
def inside(path):
    result = path.resolve(strict=True)
    result.relative_to(root)
    return result
def entry(path):
    stat = inside(path).stat()
    return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
action = request["action"]
if action == "runs":
    rows = []
    for folder in root.iterdir():
        if not folder.is_dir() or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", folder.name):
            continue
        inside(folder)
        checkpoint = folder / "latest.pt"
        replays = folder / "replays"
        if checkpoint.is_file() or replays.is_dir():
            rows.append({"name": folder.name, "checkpoint": checkpoint.is_file(),
                         "mtime_ns": entry(checkpoint if checkpoint.is_file() else folder)["mtime_ns"]})
    print(json.dumps(sorted(rows, key=lambda row: (-row["mtime_ns"], row["name"]))))
else:
    folder = inside(root / name(request["run"]))
    if action == "manifest":
        files = {}
        for filename in ("latest.pt", "config.json", "metrics.jsonl"):
            path = folder / filename
            if path.is_file():
                files[filename] = entry(path)
        replay_dir = folder / "replays"
        replays = []
        if replay_dir.is_dir():
            inside(replay_dir)
            for path in replay_dir.glob("*.jsonl"):
                name(path.name)
                if path.is_file():
                    replays.append(entry(path))
        print(json.dumps({"files": files, "replays": sorted(replays,
                         key=lambda row: (-row["mtime_ns"], row["name"]))}))
    elif action == "fetch":
        filename = request["file"]
        if filename in ("latest.pt", "config.json", "metrics.jsonl"):
            path = folder / filename
        else:
            path = folder / "replays" / name(filename)
            if path.suffix != ".jsonl":
                raise ValueError("Only replay JSONL files may be fetched")
        with inside(path).open("rb") as source:
            stat = __import__("os").fstat(source.fileno())
            limit = 64 * 1024 * 1024 if path.suffix == ".jsonl" else 512 * 1024 * 1024
            if stat.st_size > limit:
                raise ValueError("Artifact exceeds transfer size limit")
            digest = hashlib.sha256()
            remaining = stat.st_size
            while remaining:
                chunk = source.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError("Artifact changed while hashing; retry sync")
                digest.update(chunk)
                remaining -= len(chunk)
            source.seek(0)
            header = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}
            sys.stdout.buffer.write(json.dumps(header).encode() + b"\n")
            remaining = stat.st_size
            while remaining:
                chunk = source.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError("Artifact changed while reading; retry sync")
                sys.stdout.buffer.write(chunk)
                remaining -= len(chunk)
    else:
        raise ValueError("Unknown artifact operation")
'''


def valid_name(value):
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError("Invalid run/replay name; use letters, digits, dots, underscores or hyphens (max 120)")
    return value


def config_path(game):
    return Path(game).resolve() / "config" / CONFIG_NAME


def configure(game, host, user, runs_directory):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host):
        raise ValueError("SSH host must be a hostname or an SSH config alias")
    if not user or user.startswith("-") or any(character.isspace() for character in user):
        raise ValueError("A valid Rosie SSH username is required")
    path = config_path(game)
    configuration = {"python": str(Path(sys.executable).absolute()),
                     "repository": str(Path(__file__).resolve().parents[2]),
                     "ssh_host": host, "ssh_user": user, "runs_directory": runs_directory}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    return path


class Rosie:
    def __init__(self, game, offline=False):
        self.game = Path(game).resolve()
        self.configuration = json.loads(config_path(game).read_text(encoding="utf-8-sig"))
        self.cache = self.game / "sim-rosie-cache"
        self.offline = offline

    def ssh_command(self, action, **values):
        request = {"action": action, "runs_directory": self.configuration["runs_directory"], **values}
        encoded = base64.b64encode(json.dumps(request).encode()).decode()
        code = base64.b64encode(REMOTE_SCRIPT.encode()).decode()
        remote = "python3 -c " + shlex.quote("import base64;exec(base64.b64decode('" + code + "'))")
        remote += " " + shlex.quote(encoded)
        return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
                "-l", self.configuration["ssh_user"], "--", self.configuration["ssh_host"], remote]

    def request(self, action, **values):
        result = subprocess.run(self.ssh_command(action, **values), capture_output=True, timeout=90)
        if result.returncode:
            raise RuntimeError("Rosie SSH failed. Connect VPN and verify SSH key/agent login first. "
                               + result.stderr.decode("utf-8", errors="replace").strip())
        return json.loads(result.stdout)

    def runs(self):
        if self.offline:
            rows = []
            for folder in self.cache.glob("*"):
                checkpoint = folder / "latest.pt"
                if folder.is_dir() and NAME.fullmatch(folder.name):
                    rows.append({"name": folder.name, "checkpoint": checkpoint.is_file(),
                                 "mtime_ns": (checkpoint if checkpoint.exists() else folder).stat().st_mtime_ns})
            return sorted(rows, key=lambda row: (-row["mtime_ns"], row["name"]))
        return self.request("runs")

    def select(self, run, checkpoint=False):
        if run != "latest":
            return valid_name(run)
        rows = [row for row in self.runs() if not checkpoint or row["checkpoint"]]
        if not rows:
            raise ValueError("No saved runs/checkpoints found")
        return valid_name(rows[0]["name"])

    def manifest(self, run):
        path = self.cache / valid_name(run) / "manifest.json"
        if self.offline:
            return json.loads(path.read_text(encoding="utf-8"))
        manifest = self.request("manifest", run=run)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def fetch(self, run, filename, destination, expected):
        valid_name(run)
        valid_name(filename)
        destination = Path(destination)
        if destination.is_file():
            stat = destination.stat()
            if self.offline or (stat.st_size == expected["size"] and stat.st_mtime_ns // 100 == expected["mtime_ns"] // 100):
                return destination
        if self.offline:
            raise ValueError("Artifact is not cached locally; sync it while online first")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.TemporaryFile() as errors, tempfile.NamedTemporaryFile(
                    dir=destination.parent, prefix=".rosie-", suffix=".part", delete=False) as target:
                temporary = Path(target.name)
                with subprocess.Popen(self.ssh_command("fetch", run=run, file=filename),
                                      stdout=subprocess.PIPE, stderr=errors) as process:
                    try:
                        header = json.loads(process.stdout.readline(8192))
                        size = header["size"]
                        if type(size) is not int or not 0 <= size <= MAX_FILE:
                            raise ValueError("Invalid artifact size")
                        digest = hashlib.sha256()
                        remaining = size
                        while remaining:
                            chunk = process.stdout.read(min(remaining, 1024 * 1024))
                            if not chunk:
                                raise ValueError("Incomplete artifact transfer; retry sync")
                            target.write(chunk)
                            digest.update(chunk)
                            remaining -= len(chunk)
                        if process.stdout.read(1) or process.wait(timeout=30) or digest.hexdigest() != header["sha256"]:
                            raise ValueError("Artifact transfer failed integrity check; retry sync")
                    except BaseException as error:
                        process.kill()
                        process.wait()
                        errors.seek(0)
                        detail = errors.read().decode("utf-8", errors="replace").strip()
                        if detail:
                            raise RuntimeError("Rosie transfer failed: " + detail) from error
                        raise
            os.utime(temporary, ns=(header["mtime_ns"], header["mtime_ns"]))
            temporary.replace(destination)
            return destination
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def checkpoint(self, run, manifest):
        if "latest.pt" not in manifest["files"]:
            raise ValueError("This run has no saved checkpoint yet")
        for filename in ("config.json", "metrics.jsonl", "latest.pt"):
            if filename in manifest["files"]:
                self.fetch(run, filename, self.cache / run / filename, manifest["files"][filename])
        return self.cache / run / "latest.pt"

    def replay(self, run, manifest, filename="latest"):
        replays = manifest["replays"]
        if filename != "latest":
            valid_name(filename)
            replays = [entry for entry in replays if entry["name"] == filename]
        if not replays:
            raise ValueError("No matching replay. Enable SIM_REPLAY_EVERY when submitting training.")
        entry = replays[0]
        filename = valid_name(entry["name"])
        return self.fetch(run, filename, self.game / "sim-replays" / (run + "--" + filename), entry)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "runs", "replays", "replay", "sync", "duel"))
    parser.add_argument("--minecraft-directory", required=True)
    parser.add_argument("--run", default="latest")
    parser.add_argument("--replay", default="latest")
    parser.add_argument("--host")
    parser.add_argument("--user")
    parser.add_argument("--runs-directory", default=DEFAULT_RUNS)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    def emit(event, **values):
        message = {"event": event, **values}
        if args.json:
            print(json.dumps(message), flush=True)
        elif event in ("runs", "replays"):
            for row in values["items"]:
                print(row["name"], "(checkpoint)" if row.get("checkpoint") else "", flush=True)
        else:
            print(json.dumps(message, indent=2), flush=True)

    try:
        if args.action == "setup":
            if not args.host or not args.user:
                raise ValueError("Setup requires --host and --user; use your assigned Rosie management host")
            path = configure(args.minecraft_directory, args.host, args.user, args.runs_directory)
            emit("configured", path=str(path))
            return 0
        rosie = Rosie(args.minecraft_directory, args.offline)
        if args.action == "runs":
            emit("runs", items=rosie.runs())
            return 0
        run = rosie.select(args.run, checkpoint=args.action in ("duel", "sync"))
        manifest = rosie.manifest(run)
        if args.action == "replays":
            emit("replays", run=run, items=manifest["replays"])
        elif args.action == "replay":
            emit("replay", run=run, path=str(rosie.replay(run, manifest, args.replay)))
        else:
            checkpoint = rosie.checkpoint(run, manifest)
            if args.action == "sync":
                for entry in manifest["replays"]:
                    rosie.replay(run, manifest, entry["name"])
                emit("synced", run=run, checkpoint=str(checkpoint), replays=len(manifest["replays"]))
            else:
                import torch
                from simulator.training.serve import Policy, make_server

                torch.set_num_threads(1)
                with make_server(Policy(checkpoint), 0) as server:
                    emit("policy", run=run, checkpoint=str(checkpoint), port=server.server_address[1])
                    server.serve_forever()
        return 0
    except ImportError as error:
        emit("error", message="Local inference requires the training Python dependencies: " + str(error))
        return 1
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        emit("error", message=str(error))
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
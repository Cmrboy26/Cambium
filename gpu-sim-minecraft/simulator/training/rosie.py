"""On-demand SSH artifact cache and local Minecraft policy launcher."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
from http.client import HTTPConnection, HTTPException
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
from textwrap import dedent


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
        snapshots = []
        snapshot_dir = folder / "snapshots"
        if snapshot_dir.is_dir():
            inside(snapshot_dir)
            for path in snapshot_dir.glob("*.pt"):
                name(path.name)
                if path.is_file():
                    snapshots.append(entry(path))
        print(json.dumps({"files": files, "snapshots": sorted(snapshots, key=lambda row: row["name"]),
                         "replays": sorted(replays, key=lambda row: (-row["mtime_ns"], row["name"]))}))
    elif action == "fetch":
        filename = request["file"]
        if filename in ("latest.pt", "config.json", "metrics.jsonl"):
            path = folder / filename
        elif filename.startswith("snapshots/"):
            path = folder / "snapshots" / name(filename.removeprefix("snapshots/"))
            if path.suffix != ".pt":
                raise ValueError("Only policy snapshots may be fetched")
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

SESSION_SCRIPT = dedent(r'''
    import base64, hmac, json, subprocess, sys, tempfile, threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    configuration = json.loads(base64.b64decode(sys.stdin.buffer.readline(65536)))
    reader = base64.b64decode(configuration["reader"]).decode()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *arguments):
            pass
        def reply(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def authorized(self):
            return hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + configuration["token"])
        def do_GET(self):
            if not self.authorized():
                self.reply(401, {"error": "Unauthorized"})
            elif self.path == "/health":
                self.reply(200, {"connected": True})
            else:
                self.reply(404, {"error": "Not found"})
        def do_POST(self):
            if not self.authorized():
                self.reply(401, {"error": "Unauthorized"})
                return
            if self.path != "/artifact":
                self.reply(404, {"error": "Not found"})
                return
            process = None
            started = False
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 8192:
                    raise ValueError("Invalid request length")
                request = json.loads(self.rfile.read(length))
                if request.get("action") not in ("runs", "manifest", "fetch"):
                    raise ValueError("Unsupported artifact request")
                request["runs_directory"] = configuration["runs_directory"]
                encoded = base64.b64encode(json.dumps(request).encode()).decode()
                with tempfile.TemporaryFile() as errors:
                    process = subprocess.Popen([sys.executable, "-c", reader, encoded],
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors)
                    try:
                        if request["action"] != "fetch":
                            output, _ = process.communicate(timeout=90)
                            if process.returncode:
                                errors.seek(0)
                                raise ValueError(errors.read(8192).decode(errors="replace"))
                            self.reply(200, json.loads(output))
                            return
                        header = process.stdout.readline(8192)
                        if not header:
                            process.wait(timeout=10)
                            errors.seek(0)
                            raise ValueError(errors.read(8192).decode(errors="replace"))
                        size = json.loads(header)["size"]
                        if type(size) is not int or not 0 <= size <= 512 * 1024 * 1024:
                            raise ValueError("Invalid transfer size")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("Content-Length", str(len(header) + size))
                        self.end_headers()
                        started = True
                        self.wfile.write(header)
                        remaining = size
                        while remaining:
                            chunk = process.stdout.read(min(remaining, 1024 * 1024))
                            if not chunk:
                                raise ValueError("Incomplete artifact transfer")
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                        if process.wait(timeout=30):
                            raise ValueError("Artifact reader failed")
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                        process.stdout.close()
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                if started:
                    self.close_connection = True
                else:
                    self.reply(502, {"error": str(error)})
    server = ThreadingHTTPServer(("127.0.0.1", configuration["port"]), Handler)
    def watch_connection():
        sys.stdin.buffer.read(1)
        server.shutdown()
    threading.Thread(target=watch_connection, daemon=True).start()
    print(json.dumps({"ready": True, "port": server.server_port}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    ''')


def valid_name(value):
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError("Invalid run/replay name; use letters, digits, dots, underscores or hyphens (max 120)")
    return value


def config_path(game):
    return Path(game).resolve() / "config" / CONFIG_NAME


def session_path(game):
    return config_path(game).with_name("cambium-rosie-session.json")


def connection_id(configuration):
    identity = {key: configuration[key] for key in ("ssh_host", "ssh_user", "runs_directory")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def ssh_options(configuration, interactive=False):
    return ["ssh", "-T", "-o", "BatchMode=no" if interactive else "BatchMode=yes",
            "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2"]


@contextmanager
def session_response(session, request=None):
    port = session["port"]
    if type(port) is not int or not 1 <= port <= 65535 or not isinstance(session["token"], str):
        raise ValueError("Invalid terminal connection descriptor; run scripts/rosie-play.ps1 -Connect again")
    connection = HTTPConnection("127.0.0.1", port, timeout=90)
    response = None
    try:
        headers = {"Authorization": "Bearer " + session["token"], "Content-Type": "application/json"}
        connection.request("GET" if request is None else "POST", "/health" if request is None else "/artifact",
                           None if request is None else json.dumps(request).encode(), headers)
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError("Rosie terminal connection failed: " + response.read(8192).decode(errors="replace"))
        yield response
    except (OSError, HTTPException) as error:
        raise RuntimeError("Rosie terminal connection is unavailable. Run scripts/rosie-play.ps1 -Connect "
                           "in a local terminal and leave it open. " + str(error)) from error
    finally:
        if response is not None:
            response.close()
        connection.close()


def connect_terminal(game):
    if not sys.stdin.isatty():
        raise ValueError("Run scripts/rosie-play.ps1 -Connect in a local terminal; enter the SSH password there, not in Minecraft")
    configuration = json.loads(config_path(game).read_text(encoding="utf-8-sig"))
    descriptor = session_path(game)
    if descriptor.is_file():
        previous = json.loads(descriptor.read_text(encoding="utf-8"))
        try:
            with session_response(previous):
                raise ValueError("A Rosie connection is already open for this game folder. Close it with Ctrl+C before reconnecting.")
        except RuntimeError:
            pass
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        local_port = listener.getsockname()[1]
    remote_port = 20000 + secrets.randbelow(40000)
    token = secrets.token_urlsafe(32)
    request = {"runs_directory": configuration["runs_directory"], "token": token, "port": remote_port,
               "reader": base64.b64encode(REMOTE_SCRIPT.encode()).decode()}
    code = base64.b64encode(SESSION_SCRIPT.encode()).decode()
    remote = "python3 -u -c " + shlex.quote("import base64;exec(base64.b64decode('" + code + "'))")
    command = ssh_options(configuration, interactive=True) + ["-o", "ExitOnForwardFailure=yes", "-L",
        f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}", "-l", configuration["ssh_user"],
        "--", configuration["ssh_host"], remote]
    print("Connecting to Rosie. Enter any SSH password, MFA or host-key confirmation in this terminal only.", flush=True)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    input_stream = process.stdin
    output_stream = process.stdout
    if input_stream is None or output_stream is None:
        process.kill()
        process.wait()
        raise RuntimeError("Could not open SSH connection pipes")
    try:
        input_stream.write(base64.b64encode(json.dumps(request).encode()) + b"\n")
        input_stream.flush()
        line = output_stream.readline(8192)
        if not line:
            raise RuntimeError("SSH connection failed; see the terminal's SSH error above. Verify VPN and your configured host/account.")
        ready = json.loads(line)
        if ready != {"ready": True, "port": remote_port}:
            raise ValueError("Unexpected Rosie connection response")
        session = {"port": local_port, "token": token, "connection_id": connection_id(configuration)}
        with session_response(session):
            pass
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=descriptor.parent, delete=False) as target:
            temporary = Path(target.name)
            try:
                json.dump(session, target)
                target.close()
                temporary.replace(descriptor)
            finally:
                temporary.unlink(missing_ok=True)
        print("Rosie connected. Minecraft can now use /simrosie runs, replay and duel.", flush=True)
        print("Leave this terminal open. Ctrl+C disconnects; your password is not stored.", flush=True)
        if process.wait():
            raise RuntimeError("Rosie SSH connection closed; run -Connect again to reconnect")
    finally:
        try:
            input_stream.close()
        except BrokenPipeError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        output_stream.close()
        if descriptor.is_file():
            saved = json.loads(descriptor.read_text(encoding="utf-8"))
            if saved.get("token") == token:
                descriptor.unlink()


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
        self.session = None
        descriptor = session_path(game)
        if not offline and descriptor.is_file():
            self.session = json.loads(descriptor.read_text(encoding="utf-8"))
            if self.session.get("connection_id") != connection_id(self.configuration):
                raise ValueError("Rosie connection settings changed. Close the old terminal connection and run scripts/rosie-play.ps1 -Connect again.")

    def ssh_command(self, action, **values):
        request = {"action": action, "runs_directory": self.configuration["runs_directory"], **values}
        encoded = base64.b64encode(json.dumps(request).encode()).decode()
        code = base64.b64encode(REMOTE_SCRIPT.encode()).decode()
        remote = "python3 -c " + shlex.quote("import base64;exec(base64.b64decode('" + code + "'))")
        remote += " " + shlex.quote(encoded)
        return ssh_options(self.configuration) + ["-l", self.configuration["ssh_user"],
                                                  "--", self.configuration["ssh_host"], remote]

    def request(self, action, **values):
        if self.session is not None:
            with session_response(self.session, {"action": action, **values}) as response:
                return json.load(response)
        result = subprocess.run(self.ssh_command(action, **values), capture_output=True, timeout=90)
        if result.returncode:
            raise RuntimeError("Rosie SSH failed. For password login, run scripts/rosie-play.ps1 -Connect "
                               "in a local terminal, enter the password there, and leave it open. "
                               + result.stderr.decode("utf-8", errors="replace").strip())
        return json.loads(result.stdout)

    @contextmanager
    def artifact_source(self, run, filename):
        if self.session is not None:
            with session_response(self.session, {"action": "fetch", "run": run, "file": filename}) as response:
                yield response
            return
        with tempfile.TemporaryFile() as errors, subprocess.Popen(self.ssh_command("fetch", run=run, file=filename),
                stdout=subprocess.PIPE, stderr=errors) as process:
            output_stream = process.stdout
            if output_stream is None:
                process.kill()
                raise RuntimeError("Could not open artifact download pipe")
            try:
                yield output_stream
                if process.wait(timeout=30):
                    raise ValueError("Artifact transfer failed")
            except BaseException as error:
                process.kill()
                process.wait()
                errors.seek(0)
                detail = errors.read().decode("utf-8", errors="replace").strip()
                if detail:
                    raise RuntimeError("Rosie transfer failed. For password login, run scripts/rosie-play.ps1 -Connect "
                                       "in a local terminal and leave it open. " + detail) from error
                raise

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

    def fetch(self, run, filename, destination, expected, progress=None):
        valid_name(run)
        if filename.startswith("snapshots/"):
            if not valid_name(filename.removeprefix("snapshots/")).endswith(".pt"):
                raise ValueError("Snapshot must be a .pt file")
        else:
            valid_name(filename)
        destination = Path(destination)
        if destination.is_file():
            stat = destination.stat()
            if self.offline or (stat.st_size == expected["size"] and stat.st_mtime_ns // 100 == expected["mtime_ns"] // 100):
                if progress:
                    progress(stat.st_size, stat.st_size, "cached")
                return destination
        if self.offline:
            raise ValueError("Artifact is not cached locally; sync it while online first")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(0, expected.get("size", 0), "preparing")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    dir=destination.parent, prefix=".rosie-", suffix=".part", delete=False) as target:
                temporary = Path(target.name)
                with self.artifact_source(run, filename) as source:
                    header = json.loads(source.readline(8192))
                    size = header["size"]
                    if type(size) is not int or not 0 <= size <= MAX_FILE:
                        raise ValueError("Invalid artifact size")
                    if progress:
                        progress(0, size, "downloading")
                    digest = hashlib.sha256()
                    remaining = size
                    while remaining:
                        chunk = source.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            raise ValueError("Incomplete artifact transfer; retry sync")
                        target.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
                        if progress:
                            progress(size - remaining, size, "downloading")
                    if progress:
                        progress(size, size, "verifying")
                    if source.read(1) or digest.hexdigest() != header["sha256"]:
                        raise ValueError("Artifact transfer failed integrity check; retry sync")
            os.utime(temporary, ns=(header["mtime_ns"], header["mtime_ns"]))
            temporary.replace(destination)
            if progress:
                progress(size, size, "complete")
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

    def sync_snapshots(self, run, manifest):
        for entry in manifest.get("snapshots", []):
            filename = valid_name(entry["name"])
            self.fetch(run, "snapshots/" + filename, self.cache / run / "snapshots" / filename, entry)

    def sync(self, run, manifest, scope="policy", workers=3, progress=None):
        valid_name(run)
        if scope not in ("policy", "latest", "all") or type(workers) is not int or not 1 <= workers <= 4:
            raise ValueError("Sync requires scope policy/latest/all and 1 to 4 workers")
        if "latest.pt" not in manifest["files"]:
            raise ValueError("This run has no saved checkpoint yet")
        artifacts = []
        for filename in ("config.json", "latest.pt", "metrics.jsonl"):
            if filename in manifest["files"] and (filename != "metrics.jsonl" or scope == "all"):
                artifacts.append((filename, self.cache / run / filename, manifest["files"][filename]))
        for entry in manifest.get("snapshots", []):
            filename = valid_name(entry["name"])
            if not filename.endswith(".pt"):
                raise ValueError("Snapshot must be a .pt file")
            artifacts.append(("snapshots/" + filename, self.cache / run / "snapshots" / filename, entry))
        replays = manifest["replays"] if scope == "all" else manifest["replays"][:1] if scope == "latest" else []
        for entry in replays:
            filename = valid_name(entry["name"])
            artifacts.append((filename, self.game / "sim-replays" / (run + "--" + filename), entry))
        if len({str(destination) for _, destination, _ in artifacts}) != len(artifacts):
            raise ValueError("Duplicate sync artifact")
        states = {filename: {"done": 0, "size": entry["size"], "phase": "queued"} for filename, _, entry in artifacts}
        lock = threading.Lock()
        last_report = 0.0

        def report(filename, done, size, phase):
            nonlocal last_report
            with lock:
                states[filename].update(done=done, size=size, phase=phase)
                now = time.monotonic()
                if progress and (phase != "downloading" or now - last_report >= 0.15):
                    last_report = now
                    completed = sum(state["phase"] in ("complete", "cached") for state in states.values())
                    overall = ("downloading" if any(state["phase"] == "downloading" for state in states.values())
                               else "verifying" if any(state["phase"] == "verifying" for state in states.values())
                               else "preparing" if completed < len(states) else "downloaded")
                    progress(phase=overall, file=filename, file_phase=phase,
                             completed=completed,
                             total=len(states), bytes_done=sum(state["done"] for state in states.values()),
                             bytes_total=sum(state["size"] for state in states.values()),
                             cached=sum(state["phase"] == "cached" for state in states.values()),
                             active=[name + " (" + state["phase"] + ")" for name, state in states.items()
                                     if state["phase"] not in ("complete", "cached", "queued")])

        def download(artifact):
            filename, destination, entry = artifact
            return self.fetch(run, filename, destination, entry,
                              progress=lambda done, size, phase: report(filename, done, size, phase))

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rosie-sync") as executor:
            list(executor.map(download, artifacts))
        if progress:
            progress(phase="indexing", file="Local policy library", completed=len(states), total=len(states),
                     bytes_done=sum(state["done"] for state in states.values()),
                     bytes_total=sum(state["size"] for state in states.values()),
                     cached=sum(state["phase"] == "cached" for state in states.values()), active=[])
        self.index_run(run)
        return len(replays)

    def index_run(self, run):
        from simulator.training.serve import Policy

        path = self.cache / "library.json"
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"policies": [], "environments": []}
        aliases = {entry["id"]: entry["name"] for entry in previous["policies"]}
        policies = [entry for entry in previous["policies"] if entry["run"] != run]
        environments = [entry for entry in previous["environments"] if entry["run"] != run]
        folder = self.cache / valid_name(run)
        checkpoints = [folder / "latest.pt", *sorted((folder / "snapshots").glob("*.pt"))]
        for checkpoint in checkpoints:
            if not checkpoint.is_file():
                continue
            policy = Policy(checkpoint)
            identity = run + ":" + checkpoint.relative_to(folder).as_posix()
            policies.append({"id": identity, "name": aliases.get(identity, run + " / " + checkpoint.stem),
                             "run": run, "path": str(checkpoint), "update": policy.info["update"],
                             "stage": policy.info.get("experiment", {}).get("stage", ""),
                             "contract_hash": policy.info["contract_hash"], "policy": policy.contract})
            environments.append({"id": identity, "name": run + " / " + checkpoint.stem + " environment",
                                 "run": run, "policy": policy.contract, "contract_hash": policy.info["contract_hash"],
                                 "environment": policy.info["environment"]})
        config = folder / "config.json"
        if config.is_file():
            experiment = json.loads(config.read_text(encoding="utf-8")).get("experiment", {})
            for stage_index, stage in enumerate(experiment.get("stages", [])):
                for arena_index, arena in enumerate(stage["scenario"]["arenas"]):
                    from simulator.training.contract import fingerprint

                    contract = experiment["policy"]
                    environments.append({"id": f"{run}:stage:{stage_index}:{arena_index}",
                                         "name": run + " / " + stage["name"] + (f" / arena {arena_index + 1}"
                                             if len(stage["scenario"]["arenas"]) > 1 else ""),
                                         "run": run, "policy": contract,
                                         "contract_hash": fingerprint(contract), "environment": arena["environment"]})
        result = {"version": 1, "policies": policies, "environments": environments, "runs": previous.get("runs", [])}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.cache, delete=False) as target:
            temporary = Path(target.name)
            try:
                json.dump(result, target, indent=2, allow_nan=False)
                target.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "connect", "runs", "replays", "replay", "sync", "duel", "local"))
    parser.add_argument("--duel", type=Path)
    parser.add_argument("--minecraft-directory", required=True)
    parser.add_argument("--run", default="latest")
    parser.add_argument("--replay", default="latest")
    parser.add_argument("--scope", choices=("policy", "latest", "all"), default="policy")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--host")
    parser.add_argument("--user")
    parser.add_argument("--runs-directory", default=DEFAULT_RUNS)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    started = time.monotonic()

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
        if args.action == "connect":
            if args.json or args.offline:
                raise ValueError("Password login requires connect in a local terminal, without --json or --offline")
            connect_terminal(args.minecraft_directory)
            return 0
        if args.action == "local":
            emit("startup", phase="Importing PyTorch", seconds=time.monotonic() - started)
            import torch
            from simulator.training.serve import DuelPolicy, make_server

            if args.duel is None:
                raise ValueError("Local duel requires --duel <configuration.json>")
            torch.set_num_threads(1)
            emit("startup", phase="Loading local fighters", seconds=time.monotonic() - started)
            reload_token = secrets.token_urlsafe(32)
            with make_server(DuelPolicy(args.duel), 0, reload_policy=lambda: DuelPolicy(args.duel), reload_token=reload_token) as server:
                emit("policy", run="Local duel", port=server.server_address[1], reload_token=reload_token, startup_seconds=time.monotonic() - started)
                server.serve_forever()
            return 0
        rosie = Rosie(args.minecraft_directory, args.offline)
        if args.action == "runs":
            emit("runs", items=rosie.runs())
            return 0
        if args.action == "duel":
            emit("startup", phase="Reading Rosie run metadata", seconds=time.monotonic() - started)
        run = rosie.select(args.run, checkpoint=args.action in ("duel", "sync"))
        if args.action == "sync":
            emit("progress", phase="manifest", file="Reading remote file list", completed=0, total=0,
                 bytes_done=0, bytes_total=0, cached=0, active=[])
        manifest = rosie.manifest(run)
        if args.action == "replays":
            emit("replays", run=run, items=manifest["replays"])
        elif args.action == "replay":
            emit("replay", run=run, path=str(rosie.replay(run, manifest, args.replay)))
        else:
            if args.action == "sync":
                count = rosie.sync(run, manifest, args.scope, args.workers, lambda **values: emit("progress", **values))
                emit("synced", run=run, checkpoint=str(rosie.cache / run / "latest.pt"), replays=count, scope=args.scope)
            else:
                emit("startup", phase="Checking cached checkpoint / contacting Rosie", seconds=time.monotonic() - started)
                checkpoint = rosie.checkpoint(run, manifest)
                emit("startup", phase="Importing PyTorch", seconds=time.monotonic() - started)
                import torch
                from simulator.training.serve import Policy, make_server

                torch.set_num_threads(1)
                emit("startup", phase="Loading checkpoint", seconds=time.monotonic() - started)
                with make_server(Policy(checkpoint), 0) as server:
                    emit("policy", run=run, checkpoint=str(checkpoint), port=server.server_address[1], startup_seconds=time.monotonic() - started)
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
"""Read-only, loopback dashboard for standalone simulator PPO runs."""

import argparse
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from urllib.parse import urlsplit


RUNS = Path(__file__).resolve().parent / "runs"
ASSETS = Path(__file__).resolve().parent / "dashboard"
METRICS = ("update", "samples", "completed_rounds", "terminated_duels", "timeouts",
           "replays_exported", "reward_mean", "reward_total", "loss", "policy_loss", "value_loss", "entropy",
           "collection_seconds", "update_seconds", "evaluation_seconds", "evaluation_episodes",
           "replay_seconds", "iteration_seconds", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
           "player_transitions_per_second")
COUNTS = {"update", "samples", "completed_rounds", "terminated_duels", "timeouts", "replays_exported",
          "evaluation_episodes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"}
MAX_REPLAY_BYTES = 64 * 1024 * 1024
MAX_REPLAY_FRAMES = 100001


def count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Expected a nonnegative integer count")
    return value


def finite(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("Expected a finite number")
    return value


def metric_metadata(record):
    result = {}
    if "stage" in record:
        if not isinstance(record["stage"], str) or not record["stage"]:
            raise ValueError("Expected a stage name")
        result["stage"] = record["stage"]
    for name in ("stage_index", "next_stage_index"):
        if name in record:
            result[name] = None if name == "next_stage_index" and record[name] is None else count(record[name])
    if "stage_progress" in record:
        progress = record["stage_progress"]
        if not isinstance(progress, dict):
            raise ValueError("Expected stage progress")
        result["stage_progress"] = {name: count(progress.get(name)) for name in ("updates", "episodes", "transitions")}
    if "completed_curriculum" in record:
        if type(record["completed_curriculum"]) is not bool:
            raise ValueError("Expected curriculum completion boolean")
        result["completed_curriculum"] = record["completed_curriculum"]
    if "evaluations" in record:
        suites = record["evaluations"]
        if not isinstance(suites, dict):
            raise ValueError("Expected evaluation suites")
        evaluations = {}
        for label, evaluation in suites.items():
            if not label or not isinstance(evaluation, dict):
                raise ValueError("Expected a named evaluation result")
            win_rate = finite(evaluation.get("win_rate"))
            if not 0 <= win_rate <= 1:
                raise ValueError("Expected win rate between zero and one")
            evaluations[label] = {"win_rate": win_rate, "reward_mean": finite(evaluation.get("reward_mean")),
                                  "episodes": count(evaluation.get("episodes"))}
            for name in ("worlds", "snapshot_update", "stage_index"):
                if name in evaluation:
                    evaluations[label][name] = count(evaluation[name])
            if "seconds" in evaluation:
                seconds = finite(evaluation["seconds"])
                if seconds < 0:
                    raise ValueError("Expected nonnegative evaluation seconds")
                evaluations[label]["seconds"] = seconds
            if "stage" in evaluation:
                if not isinstance(evaluation["stage"], str) or not evaluation["stage"]:
                    raise ValueError("Expected an evaluation stage name")
                evaluations[label]["stage"] = evaluation["stage"]
        result["evaluations"] = evaluations
    if "curriculum_boundary" in record:
        boundary = record["curriculum_boundary"]
        if not isinstance(boundary, dict) or "next_stage" not in boundary:
            raise ValueError("Expected a curriculum boundary")
        upcoming = boundary.get("next_stage")
        if upcoming is not None and (not isinstance(upcoming, str) or not upcoming):
            raise ValueError("Expected upcoming stage name or null")
        result["curriculum_boundary"] = {"reset_worlds": count(boundary.get("reset_worlds")), "next_stage": upcoming}
    return result


class RunStore:
    def __init__(self, directory=RUNS, runs=()):
        self.directory = Path(directory).resolve()
        self.explicit = tuple(Path(run).resolve() for run in runs)

    def paths(self):
        candidates = list(self.explicit)
        if self.directory.is_dir():
            candidates.extend(path for path in self.directory.iterdir()
                              if path.is_dir() and path.resolve().parent == self.directory)
        found = {}
        for path in candidates:
            if path.is_dir() and any(self.file(path, name).is_file() for name in ("config.json", "metrics.jsonl")):
                found[sha256(str(path).encode()).hexdigest()[:16]] = path
        return found

    @staticmethod
    def file(path, name):
        result = path / name
        if result.resolve().parent != path.resolve():
            raise ValueError("Run files must stay inside their run directory")
        return result

    def list_runs(self):
        result = []
        for identity, path in self.paths().items():
            modified = max(self.file(path, name).stat().st_mtime
                           for name in ("config.json", "metrics.jsonl") if self.file(path, name).is_file())
            result.append({"id": identity, "name": path.name, "path": str(path), "modified": modified})
        return sorted(result, key=lambda run: (run["modified"], run["name"]), reverse=True)

    def replay_paths(self, identity):
        run = self.paths().get(identity)
        if run is None:
            raise FileNotFoundError("Run not found")
        directories = []
        local = run / "replays"
        if not local.is_symlink():
            directories.append(local.resolve())
        config = self.file(run, "config.json")
        if config.is_file():
            try:
                options = json.loads(config.read_text(encoding="utf-8")).get("options", {})
                configured = options.get("replay_dir")
                if isinstance(configured, str) and Path(configured).is_absolute():
                    directories.append(Path(configured).resolve())
            except (ValueError, UnicodeError, AttributeError):
                pass
        found = {}
        for directory in dict.fromkeys(directories):
            if not directory.is_dir():
                continue
            for path in directory.glob("*.jsonl"):
                if path.is_file() and not path.is_symlink() and path.resolve().parent == directory:
                    found[sha256(str(path).encode()).hexdigest()[:16]] = path
        return found

    def list_replays(self, identity):
        replays = []
        for replay_id, path in self.replay_paths(identity).items():
            size = path.stat().st_size
            if size <= MAX_REPLAY_BYTES:
                replays.append({"id": replay_id, "name": path.name, "size": size})
        return sorted(replays, key=lambda replay: replay["name"], reverse=True)

    def read_replay(self, identity, replay_id):
        path = self.replay_paths(identity).get(replay_id)
        if path is None:
            raise FileNotFoundError("Replay not found")
        with path.open("rb") as source:
            content = source.read(MAX_REPLAY_BYTES + 1)
        if len(content) > MAX_REPLAY_BYTES:
            raise ValueError("Replay exceeds the 64 MiB limit")
        if not content.endswith(b"\n"):
            raise ValueError("Replay is incomplete")
        lines = content.splitlines()
        if not 2 <= len(lines) <= MAX_REPLAY_FRAMES + 1:
            raise ValueError("Replay must contain 1 to 100001 frames")
        header, *frames = [json.loads(line) for line in lines]
        if (not isinstance(header, dict) or header.get("format") != "minecraft_sim_replay"
                or header.get("version") not in (2, 3) or header.get("players") != 2
                or not isinstance(header.get("environment"), dict)):
            raise ValueError("Unsupported replay format")
        rate = header.get("ticks_per_second", 20)
        if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= 1000:
            raise ValueError("Invalid replay tick rate")
        previous_tick = -1
        for frame in frames:
            if (not isinstance(frame, dict) or type(frame.get("tick")) is not int
                    or frame["tick"] <= previous_tick or not isinstance(frame.get("actors"), list)
                    or len(frame["actors"]) != 2):
                raise ValueError("Invalid replay frame")
            previous_tick = frame["tick"]
            actors = frame["actors"]
            for actor in actors:
                if (not isinstance(actor, dict) or type(actor.get("id")) is not int
                        or actor["id"] not in (0, 1) or not isinstance(actor.get("position"), list)
                        or len(actor["position"]) != 3
                        or not all(type(value) in (int, float) and math.isfinite(value)
                                   for value in [*actor["position"], actor.get("yaw"), actor.get("pitch"), actor.get("health")])):
                    raise ValueError("Invalid replay actor")
            if actors[0]["id"] == actors[1]["id"]:
                raise ValueError("Duplicate replay actor")
            actors.sort(key=lambda actor: actor["id"])
        json.dumps(header, allow_nan=False)
        json.dumps(frames, allow_nan=False)
        return {"name": path.name, "header": header, "frames": frames}

    def read(self, identity):
        path = self.paths().get(identity)
        if path is None:
            raise FileNotFoundError("Run not found")
        warnings = []
        config = None
        config_path = self.file(path, "config.json")
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if not isinstance(config, dict):
                    raise ValueError("Expected a configuration object")
                json.dumps(config, allow_nan=False)
            except (ValueError, UnicodeError):
                config = None
                warnings.append("Configuration is not valid JSON yet.")
        metrics = []
        invalid = 0
        metrics_path = self.file(path, "metrics.jsonl")
        if metrics_path.is_file():
            with metrics_path.open("rb") as source:
                for line in source:
                    if not line.endswith(b"\n"):
                        break
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict) or type(record.get("update")) is not int or record["update"] < 1:
                            raise ValueError("Expected a positive PPO update")
                        row = {}
                        for name in METRICS:
                            value = record.get(name)
                            if value is not None:
                                row[name] = count(value) if name in COUNTS else finite(value)
                        row.update(metric_metadata(record))
                        if "reward_types" in record:
                            contributions = record["reward_types"]
                            if not isinstance(contributions, dict) or not all(
                                    type(value) in (int, float) and math.isfinite(value)
                                    for value in contributions.values()):
                                raise ValueError("Expected finite reward contributions")
                            row["reward_types"] = contributions
                        if metrics and row["update"] <= metrics[-1]["update"]:
                            metrics = [previous for previous in metrics if previous["update"] < row["update"]]
                        metrics.append(row)
                    except (ValueError, UnicodeError, OverflowError):
                        invalid += 1
        if invalid:
            warnings.append(f"Skipped {invalid} invalid metric record(s).")
        previous = {"update": 0, "samples": 0}
        for row in metrics:
            if ("reward_total" not in row and "reward_mean" in row
                    and row["update"] == previous["update"] + 1
                    and "samples" in row and "samples" in previous):
                transitions = row["samples"] - previous["samples"]
                total = row["reward_mean"] * transitions
                if transitions > 0 and math.isfinite(total):
                    row["reward_total"] = total
            previous = row
        return {"id": identity, "name": path.name, "path": str(path), "config": config,
                "metrics": metrics, "warnings": warnings}


def make_server(store, port=8767):
    assets = {"/": ("index.html", "text/html"), "/dashboard.css": ("dashboard.css", "text/css"),
              "/dashboard.js": ("dashboard.js", "text/javascript"),
              "/chart.umd.min.js": ("chart.umd.min.js", "text/javascript")}
    for name in ("arena.js", "arena-data.js", "three.module.min.js", "three.core.min.js", "OrbitControls.js", "lucide.min.js"):
        assets[f"/{name}"] = (name, "text/javascript")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, status, value, content_type="application/json"):
            body = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlsplit(self.path).path
            try:
                if route == "/api/runs":
                    self.reply(200, {"runs": store.list_runs()})
                elif route.startswith("/api/runs/") and route.endswith("/metrics"):
                    self.reply(200, store.read(route[len("/api/runs/"):-len("/metrics")]))
                elif route.startswith("/api/runs/") and route.endswith("/replays"):
                    self.reply(200, {"replays": store.list_replays(route[len("/api/runs/"):-len("/replays")])})
                elif route.startswith("/api/runs/") and "/replays/" in route:
                    identity, replay_id = route[len("/api/runs/"):].split("/replays/", 1)
                    self.reply(200, store.read_replay(identity, replay_id))
                elif route in assets:
                    filename, content_type = assets[route]
                    self.reply(200, (ASSETS / filename).read_bytes(), content_type)
                else:
                    self.reply(404, {"error": "Not found"})
            except FileNotFoundError:
                self.reply(404, {"error": "Run or asset not found"})
            except (OSError, ValueError) as error:
                self.reply(400, {"error": str(error)})

        def do_POST(self):
            self.reply(405, {"error": "This dashboard is read-only"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS, help="Parent directory of training runs")
    parser.add_argument("--run", type=Path, action="append", default=[], help="Additional run directory; repeatable")
    parser.add_argument("--port", type=int, default=8767)
    options = parser.parse_args(argv)
    if not 0 <= options.port <= 65535:
        parser.error("Port must be between 0 and 65535")
    for path in options.run:
        if not path.is_dir():
            parser.error(f"Run directory does not exist: {path}")
    store = RunStore(options.runs_dir, options.run)
    with make_server(store, options.port) as server:
        print(f"Training dashboard: http://127.0.0.1:{server.server_port}", flush=True)
        print(f"Runs: {store.directory}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
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
           "collection_seconds", "update_seconds", "player_transitions_per_second")


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
                                if type(value) not in (int, float) or not math.isfinite(value):
                                    raise ValueError("Expected finite numeric metrics")
                                row[name] = value
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
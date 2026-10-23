"""Shared filesystem protocol for dashboard-requested training replays."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid


REQUEST_FILENAME = ".dashboard-replay-request.json"


class ReplayRequestConflict(ValueError):
    """A replay request is already waiting to be fulfilled."""


class ReplayRequestUnsupported(ValueError):
    """The selected training run predates dashboard replay requests."""


def request_path(run_directory):
    root = Path(run_directory).resolve()
    path = root / REQUEST_FILENAME
    if path.is_symlink() or path.resolve().parent != root:
        raise ValueError("Replay request must stay inside the run directory")
    return path


def read_request(run_directory):
    path = request_path(run_directory)
    if not path.is_file():
        return None
    request = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(request, dict) or not isinstance(request.get("request_id"), str)
            or request.get("status") not in ("pending", "completed", "failed")):
        raise ValueError("Replay request is not valid")
    return request


def write_request(run_directory, request):
    path = request_path(run_directory)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(request, stream, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_request(run_directory):
    existing = read_request(run_directory)
    if existing is not None and existing["status"] == "pending":
        raise ReplayRequestConflict("A replay request is already pending for this run")
    request = {
        "request_id": uuid.uuid4().hex,
        "status": "pending",
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }
    write_request(run_directory, request)
    return request


def finish_request(run_directory, request_id, status, **details):
    if status not in ("completed", "failed"):
        raise ValueError("Replay request must complete or fail")
    request = read_request(run_directory)
    if (request is None or request["request_id"] != request_id
            or request["status"] != "pending"):
        return False
    request.update(details, status=status, finished_at=datetime.now(timezone.utc).isoformat())
    write_request(run_directory, request)
    return True


def requested_replay_round(previous_rounds, rounds, scheduled_rounds):
    if rounds <= previous_rounds:
        return None
    scheduled_rounds = list(scheduled_rounds)
    return scheduled_rounds[0] if scheduled_rounds else rounds
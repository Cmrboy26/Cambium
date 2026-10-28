"""Shared filesystem protocol for dashboard-requested update profiles."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import uuid


REQUEST_FILENAME = ".dashboard-analytics-request.json"
MAX_REQUEST_BYTES = 1_000_000


class AnalyticsRequestConflict(ValueError):
    """An analytics request is already waiting or running for this run."""


class AnalyticsRequestUnsupported(ValueError):
    """The selected training run predates dashboard analytics requests."""


def request_path(run_directory):
    root = Path(run_directory).resolve()
    path = root / REQUEST_FILENAME
    if path.is_symlink() or path.resolve().parent != root:
        raise ValueError("Analytics request must stay inside the run directory")
    return path


def _validate_profile(profile):
    if not isinstance(profile, dict) or profile.get("version") != 1:
        raise ValueError("Analytics result must contain a version-1 collection profile")
    for section in ("seconds", "native_seconds", "cuda_stream_seconds", "counts"):
        values = profile.get(section)
        if not isinstance(values, dict) or any(
                not isinstance(name, str) or not name
                or type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for name, value in values.items()):
            raise ValueError(f"Analytics profile {section} must contain finite nonnegative values")
    for name in ("total_seconds", "synchronize_seconds", "unaccounted_seconds"):
        value = profile.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Analytics profile {name} must be finite and nonnegative")
    return profile


def read_request(run_directory):
    path = request_path(run_directory)
    if not path.is_file():
        return None
    if path.stat().st_size > MAX_REQUEST_BYTES:
        raise ValueError("Analytics request exceeds the size limit")
    request = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(request, dict) or not isinstance(request.get("request_id"), str)
            or request.get("status") not in ("pending", "running", "completed", "failed")):
        raise ValueError("Analytics request is not valid")
    if "profile" in request:
        _validate_profile(request["profile"])
    if "optimization_profile" in request:
        _validate_profile(request["optimization_profile"])
    if "update" in request and (type(request["update"]) is not int or request["update"] < 1):
        raise ValueError("Analytics request update is invalid")
    if "stage" in request and (not isinstance(request["stage"], str) or not request["stage"]):
        raise ValueError("Analytics request stage is invalid")
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
    if existing is not None and existing["status"] in ("pending", "running"):
        raise AnalyticsRequestConflict("An analytics request is already pending for this run")
    request = {
        "request_id": uuid.uuid4().hex,
        "status": "pending",
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }
    write_request(run_directory, request)
    return request


def mark_running(run_directory, request_id, update, stage):
    request = read_request(run_directory)
    if (request is None or request["request_id"] != request_id
            or request["status"] not in ("pending", "running")):
        return False
    request.update(status="running", update=update, stage=stage,
                   started_at=datetime.now(timezone.utc).isoformat())
    write_request(run_directory, request)
    return True


def finish_request(run_directory, request_id, status, **details):
    if status not in ("completed", "failed"):
        raise ValueError("Analytics request must complete or fail")
    request = read_request(run_directory)
    if (request is None or request["request_id"] != request_id
            or request["status"] not in ("pending", "running")):
        return False
    if status == "completed":
        _validate_profile(details.get("profile"))
        if "optimization_profile" in details:
            _validate_profile(details["optimization_profile"])
    request.update(details, status=status, finished_at=datetime.now(timezone.utc).isoformat())
    write_request(run_directory, request)
    return True
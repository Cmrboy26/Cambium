"""Summarize a standalone simulator training run for quick review."""

import argparse
import json
import math
from pathlib import Path

from simulator.training.dashboard import RUNS, RunStore


METRIC_NAMES = ("reward_mean", "loss", "policy_loss", "value_loss", "entropy",
                "player_transitions_per_second")


def read_run(path=None, include_profiles=False):
    if path is None:
        store = RunStore()
        runs = store.list_runs()
        if not runs:
            raise FileNotFoundError(f"No training runs found under {RUNS}")
        selected = Path(runs[0]["path"])
    else:
        selected = Path(path).expanduser().resolve()
        if selected.is_file() and selected.name == "metrics.jsonl":
            selected = selected.parent
        if not selected.is_dir():
            raise FileNotFoundError(f"Training run directory not found: {selected}")
        store = RunStore(selected.parent, [selected])

    runs = store.list_runs()
    if not runs:
        raise FileNotFoundError(f"No run config or metrics found under {selected}")
    selected_run = next((run for run in runs if Path(run["path"]).resolve() == selected.resolve()), None)
    if selected_run is None:
        raise FileNotFoundError(f"No run config or metrics found under {selected}")
    run = store.read(selected_run["id"])
    if include_profiles:
        profiles = {}
        metrics_path = selected / "metrics.jsonl"
        if metrics_path.is_file():
            with metrics_path.open("rb") as source:
                for line in source:
                    if not line.endswith(b"\n"):
                        break
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict) and type(row.get("update")) is int:
                            profiles[row["update"]] = row.get("collection_profile")
                    except (ValueError, UnicodeError):
                        continue
        for row in run["metrics"]:
            profile = profiles.get(row["update"])
            if profile is not None:
                row["collection_profile"] = profile
    return run


def _average(rows, name):
    values = [row[name] for row in rows if name in row]
    return sum(values) / len(values) if values else None


def _sample_deltas(rows):
    deltas = {}
    previous = None
    previous_update = None
    for row in rows:
        samples = row.get("samples")
        update = row["update"]
        if samples is not None and previous is not None and previous_update == update - 1:
            deltas[update] = max(0, samples - previous)
        elif samples is not None and update == 1:
            deltas[update] = samples
        previous = samples
        previous_update = update
    return deltas


def summarize_run(run, windows=(10, 50)):
    rows = run["metrics"]
    sample_deltas = _sample_deltas(rows)
    window_reports = []
    for width in windows:
        recent = rows[-width:]
        if not recent:
            continue
        terminated = sum(row.get("terminated_duels", 0) for row in recent)
        timeouts = sum(row.get("timeouts", 0) for row in recent)
        endings = terminated + timeouts
        transitions = sum(sample_deltas.get(row["update"], 0) for row in recent)
        reward_types = {}
        for row in recent:
            for name, value in row.get("reward_types", {}).items():
                reward_types[name] = reward_types.get(name, 0.0) + value
        if transitions:
            reward_types = {name: value / transitions for name, value in reward_types.items()}
        else:
            reward_types = {}

        window_reports.append({
            "updates": len(recent),
            "first_update": recent[0]["update"],
            "last_update": recent[-1]["update"],
            "reward_mean": _average(recent, "reward_mean"),
            "positive_reward_updates": sum(row.get("reward_mean", 0) > 0 for row in recent),
            "ended_duels": terminated,
            "timeouts": timeouts,
            "timeout_fraction": timeouts / endings if endings else None,
            "average_metrics": {name: _average(recent, name) for name in METRIC_NAMES},
            "reward_per_player_transition": reward_types,
        })

    config = run.get("config") or {}
    metadata = config.get("metadata") or {}
    options = config.get("options") or {}
    latest = rows[-1] if rows else {}
    return {
        "run": run["name"],
        "path": run["path"],
        "environment": (metadata.get("environment") or {}).get("name") or options.get("environment"),
        "device": options.get("device"),
        "worlds": options.get("envs"),
        "configured_updates": options.get("updates"),
        "updates_recorded": len(rows),
        "first_update": rows[0]["update"] if rows else None,
        "latest_update": latest.get("update"),
        "player_transitions": latest.get("samples"),
        "completed_rounds": latest.get("completed_rounds"),
        "reward_weights": metadata.get("reward_weights"),
        "windows": window_reports,
        "warnings": run.get("warnings", []),
        "interpretation_note": (
            "Shared self-play reward is not a win rate. Compare checkpoints against a fixed opponent "
            "and report both player seats separately."
        ),
    }


def summarize_timings(run, skip_updates=5, stage=None):
    rows = run["metrics"]
    deltas = _sample_deltas(rows)
    rows = rows[skip_updates:]
    if stage is not None:
        rows = [row for row in rows if row.get("stage") == stage]

    def summarize(selected):
        measured = [row for row in selected if row["update"] in deltas and row.get("iteration_seconds", 0) > 0]
        seconds = sum(row["iteration_seconds"] for row in measured)
        return {"updates": len(selected),
                "transitions_per_second": sum(deltas[row["update"]] for row in measured) / seconds if seconds else None,
                "seconds_per_update": {name: _average(selected, name) for name in
                    ("collection_seconds", "update_seconds", "evaluation_seconds", "replay_seconds", "iteration_seconds")}}

    profiles = [row["collection_profile"] for row in rows if "collection_profile" in row]
    for profile in profiles:
        if not isinstance(profile, dict) or profile.get("version") != 1:
            raise ValueError("Unsupported collection profile; expected version 1")
        for section in ("seconds", "native_seconds", "cuda_stream_seconds", "counts"):
            values = profile.get(section)
            if not isinstance(values, dict) or any(type(value) not in (int, float)
                    or not math.isfinite(value) or value < 0 for value in values.values()):
                raise ValueError(f"Collection profile {section} must contain finite nonnegative values")

    def averages(section):
        names = {name for profile in profiles for name in profile[section]}
        return {name: sum(profile[section].get(name, 0) for profile in profiles) / len(profiles)
                for name in sorted(names)}

    groups = {label: summarize([row for row in rows if ("collection_profile" in row) == profiled])
              for label, profiled in (("profiled", True), ("unprofiled", False))}
    stages = {}
    for row in rows:
        stages.setdefault(row.get("stage", "legacy"), []).append(row)
    return {"run": run["name"], "stage": stage, "skipped_updates": min(skip_updates, len(run["metrics"])),
            "groups": groups, "profiled_updates": len(profiles),
            "seconds": averages("seconds"), "native_seconds": averages("native_seconds"),
            "cuda_stream_seconds": averages("cuda_stream_seconds"), "counts_per_profile": averages("counts"),
            "synchronize_seconds": _average(profiles, "synchronize_seconds"),
            "unaccounted_seconds": _average(profiles, "unaccounted_seconds"),
            "stages": {stage: {label: summarize([row for row in selected
                if ("collection_profile" in row) == profiled]) for label, profiled in
                (("profiled", True), ("unprofiled", False))} for stage, selected in stages.items()},
            "warnings": run.get("warnings", [])}


def format_timing_report(report):
    lines = [f"Run: {report['run']} | Skipped initial updates: {report['skipped_updates']}"]
    if report["stage"] is not None:
        lines.append(f"Stage: {report['stage']}")
    for label, group in report["groups"].items():
        rate = group["transitions_per_second"]
        lines.append(f"{label}: {group['updates']} updates; " +
                     (f"{rate:,.0f} training transitions/s" if rate is not None else "throughput unavailable"))
    for stage, groups in report["stages"].items():
        for label, group in groups.items():
            rate = group["transitions_per_second"]
            if group["updates"]:
                lines.append(f"  {stage} / {label}: {group['updates']} updates; " +
                             (f"{rate:,.0f}/s" if rate is not None else "throughput unavailable"))
    for section, title in (("seconds", "Collection wall seconds/profile (dot-named children are included in parents)"),
                           ("native_seconds", "Native wall seconds/profile (included in collection phases)"),
                           ("cuda_stream_seconds", "CUDA stream seconds/profile (may include host submission gaps)")):
        if report[section]:
            lines.append(title + ":")
            lines.extend(f"  {name}: {seconds:.6f}" for name, seconds in
                         sorted(report[section].items(), key=lambda item: item[1], reverse=True))
    if report["counts_per_profile"]:
        lines.append("Counts/profile: " + ", ".join(f"{name}={value:.2f}"
                     for name, value in report["counts_per_profile"].items()))
    if not report["profiled_updates"]:
        lines.append("No collection profiles in this selection; enable --profile-every N when training.")
    lines.append("Profiled updates include instrumentation/synchronization overhead; use unprofiled updates for throughput.")
    lines.extend(f"Warning: {warning}" for warning in report["warnings"])
    return "\n".join(lines)


def format_report(report):
    lines = [
        f"Run: {report['run']}",
        f"Environment: {report['environment'] or 'unknown'} | Device: {report['device'] or 'unknown'} | "
        f"Worlds: {report['worlds'] or 'unknown'}",
        f"Updates: {report['updates_recorded']} recorded, latest {report['latest_update']} / "
        f"{report['configured_updates'] or 'unknown'} | Player transitions: {report['player_transitions'] or 0:,}",
    ]
    if report["completed_rounds"] is not None:
        lines.append(f"Completed duels: {report['completed_rounds']:,} (cumulative)")

    for window in report["windows"]:
        averages = window["average_metrics"]
        timeout_rate = window["timeout_fraction"]
        timeout_text = f"{timeout_rate:.1%}" if timeout_rate is not None else "n/a"
        lines.extend([
            f"Last {window['updates']} updates ({window['first_update']}-{window['last_update']}): "
            f"mean reward {window['reward_mean']:.5f}, positive {window['positive_reward_updates']}/"
            f"{window['updates']}, ended duels {window['ended_duels']:,}, "
            f"timeouts {window['timeouts']:,} ({timeout_text})",
            "  PPO: " + ", ".join(
                f"{name} {averages[name]:.4g}" for name in METRIC_NAMES if averages[name] is not None),
        ])
        if window["reward_per_player_transition"]:
            contributions = sorted(window["reward_per_player_transition"].items(),
                                   key=lambda item: abs(item[1]), reverse=True)
            lines.append("  Reward / player step: " + ", ".join(
                f"{name} {value:+.4g}" for name, value in contributions[:8]))

    lines.append("Note: " + report["interpretation_note"])
    lines.extend(f"Warning: {warning}" for warning in report["warnings"])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", help="Run directory or its metrics.jsonl; defaults to newest run")
    parser.add_argument("--windows", nargs="+", type=int, default=[10, 50],
                        help="Recent update window sizes to report (default: 10 50)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--timings", action="store_true", help="Report collection profiles and separate profiled/unprofiled rates")
    parser.add_argument("--skip-updates", type=int, default=5,
                        help="Initial recorded updates to exclude from --timings (default: 5)")
    parser.add_argument("--stage", help="Select one training stage for --timings")
    args = parser.parse_args(argv)
    if any(width < 1 for width in args.windows):
        parser.error("window sizes must be positive")
    if args.skip_updates < 0:
        parser.error("skip-updates must be nonnegative")

    run = read_run(args.run, include_profiles=args.timings)
    report = summarize_timings(run, args.skip_updates, args.stage) if args.timings else summarize_run(run, args.windows)
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(format_timing_report(report) if args.timings else format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
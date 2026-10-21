"""Overlay a sweep's trial metrics on a single headless PNG chart."""

import argparse
import json
from pathlib import Path

from simulator.training.analyze_metrics import read_run
from simulator.training.metrics import AXES, CATALOG, elapsed_rows, episode_summary, metric_value
from simulator.training.sweep import load_manifest


METRICS = tuple(CATALOG)


def series(path, metric="win_rate", suite=None, axis="elapsed_seconds", runs=None, opponent=None, aggregation="rollout"):
    if metric not in METRICS or axis not in AXES:
        raise ValueError("Unsupported sweep metric or x-axis")
    if aggregation not in ("rollout", "recent"):
        raise ValueError("Unsupported episode aggregation")
    manifest = load_manifest(path)
    trials = []
    warnings = set()
    available = set()
    for trial in manifest["trials"]:
        output = Path(runs) / Path(trial["output"]).name if runs else Path(trial["output"])
        try:
            result = read_run(output)
            rows = result["metrics"]
            warnings.update(result["warnings"])
        except FileNotFoundError:
            rows = []
        if axis == "elapsed_seconds" and any("elapsed_seconds" not in row for row in rows):
            warnings.add("Legacy elapsed time is estimated from recorded iterations; untimed points are omitted.")
        rows = list(elapsed_rows(rows))
        for row in rows:
            available.update(row.get("evaluations", {}))
        trials.append((trial, rows))
    if suite is None:
        suite = "combined_benchmark" if "combined_benchmark" in available else next(iter(sorted(available)), None)
    if metric in ("win_rate", "evaluation_reward") and suite is None:
        warnings.add("No evaluation results yet; curves appear after the first benchmark.")
    curves = []
    for trial, rows in trials:
        points, seen = [], set()
        boundaries = []
        previous_stage = None
        updates = {row["update"]: row for row in rows}
        for row in rows:
            stage = row.get("stage_index", row.get("stage"))
            if previous_stage is not None and stage != previous_stage and row.get(axis) is not None:
                boundaries.append({"x": row[axis], "stage": row.get("stage", str(stage))})
            previous_stage = stage
            anchor = row
            if metric in ("win_rate", "evaluation_reward"):
                evaluation = row.get("evaluations", {}).get(suite)
                if not evaluation:
                    continue
                snapshot = evaluation.get("snapshot_update")
                if snapshot is None or snapshot in seen or snapshot not in updates:
                    continue
                seen.add(snapshot)
                anchor = updates[snapshot]
            value = metric_value(row, metric, suite, opponent, aggregation)
            if value is not None and anchor.get(axis) is not None:
                points.append((anchor[axis], value))
                if CATALOG[metric]["scope"] == "episode" and episode_summary(row, aggregation).get("source") == "recovered_rewards":
                    warnings.add("Training outcomes recovered from reward counts; losses and draws are combined.")
        if not points and rows:
            warnings.add("Some curves have no data for this metric/filter; historical episode returns, lengths and opponent splits cannot be recovered.")
        curves.append({"name": trial["name"], "label": f"{manifest['parameter']}={trial['value']}",
                       "points": points, "boundaries": boundaries})
    return {"name": manifest["name"], "suite": suite, "metric": metric, "axis": axis,
            "opponent": opponent, "aggregation": aggregation, "curves": curves, "warnings": sorted(warnings)}


def plot(path, output=None, metric="win_rate", suite=None, axis="elapsed_seconds", runs=None, opponent=None, aggregation="rollout"):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    report = series(path, metric, suite, axis, runs, opponent, aggregation)
    figure = Figure(figsize=(9, 5.5), dpi=140)
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots()
    for curve in report["curves"]:
        points = curve["points"]
        separated = []
        for point in points:
            if separated and CATALOG[metric]["scope"] != "evaluation" and any(
                    separated[-1][0] < boundary["x"] <= point[0] for boundary in curve["boundaries"]):
                separated.append((point[0], float("nan")))
            separated.append(point)
        line, = axes.plot([point[0] for point in separated], [point[1] for point in separated],
                  marker="o" if metric in ("win_rate", "evaluation_reward") else None,
                  markersize=3, linewidth=1.5, label=curve["label"] + (" (no data yet)" if not points else ""))
        for boundary in curve["boundaries"]:
            axes.axvline(boundary["x"], color=line.get_color(), linestyle=":", alpha=.3, linewidth=.8)
    axes.set_title(report["name"] + (f" / {report['suite']}" if report["suite"] and metric in ("win_rate", "evaluation_reward") else ""))
    if opponent and CATALOG[metric]["scope"] == "episode":
        axes.set_title(f"{report['name']} / {opponent}")
    if CATALOG[metric]["scope"] == "episode":
        axes.set_title(axes.get_title() + (" / This rollout" if aggregation == "rollout" else " / Recent 1,000 episodes"))
    axes.set_xlabel(AXES[axis])
    axes.set_ylabel(f"{CATALOG[metric]['label']} ({CATALOG[metric]['unit']})")
    if CATALOG[metric]["unit"] == "%":
        axes.set_ylim(0, 100)
    axes.grid(alpha=.25)
    handles, labels = axes.get_legend_handles_labels()
    if any(curve["boundaries"] for curve in report["curves"]):
        handles.append(Line2D([], [], color="gray", linestyle=":", linewidth=.8))
        labels.append("Stage change")
    axes.legend(handles, labels, loc="best")
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(rect=(0, .09 if report["warnings"] else 0, 1, 1))
    if report["warnings"]:
        figure.text(.02, .02, " ".join(report["warnings"]), fontsize=8, wrap=True)
    destination = Path(output) if output else Path(path).resolve().parent / "comparison.png"
    if destination.suffix.lower() != ".png":
        raise ValueError("Sweep comparison output must be a PNG")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    canvas.print_png(str(temporary))
    temporary.replace(destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output")
    parser.add_argument("--metric", choices=METRICS, default="win_rate")
    parser.add_argument("--suite", help="Benchmark name; defaults to combined_benchmark, otherwise first available")
    parser.add_argument("--x", choices=AXES, default="elapsed_seconds")
    parser.add_argument("--runs", help="Local run cache root when plotting a copied sweep manifest")
    parser.add_argument("--opponent", help="Opponent ID for completed-episode metrics; defaults to all")
    parser.add_argument("--aggregation", choices=("rollout", "recent"), default="rollout",
                        help="Episode metrics: all completions this rollout (default) or recent 1,000")
    options = parser.parse_args(argv)
    print(plot(options.manifest, options.output, options.metric, options.suite, options.x, options.runs,
               options.opponent, options.aggregation))


if __name__ == "__main__":
    main()
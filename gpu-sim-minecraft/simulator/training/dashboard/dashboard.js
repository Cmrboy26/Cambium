"use strict";

const element = (id) => document.getElementById(id);
const colors = { green: "#18764f", blue: "#347aa2", red: "#b84242", amber: "#b27413", gray: "#6b746f" };
const rewardTypes = [
    ["doing_damage", "Damage dealt", colors.green], ["taking_damage", "Damage taken", colors.red],
    ["won_duel", "Win", colors.blue], ["death", "Death", "#8f364c"],
    ["episode_timeout", "Timeout", colors.amber], ["no_damage_penalty", "No damage", colors.gray],
    ["blocked_with_shield", "Shield block", "#29898c"],
    ["good_aim", "Good aim", "#746ca8"], ["proximity", "Proximity", "#7b8542"],
    ["pitch_control", "Pitch control", "#4f6170"], ["aim_hold", "Aim hold", "#ab7256"],
    ["extreme_pitch_penalty", "Extreme pitch", "#bc596c"],
];
const definitions = [
    ["reward-chart", [["reward_mean", "Mean reward", colors.green]]],
    ["total-reward-chart", [["reward_total", "Total reward", colors.blue]]],
    ["reward-types-chart", rewardTypes.map(([name, label, color]) => [(row) => row.reward_types?.[name], label, color])],
    ["policy-loss-chart", [["policy_loss", "Policy loss", colors.red]]],
    ["loss-chart", [["value_loss", "Value", colors.amber], ["loss", "Total", colors.blue]]],
    ["entropy-chart", [["entropy", "Entropy", colors.blue]]],
    ["throughput-chart", [["player_transitions_per_second", "Transitions / second", colors.green]]],
    ["duels-chart", [["terminated_duels", "Terminated", colors.blue], ["timeouts", "Timeouts", colors.amber]]],
    ["duration-chart", [["collection_seconds", "Collection", colors.gray], ["update_seconds", "Optimization", colors.red],
        ["evaluation_seconds", "Evaluation", colors.amber], ["replay_seconds", "Replay", colors.green]]],
];
const charts = new Map();
let records = [];
let windowSize = 100;
let smoothing = 1;
let requestNumber = 0;
let pending;
let runSignature = "";
let displayedRun = "";
let configuration = null;
let milestoneIndex = null;
let milestoneRun = "";
const milestoneCharts = new Map();
let milestoneChartSignature = "";

function sortRuns(runs) {
    return [...runs].sort((left, right) => right.name.localeCompare(left.name, undefined, { numeric: true, sensitivity: "base" }) || left.id.localeCompare(right.id));
}

function duration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "--";
    const minutes = Math.ceil(seconds / 60);
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    return hours < 24 ? `${hours}h ${minutes % 60}m` : `${Math.floor(hours / 24)}d ${hours % 24}h`;
}

function trainingTime(rows, config) {
    const options = config?.options ?? {};
    const session = config?.training;
    const start = session?.start_update ?? (options.resume ? null : 0);
    const target = session?.target_update ?? (start === 0 && Number.isFinite(options.updates) ? options.updates : null);
    const latest = rows.at(-1)?.update ?? start;
    const current = start === null ? rows : rows.filter((row) => row.update > start);
    const durations = current.map((row) => Number.isFinite(row.iteration_seconds) ? row.iteration_seconds
        : Number.isFinite(row.collection_seconds) && Number.isFinite(row.update_seconds)
            && row.collection_seconds >= 0 && row.update_seconds >= 0
            ? row.collection_seconds + row.update_seconds : null)
        .filter((seconds) => Number.isFinite(seconds) && seconds >= 0);
    const recent = durations.slice(-20);
    const average = recent.length ? recent.reduce((sum, value) => sum + value, 0) / recent.length : null;
    const remaining = rows.at(-1)?.completed_curriculum === true ? 0
        : Number.isFinite(target) && Number.isFinite(latest) ? Math.max(0, target - latest) : null;
    return {
        target, remaining, average,
        elapsed: start !== null && durations.length ? durations.reduce((sum, value) => sum + value, 0) : null,
        eta: remaining === 0 ? 0 : remaining !== null && average !== null ? remaining * average : null,
    };
}

function format(value, digits = 0) {
    return Number.isFinite(value) ? new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value) : "--";
}

function initCharts() {
    if (typeof Chart === "undefined") throw new Error("Chart library could not be loaded.");
    Chart.defaults.font.family = '"Bahnschrift", "Segoe UI", sans-serif';
    Chart.defaults.color = "#65716b";
    for (const [id, series] of definitions) {
        const stacked = id === "reward-types-chart";
        const canvas = element(id);
        const heading = canvas.closest(".chart").querySelector("h2");
        const controls = document.createElement("fieldset");
        controls.className = "segments chart-mode";
        controls.setAttribute("aria-label", `${heading.firstChild.textContent.trim()} view`);
        for (const mode of ["value", "derivative"]) {
            const label = document.createElement("label");
            const input = document.createElement("input");
            input.type = "radio";
            input.name = `${id}-mode`;
            input.value = mode;
            input.checked = mode === "value";
            input.addEventListener("change", redrawCharts);
            const text = document.createElement("span");
            text.textContent = mode === "value" ? "Value" : "Derivative";
            label.append(input, text);
            controls.append(label);
        }
        heading.after(controls);
        charts.set(id, new Chart(element(id), {
            type: stacked ? "bar" : "line",
            data: { datasets: series.map(([, label, color]) => ({
                label, data: [], borderColor: color, backgroundColor: color,
                borderWidth: 1.8, pointRadius: 0, pointHitRadius: 8, pointHoverRadius: 3,
                tension: 0, spanGaps: false,
            })) },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
                normalized: true, interaction: { mode: "index", intersect: false },
                plugins: {
                    legend: { display: series.length > 1, position: "bottom", labels: { boxWidth: 10, boxHeight: 3, font: { size: 10 } } },
                    decimation: { enabled: !stacked, algorithm: "min-max" },
                    tooltip: { callbacks: { title: (items) => `Update ${items[0]?.parsed.x ?? ""}` } },
                },
                scales: {
                    x: { type: "linear", stacked, grid: { display: false }, ticks: { maxTicksLimit: 7, precision: 0, font: { size: 10 } } },
                    y: { stacked, grid: { color: "#edf1ee" }, border: { display: false }, ticks: { maxTicksLimit: 5, font: { size: 10 } } },
                },
            },
        }));
    }
}

function smoothSeries(key, derivative = false) {
    const alpha = 2 / (smoothing + 1);
    let previous = null;
    let previousPoint = null;
    return records.map((row) => {
        const value = typeof key === "function" ? key(row) : row[key];
        if (!Number.isFinite(value)) {
            previous = previousPoint = null;
            return { x: row.update, y: null };
        }
        previous = previous === null ? value : alpha * value + (1 - alpha) * previous;
        const point = { x: row.update, y: previous };
        const slope = previousPoint && point.x > previousPoint.x ? (point.y - previousPoint.y) / (point.x - previousPoint.x) : null;
        previousPoint = point;
        return derivative ? { x: point.x, y: slope } : point;
    });
}

function redrawCharts() {
    for (const [id, series] of definitions) {
        const chart = charts.get(id);
        const derivative = document.querySelector(`input[name="${id}-mode"]:checked`)?.value === "derivative";
        series.forEach(([key], index) => {
            const values = smoothSeries(key, derivative);
            chart.data.datasets[index].data = windowSize ? values.slice(-windowSize) : values;
        });
        chart.options.scales.y.title = { display: derivative, text: "Change / update" };
        chart.update("none");
    }
    const shown = windowSize ? Math.min(windowSize, records.length) : records.length;
    element("window-info").textContent = `${format(shown)} of ${format(records.length)} updates${smoothing > 1 ? ` / EMA ${smoothing}` : ""}`;
}

function renderConfig(config) {
    configuration = config;
    const options = config?.options ?? {};
    const metadata = config?.metadata ?? {};
    const environment = metadata.scenario ?? metadata.environment ?? {};
    const experiment = resolvedExperiment(config);
    const learner = experiment?.learner ?? {};
    const policy = metadata.schema?.id ?? options.schema ?? (experiment?.policy ? "Custom layout" : "--");
    const width = metadata.hidden_dim ?? learner.hidden_dim ?? options.hidden_dim;
    element("run-title").textContent = experiment?.name ?? environment.name ?? "Training overview";
    renderFields("run-identity", [
        ["Seed", experiment?.seed ?? options.seed],
        ["Compute", `${options.device ?? "--"} / ${format(options.envs)} worlds`],
        ["Policy", `${policy}${width != null ? ` / width ${format(width)}` : ""}`],
        ["Rollout", `${format(options.rollout_steps)} steps / world`],
        ["Learning rate", learner.lr ?? options.lr],
        ["PPO epochs", learner.epochs ?? options.epochs],
        ["Minibatch", format(learner.minibatch_size ?? options.minibatch_size)],
        ["Initialization", options.resume ? "Resumed checkpoint" : options.initialize ? "Transferred weights" : config ? "Fresh weights" : "--"],
    ]);
    element("arena").scenario = environment;
    element("arena").runId = element("run").value;
    document.dispatchEvent(new CustomEvent("arena-config", { detail: { scenario: environment, runId: element("run").value } }));
    element("environment-name").textContent = config?.experiment ? `Deployment snapshot — ${environment.name ?? "Environment"}` : environment.name ?? "Environment";
    renderFields("environment", [
        ["Environment source", options.environment ?? "Checkpoint / default"],
        ["Max ticks / round", metadata.max_ticks ?? options.max_ticks ?? environment.max_ticks],
        ["Arena size", environment.bounds?.size?.join(" x ")],
        ["Policy", metadata.schema?.id ?? options.schema],
    ]);
    const fields = [
        ["Device", options.device], ["Worlds", options.envs], ["Seed", config?.experiment?.seed ?? options.seed],
        ["Learner", metadata.learner ?? (config?.plan ? "Historical research plan" : "Legacy self-play")],
        ["Learning rate", config?.experiment?.learner?.lr ?? options.lr], ["Rollout steps", options.rollout_steps],
        ["Epochs / update", config?.experiment?.learner?.epochs ?? options.epochs],
        ["Minibatch size", config?.experiment?.learner?.minibatch_size ?? options.minibatch_size],
        ["Max ticks / round", metadata.max_ticks ?? options.max_ticks],
        ["Hidden dimension", metadata.hidden_dim ?? options.hidden_dim],
        ["Policy", metadata.schema?.id ?? options.schema],
        ["Environment", options.environment ?? "Checkpoint / default"],
        ["Resume checkpoint", options.resume ?? "None"],
        ["Initial weights", options.initialize ?? "None"],
        ["Replay interval", options.replay_every === 0 ? "Off" : options.replay_every],
    ];
    for (const [index, player] of (environment.players ?? []).entries()) {
        const loadout = player.loadout ?? {};
        fields.push([`Player ${index + 1} weapons`, loadout.weapons?.join(", ")],
            [`Player ${index + 1} armor`, Object.entries(loadout.armor ?? {}).map(([slot, item]) => `${slot}: ${item}`).join(", ") || "None"],
            [`Player ${index + 1} offhand`, loadout.offhand ?? "None"]);
    }
    for (const [name, weight] of Object.entries(metadata.reward_weights ?? environment.reward_weights ?? {})) {
        fields.push([`Reward: ${name}`, weight]);
    }
    renderFields("config", fields);
    element("experiment-details").hidden = !experiment;
    element("experiment-json").textContent = experiment ? JSON.stringify(experiment, null, 2) : "";
}

function resolvedExperiment(config) {
    const experiment = config?.experiment ?? config?.plan ?? config?.metadata?.plan;
    return experiment && typeof experiment === "object" && !Array.isArray(experiment) ? experiment : null;
}

function fixedControls(controls) {
    if (controls == null) return "Not recorded";
    return Object.entries(controls?.fixed ?? controls ?? {}).map(([name, value]) => `${name}=${JSON.stringify(value)}`).join(", ") || "None (all heads learned)";
}

function stageFor(row, experiment) {
    const stages = experiment?.stages ?? [];
    return Number.isInteger(row?.stage_index) ? stages[row.stage_index] : stages.find((stage) => stage.name === row?.stage);
}

function stageName(row, experiment) {
    return row?.stage ?? stageFor(row, experiment)?.name ?? (Number.isInteger(row?.stage_index) ? `Stage ${row.stage_index + 1}` : "--");
}

function progressText(progress) {
    return progress ? ["updates", "episodes", "transitions"].map((key) => `${format(progress[key])} ${key}`).join(" / ") : "Not recorded";
}

function budgetText(advance, progress = {}) {
    if (!advance || typeof advance !== "object") return "Not recorded";
    for (const combination of ["any", "all"]) {
        if (Array.isArray(advance[combination])) return `${combination.toUpperCase()} (${advance[combination].map((item) => budgetText(item, progress)).join("; ")})`;
    }
    return Object.entries(advance).map(([key, value]) => {
        if (["updates", "episodes", "transitions"].includes(key)) return `${key}: ${format(progress[key])} / ${format(value)}`;
        if (key === "evaluation") return `${value.suite}: ${value.metric} ≥ ${format(value.at_least, 4)}`;
        return `${key}: ${JSON.stringify(value)}`;
    }).join("; ") || "Not recorded";
}

function stageConditions(stage) {
    const arenas = stage?.scenario?.arenas ?? [];
    const opponents = Array.isArray(stage?.opponents) ? stage.opponents : [];
    const total = opponents.reduce((sum, opponent) => sum + opponent.weight, 0);
    return [
        ["Environment", stage?.environment_name ?? arenas[0]?.environment?.name],
        ["Arena mixture", arenas.map((arena, index) => `${stage?.arena_names?.[index] ?? arena.environment?.name ?? `Arena ${index + 1}`} (weight ${format(arena.weight, 3)})`).join(", ") || stage?.arena_names?.join(", ")],
        ["Opponent mix", opponents.map((opponent) => `${opponent.id} (${total > 0 ? `${format(opponent.weight / total * 100, 1)}%` : `weight ${opponent.weight}`})`).join(", ") || "Not recorded"],
        ["Fixed controls", stage ? fixedControls(stage.controls) : "Not recorded"],
        ["Episode limit", arenas[0]?.environment?.max_ticks],
    ];
}

function advancementText(advance) {
    if (!advance || typeof advance !== "object") return "Not recorded";
    for (const combination of ["any", "all"]) {
        if (Array.isArray(advance[combination])) {
            return `${combination === "any" ? "Any" : "All"} of (${advance[combination].map(advancementText).join("; ")})`;
        }
    }
    return Object.entries(advance).map(([key, value]) => key === "evaluation"
        ? `${value.suite}: ${value.metric} ≥ ${format(value.at_least, 4)}`
        : `${format(value)} ${key}`).join("; ") || "Not recorded";
}

function advancementMetrics(advance) {
    if (!advance || typeof advance !== "object") return [];
    for (const combination of ["any", "all"]) {
        if (Array.isArray(advance[combination])) return advance[combination].flatMap(advancementMetrics);
    }
    const evaluation = advance.evaluation;
    if (evaluation && Number.isFinite(evaluation.at_least)) {
        return [{suite: evaluation.suite, metric: evaluation.metric, threshold: evaluation.at_least}];
    }
    return ["updates", "episodes", "transitions"].filter((metric) => Number.isFinite(advance[metric]))
        .map((metric) => ({metric, threshold: advance[metric]}));
}

function milestoneMetricHistory(stage, experiment, rows = records) {
    const stageRows = rows.filter((row) => stageFor(row, experiment) === stage);
    const snapshots = new Map(stageRows.map((row) => [row.update, row]));
    const localUpdates = stageRows.every((row) => Number.isFinite(row.stage_progress?.updates));
    const coordinate = (row) => localUpdates ? row.stage_progress.updates : row.update;
    const end = stageRows.reduce((maximum, row) => Math.max(maximum, coordinate(row)), 1);
    const start = localUpdates ? 0 : stageRows[0]?.update ?? 0;
    const conditions = new Map(advancementMetrics(stage?.advance).map((condition) => [JSON.stringify(condition), condition]));
    return [...conditions].map(([key, condition]) => {
        const data = new Map();
        const suite = stage.evaluations?.[condition.suite] ?? experiment?.evaluations?.[condition.suite];
        let previousLegacy = null;
        for (const row of stageRows) {
            if (!condition.suite) {
                const value = row.stage_progress?.[condition.metric];
                if (Number.isFinite(value)) data.set(row.update, {x: coordinate(row), y: value, update: row.update});
                continue;
            }
            const result = row.evaluations?.[condition.suite];
            if (!result || !Number.isFinite(result[condition.metric])) continue;
            if ((result.stage !== undefined || result.stage_index !== undefined) && stageFor(result, experiment) !== stage) continue;
            let snapshot = result.snapshot_update;
            if (!Number.isInteger(snapshot)) {
                const counter = stage.evaluations?.[condition.suite] ? row.stage_progress?.updates : row.update;
                if (Number.isInteger(suite?.every_updates) && Number.isFinite(counter)) {
                    if (counter % suite.every_updates !== 0) continue;
                } else {
                    const signature = JSON.stringify(result);
                    if (signature === previousLegacy) continue;
                    previousLegacy = signature;
                }
                snapshot = row.update;
            }
            const measured = snapshots.get(snapshot);
            if (measured) data.set(snapshot, {x: coordinate(measured), y: result[condition.metric], update: snapshot});
        }
        return {...condition, key, data: [...data.values()].sort((left, right) => left.x - right.x),
            thresholdData: [{x: start, y: condition.threshold}, {x: Math.max(start + 1, end), y: condition.threshold}],
            xLabel: localUpdates ? "Stage update" : "PPO update"};
    });
}

function renderMilestoneMetricCharts(stage, experiment) {
    const histories = stage ? milestoneMetricHistory(stage, experiment).filter((history) => history.metric !== "updates") : [];
    element("milestone-metrics").hidden = !histories.length;
    const signature = JSON.stringify([element("run").value, experiment?.name, stage?.name, histories.map((history) => history.key)]);
    if (signature !== milestoneChartSignature) {
        for (const entry of milestoneCharts.values()) entry.chart?.destroy();
        milestoneCharts.clear();
        milestoneChartSignature = signature;
        const fragment = document.createDocumentFragment();
        for (const [index, history] of histories.entries()) {
            const labels = {reward_mean: "Mean episode return", win_rate: "Win rate", updates: "Completed updates",
                episodes: "Completed episodes", transitions: "Learner transitions"};
            const label = labels[history.metric] ?? history.metric;
            const metricFormat = (value) => history.metric === "win_rate" ? `${format(value * 100, 2)}%` : format(value, 4);
            const section = document.createElement("section");
            section.className = "milestone-condition";
            const heading = document.createElement("div");
            heading.className = "milestone-condition-heading";
            const title = document.createElement("h5");
            title.id = `milestone-condition-${index}-title`;
            title.textContent = label;
            section.setAttribute("aria-labelledby", title.id);
            const value = document.createElement("strong");
            value.className = "milestone-condition-value";
            value.setAttribute("aria-live", "polite");
            heading.append(title, value);
            const source = document.createElement("p");
            source.className = "milestone-condition-source";
            source.textContent = history.suite ? `${history.suite} (${history.metric})` : "Stage progress";
            const plot = document.createElement("div");
            plot.className = "milestone-condition-plot";
            const canvas = document.createElement("canvas");
            canvas.setAttribute("role", "img");
            canvas.setAttribute("aria-label", `${history.suite ? `${history.suite}: ` : ""}${label} by update, threshold ${metricFormat(history.threshold)}`);
            plot.append(canvas);
            section.append(heading, source, plot);
            fragment.append(section);
            milestoneCharts.set(history.key, {canvas, value, label, metricFormat, chart: null});
        }
        element("milestone-metric-charts").replaceChildren(fragment);
    }
    for (const history of histories) {
        const entry = milestoneCharts.get(history.key);
        const latest = history.data.at(-1);
        entry.value.textContent = latest ? `${entry.metricFormat(latest.y)} / ${entry.metricFormat(history.threshold)}`
            : history.suite ? "Awaiting evaluation" : "Awaiting rollout";
        entry.value.dataset.met = latest ? String(latest.y >= history.threshold) : "unknown";
        if (!entry.chart && typeof Chart !== "undefined") {
            entry.chart = new Chart(entry.canvas, {
                type: "line",
                data: {datasets: [
                    {label: entry.label, data: [], borderColor: colors.green, backgroundColor: colors.green,
                        borderWidth: 2, pointRadius: history.suite ? 3 : 0, pointHoverRadius: 5,
                        pointHitRadius: 8, tension: 0, spanGaps: false},
                    {label: `Threshold ≥ ${entry.metricFormat(history.threshold)}`, data: [], borderColor: colors.amber,
                        backgroundColor: colors.amber, borderWidth: 1.5, borderDash: [6, 4], pointRadius: 0},
                ]},
                options: {
                    responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
                    interaction: {mode: "nearest", intersect: false},
                    plugins: {
                        legend: {position: "bottom", labels: {boxWidth: 12, boxHeight: 3, font: {size: 10}}},
                        tooltip: {callbacks: {label: (item) => `${item.dataset.label}: ${entry.metricFormat(item.parsed.y)}`}},
                    },
                    scales: {
                        x: {type: "linear", grid: {display: false}, title: {display: true, text: history.xLabel},
                            ticks: {maxTicksLimit: 7, precision: 0, font: {size: 10}}},
                        y: {suggestedMin: history.metric === "win_rate" ? 0 : undefined,
                            suggestedMax: history.metric === "win_rate" ? 1 : undefined,
                            grid: {color: "#e3e9e5"}, border: {display: false},
                            ticks: {maxTicksLimit: 5, font: {size: 10}, callback: entry.metricFormat}},
                    },
                },
            });
        }
        if (entry.chart) {
            entry.chart.data.datasets[0].data = history.data;
            entry.chart.data.datasets[1].data = history.thresholdData;
            entry.chart.options.scales.x.title.text = history.xLabel;
            entry.chart.options.scales.x.min = history.thresholdData[0].x;
            entry.chart.options.scales.x.max = history.thresholdData[1].x;
            entry.chart.options.plugins.tooltip.callbacks.title = (items) => `${history.xLabel} ${format(items[0]?.parsed.x)}`;
            entry.chart.update("none");
        }
    }
}

function rewardWeight(value) {
    const magnitude = Math.abs(value);
    const number = magnitude !== 0 && (magnitude < 0.0001 || magnitude >= 1e6)
        ? magnitude.toExponential() : format(magnitude, 8);
    return `${value > 0 ? "+" : value < 0 ? "-" : ""}${number}`;
}

function renderMilestoneRewards(stage) {
    const isMap = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
    const sources = isMap(stage?.rewards?.weights) ? [{name: "Saved stage weights", weights: stage.rewards.weights}]
        : isMap(stage?.task_override?.rewards) ? [{name: "Task override", weights: stage.task_override.rewards}]
        : (stage?.scenario?.arenas ?? []).flatMap((arena, index) => isMap(arena.environment?.reward_weights)
            ? [{name: stage.arena_names?.[index] ?? arena.environment.name ?? `Arena ${index + 1}`, weights: arena.environment.reward_weights}] : []);
    const entries = sources.flatMap((source) => Object.entries(source.weights).filter(([, value]) => Number.isFinite(value)));
    const active = entries.filter(([, value]) => value !== 0).length;
    element("milestone-reward-source").textContent = sources[0]?.name === "Saved stage weights" || sources[0]?.name === "Task override"
        ? sources[0].name : sources.length ? "Saved arena weights" : "Reward weights unavailable";
    element("milestone-reward-count").textContent = `${format(active)} active / ${format(entries.length)} recorded`;
    element("milestone-reward-count").hidden = !sources.length;
    element("milestone-reward-empty").textContent = !sources.length ? "Not recorded for this stage"
        : !entries.length ? "No reward terms configured" : "All recorded reward weights are zero";
    element("milestone-reward-empty").hidden = active > 0;
    element("milestone-reward-zero").hidden = entries.length === active;
    element("milestone-reward-zero-title").textContent = `Zero-weight terms (${format(entries.length - active)})`;
    for (const [id, inactive] of [["milestone-rewards", false], ["milestone-rewards-inactive", true]]) {
        const fragment = document.createDocumentFragment();
        for (const source of sources) {
            const weights = Object.entries(source.weights).filter(([, value]) => Number.isFinite(value));
            const selected = weights.filter(([, value]) => (value === 0) === inactive)
                .sort(([left, leftValue], [right, rightValue]) => Math.abs(rightValue) - Math.abs(leftValue) || left.localeCompare(right));
            if (sources.length > 1 && selected.length) {
                const heading = document.createElement("h5");
                heading.textContent = source.name;
                fragment.append(heading);
            }
            const maximum = Math.max(0, ...weights.map(([, value]) => Math.abs(value)));
            for (const [name, value] of selected) {
                const definition = rewardTypes.find(([key]) => key === name);
                const row = document.createElement("div");
                row.className = "reward-row";
                row.dataset.sign = value > 0 ? "positive" : value < 0 ? "negative" : "zero";
                const label = document.createElement("span");
                label.className = "reward-name";
                label.title = name;
                const swatch = document.createElement("span");
                swatch.className = "reward-swatch";
                swatch.style.backgroundColor = definition?.[2] ?? colors.gray;
                swatch.setAttribute("aria-hidden", "true");
                const text = document.createElement("span");
                text.textContent = definition?.[1] ?? name.replaceAll("_", " ");
                label.append(swatch, text);
                const weight = document.createElement("strong");
                weight.textContent = rewardWeight(value);
                const track = document.createElement("span");
                track.className = "reward-track";
                track.setAttribute("aria-hidden", "true");
                track.title = "Magnitude relative to the largest absolute weight in this reward map";
                const fill = document.createElement("span");
                fill.style.width = `${maximum > 0 ? Math.abs(value) / maximum * 100 : 0}%`;
                track.append(fill);
                row.append(label, weight, track);
                fragment.append(row);
            }
        }
        element(id).replaceChildren(fragment);
    }
}

function renderMilestoneOpponents(opponents) {
    if (!opponents.length) {
        renderFields("milestone-opponents", [["Opponent mix", "Not recorded"]]);
        return;
    }
    const total = opponents.reduce((sum, opponent) => sum + opponent.weight, 0);
    const fragment = document.createDocumentFragment();
    for (const [index, opponent] of opponents.entries()) {
        const row = document.createElement("div");
        row.style.setProperty("--mixture-color", [colors.blue, "#29898c", colors.amber, colors.red][index % 4]);
        const label = document.createElement("dt");
        label.textContent = opponent.id;
        const value = document.createElement("dd");
        value.textContent = total > 0 ? `${format(opponent.weight / total * 100, 1)}%` : `Weight ${format(opponent.weight)}`;
        const track = document.createElement("span");
        track.className = "mixture-track";
        track.setAttribute("aria-hidden", "true");
        const fill = document.createElement("span");
        fill.style.width = `${total > 0 ? opponent.weight / total * 100 : 0}%`;
        track.append(fill);
        row.append(label, value, track);
        fragment.append(row);
    }
    element("milestone-opponents").replaceChildren(fragment);
}

function renderMilestones() {
    const experiment = resolvedExperiment(configuration);
    const stages = experiment?.stages ?? [];
    const latest = records.at(-1);
    const signature = JSON.stringify([element("run").value, experiment?.name, stages.map((stage) => stage.name)]);
    if (signature !== milestoneRun) {
        milestoneIndex = null;
        milestoneRun = signature;
    }
    element("milestone-browser").hidden = !stages.length;
    element("legacy-stage-status").hidden = stages.length > 0;
    element("run-context").hidden = stages.length > 0;
    if (!stages.length) {
        renderMilestoneMetricCharts(null, experiment);
        return;
    }

    const complete = latest?.completed_curriculum === true;
    const recordedIndex = stages.indexOf(stageFor(latest, experiment));
    const nextIndex = Number.isInteger(latest?.next_stage_index) ? latest.next_stage_index
        : stages.findIndex((stage) => stage.name === latest?.curriculum_boundary?.next_stage);
    const currentIndex = !latest ? 0 : !complete && nextIndex >= 0 && nextIndex < stages.length ? nextIndex : recordedIndex;
    const selectedIndex = milestoneIndex ?? Math.max(0, currentIndex);
    const selected = stages[selectedIndex];
    const status = (index) => complete || currentIndex >= 0 && index < currentIndex ? "Completed"
        : index === currentIndex ? latest ? "Current stage" : "Initial stage"
        : currentIndex >= 0 ? "Upcoming" : "Status not recorded";
    const select = (index) => {
        if (index < 0 || index >= stages.length) return;
        milestoneIndex = index === currentIndex ? null : index;
        renderMilestones();
    };
    element("milestone-current").textContent = complete ? "Curriculum complete"
        : currentIndex >= 0 ? `${latest ? "Current stage" : "Initial stage"}: ${stages[currentIndex].name}` : "Current stage not recorded";
    element("milestone-position").textContent = `Stage ${selectedIndex + 1} of ${stages.length}`;
    element("milestone-name").textContent = selected.name;
    element("milestone-state").textContent = status(selectedIndex);
    element("milestone-state").dataset.current = String(selectedIndex === currentIndex && !complete);
    element("milestone-previous").disabled = selectedIndex === 0;
    element("milestone-next").disabled = selectedIndex === stages.length - 1;
    element("milestone-previous").onclick = () => select(selectedIndex - 1);
    element("milestone-next").onclick = () => select(selectedIndex + 1);
    element("milestone-return").hidden = currentIndex < 0 || selectedIndex === currentIndex;
    element("milestone-return").onclick = () => select(currentIndex);
    const strip = element("milestone-strip");
    const refocus = document.activeElement && strip.contains(document.activeElement);
    strip.onkeydown = (event) => {
        const index = {ArrowLeft: selectedIndex - 1, ArrowRight: selectedIndex + 1, Home: 0, End: stages.length - 1}[event.key];
        if (!Number.isInteger(index)) return;
        event.preventDefault();
        select(index);
    };
    const fragment = document.createDocumentFragment();
    for (const index of [selectedIndex - 1, selectedIndex, selectedIndex + 1]) {
        const item = document.createElement(index >= 0 && index < stages.length ? "button" : "div");
        item.className = "milestone";
        if (index >= 0 && index < stages.length) {
            item.type = "button";
            item.setAttribute("aria-pressed", String(index === selectedIndex));
            item.setAttribute("aria-controls", "milestone-details");
            if (index === currentIndex && !complete) item.setAttribute("aria-current", "step");
            const order = document.createElement("span");
            order.textContent = `Stage ${index + 1}`;
            const name = document.createElement("strong");
            name.textContent = stages[index].name;
            const state = document.createElement("span");
            state.textContent = status(index);
            item.dataset.status = status(index);
            item.append(order, name, state);
            item.onclick = () => select(index);
        } else {
            item.className = "milestone milestone-placeholder";
            item.setAttribute("aria-hidden", "true");
        }
        fragment.append(item);
    }
    strip.replaceChildren(fragment);
    if (refocus) strip.querySelector('[aria-pressed="true"]')?.focus({preventScroll: true});

    const recorded = records.filter((row) => stageFor(row, experiment) === selected).at(-1);
    element("milestone-observation").textContent = recorded ? `Latest stage rollout: update ${format(recorded.update)}` : "No rollout recorded for this stage";
    renderFields("milestone-progress", [
        ["Advance when", advancementText(selected.advance)],
        ["Recorded progress", progressText(recorded?.stage_progress)],
    ]);
    const budget = Object.entries(selected.advance ?? {});
    const [unit, limit] = budget[0] ?? [];
    const value = recorded?.stage_progress?.[unit];
    const progress = element("milestone-progress-bar");
    progress.hidden = budget.length !== 1 || !["updates", "episodes", "transitions"].includes(unit)
        || !Number.isFinite(limit) || limit <= 0 || !Number.isFinite(value);
    if (!progress.hidden) {
        progress.max = limit;
        progress.value = Math.min(limit, value);
        progress.setAttribute("aria-label", `${format(value)} of ${format(limit)} ${unit}`);
    }
    renderMilestoneMetricCharts(selected, experiment);
    renderFields("milestone-environment", stageConditions(selected).filter(([label]) => ["Environment", "Arena mixture", "Episode limit"].includes(label)));
    renderMilestoneOpponents(selected.opponents ?? []);
    const controls = selected.controls?.fixed ?? selected.controls;
    element("milestone-controls").dataset.mode = controls == null ? "unknown" : Object.keys(controls).length ? "fixed" : "learned";
    renderFields("milestone-controls", controls == null ? [["Fixed controls", "Not recorded"]]
        : Object.keys(controls).length ? Object.entries(controls).map(([name, value]) => [name, String(value)])
        : [["Action heads", "All learned"]]);
    renderMilestoneRewards(selected);
}

function renderTable(id, rows, columns, empty) {
    const fragment = document.createDocumentFragment();
    for (const values of rows.length ? rows : [[empty]]) {
        const row = document.createElement("tr");
        for (const value of values) {
            const cell = document.createElement("td");
            cell.textContent = value ?? "--";
            if (!rows.length) cell.colSpan = columns;
            row.append(cell);
        }
        fragment.append(row);
    }
    element(id).replaceChildren(fragment);
}

function renderCurriculum() {
    const experiment = resolvedExperiment(configuration);
    const latest = records.at(-1);
    renderMilestones();
    const hasResearchMetrics = records.some((row) => row.stage !== undefined || row.stage_index !== undefined
        || row.completed_curriculum !== undefined || row.evaluations !== undefined);
    element("curriculum").hidden = !experiment && !hasResearchMetrics;
    element("nav-curriculum").hidden = !experiment && !hasResearchMetrics;
    const stage = stageFor(latest, experiment);
    const initial = !latest ? experiment?.stages?.[0] : null;
    const boundary = latest?.curriculum_boundary;
    const nextIndex = Number.isInteger(latest?.next_stage_index) && latest.next_stage_index !== latest.stage_index ? latest.next_stage_index : null;
    const upcoming = nextIndex !== null ? experiment?.stages?.[nextIndex]
        : experiment?.stages?.find((item) => item.name === boundary?.next_stage);
    const upcomingName = upcoming?.name ?? boundary?.next_stage ?? (nextIndex !== null ? `Stage ${nextIndex + 1}` : null);
    const complete = latest?.completed_curriculum === true;
    element("curriculum-title").textContent = experiment?.name ?? "Experiment stages";
    element("curriculum-status").textContent = complete ? "Curriculum complete"
        : upcomingName ? `Upcoming stage: ${upcomingName} (latest rollout is from ${stageName(latest, experiment)})`
        : latest?.stage !== undefined || latest?.stage_index !== undefined ? `Active stage (latest rollout): ${stageName(latest, experiment)}`
        : initial ? `Initial stage: ${initial.name} (waiting for updates)` : "Stage status not recorded";
    element("run-stage-status").textContent = element("curriculum-status").textContent;
    element("run-budget").textContent = stage || initial ? budgetText((stage ?? initial).advance, latest?.stage_progress) : "";
    const metadata = configuration?.metadata ?? {};
    const environment = metadata.scenario ?? metadata.environment ?? {};
    const conditions = stage || initial ? stageConditions(stage ?? initial).filter(([label]) => label !== "Arena mixture") : [
        ["Environment", environment.name], ["Opponent mix", "Not recorded"],
        ["Fixed controls", fixedControls(metadata.controls)],
        ["Episode limit", metadata.max_ticks ?? configuration?.options?.max_ticks ?? environment.max_ticks],
    ];
    renderFields("run-context", conditions);
    renderFields("stage-summary", [
        ["Latest rollout stage", latest ? stageName(latest, experiment) : "No rollout yet"],
        ["Budget progress (latest rollout)", progressText(latest?.stage_progress)],
        ["Advance budget (latest rollout stage)", budgetText((stage ?? initial)?.advance, latest?.stage_progress)],
        ["Curriculum completion", complete ? "Complete (explicit metric)" : latest?.completed_curriculum === false ? "Not complete" : "Not recorded"],
        ["Upcoming stage", complete ? "None — curriculum complete" : upcomingName ?? "No transition recorded"],
        ["Boundary reset worlds", boundary?.reset_worlds],
    ]);
    element("stage-context-title").textContent = initial ? `Initial stage conditions — ${initial.name}` : `Latest rollout conditions — ${stageName(latest, experiment)}`;
    renderFields("stage-context", stageConditions(stage ?? initial));
    element("upcoming-stage").hidden = !upcomingName || complete;
    renderFields("upcoming-context", stageConditions(upcoming));

    const timeline = [];
    for (const row of records) {
        if (row.stage === undefined && row.stage_index === undefined && !row.curriculum_boundary && row.completed_curriculum === undefined) continue;
        const key = `${row.stage_index ?? ""}:${stageName(row, experiment)}`;
        const previous = timeline.at(-1);
        if (previous && previous.key === key && !previous.end.curriculum_boundary && !previous.end.completed_curriculum) previous.end = row;
        else timeline.push({key, start: row.update, end: row});
    }
    renderTable("stage-timeline", timeline.map(({start, end}) => [
        start === end.update ? format(start) : `${format(start)}–${format(end.update)}`,
        stageName(end, experiment), progressText(end.stage_progress),
        end.completed_curriculum ? "Curriculum complete" : end.curriculum_boundary
            ? `${format(end.curriculum_boundary.reset_worlds)} worlds reset → ${end.curriculum_boundary.next_stage ?? "not recorded"}` : "--",
    ]), 4, "No stage metrics recorded.");
    const suiteNames = [...Object.keys(experiment?.evaluations ?? {}),
        ...(experiment?.stages ?? []).flatMap((stage) => Object.keys(stage.evaluations ?? {}))];
    const evaluations = new Map(suiteNames.map((suite) => [suite, null]));
    for (const row of records) for (const [suite, result] of Object.entries(row.evaluations ?? {})) evaluations.set(suite, {row, result});
    renderTable("evaluations", [...evaluations].map(([suite, entry]) => [
        suite, format(entry?.result.snapshot_update ?? entry?.row.update),
        entry ? stageName(entry.result.stage ? entry.result : entry.row, experiment) : "--",
        entry ? `${format(entry.result.win_rate * 100, 2)}%` : "--", format(entry?.result.reward_mean, 4),
        format(entry?.result.episodes), format(entry?.result.worlds), format(entry?.result.seconds, 3),
    ]), 8, "No evaluation results recorded.");
}

function renderFields(id, fields) {
    const fragment = document.createDocumentFragment();
    for (const [label, value] of fields) {
        const group = document.createElement("div");
        const term = document.createElement("dt");
        term.textContent = label;
        const description = document.createElement("dd");
        description.textContent = value ?? "--";
        group.append(term, description);
        fragment.append(group);
    }
    element(id).replaceChildren(fragment);
}

function renderRecords() {
    const latest = records.at(-1) ?? {};
    renderCurriculum();
    element("reward-types-empty").hidden = records.some((row) => row.reward_types && Object.keys(row.reward_types).length > 0);
    element("stat-update").textContent = format(latest.update);
    element("stat-total").textContent = `${format(records.length)} updates recorded`;
    element("stat-samples").textContent = format(latest.samples);
    element("stat-reward").textContent = format(latest.reward_mean, 4);
    element("stat-throughput").textContent = format(latest.player_transitions_per_second);
    element("stat-rounds").textContent = format(latest.completed_rounds);
    const timeouts = records.filter((row) => Number.isFinite(row.timeouts));
    element("stat-timeouts").textContent = timeouts.length ? format(timeouts.reduce((sum, row) => sum + row.timeouts, 0)) : "--";
    const timing = trainingTime(records, configuration);
    element("stat-target").textContent = timing.target === null ? "Target unavailable" : `Target: ${format(timing.target)} updates`;
    element("stat-remaining").textContent = duration(timing.eta);
    element("stat-remaining-detail").textContent = latest.completed_curriculum === true ? "Curriculum complete"
        : timing.remaining === null ? "Session target unavailable" : `${format(timing.remaining)} updates remaining`;
    element("stat-elapsed").textContent = duration(timing.elapsed);
    element("stat-average").textContent = timing.average === null ? "--" : `${format(timing.average, 2)}s`;
    element("row-count").textContent = `${format(records.length)} recorded`;
    const fragment = document.createDocumentFragment();
    for (const row of records.slice(-12).reverse()) {
        const tr = document.createElement("tr");
        for (const [key, digits] of [["update", 0], ["samples", 0], ["reward_mean", 4], ["policy_loss", 4], ["value_loss", 4], ["entropy", 3], ["completed_rounds", 0], ["timeouts", 0]]) {
            const td = document.createElement("td");
            td.textContent = format(row[key], digits);
            tr.append(td);
        }
        fragment.append(tr);
    }
    if (!records.length) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 8;
        td.textContent = "No updates recorded.";
        tr.append(td);
        fragment.append(tr);
    }
    element("updates").replaceChildren(fragment);
    redrawCharts();
}

function notice(message) {
    element("notice").textContent = message;
    element("notice").hidden = !message;
}

async function getJson(path, signal) {
    const response = await fetch(path, { signal, cache: "no-store" });
    if (!response.ok) throw new Error(`Request failed (${response.status}).`);
    return response.json();
}

async function refresh() {
    const ticket = ++requestNumber;
    pending?.abort();
    const controller = new AbortController();
    pending = controller;
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
        const dataRuns = await getJson("/api/runs", controller.signal);
        const runs = sortRuns(dataRuns.runs);
        if (ticket !== requestNumber) return;
        const select = element("run");
        const selected = select.value;
        const signature = JSON.stringify(runs.map((run) => [run.id, run.name]));
        if (signature !== runSignature) {
            const options = runs.map((run) => new Option(run.name, run.id));
            select.replaceChildren(...(options.length ? options : [new Option("No runs found", "")]));
            select.value = runs.some((run) => run.id === selected) ? selected : runs[0]?.id ?? "";
            runSignature = signature;
        }
        select.disabled = !runs.length;
        const identity = select.value;
        if (displayedRun !== identity) {
            records = [];
            renderConfig(null);
            renderRecords();
            displayedRun = identity;
        }
        if (identity) {
            const data = await getJson(`/api/runs/${encodeURIComponent(identity)}/metrics`, controller.signal);
            if (ticket !== requestNumber || identity !== select.value) return;
            records = data.metrics;
            element("run-path").textContent = data.path;
            element("empty").hidden = records.length > 0;
            element("empty").textContent = "Waiting for the first PPO update.";
            renderConfig(data.config);
            renderRecords();
            notice(data.warnings.join(" "));
        } else {
            element("run-path").textContent = "";
            element("empty").textContent = "No training runs found.";
            element("empty").hidden = false;
            records = [];
            renderConfig(null);
            renderRecords();
            notice("");
        }
        element("connection").textContent = element("live").checked ? "Connected" : "Paused";
        element("connection").dataset.state = "connected";
        element("checked").textContent = new Date().toLocaleTimeString();
    } catch (error) {
        if (ticket !== requestNumber) return;
        element("connection").textContent = "Disconnected";
        element("connection").dataset.state = "error";
        notice(`${error.name === "AbortError" ? "Request timed out." : error.message} Displayed data may be out of date.`);
    } finally {
        clearTimeout(timeout);
    }
}

try {
    initCharts();
    element("run").addEventListener("change", refresh);
    element("refresh").addEventListener("click", refresh);
    element("live").addEventListener("change", () => {
        if (element("live").checked) refresh();
        else element("connection").textContent = "Paused";
    });
    document.querySelectorAll('input[name="window"]').forEach((input) => {
        input.addEventListener("change", () => { windowSize = Number(input.value); redrawCharts(); });
    });
    element("smooth").addEventListener("change", (event) => {
        smoothing = Number(event.target.value);
        redrawCharts();
    });
    refresh();
    setInterval(() => { if (element("live").checked && !document.hidden) refresh(); }, 3000);
    document.addEventListener("visibilitychange", () => { if (!document.hidden && element("live").checked) refresh(); });
} catch (error) {
    element("connection").textContent = "Unavailable";
    element("connection").dataset.state = "error";
    notice(error.message);
}
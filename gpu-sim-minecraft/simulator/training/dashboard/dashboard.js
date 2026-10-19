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
    ["duration-chart", [["collection_seconds", "Collection", colors.gray], ["update_seconds", "Optimization", colors.red]]],
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
    const durations = current.filter((row) => Number.isFinite(row.collection_seconds) && row.collection_seconds >= 0
        && Number.isFinite(row.update_seconds) && row.update_seconds >= 0)
        .map((row) => row.collection_seconds + row.update_seconds);
    const recent = durations.slice(-20);
    const average = recent.length ? recent.reduce((sum, value) => sum + value, 0) / recent.length : null;
    const remaining = Number.isFinite(target) && Number.isFinite(latest) ? Math.max(0, target - latest) : null;
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
    element("arena").scenario = environment;
    element("arena").runId = element("run").value;
    document.dispatchEvent(new CustomEvent("arena-config", { detail: { scenario: environment, runId: element("run").value } }));
    element("environment-name").textContent = environment.name ?? "Environment";
    renderFields("environment", [
        ["Environment source", options.environment ?? "Checkpoint / default"],
        ["Max ticks / round", metadata.max_ticks ?? options.max_ticks ?? environment.max_ticks],
        ["Arena size", environment.bounds?.size?.join(" x ")],
        ["Policy", metadata.schema?.id ?? options.schema],
    ]);
    const fields = [
        ["Device", options.device], ["Worlds", options.envs], ["Seed", options.seed],
        ["Learning rate", options.lr], ["Rollout steps", options.rollout_steps],
        ["Epochs / update", options.epochs], ["Minibatch size", options.minibatch_size],
        ["Max ticks / round", metadata.max_ticks ?? options.max_ticks],
        ["Hidden dimension", metadata.hidden_dim ?? options.hidden_dim],
        ["Policy", metadata.schema?.id ?? options.schema],
        ["Environment", options.environment ?? "Checkpoint / default"],
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
    element("stat-remaining-detail").textContent = timing.remaining === null ? "Session target unavailable" : `${format(timing.remaining)} updates remaining`;
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
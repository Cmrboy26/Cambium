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
    ["loss-chart", [["policy_loss", "Policy", colors.red], ["value_loss", "Value", colors.amber], ["loss", "Total", colors.blue]]],
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

function format(value, digits = 0) {
    return Number.isFinite(value) ? new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value) : "--";
}

function initCharts() {
    if (typeof Chart === "undefined") throw new Error("Chart library could not be loaded.");
    Chart.defaults.font.family = '"Bahnschrift", "Segoe UI", sans-serif';
    Chart.defaults.color = "#65716b";
    for (const [id, series] of definitions) {
        const stacked = id === "reward-types-chart";
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

function smoothSeries(key) {
    const alpha = 2 / (smoothing + 1);
    let previous = null;
    return records.map((row) => {
        const value = typeof key === "function" ? key(row) : row[key];
        if (!Number.isFinite(value)) return { x: row.update, y: null };
        previous = previous === null ? value : alpha * value + (1 - alpha) * previous;
        return { x: row.update, y: previous };
    });
}

function redrawCharts() {
    for (const [id, series] of definitions) {
        const chart = charts.get(id);
        series.forEach(([key], index) => {
            const values = smoothSeries(key);
            chart.data.datasets[index].data = windowSize ? values.slice(-windowSize) : values;
        });
        chart.update("none");
    }
    const shown = windowSize ? Math.min(windowSize, records.length) : records.length;
    element("window-info").textContent = `${format(shown)} of ${format(records.length)} updates${smoothing > 1 ? ` / EMA ${smoothing}` : ""}`;
}

function renderConfig(config) {
    const options = config?.options ?? {};
    const metadata = config?.metadata ?? {};
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
    element("config").replaceChildren(fragment);
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
        const { runs } = await getJson("/api/runs", controller.signal);
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
            renderRecords();
            renderConfig(null);
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
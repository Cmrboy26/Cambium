"use strict";

const element = (id) => document.getElementById(id);
const colors = { green: "#18764f", blue: "#347aa2", red: "#b84242", amber: "#b27413", gray: "#6b746f" };
const rewardTypes = [
    ["doing_damage", "Damage dealt", colors.green], ["taking_damage", "Damage taken", colors.red],
    ["won_duel", "Win", colors.blue], ["death", "Death", "#8f364c"],
    ["episode_timeout", "Timeout", colors.amber], ["no_damage_penalty", "No damage", colors.gray],
    ["missed_attack", "Missed attack", "#a86b32"],
    ["shield_disabled", "Shield disabled", "#467b9d"], ["win_speed", "Win speed", "#458541"],
    ["blocked_with_shield", "Shield block", "#29898c"],
    ["good_aim", "Good aim", "#746ca8"], ["proximity", "Proximity", "#7b8542"],
    ["pitch_control", "Pitch control", "#4f6170"], ["aim_hold", "Aim hold", "#ab7256"],
    ["extreme_pitch_penalty", "Extreme pitch", "#bc596c"],
];
const definitions = [
    ["benchmark-win-chart", [["win_rate", "Benchmark win rate", colors.green]]],
    ["benchmark-return-chart", [["evaluation_reward", "Benchmark episode return", colors.blue]]],
    ["training-outcomes-chart", [["training_win_rate", "Win", colors.green], ["training_loss_rate", "Loss", colors.red],
        ["training_timeout_rate", "Timeout", colors.amber], ["training_draw_rate", "Draw", colors.gray],
        [row => trainingSummary(row).source === "recovered_rewards" ? metricValue(row, "training_loss_or_draw_rate") : null,
            "Loss or draw (recovered)", "#8f364c"]]],
    ["training-return-chart", [["training_return", "Episode return", colors.green]]],
    ["training-length-chart", [["training_length", "All fights", colors.blue], ["training_win_length", "Wins", colors.green],
        ["training_loss_length", "Losses", colors.red]]],
    ["metric-chart", [["training_return", "Episode return", colors.blue]]],
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
let windowMode = "latest";
let windowStart = null;
let windowEnd = null;
let windowStage = "";
let smoothing = 1;
let requestNumber = 0;
let pending;
let runSignature = "";
let runEntries = [];
let runPreferences = {};
let preferenceError = "";
let labelRun = null;
let activePage = "overview";
const runPreferenceKey = "cambium.run-labels.v1";
let displayedRun = "";
let configuration = null;
let milestoneIndex = null;
let milestoneRun = "";
const milestoneCharts = new Map();
let milestoneChartSignature = "";
let metricCatalog = {};
let chartAxis = "update";
let selectedSuite = "";
let selectedOpponent = "";
let performanceMode = "benchmark";
const axisLabels = {elapsed_seconds: "Elapsed training time (seconds)", samples: "Learner transitions", update: "PPO update"};

function trainingSummary(row, opponent = selectedOpponent) {
    const source = row.training_rollout ?? {};
    return opponent ? source.training_opponents?.[opponent] ?? {} : source;
}

function renderPerformanceMode() {
    for (const id of ["benchmark-performance", "benchmark-summary-win", "benchmark-summary-return", "benchmark-suite-control"]) {
        element(id).hidden = performanceMode !== "benchmark";
    }
    for (const id of ["training-performance", "training-summary-win", "training-summary-return", "training-opponent-control",
        "reward-diagnostics"]) {
        element(id).hidden = performanceMode !== "training";
    }
    for (const input of document.querySelectorAll('input[name="performance-mode"]')) input.checked = input.value === performanceMode;
}

function metricValue(row, metric) {
    const entry = metricCatalog[metric] ?? {};
    const source = entry.scope === "evaluation" ? row.evaluations?.[selectedSuite]
        : entry.scope === "episode" ? trainingSummary(row) : row;
    let value = source?.[entry.field ?? metric];
    if (!Number.isFinite(value) && metric === "training_loss_or_draw_rate"
        && Number.isFinite(source?.training_loss_rate) && Number.isFinite(source?.training_draw_rate)) {
        value = source.training_loss_rate + source.training_draw_rate;
    }
    return Number.isFinite(value) ? value * (entry.scale ?? 1) : null;
}

function chartRows() {
    let elapsed = 0;
    return records.map(row => {
        if (Number.isFinite(row.elapsed_seconds)) elapsed = row.elapsed_seconds;
        else {
            const duration = row.iteration_seconds ?? (Number.isFinite(row.collection_seconds) && Number.isFinite(row.update_seconds)
                ? row.collection_seconds + row.update_seconds : null);
            elapsed = Number.isFinite(elapsed) && Number.isFinite(duration) ? elapsed + duration : null;
        }
        return {...row, elapsed_seconds: elapsed};
    });
}

function windowRows(rows = records) {
    if (windowMode === "range") {
        return rows.filter(row => row.update >= (windowStart ?? rows[0]?.update)
            && row.update <= (windowEnd ?? rows.at(-1)?.update));
    }
    if (windowMode === "stage") return rows.filter(row => String(row.stage_index ?? row.stage ?? "") === windowStage);
    return windowMode === "all" || !windowSize ? rows : rows.slice(-windowSize);
}

function renderWindowControls() {
    for (const input of document.querySelectorAll('input[name="window-mode"]')) input.checked = input.value === windowMode;
    for (const mode of ["latest", "range", "stage"]) element(`window-${mode}-control`).hidden = windowMode !== mode;
    const count = records.length;
    const shown = Math.min(windowSize || count, count);
    const slider = element("window-size");
    slider.value = count > 1 ? Math.round(1000 * Math.log(Math.max(1, shown)) / Math.log(count)) : 0;
    slider.disabled = count < 2;
    slider.setAttribute("aria-valuetext", `${format(shown)} updates`);
    element("window-size-value").textContent = `${format(shown)} updates`;
    const minimum = records[0]?.update ?? 0;
    const maximum = records.at(-1)?.update ?? 0;
    if (count && windowMode === "range") {
        windowStart ??= minimum;
        windowEnd ??= maximum;
    }
    if (count && windowStart !== null) windowStart = Math.max(minimum, Math.min(maximum, windowStart));
    if (count && windowEnd !== null) windowEnd = Math.max(windowStart ?? minimum, Math.min(maximum, windowEnd));
    for (const [bound, value] of [["start", windowStart ?? minimum], ["end", windowEnd ?? maximum]]) {
        for (const suffix of ["", "-slider"]) {
            const input = element(`window-${bound}${suffix}`);
            input.min = minimum;
            input.max = maximum;
            input.value = value;
            input.disabled = !count;
        }
    }
    const stages = new Map((resolvedExperiment(configuration)?.stages ?? []).map((stage, index) => [String(index), stage.name]));
    const spans = new Map();
    for (const row of records) {
        const key = String(row.stage_index ?? row.stage ?? "");
        if (!key) continue;
        stages.set(key, row.stage ?? stages.get(key) ?? `Stage ${key}`);
        const span = spans.get(key) ?? [row.update, row.update];
        span[1] = row.update;
        spans.set(key, span);
    }
    const entries = [...stages].map(([key, name]) => {
        const span = spans.get(key);
        return [key, `${name ?? `Stage ${key}`} / ${span ? `updates ${format(span[0])}-${format(span[1])}` : "No recorded updates"}`];
    });
    const stage = selectOptions("window-stage", entries.length ? entries : [["", "No curriculum stages"]], windowStage);
    if (windowMode === "stage") windowStage = stage;
    element("window-stage").disabled = !entries.length;
}

function updateWindowMode(mode) {
    windowMode = mode;
    if (mode === "range" && records.length) {
        windowStart ??= records[0].update;
        windowEnd ??= records.at(-1).update;
    }
    if (mode === "stage" && !windowStage) {
        windowStage = String(records.at(-1)?.stage_index ?? records.at(-1)?.stage ?? "");
        element("window-stage").value = windowStage;
    }
    redrawCharts();
}

function updateLatestWindow() {
    const position = Math.max(0, Math.min(1000, Number(element("window-size").value)));
    windowSize = Math.round(Math.max(1, records.length) ** (position / 1000));
    redrawCharts();
}

function updateWindowRange(bound, rawValue) {
    const value = Number(rawValue);
    if (!records.length || String(rawValue).trim() === "" || !Number.isFinite(value)) {
        renderWindowControls();
        return;
    }
    const update = Math.max(records[0].update, Math.min(records.at(-1).update, Math.round(value)));
    if (bound === "start") {
        windowStart = update;
        windowEnd = Math.max(update, windowEnd ?? records.at(-1).update);
    } else {
        windowEnd = update;
        windowStart = Math.min(update, windowStart ?? records[0].update);
    }
    redrawCharts();
}

function metricPoints(key, derivative = false) {
    const rows = chartRows();
    const updates = new Map(rows.map(row => [row.update, row]));
    const seen = new Set();
    const entry = typeof key === "string" ? metricCatalog[key] : null;
    const alpha = 2 / (smoothing + 1);
    const visibleUpdates = new Set(windowRows(rows).map(row => row.update));
    let previous = null;
    let previousPoint = null;
    let previousStage;
    const points = [];
    for (const row of rows) {
        let anchor = row;
        if (entry?.scope === "evaluation") {
            const snapshot = row.evaluations?.[selectedSuite]?.snapshot_update;
            if (!Number.isFinite(snapshot) || seen.has(snapshot) || !updates.has(snapshot)) continue;
            seen.add(snapshot);
            anchor = updates.get(snapshot);
        }
        const stage = anchor.stage_index ?? anchor.stage;
        if (entry?.scope !== "evaluation" && stage !== previousStage) previous = previousPoint = null;
        previousStage = stage;
        const value = typeof key === "function" ? key(row) : metricValue(row, key);
        const x = anchor[chartAxis];
        let y = null;
        if (Number.isFinite(value) && Number.isFinite(x)) {
            previous = entry?.scope === "evaluation" || previous === null ? value : alpha * value + (1 - alpha) * previous;
            y = derivative ? previousPoint && x > previousPoint.x ? (previous - previousPoint.y) / (x - previousPoint.x) : null : previous;
            previousPoint = {x, y: previous};
        } else previous = previousPoint = null;
        if (visibleUpdates.has(anchor.update) && Number.isFinite(x)) points.push({x, y, stage, update: anchor.update});
    }
    return points;
}

const stageMarkers = {
    id: "stageMarkers",
    afterDraw(chart) {
        const {ctx, chartArea, scales} = chart;
        if (!chartArea || !scales.x) return;
        let previous;
        ctx.save();
        ctx.strokeStyle = "#b4bdb7";
        ctx.fillStyle = colors.gray;
        ctx.font = '10px "Segoe UI", sans-serif';
        ctx.setLineDash([3, 4]);
        for (const row of windowRows(chartRows())) {
            const stage = row.stage_index ?? row.stage;
            const x = scales.x.getPixelForValue(row[chartAxis]);
            if (previous !== undefined && stage !== previous && x >= chartArea.left && x <= chartArea.right) {
                ctx.beginPath(); ctx.moveTo(x, chartArea.top); ctx.lineTo(x, chartArea.bottom); ctx.stroke();
                const width = Math.min(130, chartArea.right - chartArea.left);
                ctx.fillText(row.stage ?? String(stage), Math.min(x + 3, chartArea.right - width), chartArea.top + 12, width);
            }
            previous = stage;
        }
        ctx.restore();
    },
};

function selectOptions(id, entries, preferred = "") {
    const select = element(id);
    const current = select.value;
    const signature = JSON.stringify(entries);
    if (select.dataset.options !== signature) {
        select.replaceChildren(...entries.map(([value, label]) => new Option(label, value)));
        select.dataset.options = signature;
    }
    select.value = entries.some(([value]) => value === current) ? current
        : entries.some(([value]) => value === preferred) ? preferred : entries[0]?.[0] ?? "";
    return select.value;
}

function renderPerformance() {
    renderPerformanceMode();
    const suiteNames = [...new Set(records.flatMap(row => Object.keys(row.evaluations ?? {})))].sort();
    selectedSuite = selectOptions("benchmark-suite", suiteNames.length ? suiteNames.map(name => [name, name])
        : [["", "No evaluations yet"]], "combined_benchmark");
    const opponents = [...new Set(records.flatMap(row => Object.keys(trainingSummary(row, "").training_opponents ?? {})))].sort();
    selectedOpponent = selectOptions("training-opponent", [["", "All opponents"], ...opponents.map(name => [name, name])]);
    const metric = selectOptions("metric-select", Object.entries(metricCatalog).map(([key, entry]) =>
        [key, `${entry.label} (${entry.unit})`]), "training_return");
    if (metric) definitions.find(([id]) => id === "metric-chart")[1][0][0] = metric;
    const latest = records.at(-1) ?? {};
    const evaluation = records.map(row => row.evaluations?.[selectedSuite]).filter(Boolean).at(-1);
    const snapshot = evaluation?.snapshot_update;
    const freshness = Number.isFinite(snapshot) ? `Update ${format(snapshot)} / ${format(latest.update - snapshot)} updates ago` : "Snapshot age unavailable";
    const detail = evaluation ? `${selectedSuite} / ${format(evaluation.episodes)} episodes` : "Awaiting evaluation";
    element("stat-benchmark").textContent = evaluation ? `${format(evaluation.win_rate * 100, 2)}%` : "--";
    element("stat-benchmark-return").textContent = format(evaluation?.reward_mean, 3);
    element("stat-benchmark-detail").textContent = detail;
    element("stat-benchmark-age").textContent = freshness;
    const experiment = resolvedExperiment(configuration);
    const suite = experiment?.evaluations?.[selectedSuite] ?? experiment?.stages?.[evaluation?.stage_index]?.evaluations?.[selectedSuite];
    const moving = (suite?.opponents ?? []).some(({id}) => ["self_play", "frozen_self_play"].includes(experiment?.opponents?.[id]?.type));
    element("benchmark-status").textContent = `${detail} / ${freshness}${moving ? " / Moving-opponent evaluation" : ""}${evaluation?.win_rate === 1 ? " / Benchmark ceiling reached" : ""}`;
    const source = trainingSummary(latest, "");
    const summary = trainingSummary(latest);
    const count = summary.training_episodes ?? (selectedOpponent && source.source !== "recovered_rewards"
        && Number.isFinite(source.training_episodes) ? 0 : undefined);
    const episodeDetail = Number.isFinite(count) ? `${format(count)} completed episodes in this rollout`
        : "Episode metrics not recorded";
    const provenance = source.source === "recovered_rewards" ? " / Recovered reward counts; loss and draw combined; episode return and length unavailable" : "";
    const trainingWin = metricValue(latest, "training_win_rate");
    element("stat-training-win").textContent = trainingWin === null ? "--" : `${format(trainingWin, 2)}%`;
    element("stat-training-return").textContent = format(metricValue(latest, "training_return"), 3);
    element("stat-training-detail").textContent = episodeDetail + (source.source === "recovered_rewards" ? " / Recovered counts" : "");
    element("stat-training-opponent").textContent = selectedOpponent || "All opponents";
    element("episode-status").textContent = `${episodeDetail} / ${latest.stage ?? "Stage unavailable"} / ${selectedOpponent || "All opponents"}${provenance}`;
    renderTable("training-opponents", Object.entries(source.training_opponents ?? {}).map(([name, result]) => [
        name, format(result.training_episodes), ...["win", "loss", "timeout", "draw"].map(outcome =>
            Number.isFinite(result[`training_${outcome}_rate`]) ? `${format(result[`training_${outcome}_rate`] * 100, 2)}%` : "--"),
        format(result.training_return, 3), format(result.training_length, 1),
    ]), 8, source.source === "recovered_rewards" ? "Per-opponent episode metrics not recorded; aggregate outcomes recovered." : "Episode metrics not recorded.");
}

function runLabel(run) {
    return runPreferences[run.path] ?? {};
}

function runTitle(run) {
    return runLabel(run).name || run.experiment_name || run.environment_name || run.name;
}

function sortRuns(runs, order = "modified") {
    const compare = (left, right) => left.localeCompare(right, undefined, {numeric: true, sensitivity: "base"});
    return [...runs].sort((left, right) => order === "name"
        ? compare(runTitle(left), runTitle(right)) || compare(left.name, right.name) || compare(left.id, right.id)
        : (right.modified ?? 0) - (left.modified ?? 0) || compare(right.name, left.name) || compare(left.id, right.id));
}

function runDetails(run) {
    return [
        run.experiment_name && run.experiment_name !== runTitle(run) ? run.experiment_name : null,
        run.seed != null ? `Seed ${run.seed}` : null,
        run.hidden_dim != null ? `Width ${run.hidden_dim}` : null,
        run.lr != null ? `LR ${run.lr}` : null,
        Number.isFinite(run.modified) ? `Updated ${new Date(run.modified * 1000).toLocaleString()}` : null,
    ].filter(Boolean).join(" / ");
}

function searchRuns(runs, query) {
    const terms = query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
    return runs.filter(run => {
        const text = [runTitle(run), run.name, run.path, run.experiment_name, run.environment_name,
            run.policy_name, runDetails(run), runLabel(run).notes,
            Number.isFinite(run.lr) ? run.lr.toExponential() : ""].join(" ").toLocaleLowerCase();
        return terms.every(term => text.includes(term));
    });
}

function currentRun() {
    return runEntries.find(run => run.id === element("run").value);
}

function loadRunPreferences() {
    try {
        const saved = JSON.parse(localStorage.getItem(runPreferenceKey) ?? "{}");
        if (!saved || typeof saved !== "object" || Array.isArray(saved)
            || Object.values(saved).some(label => !label || typeof label.name !== "string"
                || typeof label.notes !== "string" || label.name.length > 160 || label.notes.length > 2000)) {
            throw new Error("Saved labels have an invalid format.");
        }
        runPreferences = saved;
    } catch (error) {
        preferenceError = `Could not load saved run labels: ${error.message}`;
        element("run-label-status").textContent = preferenceError;
    }
}

function renderRunLabel() {
    const run = currentRun();
    for (const id of ["run-display-name", "run-notes", "run-label-save", "edit-run"]) element(id).disabled = !run;
    if (labelRun === (run?.id ?? "")) return;
    labelRun = run?.id ?? "";
    element("run-display-name").value = run ? runLabel(run).name ?? "" : "";
    element("run-display-name").placeholder = run ? run.experiment_name ?? run.name : "Select a training run";
    element("run-notes").value = run ? runLabel(run).notes ?? "" : "";
    element("run-label-status").textContent = preferenceError;
}

function selectRun(identity) {
    element("run").value = identity;
    element("run-picker").open = false;
    element("run-picker-toggle").focus();
    renderRunPicker();
    refresh();
}

function renderRunPicker() {
    const run = currentRun();
    element("run-current").textContent = run ? runTitle(run) : "No training runs found";
    element("run-current-meta").textContent = run ? runDetails(run) || run.name : "Runs will appear here when training starts.";
    renderRunLabel();
    const matches = searchRuns(sortRuns(runEntries, element("run-sort").value), element("run-search").value);
    const signature = JSON.stringify([matches, runPreferences, element("run").value]);
    if (signature === runSignature) return;
    runSignature = signature;
    const list = element("run-options");
    const focused = list.contains(document.activeElement) ? document.activeElement.dataset.runId : null;
    const scroll = list.scrollTop;
    const fragment = document.createDocumentFragment();
    for (const entry of matches) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "run-option";
        button.dataset.runId = entry.id;
        button.setAttribute("aria-pressed", String(entry.id === run?.id));
        const title = document.createElement("strong");
        title.textContent = runTitle(entry);
        const details = document.createElement("span");
        details.textContent = runDetails(entry);
        const directory = document.createElement("small");
        directory.textContent = entry.name;
        directory.title = entry.path;
        button.append(title, details, directory);
        if (runLabel(entry).notes) {
            const notes = document.createElement("span");
            notes.className = "run-option-notes";
            notes.textContent = runLabel(entry).notes;
            button.append(notes);
        }
        if (entry.warnings?.length) {
            const warning = document.createElement("span");
            warning.className = "run-option-warning";
            warning.textContent = entry.warnings.join(" ");
            button.append(warning);
        }
        button.addEventListener("click", () => selectRun(entry.id));
        fragment.append(button);
    }
    if (!matches.length) {
        const empty = document.createElement("p");
        empty.className = "run-picker-empty";
        empty.textContent = runEntries.length ? "No matching runs. Try a different search." : "No training runs found.";
        fragment.append(empty);
    }
    list.replaceChildren(fragment);
    element("run-count").textContent = `${matches.length} of ${runEntries.length} runs`;
    if (focused) [...list.querySelectorAll("button")].find(button => button.dataset.runId === focused)?.focus({preventScroll: true});
    list.scrollTop = scroll;
}

function saveRunLabel(event) {
    event.preventDefault();
    const run = currentRun();
    if (!run) {
        element("run-label-status").textContent = "Select a training run before saving a label.";
        return;
    }
    const name = element("run-display-name").value.trim();
    const notes = element("run-notes").value.trim();
    if (name.length > 160 || notes.length > 2000) {
        element("run-label-status").textContent = "Use at most 160 characters for the name and 2000 for notes.";
        return;
    }
    const next = {...runPreferences};
    if (name || notes) next[run.path] = {name, notes};
    else delete next[run.path];
    try {
        localStorage.setItem(runPreferenceKey, JSON.stringify(next));
    } catch (error) {
        element("run-label-status").textContent = `Could not save run label: ${error.message}`;
        return;
    }
    runPreferences = next;
    preferenceError = "";
    renderRunPicker();
    element("run-title").textContent = runTitle(run);
    element("run-label-status").textContent = "Saved in this browser.";
}

function showPage(page) {
    activePage = page;
    for (const section of document.querySelectorAll("[data-page]")) {
        section.hidden = section.dataset.page !== page || section.dataset.available === "false";
    }
    for (const link of document.querySelectorAll("[data-view]")) {
        if (link.dataset.view === page) link.setAttribute("aria-current", "page");
        else link.removeAttribute("aria-current");
    }
    document.dispatchEvent(new CustomEvent("dashboard-page", {detail: {page}}));
    for (const chart of charts.values()) chart.resize();
    for (const entry of milestoneCharts.values()) entry.chart?.resize?.();
}

function navigateHash() {
    const target = element(window.location.hash.slice(1) || "overview");
    const section = target?.closest("[data-page]");
    showPage(section?.dataset.page ?? "overview");
}

function initRunPicker() {
    loadRunPreferences();
    element("run-search").addEventListener("input", renderRunPicker);
    element("run-sort").addEventListener("change", renderRunPicker);
    element("run-label-form").addEventListener("submit", saveRunLabel);
    element("run-picker").addEventListener("toggle", () => {
        if (element("run-picker").open) element("run-search").focus();
    });
    element("run-picker").addEventListener("keydown", event => {
        if (event.key === "Escape") {
            event.preventDefault();
            element("run-picker").open = false;
            element("run-picker-toggle").focus();
        } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            const buttons = [...element("run-options").querySelectorAll("button")];
            if (!buttons.length || event.target === element("run-sort")) return;
            event.preventDefault();
            const index = buttons.indexOf(document.activeElement);
            const next = index < 0 ? (event.key === "ArrowDown" ? 0 : buttons.length - 1)
                : (index + (event.key === "ArrowDown" ? 1 : -1) + buttons.length) % buttons.length;
            buttons[next].focus();
        }
    });
    document.addEventListener("click", event => {
        if (!element("run-picker").contains(event.target)) element("run-picker").open = false;
    });
    element("edit-run").addEventListener("click", () => {
        window.location.hash = "run-settings";
        showPage("configuration");
        element("run-display-name").focus();
    });
}

function soloLegend(chart, index) {
    if (chart.data.datasets.length < 2 || !Number.isInteger(index) || !chart.data.datasets[index]) return;
    const visible = chart.data.datasets.map((_, dataset) => chart.isDatasetVisible(dataset));
    const restore = visible[index] && visible.filter(Boolean).length === 1;
    chart.data.datasets.forEach((_, dataset) => chart.setDatasetVisibility(dataset, restore || dataset === index));
    chart.update("none");
}

const legendSoloHandlers = new WeakMap();
const legendSolo = {
    id: "legendSolo",
    afterInit(chart) {
        if (chart.data.datasets.length < 2) return;
        chart.canvas.title = "Click a legend entry to toggle it. Right-click to solo; right-click the solo entry again to show all.";
        const handler = event => {
            const legend = chart.legend;
            if (!legend || !legend.options.display) return;
            const {x, y} = Chart.helpers.getRelativePosition(event, chart);
            const hit = legend.legendHitBoxes.findIndex(box => x >= box.left && x <= box.left + box.width
                && y >= box.top && y <= box.top + box.height);
            const index = legend.legendItems[hit]?.datasetIndex;
            if (!Number.isInteger(index)) return;
            event.preventDefault();
            soloLegend(chart, index);
        };
        legendSoloHandlers.set(chart, handler);
        chart.canvas.addEventListener("contextmenu", handler);
    },
    beforeDestroy(chart) {
        const handler = legendSoloHandlers.get(chart);
        if (handler) chart.canvas.removeEventListener("contextmenu", handler);
        legendSoloHandlers.delete(chart);
    },
};

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
            plugins: [stageMarkers, legendSolo],
            data: { datasets: series.map(([, label, color]) => ({
                label, data: [], borderColor: color, backgroundColor: color,
                borderWidth: 1.8, pointRadius: 0, pointHitRadius: 8, pointHoverRadius: 3,
                tension: 0, spanGaps: false,
                segment: {borderColor: context => context.p0.raw.stage !== context.p1.raw.stage
                    && !["benchmark-win-chart", "benchmark-return-chart"].includes(id)
                    && !(id === "metric-chart" && metricCatalog[definitions.find(([name]) => name === id)[1][0][0]]?.scope === "evaluation")
                    ? "transparent" : undefined},
            })) },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
                normalized: true, interaction: { mode: "index", intersect: false },
                plugins: {
                    legend: { display: series.length > 1, position: "bottom", labels: { boxWidth: 10, boxHeight: 3, font: { size: 10 } } },
                    decimation: { enabled: !stacked, algorithm: "min-max" },
                    tooltip: { callbacks: { title: items => `Update ${items[0]?.raw.update ?? ""}` } },
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
    let previousStage;
    return records.map((row) => {
        const stage = row.stage_index ?? row.stage;
        if (stage !== previousStage) previous = previousPoint = null;
        previousStage = stage;
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

function updateSmoothing() {
    const position = Math.max(0, Math.min(1000, Number(element("smooth").value)));
    smoothing = Math.round(500 ** (position / 1000));
    const label = smoothing > 1 ? `EMA ${smoothing}` : "Off";
    element("smooth-value").textContent = label;
    element("smooth").setAttribute("aria-valuetext", label);
    redrawCharts();
}

function redrawCharts() {
    renderWindowControls();
    const visible = windowRows(chartRows());
    const domain = visible.map(row => row[chartAxis]).filter(Number.isFinite);
    for (const [id, series] of definitions) {
        const chart = charts.get(id);
        if (!chart) continue;
        const derivative = document.querySelector(`input[name="${id}-mode"]:checked`)?.value === "derivative";
        series.forEach(([key], index) => {
            chart.data.datasets[index].data = metricPoints(key, derivative);
            const entry = typeof key === "string" ? metricCatalog[key] : null;
            if (entry && series.length === 1) chart.data.datasets[index].label = `${entry.label} (${entry.unit})`;
        });
        const entry = metricCatalog[series[0][0]];
        chart.options.scales.x.min = domain[0];
        chart.options.scales.x.max = domain.at(-1);
        chart.options.scales.x.title = {display: true, text: axisLabels[chartAxis]};
        chart.options.scales.y.title = { display: true, text: derivative ? `Change / ${axisLabels[chartAxis]}` : entry?.unit ?? "Weighted reward" };
        chart.options.scales.y.min = entry?.unit === "%" && !derivative ? 0 : undefined;
        chart.options.scales.y.max = entry?.unit === "%" && !derivative ? 100 : undefined;
        chart.update("none");
    }
    const legacyClock = chartAxis === "elapsed_seconds" && windowRows().some(row => !Number.isFinite(row.elapsed_seconds));
    const bounds = visible.length ? ` / Updates ${format(visible[0].update)}-${format(visible.at(-1).update)}` : "";
    element("window-info").textContent = `${format(visible.length)} of ${format(records.length)} updates${bounds}${smoothing > 1 ? ` / EMA ${smoothing}` : ""}${legacyClock ? " / Estimated elapsed time; untimed points omitted" : ""}`;
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
    element("run-title").textContent = currentRun() ? runTitle(currentRun()) : experiment?.name ?? environment.name ?? "Training overview";
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
                plugins: [legendSolo],
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
    element("curriculum").dataset.available = String(Boolean(experiment || hasResearchMetrics));
    element("curriculum").hidden = activePage !== "overview" || !experiment && !hasResearchMetrics;
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
    renderPerformance();
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
        if (ticket !== requestNumber) return;
        runEntries = dataRuns.runs;
        const runs = sortRuns(runEntries, element("run-sort").value);
        const select = element("run");
        if (!runs.some(run => run.id === select.value)) select.value = runs[0]?.id ?? "";
        renderRunPicker();
        const identity = select.value;
        if (displayedRun !== identity) {
            windowStart = windowEnd = null;
            windowStage = "";
            element("window-stage").value = "";
            records = [];
            renderConfig(null);
            renderRecords();
            displayedRun = identity;
        }
        if (identity) {
            const data = await getJson(`/api/runs/${encodeURIComponent(identity)}/metrics`, controller.signal);
            if (ticket !== requestNumber || identity !== select.value) return;
            records = data.metrics;
            metricCatalog = data.metric_catalog ?? {};
            element("run-path").textContent = data.path;
            element("empty").hidden = records.length > 0;
            element("empty").textContent = "Waiting for the first PPO update.";
            renderConfig(data.config);
            renderRecords();
            notice([...data.warnings, preferenceError].filter(Boolean).join(" "));
        } else {
            element("run-path").textContent = "";
            element("empty").textContent = "No training runs found.";
            element("empty").hidden = false;
            records = [];
            renderConfig(null);
            renderRecords();
            notice(preferenceError);
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
    chartAxis = element("chart-axis").value in axisLabels ? element("chart-axis").value : "update";
    initCharts();
    updateSmoothing();
    initRunPicker();
    window.addEventListener("hashchange", navigateHash);
    navigateHash();
    for (const input of document.querySelectorAll('input[name="performance-mode"]')) {
        input.addEventListener("change", event => {
            performanceMode = event.target.value;
            renderPerformance();
            for (const chart of charts.values()) chart.resize?.();
            redrawCharts();
        });
    }
    element("chart-axis").addEventListener("change", event => {chartAxis = event.target.value; redrawCharts();});
    for (const id of ["benchmark-suite", "training-opponent", "metric-select"]) {
        element(id).addEventListener("change", () => {renderPerformance(); redrawCharts();});
    }
    element("run").addEventListener("change", refresh);
    element("refresh").addEventListener("click", refresh);
    element("live").addEventListener("change", () => {
        if (element("live").checked) refresh();
        else element("connection").textContent = "Paused";
    });
    document.querySelectorAll('input[name="window-mode"]').forEach((input) => {
        input.addEventListener("change", event => updateWindowMode(event.target.value));
    });
    element("window-size").addEventListener("input", updateLatestWindow);
    for (const bound of ["start", "end"]) {
        element(`window-${bound}`).addEventListener("change", event => updateWindowRange(bound, event.target.value));
        element(`window-${bound}-slider`).addEventListener("input", event => updateWindowRange(bound, event.target.value));
    }
    element("window-stage").addEventListener("change", event => {windowStage = event.target.value; redrawCharts();});
    element("smooth").addEventListener("input", updateSmoothing);
    refresh();
    setInterval(() => { if (element("live").checked && !document.hidden) refresh(); }, 3000);
    document.addEventListener("visibilitychange", () => { if (!document.hidden && element("live").checked) refresh(); });
} catch (error) {
    element("connection").textContent = "Unavailable";
    element("connection").dataset.state = "error";
    notice(error.message);
}
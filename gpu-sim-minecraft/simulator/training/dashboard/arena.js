import * as THREE from "./three.module.min.js";
import { OrbitControls } from "./OrbitControls.js";
import { buildVoxels, itemLabel, playerLoadout, replaySample, replayPlayers } from "./arena-data.js";

const host = document.getElementById("arena");
const message = document.getElementById("arena-message");
const playerColors = [0x18764f, 0x347aa2];
const materialColors = { diamond: 0x63d5d5, netherite: 0x514c57, iron: 0xc9d0d4, golden: 0xe1bc4c,
    leather: 0x985c40, copper: 0xbf7655, chainmail: 0x87969c, turtle: 0x559568, stone: 0x8b9390, wooden: 0x9d7d50 };

function equipmentColor(item, fallback) {
    return Object.entries(materialColors).find(([name]) => item?.includes(`:${name}_`))?.[1] ?? fallback;
}

function renderEquipment(scenario, playback = false) {
    for (const [index, prefix] of ["player-one", "player-two"].entries()) {
        const player = scenario.players?.[index];
        const list = document.getElementById(`${prefix}-gear`);
        const state = document.getElementById(`${prefix}-state`);
        list.replaceChildren();
        if (!player) { state.textContent = "Player unavailable"; continue; }
        const loadout = playerLoadout(player, scenario.profile);
        const shield = playback && player.shield_cooldown > 0 ? ` / Shield disabled (${player.shield_cooldown} ticks)`
            : playback && player.using_shield ? " / Shield raised" : "";
        state.textContent = `${Number((player.health ?? 20).toFixed(1))} HP / ${playback ? "Position" : "Spawn"} ${player.position?.map((value) => Number(value.toFixed(1))).join(", ") ?? "--"}${shield}`;
        const fields = [["Main hand", itemLabel(loadout.weapons[loadout.selected]), true],
            ["Weapon slots", loadout.weapons.map(itemLabel).join(" / ")], ["Offhand", itemLabel(loadout.offhand)]];
        for (const [slot, label] of [["helmet", "Helmet"], ["chestplate", "Chestplate"], ["leggings", "Leggings"], ["boots", "Boots"]]) {
            fields.push([label, itemLabel(loadout.armor[slot])]);
        }
        for (const [label, value, selected] of fields) {
            const row = document.createElement("div");
            const term = document.createElement("dt");
            const description = document.createElement("dd");
            term.textContent = label;
            description.textContent = value;
            if (selected) description.className = "gear-selected";
            row.append(term, description);
            list.append(row);
        }
    }
}

class ArenaViewer {
    constructor() {
        this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
        this.renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 2));
        this.renderer.setClearColor(0xedf2f0);
        this.renderer.domElement.setAttribute("aria-label", "Voxel arena and player positions");
        this.renderer.domElement.setAttribute("role", "img");
        host.prepend(this.renderer.domElement);
        this.scene = new THREE.Scene();
        this.scene.add(new THREE.HemisphereLight(0xffffff, 0x809389, 2.4));
        const sunlight = new THREE.DirectionalLight(0xffffff, 2.4);
        sunlight.position.set(8, 20, 12);
        this.scene.add(sunlight);
        this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 2000);
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.screenSpacePanning = true;
        this.controls.maxPolarAngle = Math.PI / 2 - 0.01;
        this.controls.addEventListener("change", () => this.draw());
        this.content = new THREE.Group();
        this.scene.add(this.content);
        this.players = new THREE.Group();
        this.content.add(this.players);
        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.resizeObserver.observe(host);
        this.renderer.domElement.addEventListener("webglcontextlost", (event) => {
            event.preventDefault();
            host.dataset.unavailable = "true";
            message.hidden = false;
            message.textContent = "3D context lost. Reload to restore the arena preview.";
        });
        this.resize();
    }

    draw() { this.renderer.render(this.scene, this.camera); }

    resize() {
        const width = host.clientWidth;
        const height = host.clientHeight;
        if (!width || !height) return;
        this.renderer.setSize(width, height, false);
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        if (this.fitPending) this.reset();
        this.draw();
    }

    clear() {
        const geometries = new Set();
        const materials = new Set();
        this.content.traverse((object) => {
            if (object.geometry) geometries.add(object.geometry);
            for (const material of Array.isArray(object.material) ? object.material : object.material ? [object.material] : []) materials.add(material);
            if (object.isInstancedMesh) object.dispose();
        });
        geometries.forEach((geometry) => geometry.dispose());
        materials.forEach((material) => { material.map?.dispose(); material.dispose(); });
        this.content.clear();
        this.players.clear();
        this.content.add(this.players);
        this.playerSignature = null;
    }

    load(scenario) {
        const voxels = buildVoxels(scenario);
        this.clear();
        const minimum = scenario.bounds.min;
        const size = scenario.bounds.size;
        this.center = new THREE.Vector3(...minimum).add(new THREE.Vector3(...size).multiplyScalar(0.5));
        this.radius = new THREE.Vector3(...size).length() / 2;
        this.wallMaterial = new THREE.MeshStandardMaterial({ color: 0x9ba7a1, roughness: 1, transparent: true, opacity: 0.32, depthWrite: false });
        const floorMaterial = new THREE.MeshStandardMaterial({ color: 0xb4bfb8, roughness: 1 });
        const geometry = new THREE.BoxGeometry(0.985, 0.985, 0.985);
        const matrix = new THREE.Matrix4();
        for (const elevated of [false, true]) {
            const positions = voxels.filter((position) => (position[1] > minimum[1]) === elevated);
            if (!positions.length) continue;
            const mesh = new THREE.InstancedMesh(geometry, elevated ? this.wallMaterial : floorMaterial, positions.length);
            positions.forEach((position, index) => {
                matrix.makeTranslation(...position.map((value) => value + 0.5));
                mesh.setMatrixAt(index, matrix);
                const variation = ((position[0] * 13 + position[1] * 7 + position[2] * 17) % 5 + 5) % 5;
                mesh.setColorAt(index, new THREE.Color().setScalar(0.88 + variation * 0.03));
            });
            mesh.instanceMatrix.needsUpdate = true;
            this.content.add(mesh);
        }
        this.setPlayers(scenario.players ?? [], scenario.profile);
        this.setWalls(document.getElementById("arena-solid-walls").checked);
        this.fitPending = true;
        this.reset();
    }

    addPlayer(player, index, profile) {
        const actor = new THREE.Group();
        actor.position.set(...player.position);
        actor.rotation.y = -THREE.MathUtils.degToRad(player.yaw ?? 0);
        this.players.add(actor);
        const loadout = playerLoadout(player, profile);
        const armor = loadout.armor;
        const box = new THREE.BoxGeometry(1, 1, 1);
        const materialCache = new Map();
        const cube = (size, position, color) => {
            if (!materialCache.has(color)) materialCache.set(color, new THREE.MeshStandardMaterial({ color, roughness: 0.85 }));
            const mesh = new THREE.Mesh(box, materialCache.get(color));
            mesh.scale.set(...size);
            mesh.position.set(...position);
            actor.add(mesh);
            return mesh;
        };
        const skin = 0xd4a083;
        const head = new THREE.Group();
        head.position.y = 1.57;
        actor.add(head);
        const headPart = (size, position, color) => {
            const part = cube(size, position, color);
            head.attach(part);
        };
        headPart([0.44, 0.43, 0.42], [0, 1.57, 0], skin);
        headPart([0.45, 0.1, 0.43], [0, 1.76, 0], equipmentColor(armor.helmet, 0x4b3932));
        if (armor.helmet && armor.helmet !== "minecraft:air") {
            for (const side of [-1, 1]) headPart([0.055, 0.26, 0.44], [side * 0.23, 1.64, 0], equipmentColor(armor.helmet, skin));
        }
        for (const side of [-1, 1]) {
            headPart([0.065, 0.045, 0.015], [side * 0.1, 1.6, 0.217], 0x293e44);
            cube([0.18, 0.62, 0.24], [side * 0.105, 0.43, 0], equipmentColor(armor.leggings, 0x455d72));
            cube([0.2, 0.19, 0.28], [side * 0.105, 0.105, 0.02], equipmentColor(armor.boots, 0x384447));
            cube([0.15, 0.55, 0.22], [side * 0.29, 1.02, 0], equipmentColor(armor.chestplate, skin));
        }
        cube([0.39, 0.58, 0.25], [0, 1.04, 0], equipmentColor(armor.chestplate, playerColors[index]));
        const weaponGroup = new THREE.Group();
        weaponGroup.position.set(-0.29, 0.8, 0.11);
        const weapon = loadout.weapons[loadout.selected];
        actor.add(weaponGroup);
        const weaponPart = (size, position, color) => { const part = cube(size, position, color); weaponGroup.attach(part); };
        if (weapon && weapon !== "minecraft:air") {
            weaponPart([0.055, 0.43, 0.055], [-0.37, 0.94, 0.26], 0x816346);
            if (weapon.endsWith("_axe")) weaponPart([0.28, 0.24, 0.08], [-0.3, 1.2, 0.26], equipmentColor(weapon, 0xc9d0d4));
            else {
                weaponPart([0.075, 0.47, 0.055], [-0.37, 1.3, 0.26], equipmentColor(weapon, 0xc9d0d4));
                weaponPart([0.24, 0.06, 0.07], [-0.37, 1.07, 0.26], 0x535d57);
            }
        }
        const shieldGroup = new THREE.Group();
        actor.add(shieldGroup);
        if (loadout.offhand === "minecraft:shield") {
            for (const part of [cube([0.4, 0.64, 0.075], [0.37, 1, 0.25], 0x6f7f80),
                cube([0.31, 0.54, 0.085], [0.37, 1, 0.26], 0xb39973)]) shieldGroup.attach(part);
        }
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(0.6, 1.8, 0.6)),
            new THREE.LineBasicMaterial({ color: playerColors[index], transparent: true, opacity: 0.6 }));
        edges.position.y = 0.9;
        actor.add(edges);
        actor.add(new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0.04, 0.2), 1.1, playerColors[index], 0.25, 0.2));
        const label = document.createElement("canvas");
        label.width = 128;
        label.height = 64;
        const context = label.getContext("2d");
        context.fillStyle = `#${playerColors[index].toString(16).padStart(6, "0")}`;
        context.font = "bold 40px sans-serif";
        context.textAlign = "center";
        context.fillText(`P${index + 1}`, 64, 45);
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(label), depthTest: false }));
        sprite.position.set(0, 2.3, 0);
        sprite.scale.set(1.2, 0.6, 1);
        actor.add(sprite);
        actor.userData = { head, weaponGroup, shieldGroup };
    }

    setPlayers(players, profile) {
        const next = JSON.stringify(players.map((player) => playerLoadout(player, profile)));
        if (next !== this.playerSignature) {
            this.players.traverse((object) => {
                object.geometry?.dispose();
                if (object.material) { object.material.map?.dispose(); object.material.dispose(); }
            });
            this.players.clear();
            players.slice(0, 2).forEach((player, index) => this.addPlayer(player, index, profile));
            this.playerSignature = next;
        }
        players.slice(0, 2).forEach((player, index) => {
            const actor = this.players.children[index];
            actor.position.set(...player.position);
            actor.rotation.y = -THREE.MathUtils.degToRad(player.yaw ?? 0);
            actor.userData.head.rotation.x = THREE.MathUtils.degToRad(player.pitch ?? 0);
            actor.userData.shieldGroup.position.set(player.using_shield ? -0.2 : 0, 0, player.using_shield ? 0.15 : 0);
            actor.userData.weaponGroup.rotation.x = player.controls?.attack ? 0.45 : 0;
            actor.scale.y = player.pose === 1 ? 0.85 : player.pose === 2 ? 0.4 : 1;
            actor.visible = player.health === undefined || player.health > 0;
        });
        this.draw();
    }

    reset(top = false) {
        if (!this.center) return;
        this.fitPending = false;
        const halfFov = THREE.MathUtils.degToRad(this.camera.fov / 2);
        const limitingFov = Math.min(halfFov, Math.atan(Math.tan(halfFov) * this.camera.aspect));
        const distance = this.radius / Math.sin(limitingFov) * 1.08;
        this.controls.target.copy(this.center);
        const direction = top ? new THREE.Vector3(0, 1, 0.001) : new THREE.Vector3(0.8, 1.25, 1);
        this.camera.position.copy(this.center).add(direction.normalize().multiplyScalar(distance));
        this.camera.near = Math.max(0.01, distance / 1000);
        this.camera.far = distance * 30;
        this.camera.updateProjectionMatrix();
        this.controls.minDistance = Math.max(0.5, this.radius * 0.1);
        this.controls.maxDistance = distance * 6;
        this.controls.update();
        this.draw();
    }

    panMode(pan) {
        this.controls.mouseButtons.LEFT = pan ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
        this.controls.mouseButtons.RIGHT = pan ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN;
        this.controls.touches.ONE = pan ? THREE.TOUCH.PAN : THREE.TOUCH.ROTATE;
    }

    zoom(factor) {
        const offset = this.camera.position.clone().sub(this.controls.target);
        offset.setLength(THREE.MathUtils.clamp(offset.length() * factor, this.controls.minDistance, this.controls.maxDistance));
        this.camera.position.copy(this.controls.target).add(offset);
        this.controls.update();
        this.draw();
    }

    setWalls(solid) {
        if (!this.wallMaterial) return;
        this.wallMaterial.opacity = solid ? 1 : 0.32;
        this.wallMaterial.transparent = !solid;
        this.wallMaterial.depthWrite = solid;
        this.wallMaterial.needsUpdate = true;
        this.draw();
    }
}

let viewer;
let signature;
function renderArena(scenario = {}, preserveCamera = false) {
    const nextSignature = JSON.stringify(scenario);
    if (signature === nextSignature) return;
    signature = nextSignature;
    renderEquipment(scenario);
    host.dataset.unavailable = "true";
    message.hidden = false;
    if (!scenario.bounds) {
        viewer?.clear();
        message.textContent = "No environment geometry recorded.";
        return;
    }
    try {
        viewer ??= new ArenaViewer();
        const camera = preserveCamera && viewer.center ? { position: viewer.camera.position.clone(), target: viewer.controls.target.clone() } : null;
        viewer.load(scenario);
        if (camera) { viewer.camera.position.copy(camera.position); viewer.controls.target.copy(camera.target); viewer.controls.update(); viewer.draw(); }
        const pan = document.querySelector('[data-camera-mode="pan"]').getAttribute("aria-pressed") === "true";
        viewer.panMode(pan);
        host.dataset.unavailable = "false";
        message.hidden = true;
    } catch (error) {
        message.textContent = error.message.includes("WebGL") ? "3D preview unavailable: WebGL 2 is required." : error.message;
    }
}

globalThis.lucide?.createIcons();
document.querySelectorAll("[data-camera-mode]").forEach((button) => {
    button.addEventListener("click", () => {
        document.querySelectorAll("[data-camera-mode]").forEach((other) => other.setAttribute("aria-pressed", String(other === button)));
        viewer?.panMode(button.dataset.cameraMode === "pan");
    });
});
document.getElementById("arena-zoom-in").addEventListener("click", () => viewer?.zoom(0.8));
document.getElementById("arena-zoom-out").addEventListener("click", () => viewer?.zoom(1.25));
document.getElementById("arena-top").addEventListener("click", () => viewer?.reset(true));
document.getElementById("arena-reset").addEventListener("click", () => viewer?.reset());
document.getElementById("arena-solid-walls").addEventListener("change", (event) => viewer?.setWalls(event.target.checked));
const replaySelect = document.getElementById("replay");
const timeline = document.getElementById("replay-timeline");
const replayStatus = document.getElementById("replay-status");
const arenaState = document.getElementById("arena-state");
let runId = "";
let spawnScenario = {};
let replay = null;
let tick = 0;
let playing = false;
let animation;
let previousTime;
let loadRequest;
let listRequest = 0;
let equipmentTick = -1;

function setPlaying(value) {
    playing = value;
    cancelAnimationFrame(animation);
    previousTime = undefined;
    const button = document.getElementById("replay-play");
    button.setAttribute("aria-label", playing ? "Pause replay" : "Play replay");
    button.title = playing ? "Pause replay" : "Play replay";
    button.replaceChildren();
    const icon = document.createElement("i");
    icon.dataset.lucide = playing ? "pause" : "play";
    button.append(icon);
    globalThis.lucide?.createIcons();
    if (playing) animation = requestAnimationFrame(animateReplay);
}

function enableReplay(enabled) {
    for (const id of ["replay-start", "replay-back", "replay-play", "replay-next", "replay-speed", "replay-loop", "replay-timeline"]) {
        document.getElementById(id).disabled = !enabled;
    }
}

function showTick(value) {
    if (!replay) return;
    const frames = replay.frames;
    tick = Math.max(frames[0].tick, Math.min(frames.at(-1).tick, value));
    const sample = replaySample(frames, tick);
    const players = replayPlayers(replay.header.environment, sample.actors);
    viewer?.setPlayers(players, replay.header.environment.profile);
    if (equipmentTick !== sample.index) {
        renderEquipment({ ...replay.header.environment, players }, true);
        equipmentTick = sample.index;
    }
    timeline.value = String(Math.floor(tick));
    const rate = replay.header.ticks_per_second ?? 20;
    document.getElementById("replay-position").textContent = `Tick ${Math.floor(tick)} / ${frames.at(-1).tick} / ${((tick - frames[0].tick) / rate).toFixed(1)}s`;
    const frame = frames[sample.index];
    replayStatus.textContent = frame.truncated ? "Timeout" : frame.terminated ? "Duel ended" : "Recorded duel";
}

function animateReplay(time) {
    if (!playing || !replay) return;
    const elapsed = previousTime === undefined ? 0 : Math.min((time - previousTime) / 1000, 0.25);
    previousTime = time;
    const first = replay.frames[0].tick;
    const last = replay.frames.at(-1).tick;
    let next = tick + elapsed * (replay.header.ticks_per_second ?? 20) * Number(document.getElementById("replay-speed").value);
    if (next >= last) {
        if (document.getElementById("replay-loop").checked && last > first) next = first + (next - first) % (last - first);
        else { showTick(last); setPlaying(false); return; }
    }
    showTick(next);
    animation = requestAnimationFrame(animateReplay);
}

async function replayJson(url, signal) {
    const response = await fetch(url, { signal, cache: "no-store" });
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.error ?? `Replay request failed (${response.status}).`);
    }
    return response.json();
}

function restoreSpawn(preserveCamera = true) {
    setPlaying(false);
    loadRequest?.abort();
    replay = null;
    equipmentTick = -1;
    enableReplay(false);
    timeline.value = "0";
    document.getElementById("replay-position").textContent = "Tick --";
    arenaState.textContent = "Spawn state";
    renderArena(spawnScenario, preserveCamera);
    renderEquipment(spawnScenario);
    if (viewer && spawnScenario.bounds) viewer.setPlayers(spawnScenario.players ?? [], spawnScenario.profile);
}

async function refreshReplays() {
    const ticket = ++listRequest;
    if (!runId) return;
    try {
        const data = await replayJson(`/api/runs/${encodeURIComponent(runId)}/replays`);
        if (ticket !== listRequest) return;
        const selected = replaySelect.value;
        replaySelect.replaceChildren(new Option("Spawn state", ""), ...data.replays.map((entry) => new Option(entry.name, entry.id)));
        replaySelect.value = data.replays.some((entry) => entry.id === selected) ? selected : "";
        if (selected && !replaySelect.value) restoreSpawn();
        if (!replaySelect.value) replayStatus.textContent = data.replays.length ? `${data.replays.length} recordings available` : "No recordings found for this run.";
    } catch (error) {
        if (ticket === listRequest) replayStatus.textContent = error.message;
    }
}

async function selectReplay() {
    restoreSpawn();
    const identity = replaySelect.value;
    if (!identity) { refreshReplays(); return; }
    const controller = new AbortController();
    loadRequest = controller;
    replayStatus.textContent = "Loading recording...";
    try {
        const data = await replayJson(`/api/runs/${encodeURIComponent(runId)}/replays/${encodeURIComponent(identity)}`, controller.signal);
        if (controller.signal.aborted) return;
        replay = data;
        renderArena(data.header.environment, true);
        const training = data.header.training;
        arenaState.textContent = training ? `Replay / Update ${training.policy_update} / Round ${training.completed_rounds}` : "Replay";
        timeline.min = String(data.frames[0].tick);
        timeline.max = String(data.frames.at(-1).tick);
        equipmentTick = -1;
        enableReplay(true);
        showTick(data.frames[0].tick);
    } catch (error) {
        if (!controller.signal.aborted) { replay = null; replayStatus.textContent = error.message; }
    }
}

function syncRun(scenario = {}, identity = "") {
    const changed = identity !== runId;
    const environmentChanged = JSON.stringify(scenario) !== JSON.stringify(spawnScenario);
    spawnScenario = scenario;
    if (changed) {
        runId = identity;
        listRequest++;
        replaySelect.replaceChildren(new Option("Spawn state", ""));
        restoreSpawn(false);
        replayStatus.textContent = identity ? "Loading recordings..." : "No run selected.";
        refreshReplays();
    } else if (!replay && environmentChanged) renderArena(spawnScenario);
}

document.getElementById("show-world").addEventListener("change", (event) => {
    document.getElementById("world-view").hidden = !event.target.checked;
    if (event.target.checked) viewer?.resize();
    else setPlaying(false);
});
document.getElementById("replay-refresh").addEventListener("click", refreshReplays);
replaySelect.addEventListener("change", selectReplay);
document.getElementById("replay-play").addEventListener("click", () => {
    if (!replay) return;
    if (!playing && tick >= replay.frames.at(-1).tick) showTick(replay.frames[0].tick);
    setPlaying(!playing);
});
document.getElementById("replay-start").addEventListener("click", () => { setPlaying(false); if (replay) showTick(replay.frames[0].tick); });
for (const [id, direction] of [["replay-back", -1], ["replay-next", 1]]) {
    document.getElementById(id).addEventListener("click", () => {
        setPlaying(false);
        if (!replay) return;
        const index = replaySample(replay.frames, tick).index;
        showTick(replay.frames[Math.max(0, Math.min(replay.frames.length - 1, index + direction))].tick);
    });
}
timeline.addEventListener("input", () => { setPlaying(false); showTick(Number(timeline.value)); });
document.addEventListener("visibilitychange", () => { previousTime = undefined; });
document.addEventListener("arena-config", (event) => syncRun(event.detail.scenario, event.detail.runId));
setInterval(() => { if (document.getElementById("live").checked && !document.hidden) refreshReplays(); }, 15000);
syncRun(host.scenario, host.runId);
renderArena(spawnScenario);
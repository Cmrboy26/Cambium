export function itemLabel(item) {
    if (!item || item === "minecraft:air") return "None";
    return item.replace(/^minecraft:/, "").replaceAll("_", " ");
}

export function playerLoadout(player, profile) {
    const fallback = profile === "unarmored_sword_axe_shield"
        ? { weapons: ["minecraft:diamond_sword", "minecraft:diamond_axe"], offhand: "minecraft:shield" }
        : { weapons: ["minecraft:diamond_sword"], offhand: "minecraft:air" };
    return { ...fallback, selected: 0, armor: {}, ...player.loadout };
}

export function replaySample(frames, tick) {
    if (!frames.length) return null;
    let first = 0;
    let last = frames.length - 1;
    while (first < last) {
        const middle = Math.ceil((first + last) / 2);
        if (frames[middle].tick <= tick) first = middle;
        else last = middle - 1;
    }
    const current = frames[first];
    const next = frames[first + 1] ?? current;
    const fraction = next.tick > current.tick ? Math.max(0, Math.min(1, (tick - current.tick) / (next.tick - current.tick))) : 0;
    const projectiles = (current.projectiles ?? []).map((arrow) => {
        const target = (next.projectiles ?? []).find((candidate) => candidate.id === arrow.id
            && candidate.owner === arrow.owner && candidate.age === arrow.age + next.tick - current.tick);
        if (!target || arrow.in_ground || arrow.in_ground !== target.in_ground
            || next.bridge?.round_reset || next.bridge?.respawn?.[arrow.owner]) return { ...arrow };
        return { ...arrow, position: arrow.position.map((value, axis) =>
            value + (target.position[axis] - value) * fraction) };
    });
    return { index: first, projectiles, actors: current.actors.map((actor, index) => {
        const target = next.actors[index];
        if (next.bridge?.round_reset || next.bridge?.respawn?.[actor.id ?? index]) return { ...actor };
        const turn = ((target.yaw - actor.yaw + 180) % 360 + 360) % 360 - 180;
        return { ...actor, position: actor.position.map((value, axis) => value + (target.position[axis] - value) * fraction),
            yaw: actor.yaw + turn * fraction, pitch: actor.pitch + (target.pitch - actor.pitch) * fraction };
    }) };
}

export function bridgeTerrain(frames) {
    let index = -1;
    const overlay = new Map();
    let snapshot = [];
    return (target) => {
        let changed = false;
        if (target < index) { overlay.clear(); index = -1; changed = true; }
        while (index < target) {
            const state = frames[++index].bridge;
            for (const block of state.initial_terrain ?? []) {
                overlay.set(block.position.join(","), block);
                changed = true;
            }
            if (state.terrain_edits.length || state.clear_overlay) changed = true;
            for (const edit of state.terrain_edits) {
                const key = edit.position.join(",");
                if (edit.action === "place") overlay.set(key, { position: edit.position, player: edit.player });
                else overlay.delete(key);
            }
            if (state.clear_overlay) overlay.clear();
        }
        if (changed) snapshot = [...overlay.values()];
        return snapshot;
    };
}

export function replayPlayers(scenario, actors, bridge = null) {
    return actors.map((actor, index) => {
        const initial = scenario.players?.[index] ?? {};
        const loadout = playerLoadout(initial, scenario.profile);
        const slots = bridge?.kits?.[index]?.slots;
        const weapons = Array.isArray(slots) ? slots.map((slot, position) => ({
            blocks: "minecraft:blocks", pickaxe: "minecraft:pickaxe", golden_apple: "minecraft:golden_apple",
            empty: "minecraft:air",
        })[slot] ?? loadout.weapons[position] ?? "minecraft:air")
            : actor.weapon && !loadout.weapons.includes(actor.weapon) ? [...loadout.weapons, actor.weapon] : loadout.weapons;
        const selected = Array.isArray(slots) && Number.isInteger(actor.weapon_slot)
            && actor.weapon_slot >= 0 && actor.weapon_slot < weapons.length ? actor.weapon_slot
            : actor.weapon ? weapons.indexOf(actor.weapon) : loadout.selected;
        return { ...initial, ...actor, loadout: { ...loadout, weapons,
            selected,
            armor: actor.armor ?? loadout.armor,
            offhand: actor.shield_equipped === undefined ? loadout.offhand : actor.shield_equipped ? "minecraft:shield" : "minecraft:air" } };
    });
}

export function buildVoxels(scenario) {
    const minimum = scenario?.bounds?.min;
    const size = scenario?.bounds?.size;
    if (![minimum, size].every((vector) => Array.isArray(vector) && vector.length === 3 && vector.every(Number.isInteger))
        || size.some((value) => value <= 0)) throw new Error("Arena bounds unavailable.");
    if (size.reduce((volume, value) => volume * value, 1) > 262144) throw new Error("Arena exceeds the preview size limit.");
    const maximum = minimum.map((value, axis) => value + size[axis] - 1);
    const occupied = new Map();
    for (const fill of scenario.blocks ?? []) {
        if (![fill.from, fill.to].every((vector) => Array.isArray(vector) && vector.length === 3 && vector.every(Number.isInteger))) {
            throw new Error("Invalid arena block fill.");
        }
        if (!["minecraft:air", "minecraft:stone"].includes(fill.block)) throw new Error("Unsupported arena block.");
        const start = fill.from.map((value, axis) => Math.max(value, minimum[axis]));
        const end = fill.to.map((value, axis) => Math.min(value, maximum[axis]));
        for (let blockX = start[0]; blockX <= end[0]; blockX++) {
            for (let blockY = start[1]; blockY <= end[1]; blockY++) {
                for (let blockZ = start[2]; blockZ <= end[2]; blockZ++) {
                    const key = `${blockX},${blockY},${blockZ}`;
                    if (fill.block === "minecraft:air") occupied.delete(key);
                    else occupied.set(key, [blockX, blockY, blockZ]);
                }
            }
        }
    }
    const neighbors = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
    return [...occupied.values()].filter((position) => neighbors.some((offset) =>
        !occupied.has(position.map((value, axis) => value + offset[axis]).join(","))));
}
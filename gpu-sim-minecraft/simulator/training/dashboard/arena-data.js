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
    return { index: first, actors: current.actors.map((actor, index) => {
        const target = next.actors[index];
        const turn = ((target.yaw - actor.yaw + 180) % 360 + 360) % 360 - 180;
        return { ...actor, position: actor.position.map((value, axis) => value + (target.position[axis] - value) * fraction),
            yaw: actor.yaw + turn * fraction, pitch: actor.pitch + (target.pitch - actor.pitch) * fraction };
    }) };
}

export function replayPlayers(scenario, actors) {
    return actors.map((actor, index) => {
        const initial = scenario.players?.[index] ?? {};
        const loadout = playerLoadout(initial, scenario.profile);
        const weapons = actor.weapon && !loadout.weapons.includes(actor.weapon) ? [...loadout.weapons, actor.weapon] : loadout.weapons;
        return { ...initial, ...actor, loadout: { ...loadout, weapons,
            selected: actor.weapon ? weapons.indexOf(actor.weapon) : loadout.selected,
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
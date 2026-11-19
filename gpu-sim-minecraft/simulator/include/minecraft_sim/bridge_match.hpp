#pragma once

#include "minecraft_sim/bridge_building.hpp"
#include "minecraft_sim/projectiles.hpp"

namespace minecraft_sim {

enum class BridgeOutcome : std::uint8_t { ongoing, team0, team1, draw };
enum class BridgeHotbarItem : std::uint8_t { empty, sword, bow, pickaxe, blocks, golden_apple };

struct BridgeKitConfig {
    int blocks = 0;
    int golden_apples = 0;
    bool explicit_hotbar = false;
    int slot_count = 0;
    BridgeHotbarItem slots[9]{};
};

struct BridgePlayerKit {
    int blocks = 0;
    int golden_apples = 0;
    int arrows = 0;
    int arrow_refill_elapsed = 0;
    int selected_slot = 0;
    BridgeBuildState building;
};

struct BridgeMatchRules {
    Aabb goals[2]{};
    double void_y = -64.0;
    int goals_to_win = 5;
    int max_ticks = 18000;
    int cage_ticks = 60;
    int arrow_refill_ticks = 0;
    BridgeKitConfig kits[2]{};
    BridgeBuildRules building;
    bool persist_blocks_on_goal = true;
};

struct BridgeMatchState {
    int scores[2]{};
    int ticks = 0;
    int cage_remaining = 0;
    BridgePlayerKit kits[2]{};
    std::uint64_t episodes = 0;
    BridgeOutcome outcome = BridgeOutcome::ongoing;
};

struct BridgeInitialPlayers {
    PlayerState players[2]{};
};

struct BridgeTickEvents {
    bool advanced = false;
    bool died[2]{};
    bool scored[2]{};
    bool respawn[2]{};
    bool placed[2]{};
    bool broke[2]{};
    TerrainEdit terrain_edits[2]{};
    bool round_reset = false;
    bool terminated = false;
    BridgeOutcome outcome = BridgeOutcome::ongoing;
};

MCSIM_HD inline BridgeMatchState initialBridgeMatch(const BridgeMatchRules& rules) {
    BridgeMatchState state;
    state.cage_remaining = rules.cage_ticks;
    return state;
}

MCSIM_HD inline void resetBridgeKit(BridgeMatchState& state, const BridgeMatchRules& rules,
                                    const PlayerState& initial, int player) {
    state.kits[player] = {rules.kits[player].blocks, rules.kits[player].golden_apples,
                          initial.ordinary_arrow_count, 0, initial.selected_weapon_slot, {}};
}

// Custom refill ordering: docs/bridge_mode_v1.md, Provisional Rules and Reset Matrix.
MCSIM_HD inline void tickBridgeKit(BridgePlayerKit& kit, PlayerState& player, BowState& bow,
                                   int arrows_before_tick, int refill_ticks) {
    if (bow.ordinary_arrows > 0 || refill_ticks <= 0 || arrows_before_tick > 0) {
        kit.arrow_refill_elapsed = 0;
    } else if (++kit.arrow_refill_elapsed >= refill_ticks) {
        bow.ordinary_arrows = 1;
        kit.arrow_refill_elapsed = 0;
    }
    player.ordinary_arrow_count = bow.ordinary_arrows;
    kit.arrows = bow.ordinary_arrows;
    kit.selected_slot = player.selected_weapon_slot;
}

MCSIM_HD inline Controls bridgeControls(const BridgeMatchState& state, Controls inputs) {
    if (state.cage_remaining <= 0) return inputs;
    Controls locked;
    locked.yaw_delta = inputs.yaw_delta;
    locked.pitch_delta = inputs.pitch_delta;
    return locked;
}

MCSIM_HD inline BridgeTickEvents finishBridgeTick(const BridgeMatchRules& rules, BridgeMatchState& state,
                                                  const PlayerState (&players)[2]) {
    BridgeTickEvents events;
    if (state.outcome != BridgeOutcome::ongoing || rules.goals_to_win <= 0
        || rules.max_ticks <= 0 || rules.cage_ticks < 0 || rules.arrow_refill_ticks < 0) return events;
    events.advanced = true;
    for (int player = 0; player < 2; ++player) {
        events.died[player] = !players[player].alive() || players[player].position.y < rules.void_y;
        events.scored[player] = !events.died[player] && players[player].bounds().intersects(rules.goals[1 - player]);
        if (events.scored[player]) ++state.scores[player];
    }
    ++state.ticks;
    if (state.cage_remaining > 0) --state.cage_remaining;

    const bool first_won = state.scores[0] >= rules.goals_to_win;
    const bool second_won = state.scores[1] >= rules.goals_to_win;
    if (first_won && second_won) state.outcome = BridgeOutcome::draw;
    else if (first_won) state.outcome = BridgeOutcome::team0;
    else if (second_won) state.outcome = BridgeOutcome::team1;
    else if (state.ticks >= rules.max_ticks) {
        state.outcome = state.scores[0] == state.scores[1] ? BridgeOutcome::draw
            : state.scores[0] > state.scores[1] ? BridgeOutcome::team0 : BridgeOutcome::team1;
    }

    events.outcome = state.outcome;
    events.terminated = state.outcome != BridgeOutcome::ongoing;
    if (events.terminated) return events;
    events.round_reset = events.scored[0] || events.scored[1];
    if (events.round_reset) state.cage_remaining = rules.cage_ticks;
    else {
        events.respawn[0] = events.died[0];
        events.respawn[1] = events.died[1];
    }
    return events;
}

MCSIM_HD inline void clearBridgePlayerProjectiles(ProjectileWorldState& projectiles, int player) {
    for (int slot = 0; projectiles.arrows && slot < projectiles.arrow_capacity; ++slot) {
        if (projectiles.arrows[slot].owner == player) projectiles.arrows[slot] = {};
    }
    for (int slot = 0; projectiles.wind_charges && slot < projectiles.wind_charge_capacity; ++slot) {
        if (projectiles.wind_charges[slot].owner == player) projectiles.wind_charges[slot] = {};
    }
    for (int slot = 0; projectiles.thrown_items && slot < projectiles.thrown_item_capacity; ++slot) {
        if (projectiles.thrown_items[slot].owner == player) projectiles.thrown_items[slot] = {};
    }
}

MCSIM_HD inline void restoreBridgePlayer(WorldView world, ProjectileWorldState& projectiles,
                                         const PlayerState& initial, int player) {
    auto& current = world.player(player).state();
    const auto random = current.random;
    current = initial;
    current.random = random;
    world.clearInputs(player);
    projectiles.bows[player] = {};
    projectiles.bows[player].ordinary_arrows = current.ordinary_arrow_count;
}

MCSIM_HD inline void respawnBridgePlayer(WorldView world, ProjectileWorldState& projectiles,
                                         const PlayerState& initial, int player) {
    clearBridgePlayerProjectiles(projectiles, player);
    restoreBridgePlayer(world, projectiles, initial, player);
}

MCSIM_HD inline void resetBridgeRound(WorldView world, ProjectileWorldState& projectiles,
                                      const PlayerState (&initial)[2], bool persist_blocks_on_goal) {
    const auto random = projectiles.random;
    resetProjectileWorld(projectiles, 0);
    projectiles.random = random;
    for (int player = 0; player < 2; ++player) restoreBridgePlayer(world, projectiles, initial[player], player);
    world.blocks().resetPlacedBlocks(TerrainReset::round, persist_blocks_on_goal);
}

MCSIM_HD inline void resetBridgeMatch(WorldView world, ProjectileWorldState& projectiles,
                                      const PlayerState (&initial)[2], const BridgeMatchRules& rules,
                                      BridgeMatchState& state, std::uint64_t seed) {
    const auto episode = state.episodes;
    state = initialBridgeMatch(rules);
    state.episodes = episode + 1;
    const auto episode_seed = seed + episode * 1000033ULL;
    const auto previous_world = world.state();
    world.mutableState() = {};
    world.mutableState().natural_regeneration = previous_world.natural_regeneration;
    world.mutableState().simulate_food = previous_world.simulate_food;
    world.mutableState().simulate_fall_damage = previous_world.simulate_fall_damage;
    world.mutableState().difficulty = previous_world.difficulty;
    resetProjectileWorld(projectiles, episode_seed);
    for (int player = 0; player < 2; ++player) {
        restoreBridgePlayer(world, projectiles, initial[player], player);
        resetBridgeKit(state, rules, initial[player], player);
        world.player(player).state().random.setSeed(episode_seed + std::uint64_t(player) * 1000003ULL);
    }
    world.blocks().resetPlacedBlocks(TerrainReset::match);
}

MCSIM_HD inline BridgeTickEvents finishBridgeWorldTick(WorldView world, ProjectileWorldState& projectiles,
                                                       const PlayerState (&initial)[2],
                                                       const BridgeMatchRules& rules, BridgeMatchState& state,
                                                       bool preserve_final_state = false) {
    if (world.playerCount() != 2) return BridgeTickEvents();
    const PlayerState players[2]{world.player(0).state(), world.player(1).state()};
    const int previous_cage = state.cage_remaining;
    auto events = finishBridgeTick(rules, state, players);
    if (!events.advanced || events.terminated) return events;
    if (preserve_final_state) {
        if (events.round_reset) state.cage_remaining = previous_cage > 0 ? previous_cage - 1 : 0;
        events.round_reset = false;
        events.respawn[0] = false;
        events.respawn[1] = false;
        return events;
    }
    if (events.round_reset) {
        resetBridgeRound(world, projectiles, initial, rules.persist_blocks_on_goal);
        for (int player = 0; player < 2; ++player) resetBridgeKit(state, rules, initial[player], player);
    }
    else {
        for (int player = 0; player < 2; ++player) {
            if (events.respawn[player]) {
                respawnBridgePlayer(world, projectiles, initial[player], player);
                resetBridgeKit(state, rules, initial[player], player);
            }
        }
    }
    return events;
}

MCSIM_HD inline BridgeTickEvents processBridgeWorldTick(WorldView world, ProjectileWorldState& projectiles,
                                                        const PlayerState (&initial)[2],
                                                        const BridgeMatchRules& rules, BridgeMatchState& state,
                                                        int external_cutoff = 0) {
    if (world.playerCount() != 2 || state.outcome != BridgeOutcome::ongoing || rules.goals_to_win <= 0
        || rules.max_ticks <= 0 || rules.cage_ticks < 0 || rules.arrow_refill_ticks < 0
        || !validBridgeBuildRules(rules.building) || (rules.building.enabled && !world.blocks().overlay)) return BridgeTickEvents();
    const bool caged = state.cage_remaining > 0;
    const bool movement_locked[2]{caged, caged};
    Vec3 locked_positions[2]{};
    int arrows_before_tick[2]{};
    bool placed[2]{}, broken[2]{};
    TerrainEdit edits[2]{};
    // Bridge action ordering: docs/bridge_mode_v1.md, Tick And Outcome Ordering.
    // Minecraft 26.3: Minecraft.startUseItem/handleKeybinds (1698-1700, 1972-1973).
    for (int player = 0; player < 2; ++player) {
        arrows_before_tick[player] = projectiles.bows[player].ordinary_arrows;
        locked_positions[player] = world.player(player).state().position;
        auto controls = bridgeControls(state, world.queuedInput(player));
        const auto& kit = rules.kits[player];
        auto& building = state.kits[player].building;
        if (building.use_delay > 0) --building.use_delay;
        const int slot = controls.select_weapon ? controls.selected_slot : world.player(player).state().selected_weapon_slot;
        if (kit.explicit_hotbar) {
            if (slot < 0 || slot >= kit.slot_count) {
                controls.select_weapon = false;
                controls.attack = false;
                controls.use_item = false;
                cancelBridgeBreaking(building);
            } else if (kit.slots[slot] == BridgeHotbarItem::pickaxe || kit.slots[slot] == BridgeHotbarItem::blocks) {
                auto source = world.player(player).state();
                source.yaw += controls.yaw_delta;
                source.pitch = static_cast<float>(maximum(-90.0, minimum(90.0, source.pitch + controls.pitch_delta)));
                if (kit.slots[slot] == BridgeHotbarItem::blocks) {
                    cancelBridgeBreaking(building);
                    if (rules.building.enabled && controls.use_item && building.use_delay == 0) {
                        building.use_delay = 4;
                        const auto target = bridgeInteractionTarget(world, player, source,
                            rules.building.block_range, rules.building.entity_range);
                        placed[player] = placeBridgeBlock(world, target, rules.building, player,
                                                          state.kits[player].blocks, edits[player]);
                    }
                } else if (rules.building.enabled && controls.attack) {
                    const auto target = bridgeInteractionTarget(world, player, source,
                        rules.building.block_range, rules.building.entity_range);
                    broken[player] = breakBridgeBlock(world, target, rules.building, building, player);
                    if (broken[player]) edits[player] = {TerrainEditAction::remove,
                        target.block.x, target.block.y, target.block.z};
                } else {
                    cancelBridgeBreaking(building);
                }
                controls.attack = false;
                controls.use_item = false;
            } else {
                cancelBridgeBreaking(building);
                if (kit.slots[slot] == BridgeHotbarItem::golden_apple) {
                    controls.attack = false;
                    controls.use_item = false;
                } else if (rules.building.enabled && controls.use_item) {
                    const auto actor = world.player(player);
                    const auto& player_state = actor.state();
                    const bool same_slot = !controls.select_weapon || controls.selected_slot == player_state.selected_weapon_slot;
                    const bool ongoing = same_slot && (projectiles.bows[player].using_bow || player_state.using_shield);
                    if (!ongoing && building.use_delay > 0) controls.use_item = false;
                    else if (!ongoing) building.use_delay = 4;
                }
            }
        }
        world.player(player).setInputs(controls);
        if (caged && projectiles.bows[player].using_bow) {
            processBowCommand(world, world.math(), player, BowCommand::cancel, projectiles);
        }
    }
    processWorldTick(world, projectiles, movement_locked);
    if (caged) {
        for (int player = 0; player < 2; ++player) {
            world.player(player).state().position = locked_positions[player];
            world.player(player).state().velocity = {};
        }
    }
    for (int player = 0; player < 2; ++player) {
        tickBridgeKit(state.kits[player], world.player(player).state(), projectiles.bows[player],
                      arrows_before_tick[player], rules.arrow_refill_ticks);
    }
    const bool preserve_final_state = external_cutoff > 0 && state.ticks >= external_cutoff - 1;
    auto events = finishBridgeWorldTick(world, projectiles, initial, rules, state, preserve_final_state);
    for (int player = 0; player < 2; ++player) {
        events.placed[player] = placed[player];
        events.broke[player] = broken[player];
        events.terrain_edits[player] = edits[player];
    }
    return events;
}

}
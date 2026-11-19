#pragma once

#include "minecraft_sim/bridge_match.hpp"

namespace bridge_match_checks {

MCSIM_HD inline bool outcomes() {
    using namespace minecraft_sim;
    BridgeMatchRules rules;
    rules.goals[0] = {{-5.0, 0.0, -1.0}, {-4.0, 2.0, 1.0}};
    rules.goals[1] = {{4.0, 0.0, -1.0}, {5.0, 2.0, 1.0}};
    rules.cage_ticks = 2;
    PlayerState players[2]{};
    auto state = initialBridgeMatch(rules);
    Controls inputs;
    inputs.forward = true;
    inputs.attack = true;
    inputs.use_item = true;
    inputs.select_weapon = true;
    inputs.yaw_delta = 12.0F;
    inputs.pitch_delta = -4.0F;
    auto filtered = bridgeControls(state, inputs);
    if (filtered.forward || filtered.attack || filtered.use_item || filtered.select_weapon
        || filtered.yaw_delta != 12.0F || filtered.pitch_delta != -4.0F) return false;
    auto events = finishBridgeTick(rules, state, players);
    if (!events.advanced || events.terminated || state.ticks != 1 || state.cage_remaining != 1) return false;
    events = finishBridgeTick(rules, state, players);
    if (state.cage_remaining != 0 || !bridgeControls(state, inputs).forward) return false;

    players[0].position.x = 4.5;
    players[1].position.x = -4.5;
    events = finishBridgeTick(rules, state, players);
    if (!events.scored[0] || !events.scored[1] || !events.round_reset || events.terminated
        || state.scores[0] != 1 || state.scores[1] != 1 || state.cage_remaining != 2) return false;

    players[0].health = 0.0F;
    events = finishBridgeTick(rules, state, players);
    if (!events.died[0] || events.scored[0] || !events.scored[1] || !events.round_reset
        || events.respawn[0] || state.scores[0] != 1 || state.scores[1] != 2) return false;

    players[0].health = 20.0F;
    state = initialBridgeMatch(rules);
    state.scores[0] = 4;
    state.scores[1] = 4;
    state.ticks = rules.max_ticks - 1;
    events = finishBridgeTick(rules, state, players);
    if (!events.terminated || events.round_reset || events.outcome != BridgeOutcome::draw
        || state.scores[0] != 5 || state.scores[1] != 5) return false;
    if (finishBridgeTick(rules, state, players).advanced) return false;

    state = initialBridgeMatch(rules);
    state.scores[0] = 4;
    state.ticks = rules.max_ticks - 1;
    players[1].position.x = 0.0;
    events = finishBridgeTick(rules, state, players);
    if (!events.terminated || events.outcome != BridgeOutcome::team0 || events.round_reset) return false;

    state = initialBridgeMatch(rules);
    state.scores[1] = 2;
    state.ticks = rules.max_ticks - 1;
    players[0].position.x = 0.0;
    events = finishBridgeTick(rules, state, players);
    if (!events.terminated || events.outcome != BridgeOutcome::team1) return false;

    state = initialBridgeMatch(rules);
    state.ticks = rules.max_ticks - 1;
    events = finishBridgeTick(rules, state, players);
    if (!events.terminated || events.outcome != BridgeOutcome::draw) return false;

    state = initialBridgeMatch(rules);
    players[0].health = 0.0F;
    events = finishBridgeTick(rules, state, players);
    if (!events.died[0] || !events.respawn[0] || events.respawn[1] || events.round_reset
        || events.terminated || state.cage_remaining != 1) return false;

    state = initialBridgeMatch(rules);
    players[0].health = 20.0F;
    players[0].position.x = 4.5;
    rules.void_y = 0.5;
    players[1].position.y = 1.0;
    events = finishBridgeTick(rules, state, players);
    if (!events.died[0] || events.scored[0] || !events.respawn[0]) return false;

    state = initialBridgeMatch(rules);
    rules.void_y = -64.0;
    players[0].position.x = 4.0 - double(0.6F) / 2.0;
    events = finishBridgeTick(rules, state, players);
    if (events.scored[0] || events.round_reset) return false;

    rules.goals_to_win = 0;
    return !finishBridgeTick(rules, state, players).advanced;
}

MCSIM_HD inline bool lifecycle() {
    using namespace minecraft_sim;
    Block base[3]{Block::solid, Block::air, Block::air};
    PlacedBlock overlay[3]{};
    BlockWorldView blocks{base, 3, 1, 1, -1, 0, 0, overlay};
    PlayerState players[2]{};
    PlayerState initial[2]{};
    initial[0].position = {0.0, 1.0, 0.0};
    initial[1].position = {0.6, 1.0, 0.0};
    initial[0].ordinary_arrow_count = 1;
    initial[1].ordinary_arrow_count = 1;
    Controls queued[2]{}, held[2]{};
    WorldState world_state;
    world_state.simulate_fall_damage = true;
    WorldView world(players, queued, held, &world_state, 2, blocks, {});
    ArrowState arrows[2]{};
    WindChargeState wind[2]{};
    ThrownProjectileState thrown[2]{};
    ProjectileWorldState projectiles;
    projectiles.arrows = arrows;
    projectiles.arrow_capacity = 2;
    projectiles.wind_charges = wind;
    projectiles.wind_charge_capacity = 2;
    projectiles.thrown_items = thrown;
    projectiles.thrown_item_capacity = 2;
    BridgeMatchRules rules;
    rules.goals[0] = {{-5.0, 0.0, -1.0}, {-4.0, 3.0, 1.0}};
    rules.goals[1] = {{4.0, 0.0, -1.0}, {5.0, 3.0, 1.0}};
    rules.cage_ticks = 2;
    BridgeMatchState state;
    resetBridgeMatch(world, projectiles, initial, rules, state, 123);
    if (state.episodes != 1 || state.cage_remaining != 2 || !world_state.simulate_fall_damage
        || projectiles.bows[0].ordinary_arrows != 1) return false;
    const auto first_seed = players[0].random.state;
    players[0].random.next(5);
    projectiles.random.next(5);
    const auto player_random = players[0].random.state;
    const auto projectile_random = projectiles.random.state;
    arrows[0].active = true;
    arrows[0].owner = 0;
    arrows[1].active = true;
    arrows[1].owner = 1;
    wind[0].active = true;
    wind[0].owner = 0;
    wind[1].active = true;
    wind[1].owner = 1;
    thrown[0].active = true;
    thrown[0].owner = 0;
    thrown[1].active = true;
    thrown[1].owner = 1;
    players[0].health = 0.0F;
    players[1].health = 13.0F;
    queued[0].forward = true;
    held[0].attack = true;
    auto events = finishBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.respawn[0] || events.respawn[1] || players[0].health != 20.0F || players[1].health != 13.0F
        || players[0].random.state != player_random || projectiles.random.state != projectile_random
        || queued[0].forward || held[0].attack || arrows[0].active || !arrows[1].active
        || wind[0].active || !wind[1].active || thrown[0].active || !thrown[1].active
        || projectiles.bows[0].ordinary_arrows != 1) return false;

    if (!blocks.setPlacedBlock(0, 0, 0, PlacedBlock::team0)
        || !blocks.setPlacedBlock(1, 0, 0, PlacedBlock::team1)) return false;
    players[0].position = {4.5, 1.0, 0.0};
    events = finishBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.round_reset || state.scores[0] != 1 || state.ticks != 2 || state.cage_remaining != 2
        || players[0].position.x != initial[0].position.x || players[1].health != 20.0F
        || arrows[1].active || wind[1].active || thrown[1].active
        || players[0].random.state != player_random || projectiles.random.state != projectile_random
        || overlay[1] != PlacedBlock::team0 || overlay[2] != PlacedBlock::team1) return false;

    rules.persist_blocks_on_goal = false;
    players[0].position = {4.5, 1.0, 0.0};
    events = finishBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.round_reset || state.scores[0] != 2 || state.ticks != 3
        || overlay[1] != PlacedBlock::empty || overlay[2] != PlacedBlock::empty || !blocks.solid(-1, 0, 0)) return false;

    state.scores[0] = 4;
    players[0].position = {4.5, 1.0, 0.0};
    arrows[0].active = true;
    arrows[0].owner = 0;
    if (!blocks.setPlacedBlock(0, 0, 0, PlacedBlock::team0)) return false;
    events = finishBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.terminated || events.outcome != BridgeOutcome::team0 || events.round_reset
        || players[0].position.x != 4.5 || !arrows[0].active || overlay[1] != PlacedBlock::team0) return false;

    resetBridgeMatch(world, projectiles, initial, rules, state, 123);
    return state.episodes == 2 && state.scores[0] == 0 && state.ticks == 0 && state.cage_remaining == 2
        && players[0].random.state != first_seed && !arrows[0].active && overlay[1] == PlacedBlock::empty
        && blocks.solid(-1, 0, 0) && world_state.simulate_fall_damage;
}

MCSIM_HD inline bool selectedWorldReset() {
    using namespace minecraft_sim;
    Block base[2]{Block::solid, Block::air};
    PlacedBlock overlays[2][2]{};
    PlayerState players[2][2]{};
    PlayerState initial[2]{};
    Controls queued[2][2]{}, held[2][2]{};
    WorldState worlds[2]{};
    ProjectileWorldState projectiles[2]{};
    BridgeMatchState matches[2]{};
    BridgeMatchRules rules;
    BlockWorldView first_blocks{base, 2, 1, 1, 0, 0, 0, overlays[0]};
    BlockWorldView second_blocks{base, 2, 1, 1, 0, 0, 0, overlays[1]};
    WorldView first(players[0], queued[0], held[0], &worlds[0], 2, first_blocks, {});
    WorldView second(players[1], queued[1], held[1], &worlds[1], 2, second_blocks, {});
    if (!first_blocks.setPlacedBlock(1, 0, 0, PlacedBlock::team0)
        || !second_blocks.setPlacedBlock(1, 0, 0, PlacedBlock::team1)) return false;
    players[1][0].health = 7.0F;
    players[1][0].random.setSeed(77);
    const auto second_random = players[1][0].random.state;
    queued[1][0].forward = true;
    worlds[1].tick = 23;
    matches[1].scores[0] = 3;
    matches[1].ticks = 15;
    matches[1].episodes = 4;
    resetBridgeMatch(first, projectiles[0], initial, rules, matches[0], 123);
    return !first_blocks.solid(1, 0, 0) && second_blocks.solid(1, 0, 0)
        && overlays[1][1] == PlacedBlock::team1 && players[1][0].health == 7.0F
        && players[1][0].random.state == second_random && second.queuedInput(0).forward
        && worlds[1].tick == 23 && matches[1].scores[0] == 3 && matches[1].ticks == 15
        && matches[1].episodes == 4;
}

MCSIM_HD inline bool tick(minecraft_sim::MathView math) {
    using namespace minecraft_sim;
    PlayerState players[2]{};
    PlayerState initial[2]{};
    initial[0].position = {0.0, 10.0, 0.0};
    initial[1].position = {5.0, 10.0, 0.0};
    Controls queued[2]{}, held[2]{};
    WorldState world_state;
    WorldView world(players, queued, held, &world_state, 2, {}, math);
    ProjectileWorldState projectiles;
    BridgeMatchRules rules;
    rules.goals[0] = {{-10.0, 0.0, -1.0}, {-9.0, 2.0, 1.0}};
    rules.goals[1] = {{10.0, 0.0, -1.0}, {11.0, 2.0, 1.0}};
    rules.cage_ticks = 2;
    BridgeMatchState state;
    resetBridgeMatch(world, projectiles, initial, rules, state, 123);
    players[0].velocity = {1.0, -1.0, 1.0};
    Controls inputs;
    inputs.forward = true;
    inputs.jump = true;
    inputs.attack = true;
    inputs.use_item = true;
    inputs.select_weapon = true;
    inputs.weapon = WeaponItem::diamond_axe;
    inputs.yaw_delta = 20.0F;
    inputs.pitch_delta = 5.0F;
    world.player(0).setInputs(inputs);
    auto events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.advanced || state.cage_remaining != 1 || players[0].position.x != 0.0
        || players[0].position.y != 10.0 || players[0].position.z != 0.0
        || players[0].velocity.lengthSquared() != 0.0 || players[0].yaw != 20.0F
        || players[0].pitch != 5.0F || players[0].weapon != initial[0].weapon
        || players[0].last_attack.attempted) return false;
    inputs.yaw_delta = -5.0F;
    inputs.pitch_delta = 0.0F;
    world.player(0).setInputs(inputs);
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.advanced || state.cage_remaining != 0 || players[0].position.y != 10.0
        || players[0].yaw != 15.0F || players[0].last_attack.attempted) return false;
    inputs = {};
    inputs.forward = true;
    world.player(0).setInputs(inputs);
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.advanced || (players[0].position.x == 0.0 && players[0].position.z == 0.0)
        || world_state.tick != 3 || state.ticks != 3) return false;
    state.outcome = BridgeOutcome::draw;
    return !processBridgeWorldTick(world, projectiles, initial, rules, state).advanced && world_state.tick == 3;
}

MCSIM_HD inline bool completeMatch(minecraft_sim::MathView math) {
    using namespace minecraft_sim;
    PlayerState players[2]{};
    PlayerState initial[2]{};
    initial[0].position = {0.0, 1.0, 0.0};
    initial[1].position = {-2.0, 1.0, 0.0};
    Controls queued[2]{}, held[2]{};
    WorldState world_state;
    WorldView world(players, queued, held, &world_state, 2, {}, math);
    ProjectileWorldState projectiles;
    BridgeMatchRules rules;
    rules.goals[0] = {{-5.0, 0.0, -1.0}, {-4.0, 3.0, 1.0}};
    rules.goals[1] = {{4.0, 0.0, -1.0}, {5.0, 3.0, 1.0}};
    rules.goals_to_win = 2;
    rules.cage_ticks = 1;
    BridgeMatchState state;
    resetBridgeMatch(world, projectiles, initial, rules, state, 123);
    auto events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.advanced || state.cage_remaining != 0 || state.ticks != 1) return false;
    players[0].position = {4.5, 1.0, 0.0};
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.scored[0] || !events.round_reset || events.terminated || state.scores[0] != 1
        || state.cage_remaining != 1 || players[0].position.x != initial[0].position.x) return false;
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.advanced || state.cage_remaining != 0 || state.ticks != 3) return false;
    players[0].position = {4.5, 1.0, 0.0};
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    return events.scored[0] && events.terminated && !events.round_reset
        && events.outcome == BridgeOutcome::team0 && state.scores[0] == 2
        && state.ticks == 4 && players[0].position.x == 4.5;
}

MCSIM_HD inline bool kitsAndRefill(minecraft_sim::MathView math) {
    using namespace minecraft_sim;
    PlayerState players[2]{}, initial[2]{};
    initial[0].position = {0.0, 10.0, 0.0};
    initial[1].position = {5.0, 10.0, 0.0};
    initial[0].weapon = WeaponItem::bow;
    initial[0].ordinary_arrow_count = 1;
    initial[1].ordinary_arrow_count = 1;
    Controls queued[2]{}, held[2]{};
    WorldState world_state;
    WorldView world(players, queued, held, &world_state, 2, {}, math);
    ArrowState arrows[4]{};
    ProjectileWorldState projectiles;
    projectiles.arrows = arrows;
    projectiles.arrow_capacity = 4;
    BridgeMatchRules rules;
    rules.goals[0] = {{-11.0, 0.0, -1.0}, {-10.0, 30.0, 1.0}};
    rules.goals[1] = {{10.0, 0.0, -1.0}, {11.0, 30.0, 1.0}};
    rules.void_y = -1000.0;
    rules.cage_ticks = 0;
    rules.arrow_refill_ticks = 3;
    rules.kits[0].blocks = 64;
    rules.kits[0].golden_apples = 2;
    rules.kits[0].explicit_hotbar = true;
    rules.kits[0].slot_count = 4;
    rules.kits[0].slots[0] = BridgeHotbarItem::bow;
    rules.kits[0].slots[1] = BridgeHotbarItem::pickaxe;
    rules.kits[0].slots[2] = BridgeHotbarItem::blocks;
    rules.kits[0].slots[3] = BridgeHotbarItem::golden_apple;
    rules.kits[1].blocks = 20;
    BridgeMatchState state;
    resetBridgeMatch(world, projectiles, initial, rules, state, 123);
    if (state.kits[0].blocks != 64 || state.kits[0].golden_apples != 2 || state.kits[0].arrows != 1) return false;
    Controls inputs;
    inputs.select_weapon = true;
    inputs.selected_slot = 1;
    inputs.weapon = WeaponItem::unarmed;
    inputs.attack = true;
    inputs.use_item = true;
    world.player(0).setInputs(inputs);
    if (!processBridgeWorldTick(world, projectiles, initial, rules, state).advanced
        || players[0].last_attack.attempted || players[0].using_shield || state.kits[0].selected_slot != 1
        || state.kits[0].blocks != 64 || state.kits[0].golden_apples != 2) return false;
    inputs = {};
    inputs.select_weapon = true;
    inputs.selected_slot = 0;
    inputs.weapon = WeaponItem::bow;
    inputs.use_item = true;
    for (int tick = 0; tick < 3; ++tick) {
        world.player(0).setInputs(inputs);
        if (!processBridgeWorldTick(world, projectiles, initial, rules, state).advanced) return false;
        inputs.select_weapon = false;
    }
    world.player(0).setInputs({});
    if (!processBridgeWorldTick(world, projectiles, initial, rules, state).advanced
        || state.kits[0].arrows != 0 || state.kits[0].arrow_refill_elapsed != 0
        || players[0].ordinary_arrow_count != 0) return false;
    for (int tick = 1; tick <= 3; ++tick) {
        world.player(0).setInputs({});
        if (!processBridgeWorldTick(world, projectiles, initial, rules, state).advanced) return false;
        if (tick < 3 && (state.kits[0].arrows != 0 || state.kits[0].arrow_refill_elapsed != tick)) return false;
    }
    if (state.kits[0].arrows != 1 || state.kits[0].arrow_refill_elapsed != 0
        || projectiles.bows[0].ordinary_arrows != 1 || players[0].ordinary_arrow_count != 1) return false;

    rules.arrow_refill_ticks = 0;
    projectiles.bows[0].ordinary_arrows = 0;
    state.kits[0].arrow_refill_elapsed = 2;
    world.player(0).setInputs({});
    if (!processBridgeWorldTick(world, projectiles, initial, rules, state).advanced
        || state.kits[0].arrows != 0 || state.kits[0].arrow_refill_elapsed != 0) return false;
    state.kits[0].blocks = 4;
    state.kits[0].golden_apples = 1;
    state.kits[1].blocks = 9;
    players[0].health = 0.0F;
    auto events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.respawn[0] || state.kits[0].blocks != 64 || state.kits[0].golden_apples != 2
        || state.kits[0].arrows != 1 || state.kits[0].selected_slot != 0 || state.kits[1].blocks != 9) return false;
    state.kits[0].blocks = 3;
    state.kits[1].blocks = 8;
    players[0].position = {10.5, 10.0, 0.0};
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.round_reset || state.kits[0].blocks != 64 || state.kits[1].blocks != 20
        || state.kits[0].arrow_refill_elapsed != 0 || state.kits[0].arrows != 1) return false;
    state.kits[0].blocks = 1;
    resetBridgeMatch(world, projectiles, initial, rules, state, 123);
    return state.kits[0].blocks == 64 && state.kits[0].golden_apples == 2 && state.kits[0].arrows == 1;
}

MCSIM_HD inline bool building(minecraft_sim::MathView math) {
    using namespace minecraft_sim;
    Block cells[7 * 4 * 7]{};
    PlacedBlock overlay[7 * 4 * 7]{};
    BlockWorldView blocks{cells, 7, 4, 7, -3, 0, -3, overlay};
    for (int z = -3; z < 4; ++z) {
        for (int x = -3; x < 4; ++x) cells[blocks.cellIndex(x, 0, z)] = Block::solid;
    }
    cells[blocks.cellIndex(0, 2, 1)] = Block::solid;
    PlayerState players[2]{}, initial[2]{};
    initial[0].position = {0.5, 1.0, -1.5};
    initial[1].position = {-2.5, 1.0, 1.5};
    initial[0].weapon = WeaponItem::unarmed;
    initial[0].on_ground = true;
    initial[1].on_ground = true;
    Controls queued[2]{}, held[2]{};
    WorldState world_state;
    WorldView world(players, queued, held, &world_state, 2, blocks, math);
    ProjectileWorldState projectiles;
    BridgeMatchRules rules;
    rules.cage_ticks = 0;
    rules.kits[0].explicit_hotbar = true;
    rules.kits[0].slot_count = 2;
    rules.kits[0].slots[0] = BridgeHotbarItem::blocks;
    rules.kits[0].slots[1] = BridgeHotbarItem::pickaxe;
    rules.kits[0].blocks = 2;
    rules.building.enabled = true;
    rules.building.zone_count = 1;
    rules.building.zones[0] = {{-3.0, 0.0, -3.0}, {4.0, 4.0, 4.0}};
    rules.building.hardness = 1.0F;
    rules.building.mining_speed = 15.0F;
    BridgeMatchState state;
    resetBridgeMatch(world, projectiles, initial, rules, state, 123);
    auto target = bridgeInteractionTarget(world, 0, players[0], 4.5, 3.0);
    if (target.kind != BridgeTargetKind::block || target.block.z != 1 || target.block.face != BlockFace::north) return false;
    players[1].position = {0.5, 1.0, -0.5};
    target = bridgeInteractionTarget(world, 0, players[0], 4.5, 3.0);
    if (target.kind != BridgeTargetKind::entity || target.entity != 1) return false;
    players[1].position = {0.5, 1.0, 1.0 + double(0.6F) / 2.0};
    target = bridgeInteractionTarget(world, 0, players[0], 4.5, 3.0);
    if (target.kind != BridgeTargetKind::block) return false;
    cells[blocks.cellIndex(0, 2, 1)] = Block::air;
    cells[blocks.cellIndex(0, 2, 2)] = Block::solid;
    players[1].position = {0.5, 1.0, 2.0};
    target = bridgeInteractionTarget(world, 0, players[0], 4.5, 3.0);
    if (target.kind != BridgeTargetKind::miss) return false;
    cells[blocks.cellIndex(0, 2, 1)] = Block::solid;
    cells[blocks.cellIndex(0, 2, 2)] = Block::air;
    players[1] = initial[1];
    target = bridgeInteractionTarget(world, 0, players[0], 4.5, 3.0);
    auto restricted = rules.building;
    restricted.zones[0].lower.x = 1.0;
    int available = 2;
    TerrainEdit rejected;
    if (placeBridgeBlock(world, target, restricted, 0, available, rejected) || available != 2) return false;

    Controls controls;
    controls.select_weapon = true;
    controls.selected_slot = 0;
    controls.weapon = WeaponItem::unarmed;
    controls.use_item = true;
    world.player(0).setInputs(controls);
    auto events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.placed[0] || state.kits[0].blocks != 1 || overlay[blocks.cellIndex(0, 2, 0)] != PlacedBlock::team0) return false;
    for (int tick = 0; tick < 3; ++tick) {
        world.player(0).setInputs(controls);
        events = processBridgeWorldTick(world, projectiles, initial, rules, state);
        if (events.placed[0] || state.kits[0].blocks != 1) return false;
    }
    world.player(0).setInputs(controls);
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.placed[0] || state.kits[0].blocks != 0 || !blocks.solid(0, 2, -1)) return false;
    state.kits[0].blocks = 1;
    for (int tick = 0; tick < 4; ++tick) {
        world.player(0).setInputs(controls);
        events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    }
    if (events.placed[0] || state.kits[0].blocks != 1 || blocks.solid(0, 2, -2)) return false;

    controls = {};
    controls.select_weapon = true;
    controls.selected_slot = 1;
    controls.weapon = WeaponItem::unarmed;
    controls.attack = true;
    players[0].on_ground = true;
    world.player(0).setInputs(controls);
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (events.broke[0] || !state.kits[0].building.breaking || state.kits[0].building.break_progress != 0.0F) return false;
    players[0].on_ground = true;
    world.player(0).setInputs(controls);
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (events.broke[0] || state.kits[0].building.break_progress != 0.5F) return false;
    controls.attack = false;
    world.player(0).setInputs(controls);
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (events.broke[0] || state.kits[0].building.breaking || state.kits[0].building.break_progress != 0.0F) return false;
    controls.attack = true;
    players[0].on_ground = true;
    world.player(0).setInputs(controls);
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (events.broke[0] || !state.kits[0].building.breaking) return false;
    players[0].on_ground = true;
    world.player(0).setInputs(controls);
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (events.broke[0] || state.kits[0].building.break_progress != 0.5F) return false;
    players[0].on_ground = true;
    world.player(0).setInputs(controls);
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.broke[0] || blocks.solid(0, 2, -1) || state.kits[0].building.break_delay != 5) return false;
    if (!blocks.setPlacedBlock(0, 2, 0, PlacedBlock::empty)) return false;
    state.kits[0].building.break_delay = 0;
    world.player(0).setInputs(controls);
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (events.broke[0] || state.kits[0].building.breaking || !blocks.solid(0, 2, 1)) return false;
    players[0].health = 0.0F;
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.respawn[0] || state.kits[0].blocks != 2 || state.kits[0].building.use_delay != 0
        || state.kits[0].building.break_delay != 0) return false;
    rules.kits[0].slot_count = 3;
    rules.kits[0].slots[2] = BridgeHotbarItem::bow;
    initial[0].ordinary_arrow_count = 1;
    resetBridgeMatch(world, projectiles, initial, rules, state, 123);
    controls = {};
    controls.use_item = true;
    world.player(0).setInputs(controls);
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.placed[0]) return false;
    controls.select_weapon = true;
    controls.selected_slot = 2;
    controls.weapon = WeaponItem::bow;
    for (int tick = 0; tick < 3; ++tick) {
        world.player(0).setInputs(controls);
        processBridgeWorldTick(world, projectiles, initial, rules, state);
        if (projectiles.bows[0].using_bow) return false;
    }
    world.player(0).setInputs(controls);
    processBridgeWorldTick(world, projectiles, initial, rules, state);
    return projectiles.bows[0].using_bow;
}

MCSIM_HD inline bool simultaneousBuilding(minecraft_sim::MathView math) {
    using namespace minecraft_sim;
    Block cells[7 * 4 * 7]{};
    PlacedBlock overlay[7 * 4 * 7]{};
    BlockWorldView blocks{cells, 7, 4, 7, -3, 0, -3, overlay};
    for (int z = -3; z < 4; ++z) {
        for (int x = -3; x < 4; ++x) cells[blocks.cellIndex(x, 0, z)] = Block::solid;
    }
    cells[blocks.cellIndex(0, 2, 1)] = Block::solid;
    PlayerState players[2]{}, initial[2]{};
    initial[0].position = {0.1, 1.0, -1.5};
    initial[1].position = {0.9, 1.0, -1.5};
    initial[0].weapon = initial[1].weapon = WeaponItem::unarmed;
    Controls queued[2]{}, held[2]{};
    WorldState world_state;
    WorldView world(players, queued, held, &world_state, 2, blocks, math);
    ProjectileWorldState projectiles;
    BridgeMatchRules rules;
    rules.cage_ticks = 0;
    rules.building.enabled = true;
    rules.building.zone_count = 1;
    rules.building.zones[0] = {{-3.0, 0.0, -3.0}, {4.0, 4.0, 4.0}};
    for (int player = 0; player < 2; ++player) {
        rules.kits[player].explicit_hotbar = true;
        rules.kits[player].slot_count = 1;
        rules.kits[player].slots[0] = BridgeHotbarItem::blocks;
        rules.kits[player].blocks = 2;
    }
    BridgeMatchState state;
    resetBridgeMatch(world, projectiles, initial, rules, state, 123);
    Controls controls;
    controls.use_item = true;
    world.player(0).setInputs(controls);
    world.player(1).setInputs(controls);
    const auto events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    return events.placed[0] && events.placed[1] && state.kits[0].blocks == 1 && state.kits[1].blocks == 1
        && overlay[blocks.cellIndex(0, 2, 0)] == PlacedBlock::team0
        && overlay[blocks.cellIndex(0, 2, -1)] == PlacedBlock::team1;
}

    MCSIM_HD inline bool externalCutoff(minecraft_sim::MathView math) {
        using namespace minecraft_sim;
        PlayerState players[2]{};
        PlayerState initial[2]{};
        initial[0].position = {0.0, 1.0, 0.0};
        initial[1].position = {-2.0, 1.0, 0.0};
        Controls queued[2]{}, held[2]{};
        WorldState world_state;
        WorldView world(players, queued, held, &world_state, 2, {}, math);
        ProjectileWorldState projectiles;
        BridgeMatchRules rules;
        rules.goals[0] = {{-5.0, 0.0, -1.0}, {-4.0, 3.0, 1.0}};
        rules.goals[1] = {{4.0, 0.0, -1.0}, {5.0, 3.0, 1.0}};
        rules.cage_ticks = 0;
        BridgeMatchState state;
        resetBridgeMatch(world, projectiles, initial, rules, state, 123);
        players[0].position = {4.5, 1.0, 0.0};
        const auto events = processBridgeWorldTick(world, projectiles, initial, rules, state, 1);
        return events.scored[0] && !events.terminated && !events.round_reset && state.scores[0] == 1
        && state.ticks == 1 && players[0].position.x == 4.5;
    }

}
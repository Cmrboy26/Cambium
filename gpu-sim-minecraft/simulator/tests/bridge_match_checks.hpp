#pragma once

#include "minecraft_sim/bridge_match.hpp"

namespace bridge_match_checks {

MCSIM_HD inline bool independentOutcomes() {
    using namespace minecraft_sim;
    BridgeMatchRules rules;
    rules.goals[0] = {{-8.0, 0.0, -2.0}, {-6.0, 3.0, 2.0}};
    rules.goals[1] = {{6.0, 0.0, -2.0}, {8.0, 3.0, 2.0}};
    rules.cage_ticks = 0;
    rules.goals_to_win = 4;
    rules.max_ticks = 19;
    rules.void_y = -7.0;
    for (int scorer = 0; scorer < 2; ++scorer) {
        const int opponent = 1 - scorer;
        const double goal_x = scorer == 0 ? 7.0 : -7.0;
        PlayerState players[2]{};
        players[0].position = {0.0, 1.0, 0.0};
        players[1].position = {0.0, 1.0, 0.0};
        players[scorer].position.x = -goal_x;
        auto state = initialBridgeMatch(rules);
        auto events = finishBridgeTick(rules, state, players);
        if (!events.advanced || events.scored[0] || events.scored[1] || events.round_reset
            || state.ticks != 1 || state.cage_remaining != 0) return false;

        players[scorer].position.x = scorer == 0 ? 6.0 - double(0.6F) / 2.0 : -6.0 + double(0.6F) / 2.0;
        state = initialBridgeMatch(rules);
        events = finishBridgeTick(rules, state, players);
        if (events.scored[scorer] || events.round_reset) return false;
        players[scorer].position.x = goal_x;
        state = initialBridgeMatch(rules);
        state.scores[scorer] = 2;
        state.scores[opponent] = 3;
        state.ticks = 18;
        events = finishBridgeTick(rules, state, players);
        if (!events.scored[scorer] || events.scored[opponent] || !events.terminated
            || events.outcome != BridgeOutcome::draw || events.round_reset
            || state.scores[scorer] != 3 || state.scores[opponent] != 3 || state.ticks != 19) return false;

        state = initialBridgeMatch(rules);
        state.scores[scorer] = 3;
        state.scores[opponent] = 3;
        state.ticks = 18;
        events = finishBridgeTick(rules, state, players);
        const auto winner = scorer == 0 ? BridgeOutcome::team0 : BridgeOutcome::team1;
        if (!events.scored[scorer] || !events.terminated || events.outcome != winner
            || events.round_reset || state.scores[scorer] != 4 || state.scores[opponent] != 3) return false;

        players[opponent].position.x = -goal_x;
        state = initialBridgeMatch(rules);
        state.scores[scorer] = 3;
        state.scores[opponent] = 1;
        events = finishBridgeTick(rules, state, players);
        if (!events.scored[0] || !events.scored[1] || !events.terminated
            || events.outcome != winner || events.round_reset
            || state.scores[scorer] != 4 || state.scores[opponent] != 2) return false;

        state = initialBridgeMatch(rules);
        state.scores[0] = state.scores[1] = 3;
        events = finishBridgeTick(rules, state, players);
        if (!events.scored[0] || !events.scored[1] || !events.terminated
            || events.outcome != BridgeOutcome::draw || state.scores[0] != 4 || state.scores[1] != 4) return false;

        players[scorer].health = 0.0F;
        state = initialBridgeMatch(rules);
        events = finishBridgeTick(rules, state, players);
        if (!events.died[scorer] || events.died[opponent] || events.scored[scorer]
            || !events.scored[opponent] || !events.round_reset || events.respawn[0] || events.respawn[1]
            || events.terminated || state.scores[scorer] != 0 || state.scores[opponent] != 1) return false;

        players[scorer].health = 20.0F;
        rules.void_y = 1.5;
        players[opponent].position.y = 2.0;
        state = initialBridgeMatch(rules);
        events = finishBridgeTick(rules, state, players);
        if (!events.died[scorer] || events.died[opponent] || events.scored[scorer]
            || !events.scored[opponent] || !events.round_reset || events.respawn[0] || events.respawn[1]
            || state.scores[scorer] != 0 || state.scores[opponent] != 1) return false;
        rules.void_y = -7.0;
        players[opponent].position.y = 1.0;
        players[scorer].position.y = -7.0;
        players[opponent].position.x = 0.0;
        state = initialBridgeMatch(rules);
        events = finishBridgeTick(rules, state, players);
        if (events.died[scorer] || events.respawn[scorer]) return false;
        players[scorer].position.y = -7.01;
        state = initialBridgeMatch(rules);
        events = finishBridgeTick(rules, state, players);
        if (!events.died[scorer] || !events.respawn[scorer] || events.terminated || events.round_reset) return false;

        players[scorer].position = {0.0, 1.0, 0.0};
        state = initialBridgeMatch(rules);
        state.scores[scorer] = 2;
        state.scores[opponent] = 1;
        state.ticks = 18;
        events = finishBridgeTick(rules, state, players);
        if (!events.terminated || events.outcome != winner || events.scored[0] || events.scored[1]
            || state.ticks != 19) return false;

        players[0].health = players[1].health = 0.0F;
        state = initialBridgeMatch(rules);
        events = finishBridgeTick(rules, state, players);
        if (!events.died[0] || !events.died[1] || !events.respawn[0] || !events.respawn[1]
            || events.scored[0] || events.scored[1] || events.round_reset || events.terminated) return false;
    }
    return true;
}

MCSIM_HD inline bool outcomes() {
    using namespace minecraft_sim;
    if (!independentOutcomes()) return false;
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

MCSIM_HD inline bool mirroredPlayerRespawns() {
    using namespace minecraft_sim;
    for (int dead = 0; dead < 2; ++dead) {
        for (int void_death = 0; void_death < 2; ++void_death) {
            const int survivor = 1 - dead;
            Block base[1]{Block::air};
            PlacedBlock overlay[1]{PlacedBlock::team1};
            BlockWorldView blocks{base, 1, 1, 1, 0, 0, 0, overlay};
            PlayerState players[2]{}, initial[2]{};
            initial[0].position = {-2.0, 1.0, 0.0};
            initial[1].position = {2.0, 1.0, 0.0};
            initial[0].ordinary_arrow_count = 1;
            initial[1].ordinary_arrow_count = 1;
            Controls queued[2]{}, held[2]{};
            WorldState world_state;
            WorldView world(players, queued, held, &world_state, 2, blocks, {});
            ArrowState arrows[2]{};
            arrows[0].active = arrows[1].active = true;
            arrows[0].owner = 0;
            arrows[1].owner = 1;
            ProjectileWorldState projectiles;
            projectiles.arrows = arrows;
            projectiles.arrow_capacity = 2;
            BridgeMatchRules rules;
            rules.void_y = -5.0;
            rules.cage_ticks = 3;
            rules.kits[0].blocks = 40;
            rules.kits[1].blocks = 50;
            BridgeMatchState state = initialBridgeMatch(rules);
            state.scores[0] = 2;
            state.scores[1] = 1;
            state.ticks = 10;
            players[0] = initial[0];
            players[1] = initial[1];
            players[dead].random.setSeed(100 + dead);
            players[survivor].random.setSeed(200 + survivor);
            projectiles.random.setSeed(300);
            const auto dead_random = players[dead].random.state;
            const auto survivor_random = players[survivor].random.state;
            const auto projectile_random = projectiles.random.state;
            players[survivor].health = 11.0F;
            players[survivor].velocity.x = 0.25;
            players[dead].ordinary_arrow_count = 0;
            projectiles.bows[dead].ordinary_arrows = 0;
            projectiles.bows[survivor].ordinary_arrows = 0;
            state.kits[dead].blocks = 3;
            state.kits[dead].arrow_refill_elapsed = 2;
            state.kits[survivor].blocks = 7;
            state.kits[survivor].arrow_refill_elapsed = 1;
            queued[dead].forward = held[dead].attack = true;
            queued[survivor].forward = held[survivor].attack = true;
            if (void_death) players[dead].position.y = -5.01;
            else players[dead].health = 0.0F;
            const auto events = finishBridgeWorldTick(world, projectiles, initial, rules, state);
            if (!events.died[dead] || !events.respawn[dead] || events.died[survivor]
                || events.respawn[survivor] || events.round_reset || events.terminated
                || state.scores[0] != 2 || state.scores[1] != 1 || state.ticks != 11
                || state.cage_remaining != 2 || players[dead].health != 20.0F
                || players[dead].position.x != initial[dead].position.x
                || players[dead].position.y != 1.0 || players[dead].random.state != dead_random
                || players[survivor].health != 11.0F || players[survivor].velocity.x != 0.25
                || players[survivor].random.state != survivor_random || projectiles.random.state != projectile_random
                || queued[dead].forward || held[dead].attack || !queued[survivor].forward || !held[survivor].attack
                || arrows[dead].active || !arrows[survivor].active || projectiles.bows[dead].ordinary_arrows != 1
                || projectiles.bows[survivor].ordinary_arrows != 0 || state.kits[dead].blocks != (dead == 0 ? 40 : 50)
                || state.kits[dead].arrows != 1 || state.kits[dead].arrow_refill_elapsed != 0
                || state.kits[survivor].blocks != 7 || state.kits[survivor].arrow_refill_elapsed != 1
                || overlay[0] != PlacedBlock::team1) return false;
        }
    }
    return true;
}

MCSIM_HD inline bool mirroredGoalDeathResets() {
    using namespace minecraft_sim;
    for (int scorer = 0; scorer < 2; ++scorer) {
        for (int winning = 0; winning < 2; ++winning) {
            const int dead = 1 - scorer;
            Block base[2]{Block::solid, Block::air};
            PlacedBlock overlay[2]{PlacedBlock::empty, PlacedBlock::team1};
            BlockWorldView blocks{base, 2, 1, 1, 0, 0, 0, overlay};
            PlayerState players[2]{}, initial[2]{};
            initial[0].position = {-2.0, 1.0, 0.0};
            initial[1].position = {2.0, 1.0, 0.0};
            initial[0].yaw = 35.0F;
            initial[1].yaw = -70.0F;
            initial[0].ordinary_arrow_count = 1;
            initial[1].ordinary_arrow_count = 2;
            Controls queued[2]{}, held[2]{};
            WorldState world_state;
            world_state.tick = 41;
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
            rules.goals[0] = {{-8.0, 0.0, -2.0}, {-6.0, 3.0, 2.0}};
            rules.goals[1] = {{6.0, 0.0, -2.0}, {8.0, 3.0, 2.0}};
            rules.goals_to_win = 3;
            rules.cage_ticks = 4;
            rules.kits[0].blocks = 31;
            rules.kits[1].blocks = 47;
            rules.persist_blocks_on_goal = false;
            auto state = initialBridgeMatch(rules);
            state.cage_remaining = 2;
            state.ticks = 12;
            state.scores[scorer] = winning ? 2 : 1;
            state.scores[dead] = 1;
            for (int player = 0; player < 2; ++player) {
                players[player] = initial[player];
                players[player].random.setSeed(500 + player);
                players[player].velocity.x = 0.25 + player;
                players[player].golden_apple_regeneration_ticks = 8 + player;
                queued[player].forward = true;
                held[player].attack = true;
                arrows[player].active = wind[player].active = thrown[player].active = true;
                arrows[player].owner = wind[player].owner = thrown[player].owner = player;
                projectiles.bows[player].ordinary_arrows = 0;
                state.kits[player].blocks = 3 + player;
                state.kits[player].arrow_refill_elapsed = 5 + player;
            }
            projectiles.random.setSeed(900);
            const auto first_random = players[0].random.state;
            const auto second_random = players[1].random.state;
            const auto projectile_random = projectiles.random.state;
            players[scorer].position.x = scorer == 0 ? 7.0 : -7.0;
            players[scorer].health = 9.0F;
            players[dead].health = 0.0F;
            const auto events = finishBridgeWorldTick(world, projectiles, initial, rules, state);
            if (!events.advanced || !events.scored[scorer] || events.scored[dead]
                || !events.died[dead] || events.died[scorer] || events.respawn[0] || events.respawn[1]
                || events.terminated != bool(winning) || events.round_reset == bool(winning)
                || state.scores[scorer] != (winning ? 3 : 2) || state.scores[dead] != 1
                || state.ticks != 13 || world_state.tick != 41
                || players[0].random.state != first_random || players[1].random.state != second_random
                || projectiles.random.state != projectile_random || !blocks.solid(0, 0, 0)) return false;
            if (winning) {
                const auto winner = scorer == 0 ? BridgeOutcome::team0 : BridgeOutcome::team1;
                if (events.outcome != winner || state.cage_remaining != 1
                    || players[scorer].position.x != (scorer == 0 ? 7.0 : -7.0)
                    || players[scorer].health != 9.0F || players[dead].health != 0.0F
                    || !queued[0].forward || !held[1].attack || !arrows[0].active || !arrows[1].active
                    || !wind[0].active || !thrown[1].active || overlay[1] != PlacedBlock::team1
                    || state.kits[0].blocks != 3 || state.kits[1].blocks != 4) return false;
            } else {
                if (events.outcome != BridgeOutcome::ongoing || state.cage_remaining != 4
                    || players[0].position.x != -2.0 || players[1].position.x != 2.0
                    || players[0].yaw != 35.0F || players[1].yaw != -70.0F
                    || players[0].health != 20.0F || players[1].health != 20.0F
                    || players[0].velocity.x != 0.0 || players[1].velocity.x != 0.0
                    || players[0].golden_apple_regeneration_ticks != 0 || players[1].golden_apple_regeneration_ticks != 0
                    || queued[0].forward || queued[1].forward || held[0].attack || held[1].attack
                    || arrows[0].active || arrows[1].active || wind[0].active || wind[1].active
                    || thrown[0].active || thrown[1].active || overlay[1] != PlacedBlock::empty
                    || projectiles.bows[0].ordinary_arrows != 1 || projectiles.bows[1].ordinary_arrows != 2
                    || state.kits[0].blocks != 31 || state.kits[1].blocks != 47
                    || state.kits[0].arrow_refill_elapsed != 0 || state.kits[1].arrow_refill_elapsed != 0) return false;
            }
        }
    }
    return true;
}

MCSIM_HD inline bool simultaneousGoalAndTimeoutState() {
    using namespace minecraft_sim;
    for (int scenario = 0; scenario < 5; ++scenario) {
        Block base[2]{Block::solid, Block::air};
        PlacedBlock overlay[2]{PlacedBlock::empty, PlacedBlock::team0};
        BlockWorldView blocks{base, 2, 1, 1, 0, 0, 0, overlay};
        PlayerState players[2]{}, initial[2]{};
        initial[0].position = {-2.0, 1.0, 0.0};
        initial[1].position = {2.0, 1.0, 0.0};
        Controls queued[2]{}, held[2]{};
        WorldState world_state;
        WorldView world(players, queued, held, &world_state, 2, blocks, {});
        ArrowState arrows[2]{};
        ProjectileWorldState projectiles;
        projectiles.arrows = arrows;
        projectiles.arrow_capacity = 2;
        BridgeMatchRules rules;
        rules.goals[0] = {{-8.0, 0.0, -2.0}, {-6.0, 3.0, 2.0}};
        rules.goals[1] = {{6.0, 0.0, -2.0}, {8.0, 3.0, 2.0}};
        rules.goals_to_win = 3;
        rules.max_ticks = 17;
        rules.cage_ticks = 5;
        auto state = initialBridgeMatch(rules);
        state.cage_remaining = 0;
        state.scores[0] = scenario == 1 || scenario == 3 ? 2 : 0;
        state.scores[1] = scenario == 2 || scenario == 3 ? 2 : 0;
        state.ticks = scenario == 4 ? 16 : 6;
        for (int player = 0; player < 2; ++player) {
            players[player] = initial[player];
            players[player].random.setSeed(1000 + player);
            players[player].position.x = player == 0 ? 7.0 : -7.0;
            players[player].health = 12.0F + player;
            queued[player].forward = true;
            held[player].attack = true;
            arrows[player].active = true;
            arrows[player].owner = player;
        }
        projectiles.random.setSeed(2000);
        const auto first_random = players[0].random.state;
        const auto second_random = players[1].random.state;
        const auto projectile_random = projectiles.random.state;
        if (scenario == 4) {
            players[0].position.x = 0.0;
            players[1].position.x = 1.0;
            players[1].health = 0.0F;
        }
        const auto events = finishBridgeWorldTick(world, projectiles, initial, rules, state);
        const auto expected = scenario == 1 ? BridgeOutcome::team0
            : scenario == 2 ? BridgeOutcome::team1 : BridgeOutcome::draw;
        if (!events.advanced || events.terminated != (scenario != 0)
            || events.round_reset != (scenario == 0) || events.respawn[0] || events.respawn[1]
            || events.scored[0] != (scenario != 4) || events.scored[1] != (scenario != 4)
            || events.died[0] || events.died[1] != (scenario == 4)
            || state.ticks != (scenario == 4 ? 17 : 7)
            || players[0].random.state != first_random || players[1].random.state != second_random
            || projectiles.random.state != projectile_random || overlay[1] != PlacedBlock::team0
            || !blocks.solid(0, 0, 0)) return false;
        if (scenario == 0) {
            if (events.outcome != BridgeOutcome::ongoing || state.scores[0] != 1 || state.scores[1] != 1
                || state.cage_remaining != 5 || players[0].position.x != -2.0 || players[1].position.x != 2.0
                || players[0].health != 20.0F || players[1].health != 20.0F
                || arrows[0].active || arrows[1].active || queued[0].forward || held[1].attack) return false;
        } else {
            if (events.outcome != expected || state.cage_remaining != 0
                || state.scores[0] != (scenario == 1 || scenario == 3 ? 3 : scenario == 4 ? 0 : 1)
                || state.scores[1] != (scenario == 2 || scenario == 3 ? 3 : scenario == 4 ? 0 : 1)
                || !arrows[0].active || !arrows[1].active || !queued[0].forward || !held[1].attack
                || players[0].health != 12.0F || players[1].health != (scenario == 4 ? 0.0F : 13.0F)
                || players[0].position.x != (scenario == 4 ? 0.0 : 7.0)
                || players[1].position.x != (scenario == 4 ? 1.0 : -7.0)) return false;
        }
    }
    return true;
}

MCSIM_HD inline bool lifecycle() {
    using namespace minecraft_sim;
    if (!mirroredPlayerRespawns() || !mirroredGoalDeathResets() || !simultaneousGoalAndTimeoutState()) return false;
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
    ArrowState arrows[2][2]{};
    WindChargeState wind[2]{};
    ThrownProjectileState thrown[2]{};
    BridgeMatchState matches[2]{};
    BridgeMatchRules rules;
    rules.cage_ticks = 6;
    rules.kits[0].blocks = 19;
    rules.kits[1].blocks = 29;
    initial[0].position = {-3.0, 2.0, 1.0};
    initial[1].position = {3.0, 2.0, -1.0};
    initial[0].velocity.x = 0.125;
    initial[1].velocity.x = -0.25;
    initial[0].yaw = 45.0F;
    initial[1].yaw = -90.0F;
    initial[0].health = 18.0F;
    initial[1].health = 16.0F;
    initial[0].ordinary_arrow_count = 2;
    initial[1].ordinary_arrow_count = 1;
    BlockWorldView first_blocks{base, 2, 1, 1, 0, 0, 0, overlays[0]};
    BlockWorldView second_blocks{base, 2, 1, 1, 0, 0, 0, overlays[1]};
    WorldView first(players[0], queued[0], held[0], &worlds[0], 2, first_blocks, {});
    WorldView second(players[1], queued[1], held[1], &worlds[1], 2, second_blocks, {});
    if (!first_blocks.setPlacedBlock(1, 0, 0, PlacedBlock::team0)
        || !second_blocks.setPlacedBlock(1, 0, 0, PlacedBlock::team1)) return false;
    players[1][0].health = 7.0F;
    players[1][0].random.setSeed(77);
    players[1][1].random.setSeed(88);
    projectiles[1].random.setSeed(99);
    const auto second_random = players[1][0].random.state;
    const auto second_player_random = players[1][1].random.state;
    const auto second_projectile_random = projectiles[1].random.state;
    for (int index = 0; index < 2; ++index) {
        projectiles[index].arrows = arrows[index];
        projectiles[index].arrow_capacity = 2;
        projectiles[index].wind_charges = &wind[index];
        projectiles[index].wind_charge_capacity = 1;
        projectiles[index].thrown_items = &thrown[index];
        projectiles[index].thrown_item_capacity = 1;
        arrows[index][0].active = arrows[index][1].active = true;
        wind[index].active = thrown[index].active = true;
        projectiles[index].bows[1].ordinary_arrows = 6;
        held[index][1].attack = true;
    }
    queued[1][0].forward = true;
    worlds[1].tick = 23;
    matches[1].scores[0] = 3;
    matches[1].ticks = 15;
    matches[1].episodes = 4;
    matches[1].cage_remaining = 7;
    matches[1].kits[1].arrow_refill_elapsed = 9;
    players[0][0].health = 2.0F;
    players[0][1].health = 3.0F;
    players[0][0].golden_apple_regeneration_ticks = 10;
    players[0][1].golden_apple_absorption_ticks = 20;
    worlds[0].tick = 31;
    worlds[0].simulate_fall_damage = true;
    worlds[0].difficulty = Difficulty::hard;
    matches[0].scores[0] = 4;
    matches[0].scores[1] = 2;
    matches[0].ticks = 11;
    matches[0].outcome = BridgeOutcome::team1;
    resetBridgeMatch(first, projectiles[0], initial, rules, matches[0], 123);
    if (first_blocks.solid(1, 0, 0) || !second_blocks.solid(1, 0, 0)
        || !first_blocks.solid(0, 0, 0) || matches[0].scores[0] != 0 || matches[0].scores[1] != 0
        || matches[0].ticks != 0 || matches[0].episodes != 1 || matches[0].cage_remaining != 6
        || matches[0].outcome != BridgeOutcome::ongoing || worlds[0].tick != 0
        || !worlds[0].simulate_fall_damage || worlds[0].difficulty != Difficulty::hard
        || players[0][0].position.x != -3.0 || players[0][1].position.x != 3.0
        || players[0][0].velocity.x != 0.125 || players[0][1].velocity.x != -0.25
        || players[0][0].yaw != 45.0F || players[0][1].yaw != -90.0F
        || players[0][0].health != 18.0F || players[0][1].health != 16.0F
        || players[0][0].golden_apple_regeneration_ticks != 0 || players[0][1].golden_apple_absorption_ticks != 0
        || matches[0].kits[0].blocks != 19 || matches[0].kits[1].blocks != 29
        || projectiles[0].bows[0].ordinary_arrows != 2 || projectiles[0].bows[1].ordinary_arrows != 1
        || overlays[1][1] != PlacedBlock::team1 || players[1][0].health != 7.0F
        || players[1][0].random.state != second_random || players[1][1].random.state != second_player_random
        || projectiles[1].random.state != second_projectile_random || !second.queuedInput(0).forward
        || !held[1][1].attack || held[0][1].attack
        || arrows[0][0].active || arrows[0][1].active || wind[0].active || thrown[0].active
        || !arrows[1][0].active || !arrows[1][1].active || !wind[1].active || !thrown[1].active
        || projectiles[1].bows[1].ordinary_arrows != 6
        || worlds[1].tick != 23 || matches[1].scores[0] != 3 || matches[1].ticks != 15
        || matches[1].episodes != 4 || matches[1].cage_remaining != 7
        || matches[1].kits[1].arrow_refill_elapsed != 9) return false;
    matches[1].episodes = 0;
    resetBridgeMatch(second, projectiles[1], initial, rules, matches[1], 123);
    if (players[0][0].random.state != players[1][0].random.state
        || players[0][1].random.state != players[1][1].random.state
        || projectiles[0].random.state != projectiles[1].random.state) return false;
    const auto first_episode_player0 = players[1][0].random.state;
    const auto first_episode_player1 = players[1][1].random.state;
    const auto first_episode_projectile = projectiles[1].random.state;
    resetBridgeMatch(first, projectiles[0], initial, rules, matches[0], 123);
    if (matches[0].episodes != 2 || matches[1].episodes != 1
        || players[0][0].random.state == first_episode_player0
        || players[0][1].random.state == first_episode_player1
        || projectiles[0].random.state == first_episode_projectile
        || players[1][0].random.state != first_episode_player0
        || players[1][1].random.state != first_episode_player1
        || projectiles[1].random.state != first_episode_projectile) return false;
    resetBridgeMatch(second, projectiles[1], initial, rules, matches[1], 123);
    return matches[1].episodes == 2
        && players[0][0].random.state == players[1][0].random.state
        && players[0][1].random.state == players[1][1].random.state
        && projectiles[0].random.state == projectiles[1].random.state;
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
    players[1].velocity = {-1.0, -2.0, -1.0};
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
    Controls other_inputs = inputs;
    other_inputs.yaw_delta = -30.0F;
    other_inputs.pitch_delta = -6.0F;
    world.player(1).setInputs(other_inputs);
    auto events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.advanced || state.cage_remaining != 1 || players[0].position.x != 0.0
        || players[0].position.y != 10.0 || players[0].position.z != 0.0
        || players[0].velocity.lengthSquared() != 0.0 || players[0].yaw != 20.0F
        || players[0].pitch != 5.0F || players[0].weapon != initial[0].weapon
        || players[0].last_attack.attempted || players[1].position.x != 5.0
        || players[1].position.y != 10.0 || players[1].position.z != 0.0
        || players[1].velocity.lengthSquared() != 0.0 || players[1].yaw != -30.0F
        || players[1].pitch != -6.0F || players[1].weapon != initial[1].weapon
        || players[1].last_attack.attempted) return false;
    inputs.yaw_delta = -5.0F;
    inputs.pitch_delta = 0.0F;
    world.player(0).setInputs(inputs);
    other_inputs.yaw_delta = 10.0F;
    other_inputs.pitch_delta = 0.0F;
    world.player(1).setInputs(other_inputs);
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.advanced || state.cage_remaining != 0 || players[0].position.y != 10.0
        || players[0].yaw != 15.0F || players[0].last_attack.attempted
        || players[1].position.y != 10.0 || players[1].yaw != -20.0F || players[1].last_attack.attempted) return false;
    inputs = {};
    inputs.forward = true;
    world.player(0).setInputs(inputs);
    world.player(1).setInputs(inputs);
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.advanced || (players[0].position.x == 0.0 && players[0].position.z == 0.0)
        || (players[1].position.x == 5.0 && players[1].position.z == 0.0)
        || world_state.tick != 3 || state.ticks != 3) return false;
    state.outcome = BridgeOutcome::draw;
    if (processBridgeWorldTick(world, projectiles, initial, rules, state).advanced || world_state.tick != 3) return false;
    rules.cage_ticks = 0;
    resetBridgeMatch(world, projectiles, initial, rules, state, 123);
    world.player(0).setInputs(inputs);
    world.player(1).setInputs(inputs);
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    return events.advanced && state.cage_remaining == 0 && state.ticks == 1
        && (players[0].position.x != 0.0 || players[0].position.z != 0.0)
        && (players[1].position.x != 5.0 || players[1].position.z != 0.0);
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

MCSIM_HD inline bool buildingConsumers(minecraft_sim::BlockWorldView blocks, minecraft_sim::MathView math, bool blocked) {
    using namespace minecraft_sim;
    const Vec3 origin{-1.5, 2.5, -0.5}, delta{4.0, 0.0, 0.0};
    const Aabb cell{{0.1, 2.1, -0.9}, {0.9, 2.9, -0.1}};
    const auto target = targetFullCube(blocks, origin, delta);
    if (target.hit != blocked || (blocked && (target.x != 0 || target.y != 2 || target.z != -1
        || target.face != BlockFace::west))
        || blockRayFraction(blocks, origin, delta) != (blocked ? 0.375 : 1.0)
        || noBlockCollision(blocks, cell) == blocked) return false;

    PlayerState players[2]{};
    players[0].position = {-0.5, 1.0, -0.5};
    players[0].yaw = -90.0F;
    players[0].velocity.x = 0.6;
    players[1].position = {1.5, 1.0, -0.5};
    Controls queued[2]{}, held[2]{};
    queued[0].attack = true;
    WorldState world_state;
    WorldView world(players, queued, held, &world_state, 2, blocks, math);
    ArrowState arrow;
    arrow.position = {-0.5, 2.5, -0.5};
    arrow.velocity = {1.0, 0.0, 0.0};
    arrow.owner = 0;
    arrow.active = true;
    ProjectileWorldState projectiles;
    projectiles.arrows = &arrow;
    projectiles.arrow_capacity = 1;
    if (world.pickAttackTarget(0) != (blocked ? -1 : 1)) return false;
    processWorldTick(world, projectiles);
    return players[0].last_attack.target == (blocked ? -1 : 1)
        && players[0].last_attack.hit_block == blocked
        && (blocked ? players[0].position.x < -0.29 : players[0].position.x > 0.0)
        && arrow.in_ground == blocked;
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

    const Vec3 ray_origin{0.5, 2.5, -1.5}, ray_delta{0.0, 0.0, 4.0};
    const Aabb placed_cell{{0.1, 2.1, 0.1}, {0.9, 2.9, 0.9}};
    const auto initial_target = targetFullCube(blocks, ray_origin, ray_delta);
    if (!initial_target.hit || initial_target.z != 1 || initial_target.face != BlockFace::north
        || blockRayFraction(blocks, ray_origin, ray_delta) != 0.625
        || !noBlockCollision(blocks, placed_cell) || !buildingConsumers(blocks, math, false)) return false;

    Controls controls;
    controls.select_weapon = true;
    controls.selected_slot = 0;
    controls.weapon = WeaponItem::unarmed;
    controls.use_item = true;
    world.player(0).setInputs(controls);
    auto events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.placed[0] || state.kits[0].blocks != 1 || overlay[blocks.cellIndex(0, 2, 0)] != PlacedBlock::team0) return false;
    const auto placed_target = targetFullCube(blocks, ray_origin, ray_delta);
    if (!placed_target.hit || placed_target.z != 0 || placed_target.face != BlockFace::north
        || blockRayFraction(blocks, ray_origin, ray_delta) != 0.375
        || noBlockCollision(blocks, placed_cell)) return false;
    for (int tick = 0; tick < 3; ++tick) {
        world.player(0).setInputs(controls);
        events = processBridgeWorldTick(world, projectiles, initial, rules, state);
        if (events.placed[0] || state.kits[0].blocks != 1) return false;
    }
    world.player(0).setInputs(controls);
    events = processBridgeWorldTick(world, projectiles, initial, rules, state);
    if (!events.placed[0] || state.kits[0].blocks != 0 || !blocks.solid(0, 2, -1)
        || !buildingConsumers(blocks, math, true)) return false;
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
    if (!events.broke[0] || blocks.solid(0, 2, -1) || state.kits[0].building.break_delay != 5
        || !buildingConsumers(blocks, math, false)) return false;
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
        auto events = processBridgeWorldTick(world, projectiles, initial, rules, state, 1);
        if (!events.scored[0] || events.terminated || events.round_reset || state.scores[0] != 1
            || state.ticks != 1 || players[0].position.x != 4.5) return false;
        resetBridgeMatch(world, projectiles, initial, rules, state, 123);
        state.scores[0] = 4;
        players[0].position = {4.5, 1.0, 0.0};
        events = processBridgeWorldTick(world, projectiles, initial, rules, state, 1);
        return events.scored[0] && events.terminated && !events.round_reset
            && events.outcome == BridgeOutcome::team0 && state.scores[0] == 5
            && state.ticks == 1 && players[0].position.x == 4.5;
    }

}
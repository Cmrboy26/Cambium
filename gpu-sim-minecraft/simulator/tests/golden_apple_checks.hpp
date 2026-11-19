#pragma once

#include "minecraft_sim/bridge_match.hpp"

namespace golden_apple_checks {

struct Fixture {
    minecraft_sim::PlayerState players[2]{}, initial[2]{};
    minecraft_sim::Controls queued[2]{}, held[2]{};
    minecraft_sim::WorldState world_state;
    minecraft_sim::WorldView world;
    minecraft_sim::ProjectileWorldState projectiles;
    minecraft_sim::BridgeMatchRules rules;
    minecraft_sim::BridgeMatchState match;

    MCSIM_HD explicit Fixture(minecraft_sim::MathView math)
        : world(players, queued, held, &world_state, 2, {}, math) {
        using namespace minecraft_sim;
        initial[0].position = {0.0, 10.0, 0.0};
        initial[1].position = {5.0, 10.0, 0.0};
        rules.goals[0] = {{-101.0, -100000.0, -1.0}, {-100.0, 100.0, 1.0}};
        rules.goals[1] = {{100.0, -100000.0, -1.0}, {101.0, 100.0, 1.0}};
        rules.void_y = -100000.0;
        rules.max_ticks = 10000;
        rules.cage_ticks = 0;
        for (int player = 0; player < 2; ++player) {
            rules.kits[player].explicit_hotbar = true;
            rules.kits[player].slot_count = 2;
            rules.kits[player].slots[0] = BridgeHotbarItem::golden_apple;
            rules.kits[player].slots[1] = BridgeHotbarItem::sword;
            rules.kits[player].golden_apples = 2;
        }
        reset();
    }

    MCSIM_HD void reset() {
        minecraft_sim::resetBridgeMatch(world, projectiles, initial, rules, match, 123);
    }

    MCSIM_HD minecraft_sim::BridgeTickEvents step(bool first, bool second) {
        queued[0] = {};
        queued[1] = {};
        queued[0].use_item = first;
        queued[1].use_item = second;
        return minecraft_sim::processBridgeWorldTick(world, projectiles, initial, rules, match);
    }
};

MCSIM_HD inline int run(minecraft_sim::MathView math) {
    using namespace minecraft_sim;
    Fixture fixture(math);
    fixture.world_state.simulate_food = true;
    fixture.players[0].health = 10.0F;
    fixture.players[0].food_level = 10;
    fixture.players[0].food_saturation = 2.0F;
    for (int tick = 0; tick < 10; ++tick) fixture.step(true, true);
    if (!fixture.world.player(0).hurt(3.0F, {}, 0.0F, true, DamageKind::fall)) return 1;
    for (int tick = 10; tick < 31; ++tick) fixture.step(true, true);
    if (fixture.players[0].health != 7.0F || fixture.players[0].golden_apple_use_ticks != 1
        || fixture.players[1].golden_apple_use_ticks != 1 || fixture.match.kits[0].golden_apples != 2) return 2;
    fixture.step(true, true);
    if (fixture.match.kits[0].golden_apples != 1 || fixture.match.kits[1].golden_apples != 1
        || fixture.players[0].absorption != 4.0F || fixture.players[1].absorption != 4.0F
        || fixture.players[0].health != 7.0F || fixture.players[0].food_level != 14
        || absolute(fixture.players[0].food_saturation - 11.6F) > 0.00001F
        || fixture.players[0].golden_apple_regeneration_ticks != 100
        || fixture.players[0].golden_apple_absorption_ticks != 2400) return 3;
    fixture.step(false, false);
    if (fixture.players[0].health != 8.0F || fixture.players[0].golden_apple_regeneration_ticks != 99
        || fixture.players[0].golden_apple_absorption_ticks != 2399) return 4;
    fixture.players[0].absorption = 1.5F;
    for (int tick = 0; tick < 32; ++tick) fixture.step(true, false);
    if (fixture.match.kits[0].golden_apples != 0 || fixture.players[0].absorption != 4.0F
        || fixture.players[0].golden_apple_regeneration_ticks != 100
        || fixture.players[0].golden_apple_absorption_ticks != 2400) return 5;
    const float health_before_damage = fixture.players[0].health;
    if (!fixture.world.player(0).hurt(4.0F, {}, 0.0F, true, DamageKind::fall)
        || fixture.players[0].health != health_before_damage || fixture.players[0].absorption != 0.0F) return 6;
    fixture.step(true, false);
    if (fixture.players[0].golden_apple_absorption_ticks != 0
        || fixture.players[0].golden_apple_use_ticks != 0 || fixture.match.kits[0].golden_apples != 0) return 7;
    fixture.players[0].absorption = 2.0F;
    fixture.players[0].golden_apple_absorption_ticks = 1;
    fixture.players[0].golden_apple_regeneration_ticks = 25;
    fixture.players[0].health = 12.0F;
    fixture.step(false, false);
    if (fixture.players[0].absorption != 0.0F || fixture.players[0].golden_apple_absorption_ticks != 0
        || fixture.players[0].health != 13.0F || fixture.players[0].golden_apple_regeneration_ticks != 24) return 8;
    for (int tick = 0; tick < 24; ++tick) fixture.step(false, false);
    if (fixture.players[0].health != 13.0F || fixture.players[0].golden_apple_regeneration_ticks != 0) return 9;

    fixture.world_state.simulate_food = false;
    fixture.reset();
    fixture.players[0].food_level = 10;
    fixture.players[0].food_saturation = 2.0F;
    for (int tick = 0; tick < 5; ++tick) fixture.step(true, false);
    fixture.step(false, false);
    if (fixture.players[0].golden_apple_use_ticks != 0 || fixture.match.kits[0].golden_apples != 2) return 10;
    for (int tick = 0; tick < 5; ++tick) fixture.step(true, false);
    fixture.queued[0] = {};
    fixture.queued[0].select_weapon = true;
    fixture.queued[0].selected_slot = 1;
    fixture.queued[0].weapon = WeaponItem::diamond_sword;
    fixture.queued[0].attack = true;
    fixture.queued[0].use_item = true;
    fixture.queued[1] = {};
    processBridgeWorldTick(fixture.world, fixture.projectiles, fixture.initial, fixture.rules, fixture.match);
    if (fixture.players[0].golden_apple_use_ticks != 0 || fixture.players[0].last_attack.attempted
        || fixture.match.kits[0].golden_apples != 2) return 11;
    fixture.queued[0] = {};
    fixture.queued[0].select_weapon = true;
    fixture.queued[0].selected_slot = 0;
    fixture.queued[0].weapon = WeaponItem::unarmed;
    fixture.queued[0].use_item = true;
    processBridgeWorldTick(fixture.world, fixture.projectiles, fixture.initial, fixture.rules, fixture.match);
    for (int tick = 1; tick < 32; ++tick) fixture.step(true, true);
    if (fixture.players[0].food_level != 10 || fixture.players[0].food_saturation != 2.0F
        || fixture.match.kits[0].golden_apples != 1) return 12;
    fixture.step(false, true);
    if (fixture.match.kits[1].golden_apples != 1) return 13;
    fixture.players[0].health = 0.0F;
    const auto death = fixture.step(false, false);
    if (!death.respawn[0] || death.respawn[1] || fixture.match.kits[0].golden_apples != 2
        || fixture.players[0].golden_apple_regeneration_ticks != 0
        || fixture.players[1].golden_apple_regeneration_ticks == 0
        || fixture.match.kits[1].golden_apples != 1) return 14;
    fixture.players[0].position = {100.5, 10.0, 0.0};
    const auto goal = fixture.step(false, false);
    if (!goal.round_reset || fixture.match.kits[0].golden_apples != 2 || fixture.match.kits[1].golden_apples != 2
        || fixture.players[0].golden_apple_regeneration_ticks != 0
        || fixture.players[1].golden_apple_absorption_ticks != 0) return 15;
    fixture.match.cage_remaining = 1;
    fixture.step(true, true);
    if (fixture.players[0].golden_apple_use_ticks != 0 || fixture.players[1].golden_apple_use_ticks != 0) return 16;
    Fixture other(math);
    other.players[0].golden_apple_regeneration_ticks = 42;
    other.match.kits[0].golden_apples = 1;
    fixture.reset();
    if (other.players[0].golden_apple_regeneration_ticks != 42 || other.match.kits[0].golden_apples != 1) return 17;

    fixture.players[0].health = 10.0F;
    for (int tick = 0; tick < 31; ++tick) fixture.step(true, false);
    fixture.players[0].position = {0.0, 10.0, 0.0};
    fixture.players[0].velocity = {};
    fixture.players[1].position = {0.0, 10.0, -2.0};
    fixture.players[1].velocity = {};
    fixture.players[1].fall_distance = 0.0;
    fixture.players[1].selected_weapon_slot = 1;
    fixture.world.player(1).equipWeapon(WeaponItem::diamond_sword);
    fixture.players[1].attack_strength_ticks = 100;
    fixture.queued[0] = {};
    fixture.queued[0].use_item = true;
    fixture.queued[1] = {};
    fixture.queued[1].attack = true;
    processBridgeWorldTick(fixture.world, fixture.projectiles, fixture.initial, fixture.rules, fixture.match);
    if (!fixture.players[1].last_attack.accepted || fixture.players[0].health != 3.0F
        || fixture.players[0].absorption != 4.0F || fixture.match.kits[0].golden_apples != 1) return 18;

    fixture.reset();
    fixture.players[0].health = 10.0F;
    for (int tick = 0; tick < 31; ++tick) fixture.step(true, false);
    fixture.players[0].position = {0.0, 10.0, 0.0};
    fixture.players[0].velocity = {};
    ArrowState arrow;
    arrow.active = true;
    arrow.owner = 1;
    arrow.left_owner = true;
    arrow.position = {0.0, 10.9, -1.0};
    arrow.velocity = {0.0, 0.0, 1.0};
    fixture.projectiles.arrows = &arrow;
    fixture.projectiles.arrow_capacity = 1;
    fixture.step(true, false);
    if (arrow.active || fixture.players[0].health != 10.0F || fixture.players[0].absorption != 2.0F
        || fixture.match.kits[0].golden_apples != 1) return 19;

    fixture.projectiles.arrows = nullptr;
    fixture.projectiles.arrow_capacity = 0;
    fixture.rules.kits[0].slots[1] = BridgeHotbarItem::bow;
    fixture.initial[0].ordinary_arrow_count = 1;
    fixture.reset();
    for (int tick = 0; tick < 5; ++tick) fixture.step(true, false);
    fixture.queued[0] = {};
    fixture.queued[0].select_weapon = true;
    fixture.queued[0].selected_slot = 1;
    fixture.queued[0].weapon = WeaponItem::bow;
    fixture.queued[0].use_item = true;
    fixture.queued[1] = {};
    processBridgeWorldTick(fixture.world, fixture.projectiles, fixture.initial, fixture.rules, fixture.match);
    if (fixture.players[0].golden_apple_use_ticks != 0 || fixture.projectiles.bows[0].using_bow
        || fixture.match.kits[0].golden_apples != 2) return 20;
    fixture.step(true, false);
    if (!fixture.projectiles.bows[0].using_bow) return 21;

    fixture.reset();
    fixture.players[0].health = 5.0F;
    fixture.players[0].golden_apple_regeneration_ticks = 100;
    for (int tick = 0; tick < 100; ++tick) fixture.step(false, false);
    if (fixture.players[0].health != 9.0F || fixture.players[0].golden_apple_regeneration_ticks != 0) return 22;

    fixture.reset();
    fixture.queued[0] = {};
    fixture.queued[1] = {};
    fixture.queued[0].forward = true;
    fixture.queued[0].use_item = true;
    fixture.queued[0].sprint = true;
    fixture.queued[1].forward = true;
    processBridgeWorldTick(fixture.world, fixture.projectiles, fixture.initial, fixture.rules, fixture.match);
    if (fixture.players[0].sprinting || fixture.players[0].velocity.z <= 0.0
        || fixture.players[0].velocity.z >= fixture.players[1].velocity.z * 0.21) return 23;

    Block base[1]{Block::air};
    PlacedBlock overlay[1]{};
    fixture.world = WorldView(fixture.players, fixture.queued, fixture.held, &fixture.world_state, 2,
        {base, 1, 1, 1, 0, 0, 0, overlay}, math);
    fixture.rules.building.enabled = true;
    fixture.rules.building.zone_count = 1;
    fixture.rules.building.zones[0] = {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}};
    fixture.reset();
    fixture.match.kits[0].building.use_delay = 2;
    fixture.step(true, false);
    if (fixture.players[0].golden_apple_use_ticks != 0 || fixture.match.kits[0].golden_apples != 2) return 24;
    fixture.step(true, false);
    if (fixture.players[0].golden_apple_use_ticks != 31 || fixture.match.kits[0].building.use_delay != 4) return 25;
    return 0;
}

}

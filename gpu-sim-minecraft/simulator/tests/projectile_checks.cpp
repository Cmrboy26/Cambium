#include "minecraft_sim/projectiles.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

using namespace minecraft_sim;

// Simulator assertion harness for the independently source-derived fixtures below.
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Simulator numeric-check invariant: non-finite values never satisfy a source expectation.
void near(double actual, double expected, double tolerance, const char* message) {
        if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

struct Fixture {
    static constexpr int size_x = 16;
    static constexpr int size_y = 8;
    static constexpr int size_z = 5;
    std::array<PlayerState, 2> players{};
    std::array<Controls, 2> queued{};
    std::array<Controls, 2> held{};
    WorldState world_state{};
    std::array<Block, size_x * size_y * size_z> blocks{};
    SineTable sine_table = makeSineTable();

        // Simulator fixture storage contract: geometry.hpp BlockWorldView uses X/Y/Z-contiguous dense cells.
    BlockWorldView blockView() const {
        return {blocks.data(), size_x, size_y, size_z, 0, 0, 0};
    }

        // Source lookup contract: Mth.sin/cos (minecraft-26.3-src/net/minecraft/util/Mth.java:50).
        MathView math() const {
        return {sine_table.data()};
    }

        // Simulator fixture wiring: core.hpp WorldView owns no player or block storage.
        WorldView world() {
        return {players.data(), queued.data(), held.data(), &world_state, 2, blockView(), math()};
    }

        // Simulator fixture storage contract: geometry.hpp BlockWorldView::solid uses this cell ordering.
        void setBlock(int x, int y, int z) {
        blocks[(z * size_y + y) * size_x + x] = Block::solid;
    }
};

// Source: minecraft-26.3-src/net/minecraft/world/entity/player/Player.java, createAttributes (207-221);
// minecraft-26.3-src/net/minecraft/world/item/BowItem.java, BowItem constructor (20-22), getPowerForTime (74-80).
void checkBowItemMetadata() {
        const auto attributes = weaponAttributes(WeaponItem::bow);
        require(attributes.damage == 1.0 && attributes.speed == 4.0 && !attributes.sweep
                        && attributes.disable_shield_seconds == 0.0F,
                        "ordinary bow keeps the player's base melee attributes without axe or sword effects");
}

void checkChargeReleaseAndCancel() {
    Fixture fixture;
    std::array<ArrowState, 1> arrows{};
    ProjectileWorldState state;
    state.arrows = arrows.data();
    state.arrow_capacity = static_cast<int>(arrows.size());
    resetProjectileWorld(state, 1234);
    state.bows[0].ordinary_arrows = 2;
        fixture.players[0].weapon = WeaponItem::bow;
    auto world = fixture.world();
    const auto math = fixture.math();

    near(bowPower(2), 0.07, 1.0e-7, "two ticks remain below release threshold");
    near(bowPower(3), 0.1075, 1.0e-7, "three ticks pass release threshold");
    near(bowPower(20), 1.0, 0.0, "full draw reaches unit power");
    near(bowPower(40), 1.0, 0.0, "overdraw is capped");
    processBowCommand(world, math, 0, BowCommand::hold, state);
    processBowCommand(world, math, 0, BowCommand::hold, state);
    auto result = processBowCommand(world, math, 0, BowCommand::release, state);
    require(result.status == BowActionStatus::too_weak, "two-tick release must not shoot");
    require(state.bows[0].ordinary_arrows == 2 && !arrows[0].active,
            "weak release must preserve finite ammunition");

    for (int tick = 0; tick < 3; ++tick) processBowCommand(world, math, 0, BowCommand::hold, state);
    result = processBowCommand(world, math, 0, BowCommand::release, state);
    require(result.status == BowActionStatus::fired && result.arrow_slot == 0,
            "three-tick release allocates one arrow");
    require(state.bows[0].ordinary_arrows == 1 && arrows[0].active,
            "ammunition is consumed once at successful release");

    processBowCommand(world, math, 0, BowCommand::hold, state);
    processBowCommand(world, math, 0, BowCommand::cancel, state);
    require(!state.bows[0].using_bow && state.bows[0].draw_ticks == 0,
            "cancel clears only the bow draw");
    require(state.bows[0].ordinary_arrows == 1, "cancel does not consume ammunition");

    fixture.players[0].using_shield = true;
    result = processBowCommand(world, math, 0, BowCommand::hold, state);
    require(result.status == BowActionStatus::shield_conflict && !state.bows[0].using_bow,
            "explicit bow API rejects simultaneous shield use");
    fixture.players[0].using_shield = false;
    processBowCommand(world, math, 0, BowCommand::hold, state);
    fixture.players[0].using_shield = true;
    result = processBowCommand(world, math, 0, BowCommand::idle, state);
    require(result.status == BowActionStatus::shield_conflict && !state.bows[0].using_bow,
            "shield activation cancels an existing bow draw");
}

// Source boundary fixture: Projectile.shootFromRotation (Projectile.java:183-194), including known shooter movement.
void checkLaunchNoiseAndShooterMotion() {
    Fixture fixture;
    std::array<ArrowState, 1> arrows{};
    ProjectileWorldState state;
    state.arrows = arrows.data();
    state.arrow_capacity = 1;
    resetProjectileWorld(state, 77);
    state.bows[0].ordinary_arrows = 1;
        fixture.players[0].weapon = WeaponItem::bow;
    fixture.players[1].position = {0.0, 0.0, 10.0};
    fixture.players[0].yaw = -90.0F;
    fixture.players[0].velocity = {0.25, 0.1, -0.5};
    fixture.players[0].on_ground = false;
    auto world = fixture.world();
    const auto math = fixture.math();
    for (int tick = 0; tick < 20; ++tick) processBowCommand(world, math, 0, BowCommand::hold, state);
    const auto result = processBowCommand(world, math, 0, BowCommand::release, state);
    require(result.status == BowActionStatus::fired && arrows[0].critical,
            "full draw creates a critical ordinary arrow");
    require(arrows[0].position.y == fixture.players[0].eyePosition().y - double(0.1F),
            "arrow spawns at eye height minus the vanilla offset");
    require(std::abs(arrows[0].velocity.x - 3.25) < 0.052,
            "launch contains aim speed and shooter horizontal velocity");
    require(std::abs(arrows[0].velocity.y - 0.1) < 0.052,
            "airborne shooter vertical motion is inherited");
    require(std::abs(arrows[0].velocity.z + 0.5) < 0.052,
            "second horizontal shooter component is inherited");
        near(arrows[0].launch_target_distance, 9.7, 1.0e-6,
            "launched arrow records its initial distance from the target hitbox");

    resetProjectileWorld(state, 78);
    state.bows[0].ordinary_arrows = 1;
    fixture.players[0].velocity.y = 1.0;
    fixture.players[0].on_ground = true;
    for (int tick = 0; tick < 20; ++tick) processBowCommand(world, math, 0, BowCommand::hold, state);
    processBowCommand(world, math, 0, BowCommand::release, state);
    require(std::abs(arrows[0].velocity.y) < 0.052,
            "grounded shooter's vertical motion is not inherited");
}

// Source fixture: AbstractArrow.tick (AbstractArrow.java:222-266), position-before-drag/gravity in air.
void checkAirFlightRecurrence() {
    Fixture fixture;
    std::array<ArrowState, 1> arrows{};
    ProjectileWorldState state;
    state.arrows = arrows.data();
    state.arrow_capacity = 1;
    arrows[0].active = true;
    arrows[0].left_owner = true;
    arrows[0].position = {10.0, 3.0, 10.0};
    arrows[0].velocity = {1.0, 0.0, 0.0};
    const auto result = tickProjectiles(fixture.world(), fixture.blockView(), state);
    require(result.player_hits == 0 && result.terrain_hits == 0, "empty air flight has no collisions");
    near(arrows[0].position.x, 11.0, 1.0e-12, "position advances using pre-drag velocity");
    near(arrows[0].position.y, 3.0, 1.0e-12, "position uses pre-gravity velocity");
    near(arrows[0].velocity.x, double(0.99F), 1.0e-12, "air drag follows movement");
    near(arrows[0].velocity.y, -0.05, 1.0e-12, "gravity follows movement");
    require(arrows[0].age == 1, "airborne tick age advances once");
}

// Source fixture: AbstractArrow.tick clips blocks before player candidates (AbstractArrow.java:245-305).
void checkTerrainOccludesPlayer() {
    Fixture fixture;
    fixture.players[0].on_ground = true;
    fixture.players[1].position = {5.0, 0.0, 2.0};
    fixture.players[1].on_ground = true;
    fixture.setBlock(3, 1, 2);
    std::array<ArrowState, 1> arrows{};
    ProjectileWorldState state;
    state.arrows = arrows.data();
    state.arrow_capacity = 1;
    arrows[0].active = true;
    arrows[0].left_owner = true;
    arrows[0].owner = 0;
    arrows[0].launch_target_distance = 2.7;
    arrows[0].position = {1.0, 1.5, 2.0};
    arrows[0].velocity = {4.0, 0.0, 0.0};
    fixture.world_state.projectile_arrow_progress[0] = 99.0F;
    const auto result = processWorldTick(fixture.world(), fixture.blockView(), state);
    require(result.terrain_hits == 1 && result.player_hits == 0,
            "full-cube terrain occludes the player hit");
    require(arrows[0].in_ground && arrows[0].velocity.lengthSquared() == 0.0,
            "block impact persists an embedded stopped arrow");
        near(result.arrow_progress[0], 1.0 - 1.75 / 2.7, 1.0e-6,
         "terrain impact progress scales distance closed toward target bounds");
        require(result.arrow_progress[1] == 0.0F, "arrow progress is credited to its shooter");
        require(fixture.world_state.projectile_arrow_progress[0] == result.arrow_progress[0],
            "world state exposes this tick's projectile progress event");
        const auto resting_result = processWorldTick(fixture.world(), fixture.blockView(), state);
        require(resting_result.arrow_progress[0] == 0.0F, "resting arrows do not pay progress repeatedly");
        require(fixture.world_state.projectile_arrow_progress[0] == 0.0F,
            "world state clears projectile progress when the next tick has no event");
    require(fixture.players[1].health == 20.0F, "occluded player takes no damage");
    require(result.health_dealt[0] == 0.0F && result.health_taken[1] == 0.0F,
            "wall impacts produce no health reward event");
}

// Source fixture: AbstractArrow.canHitEntity (591), Projectile.checkLeftOwner (141-158), tickDespawn (371-376).
void checkOwnerExclusionAndLifetime() {
    Fixture fixture;
    fixture.players[0].position = {2.0, 0.0, 2.0};
    fixture.players[1].position = {12.0, 0.0, 2.0};
    std::array<ArrowState, 1> arrows{};
    ProjectileWorldState state;
    state.arrows = arrows.data();
    state.arrow_capacity = 1;
    arrows[0].active = true;
    arrows[0].owner = 0;
    arrows[0].position = {2.0, 1.62, 2.0};
    arrows[0].velocity = {1.0, 0.0, 0.0};
    tickProjectiles(fixture.world(), fixture.blockView(), state);
    require(fixture.players[0].health == 20.0F && !arrows[0].left_owner,
            "owner remains ineligible while inside owner collision clearance");
    for (int tick = 0; tick < 5; ++tick) tickProjectiles(fixture.world(), fixture.blockView(), state);
    require(arrows[0].left_owner, "arrow becomes eligible after leaving owner clearance");

    arrows[0].in_ground = true;
        arrows[0].in_ground_ticks = arrow_lifetime_ticks - 1;
    const auto result = tickProjectiles(fixture.world(), fixture.blockView(), state);
    require(result.expired == 1 && !arrows[0].active,
            "embedded arrow expires after its source lifetime");

        for (int owner = 0; owner < projectile_player_count; ++owner) {
        arrows[0] = {};
        arrows[0].active = true;
        arrows[0].owner = owner;
        arrows[0].left_owner = true;
        arrows[0].position = {fixture.players[owner].position.x + 2.0, 1.6, 2.0};
        arrows[0].velocity = {-3.0, 0.0, 0.0};
        const auto self_hit = tickProjectiles(fixture.world(), fixture.blockView(), state);
        require(self_hit.player_hits == 1 && fixture.players[owner].health == 14.0F,
            "returning arrows still inflict ordinary self-damage after clearing their owner");
        require(self_hit.health_taken[owner] == 6.0F && self_hit.health_taken[1 - owner] == 0.0F,
            "self-hits preserve only the damaged player's health-taken event");
        require(self_hit.health_dealt[0] == 0.0F && self_hit.health_dealt[1] == 0.0F,
            "self-hits must not generate damage-dealt credit for either player");
        }
}

// Source fixture: AbstractArrow.onHitEntity and Player.hurt's shield interaction (AbstractArrow.java:423-512; LivingEntity.java:1178-1350).
void checkArrowDamageAndShieldDeflection() {
    Fixture fixture;
    fixture.players[0].position = {1.0, 0.0, 2.0};
    fixture.players[1].position = {3.0, 0.0, 2.0};
    std::array<ArrowState, 1> arrows{};
    ProjectileWorldState state;
    state.arrows = arrows.data();
    state.arrow_capacity = 1;
    arrows[0].active = true;
        arrows[0].owner = 0;
    arrows[0].position = {1.0, 1.6, 2.0};
    arrows[0].velocity = {3.0, 0.0, 0.0};
    auto result = tickProjectiles(fixture.world(), fixture.blockView(), state);
    require(result.player_hits == 1 && !arrows[0].active,
            "successful ordinary arrow hit removes the projectile");
    require(fixture.players[1].health < 20.0F,
            "arrow damage reaches existing armor, immunity and absorption handling");
    require(result.health_dealt[0] == 20.0F - fixture.players[1].health
            && result.health_taken[1] == result.health_dealt[0],
            "projectile events report only owner-attributed health loss, not absorption");
    require(fixture.players[0].health == 20.0F,
            "the shooter is excluded before its arrow clears the owner");
    require(fixture.players[1].velocity.x > 0.0,
            "successful arrow hit applies knockback along projectile travel");

    Fixture shield_fixture;
    shield_fixture.players[0].position = {1.0, 0.0, 2.0};
    shield_fixture.players[1].position = {3.0, 0.0, 2.0};
    shield_fixture.players[1].yaw = 90.0F;
    shield_fixture.players[1].shield_equipped = true;
    shield_fixture.players[1].using_shield = true;
    shield_fixture.players[1].shield_use_ticks = 5;
    std::array<ArrowState, 1> shield_arrows{};
    ProjectileWorldState shield_state;
    shield_state.arrows = shield_arrows.data();
    shield_state.arrow_capacity = 1;
    shield_arrows[0].active = true;
        shield_arrows[0].owner = 0;
    shield_arrows[0].position = {1.0, 1.6, 2.0};
    shield_arrows[0].velocity = {3.0, 0.0, 0.0};
    result = tickProjectiles(shield_fixture.world(), shield_fixture.blockView(), shield_state);
    require(result.deflections == 1 && shield_arrows[0].active,
            "fully shield-blocked arrow is deflected instead of treated as a melee hit");
    require(shield_fixture.players[1].health == 20.0F
            && shield_fixture.players[1].last_blocked_damage > 0.0F,
            "shield blocks arrow damage through the existing player damage path");
    require(result.health_dealt[0] == 0.0F && result.health_taken[1] == 0.0F,
            "shield deflection produces no health reward event");
    near(shield_arrows[0].velocity.x, -0.3 * double(0.99F), 1.0e-6,
            "reverse deflection scales velocity by negative one tenth before air drag");
}

// Regression fixture: default PlayerView::hurt remains the existing melee path (LivingEntity.hurtServer, 1178-1260).
void checkMeleeDamageCompatibility() {
    Fixture blocked_fixture;
    blocked_fixture.players[0].position = {3.0, 0.0, 2.0};
    blocked_fixture.players[0].yaw = 90.0F;
    blocked_fixture.players[0].shield_equipped = true;
    blocked_fixture.players[0].using_shield = true;
    blocked_fixture.players[0].shield_use_ticks = 5;
    const bool blocked = blocked_fixture.world().player(0).hurt(4.0F, {2.0, 1.0, 2.0});
    require(!blocked && blocked_fixture.players[0].health == 20.0F
            && blocked_fixture.players[0].last_blocked_damage == 4.0F,
            "existing default melee call still uses its shield path");

    Fixture hit_fixture;
    hit_fixture.players[1].position = {3.0, 0.0, 2.0};
    const bool accepted = hit_fixture.world().player(1).hurt(4.0F, {2.0, 1.0, 2.0});
    require(accepted && hit_fixture.players[1].health == 16.0F
            && hit_fixture.players[1].velocity.x > 0.0,
            "existing default melee call still applies damage and knockback");
}

// Source: minecraft-26.3-src/net/minecraft/world/item/BowItem.java, use (95-106), releaseUsing (26-51);
// minecraft-26.3-src/net/minecraft/world/item/component/UseEffects.java, DEFAULT (10).
void checkIntegratedBowControlsAndMovement() {
    Fixture drawing;
    Fixture walking;
        drawing.players[1].position = {10.0, 0.0, 3.0};
        walking.players[1].position = {10.0, 0.0, 3.0};
    drawing.players[0].weapon = WeaponItem::bow;
    drawing.players[0].shield_equipped = true;
    drawing.players[0].on_ground = true;
    walking.players[0].on_ground = true;
    std::array<ArrowState, projectile_arrow_capacity> arrows{};
    ProjectileWorldState projectiles;
    projectiles.arrows = arrows.data();
    projectiles.arrow_capacity = projectile_arrow_capacity;
    resetProjectileWorld(projectiles, 11);
    projectiles.bows[0].ordinary_arrows = 1;

    Controls input;
    input.forward = true;
    input.sprint = true;
    input.attack = true;
    input.select_weapon = true;
    input.selected_slot = 0;
    input.weapon = WeaponItem::bow;
    input.use_item = true;
    drawing.world().player(0).setInputs(input);
    processWorldTick(drawing.world(), drawing.blockView(), projectiles);
    auto ordinary_input = input;
    ordinary_input.select_weapon = false;
    ordinary_input.attack = false;
    ordinary_input.use_item = false;
        ordinary_input.sprint = false;
    walking.world().player(0).setInputs(ordinary_input);
    walking.world().processTick();
    require(drawing.players[0].using_bow && projectiles.bows[0].using_bow
            && projectiles.bows[0].draw_ticks == 1,
            "first held use sample starts one actual bow draw tick");
    require(!drawing.players[0].using_shield && !drawing.players[0].sprinting
            && !drawing.players[0].last_attack.attempted,
            "bow use is distinct from shield use, blocks sprint start, and suppresses melee");
        near(drawing.players[0].velocity.z, walking.players[0].velocity.z * double(0.2F), 1.0e-8,
        "actual bow draw applies the vanilla default item-use movement multiplier");

    input.attack = false;
    for (int tick = 1; tick < 3; ++tick) {
        drawing.world().player(0).setInputs(input);
        processWorldTick(drawing.world(), drawing.blockView(), projectiles);
    }
    require(projectiles.bows[0].draw_ticks == 3, "held use samples continue the same draw");
        const Vec3 release_position = drawing.players[0].position;
    input.use_item = false;
        input.yaw_delta = 90.0F;
        input.attack = true;
    drawing.world().player(0).setInputs(input);
    processWorldTick(drawing.world(), drawing.blockView(), projectiles);
    require(!drawing.players[0].using_bow && !projectiles.bows[0].using_bow
            && projectiles.bows[0].draw_ticks == 0 && projectiles.bows[0].ordinary_arrows == 0
            && arrows[0].active && arrows[0].age == 0,
            "true-to-false releases once, consumes one arrow, and leaves it for the next flight phase");
        near(arrows[0].position.x, release_position.x, 0.0, "release precedes the sample's player movement");
        require(arrows[0].velocity.z > 0.3 && std::abs(arrows[0].velocity.x) < 0.05
                        && !drawing.players[0].last_attack.attempted,
                        "release uses pre-look aim and suppresses a simultaneous melee attack like the Java adapter");

    Fixture cancelled;
    cancelled.players[0].weapon = WeaponItem::bow;
    std::array<ArrowState, projectile_arrow_capacity> cancelled_arrows{};
    ProjectileWorldState cancelled_projectiles;
    cancelled_projectiles.arrows = cancelled_arrows.data();
    cancelled_projectiles.arrow_capacity = projectile_arrow_capacity;
    resetProjectileWorld(cancelled_projectiles, 12);
    cancelled_projectiles.bows[0].ordinary_arrows = 1;
        input.selected_slot = 0;
        input.weapon = WeaponItem::bow;
    input.use_item = true;
    input.attack = false;
        input.yaw_delta = 0.0F;
    for (int tick = 0; tick < 3; ++tick) {
        cancelled.world().player(0).setInputs(input);
        processWorldTick(cancelled.world(), cancelled.blockView(), cancelled_projectiles);
    }
    input.use_item = false;
    input.selected_slot = 1;
    input.weapon = WeaponItem::diamond_sword;
    cancelled.world().player(0).setInputs(input);
    processWorldTick(cancelled.world(), cancelled.blockView(), cancelled_projectiles);
    require(!cancelled.players[0].using_bow && !cancelled_projectiles.bows[0].using_bow
            && cancelled_projectiles.bows[0].ordinary_arrows == 1 && !cancelled_arrows[0].active,
            "physical slot change cancels a draw before release without consuming ammunition");
}

// Source fixture: ProjectileWeaponItem.draw consumes on successful release; simulator fixed-capacity invariant rejects overflow.
void checkCapacityAndSelectiveReset() {
    Fixture fixture;
    std::array<ArrowState, 1> first_arrows{};
    std::array<ArrowState, 1> second_arrows{};
    ProjectileWorldState first;
    ProjectileWorldState second;
    first.arrows = first_arrows.data();
    second.arrows = second_arrows.data();
    first.arrow_capacity = second.arrow_capacity = 1;
    resetProjectileWorld(first, 1);
    resetProjectileWorld(second, 2);
    first.bows[0].ordinary_arrows = 1;
        fixture.players[0].weapon = WeaponItem::bow;
    first_arrows[0].active = true;
    const auto world = fixture.world();
    const auto math = fixture.math();
    for (int tick = 0; tick < 20; ++tick) processBowCommand(world, math, 0, BowCommand::hold, first);
    const auto result = processBowCommand(world, math, 0, BowCommand::release, first);
    require(result.status == BowActionStatus::capacity_full && first.bows[0].ordinary_arrows == 1,
            "full fixed buffer rejects shot without consuming ammunition");

    second.bows[1].ordinary_arrows = 3;
    second_arrows[0].active = true;
    resetProjectileWorld(first, 9);
    require(!first_arrows[0].active && second_arrows[0].active
            && second.bows[1].ordinary_arrows == 3,
            "selective reset clears only the supplied world's sidecar state");
}

} // namespace

// Simulator test entry point; each case above cites its controlling Java source.
int main() {
    try {
                checkBowItemMetadata();
        checkChargeReleaseAndCancel();
        checkLaunchNoiseAndShooterMotion();
                checkIntegratedBowControlsAndMovement();
        checkAirFlightRecurrence();
        checkTerrainOccludesPlayer();
        checkOwnerExclusionAndLifetime();
        checkArrowDamageAndShieldDeflection();
        checkMeleeDamageCompatibility();
        checkCapacityAndSelectiveReset();
        std::cout << "Projectile checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

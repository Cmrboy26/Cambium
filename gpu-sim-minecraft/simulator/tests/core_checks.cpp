#include "minecraft_sim/core.hpp"
#include "minecraft_sim/bridge_observation.hpp"
#include "minecraft_sim/bridge_terrain_generation.hpp"
#include "minecraft_sim/comparison.hpp"
#include "minecraft_sim/constants_mace.hpp"
#include "minecraft_sim/terrain.hpp"
#include "minecraft_sim/objective_progress.hpp"
#include "../apps/configuration.hpp"
#include "../apps/observation.hpp"
#include "../apps/rewards.hpp"
#include "../apps/replay.hpp"
#include "bridge_match_checks.hpp"
#include "golden_apple_checks.hpp"
#include "terrain_checks.hpp"
#include <fstream>

#if defined(MCSIM_HAS_CUDA)
#include "minecraft_sim/cuda_backend.hpp"
#include <cuda_runtime_api.h>
#endif

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

const minecraft_sim::SineTable& testTable() {
    static const auto table = minecraft_sim::makeSineTable();
    return table;
}

void near(double actual, double expected, double tolerance, const char* message) {
    if (!(std::abs(actual - expected) <= tolerance)) {
        std::ostringstream detail;
        detail << std::setprecision(17) << message << ": actual=" << actual
               << ", expected=" << expected << ", tolerance=" << tolerance;
        throw std::runtime_error(detail.str());
    }
}

void checkObjectiveProgress() {
    using namespace minecraft_sim;
    ObjectiveProgress progress;
    progress.reset({0, 0, 0}, {0, 0, 10});
    auto delta = progress.advance({0, 0, 2}, false, false);
    near(delta.toward, 2, 1e-12, "objective closer");
    delta = progress.advance({0, 0, 1}, false, false);
    near(delta.away, 1, 1e-12, "objective farther");
    delta = progress.advance({0, 0, 2}, false, false);
    near(delta.toward, 0, 1e-12, "objective revisit");
    delta = progress.advance({0, 0, 0}, true, false);
    near(delta.away, 0, 1e-12, "objective respawn");
    delta = progress.advance({0, 0, 2}, false, false);
    near(delta.toward, 0, 1e-12, "objective respawn revisit");
    delta = progress.advance({0, 0, 0}, true, true);
    near(delta.toward, 0, 1e-12, "objective completion teleport");
    delta = progress.advance({0, 0, 2}, false, false);
    near(delta.toward, 2, 1e-12, "new objective progress");
}

void checkBridgeTerrainGeneration() {
    using namespace minecraft_sim;
    std::array<Block, 7 * 3 * 12> base{};
    BlockWorldView view{base.data(), 7, 3, 12, -3, 0, -6};
    for (int z = -6; z < 6; ++z) base[view.cellIndex(0, 0, z)] = Block::solid;
    TerrainOverlays terrain(view, 3);
    const BridgeTerrainGeneration config{0, 0, -5, 5, 4};
    require(generateBridgeTerrain(terrain.view(0), config, 42), "generate bridge terrain");
    require(generateBridgeTerrain(terrain.view(1), config, 42), "repeat bridge terrain");
    require(generateBridgeTerrain(terrain.view(2), config, 43), "alternate bridge terrain");
    int placed = 0;
    bool differs = false;
    for (int z = -6; z < 6; ++z) {
        for (int x = -3; x <= 3; ++x) {
            for (int y = 0; y < 3; ++y) {
                const auto index = view.cellIndex(x, y, z);
                const auto value = terrain.view(0).overlay[index];
                require(value == terrain.view(1).overlay[index], "bridge terrain seed repeatability");
                differs |= value != terrain.view(2).overlay[index];
                if (value == PlacedBlock::empty) continue;
                ++placed;
                require(z >= -5 && z < 5 && x != 0 && x >= -2 && x <= 2 && y <= 1,
                        "bridge terrain stays within side structures");
                if (y == 1) require(terrain.view(0).solid(x, 0, z), "bridge rail has floor support");
                if (x == -2 || x == 2) require(terrain.view(0).solid(x / 2, 0, z), "bridge extension is connected");
            }
        }
    }
    require(placed > 0 && differs, "bridge terrain varies across seeds");
    const BridgeTerrainGeneration invalid{0, 0, -7, 5, 4};
    require(!generateBridgeTerrain(terrain.view(0), invalid, 42), "reject invalid bridge terrain region");
}

void checkBridgeLocalVoxels() {
    using namespace minecraft_sim;
    std::array<Block, 75> base{};
    base[37] = Block::solid;
    base[0] = Block::solid;
    TerrainOverlays terrain({base.data(), 5, 3, 5, -2, -1, -2}, 1);
    const auto world = terrain.view(0);
    require(world.setPlacedBlock(1, 0, 2, PlacedBlock::team0)
        && world.setPlacedBlock(-1, 1, -2, PlacedBlock::team1), "voxel fixture placements");
    std::array<std::uint8_t, bridge_local_voxel_count> output{}, expected{};
    const auto index = [](int up, int right, int forward) { return (up * 9 + right) * 9 + forward; };
    for (int up = 0; up < 5; ++up) {
        for (int right = 0; right < 9; ++right) {
            for (int forward = 0; forward < 9; ++forward) {
                expected[index(up, right, forward)] = up >= 1 && up <= 3
                    && right >= 2 && right <= 6 && forward >= 2 && forward <= 6 ? 0 : 4;
            }
        }
    }
    expected[index(2, 4, 4)] = 1;
    expected[index(1, 2, 2)] = 1;
    expected[index(2, 5, 6)] = 2;
    expected[index(3, 3, 2)] = 3;
    for (float yaw : {0.0F, -45.0F, 360.0F, -360.0F,
                      std::nextafter(45.0F, 0.0F), std::nextafter(-45.0F, 0.0F),
                      (std::numeric_limits<float>::max)(), -(std::numeric_limits<float>::max)()}) {
        require(bridgeLocalVoxels(world, {0.5, 0.5, 0.5}, yaw, 0, output.data())
            && output == expected, "voxel zero-turn grid and yaw tie");
    }
    expected[index(2, 5, 6)] = 3;
    expected[index(3, 3, 2)] = 2;
    require(bridgeLocalVoxels(world, {0.5, 0.5, 0.5}, 0.0F, 1, output.data())
        && output == expected, "voxel team-relative ownership");
    expected[index(2, 5, 6)] = 0;
    expected[index(3, 3, 2)] = 0;
    expected[index(1, 2, 2)] = 0;
    expected[index(1, 2, 6)] = 1;
    expected[index(2, 6, 3)] = 2;
    expected[index(3, 2, 5)] = 3;
    for (float yaw : {45.0F, 90.0F, 450.0F, -315.0F, 0x1p102F,
                      std::nextafter(45.0F, (std::numeric_limits<float>::infinity)())}) {
        require(bridgeLocalVoxels(world, {0.5, 0.5, 0.5}, yaw, 0, output.data())
            && output == expected, "voxel quarter-turn grid and yaw tie");
    }
    require(bridgeLocalVoxels(world, {-0.01, -0.01, -0.01}, 0.0F, 0, output.data())
        && output[index(3, 5, 5)] == 1 && output[index(2, 3, 3)] == 1
        && output[index(3, 6, 7)] == 2 && output[index(4, 4, 3)] == 3
        && output[index(2, 2, 4)] == 4 && output[index(2, 3, 4)] == 0,
        "voxel negative floor and boundary");
    for (float yaw : {-0x1p102F, -90.0F,
                      std::nextafter(-45.0F, -(std::numeric_limits<float>::infinity)())}) {
        require(bridgeLocalVoxels(world, {0.0, 0.0, 0.0}, yaw, 0, output.data())
            && output[index(2, 2, 5)] == 2 && output[index(3, 6, 3)] == 3,
            "voxel negative yaw uses nonnegative modulo");
    }
    require(bridgeLocalVoxels(world, {0.0, 0.0, 0.0}, 0x1p105F, 0, output.data())
        && output[index(2, 3, 2)] == 2 && output[index(3, 5, 6)] == 3,
        "voxel huge yaw half-turn");
    for (double coordinate : {(std::numeric_limits<double>::max)(), -(std::numeric_limits<double>::max)()}) {
        for (Vec3 feet : {Vec3{coordinate, 0.0, 0.0}, Vec3{0.0, coordinate, 0.0}, Vec3{0.0, 0.0, coordinate}}) {
            require(bridgeLocalVoxels(world, feet, (std::numeric_limits<float>::max)(), 0, output.data()),
                "voxel extreme finite input accepted");
            for (auto value : output) require(value == 4, "voxel extreme coordinates remain outside");
        }
    }
    const Block edge_cells[2]{Block::solid, Block::air};
    const BlockWorldView upper{edge_cells, 2, 1, 1, (std::numeric_limits<int>::max)() - 1, 0, 0};
    require(bridgeLocalVoxels(upper, {double((std::numeric_limits<int>::max)()), 0.0, 0.0}, 0.0F, 0, output.data())
        && output[index(2, 3, 4)] == 1 && output[index(2, 4, 4)] == 0 && output[index(2, 5, 4)] == 4,
        "voxel upper integer edge");
    require(bridgeLocalVoxels(upper, {double((std::numeric_limits<int>::max)()) + 4.0, 0.0, 0.0},
                              0.0F, 0, output.data())
        && output[index(2, 0, 4)] == 0 && output[index(2, 1, 4)] == 4,
        "voxel window reaches upper integer edge from outside");
    const BlockWorldView lower{edge_cells, 2, 1, 1, (std::numeric_limits<int>::min)(), 0, 0};
    require(bridgeLocalVoxels(lower, {double((std::numeric_limits<int>::min)()) - 0.5, 0.0, 0.0}, 0.0F, 0, output.data())
        && output[index(2, 4, 4)] == 4 && output[index(2, 5, 4)] == 1 && output[index(2, 6, 4)] == 0,
        "voxel lower integer edge and negative floor");
    require(bridgeLocalVoxels(lower, {double((std::numeric_limits<int>::min)()) - 4.0, 0.0, 0.0},
                              0.0F, 0, output.data())
        && output[index(2, 7, 4)] == 4 && output[index(2, 8, 4)] == 1,
        "voxel window reaches lower integer edge from outside");
    require(!bridgeLocalVoxels(world, {0.0, 0.0, 0.0}, (std::numeric_limits<float>::infinity)(), 0, output.data())
        && !bridgeLocalVoxels(world, {0.0, 0.0, 0.0}, (std::numeric_limits<float>::quiet_NaN)(), 0, output.data())
        && !bridgeLocalVoxels(world, {0.0, -(std::numeric_limits<double>::infinity)(), 0.0}, 0.0F, 0, output.data())
        && !bridgeLocalVoxels(world, {(std::numeric_limits<double>::quiet_NaN)(), 0.0, 0.0}, 0.0F, 0, output.data())
        && !bridgeLocalVoxels(world, {0.0, 0.0, 0.0}, 0.0F, 2, output.data())
        && !bridgeLocalVoxels({}, {0.0, 0.0, 0.0}, 0.0F, 0, output.data())
        && !bridgeLocalVoxels(world, {0.0, 0.0, 0.0}, 0.0F, 0, nullptr), "voxel invalid input rejected");
    const int maximum = (std::numeric_limits<int>::max)();
    const int minimum = (std::numeric_limits<int>::min)();
    require(!bridgeLocalVoxels({edge_cells, 2, 1, 1, maximum, 0, 0}, {0.0, 0.0, 0.0}, 0.0F, 0, output.data())
        && !bridgeLocalVoxels({edge_cells, maximum, maximum, maximum, minimum, minimum, minimum},
                              {0.0, 0.0, 0.0}, 0.0F, 0, output.data()),
        "voxel unrepresentable terrain bounds or index rejected");
    world.overlay[37] = PlacedBlock::team0;
    require(!bridgeLocalVoxels(world, {0.0, 0.0, 0.0}, 0.0F, 0, output.data()),
        "voxel invalid protected overlay rejected");
    world.overlay[37] = PlacedBlock::empty;
    world.overlay[38] = static_cast<PlacedBlock>(3);
    require(!bridgeLocalVoxels(world, {0.0, 0.0, 0.0}, 0.0F, 0, output.data()),
        "voxel unknown ownership rejected");
    world.overlay[38] = PlacedBlock::empty;
    base[38] = static_cast<Block>(2);
    require(!bridgeLocalVoxels(world, {0.0, 0.0, 0.0}, 0.0F, 0, output.data()),
        "voxel unknown base state rejected");
}

void checkGoldenAppleBoundary() {
    using namespace minecraft_sim;
    PlayerState players[2]{}, initial[2]{};
    initial[0].position = {0.0, 10.0, 0.0};
    initial[1].position = {5.0, 10.0, 0.0};
    Controls queued[2]{}, held[2]{};
    WorldState world_state;
    WorldView world(players, queued, held, &world_state, 2, {}, {testTable().data()});
    ProjectileWorldState projectiles;
    BridgeMatchRules rules;
    rules.cage_ticks = 0;
    rules.void_y = -10000.0;
    rules.goals[0] = {{-100.0, -10000.0, -1.0}, {-99.0, 100.0, 1.0}};
    rules.goals[1] = {{100.0, -10000.0, -1.0}, {101.0, 100.0, 1.0}};
    rules.kits[0].explicit_hotbar = true;
    rules.kits[0].slot_count = 1;
    rules.kits[0].slots[0] = BridgeHotbarItem::golden_apple;
    rules.kits[0].golden_apples = 2;
    BridgeMatchState match;
    resetBridgeMatch(world, projectiles, initial, rules, match, 1);
    players[0].health = 10.0F;
    for (int tick = 0; tick < 31; ++tick) {
        queued[0].use_item = true;
        processBridgeWorldTick(world, projectiles, initial, rules, match);
    }
    require(match.kits[0].golden_apples == 2 && players[0].golden_apple_use_ticks == 1
        && players[0].absorption == 0.0F, "apple remains unconsumed before tick 32");
    queued[0].use_item = true;
    processBridgeWorldTick(world, projectiles, initial, rules, match);
    require(match.kits[0].golden_apples == 1 && players[0].golden_apple_use_ticks == 0
        && players[0].absorption == 4.0F && players[0].health == 10.0F
        && players[0].golden_apple_regeneration_ticks == 100, "apple completes on tick 32");
    queued[0].use_item = false;
    processBridgeWorldTick(world, projectiles, initial, rules, match);
    require(players[0].health == 11.0F && players[0].golden_apple_regeneration_ticks == 99
        && match.kits[0].golden_apples == 1, "regeneration starts on next tick");
}

void checkStandaloneItemTick() {
    using namespace minecraft_sim;
    Block cells[7 * 4 * 7]{};
    PlacedBlock overlay[7 * 4 * 7]{};
    BlockWorldView blocks{cells, 7, 4, 7, -3, 0, -3, overlay};
    for (int z = -3; z < 4; ++z) {
        for (int x = -3; x < 4; ++x) cells[blocks.cellIndex(x, 0, z)] = Block::solid;
    }
    cells[blocks.cellIndex(0, 2, 1)] = Block::solid;
    PlayerState players[2]{};
    players[0].position = {0.5, 1.0, -1.5};
    players[1].position = {-2.5, 1.0, 1.5};
    players[0].on_ground = players[1].on_ground = true;
    Controls queued[2]{}, held[2]{};
    WorldState world_state;
    WorldView world(players, queued, held, &world_state, 2, blocks, {testTable().data()});
    ProjectileWorldState projectiles;
    ItemKitConfig configs[2]{};
    configs[0].explicit_hotbar = true;
    configs[0].slot_count = 3;
    configs[0].slots[0] = ItemHotbarItem::blocks;
    configs[0].slots[1] = ItemHotbarItem::pickaxe;
    configs[0].slots[2] = ItemHotbarItem::golden_apple;
    ItemKitState kits[2]{};
    kits[0].blocks = 1;
    kits[0].golden_apples = 1;
    ItemBuildRules building;
    building.enabled = true;
    building.zone_count = 1;
    building.zones[0] = {{-3, 0, -3}, {4, 4, 4}};
    building.hardness = 1;
    building.mining_speed = 15;
    queued[0].select_weapon = true;
    queued[0].selected_slot = 0;
    queued[0].use_item = true;
    auto events = processItemWorldTick(world, projectiles, configs, building, kits);
    require(events.placed[0] && kits[0].blocks == 0, "standalone item placement");
    queued[0] = {};
    queued[0].select_weapon = true;
    queued[0].selected_slot = 1;
    queued[0].attack = true;
    const auto mining = queued[0];
    processItemWorldTick(world, projectiles, configs, building, kits);
    queued[0] = mining;
    processItemWorldTick(world, projectiles, configs, building, kits);
    near(itemProgressState(players[0], kits[0]).mining_progress, 0.5, 1e-6, "standalone mining progress");
    queued[0] = mining;
    events = processItemWorldTick(world, projectiles, configs, building, kits);
    require(events.broke[0] && itemProgressState(players[0], kits[0]).mining_delay == 5,
            "standalone mining completion");
    queued[0] = {};
    queued[0].select_weapon = true;
    queued[0].selected_slot = 2;
    queued[0].use_item = true;
    const auto eating = queued[0];
    processItemWorldTick(world, projectiles, configs, building, kits);
    require(itemProgressState(players[0], kits[0]).eating_remaining == 31, "standalone eating progress");
    for (int tick = 0; tick < 31; ++tick) {
        queued[0] = eating;
        processItemWorldTick(world, projectiles, configs, building, kits);
    }
    require(kits[0].golden_apples == 0 && itemProgressState(players[0], kits[0]).eating_remaining == 0,
            "standalone eating completion");
}

void checkMutableTerrain() {
    using namespace minecraft_sim;
    std::array<Block, 8> base{Block::solid, Block::solid, Block::solid, Block::solid,
                              Block::air, Block::air, Block::air, Block::air};
    const BlockWorldView static_world{base.data(), 4, 2, 1, -2, 0, 0};
    TerrainOverlays terrain(static_world, 2);
    const auto first = terrain.view(0), second = terrain.view(1);
    require(terrain_checks::orderedEditsAndResets(first), "ordered terrain edits and reset policy");
    const Vec3 origin{-2.0, 1.5, 0.5}, delta{4.0, 0.0, 0.0};
    const Aabb first_cell{{0.1, 1.1, 0.1}, {0.9, 1.9, 0.9}};
    near(blockRayFraction(first, origin, delta), 1.0, 0.0, "initial first-world ray");
    near(blockRayFraction(second, origin, delta), 1.0, 0.0, "initial second-world ray");
    require(first.setPlacedBlock(0, 1, 0, PlacedBlock::team0), "first-world placement");
    require(!noBlockCollision(first, first_cell), "first-world placed collision");
    require(noBlockCollision(second, first_cell), "second world remains empty");
    require(noBlockCollision(static_world, first_cell), "static world remains empty");
    near(blockRayFraction(first, origin, delta), 0.5, 0.0, "first-world placed ray");
    near(blockRayFraction(second, origin, delta), 1.0, 0.0, "second-world isolated ray");
    require(second.setPlacedBlock(1, 1, 0, PlacedBlock::team1), "second-world placement");
    near(blockRayFraction(second, origin, delta), 0.75, 0.0, "second-world placed ray");
    require(!first.setPlacedBlock(0, 0, 0, PlacedBlock::team0), "protected base cannot be edited");
    require(!first.setPlacedBlock(2, 1, 0, PlacedBlock::team0), "outside edit rejected");
    require(!static_world.setPlacedBlock(0, 1, 0, PlacedBlock::team0), "static edit rejected");
    require(!first.setPlacedBlock(0, 1, 0, static_cast<PlacedBlock>(3)), "invalid team rejected");
    const std::array<std::uint8_t, 2> reset_mask{1, 0};
    terrain.reset(reset_mask.data(), TerrainReset::player_respawn, false);
    require(first.solid(0, 1, 0) && second.solid(1, 1, 0), "respawn preserves overlays");
    terrain.reset(reset_mask.data(), TerrainReset::round);
    require(first.solid(0, 1, 0) && second.solid(1, 1, 0), "default goal reset preserves overlays");
    terrain.reset(reset_mask.data(), TerrainReset::round, false);
    require(!first.solid(0, 1, 0) && second.solid(1, 1, 0), "goal clear affects only selected overlay");
    require(first.setPlacedBlock(0, 1, 0, PlacedBlock::team0), "replace first-world block");
    terrain.reset(reset_mask.data());
    near(blockRayFraction(first, origin, delta), 1.0, 0.0, "selected reset clears first ray");
    near(blockRayFraction(second, origin, delta), 0.75, 0.0, "selected reset preserves second ray");
    require(noBlockCollision(first, first_cell) && second.overlay[7] == PlacedBlock::team1,
            "selected reset preserves other world identity");
    require(first.solid(0, 0, 0) && second.solid(0, 0, 0), "protected base survives reset");
    require(second.setPlacedBlock(1, 1, 0, PlacedBlock::empty), "placed block removal");
    near(blockRayFraction(second, origin, delta), 1.0, 0.0, "removed block no longer occludes ray");
    bool overflow_rejected = false;
    try {
        TerrainOverlays oversized({base.data(), (std::numeric_limits<int>::max)(),
            (std::numeric_limits<int>::max)(), 3}, 2);
    } catch (const std::invalid_argument&) {
        overflow_rejected = true;
    }
    require(overflow_rejected, "terrain index overflow rejected before allocation");
        const auto maximum = (std::numeric_limits<std::size_t>::max)();
        require(checkedTerrainProduct(maximum, 1) == maximum && checkedTerrainSum(maximum - 1, 1) == maximum,
            "terrain size arithmetic accepts the exact representable limit");
        bool product_overflow_rejected = false, sum_overflow_rejected = false;
        try { (void)checkedTerrainProduct(maximum, 2); }
        catch (const std::invalid_argument&) { product_overflow_rejected = true; }
        try { (void)checkedTerrainSum(maximum, 1); }
        catch (const std::invalid_argument&) { sum_overflow_rejected = true; }
        require(product_overflow_rejected && sum_overflow_rejected,
            "terrain size multiplication and addition reject overflow");
    TerrainOverlays exact_budget(static_world, 2, 24);
    bool budget_rejected = false;
    try {
        TerrainOverlays exceeds_budget(static_world, 2, 23);
    } catch (const std::invalid_argument&) {
        budget_rejected = true;
    }
    require(budget_rejected, "terrain budget rejects allocation over the limit");
    require(parseTerrainBudget(nullptr) == default_terrain_budget_bytes && parseTerrainBudget("24") == 24,
            "terrain budget accepts default and positive decimal values");
    for (const char* value : {"", "0", "-1", "+1", "1x", "9223372036854775808"}) {
        bool invalid_rejected = false;
        try { (void)parseTerrainBudget(value); } catch (const std::invalid_argument&) { invalid_rejected = true; }
        require(invalid_rejected, "invalid terrain budget rejected");
    }
}

void checkCompositeWorldConsumers() {
    using namespace minecraft_sim;
    constexpr int environments = 3;
    std::array<Block, 3 * 4 * 5> base{};
    for (int z = 0; z < 5; ++z) {
        for (int x = 0; x < 3; ++x) base[z * 12 + x] = Block::solid;
    }
    TerrainOverlays terrain({base.data(), 3, 4, 5, -1, 0, 0}, environments);
    std::array<PlayerState, environments * 2> players{};
    std::array<Controls, environments * 2> queued{}, held{};
    std::array<WorldState, environments> worlds{};
    std::array<ProjectileWorldState, environments> projectiles{};
    std::array<ArrowState, environments> arrows{};
    BatchView batch{players.data(), queued.data(), held.data(), worlds.data(), environments, 2,
                    terrain.data(), {testTable().data()}, projectiles.data()};
    app::DuelRewardSettings settings;
    settings.max_ticks = 100;
    settings.shaping_weights[0] = 1.0F;
    settings.shaping_weights[3] = 1.0F;
    for (int environment = 0; environment < environments; ++environment) {
        players[environment * 2].position = {0.5, 1.0, 0.5};
        players[environment * 2].velocity.z = 0.6;
        players[environment * 2 + 1].position = {0.5, 1.0, 2.5};
        queued[environment * 2].attack = true;
        arrows[environment].position = {0.5, 2.5, 0.5};
        arrows[environment].velocity = {0.0, 0.0, 1.0};
        arrows[environment].owner = 0;
        arrows[environment].active = true;
        projectiles[environment].arrows = &arrows[environment];
        projectiles[environment].arrow_capacity = 1;
        if (environment != 1) {
            const auto team = environment == 0 ? PlacedBlock::team0 : PlacedBlock::team1;
            require(terrain.view(environment).setPlacedBlock(0, 1, 1, team)
                && terrain.view(environment).setPlacedBlock(0, 2, 1, team), "place composite wall");
        }
    }
    for (int environment = 0; environment < environments; ++environment) {
        const bool blocked = environment != 1;
        float observation[22]{};
        app::baseWorldObservation(players[environment * 2], players[environment * 2 + 1],
                                  terrain.view(environment), observation);
        near(observation[10], blocked ? 0.01 : 1.0, 1.0e-7, "per-world observation ray");
        app::DuelRewardState reward_state;
        const auto reward = app::duelRewards(players.data() + environment * 2, worlds[environment],
            terrain.view(environment), projectiles[environment], settings, reward_state);
        near(reward.rewards[0], blocked ? 0.0 : 2.0, 0.0, "per-world aim reward visibility");
        require(batch.world(environment).pickAttackTarget(0) == (blocked ? -1 : 1),
                "per-world melee targeting");
        processWorldTick(batch.world(environment), projectiles[environment]);
        const auto& player = players[environment * 2];
        require(player.last_attack.target == (blocked ? -1 : 1)
            && player.last_attack.hit_block == blocked, "per-world attack tick");
        require(blocked ? player.position.z < 0.71 : player.position.z > 1.0,
                "per-world movement collision");
        require(arrows[environment].in_ground == blocked, "per-world projectile collision");
        if (blocked) near(arrows[environment].position.z, 1.0 - double(0.05F), 1.0e-12,
                          "placed-block projectile impact");
    }
    const std::array<std::uint8_t, environments> mask{1, 0, 0};
    terrain.reset(mask.data());
    require(!terrain.view(0).solid(0, 2, 1) && terrain.view(2).solid(0, 2, 1)
        && terrain.view(0).solid(0, 0, 1), "selective reset preserves other overlay and protected base");
    const auto remaining = terrain.view(2);
    const auto index = remaining.cellIndex(0, 2, 1);
    require(remaining.overlay[index] == PlacedBlock::team1
        && remaining.setPlacedBlock(0, 2, 1, PlacedBlock::empty)
        && remaining.setPlacedBlock(0, 2, 1, PlacedBlock::team0)
        && remaining.overlay[index] == PlacedBlock::team0, "ordered edits preserve team identity");
}

void checkPolicyFixtures() {
    using namespace minecraft_sim;
    const auto fixture = nlohmann::json::parse(std::ifstream(MCSIM_CONTRACT_FIXTURE_PATH));
    const double tolerance = fixture.at("tolerance");
    const auto state = [](const nlohmann::json& value) {
        PlayerState player;
        const auto position = value.at("position").get<std::array<double, 3>>();
        player.position = {position[0], position[1], position[2]};
        const auto velocity = value.at("velocity").get<std::array<double, 3>>();
        player.velocity = {velocity[0], velocity[1], velocity[2]};
        player.health = value.at("health"); player.yaw = value.at("yaw"); player.pitch = value.at("pitch");
        player.on_ground = value.at("on_ground");
        player.sprinting = value.at("sprinting");
        return player;
    };
    for (const auto& sample : fixture.at("observations")) {
        const auto source = state(sample.at("source"));
        const auto target = state(sample.at("target"));
        const auto actual = app::baseObservation(source, target,
            sample.at("rays").get<std::array<float, 8>>());
        const auto expected = sample.at("expected").get<std::vector<float>>();
        require(actual.size() == expected.size(), "shared base observation size");
        for (std::size_t index = 0; index < actual.size(); ++index) {
            near(actual[index], expected[index], tolerance, sample.at("name").get_ref<const std::string&>().c_str());
        }
        std::array<float, app::feature_names.size()> catalogue{};
        app::velocityObservation(catalogue.data(), source, target);
        const auto expected_velocity = sample.at("expected_velocity").get<std::array<float, 6>>();
        for (std::size_t index = 0; index < expected_velocity.size(); ++index) {
            near(catalogue[79 + index], expected_velocity[index], tolerance, "shared velocity observation");
        }
        app::movementObservation(catalogue.data(), source, target);
        const auto expected_movement = sample.at("expected_movement").get<std::array<float, 4>>();
        for (std::size_t index = 0; index < expected_movement.size(); ++index) {
            near(catalogue[85 + index], expected_movement[index], tolerance, "shared movement observation");
        }
        app::movementObservation(catalogue.data(), target, source);
        for (std::size_t index = 0; index < expected_movement.size(); ++index) {
            near(catalogue[85 + index], expected_movement[index ^ 1], tolerance, "opponent movement perspective");
        }
    }
    std::array<std::vector<WeaponItem>, 2> slots;
    for (auto& actor : slots) for (const auto& item : fixture.at("weapons")) actor.push_back(app::parseWeapon(item));
    const app::PolicyLayout layout(fixture.at("policy"), slots);
    const auto catalogue = fixture.at("catalogue").get<std::vector<float>>();
    std::vector<float> projected(layout.observations.size());
    layout.project(catalogue.data(), projected.data());
    for (std::size_t index = 0; index < layout.observations.size(); ++index) {
        near(projected[index],
            fixture.at("projected").at(index).get<float>(), tolerance, "shared policy projection");
    }
    for (const auto& sample : fixture.at("projection_cases")) {
        auto descriptor = fixture.at("policy");
        descriptor["observations"] = sample.at("observations");
        const app::PolicyLayout encoded(descriptor, slots);
        std::vector<float> actual(encoded.observations.size());
        encoded.project(catalogue.data(), actual.data());
        const auto expected = sample.at("expected").get<std::vector<float>>();
        require(actual.size() == expected.size(), "expanded observation projection size");
        for (std::size_t index = 0; index < actual.size(); ++index) {
            near(actual[index], expected[index], tolerance, sample.at("name").get_ref<const std::string&>().c_str());
            near(app::projectObservation(catalogue.data(), encoded.projection(index)), expected[index], tolerance,
                "shared device-compatible projection");
        }
    }
    auto oversized = fixture.at("policy");
    oversized["observations"] = nlohmann::json::array();
    for (int index = 0; index < 15; ++index) oversized["observations"].push_back(
        {{"feature", "self.weapon_slot"}, {"encoding", "one_hot"}, {"categories", {0, 1, 2, 3, 4, 5, 6, 7, 8}}});
    bool overflow_rejected = false;
    try { const app::PolicyLayout unused(oversized, slots); }
    catch (const std::exception&) { overflow_rejected = true; }
    require(overflow_rejected, "expanded observation limit must be enforced");
    const auto action = fixture.at("action").get<std::vector<std::int32_t>>();
    const auto control = layout.decode(action.data(), slots[0]);
    const std::array<PlayerState, 2> actors{};
    const std::array<Controls, 2> controls{control, control};
    const auto decoded = app::replayFrame(0, actors.data(), controls.data(), true).at("actors").at(0).at("controls");
    require(decoded == fixture.at("controls"), "shared semantic action decoding");
    Controls unchanged;
    unchanged.jump = true;
    auto invalid_action = action;
    invalid_action[0] = -1;
    require(!app::decodeAction(layout.packed_actions.data(), static_cast<int>(layout.packed_actions.size()),
        invalid_action.data(), slots[0].data(), static_cast<int>(slots[0].size()), unchanged)
        && unchanged.jump, "invalid action leaves controls unchanged");
    invalid_action = action;
    invalid_action[0] = static_cast<std::int32_t>(layout.packed_actions[0].count);
    require(!app::decodeAction(layout.packed_actions.data(), static_cast<int>(layout.packed_actions.size()),
        invalid_action.data(), slots[0].data(), static_cast<int>(slots[0].size()), unchanged)
        && unchanged.jump, "out-of-range action leaves controls unchanged");
    for (const auto& invalid : fixture.at("invalid")) {
        auto descriptor = fixture.at("policy");
        descriptor[nlohmann::json::json_pointer(invalid.at("pointer").get<std::string>())] = invalid.at("value");
        bool rejected = false;
        try { const app::PolicyLayout unused(descriptor, slots); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, "shared invalid descriptor must fail");
    }
}

void checkEquipmentObservation() {
    using namespace minecraft_sim;
    require(app::parseWeapon("minecraft:mace") == WeaponItem::mace
        && std::string_view(app::weaponName(WeaponItem::mace)) == "minecraft:mace",
        "mace loadouts round-trip through the native item names");
    PlayerState source, target;
    const std::vector<WeaponItem> own{WeaponItem::wooden_sword, WeaponItem::netherite_axe, WeaponItem::unarmed};
    app::applyLoadout(source, {{"weapons", {"minecraft:wooden_sword", "minecraft:netherite_axe", "minecraft:air"}},
        {"selected", 1}, {"offhand", "minecraft:shield"}, {"armor", {
            {"boots", "minecraft:netherite_boots"}, {"leggings", "minecraft:netherite_leggings"},
            {"chestplate", "minecraft:netherite_chestplate"}, {"helmet", "minecraft:netherite_helmet"}}}});
    app::applyLoadout(target, {{"weapons", {"minecraft:iron_axe", "minecraft:diamond_sword"}},
        {"selected", 1}, {"armor", {{"chestplate", "minecraft:diamond_chestplate"}}}});
    std::array<float, app::feature_names.size()> catalogue;
    catalogue.fill(-1.0F);
    app::equipmentObservation(catalogue.data(), source, target, own);
    const auto feature = [&](const char* name) { return catalogue[app::nameIndex(app::feature_names, name)]; };
    near(feature("self.has_shield"), 1, 0, "shield ownership is independent of using/blocking");
    near(feature("opponent.has_shield"), 0, 0, "opponent without shield");
    near(feature("self.weapon_damage_fraction"), 1, 0, "netherite axe damage normalized by ten");
    near(feature("self.weapon_speed_fraction"), 0.25, 0, "netherite axe speed normalized by four");
    near(feature("opponent.weapon_damage_fraction"), 0.7, 1e-6, "selected diamond sword damage");
    near(feature("opponent.weapon_speed_fraction"), 0.4, 1e-6, "selected sword speed preserves widened float modifier");
    near(feature("self.armor_defense_fraction"), 1, 0, "netherite full defense normalized by twenty");
    near(feature("self.armor_toughness_fraction"), 1, 0, "netherite toughness normalized by twelve");
    near(feature("opponent.armor_defense_fraction"), 0.4, 1e-6, "single chestplate defense");
    near(feature("opponent.armor_toughness_fraction"), 1.0 / 6.0, 1e-6, "single diamond chestplate toughness");
    near(feature("self.weapon_slot_0.present"), 1, 0, "unselected wooden sword remains visible");
    near(feature("self.weapon_slot_0.is_axe"), 0, 0, "slot sword kind");
    near(feature("self.weapon_slot_0.damage_fraction"), 0.4, 1e-6, "slot sword damage");
    for (int slot = 2; slot < 9; ++slot) {
        for (int field = 0; field < 4; ++field) near(catalogue[43 + slot * 4 + field], 0, 0, "air/missing slots are all zero");
    }
    std::vector<WeaponItem> nine(9, WeaponItem::unarmed);
    nine[8] = WeaponItem::netherite_sword;
    app::equipmentObservation(catalogue.data(), target, source, nine);
    near(feature("self.weapon_slot_8.damage_fraction"), 0.8, 1e-6, "ninth own slot is encoded");
    source.attack_damage = 100; source.attack_speed = 100; source.armor = 100; source.armor_toughness = 100;
    target.attack_damage = -1; target.attack_speed = -1; target.armor = -1; target.armor_toughness = -1;
    app::equipmentObservation(catalogue.data(), source, target, {});
    for (int offset : {35, 37, 39, 41}) {
        near(catalogue[offset], 1, 0, "equipment upper bound");
        near(catalogue[offset + 1], 0, 0, "equipment lower bound");
    }
    PlayerState unarmed;
    app::applyLoadout(unarmed, {{"weapons", {"minecraft:air"}}});
    app::equipmentObservation(catalogue.data(), unarmed, unarmed, {WeaponItem::unarmed});
    near(feature("self.weapon_damage_fraction"), 0.1, 1e-6, "selected unarmed base attack damage");
    near(feature("self.weapon_speed_fraction"), 1, 0, "selected unarmed base attack speed");
    for (int index = 0; index < 33; ++index) near(catalogue[index], -1, 0, "legacy equipment observations remain untouched");
    PlayerState shielded, opponent;
    shielded.weapon = WeaponItem::iron_axe;
    shielded.using_shield = true;
    shielded.shield_use_ticks = 3;
    shielded.shield_cooldown = 75;
    shielded.attack_strength_ticks = 100;
    opponent.weapon = WeaponItem::diamond_sword;
    catalogue.fill(-1.0F);
    app::legacyEquipmentObservation(catalogue.data(), shielded, opponent, 99);
    const std::array<float, 9> expected{1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 1.0F, 0.75F, 1.0F, 0.6F};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        near(catalogue[22 + index], expected[index], 1e-6, "legacy equipment observation");
    }
    require(catalogue[21] == -1.0F && catalogue[31] == -1.0F, "legacy encoder touches only its own fields");
    app::legacyEquipmentObservation(catalogue.data(), opponent, shielded, 0);
    require(catalogue[22] == 0.0F && catalogue[23] == 1.0F && catalogue[24] == 0.0F
        && catalogue[25] == 1.0F && catalogue[26] == 0.0F && catalogue[27] == 0.0F
        && catalogue[30] == 0.0F, "legacy equipment observation follows actor perspective");
    std::cout << "PASS: descriptor equipment fractions, own hotbar, empty slots and legacy isolation\n";
}

void checkShapingReward() {
    using namespace minecraft_sim;
    PlayerState source, target;
    source.position = {0.0, 0.0, 0.0};
    target.position = {0.0, 0.0, 5.0};
    const std::array<float, 5> weights{2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::array<float, 5> actual{};
    app::shapingRewards(source, target, 0, {}, weights.data(), actual.data());
    near(actual[0], 2.0, 1e-6, "aim reward for centered target");
    near(actual[1], 15.0 / 7.0, 1e-6, "proximity reward for five-block distance");
    near(actual[2], 4.0, 1e-6, "pitch reward for horizontal aim");
    near(actual[3], 5.0, 1e-6, "aim-hold reward for centered target");
    near(actual[4], 0.0, 0.0, "no extreme-pitch reward below threshold");
    source.pitch = 61.0F;
    target.position = source.position;
    app::shapingRewards(source, target, 0, {}, weights.data(), actual.data());
    near(actual[0], 0.0, 0.0, "coincident targets do not produce aim reward");
    near(actual[1], 3.0, 0.0, "proximity reward clamps at one");
    near(actual[2], 4.0 * 29.0 / 90.0, 1e-6, "pitch reward follows pitch magnitude");
    near(actual[3], 0.0, 0.0, "coincident targets do not produce aim-hold reward");
    near(actual[4], 6.0, 0.0, "extreme-pitch threshold uses a strict comparison");
}

void checkDuelRewards() {
    using namespace minecraft_sim;
    std::array<PlayerState, 2> players{};
    players[0].position = {0.5, 1.0, 0.5};
    players[1].position = {0.5, 1.0, 2.5};
    WorldState world;
    world.tick = 25;
    ProjectileWorldState projectiles;
    const BlockWorldView bounds{nullptr, 8, 8, 8, -4, 0, -4};
    app::DuelRewardSettings settings;
    settings.max_ticks = 100;
    settings.weights[0] = 20.0F;
    settings.weights[1] = -10.0F;
    settings.weights[2] = 2.0F;
    settings.weights[3] = -3.0F;
    settings.weights[4] = -0.5F;
    settings.weights[5] = -0.25F;
    settings.weights[8] = 4.0F;
    players[0].last_attack.health_lost = 5.0F;
    players[1].health = 0.0F;
    app::DuelRewardState state;
    const auto win = app::duelRewards(players.data(), world, bounds, projectiles, settings, state);
    require(win.terminated && !win.truncated && !win.lost[0] && win.lost[1], "duel loss ends the episode");
    near(win.rewards[0], 10.0, 0.0, "damage, win and speed rewards combine");
    near(win.rewards[1], -5.5, 0.0, "damage taken and loss rewards combine");
    near(win.contributions[0][0], 5.0, 0.0, "damage contribution uses health removed");
    near(win.contributions[0][8], 3.0, 0.0, "speed contribution uses elapsed ticks");
    require(state.dealt_damage[0] && !state.dealt_damage[1], "damage history updates for the attacker only");
    players[1].health = 20.0F;
    players[0].last_attack = {};
    world.tick = 100;
    const auto timeout = app::duelRewards(players.data(), world, bounds, projectiles, settings, state);
    require(!timeout.terminated && timeout.truncated, "timeout truncates a surviving duel");
    near(timeout.rewards[0], -0.5, 0.0, "timeout preserves prior damage history");
    near(timeout.rewards[1], -0.75, 0.0, "timeout adds the no-damage penalty");
    players[0].position.y = -4.01;
    players[1].position.x = 4.0;
    const auto draw = app::duelRewards(players.data(), world, bounds, projectiles, settings, state);
    require(draw.terminated && !draw.truncated && draw.lost[0] && draw.lost[1],
            "simultaneous exits terminate as a double loss, not a timeout");
    near(draw.rewards[0], -3.0, 0.0, "first actor receives loss reward");
    near(draw.rewards[1], -3.0, 0.0, "second actor receives loss reward");
}

void checkComparisonStatistics() {
    minecraft_sim::ErrorStatistics statistics;
    const auto first = statistics.observe(12.0, 10.0, 10, 0.1, 1.0E-9);
    require(first.difference == 2.0 && first.absolute == 2.0
        && first.percentage == 20.0, "comparison uses absolute percentage error");
    require(!statistics.observe(0.0, 0.0, 11, 0.1, 1.0E-9).percentage,
        "zero references have no percentage, not zero percent error");
    statistics.observe(9.0, -10.0, 12, 0.1, 1.0E-9);
    require(statistics.count == 3 && statistics.percentage_count == 2,
        "percentage denominator counts only eligible references");
    near(statistics.meanAbsolute(), 7.0, 0.0, "mean absolute error");
    require(statistics.meanPercentage() == 105.0
        && statistics.absolute_min == 0.0 && statistics.absolute_max == 19.0
        && statistics.percentage_min == 20.0 && statistics.percentage_max == 190.0,
        "per-field mean/min/max do not cancel signed differences");
    require(statistics.maximum_tick == 12 && statistics.first_violation == 10,
        "comparison identifies maximum and first divergent ticks");
    minecraft_sim::ErrorStatistics near_zero;
    near_zero.observe(1.0, 1.0E-10, 1, 0.0, 1.0E-9);
    require(!near_zero.meanPercentage(), "near-zero references are excluded explicitly");
}

struct Fixture {
    std::array<minecraft_sim::PlayerState, 1> players{};
    std::array<minecraft_sim::Controls, 1> queued{};
    std::array<minecraft_sim::Controls, 1> held{};
    std::array<minecraft_sim::WorldState, 1> worlds{};
    std::array<minecraft_sim::Block, 16 * 8 * 16> cells{};
    minecraft_sim::BlockWorldView blocks{cells.data(), 16, 8, 16, -8, 0, -8};
    minecraft_sim::BatchView batch{players.data(), queued.data(), held.data(), worlds.data(),
                                 1, 1, &blocks, {testTable().data()}};

    Fixture() {
        for (int block_z = -8; block_z < 8; ++block_z) {
            for (int block_x = -8; block_x < 8; ++block_x) {
                setBlock(block_x, 0, block_z);
            }
        }
        players[0].position = {0.5, 1.0, 0.5};
        players[0].velocity.y = -0.08 * double(0.98F);
        players[0].on_ground = true;
    }

    void setBlock(int block_x, int block_y, int block_z) {
        cells[((block_z + 8) * 8 + block_y) * 16 + block_x + 8] = minecraft_sim::Block::solid;
    }

    minecraft_sim::WorldView world() { return batch.world(0); }
};

void checkMeleeTargeting() {
    Fixture terrain;
    std::array<minecraft_sim::PlayerState, 3> players{
        terrain.players[0], terrain.players[0], terrain.players[0]};
    std::array<minecraft_sim::Controls, 3> queued{};
    std::array<minecraft_sim::Controls, 3> held{};
    minecraft_sim::WorldState state;
    minecraft_sim::WorldView world{players.data(), queued.data(), held.data(), &state,
                                  3, terrain.blocks, {testTable().data()}};
    players[1].position.z = 2.5;
    players[2].position.z = 3.5;
    require(world.pickAttackTarget(0) == 1, "targeting selects nearest intersected player, not any in-range opponent");
    players[0].yaw = 180.0F;
    require(world.pickAttackTarget(0) == -1, "targeting requires aim, not feet-distance alone");
    players[0].yaw = 0.0F;
    const auto clear_path = terrain.cells;
    terrain.setBlock(0, 2, 1);
    require(world.pickAttackTarget(0) == -1, "outline full-cube block hit occludes players");
    terrain.cells = clear_path;
    players[1].position.z = 3.8001;
    players[2].position.z = 5.0;
    require(world.pickAttackTarget(0) == -1, "ordinary targeting does not exploit server packet reach buffer");
    players[0].velocity.z = 0.25;
    require(world.pickAttackTarget(0) == 1, "positive look-projected movement extends client targeting ray");
    players[0].velocity.z = -0.25;
    require(world.pickAttackTarget(0) == -1, "negative projected movement never shortens or extends reach");
    players[1].position = players[0].position;
    require(world.pickAttackTarget(0) == 1, "targeting includes a target whose box contains the ray origin");
    players[1].health = 0.0F;
    require(world.pickAttackTarget(0) == -1, "dead players are not ordinary attack targets");
    near(players[0].eyePosition().y, 1.0 + double(1.62F), 0.0, "standing eye height preserves float constant");
    players[0].pose = minecraft_sim::PlayerPose::crouching;
    near(players[0].eyePosition().y, 1.0 + double(1.27F), 0.0, "crouching changes targeting origin");
}

void checkWindExplosion() {
    using namespace minecraft_sim;
    PlayerState shooter;
    shooter.position = {2.0, 3.0, 4.0};
    shooter.velocity = {0.25, 0.5, -0.75};
    shooter.on_ground = false;
    LegacyRandom wind_random, arrow_random;
    wind_random.setSeed(42);
    arrow_random.setSeed(42);
    const auto charge = launchWindCharge(shooter, 1, {testTable().data()}, wind_random);
    const auto arrow = launchArrow(shooter, 1, 0.5F, {testTable().data()}, arrow_random);
    require(charge.active && charge.owner == 1 && charge.position.x == 2.0 && charge.position.z == 4.0
        && charge.position.y == shooter.eyePosition().y && charge.no_deflect_ticks == 5,
        "wind launch uses the eye position and five no-deflection ticks");
    require(charge.velocity.x == arrow.velocity.x && charge.velocity.y == arrow.velocity.y
        && charge.velocity.z == arrow.velocity.z,
        "equal-speed launches use the same seeded spread and inherited movement");
    near(charge.velocity.x, 0.25 + 1.5 * 0.0172275 * (0.7275636800328681 - 0.6832234717598454),
        1.0E-14, "wind launch seeded horizontal spread");
    near(charge.velocity.y, 0.5 + 1.5 * 0.0172275 * (0.30871945533265976 - 0.27707849007413665),
        1.0E-14, "wind launch inherits airborne vertical velocity");
    near(charge.velocity.z, -0.75 + 1.5 * (1.0 + 0.0172275 * (0.6655489517945736 - 0.9033722646721782)),
        1.0E-14, "wind launch speed and forward spread");
    shooter.on_ground = true;
    wind_random.setSeed(42);
    const auto grounded_charge = launchWindCharge(shooter, 1, {testTable().data()}, wind_random);
    near(grounded_charge.velocity.y, charge.velocity.y - 0.5, 1.0E-14,
        "grounded wind launch does not inherit vertical velocity");

    std::array<PlayerState, 2> players{};
    std::array<Controls, 2> queued{}, held{};
    WorldState world_state;
    WorldView world(players.data(), queued.data(), held.data(), &world_state, 2, {}, {testTable().data()});
    players[0].position = {0.0, 0.0, 1.0};
    players[1].position = {0.0, 0.0, 3.0};
    const double eye_height = double(1.62F);
    const double strength = (1.0 - 1.0 / double(1.2F * 2.0F)) * double(1.22F);
    const double length = std::sqrt(1.0 + eye_height * eye_height);
    applyWindExplosion(world, {}, {});
    near(players[0].velocity.y, strength * eye_height / length, 1.0E-12, "wind blast vertical impulse");
    near(players[0].velocity.z, strength / length, 1.0E-12, "wind blast horizontal impulse");
    require(players[0].health == 20.0F && players[1].velocity.lengthSquared() == 0.0,
        "wind blast does not damage players and respects its radius");

    std::array<Block, 125> cells{};
    for (int x = -2; x <= 2; ++x) {
        for (int y = -1; y <= 3; ++y) {
            cells[3 * 25 + (y + 1) * 5 + x + 2] = Block::solid;
        }
    }
    BlockWorldView blocks{cells.data(), 5, 5, 5, -2, -1, -2};
    players[0].position = {0.0, 0.0, 2.0};
    players[0].velocity = {};
    applyWindExplosion(world, blocks, {0.0, 1.0, 0.0});
    require(players[0].velocity.lengthSquared() == 0.0, "solid wall fully occludes a wind blast");
}

void checkMaceDamage() {
    Fixture falling;
    falling.players[0].position.y = 5.0;
    falling.players[0].on_ground = false;
    falling.players[0].velocity.y = -0.1;
    falling.world().processTick();
    require(falling.players[0].fall_distance == -double(float(-0.1)),
        "fall distance accumulates float-rounded downward displacement");

    Fixture landing;
    landing.worlds[0].simulate_fall_damage = true;
    landing.players[0].position.y = 1.1;
    landing.players[0].velocity.y = -0.2;
    landing.players[0].fall_distance = 4.0;
    landing.players[0].armor = 20.0F;
    landing.players[0].on_ground = false;
    landing.world().processTick();
    require(landing.players[0].health == 19.0F && landing.players[0].fall_distance == 0.0,
        "ordinary fall damage bypasses armor and resets fall distance");

    Fixture protected_landing;
    protected_landing.worlds[0].simulate_fall_damage = true;
    protected_landing.players[0].position.y = 1.1;
    protected_landing.players[0].velocity.y = -0.2;
    protected_landing.players[0].fall_distance = 4.0;
    protected_landing.players[0].has_impulse_impact = true;
    protected_landing.players[0].impulse_impact_y = 2.0;
    protected_landing.players[0].impulse_grace_ticks = 1;
    protected_landing.players[0].on_ground = false;
    protected_landing.world().processTick();
    require(protected_landing.players[0].health == 20.0F && protected_landing.players[0].has_impulse_impact
        && protected_landing.players[0].impulse_grace_ticks == 0,
        "impulse protection limits fall damage to distance below the impact height");

    const std::array<double, 6> distances{1.5, 2.0, 3.0, 5.0, 8.0, 9.0};
    const std::array<float, 6> expected{6.0F, 14.0F, 18.0F, 22.0F, 28.0F, 29.0F};
    const minecraft_sim::MathView math{testTable().data()};
    for (std::size_t index = 0; index < distances.size(); ++index) {
        Fixture attacker;
        Fixture victim;
        auto source = attacker.world().player(0);
        auto target = victim.world().player(0);
        source.equipWeapon(minecraft_sim::WeaponItem::mace);
        source.state().fall_distance = distances[index];
        source.state().on_ground = false;
        source.state().sprinting = true;
        source.state().attack_strength_ticks = 40;
        source.resolveMelee(target, math);
        require(source.state().last_attack.incoming_damage == expected[index],
            "charged noncritical mace damage follows the fall-distance thresholds");
    }
    Fixture attacker;
    Fixture victim;
    auto source = attacker.world().player(0);
    auto target = victim.world().player(0);
    source.equipWeapon(minecraft_sim::WeaponItem::mace);
    source.state().fall_distance = 2.0;
    source.state().on_ground = false;
    source.state().attack_strength_ticks = 40;
    source.resolveMelee(target, math);
    require(source.state().last_attack.critical && source.state().last_attack.incoming_damage == 21.0F,
        "critical multiplier applies after adding the mace fall bonus");
    require(source.state().last_attack.mace_smash && !target.state().alive() && source.state().velocity.y == double(0.01F)
        && source.state().fall_distance == 0.0 && source.state().has_impulse_impact
        && source.state().impulse_impact_y == source.state().position.y && source.state().impulse_grace_ticks == 40,
        "accepted lethal smash bounces the attacker and resets fall distance");
    target.state() = {};
    source.state().position.y = 5.0;
    source.state().fall_distance = 2.0;
    source.state().attack_strength_ticks = 40;
    source.state().impulse_grace_ticks = 60;
    source.resolveMelee(target, math);
    require(source.state().impulse_impact_y == 1.0 && source.state().impulse_grace_ticks == 60,
        "a higher repeated smash retains the lower impact height and longer grace period");
    target.state() = {};
    source.state().position.y = -2.0;
    source.state().fall_distance = 2.0;
    source.state().attack_strength_ticks = 40;
    source.resolveMelee(target, math);
    require(source.state().impulse_impact_y == -2.0 && source.state().impulse_grace_ticks == 60,
        "a lower repeated smash replaces the impact height without shortening grace");
    source.state().position.y = 1.0;
    target.state() = {};
    target.state().invulnerable = true;
    source.state().fall_distance = 2.0;
    source.state().velocity.y = -0.5;
    source.state().attack_strength_ticks = 40;
    source.resolveMelee(target, math);
    require(!source.state().last_attack.accepted && !source.state().last_attack.mace_smash && source.state().velocity.y == -0.5
        && source.state().fall_distance == 2.0,
        "rejected smash does not bounce or reset fall distance");

    target.state() = {};
    target.state().position = {0.0, 0.0, -1.0};
    target.state().shield_equipped = true;
    target.state().using_shield = true;
    target.state().shield_use_ticks = 5;
    source.state().fall_distance = 2.0;
    source.state().velocity.y = -0.5;
    source.state().attack_strength_ticks = 40;
    source.resolveMelee(target, math);
    require(!source.state().last_attack.accepted && !source.state().last_attack.mace_smash && source.state().last_attack.blocked_damage > 0.0F
        && source.state().velocity.y == -0.5 && source.state().fall_distance == 2.0,
        "shield-blocked smash does not bounce or reset fall distance");

    target.state() = {};
    target.state().absorption = 30.0F;
    source.state().fall_distance = 2.0;
    source.state().attack_strength_ticks = 40;
    source.resolveMelee(target, math);
    require(source.state().last_attack.accepted && target.state().health == 20.0F
        && target.state().absorption == 9.0F && source.state().fall_distance == 0.0
        && source.state().velocity.y == double(0.01F),
        "absorption-only smash still runs accepted-hit effects");

    target.state() = {};
    source.state().fall_distance = 2.0;
    source.state().sprinting = true;
    source.state().attack_strength_ticks = 0;
    source.resolveMelee(target, math);
    const float charge = 0.5F / static_cast<float>(20.0 / (4.0 + double(-3.4F)));
    const float expected_partial = 6.0F * (0.2F + charge * charge * 0.8F) + 8.0F;
    require(source.state().last_attack.incoming_damage == expected_partial,
        "partial charge scales base damage before adding the mace bonus");

    for (double fall : {5.0, 5.01}) {
        std::array<minecraft_sim::PlayerState, 5> players{};
        std::array<minecraft_sim::Controls, 5> queued{}, held{};
        minecraft_sim::WorldState world_state;
        minecraft_sim::WorldView world(players.data(), queued.data(), held.data(), &world_state,
            static_cast<int>(players.size()), {}, math);
        auto attacker_view = world.player(0);
        attacker_view.equipWeapon(minecraft_sim::WeaponItem::mace);
        players[0].fall_distance = fall;
        players[0].attack_strength_ticks = 40;
        players[1].position = {0.0, 0.0, 1.0};
        players[2].position = {1.0, 0.0, 1.0};
        players[3].position = {3.5, 0.0, 1.0};
        players[4].position = {-1.0, 0.0, 1.0};
        players[4].knockback_resistance = 1.0;
        attacker_view.resolveMelee(world.player(1), math, 1, players.data(), static_cast<int>(players.size()));
        near(players[2].velocity.x, 2.5 * double(0.7F) * (fall > 5.0 ? 2.0 : 1.0), 1.0E-12,
            "smash splash uses victim distance and strict heavy-fall threshold");
        require(players[2].velocity.y == double(0.7F) && players[3].velocity.lengthSquared() == 0.0
            && players[4].velocity.lengthSquared() == 0.0,
            "smash splash respects radius and knockback resistance");
    }
}

void checkMeleeDamage() {
    Fixture attacker;
    Fixture victim;
    const minecraft_sim::MathView math{testTable().data()};
    auto source = attacker.world().player(0);
    auto target = victim.world().player(0);
    auto& attack_state = source.state();
    auto& hurt_state = target.state();
    source.equipWeapon(minecraft_sim::WeaponItem::mace);
    require(attack_state.attack_damage == 6.0
        && attack_state.attack_speed == 4.0 + double(-3.4F)
        && attack_state.attack_strength_ticks == 0
        && !minecraft_sim::weaponAttributes(minecraft_sim::WeaponItem::mace).sweep,
        "equipping a mace applies vanilla base attributes without sweeping");
    attack_state.attack_strength_ticks = 32;
    near(attack_state.attackCharge(), 0.975, 1.0E-6, "mace is not fully charged at 32 ticks");
    attack_state.attack_strength_ticks = 33;
    require(attack_state.attackCharge() == 1.0F, "mace reaches full charge at 33 ticks");
    require(minecraft_sim::MACE_DAMAGE_MODIFIER == 5.0
        && minecraft_sim::MACE_SPEED_MODIFIER == double(-3.4F)
        && minecraft_sim::MACE_KNOCKBACK_POWER == double(0.7F)
        && minecraft_sim::WIND_CHARGE_EXPLOSION_POWER == 1.2F
        && minecraft_sim::WIND_CHARGE_COOLDOWN_TICKS == 10,
        "mace and wind charge constants preserve vanilla modifiers and cooldown");
    source.equipWeapon(minecraft_sim::WeaponItem::diamond_axe);
    require(attack_state.attack_damage == 9.0 && attack_state.attack_speed == 1.0
        && attack_state.attack_strength_ticks == 0, "equipping an axe updates damage, speed and attack charge");
    attack_state.attack_strength_ticks = 20;
    source.equipWeapon(minecraft_sim::WeaponItem::diamond_axe);
    require(attack_state.attack_strength_ticks == 20, "keeping the same weapon does not reset charge");
    hurt_state.position.z = 2.5;
    source.resolveMelee(target, math, 1);
    require(hurt_state.health == 11.0F && !attack_state.last_attack.sweep,
        "a charged diamond axe deals nine damage without sword sweeping");
    source.equipWeapon(minecraft_sim::WeaponItem::diamond_sword);
    hurt_state = {};
    hurt_state.position = {0.5, 1.0, 2.5};
    hurt_state.on_ground = true;
    for (int slot = 0; slot < 4; ++slot) {
        target.equipArmor(static_cast<minecraft_sim::ArmorSlot>(slot), minecraft_sim::ArmorMaterial::netherite);
    }
    require(hurt_state.armor == 20.0F && hurt_state.armor_toughness == 12.0F,
        "netherite equipment aggregates defense and toughness across all four slots");
    near(hurt_state.armor_knockback_resistance, 4.0 * double(0.1F), 0.0,
        "netherite equipment preserves per-piece float knockback modifiers");
    target.equipArmor(minecraft_sim::ArmorSlot::helmet, minecraft_sim::ArmorMaterial::leather);
    require(hurt_state.armor == 18.0F && hurt_state.armor_toughness == 9.0F,
        "replacing armor recomputes mixed equipment rather than accumulating stale attributes");
    hurt_state = {};
    hurt_state.position = {0.5, 1.0, 2.5};
    hurt_state.on_ground = true;
    hurt_state.position.z = 2.5;
    attack_state.attack_strength_ticks = 20;
    source.resolveMelee(target, math, 1);
    require(attack_state.last_attack.accepted && attack_state.last_attack.target == 1
        && attack_state.last_attack.incoming_damage == 7.0F && hurt_state.health == 13.0F,
        "charged unenchanted diamond sword deals seven unarmored health damage");
    require(hurt_state.damage_cooldown == 20 && hurt_state.hurt_time == 10
        && hurt_state.last_hurt == 7.0F && attack_state.attack_strength_ticks == 0,
        "full hit seeds damage/hurt timers and resets attacker charge");
    near(hurt_state.velocity.z, double(0.4F), 0.0, "full melee hit applies source-position default knockback");
    const auto first_velocity = hurt_state.velocity;
    attack_state.attack_strength_ticks = 20;
    source.resolveMelee(target, math);
    require(!attack_state.last_attack.accepted && hurt_state.health == 13.0F
        && hurt_state.velocity.z == first_velocity.z && attack_state.attack_strength_ticks == 0,
        "equal repeat is rejected, including knockback, but still consumes attack charge");
    hurt_state.needs_sync = false;
    attack_state.attack_strength_ticks = 20;
    attack_state.on_ground = false;
    attack_state.fall_distance = 1.0;
    source.resolveMelee(target, math);
    require(attack_state.last_attack.critical && attack_state.last_attack.accepted
        && hurt_state.health == 9.5F && hurt_state.last_hurt == 10.5F && !hurt_state.needs_sync,
        "stronger critical repeat deals only excess and does not repeat full-hit knockback");
    near(hurt_state.velocity.z, first_velocity.z, 0.0, "stronger cooldown hit preserves momentum without extra knockback");

    hurt_state = {};
    hurt_state.position = {0.5, 1.0, 2.5};
    hurt_state.on_ground = true;
    hurt_state.absorption = 5.0F;
    require(target.hurt(3.0F, attack_state.position) && hurt_state.health == 20.0F
        && hurt_state.absorption == 2.0F && hurt_state.needs_sync,
        "absorption can prevent health loss without rejecting the hit or its knockback");
    hurt_state.damage_cooldown = 10;
    require(target.hurt(7.0F, attack_state.position) && hurt_state.health == 15.0F
        && hurt_state.damage_cooldown == 20,
        "cooldown exactly ten accepts a new full hit, after remaining absorption");
    hurt_state.invulnerable = true;
    require(!target.hurt(8.0F, attack_state.position) && hurt_state.health == 15.0F,
        "invulnerability rejects damage before cooldown state changes");

    hurt_state = {};
    hurt_state.position = {0.5, 1.0, 2.5};
    hurt_state.armor = 20.0F;
    hurt_state.armor_toughness = 8.0F;
    hurt_state.absorption = 1.0F;
    require(target.hurt(7.0F, attack_state.position), "armor reduces rather than rejects a melee hit");
    near(hurt_state.health, 19.11F, 1.0E-6, "diamond armor reduction precedes absorption");
    require(hurt_state.last_hurt == 7.0F && hurt_state.absorption == 0.0F,
        "damage immunity remembers pre-armor damage");
    require(target.hurt(10.5F, attack_state.position), "stronger armored cooldown hit applies its excess");
    near(hurt_state.health, 18.2875F, 2.0E-6, "armor reduction applies to the cooldown differential");

    hurt_state = {};
    hurt_state.position = {0.5, 1.0, 2.5};
    hurt_state.on_ground = true;
    attack_state.on_ground = true;
    attack_state.fall_distance = 0.0;
    attack_state.sprinting = true;
    attack_state.velocity = {1.0, 0.0, 0.5};
    attack_state.attack_strength_ticks = 20;
    source.resolveMelee(target, math);
    require(attack_state.last_attack.sprint_bonus && !attack_state.last_attack.critical
        && !attack_state.sprinting, "charged sprint hit adds knockback and clears sprint");
    near(hurt_state.velocity.z, double(0.4F) / 2.0 + 0.5, 0.0,
        "sprint extra knockback follows and halves the default impulse");
    near(attack_state.velocity.x, 0.6, 0.0, "successful extra knockback slows attacker X");
    near(attack_state.velocity.z, 0.3, 0.0, "successful extra knockback slows attacker Z");

    hurt_state = {};
    hurt_state.position = {0.5, 1.0, 2.5};
    attack_state.on_ground = false;
    attack_state.fall_distance = 1.0;
    attack_state.attack_strength_ticks = 0;
    source.resolveMelee(target, math);
    require(!attack_state.last_attack.critical, "undercharged airborne hit is not critical");
    near(attack_state.last_attack.incoming_damage, 7.0F * (0.2F + 0.04F * 0.04F * 0.8F), 1.0E-6,
        "weak attacks retain the source quadratic charge scaling");
    hurt_state.damage_cooldown = 0;
    require(target.hurt(100.0F, attack_state.position) && !hurt_state.alive() && hurt_state.health == 0.0F,
        "lethal damage clamps health to zero and marks dead state");
    require(!target.hurt(1.0F, attack_state.position), "dead players reject further damage");

    hurt_state = {};
    hurt_state.position = {0.5, 1.0, 2.5};
    hurt_state.on_ground = true;
    hurt_state.knockback_resistance = 1.0;
    require(target.hurt(7.0F, attack_state.position) && hurt_state.needs_sync,
            "full hit marks impact even when resistance eliminates knockback");
    near(hurt_state.velocity.lengthSquared(), 0.0, 0.0, "full knockback resistance does not suppress accepted damage");
    require(!target.hurt(0.0F, attack_state.position), "player zero-damage gate does not become a successful hurt");
    hurt_state.invulnerable = true;
    attack_state.on_ground = true;
    attack_state.sprinting = true;
    attack_state.attack_strength_ticks = 20;
    attack_state.velocity.x = 1.0;
    source.resolveMelee(target, math);
    require(!attack_state.last_attack.accepted && attack_state.sprinting && attack_state.velocity.x == 1.0,
            "rejected sprint hit does not clear sprint or apply attacker slowdown");

    hurt_state = {};
    hurt_state.position = {0.5, 1.0, 2.5};
    hurt_state.yaw = 180.0F;
    target.equipShield(true);
    hurt_state.using_shield = true;
    hurt_state.shield_use_ticks = 4;
    require(!hurt_state.blocking(), "a shield is not active before five elapsed use ticks");
    hurt_state.shield_use_ticks = 5;
    require(hurt_state.blocking() && target.blocksMeleeFrom(attack_state.position),
        "a raised shield covers a source in front regardless of vertical offset");
    require(!target.blocksMeleeFrom({0.5, 1.0, 4.5}), "shield does not protect the defender's back");
    require(target.blocksMeleeFrom({2.5, 50.0, 2.5}), "shield includes the ninety-degree horizontal boundary");
    hurt_state.yaw = 0.0F;
    require(target.hurt(7.0F, attack_state.position) && hurt_state.health == 13.0F,
        "a raised shield does not block attacks from behind");
    hurt_state.yaw = 180.0F;
    hurt_state.damage_cooldown = 0;
    hurt_state.needs_sync = false;
    hurt_state.velocity = {};
    require(!target.hurt(7.0F, attack_state.position) && hurt_state.health == 13.0F
        && hurt_state.last_blocked_damage == 7.0F && hurt_state.damage_cooldown == 20
        && hurt_state.hurt_time == 10 && hurt_state.last_hurt == 0.0F && !hurt_state.needs_sync,
        "fully blocked hit seeds hurt timers but deals no damage or default knockback");
    hurt_state.damage_cooldown = 0;
    require(target.hurt(1.0F, attack_state.position, 0.0F, true) && hurt_state.health == 12.0F,
        "explicit shield-bypassing damage skips blocking");
    source.equipWeapon(minecraft_sim::WeaponItem::diamond_axe);
    attack_state.attack_strength_ticks = 0;
    attack_state.sprinting = true;
    attack_state.velocity = {1.0, 0.0, 0.5};
    hurt_state.damage_cooldown = 0;
    source.resolveMelee(target, math, 1);
    require(!attack_state.last_attack.accepted && attack_state.last_attack.blocked_damage > 0.0F
        && attack_state.last_attack.shield_disabled && hurt_state.shield_cooldown == 100
        && !hurt_state.using_shield && !hurt_state.blocking() && hurt_state.health == 12.0F,
        "even an undercharged axe hit is blocked and disables the shield for one hundred ticks");
    require(attack_state.sprinting && attack_state.velocity.x == 1.0,
        "a fully blocked axe hit does not trigger successful-hit sprint slowdown");
    target.equipShield(false);
    require(!hurt_state.shield_equipped && hurt_state.shield_use_ticks == 0,
        "unequipping a shield clears active use without clearing its cooldown");
}

void checkFoodRegeneration() {
    Fixture survival;
    survival.worlds[0].natural_regeneration = true;
    auto player_with_food = survival.world().player(0);
    auto& food_state = survival.players[0];
    food_state.health = 10.0F;
    food_state.food_saturation = 6.0F;
    for (int tick = 0; tick < 9; ++tick) player_with_food.tickFood();
    require(food_state.health == 10.0F, "saturation regeneration waits ten food ticks");
    player_with_food.tickFood();
    require(food_state.health == 11.0F && food_state.food_exhaustion == 6.0F && food_state.food_tick_timer == 0,
        "saturation regeneration heals saturation divided by six and spends exhaustion");
    player_with_food.tickFood();
    require(food_state.food_saturation == 5.0F && food_state.food_exhaustion == 2.0F,
        "regeneration exhaustion drains saturation on the following food tick");
    food_state.food_level = 18;
    food_state.food_saturation = 0.0F;
    food_state.food_exhaustion = 0.0F;
    food_state.food_tick_timer = 0;
    for (int tick = 0; tick < 79; ++tick) player_with_food.tickFood();
    require(food_state.health == 11.0F, "unsaturated regeneration waits eighty food ticks");
    player_with_food.tickFood();
    require(food_state.health == 12.0F && food_state.food_exhaustion == 6.0F,
        "food level eighteen enables one-health slow regeneration");
    food_state.health = 19.8F;
    player_with_food.heal(1.0F);
    require(food_state.health == 20.0F, "healing clamps to maximum health without adding absorption");
    food_state.health = 0.0F;
    player_with_food.heal(1.0F);
    require(food_state.health == 0.0F, "natural healing cannot resurrect a dead actor");

    food_state = {};
    food_state.position = {0.5, 1.0, 0.5};
    food_state.health = 10.0F;
    food_state.food_level = 17;
    food_state.food_saturation = 0.0F;
    for (int tick = 0; tick < 100; ++tick) player_with_food.tickFood();
    require(food_state.health == 10.0F && food_state.food_tick_timer == 0,
        "food below eighteen does not regenerate even when natural regeneration is enabled");
    survival.worlds[0].natural_regeneration = false;
    survival.worlds[0].simulate_food = true;
    food_state.food_exhaustion = 4.0F;
    player_with_food.tickFood();
    require(food_state.food_level == 17, "exhaustion at exactly four does not consume food");
    food_state.food_exhaustion = 4.1F;
    player_with_food.tickFood();
    require(food_state.food_level == 16, "exhaustion strictly above four drains food when saturation is empty");
    food_state.health = 2.0F;
    food_state.food_level = 0;
    food_state.armor = 20.0F;
    food_state.armor_toughness = 12.0F;
    food_state.food_tick_timer = 79;
    player_with_food.tickFood();
    require(food_state.health == 1.0F && food_state.velocity.lengthSquared() == 0.0,
        "normal starvation bypasses armor and knockback but stops at one health");
    food_state.damage_cooldown = 0;
    food_state.food_tick_timer = 79;
    player_with_food.tickFood();
    require(food_state.health == 1.0F, "normal difficulty starvation does not kill");
    survival.worlds[0].difficulty = minecraft_sim::Difficulty::hard;
    food_state.food_tick_timer = 79;
    player_with_food.tickFood();
    require(!food_state.alive(), "hard difficulty starvation can kill");

    food_state = {};
    food_state.position = {0.5, 1.0, 0.5};
    food_state.health = 10.0F;
    food_state.absorption = 4.0F;
    food_state.food_level = 10;
    food_state.food_saturation = 0.0F;
    survival.worlds[0].natural_regeneration = true;
    survival.worlds[0].difficulty = minecraft_sim::Difficulty::peaceful;
    survival.world().processTick();
    for (int tick = 1; tick < 20; ++tick) survival.world().processTick();
    require(food_state.health == 11.0F && food_state.food_level == 12 && food_state.food_saturation == 1.0F
        && food_state.absorption == 4.0F, "peaceful restores health, food and saturation on separate vanilla cadences");

    Fixture disabled;
    disabled.players[0].health = 10.0F;
    for (int tick = 0; tick < 100; ++tick) disabled.world().processTick();
    require(disabled.players[0].health == 10.0F && disabled.players[0].food_saturation == 5.0F,
        "legacy duel worlds keep food and regeneration frozen by default");
    Fixture jumping;
    jumping.worlds[0].simulate_food = true;
    minecraft_sim::Controls jump;
    jump.jump = true;
    jumping.world().player(0).setInputs(jump);
    jumping.world().processTick();
    require(jumping.players[0].food_exhaustion == 0.05F, "ordinary jumping spends vanilla exhaustion");
}

void checkEquipmentRules() {
    using minecraft_sim::WeaponItem;
    using minecraft_sim::ArmorMaterial;
    using minecraft_sim::ArmorSlot;
    const double damage[] = {1, 4, 5, 5, 4, 6, 7, 8, 7, 9, 9, 7, 9, 9, 10};
    const double speed[] = {4, 4 + double(-2.4F), 4 + double(-2.4F), 4 + double(-2.4F),
        4 + double(-2.4F), 4 + double(-2.4F), 4 + double(-2.4F), 4 + double(-2.4F),
        4 + double(-3.2F), 4 + double(-3.2F), 4 + double(-3.2F), 1, 4 + double(-3.1F), 1, 1};
    Fixture fixture;
    auto player = fixture.world().player(0);
    for (int index = 0; index < 15; ++index) {
        const auto item = static_cast<WeaponItem>(index);
        const auto attributes = minecraft_sim::weaponAttributes(item);
        near(attributes.damage, damage[index], 0.0, "base weapon damage matches Items and ToolMaterial");
        near(attributes.speed, speed[index], 0.0, "weapon speed preserves widened float attribute modifiers");
        require(attributes.sweep == (index >= 1 && index <= 7)
            && attributes.disable_shield_seconds == (index >= 8 ? 5.0F : 0.0F),
            "only swords sweep and only axes disable shields in this equipment subset");
        player.equipWeapon(item);
        fixture.players[0].attack_strength_ticks = 100;
        require(fixture.players[0].attackCharge() == 1.0F, "all supported weapons can fully recharge");
    }
    const float armor_totals[] = {0, 7, 10, 12, 15, 11, 20, 20};
    for (int material = 0; material < 8; ++material) {
        for (int slot = 0; slot < 4; ++slot) {
            player.equipArmor(static_cast<ArmorSlot>(slot), static_cast<ArmorMaterial>(material));
        }
        require(fixture.players[0].armor == armor_totals[material], "full armor set has source-derived defense");
    }
    player.equipArmor(ArmorSlot::helmet, ArmorMaterial::turtle);
    require(fixture.players[0].armor == 19.0F && fixture.players[0].armor_toughness == 9.0F,
        "turtle helmet contributes defense without inherited netherite toughness");
    fixture.players[0].velocity = {};
    player.knockback(1.0, 1.0, 0.0);
    near(fixture.players[0].velocity.x, -(1.0 - 3 * double(0.1F)), 0.0,
        "remaining netherite pieces reduce the physical knockback impulse");
}

void checkShieldWorld() {
    Fixture terrain;
    std::array<minecraft_sim::PlayerState, 2> players{};
    std::array<minecraft_sim::Controls, 2> queued{};
    std::array<minecraft_sim::Controls, 2> held{};
    minecraft_sim::WorldState state;
    minecraft_sim::WorldView world{players.data(), queued.data(), held.data(), &state,
        2, terrain.blocks, {testTable().data()}};
    const auto reset = [&] {
        players = {terrain.players[0], terrain.players[0]};
        players[0].attack_strength_ticks = 20;
        players[1].position.z = 2.5;
        players[1].yaw = 180.0F;
        queued = {};
        held = {};
        state = {};
        world.player(1).equipShield(true);
    };
    minecraft_sim::Controls use;
    use.use_item = true;
    minecraft_sim::Controls attack;
    attack.attack = true;
    reset();
    world.player(1).setInputs(use);
    for (int tick = 0; tick < 4; ++tick) world.processTick();
    world.player(0).setInputs(attack);
    world.processTick();
    require(players[1].health == 13.0F && players[1].shield_use_ticks == 5,
        "attack before the fifth completed use tick beats the shield raise delay");

    reset();
    world.player(1).setInputs(use);
    for (int tick = 0; tick < 5; ++tick) world.processTick();
    world.player(0).setInputs(attack);
    world.processTick();
    require(players[1].health == 20.0F && players[0].last_attack.blocked_damage == 7.0F
        && !players[0].last_attack.accepted && players[1].using_shield,
        "held item use persists and blocks attacks after five completed ticks");

    reset();
    world.player(1).setInputs(use);
    for (int tick = 0; tick < 5; ++tick) world.processTick();
    world.player(0).equipWeapon(minecraft_sim::WeaponItem::wooden_axe);
    world.player(0).setInputs(attack);
    world.processTick();
    require(players[0].last_attack.shield_disabled && players[1].shield_cooldown == 99
        && !players[1].using_shield && players[1].health == 20.0F,
        "blocked axe sets one hundred cooldown ticks before the end-of-player-tick decrement");
    for (int tick = 0; tick < 99; ++tick) world.processTick();
    require(players[1].shield_cooldown == 0 && !players[1].blocking(),
        "shield cooldown expires without an instant re-block");
    world.processTick();
    require(players[1].using_shield && players[1].shield_use_ticks == 1,
        "held use retries on the four-tick item-use cadence after disable expires");
    for (int tick = 0; tick < 4; ++tick) world.processTick();
    require(players[1].blocking(), "re-raised shield must complete the raise delay again");

    reset();
    world.player(0).equipShield(true);
    auto attack_and_use = use;
    attack_and_use.attack = true;
    world.player(0).setInputs(attack_and_use);
    world.processTick();
    require(players[0].last_attack.accepted && players[1].health == 13.0F && players[0].using_shield,
        "attack press precedes starting a new shield use in the same tick");
    world.player(0).setInputs(attack);
    world.processTick();
    require(!players[0].using_shield && !players[0].last_attack.attempted,
        "releasing item use consumes that tick's attack press instead of attacking through the shield");
    world.player(0).setInputs(attack);
    world.processTick();
    require(players[0].last_attack.attempted, "attacks resume on the following tick after release");

    Fixture walking;
    Fixture shielding;
    auto forward = use;
    forward.forward = true;
    shielding.world().player(0).equipShield(true);
    shielding.world().player(0).setInputs(forward);
    forward.use_item = false;
    walking.world().player(0).setInputs(forward);
    shielding.world().processTick();
    walking.world().processTick();
    near(shielding.players[0].position.z - 0.5, (walking.players[0].position.z - 0.5) * double(0.2F),
        1.0E-8, "shield use applies the default twenty-percent movement multiplier");
    forward.use_item = true;
    forward.sprint = true;
    shielding.world().player(0).setInputs(forward);
    shielding.world().processTick();
    require(!shielding.players[0].sprinting, "shield use prevents starting sprint from a new sprint key press");
    shielding.players[0].shield_use_ticks = 71999;
    shielding.players[0].item_use_delay = 0;
    shielding.world().processTick();
    require(!shielding.players[0].using_shield && shielding.players[0].shield_use_ticks == 0,
        "shield use completes at the source's seventy-two-thousand-tick duration");
    shielding.world().processTick();
    require(shielding.players[0].using_shield && shielding.players[0].shield_use_ticks == 1,
        "held use restarts after the long-use duration completes");
}

void checkSwordSweep() {
    Fixture terrain;
    std::array<minecraft_sim::PlayerState, 3> players{
        terrain.players[0], terrain.players[0], terrain.players[0]};
    std::array<minecraft_sim::Controls, 3> queued{};
    std::array<minecraft_sim::Controls, 3> held{};
    minecraft_sim::WorldState state;
    minecraft_sim::WorldView world{players.data(), queued.data(), held.data(), &state,
                                  3, terrain.blocks, {testTable().data()}};
    players[0].attack_strength_ticks = 20;
    players[1].position.z = 2.5;
    players[2].position = {1.2, 1.0, 2.5};
    minecraft_sim::Controls attack;
    attack.attack = true;
    world.player(0).setInputs(attack);
    world.processTick();
    require(players[0].last_attack.sweep && players[0].last_attack.sweep_hits == 1
            && players[1].health == 13.0F && players[2].health == 19.0F,
            "unenchanted sword sweep excludes primary and deals one damage to nearby eligible players");
    players[0] = terrain.players[0];
    players[0].attack_strength_ticks = 20;
    players[0].position.y = 1.1;
    players[0].on_ground = false;
    players[0].fall_distance = 0.1;
    players[1] = terrain.players[0];
    players[1].position.z = 2.5;
    players[2] = terrain.players[0];
    players[2].position = {1.2, 1.0, 2.5};
    world.player(0).setInputs(attack);
    world.processTick();
    require(players[0].last_attack.critical && !players[0].last_attack.sweep && players[2].health == 20.0F,
            "critical sword hit does not also sweep adjacent players");
}

void checkMeleeWorld() {
    Fixture terrain;
    std::array<minecraft_sim::PlayerState, 2> players{};
    std::array<minecraft_sim::Controls, 2> queued{};
    std::array<minecraft_sim::Controls, 2> held{};
    minecraft_sim::WorldState state;
    minecraft_sim::WorldView world{players.data(), queued.data(), held.data(), &state,
                  2, terrain.blocks, {testTable().data()}};
    const auto reset = [&] {
    for (auto& player : players) {
        player = terrain.players[0];
        player.attack_strength_ticks = 20;
    }
    players[1].position.z = 2.5;
    players[1].yaw = 180.0F;
    queued = {};
    held = {};
    state = {};
    };
    reset();
    minecraft_sim::Controls attack;
    attack.attack = true;
    world.player(0).setInputs(attack);
    world.player(1).setInputs(attack);
    world.processTick();
    require(players[0].health == 13.0F && players[1].health == 13.0F
        && players[0].last_attack.target == 1 && players[1].last_attack.target == 0,
        "both attacks select from the same pre-movement state");
    require(players[0].damage_cooldown == 20 && players[1].damage_cooldown == 20
        && players[0].attack_strength_ticks == 1 && players[1].attack_strength_ticks == 1,
        "all combat timers prepare once before attacks and attack charge increments after movement");
    require(!queued[0].attack && held[0].attack, "attack press latches and is consumed from the queue");
    world.processTick();
    require(!players[0].last_attack.attempted && !players[1].last_attack.attempted
        && players[0].health == 13.0F && players[1].health == 13.0F,
        "attack event does not become a held automatic attack on subsequent ticks");

    reset();
    const auto clear_path = terrain.cells;
    terrain.setBlock(0, 2, 1);
    world.player(0).setInputs(attack);
    world.processTick();
    require(players[0].last_attack.hit_block && players[1].health == 20.0F
        && players[0].attack_strength_ticks == 21,
        "block-targeted click neither attacks through walls nor resets melee charge as an air miss");
    terrain.cells = clear_path;
    reset();
    players[0].yaw = 180.0F;
    world.player(0).setInputs(attack);
    world.processTick();
    require(players[0].last_attack.attempted && players[0].last_attack.target == -1
        && players[0].attack_strength_ticks == 1 && players[0].attack_miss_time == 10,
        "air miss resets attack charge and starts miss delay");
    world.player(0).setInputs(attack);
    world.processTick();
    require(!players[0].last_attack.attempted && players[0].attack_miss_time == 9,
        "consecutive attack presses respect active miss delay");
    world.processTick();
    require(players[0].attack_miss_time == 0, "a frame without an attack press releases the miss-delay state");

    reset();
    players[0].health = 7.0F;
    players[1].health = 7.0F;
    attack.forward = true;
    attack.jump = true;
    world.player(0).setInputs(attack);
    world.player(1).setInputs(attack);
    world.processTick();
    require(players[0].health == 7.0F && !players[1].alive() && !players[1].last_attack.accepted,
        "ordered lethal attacks cancel the later dead attack rather than assuming simultaneous damage");
    require(players[1].jump_delay == 0 && !players[1].sprinting,
        "a killed player cannot jump or sprint during subsequent movement phase");
    world.processTick();
    require(players[1].death_time == 1, "dead state advances a death timer without respawn or reset");

    reset();
    players[0].yaw = 180.0F;
    attack = {};
    attack.attack = true;
    attack.yaw_delta = 180.0F;
    world.player(0).setInputs(attack);
    world.processTick();
    require(players[0].yaw == 360.0F && players[0].last_attack.accepted,
        "look events apply once before targeting, not after attack or twice before movement");

    reset();
    attack = {};
    attack.attack = true;
    attack.forward = true;
    attack.sprint = true;
    world.player(0).setInputs(attack);
    world.processTick();
    require(players[0].last_attack.accepted && !players[0].last_attack.sprint_bonus
        && players[0].sprinting,
        "starting sprint applies during movement, not retroactively to this tick's attack");

    reset();
    players[0].sprinting = true;
    attack.forward = false;
    attack.sprint = false;
    world.player(0).setInputs(attack);
    world.processTick();
    require(players[0].last_attack.accepted && players[0].last_attack.sprint_bonus
        && !players[0].sprinting,
        "releasing sprint does not remove the carried sprint bonus before this tick's attack");

        reset();
        players[1].position = players[0].position;
        players[1].position.x += 0.25;
        players[1].velocity.x = -0.5 * double(0.05F) + 0.001;
        const double initial_x = players[1].position.x;
        world.processTick();
        near(players[1].position.x, initial_x, 0.0,
            "later-player velocity thresholding stays after earlier-player contact, not at joint combat barrier");
}

void checkSprinting() {
    Fixture sprint;
    minecraft_sim::Controls controls;
    controls.forward = true;
    controls.sprint = true;
    sprint.world().player(0).setInputs(controls);
    sprint.world().processTick();
    const float friction = 0.6F;
    const float acceleration = 0.13F * (0.21600002F / (friction * friction * friction));
    near(sprint.players[0].position.z, 0.5 + double(0.98F) * acceleration, 1.0E-15,
        "sprinting uses the movement attribute modifier");
    require(sprint.players[0].sprinting, "sprint key starts sprinting");
    controls.sprint = false;
    sprint.world().player(0).setInputs(controls);
    sprint.world().processTick();
    require(sprint.players[0].sprinting, "releasing sprint does not cancel an eligible running sprint");
    sprint.players[0].food_level = 6;
    sprint.world().processTick();
    require(!sprint.players[0].sprinting, "food eligibility stops sprinting");

    Fixture jump;
    controls.sprint = true;
    controls.jump = true;
    jump.world().player(0).setInputs(controls);
    jump.world().processTick();
    near(jump.players[0].position.z, 0.5 + 0.2 + double(0.98F) * acceleration, 1.0E-15,
        "sprint jump adds a separate yaw-oriented 0.2 impulse before travel");
    near(jump.players[0].position.y, 1.0 + double(0.42F), 0.0, "sprint jump retains ordinary jump height");

    Fixture blocked;
    blocked.players[0].horizontal_collision = true;
    blocked.world().player(0).setInputs(controls);
    blocked.world().processTick();
    require(!blocked.players[0].sprinting, "non-minor collision cancels running sprint");
}

void checkCrouchingAndEdges() {
    Fixture crouch;
    minecraft_sim::Controls controls;
    controls.forward = true;
    controls.sneak = true;
    crouch.world().player(0).setInputs(controls);
    crouch.world().processTick();
    require(!crouch.players[0].crouching && crouch.players[0].pose == minecraft_sim::PlayerPose::crouching,
        "new sneak key updates pose after movement, not pre-refresh crouch flag");
    const double previous_z = crouch.players[0].position.z;
    const double previous_velocity = crouch.players[0].velocity.z;
    crouch.world().processTick();
    require(crouch.players[0].crouching, "held sneak applies local crouch slowdown next tick");
    const float friction = 0.6F;
    const float acceleration = 0.1F * (0.21600002F / (friction * friction * friction));
    near(crouch.players[0].position.z - previous_z,
     previous_velocity + double(0.98F * 0.3F) * acceleration, 1.0E-15,
     "crouch speed factor applies before square-input adjustment");
    controls.sneak = false;
    crouch.world().player(0).setInputs(controls);
    crouch.world().processTick();
    require(crouch.players[0].crouching && crouch.players[0].pose == minecraft_sim::PlayerPose::standing,
        "sneak release preserves pre-refresh slowdown for the current movement tick");

    Fixture ceiling;
    ceiling.players[0].position.y = 1.4;
    ceiling.players[0].velocity.y = 0.0;
    ceiling.players[0].on_ground = false;
    ceiling.players[0].pose = minecraft_sim::PlayerPose::crouching;
    ceiling.setBlock(0, 3, 0);
    ceiling.world().processTick();
    require(ceiling.players[0].pose == minecraft_sim::PlayerPose::crouching && ceiling.players[0].crouching,
        "low ceiling prevents standing even after sneak is released");
    ceiling.cells.fill(minecraft_sim::Block::air);
    ceiling.world().processTick();
    require(ceiling.players[0].pose == minecraft_sim::PlayerPose::standing, "pose returns to standing when clearance exists");

    Fixture edge;
    edge.cells.fill(minecraft_sim::Block::air);
    edge.setBlock(0, 0, 0);
    controls.sneak = true;
    controls.left = true;
    edge.world().player(0).setInputs(controls);
    for (int tick = 0; tick < 100; ++tick) {
    edge.world().processTick();
    require(edge.players[0].on_ground && edge.players[0].position.y == 1.0,
        "sneaking retains support at a diagonal ledge");
    }
    require(edge.players[0].position.x <= 1.0 + double(0.6F) / 2.0
        && edge.players[0].position.z <= 1.0 + double(0.6F) / 2.0
        && edge.players[0].velocity.x > 0.0 && edge.players[0].velocity.z > 0.0,
        "edge backoff clips displacement without directly zeroing horizontal velocity");

    Fixture double_tap;
    controls = {};
    controls.forward = true;
    double_tap.world().player(0).setInputs(controls);
    double_tap.world().processTick();
    double_tap.world().player(0).setInputs({});
    double_tap.world().processTick();
    double_tap.world().player(0).setInputs(controls);
    double_tap.world().processTick();
    require(double_tap.players[0].sprinting, "vanilla forward double-tap window starts sprinting");
}

void checkPlayerContact() {
    Fixture terrain;
    std::array<minecraft_sim::PlayerState, 2> players{terrain.players[0], terrain.players[0]};
    std::array<minecraft_sim::Controls, 2> queued{};
    std::array<minecraft_sim::Controls, 2> held{};
    minecraft_sim::WorldState world_state;
    players[1].position.x = 0.75;
    minecraft_sim::WorldView world{players.data(), queued.data(), held.data(), &world_state,
                   2, terrain.blocks, {testTable().data()}};
    world.processTick();
    const double first_push = 0.5 * double(0.05F);
    const double second_position = 0.75 + first_push;
    const double separation = second_position - 0.5;
    const double second_push = (separation / std::sqrt(separation)) * double(0.05F);
    near(players[0].position.x, 0.5, 0.0, "contact does not retroactively move an already-ticked player");
    near(players[1].position.x, second_position, 0.0, "later player integrates the earlier player's contact impulse");
    near(players[0].velocity.x, -first_push - second_push, 1.0E-15,
     "both player ticks process the overlapping pair, without global deduplication");
    near(players[1].velocity.x, first_push * double(0.6F * 0.91F) + second_push, 1.0E-15,
     "contact follows travel drag and precedes next player's movement");
    require(players[0].needs_sync && players[1].needs_sync, "contact marks both changed velocities");

    Fixture diagonal;
    minecraft_sim::PlayerState other = diagonal.players[0];
    other.position.x += 0.25;
    other.position.z += 0.25;
    diagonal.world().player(0).push(other);
    near(other.velocity.x, first_push, 0.0, "contact uses square-root max-axis scaling, not Euclidean normalization");
    near(other.velocity.z, first_push, 0.0, "diagonal contact has independently scaled components");
    require(diagonal.players[0].position.x == 0.5 && other.position.x == 0.75,
        "contact primitive changes velocity, not position");

    Fixture cutoff;
    other = cutoff.players[0];
    other.position.x += 0.009;
    cutoff.world().player(0).push(other);
    require(!other.needs_sync && !cutoff.players[0].needs_sync, "nearly coincident contact is ignored");
    other.position.x = 0.75;
    other.no_physics = true;
    cutoff.world().player(0).push(other);
    require(other.velocity.x == 0.0 && cutoff.players[0].velocity.x == 0.0,
        "no-physics participants suppress contact");
    other.no_physics = false;
    other.pushable = false;
    cutoff.world().player(0).push(other);
    require(other.velocity.x == 0.0 && cutoff.players[0].velocity.x < 0.0,
        "contact applies each participant's pushability independently");

    players = {terrain.players[0], terrain.players[0]};
    players[1].position.x = 4.0;
    world.processTick();
    require(players[0].velocity.x == 0.0 && players[1].velocity.x == 0.0,
        "world contact queries reject nonoverlapping player bounds");
}

void checkKnockback() {
    minecraft_sim::LegacyRandom random;
    near(random.nextDouble(), 0.730967787376657, 0.0, "Java Random seed-zero first double");
    near(random.nextDouble(), 0.24053641567148587, 0.0, "Java Random seed-zero second double");
    random.setSeed(0);
    near(random.nextDouble(), 0.730967787376657, 0.0, "Java Random reseeding scrambles the external seed");

    Fixture grounded;
    auto& state = grounded.players[0];
    state.velocity = {0.2, -0.08 * double(0.98F), 0.4};
    grounded.world().player(0).knockback(0.4, 1.0, 0.0);
    near(state.velocity.x, -0.3, 1.0E-15, "knockback subtracts direction after halving horizontal momentum");
    near(state.velocity.z, 0.2, 0.0, "knockback halves the transverse horizontal velocity too");
    near(state.velocity.y, -0.04 * double(0.98F) + 0.4, 0.0, "grounded knockback adds capped lift");
    require(state.position.x == 0.5 && state.tick == 0 && state.needs_sync,
        "knockback queues velocity without moving or ticking the player");
    grounded.world().player(0).knockback(1.0, 1.0, 0.0);
    require(state.velocity.y == 0.4, "grounded knockback lift caps at 0.4");

    Fixture airborne;
    airborne.players[0].on_ground = false;
    airborne.players[0].velocity = {0.0, 0.25, 0.0};
    airborne.world().player(0).knockback(0.5, 3.0, 4.0);
    near(airborne.players[0].velocity.x, -0.3, 0.0, "knockback normalizes direction by division");
    near(airborne.players[0].velocity.z, -0.4, 0.0, "knockback normalized Z component");
    require(airborne.players[0].velocity.y == 0.25, "airborne knockback preserves vertical velocity");

    Fixture resistant;
    resistant.players[0].knockback_resistance = 1.0;
    resistant.world().player(0).knockback(0.4, 0.0, 0.0);
    require(!resistant.players[0].needs_sync && resistant.players[0].random.state == 25214903917ULL,
        "fully resisted knockback consumes no random draws and changes no motion");
    resistant.players[0].knockback_resistance = -1.0;
    resistant.world().player(0).knockback(0.4, 1.0, 0.0);
    near(resistant.players[0].velocity.x, -0.8, 0.0, "negative resistance is not silently clamped");

    Fixture fallback;
    fallback.world().player(0).knockback(0.4, 0.0, 0.0);
    random.setSeed(0);
    const double first_x = random.nextDouble();
    const double second_x = random.nextDouble();
    const double first_z = random.nextDouble();
    const double second_z = random.nextDouble();
    const double direction_x = (first_x - second_x) * 0.01;
    const double direction_z = (first_z - second_z) * 0.01;
    const double length = std::sqrt(direction_x * direction_x + direction_z * direction_z);
    near(fallback.players[0].velocity.x, -(direction_x / length) * 0.4, 0.0,
     "degenerate knockback uses explicitly ordered Java-compatible random draws");
    near(fallback.players[0].velocity.z, -(direction_z / length) * 0.4, 0.0, "fallback direction Z");
    require(fallback.players[0].random.state == random.state, "fallback persists its consumed RNG state");
}

void checkRecordingCsv() {
    const std::string header = "\"source_line\",\"tick\",\"input\",\"forward\",\"backward\","
        "\"left\",\"right\",\"jump\",\"sneak\",\"sprint\",\"x\",\"y\",\"z\",\"yaw\",\"pitch\",\"on_ground\"";
    const std::string first = "1,100,0,0,0,0,0,0,0,0,0.5,1,-2.5,0,0,1";
    const std::string second = "2,101,1,1,0,0,0,0,0,0,0.5,1,-2.4019999966323375,0,0,1";
    std::istringstream input("\xEF\xBB\xBF" + header + "\r\n" + first + "\r\n" + second + "\r\n");
    const auto samples = minecraft_sim::readRecording(input);
    require(samples.size() == 2 && samples.front().tick == 100
            && samples.back().controls.forward && !samples.back().velocity,
            "recording parser supports UTF-8 BOM, CRLF, quotes and extra source column");
    near(samples.back().position.z, -2.4019999966323375, 0.0, "CSV retains double precision");
    std::istringstream with_velocity(header + ",vx,vy,vz\n" + first + ",0,-0.078400001525878907,0\n"
                                    + second + ",0,-0.078400001525878907,0.053508008053839436\n");
    require(minecraft_sim::readRecording(with_velocity).back().velocity->z == 0.053508008053839436,
            "reference velocity columns are optional but parsed when provided");
    const auto reject = [](const std::string& text) {
        std::istringstream invalid(text);
        try {
            minecraft_sim::readRecording(invalid);
        } catch (const std::exception&) {
            return;
        }
        throw std::runtime_error("Malformed/unsupported recording unexpectedly accepted");
    };
    reject(header + "\n" + first + "\n2,102,1,1,0,0,0,0,0,0,0.5,1,0,0,0,1\n");
    reject(header + "\n" + first + "\n" + first + "\n");
    reject(header + "\n" + first + "\n2,101,0,1,0,0,0,0,0,0,0.5,1,0,0,0,1\n");
    std::istringstream sprint_sneak(header + "\n" + first + "\n2,101,97,1,0,0,0,0,1,1,0.5,1,0,0,0,1\n");
    const auto decoded = minecraft_sim::readRecording(sprint_sneak).back().controls;
    require(decoded.forward && decoded.sprint && decoded.sneak, "CSV decodes sprint and sneak without dropping mask bits");
    const std::string movement_header = header + ",vx,vy,vz,sprinting,crouching,pose,previous_forward,previous_sneak,"
        "sprint_trigger_time,sprint_window,food_level,may_fly,fall_distance,jump_delay,horizontal_collision,minor_horizontal_collision\n";
    std::istringstream enriched(movement_header + first + ",0,-0.0784,0,0,1,1,0,1,0,7,20,0,0,0,0,0\n"
        + second + ",0,-0.0784,0,0,1,1,0,1,0,7,20,0,0,0,0,0\n");
    const auto enriched_samples = minecraft_sim::readRecording(enriched);
    require(enriched_samples.front().movement_state->pose == minecraft_sim::PlayerPose::crouching
            && enriched_samples.front().movement_state->previous_sneak,
            "CSV reads explicit initial pose and prior input state");
    reject(header + ",pose\n" + first + ",0\n" + second + ",0\n");
    reject(header + "\n" + first + "\n2,101,0,0,0,0,0,0,0,0,nan,1,0,0,0,1\n");
    reject(header + ",vx\n" + first + ",0\n" + second + ",0\n");
    reject("tick,input\n100,0\n101,0\n");
    reject(header + "\n" + first + "\n");
}

void checkRecordedReplay() {
    Fixture source;
    source.players[0].pose = minecraft_sim::PlayerPose::crouching;
    source.players[0].crouching = true;
    source.players[0].previous_sneak = true;
    source.players[0].sprint_trigger_time = 5;
    std::vector<minecraft_sim::RecordedSample> samples;
    const auto capture = [&](std::uint64_t tick, minecraft_sim::Controls controls) {
        const auto& state = source.players[0];
        samples.push_back({tick, controls, state.position, state.yaw, state.pitch,
                           state.on_ground, state.velocity, state});
    };
    capture(100, {});
    for (int tick = 1; tick <= 30; ++tick) {
        minecraft_sim::Controls controls;
        controls.forward = tick < 20;
        controls.left = tick >= 12 && tick < 25;
        controls.jump = tick == 5;
        controls.sprint = tick < 10;
        controls.sneak = tick >= 12 && tick < 18;
        source.players[0].yaw = tick >= 10 ? 15.0F : 0.0F;
        source.world().player(0).setInputs(controls);
        source.world().processTick();
        capture(static_cast<std::uint64_t>(100 + tick), controls);
    }
    Fixture replay;
    std::ostringstream details;
    const auto report = minecraft_sim::compareRecording(samples, replay.world(), details, {});
    require(report.compared_ticks == 30 && report.ground_mismatches == 0
            && report.has_reference_velocity && report.has_movement_state,
            "initial row seeds replay but is not scored");
    for (const auto mismatches : report.movement_mismatches) {
        require(mismatches == 0, "sprint/crouch/pose flags match without per-tick state corrections");
    }
    for (const auto& field : report.fields) {
        require(field.count == 30 && field.absolute_max == 0.0,
                "recorded input/rotation replay reproduces shared-core reference exactly");
    }
    Fixture previous_phase;
    minecraft_sim::ComparisonOptions options;
    options.previous_inputs = true;
    std::ostringstream previous_details;
    const auto previous = minecraft_sim::compareRecording(samples, previous_phase.world(), previous_details, options);
    require(previous.fields[2].absolute_max > 0.0, "input phase choice changes replay, not reference state");

    for (auto& sample : samples) {
        sample.velocity.reset();
    }
    samples[3].position.x += 0.25;
    samples[4].on_ground = !samples[4].on_ground;
    samples[7].movement_state->sprinting = !samples[7].movement_state->sprinting;
    Fixture perturbed;
    std::ostringstream perturbed_details;
    const auto diagnostic = minecraft_sim::compareRecording(samples, perturbed.world(), perturbed_details, {});
    require(!diagnostic.has_reference_velocity && diagnostic.fields[3].count == 0,
            "missing reference velocity is not fabricated from displacement");
    near(diagnostic.fields[0].absolute_sum, 0.25, 1.0E-15, "reference perturbation never resets simulated position");
    near(diagnostic.fields[6].absolute_sum, 0.5, 1.0E-15, "displacement compares consecutive reference positions");
    require(diagnostic.fields[0].first_violation == 103
            && diagnostic.ground_mismatches == 1 && diagnostic.first_ground_mismatch == 104,
            "per-field divergence and discrete mismatch detection");
    require(diagnostic.movement_mismatches[0] == 1 && diagnostic.first_movement_mismatch[0] == 107,
            "later recorded sprint-state perturbation is diagnosed, never copied into simulation");
    std::ostringstream summary;
    minecraft_sim::writeComparisonSummary(summary, diagnostic);
    require(summary.str().find("vx,blocks/tick,0,NA,NA,NA") != std::string::npos,
            "summary marks absent velocity reference as NA");
        require(summary.str().find("on_ground,boolean,30,NA,NA,NA,NA,NA,NA,0,30,,104,1,") != std::string::npos,
            "saved summary includes discrete mismatch counts and first mismatch tick");
}

void checkSineTable() {
    const auto& table = testTable();
    const minecraft_sim::MathView math{table.data()};
    const double scale = minecraft_sim::sine_index_scale;
    require(sizeof(table) == 262144, "one shared sine table is 256 KiB");
    require(math.sin(0.0) == 0.0F, "zero sine");
    require(math.cos(0.0) == table[16384], "cosine shares the sine table");
    require(math.sin(1.25 / scale) == table[1], "positive index truncates");
    require(math.sin(-0.75 / scale) == table[0], "negative index truncates to zero");
    require(math.sin(-1.25 / scale) == table[65535], "negative index wraps");
    require(math.cos(-0.75 / scale) == table[16383],
        "cosine offsets before truncation, not after");
    require(math.sin(65536.0 / scale) == table[0], "index wraps at table size");
    require(math.sin(std::numeric_limits<double>::infinity()) == table[65535],
        "Java positive overflow saturates before masking");
    require(math.sin(-std::numeric_limits<double>::infinity()) == table[0],
        "Java negative overflow saturates before masking");
    require(math.sin(std::numeric_limits<double>::quiet_NaN()) == table[0],
        "Java NaN converts to zero");
}

void checkQueuedInputs() {
    std::array<minecraft_sim::PlayerState, 4> players{};
    std::array<minecraft_sim::Controls, 4> queued{};
    std::array<minecraft_sim::Controls, 4> held{};
    std::array<minecraft_sim::WorldState, 2> worlds{};
    minecraft_sim::BatchView batch{players.data(), queued.data(), held.data(),
                                 worlds.data(), 2, 2, nullptr, {testTable().data()}};
    auto world = batch.world(0);
    minecraft_sim::Controls controls;
    controls.forward = true;
    controls.yaw_delta = 15.0F;
    world.player(0).setInputs(controls);
    require(!world.player(0).inputs().forward, "setInputs must not advance input");
    require(players[0].tick == 0, "setInputs must not tick a player");
    world.processTick();
    require(held[0].forward && held[0].yaw_delta == 15.0F, "inputs must latch");
    require(players[0].tick == 1 && players[1].tick == 1, "each player ticks once");
    require(worlds[0].tick == 1, "world ticks once, not once per player");
    require(worlds[1].tick == 0 && players[2].tick == 0, "environments are isolated");
    world.processTick();
    require(held[0].forward && held[0].yaw_delta == 0.0F,
            "held controls persist and look events are consumed once");
    world.player(0).setInputs({});
    world.processTick();
    require(!held[0].forward, "neutral controls release movement");
}

void checkStandingAndFalling() {
    Fixture standing;
    for (int tick = 0; tick < 20; ++tick) {
        standing.world().processTick();
        require(standing.players[0].on_ground, "settled player remains grounded");
        near(standing.players[0].position.y, 1.0, 0.0, "floor prevents sinking");
    }
    near(standing.players[0].velocity.y, -0.08 * double(0.98F), 0.0,
         "gravity and drag follow floor collision");
    Fixture falling;
    falling.players[0].position.y = 5.0;
    falling.players[0].velocity = {};
    falling.players[0].on_ground = false;
    falling.world().processTick();
    near(falling.players[0].position.y, 5.0, 0.0, "first fall displacement precedes gravity");
    falling.world().processTick();
    near(falling.players[0].position.y, 5.0 - 0.08 * double(0.98F), 1.0E-15,
         "second tick uses previous gravity-adjusted velocity");
    for (int tick = 0; tick < 30; ++tick) {
        falling.world().processTick();
    }
    require(falling.players[0].on_ground, "falling player lands");
    near(falling.players[0].position.y, 1.0, 0.0, "landing clips to floor");
}

void checkWalkingAndLook() {
    Fixture walking;
    minecraft_sim::Controls controls;
    controls.forward = true;
    walking.world().player(0).setInputs(controls);
    walking.world().processTick();
    const float friction = 0.6F;
    const float acceleration = 0.1F * (0.21600002F / (friction * friction * friction));
    const double displacement = double(0.98F) * acceleration;
    near(walking.players[0].position.z, 0.5 + displacement, 1.0E-15,
         "walking uses source-derived ground acceleration and local input");
    near(walking.players[0].velocity.z, displacement * double(0.6F * 0.91F), 1.0E-15,
         "ground drag is applied after movement");
    Fixture turned;
    controls.yaw_delta = 90.0F;
    controls.pitch_delta = 120.0F;
    turned.world().player(0).setInputs(controls);
    turned.world().processTick();
    require(turned.players[0].position.x < 0.5, "yaw rotates forward toward negative X");
    require(turned.players[0].yaw == 90.0F && turned.players[0].pitch == 90.0F,
            "look applies degree deltas and clamps pitch");
    turned.world().processTick();
    require(turned.players[0].yaw == 90.0F, "look event is not repeated");
    Fixture opposed;
    controls = {};
    controls.forward = true;
    controls.backward = true;
    controls.left = true;
    controls.right = true;
    opposed.world().player(0).setInputs(controls);
    opposed.world().processTick();
    require(opposed.players[0].position.x == 0.5 && opposed.players[0].position.z == 0.5,
            "opposing keys cancel");
}

void checkJumpAndCeiling() {
    Fixture jumping;
    minecraft_sim::Controls controls;
    controls.jump = true;
    jumping.world().player(0).setInputs(controls);
    jumping.world().processTick();
    near(jumping.players[0].position.y, 1.0 + double(0.42F), 0.0,
         "ground jump happens before travel");
    near(jumping.players[0].velocity.y, (double(0.42F) - 0.08) * double(0.98F), 0.0,
         "jump gravity and drag happen after displacement");
    require(!jumping.players[0].on_ground && jumping.players[0].jump_delay == 10,
            "jump sets repeat delay and leaves ground");
    jumping.world().processTick();
    require(jumping.players[0].jump_delay == 9, "held jump does not reset repeat timer");
    jumping.world().player(0).setInputs({});
    for (int tick = 0; tick < 30; ++tick) {
        jumping.world().processTick();
    }
    require(jumping.players[0].on_ground && jumping.players[0].jump_delay == 0,
            "released jump lands and clears delay");
    Fixture ceiling;
    ceiling.setBlock(0, 3, 0);
    ceiling.world().player(0).setInputs(controls);
    ceiling.world().processTick();
    near(ceiling.players[0].position.y, 3.0 - double(1.8F), 0.0, "ceiling clips standing height");
    require(ceiling.players[0].vertical_collision && !ceiling.players[0].on_ground,
            "ceiling contact is not ground");
}

void checkWallAndSweep() {
    Fixture wall;
    for (int block_x = -2; block_x <= 2; ++block_x) {
        wall.setBlock(block_x, 1, 2);
        wall.setBlock(block_x, 2, 2);
    }
    minecraft_sim::Controls controls;
    controls.forward = true;
    wall.world().player(0).setInputs(controls);
    for (int tick = 0; tick < 30; ++tick) {
        wall.world().processTick();
    }
    near(wall.players[0].position.z, 2.0 - double(0.6F) / 2.0, 0.0,
         "wall stops the player at the bounding-box face");
    require(wall.players[0].horizontal_collision && wall.players[0].velocity.z == 0.0,
            "ordinary walls cancel blocked velocity");
    Fixture fast;
    fast.setBlock(3, 1, 0);
    fast.setBlock(3, 2, 0);
    fast.players[0].velocity.x = 50.0;
    fast.world().processTick();
    near(fast.players[0].position.x, 3.0 - double(0.6F) / 2.0, 0.0,
         "swept collision prevents tunnelling through a full cube");
        Fixture tolerance;
        tolerance.setBlock(3, 1, 0);
        tolerance.setBlock(3, 2, 0);
        const double difference = (double(1.0E-5F) + 1.0E-5) / 2.0;
        tolerance.players[0].position.x = 3.0 - double(0.6F) / 2.0 - (0.01 - difference);
        tolerance.players[0].velocity.x = 0.01;
        tolerance.world().processTick();
        require(tolerance.players[0].horizontal_collision,
            "horizontal collision tolerance is the widened float constant");
}

void checkReplay() {
    Fixture first;
    Fixture second;
    for (int tick = 0; tick < 50; ++tick) {
        minecraft_sim::Controls controls;
        controls.forward = tick < 20;
        controls.left = tick >= 20 && tick < 30;
        controls.jump = tick == 3;
        controls.yaw_delta = tick == 10 ? 15.0F : 0.0F;
        first.world().player(0).setInputs(controls);
        second.world().player(0).setInputs(controls);
        first.world().processTick();
        second.world().processTick();
        const auto& left = first.players[0];
        const auto& right = second.players[0];
        require(left.position.x == right.position.x && left.position.y == right.position.y
                && left.position.z == right.position.z && left.velocity.x == right.velocity.x
                && left.velocity.y == right.velocity.y && left.velocity.z == right.velocity.z
                && left.on_ground == right.on_ground && left.tick == right.tick,
                "identical initial state and controls replay identically");
    }
}

    void checkInputAndAxisOrder() {
        Fixture diagonal;
        minecraft_sim::Controls controls;
        controls.forward = true;
        controls.left = true;
        diagonal.world().player(0).setInputs(controls);
        diagonal.world().processTick();
        const double diagonal_length = std::hypot(diagonal.players[0].position.x - 0.5,
                                         diagonal.players[0].position.z - 0.5);
            const double expected_component = double(0.7071067F) * double(0.1F);
            near(diagonal_length, std::hypot(expected_component, expected_component), 1.0E-15,
                "26.3 square-input adjustment retains float-rounded diagonal direction");
        Fixture small;
        small.players[0].velocity.x = 0.0025;
        small.players[0].velocity.z = 0.0025;
        small.world().processTick();
        near(small.players[0].position.x, 0.5025, 1.0E-15,
            "player horizontal dead zone tests joint squared speed");
        Fixture stopped;
        stopped.players[0].velocity.x = 0.002;
        stopped.players[0].velocity.z = 0.002;
        stopped.world().processTick();
        require(stopped.players[0].position.x == 0.5 && stopped.players[0].position.z == 0.5,
              "joint horizontal speed below threshold is cleared");
        const double half_width = double(0.6F) / 2.0;
        const minecraft_sim::Aabb bounds{{0.5 - half_width, 1.0, 0.5 - half_width},
                                  {0.5 + half_width, 1.0 + double(1.8F), 0.5 + half_width}};
        const double gap = 1.0 - (0.5 + half_width);
        Fixture corner;
        corner.setBlock(1, 1, 1);
        auto accepted = minecraft_sim::collide(corner.blocks, bounds, {0.5, 0.0, 0.5});
        near(accepted.x, 0.5, 0.0, "equal horizontal magnitudes resolve X first");
        near(accepted.z, gap, 0.0, "second axis clips against translated bounds");
        accepted = minecraft_sim::collide(corner.blocks, bounds, {0.4, 0.0, 0.5});
        near(accepted.z, 0.5, 0.0, "larger Z movement resolves Z before X");
        near(accepted.x, gap, 0.0, "X clips after Z translation");
        Fixture overhead;
        overhead.setBlock(1, 3, 0);
        accepted = minecraft_sim::collide(overhead.blocks, bounds, {0.5, 0.5, 0.0});
        near(accepted.y, 0.5, 0.0, "Y resolves before horizontal displacement");
        near(accepted.x, gap, 0.0, "horizontal clipping uses vertically translated bounds");
        Fixture negative;
        negative.setBlock(-2, 1, 0);
        accepted = minecraft_sim::collide(negative.blocks, bounds, {-5.0, 0.0, 0.0});
        near(accepted.x, -1.0 - bounds.lower.x, 0.0,
            "negative coordinates and negative sweeps clip the leading face");
    }

#if defined(MCSIM_HAS_CUDA)
void checkCudaStreamStep() {
    using namespace minecraft_sim;
    Fixture fixture;
    CudaBatch device(fixture.batch);
    Controls controls;
    controls.forward = true;
    controls.yaw_delta = 45.0F;
    fixture.world().player(0).setInputs(controls);
    cudaStream_t stream = nullptr;
    require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess, "create nonblocking CUDA stream");
    PlayerState* snapshot = nullptr;
    require(cudaMallocHost(reinterpret_cast<void**>(&snapshot), sizeof(PlayerState)) == cudaSuccess,
        "allocate pinned CUDA snapshot");
    const auto view = device.deviceView();
    require(cudaMemcpyAsync(view.queued_controls, fixture.queued.data(), sizeof(Controls), cudaMemcpyHostToDevice, stream)
        == cudaSuccess, "queue stream controls");
    device.stepOnStream(3, stream);
    require(cudaMemcpyAsync(snapshot, view.players, sizeof(PlayerState), cudaMemcpyDeviceToHost, stream)
        == cudaSuccess, "queue stream snapshot");
    require(cudaStreamSynchronize(stream) == cudaSuccess, "synchronize CUDA stream");
    for (int tick = 0; tick < 3; ++tick) fixture.world().processTick();
    const auto& expected = fixture.players[0];
    near(snapshot->position.x, expected.position.x, 1.0E-12, "stream step X diverged");
    near(snapshot->position.y, expected.position.y, 1.0E-12, "stream step Y diverged");
    near(snapshot->position.z, expected.position.z, 1.0E-12, "stream step Z diverged");
    require(snapshot->tick == expected.tick && snapshot->yaw == expected.yaw,
        "stream step tick or look diverged");
        cudaGraph_t graph = nullptr;
        require(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess,
            "begin explicit stream capture");
        device.stepOnStream(3, stream);
        require(cudaStreamEndCapture(stream, &graph) == cudaSuccess, "end explicit stream capture");
        std::size_t node_count = 0;
        require(cudaGraphGetNodes(graph, nullptr, &node_count) == cudaSuccess && node_count == 3,
            "each physics tick must be enqueued on the supplied stream");
        require(cudaGraphDestroy(graph) == cudaSuccess, "destroy captured CUDA graph");
    bool rejected = false;
    try { device.stepOnStream(0, stream); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "stream step rejects a nonpositive tick count");
    require(cudaFreeHost(snapshot) == cudaSuccess, "free pinned CUDA snapshot");
    require(cudaStreamDestroy(stream) == cudaSuccess, "destroy CUDA stream");
}

void checkCudaWindStorage() {
    using namespace minecraft_sim;
    constexpr int environments = 16;
    std::array<PlayerState, environments * 2> players{};
    std::array<Controls, environments * 2> queued{};
    std::array<Controls, environments * 2> held{};
    std::array<WorldState, environments> worlds{};
    std::array<ProjectileWorldState, environments> projectiles{};
    std::array<std::array<ArrowState, projectile_arrow_capacity>, environments> arrows{};
    std::array<std::array<WindChargeState, 2>, environments> charges{};
    std::array<std::array<ThrownProjectileState, 2>, environments> thrown{};
    for (int environment = 0; environment < environments; ++environment) {
        projectiles[environment].arrows = arrows[environment].data();
        projectiles[environment].arrow_capacity = projectile_arrow_capacity;
        projectiles[environment].wind_charges = charges[environment].data();
        projectiles[environment].wind_charge_capacity = 2;
        projectiles[environment].thrown_items = thrown[environment].data();
        projectiles[environment].thrown_item_capacity = 2;
        charges[environment][0].position = {0.0, 20.0, 0.0};
        charges[environment][0].velocity = {1.0, 0.0, 0.0};
        charges[environment][0].active = true;
        thrown[environment][0].position = {0.0, 30.0, 0.0};
        thrown[environment][0].velocity = {1.0, 0.03, 0.0};
        thrown[environment][0].active = true;
    }
    charges[8][1] = charges[8][0];
    arrows[1][0].position = {12.0, 30.0, 0.0};
    arrows[1][0].velocity = {1.0, 0.0, 0.0};
    arrows[1][0].owner = 1;
    arrows[1][0].active = true;
    arrows[1][0].left_owner = true;
    BatchView host{players.data(), queued.data(), held.data(), worlds.data(), environments, 2,
                   nullptr, {testTable().data()}, projectiles.data()};
    CudaBatch device(host, true);
    device.step();
    device.download(host);
    near(charges[0][0].position.x, 1.0, 0.0, "CUDA wind charge advances after host upload");
    near(charges[1][0].position.x, 1.0, 0.0, "CUDA wind charge advances in second world");
    near(thrown[1][0].position.x, double(0.99F), 0.0, "CUDA thrown item advances after host upload");
    const int reset_id = 0;
    device.resetEnvironments(&reset_id, 1, players.data(), projectiles.data());
    device.download(host);
    require(!charges[0][0].active && charges[1][0].active,
            "CUDA selective reset clears only the chosen wind pool");
        require(!thrown[0][0].active && thrown[1][0].active,
            "CUDA selective reset clears only the chosen thrown item pool");
    device.step();
    device.download(host);
    near(charges[1][0].position.x, 2.0, 0.0, "CUDA untouched wind charge continues after reset");
    device.step(3);
    device.download(host);
    std::array<ArrowObservationSummary, environments * 2> summaries{};
    device.downloadForTraining(host, summaries.data());
    for (int environment = 0; environment < environments; ++environment) {
        require(worlds[environment].tick == (environment == 0 ? 4 : 5),
                "CUDA multi-tick request advances each world exactly once per tick");
    }
    require(players[2].tick == 5, "CUDA multi-tick request advances player state");
    near(charges[1][0].position.x, 5.0, 0.0, "CUDA multi-tick request advances wind charges");
    require(summaries[2].found && !summaries[3].found, "CUDA final arrow summaries retain ownership");
    near(summaries[2].relative_position.x, arrows[1][0].position.x - players[2].position.x, 1.0E-12,
         "CUDA multi-tick request summarizes the final arrow state");
    projectiles[0].wind_charge_capacity = 1;
    bool rejected = false;
    try {
        device.download(host);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "CUDA rejects mismatched wind snapshot storage");
}

int checkCudaReplay() {
    if (!minecraft_sim::CudaBatch::available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    checkCudaStreamStep();
    checkCudaWindStorage();
    constexpr int environments = 137;
    constexpr int players_per_world = 2;
    Fixture terrain;
    std::array<minecraft_sim::PlayerState, environments * players_per_world> players{};
    std::array<minecraft_sim::Controls, environments * players_per_world> queued{};
    std::array<minecraft_sim::Controls, environments * players_per_world> held{};
    std::array<minecraft_sim::WorldState, environments> worlds{};
    std::array<minecraft_sim::BlockWorldView, environments> blocks{};
    for (int environment = 0; environment < environments; ++environment) {
        blocks[environment] = terrain.blocks;
        for (int player = 0; player < players_per_world; ++player) {
            auto& state = players[environment * players_per_world + player];
            state = terrain.players[0];
            state.position.x += double(player) * 0.25;
            if (environment % 4 == 0) {
                state.position.x = 0.5;
                state.position.z += player * 2.0;
                state.yaw = player * 180.0F;
                state.attack_strength_ticks = 20;
                state.knockback_resistance = environment % 3 == 0 ? 0.25 : 0.0;
                state.sprinting = environment % 8 == 0 && player == 0;
                state.previous_forward = state.sprinting;
                if (environment % 11 == 0) {
                    state.health = 7.0F;
                }
                if (environment % 8 == 4 && player == 0) {
                    state.position.y = 1.1;
                    state.on_ground = false;
                    state.fall_distance = 0.1;
                }
                if (environment == 4) {
                    state.position.z = 0.5;
                }
            }
        }
    }
    minecraft_sim::BatchView host{players.data(), queued.data(), held.data(), worlds.data(),
                                  environments, players_per_world, blocks.data(), {testTable().data()}};
    for (int environment = 0; environment < environments; environment += 8) {
        auto world = host.world(environment);
        worlds[environment].simulate_food = true;
        worlds[environment].natural_regeneration = environment % 16 == 0;
        worlds[environment].difficulty = environment % 16 == 0
            ? minecraft_sim::Difficulty::peaceful : minecraft_sim::Difficulty::hard;
        world.player(0).equipWeapon(environment % 16 == 0
            ? minecraft_sim::WeaponItem::diamond_axe : minecraft_sim::WeaponItem::netherite_sword);
        world.player(0).state().attack_strength_ticks = 100;
        world.player(1).equipShield(true);
        world.player(1).state().using_shield = environment % 16 == 0;
        world.player(1).state().shield_use_ticks = environment % 16 == 0 ? 5 : 0;
        world.player(1).state().shield_cooldown = environment % 16 == 0 ? 0 : 3;
        auto& defender = world.player(1).state();
        defender.health = 10.0F;
        defender.food_level = environment % 16 == 0 ? 20 : 0;
        defender.food_saturation = environment % 16 == 0 ? 6.0F : 0.0F;
        defender.food_tick_timer = environment % 16 == 0 ? 9 : 79;
        for (int slot = 0; slot < 4; ++slot) {
            world.player(1).equipArmor(static_cast<minecraft_sim::ArmorSlot>(slot),
                environment % 16 == 0 ? minecraft_sim::ArmorMaterial::diamond : minecraft_sim::ArmorMaterial::netherite);
        }
    }
    minecraft_sim::CudaBatch device(host);
    auto gpu_players = players;
    auto gpu_queued = queued;
    auto gpu_held = held;
    auto gpu_worlds = worlds;
    minecraft_sim::BatchView snapshot{gpu_players.data(), gpu_queued.data(), gpu_held.data(), gpu_worlds.data(),
                                      environments, players_per_world, nullptr, {testTable().data()}};
    for (int decision = 0; decision < 20; ++decision) {
        for (int environment = 0; environment < environments; ++environment) {
            for (int player = 0; player < players_per_world; ++player) {
                minecraft_sim::Controls inputs;
                inputs.forward = environment % 3 == 0 || environment % 4 == 0;
                inputs.left = environment % 3 == 1;
                inputs.jump = decision == 2 && player == 0;
                inputs.sprint = environment % 7 == 0 && decision < 8;
                inputs.sneak = environment % 7 == 1 && decision >= 5;
                inputs.attack = environment % 4 == 0 && decision % 4 == 0;
                if (environment % 8 == 0) {
                    inputs.forward = false;
                    inputs.sprint = false;
                    inputs.use_item = player == 1 && decision < 16;
                }
                inputs.yaw_delta = decision == 4 ? float(environment - 68) : 0.0F;
                host.world(environment).player(player).setInputs(inputs);
            }
        }
        device.uploadControls(queued.data(), queued.size());
        const int ticks = decision % 3 + 1;
        device.step(ticks);
        for (int tick = 0; tick < ticks; ++tick) {
            for (int environment = 0; environment < environments; ++environment) {
                host.world(environment).processTick();
            }
        }
        device.download(snapshot);
        for (std::size_t index = 0; index < players.size(); ++index) {
            const auto& expected = players[index];
            const auto& actual = gpu_players[index];
            near(actual.position.x, expected.position.x, 1.0E-12, "CUDA X diverged");
            near(actual.position.y, expected.position.y, 1.0E-12, "CUDA Y diverged");
            near(actual.position.z, expected.position.z, 1.0E-12, "CUDA Z diverged");
            near(actual.velocity.x, expected.velocity.x, 1.0E-12, "CUDA velocity X diverged");
            near(actual.velocity.y, expected.velocity.y, 1.0E-12, "CUDA velocity Y diverged");
            near(actual.velocity.z, expected.velocity.z, 1.0E-12, "CUDA velocity Z diverged");
            near(actual.fall_distance, expected.fall_distance, 1.0E-12, "CUDA fall distance diverged");
            require(actual.on_ground == expected.on_ground
                    && actual.horizontal_collision == expected.horizontal_collision
                    && actual.vertical_collision == expected.vertical_collision
                    && actual.jump_delay == expected.jump_delay && actual.tick == expected.tick
                    && actual.yaw == expected.yaw && actual.pitch == expected.pitch
                    && actual.sprinting == expected.sprinting && actual.crouching == expected.crouching
                    && actual.minor_horizontal_collision == expected.minor_horizontal_collision
                    && actual.previous_forward == expected.previous_forward
                    && actual.previous_sneak == expected.previous_sneak
                    && actual.sprint_trigger_time == expected.sprint_trigger_time
                    && actual.pose == expected.pose && actual.needs_sync == expected.needs_sync
                    && actual.random.state == expected.random.state,
                    "CUDA discrete player state diverged");
                    require(actual.health == expected.health && actual.absorption == expected.absorption
                        && actual.last_hurt == expected.last_hurt && actual.damage_cooldown == expected.damage_cooldown
                        && actual.hurt_time == expected.hurt_time && actual.hurt_duration == expected.hurt_duration
                        && actual.death_time == expected.death_time && actual.attack_strength_ticks == expected.attack_strength_ticks
                        && actual.attack_miss_time == expected.attack_miss_time,
                        "CUDA damage/health/attack timers diverged");
                    require(actual.weapon == expected.weapon && actual.attack_damage == expected.attack_damage
                        && actual.food_level == expected.food_level && actual.food_saturation == expected.food_saturation
                        && actual.food_exhaustion == expected.food_exhaustion && actual.food_tick_timer == expected.food_tick_timer
                        && actual.attack_speed == expected.attack_speed && actual.armor == expected.armor
                        && actual.armor_toughness == expected.armor_toughness
                        && actual.armor_knockback_resistance == expected.armor_knockback_resistance
                        && actual.shield_equipped == expected.shield_equipped && actual.using_shield == expected.using_shield
                        && actual.shield_use_ticks == expected.shield_use_ticks && actual.shield_cooldown == expected.shield_cooldown
                        && actual.item_use_delay == expected.item_use_delay && actual.attack_suppressed == expected.attack_suppressed
                        && actual.last_blocked_damage == expected.last_blocked_damage
                        && actual.last_shield_disabled == expected.last_shield_disabled,
                        "CUDA equipment and shield lifecycle diverged");
                    for (int slot = 0; slot < 4; ++slot) {
                        require(actual.armor_items[slot] == expected.armor_items[slot], "CUDA armor slots diverged");
                    }
                    const auto& expected_attack = expected.last_attack;
                    const auto& actual_attack = actual.last_attack;
                    require(actual_attack.target == expected_attack.target && actual_attack.attempted == expected_attack.attempted
                        && actual_attack.accepted == expected_attack.accepted && actual_attack.critical == expected_attack.critical
                        && actual_attack.sprint_bonus == expected_attack.sprint_bonus && actual_attack.sweep == expected_attack.sweep
                        && actual_attack.hit_block == expected_attack.hit_block && actual_attack.sweep_hits == expected_attack.sweep_hits
                        && actual_attack.charge == expected_attack.charge && actual_attack.incoming_damage == expected_attack.incoming_damage
                        && actual_attack.health_lost == expected_attack.health_lost
                        && actual_attack.blocked_damage == expected_attack.blocked_damage
                        && actual_attack.shield_disabled == expected_attack.shield_disabled,
                        "CUDA legal targeting or melee outcome diverged");
                    require(gpu_queued[index].attack == queued[index].attack && gpu_held[index].attack == held[index].attack
                        && gpu_queued[index].use_item == queued[index].use_item && gpu_held[index].use_item == held[index].use_item,
                        "CUDA attack-event consumption diverged");
        }
        for (int environment = 0; environment < environments; ++environment) {
            require(gpu_worlds[environment].tick == worlds[environment].tick,
                    "CUDA world tick or environment indexing diverged");
            require(gpu_worlds[environment].natural_regeneration == worlds[environment].natural_regeneration
                && gpu_worlds[environment].simulate_food == worlds[environment].simulate_food
                && gpu_worlds[environment].difficulty == worlds[environment].difficulty,
                "CUDA survival world settings diverged");
        }
    }
    std::cout << "PASS: CUDA replay, non-aligned batch size and environment isolation\n";
    return 0;
}
#endif

}

int main(int argc, char** argv) {
    try {
        if (argc > 1) {
#if defined(MCSIM_HAS_CUDA)
            if (argc == 2 && std::string_view(argv[1]) == "--cuda") {
                return checkCudaReplay();
            }
#else
            static_cast<void>(argv);
#endif
            throw std::invalid_argument("Unsupported check mode");
        }
        require(bridge_match_checks::outcomes(), "Bridge match outcome ordering");
        require(bridge_match_checks::lifecycle(), "Bridge match lifecycle resets");
        require(bridge_match_checks::selectedWorldReset(), "Bridge match selective world reset");
        require(bridge_match_checks::tick({testTable().data()}), "Bridge cage and world tick");
        require(bridge_match_checks::completeMatch({testTable().data()}), "Bridge full internal match");
        require(bridge_match_checks::kitsAndRefill({testTable().data()}), "Bridge kit and arrow refill lifecycle");
        require(bridge_match_checks::building({testTable().data()}), "Bridge block targeting, placement and breaking");
        require(bridge_match_checks::simultaneousBuilding({testTable().data()}), "Bridge ordered simultaneous building");
        require(bridge_match_checks::externalCutoff({testTable().data()}), "Bridge external cutoff retains final state");
        checkGoldenAppleBoundary();
        checkStandaloneItemTick();
        const int apple_check = golden_apple_checks::run({testTable().data()});
        if (apple_check != 0) throw std::runtime_error("Golden apple shared check " + std::to_string(apple_check));
        require(terrain_checks::blockTargeting(), "full-cube block target faces and boundaries");
        checkMutableTerrain();
        checkBridgeLocalVoxels();
        checkBridgeTerrainGeneration();
        checkObjectiveProgress();
        checkCompositeWorldConsumers();
        checkPolicyFixtures();
        checkEquipmentObservation();
        checkShapingReward();
        checkDuelRewards();
        checkComparisonStatistics();
        checkMeleeTargeting();
        checkWindExplosion();
        checkMaceDamage();
        checkMeleeDamage();
        checkFoodRegeneration();
        checkEquipmentRules();
        checkShieldWorld();
        checkSwordSweep();
        checkMeleeWorld();
        checkSprinting();
        checkCrouchingAndEdges();
        checkPlayerContact();
        checkKnockback();
        checkRecordingCsv();
        checkRecordedReplay();
        checkSineTable();
        checkQueuedInputs();
        checkStandingAndFalling();
        checkWalkingAndLook();
        checkJumpAndCeiling();
        checkWallAndSweep();
        checkReplay();
        checkInputAndAxisOrder();
        std::cout << "PASS: lookup indexing, tick ownership, movement, targeting, melee/damage, collision and CSV replay\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
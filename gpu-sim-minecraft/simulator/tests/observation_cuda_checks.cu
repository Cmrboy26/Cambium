#include "../apps/observation.hpp"
#include "../apps/actions.hpp"
#include "../apps/rewards.hpp"
#include "minecraft_sim/cuda_backend.hpp"
#include "minecraft_sim/terrain.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

constexpr int feature_count = 102;
constexpr int scenario_count = 3;

__global__ void terrainScenario(minecraft_sim::BatchView batch, double* output, int phase) {
    using namespace minecraft_sim;
    const auto first = batch.block_worlds[0], second = batch.block_worlds[1];
    if (phase == 1) {
        output[9] = first.setPlacedBlock(0, 1, 0, PlacedBlock::team0)
            && !first.setPlacedBlock(0, 0, 0, PlacedBlock::team0)
            && !first.setPlacedBlock(2, 1, 0, PlacedBlock::team0);
    } else if (phase == 2) {
        output[9] = second.setPlacedBlock(1, 1, 0, PlacedBlock::team1);
    }
    const Vec3 origin{-2.0, 1.5, 0.5}, delta{4.0, 0.0, 0.0};
    const Aabb first_cell{{0.1, 1.1, 0.1}, {0.9, 1.9, 0.9}};
    const Aabb second_cell{{1.1, 1.1, 0.1}, {1.9, 1.9, 0.9}};
    output[0] = blockRayFraction(first, origin, delta);
    output[1] = blockRayFraction(second, origin, delta);
    output[2] = !noBlockCollision(first, first_cell);
    output[3] = !noBlockCollision(second, first_cell);
    output[4] = !noBlockCollision(second, second_cell);
    output[5] = static_cast<int>(first.overlay[6]);
    output[6] = static_cast<int>(second.overlay[7]);
    output[7] = first.cells == second.cells;
    output[8] = first.overlay != second.overlay;
    output[10] = first.solid(0, 0, 0) && second.solid(0, 0, 0);
}

bool checkMutableTerrain() {
    using namespace minecraft_sim;
    std::array<Block, 8> base{Block::solid, Block::solid, Block::solid, Block::solid,
                              Block::air, Block::air, Block::air, Block::air};
    TerrainOverlays terrain({base.data(), 4, 2, 1, -2, 0, 0}, 2);
    std::array<PlayerState, 4> players{};
    std::array<Controls, 4> queued{}, held{};
    std::array<WorldState, 2> worlds{};
    const auto sine = makeSineTable();
    BatchView host{players.data(), queued.data(), held.data(), worlds.data(), 2, 2,
                   terrain.data(), {sine.data()}, nullptr};
    bool budget_rejected = false;
    try { CudaBatch too_large(host, false, 71); } catch (const std::invalid_argument&) { budget_rejected = true; }
    if (!budget_rejected) return false;
    CudaBatch exact_budget(host, false, 72);
    CudaBatch device(host);
    double* output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&output), 11 * sizeof(double)) != cudaSuccess) return false;
    std::array<double, 11> actual{};
    const auto sample = [&](int phase) {
        terrainScenario<<<1, 1>>>(device.deviceView(), output, phase);
        return cudaGetLastError() == cudaSuccess
            && cudaMemcpy(actual.data(), output, sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess;
    };
    bool result = sample(0) && actual[0] == 1.0 && actual[1] == 1.0
        && actual[2] == 0.0 && actual[3] == 0.0 && actual[4] == 0.0
        && actual[7] == 1.0 && actual[8] == 1.0 && actual[10] == 1.0;
    result = result && sample(1) && actual[0] == 0.5 && actual[1] == 1.0
        && actual[2] == 1.0 && actual[3] == 0.0 && actual[5] == 1.0 && actual[9] == 1.0;
    result = result && sample(2) && actual[0] == 0.5 && actual[1] == 0.75
        && actual[4] == 1.0 && actual[6] == 2.0 && actual[9] == 1.0;
    const int first = 0;
    device.resetEnvironments(&first, 1, players.data());
    result = result && sample(0) && actual[0] == 1.0 && actual[1] == 0.75
        && actual[2] == 0.0 && actual[4] == 1.0 && actual[5] == 0.0
        && actual[6] == 2.0 && actual[10] == 1.0;
    const int second = 1;
    device.resetEnvironments(&second, 1, players.data() + 2);
    result = result && sample(0) && actual[0] == 1.0 && actual[1] == 1.0 && actual[6] == 0.0;
    cudaFree(output);
    return result;
}

bool checkStaticTerrainSharing() {
    using namespace minecraft_sim;
    std::array<Block, 8> base{};
    std::array<BlockWorldView, 2> blocks{{{base.data(), 4, 2, 1}, {base.data(), 4, 2, 1}}};
    std::array<PlayerState, 4> players{};
    std::array<Controls, 4> queued{}, held{};
    std::array<WorldState, 2> worlds{};
    const auto sine = makeSineTable();
    BatchView host{players.data(), queued.data(), held.data(), worlds.data(), 2, 2,
                   blocks.data(), {sine.data()}, nullptr};
    bool budget_rejected = false;
    try { CudaBatch too_large(host, false, 23); } catch (const std::invalid_argument&) { budget_rejected = true; }
    if (!budget_rejected) return false;
    CudaBatch device(host, false, 24);
    std::array<BlockWorldView, 2> uploaded{};
    return cudaMemcpy(uploaded.data(), device.deviceView().block_worlds, sizeof(uploaded),
                      cudaMemcpyDeviceToHost) == cudaSuccess
        && uploaded[0].cells == uploaded[1].cells && uploaded[0].cells != nullptr;
}

bool checkFixedTerrainReset() {
    using namespace minecraft_sim;
    constexpr int environments = 2;
    std::array<Block, 8> base{Block::solid, Block::solid, Block::solid, Block::solid,
                              Block::air, Block::air, Block::air, Block::air};
    TerrainOverlays terrain({base.data(), 4, 2, 1, -2, 0, 0}, environments);
    std::array<PlayerState, environments * 2> players{};
    std::array<Controls, environments * 2> queued{}, held{};
    std::array<WorldState, environments> worlds{};
    std::array<ProjectileWorldState, environments> projectiles{};
    std::array<ArrowState, environments * projectile_arrow_capacity> arrows{};
    std::array<app::ObservationActorMetadata, environments * 2> actors{};
    std::array<app::DuelRewardState, environments> reward_states{};
    std::array<std::uint64_t, environments> episodes{};
    for (int environment = 0; environment < environments; ++environment) {
        projectiles[environment].arrows = arrows.data() + environment * projectile_arrow_capacity;
        projectiles[environment].arrow_capacity = projectile_arrow_capacity;
    }
    for (auto& actor : actors) {
        actor.slot_count = 1;
        actor.slots[0] = WeaponItem::diamond_sword;
    }
    const auto sine = makeSineTable();
    BatchView host{players.data(), queued.data(), held.data(), worlds.data(), environments, 2,
                   terrain.data(), {sine.data()}, projectiles.data()};
    CudaBatch device(host);
    const app::ObservationProjection projection{0, 1.0F, 0.0F, -1.0F, 1.0F, false};
    device.configureObservations(&projection, 1, actors.data(), actors.size());
    app::ActionHead head;
    head.kind = 0;
    head.count = 1;
    head.values[0] = 0.0F;
    device.configureActions(&head, 1);
    app::DuelRewardSettings settings;
    settings.max_ticks = 1;
    device.configureRewards(settings, reward_states.data(), reward_states.size());
    device.configureFixedResets(players.data(), actors.data(), episodes.data(), episodes.size(), 123);
    double* output = nullptr;
    std::uint8_t* device_mask = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&output), 11 * sizeof(double)) != cudaSuccess) return false;
    if (cudaMalloc(reinterpret_cast<void**>(&device_mask), environments) != cudaSuccess) {
        cudaFree(output);
        return false;
    }
    std::array<double, 11> actual{};
    const auto sample = [&](int phase) {
        terrainScenario<<<1, 1>>>(device.deviceView(), output, phase);
        return cudaGetLastError() == cudaSuccess
            && cudaMemcpy(actual.data(), output, sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess;
    };
    bool result = sample(1) && actual[9] == 1.0 && sample(2) && actual[9] == 1.0;
    std::array<std::uint8_t, environments> mask{1, 2};
    result = result && cudaMemcpy(device_mask, mask.data(), mask.size(), cudaMemcpyHostToDevice) == cudaSuccess;
    device.resetFixedOnStream(device_mask, mask.size(), nullptr);
    result = result && device.actionStatus() == 4 && sample(0) && actual[0] == 0.5 && actual[1] == 0.75;
    device.clearActionStatus();
    mask = {1, 0};
    result = result && cudaMemcpy(device_mask, mask.data(), mask.size(), cudaMemcpyHostToDevice) == cudaSuccess;
    device.resetFixedOnStream(device_mask, mask.size(), nullptr);
    result = result && device.actionStatus() == 0 && sample(0) && actual[0] == 1.0 && actual[1] == 0.75
        && actual[5] == 0.0 && actual[6] == 2.0;
    mask = {0, 1};
    result = result && cudaMemcpy(device_mask, mask.data(), mask.size(), cudaMemcpyHostToDevice) == cudaSuccess;
    device.resetFixedOnStream(device_mask, mask.size(), nullptr);
    result = result && device.actionStatus() == 0 && sample(0) && actual[0] == 1.0 && actual[1] == 1.0;
    cudaFree(device_mask);
    cudaFree(output);
    return result;
}

MCSIM_HD void encodeScenarios(float* output) {
    using namespace minecraft_sim;
    PlayerState players[2]{};
    players[0].position = {1.0, 2.0, -3.0};
    players[0].velocity = {0.25, -0.5, 0.75};
    players[0].health = 18.0F;
    players[0].yaw = 225.0F;
    players[0].pitch = -15.0F;
    players[0].on_ground = true;
    players[0].sprinting = true;
    players[0].weapon = WeaponItem::bow;
    players[0].using_bow = true;
    players[0].snowball_count = 0;
    players[0].wind_charge_count = 2;
    players[1].position = {-2.0, 3.0, 7.0};
    players[1].velocity = {-0.5, 0.25, 1.25};
    players[1].shield_equipped = true;
    const WeaponItem first_slots[] = {WeaponItem::diamond_sword, WeaponItem::wind_charge,
                                      WeaponItem::snowball, WeaponItem::bow};
    const WeaponItem second_slots[] = {WeaponItem::iron_axe, WeaponItem::unarmed};
    const float rays[] = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F, 0.8F};
    ArrowObservationSummary summaries[2]{};
    summaries[0].ordinary_arrows = 32;
    summaries[0].draw_ticks = 10;
    summaries[0].found = true;
    summaries[0].relative_position = {2.0, -1.0, 4.0};
    summaries[0].relative_velocity = {-0.25, 0.5, 0.75};
    for (int scenario = 0; scenario < scenario_count; ++scenario) {
        float* catalogue = output + scenario * feature_count;
        for (int index = 0; index < feature_count; ++index) catalogue[index] = -17.0F;
        const int actor = scenario == 1 ? 1 : 0;
        const auto& source = players[actor];
        const auto& target = players[scenario == 2 ? actor : 1 - actor];
        const WeaponItem* slots = scenario == 0 ? first_slots : (scenario == 1 ? second_slots : nullptr);
        const int slot_count = scenario == 0 ? 4 : (scenario == 1 ? 2 : 0);
        app::baseObservation(source, target, rays, catalogue);
        app::equipmentObservation(catalogue, source, target, slots, slot_count);
        app::velocityObservation(catalogue, source, target);
        app::movementObservation(catalogue, source, target);
        app::arrowObservation(catalogue, source, target, summaries[scenario == 0 ? 0 : 1]);
    }
}

__global__ void encodeScenariosOnDevice(float* output) {
    encodeScenarios(output);
}

MCSIM_HD void rewardScenarios(float* output) {
    using namespace minecraft_sim;
    PlayerState source, target;
    source.position = {0.5, 0.0, 0.5};
    target.position = {0.5, 0.0, 5.5};
    const float weights[] = {2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    app::shapingRewards(source, target, 0, {}, weights, output);
    source.pitch = 61.0F;
    target.position = source.position;
    app::shapingRewards(source, target, 0, {}, weights, output + 5);
    source.pitch = -10.0F;
    source.weapon = WeaponItem::bow;
    source.velocity = {0.1, 0.0, 0.0};
    target.position = {0.5, 0.0, 5.5};
    target.velocity = {0.2, 0.0, 0.0};
    app::shapingRewards(source, target, 1, {}, weights, output + 10);
    app::shapingRewards(source, target, 0, {}, weights, output + 15);
    Block block = Block::solid;
    BlockWorldView obstruction{&block, 1, 1, 1, 0, 1, 2};
    app::shapingRewards(source, target, 1, obstruction, weights, output + 20);
}

__global__ void rewardScenariosOnDevice(float* output) {
    rewardScenarios(output);
}

bool checkShapingRewardParity() {
    std::array<float, 25> expected{}, actual{};
    rewardScenarios(expected.data());
    float* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(actual)) != cudaSuccess) return false;
    rewardScenariosOnDevice<<<1, 1>>>(device_output);
    const auto result = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess
        && cudaMemcpy(actual.data(), device_output, sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(device_output);
    if (!result || actual[0] != 2.0F || actual[3] != 5.0F || actual[4] != 0.0F
        || actual[5] != 0.0F || actual[6] != 3.0F || actual[8] != 0.0F || actual[9] != 6.0F
        || actual[15] != 0.0F || actual[18] != 0.0F || actual[20] != 0.0F || actual[23] != 0.0F) return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || std::abs(actual[index] - expected[index]) > 1.0e-5F) {
            std::cerr << "CPU/CUDA shaping reward mismatch at index " << index << '\n';
            return false;
        }
    }
    return true;
}

bool checkBatchedDuelRewardParity() {
    using namespace minecraft_sim;
    constexpr int environments = 137;
    std::array<PlayerState, environments * 2> players{};
    std::array<Controls, environments * 2> queued{}, held{};
    std::array<WorldState, environments> worlds{};
    std::array<ProjectileWorldState, environments> projectiles{};
    std::array<ArrowState, environments * projectile_arrow_capacity> arrows{};
    std::array<Block, 8 * 8 * 8> cells{};
    std::array<BlockWorldView, environments> blocks{};
    std::array<app::ObservationActorMetadata, environments * 2> actors{};
    std::array<app::DuelRewardState, environments> states{};
    std::array<app::DuelRewardStep, environments> expected{}, actual{};
    app::DuelRewardSettings settings;
    settings.max_ticks = 1;
    for (int term = 0; term < 10; ++term) settings.weights[term] = static_cast<float>(term + 1) * 0.25F;
    settings.block_weight = 2.5F;
    settings.event_weights[0] = 0.75F;
    settings.event_weights[1] = -0.5F;
    for (int term = 0; term < 5; ++term) settings.shaping_weights[term] = static_cast<float>(term + 1) * 0.2F;
    for (int environment = 0; environment < environments; ++environment) {
        blocks[environment] = {cells.data(), 8, 8, 8, -4, 0, -4};
        projectiles[environment].arrows = arrows.data() + environment * projectile_arrow_capacity;
        projectiles[environment].arrow_capacity = projectile_arrow_capacity;
        worlds[environment].tick = 1;
        players[environment * 2].position = {0.5, 1.0, -1.0};
        players[environment * 2 + 1].position = {0.5, 1.0, 1.0};
        actors[environment * 2].slot_count = 1;
        actors[environment * 2 + 1].slot_count = 1;
        actors[environment * 2].slots[0] = WeaponItem::diamond_sword;
        actors[environment * 2 + 1].slots[0] = WeaponItem::diamond_sword;
        states[environment].dealt_damage[0] = environment % 2 == 0;
        switch (environment % 5) {
        case 0:
            players[environment * 2].last_attack.health_lost = 5.0F;
            players[environment * 2 + 1].health = 0.0F;
            break;
        case 1:
            players[environment * 2].last_attack.attempted = true;
            players[environment * 2].last_attack.target = -1;
            players[environment * 2].last_attack.attack_speed = 4.0;
            break;
        case 2:
            players[environment * 2].last_attack.blocked_damage = 2.0F;
            worlds[environment].wind_self_launch[0] = 0.5;
            break;
        case 3:
            players[environment * 2].position.x = 4.0;
            players[environment * 2 + 1].health = 0.0F;
            break;
        }
    }
    auto expected_states = states;
    for (int environment = 0; environment < environments; ++environment) {
        expected[environment] = app::duelRewards(players.data() + environment * 2, worlds[environment],
            blocks[environment], projectiles[environment], settings, expected_states[environment]);
    }
    const auto sine = makeSineTable();
    BatchView host{players.data(), queued.data(), held.data(), worlds.data(), environments, 2,
                   blocks.data(), {sine.data()}, projectiles.data()};
    CudaBatch device(host);
    const app::ObservationProjection projection{0, 1.0F, 0.0F, -1.0F, 1.0F, false};
    device.configureObservations(&projection, 1, actors.data(), actors.size());
    device.configureRewards(settings, states.data(), states.size());
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return false;
    const auto matches = [&]() {
        device.evaluateRewardsOnStream(stream);
        device.downloadRewardResults(actual.data(), actual.size());
        for (int environment = 0; environment < environments; ++environment) {
            const auto& wanted = expected[environment];
            const auto& received = actual[environment];
            if (wanted.terminated != received.terminated || wanted.truncated != received.truncated) return false;
            for (int player = 0; player < 2; ++player) {
                if (wanted.lost[player] != received.lost[player]
                    || std::abs(wanted.rewards[player] - received.rewards[player]) > 1.0e-5F) return false;
                for (int term = 0; term < 18; ++term) {
                    if (std::abs(wanted.contributions[player][term] - received.contributions[player][term]) > 1.0e-5) {
                        std::cerr << "CUDA reward mismatch at world " << environment << " player " << player
                                  << " term " << term << '\n';
                        return false;
                    }
                }
            }
        }
        return true;
    };
    bool result = matches();
    const int reset_id = 0;
    std::array<PlayerState, 2> reset_players{players[0], players[1]};
    reset_players[0].last_attack = {};
    reset_players[1].health = 20.0F;
    const std::array<app::ObservationActorMetadata, 2> reset_actors{actors[0], actors[1]};
    device.resetEnvironments(&reset_id, 1, reset_players.data(), projectiles.data(), reset_actors.data());
    device.download(host);
    expected_states[0] = {};
    for (int environment = 0; environment < environments; ++environment) {
        expected[environment] = app::duelRewards(players.data() + environment * 2, worlds[environment],
            blocks[environment], projectiles[environment], settings, expected_states[environment]);
    }
    result = result && matches();
    cudaStreamDestroy(stream);
    return result;
}

bool checkFinishedWorldProtection() {
    using namespace minecraft_sim;
    constexpr int environments = 2;
    std::array<PlayerState, environments * 2> players{};
    std::array<Controls, environments * 2> queued{}, held{};
    std::array<WorldState, environments> worlds{};
    std::array<ProjectileWorldState, environments> projectiles{};
    std::array<ArrowState, environments * projectile_arrow_capacity> arrows{};
    std::array<Block, 8 * 8 * 8> cells{};
    std::array<BlockWorldView, environments> blocks{};
    std::array<app::ObservationActorMetadata, environments * 2> actors{};
    std::array<app::DuelRewardState, environments> states{};
    std::array<app::DuelRewardStep, environments> results{};
    for (int environment = 0; environment < environments; ++environment) {
        blocks[environment] = {cells.data(), 8, 8, 8, -4, 0, -4};
        projectiles[environment].arrows = arrows.data() + environment * projectile_arrow_capacity;
        projectiles[environment].arrow_capacity = projectile_arrow_capacity;
        for (int player = 0; player < 2; ++player) {
            const int index = environment * 2 + player;
            players[index].position = {0.5, 1.0, player == 0 ? -1.0 : 1.0};
            actors[index].slot_count = 1;
            actors[index].slots[0] = WeaponItem::diamond_sword;
        }
    }
    players[0].ordinary_arrow_count = 3;
    const auto initial_players = players;
    const auto sine = makeSineTable();
    BatchView host{players.data(), queued.data(), held.data(), worlds.data(), environments, 2,
                   blocks.data(), {sine.data()}, projectiles.data()};
    CudaBatch device(host);
    const app::ObservationProjection projection{0, 1.0F, 0.0F, -1.0F, 1.0F, false};
    device.configureObservations(&projection, 1, actors.data(), actors.size());
    app::ActionHead head;
    head.kind = 0;
    head.count = 1;
    head.values[0] = 0.0F;
    device.configureActions(&head, 1);
    app::DuelRewardSettings settings;
    settings.max_ticks = 1;
    settings.weights[4] = -2.0F;
    device.configureRewards(settings, states.data(), states.size());
    const std::array<std::uint64_t, environments> episodes{1, 1};
    constexpr std::uint64_t seed = 123;
    device.configureFixedResets(initial_players.data(), actors.data(), episodes.data(), episodes.size(), seed);
    std::int32_t* actions = nullptr;
    std::uint8_t* device_mask = nullptr;
    float* device_observations = nullptr;
    float* device_rewards = nullptr;
    std::uint8_t* device_terminated = nullptr;
    std::uint8_t* device_truncated = nullptr;
    std::uint8_t* device_losses = nullptr;
    double* device_contributions = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&actions), environments * 2 * sizeof(std::int32_t)) != cudaSuccess
        || cudaMemset(actions, 0, environments * 2 * sizeof(std::int32_t)) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&device_mask), environments) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&device_observations), environments * 2 * sizeof(float)) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&device_rewards), environments * 2 * sizeof(float)) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&device_terminated), environments) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&device_truncated), environments) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&device_losses), environments * 2) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&device_contributions), environments * 2 * 18 * sizeof(double)) != cudaSuccess) return false;
    device.registerActionBuffer(actions, environments * 2);
    device.registerConfiguredObservationBuffer(device_observations, environments * 2);
    device.registerRewardBuffers(device_rewards, device_terminated, device_truncated, device_losses,
                                 device_contributions, environments);
    device.registerResetMask(device_mask, environments);
    device.stepRegisteredOnStream(nullptr);
    device.downloadRewardResults(results.data(), results.size());
    bool result = device.actionStatus() == 0 && results[0].truncated && results[1].truncated
        && results[0].rewards[0] == -2.0F;
    device.exportRewardResultsOnStream(device_rewards, device_terminated, device_truncated, device_losses,
                                       device_contributions, environments, nullptr);
    std::array<float, environments * 2> exported_rewards{};
    std::array<std::uint8_t, environments> exported_terminated{}, exported_truncated{};
    std::array<std::uint8_t, environments * 2> exported_losses{};
    std::array<double, environments * 2 * 18> exported_contributions{};
    result = result && cudaDeviceSynchronize() == cudaSuccess
        && cudaMemcpy(exported_rewards.data(), device_rewards, sizeof(exported_rewards), cudaMemcpyDeviceToHost) == cudaSuccess
        && cudaMemcpy(exported_terminated.data(), device_terminated, sizeof(exported_terminated), cudaMemcpyDeviceToHost) == cudaSuccess
        && cudaMemcpy(exported_truncated.data(), device_truncated, sizeof(exported_truncated), cudaMemcpyDeviceToHost) == cudaSuccess
        && cudaMemcpy(exported_losses.data(), device_losses, sizeof(exported_losses), cudaMemcpyDeviceToHost) == cudaSuccess
        && cudaMemcpy(exported_contributions.data(), device_contributions, sizeof(exported_contributions), cudaMemcpyDeviceToHost) == cudaSuccess;
    for (int environment = 0; environment < environments; ++environment) {
        result = result && exported_terminated[environment] == results[environment].terminated
            && exported_truncated[environment] == results[environment].truncated;
        for (int player = 0; player < 2; ++player) {
            const int actor = environment * 2 + player;
            result = result && exported_rewards[actor] == results[environment].rewards[player]
                && exported_losses[actor] == results[environment].lost[player];
            for (int term = 0; term < 18; ++term) {
                result = result && exported_contributions[actor * 18 + term] == results[environment].contributions[player][term];
            }
        }
    }
    device.stepRegisteredOnStream(nullptr);
    result = result && device.actionStatus() == 2;
    device.download(host);
    result = result && worlds[0].tick == 1 && worlds[1].tick == 1;
    device.clearActionStatus();
    std::array<std::uint8_t, environments> mask{1, 2};
    if (cudaMemcpy(device_mask, mask.data(), mask.size(), cudaMemcpyHostToDevice) != cudaSuccess) return false;
    device.resetRegisteredOnStream(nullptr);
    result = result && device.actionStatus() == 4;
    device.download(host);
    result = result && worlds[0].tick == 1 && worlds[1].tick == 1;
    device.clearActionStatus();
    mask = {1, 0};
    if (cudaMemcpy(device_mask, mask.data(), mask.size(), cudaMemcpyHostToDevice) != cudaSuccess) return false;
    device.resetRegisteredOnStream(nullptr);
    device.downloadRewardResults(results.data(), results.size());
    result = result && results[0].truncated && results[1].truncated;
    device.download(host);
    LegacyRandom expected_player_random;
    expected_player_random.setSeed(seed + 1000033ULL);
    LegacyRandom expected_projectile_random;
    expected_projectile_random.setSeed(seed + 1000033ULL);
    result = result && worlds[0].tick == 0 && worlds[1].tick == 1
        && players[0].random.state == expected_player_random.state
        && projectiles[0].random.state == expected_projectile_random.state
        && projectiles[0].bows[0].ordinary_arrows == 3;
    device.stepRegisteredOnStream(nullptr);
    result = result && device.actionStatus() == 2;
    device.download(host);
    result = result && worlds[0].tick == 0 && worlds[1].tick == 1;
    device.clearActionStatus();
    mask = {0, 1};
    if (cudaMemcpy(device_mask, mask.data(), mask.size(), cudaMemcpyHostToDevice) != cudaSuccess) return false;
    device.resetRegisteredOnStream(nullptr);
    device.stepRegisteredOnStream(nullptr);
    result = result && device.actionStatus() == 0;
    device.download(host);
    result = result && worlds[0].tick == 1 && worlds[1].tick == 1;
    mask = {1, 0};
    if (cudaMemcpy(device_mask, mask.data(), mask.size(), cudaMemcpyHostToDevice) != cudaSuccess) return false;
    device.resetRegisteredOnStream(nullptr);
    device.download(host);
    expected_player_random.setSeed(seed + 2 * 1000033ULL);
    result = result && players[0].random.state == expected_player_random.state && worlds[1].tick == 1;
    cudaFree(device_mask);
    cudaFree(device_observations);
    cudaFree(actions);
    cudaFree(device_rewards);
    cudaFree(device_terminated);
    cudaFree(device_truncated);
    cudaFree(device_losses);
    cudaFree(device_contributions);
    return result;
}

bool checkBatchedWorldObservations() {
    using namespace minecraft_sim;
    constexpr int environments = 137;
    constexpr int players_per_environment = 2;
    constexpr int values = environments * players_per_environment * 22;
    std::array<PlayerState, environments * players_per_environment> players{};
    std::array<Controls, environments * players_per_environment> queued{}, held{};
    std::array<WorldState, environments> worlds{};
    std::array<Block, environments> cells{};
    std::array<BlockWorldView, environments> blocks{};
    std::array<float, values> expected{}, actual{};
    for (int environment = 0; environment < environments; ++environment) {
        cells[environment] = environment % 2 == 0 ? Block::solid : Block::air;
        blocks[environment] = {cells.data() + environment, 1, 1, 1, 0, 1, 4};
        for (int player = 0; player < players_per_environment; ++player) {
            auto& actor = players[environment * players_per_environment + player];
            actor.position = {0.5, 0.0, 0.5 + player * 2.0};
            actor.yaw = static_cast<float>(environment + player * 180);
        }
    }
    for (int environment = 0; environment < environments; ++environment) {
        for (int player = 0; player < players_per_environment; ++player) {
            const int index = environment * players_per_environment + player;
            app::baseWorldObservation(players[index], players[index ^ 1], blocks[environment],
                                      expected.data() + index * 22);
        }
    }
    const auto sine = makeSineTable();
    BatchView host{players.data(), queued.data(), held.data(), worlds.data(), environments,
                   players_per_environment, blocks.data(), {sine.data()}};
    CudaBatch device(host);
    float* device_output = nullptr;
    cudaStream_t stream = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(actual)) != cudaSuccess
        || cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return false;
    bool rejected_null = false;
    bool rejected_size = false;
    bool rejected_host = false;
    try { device.observeBaseOnStream(nullptr, values, stream); }
    catch (const std::invalid_argument&) { rejected_null = true; }
    try { device.observeBaseOnStream(device_output, values - 1, stream); }
    catch (const std::invalid_argument&) { rejected_size = true; }
    try { device.observeBaseSynchronized(expected.data(), values, stream); }
    catch (const std::invalid_argument&) { rejected_host = true; }
    device.observeBaseOnStream(device_output, values, stream);
    bool result = cudaMemcpyAsync(actual.data(), device_output, sizeof(actual), cudaMemcpyDeviceToHost, stream)
        == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess;
    if (!result || !rejected_null || !rejected_size || !rejected_host
        || std::abs(actual[10] - 0.07F) > 1.0e-6F || actual[2 * 22 + 10] != 1.0F) return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || std::abs(actual[index] - expected[index]) > 1.0e-6F) {
            std::cerr << "CPU/CUDA batched observation mismatch at index " << index << '\n';
            return false;
        }
    }
    for (std::size_t index = 0; index < queued.size(); ++index) {
        queued[index].forward = index % 2 == 0;
        queued[index].yaw_delta = index % 2 == 0 ? 5.0F : -5.0F;
    }
    device.uploadControls(queued.data(), queued.size());
    cudaGraph_t graph = nullptr;
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return false;
    device.stepOnStream(3, stream);
    device.observeBaseOnStream(device_output, values, stream);
    std::size_t node_count = 0;
    result = cudaStreamEndCapture(stream, &graph) == cudaSuccess
        && cudaGraphGetNodes(graph, nullptr, &node_count) == cudaSuccess && node_count == 4;
    cudaGraphExec_t executable = nullptr;
    result = result && cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0) == cudaSuccess
        && cudaGraphLaunch(executable, stream) == cudaSuccess
        && cudaMemcpyAsync(actual.data(), device_output, sizeof(actual), cudaMemcpyDeviceToHost, stream) == cudaSuccess
        && cudaStreamSynchronize(stream) == cudaSuccess;
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(device_output);
    if (!result) return false;
    for (int tick = 0; tick < 3; ++tick) {
        for (int environment = 0; environment < environments; ++environment) host.world(environment).processTick();
    }
    for (int environment = 0; environment < environments; ++environment) {
        for (int player = 0; player < players_per_environment; ++player) {
            const int index = environment * players_per_environment + player;
            app::baseWorldObservation(players[index], players[index ^ 1], blocks[environment],
                                      expected.data() + index * 22);
        }
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || std::abs(actual[index] - expected[index]) > 1.0e-6F) {
            std::cerr << "CPU/CUDA stepped observation mismatch at index " << index << '\n';
            return false;
        }
    }
    return true;
}

bool checkConfiguredObservations() {
    using namespace minecraft_sim;
    constexpr int environments = 3;
    constexpr int features = 105;
    constexpr int values = environments * 2 * features;
    std::array<PlayerState, environments * 2> players{};
    std::array<Controls, environments * 2> queued{}, held{};
    std::array<WorldState, environments> worlds{};
    std::array<app::ObservationActorMetadata, environments * 2> actors{};
    std::array<app::ObservationProjection, features> projection{};
    const float infinity = std::numeric_limits<float>::infinity();
    for (int feature = 0; feature < 102; ++feature) {
        projection[feature] = {feature, 1.0F, 0.0F, -infinity, infinity, false};
    }
    projection[102] = {0, 2.0F, 0.0F, 0.25F, 0.5F, false};
    projection[103] = {31, 1.0F, 1.0F, -infinity, infinity, true};
    projection[104] = {31, -1.0F, 0.0F, -infinity, infinity, true};
    for (std::size_t index = 0; index < players.size(); ++index) {
        players[index].position = {static_cast<double>(index) * 2.0, 2.0, 0.0};
        players[index].health = index % 2 == 0 ? 15.0F : 8.0F;
        players[index].weapon = index % 2 == 0 ? WeaponItem::iron_axe : WeaponItem::diamond_sword;
        actors[index].slot_count = 2;
        actors[index].slots[0] = players[index].weapon;
        actors[index].slots[1] = WeaponItem::unarmed;
    }
    actors[0].observed_opponent_disable = 2;
    const auto sine = makeSineTable();
    BatchView host{players.data(), queued.data(), held.data(), worlds.data(), environments, 2, nullptr, {sine.data()}};
    CudaBatch device(host);
    device.configureObservations(projection.data(), projection.size(), actors.data(), actors.size());
    float* device_output = nullptr;
    std::array<float, values> actual{};
    if (cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(actual)) != cudaSuccess) return false;
    const auto matches = [&]() {
        device.observeConfiguredOnStream(device_output, values, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess
            || cudaMemcpy(actual.data(), device_output, sizeof(actual), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        for (std::size_t index = 0; index < players.size(); ++index) {
            float catalogue[102]{};
            const auto& source = players[index];
            const auto& target = players[index ^ 1U];
            const auto& metadata = actors[index];
            app::baseWorldObservation(source, target, {}, catalogue);
            app::legacyEquipmentObservation(catalogue, source, target, metadata.observed_opponent_disable);
            catalogue[31] = static_cast<float>(metadata.selected_slot);
            catalogue[32] = static_cast<float>(actors[index ^ 1U].selected_slot);
            app::velocityObservation(catalogue, source, target);
            app::movementObservation(catalogue, source, target);
            app::equipmentObservation(catalogue, source, target, metadata.slots, metadata.slot_count);
            for (int feature = 0; feature < features; ++feature) {
                const float expected = app::projectObservation(catalogue, projection[feature]);
                if (!std::isfinite(actual[index * features + feature])
                    || std::abs(actual[index * features + feature] - expected) > 1.0e-6F) return false;
            }
        }
        return true;
    };
    bool result = matches() && actual[102] == 1.0F && actual[103] == 0.0F && actual[104] == -1.0F;
    queued[0].select_weapon = true;
    queued[0].selected_slot = 1;
    queued[0].weapon = WeaponItem::unarmed;
    device.uploadControls(queued.data(), queued.size());
    device.step();
    device.download(host);
    actors[0].selected_slot = 1;
    actors[0].observed_opponent_disable = 1;
    result = result && matches() && actual[103] == 1.0F && actual[104] == 0.0F;
    const int reset_id = 1;
    std::array<PlayerState, 2> reset_players{players[2], players[3]};
    std::array<app::ObservationActorMetadata, 2> reset_actors{actors[2], actors[3]};
    reset_actors[0].selected_slot = 1;
    device.resetEnvironments(&reset_id, 1, reset_players.data(), nullptr, reset_actors.data());
    actors[2].selected_slot = 1;
    device.download(host);
    result = result && matches() && actual[2 * features + 103] == 1.0F
        && actual[103] == 1.0F && actual[27] == 1.0F;
    cudaFree(device_output);
    return result;
}

bool sameControls(const minecraft_sim::Controls& left, const minecraft_sim::Controls& right) {
    return left.forward == right.forward && left.backward == right.backward && left.left == right.left
        && left.right == right.right && left.jump == right.jump && left.yaw_delta == right.yaw_delta
        && left.pitch_delta == right.pitch_delta && left.sprint == right.sprint && left.sneak == right.sneak
        && left.attack == right.attack && left.use_item == right.use_item
        && left.select_weapon == right.select_weapon && left.weapon == right.weapon
        && left.selected_slot == right.selected_slot;
}

bool checkBatchedDeviceActions() {
    using namespace minecraft_sim;
    constexpr int environments = 137;
    constexpr int head_count = 9;
    constexpr int actor_count = environments * 2;
    std::array<PlayerState, actor_count> players{};
    std::array<Controls, actor_count> queued{}, held{}, expected{};
    std::array<WorldState, environments> worlds{};
    std::array<app::ObservationActorMetadata, actor_count> actors{};
    std::array<app::ActionHead, head_count> heads{};
    std::array<std::int32_t, actor_count * head_count> actions{};
    const int kinds[] = {0, 4, 6, 1, 2, 3, 7, 8, 5};
    for (int head = 0; head < head_count; ++head) {
        heads[head].kind = kinds[head];
        heads[head].count = kinds[head] == 0 ? 9 : (kinds[head] == 4 || kinds[head] == 5 ? 3 : 2);
        for (int choice = 0; choice < heads[head].count; ++choice) {
            heads[head].values[choice] = kinds[head] == 4 || kinds[head] == 5
                ? static_cast<float>((choice - 1) * 5) : static_cast<float>(choice);
        }
    }
    for (int actor = 0; actor < actor_count; ++actor) {
        players[actor].position = {0.0, 3.0, actor % 2 == 0 ? 0.0 : 2.0};
        actors[actor].slot_count = actor == actor_count - 1 ? 1 : 2;
        actors[actor].slots[0] = actor % 2 == 0 ? WeaponItem::iron_sword : WeaponItem::diamond_axe;
        actors[actor].slots[1] = WeaponItem::unarmed;
        for (int head = 0; head < head_count; ++head) {
            actions[actor * head_count + head] = (actor + head) % heads[head].count;
        }
        if (actor == actor_count - 1) actions[actor * head_count + 2] = 0;
        if (!app::decodeAction(heads.data(), head_count, actions.data() + actor * head_count,
                               actors[actor].slots, actors[actor].slot_count, expected[actor])) return false;
    }
    const auto sine = makeSineTable();
    BatchView host{players.data(), queued.data(), held.data(), worlds.data(), environments, 2, nullptr, {sine.data()}};
    CudaBatch device(host);
    const app::ObservationProjection projection{0, 1.0F, 0.0F, -1.0F, 1.0F, false};
    device.configureObservations(&projection, 1, actors.data(), actors.size());
    auto invalid_heads = heads;
    invalid_heads[0].values[0] = std::numeric_limits<float>::quiet_NaN();
    bool rejected_layout = false;
    try { device.configureActions(invalid_heads.data(), invalid_heads.size()); }
    catch (const std::invalid_argument&) { rejected_layout = true; }
    if (!rejected_layout) return false;
    device.configureActions(heads.data(), heads.size());
    std::int32_t* device_actions = nullptr;
    cudaStream_t stream = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_actions), sizeof(actions)) != cudaSuccess
        || cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return false;
    const auto upload = [&]() {
        return cudaMemcpy(device_actions, actions.data(), sizeof(actions), cudaMemcpyHostToDevice) == cudaSuccess;
    };
    bool result = upload();
    device.decodeActionsOnStream(device_actions, actions.size(), stream);
    result = result && device.actionStatus() == 0;
    device.download(host);
    for (int actor = 0; actor < actor_count; ++actor) result = result && sameControls(queued[actor], expected[actor]);

    actions[0] = -1;
    result = result && upload();
    device.decodeActionsOnStream(device_actions, actions.size(), stream);
    device.stepDecodedOnStream(1, stream);
    result = result && device.actionStatus() == 1;
    device.download(host);
    for (int actor = 0; actor < actor_count; ++actor) result = result && sameControls(queued[actor], expected[actor]);
    for (const auto& world : worlds) result = result && world.tick == 0;
    actions[0] = 0;
    result = result && upload();
    device.decodeActionsOnStream(device_actions, actions.size(), stream);
    device.stepDecodedOnStream(1, stream);
    result = result && device.actionStatus() == 1;
    device.download(host);
    for (const auto& world : worlds) result = result && world.tick == 0;

    device.clearActionStatus();
    actions[(actor_count - 1) * head_count + 2] = 1;
    result = result && upload();
    device.decodeActionsOnStream(device_actions, actions.size(), stream);
    device.stepDecodedOnStream(1, stream);
    result = result && device.actionStatus() == 1;
    device.download(host);
    for (const auto& world : worlds) result = result && world.tick == 0;
    for (int actor = 0; actor < actor_count; ++actor) result = result && sameControls(queued[actor], expected[actor]);

    device.clearActionStatus();
    actions[(actor_count - 1) * head_count + 2] = 0;
    auto reference_players = players;
    auto reference_queued = expected;
    auto reference_held = held;
    auto reference_worlds = worlds;
    BatchView reference{reference_players.data(), reference_queued.data(), reference_held.data(),
                        reference_worlds.data(), environments, 2, nullptr, {sine.data()}};
    result = result && upload();
    device.decodeActionsOnStream(device_actions, actions.size(), stream);
    device.stepDecodedOnStream(1, stream);
    result = result && device.actionStatus() == 0;
    device.download(host);
    for (int environment = 0; environment < environments; ++environment) {
        reference.world(environment).processTick();
        result = result && worlds[environment].tick == reference_worlds[environment].tick;
    }
    for (int actor = 0; actor < actor_count; ++actor) {
        result = result && std::abs(players[actor].position.x - reference_players[actor].position.x) < 1.0e-12
            && std::abs(players[actor].position.y - reference_players[actor].position.y) < 1.0e-12
            && std::abs(players[actor].position.z - reference_players[actor].position.z) < 1.0e-12
            && players[actor].yaw == reference_players[actor].yaw
            && sameControls(held[actor], reference_held[actor]);
    }
    cudaStreamDestroy(stream);
    cudaFree(device_actions);
    return result;
}

}

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
    if (!checkMutableTerrain() || !checkStaticTerrainSharing()) {
        std::cerr << "CUDA mutable terrain isolation check failed\n";
        return 1;
    }
    if (!checkFixedTerrainReset()) {
        std::cerr << "CUDA fixed terrain reset check failed\n";
        return 1;
    }
    std::array<float, feature_count * scenario_count> expected{};
    std::array<float, feature_count * scenario_count> actual{};
    encodeScenarios(expected.data());
    float* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(actual)) != cudaSuccess) return 1;
    encodeScenariosOnDevice<<<1, 1>>>(device_output);
    const auto result = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess
        ? cudaMemcpy(actual.data(), device_output, sizeof(actual), cudaMemcpyDeviceToHost)
        : cudaErrorUnknown;
    cudaFree(device_output);
    if (result != cudaSuccess) {
        std::cerr << "CUDA observation execution failed\n";
        return 1;
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || std::abs(actual[index] - expected[index]) > 1.0e-6F) {
            std::cerr << "CPU/CUDA observation mismatch at index " << index << '\n';
            return 1;
        }
    }
    if (actual[1] != -0.75F || actual[33] != 0.0F || actual[34] != 1.0F
        || actual[43] != 1.0F || actual[47] != 1.0F || actual[51] != 0.0F
        || actual[79] != 0.25F || actual[82] != -0.75F || actual[85] != 1.0F
        || actual[86] != 0.0F || actual[87] != 1.0F || actual[88] != 0.0F
        || actual[89] != 0.5F || actual[95] != 1.0F
        || actual[feature_count + 33] != 1.0F || actual[feature_count + 34] != 0.0F
        || actual[2 * feature_count + 19] != 0.0F || actual[2 * feature_count + 20] != 0.0F
        || actual[2 * feature_count + 21] != 0.0F || actual[2 * feature_count + 43] != 0.0F) {
        std::cerr << "CUDA observation fixture failed\n";
        return 1;
    }
    for (int scenario = 1; scenario < scenario_count; ++scenario) {
        for (int index = 95; index < feature_count; ++index) {
            if (actual[scenario * feature_count + index] != 0.0F) {
                std::cerr << "Missing arrow observation was not cleared\n";
                return 1;
            }
        }
    }
    if (!checkBatchedWorldObservations()) {
        std::cerr << "CUDA batched world observation check failed\n";
        return 1;
    }
    if (!checkShapingRewardParity()) {
        std::cerr << "CUDA shaping reward check failed\n";
        return 1;
    }
    if (!checkBatchedDuelRewardParity()) {
        std::cerr << "CUDA batched duel reward check failed\n";
        return 1;
    }
    if (!checkFinishedWorldProtection()) {
        std::cerr << "CUDA finished-world protection check failed\n";
        return 1;
    }
    if (!checkConfiguredObservations()) {
        std::cerr << "CUDA configured observation check failed\n";
        return 1;
    }
    if (!checkBatchedDeviceActions()) {
        std::cerr << "CUDA batched action check failed\n";
        return 1;
    }
    return 0;
}

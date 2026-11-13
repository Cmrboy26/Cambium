#include "../apps/observation.hpp"
#include "minecraft_sim/cuda_backend.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

constexpr int feature_count = 102;
constexpr int scenario_count = 3;

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

}

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
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
    return 0;
}

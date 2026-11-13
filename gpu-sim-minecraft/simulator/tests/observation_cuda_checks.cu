#include "../apps/observation.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <iostream>

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
    return 0;
}

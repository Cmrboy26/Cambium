#include "minecraft_sim/projectiles.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <iostream>

// Source: Projectile.shootFromRotation (Projectile.java:183-194), AbstractArrow.tick (AbstractArrow.java:173-266).
// Identical host/device inputs exercise ordered spread, inherited movement, air flight and charge boundaries.
MCSIM_HD void runProjectileScenario(const float* sine_table, double* output) {
    minecraft_sim::PlayerState players[2]{};
    minecraft_sim::Controls queued[2]{};
    minecraft_sim::Controls held[2]{};
    minecraft_sim::WorldState world_state{};
    minecraft_sim::BlockWorldView blocks{};
    minecraft_sim::MathView math{sine_table};
    minecraft_sim::WorldView world(players, queued, held, &world_state, 2, blocks, math);
    minecraft_sim::ArrowState arrows[1]{};
    minecraft_sim::ProjectileWorldState state;
    state.arrows = arrows;
    state.arrow_capacity = 1;
    arrows[0].active = true;
    arrows[0].left_owner = true;
    arrows[0].position = {20.0, 20.0, 20.0};
    arrows[0].velocity = {1.0, 0.0, 0.0};
    minecraft_sim::tickProjectiles(world, blocks, state);
    output[0] = arrows[0].position.x;
    output[1] = arrows[0].velocity.x;
    output[2] = minecraft_sim::bowPower(3);
    players[0].yaw = -90.0F;
    players[0].pitch = 15.0F;
    players[0].velocity = {0.25, 0.1, -0.5};
    minecraft_sim::LegacyRandom random;
    random.setSeed(77);
    const auto launched = minecraft_sim::launchArrow(players[0], 0, 1.0F, math, random);
    output[3] = launched.velocity.x;
    output[4] = launched.velocity.y;
    output[5] = launched.velocity.z;
    output[6] = static_cast<double>(random.state);
}

// Simulator test launch wrapper; source-derived scenario is shared with the host invocation above.
__global__ void runProjectileHeaderCheck(const float* sine_table, double* output) {
    runProjectileScenario(sine_table, output);
}

// Simulator verification harness: compare identical inputs rather than treating shared code as proof of agreement.
int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        return 77;
    }
    const auto sine_table = minecraft_sim::makeSineTable();
    std::array<double, 7> expected{};
    runProjectileScenario(sine_table.data(), expected.data());
    double* device_output = nullptr;
    float* device_sines = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(expected)) != cudaSuccess) {
        return 1;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&device_sines), sizeof(sine_table)) != cudaSuccess) {
        cudaFree(device_output);
        return 1;
    }
    if (cudaMemcpy(device_sines, sine_table.data(), sizeof(sine_table), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(device_sines);
        cudaFree(device_output);
        return 1;
    }
    runProjectileHeaderCheck<<<1, 1>>>(device_sines, device_output);
    std::array<double, 7> output{};
    const cudaError_t result = cudaDeviceSynchronize() == cudaSuccess
        ? cudaMemcpy(output.data(), device_output, sizeof(output), cudaMemcpyDeviceToHost)
        : cudaErrorUnknown;
    cudaFree(device_sines);
    cudaFree(device_output);
    if (result != cudaSuccess || output[0] != 21.0 || output[1] != double(0.99F)
        || output[2] < 0.1074 || output[2] > 0.1076) {
        std::cerr << "CUDA projectile header check failed\n";
        return 1;
    }
    for (std::size_t index = 0; index < output.size(); ++index) {
        if (!std::isfinite(output[index]) || std::abs(output[index] - expected[index]) > 1.0e-12) {
            std::cerr << "CPU/CUDA projectile mismatch at output " << index << '\n';
            return 1;
        }
    }
    return 0;
}

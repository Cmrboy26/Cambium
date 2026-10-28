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

    players[0] = {};
    players[1] = {};
    auto attacker = world.player(0);
    attacker.equipWeapon(minecraft_sim::WeaponItem::mace);
    attacker.state().fall_distance = 2.0;
    attacker.state().on_ground = false;
    attacker.state().sprinting = true;
    attacker.state().attack_strength_ticks = 40;
    attacker.resolveMelee(world.player(1), math);
    output[7] = attacker.state().last_attack.incoming_damage;
    output[8] = attacker.state().velocity.y;
    output[9] = attacker.state().fall_distance;
    output[10] = attacker.state().impulse_grace_ticks;

    players[0] = {};
    players[1] = {};
    players[0].position = {0.0, 0.0, 1.0};
    players[1].position = {0.0, 0.0, 3.0};
    minecraft_sim::applyWindExplosion(world, blocks, {});
    output[11] = players[0].velocity.y;
    output[12] = players[0].velocity.z;
    output[13] = players[1].velocity.lengthSquared();
    random.setSeed(42);
    players[0].position = {2.0, 3.0, 4.0};
    players[0].velocity = {0.25, 0.5, -0.75};
    players[0].on_ground = false;
    const auto charge = minecraft_sim::launchWindCharge(players[0], 0, math, random);
    output[14] = charge.velocity.x;
    output[15] = charge.velocity.y;
    output[16] = charge.velocity.z;
    minecraft_sim::Block block = minecraft_sim::Block::solid;
    minecraft_sim::BlockWorldView cube{&block, 1, 1, 1, 2, 2, 2};
    const auto face = minecraft_sim::fullCubeProjectileHit(cube, {0.0, 2.5, 2.5}, {4.0, 0.0, 0.0});
    const auto inside = minecraft_sim::fullCubeProjectileHit(cube, {2.5, 2.5, 2.5}, {1.0, 1.0, 0.0});
    output[17] = face.fraction;
    output[18] = face.normal.x;
    output[19] = inside.fraction;
    output[20] = inside.normal.y;
    players[0] = {};
    players[1] = {};
    players[0].position = {0.0, 0.0, 0.0};
    players[1].position = {10.0, 0.0, 0.0};
    minecraft_sim::WindChargeState moving;
    moving.active = true;
    moving.owner = 0;
    moving.position = {0.0, double(1.62F), 0.0};
    moving.velocity = {2.0, 0.0, 0.0};
    minecraft_sim::tickWindCharge(world, blocks, moving);
    output[21] = moving.position.x;
    output[22] = moving.left_owner ? 1.0 : 0.0;
    moving.velocity.x = -2.0;
    const auto impact = minecraft_sim::tickWindCharge(world, blocks, moving);
    output[23] = players[0].health;
    output[24] = impact.player_hits;

    players[0] = {};
    players[1] = {};
    players[1].position = {10.0, 0.0, 0.0};
    players[0].armor = 20.0F;
    players[0].fall_distance = 8.0;
    players[0].velocity = {0.5, -0.5, 0.0};
    minecraft_sim::ThrownProjectileState pearl;
    pearl.active = true;
    pearl.item = minecraft_sim::ThrownItem::ender_pearl;
    pearl.owner = 0;
    pearl.position = {1.0, 2.5, 2.5};
    pearl.velocity = {2.0, 0.0, 0.0};
    const auto pearl_impact = minecraft_sim::tickThrownProjectile(world, cube, pearl);
    output[25] = players[0].position.x;
    output[26] = players[0].position.y;
    output[27] = players[0].health;
    output[28] = players[0].velocity.lengthSquared();
    output[29] = pearl_impact.health_taken[0];
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
    std::array<double, 30> expected{};
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
    std::array<double, 30> output{};
    const cudaError_t result = cudaDeviceSynchronize() == cudaSuccess
        ? cudaMemcpy(output.data(), device_output, sizeof(output), cudaMemcpyDeviceToHost)
        : cudaErrorUnknown;
    cudaFree(device_sines);
    cudaFree(device_output);
    if (result != cudaSuccess || output[0] != 21.0 || output[1] != double(0.99F)
        || output[2] < 0.1074 || output[2] > 0.1076 || output[7] != 14.0
        || output[8] != double(0.01F) || output[9] != 0.0 || output[10] != 40.0
        || output[13] != 0.0 || output[17] != 0.5 || output[18] != -1.0
        || output[19] != 0.001 || output[20] != -1.0 || output[21] != 2.0
        || output[22] != 1.0 || output[23] != 19.0 || output[24] != 1.0
        || output[25] != 1.0 || output[26] != 2.5 || output[27] != 15.0
        || output[28] != 0.0 || output[29] != 5.0) {
        std::cerr << "CUDA projectile header check failed\n";
        return 1;
    }
    const double strength = (1.0 - 1.0 / double(1.2F * 2.0F)) * double(1.22F);
    const double eye_height = double(1.62F);
    const double length = std::sqrt(1.0 + eye_height * eye_height);
    if (std::abs(output[11] - strength * eye_height / length) > 1.0e-12
        || std::abs(output[12] - strength / length) > 1.0e-12
        || std::abs(output[14] - (0.25 + 1.5 * 0.0172275 * (0.7275636800328681 - 0.6832234717598454))) > 1.0e-12
        || std::abs(output[15] - (0.5 + 1.5 * 0.0172275 * (0.30871945533265976 - 0.27707849007413665))) > 1.0e-12
        || std::abs(output[16] - (-0.75 + 1.5 * (1.0 + 0.0172275 * (0.6655489517945736 - 0.9033722646721782)))) > 1.0e-12) {
        std::cerr << "CUDA mace or wind source-derived fixture failed\n";
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

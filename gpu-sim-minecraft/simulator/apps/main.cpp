#include "minecraft_sim/build_info.hpp"
#include "minecraft_sim/core.hpp"
#include "scene.hpp"
#include "replay.hpp"

#if defined(MCSIM_HAS_CUDA)
#include "minecraft_sim/cuda_backend.hpp"
#endif

#include <array>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>

#include <nlohmann/json.hpp>

namespace {

using minecraft_sim::app::Scene;

minecraft_sim::Controls scenarioInputs(int tick) {
    minecraft_sim::Controls inputs;
    inputs.forward = tick >= 2 && tick < 35;
    inputs.jump = tick == 10;
    inputs.yaw_delta = tick == 20 ? 15.0F : 0.0F;
    return inputs;
}

int runDuel(int ticks, bool gpu, const std::filesystem::path& replay_path,
            const std::filesystem::path& environment_path) {
    auto terrain = std::make_unique<Scene>(environment_path);
    auto players = terrain->initial_players;
    std::array<minecraft_sim::Controls, 2> queued{};
    std::array<minecraft_sim::Controls, 2> held{};
    minecraft_sim::BatchView batch{players.data(), queued.data(), held.data(), terrain->worlds.data(),
                                   1, 2, &terrain->blocks, {terrain->sine.data()}};
    std::ofstream replay;
    if (!replay_path.empty()) {
        if (std::filesystem::exists(replay_path)) {
            throw std::invalid_argument("Replay output already exists; choose a new path");
        }
        replay.exceptions(std::ios::failbit | std::ios::badbit);
        replay.open(replay_path);
        replay << nlohmann::json{{"format", "minecraft_sim_replay"}, {"version", 3},
            {"minecraft", minecraft_sim::target_minecraft_version()}, {"backend", gpu ? "CUDA" : "CPU"},
            {"players", 2}, {"ticks_per_second", 20}, {"environment", terrain->resolvedDefinition()}}.dump() << '\n';
    }
    const auto write_frame = [&](const std::array<minecraft_sim::Controls, 2>& actions) {
        if (!replay.is_open()) {
            return;
        }
        replay << minecraft_sim::app::replayFrame(terrain->worlds[0].tick, players.data(), actions.data(), true).dump() << '\n';
        replay.flush();
    };
    write_frame({});
#if defined(MCSIM_HAS_CUDA)
    std::unique_ptr<minecraft_sim::CudaBatch> device;
    if (gpu) {
        if (!minecraft_sim::CudaBatch::available()) {
            throw std::runtime_error("No usable CUDA device");
        }
        device = std::make_unique<minecraft_sim::CudaBatch>(batch);
    }
#endif
    std::cout << "Minecraft Java " << minecraft_sim::target_minecraft_version()
              << " synthetic configured-loadout duel (" << (gpu ? "CUDA" : "CPU") << ")\n"
              << "Not a captured Minecraft fight; both players approach melee range and press attack every 13 ticks.\n"
              << "tick,player,x,y,z,vx,vy,vz,health,alive,target,attempted,accepted,charge,damage,health_lost,critical,sprint_bonus,damage_cooldown\n"
              << std::setprecision(17);
    for (int tick = 0; tick < ticks; ++tick) {
        std::array<minecraft_sim::Controls, 2> actions{};
        for (int player = 0; player < 2; ++player) {
            minecraft_sim::Controls inputs;
            inputs.forward = batch.world(0).pickAttackTarget(player) != 1 - player;
            inputs.attack = tick % 13 == 0;
            actions[player] = inputs;
            batch.world(0).player(player).setInputs(inputs);
        }
#if defined(MCSIM_HAS_CUDA)
        if (device) {
            device->uploadControls(queued.data(), queued.size());
            device->step();
            device->download(batch);
        } else
#endif
        {
            batch.world(0).processTick();
        }
        write_frame(actions);
        for (int player = 0; player < 2; ++player) {
            const auto& state = players[player];
            const auto& attack = state.last_attack;
            std::cout << terrain->worlds[0].tick << ',' << player << ',' << state.position.x << ',' << state.position.y
                      << ',' << state.position.z << ',' << state.velocity.x << ',' << state.velocity.y << ',' << state.velocity.z
                      << ',' << state.health << ',' << state.alive() << ',' << attack.target << ',' << attack.attempted
                      << ',' << attack.accepted << ',' << attack.charge << ',' << attack.incoming_damage << ','
                      << attack.health_lost << ',' << attack.critical << ',' << attack.sprint_bonus << ',' << state.damage_cooldown << '\n';
        }
        if (!players[0].alive() || !players[1].alive()) {
            std::cout << "Duel ended: player " << (players[0].alive() ? 1 : 0) << " died\n";
            return 0;
        }
    }
    std::cout << "Duel trace reached requested tick limit\n";
    return 0;
}

}

int main(int argc, char** argv) {
    try {
        int ticks = 40;
        bool gpu = false;
        bool duel = false;
        std::filesystem::path replay_path;
        std::filesystem::path environment_path;
        for (int argument = 1; argument < argc; ++argument) {
            const std::string_view option = argv[argument];
            if (option == "--help") {
                std::cout << "Usage: minecraft_sim_runner [--ticks 1..100000] [--cuda] [--duel] [--environment FILE.json] [--replay [FILE.jsonl]]\n"
                          << "Replay export is optional. --replay without a path creates duel-replay-<timestamp>.jsonl.\n";
                return 0;
            }
            if (option == "--cuda") {
                gpu = true;
            } else if (option == "--duel") {
                duel = true;
            } else if (option == "--environment" && argument + 1 < argc) {
                environment_path = argv[++argument];
            } else if (option == "--replay") {
                if (argument + 1 < argc && !std::string_view(argv[argument + 1]).starts_with("--")) {
                    replay_path = argv[++argument];
                } else {
                    const auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
                    replay_path = "duel-replay-" + std::to_string(timestamp) + ".jsonl";
                }
            } else if (option == "--ticks" && argument + 1 < argc) {
                const std::string_view count = argv[++argument];
                const auto result = std::from_chars(count.data(), count.data() + count.size(), ticks);
                if (result.ec != std::errc{} || result.ptr != count.data() + count.size()
                    || ticks < 1 || ticks > 100000) {
                    throw std::invalid_argument("--ticks must be in 1..100000");
                }
            } else {
                throw std::invalid_argument("Unknown or incomplete option; use --help");
            }
        }
        if (!replay_path.empty() && !duel) {
            throw std::invalid_argument("--replay currently requires --duel");
        }
#if !defined(MCSIM_HAS_CUDA)
        if (gpu) {
            throw std::runtime_error("CUDA backend not built; configure with MCSIM_ENABLE_CUDA=ON");
        }
#endif
    if (duel) {
        if (!replay_path.empty()) {
            std::cout << "Replay output: " << replay_path.string() << '\n';
        }
        return runDuel(ticks, gpu, replay_path, environment_path);
    }
        auto scene = std::make_unique<Scene>(environment_path);
#if defined(MCSIM_HAS_CUDA)
        std::unique_ptr<minecraft_sim::CudaBatch> device;
        if (gpu) {
            if (!minecraft_sim::CudaBatch::available()) {
                throw std::runtime_error("No usable CUDA device");
            }
            device = std::make_unique<minecraft_sim::CudaBatch>(scene->view());
        }
#endif
        std::cout << "Minecraft shared movement core\n"
                  << "Target: Minecraft Java " << minecraft_sim::target_minecraft_version() << '\n'
                  << "Backend: " << (gpu ? "CUDA" : "CPU") << '\n'
                  << "Scenario: synthetic full-cube movement replay; not a Minecraft capture\n"
                  << "tick,x,y,z,vx,vy,vz,yaw,on_ground,horizontal_collision,vertical_collision\n"
                  << std::setprecision(17);
        for (int tick = 0; tick < ticks; ++tick) {
            const auto inputs = scenarioInputs(tick);
#if defined(MCSIM_HAS_CUDA)
            if (device) {
                device->uploadControls(&inputs, 1);
                device->step();
                device->download(scene->view());
            } else
#endif
            {
                auto world = scene->view().world(0);
                world.player(0).setInputs(inputs);
                world.processTick();
            }
            const auto& state = scene->players[0];
            std::cout << scene->worlds[0].tick << ','
                      << state.position.x << ',' << state.position.y << ',' << state.position.z << ','
                      << state.velocity.x << ',' << state.velocity.y << ',' << state.velocity.z << ','
                      << state.yaw << ',' << state.on_ground << ','
                      << state.horizontal_collision << ',' << state.vertical_collision << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
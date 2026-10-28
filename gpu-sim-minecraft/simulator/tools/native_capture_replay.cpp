#include "minecraft_sim/cuda_backend.hpp"
#include "minecraft_sim/projectiles.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(MCSIM_HAS_CUDA)
#include <cuda_runtime.h>
#endif

using namespace minecraft_sim;

template <typename Value>
std::vector<Value> readValues(std::istream& input, std::size_t count) {
    std::vector<Value> values(count);
    if (count) input.read(reinterpret_cast<char*>(values.data()), count * sizeof(Value));
    if (!input) throw std::runtime_error("Truncated capture");
    return values;
}

template <typename Value>
void checkSize(const nlohmann::json& metadata, const char* name) {
    if (metadata.at(name).get<std::size_t>() != sizeof(Value)) {
        throw std::runtime_error(std::string("Capture ABI mismatch: ") + name);
    }
}

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::invalid_argument("Usage: native_capture_replay FILE cpu|cuda|cuda-observe START COUNT");
        const std::string mode = argv[2];
        if (mode != "cpu" && mode != "cuda" && mode != "cuda-observe") throw std::invalid_argument("Invalid replay mode");
        const auto start = std::stoll(argv[3]);
        const auto count = std::stoll(argv[4]);
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) throw std::runtime_error("Cannot open capture");
        const auto header_size = readValues<std::uint64_t>(input, 1)[0];
        if (!header_size || header_size > 4 * 1024 * 1024) throw std::runtime_error("Invalid header size");
        const auto header = readValues<char>(input, header_size);
        const auto metadata = nlohmann::json::parse(header.begin(), header.end());
        if (metadata.at("format") != "mcsim_native_capture" || metadata.at("version") != 1
            || metadata.at("players_per_world") != 2) throw std::runtime_error("Unsupported capture format");
        checkSize<PlayerState>(metadata, "player_size");
        checkSize<Controls>(metadata, "controls_size");
        checkSize<WorldState>(metadata, "world_size");
        checkSize<ProjectileWorldState>(metadata, "projectile_world_size");
        checkSize<ArrowState>(metadata, "arrow_size");
        checkSize<WindChargeState>(metadata, "wind_size");
        checkSize<ThrownProjectileState>(metadata, "thrown_size");
        checkSize<Block>(metadata, "block_size");
        const auto world_count = metadata.at("worlds").get<std::size_t>();
        const auto arrows_per_world = metadata.at("arrow_capacity").get<std::size_t>();
        const auto wind_per_world = metadata.at("wind_capacity").get<std::size_t>();
        const auto thrown_per_world = metadata.at("thrown_capacity").get<std::size_t>();
        const auto cell_count = metadata.at("cell_count").get<std::size_t>();
        const auto sine_count = metadata.at("sine_count").get<std::size_t>();
        if (!world_count || world_count > 65536 || arrows_per_world != projectile_arrow_capacity
            || wind_per_world > 128 || thrown_per_world > 96 || cell_count > 32768
            || sine_count != sine_table_size || start < 0 || count < 1
            || static_cast<std::uint64_t>(start) >= world_count
            || static_cast<std::uint64_t>(count) > world_count - start) {
            throw std::invalid_argument("Invalid capture dimensions or world range");
        }
        auto players = readValues<PlayerState>(input, world_count * 2);
        auto queued = readValues<Controls>(input, world_count * 2);
        auto held = readValues<Controls>(input, world_count * 2);
        auto worlds = readValues<WorldState>(input, world_count);
        auto projectile_states = readValues<ProjectileWorldState>(input, world_count);
        auto arrows = readValues<ArrowState>(input, world_count * arrows_per_world);
        auto wind = readValues<WindChargeState>(input, world_count * wind_per_world);
        auto thrown = readValues<ThrownProjectileState>(input, world_count * thrown_per_world);
        auto cells = readValues<Block>(input, cell_count);
        auto sine = readValues<float>(input, sine_count);
        if (input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Unexpected capture data");
        for (std::size_t index = 0; index < world_count; ++index) {
            auto& state = projectile_states[index];
            if (state.arrow_capacity != static_cast<int>(arrows_per_world)
                || state.wind_charge_capacity != static_cast<int>(wind_per_world)
                || state.thrown_item_capacity != static_cast<int>(thrown_per_world)) {
                throw std::runtime_error("Projectile pool shape mismatch");
            }
            state.arrows = arrows.data() + index * arrows_per_world;
            state.wind_charges = wind_per_world ? wind.data() + index * wind_per_world : nullptr;
            state.thrown_items = thrown_per_world ? thrown.data() + index * thrown_per_world : nullptr;
        }
        const auto& bounds = metadata.at("terrain").at("bounds");
        const auto size = bounds.at("size").get<std::array<int, 3>>();
        const auto origin = bounds.at("min").get<std::array<int, 3>>();
        if (size[0] < 1 || size[1] < 1 || size[2] < 1
            || std::size_t(size[0]) * size[1] * size[2] != cell_count) {
            throw std::runtime_error("Terrain dimensions do not match capture");
        }
        const BlockWorldView block{cells.data(), size[0], size[1], size[2], origin[0], origin[1], origin[2]};
        std::vector<BlockWorldView> blocks(static_cast<std::size_t>(count), block);
        BatchView batch{players.data() + start * 2, queued.data() + start * 2, held.data() + start * 2,
                        worlds.data() + start, static_cast<int>(count), 2, blocks.data(), {sine.data()},
                        projectile_states.data() + start};
        std::cout << "CAPTURE step=" << metadata.at("step") << " source=" << metadata.at("source_hash")
                  << " start=" << start << " count=" << count << std::endl;
        if (mode == "cpu") {
            for (int index = 0; index < batch.environment_count; ++index) {
                std::cout << "BEGIN world=" << start + index << std::endl;
                const std::array<std::uint64_t, 2> random_before{
                    batch.players[index * 2].random.state, batch.players[index * 2 + 1].random.state};
                processWorldTick(batch.world(index), blocks[index], batch.projectile_states[index]);
                const auto& player = batch.players[index * 2];
                std::cout << std::setprecision(17) << "DONE world=" << start + index << " player0="
                          << player.position.x << ',' << player.position.y << ',' << player.position.z << std::endl;
                for (int actor = 0; actor < 2; ++actor) {
                    const auto after = batch.players[index * 2 + actor].random.state;
                    if (random_before[actor] != after) {
                        std::cout << "KNOCKBACK_RNG world=" << start + index << " actor=" << actor
                                  << " before=" << random_before[actor] << " after=" << after << std::endl;
                    }
                }
            }
        } else {
#if defined(MCSIM_HAS_CUDA)
            CudaBatch gpu(batch, true);
            cudaStream_t observer = nullptr;
            WorldState* observed = nullptr;
            PlayerState* observed_players = nullptr;
            if (mode == "cuda-observe") {
                if (cudaHostAlloc(reinterpret_cast<void**>(&observed), count * sizeof(*observed), 0) != cudaSuccess
                    || cudaHostAlloc(reinterpret_cast<void**>(&observed_players), count * 2 * sizeof(*observed_players), 0) != cudaSuccess
                    || cudaStreamCreateWithFlags(&observer, cudaStreamNonBlocking) != cudaSuccess) {
                    throw std::runtime_error("Cannot create CUDA observer");
                }
            }
            std::cout << "CUDA_LAUNCH" << std::endl;
            gpu.step();
            if (observer) {
                std::vector<std::uint64_t> previous(static_cast<std::size_t>(count));
                for (int sample = 0; sample < 2; ++sample) {
                    std::this_thread::sleep_for(std::chrono::seconds(sample == 0 ? 2 : 1));
                    const auto result = cudaMemcpyAsync(observed, gpu.deviceView().worlds, count * sizeof(*observed),
                                                        cudaMemcpyDeviceToHost, observer);
                    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
                    const auto player_result = cudaMemcpyAsync(observed_players, gpu.deviceView().players,
                        count * 2 * sizeof(*observed_players), cudaMemcpyDeviceToHost, observer);
                    if (player_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(player_result));
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
                    auto status = cudaStreamQuery(observer);
                    while (status == cudaErrorNotReady && std::chrono::steady_clock::now() < deadline) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        status = cudaStreamQuery(observer);
                    }
                    if (status == cudaSuccess) {
                        int unexpected = 0;
                        for (int index = 0; index < count; ++index) {
                            const auto tick = observed[index].tick;
                            if (tick != batch.worlds[index].tick + 1 || (sample && tick != previous[index])) {
                                ++unexpected;
                                std::cout << "WORLD_SAMPLE sample=" << sample + 1 << " world=" << start + index
                                          << " before=" << batch.worlds[index].tick << " observed=" << tick << '\n';
                                for (int player = 0; player < 2; ++player) {
                                    const auto player_index = index * 2 + player;
                                    std::cout << "PLAYER_SAMPLE sample=" << sample + 1 << " world=" << start + index
                                              << " player=" << player << " before=" << batch.players[player_index].tick
                                              << " observed=" << observed_players[player_index].tick << '\n';
                                }
                            }
                            previous[index] = tick;
                        }
                        std::cout << "WORLD_SNAPSHOT sample=" << sample + 1 << " unexpected=" << unexpected
                                  << " total=" << count << std::endl;
                    } else {
                        std::cout << "WORLD_SNAPSHOT unavailable=" << cudaGetErrorString(status) << std::endl;
                        break;
                    }
                }
            }
            std::cout << "CUDA_SYNCHRONIZE" << std::endl;
            gpu.synchronize();
            std::cout << "CUDA_DONE" << std::endl;
            if (observer) {
                cudaStreamDestroy(observer);
                cudaFreeHost(observed);
                cudaFreeHost(observed_players);
            }
#else
            throw std::runtime_error("Replay tool was built without CUDA");
#endif
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
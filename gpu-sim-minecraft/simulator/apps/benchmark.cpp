#include "minecraft_sim/cuda_backend.hpp"
#include "scene.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(MCSIM_HAS_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

struct Options {
    int environments = 8192;
    int launches = 1000;
    int ticks_per_launch = 1;
    int warmup_launches = 10;
};

int count(std::string_view value, bool allow_zero = false) {
    int result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()
        || result < (allow_zero ? 0 : 1)) {
        throw std::invalid_argument("Expected a " + std::string(allow_zero ? "nonnegative" : "positive")
                                    + " integer, got: " + std::string(value));
    }
    return result;
}

std::uint64_t product(std::uint64_t left, std::uint64_t right) {
    if (right != 0 && left > (std::numeric_limits<std::uint64_t>::max)() / right) {
        throw std::invalid_argument("Benchmark tick count overflow");
    }
    return left * right;
}

struct BenchmarkScene {
    minecraft_sim::app::Scene terrain{std::filesystem::path(MCSIM_BENCHMARK_ENVIRONMENT_PATH)};
    std::vector<minecraft_sim::PlayerState> players;
    std::vector<minecraft_sim::Controls> queued;
    std::vector<minecraft_sim::Controls> held;
    std::vector<minecraft_sim::WorldState> worlds;
    std::vector<minecraft_sim::BlockWorldView> blocks;

    explicit BenchmarkScene(int environments)
        : players(std::size_t(environments) * 2), queued(players.size()), held(players.size()),
          worlds(environments), blocks(environments, terrain.blocks) {
        for (int environment = 0; environment < environments; ++environment) {
            for (int player = 0; player < 2; ++player) {
                const auto index = std::size_t(environment) * 2 + player;
                players[index] = terrain.initial_players[player];
                players[index].yaw = float(environment % 4) * 90.0F;
                queued[index].forward = true;
                queued[index].sprint = environment % 3 == 0;
                queued[index].sneak = environment % 3 == 1;
                queued[index].jump = environment % 3 == 2;
            }
        }
    }

    minecraft_sim::BatchView view() {
        return {players.data(), queued.data(), held.data(), worlds.data(),
                static_cast<int>(worlds.size()), 2, blocks.data(), {terrain.sine.data()}};
    }

    void validate(std::uint64_t expected_tick) const {
        for (const auto& world : worlds) {
            if (world.tick != expected_tick) {
                throw std::runtime_error("Benchmark world tick validation failed");
            }
        }
        for (const auto& player : players) {
            if (player.tick != expected_tick || !std::isfinite(player.position.x)
                || !std::isfinite(player.position.y) || !std::isfinite(player.position.z)
                || !std::isfinite(player.velocity.x) || !std::isfinite(player.velocity.y)
                || !std::isfinite(player.velocity.z)) {
                throw std::runtime_error("Benchmark player state validation failed");
            }
        }
    }
};

#if defined(MCSIM_HAS_CUDA)
void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA benchmark: ") + cudaGetErrorString(error));
    }
}

class Event {
public:
    cudaEvent_t value = nullptr;

    Event() { checkCuda(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

void benchmark(const Options& options, std::uint64_t environment_ticks) {
    if (!minecraft_sim::CudaBatch::available()) {
        throw std::runtime_error("No usable CUDA device; benchmark has no CPU fallback");
    }
    int device_id = 0;
    cudaDeviceProp properties{};
    checkCuda(cudaGetDevice(&device_id));
    checkCuda(cudaGetDeviceProperties(&properties, device_id));
    BenchmarkScene scene(options.environments);
    minecraft_sim::CudaBatch batch(scene.view());
    Event start;
    Event stop;
    for (int launch = 0; launch < options.warmup_launches; ++launch) {
        batch.step(options.ticks_per_launch);
    }
    checkCuda(cudaDeviceSynchronize());

    const auto host_start = std::chrono::steady_clock::now();
    checkCuda(cudaEventRecord(start.value, nullptr));
    for (int launch = 0; launch < options.launches; ++launch) {
        batch.step(options.ticks_per_launch);
    }
    checkCuda(cudaEventRecord(stop.value, nullptr));
    checkCuda(cudaEventSynchronize(stop.value));
    const double host_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - host_start).count();
    float elapsed_ms = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsed_ms, start.value, stop.value));
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0F || host_seconds <= 0.0) {
        throw std::runtime_error("Invalid benchmark duration; increase --launches");
    }

    batch.download(scene.view());
    scene.validate((std::uint64_t(options.warmup_launches) + options.launches) * options.ticks_per_launch);
    const double seconds = double(elapsed_ms) / 1000.0;
    const auto threads = minecraft_sim::CudaBatch::threads_per_block;
    std::cout << std::setprecision(12)
              << "Minecraft Java 26.3 CUDA physics benchmark\n"
              << "Device: " << properties.name << "; compute capability: " << properties.major << '.' << properties.minor
              << "; SMs: " << properties.multiProcessorCount << '\n'
              << "Scenario: synthetic enclosed full-cube arena; fixed mixed forward/sprint/sneak/jump controls\n"
              << "Environments: " << options.environments << "; players/environment: 2\n"
              << "Threads/block: " << threads << "; blocks/launch: "
              << (static_cast<unsigned int>(options.environments) + threads - 1) / threads << '\n'
              << "Measured launches: " << options.launches << "; ticks/launch: " << options.ticks_per_launch
              << "; excluded warmup launches: " << options.warmup_launches << '\n'
              << "Measured environment-ticks: " << environment_ticks << '\n'
              << "CUDA-event elapsed ms: " << elapsed_ms << '\n'
              << "environment-ticks/sec (CUDA events): " << double(environment_ticks) / seconds << '\n'
              << "player-ticks/sec (CUDA events): " << double(environment_ticks) * 2.0 / seconds << '\n'
              << "Host elapsed ms (submission + completion): " << host_seconds * 1000.0 << '\n'
              << "environment-ticks/sec (host wall): " << double(environment_ticks) / host_seconds << '\n'
              << "Final tick/finite-state checks: passed\n"
              << "Timing excludes allocation, uploads, warmup, final download and validation; event interval includes launch gaps.\n"
              << "Physics only: no policy inference, PPO updates, rewards, resets, rendering or per-tick transfers.\n";
}
#endif

}

int main(int argc, char** argv) {
    try {
        Options options;
        for (int argument = 1; argument < argc; ++argument) {
            const std::string_view option = argv[argument];
            if (option == "--help") {
                std::cout << "Usage: minecraft_sim_benchmark [options]\n"
                             "CUDA-only environment-ticks/sec; two players per environment.\n"
                             "  --environments N       Independent worlds (default 8192)\n"
                             "  --launches N           Measured kernel launches (default 1000)\n"
                             "  --ticks-per-launch N   Physics ticks per kernel (default 1)\n"
                             "  --warmup-launches N    Excluded launches, may be 0 (default 10)\n"
                             "Fixed held controls; no inference or per-tick host transfers.\n";
                return 0;
            }
            if (argument + 1 >= argc) {
                throw std::invalid_argument("Missing option value; use --help");
            }
            if (option == "--environments") {
                options.environments = count(argv[++argument]);
            } else if (option == "--launches") {
                options.launches = count(argv[++argument]);
            } else if (option == "--ticks-per-launch") {
                options.ticks_per_launch = count(argv[++argument]);
            } else if (option == "--warmup-launches") {
                options.warmup_launches = count(argv[++argument], true);
            } else {
                throw std::invalid_argument("Unknown option; use --help");
            }
        }
        if (options.environments > (std::numeric_limits<int>::max)() / 2) {
            throw std::invalid_argument("Two-player batch exceeds supported index range");
        }
        const auto environment_ticks = product(product(std::uint64_t(options.environments),
            std::uint64_t(options.launches)), std::uint64_t(options.ticks_per_launch));
#if defined(MCSIM_HAS_CUDA)
        benchmark(options, environment_ticks);
#else
        static_cast<void>(environment_ticks);
        throw std::runtime_error("CUDA benchmark not built; configure with MCSIM_ENABLE_CUDA=ON. No CPU fallback.");
#endif
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
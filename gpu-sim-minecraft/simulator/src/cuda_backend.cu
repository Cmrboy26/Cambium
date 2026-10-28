#include "minecraft_sim/cuda_backend.hpp"
#include "minecraft_sim/projectiles.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace minecraft_sim {
namespace {

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(error));
    }
}

std::size_t checkedProduct(std::size_t left, std::size_t right) {
    if (right != 0 && left > (std::numeric_limits<std::size_t>::max)() / right) {
        throw std::invalid_argument("Batch allocation size overflow");
    }
    return left * right;
}

template <class Value>
struct DeviceBuffer {
    Value* data = nullptr;
    std::size_t count = 0;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data) { cudaFree(data); } }

    void allocate(std::size_t size) {
        count = size;
        if (size != 0) {
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), checkedProduct(size, sizeof(Value))));
        }
    }

    void upload(const Value* source) {
        upload(source, count);
    }

    void upload(const Value* source, std::size_t size) {
        if (size > count || (size != 0 && (!source || !data))) {
            throw std::invalid_argument("Invalid device upload size or source");
        }
        if (size != 0) {
            checkCuda(cudaMemcpy(data, source, size * sizeof(Value), cudaMemcpyHostToDevice));
        }
    }

    void download(Value* destination) const {
        if (count != 0) {
            checkCuda(cudaMemcpy(destination, data, count * sizeof(Value), cudaMemcpyDeviceToHost));
        }
    }
};

void validateHostBatch(BatchView batch) {
    if (batch.environment_count <= 0 || batch.players_per_environment <= 0
        || batch.environment_count > (std::numeric_limits<int>::max)() / batch.players_per_environment
        || !batch.players || !batch.queued_controls || !batch.held_controls || !batch.worlds) {
        throw std::invalid_argument("Invalid host batch shape or state pointers");
    }
}

std::size_t cellCount(BlockWorldView blocks) {
    if (blocks.size_x < 0 || blocks.size_y < 0 || blocks.size_z < 0
        || std::int64_t(blocks.origin_x) + blocks.size_x > (std::numeric_limits<int>::max)()
        || std::int64_t(blocks.origin_y) + blocks.size_y > (std::numeric_limits<int>::max)()
        || std::int64_t(blocks.origin_z) + blocks.size_z > (std::numeric_limits<int>::max)()) {
        throw std::invalid_argument("Invalid block world dimensions");
    }
    const auto count = checkedProduct(checkedProduct(std::size_t(blocks.size_x),
                                                     std::size_t(blocks.size_y)),
                                      std::size_t(blocks.size_z));
    if (count != 0 && !blocks.cells) {
        throw std::invalid_argument("Missing block storage");
    }
    return count;
}

__global__ void advanceEnvironments(BatchView batch, ArrowObservationSummary* arrow_summaries) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned int>(batch.environment_count)) {
        return;
    }
    auto world = batch.world(static_cast<int>(index));
    if (batch.projectile_states) {
        const auto blocks = batch.block_worlds ? batch.block_worlds[index] : BlockWorldView{};
        processWorldTick(world, blocks, batch.projectile_states[index]);
    } else {
        world.processTick();
    }
    if (arrow_summaries && batch.projectile_states) {
        for (int player = 0; player < projectile_player_count; ++player) {
            const auto player_index = index * batch.players_per_environment + player;
            arrow_summaries[player_index] = summarizeArrowObservation(
                batch.players[player_index], batch.projectile_states[index], player);
        }
    }
}

__global__ void resetSelected(BatchView batch, const int* ids, int count, const PlayerState* players,
                              const ProjectileWorldState* projectile_states,
                              ArrowObservationSummary* arrow_summaries) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned int>(count)) {
        return;
    }
    const int destination = ids[index];
    const bool simulate_fall_damage = batch.worlds[destination].simulate_fall_damage;
    batch.worlds[destination] = {};
    batch.worlds[destination].simulate_fall_damage = simulate_fall_damage;
    for (int player = 0; player < batch.players_per_environment; ++player) {
        const int offset = destination * batch.players_per_environment + player;
        batch.players[offset] = players[index * batch.players_per_environment + player];
        batch.queued_controls[offset] = {};
        batch.held_controls[offset] = {};
    }
    if (batch.projectile_states && projectile_states) {
        auto& destination_state = batch.projectile_states[destination];
        const auto& source_state = projectile_states[index];
        for (int player = 0; player < projectile_player_count; ++player) {
            destination_state.bows[player] = source_state.bows[player];
        }
        destination_state.random = source_state.random;
        for (int slot = 0; slot < destination_state.arrow_capacity; ++slot) {
            destination_state.arrows[slot] = {};
        }
        for (int slot = 0; slot < destination_state.wind_charge_capacity; ++slot) {
            destination_state.wind_charges[slot] = {};
        }
        for (int slot = 0; slot < destination_state.thrown_item_capacity; ++slot) {
            destination_state.thrown_items[slot] = {};
        }
        if (arrow_summaries) {
            for (int player = 0; player < projectile_player_count; ++player) {
                const int player_index = destination * batch.players_per_environment + player;
                arrow_summaries[player_index] = summarizeArrowObservation(
                    batch.players[player_index], destination_state, player);
            }
        }
    }
}

}

struct CudaBatch::Storage {
    DeviceBuffer<PlayerState> players;
    DeviceBuffer<Controls> queued;
    DeviceBuffer<Controls> held;
    DeviceBuffer<WorldState> worlds;
    DeviceBuffer<ProjectileWorldState> projectile_states;
    DeviceBuffer<ProjectileWorldState> reset_projectile_states;
    DeviceBuffer<ArrowState> projectile_arrows;
    DeviceBuffer<WindChargeState> wind_charges;
    int wind_charge_capacity = 0;
    DeviceBuffer<ThrownProjectileState> thrown_items;
    int thrown_item_capacity = 0;
    DeviceBuffer<ArrowObservationSummary> arrow_summaries;
    DeviceBuffer<int> reset_ids;
    DeviceBuffer<PlayerState> reset_players;
    DeviceBuffer<Block> cells;
    DeviceBuffer<BlockWorldView> block_worlds;
    DeviceBuffer<float> sine;
    BatchView view;
    bool summarize_arrows = false;
};

bool CudaBatch::available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

CudaBatch::CudaBatch(BatchView initial, bool summarize_arrows) : storage_(std::make_unique<Storage>()) {
    validateHostBatch(initial);
    if (!initial.math.sine_table) {
        throw std::invalid_argument("A shared sine table is required");
    }
    const auto player_count = std::size_t(initial.environment_count) * initial.players_per_environment;
    auto& storage = *storage_;
    storage.summarize_arrows = summarize_arrows;
    storage.players.allocate(player_count);
    storage.queued.allocate(player_count);
    storage.held.allocate(player_count);
    storage.worlds.allocate(initial.environment_count);
    storage.reset_ids.allocate(initial.environment_count);
    storage.reset_players.allocate(player_count);
    storage.sine.allocate(sine_table_size);
    storage.players.upload(initial.players);
    storage.queued.upload(initial.queued_controls);
    storage.held.upload(initial.held_controls);
    storage.worlds.upload(initial.worlds);
    storage.sine.upload(initial.math.sine_table);

    if (initial.projectile_states) {
        const auto arrow_count = checkedProduct(std::size_t(initial.environment_count), projectile_arrow_capacity);
        storage.projectile_states.allocate(initial.environment_count);
        storage.reset_projectile_states.allocate(initial.environment_count);
        storage.projectile_arrows.allocate(arrow_count);
        storage.wind_charge_capacity = initial.projectile_states[0].wind_charge_capacity;
        if (storage.wind_charge_capacity < 0) {
            throw std::invalid_argument("Invalid wind charge pool capacity");
        }
        const auto wind_count = checkedProduct(std::size_t(initial.environment_count),
                                               std::size_t(storage.wind_charge_capacity));
        if (wind_count != 0) storage.wind_charges.allocate(wind_count);
        storage.thrown_item_capacity = initial.projectile_states[0].thrown_item_capacity;
        if (storage.thrown_item_capacity < 0) {
            throw std::invalid_argument("Invalid thrown item pool capacity");
        }
        const auto thrown_count = checkedProduct(std::size_t(initial.environment_count),
                                                 std::size_t(storage.thrown_item_capacity));
        if (thrown_count != 0) storage.thrown_items.allocate(thrown_count);
        if (summarize_arrows) {
            storage.arrow_summaries.allocate(player_count);
        }
        std::vector<ProjectileWorldState> projectile_states(initial.environment_count);
        std::vector<ArrowState> arrows(arrow_count);
        std::vector<WindChargeState> wind_charges(wind_count);
        std::vector<ThrownProjectileState> thrown_items(thrown_count);
        for (int index = 0; index < initial.environment_count; ++index) {
            const auto& source = initial.projectile_states[index];
            if (!source.arrows || source.arrow_capacity != projectile_arrow_capacity) {
                throw std::invalid_argument("CUDA bow worlds require a fixed 128-arrow pool");
            }
            if (source.wind_charge_capacity != storage.wind_charge_capacity
                || (storage.wind_charge_capacity > 0 && !source.wind_charges)) {
                throw std::invalid_argument("CUDA wind charge pool shape mismatch");
            }
            if (source.thrown_item_capacity != storage.thrown_item_capacity
                || (storage.thrown_item_capacity > 0 && !source.thrown_items)) {
                throw std::invalid_argument("CUDA thrown item pool shape mismatch");
            }
            projectile_states[index] = source;
            projectile_states[index].arrows = storage.projectile_arrows.data
                + std::size_t(index) * projectile_arrow_capacity;
            projectile_states[index].wind_charges = wind_count == 0 ? nullptr : storage.wind_charges.data
                + std::size_t(index) * storage.wind_charge_capacity;
            projectile_states[index].thrown_items = thrown_count == 0 ? nullptr : storage.thrown_items.data
                + std::size_t(index) * storage.thrown_item_capacity;
            std::copy_n(source.arrows, projectile_arrow_capacity,
                        arrows.data() + std::size_t(index) * projectile_arrow_capacity);
            if (wind_count != 0) {
                std::copy_n(source.wind_charges, storage.wind_charge_capacity,
                            wind_charges.data() + std::size_t(index) * storage.wind_charge_capacity);
            }
            if (thrown_count != 0) {
                std::copy_n(source.thrown_items, storage.thrown_item_capacity,
                            thrown_items.data() + std::size_t(index) * storage.thrown_item_capacity);
            }
        }
        storage.projectile_arrows.upload(arrows.data());
        if (wind_count != 0) storage.wind_charges.upload(wind_charges.data());
        if (thrown_count != 0) storage.thrown_items.upload(thrown_items.data());
        storage.projectile_states.upload(projectile_states.data());
        if (summarize_arrows) {
            std::vector<ArrowObservationSummary> summaries(player_count);
            for (int environment = 0; environment < initial.environment_count; ++environment) {
                for (int player = 0; player < initial.players_per_environment; ++player) {
                    summaries[std::size_t(environment) * initial.players_per_environment + player]
                        = summarizeArrowObservation(
                            initial.players[std::size_t(environment) * initial.players_per_environment + player],
                            initial.projectile_states[environment], player);
                }
            }
            storage.arrow_summaries.upload(summaries.data());
        }
    }

    std::vector<Block> cells;
    std::vector<BlockWorldView> worlds(initial.environment_count);
    std::vector<std::size_t> offsets(initial.environment_count);
    for (int index = 0; index < initial.environment_count; ++index) {
        auto blocks = initial.block_worlds ? initial.block_worlds[index] : BlockWorldView{};
        const auto count = cellCount(blocks);
        offsets[index] = cells.size();
        if (count != 0) {
            cells.insert(cells.end(), blocks.cells, blocks.cells + count);
        }
        worlds[index] = blocks;
    }
    storage.cells.allocate(cells.size());
    storage.cells.upload(cells.data());
    for (int index = 0; index < initial.environment_count; ++index) {
        worlds[index].cells = cellCount(worlds[index]) != 0
            ? storage.cells.data + offsets[index] : nullptr;
    }
    storage.block_worlds.allocate(worlds.size());
    storage.block_worlds.upload(worlds.data());
    storage.view = {storage.players.data, storage.queued.data, storage.held.data,
                    storage.worlds.data, initial.environment_count, initial.players_per_environment,
                    storage.block_worlds.data, {storage.sine.data}, storage.projectile_states.data};
}

CudaBatch::~CudaBatch() = default;

BatchView CudaBatch::deviceView() const { return storage_->view; }

void CudaBatch::uploadControls(const Controls* controls, std::size_t count) {
    if (!controls || count != storage_->queued.count) {
        throw std::invalid_argument("Control batch size mismatch");
    }
    storage_->queued.upload(controls);
}

void CudaBatch::step(int ticks) {
    if (ticks <= 0) {
        throw std::invalid_argument("Tick count must be positive");
    }
    const auto count = static_cast<unsigned int>(storage_->view.environment_count);
    for (int tick = 0; tick < ticks; ++tick) {
        auto* summaries = tick == ticks - 1 ? storage_->arrow_summaries.data : nullptr;
        advanceEnvironments<<<(count + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
            storage_->view, summaries);
        checkCuda(cudaGetLastError());
    }
}

void CudaBatch::synchronize() const {
    checkCuda(cudaDeviceSynchronize());
}

void CudaBatch::resetEnvironments(const int* ids, std::size_t count, const PlayerState* players,
                                  const ProjectileWorldState* projectile_states) {
    if (count == 0) {
        return;
    }
    if (!ids || !players || count > storage_->reset_ids.count
        || (storage_->projectile_states.count != 0 && !projectile_states)) {
        throw std::invalid_argument("Invalid reset batch");
    }
    std::vector<bool> seen(storage_->view.environment_count, false);
    for (std::size_t index = 0; index < count; ++index) {
        if (ids[index] < 0 || ids[index] >= storage_->view.environment_count || seen[ids[index]]) {
            throw std::invalid_argument("Invalid or duplicate reset environment");
        }
        seen[ids[index]] = true;
    }
    checkCuda(cudaMemcpy(storage_->reset_ids.data, ids, count * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(storage_->reset_players.data, players,
        count * storage_->view.players_per_environment * sizeof(PlayerState), cudaMemcpyHostToDevice));
    if (storage_->projectile_states.count != 0) {
        storage_->reset_projectile_states.upload(projectile_states, count);
    }
    resetSelected<<<(static_cast<unsigned int>(count) + threads_per_block - 1) / threads_per_block,
                    threads_per_block>>>(storage_->view, storage_->reset_ids.data,
                                        static_cast<int>(count), storage_->reset_players.data,
                                        storage_->projectile_states.count != 0
                                            ? storage_->reset_projectile_states.data : nullptr,
                                        storage_->arrow_summaries.data);
    checkCuda(cudaGetLastError());
}

void CudaBatch::download(BatchView destination) const {
    validateHostBatch(destination);
    if (destination.environment_count != storage_->view.environment_count
        || destination.players_per_environment != storage_->view.players_per_environment) {
        throw std::invalid_argument("Snapshot batch size mismatch");
    }
    storage_->players.download(destination.players);
    storage_->queued.download(destination.queued_controls);
    storage_->held.download(destination.held_controls);
    storage_->worlds.download(destination.worlds);
    if (storage_->projectile_states.count != 0) {
        if (!destination.projectile_states) {
            throw std::invalid_argument("Snapshot is missing projectile state storage");
        }
        std::vector<ProjectileWorldState> projectile_states(storage_->projectile_states.count);
        std::vector<ArrowState> arrows(storage_->projectile_arrows.count);
        std::vector<WindChargeState> wind_charges(storage_->wind_charges.count);
        std::vector<ThrownProjectileState> thrown_items(storage_->thrown_items.count);
        storage_->projectile_states.download(projectile_states.data());
        storage_->projectile_arrows.download(arrows.data());
        if (!wind_charges.empty()) storage_->wind_charges.download(wind_charges.data());
        if (!thrown_items.empty()) storage_->thrown_items.download(thrown_items.data());
        for (std::size_t index = 0; index < projectile_states.size(); ++index) {
            auto& destination_state = destination.projectile_states[index];
            if (!destination_state.arrows || destination_state.arrow_capacity != projectile_arrow_capacity) {
                throw std::invalid_argument("Snapshot projectile pool shape mismatch");
            }
            if (destination_state.wind_charge_capacity != storage_->wind_charge_capacity
                || (storage_->wind_charge_capacity > 0 && !destination_state.wind_charges)) {
                throw std::invalid_argument("Snapshot wind charge pool shape mismatch");
            }
            if (destination_state.thrown_item_capacity != storage_->thrown_item_capacity
                || (storage_->thrown_item_capacity > 0 && !destination_state.thrown_items)) {
                throw std::invalid_argument("Snapshot thrown item pool shape mismatch");
            }
            for (int player = 0; player < projectile_player_count; ++player) {
                destination_state.bows[player] = projectile_states[index].bows[player];
            }
            destination_state.random = projectile_states[index].random;
            std::copy_n(arrows.data() + index * projectile_arrow_capacity, projectile_arrow_capacity,
                        destination_state.arrows);
            if (!wind_charges.empty()) {
                std::copy_n(wind_charges.data() + index * storage_->wind_charge_capacity,
                            storage_->wind_charge_capacity, destination_state.wind_charges);
            }
            if (!thrown_items.empty()) {
                std::copy_n(thrown_items.data() + index * storage_->thrown_item_capacity,
                            storage_->thrown_item_capacity, destination_state.thrown_items);
            }
        }
    }
}

void CudaBatch::downloadForTraining(BatchView destination, ArrowObservationSummary* arrow_summaries) const {
    validateHostBatch(destination);
    if (destination.environment_count != storage_->view.environment_count
        || destination.players_per_environment != storage_->view.players_per_environment) {
        throw std::invalid_argument("Training batch size mismatch");
    }
    if (arrow_summaries && storage_->arrow_summaries.count == 0) {
        throw std::invalid_argument("Arrow summaries were not enabled for this CUDA batch");
    }
    storage_->players.download(destination.players);
    storage_->worlds.download(destination.worlds);
    if (arrow_summaries) {
        storage_->arrow_summaries.download(arrow_summaries);
    }
}

}
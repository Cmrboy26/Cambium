#include "minecraft_sim/cuda_backend.hpp"

#include <cuda_runtime.h>

#include <limits>
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
        if (count != 0) {
            checkCuda(cudaMemcpy(data, source, count * sizeof(Value), cudaMemcpyHostToDevice));
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

__global__ void advanceEnvironments(BatchView batch, int ticks) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned int>(batch.environment_count)) {
        return;
    }
    auto world = batch.world(static_cast<int>(index));
    for (int tick = 0; tick < ticks; ++tick) {
        world.processTick();
    }
}

__global__ void resetSelected(BatchView batch, const int* ids, int count, const PlayerState* players) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned int>(count)) {
        return;
    }
    const int destination = ids[index];
    batch.worlds[destination] = {};
    for (int player = 0; player < batch.players_per_environment; ++player) {
        const int offset = destination * batch.players_per_environment + player;
        batch.players[offset] = players[index * batch.players_per_environment + player];
        batch.queued_controls[offset] = {};
        batch.held_controls[offset] = {};
    }
}

}

struct CudaBatch::Storage {
    DeviceBuffer<PlayerState> players;
    DeviceBuffer<Controls> queued;
    DeviceBuffer<Controls> held;
    DeviceBuffer<WorldState> worlds;
    DeviceBuffer<int> reset_ids;
    DeviceBuffer<PlayerState> reset_players;
    DeviceBuffer<Block> cells;
    DeviceBuffer<BlockWorldView> block_worlds;
    DeviceBuffer<float> sine;
    BatchView view;
};

bool CudaBatch::available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

CudaBatch::CudaBatch(BatchView initial) : storage_(std::make_unique<Storage>()) {
    validateHostBatch(initial);
    if (!initial.math.sine_table) {
        throw std::invalid_argument("A shared sine table is required");
    }
    const auto player_count = std::size_t(initial.environment_count) * initial.players_per_environment;
    auto& storage = *storage_;
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
                    storage.block_worlds.data, {storage.sine.data}};
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
    advanceEnvironments<<<(count + threads_per_block - 1) / threads_per_block, threads_per_block>>>(storage_->view, ticks);
    checkCuda(cudaGetLastError());
}

void CudaBatch::resetEnvironments(const int* ids, std::size_t count, const PlayerState* players) {
    if (count == 0) {
        return;
    }
    if (!ids || !players || count > storage_->reset_ids.count) {
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
    resetSelected<<<(static_cast<unsigned int>(count) + threads_per_block - 1) / threads_per_block,
                    threads_per_block>>>(storage_->view, storage_->reset_ids.data,
                                        static_cast<int>(count), storage_->reset_players.data);
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
}

}
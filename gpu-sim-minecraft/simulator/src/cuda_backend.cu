#include "minecraft_sim/cuda_backend.hpp"
#include "minecraft_sim/projectiles.hpp"
#include "../apps/actions.hpp"
#include "../apps/observation.hpp"
#include "../apps/rewards.hpp"

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

void validateCurrentDevice(int expected_device) {
    int device = -1;
    checkCuda(cudaGetDevice(&device));
    if (device != expected_device) throw std::invalid_argument("Current CUDA device differs from batch device");
}

void validateDevicePointer(const void* output, int expected_device) {
    validateCurrentDevice(expected_device);
    cudaPointerAttributes attributes{};
    const auto status = cudaPointerGetAttributes(&attributes, output);
    if (status != cudaSuccess) {
        cudaGetLastError();
        throw std::invalid_argument("Buffer must be CUDA device memory");
    }
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != expected_device) {
        throw std::invalid_argument("Buffer must be on the batch CUDA device");
    }
}

void rejectEventManagedGraphCapture(cudaStream_t stream) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    checkCuda(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        throw std::invalid_argument("Event-managed CUDA operations do not support graph capture");
    }
}

__global__ void advanceEnvironments(BatchView batch, ArrowObservationSummary* arrow_summaries,
                                    app::ObservationActorMetadata* observation_actors, const int* action_status) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned int>(batch.environment_count) || (action_status && *action_status != 0)) {
        return;
    }
    const int offset = static_cast<int>(index) * batch.players_per_environment;
    if (observation_actors) {
        for (int player = 0; player < batch.players_per_environment; ++player) {
            const auto& control = batch.queued_controls[offset + player];
            if (control.select_weapon && control.selected_slot >= 0) {
                observation_actors[offset + player].selected_slot = control.selected_slot;
            }
        }
    }
    auto world = batch.world(static_cast<int>(index));
    if (batch.projectile_states) {
        const auto blocks = batch.block_worlds ? batch.block_worlds[index] : BlockWorldView{};
        processWorldTick(world, blocks, batch.projectile_states[index]);
    } else {
        world.processTick();
    }
    if (observation_actors) {
        for (int player = 0; player < batch.players_per_environment; ++player) {
            auto& observed = observation_actors[offset + player].observed_opponent_disable;
            if (observed > 0) --observed;
            if (batch.players[offset + player].last_attack.shield_disabled) observed = 99;
        }
    }
    if (arrow_summaries && batch.projectile_states) {
        for (int player = 0; player < projectile_player_count; ++player) {
            const auto player_index = index * batch.players_per_environment + player;
            arrow_summaries[player_index] = summarizeArrowObservation(
                batch.players[player_index], batch.projectile_states[index], player);
        }
    }
}

__global__ void validateActions(BatchView batch, const app::ActionHead* heads, int head_count,
                                const app::ObservationActorMetadata* actors, const std::int32_t* actions,
                                const std::uint8_t* finished, int* status) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto count = static_cast<unsigned int>(batch.environment_count * batch.players_per_environment);
    if (index >= count) return;
    if (finished && finished[index / batch.players_per_environment]) atomicOr(status, 2);
    Controls controls;
    if (!app::decodeAction(heads, head_count, actions + std::size_t(index) * head_count,
                           actors[index].slots, actors[index].slot_count, controls)) {
        atomicOr(status, 1);
    }
}

__global__ void validateFinishedWorlds(BatchView batch, const std::uint8_t* finished, int* status) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < static_cast<unsigned int>(batch.environment_count) && finished[index]) atomicOr(status, 2);
}

__global__ void decodeActions(BatchView batch, const app::ActionHead* heads, int head_count,
                              const app::ObservationActorMetadata* actors, const std::int32_t* actions,
                              const int* status) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto count = static_cast<unsigned int>(batch.environment_count * batch.players_per_environment);
    if (index >= count || *status != 0) return;
    Controls controls;
    if (app::decodeAction(heads, head_count, actions + std::size_t(index) * head_count,
                          actors[index].slots, actors[index].slot_count, controls)) {
        batch.queued_controls[index] = controls;
    }
}

__global__ void observeBaseEnvironments(BatchView batch, float* output) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto count = static_cast<unsigned int>(batch.environment_count * batch.players_per_environment);
    if (index >= count) return;
    const auto environment = index / 2;
    const auto target = index ^ 1U;
    const auto blocks = batch.block_worlds ? batch.block_worlds[environment] : BlockWorldView{};
    app::baseWorldObservation(batch.players[index], batch.players[target], blocks, output + std::size_t(index) * 22);
}

__global__ void observeConfiguredEnvironments(BatchView batch, const app::ObservationActorMetadata* actors,
                                              const app::ObservationProjection* projection, int features,
                                              float* output) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto count = static_cast<unsigned int>(batch.environment_count * batch.players_per_environment);
    if (index >= count) return;
    const auto environment = index / 2;
    const auto target_index = index ^ 1U;
    const auto& source = batch.players[index];
    const auto& target = batch.players[target_index];
    const auto& metadata = actors[index];
    const auto blocks = batch.block_worlds ? batch.block_worlds[environment] : BlockWorldView{};
    float catalogue[102]{};
    app::baseWorldObservation(source, target, blocks, catalogue);
    app::legacyEquipmentObservation(catalogue, source, target, metadata.observed_opponent_disable);
    catalogue[31] = static_cast<float>(metadata.selected_slot);
    catalogue[32] = static_cast<float>(actors[target_index].selected_slot);
    app::velocityObservation(catalogue, source, target);
    app::movementObservation(catalogue, source, target);
    app::equipmentObservation(catalogue, source, target, metadata.slots, metadata.slot_count);
    if (batch.projectile_states) {
        app::arrowObservation(catalogue, source, target, batch.projectile_states[environment], static_cast<int>(index % 2));
    }
    float* projected = output + std::size_t(index) * features;
    for (int feature = 0; feature < features; ++feature) {
        projected[feature] = app::projectObservation(catalogue, projection[feature]);
    }
}

__global__ void evaluateDuelRewards(BatchView batch, const app::DuelRewardSettings* settings,
                                    app::DuelRewardState* states, app::DuelRewardStep* results,
                                    std::uint8_t* finished, const int* action_status) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned int>(batch.environment_count) || (action_status && *action_status != 0)) return;
    const auto blocks = batch.block_worlds ? batch.block_worlds[index] : BlockWorldView{};
    results[index] = app::duelRewards(batch.players + std::size_t(index) * 2, batch.worlds[index], blocks,
                                      batch.projectile_states[index], *settings, states[index]);
    finished[index] = results[index].terminated || results[index].truncated;
}

__global__ void exportDuelRewards(BatchView batch, const app::DuelRewardStep* results, const int* action_status,
                                  float* rewards, std::uint8_t* terminated, std::uint8_t* truncated,
                                  std::uint8_t* losses, double* contributions) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned int>(batch.environment_count) || (action_status && *action_status != 0)) return;
    const auto& result = results[index];
    terminated[index] = result.terminated;
    truncated[index] = result.truncated;
    for (int player = 0; player < 2; ++player) {
        const auto actor = std::size_t(index) * 2 + player;
        rewards[actor] = result.rewards[player];
        losses[actor] = result.lost[player];
        for (int term = 0; term < 18; ++term) contributions[actor * 18 + term] = result.contributions[player][term];
    }
}

__global__ void validateResetMask(BatchView batch, const std::uint8_t* mask, int* status) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < static_cast<unsigned int>(batch.environment_count) && mask[index] > 1) atomicOr(status, 4);
}

__global__ void resetFixedWorlds(BatchView batch, const std::uint8_t* mask,
                                 const PlayerState* initial_players,
                                 const app::ObservationActorMetadata* initial_actors,
                                 app::ObservationActorMetadata* actors,
                                 std::uint64_t* episodes, std::uint64_t seed,
                                 app::DuelRewardState* reward_states, std::uint8_t* finished,
                                 ArrowObservationSummary* summaries, const int* status) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned int>(batch.environment_count) || *status != 0 || !mask[index]) return;
    const auto episode = episodes[index]++;
    const bool simulate_fall_damage = batch.worlds[index].simulate_fall_damage;
    batch.worlds[index] = {};
    batch.worlds[index].simulate_fall_damage = simulate_fall_damage;
    resetProjectileWorld(batch.projectile_states[index], seed + std::uint64_t(index) * 1000003ULL
        + episode * 1000033ULL);
    reward_states[index] = {};
    finished[index] = 0;
    for (int player = 0; player < batch.players_per_environment; ++player) {
        const auto actor = index * batch.players_per_environment + player;
        batch.players[actor] = initial_players[actor];
        batch.players[actor].random.setSeed(seed + std::uint64_t(actor) * 1000003ULL + episode * 1000033ULL);
        batch.queued_controls[actor] = {};
        batch.held_controls[actor] = {};
        actors[actor] = initial_actors[actor];
        actors[actor].observed_opponent_disable = 0;
        batch.projectile_states[index].bows[player].ordinary_arrows = batch.players[actor].ordinary_arrow_count;
        if (summaries) {
            summaries[actor] = summarizeArrowObservation(batch.players[actor], batch.projectile_states[index], player);
        }
    }
}

__global__ void resetSelected(BatchView batch, const int* ids, int count, const PlayerState* players,
                              const ProjectileWorldState* projectile_states,
                              ArrowObservationSummary* arrow_summaries,
                              app::ObservationActorMetadata* observation_actors,
                              const app::ObservationActorMetadata* reset_observation_actors,
                              app::DuelRewardState* reward_states, std::uint8_t* finished) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned int>(count)) {
        return;
    }
    const int destination = ids[index];
    const bool simulate_fall_damage = batch.worlds[destination].simulate_fall_damage;
    batch.worlds[destination] = {};
    batch.worlds[destination].simulate_fall_damage = simulate_fall_damage;
    if (reward_states) reward_states[destination] = {};
    if (finished) finished[destination] = 0;
    for (int player = 0; player < batch.players_per_environment; ++player) {
        const int offset = destination * batch.players_per_environment + player;
        batch.players[offset] = players[index * batch.players_per_environment + player];
        batch.queued_controls[offset] = {};
        batch.held_controls[offset] = {};
        if (observation_actors) {
            observation_actors[offset] = reset_observation_actors[index * batch.players_per_environment + player];
        }
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
    DeviceBuffer<app::ObservationProjection> observation_projection;
    DeviceBuffer<app::ObservationActorMetadata> observation_actors;
    DeviceBuffer<app::ObservationActorMetadata> reset_observation_actors;
    DeviceBuffer<app::ActionHead> action_heads;
    DeviceBuffer<int> action_status;
    DeviceBuffer<app::DuelRewardSettings> reward_settings;
    DeviceBuffer<app::DuelRewardState> reward_states;
    DeviceBuffer<app::DuelRewardStep> reward_results;
    DeviceBuffer<std::uint8_t> reward_finished;
    DeviceBuffer<PlayerState> fixed_reset_players;
    DeviceBuffer<app::ObservationActorMetadata> fixed_reset_actors;
    DeviceBuffer<std::uint64_t> fixed_reset_episodes;
    std::uint64_t fixed_reset_seed = 0;
    DeviceBuffer<int> reset_ids;
    DeviceBuffer<PlayerState> reset_players;
    DeviceBuffer<Block> cells;
    DeviceBuffer<BlockWorldView> block_worlds;
    DeviceBuffer<float> sine;
    BatchView view;
    bool summarize_arrows = false;
    int device = -1;
    cudaEvent_t observation_state_ready = nullptr;
    cudaEvent_t observation_read_done = nullptr;
    bool observation_read_pending = false;
    float* registered_observation = nullptr;
    std::size_t registered_observation_count = 0;
    const std::int32_t* registered_actions = nullptr;
    std::size_t registered_action_count = 0;
    float* registered_rewards = nullptr;
    std::uint8_t* registered_terminated = nullptr;
    std::uint8_t* registered_truncated = nullptr;
    std::uint8_t* registered_losses = nullptr;
    double* registered_contributions = nullptr;
    const std::uint8_t* registered_reset_mask = nullptr;

    ~Storage() {
        if (observation_read_done && observation_read_pending) cudaEventSynchronize(observation_read_done);
        if (observation_state_ready) cudaEventSynchronize(observation_state_ready);
        if (observation_read_done) cudaEventDestroy(observation_read_done);
        if (observation_state_ready) cudaEventDestroy(observation_state_ready);
    }
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
    checkCuda(cudaGetDevice(&storage.device));
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

void CudaBatch::validateDeviceBuffer(const void* device_buffer) const {
    if (!device_buffer) throw std::invalid_argument("Null CUDA buffer");
    validateDevicePointer(device_buffer, storage_->device);
}

void CudaBatch::uploadControls(const Controls* controls, std::size_t count) {
    if (!controls || count != storage_->queued.count) {
        throw std::invalid_argument("Control batch size mismatch");
    }
    if (storage_->observation_state_ready) checkCuda(cudaEventSynchronize(storage_->observation_state_ready));
    storage_->queued.upload(controls);
}

void CudaBatch::configureObservations(const app::ObservationProjection* projection, std::size_t features,
                                      const app::ObservationActorMetadata* actors, std::size_t count) {
    if (storage_->observation_actors.data || !projection || features == 0 || features > 128
        || !actors || count != storage_->players.count || storage_->view.players_per_environment != 2) {
        throw std::invalid_argument("Invalid configured observation layout or actor metadata");
    }
    validateCurrentDevice(storage_->device);
    for (std::size_t index = 0; index < features; ++index) {
        if (projection[index].source < 0 || projection[index].source >= 102) {
            throw std::invalid_argument("Configured observation source outside catalogue");
        }
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (actors[index].slot_count < 1 || actors[index].slot_count > 9
            || actors[index].selected_slot < 0 || actors[index].selected_slot >= actors[index].slot_count
            || actors[index].observed_opponent_disable < 0) {
            throw std::invalid_argument("Invalid configured observation actor metadata");
        }
    }
    storage_->observation_projection.allocate(features);
    storage_->observation_actors.allocate(count);
    storage_->reset_observation_actors.allocate(count);
    storage_->observation_projection.upload(projection);
    storage_->observation_actors.upload(actors);
    checkCuda(cudaEventCreateWithFlags(&storage_->observation_state_ready, cudaEventDisableTiming));
    checkCuda(cudaEventCreateWithFlags(&storage_->observation_read_done, cudaEventDisableTiming));
    checkCuda(cudaEventRecord(storage_->observation_state_ready, nullptr));
}

void CudaBatch::configureActions(const app::ActionHead* heads, std::size_t count) {
    if (storage_->action_heads.data || !heads || count == 0 || count > 10 || !storage_->observation_actors.data) {
        throw std::invalid_argument("Invalid action layout or observation metadata");
    }
    validateCurrentDevice(storage_->device);
    bool seen[10]{};
    for (std::size_t index = 0; index < count; ++index) {
        const auto& head = heads[index];
        if (head.kind < 0 || head.kind > 9 || seen[head.kind] || head.count < 1
            || head.count > app::max_action_choices) throw std::invalid_argument("Invalid action head");
        seen[head.kind] = true;
        for (int choice = 0; choice < head.count; ++choice) {
            const float value = head.values[choice];
            if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite action choice");
            if ((head.kind == 0 || head.kind == 6)
                && (!(value >= 0.0F && value < 9.0F) || value != static_cast<float>(static_cast<int>(value)))) {
                throw std::invalid_argument("Invalid movement or slot choice");
            }
            if (head.kind != 0 && head.kind != 4 && head.kind != 5 && head.kind != 6
                && value != 0.0F && value != 1.0F) throw std::invalid_argument("Invalid binary action choice");
        }
    }
    if (seen[7] && seen[9]) throw std::invalid_argument("Duplicate item-use aliases");
    storage_->action_heads.allocate(count);
    storage_->action_status.allocate(1);
    storage_->action_heads.upload(heads);
    checkCuda(cudaMemset(storage_->action_status.data, 0, sizeof(int)));
    checkCuda(cudaEventRecord(storage_->observation_state_ready, nullptr));
}

void CudaBatch::decodeActionsOnStream(const std::int32_t* device_actions, std::size_t count, void* stream) {
    if (!device_actions || !storage_->action_heads.data || !storage_->action_status.data
        || count != checkedProduct(storage_->players.count, storage_->action_heads.count)) {
        throw std::invalid_argument("Invalid CUDA action buffer or batch shape");
    }
    validateCurrentDevice(storage_->device);
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    rejectEventManagedGraphCapture(cuda_stream);
    checkCuda(cudaStreamWaitEvent(cuda_stream, storage_->observation_state_ready, 0));
    const auto players = static_cast<unsigned int>(storage_->players.count);
    const auto blocks = (players + threads_per_block - 1) / threads_per_block;
    validateActions<<<blocks, threads_per_block, 0, cuda_stream>>>(storage_->view,
        storage_->action_heads.data, static_cast<int>(storage_->action_heads.count),
        storage_->observation_actors.data, device_actions, storage_->reward_finished.data, storage_->action_status.data);
    checkCuda(cudaGetLastError());
    decodeActions<<<blocks, threads_per_block, 0, cuda_stream>>>(storage_->view,
        storage_->action_heads.data, static_cast<int>(storage_->action_heads.count),
        storage_->observation_actors.data, device_actions, storage_->action_status.data);
    const auto decode_status = cudaGetLastError();
    if (decode_status != cudaSuccess) {
        cudaStreamSynchronize(cuda_stream);
        checkCuda(decode_status);
    }
    const auto status = cudaEventRecord(storage_->observation_state_ready, cuda_stream);
    if (status != cudaSuccess) {
        cudaStreamSynchronize(cuda_stream);
        checkCuda(status);
    }
}

void CudaBatch::registerActionBuffer(const std::int32_t* device_actions, std::size_t count) {
    if (!device_actions || !storage_->action_heads.data || !storage_->action_status.data
        || count != checkedProduct(storage_->players.count, storage_->action_heads.count)) {
        throw std::invalid_argument("Invalid CUDA action buffer or batch shape");
    }
    validateDevicePointer(device_actions, storage_->device);
    checkCuda(cudaEventSynchronize(storage_->observation_state_ready));
    storage_->registered_actions = device_actions;
    storage_->registered_action_count = count;
}

void CudaBatch::decodeRegisteredActionsOnStream(void* stream) {
    if (!storage_->registered_actions) throw std::invalid_argument("Register a CUDA action buffer before use");
    decodeActionsOnStream(storage_->registered_actions, storage_->registered_action_count, stream);
}

int CudaBatch::actionStatus() const {
    if (!storage_->action_status.data) throw std::invalid_argument("CUDA actions are not configured");
    validateCurrentDevice(storage_->device);
    checkCuda(cudaEventSynchronize(storage_->observation_state_ready));
    int status = 0;
    storage_->action_status.download(&status);
    return status;
}

void CudaBatch::clearActionStatus() {
    if (!storage_->action_status.data) throw std::invalid_argument("CUDA actions are not configured");
    validateCurrentDevice(storage_->device);
    checkCuda(cudaEventSynchronize(storage_->observation_state_ready));
    checkCuda(cudaMemset(storage_->action_status.data, 0, sizeof(int)));
    checkCuda(cudaEventRecord(storage_->observation_state_ready, nullptr));
}

void CudaBatch::configureRewards(const app::DuelRewardSettings& settings,
                                 const app::DuelRewardState* states, std::size_t count,
                                 const std::uint8_t* finished) {
    if (storage_->reward_settings.data || !storage_->observation_state_ready || !storage_->view.projectile_states
        || !states || count != static_cast<std::size_t>(storage_->view.environment_count)
        || settings.max_ticks < 1 || settings.max_ticks > 1000000) {
        throw std::invalid_argument("Invalid CUDA duel reward configuration");
    }
    validateCurrentDevice(storage_->device);
    for (float weight : settings.weights) if (!std::isfinite(weight)) throw std::invalid_argument("Nonfinite reward weight");
    for (float weight : settings.event_weights) if (!std::isfinite(weight)) throw std::invalid_argument("Nonfinite event weight");
    for (float weight : settings.shaping_weights) if (!std::isfinite(weight)) throw std::invalid_argument("Nonfinite shaping weight");
    if (!std::isfinite(settings.block_weight)) throw std::invalid_argument("Nonfinite block weight");
    if (finished) {
        for (std::size_t index = 0; index < count; ++index) {
            if (finished[index] > 1) throw std::invalid_argument("Invalid finished-world flag");
        }
    }
    checkCuda(cudaEventSynchronize(storage_->observation_state_ready));
    storage_->reward_settings.allocate(1);
    storage_->reward_states.allocate(count);
    storage_->reward_results.allocate(count);
    storage_->reward_finished.allocate(count);
    storage_->reward_settings.upload(&settings);
    storage_->reward_states.upload(states);
    checkCuda(cudaMemset(storage_->reward_results.data, 0, count * sizeof(app::DuelRewardStep)));
    if (finished) storage_->reward_finished.upload(finished);
    else checkCuda(cudaMemset(storage_->reward_finished.data, 0, count));
    checkCuda(cudaEventRecord(storage_->observation_state_ready, nullptr));
}

void CudaBatch::evaluateRewardsOnStream(void* stream) {
    if (!storage_->reward_settings.data) throw std::invalid_argument("CUDA duel rewards are not configured");
    validateCurrentDevice(storage_->device);
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    rejectEventManagedGraphCapture(cuda_stream);
    checkCuda(cudaStreamWaitEvent(cuda_stream, storage_->observation_state_ready, 0));
    const auto environments = static_cast<unsigned int>(storage_->view.environment_count);
    evaluateDuelRewards<<<(environments + threads_per_block - 1) / threads_per_block, threads_per_block, 0,
                          cuda_stream>>>(storage_->view, storage_->reward_settings.data, storage_->reward_states.data,
                                         storage_->reward_results.data, storage_->reward_finished.data,
                                         storage_->action_status.data);
    checkCuda(cudaGetLastError());
    const auto status = cudaEventRecord(storage_->observation_state_ready, cuda_stream);
    if (status != cudaSuccess) {
        cudaStreamSynchronize(cuda_stream);
        checkCuda(status);
    }
}

void CudaBatch::downloadRewardResults(app::DuelRewardStep* output, std::size_t count) const {
    if (!output || !storage_->reward_results.data || count != storage_->reward_results.count) {
        throw std::invalid_argument("Invalid CUDA duel reward result buffer");
    }
    validateCurrentDevice(storage_->device);
    checkCuda(cudaEventSynchronize(storage_->observation_state_ready));
    storage_->reward_results.download(output);
}

void CudaBatch::exportRewardResultsOnStream(float* rewards, std::uint8_t* terminated, std::uint8_t* truncated,
                                             std::uint8_t* losses, double* contributions,
                                             std::size_t count, void* stream) {
    if (!rewards || !terminated || !truncated || !losses || !contributions || !storage_->reward_results.data
        || count != storage_->reward_results.count) throw std::invalid_argument("Invalid CUDA reward output buffers");
    validateCurrentDevice(storage_->device);
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    rejectEventManagedGraphCapture(cuda_stream);
    checkCuda(cudaStreamWaitEvent(cuda_stream, storage_->observation_state_ready, 0));
    const auto environments = static_cast<unsigned int>(storage_->view.environment_count);
    exportDuelRewards<<<(environments + threads_per_block - 1) / threads_per_block, threads_per_block, 0,
                        cuda_stream>>>(storage_->view, storage_->reward_results.data, storage_->action_status.data,
                                       rewards, terminated, truncated, losses, contributions);
    checkCuda(cudaGetLastError());
    const auto status = cudaEventRecord(storage_->observation_state_ready, cuda_stream);
    if (status != cudaSuccess) {
        cudaStreamSynchronize(cuda_stream);
        checkCuda(status);
    }
}

void CudaBatch::configureFixedResets(const PlayerState* players, const app::ObservationActorMetadata* actors,
                                     const std::uint64_t* episodes, std::size_t count, std::uint64_t seed) {
    if (storage_->fixed_reset_players.data || !storage_->reward_settings.data || !storage_->action_status.data
        || !players || !actors || !episodes || count != static_cast<std::size_t>(storage_->view.environment_count)) {
        throw std::invalid_argument("Invalid fixed CUDA reset configuration");
    }
    validateCurrentDevice(storage_->device);
    for (std::size_t index = 0; index < storage_->players.count; ++index) {
        if (actors[index].slot_count < 1 || actors[index].slot_count > 9 || actors[index].selected_slot < 0
            || actors[index].selected_slot >= actors[index].slot_count) {
            throw std::invalid_argument("Invalid fixed CUDA reset actor metadata");
        }
    }
    checkCuda(cudaEventSynchronize(storage_->observation_state_ready));
    if (storage_->observation_read_pending) checkCuda(cudaEventSynchronize(storage_->observation_read_done));
    storage_->fixed_reset_players.allocate(storage_->players.count);
    storage_->fixed_reset_actors.allocate(storage_->players.count);
    storage_->fixed_reset_episodes.allocate(count);
    storage_->fixed_reset_players.upload(players);
    storage_->fixed_reset_actors.upload(actors);
    storage_->fixed_reset_episodes.upload(episodes);
    storage_->fixed_reset_seed = seed;
    checkCuda(cudaEventRecord(storage_->observation_state_ready, nullptr));
}

void CudaBatch::downloadFixedResetEpisodes(std::uint64_t* episodes, std::size_t count) const {
    if (!episodes || !storage_->fixed_reset_episodes.data || count != storage_->fixed_reset_episodes.count) {
        throw std::invalid_argument("Invalid fixed CUDA reset episode output");
    }
    validateCurrentDevice(storage_->device);
    checkCuda(cudaEventSynchronize(storage_->observation_state_ready));
    storage_->fixed_reset_episodes.download(episodes);
}

void CudaBatch::uploadFixedResetEpisodes(const std::uint64_t* episodes, std::size_t count) {
    if (!episodes || !storage_->fixed_reset_episodes.data || count != storage_->fixed_reset_episodes.count) {
        throw std::invalid_argument("Invalid fixed CUDA reset episode input");
    }
    validateCurrentDevice(storage_->device);
    checkCuda(cudaEventSynchronize(storage_->observation_state_ready));
    storage_->fixed_reset_episodes.upload(episodes);
    checkCuda(cudaEventRecord(storage_->observation_state_ready, nullptr));
}

void CudaBatch::resetFixedOnStream(const std::uint8_t* device_mask, std::size_t count, void* stream) {
    if (!device_mask || !storage_->fixed_reset_players.data
        || count != static_cast<std::size_t>(storage_->view.environment_count)) {
        throw std::invalid_argument("Invalid fixed CUDA reset mask or batch shape");
    }
    validateCurrentDevice(storage_->device);
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    rejectEventManagedGraphCapture(cuda_stream);
    checkCuda(cudaStreamWaitEvent(cuda_stream, storage_->observation_state_ready, 0));
    if (storage_->observation_read_pending) {
        checkCuda(cudaStreamWaitEvent(cuda_stream, storage_->observation_read_done, 0));
    }
    const auto environments = static_cast<unsigned int>(storage_->view.environment_count);
    const auto blocks = (environments + threads_per_block - 1) / threads_per_block;
    validateResetMask<<<blocks, threads_per_block, 0, cuda_stream>>>(storage_->view, device_mask,
                                                                     storage_->action_status.data);
    checkCuda(cudaGetLastError());
    resetFixedWorlds<<<blocks, threads_per_block, 0, cuda_stream>>>(storage_->view, device_mask,
        storage_->fixed_reset_players.data, storage_->fixed_reset_actors.data, storage_->observation_actors.data,
        storage_->fixed_reset_episodes.data, storage_->fixed_reset_seed, storage_->reward_states.data,
        storage_->reward_finished.data, storage_->arrow_summaries.data, storage_->action_status.data);
    const auto reset_status = cudaGetLastError();
    if (reset_status != cudaSuccess) {
        cudaStreamSynchronize(cuda_stream);
        checkCuda(reset_status);
    }
    const auto status = cudaEventRecord(storage_->observation_state_ready, cuda_stream);
    if (status != cudaSuccess) {
        cudaStreamSynchronize(cuda_stream);
        checkCuda(status);
    }
}

void CudaBatch::registerRewardBuffers(float* rewards, std::uint8_t* terminated, std::uint8_t* truncated,
                                      std::uint8_t* losses, double* contributions, std::size_t count) {
    if (!rewards || !terminated || !truncated || !losses || !contributions || !storage_->reward_results.data
        || count != storage_->reward_results.count) throw std::invalid_argument("Invalid CUDA reward output buffers");
    validateDevicePointer(rewards, storage_->device);
    validateDevicePointer(terminated, storage_->device);
    validateDevicePointer(truncated, storage_->device);
    validateDevicePointer(losses, storage_->device);
    validateDevicePointer(contributions, storage_->device);
    checkCuda(cudaEventSynchronize(storage_->observation_state_ready));
    storage_->registered_rewards = rewards;
    storage_->registered_terminated = terminated;
    storage_->registered_truncated = truncated;
    storage_->registered_losses = losses;
    storage_->registered_contributions = contributions;
}

void CudaBatch::registerResetMask(const std::uint8_t* device_mask, std::size_t count) {
    if (!device_mask || !storage_->fixed_reset_players.data
        || count != static_cast<std::size_t>(storage_->view.environment_count)) {
        throw std::invalid_argument("Invalid fixed CUDA reset mask or batch shape");
    }
    validateDevicePointer(device_mask, storage_->device);
    checkCuda(cudaEventSynchronize(storage_->observation_state_ready));
    storage_->registered_reset_mask = device_mask;
}

void CudaBatch::stepRegisteredOnStream(void* stream) {
    if (!storage_->registered_actions || !storage_->registered_observation || !storage_->registered_rewards
        || !storage_->registered_terminated || !storage_->registered_truncated || !storage_->registered_losses
        || !storage_->registered_contributions) throw std::invalid_argument("Register CUDA step buffers before use");
    decodeRegisteredActionsOnStream(stream);
    stepDecodedWithRewardsOnStream(stream);
    exportRewardResultsOnStream(storage_->registered_rewards, storage_->registered_terminated,
        storage_->registered_truncated, storage_->registered_losses, storage_->registered_contributions,
        static_cast<std::size_t>(storage_->view.environment_count), stream);
    observeRegisteredOnStream(stream);
}

void CudaBatch::resetRegisteredOnStream(void* stream) {
    if (!storage_->registered_reset_mask || !storage_->registered_observation) {
        throw std::invalid_argument("Register CUDA reset and observation buffers before use");
    }
    resetFixedOnStream(storage_->registered_reset_mask,
                       static_cast<std::size_t>(storage_->view.environment_count), stream);
    observeRegisteredOnStream(stream);
}

void CudaBatch::step(int ticks) {
    stepOnStream(ticks, nullptr);
}

void CudaBatch::stepOnStream(int ticks, void* stream) {
    advanceOnStream(ticks, stream, false);
}

void CudaBatch::stepDecodedOnStream(int ticks, void* stream) {
    if (!storage_->action_status.data) throw std::invalid_argument("CUDA actions are not configured");
    advanceOnStream(ticks, stream, true);
}

void CudaBatch::stepDecodedWithRewardsOnStream(void* stream) {
    if (!storage_->reward_settings.data) throw std::invalid_argument("CUDA duel rewards are not configured");
    stepDecodedOnStream(1, stream);
    evaluateRewardsOnStream(stream);
}

void CudaBatch::advanceOnStream(int ticks, void* stream, bool check_action_status) {
    if (ticks <= 0) {
        throw std::invalid_argument("Tick count must be positive");
    }
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    if (storage_->observation_state_ready) {
        rejectEventManagedGraphCapture(cuda_stream);
        checkCuda(cudaStreamWaitEvent(cuda_stream, storage_->observation_state_ready, 0));
        if (storage_->observation_read_pending) {
            checkCuda(cudaStreamWaitEvent(cuda_stream, storage_->observation_read_done, 0));
        }
    }
    const auto count = static_cast<unsigned int>(storage_->view.environment_count);
    if (check_action_status && storage_->reward_finished.data) {
        validateFinishedWorlds<<<(count + threads_per_block - 1) / threads_per_block, threads_per_block, 0,
                                 cuda_stream>>>(storage_->view, storage_->reward_finished.data,
                                                storage_->action_status.data);
        checkCuda(cudaGetLastError());
    }
    for (int tick = 0; tick < ticks; ++tick) {
        auto* summaries = tick == ticks - 1 ? storage_->arrow_summaries.data : nullptr;
        advanceEnvironments<<<(count + threads_per_block - 1) / threads_per_block, threads_per_block, 0,
                              cuda_stream>>>(
            storage_->view, summaries, storage_->observation_actors.data,
            check_action_status ? storage_->action_status.data : nullptr);
        checkCuda(cudaGetLastError());
    }
    if (storage_->observation_state_ready) {
        const auto status = cudaEventRecord(storage_->observation_state_ready, cuda_stream);
        if (status != cudaSuccess) {
            cudaStreamSynchronize(cuda_stream);
            checkCuda(status);
        }
    }
}

void CudaBatch::observeBaseOnStream(float* device_output, std::size_t count, void* stream) const {
    if (!device_output || storage_->view.players_per_environment != 2
        || count != checkedProduct(storage_->players.count, std::size_t(22))) {
        throw std::invalid_argument("Invalid base observation buffer or batch shape");
    }
    const auto players = static_cast<unsigned int>(storage_->players.count);
    observeBaseEnvironments<<<(players + threads_per_block - 1) / threads_per_block, threads_per_block, 0,
                              static_cast<cudaStream_t>(stream)>>>(storage_->view, device_output);
    checkCuda(cudaGetLastError());
}

void CudaBatch::observeConfiguredOnStream(float* device_output, std::size_t count, void* stream) const {
    if (!device_output || !storage_->observation_actors.data || !storage_->observation_projection.data
        || count != checkedProduct(storage_->players.count, storage_->observation_projection.count)) {
        throw std::invalid_argument("Invalid configured observation buffer or batch shape");
    }
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    rejectEventManagedGraphCapture(cuda_stream);
    checkCuda(cudaStreamWaitEvent(cuda_stream, storage_->observation_state_ready, 0));
    if (storage_->observation_read_pending) {
        checkCuda(cudaStreamWaitEvent(cuda_stream, storage_->observation_read_done, 0));
    }
    const auto players = static_cast<unsigned int>(storage_->players.count);
    observeConfiguredEnvironments<<<(players + threads_per_block - 1) / threads_per_block, threads_per_block, 0,
                                    cuda_stream>>>(storage_->view,
        storage_->observation_actors.data, storage_->observation_projection.data,
        static_cast<int>(storage_->observation_projection.count), device_output);
    checkCuda(cudaGetLastError());
    const auto status = cudaEventRecord(storage_->observation_read_done, cuda_stream);
    if (status != cudaSuccess) {
        cudaStreamSynchronize(cuda_stream);
        checkCuda(status);
    }
    storage_->observation_read_pending = true;
}

void CudaBatch::observeBaseSynchronized(float* device_output, std::size_t count, void* stream) const {
    if (!device_output || storage_->view.players_per_environment != 2
        || count != checkedProduct(storage_->players.count, std::size_t(22))) {
        throw std::invalid_argument("Invalid base observation buffer or batch shape");
    }
    validateDevicePointer(device_output, storage_->device);
    checkCuda(cudaDeviceSynchronize());
    observeBaseOnStream(device_output, count, stream);
    checkCuda(cudaDeviceSynchronize());
}

void CudaBatch::observeConfiguredSynchronized(float* device_output, std::size_t count, void* stream) const {
    if (!device_output || !storage_->observation_actors.data || !storage_->observation_projection.data
        || count != checkedProduct(storage_->players.count, storage_->observation_projection.count)) {
        throw std::invalid_argument("Invalid configured observation buffer or batch shape");
    }
    validateDevicePointer(device_output, storage_->device);
    checkCuda(cudaDeviceSynchronize());
    observeConfiguredOnStream(device_output, count, stream);
    checkCuda(cudaDeviceSynchronize());
}

void CudaBatch::registerConfiguredObservationBuffer(float* device_output, std::size_t count) {
    if (!device_output || !storage_->observation_actors.data || !storage_->observation_projection.data
        || count != checkedProduct(storage_->players.count, storage_->observation_projection.count)) {
        throw std::invalid_argument("Invalid configured observation buffer or batch shape");
    }
    validateDevicePointer(device_output, storage_->device);
    if (storage_->observation_read_pending) checkCuda(cudaEventSynchronize(storage_->observation_read_done));
    storage_->registered_observation = device_output;
    storage_->registered_observation_count = count;
}

void CudaBatch::observeRegisteredOnStream(void* stream) const {
    if (!storage_->registered_observation) {
        throw std::invalid_argument("Register a CUDA observation buffer before use");
    }
    validateCurrentDevice(storage_->device);
    observeConfiguredOnStream(storage_->registered_observation, storage_->registered_observation_count, stream);
}

void CudaBatch::waitForConfiguredObservations() {
    validateCurrentDevice(storage_->device);
    if (storage_->observation_read_pending) {
        checkCuda(cudaEventSynchronize(storage_->observation_read_done));
        storage_->observation_read_pending = false;
    }
}

void CudaBatch::synchronize() const {
    checkCuda(cudaDeviceSynchronize());
}

void CudaBatch::resetEnvironments(const int* ids, std::size_t count, const PlayerState* players,
                                  const ProjectileWorldState* projectile_states,
                                  const app::ObservationActorMetadata* observation_metadata) {
    if (count == 0) {
        return;
    }
    if (!ids || !players || count > storage_->reset_ids.count
        || (storage_->projectile_states.count != 0 && !projectile_states)
        || (storage_->observation_actors.count != 0 && !observation_metadata)) {
        throw std::invalid_argument("Invalid reset batch");
    }
    if (storage_->observation_state_ready) checkCuda(cudaEventSynchronize(storage_->observation_state_ready));
    if (storage_->observation_read_pending) checkCuda(cudaEventSynchronize(storage_->observation_read_done));
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
    if (storage_->observation_actors.count != 0) {
        storage_->reset_observation_actors.upload(observation_metadata,
            count * static_cast<std::size_t>(storage_->view.players_per_environment));
    }
    resetSelected<<<(static_cast<unsigned int>(count) + threads_per_block - 1) / threads_per_block,
                    threads_per_block>>>(storage_->view, storage_->reset_ids.data,
                                        static_cast<int>(count), storage_->reset_players.data,
                                        storage_->projectile_states.count != 0
                                            ? storage_->reset_projectile_states.data : nullptr,
                                        storage_->arrow_summaries.data, storage_->observation_actors.data,
                                        storage_->reset_observation_actors.data, storage_->reward_states.data,
                                        storage_->reward_finished.data);
    checkCuda(cudaGetLastError());
    if (storage_->observation_state_ready) checkCuda(cudaEventRecord(storage_->observation_state_ready, nullptr));
}

void CudaBatch::download(BatchView destination) const {
    validateHostBatch(destination);
    if (destination.environment_count != storage_->view.environment_count
        || destination.players_per_environment != storage_->view.players_per_environment) {
        throw std::invalid_argument("Snapshot batch size mismatch");
    }
    if (storage_->observation_state_ready) checkCuda(cudaEventSynchronize(storage_->observation_state_ready));
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
    if (storage_->observation_state_ready) checkCuda(cudaEventSynchronize(storage_->observation_state_ready));
    storage_->players.download(destination.players);
    storage_->worlds.download(destination.worlds);
    if (arrow_summaries) {
        storage_->arrow_summaries.download(arrow_summaries);
    }
}

}
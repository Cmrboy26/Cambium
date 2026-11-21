#pragma once

#include "minecraft_sim/bridge_match.hpp"
#include "minecraft_sim/bridge_terrain_generation.hpp"
#include "minecraft_sim/terrain_budget.hpp"

#include <cstddef>
#include <memory>

namespace minecraft_sim {

namespace app {
struct ActionHead;
struct DuelRewardSettings;
struct DuelRewardState;
struct DuelRewardStep;
struct ObservationProjection;
struct ObservationActorMetadata;
}

class CudaBatch {
public:
    static constexpr unsigned int threads_per_block = 128;

    explicit CudaBatch(BatchView initial, bool summarize_arrows = false,
                       std::size_t terrain_budget = terrainBudgetBytes());
    ~CudaBatch();
    CudaBatch(const CudaBatch&) = delete;
    CudaBatch& operator=(const CudaBatch&) = delete;

    static bool available();
    BatchView deviceView() const;
    void validateDeviceBuffer(const void* device_buffer) const;
    void uploadControls(const Controls* controls, std::size_t count);
    void configureBridge(const BridgeMatchRules& rules, const BridgeInitialPlayers* initial,
                         const BridgeMatchState* states, std::size_t count, int external_cutoff);
    void downloadBridge(BridgeMatchState* states, BridgeTickEvents* events, std::size_t count) const;
    void resetEnvironments(const int* ids, std::size_t count, const PlayerState* players,
                           const ProjectileWorldState* projectile_states = nullptr,
                           const app::ObservationActorMetadata* observation_metadata = nullptr,
                           const BridgeTerrainGeneration* terrain_generation = nullptr);
    void configureObservations(const app::ObservationProjection* projection, std::size_t features,
                               const app::ObservationActorMetadata* actors, std::size_t count);
    void configureActions(const app::ActionHead* heads, std::size_t count);
    void decodeActionsOnStream(const std::int32_t* device_actions, std::size_t count, void* stream);
    void stepDecodedOnStream(int ticks, void* stream);
    void registerActionBuffer(const std::int32_t* device_actions, std::size_t count);
    void decodeRegisteredActionsOnStream(void* stream);
    int actionStatus() const;
    void clearActionStatus();
    void configureRewards(const app::DuelRewardSettings& settings,
                          const app::DuelRewardState* states, std::size_t count,
                          const std::uint8_t* finished = nullptr);
    void evaluateRewardsOnStream(void* stream);
    void downloadRewardResults(app::DuelRewardStep* output, std::size_t count) const;
    void exportRewardResultsOnStream(float* rewards, std::uint8_t* terminated, std::uint8_t* truncated,
                                     std::uint8_t* losses, double* contributions, std::size_t count, void* stream);
    void stepDecodedWithRewardsOnStream(void* stream);
    void configureFixedResets(const PlayerState* players, const app::ObservationActorMetadata* actors,
                              const std::uint64_t* episodes, std::size_t count, std::uint64_t seed);
    void downloadFixedResetEpisodes(std::uint64_t* episodes, std::size_t count) const;
    void uploadFixedResetEpisodes(const std::uint64_t* episodes, std::size_t count);
    void resetFixedOnStream(const std::uint8_t* device_mask, std::size_t count, void* stream);
    void registerRewardBuffers(float* rewards, std::uint8_t* terminated, std::uint8_t* truncated,
                               std::uint8_t* losses, double* contributions, std::size_t count);
    void registerResetMask(const std::uint8_t* device_mask, std::size_t count);
    void stepRegisteredOnStream(void* stream);
    void resetRegisteredOnStream(void* stream);
    void step(int ticks = 1);
    void stepOnStream(int ticks, void* stream);
    void observeBaseOnStream(float* device_output, std::size_t count, void* stream) const;
    void observeConfiguredOnStream(float* device_output, std::size_t count, void* stream) const;
    void observeBaseSynchronized(float* device_output, std::size_t count, void* stream) const;
    void observeConfiguredSynchronized(float* device_output, std::size_t count, void* stream) const;
    void registerConfiguredObservationBuffer(float* device_output, std::size_t count);
    void observeRegisteredOnStream(void* stream) const;
    void waitForConfiguredObservations();
    void synchronize() const;
    void download(BatchView destination) const;
    void downloadForTraining(BatchView destination, ArrowObservationSummary* arrow_summaries = nullptr) const;

private:
    void advanceOnStream(int ticks, void* stream, bool check_action_status);
    struct Storage;
    std::unique_ptr<Storage> storage_;
};

}
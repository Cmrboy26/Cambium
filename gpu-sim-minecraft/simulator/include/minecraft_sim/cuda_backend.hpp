#pragma once

#include "minecraft_sim/projectiles.hpp"

#include <cstddef>
#include <memory>

namespace minecraft_sim {

namespace app {
struct ObservationProjection;
struct ObservationActorMetadata;
}

class CudaBatch {
public:
    static constexpr unsigned int threads_per_block = 128;

    explicit CudaBatch(BatchView initial, bool summarize_arrows = false);
    ~CudaBatch();
    CudaBatch(const CudaBatch&) = delete;
    CudaBatch& operator=(const CudaBatch&) = delete;

    static bool available();
    BatchView deviceView() const;
    void uploadControls(const Controls* controls, std::size_t count);
    void resetEnvironments(const int* ids, std::size_t count, const PlayerState* players,
                           const ProjectileWorldState* projectile_states = nullptr,
                           const app::ObservationActorMetadata* observation_metadata = nullptr);
    void configureObservations(const app::ObservationProjection* projection, std::size_t features,
                               const app::ObservationActorMetadata* actors, std::size_t count);
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
    struct Storage;
    std::unique_ptr<Storage> storage_;
};

}
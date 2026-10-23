#pragma once

#include "minecraft_sim/core.hpp"

#include <cstddef>
#include <memory>

namespace minecraft_sim {

class CudaBatch {
public:
    static constexpr unsigned int threads_per_block = 128;

    explicit CudaBatch(BatchView initial);
    ~CudaBatch();
    CudaBatch(const CudaBatch&) = delete;
    CudaBatch& operator=(const CudaBatch&) = delete;

    static bool available();
    BatchView deviceView() const;
    void uploadControls(const Controls* controls, std::size_t count);
    void resetEnvironments(const int* ids, std::size_t count, const PlayerState* players,
                           const ProjectileWorldState* projectile_states = nullptr);
    void step(int ticks = 1);
    void synchronize() const;
    void download(BatchView destination) const;

private:
    struct Storage;
    std::unique_ptr<Storage> storage_;
};

}
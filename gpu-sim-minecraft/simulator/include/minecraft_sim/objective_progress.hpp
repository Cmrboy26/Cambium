#pragma once

#include "minecraft_sim/geometry.hpp"

namespace minecraft_sim {

struct ObjectiveProgressDelta {
    double toward = 0.0;
    double away = 0.0;
};

struct ObjectiveProgress {
    Vec3 target{};
    double previous = 0.0;
    double best = 0.0;

    MCSIM_HD double distance(Vec3 position) const {
        const Vec3 delta{target.x - position.x, target.y - position.y, target.z - position.z};
        return ::sqrt(delta.lengthSquared());
    }

    MCSIM_HD void reset(Vec3 position, Vec3 objective) {
        target = objective;
        previous = distance(position);
        best = previous;
    }

    MCSIM_HD ObjectiveProgressDelta advance(Vec3 position, bool discontinuity, bool completed) {
        const double current = distance(position);
        ObjectiveProgressDelta result;
        if (!discontinuity) {
            result.toward = maximum(best - current, 0.0);
            result.away = maximum(current - previous, 0.0);
            best = minimum(best, current);
        }
        if (completed) best = current;
        previous = current;
        return result;
    }
};

}
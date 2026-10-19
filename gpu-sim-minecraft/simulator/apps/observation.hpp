#pragma once

#include "configuration.hpp"
#include <cmath>

namespace minecraft_sim::app {

inline std::array<float, 22> baseObservation(const PlayerState& source, const PlayerState& target,
                                           const std::array<float, 8>& rays) {
    constexpr double radians = 0.017453292519943295;
    const auto wrap = [](double angle) { return angle - 360.0 * std::floor((angle + 180.0) / 360.0); };
    const Vec3 relative{target.position.x - source.position.x, target.position.y - source.position.y,
                        target.position.z - source.position.z};
    const double distance = std::sqrt(relative.lengthSquared());
    std::array<float, 22> result{};
    result[0] = source.health / 20.0F;
    result[1] = static_cast<float>(wrap(source.yaw) / 180.0);
    result[2] = source.pitch / 90.0F;
    result[3] = target.alive() ? 1.0F : 0.0F;
    result[4] = target.health / 20.0F;
    result[5] = static_cast<float>(relative.x / 10.0);
    result[6] = static_cast<float>(relative.y / 10.0);
    result[7] = static_cast<float>(relative.z / 10.0);
    result[8] = static_cast<float>(wrap(target.yaw) / 180.0);
    result[9] = target.pitch / 90.0F;
    std::copy(rays.begin(), rays.end(), result.begin() + 10);
    result[18] = static_cast<float>(minimum(distance / 10.0, 1.0));
    if (distance > 0.01) {
        const double yaw = source.yaw * radians;
        const double pitch = source.pitch * radians;
        result[19] = static_cast<float>((-std::sin(yaw) * std::cos(pitch) * relative.x
            - std::sin(pitch) * relative.y + std::cos(yaw) * std::cos(pitch) * relative.z) / distance);
        result[20] = static_cast<float>(wrap(std::atan2(-relative.x, relative.z) / radians - source.yaw) / 180.0);
        result[21] = static_cast<float>((std::atan2(-relative.y,
            maximum(std::hypot(relative.x, relative.z), 0.01)) / radians - source.pitch) / 90.0);
    }
    return result;
}

}
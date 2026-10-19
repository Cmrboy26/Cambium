#include "minecraft_sim/comparison.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace minecraft_sim {

ErrorDifference ErrorStatistics::observe(double simulated, double recorded, std::uint64_t tick,
                                         double tolerance, double percentage_floor) {
    ErrorDifference error{simulated - recorded, std::abs(simulated - recorded), std::nullopt};
    if (!std::isfinite(simulated) || !std::isfinite(recorded) || !std::isfinite(error.absolute)
        || !std::isfinite(tolerance) || tolerance < 0.0
        || !std::isfinite(percentage_floor) || percentage_floor <= 0.0) {
        throw std::invalid_argument("Error statistics require finite values and valid thresholds");
    }
    if (count == 0 || error.absolute > absolute_max) {
        maximum_tick = tick;
    }
    ++count;
    absolute_sum += error.absolute;
    absolute_min = (std::min)(absolute_min, error.absolute);
    absolute_max = (std::max)(absolute_max, error.absolute);
    if (error.absolute > tolerance && !first_violation) {
        first_violation = tick;
    }
    if (std::abs(recorded) > percentage_floor) {
        error.percentage = error.absolute / std::abs(recorded) * 100.0;
        if (!std::isfinite(*error.percentage)) {
            throw std::overflow_error("Percentage error overflow");
        }
        ++percentage_count;
        percentage_sum += *error.percentage;
        percentage_min = (std::min)(percentage_min, *error.percentage);
        percentage_max = (std::max)(percentage_max, *error.percentage);
    }
    return error;
}

double ErrorStatistics::meanAbsolute() const {
    return count != 0 ? absolute_sum / static_cast<double>(count) : 0.0;
}

std::optional<double> ErrorStatistics::meanPercentage() const {
    return percentage_count != 0
        ? std::optional<double>(percentage_sum / static_cast<double>(percentage_count))
        : std::nullopt;
}

}
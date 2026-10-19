#pragma once

#include "minecraft_sim/core.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace minecraft_sim {

struct ErrorDifference {
    double difference = 0.0;
    double absolute = 0.0;
    std::optional<double> percentage;
};

struct ErrorStatistics {
    std::size_t count = 0;
    std::size_t percentage_count = 0;
    double absolute_sum = 0.0;
    double absolute_min = (std::numeric_limits<double>::infinity)();
    double absolute_max = 0.0;
    double percentage_sum = 0.0;
    double percentage_min = (std::numeric_limits<double>::infinity)();
    double percentage_max = 0.0;
    std::uint64_t maximum_tick = 0;
    std::optional<std::uint64_t> first_violation;

    ErrorDifference observe(double simulated, double recorded, std::uint64_t tick,
                            double tolerance, double percentage_floor);
    double meanAbsolute() const;
    std::optional<double> meanPercentage() const;
};

struct RecordedSample {
    std::uint64_t tick = 0;
    Controls controls;
    Vec3 position;
    float yaw = 0.0F;
    float pitch = 0.0F;
    bool on_ground = false;
    std::optional<Vec3> velocity;
    std::optional<PlayerState> movement_state;
};

struct ComparisonOptions {
    double tolerance = 1.0E-9;
    double percentage_floor = 1.0E-9;
    bool previous_inputs = false;
    std::optional<Vec3> initial_velocity;
};

inline constexpr std::array<std::string_view, 9> comparison_fields{
    "x", "y", "z", "vx", "vy", "vz", "dx", "dy", "dz"
};

inline constexpr std::array<std::string_view, 3> movement_flags{"sprinting", "crouching", "pose"};

struct ComparisonReport {
    std::array<ErrorStatistics, comparison_fields.size()> fields;
    std::size_t compared_ticks = 0;
    std::size_t ground_mismatches = 0;
    std::optional<std::uint64_t> first_ground_mismatch;
    Vec3 initial_velocity;
    bool has_reference_velocity = false;
    bool has_movement_state = false;
    std::array<std::size_t, movement_flags.size()> movement_mismatches{};
    std::array<std::optional<std::uint64_t>, movement_flags.size()> first_movement_mismatch;
};

std::vector<RecordedSample> readRecording(const std::string& path);
std::vector<RecordedSample> readRecording(std::istream& input);
ComparisonReport compareRecording(const std::vector<RecordedSample>& samples, WorldView world,
                                  std::ostream& details, const ComparisonOptions& options);
void writeComparisonSummary(std::ostream& output, const ComparisonReport& report);

}
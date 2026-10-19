#include "minecraft_sim/comparison.hpp"

#include <csv.h>

#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace minecraft_sim {
namespace {

template<typename Number>
Number parseNumber(const std::string& text, std::string_view field, std::size_t row) {
    Number value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()
        || !std::isfinite(static_cast<double>(value))) {
        throw std::invalid_argument("Invalid " + std::string(field) + " at data row "
                                    + std::to_string(row));
    }
    return value;
}

bool parseButton(const std::string& text, std::string_view field, std::size_t row) {
    if (text != "0" && text != "1") {
        throw std::invalid_argument("Expected 0 or 1 for " + std::string(field)
                                    + " at data row " + std::to_string(row));
    }
    return text == "1";
}

std::array<double, 9> fieldValues(Vec3 position, Vec3 velocity, Vec3 previous) {
    return {position.x, position.y, position.z, velocity.x, velocity.y, velocity.z,
            position.x - previous.x, position.y - previous.y, position.z - previous.z};
}

}

std::vector<RecordedSample> readRecording(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::invalid_argument("Cannot open recording: " + path);
    }
    return readRecording(input);
}

std::vector<RecordedSample> readRecording(std::istream& input) {
    constexpr std::array<const char*, 31> columns{
        "tick", "input", "forward", "backward", "left", "right", "jump", "sneak", "sprint",
        "x", "y", "z", "yaw", "pitch", "on_ground", "vx", "vy", "vz", "sprinting", "crouching",
        "pose", "previous_forward", "previous_sneak", "sprint_trigger_time", "sprint_window",
        "food_level", "may_fly", "fall_distance", "jump_delay", "horizontal_collision", "minor_horizontal_collision"
    };
    io::CSVReader<31, io::trim_chars<>, io::double_quote_escape<',', '"'>> reader("recording", input);
    reader.read_header(io::ignore_extra_column | io::ignore_missing_column,
        columns[0], columns[1], columns[2], columns[3], columns[4], columns[5],
        columns[6], columns[7], columns[8], columns[9], columns[10], columns[11],
        columns[12], columns[13], columns[14], columns[15], columns[16], columns[17],
        columns[18], columns[19], columns[20], columns[21], columns[22], columns[23],
        columns[24], columns[25], columns[26], columns[27], columns[28], columns[29], columns[30]);
    for (std::size_t index = 0; index < 15; ++index) {
        if (!reader.has_column(columns[index])) {
            throw std::invalid_argument("Missing required CSV column: " + std::string(columns[index]));
        }
    }
    const bool velocity = reader.has_column("vx");
    if (reader.has_column("vy") != velocity || reader.has_column("vz") != velocity) {
        throw std::invalid_argument("Reference velocity requires all three columns: vx, vy, vz");
    }
    bool movement_state = reader.has_column("minor_horizontal_collision");
    for (std::size_t index = 20; index <= 27; ++index) {
        movement_state = movement_state || reader.has_column(columns[index]);
    }
    if (movement_state) {
        for (std::size_t index = 15; index < columns.size(); ++index) {
            if (!reader.has_column(columns[index])) {
                throw std::invalid_argument("Incomplete movement-state CSV; missing " + std::string(columns[index]));
            }
        }
    }
    std::vector<RecordedSample> samples;
    std::array<std::string, 31> values;
    while (reader.read_row(values[0], values[1], values[2], values[3], values[4], values[5],
                          values[6], values[7], values[8], values[9], values[10], values[11],
                          values[12], values[13], values[14], values[15], values[16], values[17],
                          values[18], values[19], values[20], values[21], values[22], values[23],
                          values[24], values[25], values[26], values[27], values[28], values[29], values[30])) {
        const auto row = samples.size() + 1;
        RecordedSample sample;
        sample.tick = parseNumber<std::uint64_t>(values[0], "tick", row);
        if (!samples.empty()
            && (samples.back().tick == (std::numeric_limits<std::uint64_t>::max)()
                || sample.tick != samples.back().tick + 1)) {
            throw std::invalid_argument("Ticks must be consecutive and ordered at data row "
                                        + std::to_string(row));
        }
        std::array<bool, 7> buttons{};
        unsigned decoded_mask = 0;
        for (std::size_t index = 0; index < buttons.size(); ++index) {
            buttons[index] = parseButton(values[index + 2], columns[index + 2], row);
            decoded_mask |= static_cast<unsigned>(buttons[index]) << index;
        }
        if (parseNumber<unsigned>(values[1], "input", row) != decoded_mask) {
            throw std::invalid_argument("Input mask disagrees with buttons at data row "
                                        + std::to_string(row));
        }
        sample.controls = {buttons[0], buttons[1], buttons[2], buttons[3], buttons[4]};
        sample.controls.sneak = buttons[5];
        sample.controls.sprint = buttons[6];
        sample.position = {parseNumber<double>(values[9], "x", row),
                           parseNumber<double>(values[10], "y", row),
                           parseNumber<double>(values[11], "z", row)};
        sample.yaw = parseNumber<float>(values[12], "yaw", row);
        sample.pitch = parseNumber<float>(values[13], "pitch", row);
        if (sample.pitch < -90.0F || sample.pitch > 90.0F) {
            throw std::invalid_argument("Pitch outside [-90,90] at data row " + std::to_string(row));
        }
        sample.on_ground = parseButton(values[14], "on_ground", row);
        if (velocity) {
            sample.velocity = Vec3{parseNumber<double>(values[15], "vx", row),
                                   parseNumber<double>(values[16], "vy", row),
                                   parseNumber<double>(values[17], "vz", row)};
        }
        if (movement_state) {
            PlayerState state;
            state.sprinting = parseButton(values[18], "sprinting", row);
            state.crouching = parseButton(values[19], "crouching", row);
            const int pose = parseNumber<int>(values[20], "pose", row);
            state.previous_forward = parseButton(values[21], "previous_forward", row);
            state.previous_sneak = parseButton(values[22], "previous_sneak", row);
            state.sprint_trigger_time = parseNumber<int>(values[23], "sprint_trigger_time", row);
            state.sprint_window = parseNumber<int>(values[24], "sprint_window", row);
            state.food_level = parseNumber<int>(values[25], "food_level", row);
            state.may_fly = parseButton(values[26], "may_fly", row);
            state.fall_distance = parseNumber<double>(values[27], "fall_distance", row);
            state.jump_delay = parseNumber<int>(values[28], "jump_delay", row);
            state.horizontal_collision = parseButton(values[29], "horizontal_collision", row);
            state.minor_horizontal_collision = parseButton(values[30], "minor_horizontal_collision", row);
            if (pose < 0 || pose > 2 || state.sprint_trigger_time < 0 || state.sprint_window < 0
                || state.food_level < 0 || state.food_level > 20 || state.fall_distance < 0.0 || state.jump_delay < 0) {
                throw std::invalid_argument("Movement state outside supported range at data row " + std::to_string(row));
            }
            state.pose = static_cast<PlayerPose>(pose);
            sample.movement_state = state;
        }
        samples.push_back(sample);
    }
    if (samples.size() < 2) {
        throw std::invalid_argument("Recording needs an initial row and at least one predicted row");
    }
    return samples;
}

ComparisonReport compareRecording(const std::vector<RecordedSample>& samples, WorldView world,
                                  std::ostream& details, const ComparisonOptions& options) {
    if (samples.size() < 2 || !std::isfinite(options.tolerance) || options.tolerance < 0.0
        || !std::isfinite(options.percentage_floor) || options.percentage_floor <= 0.0) {
        throw std::invalid_argument("Invalid comparison samples or thresholds");
    }
    const auto& initial = samples.front();
    const auto& controls = initial.controls;
    if (controls.jump && !initial.movement_state) {
        throw std::invalid_argument("Start recording with jump released; initial jump delay is unknown");
    }
    ComparisonReport report;
    report.has_reference_velocity = initial.velocity.has_value();
    report.has_movement_state = initial.movement_state.has_value();
    if (options.initial_velocity) {
        report.initial_velocity = *options.initial_velocity;
    } else if (initial.velocity) {
        report.initial_velocity = *initial.velocity;
    } else {
        if (!initial.on_ground || controls.forward || controls.backward || controls.left || controls.right) {
            throw std::invalid_argument("Without reference velocity, start neutral/grounded or supply --initial-velocity");
        }
        report.initial_velocity = {0.0, -0.08 * double(0.98F), 0.0};
    }
    if (!std::isfinite(report.initial_velocity.x) || !std::isfinite(report.initial_velocity.y)
        || !std::isfinite(report.initial_velocity.z)) {
        throw std::invalid_argument("Initial velocity must be finite");
    }
    auto& state = world.player(0).state();
    state = initial.movement_state.value_or(PlayerState{});
    state.position = initial.position;
    state.velocity = report.initial_velocity;
    state.yaw = initial.yaw;
    state.pitch = initial.pitch;
    state.on_ground = initial.on_ground;
    state.tick = initial.tick;

    details << std::setprecision(17) << "tick,input_tick";
    for (const auto name : comparison_fields) {
        details << ',' << name << "_recorded," << name << "_simulated," << name << "_difference,"
                << name << "_absolute_error," << name << "_percent_error";
    }
    details << ",on_ground_recorded,on_ground_simulated,on_ground_mismatch,yaw_input,pitch_input";
    if (report.has_movement_state) {
        for (const auto name : movement_flags) {
            details << ',' << name << "_recorded," << name << "_simulated," << name << "_mismatch";
        }
    }
    details << '\n';
    for (std::size_t index = 1; index < samples.size(); ++index) {
        const auto& sample = samples[index];
        const auto& previous = samples[index - 1];
        if (previous.tick == (std::numeric_limits<std::uint64_t>::max)()
            || sample.tick != previous.tick + 1
            || sample.velocity.has_value() != report.has_reference_velocity
            || sample.movement_state.has_value() != report.has_movement_state) {
            throw std::invalid_argument("Inconsistent recording ticks or velocity columns");
        }
        const auto& action = options.previous_inputs ? previous : sample;
        const auto previous_position = state.position;
        state.yaw = action.yaw;
        state.pitch = action.pitch;
        world.player(0).setInputs(action.controls);
        world.processTick();
        const auto recorded = fieldValues(sample.position, sample.velocity.value_or(Vec3{}), previous.position);
        const auto simulated = fieldValues(state.position, state.velocity, previous_position);
        details << sample.tick << ',' << action.tick;
        for (std::size_t field = 0; field < comparison_fields.size(); ++field) {
            if (field >= 3 && field <= 5 && !report.has_reference_velocity) {
                details << ",," << simulated[field] << ",,,";
                continue;
            }
            const auto error = report.fields[field].observe(simulated[field], recorded[field], sample.tick,
                                                            options.tolerance, options.percentage_floor);
            details << ',' << recorded[field] << ',' << simulated[field] << ',' << error.difference
                    << ',' << error.absolute << ',';
            if (error.percentage) {
                details << *error.percentage;
            }
        }
        const bool ground_mismatch = sample.on_ground != state.on_ground;
        ++report.compared_ticks;
        if (ground_mismatch) {
            ++report.ground_mismatches;
            if (!report.first_ground_mismatch) {
                report.first_ground_mismatch = sample.tick;
            }
        }
        details << ',' << sample.on_ground << ',' << state.on_ground << ',' << ground_mismatch
                << ',' << action.yaw << ',' << action.pitch;
        if (sample.movement_state) {
            const auto& reference = *sample.movement_state;
            const std::array<int, 3> recorded_flags{reference.sprinting, reference.crouching, static_cast<int>(reference.pose)};
            const std::array<int, 3> simulated_flags{state.sprinting, state.crouching, static_cast<int>(state.pose)};
            for (std::size_t flag = 0; flag < movement_flags.size(); ++flag) {
                const bool mismatch = recorded_flags[flag] != simulated_flags[flag];
                details << ',' << recorded_flags[flag] << ',' << simulated_flags[flag] << ',' << mismatch;
                if (mismatch) {
                    ++report.movement_mismatches[flag];
                    if (!report.first_movement_mismatch[flag]) {
                        report.first_movement_mismatch[flag] = sample.tick;
                    }
                }
            }
        }
        details << '\n';
    }
    return report;
}

void writeComparisonSummary(std::ostream& output, const ComparisonReport& report) {
    output << std::setprecision(17)
           << "field,unit,samples,mean_abs,min_abs,max_abs,mean_pct,min_pct,max_pct,"
                  "pct_valid,pct_excluded,max_abs_tick,first_violation_tick,mismatches,mismatch_rate_pct\n";
    for (std::size_t index = 0; index < comparison_fields.size(); ++index) {
        const auto& field = report.fields[index];
        output << comparison_fields[index] << ',' << (index < 3 ? "blocks" : "blocks/tick")
               << ',' << field.count << ',';
        if (field.count != 0) {
            output << field.meanAbsolute() << ',' << field.absolute_min << ',' << field.absolute_max;
        } else {
            output << "NA,NA,NA";
        }
        output << ',';
        if (const auto percentage = field.meanPercentage()) {
            output << *percentage << ',' << field.percentage_min << ',' << field.percentage_max;
        } else {
            output << "NA,NA,NA";
        }
        output << ',' << field.percentage_count << ',' << field.count - field.percentage_count << ',';
        if (field.count != 0) {
            output << field.maximum_tick;
        }
        output << ',';
        if (field.first_violation) {
            output << *field.first_violation;
        }
        output << ",,\n";
    }
    const auto write_discrete = [&](std::string_view name, std::size_t mismatches,
                                    std::optional<std::uint64_t> first) {
        output << name << (name == "pose" ? ",enum," : ",boolean,") << report.compared_ticks
               << ",NA,NA,NA,NA,NA,NA,0," << report.compared_ticks << ",,";
        if (first) {
            output << *first;
        }
        output << ',' << mismatches << ',' << (report.compared_ticks != 0
            ? 100.0 * static_cast<double>(mismatches) / static_cast<double>(report.compared_ticks) : 0.0) << '\n';
    };
    write_discrete("on_ground", report.ground_mismatches, report.first_ground_mismatch);
    if (report.has_movement_state) {
        for (std::size_t flag = 0; flag < movement_flags.size(); ++flag) {
            write_discrete(movement_flags[flag], report.movement_mismatches[flag], report.first_movement_mismatch[flag]);
        }
    }
}

}
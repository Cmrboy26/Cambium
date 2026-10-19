#include "minecraft_sim/comparison.hpp"
#include "scene.hpp"

#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

double number(std::string_view text) {
    double value = 0.0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || !std::isfinite(value)) {
        throw std::invalid_argument("Expected a finite number, got: " + std::string(text));
    }
    return value;
}

bool sameFile(const std::filesystem::path& first, const std::filesystem::path& second) {
    return std::filesystem::weakly_canonical(first) == std::filesystem::weakly_canonical(second)
        || (std::filesystem::exists(first) && std::filesystem::exists(second)
            && std::filesystem::equivalent(first, second));
}

}

int main(int argc, char** argv) {
    try {
        minecraft_sim::ComparisonOptions options;
        std::filesystem::path input;
        std::filesystem::path prefix;
        bool force = false;
        for (int argument = 1; argument < argc; ++argument) {
            const std::string_view option = argv[argument];
            if (option == "--help") {
                std::cout << "Usage: minecraft_sim_compare recording.csv [options]\n"
                             "  --output-prefix PATH          Default: input path without .csv\n"
                             "  --input-phase current|previous Default: current\n"
                             "  --tolerance NUMBER            Absolute error threshold (default 1e-9)\n"
                             "  --percent-floor NUMBER        Exclude |reference| <= value (default 1e-9)\n"
                             "  --initial-velocity VX VY VZ   Override initial stored velocity\n"
                             "  --force                       Overwrite existing reports, never input\n"
                             "World: floor x/z=-8..7 at y=0; wall x=-2..2,y=1..3,z=2.\n"
                             "CPU replay uses shared physics; no recorded position snapping.\n";
                return 0;
            }
            if (option == "--force") {
                force = true;
            } else if (option == "--initial-velocity" && argument + 3 < argc) {
                const double velocity_x = number(argv[++argument]);
                const double velocity_y = number(argv[++argument]);
                const double velocity_z = number(argv[++argument]);
                options.initial_velocity = minecraft_sim::Vec3{velocity_x, velocity_y, velocity_z};
            } else if (option == "--output-prefix" && argument + 1 < argc) {
                prefix = argv[++argument];
                if (prefix.empty()) {
                    throw std::invalid_argument("Output prefix cannot be empty");
                }
            } else if (option == "--input-phase" && argument + 1 < argc) {
                const std::string_view phase = argv[++argument];
                if (phase != "current" && phase != "previous") {
                    throw std::invalid_argument("--input-phase must be current or previous");
                }
                options.previous_inputs = phase == "previous";
            } else if (option == "--tolerance" && argument + 1 < argc) {
                options.tolerance = number(argv[++argument]);
            } else if (option == "--percent-floor" && argument + 1 < argc) {
                options.percentage_floor = number(argv[++argument]);
            } else if (!option.starts_with("--") && input.empty()) {
                input = argv[argument];
            } else {
                throw std::invalid_argument("Unknown or incomplete option; use --help");
            }
        }
        if (input.empty() || options.tolerance < 0.0 || options.percentage_floor <= 0.0) {
            throw std::invalid_argument("Supply a recording, nonnegative tolerance and positive percent floor; use --help");
        }
        if (prefix.empty()) {
            prefix = input;
            prefix.replace_extension();
        }
        const std::filesystem::path details_path = prefix.string() + ".comparison.csv";
        const std::filesystem::path summary_path = prefix.string() + ".summary.csv";
        for (const auto& output : {details_path, summary_path}) {
            if (sameFile(input, output)) {
                throw std::invalid_argument("Report output must not replace the input recording");
            }
            if (std::filesystem::exists(output) && !force) {
                throw std::invalid_argument("Report already exists: " + output.string() + "; use --force");
            }
        }
        if (sameFile(details_path, summary_path)) {
            throw std::invalid_argument("Detail and summary reports must be different files");
        }
        const auto samples = minecraft_sim::readRecording(input.string());
        auto scene = std::make_unique<minecraft_sim::app::Scene>();
        scene->worlds[0].tick = samples.front().tick;
        std::ostringstream details;
        const auto report = minecraft_sim::compareRecording(samples, scene->view().world(0), details, options);
        std::ofstream details_output(details_path);
        std::ofstream summary(summary_path);
        if (!details_output || !summary) {
            throw std::runtime_error("Cannot open report output paths");
        }
        details_output << details.view();
        minecraft_sim::writeComparisonSummary(summary, report);
        details_output.flush();
        summary.flush();
        if (!details_output || !summary) {
            throw std::runtime_error("Failed to write comparison reports");
        }
        std::cout << std::setprecision(17)
                  << "Minecraft Java 26.3 trajectory comparison (CPU shared core)\n"
                  << "World: floor x/z=-8..7,y=0; wall x=-2..2,y=1..3,z=2; outside=loaded air\n"
                  << "Recording: " << input.string() << "\n"
                  << "Samples: " << samples.size() << "; compared transitions: " << report.compared_ticks
                  << "; ticks: " << samples.front().tick << ".." << samples.back().tick << '\n'
                  << "First row initializes state and is not scored; no later position/velocity correction.\n"
                  << "Inputs and absolute yaw/pitch before movement: "
                  << (options.previous_inputs ? "previous/source row" : "current/destination row") << '\n'
                  << "Initial velocity (" << (options.initial_velocity ? "explicit override" :
                      report.has_reference_velocity ? "recorded" : "ASSUMED settled/grounded") << "): "
                  << report.initial_velocity.x << ',' << report.initial_velocity.y << ','
                  << report.initial_velocity.z << '\n'
                  << "Initial movement bookkeeping: " << (report.has_movement_state
                      ? "recorded (pose, sprint timer, previous input, food, jump delay and fall distance)"
                      : "defaults (standing, no sprint, previous inputs released, food 20, jump delay/fall distance 0)") << '\n'
                  << "Absolute tolerance: " << options.tolerance
                  << "; percentage excluded when |reference| <= " << options.percentage_floor << '\n'
                  << "Percentages are absolute; position percentages depend on world origin.\n"
                  << "vx/vy/vz: " << (report.has_reference_velocity ? "reference columns supplied; phase must match"
                      : "reference unavailable (NA); simulation values only") << '\n'
                  << "dx/dy/dz: position displacement per tick, NOT stored velocity.\n"
                  << "Capture timing/world matching must be verified; server command-block samples can have phase ambiguity.\n\n";
        minecraft_sim::writeComparisonSummary(std::cout, report);
        std::cout << "\non_ground mismatches: " << report.ground_mismatches << '/' << report.compared_ticks
                  << " (" << 100.0 * static_cast<double>(report.ground_mismatches)
                      / static_cast<double>(report.compared_ticks) << "%)";
        if (report.first_ground_mismatch) {
            std::cout << "; first tick: " << *report.first_ground_mismatch;
        }
        std::cout << "\nDetails: " << details_path.string() << "\nSummary: " << summary_path.string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Comparison failed: " << error.what() << '\n';
        return 1;
    }
}
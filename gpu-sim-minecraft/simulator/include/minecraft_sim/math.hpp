#pragma once

#include "minecraft_sim/platform.hpp"

#include <array>
#include <cstdint>
#include <limits>

namespace minecraft_sim {

inline constexpr int sine_table_size = 65536;
inline constexpr double sine_index_scale = 10430.378350470453;
using SineTable = std::array<float, sine_table_size>;

SineTable makeSineTable();

struct LegacyRandom {
    std::uint64_t state = 25214903917ULL;

    MCSIM_HD void setSeed(std::uint64_t seed) {
        state = (seed ^ 25214903917ULL) & 281474976710655ULL;
    }

    MCSIM_HD std::uint32_t next(int bits) {
        state = (state * 25214903917ULL + 11ULL) & 281474976710655ULL;
        return static_cast<std::uint32_t>(state >> (48 - bits));
    }

    MCSIM_HD double nextDouble() {
        const auto upper = next(26);
        const auto lower = next(27);
        return static_cast<double>((std::uint64_t(upper) << 27) + lower) * 0x1.0p-53;
    }
};

struct MathView {
    const float* sine_table = nullptr;

    MCSIM_HD static std::int64_t javaLong(double value) {
        if (value != value) {
            return 0;
        }
        if (value >= 9223372036854775808.0) {
            return (std::numeric_limits<std::int64_t>::max)();
        }
        if (value <= -9223372036854775808.0) {
            return (std::numeric_limits<std::int64_t>::min)();
        }
        return static_cast<std::int64_t>(value);
    }

    MCSIM_HD float sin(double radians) const {
        const auto index = static_cast<std::uint64_t>(
            javaLong(radians * sine_index_scale)) & 65535ULL;
        return sine_table[index];
    }

    MCSIM_HD float cos(double radians) const {
        const auto index = static_cast<std::uint64_t>(
            javaLong(radians * sine_index_scale + 16384.0)) & 65535ULL;
        return sine_table[index];
    }
};

}
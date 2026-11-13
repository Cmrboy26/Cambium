#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace minecraft_sim {

inline constexpr std::size_t max_arena_cells = 262144;
inline constexpr std::size_t portable_arena_bytes_per_cell = 9;
inline constexpr std::size_t default_terrain_budget_bytes = 536870912;

inline std::size_t parseTerrainBudget(const char* value) {
    if (!value) return default_terrain_budget_bytes;
    const std::string_view text(value);
    std::uint64_t parsed = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size()
        || parsed == 0 || parsed > std::uint64_t((std::numeric_limits<std::int64_t>::max)())
        || parsed > std::uint64_t((std::numeric_limits<std::size_t>::max)())) {
        throw std::invalid_argument("MCSIM_TERRAIN_BUDGET_BYTES must be a positive decimal integer in range");
    }
    return static_cast<std::size_t>(parsed);
}

inline std::size_t terrainBudgetBytes() {
    return parseTerrainBudget(std::getenv("MCSIM_TERRAIN_BUDGET_BYTES"));
}

inline std::size_t checkedTerrainProduct(std::size_t left, std::size_t right) {
    if (right != 0 && left > (std::numeric_limits<std::size_t>::max)() / right) {
        throw std::invalid_argument("Terrain allocation size overflow");
    }
    return left * right;
}

inline std::size_t checkedTerrainSum(std::size_t left, std::size_t right) {
    if (left > (std::numeric_limits<std::size_t>::max)() - right) {
        throw std::invalid_argument("Terrain allocation size overflow");
    }
    return left + right;
}

inline void requireTerrainBudget(std::size_t bytes, std::size_t budget) {
    if (bytes > budget) throw std::invalid_argument("Terrain buffers exceed MCSIM_TERRAIN_BUDGET_BYTES");
}

}

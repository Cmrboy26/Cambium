#include "minecraft_sim/math.hpp"

#include <cmath>

namespace minecraft_sim {

SineTable makeSineTable() {
    SineTable table{};
    for (int index = 0; index < sine_table_size; ++index) {
        table[index] = static_cast<float>(std::sin(index / sine_index_scale));
    }
    return table;
}

}
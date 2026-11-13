#pragma once

#include "minecraft_sim/core.hpp"

#include <cstdint>
#include <type_traits>

namespace minecraft_sim::app {

inline constexpr int max_action_choices = 64;

struct ActionHead {
    int kind = 0;
    int count = 0;
    float values[max_action_choices]{};
};

static_assert(std::is_trivially_copyable_v<ActionHead>);

MCSIM_HD inline void applyActionValue(Controls& controls, int kind, float value, const WeaponItem* slots) {
    constexpr int forward[] = {0, 1, 1, 0, -1, -1, -1, 0, 1};
    constexpr int strafe[] = {0, 0, -1, -1, -1, 0, 1, 1, 1};
    switch (kind) {
    case 0: {
        const auto move = static_cast<int>(value);
        controls.forward = forward[move] > 0;
        controls.backward = forward[move] < 0;
        controls.left = strafe[move] > 0;
        controls.right = strafe[move] < 0;
        break;
    }
    case 1: controls.jump = value != 0; break;
    case 2: controls.sprint = value != 0; break;
    case 3: controls.attack = value != 0; break;
    case 4: controls.yaw_delta = value; break;
    case 5: controls.pitch_delta = value; break;
    case 6:
        controls.select_weapon = true;
        controls.weapon = slots[static_cast<int>(value)];
        controls.selected_slot = static_cast<int>(value);
        break;
    case 7: controls.use_item = value != 0; break;
    case 8: controls.sneak = value != 0; break;
    case 9: controls.use_item = value != 0; break;
    }
}

MCSIM_HD inline bool decodeAction(const ActionHead* heads, int head_count, const std::int32_t* indices,
                                  const WeaponItem* slots, int slot_count, Controls& output) {
    if (!heads || !indices || head_count < 1 || slot_count < 0 || slot_count > 9) return false;
    Controls result;
    for (int index = 0; index < head_count; ++index) {
        const auto& head = heads[index];
        const int choice = indices[index];
        if (head.kind < 0 || head.kind > 9 || head.count < 1 || head.count > max_action_choices
            || choice < 0 || choice >= head.count) return false;
        const float value = head.values[choice];
        if (head.kind == 0 && (!(value >= 0.0F && value < 9.0F)
            || value != static_cast<float>(static_cast<int>(value)))) return false;
        if (head.kind == 6 && (!slots || !(value >= 0.0F && value < static_cast<float>(slot_count))
            || value != static_cast<float>(static_cast<int>(value)))) return false;
        applyActionValue(result, head.kind, value, slots);
    }
    output = result;
    return true;
}

}

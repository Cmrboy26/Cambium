#pragma once

#include "minecraft_sim/math.hpp"

#include <cstdint>

namespace minecraft_sim {

enum class WeaponItem : std::uint8_t {
    unarmed, wooden_sword, stone_sword, copper_sword, golden_sword, iron_sword,
    diamond_sword, netherite_sword, wooden_axe, stone_axe, copper_axe, golden_axe,
    iron_axe, diamond_axe, netherite_axe
};

struct WeaponAttributes {
    double damage;
    double speed;
    bool sweep;
    float disable_shield_seconds;
};

MCSIM_HD inline WeaponAttributes weaponAttributes(WeaponItem item) {
    switch (item) {
    case WeaponItem::wooden_sword: return {4.0, 4.0 + double(-2.4F), true, 0.0F};
    case WeaponItem::stone_sword:
    case WeaponItem::copper_sword: return {5.0, 4.0 + double(-2.4F), true, 0.0F};
    case WeaponItem::golden_sword: return {4.0, 4.0 + double(-2.4F), true, 0.0F};
    case WeaponItem::iron_sword: return {6.0, 4.0 + double(-2.4F), true, 0.0F};
    case WeaponItem::diamond_sword: return {7.0, 4.0 + double(-2.4F), true, 0.0F};
    case WeaponItem::netherite_sword: return {8.0, 4.0 + double(-2.4F), true, 0.0F};
    case WeaponItem::wooden_axe: return {7.0, 4.0 + double(-3.2F), false, 5.0F};
    case WeaponItem::stone_axe:
    case WeaponItem::copper_axe: return {9.0, 4.0 + double(-3.2F), false, 5.0F};
    case WeaponItem::golden_axe: return {7.0, 1.0, false, 5.0F};
    case WeaponItem::iron_axe: return {9.0, 4.0 + double(-3.1F), false, 5.0F};
    case WeaponItem::diamond_axe: return {9.0, 1.0, false, 5.0F};
    case WeaponItem::netherite_axe: return {10.0, 1.0, false, 5.0F};
    default: return {1.0, 4.0, false, 0.0F};
    }
}

enum class ArmorMaterial : std::uint8_t { none, leather, copper, chainmail, iron, golden, diamond, netherite, turtle };
enum class ArmorSlot : std::uint8_t { boots, leggings, chestplate, helmet };

struct ArmorAttributes {
    float defense;
    float toughness;
    double knockback_resistance;
};

MCSIM_HD inline ArmorAttributes armorAttributes(ArmorMaterial material, ArmorSlot slot) {
    const int index = static_cast<int>(slot);
    float defense = 0.0F;
    switch (material) {
    case ArmorMaterial::leather: {
        const float values[] = {1, 2, 3, 1};
        defense = values[index];
        break;
    }
    case ArmorMaterial::copper: {
        const float values[] = {1, 3, 4, 2};
        defense = values[index];
        break;
    }
    case ArmorMaterial::chainmail: {
        const float values[] = {1, 4, 5, 2};
        defense = values[index];
        break;
    }
    case ArmorMaterial::iron: {
        const float values[] = {2, 5, 6, 2};
        defense = values[index];
        break;
    }
    case ArmorMaterial::golden: {
        const float values[] = {1, 3, 5, 2};
        defense = values[index];
        break;
    }
    case ArmorMaterial::diamond:
    case ArmorMaterial::netherite: {
        const float values[] = {3, 6, 8, 3};
        defense = values[index];
        break;
    }
    case ArmorMaterial::turtle: defense = slot == ArmorSlot::helmet ? 2.0F : 0.0F; break;
    default: break;
    }
    return {defense, material == ArmorMaterial::diamond ? 2.0F : material == ArmorMaterial::netherite ? 3.0F : 0.0F,
        material == ArmorMaterial::netherite ? double(0.1F) : 0.0};
}

}
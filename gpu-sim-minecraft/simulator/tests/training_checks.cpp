#include "minecraft_sim/training_api.h"
#include "../apps/policy_catalogue.generated.hpp"

#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using nlohmann::json;
using Handle = std::unique_ptr<void, decltype(&mcsim_destroy)>;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(std::string(message) + ": " + mcsim_last_error());
}

json scenario(void* handle, int world) {
    const auto* value = mcsim_scenario_for(handle, world);
    require(value != nullptr, "resolved world scenario");
    return json::parse(value);
}

json frame(void* handle, int world) {
    const auto* value = mcsim_replay_frame(handle, world);
    require(value != nullptr, "world replay frame");
    return json::parse(value);
}

void checkResetPlayers(const char* environment, int cuda) {
    json contract{{"format", "minecraft_sim_policy"}, {"version", 1}, {"minecraft", "26.3"},
        {"ticks_per_action", 1}, {"observations", json::array()},
        {"actions", {{{"name", "weapon"}, {"values", {0, 1}}}}}};
    for (const auto* name : minecraft_sim::app::feature_names) {
        contract["observations"].push_back({{"feature", name}});
    }
    const auto descriptor = contract.dump();
    const std::array<float, 6> weights{};
    Handle handle(mcsim_create_contract(3, cuda, environment, 100, 123, weights.data(), descriptor.c_str()), mcsim_destroy);
    require(handle != nullptr, "create all-equipment descriptor batch using a two-slot environment");
    require(mcsim_api_version() == 2, "additive functions preserve ABI 2");
    const auto baseline = scenario(handle.get(), 0);
    require(!baseline.contains("policy"), "resolved scenarios never advertise a stale embedded policy");
    auto own = baseline["players"];
    own[0]["health"] = 13;
    own[0]["loadout"] = {{"weapons", {"minecraft:wooden_sword", "minecraft:netherite_axe"}}, {"selected", 1},
        {"offhand", "minecraft:shield"}, {"armor", {{"chestplate", "minecraft:netherite_chestplate"}}}};
    own[1]["loadout"] = {{"weapons", {"minecraft:air", "minecraft:iron_sword"}}, {"selected", 0}};
    auto other = baseline["players"];
    other[0]["loadout"] = {{"weapons", {"minecraft:diamond_axe", "minecraft:golden_sword"}}, {"selected", 1}};
    other[1]["loadout"] = {{"weapons", {"minecraft:wooden_axe", "minecraft:stone_sword",
        "minecraft:copper_sword", "minecraft:golden_sword", "minecraft:iron_sword", "minecraft:diamond_sword",
        "minecraft:netherite_sword", "minecraft:stone_axe", "minecraft:netherite_axe"}}, {"selected", 8}};
    json definitions = {own, nullptr, other};
    constexpr std::array<uint8_t, 3> mask{1, 0, 1};
    constexpr std::size_t feature_count = minecraft_sim::app::feature_names.size();
    std::vector<float> observations(6 * feature_count);
    const auto reset = [&](const json& values) {
        const auto encoded = values.dump();
        return mcsim_reset_players(handle.get(), mask.data(), encoded.c_str(), observations.data());
    };
    require(reset(definitions) == 0, "reset two worlds with independent concrete players");
    const auto assigned = scenario(handle.get(), 0);
    require(assigned["players"][0]["health"] == 13 && assigned["players"][0]["loadout"]["selected"] == 1,
        "episode scenario captures sampled state and equipment");
    require(scenario(handle.get(), 1) == baseline, "unselected world's definitions stay unchanged");
    require(assigned["blocks"] == baseline["blocks"] && assigned["bounds"] == baseline["bounds"],
        "player resets preserve immutable terrain");
    require(json::parse(mcsim_scenario(handle.get())) == baseline, "legacy scenario remains the original shared definition");
    const auto observation = [&](int actor, const char* name) {
        const auto& names = minecraft_sim::app::feature_names;
        for (std::size_t index = 0; index < names.size(); ++index) {
            if (std::string(name) == names[index]) return observations[actor * feature_count + index];
        }
        throw std::runtime_error("Unknown test feature");
    };
    require(observation(0, "self.has_shield") == 1 && observation(1, "opponent.has_shield") == 1,
        "shield ownership is visible from both perspectives");
    require(observation(0, "self.weapon_damage_fraction") == 1 && observation(0, "self.weapon_slot") == 1,
        "selected reset weapon attributes and legacy slot agree");
    require(observation(1, "self.weapon_slot_0.present") == 0
        && std::abs(observation(1, "self.weapon_damage_fraction") - 0.1F) < 1e-6,
        "air slot is absent metadata but selected unarmed retains base damage");
    require(observation(5, "self.weapon_slot_8.damage_fraction") == 1,
        "last hotbar slot visible to its owner");
    const auto visible = observations;
    auto hidden_change = definitions;
    hidden_change[0][1]["loadout"]["weapons"][1] = "minecraft:netherite_axe";
    require(reset(hidden_change) == 0, "change only an opponent's unselected weapon");
    for (std::size_t feature = 0; feature < feature_count; ++feature) {
        require(observations[feature] == visible[feature], "opponent's hidden inventory cannot change the observer's encoding");
    }
    require(observation(1, "self.weapon_slot_1.damage_fraction") == 1,
        "hidden equipment changes remain observable to their owner");
    require(reset(definitions) == 0, "restore concrete definitions after visibility check");
    constexpr std::array<int32_t, 6> actions{0, 1, 0, 1, 0, 1};
    std::array<float, 6> rewards{};
    std::array<uint8_t, 3> terminated{}, truncated{};
    require(mcsim_step(handle.get(), actions.data(), observations.data(), rewards.data(),
        terminated.data(), truncated.data()) == 0, "step custom episode loadouts");
    require(frame(handle.get(), 0)["actors"][0]["weapon"] == "minecraft:wooden_sword"
        && frame(handle.get(), 0)["actors"][1]["weapon"] == "minecraft:iron_sword"
        && frame(handle.get(), 2)["actors"][0]["weapon"] == "minecraft:diamond_axe",
        "weapon action decoding uses each world's assigned hotbar");
    require(scenario(handle.get(), 0) == assigned, "scenario records episode starts, not current selections");
    require(!scenario(handle.get(), 0).contains("policy"), "player resets preserve policy-free resolved scenarios");
    require(frame(handle.get(), 0)["actors"][0]["position"].is_array()
        && frame(handle.get(), 0)["actors"][0]["position"].size() == 3,
        "replay positions use three-element coordinate arrays");

    const std::array<json, 3> before{frame(handle.get(), 0), frame(handle.get(), 1), frame(handle.get(), 2)};
    const auto rejects = [&](const json& invalid) {
        std::fill(observations.begin(), observations.end(), -42);
        require(reset(invalid) == -1 && *mcsim_last_error(), "invalid reset reports standard error");
        for (int world = 0; world < 3; ++world) require(frame(handle.get(), world) == before[world], "reset rejection is batch-atomic");
        require(scenario(handle.get(), 0) == assigned, "invalid reset cannot replace remembered definitions");
        for (float value : observations) require(value == -42, "invalid reset cannot overwrite observation output");
    };
    auto invalid = definitions;
    invalid[0][0]["health"] = 8;
    invalid[2][0]["loadout"]["weapons"] = {"minecraft:air"};
    invalid[2][0]["loadout"]["selected"] = 0;
    rejects(invalid);
    invalid = definitions; invalid[2][0]["position"][1] = 0; rejects(invalid);
    invalid = definitions; invalid[2][1]["position"] = invalid[2][0]["position"]; rejects(invalid);
    invalid = definitions; invalid[2][0]["break_block"] = true; rejects(invalid);
    invalid = definitions; invalid[2][0]["loadout"]["weapons"] = {"minecraft:air", "minecraft:air"}; rejects(invalid);
    invalid = definitions; invalid[2][0]["attack_strength_ticks"] = 1.5; rejects(invalid);
    invalid = definitions; invalid[1] = baseline["players"]; rejects(invalid);
    invalid = definitions; invalid[2] = nullptr; rejects(invalid);
    rejects(json::array());
    require(mcsim_reset_players(handle.get(), mask.data(), "{", observations.data()) == -1, "malformed JSON rejected");
    require(mcsim_reset_players(handle.get(), mask.data(), nullptr, observations.data()) == -1, "null JSON rejected");
    require(mcsim_reset_players(nullptr, mask.data(), definitions.dump().c_str(), observations.data()) == -1, "null handle rejected");
    require(mcsim_scenario_for(handle.get(), -1) == nullptr && *mcsim_last_error(), "negative scenario index rejected");
    require(mcsim_scenario_for(handle.get(), 3) == nullptr, "scenario index past world count rejected");
    require(mcsim_scenario_for(nullptr, 0) == nullptr, "null scenario handle rejected");
    constexpr std::array<uint8_t, 3> first{1, 0, 0};
    require(mcsim_reset(handle.get(), first.data(), observations.data()) == 0, "legacy masked reset accepts assigned definitions");
    require(frame(handle.get(), 0)["tick"] == 0 && frame(handle.get(), 0)["actors"][0]["weapon"] == "minecraft:netherite_axe",
        "legacy reset restores latest sampled selected weapon");
    require(frame(handle.get(), 1) == before[1] && frame(handle.get(), 2) == before[2], "legacy mask leaves other episodes untouched");
    require(scenario(handle.get(), 0) == assigned, "legacy reset remembers custom scenario");
    definitions[1] = baseline["players"];
    require(mcsim_reset_players(handle.get(), nullptr, definitions.dump().c_str(), observations.data()) == 0,
        "null mask selects every world");
    auto replacement = json::array({own, nullptr, nullptr});
    replacement[0][0]["loadout"] = {{"weapons", {"minecraft:air", "minecraft:iron_sword"}}, {"selected", 0}};
    require(mcsim_reset_players(handle.get(), first.data(), replacement.dump().c_str(), observations.data()) == 0,
        "assign a second concrete episode");
    require(observation(0, "self.has_shield") == 0 && observation(0, "self.armor_defense_fraction") == 0,
        "new assignment clears old shield and armor");
    const auto latest = scenario(handle.get(), 0);
    require(latest["players"][0]["loadout"]["weapons"][0] == "minecraft:air"
        && mcsim_reset(handle.get(), first.data(), observations.data()) == 0
        && scenario(handle.get(), 0) == latest && frame(handle.get(), 0)["actors"][0]["weapon"] == "minecraft:air",
        "legacy reset restores the most recent assignment, not the first custom episode");
    std::cout << "PASS: atomic concrete player resets, per-world replay/loadouts/decoding and legacy restoration\n";
}

void checkRewardWeights(int cuda) {
    std::array<float, 6> weights{};
    Handle handle(mcsim_create_schema(1, cuda, nullptr, 1, 1, weights.data(), 1), mcsim_destroy);
    require(handle != nullptr, "create legacy reward test batch");
    require(json::parse(mcsim_batch_info(handle.get()))["observation_dim"] == 22, "v1 dimension stays unchanged");
    weights[4] = 3;
    require(mcsim_set_reward_weights(handle.get(), weights.data()) == 0, "set curriculum reward weights");
    auto invalid = weights;
    invalid[4] = 7; invalid[5] = std::numeric_limits<float>::quiet_NaN();
    require(mcsim_set_reward_weights(handle.get(), invalid.data()) == -1, "nonfinite late reward rejected atomically");
    require(mcsim_set_reward_weights(nullptr, weights.data()) == -1
        && mcsim_set_reward_weights(handle.get(), nullptr) == -1, "null reward arguments rejected");
    constexpr std::array<int32_t, 12> actions{0, 0, 0, 0, 4, 2, 0, 0, 0, 0, 4, 2};
    std::array<float, 44> observations{};
    std::array<float, 2> rewards{};
    uint8_t terminated = 0, truncated = 0;
    require(mcsim_step(handle.get(), actions.data(), observations.data(), rewards.data(), &terminated, &truncated) == 0,
        "step curriculum reward batch");
    require(truncated == 1 && rewards[0] == 3 && rewards[1] == 3, "failed setter cannot partially change base reward weights");
    Handle v2(mcsim_create_schema(1, cuda, nullptr, 1, 1, weights.data(), 2), mcsim_destroy);
    require(v2 != nullptr && json::parse(mcsim_batch_info(v2.get()))["observation_dim"] == 31, "v2 dimension stays unchanged");
    std::cout << "PASS: finite curriculum reward replacement and legacy v1/v2 dimensions\n";
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 2 || (argc == 3 && std::string(argv[2]) == "--cuda"),
            "Usage: training_checks <two-slot-environment.json> [--cuda]");
        const int cuda = argc == 3 ? 1 : 0;
        if (cuda && !mcsim_cuda_available()) return 77;
        checkResetPlayers(argv[1], cuda);
        checkRewardWeights(cuda);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

#include "minecraft_sim/training_api.h"
#include "../apps/configuration.hpp"

#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
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
    invalid = definitions; invalid[2][0]["loadout"]["weapons"] = {"minecraft:iron_sword", "minecraft:iron_sword"}; rejects(invalid);
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

void checkStepRewardTotals(int cuda) {
    const std::array<float, 6> weights{0, 0, 0, 0, -3, -7};
    Handle handle(mcsim_create_schema(2, cuda, nullptr, 1, 1, weights.data(), 1), mcsim_destroy);
    require(handle != nullptr, "create selected-player reward batch");
    std::array<double, 12> totals{};
    require(mcsim_step_reward_totals(handle.get(), nullptr, totals.data()) == 0,
        "query initial step reward contributions");
    for (double amount : totals) require(amount == 0, "initial reward contributions are zero");
    std::array<int32_t, 24> actions{};
    for (int player = 0; player < 4; ++player) {
        actions[player * 6 + 4] = 4;
        actions[player * 6 + 5] = 2;
    }
    std::array<float, 88> observations{};
    std::array<float, 4> rewards{};
    std::array<uint8_t, 2> terminated{}, truncated{};
    require(mcsim_step(handle.get(), actions.data(), observations.data(), rewards.data(),
        terminated.data(), truncated.data()) == 0, "step selected-player reward batch");
    const std::array<uint8_t, 4> selected{1, 0, 0, 1};
    require(mcsim_step_reward_totals(handle.get(), selected.data(), totals.data()) == 0
        && totals[4] == -6 && totals[5] == -14,
        "reward mask selects one learner on either side of each world");
    require(mcsim_step_reward_totals(handle.get(), nullptr, totals.data()) == 0
        && totals[4] == -12 && totals[5] == -28, "null reward mask includes both players");
    const std::array<uint8_t, 4> empty{};
    require(mcsim_step_reward_totals(handle.get(), empty.data(), totals.data()) == 0,
        "empty reward mask is valid");
    for (double amount : totals) require(amount == 0, "empty reward selection has no contributions");
    auto invalid = selected;
    invalid[3] = 2;
    totals.fill(42);
    require(mcsim_step_reward_totals(handle.get(), invalid.data(), totals.data()) == -1,
        "nonboolean reward mask rejected");
    for (double amount : totals) require(amount == 42, "invalid reward mask leaves output untouched");
    require(mcsim_step_reward_totals(nullptr, nullptr, totals.data()) == -1
        && mcsim_step_reward_totals(handle.get(), nullptr, nullptr) == -1, "null reward arguments rejected");
    const std::array<uint8_t, 2> reset{1, 0};
    require(mcsim_reset(handle.get(), reset.data(), observations.data()) == 0
        && mcsim_step_reward_totals(handle.get(), nullptr, totals.data()) == 0
        && totals[4] == -6 && totals[5] == -14, "selective reset clears only selected worlds' last reward");
    std::cout << "PASS: masked per-step reward breakdown with explicit reset and input validation\n";
}

void checkMaxTicks(int cuda) {
    const std::array<float, 6> weights{};
    Handle handle(mcsim_create_schema(1, cuda, nullptr, 1, 1, weights.data(), 1), mcsim_destroy);
    require(handle != nullptr, "create timeout setter batch");
    require(mcsim_set_max_ticks(nullptr, 2) == -1, "null timeout handle rejected");
    require(mcsim_set_max_ticks(handle.get(), 1000000) == 0 && mcsim_set_max_ticks(handle.get(), 2) == 0,
        "timeout setter accepts supported limits");
    require(mcsim_set_max_ticks(handle.get(), 0) == -1 && mcsim_set_max_ticks(handle.get(), 1000001) == -1,
        "out-of-range timeout rejected without mutation");
    constexpr std::array<int32_t, 12> actions{0, 0, 0, 0, 4, 2, 0, 0, 0, 0, 4, 2};
    std::array<float, 44> observations{};
    std::array<float, 2> rewards{};
    uint8_t terminated = 0, truncated = 0;
    const auto terrain = scenario(handle.get(), 0)["blocks"];
    require(mcsim_step(handle.get(), actions.data(), observations.data(), rewards.data(), &terminated, &truncated) == 0
        && truncated == 0, "increased timeout preserves the first tick");
    require(mcsim_step(handle.get(), actions.data(), observations.data(), rewards.data(), &terminated, &truncated) == 0
        && truncated == 1, "updated timeout truncates at the requested tick");
    require(mcsim_set_max_ticks(handle.get(), 1) == 0
        && mcsim_reset(handle.get(), nullptr, observations.data()) == 0
        && mcsim_step(handle.get(), actions.data(), observations.data(), rewards.data(), &terminated, &truncated) == 0
        && truncated == 1, "stage-boundary reset uses replacement timeout");
    require(scenario(handle.get(), 0)["blocks"] == terrain, "timeout changes never rebuild terrain");
    std::cout << "PASS: bounded additive episode timeout replacement without terrain rebuild\n";
}

void checkEmptySlots(const char* environment, int cuda) {
    json contract{{"format", "minecraft_sim_policy"}, {"version", 1}, {"minecraft", "26.3"},
        {"ticks_per_action", 1},
        {"observations", {{{"feature", "self.weapon_slot"}}, {{"feature", "opponent.weapon_slot"}}}},
        {"actions", {{{"name", "weapon"}, {"values", {0, 1}}}}}};
    const std::array<float, 6> weights{};
    auto descriptor = contract.dump();
    Handle seed(mcsim_create_contract(1, cuda, environment, 100, 1, weights.data(), descriptor.c_str()), mcsim_destroy);
    require(seed != nullptr, "create concrete nine-slot test seed");
    auto definitions = json::array({scenario(seed.get(), 0)["players"]});
    for (auto& player : definitions[0]) {
        player["loadout"] = {{"weapons", {"minecraft:diamond_sword", "minecraft:air", "minecraft:air",
            "minecraft:air", "minecraft:air", "minecraft:air", "minecraft:air", "minecraft:air", "minecraft:air"}}, {"selected", 7}};
        player["attack_strength_ticks"] = 20;
    }
    std::array<float, 4> observations{};
    require(mcsim_reset_players(seed.get(), nullptr, definitions.dump().c_str(), observations.data()) == 0,
        "accept repeated empty slots");
    require(scenario(seed.get(), 0)["players"][0]["loadout"]["selected"] == 7
        && observations[0] == 7 && observations[1] == 7 && observations[2] == 7,
        "resolved initial scenario and both observation perspectives retain requested empty slot");
    require(frame(seed.get(), 0)["actors"][0]["weapon_slot"] == 7, "initial replay retains empty slot");

    contract["actions"][0]["values"] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    minecraft_sim::PlayerState state;
    auto slots = minecraft_sim::app::applyLoadout(state, definitions[0][0]["loadout"]);
    minecraft_sim::app::PolicyLayout nine(contract, {slots, slots});
    require(nine.actions[0].values.size() == 9, "nine-slot descriptor accepts padded concrete equipment");
    const std::string concrete_path = cuda ? "training-nine-slots-cuda.json" : "training-nine-slots-cpu.json";
    {
        std::ofstream output(concrete_path);
        require(static_cast<bool>(output), "open concrete runtime test environment");
        output << scenario(seed.get(), 0);
    }
    descriptor = contract.dump();
    Handle all_slots(mcsim_create_contract(1, cuda, concrete_path.c_str(), 100, 1, weights.data(), descriptor.c_str()), mcsim_destroy);
    std::filesystem::remove(concrete_path);
    require(all_slots != nullptr, "create nine-slot native inference contract from concrete runtime environment");
    require(mcsim_observe(all_slots.get(), observations.data()) == 0 && observations[0] == 7,
        "native constructor retains explicitly selected repeated air");

    std::array<int32_t, 2> actions{1, 1};
    std::array<float, 2> rewards{};
    uint8_t terminated = 0, truncated = 0;
    const std::array<int32_t, 2> last_slot{8, 8};
    require(mcsim_step(all_slots.get(), last_slot.data(), observations.data(), rewards.data(), &terminated, &truncated) == 0
        && observations[0] == 8 && observations[1] == 8
        && frame(all_slots.get(), 0)["actors"][0]["controls"]["select_slot"] == 8,
        "native nine-slot action head selects the last empty physical slot");
    require(frame(all_slots.get(), 0)["actors"][0]["attack_strength_ticks"] == 21,
        "last empty slot preserves item-based attack charge");
    require(mcsim_step(seed.get(), actions.data(), observations.data(), rewards.data(), &terminated, &truncated) == 0,
        "select another empty slot");
    const auto switched = frame(seed.get(), 0);
    require(observations[0] == 1 && observations[1] == 1
        && switched["actors"][0]["weapon_slot"] == 1
        && switched["actors"][0]["controls"]["select_slot"] == 1
        && switched["actors"][0]["weapon"] == "minecraft:air",
        "empty-slot action, observation and replay agree");
    require(switched["actors"][0]["attack_strength_ticks"] == 21, "empty-to-empty switching preserves item-based attack charge");
    require(mcsim_reset(seed.get(), nullptr, observations.data()) == 0 && observations[0] == 7,
        "reset restores requested initial empty slot");
    auto invalid = definitions;
    invalid[0][0]["loadout"]["weapons"][8] = "minecraft:diamond_sword";
    require(mcsim_reset_players(seed.get(), nullptr, invalid.dump().c_str(), observations.data()) == -1,
        "duplicate nonempty slots still rejected");
    std::cout << "PASS: repeated empty hotbar slots preserve initial selection, actions, charge, observations and replay\n";
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 2 || (argc == 3 && std::string(argv[2]) == "--cuda"),
            "Usage: training_checks <two-slot-environment.json> [--cuda]");
        const int cuda = argc == 3 ? 1 : 0;
        if (cuda && !mcsim_cuda_available()) return 77;
        checkResetPlayers(argv[1], cuda);
        checkEmptySlots(argv[1], cuda);
        checkRewardWeights(cuda);
        checkStepRewardTotals(cuda);
        checkMaxTicks(cuda);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

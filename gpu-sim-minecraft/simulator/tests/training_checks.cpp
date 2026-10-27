#include "minecraft_sim/training_api.h"
#include "../apps/configuration.hpp"
#include "../apps/observation.hpp"

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
        {"ticks_per_action", 1}, {"requires", {"bow"}}, {"observations", json::array()},
        {"actions", {{{"name", "weapon"}, {"values", {0, 1}}}}}};
    for (const auto* name : minecraft_sim::app::feature_names) {
        contract["observations"].push_back({{"feature", name}});
    }
    const auto descriptor = contract.dump();
    const std::array<float, 10> weights{};
    Handle handle(mcsim_create_contract(3, cuda, environment, 100, 123, weights.data(), descriptor.c_str()), mcsim_destroy);
    require(handle != nullptr, "create all-equipment descriptor batch using a two-slot environment");
    require(mcsim_api_version() == 4, "current training library exposes ABI 4");
    require(mcsim_set_profiling(nullptr, 1) == -1, "profiling rejects null handles");
    require(mcsim_set_profiling(handle.get(), 2) == -1, "profiling requires a boolean flag");
    require(mcsim_profile(nullptr) == nullptr, "profile rejects null handles");
    require(mcsim_set_profiling(handle.get(), 1) == 0, "enable native profiling");
    const auto baseline = scenario(handle.get(), 0);
    require(!baseline.contains("policy"), "resolved scenarios never advertise a stale embedded policy");
    auto own = baseline["players"];
    own[0]["health"] = 13;
    own[0]["velocity"] = {0.1, -0.2, 0.3};
    own[1]["velocity"] = {-0.2, 0.4, -0.1};
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
    require(std::abs(observation(0, "self.velocity.x") - 0.1F) < 1e-6F
        && std::abs(observation(0, "self.velocity.y") + 0.2F) < 1e-6F
        && std::abs(observation(0, "self.velocity.z") - 0.3F) < 1e-6F
        && std::abs(observation(0, "relative.velocity.x") + 0.3F) < 1e-6F
        && std::abs(observation(0, "relative.velocity.y") - 0.6F) < 1e-6F
        && std::abs(observation(0, "relative.velocity.z") + 0.4F) < 1e-6F,
        "training observations include self and opponent-relative velocity");
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
    const auto profile = json::parse(mcsim_profile(handle.get()));
    require(profile["version"] == 1 && profile["seconds"]["step_decode"] > 0
        && profile["seconds"]["physics"] > 0 && profile["seconds"]["reset_validate"] > 0
        && profile["seconds"]["reset_apply"] > 0 && profile["seconds"]["reset_observe"] > 0,
        "native profile records separate step and reset phases");
    require(mcsim_set_profiling(handle.get(), 0) == 0, "disable native profiling");
    require(mcsim_observe(handle.get(), observations.data()) == 0, "observe with profiling disabled");
    require(json::parse(mcsim_profile(handle.get())) == profile, "disabled profiling leaves counters unchanged");
    require(mcsim_set_profiling(handle.get(), 1) == 0, "restart native profile");
    const auto cleared = json::parse(mcsim_profile(handle.get()));
    for (const auto& value : cleared["seconds"]) require(value == 0, "new profile clears all native counters");
    require(mcsim_set_profiling(handle.get(), 0) == 0, "leave profiling disabled");

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
    std::array<float, 10> weights{};
    Handle handle(mcsim_create_schema(1, cuda, nullptr, 1, 1, weights.data(), 1), mcsim_destroy);
    require(handle != nullptr, "create legacy reward test batch");
    require(json::parse(mcsim_batch_info(handle.get()))["observation_dim"] == 22, "v1 dimension stays unchanged");
    weights[4] = 3;
    require(mcsim_set_reward_weights(handle.get(), weights.data()) == 0, "set curriculum reward weights");
    auto invalid = weights;
    invalid[4] = 7; invalid[9] = std::numeric_limits<float>::quiet_NaN();
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
    const std::array<float, 10> weights{0, 0, 0, 0, -3, -7};
    Handle handle(mcsim_create_schema(2, cuda, nullptr, 1, 1, weights.data(), 1), mcsim_destroy);
    require(handle != nullptr, "create selected-player reward batch");
    std::array<double, 16> totals{};
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
    const std::array<float, 10> weights{};
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
    const std::array<float, 10> weights{};
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

void checkArrowObservation() {
    std::array<minecraft_sim::PlayerState, 2> players{};
    players[0].position = {0.0, 0.0, 0.0};
    players[0].velocity = {0.1, 0.2, 0.3};
    players[0].weapon = minecraft_sim::WeaponItem::bow;
    players[0].using_bow = true;
    players[1].position = {0.0, 0.0, 10.0};
    players[1].weapon = minecraft_sim::WeaponItem::bow;
    players[1].using_bow = true;
    std::array<minecraft_sim::ArrowState, 5> arrows{};
    minecraft_sim::ProjectileWorldState projectiles;
    projectiles.arrows = arrows.data();
    projectiles.arrow_capacity = static_cast<int>(arrows.size());
    projectiles.bows[0].ordinary_arrows = 64;
    projectiles.bows[0].draw_ticks = 10;
    arrows[0].active = true;
    arrows[0].owner = 0;
    arrows[0].position = {1.0, 0.0, 0.0};
    arrows[0].velocity = {1.0, 0.0, 0.0};
    for (int slot = 1; slot <= 2; ++slot) {
        arrows[slot].active = true;
        arrows[slot].owner = 1;
        arrows[slot].position = {3.0, 1.0, 2.0};
        arrows[slot].velocity = slot == 1 ? minecraft_sim::Vec3{1.0, 2.0, 3.0}
                                            : minecraft_sim::Vec3{-1.0, 0.0, 0.0};
    }
    arrows[3].active = true;
    arrows[3].owner = 1;
    arrows[3].in_ground = true;
    arrows[3].position = {0.1, 0.0, 0.0};
    arrows[3].velocity = {1.0, 0.0, 0.0};
    arrows[4].active = true;
    arrows[4].owner = 1;
    arrows[4].position = {0.1, 0.0, 0.0};

    std::array<float, minecraft_sim::app::feature_names.size()> catalogue{};
    minecraft_sim::app::arrowObservation(catalogue.data(), players[0], players[1], projectiles, 0);
    require(catalogue[89] == 1.0F && catalogue[90] == 1.0F
        && std::abs(catalogue[91] - minecraft_sim::bowPower(10)) < 1.0e-6F
        && catalogue[92] == 1.0F && catalogue[93] == 1.0F && catalogue[94] == 1.0F,
        "bow observations expose own ammo and actual draw state from both players");
    require(catalogue[95] == 1.0F && std::abs(catalogue[96] - 0.3F) < 1.0e-6F
        && std::abs(catalogue[97] - 0.1F) < 1.0e-6F && std::abs(catalogue[98] - 0.2F) < 1.0e-6F
        && std::abs(catalogue[99] - 0.9F) < 1.0e-6F && std::abs(catalogue[100] - 1.8F) < 1.0e-6F
        && std::abs(catalogue[101] - 2.7F) < 1.0e-6F,
        "nearest moving opponent arrow uses slot-order tie break and opponent-minus-self velocity");

    std::array<float, minecraft_sim::app::feature_names.size()> summarized{};
    const auto summary = minecraft_sim::summarizeArrowObservation(players[0], projectiles, 0);
    minecraft_sim::app::arrowObservation(summarized.data(), players[0], players[1], summary);
    for (int feature = 89; feature <= 101; ++feature) {
        require(summarized[feature] == catalogue[feature], "compact arrow summary preserves observation features");
    }

    minecraft_sim::app::arrowObservation(catalogue.data(), players[1], players[0], projectiles, 1);
    require(catalogue[95] == 1.0F && std::abs(catalogue[96] - 0.1F) < 1.0e-6F,
        "the reverse perspective excludes its own arrow and detects the other player's arrow");
    arrows[1].in_ground = true;
    arrows[2].active = false;
    minecraft_sim::app::arrowObservation(catalogue.data(), players[0], players[1], projectiles, 0);
    for (int index = 95; index < 102; ++index) {
        require(catalogue[index] == 0.0F, "absent, stuck, stationary, and self-owned arrows encode seven zeros");
    }
}

void checkBowAdapter(const char* environment, int cuda) {
    constexpr std::array<const char*, 13> features{
        "self.arrow_count_fraction", "self.bow_using", "self.bow_power_fraction",
        "self.bow_equipped", "opponent.bow_equipped", "opponent.bow_using", "arrow.present",
        "arrow.relative.x", "arrow.relative.y", "arrow.relative.z", "arrow.relative.velocity.x",
        "arrow.relative.velocity.y", "arrow.relative.velocity.z"};
    json contract{{"format", "minecraft_sim_policy"}, {"version", 1}, {"minecraft", "26.3"},
        {"ticks_per_action", 1}, {"requires", {"bow"}}, {"observations", json::array()},
        {"actions", {{{"name", "weapon"}, {"values", {0, 1}}},
            {{"name", "use_item"}, {"values", {false, true}}}}}};
    for (const auto* feature : features) contract["observations"].push_back({{"feature", feature}});
    const std::array<float, 10> weights{2.0F, 3.0F, 0, 0, 0, 0, 0, 0, 0, 4.0F};
    const auto descriptor = contract.dump();
    Handle handle(mcsim_create_contract(2, cuda, environment, 100, 123, weights.data(), descriptor.c_str()), mcsim_destroy);
    require(handle != nullptr, "create native bow contract with appended observations and binary use_item");

    auto conflicting_contract = contract;
    conflicting_contract["actions"].push_back({{"name", "shield"}, {"values", {false, true}}});
    const auto conflicting_descriptor = conflicting_contract.dump();
    void* conflicting = mcsim_create_contract(1, cuda, environment, 100, 123, weights.data(),
                                                conflicting_descriptor.c_str());
    if (conflicting) mcsim_destroy(conflicting);
    require(conflicting == nullptr && std::string(mcsim_last_error()).find("both shield and use_item") != std::string::npos,
        "legacy shield and use_item action aliases cannot coexist in one policy");

    json definitions = json::array({scenario(handle.get(), 0)["players"], scenario(handle.get(), 1)["players"]});
    for (int world = 0; world < 2; ++world) {
        definitions[world][0]["position"] = {0.5, 1.0, -1.5};
        definitions[world][0]["yaw"] = 0.0;
        definitions[world][0]["loadout"] = {{"weapons", {"minecraft:diamond_sword", "minecraft:bow"}},
            {"selected", 1}, {"offhand", "minecraft:air"}, {"arrows", 3}};
        definitions[world][1]["position"] = {0.5, 1.0, 1.5};
        definitions[world][1]["yaw"] = 180.0;
        definitions[world][1]["loadout"] = {{"weapons", {"minecraft:bow", "minecraft:diamond_sword"}},
            {"selected", 1}, {"offhand", world == 1 ? "minecraft:shield" : "minecraft:air"}, {"arrows", 0}};
    }
    std::vector<float> observations(2 * 2 * features.size());
    require(mcsim_reset_players(handle.get(), nullptr, definitions.dump().c_str(), observations.data()) == 0,
        "reset bow loadouts with finite ordinary arrow counts and optional shield");
    require(scenario(handle.get(), 0)["players"][0]["loadout"]["arrows"] == 3,
        "scenario serialization preserves configured ordinary arrow count");
    const auto featureIndex = [&](const char* name) {
        const auto found = std::find_if(features.begin(), features.end(), [name](const char* value) {
            return std::string(value) == name;
        });
        if (found == features.end()) throw std::runtime_error("Unknown bow observation feature");
        return static_cast<std::size_t>(found - features.begin());
    };
    const auto observation = [&](int actor, const char* name) {
        return observations[static_cast<std::size_t>(actor) * features.size() + featureIndex(name)];
    };
    require(std::abs(observation(0, "self.arrow_count_fraction") - 3.0F / 64.0F) < 1.0e-6F
        && observation(0, "self.bow_equipped") == 1.0F
        && observation(0, "opponent.bow_equipped") == 0.0F
        && observation(1, "opponent.bow_equipped") == 1.0F,
        "adapter observes scalar ammo and selected bow equipment from both perspectives");
    require(observation(0, "arrow.present") == 0.0F && observation(1, "arrow.present") == 0.0F,
        "reset leaves all arrow-relative observations absent");

    std::array<int32_t, 8> actions{};
    for (int world = 0; world < 2; ++world) {
        for (int player = 0; player < 2; ++player) {
            const auto offset = static_cast<std::size_t>(world * 2 + player) * 2;
            actions[offset] = 1;
            actions[offset + 1] = 1;
        }
    }
    std::array<float, 4> rewards{};
    std::array<uint8_t, 2> terminated{}, truncated{};
    for (int tick = 0; tick < 20; ++tick) {
        require(mcsim_step(handle.get(), actions.data(), observations.data(), rewards.data(),
            terminated.data(), truncated.data()) == 0, "hold bow and legacy shield actions through their use phases");
    }
    const auto drawing_frame = frame(handle.get(), 0);
    require(drawing_frame["capabilities"]["bow"] == true
        && drawing_frame["actors"][0]["bow_using"] == true
        && drawing_frame["actors"][0]["bow_draw_ticks"] == 20
        && drawing_frame["actors"][0]["bow_power"] == 1.0
        && drawing_frame["actors"][0]["arrow_count"] == 3
        && drawing_frame["actors"][0]["controls"]["use_item"] == true,
        "replay reports actual bow draw, ammunition, power, and held semantic control");
    require(drawing_frame["actors"][1]["using_shield"] == false
        && frame(handle.get(), 1)["actors"][1]["using_shield"] == true
        && observation(0, "self.bow_using") == 1.0F
        && observation(1, "opponent.bow_using") == 1.0F,
        "use_item starts bow use or shield use according to the selected main-hand item");

    for (int world = 0; world < 2; ++world) {
        for (int player = 0; player < 2; ++player) {
            const auto offset = static_cast<std::size_t>(world * 2 + player) * 2;
            actions[offset] = 1;
            actions[offset + 1] = player == 0 ? 0 : 1;
        }
    }
    require(mcsim_step(handle.get(), actions.data(), observations.data(), rewards.data(),
        terminated.data(), truncated.data()) == 0, "release the bow once while keeping the shield held");
    const auto released = frame(handle.get(), 0);
    require(released["actors"][0]["bow_using"] == false
        && released["actors"][0]["bow_draw_ticks"] == 0
        && released["actors"][0]["arrow_count"] == 2
        && released["actors"][0]["controls"]["use_item"] == false
        && released["projectiles"].size() == 1
        && released["projectiles"][0]["owner"] == 0
        && released["projectiles"][0]["age"] == 0,
        "release creates one deferred projectile and updates replay state without advancing its age");
    require(observation(0, "arrow.present") == 0.0F && observation(1, "arrow.present") == 1.0F,
        "arrow observation excludes its owner and exposes it to the opponent");
    const auto& arrow_velocity = released["projectiles"][0]["velocity"];
    const auto& observer_velocity = released["actors"][1]["velocity"];
    require(std::abs(observation(1, "arrow.relative.velocity.x")
                - (arrow_velocity[0].get<float>() - observer_velocity[0].get<float>())) < 1.0e-6F
        && std::abs(observation(1, "arrow.relative.velocity.z")
                - (arrow_velocity[2].get<float>() - observer_velocity[2].get<float>())) < 1.0e-6F,
        "relative arrow velocity is world arrow velocity minus observer velocity");

    require(mcsim_step(handle.get(), actions.data(), observations.data(), rewards.data(),
        terminated.data(), truncated.data()) == 0, "advance delayed arrow hits and shield deflection");
    const auto hit_frame = frame(handle.get(), 0);
    const auto blocked_frame = frame(handle.get(), 1);
    const double health_lost = 20.0 - hit_frame["actors"][1]["health"].get<double>();
    require(health_lost > 0.0 && hit_frame["projectiles"].empty()
        && blocked_frame["actors"][1]["health"] == 20.0
        && blocked_frame["projectiles"].size() == 1,
        "delayed ordinary arrow damages health, while an active shield deflects without health loss");
    std::array<double, 16> totals{};
    const std::array<uint8_t, 4> hit_world{1, 1, 0, 0};
    require(mcsim_step_reward_totals(handle.get(), hit_world.data(), totals.data()) == 0
        && std::abs(totals[0] - 2.0 * health_lost / 20.0) < 1.0e-6
        && std::abs(totals[1] - 3.0 * health_lost / 20.0) < 1.0e-6
        && totals[9] == 4.0,
        "arrow reward terms use owner-attributed health loss and full progress on enemy contact");
    const std::array<uint8_t, 4> shooter_only{1, 0, 0, 0};
    require(mcsim_step_reward_totals(handle.get(), shooter_only.data(), totals.data()) == 0
        && totals[9] == 4.0,
        "the arrow-progress event is credited to its shooter only");
    const std::array<uint8_t, 4> target_only{0, 1, 0, 0};
    require(mcsim_step_reward_totals(handle.get(), target_only.data(), totals.data()) == 0
        && totals[9] == 0.0,
        "the arrow-progress event is not credited to its target");
    const std::array<uint8_t, 4> shield_world{0, 0, 1, 1};
    require(mcsim_step_reward_totals(handle.get(), shield_world.data(), totals.data()) == 0
        && totals[0] == 0.0 && totals[1] == 0.0 && totals[9] == 4.0,
        "enemy shield contact earns progress without creating health-based rewards");

    const auto untouched = frame(handle.get(), 1);
    const std::array<uint8_t, 2> reset_first{1, 0};
    require(mcsim_reset(handle.get(), reset_first.data(), observations.data()) == 0
        && frame(handle.get(), 0)["tick"] == 0 && frame(handle.get(), 0)["projectiles"].empty()
        && frame(handle.get(), 0)["actors"][0]["arrow_count"] == 3
        && frame(handle.get(), 1) == untouched,
        "selective reset restores selected ammo/draw/projectiles and leaves the other world untouched");
    std::cout << "PASS: bow adapter observations, actions, delayed rewards, replay, and selective CPU/CUDA reset\n";
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
        checkArrowObservation();
        checkBowAdapter(argv[1], cuda);
        checkRewardWeights(cuda);
        checkStepRewardTotals(cuda);
        checkMaxTicks(cuda);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

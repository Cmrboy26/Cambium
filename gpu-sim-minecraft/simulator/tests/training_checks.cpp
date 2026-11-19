#include "minecraft_sim/training_api.h"
#include "../apps/configuration.hpp"
#include "../apps/observation.hpp"

#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <cstdlib>
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

struct BudgetSetting {
    bool present = std::getenv("MCSIM_TERRAIN_BUDGET_BYTES") != nullptr;
    std::string previous = present ? std::getenv("MCSIM_TERRAIN_BUDGET_BYTES") : "";
    static void set(const char* value) {
#if defined(_WIN32)
        if (_putenv_s("MCSIM_TERRAIN_BUDGET_BYTES", value ? value : "") != 0)
            throw std::runtime_error("Unable to set terrain budget");
#else
        if ((value ? setenv("MCSIM_TERRAIN_BUDGET_BYTES", value, 1)
                   : unsetenv("MCSIM_TERRAIN_BUDGET_BYTES")) != 0)
            throw std::runtime_error("Unable to set terrain budget");
#endif
    }
    ~BudgetSetting() { set(present ? previous.c_str() : nullptr); }
};

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

void checkBridgeBatch(const char* environment, int cuda) {
    json policy{{"format", "minecraft_sim_policy"}, {"version", 1}, {"minecraft", "26.3"},
        {"ticks_per_action", 1}, {"observations", {{{"feature", "relative.z"}}}},
        {"actions", {{{"name", "movement"}, {"values", {"neutral", "forward"}}}}}};
    json config{{"mode", {{"name", "bridge"}, {"version", 1}}},
        {"goals", {{{"min", {0.0, 1.0, 2.4}}, {"max", {1.0, 3.0, 2.9}}},
                   {{"min", {0.0, 1.0, -1.1}}, {"max", {1.0, 3.0, -0.8}}}}},
        {"void_y", -2.0},
        {"match", {{"goals_to_win", 2}, {"max_ticks", 100}, {"cage_ticks", 0},
                   {"persist_blocks_on_goal", true}, {"arrow_refill_ticks", 0}}},
        {"task_rewards", {{"score", 2.0}, {"win", 5.0}, {"loss", -3.0}}}};
    const std::array<float, 10> weights{};
    const auto descriptor = policy.dump();
    Handle batch(mcsim_create_contract(2, cuda, environment, 100, 91, weights.data(), descriptor.c_str()), mcsim_destroy);
    require(batch != nullptr, "create experimental Bridge batch");
    std::array<std::uint64_t, 2> counters{91, 92};
    require(mcsim_bridge_reset_episodes(batch.get(), counters.data(), counters.size()) != 0
        && counters == std::array<std::uint64_t, 2>{91, 92}, "duel batch rejects Bridge counter access");
    require(mcsim_set_bridge_reset_episodes(batch.get(), counters.data(), counters.size()) != 0,
        "duel batch rejects Bridge counter restore");
    auto invalid = config;
    invalid["match"]["arrow_refill_ticks"] = 80;
    require(mcsim_configure_bridge(batch.get(), invalid.dump().c_str()) != 0,
        "Bridge refill without a bow must be rejected");
    require(!scenario(batch.get(), 0).contains("experimental_bridge"), "rejection leaves duel scenario unchanged");
    require(mcsim_configure_bridge(batch.get(), config.dump().c_str()) == 0, "configure static Bridge batch");
    require(scenario(batch.get(), 0)["experimental_bridge"] == config, "resolved scenario records Bridge identity");
    require(mcsim_configure_bridge(batch.get(), config.dump().c_str()) != 0, "reject repeated Bridge configuration");
    {
        Handle resumed(mcsim_create_contract(2, cuda, environment, 100, 91, weights.data(), descriptor.c_str()), mcsim_destroy);
        require(resumed && mcsim_configure_bridge(resumed.get(), config.dump().c_str()) == 0,
            "configure resumed Bridge batch");
        std::array<float, 4> expected{}, actual{}, rewards{};
        const std::array<uint8_t, 2> first{1, 0};
        require(mcsim_reset(batch.get(), first.data(), expected.data()) == 0
            && mcsim_bridge_reset_episodes(batch.get(), counters.data(), counters.size()) == 0
            && counters == std::array<std::uint64_t, 2>{3, 2}, "snapshot selective Bridge resets");
        const auto snapshot = counters;
        require(mcsim_reset(batch.get(), nullptr, expected.data()) == 0
            && mcsim_set_bridge_reset_episodes(resumed.get(), snapshot.data(), snapshot.size()) == 0
            && mcsim_observe(resumed.get(), actual.data()) == 0 && actual == expected,
            "restored counters start the next fresh matches");
        require(mcsim_bridge_reset_episodes(resumed.get(), counters.data(), counters.size()) == 0
            && counters == std::array<std::uint64_t, 2>{4, 3}, "restore increments every counter once");
        std::array<int32_t, 4> neutral{};
        std::array<uint8_t, 2> terminated{}, truncated{};
        require(mcsim_step(resumed.get(), neutral.data(), actual.data(), rewards.data(),
            terminated.data(), truncated.data()) == 0, "advance resumed match before invalid restores");
        const auto state = json::parse(mcsim_bridge_state(resumed.get()));
        const auto before = actual;
        const std::array<std::uint64_t, 2> overflow{5, (std::numeric_limits<std::uint64_t>::max)()};
        require(mcsim_bridge_reset_episodes(nullptr, counters.data(), 2) != 0
            && mcsim_bridge_reset_episodes(resumed.get(), nullptr, 2) != 0
            && mcsim_bridge_reset_episodes(resumed.get(), counters.data(), 1) != 0
            && mcsim_set_bridge_reset_episodes(nullptr, snapshot.data(), 2) != 0
            && mcsim_set_bridge_reset_episodes(resumed.get(), nullptr, 2) != 0
            && mcsim_set_bridge_reset_episodes(resumed.get(), snapshot.data(), 1) != 0
            && mcsim_set_bridge_reset_episodes(resumed.get(), overflow.data(), 2) != 0,
            "invalid Bridge counter arguments are rejected");
        require(mcsim_bridge_reset_episodes(resumed.get(), counters.data(), 2) == 0
            && counters == std::array<std::uint64_t, 2>{4, 3}
            && mcsim_observe(resumed.get(), actual.data()) == 0 && actual == before
            && json::parse(mcsim_bridge_state(resumed.get())) == state,
            "invalid restores leave counters and live matches unchanged");
        const std::array<std::uint64_t, 2> boundary{0, (std::numeric_limits<std::uint64_t>::max)() - 1};
        require(mcsim_set_bridge_reset_episodes(resumed.get(), boundary.data(), 2) == 0
            && mcsim_bridge_reset_episodes(resumed.get(), counters.data(), 2) == 0
            && counters == std::array<std::uint64_t, 2>{1, (std::numeric_limits<std::uint64_t>::max)()},
            "restore accepts the largest incrementable counter");
    }
    std::array<int32_t, 4> actions{1, 0, 0, 0};
    std::array<float, 4> observations{}, rewards{};
    std::array<uint8_t, 2> terminated{}, truncated{};
    const auto status = [&] {
        const auto* value = mcsim_bridge_state(batch.get());
        require(value != nullptr, "Bridge state is available");
        return json::parse(value);
    };
    bool first_goal = false;
    bool match_end = false;
    for (int tick = 0; tick < 30 && !match_end; ++tick) {
        require(mcsim_step(batch.get(), actions.data(), observations.data(), rewards.data(),
            terminated.data(), truncated.data()) == 0, "step static Bridge batch");
        const auto states = status();
        require(states[1]["scores"] == json::array({0, 0}) && states[1]["outcome"] == "ongoing",
            "other Bridge world remains independent");
        if (states[0]["scored"][0] == true) {
            if (!first_goal) {
                first_goal = true;
                require(!terminated[0] && !truncated[0] && rewards[0] == 2.0F
                    && states[0]["round_reset"] == true && states[0]["scores"][0] == 1,
                    "nonwinning goal rewards and resets without termination");
            } else {
                match_end = true;
                require(terminated[0] && !truncated[0] && rewards[0] == 7.0F && rewards[1] == -3.0F
                    && states[0]["outcome"] == "team0" && states[0]["round_reset"] == false,
                    "winning goal emits Bridge score and outcome rewards");
            }
        }
    }
    require(first_goal && match_end, "Bridge match reaches two goals");
    std::array<uint8_t, 4> losses{};
    require(mcsim_losses(batch.get(), losses.data()) == 0 && !losses[0] && losses[1],
        "Bridge match losses reflect the outcome, not the last death");
    std::array<double, 18> totals{};
    require(mcsim_step_reward_totals(batch.get(), nullptr, totals.data()) != 0,
        "Bridge cannot report duel reward totals");
    require(mcsim_replay_frame(batch.get(), 0) == nullptr, "Bridge replay is explicitly unsupported");
    const auto other_ticks = status()[1]["ticks"];
    const std::array<uint8_t, 2> mask{1, 0};
    require(mcsim_reset(batch.get(), mask.data(), observations.data()) == 0, "selective Bridge match reset");
    const auto reset = status();
    require(reset[0]["scores"] == json::array({0, 0}) && reset[0]["ticks"] == 0
        && reset[1]["ticks"] == other_ticks, "selected reset preserves the other match");

    Handle cutoff(mcsim_create_contract(2, cuda, environment, 1, 91, weights.data(), descriptor.c_str()), mcsim_destroy);
    require(cutoff && mcsim_configure_bridge(cutoff.get(), config.dump().c_str()) == 0,
        "configure externally truncated Bridge batch");
    auto players = scenario(cutoff.get(), 0)["players"];
    players[0]["position"][2] = -1.41;
    players[0]["velocity"][2] = 0.2;
    const auto definitions = json::array({players, nullptr}).dump();
    require(mcsim_reset_players(cutoff.get(), mask.data(), definitions.c_str(), observations.data()) == 0,
        "assign a safe spawn immediately before the goal");
    actions.fill(0);
    require(mcsim_step(cutoff.get(), actions.data(), observations.data(), rewards.data(),
        terminated.data(), truncated.data()) == 0, "step Bridge goal on external cutoff");
    const auto* cutoff_json = mcsim_bridge_state(cutoff.get());
    require(cutoff_json != nullptr, "read truncated Bridge state");
    const auto cutoff_state = json::parse(cutoff_json);
    require(!terminated[0] && truncated[0] && rewards[0] == 2.0F && cutoff_state[0]["scored"][0] == true
        && cutoff_state[0]["round_reset"] == false && std::abs(observations[0] - 0.271F) < 1e-5F,
        "external truncation retains the pre-reset goal observation");
}

void checkBridgeFixedOpponentMatch(const char* environment, int cuda) {
    const std::array<float, 10> weights{};
    const json policy{{"format", "minecraft_sim_policy"}, {"version", 1}, {"minecraft", "26.3"},
        {"ticks_per_action", 1},
        {"observations", {{{"feature", "relative.x"}}, {{"feature", "relative.z"}},
                          {{"feature", "self.health_fraction"}}, {{"feature", "self.velocity.z"}}}},
        {"actions", {{{"name", "movement"}, {"values", {"neutral", "forward"}}}}}};
    const auto descriptor = policy.dump();
    Handle seed(mcsim_create_contract(1, cuda, environment, 100, 91, weights.data(), descriptor.c_str()), mcsim_destroy);
    require(seed != nullptr, "create fixed-opponent Bridge scene source");
    auto definition = scenario(seed.get(), 0);
    definition["players"][0]["position"] = {0.5, 1.0, -1.5};
    definition["players"][1]["position"] = {-1.5, 1.0, 1.5};
    for (auto& player : definition["players"]) {
        player["velocity"] = {0.0, 0.0, 0.0};
        player["on_ground"] = true;
        player["health"] = 20.0;
    }
    definition["players"][0]["yaw"] = 0.0;
    const std::string path = cuda ? "training-bridge-match-cuda.json" : "training-bridge-match-cpu.json";
    {
        std::ofstream output(path);
        require(static_cast<bool>(output), "open fixed-opponent Bridge scene");
        output << definition;
    }
    Handle batch(mcsim_create_contract(2, cuda, path.c_str(), 100, 91, weights.data(), descriptor.c_str()), mcsim_destroy);
    std::filesystem::remove(path);
    require(batch != nullptr, "create fixed-opponent Bridge batch");
    const json config{{"mode", {{"name", "bridge"}, {"version", 1}}},
        {"goals", {{{"min", {0.0, 1.0, 2.4}}, {"max", {1.0, 3.0, 2.9}}},
                   {{"min", {0.0, 1.0, -1.16}}, {"max", {1.0, 3.0, -0.8}}}}},
        {"void_y", -2.0},
        {"match", {{"goals_to_win", 3}, {"max_ticks", 100}, {"cage_ticks", 2},
                   {"persist_blocks_on_goal", true}, {"arrow_refill_ticks", 0}}},
        {"task_rewards", {{"score", 2.0}, {"win", 5.0}, {"loss", -3.0}}}};
    require(mcsim_configure_bridge(batch.get(), config.dump().c_str()) == 0,
        "configure fixed-opponent Bridge match");
    const auto status = [&] {
        const auto* value = mcsim_bridge_state(batch.get());
        require(value != nullptr, "read fixed-opponent Bridge state");
        return json::parse(value);
    };
    std::array<int32_t, 4> actions{1, 0, 1, 0};
    std::array<float, 16> observations{};
    std::array<float, 4> rewards{};
    std::array<uint8_t, 2> terminated{}, truncated{};
    const std::array<float, 8> spawn_observations{-0.2F, 0.3F, 1.0F, 0.0F,
                                                   0.2F, -0.3F, 1.0F, 0.0F};
    const std::array<float, 8> final_observations{-0.2F, 0.2902F, 1.0F, 0.053508F,
                                                   0.2F, -0.2902F, 1.0F, 0.0F};
    const auto check_observations = [&](int world, const std::array<float, 8>& expected, const char* message) {
        for (std::size_t index = 0; index < expected.size(); ++index) {
            require(std::abs(observations[world * expected.size() + index] - expected[index]) < 2E-5F, message);
        }
    };
    require(mcsim_observe(batch.get(), observations.data()) == 0, "observe initial Bridge match");
    check_observations(0, spawn_observations, "initial first-world observations");
    check_observations(1, spawn_observations, "initial second-world observations");
    require(status()[0]["cage_remaining"] == 2 && status()[1]["cage_remaining"] == 2,
        "both matches start with two locked ticks");

    for (int tick = 1; tick <= 9; ++tick) {
        require(mcsim_step(batch.get(), actions.data(), observations.data(), rewards.data(),
            terminated.data(), truncated.data()) == 0, "step fixed-opponent Bridge match");
        const auto states = status();
        const bool scored = tick % 3 == 0;
        const bool winning = tick == 9;
        require(states[0]["ticks"] == tick && states[0]["scores"] == json::array({tick / 3, 0})
            && states[0]["scored"] == json::array({scored, false})
            && states[0]["died"] == json::array({false, false})
            && states[0]["round_reset"] == (scored && !winning)
            && states[0]["cage_remaining"] == (scored && !winning ? 2 : (scored ? 0 : 2 - tick % 3))
            && states[0]["outcome"] == (winning ? "team0" : "ongoing"),
            "first match score, clock, cage and round events follow the expected timeline");
        require(terminated[0] == winning && !truncated[0]
            && rewards[0] == (winning ? 7.0F : (scored ? 2.0F : 0.0F))
            && rewards[1] == (winning ? -3.0F : 0.0F),
            "first match emits only the expected score and outcome rewards");
        const bool other_scored = tick == 3;
        require(states[1]["ticks"] == tick && states[1]["scores"] == json::array({tick >= 3 ? 1 : 0, 0})
            && states[1]["scored"] == json::array({other_scored, false})
            && states[1]["round_reset"] == other_scored && states[1]["outcome"] == "ongoing"
            && !terminated[1] && !truncated[1]
            && rewards[2] == (other_scored ? 2.0F : 0.0F) && rewards[3] == 0.0F,
            "other match scores once and remains independent and ongoing");
        check_observations(0, winning ? final_observations : spawn_observations,
            "first match exposes reset or final observations at the correct time");
        check_observations(1, spawn_observations, "stationary other match remains at its spawns");
        if (tick == 3) actions[2] = 0;
    }

    std::array<uint8_t, 4> losses{};
    require(mcsim_losses(batch.get(), losses.data()) == 0
        && losses == std::array<uint8_t, 4>{0, 1, 0, 0}, "winning match reports only the losing opponent");
    const auto other_state = status()[1];
    require(mcsim_step(batch.get(), actions.data(), observations.data(), rewards.data(),
        terminated.data(), truncated.data()) != 0, "finished match cannot advance before reset");
    const std::array<uint8_t, 2> first_world{1, 0};
    require(mcsim_reset(batch.get(), first_world.data(), observations.data()) == 0,
        "selectively reset the finished Bridge match");
    const auto reset = status();
    require(reset[0]["ticks"] == 0 && reset[0]["scores"] == json::array({0, 0})
        && reset[0]["cage_remaining"] == 2 && reset[0]["outcome"] == "ongoing"
        && reset[1] == other_state, "full match reset preserves the other world's complete event state");
    check_observations(0, spawn_observations, "selected reset restores both player observations");
    check_observations(1, spawn_observations, "selected reset preserves other world observations");
    require(mcsim_step(batch.get(), actions.data(), observations.data(), rewards.data(),
        terminated.data(), truncated.data()) == 0, "step both matches after selective reset");
    const auto resumed = status();
    require(resumed[0]["ticks"] == 1 && resumed[0]["cage_remaining"] == 1
        && resumed[0]["scores"] == json::array({0, 0}) && !terminated[0] && !truncated[0]
        && resumed[1]["ticks"] == 10 && resumed[1]["scores"] == json::array({1, 0})
        && resumed[1]["outcome"] == "ongoing" && !terminated[1] && !truncated[1],
        "reset match restarts its cage while the other match continues");
    check_observations(0, spawn_observations, "restarted cage locks the scorer again");
    check_observations(1, spawn_observations, "other match continues with its fixed opponent");
    std::cout << "PASS: fixed-opponent Bridge match and selective CPU/CUDA reset\n";
}

void checkBridgeKits(const char* environment, int cuda) {
    const std::array<float, 10> weights{};
    json policy{{"format", "minecraft_sim_policy"}, {"version", 1}, {"minecraft", "26.3"},
        {"ticks_per_action", 1}, {"requires", {"bow"}},
        {"observations", {{{"feature", "self.health_fraction"}}}},
        {"actions", {{{"name", "weapon"}, {"values", {0, 1, 2, 3, 4}}},
                     {{"name", "use_item"}, {"values", {false, true}}},
                     {{"name", "attack"}, {"values", {false, true}}}}}};
    const auto descriptor = policy.dump();
    json seed_policy = policy;
    seed_policy["actions"][0]["values"] = {0, 1};
    const auto seed_descriptor = seed_policy.dump();
    Handle seed(mcsim_create_contract(1, cuda, environment, 100, 91, weights.data(), seed_descriptor.c_str()), mcsim_destroy);
    require(seed != nullptr, "create Bridge kit test scene");
    auto definition = scenario(seed.get(), 0);
    for (auto& player : definition["players"]) {
        player["loadout"] = {{"weapons", {"minecraft:diamond_sword", "minecraft:bow", "minecraft:air",
                                          "minecraft:air", "minecraft:air"}},
                             {"selected", 1}, {"arrows", 1}, {"offhand", "minecraft:air"}};
    }
    definition["players"][0]["health"] = 10;
    definition["players"][1]["position"][0] = -1.5;
    const std::string path = cuda ? "training-bridge-kits-cuda.json" : "training-bridge-kits-cpu.json";
    {
        std::ofstream output(path);
        require(static_cast<bool>(output), "open Bridge kit test scene");
        output << definition;
    }
    Handle batch(mcsim_create_contract(2, cuda, path.c_str(), 100, 91, weights.data(), descriptor.c_str()), mcsim_destroy);
    auto incompatible_policy = policy;
    incompatible_policy["observations"].push_back({{"feature", "self.weapon_slot_2.present"}});
    const auto incompatible_descriptor = incompatible_policy.dump();
    Handle incompatible(mcsim_create_contract(1, cuda, path.c_str(), 100, 91, weights.data(),
                                               incompatible_descriptor.c_str()), mcsim_destroy);
    auto seed_test_policy = policy;
    seed_test_policy["observations"] = {{{"feature", "arrow.present"}},
        {{"feature", "arrow.relative.velocity.x"}}, {{"feature", "arrow.relative.velocity.y"}},
        {{"feature", "arrow.relative.velocity.z"}}};
    const auto seed_test_descriptor = seed_test_policy.dump();
    Handle source(mcsim_create_contract(2, cuda, path.c_str(), 100, 91, weights.data(), seed_test_descriptor.c_str()), mcsim_destroy);
    Handle resumed(mcsim_create_contract(2, cuda, path.c_str(), 100, 91, weights.data(), seed_test_descriptor.c_str()), mcsim_destroy);
    Handle replayed(mcsim_create_contract(2, cuda, path.c_str(), 100, 91, weights.data(), seed_test_descriptor.c_str()), mcsim_destroy);
    std::filesystem::remove(path);
    require(batch && incompatible && source && resumed && replayed, "create Bridge kit training batches");
    const json kit{{"slots", {"sword", "bow", "pickaxe", "blocks", "golden_apple"}},
                   {"blocks", 64}, {"golden_apples", 2}};
    json config{{"mode", {{"name", "bridge"}, {"version", 1}}},
        {"goals", {{{"min", {0.0, 1.0, 2.4}}, {"max", {1.0, 3.0, 2.9}}},
                   {{"min", {0.0, 1.0, -1.1}}, {"max", {1.0, 3.0, -0.8}}}}},
        {"void_y", -2.0},
        {"match", {{"goals_to_win", 2}, {"max_ticks", 100}, {"cage_ticks", 0},
                   {"persist_blocks_on_goal", true}, {"arrow_refill_ticks", 3}}},
        {"kits", {kit, kit}}, {"task_rewards", {{"score", 2.0}, {"win", 5.0}, {"loss", -3.0}}}};
    {
        for (auto* handle : {source.get(), resumed.get(), replayed.get()}) {
            require(mcsim_configure_bridge(handle, config.dump().c_str()) == 0, "configure seed continuation batch");
        }
        std::array<float, 16> source_observations{}, resumed_observations{}, replayed_observations{};
        const std::array<uint8_t, 2> first{1, 0};
        std::array<std::uint64_t, 2> snapshot{};
        require(mcsim_reset(source.get(), first.data(), source_observations.data()) == 0
            && mcsim_bridge_reset_episodes(source.get(), snapshot.data(), snapshot.size()) == 0
            && mcsim_reset(source.get(), nullptr, source_observations.data()) == 0
            && mcsim_set_bridge_reset_episodes(resumed.get(), snapshot.data(), snapshot.size()) == 0,
            "resume the next per-world projectile RNG seeds");
        std::array<int32_t, 12> shoot{1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0};
        std::array<float, 4> rewards{};
        std::array<uint8_t, 2> terminated{}, truncated{};
        const auto fire_tick = [&](void* handle, std::array<float, 16>& observations) {
            require(mcsim_step(handle, shoot.data(), observations.data(), rewards.data(),
                terminated.data(), truncated.data()) == 0, "step seed continuation bow");
        };
        for (int tick = 0; tick < 3; ++tick) {
            fire_tick(source.get(), source_observations);
            fire_tick(resumed.get(), resumed_observations);
            fire_tick(replayed.get(), replayed_observations);
        }
        shoot[1] = shoot[7] = 0;
        fire_tick(source.get(), source_observations);
        fire_tick(resumed.get(), resumed_observations);
        fire_tick(replayed.get(), replayed_observations);
        require(source_observations[4] == 1.0F && source_observations == resumed_observations
            && (source_observations[5] != replayed_observations[5]
                || source_observations[6] != replayed_observations[6]
                || source_observations[7] != replayed_observations[7]),
            "resumed projectile randomness matches continuation and differs from replayed seeds");
    }
    require(mcsim_configure_bridge(incompatible.get(), config.dump().c_str()) != 0,
        "reserved Bridge items cannot masquerade as empty weapon observations");
    auto invalid = config;
    invalid["kits"][0]["slots"][2] = "bow";
    require(mcsim_configure_bridge(batch.get(), invalid.dump().c_str()) != 0,
        "reject duplicate or mismatched Bridge kit slots");
    invalid = config;
    invalid["kits"][0]["slots"][0] = "bow";
    invalid["kits"][0]["slots"][1] = "sword";
    require(mcsim_configure_bridge(batch.get(), invalid.dump().c_str()) != 0,
        "reject Bridge slots that disagree with the scene");
    require(mcsim_configure_bridge(batch.get(), config.dump().c_str()) == 0, "configure Bridge kits and refill");
    const auto kits = [&](int player) {
        const auto* value = mcsim_bridge_kits(batch.get(), player);
        require(value != nullptr, "read Bridge kits");
        return json::parse(value);
    };
    require(kits(0)[0]["arrows"] == 1 && kits(0)[0]["blocks"] == 64
        && kits(0)[0]["golden_apples"] == 2 && kits(0)[0]["selected_slot"] == 1,
        "initial Bridge kit matches its scene loadout");
    std::array<int32_t, 12> actions{1, 1, 0, 1, 0, 0, 2, 1, 1, 1, 0, 0};
    std::array<float, 4> observations{}, rewards{};
    std::array<uint8_t, 2> terminated{}, truncated{};
    const std::array<uint8_t, 2> mask{1, 0};
    auto bad_players = scenario(batch.get(), 0)["players"];
    bad_players[0]["loadout"]["weapons"][1] = "minecraft:diamond_axe";
    bad_players[0]["loadout"]["arrows"] = 0;
    const auto bad_reset = json::array({bad_players, nullptr}).dump();
    require(mcsim_reset_players(batch.get(), mask.data(), bad_reset.c_str(), observations.data()) != 0
        && kits(0)[0]["arrows"] == 1, "invalid sampled kit reset leaves existing state intact");
    const auto step = [&] {
        require(mcsim_step(batch.get(), actions.data(), observations.data(), rewards.data(),
            terminated.data(), truncated.data()) == 0 && !terminated[0] && !truncated[0],
            "step Bridge kits");
    };
    for (int tick = 0; tick < 3; ++tick) step();
    require(kits(0)[1]["selected_slot"] == 2 && kits(0)[1]["blocks"] == 64,
        "other world's unsupported pickaxe action preserves its kit");
    actions[1] = 0;
    step();
    require(kits(0)[0]["arrows"] == 0 && kits(0)[0]["arrow_refill_elapsed"] == 0,
        "last arrow firing tick does not count towards refill");
    for (int tick = 1; tick <= 3; ++tick) {
        step();
        const auto current = kits(0);
        require(current[0]["arrows"] == (tick == 3 ? 1 : 0)
            && current[0]["arrow_refill_elapsed"] == (tick == 3 ? 0 : tick),
            "Bridge arrow refill follows subsequent empty ticks");
    }
    require(mcsim_reset(batch.get(), mask.data(), observations.data()) == 0, "reset one Bridge kit world");
    require(kits(0)[0]["arrows"] == 1 && kits(0)[0]["selected_slot"] == 1
        && kits(0)[1]["selected_slot"] == 2, "kit reset leaves another world unchanged");
    require(mcsim_bridge_kits(batch.get(), 2) == nullptr, "reject invalid Bridge kit perspective");
    actions = {4, 1, 0, 1, 0, 0, 4, 1, 0, 1, 0, 0};
    for (int tick = 0; tick < 31; ++tick) {
        if (tick == 10) actions[7] = 0;
        step();
    }
    require(kits(0)[0]["golden_apples"] == 2 && kits(0)[1]["golden_apples"] == 2
        && observations[0] == 0.5F && observations[2] == 0.5F,
        "apple use does not finish early and release cancels the other world");
    step();
    require(kits(0)[0]["golden_apples"] == 1 && kits(0)[1]["golden_apples"] == 2
        && observations[0] == 0.5F, "only the held apple is consumed on tick 32");
    actions[1] = 0;
    step();
    require(std::abs(observations[0] - 0.55F) < 1E-6F && observations[2] == 0.5F,
        "golden apple regeneration affects only the consuming world");
    require(mcsim_reset(batch.get(), mask.data(), observations.data()) == 0
        && kits(0)[0]["golden_apples"] == 2 && kits(0)[1]["golden_apples"] == 2
        && observations[0] == 0.5F && observations[2] == 0.5F,
        "selected match reset restores the apple kit and clears its effects");
}

void checkBridgeBuilding(const char* environment, int cuda, bool budget_only = false) {
    const std::array<float, 10> weights{};
    json policy{{"format", "minecraft_sim_policy"}, {"version", 1}, {"minecraft", "26.3"},
        {"ticks_per_action", 1}, {"observations", {{{"feature", "arena.ray_world_0"}}}},
        {"actions", {{{"name", "weapon"}, {"values", {0, 1, 2}}},
                     {{"name", "use_item"}, {"values", {false, true}}},
                     {{"name", "attack"}, {"values", {false, true}}}}}};
    json seed_policy = policy;
    seed_policy["actions"][0]["values"] = {0, 1};
    const auto seed_descriptor = seed_policy.dump();
    Handle seed(mcsim_create_contract(1, cuda, environment, 100, 91, weights.data(), seed_descriptor.c_str()), mcsim_destroy);
    require(seed != nullptr, "create Bridge building test scene");
    auto definition = scenario(seed.get(), 0);
    definition["blocks"].push_back({{"from", {0, 2, 1}}, {"to", {0, 2, 1}}, {"block", "minecraft:stone"}});
    definition["players"][1]["position"][0] = -2.5;
    for (auto& player : definition["players"]) {
        player["velocity"][1] = -0.0784000015258789;
        player["loadout"] = {{"weapons", {"minecraft:diamond_sword", "minecraft:air", "minecraft:air"}},
                             {"selected", 1}, {"arrows", 0}, {"offhand", "minecraft:air"}};
    }
    const std::string path = cuda ? "training-bridge-building-cuda.json" : "training-bridge-building-cpu.json";
    {
        std::ofstream output(path);
        require(static_cast<bool>(output), "open Bridge building test scene");
        output << definition;
    }
    const auto descriptor = policy.dump();
    auto movement_policy = policy;
    movement_policy["observations"] = {{{"feature", "relative.x"}}};
    const auto movement_descriptor = movement_policy.dump();
    auto combat_policy = policy;
    combat_policy["observations"] = {{{"feature", "self.health_fraction"}}};
    const auto combat_descriptor = combat_policy.dump();
    Handle batch(mcsim_create_contract(2, cuda, path.c_str(), 100, 91, weights.data(), descriptor.c_str()), mcsim_destroy);
    Handle movement(mcsim_create_contract(2, cuda, path.c_str(), 100, 91, weights.data(), movement_descriptor.c_str()), mcsim_destroy);
    Handle combat(mcsim_create_contract(2, cuda, path.c_str(), 100, 91, weights.data(), combat_descriptor.c_str()), mcsim_destroy);
    Handle persistent(mcsim_create_contract(2, cuda, path.c_str(), 100, 91, weights.data(), descriptor.c_str()), mcsim_destroy);
    Handle clearing(mcsim_create_contract(2, cuda, path.c_str(), 100, 91, weights.data(), descriptor.c_str()), mcsim_destroy);
    Handle replacement(mcsim_create_contract(3, cuda, path.c_str(), 100, 91, weights.data(), descriptor.c_str()), mcsim_destroy);
    Handle reference(mcsim_create_contract(3, cuda, path.c_str(), 100, 91, weights.data(), descriptor.c_str()), mcsim_destroy);
    std::filesystem::remove(path);
    require(batch != nullptr && movement != nullptr && combat != nullptr && persistent != nullptr && clearing != nullptr
        && replacement != nullptr && reference != nullptr,
        "create Bridge building training batches");
    const json kit{{"slots", {"sword", "blocks", "pickaxe"}}, {"blocks", 4}, {"golden_apples", 0}};
    json config{{"mode", {{"name", "bridge"}, {"version", 1}}},
        {"goals", {{{"min", {0.0, 1.0, 2.4}}, {"max", {1.0, 3.0, 2.9}}},
                   {{"min", {0.0, 1.0, -1.1}}, {"max", {1.0, 3.0, -0.8}}}}},
        {"void_y", -2.0},
        {"match", {{"goals_to_win", 2}, {"max_ticks", 100}, {"cage_ticks", 0},
                   {"persist_blocks_on_goal", true}, {"arrow_refill_ticks", 0}}},
        {"kits", {kit, kit}}, {"task_rewards", {{"score", 2.0}, {"win", 5.0}, {"loss", -3.0}}},
        {"building", {{"zones", {{{"min", {-3, 0, -3}}, {"max", {4, 4, 4}}}}},
                      {"block_range", 4.5}, {"entity_range", 3.0}, {"hardness", 1.0},
                      {"mining_speed", 15.0}, {"correct_tool", true}}}};
    {
        BudgetSetting budget_setting;
        require(scenario(replacement.get(), 0)["bounds"]["size"] == json::array({7, 4, 7}),
            "replacement budget fixture has 196 cells");
        static_assert(sizeof(minecraft_sim::Block) == 1 && sizeof(minecraft_sim::PlacedBlock) == 1);
        std::array<int32_t, 18> actions{};
        std::array<float, 6> observations{}, expected_observations{}, rewards{}, expected_rewards{};
        std::array<uint8_t, 3> terminated{}, truncated{}, expected_terminated{}, expected_truncated{};
        const auto verify_rejection = [&](const char* budget) {
            std::array<json, 3> before_scenarios, before_frames;
            for (int world = 0; world < 3; ++world) {
                before_scenarios[world] = scenario(replacement.get(), world);
                before_frames[world] = frame(replacement.get(), world);
            }
            require(mcsim_observe(replacement.get(), observations.data()) == 0, "observe before budget rejection");
            const auto before_observations = observations;
            BudgetSetting::set(budget);
            require(mcsim_configure_bridge(replacement.get(), config.dump().c_str()) == -1
                && std::string(mcsim_last_error()).find("Terrain buffers exceed MCSIM_TERRAIN_BUDGET_BYTES")
                    != std::string::npos, "replacement rejects insufficient terrain budget");
            for (int world = 0; world < 3; ++world) {
                require(scenario(replacement.get(), world) == before_scenarios[world]
                    && frame(replacement.get(), world) == before_frames[world],
                    "budget rejection preserves each world's scenario and state");
            }
            require(mcsim_observe(replacement.get(), observations.data()) == 0 && observations == before_observations,
                "budget rejection preserves observations");
            require(mcsim_step(replacement.get(), actions.data(), observations.data(), rewards.data(),
                    terminated.data(), truncated.data()) == 0
                && mcsim_step(reference.get(), actions.data(), expected_observations.data(), expected_rewards.data(),
                    expected_terminated.data(), expected_truncated.data()) == 0
                && observations == expected_observations && rewards == expected_rewards
                && terminated == expected_terminated && truncated == expected_truncated,
                "rejected replacement continues stepping like an untouched batch");
            for (int world = 0; world < 3; ++world) {
                require(frame(replacement.get(), world) == frame(reference.get(), world),
                    "rejected replacement preserves each world's state");
            }
        };
        verify_rejection("783");
        if (cuda) verify_rejection("2351");
        BudgetSetting::set(cuda ? "2352" : "784");
        require(mcsim_configure_bridge(replacement.get(), config.dump().c_str()) == 0
            && scenario(replacement.get(), 0)["experimental_bridge"] == config,
            "replacement accepts exact independent terrain preflight budget");
        require(mcsim_step(replacement.get(), actions.data(), observations.data(), rewards.data(),
            terminated.data(), truncated.data()) == 0, "exact-budget replacement remains usable");
        std::cout << "PASS: terrain replacement exact/insufficient preflight budgets and rollback\n";
    }
    if (budget_only) return;
    const auto check_goal_reset = [&](Handle& goal_batch, bool persist) {
        auto goal_config = config;
        goal_config["match"]["persist_blocks_on_goal"] = persist;
        require(mcsim_configure_bridge(goal_batch.get(), goal_config.dump().c_str()) == 0,
            "configure Bridge goal terrain policy");
        auto players = scenario(goal_batch.get(), 0)["players"];
        players[0]["velocity"][2] = 0.2;
        const auto definitions = json::array({players, nullptr}).dump();
        const std::array<uint8_t, 2> first{1, 0}, second{0, 1};
        std::array<float, 4> observations{}, rewards{};
        std::array<uint8_t, 2> terminated{}, truncated{};
        const std::array<int32_t, 12> actions{1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        require(mcsim_reset_players(goal_batch.get(), first.data(), definitions.c_str(), observations.data()) == 0,
            "set up a scoring placement in one world");
        require(mcsim_step(goal_batch.get(), actions.data(), observations.data(), rewards.data(),
            terminated.data(), truncated.data()) == 0, "step scoring placement");
        const auto* raw = mcsim_bridge_state(goal_batch.get());
        require(raw != nullptr, "read scoring placement events");
        const auto events = json::parse(raw);
        require(events[0]["placed"][0] == true && events[0]["scored"][0] == true
            && events[0]["round_reset"] == true && !terminated[0] && !truncated[0],
            "a nonwinning goal follows the placement on the same tick");
        const float expected = persist ? 0.03F : 0.05F;
        require(std::abs(observations[0] - expected) < 1E-6F && std::abs(observations[2] - 0.05F) < 1E-6F,
            "goal reset applies the configured terrain policy only to its world");
        require(mcsim_reset(goal_batch.get(), second.data(), observations.data()) == 0
            && std::abs(observations[0] - expected) < 1E-6F, "other match reset preserves goal terrain");
        require(mcsim_reset(goal_batch.get(), first.data(), observations.data()) == 0
            && std::abs(observations[0] - 0.05F) < 1E-6F, "full match reset clears selected terrain");
    };
    check_goal_reset(persistent, true);
    check_goal_reset(clearing, false);
    require(mcsim_configure_bridge(movement.get(), config.dump().c_str()) == 0, "configure Bridge collision test");
    auto moving_players = scenario(movement.get(), 0)["players"];
    moving_players[1]["position"] = {-0.8, 1.0, 0.5};
    moving_players[1]["velocity"] = {0.8, -0.0784000015258789, 0.0};
    const auto moving_definitions = json::array({moving_players, moving_players}).dump();
    std::array<float, 4> moving_observations{}, moving_rewards{};
    std::array<uint8_t, 2> moving_terminated{}, moving_truncated{};
    require(mcsim_reset_players(movement.get(), nullptr, moving_definitions.c_str(), moving_observations.data()) == 0,
        "initialize matching Bridge collision worlds");
    const std::array<int32_t, 12> moving_actions{1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    require(mcsim_step(movement.get(), moving_actions.data(), moving_observations.data(), moving_rewards.data(),
        moving_terminated.data(), moving_truncated.data()) == 0, "step same-tick Bridge placement and movement");
    const auto* moving_raw = mcsim_bridge_state(movement.get());
    require(moving_raw != nullptr, "read Bridge collision events");
    const auto moving_events = json::parse(moving_raw);
    require(moving_events[0]["placed"][0] == true && moving_events[1]["placed"][0] == false
        && std::abs(moving_observations[0] - (-0.08F)) < 1E-6F
        && std::abs(moving_observations[2] - (-0.05F)) < 1E-6F,
        "placed terrain clips same-tick movement only in its world");
    require(mcsim_configure_bridge(combat.get(), config.dump().c_str()) == 0, "configure Bridge melee test");
    auto combat_players = scenario(combat.get(), 0)["players"];
    combat_players[1]["position"] = {1.31, 1.0, 1.5};
    combat_players[1]["yaw"] = 165.0;
    combat_players[1]["loadout"]["selected"] = 0;
    const auto combat_definitions = json::array({combat_players, combat_players}).dump();
    std::array<float, 4> combat_observations{}, combat_rewards{};
    std::array<uint8_t, 2> combat_terminated{}, combat_truncated{};
    require(mcsim_reset_players(combat.get(), nullptr, combat_definitions.c_str(), combat_observations.data()) == 0,
        "initialize matching Bridge melee worlds");
    const std::array<int32_t, 12> combat_actions{1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1};
    require(mcsim_step(combat.get(), combat_actions.data(), combat_observations.data(), combat_rewards.data(),
        combat_terminated.data(), combat_truncated.data()) == 0, "step same-tick Bridge placement and melee");
    const auto* combat_raw = mcsim_bridge_state(combat.get());
    require(combat_raw != nullptr, "read Bridge melee events");
    const auto combat_events = json::parse(combat_raw);
    if (!(combat_events[0]["placed"][0] == true && combat_events[1]["placed"][0] == false
        && combat_observations[0] == 1.0F && std::abs(combat_observations[2] - 0.65F) < 1E-6F)) {
        throw std::runtime_error("Bridge melee placement/health: " + combat_events.dump() + " / "
            + std::to_string(combat_observations[0]) + ", " + std::to_string(combat_observations[2]));
    }
    auto bow_definition = definition;
    for (auto& player : bow_definition["players"]) {
        player["loadout"] = {{"weapons", {"minecraft:diamond_sword", "minecraft:bow", "minecraft:air", "minecraft:air"}},
                             {"selected", 2}, {"arrows", 2}, {"offhand", "minecraft:air"}};
    }
    bow_definition["players"][1]["position"] = {1.31, 1.0, 1.5};
    bow_definition["players"][1]["yaw"] = 165.0;
    bow_definition["players"][1]["loadout"]["selected"] = 1;
    const std::string bow_path = cuda ? "training-bridge-terrain-bow-cuda.json" : "training-bridge-terrain-bow-cpu.json";
    {
        std::ofstream output(bow_path);
        require(static_cast<bool>(output), "open Bridge projectile test scene");
        output << bow_definition;
    }
    auto bow_policy = combat_policy;
    bow_policy["requires"] = {"bow"};
    bow_policy["actions"][0]["values"] = {0, 1, 2, 3};
    const auto bow_descriptor = bow_policy.dump();
    Handle bow(mcsim_create_contract(2, cuda, bow_path.c_str(), 100, 91, weights.data(), bow_descriptor.c_str()), mcsim_destroy);
    std::filesystem::remove(bow_path);
    require(bow != nullptr, "create Bridge projectile training batch");
    auto bow_config = config;
    const json bow_kit{{"slots", {"sword", "bow", "blocks", "pickaxe"}}, {"blocks", 4}, {"golden_apples", 0}};
    bow_config["kits"] = {bow_kit, bow_kit};
    require(mcsim_configure_bridge(bow.get(), bow_config.dump().c_str()) == 0, "configure Bridge projectile test");
    std::array<float, 4> bow_observations{}, bow_rewards{};
    std::array<uint8_t, 2> bow_terminated{}, bow_truncated{};
    std::array<int32_t, 12> bow_actions{2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0};
    for (int tick = 0; tick < 20; ++tick) {
        require(mcsim_step(bow.get(), bow_actions.data(), bow_observations.data(), bow_rewards.data(),
            bow_terminated.data(), bow_truncated.data()) == 0, "place Bridge block and draw bows");
        if (tick == 0) {
            const auto* raw = mcsim_bridge_state(bow.get());
            require(raw != nullptr && json::parse(raw)[0]["placed"][0] == true,
                "place projectile-blocking terrain in only one world");
            bow_actions[0] = 0;
            bow_actions[1] = 0;
        }
    }
    bow_actions[4] = 0;
    bow_actions[10] = 0;
    for (int tick = 0; tick < 4; ++tick) {
        require(mcsim_step(bow.get(), bow_actions.data(), bow_observations.data(), bow_rewards.data(),
            bow_terminated.data(), bow_truncated.data()) == 0, "release and advance Bridge arrows");
    }
    const auto* bow_kits_raw = mcsim_bridge_kits(bow.get(), 1);
    require(bow_kits_raw != nullptr, "read Bridge shooter ammunition");
    const auto bow_kits = json::parse(bow_kits_raw);
    if (!(bow_kits[0]["arrows"] == 1 && bow_kits[1]["arrows"] == 1
        && bow_observations[0] == 1.0F && bow_observations[2] < 1.0F)) {
        throw std::runtime_error("Bridge projectile ammunition/health: " + bow_kits.dump() + " / "
            + std::to_string(bow_observations[0]) + ", " + std::to_string(bow_observations[2]));
    }
    bow_actions[0] = 3;
    bow_actions[2] = 1;
    for (int tick = 0; tick < 3; ++tick) {
        require(mcsim_step(bow.get(), bow_actions.data(), bow_observations.data(), bow_rewards.data(),
            bow_terminated.data(), bow_truncated.data()) == 0, "break projectile-blocking terrain");
    }
    const auto* break_raw = mcsim_bridge_state(bow.get());
    require(break_raw != nullptr && json::parse(break_raw)[0]["broke"][0] == true,
        "break the previously placed projectile-blocking terrain");
    bow_actions[0] = 0;
    bow_actions[2] = 0;
    bow_actions[4] = 1;
    for (int tick = 0; tick < 20; ++tick) {
        require(mcsim_step(bow.get(), bow_actions.data(), bow_observations.data(), bow_rewards.data(),
            bow_terminated.data(), bow_truncated.data()) == 0, "draw bow after Bridge block break");
    }
    bow_actions[4] = 0;
    for (int tick = 0; tick < 4; ++tick) {
        require(mcsim_step(bow.get(), bow_actions.data(), bow_observations.data(), bow_rewards.data(),
            bow_terminated.data(), bow_truncated.data()) == 0, "release arrow after Bridge block break");
    }
    require(bow_observations[0] < 1.0F, "arrow reaches the player after the placed block is broken");
    auto invalid = config;
    invalid["building"]["zones"][0]["min"][0] = -4;
    require(mcsim_configure_bridge(batch.get(), invalid.dump().c_str()) != 0,
        "reject build zones outside arena bounds");
    require(mcsim_configure_bridge(batch.get(), config.dump().c_str()) == 0, "configure mutable Bridge batch");
    std::array<int32_t, 12> actions{1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::array<float, 4> observations{}, rewards{};
    std::array<uint8_t, 2> terminated{}, truncated{};
    const auto step = [&] {
        require(mcsim_step(batch.get(), actions.data(), observations.data(), rewards.data(),
            terminated.data(), truncated.data()) == 0 && !terminated[0] && !truncated[0],
            "step mutable Bridge batch");
    };
    const auto status = [&] {
        const auto* value = mcsim_bridge_state(batch.get());
        require(value != nullptr, "read Bridge building events");
        return json::parse(value);
    };
    step();
    require(status()[0]["placed"][0] == true && std::abs(observations[0] - 0.03F) < 1E-6F
        && std::abs(observations[2] - 0.05F) < 1E-6F,
        "placed terrain affects only the edited world's observation");
    std::array<std::uint64_t, 2> reset_episodes{};
    require(mcsim_bridge_reset_episodes(batch.get(), reset_episodes.data(), reset_episodes.size()) == 0
        && mcsim_set_bridge_reset_episodes(batch.get(), reset_episodes.data(), reset_episodes.size()) == 0
        && mcsim_observe(batch.get(), observations.data()) == 0 && std::abs(observations[0] - 0.05F) < 1E-6F,
        "restoring Bridge counters starts full matches and clears placed terrain");
    step();
    require(status()[0]["placed"][0] == true && std::abs(observations[0] - 0.03F) < 1E-6F,
        "building configuration remains active after restore");
    const std::array<uint8_t, 2> other{0, 1};
    require(mcsim_reset(batch.get(), other.data(), observations.data()) == 0
        && std::abs(observations[0] - 0.03F) < 1E-6F, "other-world reset preserves placed terrain");
    actions[0] = 2;
    actions[1] = 0;
    actions[2] = 1;
    step();
    require(status()[0]["broke"][0] == false, "breaking begins without immediate completion");
    step();
    require(status()[0]["broke"][0] == false, "partial break preserves the block");
    step();
    require(status()[0]["broke"][0] == true && std::abs(observations[0] - 0.05F) < 1E-6F,
        "broken terrain disappears from observations");
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
    require(mcsim_api_version() == 5, "current training library exposes ABI 5");
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
    std::array<double, 18> totals{};
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

void checkLargeArena(const char* environment, int cuda) {
    auto definition = json::parse(std::ifstream(environment));
    definition["bounds"]["size"] = {32, 48, 96};
    const std::string path = cuda ? "training-large-arena-cuda.json" : "training-large-arena-cpu.json";
    const std::array<float, 10> weights{};
    const auto create = [&](const json& value) {
        {
            std::ofstream output(path);
            require(static_cast<bool>(output), "open large-arena test environment");
            output << value;
        }
        Handle handle(mcsim_create_schema(2, cuda, path.c_str(), 100, 1, weights.data(), 2), mcsim_destroy);
        std::filesystem::remove(path);
        return handle;
    };
    auto handle = create(definition);
    require(handle != nullptr && scenario(handle.get(), 0)["bounds"]["size"] == json::array({32, 48, 96}),
            "native batch accepts the Bridge-sized static arena");
    BudgetSetting budget_setting;
    BudgetSetting::set("1327104");
    require(create(definition) != nullptr, "native scene accepts exact portable budget");
    BudgetSetting::set("1327103");
    require(create(definition) == nullptr, "native scene rejects budget one byte below portable requirement");
    BudgetSetting::set("536870912");
    for (const auto& size : {json::array({129, 1, 1}), json::array({128, 128, 17}),
                             json::array({9223372036854775807LL, 1, 1})}) {
        definition["bounds"]["size"] = size;
        require(create(definition) == nullptr, "native batch rejects invalid or oversized dimensions");
    }
    definition["bounds"]["size"] = {32, 48, 96};
    definition["blocks"] = json::array();
    for (int index = 0; index < 257; ++index) definition["blocks"].push_back(json::object());
    require(create(definition) == nullptr, "native batch rejects excessive fill count");
    std::cout << "PASS: Bridge-sized arena and native dimension/fill limits\n";
}

void checkFallDamageReset(const char* environment, int cuda) {
    json descriptor{{"format", "minecraft_sim_policy"}, {"version", 1}, {"minecraft", "26.3"},
        {"ticks_per_action", 1}, {"observations", {{{"feature", "self.health_fraction"}}}},
        {"actions", {{{"name", "jump"}, {"values", {false, true}}}}}};
    const std::array<float, 10> weights{};
    const auto contract = descriptor.dump();
    Handle seed(mcsim_create_contract(1, cuda, environment, 100, 1, weights.data(), contract.c_str()), mcsim_destroy);
    require(seed != nullptr, "create fall-damage test seed");
    auto definition = scenario(seed.get(), 0);
    definition["rules"]["simulate_fall_damage"] = true;
    definition["bounds"]["size"][1] = 12;
    definition["players"][0]["position"][1] = 8.0;
    definition["players"][0]["velocity"] = {0.0, 0.0, 0.0};
    definition["players"][0]["on_ground"] = false;
    const std::string path = cuda ? "training-fall-cuda.json" : "training-fall-cpu.json";
    {
        std::ofstream output(path);
        require(static_cast<bool>(output), "open fall-damage test environment");
        output << definition;
    }
    Handle handle(mcsim_create_contract(2, cuda, path.c_str(), 100, 1, weights.data(), contract.c_str()), mcsim_destroy);
    std::filesystem::remove(path);
    require(handle != nullptr, "create fall-damage training batch");
    require(scenario(handle.get(), 0)["rules"]["simulate_fall_damage"] == true,
        "resolved scenario preserves opt-in fall damage");
    std::array<int32_t, 4> actions{};
    std::array<float, 4> observations{}, rewards{};
    std::array<uint8_t, 2> terminated{}, truncated{};
    const auto step = [&] {
        for (int tick = 0; tick < 25; ++tick) {
            require(mcsim_step(handle.get(), actions.data(), observations.data(), rewards.data(),
                terminated.data(), truncated.data()) == 0, "step fall-damage training batch");
        }
    };
    step();
    require(frame(handle.get(), 0)["actors"][0]["health"].get<float>() < 20.0F,
        "enabled fall damage affects training gameplay");
    const auto other_world = frame(handle.get(), 1);
    const std::array<uint8_t, 2> selected{1, 0};
    require(mcsim_reset(handle.get(), selected.data(), observations.data()) == 0, "selective fall-damage reset");
    require(frame(handle.get(), 1) == other_world, "selective reset leaves the other fall-damage world unchanged");
    step();
    require(frame(handle.get(), 0)["actors"][0]["health"].get<float>() < 20.0F,
        "selective reset preserves fall-damage rule");
    require(mcsim_reset(handle.get(), nullptr, observations.data()) == 0, "full fall-damage reset");
    step();
    for (int index = 0; index < 2; ++index) {
        require(frame(handle.get(), index)["actors"][0]["health"].get<float>() < 20.0F,
            "full reset preserves fall-damage rule in each world");
    }
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

void checkThrownObservation() {
    minecraft_sim::PlayerState source, target;
    source.snowball_count = 1;
    source.egg_count = 0;
    std::array<float, minecraft_sim::app::feature_names.size()> catalogue{};
    minecraft_sim::app::equipmentObservation(catalogue.data(), source, target,
        {minecraft_sim::WeaponItem::snowball, minecraft_sim::WeaponItem::egg});
    require(catalogue[43] == 1.0F && catalogue[47] == 0.0F,
        "equipment observation hides depleted throwable stacks");
}

void checkWindAdapter(const char* environment, int cuda) {
    json descriptor{{"format", "minecraft_sim_policy"}, {"version", 1}, {"minecraft", "26.3"},
        {"ticks_per_action", 1}, {"observations", {{{"feature", "self.weapon_slot_0.present"}}}},
        {"actions", {{{"name", "use_item"}, {"values", {false, true}}}}}};
    const std::array<float, 10> weights{};
    const auto contract = descriptor.dump();
    Handle seed(mcsim_create_contract(1, cuda, environment, 100, 1, weights.data(), contract.c_str()), mcsim_destroy);
    require(seed != nullptr, "create wind test seed");
    auto definition = scenario(seed.get(), 0);
    definition["players"][0]["loadout"] = {{"weapons", {"minecraft:wind_charge", "minecraft:mace"}}, {"wind_charges", 1}};
    definition["players"][0]["pitch"] = 90;
    const std::string path = cuda ? "training-wind-cuda.json" : "training-wind-cpu.json";
    {
        std::ofstream output(path);
        require(static_cast<bool>(output), "open wind test environment");
        output << definition;
    }
    Handle handle(mcsim_create_contract(2, cuda, path.c_str(), 100, 1, weights.data(), contract.c_str()), mcsim_destroy);
    std::filesystem::remove(path);
    require(handle != nullptr, "create wind training batch");
    require(scenario(handle.get(), 0)["players"][0]["loadout"]["wind_charges"] == 1, "wind count survives scenario resolution");
    const std::array<float, 2> event_weights{2.5F, 0.0F};
    require(mcsim_set_event_rewards(handle.get(), event_weights.data()) == 0, "set wind reward weights");
    std::array<int32_t, 4> actions{1, 0, 0, 0};
    std::array<float, 4> observations{}, rewards{};
    std::array<uint8_t, 2> terminated{}, truncated{};
    require(mcsim_step(handle.get(), actions.data(), observations.data(), rewards.data(), terminated.data(), truncated.data()) == 0,
        "step wind training batch");
    const auto wind_frame = frame(handle.get(), 0);
    require(observations[0] == 0.0F && wind_frame["actors"][0]["weapon"] == "minecraft:air"
        && wind_frame["actors"][0]["wind_charge_count"] == 0 && wind_frame["actors"][0]["wind_charge_cooldown"] > 0
        && wind_frame["projectiles"].size() == 1 && wind_frame["projectiles"][0]["type"] == "wind_charge",
        "wind use depletes slot and records projectile and cooldown");
    const auto other_world = frame(handle.get(), 1);
    const std::array<uint8_t, 2> selected{1, 0};
    require(mcsim_reset(handle.get(), selected.data(), observations.data()) == 0
        && observations[0] == 1.0F && frame(handle.get(), 1) == other_world,
        "selective wind reset restores only selected world");
    require(mcsim_step(handle.get(), actions.data(), observations.data(), rewards.data(), terminated.data(), truncated.data()) == 0
        && frame(handle.get(), 0)["actors"][0]["wind_charge_count"] == 0,
        "reset wind storage permits another throw");
    actions.fill(0);
    float launch_total = 0.0F;
    for (int tick = 0; tick < 10; ++tick) {
        require(mcsim_step(handle.get(), actions.data(), observations.data(), rewards.data(), terminated.data(), truncated.data()) == 0,
            "advance wind explosion");
        std::array<float, 12> events{};
        require(mcsim_step_behavior_events(handle.get(), events.data()) == 0, "read wind behavior events");
        require(std::abs(rewards[0] - events[0] * 2.5F) < 1e-5F && events[3] == 0.0F
            && events[6] == 0.0F && events[9] == 0.0F, "wind event reward is owner and world isolated");
        launch_total += events[0];
    }
    require(launch_total > 0.2F, "downward charge records an actual upward self impulse");
    auto reset_definitions = json::array({definition["players"]});
    require(mcsim_reset_players(seed.get(), nullptr, reset_definitions.dump().c_str(), observations.data()) != 0
        && std::string(mcsim_last_error()).find("initial wind charge loadout") != std::string::npos,
        "a batch without wind storage rejects wind reset loadouts");
}

void checkThrownAdapter(const char* environment, int cuda) {
    json descriptor{{"format", "minecraft_sim_policy"}, {"version", 1}, {"minecraft", "26.3"},
        {"ticks_per_action", 1}, {"observations", {{{"feature", "self.weapon_slot_0.present"}}}},
        {"actions", {{{"name", "use_item"}, {"values", {false, true}}}}}};
    const std::array<float, 10> weights{};
    const auto contract = descriptor.dump();
    Handle seed(mcsim_create_contract(1, cuda, environment, 100, 1, weights.data(), contract.c_str()), mcsim_destroy);
    require(seed != nullptr, "create throwable test seed");
    auto definition = scenario(seed.get(), 0);
    definition["players"][0]["loadout"] = {{"weapons", {"minecraft:snowball", "minecraft:egg"}},
        {"snowballs", 1}, {"eggs", 2}};
    const std::string path = cuda ? "training-thrown-cuda.json" : "training-thrown-cpu.json";
    {
        std::ofstream output(path);
        require(static_cast<bool>(output), "open throwable test environment");
        output << definition;
    }
    Handle handle(mcsim_create_contract(2, cuda, path.c_str(), 100, 1, weights.data(), contract.c_str()), mcsim_destroy);
    std::filesystem::remove(path);
    require(handle != nullptr, "create throwable training batch");
    require(scenario(handle.get(), 0)["players"][0]["loadout"]["eggs"] == 2,
        "scenario preserves throwable stack counts");
    std::array<int32_t, 4> actions{1, 0, 1, 0};
    std::array<float, 4> observations{}, rewards{};
    std::array<uint8_t, 2> terminated{}, truncated{};
    const auto step = [&] {
        require(mcsim_step(handle.get(), actions.data(), observations.data(), rewards.data(),
            terminated.data(), truncated.data()) == 0, "step throwable training batch");
    };
    step();
    const auto thrown_frame = frame(handle.get(), 0);
    require(observations[0] == 0.0F && observations[2] == 0.0F
        && thrown_frame["actors"][0]["weapon"] == "minecraft:air"
        && thrown_frame["actors"][0]["snowball_count"] == 0
        && thrown_frame["actors"][0]["egg_count"] == 2
        && thrown_frame["projectiles"].size() == 1 && thrown_frame["projectiles"][0]["type"] == "snowball",
        "throwing the last snowball empties the selected slot in training");
    const auto other_world = frame(handle.get(), 1);
    const std::array<uint8_t, 2> selected{1, 0};
    require(mcsim_reset(handle.get(), selected.data(), observations.data()) == 0,
        "selectively reset throwable training batch");
    require(observations[0] == 1.0F && frame(handle.get(), 1) == other_world,
        "selective reset restores inventory without changing the other world");
    step();
    require(observations[0] == 0.0F, "reset projectile storage permits another throw");
    require(mcsim_reset(handle.get(), nullptr, observations.data()) == 0
        && observations[0] == 1.0F && observations[2] == 1.0F,
        "full reset restores both throwable inventories");

    auto pearl_definitions = json::array({scenario(handle.get(), 0)["players"], scenario(handle.get(), 1)["players"]});
    for (auto& players : pearl_definitions) {
        players[0]["loadout"] = {{"weapons", {"minecraft:ender_pearl"}}, {"ender_pearls", 2}};
    }
    require(mcsim_reset_players(handle.get(), nullptr, pearl_definitions.dump().c_str(), observations.data()) == 0,
        "reset throwable batch to pearl loadouts");
    step();
    const auto pearl_frame = frame(handle.get(), 0);
    require(pearl_frame["actors"][0]["ender_pearl_count"] == 1
        && pearl_frame["actors"][0]["ender_pearl_cooldown"] > 0
        && pearl_frame["projectiles"].size() == 1 && pearl_frame["projectiles"][0]["type"] == "ender_pearl",
        "pearl use and cooldown appear in training replay");
    require(mcsim_reset(handle.get(), nullptr, observations.data()) == 0
        && frame(handle.get(), 0)["actors"][0]["ender_pearl_count"] == 2
        && frame(handle.get(), 0)["actors"][0]["ender_pearl_cooldown"] == 0,
        "reset restores pearl inventory and cooldown");
    require(mcsim_reset_players(seed.get(), nullptr, json::array({pearl_definitions[0]}).dump().c_str(), observations.data()) != 0
        && std::string(mcsim_last_error()).find("initial throwable loadout") != std::string::npos,
        "a batch created without throwable storage rejects throwable reset loadouts");
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
    std::array<double, 18> totals{};
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
        if (argc >= 3 && std::string(argv[2]) == "--terrain-budget") {
            require(argc == 3 || (argc == 4 && std::string(argv[3]) == "--cuda"),
                "Usage: training_checks <environment.json> --terrain-budget [--cuda]");
            const int cuda = argc == 4 ? 1 : 0;
            if (cuda && !mcsim_cuda_available()) return 77;
            checkBridgeBuilding(argv[1], cuda, true);
            return 0;
        }
        require(argc == 2 || (argc == 3 && std::string(argv[2]) == "--cuda"),
            "Usage: training_checks <two-slot-environment.json> [--cuda]");
        const int cuda = argc == 3 ? 1 : 0;
        if (cuda && !mcsim_cuda_available()) return 77;
        checkLargeArena(argv[1], cuda);
        checkBridgeBatch(argv[1], cuda);
        checkBridgeFixedOpponentMatch(argv[1], cuda);
        checkBridgeKits(argv[1], cuda);
        checkBridgeBuilding(argv[1], cuda);
        checkResetPlayers(argv[1], cuda);
        checkFallDamageReset(argv[1], cuda);
        checkEmptySlots(argv[1], cuda);
        checkArrowObservation();
        checkThrownObservation();
        checkWindAdapter(argv[1], cuda);
        checkThrownAdapter(argv[1], cuda);
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

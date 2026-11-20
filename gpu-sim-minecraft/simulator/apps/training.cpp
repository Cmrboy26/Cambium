#include "scene.hpp"
#include "replay.hpp"
#include "observation.hpp"
#include "rewards.hpp"
#include "minecraft_sim/cuda_backend.hpp"
#include "minecraft_sim/terrain.hpp"
#include "minecraft_sim/bridge_observation.hpp"
#include "minecraft_sim/objective_progress.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
#define MCSIM_EXPORT extern "C" __declspec(dllexport)
#else
#define MCSIM_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
using namespace minecraft_sim;
thread_local std::string last_error;
constexpr std::array<float, 9> yaw_bins{-20, -10, -5, -2, 0, 2, 5, 10, 20};
constexpr std::array<float, 5> pitch_bins{-5, -2, 0, 2, 5};

void traceNative(bool enabled, const char* phase) {
    if (enabled) {
        std::printf("NATIVE_TRACE %s\n", phase);
        std::fflush(stdout);
    }
}

enum ProfilePhase : std::size_t {
    StepDecode, ControlsUpload, Physics, StateDownload, Rewards, StepObserve,
    ResetJson, ResetValidate, ResetApply, ResetObserve, Observe, ProfilePhaseCount
};
constexpr std::array<const char*, ProfilePhaseCount> profile_names{
    "step_decode", "controls_upload", "physics", "state_download", "rewards", "step_observe",
    "reset_json", "reset_validate", "reset_apply", "reset_observe", "observe"
};

struct TrainingBatch {
    app::Scene terrain;
    std::string definition = terrain.definition.dump();
    std::string scenario;
    std::unique_ptr<app::PolicyLayout> layout;
    std::vector<app::ActionHead> action_heads;
    std::vector<app::Scene::PlayerDefinitions> episode_players;
    std::vector<PlayerState> players;
    std::vector<Controls> queued;
    std::vector<Controls> held;
    std::vector<int> selected_slots;
    std::vector<WorldState> worlds;
    std::vector<ProjectileWorldState> projectile_states;
    std::vector<ArrowState> projectile_arrows;
    std::vector<ThrownProjectileState> projectile_thrown;
    std::vector<WindChargeState> projectile_wind;
    std::vector<ArrowObservationSummary> arrow_summaries;
    std::vector<BlockWorldView> blocks;
    std::unique_ptr<TerrainOverlays> bridge_terrain;
    std::vector<std::uint8_t> finished;
    bool bridge_enabled = false;
    BridgeMatchRules bridge_rules;
    std::vector<BridgeInitialPlayers> bridge_initial;
    std::vector<BridgeMatchState> bridge_states;
    std::vector<BridgeTickEvents> bridge_events;
    nlohmann::json bridge_configuration;
    float bridge_score_reward = 0.0F;
    float bridge_win_reward = 0.0F;
    float bridge_loss_reward = 0.0F;
    float bridge_opponent_score_reward = 0.0F;
    float bridge_toward_reward = 0.0F;
    float bridge_away_penalty = 0.0F;
    std::vector<ObjectiveProgress> objective_progress;
    std::vector<std::array<double, 4>> bridge_reward_terms;
    std::vector<std::uint8_t> dealt_damage;
    std::vector<std::uint64_t> episodes;
    std::vector<int> observed_opponent_disable;
    std::vector<std::array<double, 18>> step_reward_totals;
    std::array<float, 10> weights;
    float block_weight = 0.0F;
    std::array<double, 13> reward_totals{};
    std::array<float, 2> event_weights{};
    std::array<float, 5> shaping_weights{};
    std::array<double, 5> shaping_totals{};
    std::uint64_t seed;
    int max_ticks;
    int schema_version;
    int features;
    int heads;
    bool profiling = false;
    std::uint64_t capture_step_counter = 0;
    std::array<double, ProfilePhaseCount> profiling_seconds{};
#if defined(MCSIM_HAS_CUDA)
    std::unique_ptr<CudaBatch> gpu;
    bool gpu_observations_configured = false;
    bool gpu_actions_configured = false;
    bool gpu_resident_configured = false;
#endif

    static app::Scene loadTerrain(const char* path, int version) {
        auto definition = path && *path ? app::Scene::readEnvironment(path) : app::Scene::defaultEnvironment();
        if (version == 2 && definition.value("profile", "") == "unarmored_diamond_sword") {
            definition["profile"] = "unarmored_sword_axe_shield";
        }
        return app::Scene(std::move(definition));
    }

    TrainingBatch(int count, bool use_cuda, const char* path, int limit,
                  std::uint64_t initial_seed, const float* reward_weights, int version = 1, const char* contract = nullptr)
        : terrain(loadTerrain(path, version)),
          episode_players(count, terrain.parsePlayers(terrain.definition.at("players"))),
          players(std::size_t(count) * 2),
          queued(players.size()), held(players.size()), selected_slots(players.size()), worlds(count),
          projectile_states(count), projectile_arrows(std::size_t(count) * projectile_arrow_capacity),
          arrow_summaries(players.size()),
          blocks(count, terrain.blocks),
          finished(count), dealt_damage(players.size()), episodes(count), observed_opponent_disable(players.size()),
          step_reward_totals(players.size()),
          seed(initial_seed), max_ticks(limit), schema_version(version), features(version == 2 ? 31 : 22),
          heads(version == 2 ? 8 : 6) {
        if (schema_version == 2) {
            auto kit = terrain.definition;
            kit["profile"] = "unarmored_sword_axe_shield";
            definition = kit.dump();
        }
        if (schema_version == 1 && terrain.definition.value("profile", "") != "unarmored_diamond_sword") {
            throw std::invalid_argument("Configured environment profile requires a policy contract or equipment preset");
        }
        if (schema_version == 2 && (terrain.weapon_slots[0].size() != 2 || terrain.weapon_slots[1].size() != 2)) {
            throw std::invalid_argument("Equipment preset requires two weapon slots; use a policy contract for other layouts");
        }
        if (contract) {
            layout = std::make_unique<app::PolicyLayout>(nlohmann::json::parse(contract), terrain.weapon_slots);
            features = static_cast<int>(layout->observations.size());
            heads = static_cast<int>(layout->actions.size());
        }
        if (layout) {
            action_heads = layout->packed_actions;
        } else {
            constexpr int sizes[] = {9, 2, 2, 2, 9, 5, 2, 2};
            for (int index = 0; index < heads; ++index) {
                app::ActionHead head;
                head.kind = index;
                head.count = sizes[index];
                for (int choice = 0; choice < head.count; ++choice) {
                    head.values[choice] = index == 4 ? yaw_bins[choice]
                        : (index == 5 ? pitch_bins[choice] : static_cast<float>(choice));
                }
                action_heads.push_back(head);
            }
        }
        scenario = terrain.resolvedDefinition().dump();
        std::copy_n(reward_weights, weights.size(), weights.begin());
        bool has_throwables = false;
        bool has_wind = false;
        for (const auto& player : terrain.initial_players) {
            has_throwables = has_throwables || player.snowball_count > 0 || player.egg_count > 0 || player.ender_pearl_count > 0;
            has_wind = has_wind || player.wind_charge_count > 0;
        }
        constexpr int thrown_capacity = 96;
        constexpr int wind_capacity = 128;
        if (has_throwables) projectile_thrown.resize(std::size_t(count) * thrown_capacity);
        if (has_wind) projectile_wind.resize(std::size_t(count) * wind_capacity);
        for (int environment = 0; environment < count; ++environment) {
            projectile_states[environment].arrows = projectile_arrows.data()
                + std::size_t(environment) * projectile_arrow_capacity;
            projectile_states[environment].arrow_capacity = projectile_arrow_capacity;
            if (has_throwables) {
                projectile_states[environment].thrown_items = projectile_thrown.data() + std::size_t(environment) * thrown_capacity;
                projectile_states[environment].thrown_item_capacity = thrown_capacity;
            }
            if (has_wind) {
                projectile_states[environment].wind_charges = projectile_wind.data() + std::size_t(environment) * wind_capacity;
                projectile_states[environment].wind_charge_capacity = wind_capacity;
            }
        }
        reset(nullptr);
        if (use_cuda) {
#if defined(MCSIM_HAS_CUDA)
            if (!CudaBatch::available()) {
                throw std::runtime_error("CUDA requested but no CUDA device is available");
            }
            gpu = std::make_unique<CudaBatch>(view(), layout != nullptr);
#else
            throw std::runtime_error("CUDA requested but library was built without MCSIM_ENABLE_CUDA");
#endif
        }
    }

    std::chrono::steady_clock::time_point profileStart() const {
        return profiling ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    }

    void profileEnd(ProfilePhase phase, std::chrono::steady_clock::time_point start) {
        if (profiling) {
            profiling_seconds[phase] += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
    }

    BatchView view() {
        return {players.data(), queued.data(), held.data(), worlds.data(),
                static_cast<int>(worlds.size()), 2, blocks.data(), {terrain.sine.data()}, projectile_states.data()};
    }

    void requireHostState() const {
#if defined(MCSIM_HAS_CUDA)
        if (gpu_resident_configured) throw std::invalid_argument("Host state is unavailable after CUDA resident mode activation");
#endif
    }

    void requireDuelRewards() const {
        if (bridge_enabled) throw std::invalid_argument("Duel reward metrics are unavailable for Bridge matches");
    }

#if defined(MCSIM_HAS_CUDA)
    app::ObservationActorMetadata observationActor(int environment, int player) const {
        const auto index = std::size_t(environment) * 2 + player;
        app::ObservationActorMetadata result;
        const auto& slots = episode_players[environment].weapon_slots[player];
        result.slot_count = static_cast<int>(slots.size());
        std::copy(slots.begin(), slots.end(), result.slots);
        result.selected_slot = selected_slots[index];
        result.observed_opponent_disable = observed_opponent_disable[index];
        return result;
    }

    void prepareCudaObservations() {
        if (!gpu) throw std::invalid_argument("Configured CUDA observations require a CUDA batch");
        if (gpu_observations_configured) return;
        std::vector<app::ObservationProjection> projection;
        projection.reserve(features);
        const float infinity = std::numeric_limits<float>::infinity();
        for (int feature = 0; feature < features; ++feature) {
            projection.push_back(layout ? layout->projection(feature)
                : app::ObservationProjection{feature, 1.0F, 0.0F, -infinity, infinity, false});
        }
        std::vector<app::ObservationActorMetadata> actors;
        actors.reserve(players.size());
        for (int environment = 0; environment < static_cast<int>(worlds.size()); ++environment) {
            for (int player = 0; player < 2; ++player) actors.push_back(observationActor(environment, player));
        }
        gpu->configureObservations(projection.data(), projection.size(), actors.data(), actors.size());
        gpu_observations_configured = true;
    }

    void prepareCudaActions() {
        if (gpu_actions_configured) return;
        prepareCudaObservations();
        gpu->configureActions(action_heads.data(), action_heads.size());
        gpu_actions_configured = true;
    }

    void prepareCudaResident() {
        if (gpu_resident_configured) return;
        if (bridge_enabled) throw std::invalid_argument("CUDA resident Bridge collection is not implemented");
        prepareCudaActions();
        app::DuelRewardSettings settings;
        std::copy(weights.begin(), weights.end(), settings.weights);
        settings.block_weight = block_weight;
        std::copy(event_weights.begin(), event_weights.end(), settings.event_weights);
        std::copy(shaping_weights.begin(), shaping_weights.end(), settings.shaping_weights);
        settings.max_ticks = max_ticks;
        std::vector<app::DuelRewardState> states(worlds.size());
        std::vector<PlayerState> initial_players;
        std::vector<app::ObservationActorMetadata> initial_actors;
        initial_players.reserve(players.size());
        initial_actors.reserve(players.size());
        for (int environment = 0; environment < static_cast<int>(worlds.size()); ++environment) {
            for (int player = 0; player < 2; ++player) {
                const auto index = std::size_t(environment) * 2 + player;
                states[environment].dealt_damage[player] = dealt_damage[index] != 0;
                initial_players.push_back(episode_players[environment].players[player]);
                auto metadata = observationActor(environment, player);
                metadata.selected_slot = episode_players[environment].resolved[player]["loadout"]["selected"].get<int>();
                metadata.observed_opponent_disable = 0;
                initial_actors.push_back(metadata);
            }
        }
        gpu->configureRewards(settings, states.data(), states.size(), finished.data());
        gpu->configureFixedResets(initial_players.data(), initial_actors.data(), episodes.data(), episodes.size(), seed);
        gpu_resident_configured = true;
    }
#endif

    std::uint64_t captureTarget() const {
        const char* target = std::getenv("MCSIM_CAPTURE_STEP");
        if (!target || !*target) return 0;
        char* end = nullptr;
        const auto requested = std::strtoull(target, &end, 10);
        if (*end || requested == 0) throw std::invalid_argument("MCSIM_CAPTURE_STEP must be a positive integer");
        return requested;
    }

    void writeCapture(const char* path, std::uint64_t requested) {
        std::ofstream output(path, std::ios::binary);
        if (!output) throw std::runtime_error("Cannot open capture path");
        const nlohmann::json metadata{
            {"format", "mcsim_native_capture"}, {"version", 1}, {"step", requested},
            {"source_hash", MCSIM_SOURCE_HASH}, {"worlds", worlds.size()},
            {"players_per_world", 2}, {"terrain", terrain.definition},
            {"player_size", sizeof(PlayerState)}, {"controls_size", sizeof(Controls)},
            {"world_size", sizeof(WorldState)}, {"projectile_world_size", sizeof(ProjectileWorldState)},
            {"arrow_size", sizeof(ArrowState)}, {"wind_size", sizeof(WindChargeState)},
            {"thrown_size", sizeof(ThrownProjectileState)}, {"block_size", sizeof(Block)},
            {"arrow_capacity", projectile_states[0].arrow_capacity},
            {"wind_capacity", projectile_states[0].wind_charge_capacity},
            {"thrown_capacity", projectile_states[0].thrown_item_capacity},
            {"cell_count", terrain.cells.size()}, {"sine_count", terrain.sine.size()}
        };
        const auto header = metadata.dump();
        const auto header_size = std::uint64_t(header.size());
        const auto write = [&output](const auto* data, std::size_t count) {
            if (count) output.write(reinterpret_cast<const char*>(data), count * sizeof(*data));
        };
        write(&header_size, 1);
        output.write(header.data(), static_cast<std::streamsize>(header.size()));
        write(players.data(), players.size());
        write(queued.data(), queued.size());
        write(held.data(), held.size());
        write(worlds.data(), worlds.size());
        auto states = projectile_states;
        for (auto& state : states) {
            state.arrows = nullptr;
            state.wind_charges = nullptr;
            state.thrown_items = nullptr;
        }
        write(states.data(), states.size());
        write(projectile_arrows.data(), projectile_arrows.size());
        write(projectile_wind.data(), projectile_wind.size());
        write(projectile_thrown.data(), projectile_thrown.size());
        write(terrain.cells.data(), terrain.cells.size());
        write(terrain.sine.data(), terrain.sine.size());
        output.close();
        if (!output) throw std::runtime_error("Failed to write native capture");
        std::printf("NATIVE_CAPTURE step=%llu path=%s worlds=%zu\n",
                    static_cast<unsigned long long>(requested), path, worlds.size());
        std::fflush(stdout);
    }

    const ArrowObservationSummary* currentArrowSummaries() const {
#if defined(MCSIM_HAS_CUDA)
        if (gpu && layout) return arrow_summaries.data();
#endif
        return nullptr;
    }

    void validateBridgeSpawns(const app::Scene::PlayerDefinitions& definitions,
                              const BridgeMatchRules& rules) const {
        if (definitions.players[0].bounds().intersects(definitions.players[1].bounds())) {
            throw std::invalid_argument("Bridge spawns must not overlap");
        }
        for (const auto& player : definitions.players) {
            if (player.position.y < rules.void_y || !noBlockCollision(terrain.blocks, player.bounds())
                || player.bounds().intersects(rules.goals[0]) || player.bounds().intersects(rules.goals[1])) {
                throw std::invalid_argument("Bridge spawns must be safe and outside both goals");
            }
        }
    }

    void validateBridgeKits(const app::Scene::PlayerDefinitions& definitions,
                            const BridgeMatchRules& rules) const {
        for (int player = 0; player < 2; ++player) {
            const auto& kit = rules.kits[player];
            const auto& slots = definitions.weapon_slots[player];
            if (rules.arrow_refill_ticks > 0 && std::find(slots.begin(), slots.end(), WeaponItem::bow) == slots.end()) {
                throw std::invalid_argument("Bridge arrow refill requires a bow for both players");
            }
            if (!kit.explicit_hotbar) continue;
            if (kit.slot_count != static_cast<int>(slots.size())) {
                throw std::invalid_argument("Bridge kit slots must match the scene loadout size");
            }
            for (int slot = 0; slot < kit.slot_count; ++slot) {
                const auto item = kit.slots[slot];
                const auto weapon = slots[slot];
                const bool valid_sword = weapon >= WeaponItem::wooden_sword && weapon <= WeaponItem::netherite_sword;
                if ((item == BridgeHotbarItem::sword && !valid_sword)
                    || (item == BridgeHotbarItem::bow && weapon != WeaponItem::bow)
                    || (item != BridgeHotbarItem::sword && item != BridgeHotbarItem::bow && weapon != WeaponItem::unarmed)) {
                    throw std::invalid_argument("Bridge kit slot disagrees with the scene weapon loadout");
                }
            }
        }
    }

    void validateBridgeKitObservations(const BridgeMatchRules& rules) const {
        bool has_reserved = false;
        for (const auto& kit : rules.kits) {
            for (int slot = 0; kit.explicit_hotbar && slot < kit.slot_count; ++slot) {
                const auto item = kit.slots[slot];
                has_reserved = has_reserved || item == BridgeHotbarItem::pickaxe
                    || item == BridgeHotbarItem::blocks || item == BridgeHotbarItem::golden_apple;
            }
        }
        if (!has_reserved) return;
        if (!layout && schema_version == 2) {
            throw std::invalid_argument("Legacy equipment observations cannot represent reserved Bridge kit items");
        }
        if (!layout) return;
        for (const int source : layout->observations) {
            if (source == 29 || (source >= 35 && source <= 38)) {
                throw std::invalid_argument("Weapon charge/attributes cannot represent reserved Bridge kit items");
            }
            if (source < 43 || source > 78) continue;
            const int slot = (source - 43) / 4;
            for (const auto& kit : rules.kits) {
                if (!kit.explicit_hotbar || slot >= kit.slot_count) continue;
                const auto item = kit.slots[slot];
                if (item == BridgeHotbarItem::pickaxe || item == BridgeHotbarItem::blocks
                    || item == BridgeHotbarItem::golden_apple) {
                    throw std::invalid_argument("Weapon slot observations cannot represent reserved Bridge kit items");
                }
            }
        }
    }

    void configureBridge(const nlohmann::json& config) {
        requireHostState();
        if (bridge_enabled || !config.is_object()) throw std::invalid_argument("Invalid or repeated Bridge configuration");
        for (const auto& field : config.items()) {
            if (field.key() != "mode" && field.key() != "goals" && field.key() != "void_y"
                && field.key() != "match" && field.key() != "task_rewards" && field.key() != "kits"
                && field.key() != "building") {
                throw std::invalid_argument("Unsupported experimental Bridge field: " + field.key());
            }
        }
        if (config.at("mode") != nlohmann::json{{"name", "bridge"}, {"version", 1}}) {
            throw std::invalid_argument("Experimental Bridge requires mode bridge version 1");
        }
        if (std::any_of(blocks.begin(), blocks.end(), [](const auto& view) { return view.overlay != nullptr; })) {
            throw std::invalid_argument("Experimental host-staged Bridge supports static terrain only");
        }
        const auto& goals = config.at("goals");
        if (!goals.is_array() || goals.size() != 2) throw std::invalid_argument("Bridge requires two goal boxes");
        BridgeMatchRules rules;
        for (int team = 0; team < 2; ++team) {
            const auto& box = goals[team];
            if (!box.is_object() || box.size() != 2 || !box.contains("min") || !box.contains("max")
                || !box["min"].is_array() || !box["max"].is_array()
                || box["min"].size() != 3 || box["max"].size() != 3) {
                throw std::invalid_argument("Bridge goals require min and max vectors");
            }
            double lower[3]{}, upper[3]{};
            for (int axis = 0; axis < 3; ++axis) {
                if (!box["min"][axis].is_number() || !box["max"][axis].is_number()) {
                    throw std::invalid_argument("Bridge goal coordinates must be finite numbers");
                }
                lower[axis] = box["min"][axis].get<double>();
                upper[axis] = box["max"][axis].get<double>();
                const int origin = axis == 0 ? terrain.blocks.origin_x
                    : axis == 1 ? terrain.blocks.origin_y : terrain.blocks.origin_z;
                const int size = axis == 0 ? terrain.blocks.size_x
                    : axis == 1 ? terrain.blocks.size_y : terrain.blocks.size_z;
                if (!std::isfinite(lower[axis]) || !std::isfinite(upper[axis]) || lower[axis] >= upper[axis]
                    || lower[axis] < origin || upper[axis] > origin + size) {
                    throw std::invalid_argument("Bridge goal box is invalid or outside terrain bounds");
                }
            }
            rules.goals[team] = {{lower[0], lower[1], lower[2]}, {upper[0], upper[1], upper[2]}};
        }
        if (rules.goals[0].intersects(rules.goals[1])) throw std::invalid_argument("Bridge goals must not overlap");
        if (!config.at("void_y").is_number() || !std::isfinite(config.at("void_y").get<double>())) {
            throw std::invalid_argument("Bridge void_y must be finite");
        }
        rules.void_y = config.at("void_y").get<double>();
        const auto& match = config.at("match");
        if (!match.is_object() || match.size() != 5 || !match.contains("goals_to_win") || !match.contains("max_ticks")
            || !match.contains("cage_ticks") || !match.contains("persist_blocks_on_goal")
            || !match.contains("arrow_refill_ticks") || !match["goals_to_win"].is_number_integer()
            || !match["max_ticks"].is_number_integer() || !match["cage_ticks"].is_number_integer()
            || !match["persist_blocks_on_goal"].is_boolean() || !match["arrow_refill_ticks"].is_number_integer()
            || match["goals_to_win"] < 1 || match["goals_to_win"] > 1000000
            || match["max_ticks"] < 1 || match["max_ticks"] > 1000000
            || match["cage_ticks"] < 0 || match["cage_ticks"] > 1000000
            || match["arrow_refill_ticks"] < 0 || match["arrow_refill_ticks"] > 1000000) {
            throw std::invalid_argument("Invalid Bridge match rules");
        }
        rules.goals_to_win = match["goals_to_win"].get<int>();
        rules.max_ticks = match["max_ticks"].get<int>();
        rules.cage_ticks = match["cage_ticks"].get<int>();
        rules.arrow_refill_ticks = match["arrow_refill_ticks"].get<int>();
        rules.persist_blocks_on_goal = match["persist_blocks_on_goal"].get<bool>();
        if (config.contains("kits")) {
            const auto& kits = config.at("kits");
            if (!kits.is_array() || kits.size() != 2) throw std::invalid_argument("Bridge requires two kit definitions");
            for (int player = 0; player < 2; ++player) {
                const auto& source = kits[player];
                if (!source.is_object() || source.size() != 3 || !source.contains("slots")
                    || !source.contains("blocks") || !source.contains("golden_apples")
                    || !source["slots"].is_array() || source["slots"].empty() || source["slots"].size() > 9
                    || !source["blocks"].is_number_integer() || !source["golden_apples"].is_number_integer()
                    || source["blocks"] < 0 || source["blocks"] > 64
                    || source["golden_apples"] < 0 || source["golden_apples"] > 64) {
                    throw std::invalid_argument("Invalid Bridge kit slots or item counts");
                }
                auto& kit = rules.kits[player];
                kit.explicit_hotbar = true;
                kit.slot_count = static_cast<int>(source["slots"].size());
                kit.blocks = source["blocks"].get<int>();
                kit.golden_apples = source["golden_apples"].get<int>();
                bool seen[6]{};
                for (int slot = 0; slot < kit.slot_count; ++slot) {
                    if (!source["slots"][slot].is_string()) throw std::invalid_argument("Bridge kit items must be strings");
                    const auto name = source["slots"][slot].get<std::string>();
                    const auto item = name == "empty" ? BridgeHotbarItem::empty
                        : name == "sword" ? BridgeHotbarItem::sword : name == "bow" ? BridgeHotbarItem::bow
                        : name == "pickaxe" ? BridgeHotbarItem::pickaxe : name == "blocks" ? BridgeHotbarItem::blocks
                        : name == "golden_apple" ? BridgeHotbarItem::golden_apple
                        : throw std::invalid_argument("Unsupported Bridge kit item: " + name);
                    const auto item_index = static_cast<int>(item);
                    if (item != BridgeHotbarItem::empty && seen[item_index]) {
                        throw std::invalid_argument("Bridge kit items must occupy distinct slots");
                    }
                    seen[item_index] = true;
                    kit.slots[slot] = item;
                }
                if ((kit.blocks > 0 && !seen[static_cast<int>(BridgeHotbarItem::blocks)])
                    || (kit.golden_apples > 0 && !seen[static_cast<int>(BridgeHotbarItem::golden_apple)])) {
                    throw std::invalid_argument("Bridge item counts require corresponding hotbar slots");
                }
            }
        }
        if (config.contains("building")) {
            if (!config.contains("kits")) throw std::invalid_argument("Bridge building requires explicit kits");
            const auto& building = config.at("building");
            if (!building.is_object() || building.size() != 6 || !building.contains("zones")
                || !building.contains("block_range") || !building.contains("entity_range")
                || !building.contains("hardness") || !building.contains("mining_speed")
                || !building.contains("correct_tool") || !building["zones"].is_array()
                || building["zones"].empty() || building["zones"].size() > 8
                || !building["block_range"].is_number() || !building["entity_range"].is_number()
                || !building["hardness"].is_number() || !building["mining_speed"].is_number()
                || !building["correct_tool"].is_boolean()) {
                throw std::invalid_argument("Invalid experimental Bridge building configuration");
            }
            auto& target = rules.building;
            target.enabled = true;
            target.block_range = building["block_range"].get<double>();
            target.entity_range = building["entity_range"].get<double>();
            target.hardness = building["hardness"].get<float>();
            target.mining_speed = building["mining_speed"].get<float>();
            target.correct_tool = building["correct_tool"].get<bool>();
            target.zone_count = static_cast<int>(building["zones"].size());
            for (int zone = 0; zone < target.zone_count; ++zone) {
                const auto& source = building["zones"][zone];
                if (!source.is_object() || source.size() != 2 || !source.contains("min") || !source.contains("max")
                    || !source["min"].is_array() || !source["max"].is_array()
                    || source["min"].size() != 3 || source["max"].size() != 3) {
                    throw std::invalid_argument("Bridge build zones require min and max vectors");
                }
                double lower[3]{}, upper[3]{};
                for (int axis = 0; axis < 3; ++axis) {
                    if (!source["min"][axis].is_number() || !source["max"][axis].is_number()) {
                        throw std::invalid_argument("Bridge build-zone coordinates must be finite numbers");
                    }
                    lower[axis] = source["min"][axis].get<double>();
                    upper[axis] = source["max"][axis].get<double>();
                    const int origin = axis == 0 ? terrain.blocks.origin_x
                        : axis == 1 ? terrain.blocks.origin_y : terrain.blocks.origin_z;
                    const int size = axis == 0 ? terrain.blocks.size_x
                        : axis == 1 ? terrain.blocks.size_y : terrain.blocks.size_z;
                    if (!std::isfinite(lower[axis]) || !std::isfinite(upper[axis])
                        || lower[axis] >= upper[axis] || lower[axis] < origin || upper[axis] > origin + size) {
                        throw std::invalid_argument("Bridge build zone is invalid or outside terrain bounds");
                    }
                }
                target.zones[zone] = {{lower[0], lower[1], lower[2]}, {upper[0], upper[1], upper[2]}};
            }
            if (!validBridgeBuildRules(target)) throw std::invalid_argument("Invalid Bridge building physics parameters");
#if defined(MCSIM_HAS_CUDA)
            if (gpu && gpu_observations_configured) {
                throw std::invalid_argument("Configure Bridge building before registering CUDA observation buffers");
            }
#endif
        }
        const auto& task = config.at("task_rewards");
        if (weights[2] != 0.0F || weights[8] != 0.0F) {
            throw std::invalid_argument("Bridge does not support duel win rewards");
        }
        if (!task.is_object() || !task.contains("score")
            || !task.contains("win") || !task.contains("loss")) {
            throw std::invalid_argument("Bridge requires explicit score, win and loss rewards");
        }
        for (const auto& field : task.items()) {
            if (field.key() != "score" && field.key() != "win" && field.key() != "loss"
                && field.key() != "opponent_score" && field.key() != "toward" && field.key() != "away") {
                throw std::invalid_argument("Unsupported Bridge reward field");
            }
        }
        float task_weights[3]{};
        const char* names[3]{"score", "win", "loss"};
        for (int index = 0; index < 3; ++index) {
            if (!task[names[index]].is_number()) throw std::invalid_argument("Bridge reward weights must be finite");
            const double value = task[names[index]].get<double>();
            if (!std::isfinite(value) || std::abs(value) > (std::numeric_limits<float>::max)()) {
                throw std::invalid_argument("Bridge reward weights must be finite floats");
            }
            task_weights[index] = static_cast<float>(value);
        }
        double opponent_score = 0.0;
        if (task.contains("opponent_score")) {
            if (!task["opponent_score"].is_number()) throw std::invalid_argument("Bridge reward weights must be finite");
            opponent_score = task["opponent_score"].get<double>();
            if (!std::isfinite(opponent_score) || std::abs(opponent_score) > (std::numeric_limits<float>::max)()) {
                throw std::invalid_argument("Bridge reward weights must be finite floats");
            }
        }
        for (int scored = 0; scored < 2; ++scored) {
            for (int opposed = 0; opposed < 2; ++opposed) {
                for (int outcome = 0; outcome < 3; ++outcome) {
                    const double total = scored * double(task_weights[0]) + opposed * opponent_score
                        + (outcome == 0 ? 0.0 : double(task_weights[outcome]));
                    if (std::abs(total) > (std::numeric_limits<float>::max)()) {
                        throw std::invalid_argument("Combined Bridge score and outcome rewards must be finite floats");
                    }
                }
            }
        }
        float progress_weights[2]{};
        const char* progress_names[2]{"toward", "away"};
        for (int index = 0; index < 2; ++index) {
            if (!task.contains(progress_names[index])) continue;
            if (!task[progress_names[index]].is_number()) throw std::invalid_argument("Invalid objective progress weight");
            const double value = task[progress_names[index]].get<double>();
            if (!std::isfinite(value) || value < 0.0 || value > 1000.0) {
                throw std::invalid_argument("Objective progress weights must be in 0..1000");
            }
            progress_weights[index] = static_cast<float>(value);
        }
        validateBridgeKitObservations(rules);
        for (const auto& definitions : episode_players) {
            validateBridgeSpawns(definitions, rules);
            validateBridgeKits(definitions, rules);
        }
        if (rules.building.enabled) {
            auto overlays = std::make_unique<TerrainOverlays>(terrain.blocks, worlds.size());
            auto previous = blocks;
            for (std::size_t environment = 0; environment < worlds.size(); ++environment) {
                blocks[environment] = overlays->view(environment);
            }
            try {
#if defined(MCSIM_HAS_CUDA)
                if (gpu) gpu = std::make_unique<CudaBatch>(view(), layout != nullptr);
#endif
            } catch (...) {
                blocks = std::move(previous);
                throw;
            }
            bridge_terrain = std::move(overlays);
        }
        bridge_initial.resize(worlds.size());
        bridge_states.resize(worlds.size());
        bridge_events.resize(worlds.size());
        objective_progress.resize(players.size());
        bridge_reward_terms.resize(players.size());
        bridge_rules = rules;
        bridge_score_reward = task_weights[0];
        bridge_win_reward = task_weights[1];
        bridge_loss_reward = task_weights[2];
        bridge_opponent_score_reward = static_cast<float>(opponent_score);
        bridge_toward_reward = progress_weights[0];
        bridge_away_penalty = progress_weights[1];
        bridge_configuration = config;
        auto resolved = nlohmann::json::parse(scenario);
        resolved.erase("reward_weights");
        resolved["experimental_bridge"] = config;
        scenario = resolved.dump();
        auto original = nlohmann::json::parse(definition);
        original.erase("reward_weights");
        original["experimental_bridge"] = config;
        definition = original.dump();
        bridge_enabled = true;
        reset(nullptr);
    }

    void reset(const std::uint8_t* mask) {
        requireHostState();
        const auto start = profileStart();
        std::vector<int> ids;
        std::vector<PlayerState> initial;
        std::vector<ProjectileWorldState> initial_projectiles;
    #if defined(MCSIM_HAS_CUDA)
        std::vector<app::ObservationActorMetadata> initial_observation_actors;
    #endif
        for (int environment = 0; environment < static_cast<int>(worlds.size()); ++environment) {
            if (mask && !mask[environment]) {
                continue;
            }
            ids.push_back(environment);
            blocks[environment].clearPlacedBlocks();
            worlds[environment] = {};
            worlds[environment].simulate_fall_damage = terrain.worlds[0].simulate_fall_damage;
            finished[environment] = 0;
            const auto episode = episodes[environment]++;
            if (bridge_enabled) {
                bridge_states[environment] = initialBridgeMatch(bridge_rules);
                bridge_states[environment].episodes = episodes[environment];
                bridge_events[environment] = BridgeTickEvents();
                for (int player = 0; player < 2; ++player) {
                    bridge_initial[environment].players[player] = episode_players[environment].players[player];
                    resetBridgeKit(bridge_states[environment], bridge_rules,
                                   bridge_initial[environment].players[player], player);
                }
            }
            resetProjectileWorld(projectile_states[environment],
                seed + std::uint64_t(environment) * 1000003ULL + episode * 1000033ULL);
            for (int player = 0; player < 2; ++player) {
                const auto index = std::size_t(environment) * 2 + player;
                players[index] = episode_players[environment].players[player];
                if (bridge_enabled) {
                    const auto& goal = bridge_rules.goals[1 - player];
                    const Vec3 target{(goal.lower.x + goal.upper.x) / 2.0,
                                      (goal.lower.y + goal.upper.y) / 2.0,
                                      (goal.lower.z + goal.upper.z) / 2.0};
                    objective_progress[index].reset(players[index].position, target);
                    bridge_reward_terms[index].fill(0.0);
                }
                players[index].random.setSeed(seed + index * 1000003ULL + episode * 1000033ULL);
                projectile_states[environment].bows[player].ordinary_arrows = players[index].ordinary_arrow_count;
                arrow_summaries[index] = summarizeArrowObservation(
                    players[index], projectile_states[environment], player);
                queued[index] = {};
                held[index] = {};
                selected_slots[index] = episode_players[environment].resolved[player]["loadout"]["selected"].get<int>();
                dealt_damage[index] = 0;
                observed_opponent_disable[index] = 0;
                step_reward_totals[index].fill(0.0);
                initial.push_back(players[index]);
#if defined(MCSIM_HAS_CUDA)
                if (gpu_observations_configured) initial_observation_actors.push_back(observationActor(environment, player));
#endif
            }
            initial_projectiles.push_back(projectile_states[environment]);
        }
#if defined(MCSIM_HAS_CUDA)
        if (gpu) {
            gpu->resetEnvironments(ids.data(), ids.size(), initial.data(), initial_projectiles.data(),
                gpu_observations_configured ? initial_observation_actors.data() : nullptr);
            if (bridge_enabled && !ids.empty()) {
                gpu->configureBridge(bridge_rules, bridge_initial.data(), bridge_states.data(), worlds.size(), max_ticks);
            }
            if (profiling) gpu->synchronize();
        }
#endif
        profileEnd(ResetApply, start);
    }

    void resetPlayers(const std::uint8_t* mask, const nlohmann::json& definitions) {
        requireHostState();
        const auto start = profileStart();
        if (!definitions.is_array() || definitions.size() != worlds.size()) {
            throw std::invalid_argument("Reset players must be an array with one entry per world");
        }
        std::vector<std::pair<std::size_t, app::Scene::PlayerDefinitions>> pending;
        for (std::size_t environment = 0; environment < worlds.size(); ++environment) {
            const bool selected = !mask || mask[environment] != 0;
            if (!selected) {
                if (!definitions[environment].is_null()) {
                    throw std::invalid_argument("Unselected reset worlds must have null player definitions");
                }
                continue;
            }
            auto parsed = terrain.parsePlayers(definitions[environment], true);
            if (bridge_enabled) {
                validateBridgeSpawns(parsed, bridge_rules);
                validateBridgeKits(parsed, bridge_rules);
            }
            if (projectile_thrown.empty()) {
                for (const auto& player : parsed.players) {
                    if (player.snowball_count > 0 || player.egg_count > 0 || player.ender_pearl_count > 0) {
                        throw std::invalid_argument("Throwable reset loadouts require an initial throwable loadout");
                    }
                }
            }
            if (projectile_wind.empty()) {
                for (const auto& player : parsed.players) {
                    if (player.wind_charge_count > 0) {
                        throw std::invalid_argument("Wind charge reset loadouts require an initial wind charge loadout");
                    }
                }
            }
            for (const auto& slots : parsed.weapon_slots) {
                if (schema_version == 2 && slots.size() != 2) {
                    throw std::invalid_argument("Equipment preset requires two weapon slots");
                }
                if (layout) {
                    for (const auto& head : layout->actions) {
                        if (head.kind != 6) continue;
                        for (float value : head.values) {
                            if (static_cast<std::size_t>(value) >= slots.size()) {
                                throw std::invalid_argument("Weapon action outside reset player's loadout");
                            }
                        }
                    }
                }
            }
            pending.emplace_back(environment, std::move(parsed));
        }
        for (auto& [environment, parsed] : pending) episode_players[environment] = std::move(parsed);
        profileEnd(ResetValidate, start);
        reset(mask);
    }

    std::string scenarioFor(int environment) const {
        if (environment < 0 || environment >= static_cast<int>(worlds.size())) {
            throw std::invalid_argument("Scenario environment index outside batch");
        }
        auto resolved = nlohmann::json::parse(scenario);
        resolved["players"] = episode_players[environment].resolved;
        if (bridge_enabled) resolved["experimental_bridge"] = bridge_configuration;
        return resolved.dump();
    }

    bool outside(const PlayerState& player, std::size_t environment) const {
        return app::duelOutside(player, blocks[environment]);
    }

    void observe(float* output, const ArrowObservationSummary* summaries = nullptr) const {
        requireHostState();
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (int environment = 0; environment < static_cast<int>(worlds.size()); ++environment) {
            for (int player = 0; player < 2; ++player) {
                const auto index = std::size_t(environment) * 2 + player;
                const auto& source = players[index];
                const auto& target = players[std::size_t(environment) * 2 + (1 - player)];
                float catalogue[app::feature_names.size()]{};
                float* observation = layout ? catalogue : output + index * features;
                app::baseWorldObservation(source, target, blocks[environment], observation);
                if (schema_version != 1) {
                    app::legacyEquipmentObservation(observation, source, target, observed_opponent_disable[index]);
                }
                if (layout) {
                    catalogue[31] = static_cast<float>(selected_slots[index]);
                    catalogue[32] = static_cast<float>(selected_slots[index ^ 1]);
                    app::velocityObservation(catalogue, source, target);
                    app::movementObservation(catalogue, source, target);
                    app::equipmentObservation(catalogue, source, target,
                        episode_players[environment].weapon_slots[player]);
                    if (summaries) {
                        app::arrowObservation(catalogue, source, target, summaries[index]);
                    } else {
                        app::arrowObservation(catalogue, source, target, projectile_states[environment], player);
                    }
                    layout->project(catalogue, output + index * features);
                }
            }
        }
    }

    void recordReward(std::size_t index, std::size_t term, double amount) {
        step_reward_totals[index][term] += amount;
        if (term < reward_totals.size()) {
            reward_totals[term] += amount;
        } else {
            shaping_totals[term - reward_totals.size()] += amount;
        }
    }

    void step(const std::int32_t* actions, float* observations, float* rewards,
              std::uint8_t* terminated, std::uint8_t* truncated) {
        requireHostState();
        if (bridge_enabled && captureTarget()) {
            throw std::invalid_argument("Native capture does not support Bridge state");
        }
        const char* trace_value = std::getenv("MCSIM_TRACE_NATIVE");
        const bool trace = trace_value && trace_value[0] == '1' && trace_value[1] == '\0';
        const auto decode_start = profileStart();
        if (std::any_of(finished.begin(), finished.end(), [](auto value) { return value != 0; })) {
            throw std::invalid_argument("Reset finished environments before the next step");
        }
        for (std::size_t index = 0; index < players.size(); ++index) {
            const auto* action = actions + index * heads;
            for (int head = 0; head < heads; ++head) {
                const auto& descriptor = action_heads[head];
                if (action[head] < 0 || action[head] >= descriptor.count) {
                    throw std::invalid_argument("Action index outside declared policy head");
                }
                if (descriptor.kind == 6 && static_cast<std::size_t>(descriptor.values[action[head]])
                    >= episode_players[index / 2].weapon_slots[index % 2].size()) {
                    throw std::invalid_argument("Weapon action outside loadout");
                }
            }
        }
        for (std::size_t index = 0; index < players.size(); ++index) {
            const auto* action = actions + index * heads;
            Controls controls;
            const auto& slots = episode_players[index / 2].weapon_slots[index % 2];
            if (!app::decodeAction(action_heads.data(), heads, action, slots.data(),
                                   static_cast<int>(slots.size()), controls)) {
                throw std::invalid_argument("Action index or selected slot outside declared policy");
            }
            if (controls.select_weapon && !bridge_enabled) selected_slots[index] = controls.selected_slot;
            queued[index] = controls;
        }
            profileEnd(StepDecode, decode_start);
        const auto capture_target = captureTarget();
        const bool capture_now = capture_target && ++capture_step_counter == capture_target;
        const char* capture_path = capture_now ? std::getenv("MCSIM_CAPTURE_PATH") : nullptr;
        if (capture_now && (!capture_path || !*capture_path)) {
            throw std::invalid_argument("MCSIM_CAPTURE_PATH is required");
        }
        if (capture_now && std::filesystem::exists(capture_path)) {
            throw std::runtime_error("Capture path already exists");
        }
#if defined(MCSIM_HAS_CUDA)
        if (gpu) {
                const auto upload_start = profileStart();
            traceNative(trace, "controls_upload_begin");
            gpu->uploadControls(queued.data(), queued.size());
            traceNative(trace, "controls_upload_done");
                profileEnd(ControlsUpload, upload_start);
            if (capture_now && capture_target == 1) gpu->download(view());
                const auto physics_start = profileStart();
            traceNative(trace, "kernel_launch_begin");
            gpu->step();
            traceNative(trace, "kernel_launch_done");
            if (capture_now) writeCapture(capture_path, capture_target);
            if (profiling) {
                traceNative(trace, "cuda_synchronize_begin");
                gpu->synchronize();
                traceNative(trace, "cuda_synchronize_done");
            }
                profileEnd(Physics, physics_start);
                const auto download_start = profileStart();
            traceNative(trace, "state_download_begin");
            gpu->downloadForTraining(view(), layout ? arrow_summaries.data() : nullptr);
            if (bridge_enabled) gpu->downloadBridge(bridge_states.data(), bridge_events.data(), worlds.size());
            if (bridge_enabled && bridge_rules.building.enabled) {
                for (std::size_t environment = 0; environment < worlds.size(); ++environment) {
                    const auto& events = bridge_events[environment];
                    const auto results = applyTerrainEdits(blocks[environment], events.terrain_edits);
                    for (int player = 0; player < 2; ++player) {
                        if (results.applied[player] != (events.terrain_edits[player].action != TerrainEditAction::none)) {
                            throw std::runtime_error("CUDA Bridge terrain delta disagrees with host overlay");
                        }
                    }
                    if (events.round_reset && !bridge_rules.persist_blocks_on_goal) {
                        blocks[environment].resetPlacedBlocks(TerrainReset::round, false);
                    }
                }
            }
            traceNative(trace, "state_download_done");
            if (capture_target > 1 && capture_step_counter == capture_target - 1) {
                gpu->download(view());
                traceNative(trace, "capture_prefetch_done");
            }
                profileEnd(StateDownload, download_start);
        } else
#endif
        {
            if (capture_now) writeCapture(capture_path, capture_target);
                const auto physics_start = profileStart();
            traceNative(trace, "cpu_physics_begin");
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (int environment = 0; environment < static_cast<int>(worlds.size()); ++environment) {
                if (bridge_enabled) {
                    bridge_events[environment] = processBridgeWorldTick(view().world(environment),
                        projectile_states[environment], bridge_initial[environment].players,
                        bridge_rules, bridge_states[environment], max_ticks);
                } else {
                    processWorldTick(view().world(environment), projectile_states[environment]);
                }
            }
            traceNative(trace, "cpu_physics_done");
            profileEnd(Physics, physics_start);
        }
        const auto rewards_start = profileStart();
        std::fill_n(rewards, players.size(), 0.0F);
        for (auto& totals : step_reward_totals) totals.fill(0.0);
        app::DuelRewardSettings reward_settings;
        std::copy(weights.begin(), weights.end(), reward_settings.weights);
        reward_settings.block_weight = block_weight;
        std::copy(event_weights.begin(), event_weights.end(), reward_settings.event_weights);
        std::copy(shaping_weights.begin(), shaping_weights.end(), reward_settings.shaping_weights);
        reward_settings.max_ticks = max_ticks;
        for (std::size_t environment = 0; environment < worlds.size(); ++environment) {
            const auto offset = environment * 2;
            if (bridge_enabled) {
                const auto& events = bridge_events[environment];
                if (!events.advanced) throw std::runtime_error("Bridge tick did not advance");
                terminated[environment] = events.terminated;
                truncated[environment] = !events.terminated && bridge_states[environment].ticks >= max_ticks;
                finished[environment] = terminated[environment] || truncated[environment];
                PlayerState combat_players[2]{players[offset], players[offset + 1]};
                for (int player = 0; player < 2; ++player) {
                    combat_players[player].last_attack = events.attacks[player];
                    combat_players[player].weapon = events.attack_weapons[player];
                }
                app::DuelRewardState reward_state{{dealt_damage[offset] != 0, dealt_damage[offset + 1] != 0}};
                auto shared = app::sharedRewardEvents(combat_players, worlds[environment], reward_settings, reward_state);
                const bool timeout = truncated[environment] || (events.terminated
                    && bridge_states[environment].ticks >= bridge_rules.max_ticks
                    && bridge_states[environment].scores[0] < bridge_rules.goals_to_win
                    && bridge_states[environment].scores[1] < bridge_rules.goals_to_win);
                for (int player = 0; player < 2; ++player) {
                    const auto index = offset + player;
                    dealt_damage[index] = reward_state.dealt_damage[player];
                    if (events.died[player]) app::addDuelReward(shared, player, 3, reward_settings.weights[3]);
                    if (timeout) {
                        app::addDuelReward(shared, player, 4, reward_settings.weights[4]);
                        if (!reward_state.dealt_damage[player]) {
                            app::addDuelReward(shared, player, 5, reward_settings.weights[5]);
                        }
                    }
                    if (!finished[environment] && !events.round_reset && !events.respawn[0] && !events.respawn[1]
                        && std::any_of(shaping_weights.begin(), shaping_weights.end(), [](float weight) { return weight != 0.0F; })) {
                        float shaping[5]{};
                        app::shapingRewards(players[index], players[offset + 1 - player],
                            players[index].ordinary_arrow_count,
                            blocks[environment], reward_settings.shaping_weights, shaping);
                        for (int term = 0; term < 5; ++term) app::addDuelReward(shared, player, 13 + term, shaping[term]);
                    }
                    selected_slots[index] = players[index].selected_weapon_slot;
                    const bool restored = !finished[environment]
                        && (events.round_reset || events.respawn[0] || events.respawn[1]);
                    if (restored) observed_opponent_disable[index] = 0;
                    else {
                        if (observed_opponent_disable[index] > 0) --observed_opponent_disable[index];
                        if (players[index].last_attack.shield_disabled) observed_opponent_disable[index] = 99;
                    }
                    const bool teleported = events.round_reset || events.respawn[player];
                    const auto progress = objective_progress[index].advance(players[index].position,
                        teleported, events.round_reset && events.scored[player]);
                    auto& terms = bridge_reward_terms[index];
                    terms = {events.scored[player] ? double(bridge_score_reward) : 0.0,
                             events.scored[1 - player] ? double(bridge_opponent_score_reward) : 0.0,
                             progress.toward / 50.0 * bridge_toward_reward,
                             -progress.away / 50.0 * bridge_away_penalty};
                    rewards[index] = static_cast<float>(terms[0] + terms[1] + terms[2] + terms[3]) + shared.rewards[player];
                    for (std::size_t term = 0; term < step_reward_totals[index].size(); ++term) {
                        recordReward(index, term, shared.contributions[player][term]);
                    }
                    const auto winner = player == 0 ? BridgeOutcome::team0 : BridgeOutcome::team1;
                    if (events.terminated && events.outcome != BridgeOutcome::draw) {
                        rewards[index] += events.outcome == winner ? bridge_win_reward : bridge_loss_reward;
                    }
                }
                continue;
            }
            app::DuelRewardState reward_state{{dealt_damage[offset] != 0, dealt_damage[offset + 1] != 0}};
            for (int player = 0; player < 2; ++player) {
                const auto index = offset + player;
                const auto& state = players[index];
                if (schema_version != 1) {
                    if (observed_opponent_disable[index] > 0) --observed_opponent_disable[index];
                    if (state.last_attack.shield_disabled) observed_opponent_disable[index] = 99;
                }
            }
            const auto outcome = app::duelRewards(players.data() + offset, worlds[environment], blocks[environment],
                                                   projectile_states[environment], reward_settings, reward_state);
            terminated[environment] = outcome.terminated;
            truncated[environment] = outcome.truncated;
            finished[environment] = terminated[environment] || truncated[environment];
            for (int player = 0; player < 2; ++player) {
                const auto index = offset + player;
                rewards[index] = outcome.rewards[player];
                dealt_damage[index] = reward_state.dealt_damage[player];
                for (std::size_t term = 0; term < step_reward_totals[index].size(); ++term) {
                    recordReward(index, term, outcome.contributions[player][term]);
                }
            }
        }
        profileEnd(Rewards, rewards_start);
        const auto observation_start = profileStart();
        observe(observations, currentArrowSummaries());
        profileEnd(StepObserve, observation_start);
    }
};

nlohmann::json bridgeStateDescription(const TrainingBatch& batch, std::size_t index) {
    const auto& state = batch.bridge_states[index];
    const auto& events = batch.bridge_events[index];
    const char* outcome = state.outcome == BridgeOutcome::team0 ? "team0"
        : state.outcome == BridgeOutcome::team1 ? "team1"
        : state.outcome == BridgeOutcome::draw ? "draw" : "ongoing";
    return {{"scores", {state.scores[0], state.scores[1]}}, {"ticks", state.ticks},
        {"cage_remaining", state.cage_remaining}, {"outcome", outcome},
        {"terminated", events.terminated},
        {"truncated", !events.terminated && batch.finished[index] != 0},
        {"scored", {events.scored[0], events.scored[1]}},
        {"died", {events.died[0], events.died[1]}},
        {"placed", {events.placed[0], events.placed[1]}},
        {"broke", {events.broke[0], events.broke[1]}},
        {"respawn", {events.respawn[0] && !batch.finished[index], events.respawn[1] && !batch.finished[index]}},
        {"round_reset", events.round_reset && !batch.finished[index]}};
}

template <class Function>
int protect(Function function) {
    try {
        function();
        last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}
}

MCSIM_EXPORT const char* mcsim_last_error() { return last_error.c_str(); }
MCSIM_EXPORT int mcsim_api_version() { return 5; }
MCSIM_EXPORT const char* mcsim_build_info() {
    static const std::string description = nlohmann::json{
        {"abi", 5}, {"minecraft", "26.3"}, {"version", MCSIM_BUILD_VERSION},
        {"source_hash", MCSIM_SOURCE_HASH}, {"revision", MCSIM_SOURCE_REVISION},
        {"compiler", MCSIM_BUILD_COMPILER}, {"compiled", __DATE__ " " __TIME__},
        {"catalogue_hash", app::catalogue_hash}, {"capabilities", app::capability_names},
        {"observation_projection_version", 1},
        {"cuda_base_observation_version", 1},
        {"cuda_configured_observation_version", 1},
        {"cuda_async_observation_version", 1},
        {"cuda_action_decode_version", 1},
        {"cuda_resident_step_version", 1},
        {"cuda_resident_episode_version", 1},
        {"bridge_host_batch_version", 3},
        {"bridge_shared_reward_version", 1},
        {"bridge_local_voxels_version", 1},
        {"bridge_opponent_score_version", 1},
        {"bridge_objective_reward_version", 1},
        {"actor_task_state_version", 1},
        {"behavior_events_version", 1},
        {"players_per_environment", 2},
        {"scalar_bytes", {{"observation", sizeof(float)}, {"action", sizeof(std::int32_t)},
                           {"reward_total", sizeof(double)}, {"done", sizeof(std::uint8_t)}}},
        {"reward_inputs", {{"base", 10}, {"events", 2}, {"shaping", 5}}},
        {"reward_totals", {{"base", 13}, {"shaping", 5}}},
#if defined(MCSIM_HAS_CUDA)
        {"cuda_compiled", true}
#else
        {"cuda_compiled", false}
#endif
    }.dump();
    return description.c_str();
}
MCSIM_EXPORT const char* mcsim_batch_info(void* handle) {
    if (!handle) return nullptr;
    const auto& batch = *static_cast<TrainingBatch*>(handle);
    std::vector<std::size_t> sizes{9, 2, 2, 2, 9, 5};
    if (batch.schema_version == 2) { sizes.push_back(2); sizes.push_back(2); }
    if (batch.layout) {
        sizes.clear();
        for (const auto& head : batch.layout->actions) sizes.push_back(head.values.size());
    }
    static thread_local std::string description;
    description = nlohmann::json{{"environments", batch.worlds.size()}, {"players_per_environment", 2},
        {"observation_dim", batch.features}, {"action_sizes", sizes}}.dump();
    return description.c_str();
}
MCSIM_EXPORT const char* mcsim_environment(void* handle) {
    if (!handle) { return nullptr; }
    return static_cast<TrainingBatch*>(handle)->definition.c_str();
}
MCSIM_EXPORT const char* mcsim_scenario(void* handle) {
    return handle ? static_cast<TrainingBatch*>(handle)->scenario.c_str() : nullptr;
}
MCSIM_EXPORT const char* mcsim_scenario_for(void* handle, int environment) {
    thread_local std::string scenario;
    const int result = protect([&] {
        if (!handle) throw std::invalid_argument("Null scenario handle");
        scenario = static_cast<TrainingBatch*>(handle)->scenarioFor(environment);
    });
    return result == 0 ? scenario.c_str() : nullptr;
}
MCSIM_EXPORT int mcsim_cuda_available() {
#if defined(MCSIM_HAS_CUDA)
    return minecraft_sim::CudaBatch::available() ? 1 : 0;
#else
    return 0;
#endif
}
MCSIM_EXPORT void* mcsim_create(int environments, int use_cuda, const char* path, int max_ticks,
                               std::uint64_t seed, const float* weights) {
    TrainingBatch* result = nullptr;
    protect([&] {
        if (environments < 1 || environments > 65536 || max_ticks < 1 || max_ticks > 1000000
            || (use_cuda != 0 && use_cuda != 1) || !weights) {
            throw std::invalid_argument("Invalid training batch configuration");
        }
        for (int index = 0; index < 10; ++index) {
            if (!std::isfinite(weights[index])) {
                throw std::invalid_argument("Reward weights must be finite");
            }
        }
        result = new TrainingBatch(environments, use_cuda != 0, path, max_ticks, seed, weights);
    });
    return result;
}
MCSIM_EXPORT void* mcsim_create_schema(int environments, int use_cuda, const char* path, int max_ticks,
                                      std::uint64_t seed, const float* weights, int schema_version) {
    if (schema_version == 1) return mcsim_create(environments, use_cuda, path, max_ticks, seed, weights);
    TrainingBatch* result = nullptr;
    protect([&] {
        if (schema_version != 2 || environments < 1 || environments > 65536 || max_ticks < 1 || max_ticks > 1000000
            || (use_cuda != 0 && use_cuda != 1) || !weights) {
            throw std::invalid_argument("Invalid training schema or batch configuration");
        }
        for (int index = 0; index < 10; ++index) {
            if (!std::isfinite(weights[index])) throw std::invalid_argument("Reward weights must be finite");
        }
        result = new TrainingBatch(environments, use_cuda != 0, path, max_ticks, seed, weights, 2);
    });
    return result;
}
MCSIM_EXPORT void mcsim_destroy(void* handle) { delete static_cast<TrainingBatch*>(handle); }
MCSIM_EXPORT int mcsim_set_profiling(void* handle, int enabled) {
    return protect([&] {
        if (!handle || (enabled != 0 && enabled != 1)) throw std::invalid_argument("Invalid profiling configuration");
        auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.profiling = enabled != 0;
        if (enabled) batch.profiling_seconds.fill(0.0);
    });
}
MCSIM_EXPORT const char* mcsim_profile(void* handle) {
    static thread_local std::string description;
    const auto result = protect([&] {
        if (!handle) throw std::invalid_argument("Null profiling handle");
        const auto& batch = *static_cast<TrainingBatch*>(handle);
        nlohmann::json seconds = nlohmann::json::object();
        for (std::size_t index = 0; index < profile_names.size(); ++index) {
            seconds[profile_names[index]] = batch.profiling_seconds[index];
        }
        description = nlohmann::json{{"version", 1}, {"seconds", seconds}}.dump();
    });
    return result == 0 ? description.c_str() : nullptr;
}
MCSIM_EXPORT int mcsim_set_max_ticks(void* handle, int max_ticks) {
    return protect([&] {
        if (!handle || max_ticks < 1 || max_ticks > 1000000) {
            throw std::invalid_argument("Episode timeout must be in 1..1000000 with a valid handle");
        }
        auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        if (batch.bridge_enabled) throw std::invalid_argument("Bridge external cutoff is fixed at configuration time");
        batch.max_ticks = max_ticks;
    });
}
MCSIM_EXPORT int mcsim_set_reward_weights(void* handle, const float* weights) {
    return protect([&] {
        if (!handle || !weights) throw std::invalid_argument("Null base reward argument");
        auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        if (batch.bridge_enabled && (weights[2] != 0.0F || weights[8] != 0.0F)) {
            throw std::invalid_argument("Bridge does not support duel win rewards");
        }
        for (std::size_t index = 0; index < batch.weights.size(); ++index) {
            if (!std::isfinite(weights[index])) throw std::invalid_argument("Reward weights must be finite");
        }
        std::copy_n(weights, batch.weights.size(), batch.weights.begin());
    });
}
MCSIM_EXPORT int mcsim_set_block_reward(void* handle, float weight) {
    return protect([&] {
        if (!handle) throw std::invalid_argument("Null block reward argument");
        if (!std::isfinite(weight)) throw std::invalid_argument("Reward weights must be finite");
        auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        batch.block_weight = weight;
    });
}
MCSIM_EXPORT int mcsim_set_event_rewards(void* handle, const float* weights) {
    return protect([&] {
        if (!handle || !weights) throw std::invalid_argument("Null event reward argument");
        auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        for (std::size_t index = 0; index < batch.event_weights.size(); ++index) {
            if (!std::isfinite(weights[index])) throw std::invalid_argument("Reward weights must be finite");
        }
        std::copy_n(weights, batch.event_weights.size(), batch.event_weights.begin());
    });
}
MCSIM_EXPORT int mcsim_set_shaping_rewards(void* handle, const float* weights) {
    return protect([&] {
        if (!handle || !weights) throw std::invalid_argument("Null shaping reward argument");
        auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        for (std::size_t index = 0; index < batch.shaping_weights.size(); ++index) {
            if (!std::isfinite(weights[index])) throw std::invalid_argument("Reward weights must be finite");
        }
        std::copy_n(weights, batch.shaping_weights.size(), batch.shaping_weights.begin());
    });
}
MCSIM_EXPORT void* mcsim_create_contract(int environments, int use_cuda, const char* path, int max_ticks,
                                        std::uint64_t seed, const float* weights, const char* contract) {
    TrainingBatch* result = nullptr;
    protect([&] {
        if (environments < 1 || environments > 65536 || max_ticks < 1 || max_ticks > 1000000
            || (use_cuda != 0 && use_cuda != 1) || !weights || !contract) {
            throw std::invalid_argument("Invalid configured training batch");
        }
        for (int index = 0; index < 10; ++index) if (!std::isfinite(weights[index])) throw std::invalid_argument("Reward weights must be finite");
        result = new TrainingBatch(environments, use_cuda != 0, path, max_ticks, seed, weights, 3, contract);
    });
    return result;
}
MCSIM_EXPORT int mcsim_configure_bridge(void* handle, const char* json_utf8) {
    return protect([&] {
        if (!handle || !json_utf8) throw std::invalid_argument("Null Bridge configuration argument");
        static_cast<TrainingBatch*>(handle)->configureBridge(nlohmann::json::parse(json_utf8));
    });
}
MCSIM_EXPORT const char* mcsim_bridge_state(void* handle) {
    static thread_local std::string description;
    const auto result = protect([&] {
        if (!handle) throw std::invalid_argument("Null Bridge state handle");
        const auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        if (!batch.bridge_enabled) throw std::invalid_argument("Bridge is not configured");
        auto worlds = nlohmann::json::array();
        for (std::size_t index = 0; index < batch.worlds.size(); ++index) {
            worlds.push_back(bridgeStateDescription(batch, index));
        }
        description = worlds.dump();
    });
    return result == 0 ? description.c_str() : nullptr;
}
MCSIM_EXPORT const char* mcsim_bridge_kits(void* handle, int player) {
    static thread_local std::string description;
    const auto result = protect([&] {
        if (!handle || player < 0 || player >= 2) throw std::invalid_argument("Invalid Bridge kit query");
        const auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        if (!batch.bridge_enabled) throw std::invalid_argument("Bridge is not configured");
        auto worlds = nlohmann::json::array();
        for (const auto& state : batch.bridge_states) {
            const auto& kit = state.kits[player];
            worlds.push_back({{"blocks", kit.blocks}, {"golden_apples", kit.golden_apples},
                {"arrows", kit.arrows}, {"arrow_refill_elapsed", kit.arrow_refill_elapsed},
                {"selected_slot", kit.selected_slot}, {"use_delay", kit.building.use_delay},
                {"break_delay", kit.building.break_delay}, {"break_progress", kit.building.break_progress}});
        }
        description = worlds.dump();
    });
    return result == 0 ? description.c_str() : nullptr;
}
MCSIM_EXPORT int mcsim_actor_task_state(void* handle, double* output, std::size_t count) {
    return protect([&] {
        if (!handle || !output) throw std::invalid_argument("Null actor task state argument");
        const auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        if (count != batch.players.size() * 9) throw std::invalid_argument("Invalid actor task state buffer size");
        for (std::size_t index = 0; index < batch.players.size(); ++index) {
            const auto& actor = batch.players[index];
            const auto world = index / 2;
            const auto player = index % 2;
            double* row = output + index * 9;
            row[0] = actor.position.x;
            row[1] = actor.position.y;
            row[2] = actor.position.z;
            row[3] = actor.yaw;
            row[4] = actor.pitch;
            const ItemKitState empty_kit;
            const auto progress = itemProgressState(actor, batch.bridge_enabled ? batch.bridge_states[world].kits[player] : empty_kit);
            row[5] = progress.eating_remaining;
            row[6] = progress.mining_progress;
            row[7] = progress.use_delay;
            row[8] = progress.mining_delay;
        }
    });
}
MCSIM_EXPORT int mcsim_bridge_reward_terms(void* handle, double* output, std::size_t count) {
    return protect([&] {
        if (!handle || !output) throw std::invalid_argument("Null Bridge reward terms argument");
        const auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        if (!batch.bridge_enabled || count != batch.players.size() * 4) {
            throw std::invalid_argument("Invalid Bridge reward terms buffer");
        }
        for (std::size_t index = 0; index < batch.players.size(); ++index) {
            std::copy(batch.bridge_reward_terms[index].begin(), batch.bridge_reward_terms[index].end(), output + index * 4);
        }
    });
}
MCSIM_EXPORT int mcsim_observe(void* handle, float* observations) {
    return protect([&] {
        if (!handle || !observations) { throw std::invalid_argument("Null observation argument"); }
        auto& batch = *static_cast<TrainingBatch*>(handle);
        const auto start = batch.profileStart();
        batch.observe(observations, batch.currentArrowSummaries());
        batch.profileEnd(Observe, start);
    });
}
MCSIM_EXPORT int mcsim_bridge_local_voxels(void* handle, std::uint8_t* output, std::size_t count) {
    return protect([&] {
        if (!handle || !output) throw std::invalid_argument("Null Bridge local voxel argument");
        const auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        if (!batch.bridge_enabled) throw std::invalid_argument("Bridge is not configured");
        const auto expected = batch.players.size() * std::size_t(bridge_local_voxel_count);
        if (count != expected) throw std::invalid_argument("Invalid Bridge local voxel buffer size");
        std::vector<std::uint8_t> voxels(expected);
        for (std::size_t world = 0; world < batch.worlds.size(); ++world) {
            for (int player = 0; player < 2; ++player) {
                const auto index = world * 2 + player;
                const auto& actor = batch.players[index];
                if (!bridgeLocalVoxels(batch.blocks[world], actor.position, actor.yaw, player,
                                       voxels.data() + index * bridge_local_voxel_count)) {
                    throw std::runtime_error("Invalid Bridge local voxel snapshot");
                }
            }
        }
        std::copy(voxels.begin(), voxels.end(), output);
    });
}
MCSIM_EXPORT int mcsim_observe_base_cuda_sync(void* handle, float* observations, std::size_t count, void* stream) {
    return protect([&] {
        if (!handle) throw std::invalid_argument("Null observation handle");
#if defined(MCSIM_HAS_CUDA)
        auto& batch = *static_cast<TrainingBatch*>(handle);
        if (!batch.gpu) throw std::invalid_argument("Base CUDA observations require a CUDA batch");
        batch.gpu->observeBaseSynchronized(observations, count, stream);
#else
        static_cast<void>(observations);
        static_cast<void>(count);
        static_cast<void>(stream);
        throw std::invalid_argument("Base CUDA observations require a CUDA-enabled library");
#endif
    });
}
MCSIM_EXPORT int mcsim_prepare_cuda_observations(void* handle) {
    return protect([&] {
        if (!handle) throw std::invalid_argument("Null observation handle");
#if defined(MCSIM_HAS_CUDA)
        static_cast<TrainingBatch*>(handle)->prepareCudaObservations();
#else
        throw std::invalid_argument("Configured CUDA observations require a CUDA-enabled library");
#endif
    });
}
MCSIM_EXPORT int mcsim_observe_cuda_sync(void* handle, float* observations, std::size_t count, void* stream) {
    return protect([&] {
        if (!handle) throw std::invalid_argument("Null observation handle");
#if defined(MCSIM_HAS_CUDA)
        auto& batch = *static_cast<TrainingBatch*>(handle);
        if (!batch.gpu || !batch.gpu_observations_configured) {
            throw std::invalid_argument("Prepare configured CUDA observations before use");
        }
        batch.gpu->observeConfiguredSynchronized(observations, count, stream);
#else
        static_cast<void>(observations);
        static_cast<void>(count);
        static_cast<void>(stream);
        throw std::invalid_argument("Configured CUDA observations require a CUDA-enabled library");
#endif
    });
}
MCSIM_EXPORT int mcsim_register_cuda_observations(void* handle, float* observations, std::size_t count) {
    return protect([&] {
        if (!handle) throw std::invalid_argument("Null observation handle");
#if defined(MCSIM_HAS_CUDA)
        auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.prepareCudaObservations();
        batch.gpu->registerConfiguredObservationBuffer(observations, count);
#else
        static_cast<void>(observations);
        static_cast<void>(count);
        throw std::invalid_argument("Configured CUDA observations require a CUDA-enabled library");
#endif
    });
}
MCSIM_EXPORT int mcsim_observe_cuda_async(void* handle, void* stream) {
    return protect([&] {
        if (!handle) throw std::invalid_argument("Null observation handle");
#if defined(MCSIM_HAS_CUDA)
        auto& batch = *static_cast<TrainingBatch*>(handle);
        if (!batch.gpu || !batch.gpu_observations_configured) {
            throw std::invalid_argument("Register a CUDA observation buffer before use");
        }
        batch.gpu->observeRegisteredOnStream(stream);
#else
        static_cast<void>(stream);
        throw std::invalid_argument("Configured CUDA observations require a CUDA-enabled library");
#endif
    });
}
MCSIM_EXPORT int mcsim_wait_cuda_observations(void* handle) {
    return protect([&] {
        if (!handle) throw std::invalid_argument("Null observation handle");
#if defined(MCSIM_HAS_CUDA)
        auto& batch = *static_cast<TrainingBatch*>(handle);
        if (!batch.gpu || !batch.gpu_observations_configured) {
            throw std::invalid_argument("Register a CUDA observation buffer before use");
        }
        batch.gpu->waitForConfiguredObservations();
#else
        throw std::invalid_argument("Configured CUDA observations require a CUDA-enabled library");
#endif
    });
}
MCSIM_EXPORT int mcsim_register_cuda_actions(void* handle, const std::int32_t* actions, std::size_t count) {
    return protect([&] {
        if (!handle) throw std::invalid_argument("Null action handle");
#if defined(MCSIM_HAS_CUDA)
        auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.prepareCudaActions();
        batch.gpu->registerActionBuffer(actions, count);
#else
        static_cast<void>(actions);
        static_cast<void>(count);
        throw std::invalid_argument("CUDA action decoding requires a CUDA-enabled library");
#endif
    });
}
MCSIM_EXPORT int mcsim_decode_cuda_actions_async(void* handle, void* stream) {
    return protect([&] {
        if (!handle) throw std::invalid_argument("Null action handle");
#if defined(MCSIM_HAS_CUDA)
        auto& batch = *static_cast<TrainingBatch*>(handle);
        if (!batch.gpu || !batch.gpu_actions_configured) {
            throw std::invalid_argument("Register a CUDA action buffer before use");
        }
        batch.gpu->decodeRegisteredActionsOnStream(stream);
#else
        static_cast<void>(stream);
        throw std::invalid_argument("CUDA action decoding requires a CUDA-enabled library");
#endif
    });
}
MCSIM_EXPORT int mcsim_cuda_action_status(void* handle, int* status) {
    return protect([&] {
        if (!handle || !status) throw std::invalid_argument("Null action status argument");
#if defined(MCSIM_HAS_CUDA)
        auto& batch = *static_cast<TrainingBatch*>(handle);
        if (!batch.gpu || !batch.gpu_actions_configured) {
            throw std::invalid_argument("Register a CUDA action buffer before use");
        }
        *status = batch.gpu->actionStatus();
#else
        throw std::invalid_argument("CUDA action decoding requires a CUDA-enabled library");
#endif
    });
}
MCSIM_EXPORT int mcsim_clear_cuda_action_status(void* handle) {
    return protect([&] {
        if (!handle) throw std::invalid_argument("Null action handle");
#if defined(MCSIM_HAS_CUDA)
        auto& batch = *static_cast<TrainingBatch*>(handle);
        if (!batch.gpu || !batch.gpu_actions_configured) {
            throw std::invalid_argument("Register a CUDA action buffer before use");
        }
        batch.gpu->clearActionStatus();
#else
        throw std::invalid_argument("CUDA action decoding requires a CUDA-enabled library");
#endif
    });
}
MCSIM_EXPORT int mcsim_register_cuda_step_outputs(void* handle, float* rewards, std::uint8_t* terminated,
                                                  std::uint8_t* truncated, std::uint8_t* losses,
                                                  double* contributions, std::size_t worlds) {
    return protect([&] {
        if (!handle) throw std::invalid_argument("Null resident step handle");
#if defined(MCSIM_HAS_CUDA)
        auto& batch = *static_cast<TrainingBatch*>(handle);
        if (!batch.gpu || !rewards || !terminated || !truncated || !losses || !contributions
            || worlds != batch.worlds.size()) throw std::invalid_argument("Invalid CUDA resident output buffers");
        batch.gpu->validateDeviceBuffer(rewards);
        batch.gpu->validateDeviceBuffer(terminated);
        batch.gpu->validateDeviceBuffer(truncated);
        batch.gpu->validateDeviceBuffer(losses);
        batch.gpu->validateDeviceBuffer(contributions);
        batch.prepareCudaResident();
        batch.gpu->registerRewardBuffers(rewards, terminated, truncated, losses, contributions, worlds);
#else
        static_cast<void>(rewards);
        static_cast<void>(terminated);
        static_cast<void>(truncated);
        static_cast<void>(losses);
        static_cast<void>(contributions);
        static_cast<void>(worlds);
        throw std::invalid_argument("CUDA resident stepping requires a CUDA-enabled library");
#endif
    });
}
MCSIM_EXPORT int mcsim_register_cuda_reset_mask(void* handle, const std::uint8_t* mask, std::size_t worlds) {
    return protect([&] {
        if (!handle) throw std::invalid_argument("Null resident reset handle");
#if defined(MCSIM_HAS_CUDA)
        auto& batch = *static_cast<TrainingBatch*>(handle);
        if (!batch.gpu || !batch.gpu_resident_configured) throw std::invalid_argument("Prepare CUDA resident stepping first");
        batch.gpu->registerResetMask(mask, worlds);
#else
        static_cast<void>(mask);
        static_cast<void>(worlds);
        throw std::invalid_argument("CUDA resident stepping requires a CUDA-enabled library");
#endif
    });
}
MCSIM_EXPORT int mcsim_step_cuda_resident_async(void* handle, void* stream) {
    return protect([&] {
        if (!handle) throw std::invalid_argument("Null resident step handle");
#if defined(MCSIM_HAS_CUDA)
        auto& batch = *static_cast<TrainingBatch*>(handle);
        if (!batch.gpu || !batch.gpu_resident_configured) throw std::invalid_argument("Prepare CUDA resident stepping first");
        batch.gpu->stepRegisteredOnStream(stream);
#else
        static_cast<void>(stream);
        throw std::invalid_argument("CUDA resident stepping requires a CUDA-enabled library");
#endif
    });
}
MCSIM_EXPORT int mcsim_reset_cuda_resident_async(void* handle, void* stream) {
    return protect([&] {
        if (!handle) throw std::invalid_argument("Null resident reset handle");
#if defined(MCSIM_HAS_CUDA)
        auto& batch = *static_cast<TrainingBatch*>(handle);
        if (!batch.gpu || !batch.gpu_resident_configured) throw std::invalid_argument("Prepare CUDA resident stepping first");
        batch.gpu->resetRegisteredOnStream(stream);
#else
        static_cast<void>(stream);
        throw std::invalid_argument("CUDA resident stepping requires a CUDA-enabled library");
#endif
    });
}
MCSIM_EXPORT int mcsim_bridge_reset_episodes(void* handle, std::uint64_t* episodes, std::size_t worlds) {
    return protect([&] {
        if (!handle || !episodes) throw std::invalid_argument("Null Bridge reset episode argument");
        const auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        if (!batch.bridge_enabled) throw std::invalid_argument("Bridge is not configured");
        if (worlds != batch.episodes.size()) throw std::invalid_argument("Invalid Bridge reset episode count");
        std::copy(batch.episodes.begin(), batch.episodes.end(), episodes);
    });
}
MCSIM_EXPORT int mcsim_set_bridge_reset_episodes(void* handle, const std::uint64_t* episodes, std::size_t worlds) {
    return protect([&] {
        if (!handle || !episodes) throw std::invalid_argument("Null Bridge reset episode argument");
        auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        if (!batch.bridge_enabled) throw std::invalid_argument("Bridge is not configured");
        if (worlds != batch.episodes.size()) throw std::invalid_argument("Invalid Bridge reset episode count");
        if (std::any_of(episodes, episodes + worlds, [](auto episode) {
                return episode == (std::numeric_limits<std::uint64_t>::max)();
            })) throw std::invalid_argument("Bridge reset episode counter cannot be incremented");
        std::copy_n(episodes, worlds, batch.episodes.begin());
        batch.reset(nullptr);
    });
}
MCSIM_EXPORT int mcsim_cuda_reset_episodes(void* handle, std::uint64_t* episodes, std::size_t worlds) {
    return protect([&] {
        if (!handle || !episodes) throw std::invalid_argument("Null CUDA reset episode argument");
#if defined(MCSIM_HAS_CUDA)
        auto& batch = *static_cast<TrainingBatch*>(handle);
        if (!batch.gpu || !batch.gpu_resident_configured) throw std::invalid_argument("Prepare CUDA resident stepping first");
        batch.gpu->downloadFixedResetEpisodes(episodes, worlds);
#else
        static_cast<void>(worlds);
        throw std::invalid_argument("CUDA resident stepping requires a CUDA-enabled library");
#endif
    });
}
MCSIM_EXPORT int mcsim_set_cuda_reset_episodes(void* handle, const std::uint64_t* episodes, std::size_t worlds) {
    return protect([&] {
        if (!handle || !episodes) throw std::invalid_argument("Null CUDA reset episode argument");
#if defined(MCSIM_HAS_CUDA)
        auto& batch = *static_cast<TrainingBatch*>(handle);
        if (!batch.gpu || !batch.gpu_resident_configured) throw std::invalid_argument("Prepare CUDA resident stepping first");
        batch.gpu->uploadFixedResetEpisodes(episodes, worlds);
#else
        static_cast<void>(worlds);
        throw std::invalid_argument("CUDA resident stepping requires a CUDA-enabled library");
#endif
    });
}
MCSIM_EXPORT int mcsim_reward_totals(void* handle, double* totals) {
    return protect([&] {
        if (!handle || !totals) throw std::invalid_argument("Null reward totals argument");
        const auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        batch.requireDuelRewards();
        std::copy(batch.reward_totals.begin(), batch.reward_totals.end(), totals);
    });
}
MCSIM_EXPORT int mcsim_step_reward_totals(void* handle, const std::uint8_t* mask, double* totals) {
    return protect([&] {
        if (!handle || !totals) throw std::invalid_argument("Null step reward totals argument");
        const auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        if (mask && std::any_of(mask, mask + batch.players.size(), [](auto value) { return value > 1; })) {
            throw std::invalid_argument("Step reward mask must contain one boolean per player");
        }
        std::array<double, 18> selected{};
        for (std::size_t index = 0; index < batch.players.size(); ++index) {
            if (mask && !mask[index]) continue;
            for (std::size_t term = 0; term < selected.size(); ++term) {
                selected[term] += batch.step_reward_totals[index][term];
            }
        }
        std::copy(selected.begin(), selected.end(), totals);
    });
}
MCSIM_EXPORT int mcsim_step_behavior_events(void* handle, float* events) {
    return protect([&] {
        if (!handle || !events) throw std::invalid_argument("Null behavior events argument");
        const auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        if (batch.bridge_enabled) throw std::invalid_argument("Bridge behavior metrics are not implemented");
        for (std::size_t index = 0; index < batch.players.size(); ++index) {
            const auto& player = batch.players[index];
            events[index * 3] = static_cast<float>(batch.worlds[index / 2].wind_self_launch[index % 2]);
            events[index * 3 + 1] = player.last_attack.mace_smash ? 1.0F : 0.0F;
            events[index * 3 + 2] = player.on_ground ? 1.0F : 0.0F;
        }
    });
}
MCSIM_EXPORT int mcsim_losses(void* handle, std::uint8_t* losses) {
    return protect([&] {
        if (!handle || !losses) throw std::invalid_argument("Null losses argument");
        const auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        for (std::size_t index = 0; index < batch.players.size(); ++index) {
            const auto& state = batch.players[index];
            if (batch.bridge_enabled) {
                const auto outcome = batch.bridge_states[index / 2].outcome;
                losses[index] = (outcome == BridgeOutcome::team0 && index % 2 == 1)
                    || (outcome == BridgeOutcome::team1 && index % 2 == 0);
            } else {
                losses[index] = !state.alive() || batch.outside(state, index / 2);
            }
        }
    });
}
MCSIM_EXPORT int mcsim_shaping_reward_totals(void* handle, double* totals) {
    return protect([&] {
        if (!handle || !totals) throw std::invalid_argument("Null shaping totals argument");
        const auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        batch.requireDuelRewards();
        std::copy(batch.shaping_totals.begin(), batch.shaping_totals.end(), totals);
    });
}
MCSIM_EXPORT const char* mcsim_replay_frame(void* handle, int environment) {
    thread_local std::string frame;
    const auto result = protect([&] {
        if (!handle) { throw std::invalid_argument("Null replay handle"); }
        auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.requireHostState();
        if (environment < 0 || environment >= static_cast<int>(batch.worlds.size())) {
            throw std::invalid_argument("Replay environment index outside batch");
        }
    #if defined(MCSIM_HAS_CUDA)
        if (batch.gpu) batch.gpu->download(batch.view());
    #endif
        auto recorded = minecraft_sim::app::replayFrame(batch.worlds[environment].tick,
            batch.players.data() + environment * 2, batch.held.data() + environment * 2, true,
            batch.selected_slots.data() + environment * 2,
            &batch.projectile_states[environment]);
        if (batch.bridge_enabled) {
            auto bridge = bridgeStateDescription(batch, environment);
            bridge["terrain_edits"] = nlohmann::json::array();
            for (int player = 0; player < 2; ++player) {
                const auto& edit = batch.bridge_events[environment].terrain_edits[player];
                if (edit.action == TerrainEditAction::none) continue;
                bridge["terrain_edits"].push_back({{"player", player},
                    {"action", edit.action == TerrainEditAction::place ? "place" : "remove"},
                    {"position", {edit.x, edit.y, edit.z}}});
            }
            bridge["clear_overlay"] = bridge["round_reset"].get<bool>() && !batch.bridge_rules.persist_blocks_on_goal;
            recorded["bridge"] = std::move(bridge);
        }
        frame = recorded.dump();
    });
    return result == 0 ? frame.c_str() : nullptr;
}
MCSIM_EXPORT int mcsim_reset(void* handle, const std::uint8_t* mask, float* observations) {
    return protect([&] {
        if (!handle || !observations) { throw std::invalid_argument("Null reset argument"); }
        auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.reset(mask);
        const auto start = batch.profileStart();
        batch.observe(observations, batch.currentArrowSummaries());
        batch.profileEnd(ResetObserve, start);
    });
}
MCSIM_EXPORT int mcsim_reset_players(void* handle, const std::uint8_t* mask,
                                   const char* json_utf8, float* observations) {
    return protect([&] {
        if (!handle || !json_utf8 || !observations) throw std::invalid_argument("Null player reset argument");
        auto& batch = *static_cast<TrainingBatch*>(handle);
        const auto parse_start = batch.profileStart();
        const auto definitions = nlohmann::json::parse(json_utf8);
        batch.profileEnd(ResetJson, parse_start);
        batch.resetPlayers(mask, definitions);
        const auto observation_start = batch.profileStart();
        batch.observe(observations, batch.currentArrowSummaries());
        batch.profileEnd(ResetObserve, observation_start);
    });
}
MCSIM_EXPORT int mcsim_step(void* handle, const std::int32_t* actions, float* observations, float* rewards,
                          std::uint8_t* terminated, std::uint8_t* truncated) {
    return protect([&] {
        if (!handle || !actions || !observations || !rewards || !terminated || !truncated) {
            throw std::invalid_argument("Null step argument");
        }
        static_cast<TrainingBatch*>(handle)->step(actions, observations, rewards, terminated, truncated);
    });
}
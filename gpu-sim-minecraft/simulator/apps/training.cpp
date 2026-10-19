#include "scene.hpp"
#include "replay.hpp"
#include "observation.hpp"
#include "minecraft_sim/cuda_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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
constexpr double radians = 0.017453292519943295;
constexpr std::array<float, 9> yaw_bins{-20, -10, -5, -2, 0, 2, 5, 10, 20};
constexpr std::array<float, 5> pitch_bins{-5, -2, 0, 2, 5};
constexpr std::array<int, 9> forward{0, 1, 1, 0, -1, -1, -1, 0, 1};
constexpr std::array<int, 9> strafe{0, 0, -1, -1, -1, 0, 1, 1, 1};

double wrapYaw(double angle) {
    return angle - 360.0 * std::floor((angle + 180.0) / 360.0);
}

struct TrainingBatch {
    app::Scene terrain;
    std::string definition = terrain.definition.dump();
    std::string scenario;
    std::unique_ptr<app::PolicyLayout> layout;
    std::vector<app::Scene::PlayerDefinitions> episode_players;
    std::vector<PlayerState> players;
    std::vector<Controls> queued;
    std::vector<Controls> held;
    std::vector<int> selected_slots;
    std::vector<WorldState> worlds;
    std::vector<BlockWorldView> blocks;
    std::vector<std::uint8_t> finished;
    std::vector<std::uint8_t> dealt_damage;
    std::vector<std::uint64_t> episodes;
    std::vector<int> observed_opponent_disable;
    std::array<float, 6> weights;
    float block_weight = 0.0F;
    std::array<double, 7> reward_totals{};
    std::array<float, 5> shaping_weights{};
    std::array<double, 5> shaping_totals{};
    std::uint64_t seed;
    int max_ticks;
    int schema_version;
    int features;
    int heads;
#if defined(MCSIM_HAS_CUDA)
    std::unique_ptr<CudaBatch> gpu;
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
          queued(players.size()), held(players.size()), selected_slots(players.size()), worlds(count), blocks(count, terrain.blocks),
          finished(count), dealt_damage(players.size()), episodes(count), observed_opponent_disable(players.size()),
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
        scenario = terrain.resolvedDefinition().dump();
        std::copy_n(reward_weights, weights.size(), weights.begin());
        reset(nullptr);
        if (use_cuda) {
#if defined(MCSIM_HAS_CUDA)
            if (!CudaBatch::available()) {
                throw std::runtime_error("CUDA requested but no CUDA device is available");
            }
            gpu = std::make_unique<CudaBatch>(view());
#else
            throw std::runtime_error("CUDA requested but library was built without MCSIM_ENABLE_CUDA");
#endif
        }
    }

    BatchView view() {
        return {players.data(), queued.data(), held.data(), worlds.data(),
                static_cast<int>(worlds.size()), 2, blocks.data(), {terrain.sine.data()}};
    }

    void reset(const std::uint8_t* mask) {
        std::vector<int> ids;
        std::vector<PlayerState> initial;
        for (int environment = 0; environment < static_cast<int>(worlds.size()); ++environment) {
            if (mask && !mask[environment]) {
                continue;
            }
            ids.push_back(environment);
            worlds[environment] = {};
            finished[environment] = 0;
            const auto episode = episodes[environment]++;
            for (int player = 0; player < 2; ++player) {
                const auto index = std::size_t(environment) * 2 + player;
                players[index] = episode_players[environment].players[player];
                players[index].random.setSeed(seed + index * 1000003ULL + episode * 1000033ULL);
                queued[index] = {};
                held[index] = {};
                selected_slots[index] = episode_players[environment].resolved[player]["loadout"]["selected"].get<int>();
                dealt_damage[index] = 0;
                observed_opponent_disable[index] = 0;
                initial.push_back(players[index]);
            }
        }
#if defined(MCSIM_HAS_CUDA)
        if (gpu) {
            gpu->resetEnvironments(ids.data(), ids.size(), initial.data());
        }
#endif
    }

    void resetPlayers(const std::uint8_t* mask, const nlohmann::json& definitions) {
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
        reset(mask);
    }

    std::string scenarioFor(int environment) const {
        if (environment < 0 || environment >= static_cast<int>(worlds.size())) {
            throw std::invalid_argument("Scenario environment index outside batch");
        }
        auto resolved = nlohmann::json::parse(scenario);
        resolved["players"] = episode_players[environment].resolved;
        return resolved.dump();
    }

    bool outside(const PlayerState& player) const {
        const auto& position = player.position;
        const auto& bounds = terrain.blocks;
        return position.x < bounds.origin_x || position.x >= bounds.origin_x + bounds.size_x
            || position.z < bounds.origin_z || position.z >= bounds.origin_z + bounds.size_z
            || position.y < bounds.origin_y - 4;
    }

    void observe(float* output) const {
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
                std::array<float, 8> rays{};
                for (int ray = 0; ray < 8; ++ray) {
                    const double angle = ray * 45.0 * radians;
                    rays[ray] = static_cast<float>(blockRayFraction(terrain.blocks,
                        source.eyePosition(), {-std::sin(angle) * 50.0, 0.0, std::cos(angle) * 50.0}));
                }
                const auto base = app::baseObservation(source, target, rays);
                std::copy(base.begin(), base.end(), observation);
                if (schema_version != 1) {
                    observation[22] = weaponAttributes(source.weapon).disable_shield_seconds > 0 ? 1.0F : 0.0F;
                    observation[23] = weaponAttributes(target.weapon).disable_shield_seconds > 0 ? 1.0F : 0.0F;
                    observation[24] = source.using_shield ? 1.0F : 0.0F;
                    observation[25] = target.using_shield ? 1.0F : 0.0F;
                    observation[26] = source.shield_cooldown > 0 ? 1.0F : 0.0F;
                    observation[27] = observed_opponent_disable[index] > 0 ? 1.0F : 0.0F;
                    observation[28] = static_cast<float>(minimum(source.shield_cooldown / 100.0F, 1.0F));
                    observation[29] = source.attackCharge();
                    observation[30] = source.using_shield
                        ? static_cast<float>(minimum(source.shield_use_ticks / 5.0F, 1.0F)) : 0.0F;
                }
                if (layout) {
                    catalogue[31] = static_cast<float>(selected_slots[index]);
                    catalogue[32] = static_cast<float>(selected_slots[index ^ 1]);
                    app::equipmentObservation(catalogue, source, target,
                        episode_players[environment].weapon_slots[player]);
                    for (int feature = 0; feature < features; ++feature) {
                        output[index * features + feature] = catalogue[layout->observations[feature]] * layout->scales[feature];
                    }
                }
            }
        }
    }

    void addShapingReward(std::size_t index, float* rewards) {
        const auto& source = players[index];
        const auto& target = players[index ^ 1];
        const Vec3 relative{target.position.x - source.position.x,
                            target.position.y - source.position.y,
                            target.position.z - source.position.z};
        const double distance = std::sqrt(relative.lengthSquared());
        const double pitch_error = std::abs(source.pitch);
        std::array<double, 5> amounts{0.0, std::clamp((10.0 - distance) / 7.0, 0.0, 1.0),
            std::max(0.0, 1.0 - pitch_error / 90.0), 0.0, pitch_error > 60.0 ? 1.0 : 0.0};
        const auto source_eye = source.eyePosition();
        const auto target_eye = target.eyePosition();
        const Vec3 aim{target_eye.x - source_eye.x, target_eye.y - source_eye.y,
                       target_eye.z - source_eye.z};
        if (aim.lengthSquared() > 0.0001 && blockRayFraction(terrain.blocks, source_eye, aim) >= 1.0) {
            const double yaw_error = std::abs(wrapYaw(std::atan2(-aim.x, aim.z) / radians - source.yaw));
            const double aim_pitch = -std::atan2(aim.y, std::hypot(aim.x, aim.z)) / radians;
            const double aim_error = std::max(yaw_error, std::abs(aim_pitch - source.pitch));
            amounts[0] = std::max(0.0, 1.0 - aim_error / 30.0);
            amounts[3] = aim_error < 20.0 ? 1.0 : 0.0;
        }
        for (std::size_t term = 0; term < amounts.size(); ++term) {
            const float amount = static_cast<float>(amounts[term] * shaping_weights[term]);
            rewards[index] += amount;
            shaping_totals[term] += amount;
        }
    }

    void step(const std::int32_t* actions, float* observations, float* rewards,
              std::uint8_t* terminated, std::uint8_t* truncated) {
        if (std::any_of(finished.begin(), finished.end(), [](auto value) { return value != 0; })) {
            throw std::invalid_argument("Reset finished environments before the next step");
        }
        constexpr std::array<int, 8> dimensions{9, 2, 2, 2, 9, 5, 2, 2};
        for (std::size_t index = 0; index < players.size(); ++index) {
            const auto* action = actions + index * heads;
            for (int head = 0; head < heads; ++head) {
                const auto size = layout ? static_cast<int>(layout->actions[head].values.size()) : dimensions[head];
                if (action[head] < 0 || action[head] >= size) {
                    throw std::invalid_argument("Action index outside declared policy head");
                }
            }
        }
        for (std::size_t index = 0; index < players.size(); ++index) {
            const auto* action = actions + index * heads;
            if (layout) {
                queued[index] = layout->decode(action, episode_players[index / 2].weapon_slots[index % 2]);
                for (std::size_t head = 0; head < layout->actions.size(); ++head) {
                    if (layout->actions[head].kind == 6) {
                        selected_slots[index] = static_cast<int>(layout->actions[head].values[action[head]]);
                    }
                }
                continue;
            }
            Controls controls;
            controls.forward = forward[action[0]] > 0;
            controls.backward = forward[action[0]] < 0;
            controls.left = strafe[action[0]] > 0;
            controls.right = strafe[action[0]] < 0;
            controls.jump = action[1] != 0;
            controls.sprint = action[2] != 0;
            controls.attack = action[3] != 0;
            controls.yaw_delta = yaw_bins[action[4]];
            controls.pitch_delta = pitch_bins[action[5]];
            if (schema_version == 2) {
                controls.select_weapon = true;
                const auto& slots = episode_players[index / 2].weapon_slots[index % 2];
                if (action[6] >= static_cast<int>(slots.size())) throw std::invalid_argument("Weapon action outside loadout");
                controls.weapon = slots[action[6]];
                selected_slots[index] = action[6];
                controls.use_item = action[7] != 0;
            }
            queued[index] = controls;
        }
#if defined(MCSIM_HAS_CUDA)
        if (gpu) {
            gpu->uploadControls(queued.data(), queued.size());
            gpu->step();
            gpu->download(view());
        } else
#endif
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (int environment = 0; environment < static_cast<int>(worlds.size()); ++environment) {
                view().world(environment).processTick();
            }
        }
        std::fill_n(rewards, players.size(), 0.0F);
        for (std::size_t environment = 0; environment < worlds.size(); ++environment) {
            const auto offset = environment * 2;
            bool lost[2]{};
            for (int player = 0; player < 2; ++player) {
                const auto index = offset + player;
                const auto& state = players[index];
                if (schema_version != 1) {
                    if (observed_opponent_disable[index] > 0) --observed_opponent_disable[index];
                    if (state.last_attack.shield_disabled) observed_opponent_disable[index] = 99;
                }
                if (state.last_attack.health_lost > 0.0F) {
                    const float health_fraction = state.last_attack.health_lost / 20.0F;
                    rewards[index] += weights[0] * health_fraction;
                    rewards[offset + (1 - player)] += weights[1] * health_fraction;
                    reward_totals[0] += weights[0] * health_fraction;
                    reward_totals[1] += weights[1] * health_fraction;
                    dealt_damage[index] = 1;
                }
                if (state.last_attack.blocked_damage > 0.0F
                    && weaponAttributes(state.weapon).disable_shield_seconds == 0.0F) {
                    rewards[offset + (1 - player)] += block_weight;
                    reward_totals[6] += block_weight;
                }
                lost[player] = !state.alive() || outside(state);
            }
            terminated[environment] = lost[0] || lost[1];
            truncated[environment] = !terminated[environment] && worlds[environment].tick >= std::uint64_t(max_ticks);
            finished[environment] = terminated[environment] || truncated[environment];
            for (int player = 0; player < 2; ++player) {
                const auto index = offset + player;
                if (!terminated[environment]) addShapingReward(index, rewards);
                if (terminated[environment]) {
                    rewards[index] += lost[player] ? weights[3] : weights[2];
                    reward_totals[lost[player] ? 3 : 2] += lost[player] ? weights[3] : weights[2];
                } else if (truncated[environment]) {
                    rewards[index] += weights[4] + (dealt_damage[index] ? 0.0F : weights[5]);
                    reward_totals[4] += weights[4];
                    if (!dealt_damage[index]) reward_totals[5] += weights[5];
                }
            }
        }
        observe(observations);
    }
};

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
MCSIM_EXPORT int mcsim_api_version() { return 2; }
MCSIM_EXPORT const char* mcsim_build_info() {
    static const std::string description = nlohmann::json{
        {"abi", 2}, {"minecraft", "26.3"}, {"version", MCSIM_BUILD_VERSION},
        {"source_hash", MCSIM_SOURCE_HASH}, {"revision", MCSIM_SOURCE_REVISION},
        {"compiler", MCSIM_BUILD_COMPILER}, {"compiled", __DATE__ " " __TIME__},
        {"catalogue_hash", app::catalogue_hash}, {"capabilities", app::capability_names},
        {"players_per_environment", 2},
        {"scalar_bytes", {{"observation", sizeof(float)}, {"action", sizeof(std::int32_t)},
                           {"reward_total", sizeof(double)}, {"done", sizeof(std::uint8_t)}}},
        {"reward_inputs", {{"base", 6}, {"shaping", 5}}},
        {"reward_totals", {{"base", 7}, {"shaping", 5}}},
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
        for (int index = 0; index < 6; ++index) {
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
        for (int index = 0; index < 6; ++index) {
            if (!std::isfinite(weights[index])) throw std::invalid_argument("Reward weights must be finite");
        }
        result = new TrainingBatch(environments, use_cuda != 0, path, max_ticks, seed, weights, 2);
    });
    return result;
}
MCSIM_EXPORT void mcsim_destroy(void* handle) { delete static_cast<TrainingBatch*>(handle); }
MCSIM_EXPORT int mcsim_set_max_ticks(void* handle, int max_ticks) {
    return protect([&] {
        if (!handle || max_ticks < 1 || max_ticks > 1000000) {
            throw std::invalid_argument("Episode timeout must be in 1..1000000 with a valid handle");
        }
        static_cast<TrainingBatch*>(handle)->max_ticks = max_ticks;
    });
}
MCSIM_EXPORT int mcsim_set_reward_weights(void* handle, const float* weights) {
    return protect([&] {
        if (!handle || !weights) throw std::invalid_argument("Null base reward argument");
        auto& batch = *static_cast<TrainingBatch*>(handle);
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
        static_cast<TrainingBatch*>(handle)->block_weight = weight;
    });
}
MCSIM_EXPORT int mcsim_set_shaping_rewards(void* handle, const float* weights) {
    return protect([&] {
        if (!handle || !weights) throw std::invalid_argument("Null shaping reward argument");
        auto& batch = *static_cast<TrainingBatch*>(handle);
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
        for (int index = 0; index < 6; ++index) if (!std::isfinite(weights[index])) throw std::invalid_argument("Reward weights must be finite");
        result = new TrainingBatch(environments, use_cuda != 0, path, max_ticks, seed, weights, 3, contract);
    });
    return result;
}
MCSIM_EXPORT int mcsim_observe(void* handle, float* observations) {
    return protect([&] {
        if (!handle || !observations) { throw std::invalid_argument("Null observation argument"); }
        static_cast<TrainingBatch*>(handle)->observe(observations);
    });
}
MCSIM_EXPORT int mcsim_reward_totals(void* handle, double* totals) {
    return protect([&] {
        if (!handle || !totals) throw std::invalid_argument("Null reward totals argument");
        const auto& batch = *static_cast<TrainingBatch*>(handle);
        std::copy(batch.reward_totals.begin(), batch.reward_totals.end(), totals);
    });
}
MCSIM_EXPORT int mcsim_shaping_reward_totals(void* handle, double* totals) {
    return protect([&] {
        if (!handle || !totals) throw std::invalid_argument("Null shaping totals argument");
        const auto& batch = *static_cast<TrainingBatch*>(handle);
        std::copy(batch.shaping_totals.begin(), batch.shaping_totals.end(), totals);
    });
}
MCSIM_EXPORT const char* mcsim_replay_frame(void* handle, int environment) {
    thread_local std::string frame;
    const auto result = protect([&] {
        if (!handle) { throw std::invalid_argument("Null replay handle"); }
        const auto& batch = *static_cast<TrainingBatch*>(handle);
        if (environment < 0 || environment >= static_cast<int>(batch.worlds.size())) {
            throw std::invalid_argument("Replay environment index outside batch");
        }
        frame = minecraft_sim::app::replayFrame(batch.worlds[environment].tick,
            batch.players.data() + environment * 2, batch.held.data() + environment * 2, true,
            batch.selected_slots.data() + environment * 2).dump();
    });
    return result == 0 ? frame.c_str() : nullptr;
}
MCSIM_EXPORT int mcsim_reset(void* handle, const std::uint8_t* mask, float* observations) {
    return protect([&] {
        if (!handle || !observations) { throw std::invalid_argument("Null reset argument"); }
        static_cast<TrainingBatch*>(handle)->reset(mask);
        static_cast<TrainingBatch*>(handle)->observe(observations);
    });
}
MCSIM_EXPORT int mcsim_reset_players(void* handle, const std::uint8_t* mask,
                                   const char* json_utf8, float* observations) {
    return protect([&] {
        if (!handle || !json_utf8 || !observations) throw std::invalid_argument("Null player reset argument");
        auto& batch = *static_cast<TrainingBatch*>(handle);
        batch.resetPlayers(mask, nlohmann::json::parse(json_utf8));
        batch.observe(observations);
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
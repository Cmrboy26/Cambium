#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* mcsim_last_error(void);
int mcsim_api_version(void);
int mcsim_cuda_available(void);
const char* mcsim_batch_info(void* handle);
const char* mcsim_scenario(void* handle);
const char* mcsim_replay_frame(void* handle, int world);
/* ABI 5 requires ten base reward weights in the build-info reward_inputs order. */
void* mcsim_create_schema(int worlds, int use_cuda, const char* path, int max_ticks,
                         uint64_t seed, const float* weights, int schema_version);
void* mcsim_create_contract(int worlds, int use_cuda, const char* path, int max_ticks,
                           uint64_t seed, const float* weights, const char* contract);
void mcsim_destroy(void* handle);
int mcsim_configure_bridge(void* handle, const char* json_utf8);
const char* mcsim_bridge_state(void* handle);
const char* mcsim_bridge_kits(void* handle, int player);
/* Host actor state [worlds, 2, 9]: position xyz, yaw, pitch, remaining
   golden-apple use ticks, active mining fraction, use and mining delays. Ordinary duel mode
   implements no mining, and therefore reports zero for its mining fraction. */
int mcsim_actor_task_state(void* handle, double* output, size_t count);
/* Last tick's Bridge score, opponent score, toward and away contributions,
   in world/player order. Outcome rewards are not included. */
int mcsim_bridge_reward_terms(void* handle, double* output, size_t count);
/* Host-staged Bridge batches only. Read next-reset counters, or restore them
   and immediately reset every world. Buffers contain exactly worlds entries;
   UINT64_MAX is invalid for restore because the reset increments each value. */
int mcsim_bridge_reset_episodes(void* handle, uint64_t* episodes, size_t worlds);
int mcsim_set_bridge_reset_episodes(void* handle, const uint64_t* episodes, size_t worlds);
int mcsim_observe(void* handle, float* observations);
/* Host diagnostic: version 1 or 2 Bridge match features [worlds, 2, 10 or 17].
   count is the exact number of floats; output is unchanged on error. */
int mcsim_bridge_match_features(void* handle, int version, float* output, size_t count);
/* Host-staged Bridge terrain [worlds, 2, 5, 9, 9], uint8. count is the
   exact number of bytes; output is unchanged on error. Finished worlds retain
   their final snapshot until reset. Caller serializes access to the batch. */
int mcsim_bridge_local_voxels(void* handle, uint8_t* output, size_t count);
/* Blocking diagnostic: raw 22-feature rows [worlds, 2, 22] in caller-owned
   CUDA device memory on the batch device. The caller must allocate count floats.
   stream is a cudaStream_t, or null for the default stream. This synchronizes
   the device before and after the kernel and does not apply the policy layout. */
int mcsim_observe_base_cuda_sync(void* handle, float* observations, size_t count, void* stream);
/* Prepare the current policy layout and per-episode metadata for CUDA observations. */
int mcsim_prepare_cuda_observations(void* handle);
/* Blocking diagnostic: configured float32 policy observations in caller-owned
   CUDA memory [worlds, 2, observation_dim]. The caller owns buffer capacity. */
int mcsim_observe_cuda_sync(void* handle, float* observations, size_t count, void* stream);
/* Register caller-owned CUDA float32 memory of exactly worlds*2*observation_dim
   elements. The caller must retain it until replacement or batch destruction. */
int mcsim_register_cuda_observations(void* handle, float* observations, size_t count);
/* Enqueue configured observations into the registered buffer on stream, which
   must belong to the batch device. The caller orders cross-stream consumers. */
int mcsim_observe_cuda_async(void* handle, void* stream);
/* Wait for the most recent configured observation and report asynchronous errors. */
int mcsim_wait_cuda_observations(void* handle);
/* Register caller-owned int32 CUDA actions [worlds, 2, action_heads]. The
   caller retains the allocation until replacement or batch destruction. */
int mcsim_register_cuda_actions(void* handle, const int32_t* actions, size_t count);
/* Enqueue batch-atomic action validation and decoding. Invalid input sets a
   sticky status and leaves all queued controls unchanged. */
int mcsim_decode_cuda_actions_async(void* handle, void* stream);
/* Wait for queued decoding, then report or explicitly clear the sticky status. */
int mcsim_cuda_action_status(void* handle, int* status);
int mcsim_clear_cuda_action_status(void* handle);
/* Register caller-owned CUDA buffers for fixed-reset resident stepping.
   Shapes: rewards/losses [worlds,2], terminated/truncated [worlds],
   contributions [worlds,2,18]. Activating this mode disables host-state APIs. */
int mcsim_register_cuda_step_outputs(void* handle, float* rewards, uint8_t* terminated,
                                     uint8_t* truncated, uint8_t* losses, double* contributions, size_t worlds);
/* Resident Bridge: caller-owned match [worlds,2,10|17], voxel
   [worlds,2,5,9,9], scored [worlds,2] and task terms [worlds,2,4] buffers.
   Counts are elements, and version selects the match feature layout. */
int mcsim_register_cuda_bridge_buffers(void* handle, float* match, size_t match_count,
                                       uint8_t* voxels, size_t voxel_count, uint8_t* scored, size_t scored_count,
                                       double* task_terms, size_t task_count, int match_version);
int mcsim_register_cuda_reset_mask(void* handle, const uint8_t* mask, size_t worlds);
int mcsim_step_cuda_resident_async(void* handle, void* stream);
int mcsim_reset_cuda_resident_async(void* handle, void* stream);
/* Blocking checkpoint-boundary access to the next per-world reset counters. */
int mcsim_cuda_reset_episodes(void* handle, uint64_t* episodes, size_t worlds);
int mcsim_set_cuda_reset_episodes(void* handle, const uint64_t* episodes, size_t worlds);
int mcsim_reset(void* handle, const uint8_t* mask, float* observations);
int mcsim_step(void* handle, const int32_t* actions, float* observations, float* rewards,
               uint8_t* terminated, uint8_t* truncated);
int mcsim_losses(void* handle, uint8_t* losses);

/* Additive ABI 2 calls. Return 0 on success, -1 on error; obtain the
   thread-local error with mcsim_last_error(). The caller owns all buffers. */
int mcsim_set_reward_weights(void* handle, const float* weights);
int mcsim_set_event_rewards(void* handle, const float* weights);
/* Last successful tick's reward contributions for selected players.
   mask has worlds*2 boolean entries in world/player order; null selects all.
   ABI 5 totals has 18 doubles: the thirteen base types followed by five shaping types
   in the build-info reward_totals order. Reset clears selected worlds' last
   step contributions, without changing the cumulative batch reward totals. */
int mcsim_step_reward_totals(void* handle, const uint8_t* mask, double* totals);
/* Three floats per player in world-major order: positive own wind impulse,
   accepted mace-smash flag, and grounded flag. Read after step, before reset. */
int mcsim_step_behavior_events(void* handle, float* events);
/* Replace the episode timeout in 1..1000000 without rebuilding terrain.
   Call at a stage reset boundary; this does not reset worlds or counters. */
int mcsim_set_max_ticks(void* handle, int max_ticks);
int mcsim_set_profiling(void* handle, int enabled);
const char* mcsim_profile(void* handle);

/* json_utf8 is a world-count array. Selected entries are two concrete Scene
   player definitions; unselected entries must be null. A null mask selects all.
   Every definition/loadout/action slot/spawn is validated before any mutation.
   Legacy mcsim_reset subsequently restores these assigned definitions.
   observations receives the entire batch, including unchanged worlds. */
int mcsim_reset_players(void* handle, const uint8_t* mask, const char* json_utf8,
                       float* observations);

/* Resolved initial scenario for this world's current episode, not live state.
   The returned thread-local string is valid until the next scenario_for call
   on the same thread. Null indicates an invalid handle/index or other error.
   Terrain is immutable and shared; player resets do not rebuild or edit it. */
const char* mcsim_scenario_for(void* handle, int world);

#ifdef __cplusplus
}
#endif

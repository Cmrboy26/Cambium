#pragma once

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
void* mcsim_create_schema(int worlds, int use_cuda, const char* path, int max_ticks,
                         uint64_t seed, const float* weights, int schema_version);
void* mcsim_create_contract(int worlds, int use_cuda, const char* path, int max_ticks,
                           uint64_t seed, const float* weights, const char* contract);
void mcsim_destroy(void* handle);
int mcsim_observe(void* handle, float* observations);
int mcsim_reset(void* handle, const uint8_t* mask, float* observations);
int mcsim_step(void* handle, const int32_t* actions, float* observations, float* rewards,
               uint8_t* terminated, uint8_t* truncated);

/* Additive ABI 2 calls. Return 0 on success, -1 on error; obtain the
   thread-local error with mcsim_last_error(). The caller owns all buffers. */
int mcsim_set_reward_weights(void* handle, const float* weights);

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

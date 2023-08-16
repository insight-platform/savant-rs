#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

bool check_version(const char *external_version);

void pipeline_move_as_is(uintptr_t handle,
                         const char *dest_stage,
                         const int64_t *ids,
                         uintptr_t len);

void pipeline_move_and_pack_frames(uintptr_t handle,
                                   const char *dest_stage,
                                   const int64_t *frame_ids,
                                   uintptr_t len,
                                   int64_t *batch_id);

uintptr_t pipeline_move_and_unpack_batch(uintptr_t handle,
                                         const char *dest_stage,
                                         int64_t batch_id,
                                         int64_t *resulting_ids,
                                         uintptr_t resulting_ids_len);

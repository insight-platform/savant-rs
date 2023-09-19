#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum BoxSource {
  Detection = 0,
  Tracking = 1,
  TrackingWhenAbsentDetection = 2,
} BoxSource;

typedef struct InferenceMeta {
  int64_t id;
  int64_t parent_id;
  int64_t namespace_;
  int64_t label;
  int64_t left;
  int64_t top;
  int64_t width;
  int64_t height;
} InferenceMeta;

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
bool check_version(const char *external_version);

/**
 * # Safety
 *
 * The function is unsafe because it is exported to C-ABI and works with raw pointers.
 *
 */
uintptr_t build_inference_meta(uintptr_t handle,
                               enum BoxSource box_source,
                               struct InferenceMeta *meta,
                               uintptr_t meta_capacity);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
void pipeline2_move_as_is(uintptr_t handle,
                          const char *dest_stage,
                          const int64_t *ids,
                          uintptr_t len);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
int64_t pipeline2_move_and_pack_frames(uintptr_t handle,
                                       const char *dest_stage,
                                       const int64_t *frame_ids,
                                       uintptr_t len);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
uintptr_t pipeline2_move_and_unpack_batch(uintptr_t handle,
                                          const char *dest_stage,
                                          int64_t batch_id,
                                          int64_t *resulting_ids,
                                          uintptr_t resulting_ids_len);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 *
 * Arguments
 * ---------
 * handle: usize
 *   The pipeline handle
 * id: i64
 *   The batch or frame id to apply updates to
 *
 * Returns
 * -------
 * bool
 *   True if the updates were applied, false otherwise
 *
 */
bool pipeline2_apply_updates(uintptr_t handle, int64_t id);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 *
 * Arguments
 * ---------
 * handle: usize
 *   The pipeline handle
 * id: i64
 *   The batch or frame id to clear updates from
 *
 * Returns
 * -------
 * bool
 *   True if the updates were cleared, false otherwise
 *
 */
bool pipeline2_clear_updates(uintptr_t handle, int64_t id);

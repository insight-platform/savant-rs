#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * When BBox is not defined, its elements are set to this value.
 */
typedef struct VideoObjectInferenceMeta {
  int64_t id;
  int64_t namespace_id;
  int64_t label_id;
  float confidence;
  int64_t track_id;
  float xc;
  float yc;
  float width;
  float height;
  float angle;
} VideoObjectInferenceMeta;

/**
 * Updates frame meta from inference meta
 *
 * # Safety
 *
 * This function is unsafe because it transforms raw pointer to VideoFrame
 *
 */
void update_frame_meta(uintptr_t frame_handle,
                       const struct VideoObjectInferenceMeta *ffi_inference_meta,
                       uintptr_t count,
                       VideoObjectBBoxType t);

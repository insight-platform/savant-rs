#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * When BBox is not defined, its elements are set to this value.
 */
#define BBOX_ELEMENT_UNDEFINED 1.7976931348623157e308

#define EPS 0.00001

#define NATIVE_MESSAGE_MARKER_LEN 4

#define VERSION_LEN 4

/**
 * Determines which object bbox is a subject of the operation
 *
 */
typedef enum VideoObjectBBoxType {
  Detection,
  TrackingInfo,
} VideoObjectBBoxType;

typedef struct VideoObjectInferenceMeta {
  int64_t id;
  int64_t creator_id;
  int64_t label_id;
  double confidence;
  int64_t track_id;
  double xc;
  double yc;
  double width;
  double height;
  double angle;
} VideoObjectInferenceMeta;

/**
 * Returns the object vector length
 *
 * # Safety
 *
 * This function is unsafe because it dereferences a raw pointer
 *
 */
uintptr_t object_vector_len(uintptr_t handle);

/**
 * Returns the object data casted to InferenceObjectMeta by index
 *
 * # Safety
 *
 * This function is unsafe because it dereferences a raw pointer
 *
 */
struct VideoObjectInferenceMeta get_inference_meta(uintptr_t handle,
                                                   uintptr_t pos,
                                                   enum VideoObjectBBoxType t);

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
                       enum VideoObjectBBoxType t);

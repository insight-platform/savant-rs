#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct Arc_Vec_BorrowedVideoObject Arc_Vec_BorrowedVideoObject;

typedef struct BorrowedVideoObject BorrowedVideoObject;

typedef struct VideoObjectsView {
  struct Arc_Vec_BorrowedVideoObject inner;
} VideoObjectsView;

typedef struct CAPI_BoundingBox {
  float xc;
  float yc;
  float width;
  float height;
  float angle;
  bool oriented;
} CAPI_BoundingBox;

typedef struct CAPI_ObjectCreateSpecification {
  const char *namespace_;
  const char *label;
  int64_t parent_id;
  bool parent_id_defined;
  struct CAPI_BoundingBox bounding_box;
  int64_t resulting_object_id;
} CAPI_ObjectCreateSpecification;

typedef struct CAPI_ObjectIds {
  int64_t id;
  int64_t namespace_id;
  int64_t label_id;
  int64_t tracking_id;
  bool namespace_id_set;
  bool label_id_set;
  bool tracking_id_set;
} CAPI_ObjectIds;

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
bool check_version(const char *external_version);

struct VideoObjectsView *savant_get_object_view_from_frame_handle(uintptr_t handle);

struct VideoObjectsView *savant_get_object_view_from_object_view_handle(uintptr_t handle);

void savant_release_object_view(struct VideoObjectsView *view);

struct BorrowedVideoObject *savant_get_view_object(const struct VideoObjectsView *view,
                                                   int64_t object_id);

void savant_release_object(struct BorrowedVideoObject *object);

void savant_create_objects(uintptr_t _frame_handle,
                           struct CAPI_ObjectCreateSpecification *_objects,
                           uintptr_t _len);

struct CAPI_ObjectIds savant_get_object_ids(const struct BorrowedVideoObject *object);

bool savant_get_object_confidence(const struct BorrowedVideoObject *object, float *conf);

void savant_set_object_confidence(struct BorrowedVideoObject *object, float conf);

void savant_clear_object_confidence(struct BorrowedVideoObject *object);

uintptr_t savant_get_object_namespace(const struct BorrowedVideoObject *object,
                                      char *caller_allocated_buf,
                                      uintptr_t len);

uintptr_t savant_get_object_label(const struct BorrowedVideoObject *object,
                                  char *caller_allocated_buf,
                                  uintptr_t len);

uintptr_t savant_get_object_draw_label(const struct BorrowedVideoObject *object,
                                       char *caller_allocated_buf,
                                       uintptr_t len);

void savant_get_object_detection_box(const struct BorrowedVideoObject *object,
                                     struct CAPI_BoundingBox *caller_allocated_bb);

void savant_set_object_detection_box(struct BorrowedVideoObject *object,
                                     const struct CAPI_BoundingBox *bb);

bool savant_get_object_tracking_info(const struct BorrowedVideoObject *object,
                                     struct CAPI_BoundingBox *caller_allocated_bb,
                                     int64_t *caller_allocated_tracking_id);

void savant_set_object_tracking_info(struct BorrowedVideoObject *object,
                                     const struct CAPI_BoundingBox *bb,
                                     int64_t tracking_id);

void savant_clear_object_tracking_info(struct BorrowedVideoObject *object);

bool savant_get_object_float_vec_attribute_value(const struct BorrowedVideoObject *object,
                                                 const char *namespace_,
                                                 const char *name,
                                                 uintptr_t value_index,
                                                 double *caller_allocated_result,
                                                 uintptr_t *caller_allocated_result_len,
                                                 float *caller_allocated_confidence,
                                                 bool *caller_allocated_confidence_set);

void savant_set_object_float_vec_attribute_value(struct BorrowedVideoObject *object,
                                                 const char *namespace_,
                                                 const char *name,
                                                 const char *hint,
                                                 const double *values,
                                                 uintptr_t values_len,
                                                 const float *confidence,
                                                 bool persistent,
                                                 bool hidden);

bool savant_get_object_int_vec_attribute_value(const struct BorrowedVideoObject *object,
                                               const char *namespace_,
                                               const char *name,
                                               uintptr_t value_index,
                                               int64_t *caller_allocated_result,
                                               uintptr_t *caller_allocated_result_len,
                                               float *caller_allocated_confidence,
                                               bool *caller_allocated_confidence_set);

void savant_set_object_int_vec_attribute_value(struct BorrowedVideoObject *object,
                                               const char *namespace_,
                                               const char *name,
                                               const char *hint,
                                               const int64_t *values,
                                               uintptr_t values_len,
                                               const float *confidence,
                                               bool persistent,
                                               bool hidden);

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

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct Arc_Vec_BorrowedVideoObject Arc_Vec_BorrowedVideoObject;

typedef struct BorrowedVideoObject BorrowedVideoObject;

typedef struct VideoFrame VideoFrame;

typedef struct VideoObjectsView {
  struct Arc_Vec_BorrowedVideoObject _0;
} VideoObjectsView;

typedef struct CAPIBoundingBox {
  float xc;
  float yc;
  float width;
  float height;
  float angle;
  bool oriented;
} CAPIBoundingBox;

typedef struct CAPIObjectCreateSpecification {
  const char *namespace_;
  const char *label;
  float confidence;
  bool confidence_defined;
  int64_t parent_id;
  bool parent_id_defined;
  struct CAPIBoundingBox detection_box_box;
  int64_t tracking_id;
  struct CAPIBoundingBox tracking_box;
  bool tracking_id_defined;
  int64_t resulting_object_id;
} CAPIObjectCreateSpecification;

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

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
struct VideoFrame *savant_frame_from_handle(uintptr_t handle);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
void savant_release_frame(struct VideoFrame *frame);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
struct VideoObjectsView *savant_frame_get_all_objects(const struct VideoFrame *frame);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
struct VideoObjectsView *savant_object_view_from_handle(uintptr_t handle);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
void savant_release_object_view(struct VideoObjectsView *view);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
struct BorrowedVideoObject *savant_frame_get_object(const struct VideoFrame *frame,
                                                    int64_t object_id);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
void savant_frame_delete_objects_with_ids(struct VideoFrame *frame,
                                          const int64_t *object_ids,
                                          uintptr_t len);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
struct BorrowedVideoObject *savant_object_view_get_object(const struct VideoObjectsView *view,
                                                          int64_t object_id);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
void savant_release_object(struct BorrowedVideoObject *object);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
void savant_create_objects(struct VideoFrame *frame,
                           struct CAPIObjectCreateSpecification *objects,
                           uintptr_t len);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
struct BorrowedVideoObject *savant_get_borrowed_object_from_handle(uintptr_t handle);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
void savant_release_borrowed_object(struct BorrowedVideoObject *object);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
struct CAPI_ObjectIds savant_object_get_ids(const struct BorrowedVideoObject *object);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
bool savant_object_get_confidence(const struct BorrowedVideoObject *object, float *conf);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
void savant_object_set_confidence(struct BorrowedVideoObject *object, float conf);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
void savant_object_clear_confidence(struct BorrowedVideoObject *object);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
uintptr_t savant_object_get_namespace(const struct BorrowedVideoObject *object,
                                      char *caller_allocated_buf,
                                      uintptr_t len);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
uintptr_t savant_object_get_label(const struct BorrowedVideoObject *object,
                                  char *caller_allocated_buf,
                                  uintptr_t len);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
uintptr_t savant_object_get_draw_label(const struct BorrowedVideoObject *object,
                                       char *caller_allocated_buf,
                                       uintptr_t len);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
void savant_object_get_detection_box(const struct BorrowedVideoObject *object,
                                     struct CAPIBoundingBox *caller_allocated_bb);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
void savant_object_set_detection_box(struct BorrowedVideoObject *object,
                                     const struct CAPIBoundingBox *bb);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
bool savant_object_get_tracking_info(const struct BorrowedVideoObject *object,
                                     struct CAPIBoundingBox *caller_allocated_bb,
                                     int64_t *caller_allocated_tracking_id);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
void savant_object_set_tracking_info(struct BorrowedVideoObject *object,
                                     const struct CAPIBoundingBox *bb,
                                     int64_t tracking_id);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
void savant_object_clear_tracking_info(struct BorrowedVideoObject *object);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
bool savant_object_get_float_vec_attribute_value(const struct BorrowedVideoObject *object,
                                                 const char *namespace_,
                                                 const char *name,
                                                 uintptr_t value_index,
                                                 double *caller_allocated_result,
                                                 uintptr_t *caller_allocated_result_len,
                                                 float *caller_allocated_confidence,
                                                 bool *caller_allocated_confidence_set);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
void savant_object_set_float_vec_attribute_value(struct BorrowedVideoObject *object,
                                                 const char *namespace_,
                                                 const char *name,
                                                 const char *hint,
                                                 const double *values,
                                                 uintptr_t values_len,
                                                 const float *confidence,
                                                 bool persistent,
                                                 bool hidden);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
bool savant_object_get_int_vec_attribute_value(const struct BorrowedVideoObject *object,
                                               const char *namespace_,
                                               const char *name,
                                               uintptr_t value_index,
                                               int64_t *caller_allocated_result,
                                               uintptr_t *caller_allocated_result_len,
                                               float *caller_allocated_confidence,
                                               bool *caller_allocated_confidence_set);

/**
 * # Safety
 *
 * The function is intended for invocation from C/C++, so it is unsafe by design.
 */
void savant_object_set_int_vec_attribute_value(struct BorrowedVideoObject *object,
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

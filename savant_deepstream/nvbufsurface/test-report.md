# Test Report: deepstream_nvbufsurface

**Date:** 2026-03-15
**Total:** 155 | **Passed:** 155 | **Failed:** 0 | **Timeout:** 0

## Summary

All 155 tests in the `deepstream_nvbufsurface` crate passed successfully. This includes 27 unit tests across 4 modules (`cuda_stream`, `shared_buffer`, `surface_view`, `transform`), 115 integration tests across 7 test binaries (`batched`, `bridge_meta`, `generator`, `heterogeneous`, `slot_view`, `surface_view_gpu`, `transform`), and 13 doc-tests. Tests were run individually with `--exact` matching using `--features testing`. Total execution time: ~171 seconds.

## Failures

None.

## Passing Tests

<details>
<summary>All passing tests (155)</summary>

### Unit tests (27)

- `cuda_stream::tests::clone_is_non_owning`
- `cuda_stream::tests::debug_format`
- `cuda_stream::tests::default_is_null_and_non_owning`
- `cuda_stream::tests::from_raw_is_non_owning`
- `shared_buffer::tests::test_clone_is_cheap`
- `shared_buffer::tests::test_debug_format`
- `shared_buffer::tests::test_from_buffer_and_into_buffer`
- `shared_buffer::tests::test_into_buffer_fails_with_clones`
- `shared_buffer::tests::test_into_buffer_succeeds_after_clone_dropped`
- `shared_buffer::tests::test_lock_read_write`
- `surface_view::tests::test_buffer_accessible`
- `surface_view::tests::test_color_format_channels`
- `surface_view::tests::test_cuda_stream_default`
- `surface_view::tests::test_from_cuda_ptr_gray8`
- `surface_view::tests::test_from_cuda_ptr_null_rejected`
- `surface_view::tests::test_from_cuda_ptr_rgba`
- `surface_view::tests::test_from_cuda_ptr_with_keepalive`
- `surface_view::tests::test_into_buffer_fails_with_sibling`
- `surface_view::tests::test_into_buffer_sole_owner`
- `surface_view::tests::test_shared_buffer_clone`
- `surface_view::tests::test_surface_view_is_send`
- `surface_view::tests::test_with_cuda_stream`
- `surface_view::tests::test_wrap_plain_buffer`
- `transform::tests::compute_letterbox_rect_none_with_dst_padding`
- `transform::tests::compute_letterbox_rect_right_bottom_with_dst_padding`
- `transform::tests::compute_letterbox_rect_symmetric_no_dst_padding`
- `transform::tests::compute_letterbox_rect_symmetric_with_dst_padding`

### Integration: batched (22)

- `test_acquire_batched_surface`
- `test_auto_propagate_id_from_source`
- `test_batched_generator_batch_size_1`
- `test_batched_generator_builder`
- `test_create_batched_generator`
- `test_explicit_ids_in_order`
- `test_fill_all_slots`
- `test_fill_different_source_resolutions`
- `test_fill_exceeds_batch_size`
- `test_fill_nv12_to_rgba`
- `test_fill_partial_batch`
- `test_fill_reuse_after_finalize`
- `test_fill_single_slot`
- `test_fill_with_no_roi`
- `test_fill_with_src_roi`
- `test_finalize_empty_batch`
- `test_mixed_explicit_and_none_ids`
- `test_no_ids_at_all`
- `test_set_num_filled_overflow`
- `test_set_num_filled_standalone`
- `test_slot_ptr_out_of_bounds`
- `test_slot_ptrs_are_distinct`

### Integration: bridge_meta (3)

- `test_bridge_meta_nvjpegenc`
- `test_bridge_meta_nvv4l2h264enc`
- `test_bridge_meta_nvv4l2h265enc`

### Integration: generator (18)

- `test_acquire_multiple_surfaces`
- `test_acquire_surface`
- `test_acquire_surface_with_id`
- `test_acquire_surface_with_ptr_and_id`
- `test_acquire_surface_without_id_has_no_meta`
- `test_builder_defaults`
- `test_builder_full`
- `test_create_generator`
- `test_create_surface`
- `test_create_surface_raw`
- `test_create_surface_with_id`
- `test_cuda_device_mem_type`
- `test_different_resolutions`
- `test_from_caps`
- `test_mem_type_conversions`
- `test_null_pointer_error`
- `test_nvmm_caps`
- `test_raw_caps`

### Integration: heterogeneous (12)

- `test_heterogeneous_add_different_formats`
- `test_heterogeneous_add_different_sizes`
- `test_heterogeneous_add_exceeds_capacity`
- `test_heterogeneous_add_partial`
- `test_heterogeneous_auto_propagate_ids`
- `test_heterogeneous_batch_create`
- `test_heterogeneous_batch_size_1`
- `test_heterogeneous_finalize_empty`
- `test_heterogeneous_ids_preserved`
- `test_heterogeneous_parent_buffer_meta`
- `test_heterogeneous_slot_ptr_returns_correct_dims`
- `test_heterogeneous_source_buffer_not_leaked`

### Integration: slot_view (31)

- `test_heterogeneous_add_after_finalize_fails`
- `test_heterogeneous_as_gst_buffer_before_finalize_fails`
- `test_heterogeneous_as_gst_buffer_no_leak`
- `test_heterogeneous_buffer_valid_after_cow`
- `test_heterogeneous_buffer_valid_after_struct_drop`
- `test_heterogeneous_extract_preserves_dimensions`
- `test_heterogeneous_extract_slot_after_finalize_works`
- `test_heterogeneous_extract_slot_before_finalize_fails`
- `test_heterogeneous_id_propagated_per_slot`
- `test_heterogeneous_slot_out_of_bounds`
- `test_heterogeneous_slot_view_no_leak`
- `test_heterogeneous_timestamps_propagated`
- `test_no_id_meta_on_batch`
- `test_uniform_as_gst_buffer_before_finalize_fails`
- `test_uniform_as_gst_buffer_cow_no_pool_leak`
- `test_uniform_as_gst_buffer_no_pool_leak`
- `test_uniform_buffer_id_meta_survives_struct_drop`
- `test_uniform_buffer_valid_after_cow`
- `test_uniform_buffer_valid_after_struct_drop`
- `test_uniform_extract_all_slots_have_distinct_ptrs`
- `test_uniform_extract_first_slot`
- `test_uniform_extract_last_slot`
- `test_uniform_extract_slot_after_finalize_works`
- `test_uniform_extract_slot_before_finalize_fails`
- `test_uniform_fill_after_finalize_fails`
- `test_uniform_id_propagated_per_slot`
- `test_uniform_slot_out_of_bounds`
- `test_uniform_slot_view_from_detached_buffer`
- `test_uniform_slot_view_no_pool_leak`
- `test_uniform_timestamps_propagated`
- `test_view_survives_batch_drop`

### Integration: surface_view_gpu (14)

- `map_unmap_cycle::test_implicit_map_is_permanent`
- `test_data_ptr_is_cuda_addressable`
- `test_into_buffer_roundtrip`
- `test_pitch_matches_surface`
- `test_recycled_buffer_keeps_mapping`
- `test_shared_buffer_into_buffer_fails_with_siblings`
- `test_uniform_batch_slot_cuda_readback`
- `test_uniform_batch_slot_out_of_bounds`
- `test_uniform_batch_slot_views_distinct`
- `test_write_read_roundtrip`
- `tracking::test_meta_balanced_across_recycles`
- `tracking::test_meta_deregistration_on_pool_destroy`
- `tracking::test_multi_slot_meta_lifecycle`
- `tracking::test_registration_persists_across_views`

### Integration: transform (15)

- `transform_all_interpolation_methods`
- `transform_gpu_compute_mode`
- `transform_nv12_to_rgba`
- `transform_preserves_savant_id_meta`
- `transform_rejects_dst_padding_too_tall`
- `transform_rejects_dst_padding_too_wide`
- `transform_rgba_downscale_right_bottom`
- `transform_rgba_downscale_symmetric`
- `transform_rgba_no_padding`
- `transform_rgba_to_nv12`
- `transform_rgba_to_rgba_same_size`
- `transform_upscale_symmetric`
- `transform_with_dst_padding`
- `transform_with_ptr_returns_valid_data`
- `transform_with_src_crop`

### Doc-tests (13)

- `buffers::batched::non_uniform::DsNvNonUniformSurfaceBuffer (line 20)`
- `buffers::batched::slot_view::extract_slot_view (line 34)`
- `buffers::batched::uniform::DsNvUniformSurfaceBufferGenerator (line 30)`
- `buffers::batched::uniform::DsNvUniformSurfaceBufferGeneratorBuilder (line 68)`
- `buffers::single::DsNvSurfaceBufferGenerator (line 30)`
- `buffers::single::DsNvSurfaceBufferGenerator::nvmm_caps (line 370)`
- `buffers::single::DsNvSurfaceBufferGeneratorBuilder (line 59)`
- `lib.rs (line 15)`
- `bridge_savant_id_meta (line 297)`
- `shared_buffer::SharedMutableGstBuffer (line 22)`
- `surface_view::SurfaceView (line 51)`
- `surface_view::SurfaceView::from_shared (line 223)`
- `surface_view::SurfaceView::with_cuda_stream (line 481)`

</details>

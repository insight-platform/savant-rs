//! Internal `GstMeta` marker that flags an NvTracker service batch.
//!
//! A *service batch* is a one-frame batch the crate submits internally during
//! [`reset_stream_with_reason`](crate::pipeline::NvTracker::reset_stream_with_reason)
//! to drive the DeepStream `nvtracker` element through the per-source release
//! sequence (synthetic frame for the source → `GST_NVEVENT_PAD_DELETED` →
//! tracker drops the prev-frame pin on the next regular batch).
//!
//! The marker carries the `source_pad` (crc32(source_id)) so future rescue
//! paths can correlate output buffers back to the source they were submitted
//! against, but its sole runtime role today is binary: if the meta is present
//! on a buffer that exits `nvtracker`, the output drain discards the buffer
//! instead of surfacing it as a [`TrackerOutput`](crate::output::TrackerOutput).
//!
//! Why a dedicated meta and not "absence of [`SavantIdMeta`](deepstream_buffers::SavantIdMeta)"?
//! [`SavantIdMeta`] is optional on user batches (callers may legitimately
//! submit without ids), so its absence is not a reliable service marker.  An
//! explicit meta is positive and unambiguous and survives any future change
//! to the `SavantIdMeta` policy.
//!
//! The meta is registered crate-locally and is **not** part of the public
//! API.  External code must not rely on its existence.

use gstreamer::{glib, MetaAPI, MetaAPIExt};
use std::{fmt, mem};

/// Service-batch marker for [`NvTracker`](crate::pipeline::NvTracker).
///
/// See module-level docs.  Only constructed via [`Self::add`].
#[repr(transparent)]
pub(crate) struct NvTrackerServiceMeta(imp::NvTrackerServiceMeta);

impl NvTrackerServiceMeta {
    /// `crc32(source_id)` of the source this service batch targets.
    #[allow(dead_code)] // exposed for parity / debugging; runtime reads only the marker bit.
    pub(crate) fn source_pad(&self) -> u32 {
        self.0.source_pad
    }

    /// Attach a fresh service meta to `buffer`.
    pub(crate) fn add(
        buffer: &mut gstreamer::BufferRef,
        source_pad: u32,
    ) -> gstreamer::MetaRefMut<'_, Self, gstreamer::meta::Standalone> {
        unsafe {
            // gst_buffer_add_meta() takes ownership of the contents of
            // `params`; defuse Drop until the init callback has copied
            // the fields out.
            let mut params = mem::ManuallyDrop::new(imp::NvTrackerServiceMetaParams { source_pad });

            let meta = gstreamer::ffi::gst_buffer_add_meta(
                buffer.as_mut_ptr(),
                imp::nvtracker_service_meta_get_info(),
                &mut *params as *mut imp::NvTrackerServiceMetaParams as glib::ffi::gpointer,
            ) as *mut imp::NvTrackerServiceMeta;

            Self::from_mut_ptr(buffer, meta)
        }
    }

    /// `true` iff `buffer` carries an [`NvTrackerServiceMeta`].
    pub(crate) fn is_present(buffer: &gstreamer::BufferRef) -> bool {
        buffer.meta::<NvTrackerServiceMeta>().is_some()
    }
}

unsafe impl Send for NvTrackerServiceMeta {}
unsafe impl Sync for NvTrackerServiceMeta {}

unsafe impl MetaAPI for NvTrackerServiceMeta {
    type GstType = imp::NvTrackerServiceMeta;

    fn meta_api() -> glib::Type {
        imp::nvtracker_service_meta_api_get_type()
    }
}

impl fmt::Debug for NvTrackerServiceMeta {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("NvTrackerServiceMeta")
            .field("source_pad", &self.0.source_pad)
            .finish()
    }
}

mod imp {
    use gstreamer::glib;
    use gstreamer::glib::translate::{from_glib, IntoGlib};
    use std::mem::size_of;
    use std::ptr;
    use std::sync::LazyLock;

    pub struct NvTrackerServiceMetaParams {
        pub source_pad: u32,
    }

    #[repr(C)]
    pub struct NvTrackerServiceMeta {
        parent: gstreamer::ffi::GstMeta,
        pub(super) source_pad: u32,
    }

    pub(super) fn nvtracker_service_meta_api_get_type() -> glib::Type {
        // Same multi-shared-library guard as `SavantIdMeta`: the GType
        // registry is process-global, so we must not re-register if
        // another statically-linked copy already did.
        static TYPE: LazyLock<glib::Type> = LazyLock::new(|| unsafe {
            if let Some(existing) = glib::Type::from_name("GstNvTrackerServiceMetaAPI") {
                return existing;
            }

            let t = from_glib(gstreamer::ffi::gst_meta_api_type_register(
                c"GstNvTrackerServiceMetaAPI".as_ptr() as *const _,
                [ptr::null::<std::os::raw::c_char>()].as_ptr() as *mut *const _,
            ));

            assert_ne!(t, glib::Type::INVALID);

            t
        });

        *TYPE
    }

    unsafe extern "C" fn nvtracker_service_meta_init(
        meta: *mut gstreamer::ffi::GstMeta,
        params: glib::ffi::gpointer,
        _buffer: *mut gstreamer::ffi::GstBuffer,
    ) -> glib::ffi::gboolean {
        assert!(!params.is_null());

        let meta = &mut *(meta as *mut NvTrackerServiceMeta);
        let params = ptr::read(params as *const NvTrackerServiceMetaParams);

        ptr::write(&mut meta.source_pad, params.source_pad);

        true.into_glib()
    }

    unsafe extern "C" fn nvtracker_service_meta_free(
        _meta: *mut gstreamer::ffi::GstMeta,
        _buffer: *mut gstreamer::ffi::GstBuffer,
    ) {
        // `source_pad: u32` is `Copy`; nothing to drop.
    }

    unsafe extern "C" fn nvtracker_service_meta_transform(
        dest: *mut gstreamer::ffi::GstBuffer,
        meta: *mut gstreamer::ffi::GstMeta,
        _buffer: *mut gstreamer::ffi::GstBuffer,
        _type_: glib::ffi::GQuark,
        _data: glib::ffi::gpointer,
    ) -> glib::ffi::gboolean {
        let meta = &*(meta as *mut NvTrackerServiceMeta);

        super::NvTrackerServiceMeta::add(gstreamer::BufferRef::from_mut_ptr(dest), meta.source_pad);

        true.into_glib()
    }

    pub(super) unsafe fn nvtracker_service_meta_get_info() -> *const gstreamer::ffi::GstMetaInfo {
        struct MetaInfo(ptr::NonNull<gstreamer::ffi::GstMetaInfo>);
        unsafe impl Send for MetaInfo {}
        unsafe impl Sync for MetaInfo {}

        static META_INFO: LazyLock<MetaInfo> = LazyLock::new(|| unsafe {
            let impl_name = c"GstNvTrackerServiceMeta";

            let existing = gstreamer::ffi::gst_meta_get_info(impl_name.as_ptr() as *const _);
            if !existing.is_null() {
                return MetaInfo(
                    ptr::NonNull::new(existing as *mut gstreamer::ffi::GstMetaInfo)
                        .expect("gst_meta_get_info returned non-null but cast failed"),
                );
            }

            MetaInfo(
                ptr::NonNull::new(gstreamer::ffi::gst_meta_register(
                    nvtracker_service_meta_api_get_type().into_glib(),
                    impl_name.as_ptr() as *const _,
                    size_of::<NvTrackerServiceMeta>(),
                    Some(nvtracker_service_meta_init),
                    Some(nvtracker_service_meta_free),
                    Some(nvtracker_service_meta_transform),
                ) as *mut gstreamer::ffi::GstMetaInfo)
                .expect("Failed to register NvTrackerServiceMeta"),
            )
        });

        META_INFO.0.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gstreamer as gst;

    fn ensure_gst() {
        let _ = gst::init();
    }

    #[test]
    fn add_marks_buffer_and_round_trips_source_pad() {
        ensure_gst();
        let mut buffer = gst::Buffer::new();
        {
            let buf_ref = buffer.make_mut();
            NvTrackerServiceMeta::add(buf_ref, 0xdead_beef);
        }
        assert!(NvTrackerServiceMeta::is_present(buffer.as_ref()));
        let meta = buffer.meta::<NvTrackerServiceMeta>().expect("meta present");
        assert_eq!(meta.source_pad(), 0xdead_beef);
    }

    #[test]
    fn fresh_buffer_does_not_have_service_meta() {
        ensure_gst();
        let buffer = gst::Buffer::new();
        assert!(!NvTrackerServiceMeta::is_present(buffer.as_ref()));
    }
}

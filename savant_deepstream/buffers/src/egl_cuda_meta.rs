//! Custom `GstMeta` for EGL-CUDA interop on Jetson (aarch64).
//!
//! Attaches per-buffer, per-slot `CUgraphicsResource` + CUDA device pointers
//! so that VIC-managed NvBufSurface memory is directly CUDA-addressable.
//! The meta is created lazily via [`ensure_meta`] and cleaned up automatically
//! by the GStreamer meta `free` callback when the buffer is finalized.
//!
//! Supports batched NvBufSurface buffers (up to [`MAX_BATCH_SLOTS`] slots).
//! Each slot is registered independently on first access, and deregistered
//! in `meta_free` when the buffer is destroyed.

use crate::{ffi, transform, NvBufSurfaceError};
use gstreamer::{glib, MetaAPI};
use log::debug;
use std::sync::atomic::{AtomicUsize, Ordering};

const MAX_PLANES: usize = 3;

/// Maximum number of batch slots supported by [`EglCudaMeta`].
///
/// **Not a hardware or NvBufSurface API limit.** In NVIDIA's `nvbufsurface.h`, the
/// `batchSize` field on [`NvBufSurface`](crate::ffi::NvBufSurface) is a `uint32_t` with no
/// documented maximum. DeepStream uses other symbolic caps elsewhere (e.g.
/// `NVDSINFER_MAX_BATCH_SIZE` in the inference headers). This constant only bounds the
/// fixed-size `slots` array embedded in our `GstMeta` so `gst_meta_register` gets a
/// stable layout without heap-allocating the per-slot registration table.
///
/// Raising the value increases per-buffer `GstMeta` size (each slot holds
/// `CUgraphicsResource` plus plane pointers/pitches). The default is chosen to cover
/// typical batched pipelines (including nvinfer-style batch sizes) while keeping meta
/// overhead predictable.
pub const MAX_BATCH_SLOTS: usize = 64;

// ─── Registration tracking (diagnostics / tests) ─────────────────────────────

static REGISTRATIONS: AtomicUsize = AtomicUsize::new(0);
static DEREGISTRATIONS: AtomicUsize = AtomicUsize::new(0);

/// Return `(registrations, deregistrations)` since last [`reset_tracking`].
///
/// Counts are per individual slot registration/deregistration.
pub fn tracking_counts() -> (usize, usize) {
    (
        REGISTRATIONS.load(Ordering::SeqCst),
        DEREGISTRATIONS.load(Ordering::SeqCst),
    )
}

/// Reset both counters to zero.
pub fn reset_tracking() {
    REGISTRATIONS.store(0, Ordering::SeqCst);
    DEREGISTRATIONS.store(0, Ordering::SeqCst);
}

// ─── Public wrapper ──────────────────────────────────────────────────────────

/// Lightweight value type holding the CUDA pointers extracted from one slot
/// of an [`EglCudaMeta`] on the buffer.  Avoids lifetime issues with
/// `MetaRef` temporaries.
#[derive(Debug, Clone)]
pub struct EglCudaMapping {
    pub cuda_ptrs: [*mut std::ffi::c_void; MAX_PLANES],
    pub pitches: [u32; MAX_PLANES],
    pub plane_count: u32,
}

impl EglCudaMapping {
    /// CUDA device pointer for the given plane.
    pub fn cuda_ptr(&self, plane: usize) -> *mut std::ffi::c_void {
        self.cuda_ptrs[plane]
    }

    /// Row pitch in bytes for the given plane.
    pub fn pitch(&self, plane: usize) -> u32 {
        self.pitches[plane]
    }
}

// SAFETY: The pointers reference GPU memory owned by the GstBuffer.
unsafe impl Send for EglCudaMapping {}
unsafe impl Sync for EglCudaMapping {}

/// Custom GstMeta storing EGL-CUDA interop resources for an NvBufSurface.
///
/// Supports multiple batch slots — each slot's registration is stored in a
/// fixed-size array indexed by slot number.
#[repr(transparent)]
pub struct EglCudaMeta(imp::EglCudaMetaInner);

// SAFETY: `EglCudaMeta` is stored inline as `GstMeta` on a refcounted `GstBuffer`. `Send` is
// required because GStreamer may pass buffers between threads.
//
// `Sync`: New slots are registered and existing slots are updated only while the caller holds
// `&mut BufferRef` in `ensure_meta` (see SAFETY there for why interior mutation through a
// shared `MetaRef` is still exclusive to this buffer). `read_meta` and the read-only path in
// `ensure_meta` only copy POD fields (`cuda_ptrs`, `pitches`, `plane_count`) and check
// whether `resource` is null — values that are stable after a successful registration until
// `meta_free` runs on buffer teardown. `CUgraphicsResource` lifetime is tied to a CUDA context;
// registration uses `ensure_cuda_egl_context` before driver calls, and `meta_free` unregisters
// on the same buffer-finalization path. This relies on the usual GstBuffer contract: no
// concurrent mutation of the same buffer from multiple threads without external synchronization.
unsafe impl Send for EglCudaMeta {}
unsafe impl Sync for EglCudaMeta {}

unsafe impl MetaAPI for EglCudaMeta {
    type GstType = imp::EglCudaMetaInner;

    fn meta_api() -> glib::Type {
        imp::egl_cuda_meta_api_get_type()
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Ensure the buffer has an [`EglCudaMeta`] with the given `slot_index`
/// registered, and return a snapshot of that slot's CUDA pointers.
///
/// - If the meta exists and the slot is already registered, the cached
///   pointers are returned in O(1).
/// - If the meta exists but the slot is not registered, only that slot is
///   registered (mutated in-place).
/// - If no meta exists, a new one is created with `batch_size` from the
///   NvBufSurface header and the requested slot is registered.
///
/// # Safety
///
/// `buf` must be backed by a valid NvBufSurface with `numFilled > slot_index`.
pub unsafe fn ensure_meta(
    buf: &mut gstreamer::BufferRef,
    slot_index: u32,
) -> Result<EglCudaMapping, NvBufSurfaceError> {
    if slot_index as usize >= MAX_BATCH_SLOTS {
        return Err(NvBufSurfaceError::SlotOutOfBounds {
            index: slot_index,
            max: MAX_BATCH_SLOTS as u32,
        });
    }

    // If meta already exists, check if this slot is registered.
    if let Some(existing) = buf.meta::<EglCudaMeta>() {
        let slot = &existing.0.slots[slot_index as usize];
        if !slot.resource.is_null() {
            return Ok(EglCudaMapping {
                cuda_ptrs: slot.cuda_ptrs,
                pitches: slot.pitches,
                plane_count: slot.plane_count,
            });
        }

        // Meta exists but slot not registered — register in-place.
        let surf_ptr = existing.0.surf_ptr;
        let reg = imp::register_egl_cuda_slot(surf_ptr, slot_index)?;
        let mapping = EglCudaMapping {
            cuda_ptrs: reg.cuda_ptrs,
            pitches: reg.pitches,
            plane_count: reg.plane_count,
        };

        // SAFETY: The caller holds `&mut BufferRef`, which guarantees exclusive
        // access to the underlying GstBuffer.  The `MetaRef` returned by
        // `buf.meta::<T>()` only provides a shared `&EglCudaMeta`, but the
        // actual buffer is exclusively owned — no other code can read or write
        // this meta concurrently.  This is a known gstreamer-rs limitation:
        // `MetaAPI` only exposes shared references even when the buffer is
        // mutably borrowed.  Because we have exclusive buffer access, there
        // are no data races and the interior mutation is sound.
        let meta_ptr = &existing.0 as *const imp::EglCudaMetaInner as *mut imp::EglCudaMetaInner;
        (*meta_ptr).slots[slot_index as usize] = reg;

        return Ok(mapping);
    }

    // No meta — create one.
    let surf_ptr = transform::extract_nvbufsurface(buf)
        .map_err(|e| NvBufSurfaceError::InvalidInput(e.to_string()))?;
    let batch_size = (*surf_ptr).batchSize;

    if batch_size as usize > MAX_BATCH_SLOTS {
        return Err(NvBufSurfaceError::InvalidInput(format!(
            "batchSize {} exceeds MAX_BATCH_SLOTS {}",
            batch_size, MAX_BATCH_SLOTS
        )));
    }

    let reg = imp::register_egl_cuda_slot(surf_ptr, slot_index)?;
    let mapping = EglCudaMapping {
        cuda_ptrs: reg.cuda_ptrs,
        pitches: reg.pitches,
        plane_count: reg.plane_count,
    };

    let params = imp::EglCudaMetaParams {
        surf_ptr,
        batch_size,
        initial_slot_index: slot_index,
        initial_reg: reg,
    };
    imp::attach(buf, params);

    // Mark as POOLED + LOCKED so the meta survives pool recycle.
    if let Some(meta) = buf.meta::<EglCudaMeta>() {
        let raw = &meta.0 as *const imp::EglCudaMetaInner as *mut gstreamer::ffi::GstMeta;
        (*raw).flags |= gstreamer::ffi::GST_META_FLAG_POOLED | gstreamer::ffi::GST_META_FLAG_LOCKED;
    }

    Ok(mapping)
}

/// Read an existing [`EglCudaMeta`] slot from a buffer (no registration).
///
/// Returns `None` if the meta does not exist or the slot has not been
/// registered yet.
pub fn read_meta(buf: &gstreamer::BufferRef, slot_index: u32) -> Option<EglCudaMapping> {
    let meta = buf.meta::<EglCudaMeta>()?;
    let idx = slot_index as usize;
    if idx >= MAX_BATCH_SLOTS {
        return None;
    }
    let slot = &meta.0.slots[idx];
    if slot.resource.is_null() {
        return None;
    }
    Some(EglCudaMapping {
        cuda_ptrs: slot.cuda_ptrs,
        pitches: slot.pitches,
        plane_count: slot.plane_count,
    })
}

// ─── Implementation module ───────────────────────────────────────────────────

mod imp {
    use super::*;
    use gstreamer::glib::translate::{from_glib, IntoGlib};
    use std::mem::size_of;
    use std::ptr;
    use std::sync::LazyLock;

    /// Per-slot EGL-CUDA registration data.
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub(super) struct SlotRegistration {
        pub resource: ffi::CUgraphicsResource,
        pub cuda_ptrs: [*mut std::ffi::c_void; MAX_PLANES],
        pub pitches: [u32; MAX_PLANES],
        pub plane_count: u32,
    }

    impl SlotRegistration {
        pub const EMPTY: Self = Self {
            resource: ptr::null_mut(),
            cuda_ptrs: [ptr::null_mut(); MAX_PLANES],
            pitches: [0; MAX_PLANES],
            plane_count: 0,
        };
    }

    pub(super) struct EglCudaMetaParams {
        pub surf_ptr: *mut ffi::NvBufSurface,
        pub batch_size: u32,
        pub initial_slot_index: u32,
        pub initial_reg: SlotRegistration,
    }

    #[repr(C)]
    pub struct EglCudaMetaInner {
        parent: gstreamer::ffi::GstMeta,
        pub(super) surf_ptr: *mut ffi::NvBufSurface,
        pub(super) batch_size: u32,
        pub(super) slots: [SlotRegistration; MAX_BATCH_SLOTS],
    }

    pub(super) fn egl_cuda_meta_api_get_type() -> glib::Type {
        static TYPE: LazyLock<glib::Type> = LazyLock::new(|| unsafe {
            if let Some(existing) = glib::Type::from_name("GstEglCudaMetaAPI") {
                return existing;
            }
            let t = from_glib(gstreamer::ffi::gst_meta_api_type_register(
                c"GstEglCudaMetaAPI".as_ptr() as *const _,
                [ptr::null::<std::os::raw::c_char>()].as_ptr() as *mut *const _,
            ));
            assert_ne!(t, glib::Type::INVALID);
            t
        });
        *TYPE
    }

    unsafe extern "C" fn meta_init(
        meta: *mut gstreamer::ffi::GstMeta,
        params: glib::ffi::gpointer,
        _buffer: *mut gstreamer::ffi::GstBuffer,
    ) -> glib::ffi::gboolean {
        assert!(!params.is_null());
        let meta = &mut *(meta as *mut EglCudaMetaInner);
        let p = ptr::read(params as *const EglCudaMetaParams);

        meta.surf_ptr = p.surf_ptr;
        meta.batch_size = p.batch_size;
        meta.slots = [SlotRegistration::EMPTY; MAX_BATCH_SLOTS];
        meta.slots[p.initial_slot_index as usize] = p.initial_reg;

        // Tracking is handled by register_egl_cuda_slot, not here.
        true.into_glib()
    }

    unsafe extern "C" fn meta_free(
        meta: *mut gstreamer::ffi::GstMeta,
        _buffer: *mut gstreamer::ffi::GstBuffer,
    ) {
        let meta = &mut *(meta as *mut EglCudaMetaInner);
        let n = (meta.batch_size as usize).min(MAX_BATCH_SLOTS);

        for i in 0..n {
            let slot = &mut meta.slots[i];
            if !slot.resource.is_null() {
                let _ = ffi::cuGraphicsUnregisterResource(slot.resource);
                slot.resource = ptr::null_mut();

                let _ = ffi::NvBufSurfaceUnMapEglImage(meta.surf_ptr, i as i32);

                DEREGISTRATIONS.fetch_add(1, Ordering::SeqCst);
            }
        }
        meta.surf_ptr = ptr::null_mut();
    }

    unsafe extern "C" fn meta_transform(
        _dest: *mut gstreamer::ffi::GstBuffer,
        _meta: *mut gstreamer::ffi::GstMeta,
        _buffer: *mut gstreamer::ffi::GstBuffer,
        _type_: glib::ffi::GQuark,
        _data: glib::ffi::gpointer,
    ) -> glib::ffi::gboolean {
        false.into_glib()
    }

    unsafe fn get_info() -> *const gstreamer::ffi::GstMetaInfo {
        struct MetaInfo(ptr::NonNull<gstreamer::ffi::GstMetaInfo>);
        unsafe impl Send for MetaInfo {}
        unsafe impl Sync for MetaInfo {}

        static META_INFO: LazyLock<MetaInfo> = LazyLock::new(|| unsafe {
            let impl_name = c"GstEglCudaMeta";

            let existing = gstreamer::ffi::gst_meta_get_info(impl_name.as_ptr() as *const _);
            if !existing.is_null() {
                return MetaInfo(
                    ptr::NonNull::new(existing as *mut gstreamer::ffi::GstMetaInfo)
                        .expect("gst_meta_get_info returned non-null but cast failed"),
                );
            }

            MetaInfo(
                ptr::NonNull::new(gstreamer::ffi::gst_meta_register(
                    egl_cuda_meta_api_get_type().into_glib(),
                    impl_name.as_ptr() as *const _,
                    size_of::<EglCudaMetaInner>(),
                    Some(meta_init),
                    Some(meta_free),
                    Some(meta_transform),
                ) as *mut gstreamer::ffi::GstMetaInfo)
                .expect("Failed to register EglCudaMeta"),
            )
        });

        META_INFO.0.as_ptr()
    }

    pub(super) unsafe fn attach(buf: &mut gstreamer::BufferRef, params: EglCudaMetaParams) {
        let mut p = std::mem::ManuallyDrop::new(params);
        gstreamer::ffi::gst_buffer_add_meta(
            buf.as_mut_ptr(),
            get_info(),
            &mut *p as *mut EglCudaMetaParams as glib::ffi::gpointer,
        );
    }

    type CUcontext = *mut std::ffi::c_void;

    extern "C" {
        fn cudaSetDevice(device: i32) -> i32;
        fn eglGetDisplay(display_id: *const std::ffi::c_void) -> *mut std::ffi::c_void;
        fn eglInitialize(display: *mut std::ffi::c_void, major: *mut i32, minor: *mut i32) -> u32;
        fn cuCtxGetCurrent(ctx: *mut CUcontext) -> u32;
        fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, dev: i32) -> u32;
        fn cuCtxSetCurrent(ctx: CUcontext) -> u32;
    }

    /// Ensure the current thread has a CUDA context and EGL is initialized.
    unsafe fn ensure_cuda_egl_context(gpu_id: u32) {
        let rc = cudaSetDevice(gpu_id as i32);
        if rc != 0 {
            log::warn!("cudaSetDevice({}) failed with error code {}", gpu_id, rc);
        }

        let mut ctx: CUcontext = ptr::null_mut();
        let rc = cuCtxGetCurrent(&mut ctx);
        if rc != 0 {
            log::warn!("cuCtxGetCurrent failed with error code {}", rc);
        }
        if ctx.is_null() {
            let rc = cuDevicePrimaryCtxRetain(&mut ctx, gpu_id as i32);
            if rc != 0 {
                log::warn!(
                    "cuDevicePrimaryCtxRetain({}) failed with error code {}",
                    gpu_id,
                    rc,
                );
            }
            if !ctx.is_null() {
                let rc = cuCtxSetCurrent(ctx);
                if rc != 0 {
                    log::warn!("cuCtxSetCurrent failed with error code {}", rc);
                }
            }
        }

        let display = eglGetDisplay(std::ptr::null());
        if !display.is_null() {
            let mut major = 0i32;
            let mut minor = 0i32;
            let rc = eglInitialize(display, &mut major, &mut minor);
            if rc != 1 {
                log::warn!("eglInitialize failed (returned {})", rc);
            }
        } else {
            log::warn!("eglGetDisplay returned null");
        }
    }

    /// Perform the full EGL-CUDA registration chain for a specific slot.
    pub(super) unsafe fn register_egl_cuda_slot(
        surf_ptr: *mut ffi::NvBufSurface,
        slot_index: u32,
    ) -> Result<SlotRegistration, NvBufSurfaceError> {
        let surf_list = (*surf_ptr).surfaceList;
        let slot_params = &*surf_list.add(slot_index as usize);

        ensure_cuda_egl_context((*surf_ptr).gpuId);

        debug!(
            "register_egl_cuda_slot[{}]: bufferDesc={}, memType={}, dataPtr={:?}",
            slot_index,
            slot_params.bufferDesc,
            (*surf_ptr).memType,
            slot_params.dataPtr
        );

        let ret = ffi::NvBufSurfaceMapEglImage(surf_ptr, slot_index as i32);
        debug!("NvBufSurfaceMapEglImage[{}] returned {}", slot_index, ret);
        if ret != 0 {
            return Err(NvBufSurfaceError::SurfaceMapFailed(ret));
        }

        // Re-read after NvBufSurfaceMapEglImage mutated the struct.
        let slot_params = &*surf_list.add(slot_index as usize);
        let egl_image: *mut std::ffi::c_void = std::ptr::read_volatile(
            &slot_params.mappedAddr.eglImage as *const *mut std::ffi::c_void,
        );
        debug!("eglImage[{}] = {:?}", slot_index, egl_image);
        if egl_image.is_null() {
            let _ = ffi::NvBufSurfaceUnMapEglImage(surf_ptr, slot_index as i32);
            return Err(NvBufSurfaceError::NullPointer(
                "mappedAddr.eglImage is null after NvBufSurfaceMapEglImage".into(),
            ));
        }

        let mut resource: ffi::CUgraphicsResource = ptr::null_mut();
        let rc = ffi::cuGraphicsEGLRegisterImage(
            &mut resource,
            egl_image,
            ffi::CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE,
        );
        debug!(
            "cuGraphicsEGLRegisterImage[{}] rc={}, resource={:?}",
            slot_index, rc, resource
        );
        if rc != 0 {
            let _ = ffi::NvBufSurfaceUnMapEglImage(surf_ptr, slot_index as i32);
            return Err(NvBufSurfaceError::CudaDriverError {
                function: "cuGraphicsEGLRegisterImage",
                code: rc,
            });
        }

        let mut egl_frame = std::mem::MaybeUninit::<ffi::CUeglFrame>::zeroed();
        let rc = ffi::cuGraphicsResourceGetMappedEglFrame(egl_frame.as_mut_ptr(), resource, 0, 0);
        debug!(
            "cuGraphicsResourceGetMappedEglFrame[{}] rc={}",
            slot_index, rc
        );
        if rc != 0 {
            let _ = ffi::cuGraphicsUnregisterResource(resource);
            let _ = ffi::NvBufSurfaceUnMapEglImage(surf_ptr, slot_index as i32);
            return Err(NvBufSurfaceError::CudaDriverError {
                function: "cuGraphicsResourceGetMappedEglFrame",
                code: rc,
            });
        }
        let egl_frame = egl_frame.assume_init();

        if egl_frame.frame_type != ffi::CU_EGL_FRAME_TYPE_PITCH {
            let _ = ffi::cuGraphicsUnmapResources(1, &mut resource, ptr::null_mut());
            let _ = ffi::cuGraphicsUnregisterResource(resource);
            let _ = ffi::NvBufSurfaceUnMapEglImage(surf_ptr, slot_index as i32);
            return Err(NvBufSurfaceError::InvalidInput(format!(
                "EGL frame type is {} (expected CU_EGL_FRAME_TYPE_PITCH=1)",
                egl_frame.frame_type,
            )));
        }

        let plane_count = egl_frame.plane_count.min(MAX_PLANES as u32);
        let mut cuda_ptrs = [ptr::null_mut(); MAX_PLANES];
        let mut pitches = [0u32; MAX_PLANES];

        for i in 0..plane_count as usize {
            cuda_ptrs[i] = egl_frame.frame.p_pitch[i];
            pitches[i] = if i == 0 {
                egl_frame.pitch
            } else {
                slot_params.planeParams.pitch[i]
            };
        }

        debug!(
            "EGL-CUDA registered slot[{}] bufferDesc={}: {} plane(s), ptr={:?}, pitch={}",
            slot_index, slot_params.bufferDesc, plane_count, cuda_ptrs[0], pitches[0]
        );

        REGISTRATIONS.fetch_add(1, Ordering::SeqCst);

        Ok(SlotRegistration {
            resource,
            cuda_ptrs,
            pitches,
            plane_count,
        })
    }
}

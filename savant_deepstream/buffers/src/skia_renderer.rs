//! GPU-accelerated Skia renderer with CUDA-GL interop for NvBufSurface.
//!
//! [`SkiaRenderer`] combines:
//! - A headless EGL context ([`EglHeadlessContext`])
//! - An OpenGL texture + FBO as the render target
//! - A Skia GPU `DirectContext` + `Surface` for hardware-accelerated 2D drawing
//! - CUDA-GL interop to copy rendered pixels into NvBufSurface GPU memory
//!
//! # Data flow
//!
//! ## Render to NvBufSurface (Mode 2a / 2b)
//!
//! ```text
//! Skia Canvas ──draw──▸ GL Texture (FBO)
//!                          │
//!                    flush_and_submit
//!                          │
//!               CUDA-GL interop map ──▸ cudaArray
//!                          │
//!            cudaMemcpy2DFromArray ──▸ NvBufSurface.dataPtr (fast path)
//!                                      OR
//!            cudaMemcpy2DFromArray ──▸ temp NvBufSurface ──▸ NvBufSurfTransform ──▸ dst (scaled path)
//! ```
//!
//! ## Load from NvBufSurface (Mode 2b overlay)
//!
//! ```text
//! NvBufSurface.dataPtr ──▸ cudaMemcpy2DToArray ──▸ cudaArray ──▸ GL Texture
//!                                                                    │
//!                                                             Skia draws on top
//! ```
//!
//! On **dGPU** the entire path is GPU-side; no CPU pixel copies occur.
//!
//! On **Jetson** (`NvBufSurfaceMemType::Default` = `SurfaceArray`, VIC-managed)
//! the surface memory is accessed via EGL-CUDA interop (`EglCudaMeta`) to
//! obtain a CUDA device pointer, then uses `DeviceToDevice` copies — the same
//! code path as dGPU.

use crate::cuda_stream::CudaStream;
use crate::egl_context::{EglError, EglHeadlessContext};
use crate::transform::{self, TransformConfig};
use crate::{ffi, BufferGenerator, NvBufSurfaceMemType, VideoFormat};
use gstreamer as gst;
use gstreamer::glib;
use thiserror::Error;

// ─── CUDA-GL Interop FFI ────────────────────────────────────────────────────

#[allow(non_camel_case_types)]
type cudaGraphicsResource_t = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
type cudaArray_t = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
type cudaStream_t = *mut std::ffi::c_void;

extern "C" {
    fn cudaGraphicsGLRegisterImage(
        resource: *mut cudaGraphicsResource_t,
        image: u32,  // GL texture name
        target: u32, // GL_TEXTURE_2D
        flags: u32,
    ) -> i32;

    fn cudaGraphicsMapResources(
        count: i32,
        resources: *mut cudaGraphicsResource_t,
        stream: cudaStream_t,
    ) -> i32;

    fn cudaGraphicsUnmapResources(
        count: i32,
        resources: *mut cudaGraphicsResource_t,
        stream: cudaStream_t,
    ) -> i32;

    fn cudaGraphicsSubResourceGetMappedArray(
        array: *mut cudaArray_t,
        resource: cudaGraphicsResource_t,
        array_index: u32,
        mip_level: u32,
    ) -> i32;

    fn cudaGraphicsUnregisterResource(resource: cudaGraphicsResource_t) -> i32;

    fn cudaMemcpy2DFromArrayAsync(
        dst: *mut std::ffi::c_void,
        dpitch: usize,
        src: cudaArray_t,
        w_offset: usize,
        h_offset: usize,
        width: usize,
        height: usize,
        kind: i32,
        stream: cudaStream_t,
    ) -> i32;

    fn cudaMemcpy2DToArrayAsync(
        dst: cudaArray_t,
        w_offset: usize,
        h_offset: usize,
        src: *const std::ffi::c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: i32,
        stream: cudaStream_t,
    ) -> i32;
}

const GL_TEXTURE_2D: u32 = 0x0DE1;
const GL_RGBA: u32 = 0x1908;
const GL_RGBA8: u32 = 0x8058;
const GL_UNSIGNED_BYTE: u32 = 0x1401;
const GL_FRAMEBUFFER: u32 = 0x8D40;
const GL_COLOR_ATTACHMENT0: u32 = 0x8CE0;
const GL_FRAMEBUFFER_COMPLETE: u32 = 0x8CD5;
const GL_LINEAR: i32 = 0x2601;
const GL_TEXTURE_MIN_FILTER: u32 = 0x2801;
const GL_TEXTURE_MAG_FILTER: u32 = 0x2800;

const CUDA_GRAPHICS_REGISTER_FLAGS_NONE: u32 = 0;

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors from the Skia renderer.
#[derive(Debug, Error)]
pub enum SkiaRendererError {
    #[error("EGL error: {0}")]
    Egl(#[from] EglError),

    #[error("OpenGL error: {0}")]
    Gl(String),

    #[error("CUDA error code {0}: {1}")]
    Cuda(i32, String),

    #[error("Skia error: {0}")]
    Skia(String),

    #[error("NvBufSurface error: {0}")]
    NvBuf(String),
}

// ─── SkiaRenderer ────────────────────────────────────────────────────────────

/// GPU-accelerated Skia renderer that copies rendered frames into NvBufSurface
/// buffers via CUDA-GL interop.
///
/// This type is `!Send` and `!Sync` because the underlying EGL/GL context has
/// thread affinity — all operations must occur on the thread that created the
/// renderer.
///
/// # Drop order
///
/// Fields are declared so that Rust's implicit drop order (declaration order)
/// destroys the Skia surface first, then the Skia GPU context, and finally the
/// EGL context.  Do **not** reorder `surface`, `gr_context`, or `_egl`.
///
/// # Example
///
/// ```rust,no_run
/// use deepstream_buffers::{
///     SkiaRenderer, BufferGenerator, NvBufSurfaceMemType,
///     SurfaceView, VideoFormat, cuda_init,
/// };
/// use skia_safe::Color;
///
/// cuda_init(0).unwrap();
/// let gen = BufferGenerator::new(
///     VideoFormat::RGBA, 1920, 1080, 30, 1, 0, NvBufSurfaceMemType::Default,
/// ).unwrap();
///
/// let mut renderer = SkiaRenderer::new(1920, 1080, 0).unwrap();
/// renderer.canvas().clear(Color::from_argb(255, 30, 40, 60));
///
/// let shared = gen.acquire(None).unwrap();
/// let view = SurfaceView::from_buffer(&shared, 0).unwrap();
/// let mut guard = view.gst_buffer();
/// renderer.render_to_nvbuf(guard.make_mut(), None).unwrap();
/// ```
pub struct SkiaRenderer {
    // Drop order: surface → gr_context → _egl.  Do not reorder.
    surface: skia_safe::Surface,
    gr_context: skia_safe::gpu::DirectContext,
    _egl: EglHeadlessContext,
    gl_texture: u32,
    gl_fbo: u32,
    cuda_resource: cudaGraphicsResource_t,
    width: u32,
    height: u32,
    gpu_id: u32,
    /// Lazily-created temporary generator for the scaled path
    /// (when canvas dimensions != destination dimensions).
    temp_gen: Option<BufferGenerator>,
    cuda_stream: CudaStream,
}

impl SkiaRenderer {
    /// Create a new GPU-accelerated Skia renderer with an empty (transparent)
    /// canvas.
    ///
    /// Sets up a headless EGL context, creates an RGBA8 GL texture and FBO,
    /// registers the texture with CUDA for interop, and creates a Skia
    /// `DirectContext` + `Surface` backed by the FBO.
    ///
    /// # Arguments
    ///
    /// * `width`  — render target width in pixels
    /// * `height` — render target height in pixels
    /// * `gpu_id` — CUDA GPU device ID (usually 0)
    pub fn new(width: u32, height: u32, gpu_id: u32) -> Result<Self, SkiaRendererError> {
        // 1. Create headless EGL context
        let egl = EglHeadlessContext::new()?;

        // 2. Load GL function pointers via the `gl` crate
        gl::load_with(|name| egl.get_proc_address(name));

        // 3. Create GL texture
        let mut texture: u32 = 0;
        unsafe {
            gl::GenTextures(1, &mut texture);
            gl::BindTexture(GL_TEXTURE_2D, texture);
            gl::TexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA8 as i32,
                width as i32,
                height as i32,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                std::ptr::null(),
            );
            gl::TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            gl::TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }
        check_gl_error("texture creation")?;

        // 4. Create FBO and attach texture
        let mut fbo: u32 = 0;
        unsafe {
            gl::GenFramebuffers(1, &mut fbo);
            gl::BindFramebuffer(GL_FRAMEBUFFER, fbo);
            gl::FramebufferTexture2D(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D,
                texture,
                0,
            );
        }
        let status = unsafe { gl::CheckFramebufferStatus(GL_FRAMEBUFFER) };
        if status != GL_FRAMEBUFFER_COMPLETE {
            return Err(SkiaRendererError::Gl(format!(
                "Framebuffer incomplete: 0x{:X}",
                status
            )));
        }
        check_gl_error("FBO creation")?;

        // 5. Register GL texture with CUDA (read+write: we may load INTO the texture)
        let mut cuda_resource: cudaGraphicsResource_t = std::ptr::null_mut();
        let rc = unsafe {
            cudaGraphicsGLRegisterImage(
                &mut cuda_resource,
                texture,
                GL_TEXTURE_2D,
                CUDA_GRAPHICS_REGISTER_FLAGS_NONE,
            )
        };
        if rc != 0 {
            return Err(SkiaRendererError::Cuda(
                rc,
                "cudaGraphicsGLRegisterImage failed".into(),
            ));
        }

        // 6. Create Skia GL interface + DirectContext
        let interface = skia_safe::gpu::gl::Interface::new_load_with(|name| {
            if name == "eglGetCurrentDisplay" {
                return std::ptr::null();
            }
            egl.get_proc_address(name)
        })
        .ok_or_else(|| SkiaRendererError::Skia("Failed to create GL interface".into()))?;

        let mut gr_context = skia_safe::gpu::direct_contexts::make_gl(interface, None)
            .ok_or_else(|| SkiaRendererError::Skia("Failed to create Skia DirectContext".into()))?;

        // 7. Wrap the FBO as a Skia surface
        let fb_info = skia_safe::gpu::gl::FramebufferInfo {
            fboid: fbo,
            format: skia_safe::gpu::gl::Format::RGBA8.into(),
            ..Default::default()
        };

        let backend_rt = skia_safe::gpu::backend_render_targets::make_gl(
            (width as i32, height as i32),
            0, // samples
            8, // stencil bits
            fb_info,
        );

        let surface = skia_safe::gpu::surfaces::wrap_backend_render_target(
            &mut gr_context,
            &backend_rt,
            skia_safe::gpu::SurfaceOrigin::TopLeft,
            skia_safe::ColorType::RGBA8888,
            None,
            None,
        )
        .ok_or_else(|| SkiaRendererError::Skia("Failed to create Skia surface from FBO".into()))?;

        Ok(Self {
            surface,
            gr_context,
            _egl: egl,
            gl_texture: texture,
            gl_fbo: fbo,
            cuda_resource,
            width,
            height,
            gpu_id,
            temp_gen: None,
            cuda_stream: CudaStream::default(),
        })
    }

    /// Construct a SkiaRenderer pre-loaded with content from GPU memory.
    ///
    /// Creates the EGL/GL/CUDA setup at `width x height` and copies the
    /// existing pixels into the GL texture so Skia can draw on top.
    ///
    /// The caller must provide the CUDA device pointer and pitch (row stride
    /// in bytes). On dGPU this is the NvBufSurface `dataPtr`; on Jetson use
    /// a [`SurfaceView`] to resolve the pointer via EGL-CUDA interop.
    ///
    /// [`SurfaceView`]: crate::SurfaceView
    ///
    /// # Arguments
    ///
    /// * `width`    — source buffer width (canvas will be this size)
    /// * `height`   — source buffer height (canvas will be this size)
    /// * `gpu_id`   — CUDA GPU device ID
    /// * `data_ptr` — CUDA device pointer to the first pixel (RGBA)
    /// * `pitch`    — row stride in bytes
    /// # Safety
    ///
    /// `data_ptr` must be a valid CUDA device pointer with at least
    /// `pitch * height` readable bytes.
    pub unsafe fn from_nvbuf(
        width: u32,
        height: u32,
        gpu_id: u32,
        data_ptr: *const std::ffi::c_void,
        pitch: usize,
    ) -> Result<Self, SkiaRendererError> {
        let mut renderer = Self::new(width, height, gpu_id)?;
        renderer.load_from_nvbuf(data_ptr, pitch)?;
        Ok(renderer)
    }

    /// Copy GPU pixels INTO the GL texture (reverse direction) so Skia can
    /// draw on top of existing content.
    ///
    /// Performs a GPU-to-GPU `cudaMemcpy2DToArray` (device-to-device) from
    /// the caller-supplied CUDA pointer into the CUDA-GL interop array.
    ///
    /// # Safety
    ///
    /// `data_ptr` must be a valid CUDA device pointer with at least
    /// `pitch * self.height` readable bytes.
    ///
    /// # Arguments
    ///
    /// * `data_ptr` — CUDA device pointer to the source RGBA pixels
    /// * `pitch`    — row stride in bytes
    pub unsafe fn load_from_nvbuf(
        &mut self,
        data_ptr: *const std::ffi::c_void,
        pitch: usize,
    ) -> Result<(), SkiaRendererError> {
        let rc = unsafe {
            cudaGraphicsMapResources(1, &mut self.cuda_resource, self.cuda_stream.as_raw())
        };
        if rc != 0 {
            return Err(SkiaRendererError::Cuda(
                rc,
                "cudaGraphicsMapResources failed (load)".into(),
            ));
        }

        let mut cuda_array: cudaArray_t = std::ptr::null_mut();
        let rc = unsafe {
            cudaGraphicsSubResourceGetMappedArray(&mut cuda_array, self.cuda_resource, 0, 0)
        };
        if rc != 0 {
            unsafe {
                cudaGraphicsUnmapResources(1, &mut self.cuda_resource, self.cuda_stream.as_raw());
            }
            return Err(SkiaRendererError::Cuda(
                rc,
                "cudaGraphicsSubResourceGetMappedArray failed (load)".into(),
            ));
        }

        let width_bytes = (self.width as usize) * 4;

        let copy_rc = unsafe {
            cudaMemcpy2DToArrayAsync(
                cuda_array,
                0,
                0,
                data_ptr,
                pitch,
                width_bytes,
                self.height as usize,
                ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                self.cuda_stream.as_raw(),
            )
        };

        let unmap_rc = unsafe {
            cudaGraphicsUnmapResources(1, &mut self.cuda_resource, self.cuda_stream.as_raw())
        };

        if copy_rc != 0 {
            return Err(SkiaRendererError::Cuda(
                copy_rc,
                "cudaMemcpy2DToArray failed".into(),
            ));
        }
        if unmap_rc != 0 {
            return Err(SkiaRendererError::Cuda(
                unmap_rc,
                "cudaGraphicsUnmapResources failed (load)".into(),
            ));
        }

        self.gr_context.reset(None);
        Ok(())
    }

    /// Get the Skia canvas for drawing.
    ///
    /// Draw shapes, text, images, gradients, etc. using the standard Skia API.
    /// After drawing, call [`render_to_nvbuf`](Self::render_to_nvbuf) to flush
    /// and copy the result into an NvBufSurface buffer.
    pub fn canvas(&mut self) -> &skia_safe::Canvas {
        self.surface.canvas()
    }

    /// Render target width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Render target height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// OpenGL FBO ID (needed by Python skia-python to create its own Surface).
    pub fn fbo_id(&self) -> u32 {
        self.gl_fbo
    }

    /// Set the CUDA stream used for GPU interop copies (builder).
    ///
    /// When set, `cudaGraphicsMapResources`, `cudaGraphicsUnmapResources`,
    /// and all async CUDA copies use this stream instead of the default.
    pub fn with_cuda_stream(mut self, stream: CudaStream) -> Self {
        self.cuda_stream = stream;
        self
    }

    /// Replace the CUDA stream on an existing renderer.
    ///
    /// Same effect as [`with_cuda_stream`](Self::with_cuda_stream) but
    /// takes `&mut self` so it can be called on a reused renderer without
    /// consuming it.
    pub fn set_cuda_stream(&mut self, stream: CudaStream) {
        self.cuda_stream = stream;
    }

    /// Flush Skia rendering and copy the result into a destination NvBufSurface.
    ///
    /// Convenience wrapper that creates a [`SurfaceView`] internally to
    /// resolve the CUDA device pointer, then delegates to
    /// [`render_to_nvbuf_with_ptr`](Self::render_to_nvbuf_with_ptr).
    ///
    /// [`SurfaceView`]: crate::SurfaceView
    pub fn render_to_nvbuf(
        &mut self,
        dst_buf: &mut gst::BufferRef,
        transform_config: Option<&TransformConfig>,
    ) -> Result<(), SkiaRendererError> {
        // Obtain a temporary owned reference (increments GstBuffer refcount)
        // so we can create a SurfaceView to resolve the CUDA pointer.
        let owned: gst::Buffer = unsafe { glib::translate::from_glib_none(dst_buf.as_mut_ptr()) };
        let view = crate::SurfaceView::from_gst_buffer(owned, 0)
            .map_err(|e| SkiaRendererError::NvBuf(e.to_string()))?;
        let data_ptr = view.data_ptr();
        let pitch = view.pitch() as usize;
        // Keep `view` alive until after the pointer is used: on Jetson,
        // `from_glib_none` bumps the refcount causing a COW copy, so the
        // EGL-CUDA meta lives on the copy held by `view`. Dropping early
        // would invalidate `data_ptr`.
        let result =
            unsafe { self.render_to_nvbuf_with_ptr(dst_buf, data_ptr, pitch, transform_config) };
        drop(view);
        result
    }

    /// Flush Skia rendering and copy the result into a destination NvBufSurface.
    ///
    /// ## Fast path (no scaling)
    ///
    /// When `transform_config` is `None` **and** the canvas dimensions equal
    /// the destination buffer dimensions, a direct CUDA-GL copy into
    /// `dst_ptr` is performed (GPU-to-GPU, no intermediate buffer).
    ///
    /// ## Scaled path (letterboxing)
    ///
    /// When `transform_config` is `Some` **or** dimensions differ:
    /// 1. Copies the GL texture into an internal temporary RGBA NvBufSurface
    ///    at canvas resolution (using a [`SurfaceView`] for pointer resolution)
    /// 2. Uses `NvBufSurfTransform` to scale/letterbox from the temp buffer
    ///    into the destination
    ///
    /// [`SurfaceView`]: crate::SurfaceView
    ///
    /// # Arguments
    ///
    /// * `dst_buf`  — Mutable reference to the destination GstBuffer (needed
    ///   for the scaled-path transform; ignored on the fast path).
    /// * `dst_ptr`  — CUDA device pointer to the destination's first pixel.
    /// * `dst_pitch`— Row stride in bytes.
    /// * `transform_config` — Optional scaling/padding configuration.
    ///   When `None` and dimensions match, the fast path is used.
    ///
    /// # Safety
    ///
    /// `dst_ptr` must be a valid CUDA device pointer with at least
    /// `dst_pitch * height` writable bytes (where `height` is the
    /// destination buffer's height).
    pub unsafe fn render_to_nvbuf_with_ptr(
        &mut self,
        dst_buf: &mut gst::BufferRef,
        dst_ptr: *mut std::ffi::c_void,
        dst_pitch: usize,
        transform_config: Option<&TransformConfig>,
    ) -> Result<(), SkiaRendererError> {
        let dst_surf = unsafe {
            transform::extract_nvbufsurface(dst_buf)
                .map_err(|e| SkiaRendererError::NvBuf(e.to_string()))?
        };
        let dst_surface = unsafe { &*(*dst_surf).surfaceList };
        let dst_w = dst_surface.width;
        let dst_h = dst_surface.height;

        let needs_scaling =
            transform_config.is_some() || self.width != dst_w || self.height != dst_h;

        if !needs_scaling {
            return self.copy_gl_to_nvbuf(dst_ptr, dst_pitch);
        }

        // Scaled path: GL → temp buffer → NvBufSurfTransform → dst
        if self.temp_gen.is_none() {
            let surface_generator =
                BufferGenerator::builder(VideoFormat::RGBA, self.width, self.height)
                    .gpu_id(self.gpu_id)
                    .mem_type(NvBufSurfaceMemType::Default)
                    .min_buffers(1)
                    .max_buffers(1)
                    .build()
                    .map_err(|e| SkiaRendererError::NvBuf(e.to_string()))?;
            self.temp_gen = Some(surface_generator);
        }
        let temp_gen = self.temp_gen.as_ref().unwrap();

        let temp_shared = temp_gen
            .acquire(None)
            .map_err(|e| SkiaRendererError::NvBuf(e.to_string()))?;

        let temp_view = crate::SurfaceView::from_buffer(&temp_shared, 0)
            .map_err(|e| SkiaRendererError::NvBuf(e.to_string()))?;

        self.copy_gl_to_nvbuf(temp_view.data_ptr(), temp_view.pitch() as usize)?;

        let temp_guard = temp_view.gst_buffer();
        let temp_surf = unsafe {
            transform::extract_nvbufsurface(temp_guard.as_ref())
                .map_err(|e| SkiaRendererError::NvBuf(e.to_string()))?
        };

        let config = transform_config.cloned().unwrap_or_default();
        unsafe {
            transform::do_transform(temp_surf, dst_surf, &config, None)
                .map_err(|e| SkiaRendererError::NvBuf(e.to_string()))?;
        }

        Ok(())
    }

    /// Flush Skia and copy the GL texture into GPU memory at `(data_ptr, pitch)`.
    ///
    /// Performs a `cudaMemcpy2DFromArray` device-to-device copy from the
    /// CUDA-GL interop array into the caller-supplied CUDA device pointer.
    fn copy_gl_to_nvbuf(
        &mut self,
        data_ptr: *mut std::ffi::c_void,
        pitch: usize,
    ) -> Result<(), SkiaRendererError> {
        self.gr_context.flush_and_submit();

        // glFinish() guarantees the GL texture contains current pixels
        // before CUDA reads from it.
        unsafe {
            gl::Finish();
        }

        let rc = unsafe {
            cudaGraphicsMapResources(1, &mut self.cuda_resource, self.cuda_stream.as_raw())
        };
        if rc != 0 {
            return Err(SkiaRendererError::Cuda(
                rc,
                "cudaGraphicsMapResources failed".into(),
            ));
        }

        let mut cuda_array: cudaArray_t = std::ptr::null_mut();
        let rc = unsafe {
            cudaGraphicsSubResourceGetMappedArray(&mut cuda_array, self.cuda_resource, 0, 0)
        };
        if rc != 0 {
            unsafe {
                cudaGraphicsUnmapResources(1, &mut self.cuda_resource, self.cuda_stream.as_raw());
            }
            return Err(SkiaRendererError::Cuda(
                rc,
                "cudaGraphicsSubResourceGetMappedArray failed".into(),
            ));
        }

        let width_bytes = (self.width as usize) * 4;

        let copy_rc = unsafe {
            cudaMemcpy2DFromArrayAsync(
                data_ptr,
                pitch,
                cuda_array,
                0,
                0,
                width_bytes,
                self.height as usize,
                ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                self.cuda_stream.as_raw(),
            )
        };

        let unmap_rc = unsafe {
            cudaGraphicsUnmapResources(1, &mut self.cuda_resource, self.cuda_stream.as_raw())
        };

        if copy_rc != 0 {
            return Err(SkiaRendererError::Cuda(
                copy_rc,
                "cudaMemcpy2DFromArray failed".into(),
            ));
        }
        if unmap_rc != 0 {
            return Err(SkiaRendererError::Cuda(
                unmap_rc,
                "cudaGraphicsUnmapResources failed".into(),
            ));
        }

        Ok(())
    }

    /// Flush Skia and copy the GL texture into GPU memory at `(data_ptr, pitch)`.
    ///
    /// This always does a direct 1:1 GPU-to-GPU copy (no scaling). The caller
    /// must resolve the CUDA pointer beforehand (e.g. via [`SurfaceView`]).
    ///
    /// [`SurfaceView`]: crate::SurfaceView
    ///
    /// # Safety
    ///
    /// `data_ptr` must be a valid CUDA device pointer with at least
    /// `pitch * self.height` writable bytes.
    pub unsafe fn render_to_nvbuf_raw(
        &mut self,
        data_ptr: *mut std::ffi::c_void,
        pitch: u32,
    ) -> Result<(), SkiaRendererError> {
        if data_ptr.is_null() {
            return Err(SkiaRendererError::NvBuf(
                "NvBufSurface dataPtr is null".into(),
            ));
        }
        self.copy_gl_to_nvbuf(data_ptr, pitch as usize)
    }
}

impl Drop for SkiaRenderer {
    fn drop(&mut self) {
        // Unregister CUDA resource before GL objects are destroyed
        if !self.cuda_resource.is_null() {
            unsafe {
                cudaGraphicsUnregisterResource(self.cuda_resource);
            }
        }
        // Explicit GL cleanup; struct field drop order (surface → gr_context → _egl)
        // handles Skia → EGL teardown automatically after this method returns.
        unsafe {
            gl::DeleteFramebuffers(1, &self.gl_fbo);
            gl::DeleteTextures(1, &self.gl_texture);
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn check_gl_error(context: &str) -> Result<(), SkiaRendererError> {
    let err = unsafe { gl::GetError() };
    if err != 0 {
        Err(SkiaRendererError::Gl(format!(
            "GL error 0x{:X} during {}",
            err, context
        )))
    } else {
        Ok(())
    }
}

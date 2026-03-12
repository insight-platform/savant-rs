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
//! the surface memory is not directly CUDA-addressable.  The copies use
//! `NvBufSurfaceMap` / `NvBufSurfaceSyncForCpu` / `NvBufSurfaceSyncForDevice`
//! / `NvBufSurfaceUnMap` with `cudaMemcpyHostToDevice` / `DeviceToHost` via
//! the CPU-mapped pointer.

use crate::egl_context::{EglError, EglHeadlessContext};
use crate::transform::{self, TransformConfig};
use crate::{ffi, DsNvSurfaceBufferGenerator, NvBufSurfaceMemType, VideoFormat};
use gstreamer as gst;
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

    fn cudaMemcpy2DFromArray(
        dst: *mut std::ffi::c_void,
        dpitch: usize,
        src: cudaArray_t,
        w_offset: usize,
        h_offset: usize,
        width: usize,
        height: usize,
        kind: i32, // cudaMemcpyDeviceToDevice = 3
    ) -> i32;

    fn cudaMemcpy2DToArray(
        dst: cudaArray_t,
        w_offset: usize,
        h_offset: usize,
        src: *const std::ffi::c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: i32, // cudaMemcpyDeviceToDevice = 3
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
/// # Example
///
/// ```rust,no_run
/// use deepstream_nvbufsurface::{
///     SkiaRenderer, DsNvSurfaceBufferGenerator, NvBufSurfaceMemType, VideoFormat, cuda_init,
/// };
/// use skia_safe::Color;
///
/// cuda_init(0).unwrap();
/// let gen = DsNvSurfaceBufferGenerator::new(
///     VideoFormat::RGBA, 1920, 1080, 30, 1, 0, NvBufSurfaceMemType::Default,
/// ).unwrap();
///
/// let mut renderer = SkiaRenderer::new(1920, 1080, 0).unwrap();
/// renderer.canvas().clear(Color::from_argb(255, 30, 40, 60));
///
/// let mut buf = gen.acquire_surface(None).unwrap();
/// renderer.render_to_nvbuf(buf.make_mut(), None).unwrap();
/// ```
pub struct SkiaRenderer {
    _egl: EglHeadlessContext,
    gr_context: skia_safe::gpu::DirectContext,
    surface: skia_safe::Surface,
    gl_texture: u32,
    gl_fbo: u32,
    cuda_resource: cudaGraphicsResource_t,
    width: u32,
    height: u32,
    gpu_id: u32,
    /// Lazily-created temporary generator for the scaled path
    /// (when canvas dimensions != destination dimensions).
    temp_gen: Option<DsNvSurfaceBufferGenerator>,
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
            _egl: egl,
            gr_context,
            surface,
            gl_texture: texture,
            gl_fbo: fbo,
            cuda_resource,
            width,
            height,
            gpu_id,
            temp_gen: None,
        })
    }

    /// Construct a SkiaRenderer pre-loaded with content from an NvBufSurface.
    ///
    /// Creates the EGL/GL/CUDA setup at `width x height` (the source buffer's
    /// dimensions) and copies the existing pixels into the GL texture so Skia
    /// can draw on top.
    ///
    /// On Jetson the copy goes through `NvBufSurfaceMap` (CPU staging) because
    /// `SurfaceArray` memory is not directly CUDA-addressable.
    ///
    /// # Arguments
    ///
    /// * `width`  — source buffer width (canvas will be this size)
    /// * `height` — source buffer height (canvas will be this size)
    /// * `gpu_id` — CUDA GPU device ID
    /// * `buf`    — GstBuffer backed by an NvBufSurface (RGBA)
    pub fn from_nvbuf(
        width: u32,
        height: u32,
        gpu_id: u32,
        buf: &gst::BufferRef,
    ) -> Result<Self, SkiaRendererError> {
        let mut renderer = Self::new(width, height, gpu_id)?;
        renderer.load_from_nvbuf(buf)?;
        Ok(renderer)
    }

    /// Copy NvBufSurface pixels INTO the GL texture (reverse direction).
    ///
    /// On dGPU this is a GPU-to-GPU `cudaMemcpy2DToArray`.  On Jetson the
    /// VIC-managed surface is mapped to CPU and copied with
    /// `cudaMemcpyHostToDevice`.
    ///
    /// After this call Skia can draw on top of the loaded content.
    ///
    /// # Arguments
    ///
    /// * `buf` — GstBuffer backed by an NvBufSurface (RGBA)
    pub fn load_from_nvbuf(&mut self, buf: &gst::BufferRef) -> Result<(), SkiaRendererError> {
        let surf_ptr = unsafe {
            transform::extract_nvbufsurface(buf)
                .map_err(|e| SkiaRendererError::NvBuf(e.to_string()))?
        };

        // 1. Map CUDA-GL resource → get cudaArray
        let rc =
            unsafe { cudaGraphicsMapResources(1, &mut self.cuda_resource, std::ptr::null_mut()) };
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
                cudaGraphicsUnmapResources(1, &mut self.cuda_resource, std::ptr::null_mut());
            }
            return Err(SkiaRendererError::Cuda(
                rc,
                "cudaGraphicsSubResourceGetMappedArray failed (load)".into(),
            ));
        }

        let width_bytes = (self.width as usize) * 4; // RGBA = 4 bpp

        #[cfg(not(target_arch = "aarch64"))]
        let copy_rc = {
            let surface = unsafe { &*(*surf_ptr).surfaceList };
            let data_ptr = surface.dataPtr;
            if data_ptr.is_null() {
                unsafe {
                    cudaGraphicsUnmapResources(1, &mut self.cuda_resource, std::ptr::null_mut());
                }
                return Err(SkiaRendererError::NvBuf(
                    "NvBufSurface dataPtr is null".into(),
                ));
            }
            unsafe {
                cudaMemcpy2DToArray(
                    cuda_array,
                    0,
                    0,
                    data_ptr as *const std::ffi::c_void,
                    surface.pitch as usize,
                    width_bytes,
                    self.height as usize,
                    ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                )
            }
        };

        #[cfg(target_arch = "aarch64")]
        let copy_rc = unsafe { self.load_from_nvbuf_mapped(surf_ptr, cuda_array, width_bytes)? };

        // Unmap CUDA-GL resource (always)
        let unmap_rc =
            unsafe { cudaGraphicsUnmapResources(1, &mut self.cuda_resource, std::ptr::null_mut()) };

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

        // Invalidate Skia's cached GL state so it re-reads the texture
        self.gr_context.reset(None);

        Ok(())
    }

    /// Jetson: Map NvBufSurface → SyncForCpu → cudaMemcpy2DToArray (H2D) → UnMap.
    ///
    /// Returns the CUDA return code from `cudaMemcpy2DToArray` (0 = success).
    #[cfg(target_arch = "aarch64")]
    unsafe fn load_from_nvbuf_mapped(
        &self,
        surf_ptr: *mut ffi::NvBufSurface,
        cuda_array: cudaArray_t,
        width_bytes: usize,
    ) -> Result<i32, SkiaRendererError> {
        let ret = ffi::NvBufSurfaceMap(
            surf_ptr,
            0,
            -1,
            ffi::NvBufSurfaceMemMapFlags_NVBUF_MAP_READ_WRITE,
        );
        if ret != 0 {
            return Err(SkiaRendererError::NvBuf(format!(
                "NvBufSurfaceMap failed (code {})",
                ret
            )));
        }

        let ret = ffi::NvBufSurfaceSyncForCpu(surf_ptr, 0, -1);
        if ret != 0 {
            let _ = ffi::NvBufSurfaceUnMap(surf_ptr, 0, -1);
            return Err(SkiaRendererError::NvBuf(format!(
                "NvBufSurfaceSyncForCpu failed (code {})",
                ret
            )));
        }

        let params = &*(*surf_ptr).surfaceList;
        let mapped = params.mappedAddr.addr[0];
        if mapped.is_null() {
            let _ = ffi::NvBufSurfaceUnMap(surf_ptr, 0, -1);
            return Err(SkiaRendererError::NvBuf(
                "mappedAddr.addr[0] is null after NvBufSurfaceMap".into(),
            ));
        }
        let mapped_pitch = params.planeParams.pitch[0] as usize;

        let rc = cudaMemcpy2DToArray(
            cuda_array,
            0,
            0,
            mapped as *const std::ffi::c_void,
            mapped_pitch,
            width_bytes,
            self.height as usize,
            ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
        );

        let _ = ffi::NvBufSurfaceUnMap(surf_ptr, 0, -1);
        Ok(rc)
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

    /// Flush Skia rendering and copy the result into a destination NvBufSurface.
    ///
    /// ## Fast path (no scaling)
    ///
    /// When `transform_config` is `None` **and** the canvas dimensions equal
    /// the destination buffer dimensions, a direct CUDA-GL copy is performed
    /// (GPU-to-GPU, no intermediate buffer).
    ///
    /// ## Scaled path (letterboxing)
    ///
    /// When `transform_config` is `Some` **or** dimensions differ:
    /// 1. Copies the GL texture into an internal temporary RGBA NvBufSurface
    ///    at canvas resolution
    /// 2. Uses `NvBufSurfTransform` to scale/letterbox from the temp buffer
    ///    into the destination
    ///
    /// # Arguments
    ///
    /// * `dst_buf` — Mutable reference to the destination GstBuffer
    ///   (from [`DsNvSurfaceBufferGenerator::acquire_surface`]).
    /// * `transform_config` — Optional scaling/padding configuration.
    ///   When `None` and dimensions match, fast path is used.
    pub fn render_to_nvbuf(
        &mut self,
        dst_buf: &mut gst::BufferRef,
        transform_config: Option<&TransformConfig>,
    ) -> Result<(), SkiaRendererError> {
        // Extract destination NvBufSurface
        let dst_surf = unsafe {
            transform::extract_nvbufsurface(dst_buf)
                .map_err(|e| SkiaRendererError::NvBuf(e.to_string()))?
        };
        let dst_surface = unsafe { &*(*dst_surf).surfaceList };
        let dst_w = dst_surface.width;
        let dst_h = dst_surface.height;

        // Decide: fast path or scaled path
        let needs_scaling =
            transform_config.is_some() || self.width != dst_w || self.height != dst_h;

        if !needs_scaling {
            return self.copy_gl_to_nvbuf(dst_surf);
        }

        // Scaled path: copy GL → temp NvBufSurface → NvBufSurfTransform → dst
        if self.temp_gen.is_none() {
            let surface_generator =
                DsNvSurfaceBufferGenerator::builder(VideoFormat::RGBA, self.width, self.height)
                    .gpu_id(self.gpu_id)
                    .mem_type(NvBufSurfaceMemType::Default)
                    .min_buffers(1)
                    .max_buffers(1)
                    .build()
                    .map_err(|e| SkiaRendererError::NvBuf(e.to_string()))?;
            self.temp_gen = Some(surface_generator);
        }
        let temp_gen = self.temp_gen.as_ref().unwrap();

        let temp_buf = temp_gen
            .acquire_surface(None)
            .map_err(|e| SkiaRendererError::NvBuf(e.to_string()))?;

        let temp_surf = unsafe {
            transform::extract_nvbufsurface(temp_buf.as_ref())
                .map_err(|e| SkiaRendererError::NvBuf(e.to_string()))?
        };

        self.copy_gl_to_nvbuf(temp_surf)?;

        // Perform transform: temp → dst (no crop)
        let config = transform_config.cloned().unwrap_or_default();
        unsafe {
            transform::do_transform(temp_surf, dst_surf, &config, None)
                .map_err(|e| SkiaRendererError::NvBuf(e.to_string()))?;
        }

        Ok(())
    }

    /// Flush Skia and copy the GL texture into an NvBufSurface.
    ///
    /// On dGPU this is a direct GPU-to-GPU `cudaMemcpy2DFromArray`.
    /// On Jetson the VIC-managed surface is mapped to CPU and the copy
    /// uses `cudaMemcpyDeviceToHost` followed by `NvBufSurfaceSyncForDevice`.
    fn copy_gl_to_nvbuf(
        &mut self,
        surf_ptr: *mut ffi::NvBufSurface,
    ) -> Result<(), SkiaRendererError> {
        // 1. Flush Skia to GL
        self.gr_context.flush_and_submit();

        // glFinish() guarantees that the GL texture contains the current
        // frame's pixels before CUDA reads from it. flushAndSubmit() only
        // calls glFlush() which does not block.
        unsafe {
            gl::Finish();
        }

        // 2. Map CUDA-GL resource → get cudaArray
        let rc =
            unsafe { cudaGraphicsMapResources(1, &mut self.cuda_resource, std::ptr::null_mut()) };
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
                cudaGraphicsUnmapResources(1, &mut self.cuda_resource, std::ptr::null_mut());
            }
            return Err(SkiaRendererError::Cuda(
                rc,
                "cudaGraphicsSubResourceGetMappedArray failed".into(),
            ));
        }

        let width_bytes = (self.width as usize) * 4; // RGBA = 4 bpp

        #[cfg(not(target_arch = "aarch64"))]
        let copy_rc = {
            let surface = unsafe { &*(*surf_ptr).surfaceList };
            let data_ptr = surface.dataPtr;
            if data_ptr.is_null() {
                unsafe {
                    cudaGraphicsUnmapResources(1, &mut self.cuda_resource, std::ptr::null_mut());
                }
                return Err(SkiaRendererError::NvBuf(
                    "NvBufSurface dataPtr is null".into(),
                ));
            }
            unsafe {
                cudaMemcpy2DFromArray(
                    data_ptr,
                    surface.pitch as usize,
                    cuda_array,
                    0,
                    0,
                    width_bytes,
                    self.height as usize,
                    ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                )
            }
        };

        #[cfg(target_arch = "aarch64")]
        let copy_rc = unsafe { self.copy_gl_to_nvbuf_mapped(surf_ptr, cuda_array, width_bytes)? };

        // 4. Unmap CUDA-GL resource (always, even on copy error)
        let unmap_rc =
            unsafe { cudaGraphicsUnmapResources(1, &mut self.cuda_resource, std::ptr::null_mut()) };

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

    /// Jetson: Map NvBufSurface → cudaMemcpy2DFromArray (D2H) → SyncForDevice → UnMap.
    ///
    /// Returns the CUDA return code from `cudaMemcpy2DFromArray` (0 = success).
    #[cfg(target_arch = "aarch64")]
    unsafe fn copy_gl_to_nvbuf_mapped(
        &self,
        surf_ptr: *mut ffi::NvBufSurface,
        cuda_array: cudaArray_t,
        width_bytes: usize,
    ) -> Result<i32, SkiaRendererError> {
        let ret = ffi::NvBufSurfaceMap(
            surf_ptr,
            0,
            -1,
            ffi::NvBufSurfaceMemMapFlags_NVBUF_MAP_READ_WRITE,
        );
        if ret != 0 {
            return Err(SkiaRendererError::NvBuf(format!(
                "NvBufSurfaceMap failed (code {})",
                ret
            )));
        }

        let params = &*(*surf_ptr).surfaceList;
        let mapped = params.mappedAddr.addr[0];
        if mapped.is_null() {
            let _ = ffi::NvBufSurfaceUnMap(surf_ptr, 0, -1);
            return Err(SkiaRendererError::NvBuf(
                "mappedAddr.addr[0] is null after NvBufSurfaceMap".into(),
            ));
        }
        let mapped_pitch = params.planeParams.pitch[0] as usize;

        let rc = cudaMemcpy2DFromArray(
            mapped,
            mapped_pitch,
            cuda_array,
            0,
            0,
            width_bytes,
            self.height as usize,
            ffi::CUDA_MEMCPY_DEVICE_TO_HOST,
        );

        let sync_ret = ffi::NvBufSurfaceSyncForDevice(surf_ptr, 0, -1);
        let unmap_ret = ffi::NvBufSurfaceUnMap(surf_ptr, 0, -1);

        if rc == 0 && sync_ret != 0 {
            return Err(SkiaRendererError::NvBuf(format!(
                "NvBufSurfaceSyncForDevice failed (code {})",
                sync_ret
            )));
        }
        if rc == 0 && unmap_ret != 0 {
            return Err(SkiaRendererError::NvBuf(format!(
                "NvBufSurfaceUnMap failed (code {})",
                unmap_ret
            )));
        }

        Ok(rc)
    }

    /// Legacy render_to_nvbuf variant that accepts raw (data_ptr, pitch).
    ///
    /// This always does a direct 1:1 GPU-to-GPU copy (no scaling).
    /// **Not available on Jetson** where `dataPtr` is VIC-managed.
    /// Prefer [`render_to_nvbuf`](Self::render_to_nvbuf).
    #[cfg(not(target_arch = "aarch64"))]
    pub fn render_to_nvbuf_raw(
        &mut self,
        data_ptr: *mut std::ffi::c_void,
        pitch: u32,
    ) -> Result<(), SkiaRendererError> {
        if data_ptr.is_null() {
            return Err(SkiaRendererError::NvBuf(
                "NvBufSurface dataPtr is null".into(),
            ));
        }

        self.gr_context.flush_and_submit();
        unsafe {
            gl::Finish();
        }

        let rc =
            unsafe { cudaGraphicsMapResources(1, &mut self.cuda_resource, std::ptr::null_mut()) };
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
                cudaGraphicsUnmapResources(1, &mut self.cuda_resource, std::ptr::null_mut());
            }
            return Err(SkiaRendererError::Cuda(
                rc,
                "cudaGraphicsSubResourceGetMappedArray failed".into(),
            ));
        }

        let width_bytes = (self.width as usize) * 4;
        let rc = unsafe {
            cudaMemcpy2DFromArray(
                data_ptr,
                pitch as usize,
                cuda_array,
                0,
                0,
                width_bytes,
                self.height as usize,
                ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
            )
        };

        let unmap_rc =
            unsafe { cudaGraphicsUnmapResources(1, &mut self.cuda_resource, std::ptr::null_mut()) };

        if rc != 0 {
            return Err(SkiaRendererError::Cuda(
                rc,
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
}

impl Drop for SkiaRenderer {
    fn drop(&mut self) {
        // Unregister CUDA resource before GL objects are destroyed
        if !self.cuda_resource.is_null() {
            unsafe {
                cudaGraphicsUnregisterResource(self.cuda_resource);
            }
        }
        // Drop Skia surface before the context (implicit via struct field order)
        // Then GL cleanup
        unsafe {
            gl::DeleteFramebuffers(1, &self.gl_fbo);
            gl::DeleteTextures(1, &self.gl_texture);
        }
        // EGL context destroyed via _egl Drop
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

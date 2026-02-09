//! GPU-accelerated Skia renderer with CUDA-GL interop for NvBufSurface.
//!
//! [`SkiaRenderer`] combines:
//! - A headless EGL context ([`EglHeadlessContext`])
//! - An OpenGL texture + FBO as the render target
//! - A Skia GPU `DirectContext` + `Surface` for hardware-accelerated 2D drawing
//! - CUDA-GL interop to copy rendered pixels into NvBufSurface GPU memory
//!
//! # Data flow (per frame)
//!
//! ```text
//! Skia Canvas ──draw──▸ GL Texture (FBO)
//!                          │
//!                    flush_and_submit
//!                          │
//!               CUDA-GL interop map ──▸ cudaArray (implicit GL sync)
//!                          │
//!             cudaMemcpy2DFromArray ──▸ NvBufSurface.dataPtr (GPU linear)
//!                          │
//!                    push to appsrc
//! ```
//!
//! The entire path is GPU-side; no CPU pixel copies occur.

use crate::egl_context::{EglError, EglHeadlessContext};
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
        image: u32, // GL texture name
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

const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;
const CUDA_GRAPHICS_REGISTER_FLAGS_READ_ONLY: u32 = 1;

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
///     SkiaRenderer, NvBufSurfaceGenerator, NvBufSurfaceMemType, cuda_init,
/// };
/// use skia_safe::Color;
///
/// cuda_init(0).unwrap();
/// let generator = NvBufSurfaceGenerator::new(
///     "RGBA", 1920, 1080, 30, 1, 0, NvBufSurfaceMemType::Default,
/// ).unwrap();
///
/// let mut renderer = SkiaRenderer::new(1920, 1080, 0).unwrap();
///
/// // Draw something
/// renderer.canvas().clear(Color::from_argb(255, 30, 40, 60));
///
/// // Acquire buffer and copy rendered pixels into it
/// let (buffer, data_ptr, pitch) = generator.acquire_surface_with_ptr().unwrap();
/// renderer.render_to_nvbuf(data_ptr, pitch).unwrap();
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
}

impl SkiaRenderer {
    /// Create a new GPU-accelerated Skia renderer.
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
    pub fn new(width: u32, height: u32, _gpu_id: u32) -> Result<Self, SkiaRendererError> {
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

        // 5. Register GL texture with CUDA (read-only: we copy FROM the texture)
        let mut cuda_resource: cudaGraphicsResource_t = std::ptr::null_mut();
        let rc = unsafe {
            cudaGraphicsGLRegisterImage(
                &mut cuda_resource,
                texture,
                GL_TEXTURE_2D,
                CUDA_GRAPHICS_REGISTER_FLAGS_READ_ONLY,
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

        let mut gr_context =
            skia_safe::gpu::direct_contexts::make_gl(interface, None).ok_or_else(|| {
                SkiaRendererError::Skia("Failed to create Skia DirectContext".into())
            })?;

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
        .ok_or_else(|| {
            SkiaRendererError::Skia("Failed to create Skia surface from FBO".into())
        })?;

        Ok(Self {
            _egl: egl,
            gr_context,
            surface,
            gl_texture: texture,
            gl_fbo: fbo,
            cuda_resource,
            width,
            height,
        })
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

    /// Flush Skia rendering and copy the result into an NvBufSurface buffer.
    ///
    /// The caller is responsible for acquiring the buffer from a
    /// [`NvBufSurfaceGenerator`](crate::NvBufSurfaceGenerator) (e.g. via
    /// [`acquire_surface_with_ptr`](crate::NvBufSurfaceGenerator::acquire_surface_with_ptr))
    /// and passing the GPU `data_ptr` and row `pitch` obtained from it.
    ///
    /// This method:
    /// 1. Flushes all pending Skia draw commands to the GL texture
    /// 2. Maps the GL texture via CUDA-GL interop (implicit GL sync)
    /// 3. Copies pixels from the GL texture (cudaArray) to the provided
    ///    NvBufSurface GPU memory via `cudaMemcpy2DFromArray` (GPU-to-GPU)
    /// 4. Unmaps the CUDA resource
    ///
    /// No `glFinish()` is used — `cudaGraphicsMapResources` implicitly waits
    /// for all pending GL operations on the registered texture to complete.
    ///
    /// # Arguments
    ///
    /// * `data_ptr` — GPU pointer to the NvBufSurface's pixel memory
    ///   (`NvBufSurfaceParams::dataPtr`).
    /// * `pitch` — Row stride in bytes (`NvBufSurfaceParams::pitch`).
    ///
    /// # Safety
    ///
    /// `data_ptr` must point to a valid CUDA device allocation with at least
    /// `pitch * height` bytes. The caller must ensure the buffer is not
    /// freed while this method is running.
    pub fn render_to_nvbuf(
        &mut self,
        data_ptr: *mut std::ffi::c_void,
        pitch: u32,
    ) -> Result<(), SkiaRendererError> {
        if data_ptr.is_null() {
            return Err(SkiaRendererError::NvBuf(
                "NvBufSurface dataPtr is null".into(),
            ));
        }

        // 1. Flush Skia to GL (no glFinish — CUDA map provides implicit sync)
        self.gr_context.flush_and_submit();

        // 2. Map CUDA resource → get cudaArray
        //    This implicitly synchronizes with pending GL commands on the texture.
        let rc = unsafe {
            cudaGraphicsMapResources(1, &mut self.cuda_resource, std::ptr::null_mut())
        };
        if rc != 0 {
            return Err(SkiaRendererError::Cuda(
                rc,
                "cudaGraphicsMapResources failed".into(),
            ));
        }

        let mut cuda_array: cudaArray_t = std::ptr::null_mut();
        let rc = unsafe {
            cudaGraphicsSubResourceGetMappedArray(
                &mut cuda_array,
                self.cuda_resource,
                0,
                0,
            )
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

        // 3. GPU-to-GPU copy: cudaArray → NvBufSurface linear CUDA memory
        let width_bytes = (self.width as usize) * 4; // RGBA = 4 bytes per pixel
        let rc = unsafe {
            cudaMemcpy2DFromArray(
                data_ptr,
                pitch as usize,
                cuda_array,
                0,
                0,
                width_bytes,
                self.height as usize,
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
            )
        };

        // 4. Unmap CUDA resource (always, even on copy error)
        let unmap_rc = unsafe {
            cudaGraphicsUnmapResources(1, &mut self.cuda_resource, std::ptr::null_mut())
        };

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

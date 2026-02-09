//! Headless EGL context for GPU-accelerated OpenGL rendering without a display.
//!
//! Uses the EGL device platform (`EGL_PLATFORM_DEVICE_EXT`) and surfaceless
//! context (`EGL_KHR_surfaceless_context`) to create a fully functional OpenGL
//! context without X11, Wayland, or any window system.
//!
//! Implemented via raw FFI since the standard Rust EGL crates don't expose
//! the device-platform extensions needed for headless GPU rendering.

use std::ffi::CString;
use thiserror::Error;

// ─── EGL Types and Constants ────────────────────────────────────────────────

#[allow(non_camel_case_types)]
type EGLDisplay = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
type EGLConfig = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
type EGLContext = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
type EGLSurface = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
type EGLDeviceEXT = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
type EGLint = i32;
#[allow(non_camel_case_types)]
type EGLBoolean = u32;

const EGL_NO_DISPLAY: EGLDisplay = std::ptr::null_mut();
const EGL_NO_CONTEXT: EGLContext = std::ptr::null_mut();
const EGL_NO_SURFACE: EGLSurface = std::ptr::null_mut();

const EGL_TRUE: EGLBoolean = 1;
const EGL_NONE: EGLint = 0x3038;
const EGL_OPENGL_API: u32 = 0x30A2;
const EGL_SURFACE_TYPE: EGLint = 0x3033;
const EGL_PBUFFER_BIT: EGLint = 0x0001;
const EGL_RENDERABLE_TYPE: EGLint = 0x3040;
const EGL_OPENGL_BIT: EGLint = 0x0008;
const EGL_RED_SIZE: EGLint = 0x3024;
const EGL_GREEN_SIZE: EGLint = 0x3023;
const EGL_BLUE_SIZE: EGLint = 0x3022;
const EGL_ALPHA_SIZE: EGLint = 0x3021;
const EGL_SUCCESS: EGLint = 0x3000;

// EXT constants not in core EGL
const EGL_PLATFORM_DEVICE_EXT: u32 = 0x313F;

// ─── EGL FFI ────────────────────────────────────────────────────────────────

#[allow(non_snake_case)]
extern "C" {
    fn eglGetError() -> EGLint;
    fn eglInitialize(display: EGLDisplay, major: *mut EGLint, minor: *mut EGLint) -> EGLBoolean;
    fn eglTerminate(display: EGLDisplay) -> EGLBoolean;
    fn eglBindAPI(api: u32) -> EGLBoolean;
    fn eglChooseConfig(
        display: EGLDisplay,
        attrib_list: *const EGLint,
        configs: *mut EGLConfig,
        config_size: EGLint,
        num_config: *mut EGLint,
    ) -> EGLBoolean;
    fn eglCreateContext(
        display: EGLDisplay,
        config: EGLConfig,
        share_context: EGLContext,
        attrib_list: *const EGLint,
    ) -> EGLContext;
    fn eglDestroyContext(display: EGLDisplay, context: EGLContext) -> EGLBoolean;
    fn eglMakeCurrent(
        display: EGLDisplay,
        draw: EGLSurface,
        read: EGLSurface,
        ctx: EGLContext,
    ) -> EGLBoolean;
    fn eglGetProcAddress(procname: *const std::ffi::c_char) -> *const std::ffi::c_void;
}

// EGL extension function pointer types
#[allow(non_camel_case_types)]
type PFNEGLQUERYDEVICESEXTPROC =
    unsafe extern "C" fn(max_devices: EGLint, devices: *mut EGLDeviceEXT, num_devices: *mut EGLint)
        -> EGLBoolean;
#[allow(non_camel_case_types)]
type PFNEGLGETPLATFORMDISPLAYEXTPROC = unsafe extern "C" fn(
    platform: u32,
    native_display: EGLDeviceEXT,
    attrib_list: *const EGLint,
) -> EGLDisplay;

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors that can occur during EGL context creation.
#[derive(Debug, Error)]
pub enum EglError {
    #[error("EGL extension not available: {0}")]
    MissingExtension(String),

    #[error("No EGL devices found")]
    NoDevices,

    #[error("EGL error 0x{0:x}: {1}")]
    Egl(u32, String),
}

fn egl_check(context: &str) -> Result<(), EglError> {
    let err = unsafe { eglGetError() };
    if err != EGL_SUCCESS {
        Err(EglError::Egl(err as u32, context.into()))
    } else {
        Ok(())
    }
}

// ─── EglHeadlessContext ─────────────────────────────────────────────────────

/// A headless EGL OpenGL context backed by the NVIDIA GPU.
///
/// Creates a surfaceless OpenGL context via the EGL device platform,
/// suitable for GPU-accelerated rendering without any display server.
/// The context is made current on the creating thread.
///
/// # Drop
///
/// Destroys the EGL context and terminates the display on drop.
pub struct EglHeadlessContext {
    display: EGLDisplay,
    context: EGLContext,
}

impl EglHeadlessContext {
    /// Create a new headless EGL OpenGL context.
    ///
    /// Queries available EGL devices, picks the first one (typically the
    /// NVIDIA GPU), creates a surfaceless OpenGL context, and makes it
    /// current on the calling thread.
    pub fn new() -> Result<Self, EglError> {
        unsafe {
            // ── Load extension function pointers ─────────────────────────
            let query_devices_name = CString::new("eglQueryDevicesEXT").unwrap();
            let query_devices_ptr = eglGetProcAddress(query_devices_name.as_ptr());
            if query_devices_ptr.is_null() {
                return Err(EglError::MissingExtension(
                    "eglQueryDevicesEXT".into(),
                ));
            }
            let egl_query_devices: PFNEGLQUERYDEVICESEXTPROC =
                std::mem::transmute(query_devices_ptr);

            let get_platform_display_name =
                CString::new("eglGetPlatformDisplayEXT").unwrap();
            let get_platform_display_ptr =
                eglGetProcAddress(get_platform_display_name.as_ptr());
            if get_platform_display_ptr.is_null() {
                return Err(EglError::MissingExtension(
                    "eglGetPlatformDisplayEXT".into(),
                ));
            }
            let egl_get_platform_display: PFNEGLGETPLATFORMDISPLAYEXTPROC =
                std::mem::transmute(get_platform_display_ptr);

            // ── Query EGL devices ────────────────────────────────────────
            let mut devices: [EGLDeviceEXT; 4] = [std::ptr::null_mut(); 4];
            let mut num_devices: EGLint = 0;
            egl_query_devices(4, devices.as_mut_ptr(), &mut num_devices);
            if num_devices == 0 {
                return Err(EglError::NoDevices);
            }

            // ── Get device-backed display (headless) ─────────────────────
            let display =
                egl_get_platform_display(EGL_PLATFORM_DEVICE_EXT, devices[0], std::ptr::null());
            if display == EGL_NO_DISPLAY {
                return Err(EglError::Egl(0, "eglGetPlatformDisplayEXT failed".into()));
            }

            // ── Initialize ───────────────────────────────────────────────
            let mut major: EGLint = 0;
            let mut minor: EGLint = 0;
            if eglInitialize(display, &mut major, &mut minor) != EGL_TRUE {
                egl_check("eglInitialize")?;
            }

            // ── Bind OpenGL API ──────────────────────────────────────────
            if eglBindAPI(EGL_OPENGL_API) != EGL_TRUE {
                egl_check("eglBindAPI(OPENGL_API)")?;
            }

            // ── Choose config ────────────────────────────────────────────
            let config_attribs: [EGLint; 13] = [
                EGL_SURFACE_TYPE,
                EGL_PBUFFER_BIT,
                EGL_RENDERABLE_TYPE,
                EGL_OPENGL_BIT,
                EGL_RED_SIZE,
                8,
                EGL_GREEN_SIZE,
                8,
                EGL_BLUE_SIZE,
                8,
                EGL_ALPHA_SIZE,
                8,
                EGL_NONE,
            ];
            let mut config: EGLConfig = std::ptr::null_mut();
            let mut num_configs: EGLint = 0;
            if eglChooseConfig(
                display,
                config_attribs.as_ptr(),
                &mut config,
                1,
                &mut num_configs,
            ) != EGL_TRUE
                || num_configs == 0
            {
                return Err(EglError::Egl(0, "No suitable EGL config found".into()));
            }

            // ── Create context ───────────────────────────────────────────
            let ctx_attribs: [EGLint; 1] = [EGL_NONE];
            let context =
                eglCreateContext(display, config, EGL_NO_CONTEXT, ctx_attribs.as_ptr());
            if context == EGL_NO_CONTEXT {
                egl_check("eglCreateContext")?;
            }

            // ── Make current (surfaceless) ───────────────────────────────
            if eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context)
                != EGL_TRUE
            {
                egl_check("eglMakeCurrent (surfaceless)")?;
            }

            Ok(Self { display, context })
        }
    }

    /// Get the EGL proc address for a given function name.
    ///
    /// Used by Skia and the `gl` crate to load OpenGL function pointers.
    pub fn get_proc_address(&self, name: &str) -> *const std::ffi::c_void {
        let c_name = CString::new(name).unwrap();
        unsafe { eglGetProcAddress(c_name.as_ptr()) }
    }
}

impl Drop for EglHeadlessContext {
    fn drop(&mut self) {
        unsafe {
            eglMakeCurrent(
                self.display,
                EGL_NO_SURFACE,
                EGL_NO_SURFACE,
                EGL_NO_CONTEXT,
            );
            eglDestroyContext(self.display, self.context);
            eglTerminate(self.display);
        }
    }
}

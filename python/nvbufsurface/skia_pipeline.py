#!/usr/bin/env python3
"""Skia GPU-rendered NVMM encoding pipeline example (Python port).

Demonstrates GPU-accelerated Skia rendering + NvBufSurfaceGenerator:
each frame is drawn with skia-python on an OpenGL texture (GPU), copied
into the NvBufSurface via CUDA-GL interop (GPU-to-GPU, no CPU), and
encoded with NVENC.

The entire pixel data path is GPU-side -- no CPU copies occur.

Data flow (per frame)::

    Skia Canvas --draw--> GL Texture (FBO)
                              |
                        flush + submit
                              |
                   CUDA-GL interop map --> cudaArray (implicit GL sync)
                              |
                 cudaMemcpy2DFromArray --> NvBufSurface.dataPtr (GPU linear)
                              |
                        push to appsrc

Pipeline::

    Skia GPU -> CUDA-GL copy -> NvBufSurface (RGBA NVMM)
        -> appsrc -> nvvideoconvert -> nvv4l2h26Xenc -> h26Xparse -> sink

Usage::

    # 300 frames to MP4
    python skia_pipeline.py --num-frames 300 --output /tmp/skia_demo.mp4

    # Infinite run (Ctrl-C to stop)
    python skia_pipeline.py

    # Custom resolution and codec
    python skia_pipeline.py --width 1280 --height 720 --codec h264

    # JPEG output
    python skia_pipeline.py --codec jpeg --num-frames 100
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass

import skia

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst

from deepstream_nvbufsurface import NvBufSurfaceGenerator, init_cuda, bridge_savant_id_meta_py

# ===========================================================================
# EGL headless context via ctypes
# ===========================================================================

_libEGL = ctypes.CDLL("libEGL.so.1")
_libGL = ctypes.CDLL("libGL.so.1")

# EGL types
EGLDisplay = ctypes.c_void_p
EGLConfig = ctypes.c_void_p
EGLContext = ctypes.c_void_p
EGLSurface = ctypes.c_void_p
EGLDeviceEXT = ctypes.c_void_p
EGLint = ctypes.c_int32
EGLBoolean = ctypes.c_uint32

EGL_NO_DISPLAY = ctypes.c_void_p(None)
EGL_NO_CONTEXT = ctypes.c_void_p(None)
EGL_NO_SURFACE = ctypes.c_void_p(None)

EGL_TRUE = 1
EGL_NONE = 0x3038
EGL_OPENGL_API = 0x30A2
EGL_SURFACE_TYPE = 0x3033
EGL_PBUFFER_BIT = 0x0001
EGL_RENDERABLE_TYPE = 0x3040
EGL_OPENGL_BIT = 0x0008
EGL_RED_SIZE = 0x3024
EGL_GREEN_SIZE = 0x3023
EGL_BLUE_SIZE = 0x3022
EGL_ALPHA_SIZE = 0x3021
EGL_SUCCESS = 0x3000
EGL_PLATFORM_DEVICE_EXT = 0x313F

# EGL functions
_eglGetError = _libEGL.eglGetError
_eglGetError.restype = EGLint

_eglGetProcAddress = _libEGL.eglGetProcAddress
_eglGetProcAddress.restype = ctypes.c_void_p
_eglGetProcAddress.argtypes = [ctypes.c_char_p]

_eglInitialize = _libEGL.eglInitialize
_eglInitialize.restype = EGLBoolean
_eglInitialize.argtypes = [EGLDisplay, ctypes.POINTER(EGLint), ctypes.POINTER(EGLint)]

_eglTerminate = _libEGL.eglTerminate
_eglTerminate.restype = EGLBoolean
_eglTerminate.argtypes = [EGLDisplay]

_eglBindAPI = _libEGL.eglBindAPI
_eglBindAPI.restype = EGLBoolean
_eglBindAPI.argtypes = [ctypes.c_uint32]

_eglChooseConfig = _libEGL.eglChooseConfig
_eglChooseConfig.restype = EGLBoolean
_eglChooseConfig.argtypes = [
    EGLDisplay, ctypes.POINTER(EGLint), ctypes.POINTER(EGLConfig),
    EGLint, ctypes.POINTER(EGLint),
]

_eglCreateContext = _libEGL.eglCreateContext
_eglCreateContext.restype = EGLContext
_eglCreateContext.argtypes = [EGLDisplay, EGLConfig, EGLContext, ctypes.POINTER(EGLint)]

_eglDestroyContext = _libEGL.eglDestroyContext
_eglDestroyContext.restype = EGLBoolean
_eglDestroyContext.argtypes = [EGLDisplay, EGLContext]

_eglMakeCurrent = _libEGL.eglMakeCurrent
_eglMakeCurrent.restype = EGLBoolean
_eglMakeCurrent.argtypes = [EGLDisplay, EGLSurface, EGLSurface, EGLContext]

# EGL extension function pointer types
PFNEGLQUERYDEVICESEXTPROC = ctypes.CFUNCTYPE(
    EGLBoolean, EGLint, ctypes.POINTER(EGLDeviceEXT), ctypes.POINTER(EGLint),
)
PFNEGLGETPLATFORMDISPLAYEXTPROC = ctypes.CFUNCTYPE(
    EGLDisplay, ctypes.c_uint32, EGLDeviceEXT, ctypes.POINTER(EGLint),
)


class EglHeadlessContext:
    """Headless EGL OpenGL context for GPU rendering without a display."""

    def __init__(self):
        # Load extension function pointers
        query_devices_ptr = _eglGetProcAddress(b"eglQueryDevicesEXT")
        if not query_devices_ptr:
            raise RuntimeError("eglQueryDevicesEXT not available")
        egl_query_devices = PFNEGLQUERYDEVICESEXTPROC(query_devices_ptr)

        get_platform_display_ptr = _eglGetProcAddress(b"eglGetPlatformDisplayEXT")
        if not get_platform_display_ptr:
            raise RuntimeError("eglGetPlatformDisplayEXT not available")
        egl_get_platform_display = PFNEGLGETPLATFORMDISPLAYEXTPROC(get_platform_display_ptr)

        # Query EGL devices
        devices = (EGLDeviceEXT * 4)()
        num_devices = EGLint(0)
        egl_query_devices(4, devices, ctypes.byref(num_devices))
        if num_devices.value == 0:
            raise RuntimeError("No EGL devices found")

        # Get device-backed display (headless)
        self.display = egl_get_platform_display(
            EGL_PLATFORM_DEVICE_EXT, devices[0], None,
        )
        if not self.display:
            raise RuntimeError("eglGetPlatformDisplayEXT failed")

        # Initialize
        major, minor = EGLint(0), EGLint(0)
        if _eglInitialize(self.display, ctypes.byref(major), ctypes.byref(minor)) != EGL_TRUE:
            raise RuntimeError(f"eglInitialize failed: 0x{_eglGetError():X}")

        # Bind OpenGL API
        if _eglBindAPI(EGL_OPENGL_API) != EGL_TRUE:
            raise RuntimeError(f"eglBindAPI failed: 0x{_eglGetError():X}")

        # Choose config
        config_attribs = (EGLint * 13)(
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_RED_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_BLUE_SIZE, 8,
            EGL_ALPHA_SIZE, 8,
            EGL_NONE,
        )
        config = EGLConfig()
        num_configs = EGLint(0)
        if _eglChooseConfig(self.display, config_attribs, ctypes.byref(config), 1,
                            ctypes.byref(num_configs)) != EGL_TRUE or num_configs.value == 0:
            raise RuntimeError("No suitable EGL config found")

        # Create context
        ctx_attribs = (EGLint * 1)(EGL_NONE)
        self.context = _eglCreateContext(self.display, config, EGL_NO_CONTEXT, ctx_attribs)
        if not self.context:
            raise RuntimeError(f"eglCreateContext failed: 0x{_eglGetError():X}")

        # Make current (surfaceless)
        if _eglMakeCurrent(self.display, EGL_NO_SURFACE, EGL_NO_SURFACE, self.context) != EGL_TRUE:
            raise RuntimeError(f"eglMakeCurrent failed: 0x{_eglGetError():X}")

    def destroy(self):
        _eglMakeCurrent(self.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)
        _eglDestroyContext(self.display, self.context)
        _eglTerminate(self.display)


# ===========================================================================
# OpenGL helpers via ctypes
# ===========================================================================

GL_TEXTURE_2D = 0x0DE1
GL_RGBA = 0x1908
GL_RGBA8 = 0x8058
GL_UNSIGNED_BYTE = 0x1401
GL_FRAMEBUFFER = 0x8D40
GL_COLOR_ATTACHMENT0 = 0x8CE0
GL_FRAMEBUFFER_COMPLETE = 0x8CD5
GL_LINEAR = 0x2601
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_MAG_FILTER = 0x2800
GL_NO_ERROR = 0

_glGenTextures = _libGL.glGenTextures
_glGenTextures.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
_glBindTexture = _libGL.glBindTexture
_glBindTexture.argtypes = [ctypes.c_uint, ctypes.c_uint]
_glTexImage2D = _libGL.glTexImage2D
_glTexImage2D.argtypes = [
    ctypes.c_uint, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
]
_glTexParameteri = _libGL.glTexParameteri
_glTexParameteri.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_int]
_glGenFramebuffers = _libGL.glGenFramebuffers
_glGenFramebuffers.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
_glBindFramebuffer = _libGL.glBindFramebuffer
_glBindFramebuffer.argtypes = [ctypes.c_uint, ctypes.c_uint]
_glFramebufferTexture2D = _libGL.glFramebufferTexture2D
_glFramebufferTexture2D.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_int]
_glCheckFramebufferStatus = _libGL.glCheckFramebufferStatus
_glCheckFramebufferStatus.restype = ctypes.c_uint
_glCheckFramebufferStatus.argtypes = [ctypes.c_uint]
_glGetError = _libGL.glGetError
_glGetError.restype = ctypes.c_uint
_glDeleteFramebuffers = _libGL.glDeleteFramebuffers
_glDeleteFramebuffers.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
_glDeleteTextures = _libGL.glDeleteTextures
_glDeleteTextures.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]


def _check_gl(context: str):
    err = _glGetError()
    if err != GL_NO_ERROR:
        raise RuntimeError(f"GL error 0x{err:X} during {context}")


# ===========================================================================
# CUDA-GL interop via ctypes
# ===========================================================================

_cudart = ctypes.CDLL("libcudart.so")

# cudaGraphicsGLRegisterImage
_cudaGraphicsGLRegisterImage = _cudart.cudaGraphicsGLRegisterImage
_cudaGraphicsGLRegisterImage.restype = ctypes.c_int
_cudaGraphicsGLRegisterImage.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),  # resource
    ctypes.c_uint,   # image (GL texture)
    ctypes.c_uint,   # target (GL_TEXTURE_2D)
    ctypes.c_uint,   # flags
]

# cudaGraphicsMapResources
_cudaGraphicsMapResources = _cudart.cudaGraphicsMapResources
_cudaGraphicsMapResources.restype = ctypes.c_int
_cudaGraphicsMapResources.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]

# cudaGraphicsUnmapResources
_cudaGraphicsUnmapResources = _cudart.cudaGraphicsUnmapResources
_cudaGraphicsUnmapResources.restype = ctypes.c_int
_cudaGraphicsUnmapResources.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]

# cudaGraphicsSubResourceGetMappedArray
_cudaGraphicsSubResourceGetMappedArray = _cudart.cudaGraphicsSubResourceGetMappedArray
_cudaGraphicsSubResourceGetMappedArray.restype = ctypes.c_int
_cudaGraphicsSubResourceGetMappedArray.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),  # array
    ctypes.c_void_p,  # resource
    ctypes.c_uint,    # arrayIndex
    ctypes.c_uint,    # mipLevel
]

# cudaGraphicsUnregisterResource
_cudaGraphicsUnregisterResource = _cudart.cudaGraphicsUnregisterResource
_cudaGraphicsUnregisterResource.restype = ctypes.c_int
_cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]

# cudaMemcpy2DFromArray
_cudaMemcpy2DFromArray = _cudart.cudaMemcpy2DFromArray
_cudaMemcpy2DFromArray.restype = ctypes.c_int
_cudaMemcpy2DFromArray.argtypes = [
    ctypes.c_void_p,  # dst
    ctypes.c_size_t,  # dpitch
    ctypes.c_void_p,  # src (cudaArray)
    ctypes.c_size_t,  # wOffset
    ctypes.c_size_t,  # hOffset
    ctypes.c_size_t,  # width (bytes)
    ctypes.c_size_t,  # height (rows)
    ctypes.c_int,     # kind
]

CUDA_MEMCPY_DEVICE_TO_DEVICE = 3
CUDA_GRAPHICS_REGISTER_FLAGS_READ_ONLY = 1


# ===========================================================================
# GPU-accelerated Skia renderer with CUDA-GL interop
# ===========================================================================

class SkiaGpuRenderer:
    """GPU-accelerated Skia renderer that copies rendered frames into
    NvBufSurface buffers via CUDA-GL interop.

    The entire pixel path is GPU-side: Skia draws to a GL texture,
    CUDA-GL interop maps it, and cudaMemcpy2DFromArray copies to the
    NvBufSurface linear CUDA memory.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # 1. EGL headless context
        self._egl = EglHeadlessContext()

        # 2. Create GL texture
        self._gl_texture = ctypes.c_uint(0)
        _glGenTextures(1, ctypes.byref(self._gl_texture))
        _glBindTexture(GL_TEXTURE_2D, self._gl_texture.value)
        _glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA8,
            width, height, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, None,
        )
        _glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        _glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        _check_gl("texture creation")

        # 3. Create FBO and attach texture
        self._gl_fbo = ctypes.c_uint(0)
        _glGenFramebuffers(1, ctypes.byref(self._gl_fbo))
        _glBindFramebuffer(GL_FRAMEBUFFER, self._gl_fbo.value)
        _glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D, self._gl_texture.value, 0,
        )
        status = _glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer incomplete: 0x{status:X}")
        _check_gl("FBO creation")

        # 4. Register GL texture with CUDA (read-only)
        self._cuda_resource = ctypes.c_void_p(0)
        rc = _cudaGraphicsGLRegisterImage(
            ctypes.byref(self._cuda_resource),
            self._gl_texture.value,
            GL_TEXTURE_2D,
            CUDA_GRAPHICS_REGISTER_FLAGS_READ_ONLY,
        )
        if rc != 0:
            raise RuntimeError(f"cudaGraphicsGLRegisterImage failed: {rc}")

        # 5. Create Skia GL interface and DirectContext
        gl_interface = skia.GrGLInterface.MakeEGL()
        if gl_interface is None:
            raise RuntimeError("Failed to create Skia GrGLInterface")

        self._gr_context = skia.GrDirectContext.MakeGL(gl_interface)
        if self._gr_context is None:
            raise RuntimeError("Failed to create Skia GrDirectContext")

        # 6. Wrap the FBO as a Skia surface
        fb_info = skia.GrGLFramebufferInfo(self._gl_fbo.value, GL_RGBA8)
        backend_rt = skia.GrBackendRenderTarget(
            width, height,
            0,  # samples
            8,  # stencil bits
            fb_info,
        )
        self._surface = skia.Surface.MakeFromBackendRenderTarget(
            self._gr_context,
            backend_rt,
            skia.kTopLeft_GrSurfaceOrigin,
            skia.kRGBA_8888_ColorType,
            None,  # colorSpace
        )
        if self._surface is None:
            raise RuntimeError("Failed to create GPU-backed Skia Surface")

        # Cache fonts
        bold_tf = skia.Typeface("monospace", skia.FontStyle.Bold())
        normal_tf = skia.Typeface("monospace", skia.FontStyle.Normal())
        self.label_font = skia.Font(bold_tf, 16)
        self.title_font = skia.Font(bold_tf, 18)
        self.legend_font = skia.Font(normal_tf, 13)
        self.footer_font = skia.Font(bold_tf, 14)

        # Cache paints
        self.white_paint = skia.Paint(Color=skia.ColorWHITE, AntiAlias=True)
        self.black_paint = skia.Paint(Color=skia.ColorBLACK, AntiAlias=True)
        self.sidebar_bg_paint = skia.Paint(Color=skia.Color(15, 18, 25, 210))
        self.separator_paint = skia.Paint(
            Color=skia.Color(255, 255, 255, 100),
            StrokeWidth=1.0, Style=skia.Paint.kStroke_Style,
        )
        self.divider_paint = skia.Paint(
            Color=skia.Color(255, 255, 255, 60),
            StrokeWidth=1.0, Style=skia.Paint.kStroke_Style,
        )
        self.footer_bg_paint = skia.Paint(Color=skia.Color(0, 0, 0, 180))
        self.footer_text_paint = skia.Paint(Color=skia.Color(200, 200, 200, 200), AntiAlias=True)
        self.fill_paint = skia.Paint(AntiAlias=True)
        self.stroke_paint = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2.0)
        self.label_bg_paint = skia.Paint()
        self.dot_paint = skia.Paint(AntiAlias=True)
        self.legend_text_paint = skia.Paint(AntiAlias=True, Color=skia.Color(255, 255, 255, 220))
        self.bg_paint = skia.Paint()

        self.boxes: list[BBox] = []

    def canvas(self) -> skia.Canvas:
        return self._surface.getCanvas()

    def render_to_nvbuf(self, data_ptr: int, pitch: int) -> None:
        """Flush Skia and GPU-to-GPU copy to NvBufSurface."""
        # 1. Flush Skia to GL
        self._gr_context.flushAndSubmit()

        # 2. Map CUDA resource (implicit GL sync)
        rc = _cudaGraphicsMapResources(
            1, ctypes.byref(self._cuda_resource), None,
        )
        if rc != 0:
            raise RuntimeError(f"cudaGraphicsMapResources failed: {rc}")

        try:
            # Get cudaArray
            cuda_array = ctypes.c_void_p(0)
            rc = _cudaGraphicsSubResourceGetMappedArray(
                ctypes.byref(cuda_array), self._cuda_resource, 0, 0,
            )
            if rc != 0:
                raise RuntimeError(f"cudaGraphicsSubResourceGetMappedArray failed: {rc}")

            # GPU-to-GPU copy: cudaArray -> NvBufSurface linear memory
            width_bytes = self.width * 4
            rc = _cudaMemcpy2DFromArray(
                ctypes.c_void_p(data_ptr),
                pitch,
                cuda_array,
                0, 0,
                width_bytes,
                self.height,
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
            )
            if rc != 0:
                raise RuntimeError(f"cudaMemcpy2DFromArray failed: {rc}")
        finally:
            # 3. Always unmap
            _cudaGraphicsUnmapResources(
                1, ctypes.byref(self._cuda_resource), None,
            )

    def destroy(self):
        """Clean up GPU resources."""
        if self._cuda_resource:
            _cudaGraphicsUnregisterResource(self._cuda_resource)
            self._cuda_resource = ctypes.c_void_p(0)
        # Skia surface/context are cleaned by Python GC
        self._surface = None
        self._gr_context = None
        # GL cleanup
        _glDeleteFramebuffers(1, ctypes.byref(self._gl_fbo))
        _glDeleteTextures(1, ctypes.byref(self._gl_texture))
        self._egl.destroy()


# ===========================================================================
# Detection class definitions
# ===========================================================================

NUM_BOXES = 20


@dataclass
class DetectionClass:
    name: str
    color: int  # ARGB as a 32-bit int


CLASSES = [
    DetectionClass("person",  0xFFFF5050),
    DetectionClass("car",     0xFF50C8FF),
    DetectionClass("truck",   0xFFFFB428),
    DetectionClass("bicycle", 0xFF50FF78),
    DetectionClass("dog",     0xFFDC64FF),
    DetectionClass("bus",     0xFFFFFF50),
    DetectionClass("bike",    0xFF50FFFF),
    DetectionClass("sign",    0xFFFF8C8C),
]


def _with_alpha(c: int, a: int) -> int:
    return (a << 24) | (c & 0x00FFFFFF)


# ===========================================================================
# Pseudo-random
# ===========================================================================

def pseudo_rand(seed1: int, seed2: int) -> float:
    MASK64 = 0xFFFF_FFFF_FFFF_FFFF
    h = ((seed1 * 6364136223846793005) + seed2) & MASK64
    h ^= (h >> 33)
    h = (h * 0xFF51AFD7ED558CCD) & MASK64
    h ^= (h >> 33)
    return (h & 0x00FF_FFFF) / 0x0100_0000


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float
    class_idx: int
    confidence: float
    id: int


def hsv_to_color(h_deg: float, s: float, v: float) -> int:
    h = ((h_deg % 360.0) + 360.0) % 360.0
    c = v * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = v - c
    sector = int(h / 60.0) % 6
    if sector == 0:   r, g, b = c, x, 0.0
    elif sector == 1: r, g, b = x, c, 0.0
    elif sector == 2: r, g, b = 0.0, c, x
    elif sector == 3: r, g, b = 0.0, x, c
    elif sector == 4: r, g, b = x, 0.0, c
    else:             r, g, b = c, 0.0, x
    return (0xFF << 24) | (int((r+m)*255) << 16) | (int((g+m)*255) << 8) | int((b+m)*255)


# ===========================================================================
# draw_frame (uses the GPU-backed canvas from SkiaGpuRenderer)
# ===========================================================================

def draw_frame(renderer: SkiaGpuRenderer, frame_idx: int, width: float, height: float) -> None:
    canvas = renderer.canvas()

    # -- Background gradient -----------------------------------------------
    hue_shift = (frame_idx * 0.3) % 360.0
    bg1 = hsv_to_color(hue_shift, 0.15, 0.10)
    bg2 = hsv_to_color(hue_shift + 40.0, 0.20, 0.18)
    shader = skia.GradientShader.MakeLinear(
        points=[skia.Point(0, 0), skia.Point(width, height)],
        colors=[bg1, bg2],
    )
    if shader:
        bg_with_shader = skia.Paint(renderer.bg_paint)
        bg_with_shader.setShader(shader)
        canvas.drawRect(skia.Rect.MakeWH(width, height), bg_with_shader)
    else:
        canvas.clear(skia.Color(18, 20, 28))

    # -- Sidebar dimensions ------------------------------------------------
    sidebar_w = min(340.0, width * 0.22)
    scene_w = width - sidebar_w
    t = frame_idx / 60.0

    # -- Generate bounding boxes -------------------------------------------
    renderer.boxes.clear()
    for i in range(NUM_BOXES):
        seed = i
        cx_base = pseudo_rand(seed, 100) * scene_w * 0.7 + scene_w * 0.15
        cy_base = pseudo_rand(seed, 200) * height * 0.7 + height * 0.15
        orbit_rx = pseudo_rand(seed, 300) * scene_w * 0.12 + 20.0
        orbit_ry = pseudo_rand(seed, 400) * height * 0.10 + 15.0
        speed = 0.3 + pseudo_rand(seed, 500) * 0.7
        phase = pseudo_rand(seed, 600) * math.tau

        cx = cx_base + math.cos(t * speed + phase) * orbit_rx
        cy = cy_base + math.sin(t * speed * 0.8 + phase) * orbit_ry

        bw = 50.0 + pseudo_rand(seed, 700) * 140.0
        bh = 40.0 + pseudo_rand(seed, 800) * 160.0
        class_idx = int(pseudo_rand(seed, 900) * len(CLASSES)) % len(CLASSES)
        confidence = 0.55 + pseudo_rand(seed, 1000) * 0.44

        renderer.boxes.append(BBox(
            x=max(0.0, min(cx - bw / 2.0, scene_w - bw)),
            y=max(0.0, min(cy - bh / 2.0, height - bh)),
            w=bw, h=bh,
            class_idx=class_idx,
            confidence=confidence,
            id=i,
        ))

    # -- Draw bounding boxes -----------------------------------------------
    for b in renderer.boxes:
        cls = CLASSES[b.class_idx]
        rect = skia.Rect.MakeXYWH(b.x, b.y, b.w, b.h)

        renderer.fill_paint.setColor(_with_alpha(cls.color, 50))
        canvas.drawRect(rect, renderer.fill_paint)

        renderer.stroke_paint.setColor(cls.color)
        canvas.drawRect(rect, renderer.stroke_paint)

        label_text = f"{cls.name} #{b.id} {b.confidence * 100:.0f}%"
        tw = renderer.label_font.measureText(label_text)
        lh = 22.0
        lx = b.x
        ly = b.y - lh - 2.0 if b.y >= lh + 2.0 else b.y

        renderer.label_bg_paint.setColor(_with_alpha(cls.color, 200))
        canvas.drawRect(skia.Rect.MakeXYWH(lx, ly, tw + 10, lh), renderer.label_bg_paint)

        canvas.drawString(label_text, lx + 5, ly + lh - 5, renderer.label_font, renderer.black_paint)

    # -- Sidebar -----------------------------------------------------------
    sx = scene_w
    canvas.drawRect(skia.Rect.MakeXYWH(sx, 0, sidebar_w, height), renderer.sidebar_bg_paint)
    canvas.drawLine(sx, 0, sx, height, renderer.separator_paint)

    canvas.drawString("DETECTIONS", sx + 12, 28, renderer.title_font, renderer.white_paint)
    canvas.drawLine(sx + 8, 36, sx + sidebar_w - 8, 36, renderer.divider_paint)

    y_off = 52.0
    row_h = 18.0

    for i, b in enumerate(renderer.boxes):
        if y_off + row_h > height - 40.0:
            renderer.legend_text_paint.setColor(skia.Color(255, 255, 255, 180))
            canvas.drawString(
                f"... +{NUM_BOXES - i} more",
                sx + 12, y_off + 14, renderer.legend_font, renderer.legend_text_paint,
            )
            break

        cls = CLASSES[b.class_idx]
        renderer.dot_paint.setColor(cls.color)
        canvas.drawCircle(sx + 16, y_off + 6, 4.0, renderer.dot_paint)

        entry = f"{cls.name:<8} #{b.id:<2} ({int(b.x):>4},{int(b.y):>4}) {b.confidence * 100:>3.0f}%"
        renderer.legend_text_paint.setColor(skia.Color(255, 255, 255, 220))
        canvas.drawString(entry, sx + 26, y_off + 10, renderer.legend_font, renderer.legend_text_paint)

        y_off += row_h

    # -- Footer ------------------------------------------------------------
    canvas.drawRect(skia.Rect.MakeXYWH(sx, height - 32, sidebar_w, 32), renderer.footer_bg_paint)
    footer = f"F:{frame_idx:>6} {int(width)}x{int(height)} {NUM_BOXES}obj"
    canvas.drawString(footer, sx + 10, height - 11, renderer.footer_font, renderer.footer_text_paint)


# ===========================================================================
# Helpers
# ===========================================================================

def rss_kb() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def container_mux_for_ext(ext: str) -> str | None:
    return {
        "mp4": "qtmux", "m4v": "qtmux",
        "mkv": "matroskamux", "webm": "matroskamux",
        "ts": "mpegtsmux",
    }.get(ext)


# ===========================================================================
# Low-level GStreamer helpers
# ===========================================================================

_libgst: ctypes.CDLL | None = None
_libgstapp: ctypes.CDLL | None = None


def _ensure_gst_libs():
    global _libgst, _libgstapp
    if _libgst is None:
        _libgst = ctypes.CDLL("libgstreamer-1.0.so.0")
        _libgstapp = ctypes.CDLL("libgstapp-1.0.so.0")


def _push_buffer_with_ts(appsrc_ptr: int, buf_ptr: int, pts_ns: int, duration_ns: int):
    """Set PTS/duration on a raw GstBuffer and push to AppSrc."""
    _ensure_gst_libs()
    pts_field = ctypes.c_uint64.from_address(buf_ptr + 72)
    pts_field.value = pts_ns
    dur_field = ctypes.c_uint64.from_address(buf_ptr + 88)
    dur_field.value = duration_ns

    _libgstapp.gst_app_src_push_buffer.restype = ctypes.c_int
    _libgstapp.gst_app_src_push_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    ret = _libgstapp.gst_app_src_push_buffer(ctypes.c_void_p(appsrc_ptr), ctypes.c_void_p(buf_ptr))
    if ret != 0:
        raise RuntimeError(f"gst_app_src_push_buffer returned {ret}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Skia GPU-rendered NVMM encoding pipeline")
    parser.add_argument("--width", type=int, default=1920, help="Frame width")
    parser.add_argument("--height", type=int, default=1080, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Framerate numerator")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--pool-size", type=int, default=4, help="Buffer pool size")
    parser.add_argument("--codec", type=str, default="h265",
                        choices=["h264", "h265", "hevc", "jpeg"],
                        help="Video codec")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (.mp4, .mkv, ...)")
    parser.add_argument("--num-frames", "-n", type=int, default=None,
                        help="Number of frames (omit for infinite)")
    args = parser.parse_args()

    # -- Init --------------------------------------------------------------
    Gst.init(None)
    init_cuda(args.gpu_id)

    frame_duration_ns = 1_000_000_000 // args.fps if args.fps > 0 else 33_333_333
    w, h = args.width, args.height

    # -- GPU Skia renderer -------------------------------------------------
    renderer = SkiaGpuRenderer(w, h)
    print(f"SkiaGpuRenderer created: {w}x{h} (gpu {args.gpu_id})")

    # -- Generator (RGBA - Skia's native format) ---------------------------
    gen = NvBufSurfaceGenerator(
        "RGBA", w, h,
        fps_num=args.fps,
        fps_den=1,
        gpu_id=args.gpu_id,
        pool_size=args.pool_size,
    )

    # -- Pipeline ----------------------------------------------------------
    codec = "h265" if args.codec == "hevc" else args.codec
    if codec == "jpeg":
        enc_name = "nvjpegenc"
        parse_name = "jpegparse"
    else:
        enc_name = f"nvv4l2{codec}enc"
        parse_name = f"{codec}parse"

    pipeline = Gst.Pipeline.new("pipeline")
    appsrc = Gst.ElementFactory.make("appsrc", "src")
    convert = Gst.ElementFactory.make("nvvideoconvert", "convert")
    enc = Gst.ElementFactory.make(enc_name, "enc")
    parse = Gst.ElementFactory.make(parse_name, "parse")
    assert appsrc and convert and enc and parse, "Failed to create pipeline elements"

    # Bridge SavantIdMeta across the encoder (PTS-keyed pad probes)
    bridge_savant_id_meta_py(hash(enc))

    container_mux = None
    if args.output:
        sink = Gst.ElementFactory.make("filesink", "sink")
        assert sink, "Failed to create filesink"
        sink.set_property("location", args.output)
        ext = os.path.splitext(args.output)[1].lstrip(".").lower()
        mux_factory = container_mux_for_ext(ext)
        if mux_factory:
            container_mux = Gst.ElementFactory.make(mux_factory, "cmux")
            assert container_mux, f"Failed to create {mux_factory}"
    else:
        sink = Gst.ElementFactory.make("fakesink", "sink")
        assert sink, "Failed to create fakesink"
        sink.set_property("sync", False)

    caps = Gst.Caps.from_string(gen.nvmm_caps_str())
    appsrc.set_property("caps", caps)
    appsrc.set_property("format", Gst.Format.TIME)
    appsrc.set_property("stream-type", 0)

    chain: list[Gst.Element] = [appsrc, convert, enc, parse]
    if container_mux:
        chain.append(container_mux)
    chain.append(sink)

    for elem in chain:
        pipeline.add(elem)
    for idx in range(len(chain) - 1):
        assert chain[idx].link(chain[idx + 1])

    if args.output:
        cmux_label = f"{container_mux.get_factory().get_name()} -> " if container_mux else ""
        sink_label = f"{cmux_label}filesink({args.output})"
    else:
        sink_label = "fakesink"
    print(f"Pipeline: Skia GPU -> appsrc(RGBA) -> nvvideoconvert -> {enc_name} -> {parse_name} -> {sink_label}")

    ret = pipeline.set_state(Gst.State.PLAYING)
    assert ret != Gst.StateChangeReturn.FAILURE, "Failed to start pipeline"

    limit = args.num_frames if args.num_frames is not None else sys.maxsize
    if args.num_frames is not None:
        print(f"Pipeline running ({args.num_frames} frames)...\n")
    else:
        print("Pipeline running (Ctrl-C to stop)...\n")

    # -- Ctrl-C ------------------------------------------------------------
    running = True

    def _sigint(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sigint)

    # -- Stats reporter ----------------------------------------------------
    frame_count = 0
    count_lock = threading.Lock()

    def stats_reporter():
        nonlocal frame_count
        last_count = 0
        last_time = time.monotonic()
        while running:
            time.sleep(1.0)
            now = time.monotonic()
            with count_lock:
                count = frame_count
            elapsed = now - last_time
            delta = count - last_count
            fps = delta / elapsed if elapsed > 0 else 0.0
            rss = rss_kb()
            print(f"frames: {count:>8}  |  fps: {fps:>8.1f}  |  RSS: {rss // 1024} MB")
            last_count = count
            last_time = now

    stats_thread = threading.Thread(target=stats_reporter, daemon=True)
    stats_thread.start()

    # -- Push loop ---------------------------------------------------------
    appsrc_ptr = hash(appsrc)
    bus = pipeline.get_bus()
    i = 0

    while i < limit and running:
        # 1. Draw with Skia (GPU)
        draw_frame(renderer, i, float(w), float(h))

        # 2. Acquire NvBufSurface buffer
        try:
            buf_ptr, data_ptr, pitch = gen.acquire_surface_with_ptr(id=i)
        except Exception as e:
            print(f"acquire_surface_with_ptr failed at frame {i}: {e}", file=sys.stderr)
            break

        # 3. GPU-to-GPU copy (CUDA-GL interop)
        try:
            renderer.render_to_nvbuf(data_ptr, pitch)
        except Exception as e:
            print(f"render_to_nvbuf failed at frame {i}: {e}", file=sys.stderr)
            break

        # 4. Set timestamps and push
        pts_ns = i * frame_duration_ns
        try:
            _push_buffer_with_ts(appsrc_ptr, buf_ptr, pts_ns, frame_duration_ns)
            with count_lock:
                frame_count += 1
            i += 1
        except Exception as e:
            print(f"Push failed at frame {i}: {e}", file=sys.stderr)
            break

        msg = bus.pop_filtered(Gst.MessageType.ERROR)
        if msg:
            err, debug = msg.parse_error()
            print(f"Pipeline error: {err}", file=sys.stderr)
            break

    # -- Shutdown ----------------------------------------------------------
    print("\nStopping...")
    running = False

    NvBufSurfaceGenerator.send_eos(appsrc_ptr)
    bus.timed_pop_filtered(5 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
    pipeline.set_state(Gst.State.NULL)
    stats_thread.join(timeout=2)

    renderer.destroy()

    with count_lock:
        total = frame_count
    print(f"Total frames pushed: {total}")


if __name__ == "__main__":
    main()

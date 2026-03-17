//! `SurfaceView` — zero-copy view of a single GPU surface.

use super::buffer::extract_shared_buffer;
use numpy::PyReadonlyArray3;
use pyo3::prelude::*;

/// Zero-copy view of a single GPU surface.
///
/// Wraps an NvBufSurface-backed buffer or arbitrary CUDA memory with cached
/// surface parameters.  Implements ``__cuda_array_interface__`` for
/// single-plane formats (RGBA, BGRx, GRAY8) so the surface can be consumed
/// by CuPy, PyTorch, and other CUDA-aware libraries.
///
/// Construction:
///
/// - ``SurfaceView.from_buffer(buf, slot_index)`` — from a ``GstBuffer``.
/// - ``SurfaceView.from_cuda_array(obj)`` — from any object exposing
///   ``__cuda_array_interface__`` (CuPy array, PyTorch CUDA tensor, etc.).
#[pyclass(name = "SurfaceView", module = "savant_rs.deepstream")]
pub struct PySurfaceView {
    inner: Option<deepstream_nvbufsurface::SurfaceView>,
}

impl PySurfaceView {
    pub fn new(view: deepstream_nvbufsurface::SurfaceView) -> Self {
        Self { inner: Some(view) }
    }

    /// Consume the inner SurfaceView (e.g. for passing to Picasso).
    pub fn take(&mut self) -> PyResult<deepstream_nvbufsurface::SurfaceView> {
        self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("SurfaceView has been consumed")
        })
    }

    pub(crate) fn inner_ref(&self) -> PyResult<&deepstream_nvbufsurface::SurfaceView> {
        self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("SurfaceView has been consumed")
        })
    }

    /// Create a SurfaceView from a `__cuda_array_interface__` object.
    /// Callable from Rust code (e.g. Picasso `send_frame` dispatch).
    pub(crate) fn from_cuda_iface(
        py: Python<'_>,
        obj: Bound<'_, PyAny>,
        gpu_id: u32,
    ) -> PyResult<Self> {
        let iface = obj.getattr("__cuda_array_interface__").map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "object does not expose __cuda_array_interface__",
            )
        })?;

        let shape: Vec<u64> = iface.get_item("shape")?.extract()?;
        let typestr: String = iface.get_item("typestr")?.extract()?;
        let data_tuple: (usize, bool) = iface.get_item("data")?.extract()?;
        let data_ptr = data_tuple.0;

        if data_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "data pointer is null",
            ));
        }
        if typestr != "|u1" && typestr != "<u1" {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported dtype '{}'; only uint8 ('|u1' / '<u1') is supported",
                typestr
            )));
        }

        let (height, width, channels) = match shape.len() {
            2 => (shape[0] as u32, shape[1] as u32, 1u32),
            3 => (shape[0] as u32, shape[1] as u32, shape[2] as u32),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unsupported shape {:?}; expected (H, W) or (H, W, C)",
                    shape
                )));
            }
        };

        let color_format: u32 = match channels {
            1 => 1,  // NVBUF_COLOR_FORMAT_GRAY8
            4 => 19, // NVBUF_COLOR_FORMAT_RGBA
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unsupported channel count {}; expected 1 (GRAY8) or 4 (RGBA)",
                    channels
                )));
            }
        };

        let pitch = if let Ok(strides_obj) = iface.get_item("strides") {
            if !strides_obj.is_none() {
                let strides: Vec<u64> = strides_obj.extract()?;
                if strides.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "strides must be non-empty when present",
                    ));
                }
                strides[0] as u32
            } else {
                width * channels
            }
        } else {
            width * channels
        };

        let keepalive: pyo3::Py<PyAny> = obj.unbind();
        let boxed: Box<dyn std::any::Any + Send + Sync> = Box::new(keepalive);

        py.detach(|| {
            let view = deepstream_nvbufsurface::SurfaceView::from_cuda_ptr(
                data_ptr as *mut std::ffi::c_void,
                pitch,
                width,
                height,
                gpu_id,
                channels,
                color_format,
                Some(boxed),
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(PySurfaceView::new(view))
        })
    }
}

#[pymethods]
impl PySurfaceView {
    /// Create a view from an NvBufSurface-backed buffer.
    ///
    /// Args:
    ///     buf (GstBuffer | int): Source buffer.
    ///     slot_index (int): Zero-based slot index (default 0).
    ///
    /// Raises:
    ///     ValueError: If ``buf`` is null or ``slot_index`` is out of bounds.
    ///     RuntimeError: If the buffer is not a valid NvBufSurface or uses
    ///         a multi-plane format (NV12, I420, etc.).
    #[staticmethod]
    #[pyo3(signature = (buf, slot_index=0, cuda_stream=0))]
    fn from_buffer(
        py: Python<'_>,
        buf: &Bound<'_, PyAny>,
        slot_index: u32,
        cuda_stream: usize,
    ) -> PyResult<Self> {
        let shared = extract_shared_buffer(buf)?;
        py.detach(|| {
            let mut view = deepstream_nvbufsurface::SurfaceView::from_shared(shared, slot_index)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            if cuda_stream != 0 {
                view = view.with_cuda_stream(unsafe {
                    deepstream_nvbufsurface::CudaStream::from_raw(
                        cuda_stream as *mut std::ffi::c_void,
                    )
                });
            }
            Ok(PySurfaceView::new(view))
        })
    }

    /// Create a view from any object exposing ``__cuda_array_interface__``.
    ///
    /// Supported shapes:
    ///
    /// - ``(H, W, C)`` — interleaved: C must be 1 (GRAY8) or 4 (RGBA).
    /// - ``(H, W)``    — grayscale (GRAY8).
    ///
    /// The source object is kept alive for the lifetime of this view.
    ///
    /// Args:
    ///     obj: A CuPy array, PyTorch CUDA tensor, or any object with
    ///         ``__cuda_array_interface__``.
    ///     gpu_id (int): CUDA device ID (default 0).
    ///
    /// Raises:
    ///     TypeError: If *obj* has no ``__cuda_array_interface__``.
    ///     ValueError: If shape, dtype, or strides are unsupported.
    #[staticmethod]
    #[pyo3(signature = (obj, gpu_id=0, cuda_stream=0))]
    fn from_cuda_array(
        py: Python<'_>,
        obj: Bound<'_, PyAny>,
        gpu_id: u32,
        cuda_stream: usize,
    ) -> PyResult<Self> {
        let mut py_view = Self::from_cuda_iface(py, obj, gpu_id)?;
        if cuda_stream != 0 {
            let view = py_view.inner.take().unwrap();
            py_view.inner = Some(view.with_cuda_stream(unsafe {
                deepstream_nvbufsurface::CudaStream::from_raw(cuda_stream as *mut std::ffi::c_void)
            }));
        }
        Ok(py_view)
    }

    /// CUDA data pointer to the first pixel.
    #[getter]
    fn data_ptr(&self) -> PyResult<usize> {
        Ok(self.inner_ref()?.data_ptr() as usize)
    }

    /// Row stride in bytes.
    #[getter]
    fn pitch(&self) -> PyResult<u32> {
        Ok(self.inner_ref()?.pitch())
    }

    /// Surface width in pixels.
    #[getter]
    fn width(&self) -> PyResult<u32> {
        Ok(self.inner_ref()?.width())
    }

    /// Surface height in pixels.
    #[getter]
    fn height(&self) -> PyResult<u32> {
        Ok(self.inner_ref()?.height())
    }

    /// GPU device ID.
    #[getter]
    fn gpu_id(&self) -> PyResult<u32> {
        Ok(self.inner_ref()?.gpu_id())
    }

    /// Number of interleaved channels per pixel.
    #[getter]
    fn channels(&self) -> PyResult<u32> {
        Ok(self.inner_ref()?.channels())
    }

    /// Raw ``NvBufSurfaceColorFormat`` value.
    #[getter]
    fn color_format(&self) -> PyResult<u32> {
        Ok(self.inner_ref()?.color_format())
    }

    /// CUDA stream handle associated with this view (as an integer pointer).
    ///
    /// Returns 0 for the default (legacy) stream.
    #[getter]
    fn cuda_stream(&self) -> PyResult<usize> {
        Ok(self.inner_ref()?.cuda_stream().as_raw() as usize)
    }

    /// The ``__cuda_array_interface__`` descriptor (v3).
    ///
    /// Exposes the GPU surface as a CUDA array so that CuPy, PyTorch, and
    /// other external Python consumers can access the data without copies
    /// (e.g. ``cupy.asarray(surface_view)``, ``torch.as_tensor(surface_view)``).
    ///
    /// Only available for single-plane formats (RGBA, BGRx, GRAY8).
    #[getter]
    #[pyo3(name = "__cuda_array_interface__")]
    fn __cuda_array_interface__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let view = self.inner_ref()?;
        let dict = pyo3::types::PyDict::new(py);

        let channels = view.channels();
        let height = view.height();
        let width = view.width();

        let shape = if channels == 1 {
            (height as u64, width as u64)
                .into_pyobject(py)?
                .into_any()
                .unbind()
        } else {
            (height as u64, width as u64, channels as u64)
                .into_pyobject(py)?
                .into_any()
                .unbind()
        };

        let strides = if channels == 1 {
            (view.pitch() as u64, 1u64)
                .into_pyobject(py)?
                .into_any()
                .unbind()
        } else {
            (view.pitch() as u64, channels as u64, 1u64)
                .into_pyobject(py)?
                .into_any()
                .unbind()
        };

        dict.set_item("shape", shape)?;
        dict.set_item("typestr", "|u1")?;
        dict.set_item("descr", vec![("", "|u1")])?;
        dict.set_item("data", (view.data_ptr() as usize, false))?;
        dict.set_item("strides", strides)?;
        dict.set_item("version", 3)?;

        Ok(dict)
    }

    /// Fill the surface with a constant byte value.
    ///
    /// Every byte of the surface (up to ``pitch × height``) is set to
    /// *value*.  This is the fastest fill but only produces a uniform
    /// colour when all channels share the same byte (e.g. 0 for black,
    /// 255 for white on RGBA).  Use :meth:`fill` for arbitrary colours.
    ///
    /// Args:
    ///     value (int): Byte value (0–255) to fill every byte with.
    ///
    /// Raises:
    ///     RuntimeError: If the view has been consumed or the GPU operation
    ///         fails.
    fn memset(&self, value: u8) -> PyResult<()> {
        let view = self.inner_ref()?;
        deepstream_nvbufsurface::memset_surface(view, value)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Fill the surface with a repeating pixel colour.
    ///
    /// *color* must have exactly as many elements as the surface has
    /// channels (e.g. ``[R, G, B, A]`` for RGBA, ``[Y]`` for GRAY8).
    ///
    /// Example::
    ///
    ///     view.fill([128, 0, 255, 255])   # semi-blue, opaque RGBA
    ///
    /// Args:
    ///     color (list[int]): Per-channel byte values.
    ///
    /// Raises:
    ///     ValueError: If *color* length does not match the surface's
    ///         channel count.
    ///     RuntimeError: If the view has been consumed or the GPU
    ///         operation fails.
    fn fill(&self, color: Vec<u8>) -> PyResult<()> {
        let view = self.inner_ref()?;
        deepstream_nvbufsurface::fill_surface(view, &color).map_err(|e| match e {
            deepstream_nvbufsurface::NvBufSurfaceError::InvalidInput(_) => {
                pyo3::exceptions::PyValueError::new_err(e.to_string())
            }
            _ => pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
        })
    }

    /// Upload pixel data from a NumPy array to the surface.
    ///
    /// Args:
    ///     data (numpy.ndarray): A 3-D ``uint8`` array with shape
    ///         ``(height, width, channels)`` matching the surface dimensions
    ///         and color format (e.g. 4 channels for RGBA).
    ///
    /// Raises:
    ///     ValueError: If *data* has wrong shape, dtype, or dimensions.
    ///     RuntimeError: If the view has been consumed or the GPU operation
    ///         fails.
    fn upload<'py>(&self, data: PyReadonlyArray3<'py, u8>) -> PyResult<()> {
        let view = self.inner_ref()?;
        let arr = data.as_array();
        let shape = arr.shape();
        let height = shape[0] as u32;
        let width = shape[1] as u32;
        let channels = shape[2] as u32;
        let slice = data.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "array must be contiguous in memory: {e}"
            ))
        })?;
        deepstream_nvbufsurface::upload_to_surface(view, slice, width, height, channels)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(v) => format!(
                "SurfaceView({}x{}, ch={}, gpu={})",
                v.width(),
                v.height(),
                v.channels(),
                v.gpu_id()
            ),
            None => "SurfaceView(<consumed>)".to_string(),
        }
    }

    fn __bool__(&self) -> bool {
        self.inner.is_some()
    }
}

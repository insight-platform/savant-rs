use super::message::{PyBypassOutput, PyEncodedOutput};
use super::spec::general::PyEvictionDecision;
use deepstream_nvbufsurface::SkiaRenderer;
use picasso::prelude::*;
use pyo3::prelude::*;
use savant_core::draw::ObjectDraw;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::primitives::object::BorrowedVideoObject;
use std::sync::Arc;

struct PyOnEncodedFrame(Py<PyAny>);
unsafe impl Send for PyOnEncodedFrame {}
unsafe impl Sync for PyOnEncodedFrame {}

impl OnEncodedFrame for PyOnEncodedFrame {
    fn call(&self, output: EncodedOutput) {
        Python::attach(|py| {
            let py_output = PyEncodedOutput::from_rust(output);
            if let Err(e) = self.0.call1(py, (py_output,)) {
                log::error!("on_encoded_frame callback error: {e}");
            }
        });
    }
}

struct PyOnBypassFrame(Py<PyAny>);
unsafe impl Send for PyOnBypassFrame {}
unsafe impl Sync for PyOnBypassFrame {}

impl OnBypassFrame for PyOnBypassFrame {
    fn call(&self, output: BypassOutput) {
        Python::attach(|py| {
            let py_output = PyBypassOutput::from_rust(output);
            if let Err(e) = self.0.call1(py, (py_output,)) {
                log::error!("on_bypass_frame callback error: {e}");
            }
        });
    }
}

struct PyOnRender(Py<PyAny>);
unsafe impl Send for PyOnRender {}
unsafe impl Sync for PyOnRender {}

impl OnRender for PyOnRender {
    fn call(&self, source_id: &str, renderer: &mut SkiaRenderer, frame: &VideoFrameProxy) {
        let fbo_id = renderer.fbo_id();
        let width = renderer.width();
        let height = renderer.height();
        Python::attach(|py| {
            let py_frame = crate::primitives::frame::VideoFrame(frame.clone());
            if let Err(e) = self
                .0
                .call1(py, (source_id, fbo_id, width, height, py_frame))
            {
                log::error!("on_render callback error: {e}");
            }
        });
    }
}

struct PyOnObjectDrawSpec(Py<PyAny>);
unsafe impl Send for PyOnObjectDrawSpec {}
unsafe impl Sync for PyOnObjectDrawSpec {}

impl OnObjectDrawSpec for PyOnObjectDrawSpec {
    fn call(
        &self,
        source_id: &str,
        object: &BorrowedVideoObject,
        current_spec: Option<&ObjectDraw>,
    ) -> Option<ObjectDraw> {
        Python::attach(|py| {
            let py_object = crate::primitives::object::BorrowedVideoObject(object.clone());
            let py_spec = current_spec.map(super::spec::draw::rebuild_py_object_draw);
            match self.0.call1(py, (source_id, py_object, py_spec)) {
                Ok(result) => {
                    if result.is_none(py) {
                        return None;
                    }
                    match result.extract::<crate::draw_spec::ObjectDraw>(py) {
                        Ok(draw) => Some(draw.0.clone()),
                        Err(e) => {
                            log::error!("on_object_draw_spec return type error: {e}");
                            None
                        }
                    }
                }
                Err(e) => {
                    log::error!("on_object_draw_spec callback error: {e}");
                    None
                }
            }
        })
    }
}

struct PyOnGpuMat(Py<PyAny>);
unsafe impl Send for PyOnGpuMat {}
unsafe impl Sync for PyOnGpuMat {}

impl OnGpuMat for PyOnGpuMat {
    fn call(
        &self,
        source_id: &str,
        frame: &VideoFrameProxy,
        data_ptr: usize,
        pitch: u32,
        width: u32,
        height: u32,
        cuda_stream: usize,
    ) {
        Python::attach(|py| {
            let py_frame = crate::primitives::frame::VideoFrame(frame.clone());
            if let Err(e) = self.0.call1(
                py,
                (
                    source_id,
                    py_frame,
                    data_ptr,
                    pitch,
                    width,
                    height,
                    cuda_stream,
                ),
            ) {
                log::error!("on_gpumat callback error: {e}");
            }
        });
    }
}

struct PyOnEviction(Py<PyAny>);
unsafe impl Send for PyOnEviction {}
unsafe impl Sync for PyOnEviction {}

impl OnEviction for PyOnEviction {
    fn call(&self, source_id: &str) -> EvictionDecision {
        Python::attach(|py| match self.0.call1(py, (source_id,)) {
            Ok(result) => match result.extract::<PyEvictionDecision>(py) {
                Ok(d) => d.to_rust(),
                Err(e) => {
                    log::error!("on_eviction return type error: {e}");
                    EvictionDecision::Terminate
                }
            },
            Err(e) => {
                log::error!("on_eviction callback error: {e}");
                EvictionDecision::Terminate
            }
        })
    }
}

/// Aggregate holder for all optional Python callbacks.
#[pyclass(name = "Callbacks", module = "savant_rs.picasso")]
pub struct PyCallbacks {
    pub(crate) on_encoded_frame: Option<Py<PyAny>>,
    pub(crate) on_bypass_frame: Option<Py<PyAny>>,
    pub(crate) on_render: Option<Py<PyAny>>,
    pub(crate) on_object_draw_spec: Option<Py<PyAny>>,
    pub(crate) on_gpumat: Option<Py<PyAny>>,
    pub(crate) on_eviction: Option<Py<PyAny>>,
}

#[pymethods]
impl PyCallbacks {
    #[new]
    #[pyo3(signature = (
        on_encoded_frame = None,
        on_bypass_frame = None,
        on_render = None,
        on_object_draw_spec = None,
        on_gpumat = None,
        on_eviction = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        on_encoded_frame: Option<Py<PyAny>>,
        on_bypass_frame: Option<Py<PyAny>>,
        on_render: Option<Py<PyAny>>,
        on_object_draw_spec: Option<Py<PyAny>>,
        on_gpumat: Option<Py<PyAny>>,
        on_eviction: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            on_encoded_frame,
            on_bypass_frame,
            on_render,
            on_object_draw_spec,
            on_gpumat,
            on_eviction,
        }
    }

    #[getter]
    fn get_on_encoded_frame(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.on_encoded_frame.as_ref().map(|cb| cb.clone_ref(py))
    }

    #[setter]
    fn set_on_encoded_frame(&mut self, cb: Option<Py<PyAny>>) {
        self.on_encoded_frame = cb;
    }

    #[getter]
    fn get_on_bypass_frame(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.on_bypass_frame.as_ref().map(|cb| cb.clone_ref(py))
    }

    #[setter]
    fn set_on_bypass_frame(&mut self, cb: Option<Py<PyAny>>) {
        self.on_bypass_frame = cb;
    }

    #[getter]
    fn get_on_render(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.on_render.as_ref().map(|cb| cb.clone_ref(py))
    }

    #[setter]
    fn set_on_render(&mut self, cb: Option<Py<PyAny>>) {
        self.on_render = cb;
    }

    #[getter]
    fn get_on_object_draw_spec(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.on_object_draw_spec.as_ref().map(|cb| cb.clone_ref(py))
    }

    #[setter]
    fn set_on_object_draw_spec(&mut self, cb: Option<Py<PyAny>>) {
        self.on_object_draw_spec = cb;
    }

    #[getter]
    fn get_on_gpumat(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.on_gpumat.as_ref().map(|cb| cb.clone_ref(py))
    }

    #[setter]
    fn set_on_gpumat(&mut self, cb: Option<Py<PyAny>>) {
        self.on_gpumat = cb;
    }

    #[getter]
    fn get_on_eviction(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.on_eviction.as_ref().map(|cb| cb.clone_ref(py))
    }

    #[setter]
    fn set_on_eviction(&mut self, cb: Option<Py<PyAny>>) {
        self.on_eviction = cb;
    }

    fn __repr__(&self) -> String {
        let slots: Vec<&str> = [
            self.on_encoded_frame.as_ref().map(|_| "on_encoded_frame"),
            self.on_bypass_frame.as_ref().map(|_| "on_bypass_frame"),
            self.on_render.as_ref().map(|_| "on_render"),
            self.on_object_draw_spec
                .as_ref()
                .map(|_| "on_object_draw_spec"),
            self.on_gpumat.as_ref().map(|_| "on_gpumat"),
            self.on_eviction.as_ref().map(|_| "on_eviction"),
        ]
        .iter()
        .filter_map(|x| *x)
        .collect();
        format!("Callbacks(active=[{}])", slots.join(", "))
    }
}

impl PyCallbacks {
    /// Convert into the Rust [`Callbacks`] struct.
    pub(crate) fn to_rust(&self, py: Python<'_>) -> Callbacks {
        Callbacks {
            on_encoded_frame: self
                .on_encoded_frame
                .as_ref()
                .map(|cb| Arc::new(PyOnEncodedFrame(cb.clone_ref(py))) as Arc<dyn OnEncodedFrame>),
            on_bypass_frame: self
                .on_bypass_frame
                .as_ref()
                .map(|cb| Arc::new(PyOnBypassFrame(cb.clone_ref(py))) as Arc<dyn OnBypassFrame>),
            on_render: self
                .on_render
                .as_ref()
                .map(|cb| Arc::new(PyOnRender(cb.clone_ref(py))) as Arc<dyn OnRender>),
            on_object_draw_spec: self.on_object_draw_spec.as_ref().map(|cb| {
                Arc::new(PyOnObjectDrawSpec(cb.clone_ref(py))) as Arc<dyn OnObjectDrawSpec>
            }),
            on_gpumat: self
                .on_gpumat
                .as_ref()
                .map(|cb| Arc::new(PyOnGpuMat(cb.clone_ref(py))) as Arc<dyn OnGpuMat>),
            on_eviction: self
                .on_eviction
                .as_ref()
                .map(|cb| Arc::new(PyOnEviction(cb.clone_ref(py))) as Arc<dyn OnEviction>),
        }
    }
}

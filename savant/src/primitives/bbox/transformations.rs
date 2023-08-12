use pyo3::prelude::*;

#[derive(Debug, Clone, Copy)]
pub enum VideoObjectBBoxTransformation {
    Scale(f32, f32),
    Shift(f32, f32),
}

#[pyclass]
#[derive(Debug, Clone, Copy)]
#[pyo3(name = "VideoObjectBBoxTransformation")]
pub(crate) struct VideoObjectBBoxTransformationProxy {
    pub transformation: VideoObjectBBoxTransformation,
}

impl VideoObjectBBoxTransformationProxy {
    pub fn get_ref(&self) -> &VideoObjectBBoxTransformation {
        &self.transformation
    }
}

#[pymethods]
impl VideoObjectBBoxTransformationProxy {
    #[staticmethod]
    fn scale(x: f32, y: f32) -> Self {
        Self {
            transformation: VideoObjectBBoxTransformation::Scale(x, y),
        }
    }

    #[staticmethod]
    fn shift(x: f32, y: f32) -> Self {
        Self {
            transformation: VideoObjectBBoxTransformation::Shift(x, y),
        }
    }
}

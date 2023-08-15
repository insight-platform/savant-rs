use crate::np::np_nalgebra::NalgebraDMatrix;
use crate::np::np_ndarray::NDarray;
use pyo3::prelude::*;
use pyo3::pyclass;

/// The format of a bounding box passed as a parameter or requested as a return type.
///
/// LeftTopRightBottom
///   The format is [left, top, right, bottom].
/// LeftTopWidthHeight
///   The format is [left, top, width, height].
/// XcYcWidthHeight
///   The format is [xcenter, ycenter, width, height].
///
#[pyclass]
#[derive(Debug, Clone)]
pub enum BBoxFormat {
    LeftTopRightBottom,
    LeftTopWidthHeight,
    XcYcWidthHeight,
}

pub(crate) mod conversions;
mod np;

use np::bbox::*;
use np::np_nalgebra::*;
use np::np_ndarray::*;

#[pymodule]
pub fn savant_nalgebra(_py: Python, m: &PyModule) -> PyResult<()> {
    // bbox batch ops
    m.add_function(wrap_pyfunction!(rotated_bboxes_to_ndarray_gil, m)?)?;
    m.add_function(wrap_pyfunction!(bboxes_to_ndarray_gil, m)?)?;
    m.add_function(wrap_pyfunction!(ndarray_to_bboxes_py, m)?)?;
    m.add_function(wrap_pyfunction!(ndarray_to_rotated_bboxes_py, m)?)?;
    //
    // // numpy utils
    m.add_function(wrap_pyfunction!(np_to_matrix_gil, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_to_np_py, m)?)?;
    m.add_function(wrap_pyfunction!(np_to_ndarray_gil, m)?)?;
    m.add_function(wrap_pyfunction!(ndarray_to_np_py, m)?)?;

    m.add_class::<NalgebraDMatrix>()?;
    m.add_class::<NDarray>()?;
    m.add_class::<BBoxFormat>()?;

    Ok(())
}

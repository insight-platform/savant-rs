use crate::np::np_nalgebra::NalgebraDMatrix;
use crate::np::np_ndarray::NDarray;
use pyo3::prelude::*;
use savant_rs::utils::bbox::BBoxFormat;

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

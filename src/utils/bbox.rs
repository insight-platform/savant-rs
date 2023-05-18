use crate::primitives::{BBox, RBBox};
use crate::utils::np::{ConvF64, ElementType, RConvF64};
use numpy::ndarray::ArrayD;
use numpy::{IxDyn, PyArray, PyReadonlyArrayDyn};
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub enum BBoxFormat {
    LeftTopRightBottom,
    LeftTopWidthHeight,
    XcYcWidthHeight,
}

pub fn ndarray_to_rotated_bboxes<T: ElementType + ConvF64>(
    arr: PyReadonlyArrayDyn<T>,
) -> Vec<RBBox> {
    let dims = arr.shape();
    assert!(dims.len() == 2 && dims[1] >= 5);
    arr.as_array()
        .rows()
        .into_iter()
        .map(|r| {
            RBBox::new(
                r[0].conv_f64(),
                r[1].conv_f64(),
                r[2].conv_f64(),
                r[3].conv_f64(),
                Some(r[4].conv_f64()),
            )
        })
        .collect::<Vec<_>>()
}

pub fn ndarray_to_bboxes<T: ElementType + ConvF64>(
    arr: PyReadonlyArrayDyn<T>,
    format: BBoxFormat,
) -> Vec<BBox> {
    let dims = arr.shape();
    assert!(dims.len() == 2 && dims[1] >= 4);
    arr.as_array()
        .rows()
        .into_iter()
        .map(|r| match format {
            BBoxFormat::LeftTopRightBottom => BBox::ltrb(
                r[0].conv_f64(),
                r[1].conv_f64(),
                r[2].conv_f64(),
                r[3].conv_f64(),
            ),
            BBoxFormat::LeftTopWidthHeight => BBox::ltwh(
                r[0].conv_f64(),
                r[1].conv_f64(),
                r[2].conv_f64(),
                r[3].conv_f64(),
            ),
            BBoxFormat::XcYcWidthHeight => BBox::new(
                r[0].conv_f64(),
                r[1].conv_f64(),
                r[2].conv_f64(),
                r[3].conv_f64(),
            ),
        })
        .collect::<Vec<_>>()
}

pub fn bboxes_to_ndarray<T: ElementType + RConvF64 + num_traits::identities::Zero>(
    boxes: Vec<BBox>,
    format: BBoxFormat,
) -> Py<PyArray<T, IxDyn>> {
    Python::with_gil(|py| {
        let arr = py.allow_threads(|| {
            let mut arr = ArrayD::<T>::zeros(IxDyn(&[boxes.len(), 4]));
            for (i, bbox) in boxes.iter().enumerate() {
                let (v0, v1, v2, v3) = match format {
                    BBoxFormat::LeftTopRightBottom => bbox.as_ltrb(),
                    BBoxFormat::LeftTopWidthHeight => bbox.as_ltwh(),
                    BBoxFormat::XcYcWidthHeight => bbox.as_xcycwh(),
                };

                arr[[i, 0]] = RConvF64::conv_from_f64(v0);
                arr[[i, 1]] = RConvF64::conv_from_f64(v1);
                arr[[i, 2]] = RConvF64::conv_from_f64(v2);
                arr[[i, 3]] = RConvF64::conv_from_f64(v3);
            }
            arr
        });
        let arr = PyArray::from_array(py, &arr);
        arr.into_py(py)
    })
}

pub fn rotated_bboxes_to_ndarray<T: ElementType + RConvF64 + num_traits::identities::Zero>(
    boxes: Vec<RBBox>,
) -> Py<PyArray<T, IxDyn>> {
    Python::with_gil(|py| {
        let arr = py.allow_threads(|| {
            let mut arr = ArrayD::<T>::zeros(IxDyn(&[boxes.len(), 5]));
            for (i, bbox) in boxes.iter().enumerate() {
                arr[[i, 0]] = RConvF64::conv_from_f64(bbox.xc);
                arr[[i, 1]] = RConvF64::conv_from_f64(bbox.yc);
                arr[[i, 2]] = RConvF64::conv_from_f64(bbox.width);
                arr[[i, 3]] = RConvF64::conv_from_f64(bbox.height);
                arr[[i, 4]] = RConvF64::conv_from_f64(bbox.angle.unwrap_or(0.0));
            }
            arr
        });
        let arr = PyArray::from_array(py, &arr);
        arr.into_py(py)
    })
}

#[pyfunction]
#[pyo3(name = "rotated_bboxes_to_ndarray")]
pub fn rotated_bboxes_to_ndarray_py(boxes: Vec<RBBox>, dtype: String) -> PyObject {
    match dtype.as_str() {
        "float32" => {
            let arr = rotated_bboxes_to_ndarray::<f32>(boxes);
            Python::with_gil(|py| arr.to_object(py))
        }
        "float64" => {
            let arr = rotated_bboxes_to_ndarray::<f64>(boxes);
            Python::with_gil(|py| arr.to_object(py))
        }
        "int32" => {
            let arr = rotated_bboxes_to_ndarray::<i32>(boxes);
            Python::with_gil(|py| arr.to_object(py))
        }
        "int64" => {
            let arr = rotated_bboxes_to_ndarray::<i64>(boxes);
            Python::with_gil(|py| arr.to_object(py))
        }
        _ => panic!("Unsupported dtype"),
    }
}

#[pyfunction]
#[pyo3(name = "ndarray_to_bboxes")]
pub fn ndarray_to_bboxes_py(arr: &PyAny, format: BBoxFormat) -> PyResult<Vec<BBox>> {
    if let Ok(arr) = arr.downcast::<PyArray<f32, IxDyn>>() {
        return Ok(ndarray_to_bboxes(arr.readonly(), format));
    }

    if let Ok(arr) = arr.downcast::<PyArray<f64, IxDyn>>() {
        return Ok(ndarray_to_bboxes(arr.readonly(), format));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i32, IxDyn>>() {
        return Ok(ndarray_to_bboxes(arr.readonly(), format));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i64, IxDyn>>() {
        return Ok(ndarray_to_bboxes(arr.readonly(), format));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected ndarray of type f32, f64, i32 or i64",
    ))
}

#[pyfunction]
#[pyo3(name = "ndarray_to_rotated_bboxes")]
pub fn ndarray_to_rotated_bboxes_py(arr: &PyAny) -> PyResult<Vec<RBBox>> {
    if let Ok(arr) = arr.downcast::<PyArray<f32, IxDyn>>() {
        return Ok(ndarray_to_rotated_bboxes(arr.readonly()));
    }

    if let Ok(arr) = arr.downcast::<PyArray<f64, IxDyn>>() {
        return Ok(ndarray_to_rotated_bboxes(arr.readonly()));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i32, IxDyn>>() {
        return Ok(ndarray_to_rotated_bboxes(arr.readonly()));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i64, IxDyn>>() {
        return Ok(ndarray_to_rotated_bboxes(arr.readonly()));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected ndarray of type f32, f64, i32 or i64",
    ))
}

#[pyfunction]
#[pyo3(name = "bboxes_to_ndarray")]
pub fn bboxes_to_ndarray_py(boxes: Vec<BBox>, format: BBoxFormat, dtype: String) -> PyObject {
    match dtype.as_str() {
        "float32" => {
            let arr = bboxes_to_ndarray::<f32>(boxes, format);
            Python::with_gil(|py| arr.to_object(py))
        }
        "float64" => {
            let arr = bboxes_to_ndarray::<f64>(boxes, format);
            Python::with_gil(|py| arr.to_object(py))
        }
        "int32" => {
            let arr = bboxes_to_ndarray::<i32>(boxes, format);
            Python::with_gil(|py| arr.to_object(py))
        }
        "int64" => {
            let arr = bboxes_to_ndarray::<i64>(boxes, format);
            Python::with_gil(|py| arr.to_object(py))
        }
        _ => panic!("Unsupported dtype"),
    }
}

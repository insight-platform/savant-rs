use crate::primitives::{PythonBBox, RBBox};
use crate::utils::np::{ConvF32, ElementType, RConvF32};
use crate::with_gil;
use numpy::ndarray::ArrayD;
use numpy::{IxDyn, PyArray, PyReadonlyArrayDyn};
use pyo3::prelude::*;

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

pub fn ndarray_to_rotated_bboxes<T: ElementType + ConvF32>(
    arr: &PyReadonlyArrayDyn<T>,
) -> Vec<RBBox> {
    let dims = arr.shape();
    assert!(dims.len() == 2 && dims[1] >= 5);
    arr.as_array()
        .rows()
        .into_iter()
        .map(|r| {
            RBBox::new(
                r[0].conv_f32(),
                r[1].conv_f32(),
                r[2].conv_f32(),
                r[3].conv_f32(),
                Some(r[4].conv_f32()),
            )
        })
        .collect::<Vec<_>>()
}

pub fn ndarray_to_bboxes<T: ElementType + ConvF32>(
    arr: &PyReadonlyArrayDyn<T>,
    format: &BBoxFormat,
) -> Vec<PythonBBox> {
    let dims = arr.shape();
    assert!(dims.len() == 2 && dims[1] >= 4);
    arr.as_array()
        .rows()
        .into_iter()
        .map(|r| match format {
            BBoxFormat::LeftTopRightBottom => PythonBBox::ltrb(
                r[0].conv_f32(),
                r[1].conv_f32(),
                r[2].conv_f32(),
                r[3].conv_f32(),
            ),
            BBoxFormat::LeftTopWidthHeight => PythonBBox::ltwh(
                r[0].conv_f32(),
                r[1].conv_f32(),
                r[2].conv_f32(),
                r[3].conv_f32(),
            ),
            BBoxFormat::XcYcWidthHeight => PythonBBox::new(
                r[0].conv_f32(),
                r[1].conv_f32(),
                r[2].conv_f32(),
                r[3].conv_f32(),
            ),
        })
        .collect::<Vec<_>>()
}

pub fn bboxes_to_ndarray<T: ElementType + RConvF32 + num_traits::identities::Zero>(
    boxes: &Vec<PythonBBox>,
    format: &BBoxFormat,
) -> Py<PyArray<T, IxDyn>> {
    let arr = {
        let mut arr = ArrayD::<T>::zeros(IxDyn(&[boxes.len(), 4]));
        for (i, bbox) in boxes.iter().enumerate() {
            let (v0, v1, v2, v3) = match format {
                BBoxFormat::LeftTopRightBottom => bbox.as_ltrb(),
                BBoxFormat::LeftTopWidthHeight => bbox.as_ltwh(),
                BBoxFormat::XcYcWidthHeight => bbox.as_xcycwh(),
            };

            arr[[i, 0]] = RConvF32::conv_from_f32(v0);
            arr[[i, 1]] = RConvF32::conv_from_f32(v1);
            arr[[i, 2]] = RConvF32::conv_from_f32(v2);
            arr[[i, 3]] = RConvF32::conv_from_f32(v3);
        }
        arr
    };

    with_gil!(|py| {
        let arr = PyArray::from_array(py, &arr);
        arr.into_py(py)
    })
}

pub fn rotated_bboxes_to_ndarray<T: ElementType + RConvF32 + num_traits::identities::Zero>(
    boxes: Vec<RBBox>,
) -> Py<PyArray<T, IxDyn>> {
    let arr = {
        let mut arr = ArrayD::<T>::zeros(IxDyn(&[boxes.len(), 5]));
        for (i, bbox) in boxes.iter().enumerate() {
            arr[[i, 0]] = RConvF32::conv_from_f32(bbox.get_xc());
            arr[[i, 1]] = RConvF32::conv_from_f32(bbox.get_yc());
            arr[[i, 2]] = RConvF32::conv_from_f32(bbox.get_width());
            arr[[i, 3]] = RConvF32::conv_from_f32(bbox.get_height());
            arr[[i, 4]] = RConvF32::conv_from_f32(bbox.get_angle().unwrap_or(0.0));
        }
        arr
    };

    with_gil!(|py| {
        let arr = PyArray::from_array(py, &arr);
        arr.into_py(py)
    })
}

/// Converts a list of :class:`savant_rs.primitives.geometry.RBBox`-es to a numpy
/// array with rows represented by ``[xc, yc, width, height, angle]``.
///
/// Parameters
/// ----------
/// arr : List[savant_rs.primitives.geometry.RBBox]
///   The numpy array with rows represented by ``[xc, yc, width, height, angle]``.
/// dtype : str
///   The data type of the numpy array. Can be ``float32``, ``float64``, ``int32`` or ``int64``.
///
/// Returns
/// -------
/// numpy.ndarray
///   The numpy array with rows represented by ``[xc, yc, width, height, angle]``.
///
///
/// Panics when a data type is not ``float32``, ``float64``, ``int32`` or ``int64``.
///
#[pyfunction]
#[pyo3(name = "rotated_bboxes_to_ndarray")]
pub fn rotated_bboxes_to_ndarray_gil(boxes: Vec<RBBox>, dtype: String) -> PyObject {
    match dtype.as_str() {
        "float32" => {
            let arr = rotated_bboxes_to_ndarray::<f32>(boxes);
            with_gil!(|py| arr.to_object(py))
        }
        "float64" => {
            let arr = rotated_bboxes_to_ndarray::<f64>(boxes);
            with_gil!(|py| arr.to_object(py))
        }
        "int32" => {
            let arr = rotated_bboxes_to_ndarray::<i32>(boxes);
            with_gil!(|py| arr.to_object(py))
        }
        "int64" => {
            let arr = rotated_bboxes_to_ndarray::<i64>(boxes);
            with_gil!(|py| arr.to_object(py))
        }
        _ => panic!("Unsupported dtype"),
    }
}

/// Converts a numpy array with rows in a format specified by ``format`` to a list of :class:`savant_rs.primitives.geometry.BBox`-es.
///
/// Parameters
/// ----------
/// arr : numpy.ndarray
///   The numpy array with rows in a format specified by ``format``.
/// format : BBoxFormat
///   The format of the numpy array. Can be ``BBoxFormat.LeftTopRightBottom``, ``BBoxFormat.LeftTopWidthHeight`` or ``BBoxFormat.XcYcWidthHeight``.
///
/// Returns
/// -------
/// List[savant_rs.primitives.geometry.BBox]
///   The list of :class:`savant_rs.primitives.geometry.BBox`-es.
///
#[pyfunction]
#[pyo3(name = "ndarray_to_bboxes")]
pub fn ndarray_to_bboxes_gil(arr: &PyAny, format: &BBoxFormat) -> PyResult<Vec<PythonBBox>> {
    if let Ok(arr) = arr.downcast::<PyArray<f32, IxDyn>>() {
        return Ok(ndarray_to_bboxes(&arr.readonly(), format));
    }

    if let Ok(arr) = arr.downcast::<PyArray<f64, IxDyn>>() {
        return Ok(ndarray_to_bboxes(&arr.readonly(), format));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i32, IxDyn>>() {
        return Ok(ndarray_to_bboxes(&arr.readonly(), format));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i64, IxDyn>>() {
        return Ok(ndarray_to_bboxes(&arr.readonly(), format));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected ndarray of type f32, f64, i32 or i64",
    ))
}

/// Converts numpy array with rows represented by ``[xc, yc, width, height, angle]`` to a list of :class:`savant_rs.primitives.geometry.RBBox`-es.
///
/// Parameters
/// ----------
/// arr : numpy.ndarray
///   The numpy array with rows represented by ``[xc, yc, width, height, angle]``.
///
/// Returns
/// -------
/// List[savant_rs.primitives.geometry.RBBox]
///   The list of :class:`savant_rs.primitives.geometry.RBBox`-es.
///
/// Raises
/// ------
/// TypeError
///   If the numpy array is not of type ``float32``, ``float64``, ``int32`` or ``int64``.
///
#[pyfunction]
#[pyo3(name = "ndarray_to_rotated_bboxes")]
pub fn ndarray_to_rotated_bboxes_gil(arr: &PyAny) -> PyResult<Vec<RBBox>> {
    if let Ok(arr) = arr.downcast::<PyArray<f32, IxDyn>>() {
        return Ok(ndarray_to_rotated_bboxes(&arr.readonly()));
    }

    if let Ok(arr) = arr.downcast::<PyArray<f64, IxDyn>>() {
        return Ok(ndarray_to_rotated_bboxes(&arr.readonly()));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i32, IxDyn>>() {
        return Ok(ndarray_to_rotated_bboxes(&arr.readonly()));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i64, IxDyn>>() {
        return Ok(ndarray_to_rotated_bboxes(&arr.readonly()));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected ndarray of type f32, f64, i32 or i64",
    ))
}

/// Converts a list of :class:`savant_rs.primitives.geometry.BBox`-es to a numpy ndarray. The format of the ndarray is determined by the ``format`` parameter.
///
/// Parameters
/// ----------
/// boxes : List[savant_rs.primitives.geometry.BBox]
///   The list of :class:`savant_rs.primitives.geometry.BBox`-es.
/// format : BBoxFormat
///   The format of bbox representation. One
///   of :class:`BBoxFormat.LeftTopRightBottom`, :class:`BBoxFormat.LeftTopWidthHeight`, or :class:`BBoxFormat.XcYcWidthHeight`.
/// dtype : str
///   The data type of the numpy array. Can be ``float32``, ``float64``, ``int32`` or ``int64``.
///
/// Returns
/// -------
/// numpy.ndarray
///   The numpy array with rows in a specified format.
///
#[pyfunction]
#[pyo3(name = "bboxes_to_ndarray")]
pub fn bboxes_to_ndarray_gil(
    boxes: Vec<PythonBBox>,
    format: &BBoxFormat,
    dtype: String,
) -> PyObject {
    match dtype.as_str() {
        "float32" => {
            let arr = bboxes_to_ndarray::<f32>(&boxes, format);
            with_gil!(|py| arr.to_object(py))
        }
        "float64" => {
            let arr = bboxes_to_ndarray::<f64>(&boxes, format);
            with_gil!(|py| arr.to_object(py))
        }
        "int32" => {
            let arr = bboxes_to_ndarray::<i32>(&boxes, format);
            with_gil!(|py| arr.to_object(py))
        }
        "int64" => {
            let arr = bboxes_to_ndarray::<i64>(&boxes, format);
            with_gil!(|py| arr.to_object(py))
        }
        _ => panic!("Unsupported dtype"),
    }
}

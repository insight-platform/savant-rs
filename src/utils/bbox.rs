use crate::primitives::{BBox, RBBox};
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

#[pyfunction]
pub fn bboxes_to_ndarray_float(boxes: Vec<BBox>, format: BBoxFormat) -> Py<PyArray<f64, IxDyn>> {
    Python::with_gil(|py| {
        let arr = py.allow_threads(|| {
            let mut arr = ArrayD::<f64>::zeros(IxDyn(&[boxes.len(), 4]));
            for (i, bbox) in boxes.iter().enumerate() {
                let (v0, v1, v2, v3) = match format {
                    BBoxFormat::LeftTopRightBottom => bbox.as_ltrb(),
                    BBoxFormat::LeftTopWidthHeight => bbox.as_ltwh(),
                    BBoxFormat::XcYcWidthHeight => bbox.as_xcycwh(),
                };

                arr[[i, 0]] = v0;
                arr[[i, 1]] = v1;
                arr[[i, 2]] = v2;
                arr[[i, 3]] = v3;
            }
            arr
        });
        let arr = PyArray::from_array(py, &arr);
        arr.into_py(py)
    })
}

#[pyfunction]
pub fn bboxes_to_ndarray_int(boxes: Vec<BBox>, format: BBoxFormat) -> Py<PyArray<i64, IxDyn>> {
    Python::with_gil(|py| {
        let arr = py.allow_threads(|| {
            let mut arr = ArrayD::<i64>::zeros(IxDyn(&[boxes.len(), 4]));
            for (i, bbox) in boxes.iter().enumerate() {
                let (v0, v1, v2, v3) = match format {
                    BBoxFormat::LeftTopRightBottom => bbox.as_ltrb_int(),
                    BBoxFormat::LeftTopWidthHeight => bbox.as_ltwh_int(),
                    BBoxFormat::XcYcWidthHeight => bbox.as_xcycwh_int(),
                };

                arr[[i, 0]] = v0;
                arr[[i, 1]] = v1;
                arr[[i, 2]] = v2;
                arr[[i, 3]] = v3;
            }
            arr
        });
        let arr = PyArray::from_array(py, &arr);
        arr.into_py(py)
    })
}

#[pyfunction]
pub fn ndarray_float_to_bboxes(arr: PyReadonlyArrayDyn<f64>, format: BBoxFormat) -> Vec<BBox> {
    let dims = arr.shape();
    assert!(dims.len() == 2 && dims[1] >= 4);
    arr.as_array()
        .rows()
        .into_iter()
        .map(|r| match format {
            BBoxFormat::LeftTopRightBottom => BBox::ltrb(r[0], r[1], r[2], r[3]),
            BBoxFormat::LeftTopWidthHeight => BBox::ltwh(r[0], r[1], r[2], r[3]),
            BBoxFormat::XcYcWidthHeight => BBox::new(r[0], r[1], r[2], r[3]),
        })
        .collect::<Vec<_>>()
}

#[pyfunction]
pub fn ndarray_int_to_bboxes(arr: PyReadonlyArrayDyn<i64>, format: BBoxFormat) -> Vec<BBox> {
    let dims = arr.shape();
    assert!(dims.len() == 2 && dims[1] >= 4);
    arr.as_array()
        .rows()
        .into_iter()
        .map(|r| match format {
            BBoxFormat::LeftTopRightBottom => {
                BBox::ltrb(r[0] as f64, r[1] as f64, r[2] as f64, r[3] as f64)
            }
            BBoxFormat::LeftTopWidthHeight => {
                BBox::ltwh(r[0] as f64, r[1] as f64, r[2] as f64, r[3] as f64)
            }
            BBoxFormat::XcYcWidthHeight => {
                BBox::new(r[0] as f64, r[1] as f64, r[2] as f64, r[3] as f64)
            }
        })
        .collect::<Vec<_>>()
}

#[pyfunction]
pub fn ndarray_float_to_rotated_bboxes(arr: PyReadonlyArrayDyn<f64>) -> Vec<RBBox> {
    let dims = arr.shape();
    assert!(dims.len() == 2 && dims[1] >= 5);
    arr.as_array()
        .rows()
        .into_iter()
        .map(|r| RBBox::new(r[0], r[1], r[2], r[3], Some(r[4])))
        .collect::<Vec<_>>()
}

#[pyfunction]
pub fn ndarray_int_to_rotated_bboxes(arr: PyReadonlyArrayDyn<i64>) -> Vec<RBBox> {
    let dims = arr.shape();
    assert!(dims.len() == 2 && dims[1] >= 5);
    arr.as_array()
        .rows()
        .into_iter()
        .map(|r| {
            RBBox::new(
                r[0] as f64,
                r[1] as f64,
                r[2] as f64,
                r[3] as f64,
                Some(r[4] as f64),
            )
        })
        .collect::<Vec<_>>()
}

#[pyfunction]
pub fn rotated_bboxes_to_ndarray_float(boxes: Vec<RBBox>) -> Py<PyArray<f64, IxDyn>> {
    Python::with_gil(|py| {
        let arr = py.allow_threads(|| {
            let mut arr = ArrayD::<f64>::zeros(IxDyn(&[boxes.len(), 5]));
            for (i, bbox) in boxes.iter().enumerate() {
                arr[[i, 0]] = bbox.xc;
                arr[[i, 1]] = bbox.yc;
                arr[[i, 2]] = bbox.width;
                arr[[i, 3]] = bbox.height;
                arr[[i, 4]] = bbox.angle.unwrap_or(0.0);
            }
            arr
        });
        let arr = PyArray::from_array(py, &arr);
        arr.into_py(py)
    })
}

#[pyfunction]
pub fn rotated_bboxes_to_ndarray_int(boxes: Vec<RBBox>) -> Py<PyArray<i64, IxDyn>> {
    Python::with_gil(|py| {
        let arr = py.allow_threads(|| {
            let mut arr = ArrayD::<i64>::zeros(IxDyn(&[boxes.len(), 5]));
            for (i, bbox) in boxes.iter().enumerate() {
                arr[[i, 0]] = bbox.xc as i64;
                arr[[i, 1]] = bbox.yc as i64;
                arr[[i, 2]] = bbox.width as i64;
                arr[[i, 3]] = bbox.height as i64;
                arr[[i, 4]] = bbox.angle.unwrap_or(0.0) as i64;
            }
            arr
        });
        let arr = PyArray::from_array(py, &arr);
        arr.into_py(py)
    })
}

#[cfg(test)]
mod tests {
    use crate::primitives::BBox;
    use crate::utils::bbox::BBoxFormat;
    use crate::utils::bboxes_to_ndarray_float;

    #[test]
    fn test_to_ndarray() {
        pyo3::prepare_freethreaded_python();
        let _res = bboxes_to_ndarray_float(
            vec![BBox::new(0.0, 0.0, 10.0, 20.0)],
            BBoxFormat::LeftTopRightBottom,
        );
    }
}

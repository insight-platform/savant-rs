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

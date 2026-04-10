use crate::attach;
use crate::draw_spec::SetDrawLabelKind;
use crate::match_query::MatchQuery;
use crate::primitives::attribute::Attribute;
use crate::primitives::attribute_value::AttributeValue;
use crate::primitives::bbox::{RBBox, VideoObjectBBoxTransformation};
use crate::primitives::frame_update::VideoFrameUpdate;
use crate::primitives::message::Message;
use crate::primitives::object::{BorrowedVideoObject, IdCollisionResolutionPolicy, VideoObject};
use crate::primitives::objects_view::VideoObjectsView;
use crate::utils::bigint::fit_i64;
use crate::{detach, err_to_pyerr};
use pyo3::exceptions::{PyRuntimeError, PySystemError, PyValueError};
use pyo3::types::{PyBytes, PyBytesMethods};
use pyo3::{pyclass, pymethods, Bound, Py, PyAny, PyResult};
use savant_core::json_api::ToSerdeJsonValue;
use savant_core::primitives::object::ObjectOperations;
use savant_core::primitives::{rust, WithAttributes};
use savant_core::protobuf::{from_pb, ToProtobuf};
use serde_json::Value;
use std::fmt::Debug;

use super::object::object_tree::VideoObjectTree;

#[pyclass]
pub struct ExternalFrame(pub(crate) rust::ExternalFrame);

#[pymethods]
impl ExternalFrame {
    #[new]
    #[pyo3(signature = (method, location=None))]
    pub fn new(method: &str, location: Option<String>) -> Self {
        Self(rust::ExternalFrame::new(method, &location.as_deref()))
    }

    #[getter]
    pub fn method(&self) -> String {
        self.0.method.clone()
    }

    #[getter]
    pub fn location(&self) -> Option<String> {
        self.0.location.clone()
    }

    #[setter]
    pub fn set_method(&mut self, method: String) {
        self.0.method = method;
    }

    #[setter]
    pub fn set_location(&mut self, location: Option<String>) {
        self.0.location = location;
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl ToSerdeJsonValue for ExternalFrame {
    fn to_serde_json_value(&self) -> Value {
        self.0.to_serde_json_value()
    }
}

/// Represents the structure for accessing primary video content for the frame.
///
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct VideoFrameContent(rust::VideoFrameContent);

#[pymethods]
impl VideoFrameContent {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[staticmethod]
    #[pyo3(signature = (method, location=None))]
    pub fn external(method: String, location: Option<String>) -> Self {
        Self(rust::VideoFrameContent::External(rust::ExternalFrame {
            method,
            location,
        }))
    }

    #[staticmethod]
    pub fn internal(data: &Bound<'_, PyBytes>) -> Self {
        let bytes = data.as_bytes();
        Self(rust::VideoFrameContent::Internal(bytes.to_vec()))
    }

    #[staticmethod]
    pub fn none() -> Self {
        Self(rust::VideoFrameContent::None)
    }

    pub fn is_external(&self) -> bool {
        matches!(&self.0, rust::VideoFrameContent::External(_))
    }

    pub fn is_internal(&self) -> bool {
        matches!(&self.0, rust::VideoFrameContent::Internal(_))
    }

    pub fn is_none(&self) -> bool {
        matches!(&self.0, rust::VideoFrameContent::None)
    }

    /// Returns the video data as a Python bytes object if the content is internal,
    /// otherwise results in the TypeError exception.
    ///
    /// Returns
    /// -------
    /// bytes
    ///   The video data as a Python bytes object.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///   If the content is not internal.
    ///
    pub fn get_data(&self) -> PyResult<Py<PyAny>> {
        match &self.0 {
            rust::VideoFrameContent::Internal(data) => {
                attach!(|py| {
                    let bytes = PyBytes::new_with(py, data.len(), |b: &mut [u8]| {
                        b.copy_from_slice(data);
                        Ok(())
                    })?;
                    Ok(Py::from(bytes))
                })
            }
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Video data is not stored internally",
            )),
        }
    }

    /// Returns the method for external video data if the content is external,
    /// otherwise results in the TypeError exception.
    ///
    /// Returns
    /// -------
    /// str
    ///   The method for external video data.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///   If the content is not external.
    ///
    pub fn get_method(&self) -> PyResult<String> {
        match &self.0 {
            rust::VideoFrameContent::External(data) => Ok(data.method.clone()),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Video data is not stored externally",
            )),
        }
    }

    /// Returns the location for external video data if the content is external,
    /// otherwise results in the TypeError exception.
    ///
    /// Returns
    /// -------
    /// str
    ///   The location for external video data.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///   If the content is not external.
    ///
    pub fn get_location(&self) -> PyResult<Option<String>> {
        match &self.0 {
            rust::VideoFrameContent::External(data) => Ok(data.location.clone()),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Video data is not stored externally",
            )),
        }
    }
}

/// Represents the structure for accessing primary video content encoding information
/// for the frame.
#[pyclass(from_py_object, eq, eq_int)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum VideoFrameTranscodingMethod {
    Copy,
    Encoded,
}

impl From<VideoFrameTranscodingMethod> for rust::VideoFrameTranscodingMethod {
    fn from(value: VideoFrameTranscodingMethod) -> Self {
        match value {
            VideoFrameTranscodingMethod::Copy => rust::VideoFrameTranscodingMethod::Copy,
            VideoFrameTranscodingMethod::Encoded => rust::VideoFrameTranscodingMethod::Encoded,
        }
    }
}

impl From<rust::VideoFrameTranscodingMethod> for VideoFrameTranscodingMethod {
    fn from(value: rust::VideoFrameTranscodingMethod) -> Self {
        match value {
            rust::VideoFrameTranscodingMethod::Copy => VideoFrameTranscodingMethod::Copy,
            rust::VideoFrameTranscodingMethod::Encoded => VideoFrameTranscodingMethod::Encoded,
        }
    }
}

impl ToSerdeJsonValue for VideoFrameTranscodingMethod {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(format!(
            "{:?}",
            rust::VideoFrameTranscodingMethod::from(*self)
        ))
    }
}

/// Video codec on a :class:`VideoFrame` (includes ``SwJpeg`` for software JPEG).
#[pyclass(from_py_object, eq, eq_int, frozen)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VideoFrameCodec {
    H264,
    Hevc,
    Jpeg,
    SwJpeg,
    Av1,
    Png,
    Vp8,
    Vp9,
    RawRgba,
    RawRgb,
    RawNv12,
}

impl From<VideoFrameCodec> for rust::VideoCodec {
    fn from(value: VideoFrameCodec) -> Self {
        match value {
            VideoFrameCodec::H264 => rust::VideoCodec::H264,
            VideoFrameCodec::Hevc => rust::VideoCodec::Hevc,
            VideoFrameCodec::Jpeg => rust::VideoCodec::Jpeg,
            VideoFrameCodec::SwJpeg => rust::VideoCodec::SwJpeg,
            VideoFrameCodec::Av1 => rust::VideoCodec::Av1,
            VideoFrameCodec::Png => rust::VideoCodec::Png,
            VideoFrameCodec::Vp8 => rust::VideoCodec::Vp8,
            VideoFrameCodec::Vp9 => rust::VideoCodec::Vp9,
            VideoFrameCodec::RawRgba => rust::VideoCodec::RawRgba,
            VideoFrameCodec::RawRgb => rust::VideoCodec::RawRgb,
            VideoFrameCodec::RawNv12 => rust::VideoCodec::RawNv12,
        }
    }
}

impl From<rust::VideoCodec> for VideoFrameCodec {
    fn from(value: rust::VideoCodec) -> Self {
        match value {
            rust::VideoCodec::H264 => VideoFrameCodec::H264,
            rust::VideoCodec::Hevc => VideoFrameCodec::Hevc,
            rust::VideoCodec::Jpeg => VideoFrameCodec::Jpeg,
            rust::VideoCodec::SwJpeg => VideoFrameCodec::SwJpeg,
            rust::VideoCodec::Av1 => VideoFrameCodec::Av1,
            rust::VideoCodec::Png => VideoFrameCodec::Png,
            rust::VideoCodec::Vp8 => VideoFrameCodec::Vp8,
            rust::VideoCodec::Vp9 => VideoFrameCodec::Vp9,
            rust::VideoCodec::RawRgba => VideoFrameCodec::RawRgba,
            rust::VideoCodec::RawRgb => VideoFrameCodec::RawRgb,
            rust::VideoCodec::RawNv12 => VideoFrameCodec::RawNv12,
        }
    }
}

#[pymethods]
impl VideoFrameCodec {
    #[staticmethod]
    pub fn from_name(name: &str) -> PyResult<Self> {
        rust::VideoCodec::from_name(name.trim())
            .map(Into::into)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown video frame codec: '{name}'")))
    }

    /// Canonical wire name (e.g. ``\"hevc\"``, ``\"swjpeg\"``).
    pub fn name(&self) -> &'static str {
        rust::VideoCodec::from(*self).name()
    }
}

/// Represents the structure for accessing/defining video frame transformation information.
///
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct VideoFrameTransformation(rust::VideoFrameTransformation);

#[pymethods]
impl VideoFrameTransformation {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Creates an ``InitialSize`` transformation recording original frame
    /// dimensions.
    ///
    /// Parameters
    /// ----------
    /// width : int
    ///   The width of the frame.
    /// height : int
    ///   The height of the frame.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the width or height is less than or equal to 0.
    ///
    #[staticmethod]
    pub fn initial_size(width: i64, height: i64) -> PyResult<Self> {
        if width <= 0 || height <= 0 {
            return Err(PyValueError::new_err(format!(
                "Width and height must be greater than 0, got {:?}x{:?}",
                width, height,
            )));
        }
        Ok(Self(rust::VideoFrameTransformation::InitialSize(
            u64::try_from(width).unwrap(),
            u64::try_from(height).unwrap(),
        )))
    }

    /// Creates a ``LetterBox`` transformation: scales the image to fit inside
    /// ``(outer_width - padding_left - padding_right) × (outer_height -
    /// padding_top - padding_bottom)`` and then pads it to
    /// ``outer_width × outer_height``.
    ///
    /// Parameters
    /// ----------
    /// outer_width : int
    ///   The total width after letterboxing.
    /// outer_height : int
    ///   The total height after letterboxing.
    /// padding_left : int
    ///   Inner padding on the left.
    /// padding_top : int
    ///   Inner padding on the top.
    /// padding_right : int
    ///   Inner padding on the right.
    /// padding_bottom : int
    ///   Inner padding on the bottom.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If any dimension is invalid.
    ///
    #[staticmethod]
    pub fn letter_box(
        outer_width: i64,
        outer_height: i64,
        padding_left: i64,
        padding_top: i64,
        padding_right: i64,
        padding_bottom: i64,
    ) -> PyResult<Self> {
        if outer_width <= 0 || outer_height <= 0 {
            return Err(PyValueError::new_err(format!(
                "Outer width and height must be > 0, got {outer_width}x{outer_height}",
            )));
        }
        if padding_left < 0 || padding_top < 0 || padding_right < 0 || padding_bottom < 0 {
            return Err(PyValueError::new_err(format!(
                "Padding values must be >= 0, got {padding_left},{padding_top},{padding_right},{padding_bottom}",
            )));
        }
        let inner_w = outer_width - padding_left - padding_right;
        let inner_h = outer_height - padding_top - padding_bottom;
        if inner_w <= 0 || inner_h <= 0 {
            return Err(PyValueError::new_err(format!(
                "Inner dimensions must be > 0, got {inner_w}x{inner_h}",
            )));
        }
        Ok(Self(rust::VideoFrameTransformation::LetterBox(
            u64::try_from(outer_width).unwrap(),
            u64::try_from(outer_height).unwrap(),
            u64::try_from(padding_left).unwrap(),
            u64::try_from(padding_top).unwrap(),
            u64::try_from(padding_right).unwrap(),
            u64::try_from(padding_bottom).unwrap(),
        )))
    }

    /// Creates a ``Padding`` transformation that adds border pixels.
    ///
    /// Parameters
    /// ----------
    /// left : int
    ///   The left padding.
    /// top : int
    ///   The top padding.
    /// right : int
    ///   The right padding.
    /// bottom : int
    ///   The bottom padding.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If any padding value is negative.
    ///
    #[staticmethod]
    pub fn padding(left: i64, top: i64, right: i64, bottom: i64) -> PyResult<Self> {
        if left < 0 || top < 0 || right < 0 || bottom < 0 {
            return Err(PyValueError::new_err(format!(
                "Padding must be >= 0, got {left},{top},{right},{bottom}",
            )));
        }
        Ok(Self(rust::VideoFrameTransformation::Padding(
            u64::try_from(left).unwrap(),
            u64::try_from(top).unwrap(),
            u64::try_from(right).unwrap(),
            u64::try_from(bottom).unwrap(),
        )))
    }

    /// Creates a ``Crop`` transformation that removes border pixels.
    ///
    /// Parameters
    /// ----------
    /// left : int
    ///   Pixels to remove from the left.
    /// top : int
    ///   Pixels to remove from the top.
    /// right : int
    ///   Pixels to remove from the right.
    /// bottom : int
    ///   Pixels to remove from the bottom.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If any value is negative.
    ///
    #[staticmethod]
    pub fn crop(left: i64, top: i64, right: i64, bottom: i64) -> PyResult<Self> {
        if left < 0 || top < 0 || right < 0 || bottom < 0 {
            return Err(PyValueError::new_err(format!(
                "Crop values must be >= 0, got {left},{top},{right},{bottom}",
            )));
        }
        Ok(Self(rust::VideoFrameTransformation::Crop(
            u64::try_from(left).unwrap(),
            u64::try_from(top).unwrap(),
            u64::try_from(right).unwrap(),
            u64::try_from(bottom).unwrap(),
        )))
    }

    /// Returns true if the transformation is initial size.
    #[getter]
    pub fn is_initial_size(&self) -> bool {
        matches!(self.0, rust::VideoFrameTransformation::InitialSize(_, _))
    }

    /// Returns true if the transformation is a letterbox.
    #[getter]
    pub fn is_letter_box(&self) -> bool {
        matches!(
            self.0,
            rust::VideoFrameTransformation::LetterBox(_, _, _, _, _, _)
        )
    }

    /// Returns true if the transformation is padding.
    #[getter]
    pub fn is_padding(&self) -> bool {
        matches!(self.0, rust::VideoFrameTransformation::Padding(_, _, _, _))
    }

    /// Returns true if the transformation is a crop.
    #[getter]
    pub fn is_crop(&self) -> bool {
        matches!(self.0, rust::VideoFrameTransformation::Crop(_, _, _, _))
    }

    /// Returns ``(width, height)`` if this is an ``InitialSize``, else ``None``.
    #[getter]
    pub fn as_initial_size(&self) -> Option<(u64, u64)> {
        match &self.0 {
            rust::VideoFrameTransformation::InitialSize(w, h) => Some((*w, *h)),
            _ => None,
        }
    }

    /// Returns ``(outer_w, outer_h, pad_left, pad_top, pad_right, pad_bottom)``
    /// if this is a ``LetterBox``, else ``None``.
    #[getter]
    pub fn as_letter_box(&self) -> Option<(u64, u64, u64, u64, u64, u64)> {
        match &self.0 {
            rust::VideoFrameTransformation::LetterBox(ow, oh, pl, pt, pr, pb) => {
                Some((*ow, *oh, *pl, *pt, *pr, *pb))
            }
            _ => None,
        }
    }

    /// Returns ``(left, top, right, bottom)`` if this is a ``Padding``, else
    /// ``None``.
    #[getter]
    pub fn as_padding(&self) -> Option<(u64, u64, u64, u64)> {
        match &self.0 {
            rust::VideoFrameTransformation::Padding(l, t, r, b) => Some((*l, *t, *r, *b)),
            _ => None,
        }
    }

    /// Returns ``(left, top, right, bottom)`` if this is a ``Crop``, else
    /// ``None``.
    #[getter]
    pub fn as_crop(&self) -> Option<(u64, u64, u64, u64)> {
        match &self.0 {
            rust::VideoFrameTransformation::Crop(l, t, r, b) => Some((*l, *t, *r, *b)),
            _ => None,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct VideoFrame(pub rust::VideoFrameProxy);

impl ToSerdeJsonValue for VideoFrame {
    fn to_serde_json_value(&self) -> Value {
        self.0.to_serde_json_value()
    }
}

#[pymethods]
impl VideoFrame {
    /// Low-level: apply a list of bbox operations to every object in the
    /// frame.  Prefer :py:meth:`transform_backward` or
    /// :py:meth:`transform_forward`.
    ///
    /// Parameters
    /// ----------
    /// ops : List[:py:class:`savant_rs.primitives.VideoObjectBBoxTransformation`]
    ///   The list of transformation operations to apply.
    /// no_gil : bool
    ///   Whether to release the GIL while applying the transformations.
    ///
    #[pyo3(name = "transform_geometry")]
    #[pyo3(signature = (ops, no_gil=true))]
    fn transform_geometry_gil(&self, ops: Vec<VideoObjectBBoxTransformation>, no_gil: bool) {
        detach!(no_gil, || {
            let ops_ref = ops.iter().map(|op| op.0).collect();
            self.0.transform_geometry(&ops_ref);
        })
    }

    /// Map all object bounding boxes from the current (post-transform)
    /// coordinate space back to the original coordinate space recorded by
    /// ``VideoFrameTransformation.initial_size``.
    ///
    /// After the call the transformation chain is reset to a single
    /// ``initial_size`` entry and the frame's ``width``/``height`` are
    /// updated to the original dimensions.
    ///
    /// Parameters
    /// ----------
    /// no_gil : bool
    ///   Whether to release the GIL while transforming.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///   If the chain has no ``InitialSize`` entry or the affine is
    ///   degenerate.
    ///
    #[pyo3(name = "transform_backward")]
    #[pyo3(signature = (no_gil=true))]
    fn transform_backward_gil(&mut self, no_gil: bool) -> PyResult<()> {
        detach!(no_gil, || {
            self.0
                .transform_backward()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Map all object bounding boxes from the **initial** coordinate space
    /// through the transformation chain into the **target** (final/current)
    /// coordinate space.
    ///
    /// Objects are assumed to be defined in the ``InitialSize`` space.
    /// The forward affine built from the chain is applied directly.
    /// Target dimensions are determined by the chain itself.
    ///
    /// After the call the transformation chain is reset to a single
    /// ``initial_size(target_w, target_h)`` and the frame's
    /// ``width``/``height`` are updated.
    ///
    /// Parameters
    /// ----------
    /// no_gil : bool
    ///   Whether to release the GIL while transforming.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///   If the chain has no ``InitialSize`` entry or has no computable
    ///   current size.
    ///
    #[pyo3(name = "transform_forward")]
    #[pyo3(signature = (no_gil=true))]
    fn transform_forward_gil(&mut self, no_gil: bool) -> PyResult<()> {
        detach!(no_gil, || {
            self.0
                .transform_forward()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Allows getting raw pointer to a frame for 3rd-party integration.
    ///
    /// Returns
    /// -------
    /// int
    ///   The pointer to the frame.
    ///
    #[getter]
    pub fn memory_handle(&self) -> usize {
        self.0.memory_handle()
    }

    fn __hash__(&self) -> usize {
        self.memory_handle()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Creates a new video frame.
    ///
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(
        signature = (source_id, fps, width, height, content, transcoding_method=VideoFrameTranscodingMethod::Copy, codec=None, keyframe=None, time_base=(1, 1000000), pts=0, dts=None, duration=None)
    )]
    pub fn new(
        source_id: &str,
        fps: (i64, i64),
        width: i64,
        height: i64,
        content: VideoFrameContent,
        transcoding_method: VideoFrameTranscodingMethod,
        codec: Option<VideoFrameCodec>,
        keyframe: Option<bool>,
        time_base: (i64, i64),
        pts: i64,
        dts: Option<i64>,
        duration: Option<i64>,
    ) -> PyResult<Self> {
        Ok(VideoFrame(err_to_pyerr!(
            rust::VideoFrameProxy::new(
                source_id,
                fps,
                width,
                height,
                content.0,
                transcoding_method.into(),
                codec.map(Into::into),
                keyframe,
                time_base,
                pts,
                dts,
                duration,
            ),
            PySystemError
        )?))
    }

    /// Creates protocol message (:py:class:`savant_rs.utils.serialization.Message`) from the frame.
    ///
    /// Returns
    /// -------
    /// :py:class:`savant_rs.utils.serialization.Message`
    ///   The protocol message.
    ///
    pub fn to_message(&self) -> Message {
        Message::video_frame(self)
    }

    /// Returns the source ID for the frame.
    ///
    /// Returns
    /// -------
    /// str
    ///   The source ID for the frame.
    ///
    #[getter]
    pub fn get_source_id(&self) -> String {
        self.0.get_source_id()
    }

    #[setter]
    pub fn set_source_id(&mut self, source_id: &str) {
        self.0.set_source_id(source_id)
    }

    /// Returns stream time base for the frame.
    ///
    /// Returns
    /// -------
    /// Tuple[int, int]
    ///   The stream time base ``(numerator, denominator)`` for the frame.
    ///
    #[getter]
    pub fn get_time_base(&self) -> (i64, i64) {
        self.0.get_time_base()
    }

    /// Sets stream time base for the frame.
    ///
    /// Parameters
    /// ----------
    /// time_base : Tuple[int, int]
    ///   The stream time base ``(numerator, denominator)``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the time base value is invalid.
    ///
    #[setter]
    pub fn set_time_base(&mut self, time_base: (i64, i64)) -> PyResult<()> {
        Ok(err_to_pyerr!(
            self.0.set_time_base(time_base),
            PyValueError
        )?)
    }

    /// Returns frame PTS
    ///
    /// Returns
    /// -------
    /// int
    ///   The frame PTS
    ///
    #[getter]
    pub fn get_pts(&self) -> i64 {
        self.0.get_pts()
    }

    /// Sets frame PTS
    ///
    /// Parameters
    /// ----------
    /// pts : int
    ///   The frame PTS to set
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the PTS value is invalid.
    ///
    #[setter]
    pub fn set_pts(&mut self, pts: i64) -> PyResult<()> {
        Ok(err_to_pyerr!(self.0.set_pts(pts), PyValueError)?)
    }

    #[getter]
    pub fn get_uuid(&self) -> String {
        self.0.get_uuid_as_string()
    }

    #[getter]
    pub fn get_creation_timestamp_ns(&self) -> u128 {
        self.0.get_creation_timestamp_ns()
    }

    #[setter]
    pub fn set_creation_timestamp_ns(&mut self, timestamp: u128) {
        self.0.set_creation_timestamp_ns(timestamp)
    }

    #[getter]
    pub fn get_fps(&self) -> (i64, i64) {
        self.0.get_fps()
    }

    #[setter]
    pub fn set_fps(&mut self, fps: (i64, i64)) -> PyResult<()> {
        Ok(err_to_pyerr!(self.0.set_fps(fps), PyValueError)?)
    }

    #[getter]
    pub fn get_width(&self) -> i64 {
        self.0.get_width()
    }

    /// Sets frame width
    ///
    /// Parameters
    /// ----------
    /// width : int
    ///   The frame width to set
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the width value is invalid.
    ///
    #[setter]
    pub fn set_width(&mut self, width: i64) -> PyResult<()> {
        Ok(err_to_pyerr!(self.0.set_width(width), PyValueError)?)
    }

    #[getter]
    pub fn get_height(&self) -> i64 {
        self.0.get_height()
    }

    /// Sets frame height
    ///
    /// Parameters
    /// ----------
    /// height : int
    ///   The frame height to set
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the height value is invalid.
    ///
    #[setter]
    pub fn set_height(&mut self, height: i64) -> PyResult<()> {
        Ok(err_to_pyerr!(self.0.set_height(height), PyValueError)?)
    }

    #[getter]
    pub fn get_dts(&self) -> Option<i64> {
        self.0.get_dts()
    }

    /// Sets frame DTS
    ///
    /// Parameters
    /// ----------
    /// dts : int or None
    ///   The frame DTS to set
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the DTS value is invalid.
    ///
    #[setter]
    pub fn set_dts(&mut self, dts: Option<i64>) -> PyResult<()> {
        Ok(err_to_pyerr!(self.0.set_dts(dts), PyValueError)?)
    }

    #[getter]
    pub fn get_duration(&self) -> Option<i64> {
        self.0.get_duration()
    }

    /// Sets frame duration
    ///
    /// Parameters
    /// ----------
    /// duration : int or None
    ///   The frame duration to set
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the duration value is invalid.
    ///
    #[setter]
    pub fn set_duration(&mut self, duration: Option<i64>) -> PyResult<()> {
        Ok(err_to_pyerr!(self.0.set_duration(duration), PyValueError)?)
    }

    #[getter]
    pub fn get_transcoding_method(&self) -> VideoFrameTranscodingMethod {
        self.0.get_transcoding_method().into()
    }

    #[setter]
    pub fn set_transcoding_method(&mut self, transcoding_method: VideoFrameTranscodingMethod) {
        self.0.set_transcoding_method(transcoding_method.into())
    }

    #[getter]
    pub fn get_codec(&self) -> Option<VideoFrameCodec> {
        self.0.get_codec().map(Into::into)
    }

    #[setter]
    pub fn set_codec(&mut self, codec: Option<VideoFrameCodec>) {
        self.0.set_codec(codec.map(Into::into));
    }

    #[getter]
    pub fn get_keyframe(&self) -> Option<bool> {
        self.0.get_keyframe()
    }

    #[setter]
    pub fn set_keyframe(&mut self, keyframe: Option<bool>) {
        self.0.set_keyframe(keyframe)
    }

    #[getter]
    pub fn get_content(&self) -> VideoFrameContent {
        VideoFrameContent(self.0.get_content().as_ref().clone())
    }

    #[setter]
    pub fn set_content(&mut self, content: VideoFrameContent) {
        self.0.set_content(content.0)
    }

    /// Returns the previous frame sequence ID for the frame within the
    /// sender connection.
    ///
    /// Returns
    /// -------
    /// Optional[int]
    ///   The previous frame sequence ID for the frame.
    ///
    #[getter]
    pub fn get_previous_frame_seq_id(&self) -> Option<i64> {
        self.0.get_previous_frame_seq_id()
    }

    #[getter]
    pub fn get_previous_keyframe_uuid(&self) -> Option<String> {
        self.0.get_previous_keyframe_as_string()
    }

    #[getter]
    #[pyo3(name = "json")]
    pub fn json_gil(&self) -> PyResult<String> {
        detach!(true, || {
            serde_json::to_string(&self.to_serde_json_value())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    #[getter]
    #[pyo3(name = "json_pretty")]
    fn json_pretty_gil(&self) -> PyResult<String> {
        detach!(true, || {
            serde_json::to_string_pretty(&self.to_serde_json_value())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    /// Resets all transformation records for a frame
    ///
    pub fn clear_transformations(&mut self) {
        self.0.clear_transformations()
    }

    /// Adds transformation record for a frame
    ///
    /// Parameters
    /// ----------
    /// transformation : :py:class:`savant_rs.primitives.VideoFrameTransformation`
    ///
    pub fn add_transformation(&mut self, transformation: VideoFrameTransformation) {
        self.0.add_transformation(transformation.0)
    }

    /// Returns the list of transformations
    ///
    /// Returns
    /// -------
    /// List[:py:class:`savant_rs.primitives.VideoFrameTransformation`]
    ///   The list of transformations
    ///
    #[getter]
    pub fn get_transformations(&self) -> Vec<VideoFrameTransformation> {
        self.0
            .get_transformations()
            .into_iter()
            .map(VideoFrameTransformation)
            .collect()
    }

    /// Returns the list of attributes
    ///
    /// Returns
    /// -------
    /// List[Tuple[str, str]]
    ///   The list of attributes (namespace, name)
    ///
    #[getter]
    pub fn attributes(&self) -> Vec<(String, String)> {
        self.0.get_attributes()
    }

    /// Returns the parent chain for the object.
    ///
    /// Parameters
    /// ----------
    /// obj : :py:class:`savant_rs.primitives.BorrowedVideoObject`
    ///   The object to get the parent chain for.
    ///
    /// Returns
    /// -------
    /// List[int]
    ///   The parent chain for the object, from closer to farther.
    ///
    pub fn get_parent_chain(&self, obj: &BorrowedVideoObject) -> Vec<i64> {
        self.0.get_parent_chain(&obj.0)
    }

    /// Returns the attribute object by namespace and name
    ///
    /// Parameters
    /// ----------
    /// namespace : str
    ///   Attribute namespace.
    /// name : str
    ///   Attribute name.
    ///
    /// Returns
    /// -------
    /// :py:class:`savant_rs.primitives.Attribute`
    ///   The attribute object
    ///
    pub fn get_attribute(&self, namespace: &str, name: &str) -> Option<Attribute> {
        self.0.get_attribute(namespace, name).map(Attribute)
    }

    pub fn find_attributes_with_ns(&mut self, namespace: &str) -> Vec<(String, String)> {
        self.0.find_attributes_with_ns(namespace)
    }

    pub fn find_attributes_with_names(&mut self, names: Vec<String>) -> Vec<(String, String)> {
        let label_refs = names.iter().map(|v| v.as_ref()).collect::<Vec<&str>>();
        self.0.find_attributes_with_names(&label_refs)
    }
    pub fn find_attributes_with_hints(
        &mut self,
        hints: Vec<Option<String>>,
    ) -> Vec<(String, String)> {
        let hint_opts_refs = hints
            .iter()
            .map(|v| v.as_deref())
            .collect::<Vec<Option<&str>>>();
        let hint_refs = hint_opts_refs.iter().collect::<Vec<_>>();

        self.0.find_attributes_with_hints(&hint_refs)
    }

    /// Deletes the attribute object by namespace and name
    ///
    /// Parameters
    /// ----------
    /// namespace : str
    ///   Attribute namespace.
    /// name : str
    ///   Attribute name.
    ///
    /// Returns
    /// -------
    /// Optional[:py:class:`savant_rs.primitives.Attribute`]
    ///   The deleted attribute object
    ///
    pub fn delete_attribute(&mut self, namespace: &str, name: &str) -> Option<Attribute> {
        self.0.delete_attribute(namespace, name).map(Attribute)
    }

    pub fn clear_attributes(&mut self) {
        self.0.clear_attributes()
    }

    pub fn delete_attributes_with_ns(&mut self, namespace: &str) {
        self.0.delete_attributes_with_ns(namespace)
    }

    pub fn delete_attributes_with_names(&mut self, names: Vec<String>) {
        let label_refs = names.iter().map(|v| v.as_ref()).collect::<Vec<&str>>();
        self.0.delete_attributes_with_names(&label_refs)
    }

    pub fn delete_attributes_with_hints(&mut self, hints: Vec<Option<String>>) {
        let hint_opts_refs = hints
            .iter()
            .map(|v| v.as_deref())
            .collect::<Vec<Option<&str>>>();
        let hint_refs = hint_opts_refs.iter().collect::<Vec<_>>();

        self.0.delete_attributes_with_hints(&hint_refs)
    }

    /// Sets new attribute for the frame. If the attribute is already set, it is replaced.
    ///
    /// Parameters
    /// ----------
    /// attribute : :py:class:`Attribute`
    ///   The attribute to set.
    ///
    /// Returns
    /// -------
    /// :py:class:`Attribute`
    ///   The set attribute.
    ///
    pub fn set_attribute(&mut self, attribute: Attribute) -> Option<Attribute> {
        self.0.set_attribute(attribute.0).map(Attribute)
    }

    /// Sets multiple attributes for the frame.
    ///
    /// Parameters
    /// ----------
    /// attributes : List[:py:class:`Attribute`]
    ///   The attributes to set.
    ///
    pub fn set_attributes(&mut self, attributes: Vec<Attribute>) {
        for attribute in attributes {
            self.set_attribute(attribute);
        }
    }

    /// Sets new persistent attribute for the frame. If the attribute is already set, it is replaced.
    ///
    /// Parameters
    /// ----------
    /// namespace : str
    ///   Attribute namespace.
    /// name : str
    ///   Attribute name.
    /// hint : str or None
    ///   Attribute hint.
    /// is_hidden : bool
    ///   Attribute hidden flag.
    /// values : List[:py:class:`AttributeValue`] or None
    ///   Attribute values.
    ///
    #[pyo3(signature = (namespace, name, is_hidden = false, hint = None, values = vec![]))]
    pub fn set_persistent_attribute(
        &mut self,
        namespace: &str,
        name: &str,
        is_hidden: bool,
        hint: Option<String>,
        values: Option<Vec<AttributeValue>>,
    ) {
        let values = match values {
            Some(values) => values.into_iter().map(|v| v.0).collect::<Vec<_>>(),
            None => vec![],
        };
        let hint = hint.as_deref();
        self.0
            .set_persistent_attribute(namespace, name, &hint, is_hidden, values)
    }

    /// Sets new temporary attribute for the frame. If the attribute is already set, it is replaced.
    ///
    /// Parameters
    /// ----------
    /// namespace : str
    ///   Attribute namespace.
    /// name : str
    ///   Attribute name.
    /// hint : str or None
    ///   Attribute hint.
    /// is_hidden : bool
    ///   Attribute hidden flag.
    /// values : List[:py:class:`AttributeValue`] or None
    ///   Attribute values.
    ///
    #[pyo3(signature = (namespace, name, is_hidden = false, hint = None, values = vec![]))]
    pub fn set_temporary_attribute(
        &mut self,
        namespace: &str,
        name: &str,
        is_hidden: bool,
        hint: Option<String>,
        values: Option<Vec<AttributeValue>>,
    ) {
        let values = match values {
            Some(values) => values.into_iter().map(|v| v.0).collect::<Vec<_>>(),
            None => vec![],
        };
        let hint = hint.as_deref();
        self.0
            .set_temporary_attribute(namespace, name, &hint, is_hidden, values)
    }

    #[pyo3(name = "set_draw_label")]
    #[pyo3(signature = (q, draw_label, no_gil = false))]
    pub fn set_draw_label_gil(&self, q: &MatchQuery, draw_label: SetDrawLabelKind, no_gil: bool) {
        detach!(no_gil, || self.0.set_draw_label(&q.0, draw_label.0))
    }

    pub fn add_object(
        &self,
        o: VideoObject,
        policy: IdCollisionResolutionPolicy,
    ) -> PyResult<BorrowedVideoObject> {
        self.0
            .add_object(o.0, policy.into())
            .map(BorrowedVideoObject)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (namespace, label, parent_id=None, confidence=None, detection_box=None, track_id=None, track_box=None, attributes=None))]
    pub fn create_object(
        &self,
        namespace: &str,
        label: &str,
        parent_id: Option<i64>,
        confidence: Option<f32>,
        detection_box: Option<RBBox>,
        track_id: Option<num_bigint::BigInt>,
        track_box: Option<RBBox>,
        attributes: Option<Vec<Attribute>>,
    ) -> PyResult<BorrowedVideoObject> {
        let native_attributes = match attributes {
            None => vec![],
            Some(_) => attributes
                .unwrap()
                .into_iter()
                .map(|a| a.0)
                .collect::<Vec<_>>(),
        };

        if detection_box.is_none() {
            return Err(PyValueError::new_err(
                "Detection box must be specified for new objects",
            ));
        }

        let track_id = track_id
            .map(fit_i64)
            .transpose()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        self.0
            .create_object(
                namespace,
                label,
                parent_id,
                detection_box.unwrap().0,
                confidence,
                track_id,
                track_box.map(|b| b.0),
                native_attributes,
            )
            .map(BorrowedVideoObject)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn get_object(&self, id: i64) -> Option<BorrowedVideoObject> {
        self.0.get_object(id).map(BorrowedVideoObject)
    }

    pub fn get_all_objects(&self) -> VideoObjectsView {
        self.0.get_all_objects().into()
    }

    pub fn has_objects(&self) -> bool {
        self.0.has_objects()
    }

    #[pyo3(name = "access_objects")]
    #[pyo3(signature = (q, no_gil = true))]
    pub fn access_objects_gil(&self, q: &MatchQuery, no_gil: bool) -> VideoObjectsView {
        detach!(no_gil, || VideoObjectsView::from(
            self.0.access_objects(&q.0)
        ))
    }

    pub fn access_objects_with_ids(&self, ids: Vec<i64>) -> VideoObjectsView {
        self.0.access_objects_with_id(&ids).into()
    }

    #[pyo3(name = "delete_objects")]
    #[pyo3(signature = (q, no_gil = true))]
    pub fn delete_objects_gil(&self, q: &MatchQuery, no_gil: bool) -> Vec<VideoObject> {
        detach!(no_gil, || self
            .0
            .delete_objects(&q.0)
            .into_iter()
            .map(VideoObject)
            .collect())
    }

    pub fn delete_objects_with_ids(&self, ids: Vec<i64>) -> Vec<VideoObject> {
        self.0
            .delete_objects_with_ids(&ids)
            .into_iter()
            .map(VideoObject)
            .collect()
    }

    /// Export complete object trees for objects matching the query.
    /// If delete_exported is true, the exported objects will be deleted from the frame.
    ///
    /// Parameters
    /// ----------
    /// q : :py:class:`savant_rs.match_query.MatchQuery`
    ///   The query to match the objects to export.
    /// delete_exported : bool
    ///   If true, the exported objects will be deleted from the frame.
    ///
    /// Returns
    /// -------
    /// List[:py:class:`savant_rs.primitives.VideoObjectTree`]
    ///   A vector of exported object trees. Those trees can be imported back to the frame by using
    ///   :py:meth:`savant_rs.primitives.VideoFrame.import_object_trees`. Object trees can be walked
    ///   by using :py:meth:`savant_rs.primitives.VideoObjectTree.walk_objects`.
    ///
    /// A tree can be safely imported in any other frame. Objects receive new identifiers and never
    /// refer to the objects in the frame as their parents.
    ///
    /// Raises
    /// ------
    /// PyRuntimeError
    ///   If the export fails.
    ///
    pub fn export_complete_object_trees(
        &self,
        q: &MatchQuery,
        delete_exported: bool,
    ) -> PyResult<Vec<VideoObjectTree>> {
        detach!(true, || {
            Ok(err_to_pyerr!(
                self.0.export_complete_object_trees(&q.0, delete_exported),
                PyRuntimeError
            )?
            .into_iter()
            .map(VideoObjectTree)
            .collect())
        })
    }

    /// Import object trees into the frame.
    ///
    /// Parameters
    /// ----------
    /// trees : List[:py:class:`savant_rs.primitives.VideoObjectTree`]
    ///   The object trees to import.
    ///
    /// Returns
    /// -------
    /// None
    ///
    /// Raises
    /// ------
    /// PyRuntimeError
    ///   If the import fails.
    ///
    pub fn import_object_trees(&self, trees: Vec<VideoObjectTree>) -> PyResult<()> {
        let trees = trees.into_iter().map(|t| t.0).collect::<Vec<_>>();
        detach!(true, || {
            Ok(err_to_pyerr!(
                self.0.import_object_trees(trees),
                PyRuntimeError
            )?)
        })
    }

    #[pyo3(name = "set_parent")]
    #[pyo3(signature = (q, parent, no_gil = true))]
    pub fn set_parent_gil(
        &self,
        q: &MatchQuery,
        parent: &BorrowedVideoObject,
        no_gil: bool,
    ) -> PyResult<VideoObjectsView> {
        let fun = || {
            self.0
                .set_parent(&q.0, &parent.0)
                .map(|o| o.into())
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Cannot set parent ID={} for objects matching query {:?}: {}",
                        parent.0.get_id(),
                        q,
                        e
                    ))
                })
        };
        detach!(no_gil, fun)
    }

    pub fn set_parent_by_id(&self, object_id: i64, parent_id: i64) -> PyResult<()> {
        self.0
            .set_parent_by_id(object_id, parent_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(name = "clear_parent")]
    #[pyo3(signature = (q, no_gil = true))]
    pub fn clear_parent_gil(&self, q: &MatchQuery, no_gil: bool) -> VideoObjectsView {
        detach!(no_gil, || VideoObjectsView::from(self.0.clear_parent(&q.0)))
    }

    pub fn clear_objects(&self) {
        self.0.clear_objects()
    }

    pub fn get_children(&self, id: i64) -> VideoObjectsView {
        self.0.get_children(id).into()
    }

    #[pyo3(name = "copy")]
    #[pyo3(signature = (no_gil = true))]
    pub fn copy_gil(&self, no_gil: bool) -> VideoFrame {
        detach!(no_gil, || VideoFrame(self.0.smart_copy()))
    }

    /// Updates the frame with the given update. The function is GIL-free.
    ///
    /// The order of execution:
    /// - frame attributes are updated
    /// - existing objects are updated with attributes
    /// - new objects are added
    ///
    /// Parameters
    /// ----------
    /// update: :py:class:`savant_rs.primitives.VideoFrameUpdate`
    ///   The update to apply
    ///
    /// Returns
    /// -------
    /// None
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the update cannot be applied to the frame
    ///
    #[pyo3(name = "update")]
    #[pyo3(signature = (update, no_gil = true))]
    pub fn update_gil(&self, update: &VideoFrameUpdate, no_gil: bool) -> PyResult<()> {
        detach!(no_gil, || self.0.update(&update.0))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(name = "to_protobuf")]
    #[pyo3(signature = (no_gil = true))]
    fn to_protobuf_gil(&self, no_gil: bool) -> PyResult<Py<PyAny>> {
        let bytes = detach!(no_gil, || {
            self.0.to_pb().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to serialize video frame to protobuf: {e}"))
            })
        })?;
        attach!(|py| {
            let bytes = PyBytes::new_with(py, bytes.len(), |b: &mut [u8]| {
                b.copy_from_slice(&bytes);
                Ok(())
            })?;
            Ok(Py::from(bytes))
        })
    }

    #[staticmethod]
    #[pyo3(name = "from_protobuf")]
    #[pyo3(signature = (bytes, no_gil = true))]
    fn from_protobuf_gil(bytes: &Bound<'_, PyBytes>, no_gil: bool) -> PyResult<Self> {
        let bytes = bytes.as_bytes();
        detach!(no_gil, || {
            let obj = from_pb::<savant_core::protobuf::VideoFrame, rust::VideoFrameProxy>(bytes)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Failed to deserialize video frame from protobuf: {e}"
                    ))
                })?;
            Ok(Self(obj))
        })
    }
}

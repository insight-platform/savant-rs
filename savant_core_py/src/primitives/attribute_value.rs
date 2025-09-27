use crate::attach;
use crate::primitives::segment::Intersection;
use crate::primitives::{BBox, Point, PolygonalArea, RBBox};
use pyo3::exceptions::{PyIndexError, PySystemError, PyValueError};
use pyo3::types::{PyBytes, PyBytesMethods};
use pyo3::{pyclass, pymethods, Bound, Py, PyAny, PyResult};
use savant_core::primitives::any_object::AnyObject;
use savant_core::primitives::attribute_value::AttributeValueVariant;
use savant_core::primitives::rust;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::mem;
use std::sync::Arc;

#[pyclass]
#[derive(Debug, Clone)]
pub struct AttributeValue(pub rust::AttributeValue);

#[cfg(test)]
impl AttributeValue {
    pub fn get_value(&self) -> AttributeValueVariant {
        self.0.value.clone()
    }

    pub fn set_value(&mut self, value: AttributeValueVariant) {
        self.0.value = value;
    }
}

#[pymethods]
impl AttributeValue {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[getter]
    fn get_confidence(&self) -> Option<f32> {
        self.0.confidence
    }

    #[setter]
    fn set_confidence(&mut self, confidence: Option<f32>) {
        self.0.confidence = confidence;
    }

    /// Returns the confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValueType`
    ///    The type of the attribute value.
    ///
    #[getter]
    fn get_value_type(&self) -> AttributeValueType {
        match &self.0.value {
            AttributeValueVariant::Bytes(_, _) => AttributeValueType::Bytes,
            AttributeValueVariant::String(_) => AttributeValueType::String,
            AttributeValueVariant::StringVector(_) => AttributeValueType::StringList,
            AttributeValueVariant::Integer(_) => AttributeValueType::Integer,
            AttributeValueVariant::IntegerVector(_) => AttributeValueType::IntegerList,
            AttributeValueVariant::Float(_) => AttributeValueType::Float,
            AttributeValueVariant::FloatVector(_) => AttributeValueType::FloatList,
            AttributeValueVariant::Boolean(_) => AttributeValueType::Boolean,
            AttributeValueVariant::BooleanVector(_) => AttributeValueType::BooleanList,
            AttributeValueVariant::BBox(_) => AttributeValueType::BBox,
            AttributeValueVariant::BBoxVector(_) => AttributeValueType::BBoxList,
            AttributeValueVariant::Point(_) => AttributeValueType::Point,
            AttributeValueVariant::PointVector(_) => AttributeValueType::PointList,
            AttributeValueVariant::Polygon(_) => AttributeValueType::Polygon,
            AttributeValueVariant::PolygonVector(_) => AttributeValueType::PolygonList,
            AttributeValueVariant::Intersection(_) => AttributeValueType::Intersection,
            AttributeValueVariant::None => AttributeValueType::None_,
            AttributeValueVariant::TemporaryValue(_) => AttributeValueType::TemporaryValue,
        }
    }

    /// Creates a new attribute value of type :class:`savant_rs.primitives.geometry.Intersection`.
    ///
    /// Parameters
    /// ----------
    /// int : :class:`savant_rs.primitives.geometry.Intersection`
    ///   The intersection value.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (int, confidence = None))]
    pub fn intersection(int: Intersection, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::Intersection(int.0),
        })
    }

    /// Creates a new attribute value of type None
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    pub fn none() -> Self {
        Self(rust::AttributeValue {
            confidence: None,
            value: AttributeValueVariant::None,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (pyobj, confidence = None))]
    pub fn temporary_python_object(pyobj: Py<PyAny>, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::TemporaryValue(AnyObject::new(Box::new(pyobj))),
        })
    }

    /// Creates a new attribute value of blob type.
    ///
    /// Parameters
    /// ----------
    /// dims : list of int
    ///   The dimensions of the blob.
    /// blob : List[int]
    ///   The blob.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   
    #[staticmethod]
    #[pyo3(signature = (dims, blob, confidence = None))]
    pub fn bytes_from_list(dims: Vec<i64>, blob: Vec<u8>, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::Bytes(dims, blob),
        })
    }

    /// Creates a new attribute value of blob type.
    ///
    /// Parameters
    /// ----------
    /// dims : list of int
    ///   The dimensions of the blob.
    /// blob : bytes
    ///   The blob.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   
    #[staticmethod]
    #[pyo3(signature = (dims, blob, confidence = None))]
    pub fn bytes(dims: Vec<i64>, blob: &Bound<'_, PyBytes>, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::Bytes(dims, blob.as_bytes().to_vec()),
        })
    }

    /// Creates a new attribute value of string type.
    ///
    /// Parameters
    /// ----------
    /// s : str
    ///   The string.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (s, confidence = None))]
    pub fn string(s: String, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::String(s),
        })
    }

    /// Creates a new attribute value of list of strings type.
    ///
    /// Parameters
    /// ----------
    /// ss : List[str]
    ///   The list of strings.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (ss, confidence = None))]
    pub fn strings(ss: Vec<String>, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::StringVector(ss),
        })
    }

    /// Creates a new attribute value of integer type.
    ///
    /// Parameters
    /// ----------
    /// i : int
    ///   The integer value.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (i, confidence = None))]
    pub fn integer(i: i64, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::Integer(i),
        })
    }

    /// Creates a new attribute value of list of integers type.
    ///
    /// Parameters
    /// ----------
    /// ii : List[int]
    ///   The list of integers.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (ii, confidence = None))]
    pub fn integers(ii: Vec<i64>, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::IntegerVector(ii),
        })
    }

    /// Creates a new attribute value of float type.
    ///
    /// Parameters
    /// ----------
    /// f : float
    ///   The float value.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (f, confidence = None))]
    pub fn float(f: f64, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::Float(f),
        })
    }

    /// Creates a new attribute value of list of floats type.
    ///
    /// Parameters
    /// ----------
    /// ff : List[float]
    ///   The list of floats.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (ff, confidence = None))]
    pub fn floats(ff: Vec<f64>, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::FloatVector(ff),
        })
    }

    /// Creates a new attribute value of boolean type.
    ///
    /// Parameters
    /// ----------
    /// b : bool
    ///   The boolean value.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (b, confidence = None))]
    pub fn boolean(b: bool, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::Boolean(b),
        })
    }

    /// Creates a new attribute value of list of booleans type.
    ///
    /// Parameters
    /// ----------
    /// bb : List[bool]
    ///   The list of booleans.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (bb, confidence = None))]
    pub fn booleans(bb: Vec<bool>, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::BooleanVector(bb),
        })
    }

    /// Creates a new attribute value of bounding box type.
    ///
    /// Parameters
    /// ----------
    /// bbox : :class:`savant_rs.primitives.geometry.BBox`
    ///   The bounding box.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (bbox, confidence = None))]
    pub fn bbox(bbox: BBox, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::BBox(bbox.0 .0.into()),
        })
    }

    /// Creates a new attribute value of rotated bounding box type.
    ///
    /// Parameters
    /// ----------
    /// rbbox : :class:`savant_rs.primitives.geometry.RBBox`
    ///   The rotated bounding box.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (rbbox, confidence = None))]
    pub fn rbbox(rbbox: RBBox, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::BBox(rbbox.0.into()),
        })
    }

    /// Creates a new attribute value of list of bounding boxes type.
    ///
    /// Parameters
    /// ----------
    /// bboxes : List[:class:`savant_rs.primitives.geometry.BBox`]
    ///   The list of bounding boxes.
    /// confidence : float, optional
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (bboxes, confidence = None))]
    pub fn rbboxes(bboxes: Vec<RBBox>, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::BBoxVector(
                bboxes.into_iter().map(|b| b.0.into()).collect(),
            ),
        })
    }

    /// Creates a new attribute value of list of bounding boxes type.
    ///
    /// Parameters
    /// ----------
    /// bboxes : List[:class:`savant_rs.primitives.geometry.RBBox`]
    ///   The list of rotated bounding boxes.
    /// confidence : float, optional
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (bboxes, confidence = None))]
    pub fn bboxes(bboxes: Vec<BBox>, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::BBoxVector(
                bboxes.into_iter().map(|b| b.0 .0.into()).collect(),
            ),
        })
    }

    /// Creates a new attribute value of point type.
    ///
    /// Parameters
    /// ----------
    /// point : :class:`savant_rs.primitives.geometry.Point`
    ///   The point.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (point, confidence = None))]
    pub fn point(point: Point, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::Point(point.0),
        })
    }

    /// Creates a new attribute value of list of points type.
    ///
    /// Parameters
    /// ----------
    /// points : List[:class:`savant_rs.primitives.geometry.Point`]
    ///   The list of points.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (points, confidence = None))]
    pub fn points(points: Vec<Point>, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::PointVector(unsafe {
                mem::transmute::<Vec<Point>, Vec<rust::Point>>(points)
            }),
        })
    }

    /// Creates a new attribute value of polygon type.
    ///
    /// Parameters
    /// ----------
    /// polygon : :class:`savant_rs.primitives.geometry.PolygonalArea`
    ///   The polygon.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (polygon, confidence = None))]
    pub fn polygon(polygon: PolygonalArea, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::Polygon(polygon.0),
        })
    }

    /// Creates a new attribute value of list of polygons type.
    ///
    /// Parameters
    /// ----------
    /// polygons : List[:class:`savant_rs.primitives.geometry.PolygonalArea`]
    ///   The list of polygons.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (polygons, confidence = None))]
    pub fn polygons(polygons: Vec<PolygonalArea>, confidence: Option<f32>) -> Self {
        Self(rust::AttributeValue {
            confidence,
            value: AttributeValueVariant::PolygonVector(unsafe {
                mem::transmute::<Vec<PolygonalArea>, Vec<rust::PolygonalArea>>(polygons)
            }),
        })
    }

    /// Checks if the attribute valus if of None type.
    ///
    pub fn is_none(&self) -> bool {
        matches!(&self.0.value, AttributeValueVariant::None)
    }

    /// Returns the value of attribute as ``(dims, bytes)`` tuple or None if not a bytes type.
    ///
    /// Returns
    /// -------
    /// Optional[Tuple[List[int], bytes]]
    ///   The value of attribute as ``(dims, bytes)`` tuple or None if not a bytes type.
    ///
    pub fn as_bytes(&self) -> Option<(Vec<i64>, Py<PyAny>)> {
        match &self.0.value {
            AttributeValueVariant::Bytes(dims, bytes) => Some((
                dims.clone(),
                attach!(|py| Py::from(PyBytes::new(py, bytes))),
            )),
            _ => None,
        }
    }

    /// Returns the value of attribute as an :class:`savant_rs.primitives.geometry.Intersection` or None if not an intersection type.
    ///
    /// Returns
    /// -------
    /// Optional[:class:`savant_rs.primitives.geometry.Intersection`]
    ///   The value of attribute as an :class:`savant_rs.primitives.geometry.Intersection` or None if not an intersection type.
    ///
    pub fn as_intersection(&self) -> Option<Intersection> {
        match &self.0.value {
            AttributeValueVariant::Intersection(i) => Some(Intersection(i.clone())),
            _ => None,
        }
    }

    /// Returns the value of attribute as a string or None if not a string type.
    ///
    /// Returns
    /// -------
    /// Optional[str]
    ///   The value of attribute as a string or None if not a string type.
    ///
    pub fn as_string(&self) -> Option<String> {
        match &self.0.value {
            AttributeValueVariant::String(s) => Some(s.clone()),
            _ => None,
        }
    }

    /// Returns the value of attribute as a list of strings or None if not a list of strings type.
    ///
    /// Returns
    /// -------
    /// Optional[List[str]]
    ///   The value of attribute as a list of strings or None if not a list of strings type.
    ///
    pub fn as_strings(&self) -> Option<Vec<String>> {
        match &self.0.value {
            AttributeValueVariant::StringVector(s) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn as_temporary_python_object(&self) -> Option<Py<PyAny>> {
        match &self.0.value {
            AttributeValueVariant::TemporaryValue(obj) => {
                let obj = obj.take()?;
                let obj = obj.downcast::<Py<PyAny>>().ok()?;
                Some(*obj)
            }
            _ => None,
        }
    }

    /// Returns the value of attribute as an integer or None if not an integer type.
    ///
    /// Returns
    /// -------
    /// Optional[int]
    ///   The value of attribute as an integer or None if not an integer type.
    ///
    pub fn as_integer(&self) -> Option<i64> {
        match &self.0.value {
            AttributeValueVariant::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the value of attribute as a list of integers or None if not a list of integers type.
    ///
    /// Returns
    /// -------
    /// Optional[List[int]]
    ///   The value of attribute as a list of integers or None if not a list of integers type.
    ///
    pub fn as_integers(&self) -> Option<Vec<i64>> {
        match &self.0.value {
            AttributeValueVariant::IntegerVector(i) => Some(i.clone()),
            _ => None,
        }
    }

    /// Returns the value of attribute as a float or None if not a float type.
    ///
    /// Returns
    /// -------
    /// Optional[float]
    ///   The value of attribute as a float or None if not a float type.
    ///
    pub fn as_float(&self) -> Option<f64> {
        match &self.0.value {
            AttributeValueVariant::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns the value of attribute as a list of floats or None if not a list of floats type.
    ///
    /// Returns
    /// -------
    /// Optional[List[float]]
    ///   The value of attribute as a list of floats or None if not a list of floats type.
    ///
    pub fn as_floats(&self) -> Option<Vec<f64>> {
        match &self.0.value {
            AttributeValueVariant::FloatVector(f) => Some(f.clone()),
            _ => None,
        }
    }

    /// Returns the value of attribute as a boolean or None if not a boolean type.
    ///
    /// Returns
    /// -------
    /// Optional[bool]
    ///   The value of attribute as a boolean or None if not a boolean type.
    ///
    pub fn as_boolean(&self) -> Option<bool> {
        match &self.0.value {
            AttributeValueVariant::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns the value of attribute as a list of booleans or None if not a list of booleans type.
    ///
    /// Returns
    /// -------
    /// Optional[List[bool]]
    ///   The value of attribute as a list of booleans or None if not a list of booleans type.
    ///
    pub fn as_booleans(&self) -> Option<Vec<bool>> {
        match &self.0.value {
            AttributeValueVariant::BooleanVector(b) => Some(b.clone()),
            _ => None,
        }
    }

    /// This method always raises an error because there is a chance that it is a RBBox. Use as_rbbox instead.
    ///
    /// Returns
    /// -------
    /// PyResult[Option[:class:`savant_rs.primitives.geometry.BBox`]]
    ///   The value of attribute as a :class:`savant_rs.primitives.geometry.BBox` or None if not a bounding box type.
    ///
    pub fn as_bbox(&self) -> PyResult<Option<BBox>> {
        Err(PySystemError::new_err(
            "Not implemented because there is a chance that it is a RBBox. Use as_rbbox instead.",
        ))
    }

    /// Returns the value of attribute as a :class:`savant_rs.primitives.geometry.RBBox` or None if not a bounding box type.
    ///
    /// Returns
    /// -------
    /// Optional[:class:`savant_rs.primitives.geometry.RBBox`]
    ///   The value of attribute as a :class:`savant_rs.primitives.geometry.RBBox` or None if not a bounding box type.
    ///
    pub fn as_rbbox(&self) -> Option<RBBox> {
        match &self.0.value {
            AttributeValueVariant::BBox(bbox) => Some(RBBox(rust::RBBox::from(bbox.clone()))),
            _ => None,
        }
    }

    /// This method always raises an error because there is a chance that it is a list of RBBox-es. Use as_rbboxes instead.
    ///
    /// Returns
    /// -------
    /// PyResult[Option[List[:class:`savant_rs.primitives.geometry.BBox`]]]
    ///   The value of attribute as a list of :class:`savant_rs.primitives.geometry.BBox` or None if not a list of bounding boxes type.
    ///
    pub fn as_bboxes(&self) -> PyResult<Option<Vec<BBox>>> {
        Err(PySystemError::new_err(
            "Not implemented because there is a chance that it is a list of RBBox-es. Use as_rbboxes instead."
        ))
    }

    /// Returns the value of attribute as a list of :class:`savant_rs.primitives.geometry.RBBox` or None if not a list of bounding boxes type.
    ///
    /// Returns
    /// -------
    /// Optional[List[:class:`savant_rs.primitives.geometry.RBBox`]]
    ///   The value of attribute as a list of :class:`savant_rs.primitives.geometry.RBBox` or None if not a list of bounding boxes type.
    ///
    pub fn as_rbboxes(&self) -> Option<Vec<RBBox>> {
        match &self.0.value {
            AttributeValueVariant::BBoxVector(bboxes) => Some(
                bboxes
                    .iter()
                    .map(|bbox| RBBox(rust::RBBox::from(bbox.clone())))
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Returns the value of attribute as a :class:`savant_rs.primitives.geometry.Point` or None if not a point type.
    ///
    /// Returns
    /// -------
    /// Optional[:class:`savant_rs.primitives.geometry.Point`]
    ///   The value of attribute as a :class:`savant_rs.primitives.geometry.Point` or None if not a point type.
    ///
    pub fn as_point(&self) -> Option<Point> {
        match &self.0.value {
            AttributeValueVariant::Point(point) => Some(Point(point.clone())),
            _ => None,
        }
    }

    /// Returns the value of attribute as a list of :class:`savant_rs.primitives.geometry.Point` or None if not a list of points type.
    ///
    /// Returns
    /// -------
    /// Optional[List[:class:`savant_rs.primitives.geometry.Point`]]
    ///   The value of attribute as a list of :class:`savant_rs.primitives.geometry.Point` or None if not a list of points type.
    ///
    pub fn as_points(&self) -> Option<Vec<Point>> {
        match &self.0.value {
            AttributeValueVariant::PointVector(points) => {
                Some(unsafe { mem::transmute::<Vec<rust::Point>, Vec<Point>>(points.clone()) })
            }
            _ => None,
        }
    }

    /// Returns the value of attribute as a :class:`savant_rs.primitives.geometry.PolygonalArea` or None if not a polygon type.
    ///
    /// Returns
    /// -------
    /// Optional[:class:`savant_rs.primitives.geometry.PolygonalArea`]
    ///   The value of attribute as a :class:`savant_rs.primitives.geometry.PolygonalArea` or None if not a polygon type.
    ///
    pub fn as_polygon(&self) -> Option<PolygonalArea> {
        match &self.0.value {
            AttributeValueVariant::Polygon(polygon) => Some(PolygonalArea(polygon.clone())),
            _ => None,
        }
    }

    /// Returns the value of attribute as a list of :class:`savant_rs.primitives.geometry.PolygonalArea` or None if not a list of polygons type.
    ///
    /// Returns
    /// -------
    /// Optional[List[:class:`savant_rs.primitives.geometry.PolygonalArea`]]
    ///   The value of attribute as a list of :class:`savant_rs.primitives.geometry.PolygonalArea` or None if not a list of polygons type.
    ///
    pub fn as_polygons(&self) -> Option<Vec<PolygonalArea>> {
        match &self.0.value {
            AttributeValueVariant::PolygonVector(polygons) => Some(unsafe {
                mem::transmute::<Vec<rust::PolygonalArea>, Vec<PolygonalArea>>(polygons.clone())
            }),
            _ => None,
        }
    }

    #[getter]
    pub fn json(&self) -> PyResult<String> {
        let res = self
            .0
            .to_json()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(res)
    }

    #[staticmethod]
    pub fn from_json(json: &str) -> PyResult<Self> {
        let res = rust::AttributeValue::from_json(json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self(res))
    }
}

/// Represents attribute value types for matching
///
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Hash, PartialEq)]
pub enum AttributeValueType {
    Bytes,
    String,
    StringList,
    Integer,
    IntegerList,
    Float,
    FloatList,
    Boolean,
    BooleanList,
    BBox,
    BBoxList,
    RBBox,
    RBBoxList,
    Point,
    PointList,
    Polygon,
    PolygonList,
    Intersection,
    TemporaryValue,
    None_,
}

#[pymethods]
impl AttributeValueType {
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Helper class allowing access attribute values without copying them from the object to a requesting party. The class suits well if you
/// work with compound values and want to check value partially before accessing costly operations. It supports Python's ``len(obj)`` and ``obj[i]``
/// operations, but only on reading.
///
#[pyclass]
#[derive(Debug, Clone)]
pub struct AttributeValuesView(pub Arc<Vec<rust::AttributeValue>>);

#[pymethods]
impl AttributeValuesView {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __getitem__(&self, index: usize) -> PyResult<AttributeValue> {
        let v = self
            .0
            .get(index)
            .ok_or(PyIndexError::new_err("index out of range"))
            .cloned()?;
        Ok(AttributeValue(v))
    }

    #[getter]
    fn memory_handle(&self) -> usize {
        self as *const Self as usize
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.0.len())
    }
}

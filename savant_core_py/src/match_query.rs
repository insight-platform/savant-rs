use crate::primitives::bbox::{BBoxMetricType, RBBox};
use crate::primitives::objects_view::QueryFunctions;
use crate::utils::eval_resolvers::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use savant_core::match_query as rust;

/**
Module for defining queries on video objects.

JMES Query Syntax can be found here: `JMESPath <https://jmespath.org/>`__.

 */

/// A class allowing to define a float expression
///
#[pyclass]
#[derive(Debug, Clone)]
pub struct FloatExpression(rust::FloatExpression);

#[pymethods]
impl FloatExpression {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Eq expression
    ///
    /// In JSON/YAML: eq
    ///
    /// Parameters
    /// ----------
    /// v: float
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`FloatExpression`
    ///   Float expression
    ///
    #[staticmethod]
    fn eq(v: f32) -> FloatExpression {
        FloatExpression(rust::FloatExpression::EQ(v))
    }

    /// Ne expression
    ///
    /// In JSON/YAML: ne
    ///
    /// Parameters
    /// ----------
    /// v: float
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`FloatExpression`
    ///   Float expression
    ///
    #[staticmethod]
    fn ne(v: f32) -> FloatExpression {
        FloatExpression(rust::FloatExpression::NE(v))
    }

    /// Lt expression
    ///
    /// In JSON/YAML: lt
    ///
    /// Parameters
    /// ----------
    /// v: float
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`FloatExpression`
    ///   Float expression
    ///
    #[staticmethod]
    fn lt(v: f32) -> FloatExpression {
        FloatExpression(rust::FloatExpression::LT(v))
    }

    /// Le expression
    ///
    /// In JSON/YAML: le
    ///
    /// Parameters
    /// ----------
    /// v: float
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`FloatExpression`
    ///   Float expression
    ///
    #[staticmethod]
    fn le(v: f32) -> FloatExpression {
        FloatExpression(rust::FloatExpression::LE(v))
    }

    /// Gt expression
    ///
    /// In JSON/YAML: gt
    ///
    /// Parameters
    /// ----------
    /// v: float
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`FloatExpression`
    ///   Float expression
    ///
    #[staticmethod]
    fn gt(v: f32) -> FloatExpression {
        FloatExpression(rust::FloatExpression::GT(v))
    }

    /// Ge expression
    ///
    /// In JSON/YAML: ge
    ///
    /// Parameters
    /// ----------
    /// v: float
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`FloatExpression`
    ///   Float expression
    ///
    #[staticmethod]
    fn ge(v: f32) -> FloatExpression {
        FloatExpression(rust::FloatExpression::GE(v))
    }

    /// Between expression
    ///
    /// In JSON/YAML: between
    ///
    /// Parameters
    /// ----------
    /// a: float
    ///   Lower bound
    /// b: float
    ///   Upper bound
    ///
    /// Returns
    /// -------
    /// :py:class:`FloatExpression`
    ///   Float expression
    ///
    #[staticmethod]
    fn between(a: f32, b: f32) -> FloatExpression {
        FloatExpression(rust::FloatExpression::Between(a, b))
    }

    /// One of expression
    ///
    /// In JSON/YAML: one_of
    ///
    ///  
    ///
    /// Parameters
    /// ----------
    /// \*list: \*list of float
    ///   List of values to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`FloatExpression`
    ///   Float expression
    ///
    #[staticmethod]
    #[pyo3(signature = (*list))]
    fn one_of(list: &PyTuple) -> FloatExpression {
        let mut vals = Vec::with_capacity(list.len());
        for arg in list {
            let v = arg
                .extract::<f32>()
                .expect("Invalid argument. Only f32 values are allowed.");
            vals.push(v);
        }
        FloatExpression(rust::FloatExpression::OneOf(vals))
    }
}

/// A class allowing to define an integer expression
///
#[pyclass]
#[derive(Debug, Clone)]
pub struct IntExpression(rust::IntExpression);

#[pymethods]
impl IntExpression {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Eq expression
    ///
    /// In JSON/YAML: eq
    ///
    /// Parameters
    /// ----------
    /// v: int
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`IntExpression`
    ///   Int expression
    ///
    #[staticmethod]
    fn eq(v: i64) -> IntExpression {
        IntExpression(rust::IntExpression::EQ(v))
    }

    /// Ne expression
    ///
    /// In JSON/YAML: ne
    ///
    /// Parameters
    /// ----------
    /// v: int
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`IntExpression`
    ///   Int expression
    ///
    #[staticmethod]
    fn ne(v: i64) -> IntExpression {
        IntExpression(rust::IntExpression::NE(v))
    }

    /// Lt expression
    ///
    /// In JSON/YAML: lt
    ///
    /// Parameters
    /// ----------
    /// v: int
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`IntExpression`
    ///   Int expression
    ///
    #[staticmethod]
    fn lt(v: i64) -> IntExpression {
        IntExpression(rust::IntExpression::LT(v))
    }

    /// Le expression
    ///
    /// In JSON/YAML: le
    ///
    /// Parameters
    /// ----------
    /// v: int
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`IntExpression`
    ///   Int expression
    ///
    #[staticmethod]
    fn le(v: i64) -> IntExpression {
        IntExpression(rust::IntExpression::LE(v))
    }

    /// Gt expression
    ///
    /// In JSON/YAML: gt
    ///
    /// Parameters
    /// ----------
    /// v: int
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`IntExpression`
    ///   Int expression
    ///
    #[staticmethod]
    fn gt(v: i64) -> IntExpression {
        IntExpression(rust::IntExpression::GT(v))
    }

    /// Ge expression
    ///
    /// In JSON/YAML: ge
    ///
    /// Parameters
    /// ----------
    /// v: int
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`IntExpression`
    ///   Int expression
    ///
    #[staticmethod]
    fn ge(v: i64) -> IntExpression {
        IntExpression(rust::IntExpression::GE(v))
    }

    /// Between expression
    ///
    /// In JSON/YAML: between
    ///
    /// Parameters
    /// ----------
    /// a: int
    ///   Lower bound
    /// b: int
    ///   Upper bound
    ///
    /// Returns
    /// -------
    /// :py:class:`IntExpression`
    ///   Int expression
    ///
    #[staticmethod]
    fn between(a: i64, b: i64) -> IntExpression {
        IntExpression(rust::IntExpression::Between(a, b))
    }

    /// One of expression
    ///
    /// In JSON/YAML: one_of
    ///
    /// Parameters
    /// ----------
    /// \*list: \*list of int
    ///   List of values to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`IntExpression`
    ///   Int expression
    ///
    #[staticmethod]
    #[pyo3(signature = (*list))]
    fn one_of(list: &PyTuple) -> IntExpression {
        let mut vals = Vec::with_capacity(list.len());
        for arg in list {
            let v = arg
                .extract::<i64>()
                .expect("Invalid argument. Only i64 values are allowed.");
            vals.push(v);
        }
        IntExpression(rust::IntExpression::OneOf(vals))
    }
}

/// A class allowing to define a string expression
///
#[pyclass]
#[derive(Debug, Clone)]
pub struct StringExpression(rust::StringExpression);

#[pymethods]
impl StringExpression {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Eq expression
    ///
    /// In JSON/YAML: eq
    ///
    /// Parameters
    /// ----------
    /// v: str
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`StringExpression`
    ///   String expression
    ///
    #[staticmethod]
    fn eq(v: String) -> StringExpression {
        StringExpression(rust::StringExpression::EQ(v))
    }

    /// Ne expression
    ///
    /// In JSON/YAML: ne
    ///
    /// Parameters
    /// ----------
    /// v: str
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`StringExpression`
    ///   String expression
    ///
    #[staticmethod]
    fn ne(v: String) -> StringExpression {
        StringExpression(rust::StringExpression::NE(v))
    }

    /// Contains expression
    ///
    /// In JSON/YAML: contains
    ///
    /// Parameters
    /// ----------
    /// v: str
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`StringExpression`
    ///   String expression
    ///
    #[staticmethod]
    fn contains(v: String) -> StringExpression {
        StringExpression(rust::StringExpression::Contains(v))
    }

    /// Not contains expression
    ///
    /// In JSON/YAML: not_contains
    ///
    /// Parameters
    /// ----------
    /// v: str
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`StringExpression`
    ///   String expression
    ///
    #[staticmethod]
    fn not_contains(v: String) -> StringExpression {
        StringExpression(rust::StringExpression::NotContains(v))
    }

    /// Starts with expression
    ///
    /// In JSON/YAML: starts_with
    ///
    /// Parameters
    /// ----------
    /// v: str
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`StringExpression`
    ///   String expression
    ///
    #[staticmethod]
    fn starts_with(v: String) -> StringExpression {
        StringExpression(rust::StringExpression::StartsWith(v))
    }

    /// Ends with expression
    ///
    /// In JSON/YAML: ends_with
    ///
    /// Parameters
    /// ----------
    /// v: str
    ///   Value to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`StringExpression`
    ///   String expression
    ///
    #[staticmethod]
    fn ends_with(v: String) -> StringExpression {
        StringExpression(rust::StringExpression::EndsWith(v))
    }

    /// One of expression
    ///
    /// In JSON/YAML: one_of
    ///
    /// Parameters
    /// ----------
    /// \*list: \*list of str
    ///   List of values to compare with
    ///
    /// Returns
    /// -------
    /// :py:class:`StringExpression`
    ///   String expression
    ///
    #[staticmethod]
    #[pyo3(signature = (*list))]
    fn one_of(list: &PyTuple) -> StringExpression {
        let mut vals = Vec::with_capacity(list.len());
        for arg in list {
            let v = arg
                .extract::<String>()
                .expect("Invalid argument. Only String values are allowed.");
            vals.push(v);
        }
        StringExpression(rust::StringExpression::OneOf(vals))
    }
}

/// A class allowing to define a Query based on expressions
///
#[pyclass]
#[derive(Debug, Clone)]
pub struct MatchQuery(pub(crate) rust::MatchQuery);

#[pymethods]
impl MatchQuery {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// And predicate
    ///
    /// In JSON/YAML: and
    ///
    /// Parameters
    /// ----------
    /// \*list: \*list of :py:class:`Query`
    ///
    /// Returns
    /// -------
    /// :py:class:`Query`
    ///   Query
    ///
    #[staticmethod]
    #[pyo3(signature = (*list))]
    fn and_(list: &PyTuple) -> MatchQuery {
        let mut v = Vec::with_capacity(list.len());
        for arg in list {
            let q = arg
                .extract::<MatchQuery>()
                .expect("Invalid argument. Only Query values are allowed.");
            v.push(q.0.clone());
        }
        MatchQuery(rust::MatchQuery::And(v))
    }

    /// Or predicate
    ///
    /// In JSON/YAML: or
    ///
    /// Parameters
    /// ----------
    /// \*list: \*list of :py:class:`Query`
    ///
    /// Returns
    /// -------
    /// :py:class:`Query`
    ///   Query
    ///
    #[staticmethod]
    #[pyo3(signature = (*list))]
    fn or_(list: &PyTuple) -> MatchQuery {
        let mut v = Vec::with_capacity(list.len());
        for arg in list {
            let q = arg
                .extract::<MatchQuery>()
                .expect("Invalid argument. Only Query values are allowed.");
            v.push(q.0.clone());
        }
        MatchQuery(rust::MatchQuery::Or(v))
    }

    /// Not predicate
    ///
    /// In JSON/YAML: not
    ///
    /// Parameters
    /// ----------
    /// a: :py:class:`Query`
    ///   Query
    ///
    /// Returns
    /// -------
    /// :py:class:`Query`
    ///   Query
    ///
    #[staticmethod]
    fn not_(a: &MatchQuery) -> MatchQuery {
        MatchQuery(rust::MatchQuery::Not(Box::new(a.0.clone())))
    }

    /// Stop searching If False predicate (short-circuit)
    ///
    /// In JSON/YAML: stop_if_false
    ///
    /// Parameters
    /// ----------
    /// a: :py:class:`Query`
    ///  Query
    ///
    /// Returns
    /// -------
    /// :py:class:`Query`
    ///   Query
    ///
    #[staticmethod]
    fn stop_if_false(a: &MatchQuery) -> MatchQuery {
        MatchQuery(rust::MatchQuery::StopIfFalse(Box::new(a.0.clone())))
    }

    /// Stop searching If True predicate (short-circuit)
    ///
    /// In JSON/YAML: stop_if_true
    ///
    /// Parameters
    /// ----------
    /// a: :py:class:`Query`
    ///  Query
    ///
    /// Returns
    /// -------
    /// :py:class:`Query`
    ///   Query
    ///
    #[staticmethod]
    fn stop_if_true(a: &MatchQuery) -> MatchQuery {
        MatchQuery(rust::MatchQuery::StopIfTrue(Box::new(a.0.clone())))
    }

    /// True if query executed on children objects of an object returns a number of results
    /// matching the given integer expression.
    ///
    /// E.g. If a person has at least one hand visible.
    ///
    /// In JSON/YAML: with_children
    ///
    /// Parameters
    /// ----------
    /// a: :py:class:`Query`
    ///   Query to run on children objects to get the number of matching results
    /// n: :py:class:`IntExpression`
    ///   Integer expression to compare the number retrieved for children with
    ///
    /// Returns
    /// -------
    /// :py:class:`Query`
    ///   Query
    ///
    #[staticmethod]
    fn with_children(a: MatchQuery, n: IntExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::WithChildren(Box::new(a.0.clone()), n.0))
    }

    /// True, when expression defined by evalexpr is computed.
    ///
    /// In JSON/YAML: eval
    ///
    /// Parameters
    /// ----------
    /// exp: str
    ///   Expression language format
    /// resolvers: List[str]
    ///   Resolvers enabled for evaluation
    ///
    /// Returns
    /// -------
    /// :py:class:`Query`
    ///   Query
    ///
    #[staticmethod]
    fn eval(exp: String) -> MatchQuery {
        MatchQuery(rust::MatchQuery::EvalExpr(exp))
    }

    /// True if object's Id matches the given integer expression.
    ///
    /// In JSON/YAML: object.id
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`IntExpression`
    ///   Integer expression to compare the object's Id with
    ///
    /// Returns
    /// -------
    /// :py:class:`Query`
    ///   Query
    ///
    #[staticmethod]
    fn id(e: IntExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::Id(e.0))
    }

    #[staticmethod]
    fn box_metric(bbox: &RBBox, metric_type: BBoxMetricType, e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::BoxMetric {
            other: (
                bbox.get_xc(),
                bbox.get_yc(),
                bbox.get_width(),
                bbox.get_height(),
                bbox.get_angle(),
            ),
            metric_type: metric_type.into(),
            threshold_expr: e.0,
        })
    }

    #[staticmethod]
    fn track_box_metric(
        bbox: &RBBox,
        metric_type: BBoxMetricType,
        e: FloatExpression,
    ) -> MatchQuery {
        MatchQuery(rust::MatchQuery::TrackBoxMetric {
            other: (
                bbox.get_xc(),
                bbox.get_yc(),
                bbox.get_width(),
                bbox.get_height(),
                bbox.get_angle(),
            ),
            metric_type: metric_type.into(),
            threshold_expr: e.0,
        })
    }

    /// True if object's namespace matches the given string expression.
    ///
    /// In JSON/YAML: namespace
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`StringExpression`
    ///   String expression to compare the object's namespace with
    ///
    /// Returns
    /// -------
    /// :py:class:`Query`
    ///   Query
    ///
    #[staticmethod]
    fn namespace(e: StringExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::Namespace(e.0))
    }

    /// True if object's label matches the given string expression.
    ///
    /// In JSON/YAML: label
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`StringExpression`
    ///   String expression to compare the object's label with
    ///
    /// Returns
    /// -------
    /// :py:class:`Query`
    ///   Query
    ///
    #[staticmethod]
    fn label(e: StringExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::Label(e.0))
    }

    /// True if object's confidence matches the given float expression.
    ///
    /// In JSON/YAML: confidence
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`FloatExpression`
    ///   Float expression to compare the object's confidence with
    ///
    /// Returns
    /// -------
    /// :py:class:`Query`
    ///   Query
    ///
    #[staticmethod]
    fn confidence(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::Confidence(e.0))
    }

    /// True if object's track_id matches the given int expression.
    ///
    /// In JSON/YAML: track.id
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`IntExpression`
    ///   Integer expression to compare the object's track_id with
    ///
    /// Returns
    /// -------
    /// :py:class:`Query`
    ///   Query
    ///
    #[staticmethod]
    fn track_id(e: IntExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::TrackId(e.0))
    }

    /// True if object's track bbox xc matches the given float expression.
    ///
    /// In JSON/YAML: track.bbox.xc
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`FloatExpression`
    ///   Float expression to compare the object's track bbox xc with
    ///
    /// Returns
    /// -------
    /// :py:class:`Query`
    ///   Query
    ///
    #[staticmethod]
    fn track_box_x_center(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::TrackBoxXCenter(e.0))
    }

    /// True if object's track bbox yc matches the given float expression.
    ///
    /// In JSON/YAML: track.bbox.yc
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`FloatExpression`
    ///   Float expression to compare the object's track bbox yc with
    ///
    #[staticmethod]
    fn track_box_y_center(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::TrackBoxYCenter(e.0))
    }

    /// True if object's track bbox width matches the given float expression.
    ///
    /// In JSON/YAML: track.bbox.width
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`FloatExpression`
    ///   Float expression to compare the object's track bbox width with
    ///
    #[staticmethod]
    fn track_box_width(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::TrackBoxWidth(e.0))
    }

    /// True if object's track bbox height matches the given float expression.
    ///
    /// In JSON/YAML: track.bbox.height
    ///
    #[staticmethod]
    fn track_box_height(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::TrackBoxHeight(e.0))
    }

    /// True if object's track bbox area (width x height) matches the given float expression.
    ///
    /// In JSON/YAML: track.bbox.area
    ///
    #[staticmethod]
    fn track_box_area(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::TrackBoxArea(e.0))
    }

    /// True if object's track bbox aspect ratio (width / height) matches the given float expression.
    ///
    /// In JSON/YAML: track.bbox.width_to_height_ratio
    ///
    #[staticmethod]
    fn track_box_width_to_height_ratio(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::TrackBoxWidthToHeightRatio(e.0))
    }

    /// True if object's track bbox angle matches the given float expression.
    ///
    /// In JSON/YAML: track.bbox.angle
    ///
    #[staticmethod]
    fn track_box_angle(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::TrackBoxAngle(e.0))
    }

    /// True if an UDF predicate executed on object returned true.
    ///
    /// In JSON/YAML: user_defined_object_predicate
    ///
    /// Parameters
    /// ----------
    /// plugin: str
    ///   Name of the plugin to execute
    /// function: str
    ///   Name of the function to execute
    ///
    #[staticmethod]
    fn user_defined_rust_plugin_object_predicate(plugin: String, function: String) -> MatchQuery {
        MatchQuery(rust::MatchQuery::UserDefinedObjectPredicate(
            plugin, function,
        ))
    }

    /// True if object's parent id matches the given int expression.
    ///
    /// In JSON/YAML: parent.id
    ///
    #[staticmethod]
    fn parent_id(e: IntExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::ParentId(e.0))
    }

    /// True if object's parent namespace matches the given string expression.
    ///
    /// In JSON/YAML: parent.namespace
    ///
    #[staticmethod]
    fn parent_namespace(e: StringExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::ParentNamespace(e.0))
    }

    /// True if object's parent label matches the given string expression.
    ///
    /// In JSON/YAML: parent.label
    ///
    #[staticmethod]
    fn parent_label(e: StringExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::ParentLabel(e.0))
    }

    /// True if object's box xc matches the given float expression.
    ///
    /// In JSON/YAML: bbox.xc
    ///
    #[staticmethod]
    fn box_x_center(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::BoxXCenter(e.0))
    }

    /// True if object's box yc matches the given float expression.
    ///
    /// In JSON/YAML: bbox.yc
    ///
    #[staticmethod]
    fn box_y_center(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::BoxYCenter(e.0))
    }

    /// True if object's box width matches the given float expression.
    ///
    /// In JSON/YAML: bbox.width
    ///
    #[staticmethod]
    fn box_width(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::BoxWidth(e.0))
    }

    /// True if object's box height matches the given float expression.
    ///
    /// In JSON/YAML: bbox.height
    ///
    #[staticmethod]
    fn box_height(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::BoxHeight(e.0))
    }

    /// True if object's box area (width x height) matches the given float expression.
    ///
    /// In JSON/YAML: bbox.area
    ///
    #[staticmethod]
    fn box_area(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::BoxArea(e.0))
    }

    /// True if object's box aspect ratio (width / height) matches the given float expression.
    ///
    /// In JSON/YAML: bbox.width_to_height_ratio
    ///
    #[staticmethod]
    fn box_width_to_height_ratio(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::BoxWidthToHeightRatio(e.0))
    }

    /// True if object's box angle matches the given float expression.
    ///
    /// In JSON/YAML: bbox.angle
    ///
    #[staticmethod]
    fn box_angle(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::BoxAngle(e.0))
    }

    /// Always true
    ///
    #[staticmethod]
    pub fn idle() -> MatchQuery {
        MatchQuery(rust::MatchQuery::Idle)
    }

    /// True if JMES Query executed on attributes converted in JSON format returns True.
    ///
    /// In JSON/YAML: attributes.jmes_query
    ///
    #[staticmethod]
    fn attributes_jmes_query(e: String) -> MatchQuery {
        MatchQuery(rust::MatchQuery::AttributesJMESQuery(e))
    }

    /// True if object's parent is defined.
    ///
    /// In JSON/YAML: parent.defined
    #[staticmethod]
    fn parent_defined() -> MatchQuery {
        MatchQuery(rust::MatchQuery::ParentDefined)
    }

    /// True if object's confidence is defined.
    ///
    /// In JSON/YAML: confidence.defined
    ///
    #[staticmethod]
    fn confidence_defined() -> MatchQuery {
        MatchQuery(rust::MatchQuery::ConfidenceDefined)
    }

    /// True if object's track id is defined.
    ///
    /// In JSON/YAML: track.id.defined
    ///
    #[staticmethod]
    fn track_id_defined() -> MatchQuery {
        MatchQuery(rust::MatchQuery::TrackDefined)
    }

    /// True if object's box has angle defined.
    ///
    /// In JSON/YAML: bbox.angle.defined
    ///
    #[staticmethod]
    fn box_angle_defined() -> MatchQuery {
        MatchQuery(rust::MatchQuery::BoxAngleDefined)
    }

    /// True if object doesn't have attributes
    ///
    #[staticmethod]
    fn attributes_empty() -> MatchQuery {
        MatchQuery(rust::MatchQuery::AttributesEmpty)
    }

    #[staticmethod]
    fn attribute_defined(namespace: String, label: String) -> MatchQuery {
        MatchQuery(rust::MatchQuery::AttributeExists(namespace, label))
    }

    #[staticmethod]
    fn frame_source_id(e: StringExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameSourceId(e.0))
    }

    #[staticmethod]
    fn frame_is_key_frame() -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameIsKeyFrame)
    }

    #[staticmethod]
    fn frame_width(e: IntExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameWidth(e.0))
    }

    #[staticmethod]
    fn frame_height(e: IntExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameHeight(e.0))
    }

    /// When the frame does not have associated video, because of sparsity, for example
    ///
    #[staticmethod]
    fn frame_no_video() -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameNoVideo)
    }

    /// When the processing is configured in pass-through mode
    ///
    #[staticmethod]
    fn frame_transcoding_is_copy() -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameTranscodingIsCopy)
    }

    #[staticmethod]
    fn frame_attribute_exists(namespace: String, label: String) -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameAttributeExists(namespace, label))
    }

    #[staticmethod]
    fn frame_attributes_empty() -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameAttributesEmpty)
    }

    #[staticmethod]
    fn frame_attributes_jmes_query(e: String) -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameAttributesJMESQuery(e))
    }

    /// Dumps query to JSON string.
    ///
    #[getter]
    fn json(&self) -> String {
        self.0.to_json()
    }

    /// Dumps query to pretty JSON string.
    ///
    #[getter]
    fn json_pretty(&self) -> String {
        self.0.to_json_pretty()
    }

    /// Dumps query to YAML string.
    ///
    #[getter]
    fn yaml(&self) -> String {
        self.0.to_yaml()
    }

    /// Loads query from JSON string.
    ///
    #[staticmethod]
    fn from_json(json: String) -> MatchQuery {
        MatchQuery(
            rust::MatchQuery::from_json(&json)
                .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {}", e)))
                .unwrap(),
        )
    }

    /// Loads query from YAML string.
    ///
    #[staticmethod]
    fn from_yaml(yaml: String) -> MatchQuery {
        MatchQuery(
            rust::MatchQuery::from_yaml(&yaml)
                .map_err(|e| PyValueError::new_err(format!("Invalid YAML: {}", e)))
                .unwrap(),
        )
    }
}

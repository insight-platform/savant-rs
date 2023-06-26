use crate::primitives::bbox::BBoxMetricType;
use crate::primitives::message::video::object::objects_view::QueryFunctions;
use crate::primitives::message::video::query::match_query::{
    FloatExpression, IntExpression, MatchQuery, StringExpression,
};
use crate::primitives::message::video::query::MatchQuery::{BoxMetric, TrackBoxMetric};
use crate::primitives::RBBox;
use crate::utils::eval_resolvers::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::ops::Deref;
use std::sync::Arc;

/**
Module for defining queries on video objects.

JMES Query Syntax can be found here: `JMESPath <https://jmespath.org/>`__.

 */
#[pymodule]
pub fn video_object_query(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FloatExpressionProxy>()?;
    m.add_class::<IntExpressionProxy>()?;
    m.add_class::<StringExpressionProxy>()?;
    m.add_class::<MatchQueryProxy>()?;
    m.add_class::<QueryFunctions>()?;

    m.add_function(wrap_pyfunction!(utility_resolver_name, m)?)?;
    m.add_function(wrap_pyfunction!(etcd_resolver_name, m)?)?;
    m.add_function(wrap_pyfunction!(env_resolver_name, m)?)?;
    m.add_function(wrap_pyfunction!(config_resolver_name, m)?)?;

    m.add_function(wrap_pyfunction!(register_utility_resolver, m)?)?;
    m.add_function(wrap_pyfunction!(register_env_resolver, m)?)?;
    m.add_function(wrap_pyfunction!(register_etcd_resolver, m)?)?;
    m.add_function(wrap_pyfunction!(register_config_resolver, m)?)?;
    m.add_function(wrap_pyfunction!(update_config_resolver, m)?)?;

    m.add_function(wrap_pyfunction!(unregister_resolver, m)?)?;
    Ok(())
}

/// A class allowing to define a float expression
///
#[pyclass]
#[derive(Debug, Clone)]
#[pyo3(name = "FloatExpression")]
pub struct FloatExpressionProxy {
    inner: FloatExpression,
}

#[pymethods]
impl FloatExpressionProxy {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:#?}", self)
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
    fn eq(v: f64) -> FloatExpressionProxy {
        FloatExpressionProxy {
            inner: FloatExpression::EQ(v),
        }
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
    fn ne(v: f64) -> FloatExpressionProxy {
        FloatExpressionProxy {
            inner: FloatExpression::NE(v),
        }
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
    fn lt(v: f64) -> FloatExpressionProxy {
        FloatExpressionProxy {
            inner: FloatExpression::LT(v),
        }
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
    fn le(v: f64) -> FloatExpressionProxy {
        FloatExpressionProxy {
            inner: FloatExpression::LE(v),
        }
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
    fn gt(v: f64) -> FloatExpressionProxy {
        FloatExpressionProxy {
            inner: FloatExpression::GT(v),
        }
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
    fn ge(v: f64) -> FloatExpressionProxy {
        FloatExpressionProxy {
            inner: FloatExpression::GE(v),
        }
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
    fn between(a: f64, b: f64) -> FloatExpressionProxy {
        FloatExpressionProxy {
            inner: FloatExpression::Between(a, b),
        }
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
    fn one_of(list: &PyTuple) -> FloatExpressionProxy {
        let mut vals = Vec::with_capacity(list.len());
        for arg in list {
            let v = arg
                .extract::<f64>()
                .expect("Invalid argument. Only f64 values are allowed.");
            vals.push(v);
        }
        FloatExpressionProxy {
            inner: FloatExpression::OneOf(vals),
        }
    }
}

/// A class allowing to define an integer expression
///
#[pyclass]
#[derive(Debug, Clone)]
#[pyo3(name = "IntExpression")]
pub struct IntExpressionProxy {
    inner: IntExpression,
}

#[pymethods]
impl IntExpressionProxy {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:#?}", self)
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
    fn eq(v: i64) -> IntExpressionProxy {
        IntExpressionProxy {
            inner: IntExpression::EQ(v),
        }
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
    fn ne(v: i64) -> IntExpressionProxy {
        IntExpressionProxy {
            inner: IntExpression::NE(v),
        }
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
    fn lt(v: i64) -> IntExpressionProxy {
        IntExpressionProxy {
            inner: IntExpression::LT(v),
        }
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
    fn le(v: i64) -> IntExpressionProxy {
        IntExpressionProxy {
            inner: IntExpression::LE(v),
        }
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
    fn gt(v: i64) -> IntExpressionProxy {
        IntExpressionProxy {
            inner: IntExpression::GT(v),
        }
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
    fn ge(v: i64) -> IntExpressionProxy {
        IntExpressionProxy {
            inner: IntExpression::GE(v),
        }
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
    fn between(a: i64, b: i64) -> IntExpressionProxy {
        IntExpressionProxy {
            inner: IntExpression::Between(a, b),
        }
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
    fn one_of(list: &PyTuple) -> IntExpressionProxy {
        let mut vals = Vec::with_capacity(list.len());
        for arg in list {
            let v = arg
                .extract::<i64>()
                .expect("Invalid argument. Only i64 values are allowed.");
            vals.push(v);
        }
        IntExpressionProxy {
            inner: IntExpression::OneOf(vals),
        }
    }
}

/// A class allowing to define a string expression
///
#[pyclass]
#[derive(Debug, Clone)]
#[pyo3(name = "StringExpression")]
pub struct StringExpressionProxy {
    inner: StringExpression,
}

#[pymethods]
impl StringExpressionProxy {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:#?}", self)
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
    fn eq(v: String) -> StringExpressionProxy {
        StringExpressionProxy {
            inner: StringExpression::EQ(v),
        }
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
    fn ne(v: String) -> StringExpressionProxy {
        StringExpressionProxy {
            inner: StringExpression::NE(v),
        }
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
    fn contains(v: String) -> StringExpressionProxy {
        StringExpressionProxy {
            inner: StringExpression::Contains(v),
        }
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
    fn not_contains(v: String) -> StringExpressionProxy {
        StringExpressionProxy {
            inner: StringExpression::NotContains(v),
        }
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
    fn starts_with(v: String) -> StringExpressionProxy {
        StringExpressionProxy {
            inner: StringExpression::StartsWith(v),
        }
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
    fn ends_with(v: String) -> StringExpressionProxy {
        StringExpressionProxy {
            inner: StringExpression::EndsWith(v),
        }
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
    fn one_of(list: &PyTuple) -> StringExpressionProxy {
        let mut vals = Vec::with_capacity(list.len());
        for arg in list {
            let v = arg
                .extract::<String>()
                .expect("Invalid argument. Only String values are allowed.");
            vals.push(v);
        }
        StringExpressionProxy {
            inner: StringExpression::OneOf(vals),
        }
    }
}

/// A class allowing to define a Query based on expressions
///
#[pyclass]
#[pyo3(name = "MatchQuery")]
#[derive(Debug, Clone)]
pub struct MatchQueryProxy {
    pub inner: Arc<MatchQuery>,
}

#[pymethods]
impl MatchQueryProxy {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:#?}", self.inner.deref())
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
    fn and_(list: &PyTuple) -> MatchQueryProxy {
        let mut v = Vec::with_capacity(list.len());
        for arg in list {
            let q = arg
                .extract::<MatchQueryProxy>()
                .expect("Invalid argument. Only Query values are allowed.");
            v.push(q.inner.deref().clone());
        }
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::And(v)),
        }
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
    fn or_(list: &PyTuple) -> MatchQueryProxy {
        let mut v = Vec::with_capacity(list.len());
        for arg in list {
            let q = arg
                .extract::<MatchQueryProxy>()
                .expect("Invalid argument. Only Query values are allowed.");
            v.push(q.inner.deref().clone());
        }
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::Or(v)),
        }
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
    fn not_(a: MatchQueryProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::Not(Box::new(a.inner.deref().clone()))),
        }
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
    fn with_children(a: MatchQueryProxy, n: IntExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::WithChildren(
                Box::new(a.inner.deref().clone()),
                n.inner,
            )),
        }
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
    fn eval(exp: String) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::EvalExpr(exp)),
        }
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
    fn id(e: IntExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::Id(e.inner)),
        }
    }

    #[staticmethod]
    fn box_metric(
        bbox: &RBBox,
        metric_type: BBoxMetricType,
        e: FloatExpressionProxy,
    ) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(BoxMetric {
                other: (
                    bbox.get_xc(),
                    bbox.get_yc(),
                    bbox.get_width(),
                    bbox.get_height(),
                    bbox.get_angle(),
                ),
                metric_type,
                threshold_expr: e.inner,
            }),
        }
    }

    #[staticmethod]
    fn track_box_metric(
        bbox: &RBBox,
        metric_type: BBoxMetricType,
        e: FloatExpressionProxy,
    ) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(TrackBoxMetric {
                other: (
                    bbox.get_xc(),
                    bbox.get_yc(),
                    bbox.get_width(),
                    bbox.get_height(),
                    bbox.get_angle(),
                ),
                metric_type,
                threshold_expr: e.inner,
            }),
        }
    }

    /// True if object's creator matches the given string expression.
    ///
    /// In JSON/YAML: creator
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`StringExpression`
    ///   String expression to compare the object's creator with
    ///
    /// Returns
    /// -------
    /// :py:class:`Query`
    ///   Query
    ///
    #[staticmethod]
    fn creator(e: StringExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::Creator(e.inner)),
        }
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
    fn label(e: StringExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::Label(e.inner)),
        }
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
    fn confidence(e: FloatExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::Confidence(e.inner)),
        }
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
    fn track_id(e: IntExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::TrackId(e.inner)),
        }
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
    fn track_box_x_center(e: FloatExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::TrackBoxXCenter(e.inner)),
        }
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
    fn track_box_y_center(e: FloatExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::TrackBoxYCenter(e.inner)),
        }
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
    fn track_box_width(e: FloatExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::TrackBoxWidth(e.inner)),
        }
    }

    /// True if object's track bbox height matches the given float expression.
    ///
    /// In JSON/YAML: track.bbox.height
    ///
    #[staticmethod]
    fn track_box_height(e: FloatExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::TrackBoxHeight(e.inner)),
        }
    }

    /// True if object's track bbox area (width x height) matches the given float expression.
    ///
    /// In JSON/YAML: track.bbox.area
    ///
    #[staticmethod]
    fn track_box_area(e: FloatExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::TrackBoxArea(e.inner)),
        }
    }

    /// True if object's track bbox aspect ratio (width / height) matches the given float expression.
    ///
    /// In JSON/YAML: track.bbox.width_to_height_ratio
    ///
    #[staticmethod]
    fn track_box_width_to_height_ratio(e: FloatExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::TrackBoxWidthToHeightRatio(e.inner)),
        }
    }

    /// True if object's track bbox angle matches the given float expression.
    ///
    /// In JSON/YAML: track.bbox.angle
    ///
    #[staticmethod]
    fn track_box_angle(e: FloatExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::TrackBoxAngle(e.inner)),
        }
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
    fn user_defined_rust_plugin_object_predicate(
        plugin: String,
        function: String,
    ) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::UserDefinedObjectPredicate(plugin, function)),
        }
    }

    /// True if object's parent id matches the given int expression.
    ///
    /// In JSON/YAML: parent.id
    ///
    #[staticmethod]
    fn parent_id(e: IntExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::ParentId(e.inner)),
        }
    }

    /// True if object's parent creator matches the given string expression.
    ///
    /// In JSON/YAML: parent.creator
    ///
    #[staticmethod]
    fn parent_creator(e: StringExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::ParentCreator(e.inner)),
        }
    }

    /// True if object's parent label matches the given string expression.
    ///
    /// In JSON/YAML: parent.label
    ///
    #[staticmethod]
    fn parent_label(e: StringExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::ParentLabel(e.inner)),
        }
    }

    /// True if object's box xc matches the given float expression.
    ///
    /// In JSON/YAML: bbox.xc
    ///
    #[staticmethod]
    fn box_x_center(e: FloatExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::BoxXCenter(e.inner)),
        }
    }

    /// True if object's box yc matches the given float expression.
    ///
    /// In JSON/YAML: bbox.yc
    ///
    #[staticmethod]
    fn box_y_center(e: FloatExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::BoxYCenter(e.inner)),
        }
    }

    /// True if object's box width matches the given float expression.
    ///
    /// In JSON/YAML: bbox.width
    ///
    #[staticmethod]
    fn box_width(e: FloatExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::BoxWidth(e.inner)),
        }
    }

    /// True if object's box height matches the given float expression.
    ///
    /// In JSON/YAML: bbox.height
    ///
    #[staticmethod]
    fn box_height(e: FloatExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::BoxHeight(e.inner)),
        }
    }

    /// True if object's box area (width x height) matches the given float expression.
    ///
    /// In JSON/YAML: bbox.area
    ///
    #[staticmethod]
    fn box_area(e: FloatExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::BoxArea(e.inner)),
        }
    }

    /// True if object's box aspect ratio (width / height) matches the given float expression.
    ///
    /// In JSON/YAML: bbox.width_to_height_ratio
    ///
    #[staticmethod]
    fn box_width_to_height_ratio(e: FloatExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::BoxWidthToHeightRatio(e.inner)),
        }
    }

    /// True if object's box angle matches the given float expression.
    ///
    /// In JSON/YAML: bbox.angle
    ///
    #[staticmethod]
    fn box_angle(e: FloatExpressionProxy) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::BoxAngle(e.inner)),
        }
    }

    /// Always true
    ///
    #[staticmethod]
    fn idle() -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::Idle),
        }
    }

    /// True if JMES Query executed on attributes converted in JSON format returns True.
    ///
    /// In JSON/YAML: attributes.jmes_query
    ///
    #[staticmethod]
    fn attributes_jmes_query(e: String) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::AttributesJMESQuery(e)),
        }
    }

    /// True if object's parent is defined.
    ///
    /// In JSON/YAML: parent.defined
    #[staticmethod]
    fn parent_defined() -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::ParentDefined),
        }
    }

    /// True if object's confidence is defined.
    ///
    /// In JSON/YAML: confidence.defined
    ///
    #[staticmethod]
    fn confidence_defined() -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::ConfidenceDefined),
        }
    }

    /// True if object's track id is defined.
    ///
    /// In JSON/YAML: track.id.defined
    ///
    #[staticmethod]
    fn track_id_defined() -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::TrackDefined),
        }
    }

    /// True if object's box has angle defined.
    ///
    /// In JSON/YAML: bbox.angle.defined
    ///
    #[staticmethod]
    fn box_angle_defined() -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::BoxAngleDefined),
        }
    }

    /// True if object doesn't have attributes
    ///
    #[staticmethod]
    fn attributes_empty() -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::AttributesEmpty),
        }
    }

    #[staticmethod]
    fn attribute_defined(creator: String, label: String) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(MatchQuery::AttributeDefined(creator, label)),
        }
    }

    /// Dumps query to JSON string.
    ///
    #[getter]
    fn json(&self) -> String {
        self.inner.to_json()
    }

    /// Dumps query to pretty JSON string.
    ///
    #[getter]
    fn json_pretty(&self) -> String {
        self.inner.to_json_pretty()
    }

    /// Dumps query to YAML string.
    ///
    #[getter]
    fn yaml(&self) -> String {
        self.inner.to_yaml()
    }

    /// Loads query from JSON string.
    ///
    #[staticmethod]
    fn from_json(json: String) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(
                MatchQuery::from_json(&json)
                    .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {}", e)))
                    .unwrap(),
            ),
        }
    }

    /// Loads query from YAML string.
    ///
    #[staticmethod]
    fn from_yaml(yaml: String) -> MatchQueryProxy {
        MatchQueryProxy {
            inner: Arc::new(
                MatchQuery::from_yaml(&yaml)
                    .map_err(|e| PyValueError::new_err(format!("Invalid YAML: {}", e)))
                    .unwrap(),
            ),
        }
    }
}

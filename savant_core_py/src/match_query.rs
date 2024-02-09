use crate::primitives::bbox::{BBoxMetricType, RBBox};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use savant_core::match_query as rust;

// /**
// Module for defining queries on video objects.
//
// JMES Query Syntax can be found here: `JMESPath <https://jmespath.org/>`__.
//
//  */
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import FloatExpression as FE
    ///    FE.eq(0.5)
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import FloatExpression as FE
    ///    FE.ne(0.5)
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import FloatExpression as FE
    ///    FE.lt(0.5)
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///    
    ///    from savant_rs.match_query import FloatExpression as FE
    ///    FE.le(0.5)
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///    
    ///    from savant_rs.match_query import FloatExpression as FE
    ///    FE.gt(0.5)
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///    
    ///    from savant_rs.match_query import FloatExpression as FE
    ///    FE.ge(0.5)
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///   
    ///    from savant_rs.match_query import FloatExpression as FE
    ///    FE.between(0.5, 0.7)
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import FloatExpression as FE
    ///    FE.one_of(0.5, 0.7, 0.9)
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///  
    ///    from savant_rs.match_query import IntExpression as IE
    ///    IE.eq(5)
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import IntExpression as IE
    ///    IE.ne(5)
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///  
    ///    from savant_rs.match_query import IntExpression as IE
    ///    IE.lt(5)
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///   
    ///    from savant_rs.match_query import IntExpression as IE
    ///    IE.le(5)
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
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///   
    ///    from savant_rs.match_query import IntExpression as IE
    ///    IE.gt(5)
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import IntExpression as IE
    ///    IE.ge(5)
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import IntExpression as IE
    ///    IE.between(5, 7)
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import IntExpression as IE
    ///    IE.one_of(5, 7, 9)
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///    
    ///    from savant_rs.match_query import StringExpression as SE
    ///    SE.eq("hello")
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///   
    ///    from savant_rs.match_query import StringExpression as SE
    ///    SE.ne("hello")
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///  
    ///    from savant_rs.match_query import StringExpression as SE
    ///    SE.contains("hello")
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import StringExpression as SE
    ///    SE.not_contains("hello")
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///  
    ///    from savant_rs.match_query import StringExpression as SE
    ///    SE.starts_with("hello")
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import StringExpression as SE
    ///    SE.ends_with("hello")
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
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import StringExpression as SE
    ///    SE.one_of("hello", "world")
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
    /// \*list: \*list of :py:class:`MatchQuery`
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import IntExpression as IE
    ///    from savant_rs.match_query import StringExpression as SE
    ///
    ///    q = MQ.and_(
    ///        MQ.id(IE.eq(5)),
    ///        MQ.label(SE.eq("hello"))
    ///    )
    ///    print(q.yaml, "\n", q.json)
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
    /// \*list: \*list of :py:class:`MatchQuery`
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import IntExpression as IE
    ///    from savant_rs.match_query import StringExpression as SE
    ///
    ///    q = MQ.or_(
    ///        MQ.namespace(SE.eq("model1")),
    ///        MQ.namespace(SE.eq("model2"))
    ///    )
    ///    print(q.yaml, "\n", q.json)
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
    /// a: :py:class:`MatchQuery`
    ///   Query
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///  
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import IntExpression as IE
    ///
    ///    q = MQ.not_(MQ.id(IE.eq(5)))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn not_(a: &MatchQuery) -> MatchQuery {
        MatchQuery(rust::MatchQuery::Not(Box::new(a.0.clone())))
    }

    /// Stop checking next objects If False predicate (short-circuit)
    ///
    /// In JSON/YAML: stop_if_false
    ///
    /// Parameters
    /// ----------
    /// a: :py:class:`MatchQuery`
    ///  Query
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///    q = MQ.stop_if_false(MQ.frame_is_key_frame())
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn stop_if_false(a: &MatchQuery) -> MatchQuery {
        MatchQuery(rust::MatchQuery::StopIfFalse(Box::new(a.0.clone())))
    }

    /// Stop checking next objects If True predicate (short-circuit)
    ///
    /// In JSON/YAML: stop_if_true
    ///
    /// Parameters
    /// ----------
    /// a: :py:class:`MatchQuery`
    ///  Query
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///    q = MQ.stop_if_true(MQ.not_(MQ.frame_is_key_frame()))
    ///    print(q.yaml, "\n", q.json)
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
    /// a: :py:class:`MatchQuery`
    ///   Query to run on children objects to get the number of matching results
    /// n: :py:class:`IntExpression`
    ///   Integer expression to compare the number retrieved for children with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import IntExpression as IE
    ///    from savant_rs.match_query import StringExpression as SE
    ///
    ///    # More than one person among the children of the object
    ///
    ///    q = MQ.with_children(
    ///        MQ.label(SE.eq("person")),
    ///        IE.ge(1)
    ///    )
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn with_children(a: MatchQuery, n: IntExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::WithChildren(Box::new(a.0.clone()), n.0))
    }

    /// True, when expression defined by evalexpr is computed. EvalExpr is a powerful way to
    /// define complex queries but is slower than explicit definition of expressions.
    ///
    /// In JSON/YAML: eval
    ///
    /// Parameters
    /// ----------
    /// exp: str
    ///   Expression language format: https://docs.rs/evalexpr/11.3.0/evalexpr/index.html
    /// resolvers: List[str]
    ///   Resolvers enabled for evaluation
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    
    ///    # Available symbols:
    ///    #
    ///    #   - id: object id
    ///    #   - namespace: object namespace
    ///    #   - label: object label
    ///    #   - confidence: object confidence
    ///    #   - parent.id: parent object id
    ///    #   - parent.namespace: parent object namespace
    ///    #   - parent.label: parent object label
    ///    #   - bbox.xc: object bbox x center
    ///    #   - bbox.yc: object bbox y center
    ///    #   - bbox.width: object bbox width
    ///    #   - bbox.height: object bbox height
    ///    #   - bbox.angle: object bbox angle
    ///    #   - tracking_info.id: tracking id
    ///    #   - tracking_info.bbox.xc: tracking bbox x center
    ///    #   - tracking_info.bbox.yc: tracking bbox y center
    ///    #   - tracking_info.bbox.width: tracking bbox width
    ///    #   - tracking_info.bbox.height: tracking bbox height
    ///    #   - tracking_info.bbox.angle: tracking bbox angle
    ///    #   - frame.source: frame source
    ///    #   - frame.rate: frame rate
    ///    #   - frame.width: frame width
    ///    #   - frame.height: frame height
    ///    #   - frame.keyframe: frame keyframe
    ///    #   - frame.dts: frame dts
    ///    #   - frame.pts: frame pts
    ///    #   - frame.time_base.nominator: frame time base nominator
    ///    #   - frame.time_base.denominator: frame time base denominator
    ///    #
    ///    # Available functions:
    ///    #   - standard functions: https://docs.rs/evalexpr/11.3.0/evalexpr/index.html
    ///    #
    ///    #   - env("NAME", default): get environment variable, default also represents the type to cast env to
    ///    #   - config("NAME", default): get config variable, default also represents the type to cast config to
    ///    #
    ///    #   - is_boolean(value): check if value is boolean
    ///    #   - is_float(value): check if value is float
    ///    #   - is_int(value): check if value is int
    ///    #   - is_string(value): check if value is string
    ///    #   - is_tuple(value): check if value is tuple
    ///    #   - is_empty(value): check if value is empty
    ///    #
    ///    #   - ends_with(value, suffix): check if value ends with suffix
    ///    #   - starts_with(value, prefix): check if value starts with prefix
    ///    #
    ///    #   - etcd("KEY", default): get etcd variable, default also represents the type to cast etcd to
    ///    #
    ///    q = MQ.eval("""(etcd("pipeline_status", false) == true || env("PIPELINE_STATUS", false) == true) && frame.keyframe""")
    ///    print(q.yaml, "\n", q.json)
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
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import IntExpression as IE
    ///
    ///    q = MQ.id(IE.eq(5))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn id(e: IntExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::Id(e.0))
    }

    /// True if the metric calculated between the current object box and specified box matches the expression.
    ///
    /// Parameters
    /// ----------
    /// bbox: :py:class:`RBBox`
    ///   Bounding box to compare with
    /// metric_type: :py:class:`savant_rs.utils.BBoxMetricType`
    ///   Metric type to use for comparison
    /// e: :py:class:`FloatExpression`
    ///   Float expression to compare the metric with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///    Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import FloatExpression as FE
    ///    from savant_rs.utils import BBoxMetricType
    ///    from savant_rs.primitives.geometry import RBBox
    ///
    ///    q = MQ.box_metric(RBBox(0.5, 0.5, 0.5, 0.5, 0.0), BBoxMetricType.IoU, FE.gt(0.5))
    ///    print(q.yaml, "\n", q.json)
    ///
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

    /// True if the metric calculated between the current object track box and specified box matches the expression.
    ///
    /// Parameters
    /// ----------
    /// bbox: :py:class:`RBBox`
    ///   Bounding box to compare with
    /// metric_type: :py:class:`savant_rs.utils.BBoxMetricType`
    ///   Metric type to use for comparison
    /// e: :py:class:`FloatExpression`
    ///   Float expression to compare the metric with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import FloatExpression as FE
    ///    from savant_rs.utils import BBoxMetricType
    ///    from savant_rs.primitives.geometry import RBBox
    ///    
    ///    q = MQ.track_box_metric(RBBox(0.5, 0.5, 0.5, 0.5, 0.0), BBoxMetricType.IoU, FE.gt(0.5))
    ///    print(q.yaml, "\n", q.json)
    ///
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
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///   
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import StringExpression as SE
    ///
    ///    q = MQ.namespace(SE.eq("model1"))
    ///    print(q.yaml, "\n", q.json)
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
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import StringExpression as SE
    ///   
    ///    q = MQ.label(SE.eq("person"))
    ///    print(q.yaml, "\n", q.json)
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
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///   
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import FloatExpression as FE
    ///
    ///    q = MQ.confidence(FE.gt(0.5))
    ///    print(q.yaml, "\n", q.json)
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
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import IntExpression as IE
    ///
    ///    q = MQ.track_id(IE.eq(5))
    ///    print(q.yaml, "\n", q.json)
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
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import FloatExpression as FE
    ///
    ///    q = MQ.track_box_x_center(FE.gt(0.5))
    ///    print(q.yaml, "\n", q.json)
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
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import FloatExpression as FE
    ///
    ///    q = MQ.track_box_y_center(FE.gt(0.5))
    ///    print(q.yaml, "\n", q.json)
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
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import FloatExpression as FE
    ///   
    ///    q = MQ.track_box_width(FE.gt(0.5))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn track_box_width(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::TrackBoxWidth(e.0))
    }

    /// True if object's track bbox height matches the given float expression.
    ///
    /// In JSON/YAML: track.bbox.height
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`FloatExpression`
    ///   Float expression to compare the object's track bbox height with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import FloatExpression as FE
    ///   
    ///    q = MQ.track_box_height(FE.gt(0.5))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn track_box_height(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::TrackBoxHeight(e.0))
    }

    /// True if object's track bbox area (width x height) matches the given float expression.
    ///
    /// In JSON/YAML: track.bbox.area
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`FloatExpression`
    ///   Float expression to compare the object's track bbox area with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///     
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import FloatExpression as FE
    ///
    ///    q = MQ.track_box_area(FE.gt(0.5))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn track_box_area(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::TrackBoxArea(e.0))
    }

    /// True if object's track bbox aspect ratio (width / height) matches the given float expression.
    ///
    /// In JSON/YAML: track.bbox.width_to_height_ratio
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`FloatExpression`
    ///   Float expression to compare the object's track bbox aspect ratio with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///    
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import FloatExpression as FE
    ///  
    ///    q = MQ.track_box_width_to_height_ratio(FE.gt(0.5))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn track_box_width_to_height_ratio(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::TrackBoxWidthToHeightRatio(e.0))
    }

    /// True if object's track bbox angle matches the given float expression.
    ///
    /// In JSON/YAML: track.bbox.angle
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`FloatExpression`
    ///   Float expression to compare the object's track bbox angle with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///  
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import FloatExpression as FE
    ///
    ///    q = MQ.track_box_angle(FE.gt(0.5))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn track_box_angle(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::TrackBoxAngle(e.0))
    }

    /// True if object's parent id matches the given int expression.
    ///
    /// In JSON/YAML: parent.id
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`IntExpression`
    ///   Integer expression to compare the object's parent id with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import IntExpression as IE
    ///
    ///    q = MQ.parent_id(IE.eq(5))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn parent_id(e: IntExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::ParentId(e.0))
    }

    /// True if object's parent namespace matches the given string expression.
    ///
    /// In JSON/YAML: parent.namespace
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`StringExpression`
    ///   String expression to compare the object's parent namespace with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///   
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import StringExpression as SE
    ///   
    ///    q = MQ.parent_namespace(SE.eq("model1"))
    ///    print(q.yaml, "\n", q.json)
    ///    
    #[staticmethod]
    fn parent_namespace(e: StringExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::ParentNamespace(e.0))
    }

    /// True if object's parent label matches the given string expression.
    ///
    /// In JSON/YAML: parent.label
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`StringExpression`
    ///   String expression to compare the object's parent label with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import StringExpression as SE
    ///   
    ///    q = MQ.parent_label(SE.eq("person"))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn parent_label(e: StringExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::ParentLabel(e.0))
    }

    /// True if object's box xc matches the given float expression.
    ///
    /// In JSON/YAML: bbox.xc
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`FloatExpression`
    ///   Float expression to compare the object's box xc with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///   
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import FloatExpression as FE
    ///
    ///    q = MQ.box_x_center(FE.gt(0.5))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn box_x_center(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::BoxXCenter(e.0))
    }

    /// True if object's box yc matches the given float expression.
    ///
    /// In JSON/YAML: bbox.yc
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`FloatExpression`
    ///   Float expression to compare the object's box yc with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import FloatExpression as FE
    ///
    ///    q = MQ.box_y_center(FE.gt(0.5))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn box_y_center(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::BoxYCenter(e.0))
    }

    /// True if object's box width matches the given float expression.
    ///
    /// In JSON/YAML: bbox.width
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`FloatExpression`
    ///   Float expression to compare the object's box width with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import FloatExpression as FE
    ///    
    ///    q = MQ.box_width(FE.gt(0.5))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn box_width(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::BoxWidth(e.0))
    }

    /// True if object's box height matches the given float expression.
    ///
    /// In JSON/YAML: bbox.height
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`FloatExpression`
    ///   Float expression to compare the object's box height with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///  
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import FloatExpression as FE
    ///
    ///    q = MQ.box_height(FE.gt(0.5))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn box_height(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::BoxHeight(e.0))
    }

    /// True if object's box area (width x height) matches the given float expression.
    ///
    /// In JSON/YAML: bbox.area
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`FloatExpression`
    ///   Float expression to compare the object's box area with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import FloatExpression as FE
    ///
    ///    q = MQ.box_area(FE.gt(0.5))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn box_area(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::BoxArea(e.0))
    }

    /// True if object's box aspect ratio (width / height) matches the given float expression.
    ///
    /// In JSON/YAML: bbox.width_to_height_ratio
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`FloatExpression`
    ///   Float expression to compare the object's box aspect ratio with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///  
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import FloatExpression as FE
    ///  
    ///    q = MQ.box_width_to_height_ratio(FE.gt(0.5))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn box_width_to_height_ratio(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::BoxWidthToHeightRatio(e.0))
    }

    /// True if object's box angle matches the given float expression.
    ///
    /// In JSON/YAML: bbox.angle
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`FloatExpression`
    ///   Float expression to compare the object's box angle with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import FloatExpression as FE
    ///
    ///    q = MQ.box_angle(FE.gt(0.5))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn box_angle(e: FloatExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::BoxAngle(e.0))
    }

    /// Always true
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///    q = MQ.idle()
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    pub fn idle() -> MatchQuery {
        MatchQuery(rust::MatchQuery::Idle)
    }

    /// True if JMES Query executed on attributes converted in JSON format returns True.
    ///
    /// Syntax: https://jmespath.org/specification.html
    ///
    /// In JSON/YAML: attributes.jmes_query
    ///
    /// Parameters
    /// ----------
    /// e: str
    ///   JMES Query to run on object's attributes
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///    q = MQ.attributes_jmes_query("[? (hint == 'morphological-classifier') && (namespace == 'classifier')]")
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn attributes_jmes_query(e: String) -> MatchQuery {
        MatchQuery(rust::MatchQuery::AttributesJMESQuery(e))
    }

    /// True if object's parent is defined.
    ///
    /// In JSON/YAML: parent.defined
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///    q = MQ.parent_defined()
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn parent_defined() -> MatchQuery {
        MatchQuery(rust::MatchQuery::ParentDefined)
    }

    /// True if object's confidence is defined.
    ///
    /// In JSON/YAML: confidence.defined
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///    q = MQ.confidence_defined()
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn confidence_defined() -> MatchQuery {
        MatchQuery(rust::MatchQuery::ConfidenceDefined)
    }

    /// True if object's track id is defined.
    ///
    /// In JSON/YAML: track.id.defined
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///    q = MQ.track_id_defined()
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn track_id_defined() -> MatchQuery {
        MatchQuery(rust::MatchQuery::TrackDefined)
    }

    /// True if object's box has angle defined.
    ///
    /// In JSON/YAML: bbox.angle.defined
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///    q = MQ.box_angle_defined()
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn box_angle_defined() -> MatchQuery {
        MatchQuery(rust::MatchQuery::BoxAngleDefined)
    }

    /// True if object doesn't have attributes
    ///
    /// In JSON/YAML: attributes.empty
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///    q = MQ.attributes_empty()
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn attributes_empty() -> MatchQuery {
        MatchQuery(rust::MatchQuery::AttributesEmpty)
    }

    /// True if object's attribute is defined.
    ///
    /// In JSON/YAML: attribute.defined
    ///
    /// Parameters
    /// ----------
    /// namespace: str
    ///   Attribute namespace
    /// label: str
    ///   Attribute label
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///    q = MQ.attribute_defined("classifier", "hint")
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn attribute_defined(namespace: String, label: String) -> MatchQuery {
        MatchQuery(rust::MatchQuery::AttributeExists(namespace, label))
    }

    /// True if frame source_id matches the given string expression.
    ///
    /// In JSON/YAML: frame.source_id
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`StringExpression`
    ///   String expression to compare the frame's source_id with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import StringExpression as SE
    ///
    ///    q = MQ.frame_source_id(SE.eq("source1"))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn frame_source_id(e: StringExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameSourceId(e.0))
    }

    /// True if frame is key frame.
    ///
    /// In JSON/YAML: frame.is_key_frame
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///   q = MQ.frame_is_key_frame()
    ///   print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn frame_is_key_frame() -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameIsKeyFrame)
    }

    /// True if frame width matches the given int expression.
    ///
    /// In JSON/YAML: frame.width
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`IntExpression`
    ///   Integer expression to compare the frame's width with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import IntExpression as IE
    ///
    ///    q = MQ.frame_width(IE.eq(1920))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn frame_width(e: IntExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameWidth(e.0))
    }

    /// True if frame height matches the given int expression.
    ///
    /// In JSON/YAML: frame.height
    ///
    /// Parameters
    /// ----------
    /// e: :py:class:`IntExpression`
    ///   Integer expression to compare the frame's height with
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///    from savant_rs.match_query import IntExpression as IE
    ///
    ///    q = MQ.frame_height(IE.eq(1080))
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn frame_height(e: IntExpression) -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameHeight(e.0))
    }

    /// When the frame does not have associated video, because of sparsity, for example
    ///
    /// In JSON/YAML: frame.no_video
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///    q = MQ.frame_no_video()
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn frame_no_video() -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameNoVideo)
    }

    /// When the processing is configured in pass-through mode
    ///
    /// In JSON/YAML: frame.transcoding.is_copy
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///    q = MQ.frame_transcoding_is_copy()
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn frame_transcoding_is_copy() -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameTranscodingIsCopy)
    }

    /// True if frame's attribute is defined.
    ///
    /// In JSON/YAML: frame.attribute.exists
    ///
    /// Parameters
    /// ----------
    /// namespace: str
    ///   Attribute namespace
    /// label: str
    ///   Attribute label
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///    q = MQ.frame_attribute_exists("age_gender", "age")
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn frame_attribute_exists(namespace: String, label: String) -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameAttributeExists(namespace, label))
    }

    /// True if frame's attribute is not defined.
    ///
    /// In JSON/YAML: frame.attribute.empty
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///    q = MQ.frame_attribute_empty()
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn frame_attributes_empty() -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameAttributesEmpty)
    }

    /// True if JMES Query executed on frame attributes converted in JSON format returns True.
    ///
    /// Syntax: https://jmespath.org/specification.html
    ///
    /// In JSON/YAML: frame.attributes.jmes_query
    ///
    /// Parameters
    /// ----------
    /// e: str
    ///   JMES Query to run on frame's attributes
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///    q = MQ.frame_attributes_jmes_query("[? (source_id == 'source1')]")
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn frame_attributes_jmes_query(e: String) -> MatchQuery {
        MatchQuery(rust::MatchQuery::FrameAttributesJMESQuery(e))
    }

    /// Dumps query to JSON string.
    ///
    /// Returns
    /// -------
    /// str
    ///   JSON string
    ///
    #[getter]
    fn json(&self) -> String {
        self.0.to_json()
    }

    /// Dumps query to pretty JSON string.
    ///
    /// Returns
    /// -------
    /// str
    ///   Pretty JSON string
    ///
    #[getter]
    fn json_pretty(&self) -> String {
        self.0.to_json_pretty()
    }

    /// Dumps query to YAML string.
    ///
    /// Returns
    /// -------
    /// str
    ///   YAML string
    ///
    #[getter]
    fn yaml(&self) -> String {
        self.0.to_yaml()
    }

    /// Loads query from JSON string.
    ///
    /// Parameters
    /// ----------
    /// json: str
    ///   JSON string
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the JSON string is invalid
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///    q = MQ.from_json('{"not":{"id":{"eq":5}}}')
    ///    print(q.yaml, "\n", q.json)
    ///
    #[staticmethod]
    fn from_json(json: String) -> PyResult<MatchQuery> {
        Ok(MatchQuery(rust::MatchQuery::from_json(&json).map_err(
            |e| PyValueError::new_err(format!("Invalid JSON: {}", e)),
        )?))
    }

    /// Loads query from YAML string.
    ///
    /// Parameters
    /// ----------
    /// yaml: str
    ///   YAML string
    ///
    /// Returns
    /// -------
    /// :py:class:`MatchQuery`
    ///   Query
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the YAML string is invalid
    ///
    /// Example
    /// -------
    ///
    /// .. code-block:: python
    ///
    ///    from savant_rs.match_query import MatchQuery as MQ
    ///
    ///    q = MQ.from_yaml("""
    ///    not:
    ///      id:
    ///        eq: 5
    ///    """)
    ///
    #[staticmethod]
    fn from_yaml(yaml: String) -> PyResult<MatchQuery> {
        Ok(MatchQuery(rust::MatchQuery::from_yaml(&yaml).map_err(
            |e| PyValueError::new_err(format!("Invalid YAML: {}", e)),
        )?))
    }
}

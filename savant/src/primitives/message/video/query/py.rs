use super::{FloatExpression, IntExpression, Query, StringExpression};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::ops::Deref;
use std::sync::Arc;

#[pymodule]
pub fn video_object_query(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FloatExpressionProxy>()?;
    m.add_class::<IntExpressionProxy>()?;
    m.add_class::<StringExpressionProxy>()?;
    m.add_class::<QueryProxy>()?;
    Ok(())
}

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

    #[staticmethod]
    fn eq(v: f64) -> FloatExpressionProxy {
        FloatExpressionProxy {
            inner: FloatExpression::EQ(v),
        }
    }

    #[staticmethod]
    fn ne(v: f64) -> FloatExpressionProxy {
        FloatExpressionProxy {
            inner: FloatExpression::NE(v),
        }
    }

    #[staticmethod]
    fn lt(v: f64) -> FloatExpressionProxy {
        FloatExpressionProxy {
            inner: FloatExpression::LT(v),
        }
    }

    #[staticmethod]
    fn le(v: f64) -> FloatExpressionProxy {
        FloatExpressionProxy {
            inner: FloatExpression::LE(v),
        }
    }

    #[staticmethod]
    fn gt(v: f64) -> FloatExpressionProxy {
        FloatExpressionProxy {
            inner: FloatExpression::GT(v),
        }
    }

    #[staticmethod]
    fn ge(v: f64) -> FloatExpressionProxy {
        FloatExpressionProxy {
            inner: FloatExpression::GE(v),
        }
    }

    #[staticmethod]
    fn between(a: f64, b: f64) -> FloatExpressionProxy {
        FloatExpressionProxy {
            inner: FloatExpression::Between(a, b),
        }
    }

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

    #[staticmethod]
    fn eq(v: i64) -> IntExpressionProxy {
        IntExpressionProxy {
            inner: IntExpression::EQ(v),
        }
    }

    #[staticmethod]
    fn ne(v: i64) -> IntExpressionProxy {
        IntExpressionProxy {
            inner: IntExpression::NE(v),
        }
    }

    #[staticmethod]
    fn lt(v: i64) -> IntExpressionProxy {
        IntExpressionProxy {
            inner: IntExpression::LT(v),
        }
    }

    #[staticmethod]
    fn le(v: i64) -> IntExpressionProxy {
        IntExpressionProxy {
            inner: IntExpression::LE(v),
        }
    }

    #[staticmethod]
    fn gt(v: i64) -> IntExpressionProxy {
        IntExpressionProxy {
            inner: IntExpression::GT(v),
        }
    }

    #[staticmethod]
    fn ge(v: i64) -> IntExpressionProxy {
        IntExpressionProxy {
            inner: IntExpression::GE(v),
        }
    }

    #[staticmethod]
    fn between(a: i64, b: i64) -> IntExpressionProxy {
        IntExpressionProxy {
            inner: IntExpression::Between(a, b),
        }
    }

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

    #[staticmethod]
    fn eq(v: String) -> StringExpressionProxy {
        StringExpressionProxy {
            inner: StringExpression::EQ(v),
        }
    }

    #[staticmethod]
    fn ne(v: String) -> StringExpressionProxy {
        StringExpressionProxy {
            inner: StringExpression::NE(v),
        }
    }

    #[staticmethod]
    fn contains(v: String) -> StringExpressionProxy {
        StringExpressionProxy {
            inner: StringExpression::Contains(v),
        }
    }

    #[staticmethod]
    fn not_contains(v: String) -> StringExpressionProxy {
        StringExpressionProxy {
            inner: StringExpression::NotContains(v),
        }
    }

    #[staticmethod]
    fn starts_with(v: String) -> StringExpressionProxy {
        StringExpressionProxy {
            inner: StringExpression::StartsWith(v),
        }
    }

    #[staticmethod]
    fn ends_with(v: String) -> StringExpressionProxy {
        StringExpressionProxy {
            inner: StringExpression::EndsWith(v),
        }
    }

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

#[pyclass]
#[pyo3(name = "Query")]
#[derive(Debug, Clone)]
pub struct QueryProxy {
    pub inner: Arc<Query>,
}

#[pymethods]
impl QueryProxy {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:#?}", self.inner.deref())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[staticmethod]
    #[pyo3(signature = (*list))]
    fn and_(list: &PyTuple) -> QueryProxy {
        let mut v = Vec::with_capacity(list.len());
        for arg in list {
            let q = arg
                .extract::<QueryProxy>()
                .expect("Invalid argument. Only Query values are allowed.");
            v.push(q.inner.deref().clone());
        }
        QueryProxy {
            inner: Arc::new(Query::And(v)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (*list))]
    fn or_(list: &PyTuple) -> QueryProxy {
        let mut v = Vec::with_capacity(list.len());
        for arg in list {
            let q = arg
                .extract::<QueryProxy>()
                .expect("Invalid argument. Only Query values are allowed.");
            v.push(q.inner.deref().clone());
        }
        QueryProxy {
            inner: Arc::new(Query::Or(v)),
        }
    }

    #[staticmethod]
    fn not_(a: QueryProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::Not(Box::new(a.inner.deref().clone()))),
        }
    }

    #[staticmethod]
    fn with_children(a: QueryProxy, n: IntExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::WithChildren(
                Box::new(a.inner.deref().clone()),
                n.inner,
            )),
        }
    }

    #[staticmethod]
    fn id(e: IntExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::Id(e.inner)),
        }
    }

    #[staticmethod]
    fn creator(e: StringExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::Creator(e.inner)),
        }
    }

    #[staticmethod]
    fn label(e: StringExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::Label(e.inner)),
        }
    }

    #[staticmethod]
    fn confidence(e: FloatExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::Confidence(e.inner)),
        }
    }

    #[staticmethod]
    fn track_id(e: IntExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::TrackId(e.inner)),
        }
    }

    #[staticmethod]
    fn track_box_x_center(e: FloatExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::TrackBoxXCenter(e.inner)),
        }
    }

    #[staticmethod]
    fn track_box_y_center(e: FloatExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::TrackBoxYCenter(e.inner)),
        }
    }

    #[staticmethod]
    fn track_box_width(e: FloatExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::TrackBoxWidth(e.inner)),
        }
    }

    #[staticmethod]
    fn track_box_height(e: FloatExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::TrackBoxHeight(e.inner)),
        }
    }

    #[staticmethod]
    fn track_box_area(e: FloatExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::TrackBoxArea(e.inner)),
        }
    }

    #[staticmethod]
    fn track_box_angle(e: FloatExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::TrackBoxAngle(e.inner)),
        }
    }

    #[staticmethod]
    fn user_defined_rust_plugin_object_predicate(plugin: String, function: String) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::UserDefinedObjectPredicate(plugin, function)),
        }
    }

    #[staticmethod]
    fn parent_id(e: IntExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::ParentId(e.inner)),
        }
    }

    #[staticmethod]
    fn parent_creator(e: StringExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::ParentCreator(e.inner)),
        }
    }

    #[staticmethod]
    fn parent_label(e: StringExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::ParentLabel(e.inner)),
        }
    }

    #[staticmethod]
    fn box_x_center(e: FloatExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::BoxXCenter(e.inner)),
        }
    }

    #[staticmethod]
    fn box_y_center(e: FloatExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::BoxYCenter(e.inner)),
        }
    }

    #[staticmethod]
    fn box_width(e: FloatExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::BoxWidth(e.inner)),
        }
    }

    #[staticmethod]
    fn box_height(e: FloatExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::BoxHeight(e.inner)),
        }
    }

    #[staticmethod]
    fn box_area(e: FloatExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::BoxArea(e.inner)),
        }
    }

    #[staticmethod]
    fn box_angle(e: FloatExpressionProxy) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::BoxAngle(e.inner)),
        }
    }

    #[staticmethod]
    fn idle() -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::Idle),
        }
    }

    #[staticmethod]
    fn attributes_jmes_query(e: String) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::AttributesJMESQuery(e)),
        }
    }

    #[staticmethod]
    fn parent_defined() -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::ParentDefined),
        }
    }

    #[staticmethod]
    fn confidence_defined() -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::ConfidenceDefined),
        }
    }

    #[staticmethod]
    fn track_id_defined() -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::TrackDefined),
        }
    }

    #[staticmethod]
    fn box_angle_defined() -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::BoxAngleDefined),
        }
    }

    #[staticmethod]
    fn attributes_empty() -> QueryProxy {
        QueryProxy {
            inner: Arc::new(Query::AttributesEmpty),
        }
    }

    #[getter]
    fn json(&self) -> String {
        self.inner.to_json()
    }

    #[getter]
    fn json_pretty(&self) -> String {
        self.inner.to_json_pretty()
    }

    #[getter]
    fn yaml(&self) -> String {
        self.inner.to_yaml()
    }

    #[staticmethod]
    fn from_json(json: String) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(
                Query::from_json(&json)
                    .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {}", e)))
                    .unwrap(),
            ),
        }
    }

    #[staticmethod]
    fn from_yaml(yaml: String) -> QueryProxy {
        QueryProxy {
            inner: Arc::new(
                Query::from_yaml(&yaml)
                    .map_err(|e| PyValueError::new_err(format!("Invalid YAML: {}", e)))
                    .unwrap(),
            ),
        }
    }
}

use super::{FloatExpression, IntExpression, Query, StringExpression};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::ops::Deref;
use std::sync::Arc;

#[pymodule]
pub fn video_object_query(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FloatExpressionWrapper>()?;
    m.add_class::<IntExpressionWrapper>()?;
    m.add_class::<StringExpressionWrapper>()?;
    m.add_class::<QueryWrapper>()?;
    Ok(())
}

#[pyclass]
#[derive(Debug, Clone)]
#[pyo3(name = "FloatExpression")]
pub struct FloatExpressionWrapper {
    inner: FloatExpression,
}

#[pymethods]
impl FloatExpressionWrapper {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[staticmethod]
    fn eq(v: f64) -> FloatExpressionWrapper {
        FloatExpressionWrapper {
            inner: FloatExpression::EQ(v),
        }
    }

    #[staticmethod]
    fn ne(v: f64) -> FloatExpressionWrapper {
        FloatExpressionWrapper {
            inner: FloatExpression::NE(v),
        }
    }

    #[staticmethod]
    fn lt(v: f64) -> FloatExpressionWrapper {
        FloatExpressionWrapper {
            inner: FloatExpression::LT(v),
        }
    }

    #[staticmethod]
    fn le(v: f64) -> FloatExpressionWrapper {
        FloatExpressionWrapper {
            inner: FloatExpression::LE(v),
        }
    }

    #[staticmethod]
    fn gt(v: f64) -> FloatExpressionWrapper {
        FloatExpressionWrapper {
            inner: FloatExpression::GT(v),
        }
    }

    #[staticmethod]
    fn ge(v: f64) -> FloatExpressionWrapper {
        FloatExpressionWrapper {
            inner: FloatExpression::GE(v),
        }
    }

    #[staticmethod]
    fn between(a: f64, b: f64) -> FloatExpressionWrapper {
        FloatExpressionWrapper {
            inner: FloatExpression::Between(a, b),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (*list))]
    fn one_of(list: &PyTuple) -> FloatExpressionWrapper {
        let mut vals = Vec::with_capacity(list.len());
        for arg in list {
            let v = arg
                .extract::<f64>()
                .expect("Invalid argument. Only f64 values are allowed.");
            vals.push(v);
        }
        FloatExpressionWrapper {
            inner: FloatExpression::OneOf(vals),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
#[pyo3(name = "IntExpression")]
pub struct IntExpressionWrapper {
    inner: IntExpression,
}

#[pymethods]
impl IntExpressionWrapper {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[staticmethod]
    fn eq(v: i64) -> IntExpressionWrapper {
        IntExpressionWrapper {
            inner: IntExpression::EQ(v),
        }
    }

    #[staticmethod]
    fn ne(v: i64) -> IntExpressionWrapper {
        IntExpressionWrapper {
            inner: IntExpression::NE(v),
        }
    }

    #[staticmethod]
    fn lt(v: i64) -> IntExpressionWrapper {
        IntExpressionWrapper {
            inner: IntExpression::LT(v),
        }
    }

    #[staticmethod]
    fn le(v: i64) -> IntExpressionWrapper {
        IntExpressionWrapper {
            inner: IntExpression::LE(v),
        }
    }

    #[staticmethod]
    fn gt(v: i64) -> IntExpressionWrapper {
        IntExpressionWrapper {
            inner: IntExpression::GT(v),
        }
    }

    #[staticmethod]
    fn ge(v: i64) -> IntExpressionWrapper {
        IntExpressionWrapper {
            inner: IntExpression::GE(v),
        }
    }

    #[staticmethod]
    fn between(a: i64, b: i64) -> IntExpressionWrapper {
        IntExpressionWrapper {
            inner: IntExpression::Between(a, b),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (*list))]
    fn one_of(list: &PyTuple) -> IntExpressionWrapper {
        let mut vals = Vec::with_capacity(list.len());
        for arg in list {
            let v = arg
                .extract::<i64>()
                .expect("Invalid argument. Only i64 values are allowed.");
            vals.push(v);
        }
        IntExpressionWrapper {
            inner: IntExpression::OneOf(vals),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
#[pyo3(name = "StringExpression")]
pub struct StringExpressionWrapper {
    inner: StringExpression,
}

#[pymethods]
impl StringExpressionWrapper {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[staticmethod]
    fn eq(v: String) -> StringExpressionWrapper {
        StringExpressionWrapper {
            inner: StringExpression::EQ(v),
        }
    }

    #[staticmethod]
    fn ne(v: String) -> StringExpressionWrapper {
        StringExpressionWrapper {
            inner: StringExpression::NE(v),
        }
    }

    #[staticmethod]
    fn contains(v: String) -> StringExpressionWrapper {
        StringExpressionWrapper {
            inner: StringExpression::Contains(v),
        }
    }

    #[staticmethod]
    fn not_contains(v: String) -> StringExpressionWrapper {
        StringExpressionWrapper {
            inner: StringExpression::NotContains(v),
        }
    }

    #[staticmethod]
    fn starts_with(v: String) -> StringExpressionWrapper {
        StringExpressionWrapper {
            inner: StringExpression::StartsWith(v),
        }
    }

    #[staticmethod]
    fn ends_with(v: String) -> StringExpressionWrapper {
        StringExpressionWrapper {
            inner: StringExpression::EndsWith(v),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (*list))]
    fn one_of(list: &PyTuple) -> StringExpressionWrapper {
        let mut vals = Vec::with_capacity(list.len());
        for arg in list {
            let v = arg
                .extract::<String>()
                .expect("Invalid argument. Only String values are allowed.");
            vals.push(v);
        }
        StringExpressionWrapper {
            inner: StringExpression::OneOf(vals),
        }
    }
}

#[pyclass]
#[pyo3(name = "Query")]
#[derive(Debug, Clone)]
pub struct QueryWrapper {
    pub inner: Arc<Query>,
}

#[pymethods]
impl QueryWrapper {
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
    fn and_(list: &PyTuple) -> QueryWrapper {
        let mut v = Vec::with_capacity(list.len());
        for arg in list {
            let q = arg
                .extract::<QueryWrapper>()
                .expect("Invalid argument. Only Query values are allowed.");
            v.push(q.inner.deref().clone());
        }
        QueryWrapper {
            inner: Arc::new(Query::And(v)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (*list))]
    fn or_(list: &PyTuple) -> QueryWrapper {
        let mut v = Vec::with_capacity(list.len());
        for arg in list {
            let q = arg
                .extract::<QueryWrapper>()
                .expect("Invalid argument. Only Query values are allowed.");
            v.push(q.inner.deref().clone());
        }
        QueryWrapper {
            inner: Arc::new(Query::Or(v)),
        }
    }

    #[staticmethod]
    fn not_(a: QueryWrapper) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::Not(Box::new(a.inner.deref().clone()))),
        }
    }

    #[staticmethod]
    fn id(e: IntExpressionWrapper) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::Id(e.inner)),
        }
    }

    #[staticmethod]
    fn creator(e: StringExpressionWrapper) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::Creator(e.inner)),
        }
    }

    #[staticmethod]
    fn label(e: StringExpressionWrapper) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::Label(e.inner)),
        }
    }

    #[staticmethod]
    fn confidence(e: FloatExpressionWrapper) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::Confidence(e.inner)),
        }
    }

    #[staticmethod]
    fn track_id(e: IntExpressionWrapper) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::TrackId(e.inner)),
        }
    }

    #[staticmethod]
    fn parent_id(e: IntExpressionWrapper) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::ParentId(e.inner)),
        }
    }

    #[staticmethod]
    fn parent_creator(e: StringExpressionWrapper) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::ParentCreator(e.inner)),
        }
    }

    #[staticmethod]
    fn parent_label(e: StringExpressionWrapper) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::ParentLabel(e.inner)),
        }
    }

    #[staticmethod]
    fn box_x_center(e: FloatExpressionWrapper) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::BoxXCenter(e.inner)),
        }
    }

    #[staticmethod]
    fn box_y_center(e: FloatExpressionWrapper) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::BoxYCenter(e.inner)),
        }
    }

    #[staticmethod]
    fn box_width(e: FloatExpressionWrapper) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::BoxWidth(e.inner)),
        }
    }

    #[staticmethod]
    fn box_height(e: FloatExpressionWrapper) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::BoxHeight(e.inner)),
        }
    }

    #[staticmethod]
    fn box_area(e: FloatExpressionWrapper) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::BoxArea(e.inner)),
        }
    }

    #[staticmethod]
    fn box_angle(e: FloatExpressionWrapper) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::BoxAngle(e.inner)),
        }
    }

    #[staticmethod]
    fn idle() -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::Idle),
        }
    }

    #[staticmethod]
    fn attributes_jmes_query(e: String) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::AttributesJMESQuery(e)),
        }
    }

    #[staticmethod]
    fn parent_defined() -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::ParentDefined),
        }
    }

    #[staticmethod]
    fn confidence_defined() -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::ConfidenceDefined),
        }
    }

    #[staticmethod]
    fn track_id_defined() -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::TrackIdDefined),
        }
    }

    #[staticmethod]
    fn box_angle_defined() -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(Query::BoxAngleDefined),
        }
    }

    #[staticmethod]
    fn attributes_empty() -> QueryWrapper {
        QueryWrapper {
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
    fn from_json(json: String) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(
                Query::from_json(&json)
                    .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {}", e)))
                    .unwrap(),
            ),
        }
    }

    #[staticmethod]
    fn from_yaml(yaml: String) -> QueryWrapper {
        QueryWrapper {
            inner: Arc::new(
                Query::from_yaml(&yaml)
                    .map_err(|e| PyValueError::new_err(format!("Invalid YAML: {}", e)))
                    .unwrap(),
            ),
        }
    }
}

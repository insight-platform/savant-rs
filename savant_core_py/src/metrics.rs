use prometheus_client::registry::Unit;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
pub struct CounterFamily(pub(crate) savant_core::metrics::SharedCounterFamily);

#[pymethods]
impl CounterFamily {
    /// Returns the value of the counter with the given labels.
    ///
    /// Parameters
    /// ----------
    /// label_values : List[str]
    ///  The list of label values.
    ///
    /// Returns
    /// -------
    /// Optional[int]
    ///  The value of the counter.
    ///
    /// Raises
    /// ------
    /// PyValueError
    ///  If the counter does not exist.
    #[pyo3(signature = (label_values=vec![]))]
    pub fn get(&self, label_values: Vec<String>) -> PyResult<Option<u64>> {
        let l_ref = label_values
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();
        self.0
            .lock()
            .get(&l_ref)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Deletes the counter with the given labels.
    ///
    /// Parameters
    /// ----------
    /// label_values : List[str]
    ///   The list of label values.
    ///
    #[pyo3(signature = (label_values=vec![]))]
    pub fn delete(&self, label_values: Vec<String>) -> PyResult<Option<u64>> {
        let l_ref = label_values
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();
        self.0
            .lock()
            .delete(&l_ref)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (value=1, label_values=vec![]))]
    pub fn inc(&self, value: u64, label_values: Vec<String>) -> PyResult<u64> {
        let l_ref = label_values
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();
        self.0
            .lock()
            .inc(value, &l_ref)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (value, label_values=vec![]))]
    pub fn set(&self, value: u64, label_values: Vec<String>) -> PyResult<u64> {
        let l_ref = label_values
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();
        self.0
            .lock()
            .set(value, &l_ref)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Creates or returns a counter with the given name.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///   The name of the counter.
    /// description : str, optional
    ///   The description of the counter.
    /// label_names : List[str], optional
    ///   The list of label names.
    /// unit : str, optional
    ///   The unit of the counter.
    ///
    /// Returns
    /// -------
    /// CounterFamily
    ///   The counter.
    ///
    #[staticmethod]
    #[pyo3(signature = (name, description=None, label_names=vec![], unit=None))]
    pub fn get_or_create_counter_family(
        name: &str,
        description: Option<&str>,
        label_names: Vec<String>,
        unit: Option<String>,
    ) -> CounterFamily {
        let ln_ref = label_names
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();
        CounterFamily(savant_core::metrics::get_or_create_counter_family(
            name,
            description,
            &ln_ref,
            unit.map(Unit::Other),
        ))
    }

    /// Returns a counter with the given name.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///   The name of the counter.
    ///
    /// Returns
    /// -------
    /// Optional[CounterFamily]
    ///   The counter.
    ///
    #[staticmethod]
    pub fn get_counter_family(name: &str) -> Option<CounterFamily> {
        savant_core::metrics::get_counter_family(name).map(CounterFamily)
    }
}

#[pyclass]
pub struct GaugeFamily(pub(crate) savant_core::metrics::SharedGaugeFamily);

#[pymethods]
impl GaugeFamily {
    #[pyo3(signature = (label_values=vec![]))]
    pub fn get(&self, label_values: Vec<String>) -> PyResult<Option<f64>> {
        let l_ref = label_values
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();
        self.0
            .lock()
            .get(&l_ref)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (label_values=vec![]))]
    pub fn delete(&self, label_values: Vec<String>) -> PyResult<Option<f64>> {
        let l_ref = label_values
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();
        self.0
            .lock()
            .delete(&l_ref)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (value, label_values=vec![]))]
    pub fn set(&self, value: f64, label_values: Vec<String>) -> PyResult<f64> {
        let l_ref = label_values
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();
        self.0
            .lock()
            .set(value, &l_ref)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Creates or returns a gauge with the given name.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///   The name of the gauge.
    /// description : str, optional
    ///   The description of the gauge.
    /// label_names : List[str], optional
    ///   The list of label names.
    /// unit : str, optional
    ///
    /// Returns
    /// -------
    /// GaugeFamily
    ///   The gauge.
    ///
    #[staticmethod]
    #[pyo3(signature = (name, description=None, label_names=vec![], unit=None))]
    pub fn get_or_create_gauge_family(
        name: &str,
        description: Option<&str>,
        label_names: Vec<String>,
        unit: Option<String>,
    ) -> GaugeFamily {
        let ln_ref = label_names
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();
        GaugeFamily(savant_core::metrics::get_or_create_gauge_family(
            name,
            description,
            &ln_ref,
            unit.map(Unit::Other),
        ))
    }

    /// Returns a counter with the given name.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///   The name of the counter.
    ///
    /// Returns
    /// -------
    /// Optional[GaugeFamily]
    ///   The counter.
    ///
    #[staticmethod]
    pub fn get_gauge_family(name: &str) -> Option<GaugeFamily> {
        savant_core::metrics::get_gauge_family(name).map(GaugeFamily)
    }
}

/// Histogram state as returned to Python: ``(sum, count, buckets)``, where
/// ``buckets`` is a list of ``(upper_bound, cumulative_count)`` and the last
/// upper bound is ``inf``.
type PyHistogramValue = (f64, u64, Vec<(f64, u64)>);

fn to_py_histogram_value(value: savant_core::metrics::HistogramValue) -> PyHistogramValue {
    (value.sum, value.count, value.cumulative_buckets())
}

#[pyclass]
pub struct HistogramFamily(pub(crate) savant_core::metrics::SharedHistogramFamily);

#[pymethods]
impl HistogramFamily {
    /// Records a value in the histogram with the given labels.
    ///
    /// Parameters
    /// ----------
    /// value : float
    ///   The observed value.
    /// label_values : List[str]
    ///   The list of label values.
    ///
    /// Raises
    /// ------
    /// PyValueError
    ///   If the labels do not match the label names or the value is not finite
    ///   (NaN or infinity).
    ///
    #[pyo3(signature = (value, label_values=vec![]))]
    pub fn observe(&self, value: f64, label_values: Vec<String>) -> PyResult<()> {
        let l_ref = label_values
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();
        self.0
            .lock()
            .observe(value, &l_ref)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Returns the state of the histogram with the given labels.
    ///
    /// Parameters
    /// ----------
    /// label_values : List[str]
    ///   The list of label values.
    ///
    /// Returns
    /// -------
    /// Optional[Tuple[float, int, List[Tuple[float, int]]]]
    ///   ``(sum, count, buckets)``, where ``buckets`` holds
    ///   ``(upper_bound, cumulative_count)`` pairs and ends with ``inf``.
    ///
    #[pyo3(signature = (label_values=vec![]))]
    pub fn get(&self, label_values: Vec<String>) -> PyResult<Option<PyHistogramValue>> {
        let l_ref = label_values
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();
        self.0
            .lock()
            .get(&l_ref)
            .map(|v| v.map(to_py_histogram_value))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Deletes the histogram with the given labels.
    ///
    /// Parameters
    /// ----------
    /// label_values : List[str]
    ///   The list of label values.
    ///
    /// Returns
    /// -------
    /// Optional[Tuple[float, int, List[Tuple[float, int]]]]
    ///   The deleted state, in the same form as :py:meth:`get`.
    ///
    #[pyo3(signature = (label_values=vec![]))]
    pub fn delete(&self, label_values: Vec<String>) -> PyResult<Option<PyHistogramValue>> {
        let l_ref = label_values
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();
        self.0
            .lock()
            .delete(&l_ref)
            .map(|v| v.map(to_py_histogram_value))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// The bucket upper bounds, without the implicit ``+Inf`` bucket.
    ///
    #[getter]
    pub fn buckets(&self) -> Vec<f64> {
        self.0.lock().get_buckets().to_vec()
    }

    /// Creates or returns a histogram with the given name.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///   The name of the histogram.
    /// description : str, optional
    ///   The description of the histogram.
    /// label_names : List[str], optional
    ///   The list of label names. ``le`` is reserved.
    /// buckets : List[float], optional
    ///   Finite, strictly increasing bucket upper bounds below the maximum
    ///   float (it is reserved for ``+Inf``). Defaults to the
    ///   Prometheus defaults ``[0.005, 0.01, ..., 10.0]``. Ignored if the
    ///   histogram already exists.
    /// unit : str, optional
    ///   The unit of the histogram.
    ///
    /// Returns
    /// -------
    /// HistogramFamily
    ///   The histogram.
    ///
    /// Raises
    /// ------
    /// PyValueError
    ///   If the buckets or label names are invalid.
    ///
    #[staticmethod]
    #[pyo3(signature = (name, description=None, label_names=vec![], buckets=None, unit=None))]
    pub fn get_or_create_histogram_family(
        name: &str,
        description: Option<&str>,
        label_names: Vec<String>,
        buckets: Option<Vec<f64>>,
        unit: Option<String>,
    ) -> PyResult<HistogramFamily> {
        let ln_ref = label_names
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();
        savant_core::metrics::get_or_create_histogram_family(
            name,
            description,
            &ln_ref,
            buckets.as_deref(),
            unit.map(Unit::Other),
        )
        .map(HistogramFamily)
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Returns a histogram with the given name.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///   The name of the histogram.
    ///
    /// Returns
    /// -------
    /// Optional[HistogramFamily]
    ///   The histogram.
    ///
    #[staticmethod]
    pub fn get_histogram_family(name: &str) -> Option<HistogramFamily> {
        savant_core::metrics::get_histogram_family(name).map(HistogramFamily)
    }
}

/// Deletes a counter with the given name.
///
/// Parameters
/// ----------
/// name : str
///   The name of the counter or gauge.
///
#[pyfunction]
pub fn delete_metric_family(name: &str) {
    savant_core::metrics::delete_metric_family(name);
}
#[pyfunction]
pub fn set_extra_labels(labels: HashMap<String, String>) {
    let labels = labels.into_iter().collect::<hashbrown::HashMap<_, _>>();
    savant_core::metrics::set_extra_labels(labels);
}

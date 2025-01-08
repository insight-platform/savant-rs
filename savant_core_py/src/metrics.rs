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

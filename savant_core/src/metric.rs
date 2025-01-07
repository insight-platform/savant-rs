use anyhow::bail;
use hashbrown::HashMap;
use lazy_static::lazy_static;
use parking_lot::Mutex;
use prometheus_client::metrics::counter::Counter as TypedPrometheusCounter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge as TypedPrometheusGauge;
use prometheus_client::registry::Unit;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

pub(crate) mod pipeline_metric_collector;
pub(crate) mod user_metric_collector;

type PrometheusCounter = TypedPrometheusCounter<u64>;
type PrometheusCounterFn = fn() -> PrometheusCounter;
type PrometheusGauge = TypedPrometheusGauge<f64, AtomicU64>;
type PrometheusGaugeFn = fn() -> PrometheusGauge;
type PrometheusLabels = Vec<(String, String)>;

pub struct Counter {
    name: String,
    description: Option<String>,
    label_names: Vec<String>,
    unit: Option<Unit>,
    values: HashMap<Vec<String>, u64>,
}

pub struct Gauge {
    name: String,
    description: Option<String>,
    label_names: Vec<String>,
    unit: Option<Unit>,
    values: HashMap<Vec<String>, f64>,
}

pub type SharedCounter = Arc<Mutex<Counter>>;
pub type SharedGauge = Arc<Mutex<Gauge>>;

enum MetricType {
    Counter(SharedCounter),
    Gauge(SharedGauge),
}

lazy_static! {
    static ref REGISTRY: Mutex<HashMap<String, MetricType>> = Mutex::new(HashMap::new());
}

fn build_labels(names: &[String], values: &[String]) -> Vec<(String, String)> {
    names.iter().cloned().zip(values.iter().cloned()).collect()
}

pub fn new_counter(
    name: &str,
    description: Option<&str>,
    label_names: &[&str],
    unit: Option<Unit>,
) -> SharedCounter {
    let mut registry = REGISTRY.lock();
    let counter = Arc::new(Mutex::new(Counter {
        name: name.to_string(),
        description: description.map(|s| s.to_string()),
        label_names: label_names.iter().map(|s| s.to_string()).collect(),
        unit,
        values: HashMap::new(),
    }));
    registry.insert(name.to_string(), MetricType::Counter(counter.clone()));
    counter
}

pub fn get_or_create_counter(
    name: &str,
    description: Option<&str>,
    label_names: &[&str],
    unit: Option<Unit>,
) -> SharedCounter {
    match get_counter(name) {
        Some(counter) => counter,
        None => new_counter(name, description, label_names, unit),
    }
}

pub fn get_counter(name: &str) -> Option<SharedCounter> {
    let registry = REGISTRY.lock();
    match registry.get(name) {
        Some(MetricType::Counter(counter)) => Some(counter.clone()),
        _ => None,
    }
}

pub fn new_gauge(
    name: &str,
    description: Option<&str>,
    label_names: &[&str],
    unit: Option<Unit>,
) -> SharedGauge {
    let mut registry = REGISTRY.lock();
    let gauge = Arc::new(Mutex::new(Gauge {
        name: name.to_string(),
        description: description.map(|s| s.to_string()),
        label_names: label_names.iter().map(|s| s.to_string()).collect(),
        unit,
        values: HashMap::new(),
    }));
    registry.insert(name.to_string(), MetricType::Gauge(gauge.clone()));
    gauge
}

pub fn get_or_create_gauge(
    name: &str,
    description: Option<&str>,
    label_names: &[&str],
    unit: Option<Unit>,
) -> SharedGauge {
    match get_gauge(name) {
        Some(gauge) => gauge,
        None => new_gauge(name, description, label_names, unit),
    }
}

pub fn get_gauge(name: &str) -> Option<SharedGauge> {
    let registry = REGISTRY.lock();
    match registry.get(name) {
        Some(MetricType::Gauge(gauge)) => Some(gauge.clone()),
        _ => None,
    }
}

pub fn del_metric(name: &str) {
    let mut registry = REGISTRY.lock();
    registry.remove(name);
}

fn collect_labels(labels: &[&str]) -> Vec<String> {
    labels.iter().map(|s| s.to_string()).collect()
}

impl Counter {
    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    pub fn get_label_names(&self) -> &[String] {
        &self.label_names
    }

    pub fn get_unit(&self) -> &Option<Unit> {
        &self.unit
    }

    pub fn inc(&mut self, increment: u64, labels: &[&str]) -> anyhow::Result<u64> {
        let labels = collect_labels(labels);
        if labels.len() != self.label_names.len() {
            bail!("Invalid labels: {:?} != {:?}", &labels, &self.label_names);
        }
        let counter = self.values.entry(labels).or_insert(0);
        let last_value = *counter;
        *counter += increment;
        Ok(last_value)
    }

    pub fn set(&mut self, value: u64, labels: &[&str]) -> anyhow::Result<u64> {
        let labels = collect_labels(labels);
        if labels.len() != self.label_names.len() {
            bail!("Invalid labels: {:?} != {:?}", &labels, &self.label_names);
        }
        let counter = self.values.entry(labels).or_insert(value);
        let last_value = *counter;
        *counter = value;
        Ok(last_value)
    }

    pub fn get(&self, labels: &[&str]) -> anyhow::Result<Option<u64>> {
        let labels = collect_labels(labels);
        if labels.len() != self.label_names.len() {
            bail!("Invalid labels: {:?} != {:?}", &labels, &self.label_names);
        }
        Ok(self.values.get(&labels).cloned())
    }

    pub fn delete(&mut self, labels: &[&str]) -> anyhow::Result<Option<u64>> {
        let labels = collect_labels(labels);
        if labels.len() != self.label_names.len() {
            bail!("Invalid labels: {:?} != {:?}", &labels, &self.label_names);
        }
        Ok(self.values.remove(&labels))
    }

    pub fn get_all(&self) -> &HashMap<Vec<String>, u64> {
        &self.values
    }

    pub fn export(&self) -> Family<PrometheusLabels, PrometheusCounter, PrometheusCounterFn> {
        let fam = Family::<PrometheusLabels, PrometheusCounter>::default();
        for (labels, value) in &self.values {
            let label_map = build_labels(&self.label_names, &labels);
            let c = fam.get_or_create(&label_map);
            c.inc_by(*value);
        }
        fam
    }
}

impl Gauge {
    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    pub fn get_label_names(&self) -> &[String] {
        &self.label_names
    }

    pub fn get_unit(&self) -> &Option<Unit> {
        &self.unit
    }

    pub fn set(&mut self, value: f64, labels: &[&str]) -> anyhow::Result<f64> {
        let labels = collect_labels(labels);
        if labels.len() != self.label_names.len() {
            bail!("Invalid labels: {:?} != {:?}", &labels, &self.label_names);
        }
        let gauge = self.values.entry(labels).or_insert(value);
        let last_value = *gauge;
        *gauge = value;
        Ok(last_value)
    }

    pub fn get(&self, labels: &[&str]) -> anyhow::Result<Option<f64>> {
        let labels = collect_labels(labels);
        if labels.len() != self.label_names.len() {
            bail!("Invalid labels: {:?} != {:?}", &labels, &self.label_names);
        }
        Ok(self.values.get(&labels).cloned())
    }

    pub fn delete(&mut self, labels: &[&str]) -> anyhow::Result<Option<f64>> {
        let labels = collect_labels(labels);
        if labels.len() != self.label_names.len() {
            bail!("Invalid labels: {:?} != {:?}", &labels, &self.label_names);
        }
        Ok(self.values.remove(&labels))
    }

    pub fn get_all(&self) -> &HashMap<Vec<String>, f64> {
        &self.values
    }

    pub fn export(&self) -> Family<PrometheusLabels, PrometheusGauge, PrometheusGaugeFn> {
        let fam = Family::<PrometheusLabels, PrometheusGauge>::default();
        for (labels, value) in &self.values {
            let label_map = build_labels(&self.label_names, &labels);
            let g = fam.get_or_create(&label_map);
            g.set(*value);
        }
        fam
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Labels {
    labels: HashMap<String, String>,
}

// impl EncodeLabelSet for Labels {
//     fn encode(&self, encoder: &mut LabelSetEncoder) -> Result<(), std::fmt::Error> {
//         for (k, v) in &self.labels {
//             let mut le = encoder.encode_label();
//             let mut lke = le.encode_label_key()?;
//             EncodeLabelKey::encode(&k, &mut lke)?;
//             let mut lve = lke.encode_label_value()?;
//             EncodeLabelValue::encode(&v, &mut lve)?;
//             lve.finish()?;
//         }
//         Ok(())
//     }
// }

pub enum ConstMetric {
    Counter(Family<PrometheusLabels, PrometheusCounter, PrometheusCounterFn>),
    Gauge(Family<PrometheusLabels, PrometheusGauge, PrometheusGaugeFn>),
}

pub struct MetricExport {
    pub name: String,
    pub description: Option<String>,
    pub unit: Option<Unit>,
    pub metric: ConstMetric,
}

pub fn export_metrics() -> Vec<MetricExport> {
    let registry = REGISTRY.lock();
    registry
        .iter()
        .map(|(name, metric)| match metric {
            MetricType::Counter(shared_counter) => {
                let counter = shared_counter.lock();
                MetricExport {
                    name: name.clone(),
                    description: counter.get_description().map(|s| s.to_string()),
                    unit: counter.get_unit().clone(),
                    metric: ConstMetric::Counter(counter.export()),
                }
            }
            MetricType::Gauge(shared_gauge) => {
                let gauge = shared_gauge.lock();
                MetricExport {
                    name: name.clone(),
                    description: gauge.get_description().map(|s| s.to_string()),
                    unit: gauge.get_unit().clone(),
                    metric: ConstMetric::Gauge(gauge.export()),
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn test_new_counter() -> anyhow::Result<()> {
        let shared_counter = new_counter(
            "test_counter",
            Some("Test counter"),
            &["label1", "label2"],
            None,
        );
        let mut counter = shared_counter.lock();
        assert_eq!(counter.get_name(), "test_counter");
        assert_eq!(counter.get_description(), Some("Test counter"));
        assert_eq!(
            counter.get_label_names(),
            &["label1".to_string(), "label2".to_string()]
        );
        let last = counter.inc(1, &["a", "b"])?;
        assert_eq!(last, 0);
        assert_eq!(counter.get(&["a", "b"])?, Some(1));
        let last = counter.set(20, &["a", "b"])?;
        assert_eq!(last, 1);
        assert_eq!(counter.get(&["a", "b"])?, Some(20));
        let last = counter.delete(&["a", "b"])?;
        assert_eq!(last, Some(20));
        let last = counter.delete(&["a", "b"])?;
        assert_eq!(last, None);
        counter.inc(1, &["a", "b"])?;
        counter.inc(2, &["c", "d"])?;
        let counters = counter.get_all();
        assert_eq!(counters.len(), 2);
        assert_eq!(counters.values().sum::<u64>(), 3);
        del_metric("test_counter");
        Ok(())
    }

    #[test]
    #[serial_test::serial]
    fn test_counter_wrong_labels() -> anyhow::Result<()> {
        let shared_counter = new_counter(
            "test_counter",
            Some("Test counter"),
            &["label1", "label2"],
            None,
        );
        let mut counter = shared_counter.lock();
        let err = counter.inc(1, &["a"]);
        assert!(err.is_err());
        let err = counter.set(1, &["a"]);
        assert!(err.is_err());
        let err = counter.get(&["a"]);
        assert!(err.is_err());
        let err = counter.delete(&["a"]);
        assert!(err.is_err());
        del_metric("test_counter");
        Ok(())
    }

    #[test]
    #[serial_test::serial]
    fn test_new_gauge() -> anyhow::Result<()> {
        let shared_gauge = new_gauge(
            "test_gauge",
            Some("Test gauge"),
            &["label1", "label2"],
            None,
        );
        let mut gauge = shared_gauge.lock();
        assert_eq!(gauge.get_name(), "test_gauge");
        assert_eq!(gauge.get_description(), Some("Test gauge"));
        assert_eq!(
            gauge.get_label_names(),
            &["label1".to_string(), "label2".to_string()]
        );
        let last = gauge.set(1.0, &["a", "b"])?;
        assert_eq!(last, 1.0);
        assert_eq!(gauge.get(&["a", "b"])?, Some(1.0));
        let last = gauge.set(20.0, &["a", "b"])?;
        assert_eq!(last, 1.0);
        assert_eq!(gauge.get(&["a", "b"])?, Some(20.0));
        let last = gauge.delete(&["a", "b"])?;
        assert_eq!(last, Some(20.0));
        let last = gauge.delete(&["a", "b"])?;
        assert_eq!(last, None);
        gauge.set(1.0, &["a", "b"])?;
        gauge.set(2.0, &["c", "d"])?;
        let gauges = gauge.get_all();
        assert_eq!(gauges.len(), 2);
        assert_eq!(gauges.values().sum::<f64>(), 3.0);
        del_metric("test_gauge");
        Ok(())
    }

    #[test]
    #[serial_test::serial]
    fn test_gauge_wrong_labels() -> anyhow::Result<()> {
        let shared_gauge = new_gauge(
            "test_gauge",
            Some("Test gauge"),
            &["label1", "label2"],
            None,
        );
        let mut gauge = shared_gauge.lock();
        let err = gauge.set(1.0, &["a"]);
        assert!(err.is_err());
        let err = gauge.get(&["a"]);
        assert!(err.is_err());
        let err = gauge.delete(&["a"]);
        assert!(err.is_err());
        del_metric("test_gauge");
        Ok(())
    }
}

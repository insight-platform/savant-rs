use anyhow::bail;
use hashbrown::HashMap;
use lazy_static::lazy_static;
use parking_lot::Mutex;
use prometheus_client::encoding::{EncodeMetric, MetricEncoder, NoLabelSet};
use prometheus_client::metrics::counter::Counter as TypedPrometheusCounter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge as TypedPrometheusGauge;
use prometheus_client::metrics::MetricType as PrometheusMetricType;
use prometheus_client::registry::Unit;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock};

pub(crate) mod metric_collector;
pub(crate) mod pipeline_metric_builder;

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

/// Default bucket upper bounds, the same as the Prometheus Go client uses.
pub const DEFAULT_BUCKETS: [f64; 11] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// State of a single labelled histogram.
///
/// `buckets` holds the raw storage layout, the same one `prometheus_client`
/// uses: counts are **non-cumulative** and the last bucket has the upper bound
/// `f64::MAX`, which stands for `+Inf`. Use [`HistogramValue::cumulative_buckets`]
/// for the Prometheus view (cumulative counts, last bound `f64::INFINITY`),
/// which is also what the Python binding returns.
#[derive(Clone, Debug, PartialEq)]
pub struct HistogramValue {
    pub sum: f64,
    pub count: u64,
    pub buckets: Vec<(f64, u64)>,
}

impl HistogramValue {
    /// Returns `(upper_bound, cumulative_count)` pairs as Prometheus reports
    /// them; the last upper bound is `f64::INFINITY` and its count equals `count`.
    pub fn cumulative_buckets(&self) -> Vec<(f64, u64)> {
        let mut cumulative = 0;
        self.buckets
            .iter()
            .map(|&(upper_bound, count)| {
                cumulative += count;
                let upper_bound = if upper_bound == f64::MAX {
                    f64::INFINITY
                } else {
                    upper_bound
                };
                (upper_bound, cumulative)
            })
            .collect()
    }
}

pub struct Histogram {
    name: String,
    description: Option<String>,
    label_names: Vec<String>,
    unit: Option<Unit>,
    bounds: Vec<f64>,
    values: HashMap<Vec<String>, HistogramValue>,
}

pub type SharedCounterFamily = Arc<Mutex<Counter>>;
pub type SharedGaugeFamily = Arc<Mutex<Gauge>>;
pub type SharedHistogramFamily = Arc<Mutex<Histogram>>;

enum MetricType {
    Counter(SharedCounterFamily),
    Gauge(SharedGaugeFamily),
    Histogram(SharedHistogramFamily),
}

lazy_static! {
    static ref REGISTRY: Mutex<HashMap<String, MetricType>> = Mutex::new(HashMap::new());
    static ref EXTRA_LABELS: OnceLock<HashMap<String, String>> = OnceLock::new();
}

pub fn set_extra_labels(labels: HashMap<String, String>) {
    EXTRA_LABELS.get_or_init(|| labels);
}

fn build_labels(names: &[String], values: &[String]) -> Vec<(String, String)> {
    let labels = names
        .iter()
        .cloned()
        .zip(values.iter().cloned())
        .collect::<Vec<(_, _)>>();
    let extra_labels = EXTRA_LABELS.get();
    if let Some(el) = extra_labels {
        labels
            .into_iter()
            .chain(el.iter().map(|(a, b)| (a.to_string(), b.to_string())))
            .collect()
    } else {
        labels
    }
}

pub fn new_counter(
    name: &str,
    description: Option<&str>,
    label_names: &[&str],
    unit: Option<Unit>,
) -> SharedCounterFamily {
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

pub fn get_or_create_counter_family(
    name: &str,
    description: Option<&str>,
    label_names: &[&str],
    unit: Option<Unit>,
) -> SharedCounterFamily {
    match get_counter_family(name) {
        Some(counter) => counter,
        None => new_counter(name, description, label_names, unit),
    }
}

pub fn get_counter_family(name: &str) -> Option<SharedCounterFamily> {
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
) -> SharedGaugeFamily {
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

pub fn get_or_create_gauge_family(
    name: &str,
    description: Option<&str>,
    label_names: &[&str],
    unit: Option<Unit>,
) -> SharedGaugeFamily {
    match get_gauge_family(name) {
        Some(gauge) => gauge,
        None => new_gauge(name, description, label_names, unit),
    }
}

pub fn get_gauge_family(name: &str) -> Option<SharedGaugeFamily> {
    let registry = REGISTRY.lock();
    match registry.get(name) {
        Some(MetricType::Gauge(gauge)) => Some(gauge.clone()),
        _ => None,
    }
}

fn validate_histogram(label_names: &[&str], bounds: &[f64]) -> anyhow::Result<()> {
    if label_names.contains(&"le") {
        bail!("Label name 'le' is reserved for histogram buckets");
    }
    // f64::MAX is reserved as the implicit +Inf bucket bound.
    if bounds.iter().any(|b| !b.is_finite() || *b == f64::MAX) {
        bail!(
            "Histogram bucket bounds must be finite and less than f64::MAX: {:?}",
            bounds
        );
    }
    if bounds.windows(2).any(|w| w[0] >= w[1]) {
        bail!(
            "Histogram bucket bounds must be strictly increasing: {:?}",
            bounds
        );
    }
    Ok(())
}

pub fn new_histogram(
    name: &str,
    description: Option<&str>,
    label_names: &[&str],
    buckets: Option<&[f64]>,
    unit: Option<Unit>,
) -> anyhow::Result<SharedHistogramFamily> {
    let bounds = buckets.unwrap_or(&DEFAULT_BUCKETS).to_vec();
    validate_histogram(label_names, &bounds)?;
    let mut registry = REGISTRY.lock();
    let histogram = Arc::new(Mutex::new(Histogram {
        name: name.to_string(),
        description: description.map(|s| s.to_string()),
        label_names: label_names.iter().map(|s| s.to_string()).collect(),
        unit,
        bounds,
        values: HashMap::new(),
    }));
    registry.insert(name.to_string(), MetricType::Histogram(histogram.clone()));
    Ok(histogram)
}

pub fn get_or_create_histogram_family(
    name: &str,
    description: Option<&str>,
    label_names: &[&str],
    buckets: Option<&[f64]>,
    unit: Option<Unit>,
) -> anyhow::Result<SharedHistogramFamily> {
    match get_histogram_family(name) {
        Some(histogram) => Ok(histogram),
        None => new_histogram(name, description, label_names, buckets, unit),
    }
}

pub fn get_histogram_family(name: &str) -> Option<SharedHistogramFamily> {
    let registry = REGISTRY.lock();
    match registry.get(name) {
        Some(MetricType::Histogram(histogram)) => Some(histogram.clone()),
        _ => None,
    }
}

pub fn delete_metric_family(name: &str) {
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

    pub fn inc(&mut self, increment: u64, label_values: &[&str]) -> anyhow::Result<u64> {
        let labels = collect_labels(label_values);
        if labels.len() != self.label_names.len() {
            bail!("Invalid labels: {:?} != {:?}", &labels, &self.label_names);
        }
        let counter = self.values.entry(labels).or_insert(0);
        let last_value = *counter;
        *counter += increment;
        Ok(last_value)
    }

    pub fn set(&mut self, value: u64, label_values: &[&str]) -> anyhow::Result<u64> {
        let labels = collect_labels(label_values);
        if labels.len() != self.label_names.len() {
            bail!("Invalid labels: {:?} != {:?}", &labels, &self.label_names);
        }
        let counter = self.values.entry(labels).or_insert(value);
        let last_value = *counter;
        *counter = value;
        Ok(last_value)
    }

    pub fn get(&self, label_values: &[&str]) -> anyhow::Result<Option<u64>> {
        let labels = collect_labels(label_values);
        if labels.len() != self.label_names.len() {
            bail!("Invalid labels: {:?} != {:?}", &labels, &self.label_names);
        }
        Ok(self.values.get(&labels).cloned())
    }

    pub fn delete(&mut self, label_values: &[&str]) -> anyhow::Result<Option<u64>> {
        let labels = collect_labels(label_values);
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
            let label_map = build_labels(&self.label_names, labels);
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

    pub fn set(&mut self, value: f64, label_values: &[&str]) -> anyhow::Result<f64> {
        let labels = collect_labels(label_values);
        if labels.len() != self.label_names.len() {
            bail!("Invalid labels: {:?} != {:?}", &labels, &self.label_names);
        }
        let gauge = self.values.entry(labels).or_insert(value);
        let last_value = *gauge;
        *gauge = value;
        Ok(last_value)
    }

    pub fn get(&self, label_values: &[&str]) -> anyhow::Result<Option<f64>> {
        let labels = collect_labels(label_values);
        if labels.len() != self.label_names.len() {
            bail!("Invalid labels: {:?} != {:?}", &labels, &self.label_names);
        }
        Ok(self.values.get(&labels).cloned())
    }

    pub fn delete(&mut self, label_values: &[&str]) -> anyhow::Result<Option<f64>> {
        let labels = collect_labels(label_values);
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
            let label_map = build_labels(&self.label_names, labels);
            let g = fam.get_or_create(&label_map);
            g.set(*value);
        }
        fam
    }
}

impl Histogram {
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

    pub fn get_buckets(&self) -> &[f64] {
        &self.bounds
    }

    pub fn observe(&mut self, value: f64, label_values: &[&str]) -> anyhow::Result<()> {
        // An infinite value would land in no bucket and poison the sum forever.
        if !value.is_finite() {
            bail!("Cannot observe non-finite value: {}", value);
        }
        let labels = collect_labels(label_values);
        if labels.len() != self.label_names.len() {
            bail!("Invalid labels: {:?} != {:?}", &labels, &self.label_names);
        }
        let bounds = &self.bounds;
        let histogram = self.values.entry(labels).or_insert_with(|| HistogramValue {
            sum: 0.0,
            count: 0,
            buckets: bounds
                .iter()
                .copied()
                .chain(std::iter::once(f64::MAX))
                .map(|upper_bound| (upper_bound, 0))
                .collect(),
        });
        histogram.sum += value;
        histogram.count += 1;
        if let Some((_, count)) = histogram
            .buckets
            .iter_mut()
            .find(|(upper_bound, _)| *upper_bound >= value)
        {
            *count += 1;
        }
        Ok(())
    }

    /// Returns the raw (non-cumulative) state; see [`HistogramValue::cumulative_buckets`].
    pub fn get(&self, label_values: &[&str]) -> anyhow::Result<Option<HistogramValue>> {
        let labels = collect_labels(label_values);
        if labels.len() != self.label_names.len() {
            bail!("Invalid labels: {:?} != {:?}", &labels, &self.label_names);
        }
        Ok(self.values.get(&labels).cloned())
    }

    pub fn delete(&mut self, label_values: &[&str]) -> anyhow::Result<Option<HistogramValue>> {
        let labels = collect_labels(label_values);
        if labels.len() != self.label_names.len() {
            bail!("Invalid labels: {:?} != {:?}", &labels, &self.label_names);
        }
        Ok(self.values.remove(&labels))
    }

    pub fn get_all(&self) -> &HashMap<Vec<String>, HistogramValue> {
        &self.values
    }

    pub fn export(&self) -> HistogramExport {
        HistogramExport(
            self.values
                .iter()
                .map(|(labels, value)| (build_labels(&self.label_names, labels), value.clone()))
                .collect(),
        )
    }
}

/// Snapshot of a histogram family, encoded directly because a
/// `prometheus_client` histogram cannot be rebuilt from stored sums and counts.
pub struct HistogramExport(Vec<(PrometheusLabels, HistogramValue)>);

impl EncodeMetric for HistogramExport {
    fn encode(&self, mut encoder: MetricEncoder) -> Result<(), std::fmt::Error> {
        for (labels, value) in &self.0 {
            encoder
                .encode_family(labels)?
                .encode_histogram::<NoLabelSet>(value.sum, value.count, &value.buckets, None)?;
        }
        Ok(())
    }

    fn metric_type(&self) -> PrometheusMetricType {
        PrometheusMetricType::Histogram
    }
}

pub enum ConstMetric {
    Counter(Family<PrometheusLabels, PrometheusCounter, PrometheusCounterFn>),
    Gauge(Family<PrometheusLabels, PrometheusGauge, PrometheusGaugeFn>),
    Histogram(HistogramExport),
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
            MetricType::Histogram(shared_histogram) => {
                let histogram = shared_histogram.lock();
                MetricExport {
                    name: name.clone(),
                    description: histogram.get_description().map(|s| s.to_string()),
                    unit: histogram.get_unit().clone(),
                    metric: ConstMetric::Histogram(histogram.export()),
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
        delete_metric_family("test_counter");
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
        delete_metric_family("test_counter");
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
        delete_metric_family("test_gauge");
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
        delete_metric_family("test_gauge");
        Ok(())
    }

    #[test]
    #[serial_test::serial]
    fn test_new_histogram() -> anyhow::Result<()> {
        let shared_histogram = new_histogram(
            "test_histogram",
            Some("Test histogram"),
            &["label1", "label2"],
            Some(&[1.0, 5.0, 10.0]),
            None,
        )?;
        let mut histogram = shared_histogram.lock();
        assert_eq!(histogram.get_name(), "test_histogram");
        assert_eq!(histogram.get_description(), Some("Test histogram"));
        assert_eq!(
            histogram.get_label_names(),
            &["label1".to_string(), "label2".to_string()]
        );
        assert_eq!(histogram.get_buckets(), &[1.0, 5.0, 10.0]);
        assert_eq!(histogram.get(&["a", "b"])?, None);
        histogram.observe(0.5, &["a", "b"])?;
        histogram.observe(1.0, &["a", "b"])?;
        histogram.observe(7.0, &["a", "b"])?;
        histogram.observe(100.0, &["a", "b"])?;
        let value = histogram.get(&["a", "b"])?.unwrap();
        assert_eq!(value.sum, 108.5);
        assert_eq!(value.count, 4);
        assert_eq!(
            value.buckets,
            vec![(1.0, 2), (5.0, 0), (10.0, 1), (f64::MAX, 1)]
        );
        assert_eq!(
            value.cumulative_buckets(),
            vec![(1.0, 2), (5.0, 2), (10.0, 3), (f64::INFINITY, 4)]
        );
        histogram.observe(2.0, &["c", "d"])?;
        assert_eq!(histogram.get_all().len(), 2);
        let deleted = histogram.delete(&["a", "b"])?;
        assert_eq!(deleted, Some(value));
        assert_eq!(histogram.delete(&["a", "b"])?, None);
        drop(histogram);
        assert!(get_histogram_family("test_histogram").is_some());
        delete_metric_family("test_histogram");
        assert!(get_histogram_family("test_histogram").is_none());
        Ok(())
    }

    #[test]
    #[serial_test::serial]
    fn test_histogram_default_buckets() -> anyhow::Result<()> {
        let shared_histogram =
            get_or_create_histogram_family("test_histogram", None, &[], None, None)?;
        assert_eq!(shared_histogram.lock().get_buckets(), &DEFAULT_BUCKETS);
        delete_metric_family("test_histogram");
        Ok(())
    }

    #[test]
    #[serial_test::serial]
    fn test_histogram_wrong_labels() -> anyhow::Result<()> {
        let shared_histogram = new_histogram(
            "test_histogram",
            Some("Test histogram"),
            &["label1", "label2"],
            None,
            None,
        )?;
        let mut histogram = shared_histogram.lock();
        assert!(histogram.observe(1.0, &["a"]).is_err());
        assert!(histogram.observe(f64::NAN, &["a", "b"]).is_err());
        assert!(histogram.observe(f64::INFINITY, &["a", "b"]).is_err());
        assert!(histogram.observe(f64::NEG_INFINITY, &["a", "b"]).is_err());
        assert_eq!(histogram.get(&["a", "b"])?, None);
        assert!(histogram.get(&["a"]).is_err());
        assert!(histogram.delete(&["a"]).is_err());
        delete_metric_family("test_histogram");
        Ok(())
    }

    #[test]
    #[serial_test::serial]
    fn test_histogram_invalid_config() {
        assert!(new_histogram("test_histogram", None, &["le"], None, None).is_err());
        assert!(new_histogram("test_histogram", None, &[], Some(&[2.0, 1.0]), None).is_err());
        assert!(new_histogram("test_histogram", None, &[], Some(&[1.0, 1.0]), None).is_err());
        assert!(new_histogram(
            "test_histogram",
            None,
            &[],
            Some(&[1.0, f64::INFINITY]),
            None
        )
        .is_err());
        assert!(new_histogram("test_histogram", None, &[], Some(&[1.0, f64::MAX]), None).is_err());
        assert!(get_histogram_family("test_histogram").is_none());
    }
}

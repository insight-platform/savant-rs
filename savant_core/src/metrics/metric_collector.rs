use crate::metrics::{export_metrics, ConstMetric};
use prometheus_client::collector::Collector;
use prometheus_client::encoding::{DescriptorEncoder, EncodeMetric};
use prometheus_client::metrics::MetricType;

#[derive(Debug)]
pub struct SystemMetricCollector;

impl Collector for SystemMetricCollector {
    fn encode(&self, mut encoder: DescriptorEncoder) -> Result<(), std::fmt::Error> {
        let exported_user_metrics = export_metrics();
        for m in exported_user_metrics {
            let (name, description, unit, metric) = (m.name, m.description, m.unit, m.metric);
            let desc_str = description.unwrap_or("".to_string());
            match metric {
                ConstMetric::Counter(c) => {
                    let metric_encoder = encoder.encode_descriptor(
                        &name,
                        &desc_str,
                        unit.as_ref(),
                        MetricType::Counter,
                    )?;

                    c.encode(metric_encoder)?;
                }
                ConstMetric::Gauge(g) => {
                    let metric_encoder = encoder.encode_descriptor(
                        &name,
                        &desc_str,
                        unit.as_ref(),
                        MetricType::Gauge,
                    )?;
                    g.encode(metric_encoder)?;
                }
            }
        }
        Ok(())
    }
}

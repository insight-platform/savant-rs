use opentelemetry::global;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{Config, TracerProvider};
use opentelemetry_sdk::Resource;

pub fn init_jaeger_tracer(service_name: &str, endpoint: &str) {
    global::set_text_map_propagator(opentelemetry_jaeger::Propagator::new());
    opentelemetry_jaeger::new_agent_pipeline()
        .with_endpoint(endpoint)
        .with_service_name(service_name)
        .with_trace_config(Config::default().with_resource(Resource::new(vec![
            opentelemetry::KeyValue::new("service.namespace", "savant-core"),
            opentelemetry::KeyValue::new("exporter", "jaeger"),
        ])))
        .install_simple()
        .expect("Failed to install Jaeger tracer globally");
}

pub fn init_noop_tracer() {
    let exporter = opentelemetry_stdout::SpanExporter::builder()
        .with_writer(std::io::sink())
        .build();
    let p = TracerProvider::builder()
        .with_simple_exporter(exporter)
        .build();
    global::set_tracer_provider(p);
    global::set_text_map_propagator(TraceContextPropagator::new());
}

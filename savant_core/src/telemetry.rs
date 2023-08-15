use opentelemetry::global;
pub fn init_jaeger_tracer(service_name: &str, endpoint: &str) {
    global::set_text_map_propagator(opentelemetry_jaeger::Propagator::new());
    opentelemetry_jaeger::new_agent_pipeline()
        .with_endpoint(endpoint)
        .with_service_name(service_name)
        .with_trace_config(opentelemetry::sdk::trace::config().with_resource(
            opentelemetry::sdk::Resource::new(vec![
                opentelemetry::KeyValue::new("service.name", "savant"), // this will not override the trace-udp-demo
                opentelemetry::KeyValue::new("service.namespace", "savant-core"),
                opentelemetry::KeyValue::new("exporter", "jaeger"),
            ]),
        ))
        .install_simple()
        .expect("Failed to install Jaeger tracer globally");
}

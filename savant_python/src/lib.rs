use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pymodule;

use savant_core_py::atomic_counter::AtomicCounter;
use savant_core_py::draw_spec::*;
use savant_core_py::logging::*;
use savant_core_py::match_query::*;
use savant_core_py::metrics::*;
use savant_core_py::pipeline::{
    load_stage_function_plugin, FrameProcessingStatRecord, FrameProcessingStatRecordType, Pipeline,
    PipelineConfiguration, StageFunction, StageLatencyMeasurements, StageLatencyStat,
    StageProcessingStat, VideoPipelineStagePayloadType,
};
use savant_core_py::primitives::attribute::Attribute;
use savant_core_py::primitives::attribute_value::{
    AttributeValue, AttributeValueType, AttributeValuesView,
};
use savant_core_py::primitives::batch::VideoFrameBatch;
use savant_core_py::primitives::bbox::utils::*;
use savant_core_py::primitives::bbox::{
    BBox, BBoxMetricType, RBBox, VideoObjectBBoxTransformation,
};
use savant_core_py::primitives::eos::EndOfStream;
use savant_core_py::primitives::frame::{
    VideoFrame, VideoFrameContent, VideoFrameTranscodingMethod, VideoFrameTransformation,
};
use savant_core_py::primitives::frame_update::{
    AttributeUpdatePolicy, ObjectUpdatePolicy, VideoFrameUpdate,
};
use savant_core_py::primitives::message::loader::*;
use savant_core_py::primitives::message::saver::*;
use savant_core_py::primitives::message::*;
use savant_core_py::primitives::object::object_tree::VideoObjectTree;
use savant_core_py::primitives::object::{
    BorrowedVideoObject, IdCollisionResolutionPolicy, VideoObject,
};
use savant_core_py::primitives::objects_view::{
    QueryFunctions, VideoObjectBBoxType, VideoObjectsView,
};
use savant_core_py::primitives::point::Point;
use savant_core_py::primitives::polygonal_area::PolygonalArea;
use savant_core_py::primitives::segment::{Intersection, IntersectionKind, Segment};
use savant_core_py::primitives::shutdown::Shutdown;
use savant_core_py::primitives::user_data::UserData;
use savant_core_py::telemetry::*;
use savant_core_py::test::utils::*;
use savant_core_py::utils::byte_buffer::ByteBuffer;
use savant_core_py::utils::eval_resolvers::*;
use savant_core_py::utils::otlp::*;
use savant_core_py::utils::symbol_mapper::*;
use savant_core_py::utils::*;
use savant_core_py::webserver::kvs::*;
use savant_core_py::webserver::*;
use savant_core_py::zmq::basic_types::{ReaderSocketType, TopicPrefixSpec, WriterSocketType};
use savant_core_py::zmq::configs::{
    ReaderConfig, ReaderConfigBuilder, WriterConfig, WriterConfigBuilder,
};
use savant_core_py::zmq::results::{
    ReaderResultBlacklisted, ReaderResultMessage, ReaderResultPrefixMismatch, ReaderResultTimeout,
    WriterResultAck, WriterResultAckTimeout, WriterResultSendTimeout, WriterResultSuccess,
};
use savant_core_py::zmq::{blocking, nonblocking};
use savant_core_py::*;

#[pymodule(gil_used = false)]
pub fn metrics(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CounterFamily>()?;
    m.add_class::<GaugeFamily>()?;
    m.add_function(wrap_pyfunction!(delete_metric_family, m)?)?;
    m.add_function(wrap_pyfunction!(set_extra_labels, m)?)?;
    Ok(())
}

#[pymodule(gil_used = false)]
pub fn gstreamer(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<gst::FlowResult>()?;
    m.add_class::<gst::InvocationReason>()?;
    Ok(())
}

#[pymodule(gil_used = false)]
pub fn webserver(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_webserver, m)?)?;
    m.add_function(wrap_pyfunction!(stop_webserver, m)?)?;
    m.add_function(wrap_pyfunction!(set_shutdown_token, m)?)?;
    m.add_function(wrap_pyfunction!(is_shutdown_set, m)?)?;
    m.add_function(wrap_pyfunction!(set_status_running, m)?)?;
    m.add_function(wrap_pyfunction!(set_shutdown_signal, m)?)?;
    m.add_wrapped(wrap_pymodule!(self::kvs))?;
    Ok(())
}

#[pymodule(gil_used = false)]
pub fn kvs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_attributes, m)?)?;
    m.add_function(wrap_pyfunction!(get_attribute, m)?)?;
    m.add_function(wrap_pyfunction!(search_attributes, m)?)?;
    m.add_function(wrap_pyfunction!(search_keys, m)?)?;
    m.add_function(wrap_pyfunction!(del_attributes, m)?)?;
    m.add_function(wrap_pyfunction!(del_attribute, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_attributes, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize_attributes, m)?)?;

    m.add_class::<KvsSubscription>()?;
    m.add_class::<KvsSetOperation>()?;
    m.add_class::<KvsDeleteOperation>()?;

    Ok(())
}

#[pymodule(gil_used = false)]
pub fn zmq(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WriterSocketType>()?;
    m.add_class::<WriterConfigBuilder>()?;
    m.add_class::<WriterConfig>()?;
    m.add_class::<WriterResultSendTimeout>()?;
    m.add_class::<WriterResultAckTimeout>()?;
    m.add_class::<WriterResultAck>()?;
    m.add_class::<WriterResultSuccess>()?;

    m.add_class::<blocking::BlockingWriter>()?;
    m.add_class::<nonblocking::NonBlockingWriter>()?;
    m.add_class::<nonblocking::WriteOperationResult>()?;

    m.add_class::<ReaderSocketType>()?;
    m.add_class::<TopicPrefixSpec>()?;
    m.add_class::<ReaderConfigBuilder>()?;
    m.add_class::<ReaderConfig>()?;
    m.add_class::<ReaderResultMessage>()?;
    m.add_class::<ReaderResultBlacklisted>()?;
    m.add_class::<ReaderResultTimeout>()?;
    m.add_class::<ReaderResultPrefixMismatch>()?;

    m.add_class::<blocking::BlockingReader>()?;
    m.add_class::<nonblocking::NonBlockingReader>()?;

    Ok(())
}

#[pymodule(gil_used = false)]
pub fn symbol_mapper(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_model_object_key_py, m)?)?;
    m.add_function(wrap_pyfunction!(clear_symbol_maps_py, m)?)?;
    m.add_function(wrap_pyfunction!(dump_registry_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_model_id_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_model_name_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_object_id_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_object_ids_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_object_label_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_object_labels_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_model_registered_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_object_registered_py, m)?)?;
    m.add_function(wrap_pyfunction!(parse_compound_key_py, m)?)?;
    m.add_function(wrap_pyfunction!(register_model_objects_py, m)?)?;
    m.add_function(wrap_pyfunction!(validate_base_key_py, m)?)?;

    m.add_class::<RegistrationPolicy>()?;

    Ok(())
}

#[pymodule(gil_used = false)]
pub fn serialization(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ser deser
    m.add_function(wrap_pyfunction!(save_message_gil, m)?)?;
    m.add_function(wrap_pyfunction!(save_message_to_bytebuffer_gil, m)?)?;
    m.add_function(wrap_pyfunction!(save_message_to_bytes_gil, m)?)?;

    m.add_function(wrap_pyfunction!(load_message_gil, m)?)?;
    m.add_function(wrap_pyfunction!(load_message_from_bytebuffer_gil, m)?)?;
    m.add_function(wrap_pyfunction!(load_message_from_bytes_gil, m)?)?;

    m.add_class::<Message>()?;
    m.add_function(wrap_pyfunction!(clear_source_seq_id, m)?)?;
    Ok(())
}

#[pymodule(gil_used = false)]
pub fn utils(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eval_expr, m)?)?;
    m.add_function(wrap_pyfunction!(gen_frame, m)?)?;
    m.add_function(wrap_pyfunction!(gen_empty_frame, m)?)?;
    // utility
    m.add_function(wrap_pyfunction!(round_2_digits, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_gil_contention, m)?)?;
    m.add_function(wrap_pyfunction!(enable_dl_detection, m)?)?;
    m.add_function(wrap_pyfunction!(incremental_uuid_v7, m)?)?;
    m.add_function(wrap_pyfunction!(relative_time_uuid_v7, m)?)?;

    m.add_class::<PropagatedContext>()?;
    m.add_class::<TelemetrySpan>()?;
    m.add_class::<MaybeTelemetrySpan>()?;
    m.add_class::<ByteBuffer>()?;
    m.add_class::<VideoObjectBBoxType>()?;
    m.add_class::<VideoObjectBBoxTransformation>()?;
    m.add_class::<BBoxMetricType>()?;
    m.add_class::<AtomicCounter>()?;

    m.add_wrapped(wrap_pymodule!(self::symbol_mapper))?;
    m.add_wrapped(wrap_pymodule!(self::serialization))?;

    Ok(())
}

#[pymodule(gil_used = false)]
pub fn geometry(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_class::<Segment>()?;
    m.add_class::<IntersectionKind>()?;
    m.add_class::<Intersection>()?;
    m.add_class::<PolygonalArea>()?;
    m.add_class::<RBBox>()?;
    m.add_class::<BBox>()?;

    m.add_function(wrap_pyfunction!(solely_owned_areas, m)?)?;
    m.add_function(wrap_pyfunction!(associate_bboxes, m)?)?;

    Ok(())
}

#[pymodule(gil_used = false)]
pub fn primitives(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Attribute>()?;
    m.add_class::<AttributeUpdatePolicy>()?;
    m.add_class::<ObjectUpdatePolicy>()?;
    m.add_class::<AttributeValue>()?;
    m.add_class::<AttributeValueType>()?;
    m.add_class::<AttributeValuesView>()?;
    m.add_class::<EndOfStream>()?;
    m.add_class::<Shutdown>()?;
    m.add_class::<UserData>()?;

    m.add_class::<VideoFrame>()?;
    m.add_class::<VideoFrameBatch>()?;
    m.add_class::<VideoFrameContent>()?;
    m.add_class::<VideoFrameTranscodingMethod>()?;
    m.add_class::<VideoFrameUpdate>()?;
    m.add_class::<VideoFrameTransformation>()?;

    m.add_class::<BorrowedVideoObject>()?;
    m.add_class::<VideoObject>()?;
    m.add_class::<VideoObjectTree>()?;
    m.add_class::<VideoObjectsView>()?;

    m.add_class::<IdCollisionResolutionPolicy>()?;

    m.add_wrapped(wrap_pymodule!(self::geometry))?;
    Ok(())
}

#[pymodule(gil_used = false)]
pub(crate) fn pipeline(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VideoPipelineStagePayloadType>()?;
    m.add_class::<PipelineConfiguration>()?;
    m.add_class::<Pipeline>()?;
    m.add_class::<FrameProcessingStatRecord>()?;
    m.add_class::<StageLatencyStat>()?;
    m.add_class::<StageProcessingStat>()?;
    m.add_class::<StageLatencyMeasurements>()?;
    m.add_class::<FrameProcessingStatRecordType>()?;
    m.add_class::<StageFunction>()?;
    m.add_function(wrap_pyfunction!(load_stage_function_plugin, m)?)?;
    Ok(())
}

#[pymodule(gil_used = false)]
pub fn match_query(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FloatExpression>()?;
    m.add_class::<IntExpression>()?;
    m.add_class::<StringExpression>()?;
    m.add_class::<MatchQuery>()?;
    m.add_class::<QueryFunctions>()?;
    m.add_class::<EtcdCredentials>()?;
    m.add_class::<TlsConfig>()?;

    m.add_function(wrap_pyfunction!(utility_resolver_name, m)?)?;
    m.add_function(wrap_pyfunction!(etcd_resolver_name, m)?)?;
    m.add_function(wrap_pyfunction!(env_resolver_name, m)?)?;
    m.add_function(wrap_pyfunction!(config_resolver_name, m)?)?;

    m.add_function(wrap_pyfunction!(register_utility_resolver, m)?)?;
    m.add_function(wrap_pyfunction!(register_env_resolver, m)?)?;
    m.add_function(wrap_pyfunction!(register_etcd_resolver, m)?)?;
    m.add_function(wrap_pyfunction!(register_config_resolver, m)?)?;
    m.add_function(wrap_pyfunction!(update_config_resolver, m)?)?;

    m.add_function(wrap_pyfunction!(unregister_resolver, m)?)?;
    Ok(())
}

#[pymodule(gil_used = false)]
pub(crate) fn logging(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LogLevel>()?;
    m.add_function(wrap_pyfunction!(set_log_level, m)?)?;
    m.add_function(wrap_pyfunction!(get_log_level, m)?)?;
    m.add_function(wrap_pyfunction!(log_level_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(log_message_gil, m)?)?;
    Ok(())
}

#[pymodule(gil_used = false)]
pub fn draw_spec(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ColorDraw>()?;
    m.add_class::<BoundingBoxDraw>()?;
    m.add_class::<DotDraw>()?;
    m.add_class::<LabelDraw>()?;
    m.add_class::<LabelPositionKind>()?;
    m.add_class::<LabelPosition>()?;
    m.add_class::<PaddingDraw>()?;
    m.add_class::<ObjectDraw>()?;
    m.add_class::<SetDrawLabelKind>()?;
    Ok(())
}

#[pymodule(gil_used = false)]
pub fn telemetry(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ContextPropagationFormat>()?;
    m.add_class::<Protocol>()?;
    m.add_class::<Identity>()?;
    m.add_class::<ClientTlsConfig>()?;
    m.add_class::<TracerConfiguration>()?;
    m.add_class::<TelemetryConfiguration>()?;
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_function(wrap_pyfunction!(init_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(shutdown, m)?)?;
    Ok(())
}

#[pymodule(gil_used = false)]
fn savant_rs(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_logs(LogLevel::Trace)?;
    init_all(py, m)
}

pub use savant_core_py::logging::LogLevel;

pub fn init_logs(log_level: LogLevel) -> PyResult<()> {
    let log_env_var_name = "LOGLEVEL";
    let log_env_var_level = log_level.__str__();
    if std::env::var(log_env_var_name).is_err() {
        unsafe {
            std::env::set_var(log_env_var_name, log_env_var_level);
        }
    }
    err_to_pyo3!(
        pretty_env_logger::try_init_timed_custom_env(log_env_var_name),
        PyRuntimeError
    )?;

    //set_log_level(LogLevel::Error);
    Ok(())
}

pub fn init_all(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(register_handler, m)?)?;
    m.add_function(wrap_pyfunction!(unregister_handler, m)?)?;

    m.add_wrapped(wrap_pymodule!(self::primitives))?;
    m.add_wrapped(wrap_pymodule!(self::pipeline))?;
    m.add_wrapped(wrap_pymodule!(self::geometry))?;
    m.add_wrapped(wrap_pymodule!(self::draw_spec))?;
    m.add_wrapped(wrap_pymodule!(self::utils))?;
    m.add_wrapped(wrap_pymodule!(self::symbol_mapper))?;
    m.add_wrapped(wrap_pymodule!(self::serialization))?;
    m.add_wrapped(wrap_pymodule!(self::match_query))?;
    m.add_wrapped(wrap_pymodule!(self::logging))?;
    m.add_wrapped(wrap_pymodule!(self::zmq))?;
    m.add_wrapped(wrap_pymodule!(self::telemetry))?;
    m.add_wrapped(wrap_pymodule!(self::webserver))?;
    m.add_wrapped(wrap_pymodule!(self::metrics))?;
    m.add_wrapped(wrap_pymodule!(self::kvs))?;
    m.add_wrapped(wrap_pymodule!(self::gstreamer))?;

    let sys = PyModule::import(py, "sys")?;
    let sys_modules_bind = sys.getattr("modules")?;
    let sys_modules = sys_modules_bind.downcast::<PyDict>()?;

    sys_modules.set_item("savant_rs.gstreamer", m.getattr("gstreamer")?)?;
    sys_modules.set_item("savant_rs.primitives", m.getattr("primitives")?)?;
    sys_modules.set_item("savant_rs.pipeline", m.getattr("pipeline")?)?;
    sys_modules.set_item("savant_rs.pipeline2", m.getattr("pipeline")?)?;

    sys_modules.set_item("savant_rs.primitives.geometry", m.getattr("geometry")?)?;
    sys_modules.set_item("savant_rs.draw_spec", m.getattr("draw_spec")?)?;
    sys_modules.set_item("savant_rs.utils", m.getattr("utils")?)?;
    sys_modules.set_item("savant_rs.logging", m.getattr("logging")?)?;
    sys_modules.set_item("savant_rs.zmq", m.getattr("zmq")?)?;
    sys_modules.set_item("savant_rs.telemetry", m.getattr("telemetry")?)?;

    sys_modules.set_item("savant_rs.webserver", m.getattr("webserver")?)?;
    sys_modules.set_item("savant_rs.webserver.kvs", m.getattr("kvs")?)?;
    sys_modules.set_item("savant_rs.metrics", m.getattr("metrics")?)?;

    sys_modules.set_item("savant_rs.utils.symbol_mapper", m.getattr("symbol_mapper")?)?;

    sys_modules.set_item("savant_rs.utils.serialization", m.getattr("serialization")?)?;

    sys_modules.set_item("savant_rs.match_query", m.getattr("match_query")?)?;

    Ok(())
}

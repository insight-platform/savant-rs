use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pymodule;
use savant_core_py::draw_spec::*;
use savant_core_py::logging::*;
use savant_core_py::match_query::*;
use savant_core_py::pipeline::{
    FrameProcessingStatRecord, FrameProcessingStatRecordType, Pipeline, PipelineConfiguration,
    StageStat, VideoPipelineStagePayloadType,
};
use savant_core_py::primitives::attribute::Attribute;
use savant_core_py::primitives::attribute_value::{
    AttributeValue, AttributeValueType, AttributeValuesView,
};
use savant_core_py::primitives::batch::VideoFrameBatch;
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
use savant_core_py::test::utils::*;
use savant_core_py::utils::byte_buffer::ByteBuffer;
use savant_core_py::utils::eval_resolvers::*;
use savant_core_py::utils::otlp::*;
use savant_core_py::utils::symbol_mapper::*;
use savant_core_py::utils::*;
use savant_core_py::zmq::basic_types::{ReaderSocketType, TopicPrefixSpec, WriterSocketType};
use savant_core_py::zmq::configs::{
    ReaderConfig, ReaderConfigBuilder, WriterConfig, WriterConfigBuilder,
};
use savant_core_py::zmq::results::{
    ReaderResultMessage, ReaderResultPrefixMismatch, ReaderResultTimeout, WriterResultAck,
    WriterResultAckTimeout, WriterResultSendTimeout, WriterResultSuccess,
};
use savant_core_py::zmq::{blocking, nonblocking};
use savant_core_py::*;

#[pymodule]
pub fn zmq(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<WriterSocketType>()?; // PYI
    m.add_class::<WriterConfigBuilder>()?; // PYI
    m.add_class::<WriterConfig>()?; // PYI
    m.add_class::<WriterResultSendTimeout>()?; // PYI
    m.add_class::<WriterResultAckTimeout>()?; // PYI
    m.add_class::<WriterResultAck>()?; // PYI
    m.add_class::<WriterResultSuccess>()?; // PYI

    m.add_class::<blocking::BlockingWriter>()?; // PYI
    m.add_class::<nonblocking::NonBlockingWriter>()?; // PYI
    m.add_class::<nonblocking::WriteOperationResult>()?; // PYI

    m.add_class::<ReaderSocketType>()?; // PYI
    m.add_class::<TopicPrefixSpec>()?; // PYI
    m.add_class::<ReaderConfigBuilder>()?; // PYI
    m.add_class::<ReaderConfig>()?; // PYI
    m.add_class::<ReaderResultMessage>()?; // PYI
    m.add_class::<ReaderResultTimeout>()?; // PYI
    m.add_class::<ReaderResultPrefixMismatch>()?; // PYI

    m.add_class::<blocking::BlockingReader>()?; // PYI
    m.add_class::<nonblocking::NonBlockingReader>()?;

    Ok(())
}

#[pymodule]
pub fn symbol_mapper_module(_py: Python, m: &PyModule) -> PyResult<()> {
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

#[pymodule]
pub fn serialization_module(_py: Python, m: &PyModule) -> PyResult<()> {
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

#[pymodule]
pub fn utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eval_expr, m)?)?; // PYI
    m.add_function(wrap_pyfunction!(gen_frame, m)?)?; // PYI
    m.add_function(wrap_pyfunction!(gen_empty_frame, m)?)?; // PYI
                                                            // utility
    m.add_function(wrap_pyfunction!(round_2_digits, m)?)?; // PYI
    m.add_function(wrap_pyfunction!(estimate_gil_contention, m)?)?; // PYI
    m.add_function(wrap_pyfunction!(enable_dl_detection, m)?)?; // PYI

    m.add_class::<PropagatedContext>()?; // PYI
    m.add_class::<TelemetrySpan>()?; // PYI
    m.add_class::<MaybeTelemetrySpan>()?; // PYI
    m.add_class::<ByteBuffer>()?; // PYI
    m.add_class::<VideoObjectBBoxType>()?; // PYI
    m.add_class::<VideoObjectBBoxTransformation>()?; // PYI
    m.add_class::<BBoxMetricType>()?; // PYI

    Ok(())
}

#[pymodule]
pub fn geometry(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_class::<Segment>()?;
    m.add_class::<IntersectionKind>()?;
    m.add_class::<Intersection>()?;
    m.add_class::<PolygonalArea>()?;
    m.add_class::<RBBox>()?;
    m.add_class::<BBox>()?;
    Ok(())
}

#[pymodule]
pub fn primitives(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Attribute>()?; // PYI
    m.add_class::<AttributeUpdatePolicy>()?; // PYI
    m.add_class::<ObjectUpdatePolicy>()?; // PYI
    m.add_class::<AttributeValue>()?; // PYI
    m.add_class::<AttributeValueType>()?; // PYI
    m.add_class::<AttributeValuesView>()?; // PYI
    m.add_class::<EndOfStream>()?; // PYI
    m.add_class::<Shutdown>()?; // PYI
    m.add_class::<UserData>()?; // PYI

    m.add_class::<VideoFrame>()?; // PYI
    m.add_class::<VideoFrameBatch>()?; // PYI
    m.add_class::<VideoFrameContent>()?; // PYI
    m.add_class::<VideoFrameTranscodingMethod>()?; // PYI
    m.add_class::<VideoFrameUpdate>()?; // PYI
    m.add_class::<VideoFrameTransformation>()?; // PYI

    m.add_class::<BorrowedVideoObject>()?; // PYI
    m.add_class::<VideoObject>()?; // PYI
    m.add_class::<VideoObjectsView>()?; // PYI

    m.add_class::<IdCollisionResolutionPolicy>()?; // PYI

    Ok(())
}

#[pymodule]
pub(crate) fn pipeline(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<VideoPipelineStagePayloadType>()?;
    m.add_class::<PipelineConfiguration>()?;
    m.add_class::<Pipeline>()?;
    m.add_class::<FrameProcessingStatRecord>()?;
    m.add_class::<StageStat>()?;
    m.add_class::<FrameProcessingStatRecordType>()?;
    Ok(())
}

#[pymodule]
pub fn match_query(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FloatExpression>()?;
    m.add_class::<IntExpression>()?;
    m.add_class::<StringExpression>()?;
    m.add_class::<MatchQuery>()?;
    m.add_class::<QueryFunctions>()?;

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

#[pymodule]
pub(crate) fn logging(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LogLevel>()?;
    m.add_function(wrap_pyfunction!(set_log_level, m)?)?;
    m.add_function(wrap_pyfunction!(get_log_level, m)?)?;
    m.add_function(wrap_pyfunction!(log_level_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(log_message_gil, m)?)?;
    Ok(())
}

#[pymodule]
pub fn draw_spec(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ColorDraw>()?; // PYI
    m.add_class::<BoundingBoxDraw>()?; // PYI
    m.add_class::<DotDraw>()?; // PYI
    m.add_class::<LabelDraw>()?; // PYI
    m.add_class::<LabelPositionKind>()?; // PYI
    m.add_class::<LabelPosition>()?; // PYI
    m.add_class::<PaddingDraw>()?; // PYI
    m.add_class::<ObjectDraw>()?; // PYI
    m.add_class::<SetDrawLabelKind>()?;
    Ok(())
}

#[pymodule]
fn savant_rs(py: Python, m: &PyModule) -> PyResult<()> {
    let log_env_var_name = "LOGLEVEL";
    let log_env_var_level = "trace";
    if std::env::var(log_env_var_name).is_err() {
        std::env::set_var(log_env_var_name, log_env_var_level);
    }
    pretty_env_logger::try_init_custom_env(log_env_var_name)
        .map_err(|_| PyRuntimeError::new_err("Failed to initialize logger"))?;
    set_log_level(LogLevel::Error);

    m.add_function(wrap_pyfunction!(init_jaeger_tracer, m)?)?; // PYI
    m.add_function(wrap_pyfunction!(init_noop_tracer, m)?)?; // PYI
    m.add_function(wrap_pyfunction!(version, m)?)?; // PYI

    m.add_wrapped(wrap_pymodule!(primitives))?;
    m.add_wrapped(wrap_pymodule!(pipeline))?;
    m.add_wrapped(wrap_pymodule!(geometry))?;
    m.add_wrapped(wrap_pymodule!(draw_spec))?; // PYI
    m.add_wrapped(wrap_pymodule!(utils))?; // PYI
    m.add_wrapped(wrap_pymodule!(symbol_mapper_module))?;
    m.add_wrapped(wrap_pymodule!(serialization_module))?;
    m.add_wrapped(wrap_pymodule!(match_query))?;
    m.add_wrapped(wrap_pymodule!(logging))?; // PYI
    m.add_wrapped(wrap_pymodule!(zmq))?; // PYI

    let sys = PyModule::import(py, "sys")?;
    let sys_modules: &PyDict = sys.getattr("modules")?.downcast()?;

    sys_modules.set_item("savant_rs.primitives", m.getattr("primitives")?)?;
    sys_modules.set_item("savant_rs.pipeline", m.getattr("pipeline")?)?;
    sys_modules.set_item("savant_rs.pipeline2", m.getattr("pipeline")?)?;

    sys_modules.set_item("savant_rs.primitives.geometry", m.getattr("geometry")?)?;
    sys_modules.set_item("savant_rs.draw_spec", m.getattr("draw_spec")?)?;
    sys_modules.set_item("savant_rs.utils", m.getattr("utils")?)?;
    sys_modules.set_item("savant_rs.logging", m.getattr("logging")?)?;
    sys_modules.set_item("savant_rs.zmq", m.getattr("zmq")?)?;

    sys_modules.set_item(
        "savant_rs.utils.symbol_mapper",
        m.getattr("symbol_mapper_module")?,
    )?;

    sys_modules.set_item(
        "savant_rs.utils.serialization",
        m.getattr("serialization_module")?,
    )?;

    sys_modules.set_item("savant_rs.match_query", m.getattr("match_query")?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {}
}

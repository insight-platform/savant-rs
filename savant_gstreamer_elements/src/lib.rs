mod rspy;
// mod zeromq_src;

use gst::glib;
use gst::prelude::StaticType;
// The public Rust wrapper type for our element
glib::wrapper! {
    pub struct RsPy(ObjectSubclass<rspy::RsPy>) @extends gst::Element, gst::Object;
}

// glib::wrapper! {
//     pub struct ZeromqSrc(ObjectSubclass<zeromq_src::ZeromqSrc>) @extends gst_base::PushSrc, gst_base::BaseSrc, gst::Element, gst::Object;
// }

// Registers the type for our element, and then registers in GStreamer under
// the name "rspy" for being able to instantiate it via e.g.
// gst::ElementFactory::make().
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(Some(plugin), "rspy", gst::Rank::NONE, RsPy::static_type())?;
    // gst::Element::register(
    //     Some(plugin),
    //     "zeromq_src",
    //     gst::Rank::NONE,
    //     ZeromqSrc::static_type(),
    // )?;
    Ok(())
}

gst::plugin_define!(
    savant_gstreamer_elements,
    env!("CARGO_PKG_DESCRIPTION"),
    register,
    env!("CARGO_PKG_VERSION"),
    "APACHE-2.0",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY")
);

// use pyo3::prelude::*;
// use savant_core::pipeline::{
//     Pipeline, PipelinePayload, PipelineStageFunction, PipelineStageFunctionOrder, PluginParams,
// };
// use pyo3::types::PyDict;
// use savant_core::pipeline::stage::PipelineStage;
// use savant_core_py::logging::LogLevel;
// use savant_core_py::pipeline::StageFunction;
// use savant_core_py::primitives::attribute_value::AttributeValue;
// use savant_core_py::primitives::frame::VideoFrame;
// use savant_core_py::primitives::object::BorrowedVideoObject;
// use savant_core_py::utils::check_pybound_name;

// #[no_mangle]
// pub fn init_plugin(_: &str, pp: PluginParams) -> *mut dyn PipelineStageFunction {
//     let plugin = Plugin {
//         pipeline: None,
//         params: pp,
//     };
//     Box::into_raw(Box::new(plugin))
// }
//
// pub struct Plugin {
//     params: PluginParams,
//     pipeline: Option<Pipeline>,
// }
//
// impl PipelineStageFunction for Plugin {
//     fn set_pipeline(&mut self, pipeline: Pipeline) {
//         self.pipeline = Some(pipeline);
//     }
//     fn get_pipeline(&self) -> &Option<Pipeline> {
//         &self.pipeline
//     }
//     fn call(
//         &self,
//         id: i64,
//         stage: &PipelineStage,
//         order: PipelineStageFunctionOrder,
//         _: &mut PipelinePayload,
//     ) -> anyhow::Result<()> {
//         savant_core_py::logging::log_message(
//             LogLevel::Trace,
//             "savant_plugin_sample",
//             format!(
//                 "Object {}, stage {}, order={:?}, params len={}",
//                 id,
//                 stage.name,
//                 order,
//                 self.params.params.len()
//             )
//             .as_str(),
//             None,
//         );
//         Ok(())
//     }
// }
//
// #[pyfunction]
// pub fn get_instance(name: &str, params: &Bound<'_, PyDict>) -> PyResult<StageFunction> {
//     // HashMap<String, AttributeValue>
//     let params = params
//         .into_iter()
//         .map(|(k, v)| {
//             check_pybound_name(&v, "AttributeValue")?;
//             let bound_attr = unsafe { v.downcast_unchecked::<AttributeValue>() };
//             let attr = bound_attr.borrow().clone();
//             Ok((k.to_string(), attr.0))
//         })
//         .collect::<PyResult<hashbrown::HashMap<_, _>>>()?;
//
//     let pp = PluginParams { params };
//     Ok(StageFunction::new(unsafe {
//         Box::from_raw(init_plugin(name, pp))
//     }))
// }
//
// #[pyfunction]
// pub fn access_frame(f: &Bound<'_, PyAny>) -> PyResult<()> {
//     check_pybound_name(f, "VideoFrame")?;
//     let frame = unsafe { f.downcast_unchecked::<VideoFrame>() };
//     println!("Frame: {:?}", frame.borrow().get_uuid());
//     Ok(())
// }
//
// #[pyfunction]
// pub fn access_object(o: &Bound<'_, PyAny>) -> PyResult<()> {
//     check_pybound_name(o, "BorrowedVideoObject")?;
//     let obj = unsafe { o.downcast_unchecked::<BorrowedVideoObject>() };
//     println!("Object: {:?}", obj.borrow().get_id());
//     Ok(())
// }
//
// #[pymodule(gil_used = false)]
// fn savant_plugin_sample(_: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(access_frame, m)?)?;
//     m.add_function(wrap_pyfunction!(access_object, m)?)?;
//     m.add_function(wrap_pyfunction!(get_instance, m)?)?;
//     m.add_class::<VideoFrame>()?;
//     m.add_class::<BorrowedVideoObject>()?;
//     Ok(())
// }

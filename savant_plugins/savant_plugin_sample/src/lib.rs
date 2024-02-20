use pyo3::prelude::*;
use savant_core::pipeline::stage::PipelineStage;
use savant_core::pipeline::{
    Pipeline, PipelinePayload, PipelineStageFunction, PipelineStageFunctionOrder, PluginParams,
};
use savant_core_py::logging::LogLevel;
use savant_core_py::pipeline::StageFunction;
use savant_core_py::primitives::frame::VideoFrame;
use savant_core_py::primitives::object::BorrowedVideoObject;

#[no_mangle]
pub fn init_plugin(_: &str, _: PluginParams) -> *mut dyn PipelineStageFunction {
    let plugin = Plugin { pipeline: None };
    Box::into_raw(Box::new(plugin))
}

pub struct Plugin {
    pipeline: Option<Pipeline>,
}

impl PipelineStageFunction for Plugin {
    fn set_pipeline(&mut self, pipeline: Pipeline) {
        self.pipeline = Some(pipeline);
    }
    fn get_pipeline(&self) -> &Option<Pipeline> {
        &self.pipeline
    }
    fn call(
        &self,
        _: i64,
        _: &PipelineStage,
        _: PipelineStageFunctionOrder,
        _: &mut PipelinePayload,
    ) -> anyhow::Result<()> {
        savant_core_py::logging::log_message(
            LogLevel::Info,
            "savant_plugin",
            "Hello from Rust Plugin",
            None,
        );
        Ok(())
    }
}

#[pyfunction]
pub fn get_stage_function(name: &str) -> StageFunction {
    StageFunction::new(unsafe { Box::from_raw(init_plugin(name, PluginParams::default())) })
}

#[pyfunction]
pub fn access_frame(f: &VideoFrame) {
    println!("Frame: {:?}", f.get_uuid());
}

#[pyfunction]
pub fn access_object(o: &BorrowedVideoObject) {
    println!("Object: {:?}", o.get_id());
}

#[pymodule]
fn savant_plugin_sample(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(access_frame, m)?)?;
    m.add_function(wrap_pyfunction!(access_object, m)?)?;
    m.add_function(wrap_pyfunction!(get_stage_function, m)?)?;
    Ok(())
}

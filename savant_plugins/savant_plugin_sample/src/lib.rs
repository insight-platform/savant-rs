use pyo3::prelude::*;
use savant_core::pipeline::stage::PipelineStage;
use savant_core::pipeline::{
    Pipeline, PipelinePayload, PipelineStageFunction, PipelineStageFunctionOrder, PluginParams,
};
use savant_core_py::logging::LogLevel;
use savant_core_py::pipeline::StageFunction;
use savant_core_py::primitives::attribute_value::AttributeValue;
use savant_core_py::primitives::frame::VideoFrame;
use savant_core_py::primitives::object::BorrowedVideoObject;
use std::collections::HashMap;

#[no_mangle]
pub fn init_plugin(_: &str, pp: PluginParams) -> *mut dyn PipelineStageFunction {
    let plugin = Plugin {
        pipeline: None,
        params: pp,
    };
    Box::into_raw(Box::new(plugin))
}

pub struct Plugin {
    params: PluginParams,
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
        id: i64,
        stage: &PipelineStage,
        order: PipelineStageFunctionOrder,
        _: &mut PipelinePayload,
    ) -> anyhow::Result<()> {
        savant_core_py::logging::log_message(
            LogLevel::Trace,
            "savant_plugin_sample",
            format!(
                "Object {}, stage {}, order={:?}, params len={}",
                id,
                stage.name,
                order,
                self.params.params.len()
            )
            .as_str(),
            None,
        );
        Ok(())
    }
}

#[pyfunction]
pub fn get_instance(name: &str, params: HashMap<String, AttributeValue>) -> StageFunction {
    let pp = PluginParams {
        params: params.into_iter().map(|(k, v)| (k, v.0)).collect(),
    };
    StageFunction::new(unsafe { Box::from_raw(init_plugin(name, pp)) })
}

#[pyfunction]
pub fn access_frame(f: &VideoFrame) {
    println!("Frame: {:?}", f.get_uuid());
}

#[pyfunction]
pub fn access_object(o: &BorrowedVideoObject) {
    println!("Object: {:?}", o.get_id());
}

#[pymodule(gil_used = false)]
fn savant_plugin_sample(_: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(access_frame, m)?)?;
    m.add_function(wrap_pyfunction!(access_object, m)?)?;
    m.add_function(wrap_pyfunction!(get_instance, m)?)?;
    Ok(())
}

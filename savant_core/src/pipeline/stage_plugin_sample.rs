use crate::pipeline::stage::PipelineStage;
use crate::pipeline::{
    Pipeline, PipelinePayload, PipelineStageFunction, PipelineStageFunctionOrder, PluginParams,
};

#[no_mangle]
pub fn init_plugin_test(_: &str, params: PluginParams) -> *mut dyn PipelineStageFunction {
    let plugin = Plugin {
        pipeline: None,
        params,
    };
    Box::into_raw(Box::new(plugin))
}

pub struct Plugin {
    pipeline: Option<Pipeline>,
    params: PluginParams,
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
        _: &PipelineStage,
        _: PipelineStageFunctionOrder,
        payload: &mut PipelinePayload,
    ) -> anyhow::Result<()> {
        dbg!(id, payload, &self.params);
        Ok(())
    }
}

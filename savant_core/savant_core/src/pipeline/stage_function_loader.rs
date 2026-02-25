use crate::pipeline::{PipelineStageFunction, PluginParams};
use hashbrown::HashMap;
use lazy_static::lazy_static;
use parking_lot::Mutex;

lazy_static! {
    static ref LIBRARIES: Mutex<HashMap<String, libloading::Library>> = Mutex::new(HashMap::new());
}

pub fn load_stage_function_plugin(
    libname: &str,
    init_name: &str,
    plugin_name: &str,
    params: PluginParams,
) -> anyhow::Result<Box<dyn PipelineStageFunction>> {
    let mut libs = LIBRARIES.lock();
    if !libs.contains_key(libname) {
        let lib = unsafe { libloading::Library::new(libname)? };
        libs.insert(libname.to_string(), lib);
    }
    let lib = libs
        .get(libname)
        .expect("Library must be available according to the code logic");
    let init: libloading::Symbol<super::PipelineStageFunctionFactory> =
        unsafe { lib.get(init_name.as_bytes())? };
    let raw = init(plugin_name, params);
    Ok(unsafe { Box::from_raw(raw) })
}

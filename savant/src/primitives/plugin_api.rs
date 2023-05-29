use crate::primitives::Object;
use hashbrown::HashMap;
use lazy_static::lazy_static;
use parking_lot::{const_mutex, Mutex};
use pyo3::prelude::*;

pub type ObjectPredicate = libloading::Symbol<'static, extern "C" fn(o: &Object) -> bool>;
pub type ObjectMatchPredicate =
    libloading::Symbol<'static, extern "C" fn(left: &Object, right: &Object) -> bool>;

#[derive(Clone, Debug)]
pub enum PluginFunction {
    ObjectPredicate(ObjectPredicate),
    ObjectMatchPredicate(ObjectMatchPredicate),
}

#[derive(Debug)]
pub struct Plugin {
    pub lib: libloading::Library,
    pub functions: HashMap<String, PluginFunction>,
}

lazy_static! {
    static ref PLUGIN_API: Mutex<HashMap<String, Plugin>> = const_mutex(HashMap::default());
}

pub fn get_plugin_function(plugin: &str, function: &str) -> Option<PluginFunction> {
    let plugins = PLUGIN_API.lock();
    let plugin = plugins.get(plugin)?;
    plugin.functions.get(function).cloned()
}

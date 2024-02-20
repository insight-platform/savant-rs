use savant_core::pipeline::stage_function_loader::load_stage_function_plugin;
use savant_core::pipeline::PluginParams;

pub fn main() {
    let cargo_target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or("target".to_string());
    let p = load_stage_function_plugin(
        format!("{}/debug/libsavant_core.so", cargo_target_dir).as_str(),
        "init_plugin_test",
        "plugin",
        PluginParams::default(),
    )
    .unwrap();
    drop(p);
}

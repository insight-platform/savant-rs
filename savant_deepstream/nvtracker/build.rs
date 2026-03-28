//! Link NVIDIA helper library for `gst_nvevent_new_stream_reset`.

fn main() {
    println!("cargo:rustc-link-search=native=/opt/nvidia/deepstream/deepstream/lib/");
    println!("cargo:rustc-link-lib=nvdsgst_helper");
}

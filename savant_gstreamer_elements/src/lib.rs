mod rspy;
pub mod utils;
mod zeromq_src;

const SAVANT_EOS_EVENT_NAME: &str = "savant-eos";
const SAVANT_EOS_EVENT_SOURCE_ID_PROPERTY: &str = "source-id";

const SAVANT_USERDATA_EVENT_NAME: &str = "savant-userdata";
const SAVANT_USERDATA_EVENT_DATA_PROPERTY: &str = "data";

use gstreamer::prelude::StaticType;
use gstreamer::{glib, FlowError};
use gstreamer_base::subclass::base_src::CreateSuccess;
// The public Rust wrapper type for our element
glib::wrapper! {
    pub struct RsPy(ObjectSubclass<rspy::RsPy>) @extends gstreamer::Element, gstreamer::Object;
}

glib::wrapper! {
    pub struct ZeromqSrc(ObjectSubclass<zeromq_src::ZeromqSrc>) @extends gstreamer_base::PushSrc, gstreamer_base::BaseSrc, gstreamer::Element, gstreamer::Object;
}

// Registers the type for our element, and then registers in GStreamer under
// the name "rspy" for being able to instantiate it via e.g.
// gstreamer::ElementFactory::make().
pub fn register(plugin: &gstreamer::Plugin) -> Result<(), glib::BoolError> {
    gstreamer::Element::register(
        Some(plugin),
        "rspy",
        gstreamer::Rank::NONE,
        RsPy::static_type(),
    )?;
    gstreamer::Element::register(
        Some(plugin),
        "zeromq_src",
        gstreamer::Rank::NONE,
        ZeromqSrc::static_type(),
    )?;
    Ok(())
}

gstreamer::plugin_define!(
    savant_gstreamer_elements,
    env!("CARGO_PKG_DESCRIPTION"),
    register,
    env!("CARGO_PKG_VERSION"),
    "APACHE-2.0",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY")
);

pub type OptionalGstFlowReturn = Option<Result<CreateSuccess, FlowError>>;
pub type OptionalGstBufferReturn = Option<Result<gstreamer::Buffer, FlowError>>;

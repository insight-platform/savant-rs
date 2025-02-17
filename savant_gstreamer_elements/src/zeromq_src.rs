use crate::py_handler::PyHandler;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst::{glib, Buffer};
use parking_lot::RwLock;
use pyo3::prelude::*;
use savant_core::transport::zeromq::{NonBlockingReader, ReaderSocketType, TopicPrefixSpec};
use savant_core_py::gst::FlowResult;
use std::sync::{LazyLock, OnceLock};
use std::time::Instant;
// This module contains the private implementation details of our element

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "zeromq_src",
        gst::DebugColorFlags::empty(),
        Some("ZeroMQ Source Element"),
    )
});

#[derive(Debug, Clone)]
struct Settings {
    socket: Option<String>,
    bind: Option<bool>,
    socket_type: Option<ReaderSocketType>,
    receive_hwm: Option<i32>,
    prefix_spec: Option<TopicPrefixSpec>,
    pipeline: Option<String>,
    pipeline_stage_name: Option<String>,
    shutdown_authorization: Option<String>,
    max_width: Option<usize>,
    max_height: Option<usize>,
    pass_through_mode: Option<bool>,
    ingress_module: Option<String>,
    ingress_class: Option<String>,
    ingress_kwargs: Option<String>,
    ingress_dev_mode: Option<bool>,
    blacklist_size: Option<usize>,
    blacklist_ttl: Option<u64>,
}

pub struct ZeromqSrc {
    reader: OnceLock<NonBlockingReader>,
}

#[glib::object_subclass]
impl ObjectSubclass for ZeromqSrc {
    const NAME: &'static str = "GstZeroMqSrc";
    type Type = super::ZeromqSrc;
    type ParentType = gst_base::PushSrc;
}

impl ObjectImpl for ZeromqSrc {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| vec![]);
        PROPERTIES.as_ref()
    }
}

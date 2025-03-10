// This module contains the private implementation details of our element

use gst::glib;
use gst::glib::ParamFlags;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::prelude::BaseSrcExt;
use savant_core::transport::zeromq::TopicPrefixSpec;
use std::sync::LazyLock;

use crate::zeromq_src::{
    CAT, DEFAULT_FILTER_FRAMES, DEFAULT_FIX_IPC_PERMISSIONS, DEFAULT_INVOKE_ON_MESSAGE,
    DEFAULT_IS_LIVE,
};

use super::ZeromqSrc;

#[glib::object_subclass]
impl ObjectSubclass for ZeromqSrc {
    const NAME: &'static str = "GstZeroMqSrc";
    type Type = crate::ZeromqSrc;
    type ParentType = gst_base::PushSrc;
}

impl ObjectImpl for ZeromqSrc {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            vec![
                glib::ParamSpecString::builder("savant-pipeline-name")
                    .nick("PipelineName")
                    .blurb("The pipeline to work with")
                    .flags(ParamFlags::WRITABLE)
                    .build(),
                glib::ParamSpecString::builder("savant-pipeline-stage")
                    .nick("PipelineStage")
                    .blurb("The stage to work with")
                    .flags(ParamFlags::WRITABLE)
                    .build(),
                glib::ParamSpecString::builder("zmq-socket-uri")
                    .nick("SocketURI")
                    .blurb("The ZeroMQ socket URI")
                    .flags(ParamFlags::WRITABLE)
                    .build(),
                glib::ParamSpecInt::builder("zmq-receive-hwm")
                    .nick("ReceiveHWM")
                    .blurb("The ZeroMQ receive high water mark")
                    .flags(ParamFlags::WRITABLE)
                    .build(),
                glib::ParamSpecString::builder("zmq-topic")
                    .nick("Topic")
                    .blurb("The ZeroMQ topic prefix")
                    .flags(ParamFlags::WRITABLE)
                    .build(),
                glib::ParamSpecString::builder("zmq-topic-prefix")
                    .nick("TopicPrefix")
                    .blurb("The ZeroMQ topic prefix")
                    .flags(ParamFlags::WRITABLE)
                    .build(),
                glib::ParamSpecString::builder("shutdown-authorization")
                    .nick("ShutdownAuthorization")
                    .blurb("The authorization token to shutdown the pipeline")
                    .flags(ParamFlags::WRITABLE)
                    .build(),
                glib::ParamSpecUInt::builder("max-width")
                    .nick("MaxWidth")
                    .blurb("The maximum width of the video frame")
                    .flags(ParamFlags::WRITABLE)
                    .build(),
                glib::ParamSpecUInt::builder("max-height")
                    .nick("MaxHeight")
                    .blurb("The maximum height of the video frame")
                    .flags(ParamFlags::WRITABLE)
                    .build(),
                glib::ParamSpecBoolean::builder("pass-through-mode")
                    .nick("PassThroughMode")
                    .blurb("Whether to pass through the video frame")
                    .flags(ParamFlags::WRITABLE)
                    .build(),
                glib::ParamSpecUInt::builder("blacklist-size")
                    .nick("BlacklistSize")
                    .blurb("The size of the blacklist")
                    .flags(ParamFlags::WRITABLE)
                    .build(),
                glib::ParamSpecUInt::builder("blacklist-ttl")
                    .nick("BlacklistTTL")
                    .blurb("The time-to-live of the blacklist")
                    .flags(ParamFlags::WRITABLE)
                    .build(),
                glib::ParamSpecUInt::builder("fix-ipc-permissions")
                    .nick("FixIPCPermissions")
                    .blurb("Fix IPC permissions")
                    .default_value(DEFAULT_FIX_IPC_PERMISSIONS)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecUInt::builder("receive-timeout")
                    .nick("ReceiveTimeout")
                    .blurb("The ZeroMQ receive timeout")
                    .flags(ParamFlags::WRITABLE)
                    .build(),
                glib::ParamSpecBoolean::builder("invoke-on-message")
                    .nick("InvokeOnMessage")
                    .blurb("Invoke on message")
                    .default_value(DEFAULT_INVOKE_ON_MESSAGE)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecBoolean::builder("filter-frames")
                    .nick("FilterFrames")
                    .blurb("Filter frames")
                    .default_value(DEFAULT_FILTER_FRAMES)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecBoolean::builder("is-live")
                    .nick("IsLive")
                    .blurb("Live output")
                    .default_value(DEFAULT_IS_LIVE)
                    .mutable_ready()
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut settings = self.settings.lock();
        match pspec.name() {
            "savant-pipeline-name" => {
                let pipeline_name = value.get().expect("type checked upstream");
                gst::info!(
                    CAT,
                    imp = self,
                    "Changing pipeline name to {}",
                    pipeline_name
                );
                settings.pipeline_name = Some(pipeline_name);
            }
            "savant-pipeline-stage" => {
                let pipeline_stage_name = value.get().expect("type checked upstream");
                gst::info!(
                    CAT,
                    imp = self,
                    "Changing pipeline stage name to {}",
                    pipeline_stage_name
                );
                settings.pipeline_stage_name = Some(pipeline_stage_name);
            }
            "zmq-socket-uri" => {
                let socket_uri = value.get().expect("type checked upstream");
                gst::info!(CAT, imp = self, "Changing socket URI to {}", socket_uri);
                settings.socket_uri = Some(socket_uri);
            }
            "zmq-receive-hwm" => {
                let receive_hwm = value.get().expect("type checked upstream");
                gst::info!(CAT, imp = self, "Changing receive HWM to {}", receive_hwm);
                assert!(receive_hwm > 0, "Receive HWM must be non-negative",);
                settings.receive_hwm = receive_hwm;
            }
            "zmq-topic" => {
                let topic = value.get().expect("type checked upstream");
                gst::info!(CAT, imp = self, "Changing topic to {}", topic);
                settings.topic_prefix_spec = TopicPrefixSpec::source_id(topic);
            }
            "zmq-topic-prefix" => {
                let topic_prefix = value.get().expect("type checked upstream");
                gst::info!(CAT, imp = self, "Changing topic prefix to {}", topic_prefix);
                settings.topic_prefix_spec = TopicPrefixSpec::prefix(topic_prefix);
            }
            "shutdown-authorization" => {
                let shutdown_authorization = value.get().expect("type checked upstream");
                gst::info!(
                    CAT,
                    imp = self,
                    "Changing shutdown authorization to {}",
                    shutdown_authorization
                );
                settings.shutdown_authorization = Some(shutdown_authorization);
            }
            "max-width" => {
                let max_width: u64 = value.get().expect("type checked upstream");
                gst::info!(CAT, imp = self, "Changing max width to {}", max_width);
                settings.max_width = max_width;
            }
            "max-height" => {
                let max_height: u64 = value.get().expect("type checked upstream");
                gst::info!(CAT, imp = self, "Changing max height to {}", max_height);
                settings.max_height = max_height;
            }
            "pass-through-mode" => {
                let pass_through_mode = value.get().expect("type checked upstream");
                gst::info!(
                    CAT,
                    imp = self,
                    "Changing pass-through mode to {}",
                    pass_through_mode
                );
                settings.pass_through_mode = pass_through_mode;
            }
            "blacklist-size" => {
                let blacklist_size = value.get().expect("type checked upstream");
                gst::info!(
                    CAT,
                    imp = self,
                    "Changing blacklist size to {}",
                    blacklist_size
                );
                assert!(blacklist_size > 0, "Blacklist size must be non-negative",);
                settings.blacklist_size = blacklist_size;
            }
            "blacklist-ttl" => {
                let blacklist_ttl = value.get().expect("type checked upstream");
                gst::info!(
                    CAT,
                    imp = self,
                    "Changing blacklist TTL to {}",
                    blacklist_ttl
                );
                assert!(blacklist_ttl > 0, "Blacklist TTL must be non-negative",);
                settings.blacklist_ttl = blacklist_ttl;
            }
            "fix-ipc-permissions" => {
                let fix_ipc_permissions = value.get().expect("type checked upstream");
                gst::info!(
                    CAT,
                    imp = self,
                    "Changing fix IPC permissions to {}",
                    fix_ipc_permissions
                );
                assert!(
                    fix_ipc_permissions <= 0o777,
                    "IPC permissions must not exceed 0o777",
                );
                settings.fix_ipc_permissions = Some(fix_ipc_permissions);
            }
            "receive-timeout" => {
                let receive_timeout = value.get().expect("type checked upstream");
                gst::info!(
                    CAT,
                    imp = self,
                    "Changing receive timeout to {}",
                    receive_timeout
                );
                assert!(receive_timeout > 0, "Receive timeout must be non-negative",);
                settings.receive_timeout = receive_timeout;
            }
            "invoke-on-message" => {
                let invoke_on_message = value.get().expect("type checked upstream");
                gst::info!(
                    CAT,
                    imp = self,
                    "Changing invoke on message to {}",
                    invoke_on_message
                );
                settings.invoke_on_message = invoke_on_message;
            }
            "filter-frames" => {
                let filter_frames = value.get().expect("type checked upstream");
                gst::info!(
                    CAT,
                    imp = self,
                    "Changing filter frames to {}",
                    filter_frames
                );
                settings.filter_frames = filter_frames;
            }
            "is-live" => {
                let is_live = value.get().expect("type checked upstream");
                gst::info!(CAT, imp = self, "Changing is live to {}", is_live);
                settings.is_live = is_live;
            }
            _ => unimplemented!(),
        }
    }

    fn constructed(&self) {
        // Call the parent class' ::constructed() implementation first
        self.parent_constructed();

        let obj = self.obj();
        // Initialize live-ness and notify the base class that
        // we'd like to operate in Time format
        obj.set_live(DEFAULT_IS_LIVE);
        obj.set_format(gst::Format::Time);
    }

    fn property(&self, _id: usize, _pspec: &glib::ParamSpec) -> glib::Value {
        unimplemented!("Getting properties is not implemented yet");
    }
}

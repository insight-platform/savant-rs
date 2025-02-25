use gst::glib::ParamFlags;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst::{glib, Buffer};
use gst_base::prelude::BaseSrcExt;
use parking_lot::RwLock;
use pyo3::prelude::*;
use savant_core::transport::zeromq::{ReaderSocketType, SyncReader, TopicPrefixSpec};
use savant_core_py::gst::FlowResult;
use std::sync::{LazyLock, OnceLock};
use std::time::Instant;
// This module contains the private implementation details of our element

const DEFAULT_IS_LIVE: bool = false;
const DEFAULT_BLACKLIST_SIZE: u64 = 256;
const DEFAULT_BLACKLIST_TTL: u64 = 60;
const DEFAULT_MAX_WIDTH: u64 = 15360; // 16K resolution
const DEFAULT_MAX_HEIGHT: u64 = 8640; // 16K resolution
const DEFAULT_PASS_THROUGH_MODE: bool = false;

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "zeromq_src",
        gst::DebugColorFlags::empty(),
        Some("ZeroMQ Source Element"),
    )
});

#[derive(Debug, Clone)]
struct Settings {
    socket_uri: Option<String>,
    receive_hwm: Option<i32>,
    topic_prefix_spec: TopicPrefixSpec,
    pipeline_name: Option<String>,
    pipeline_stage_name: Option<String>,
    shutdown_authorization: Option<String>,
    max_width: u64,
    max_height: u64,
    pass_through_mode: bool,
    blacklist_size: u64,
    blacklist_ttl: u64,
    is_live: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            socket_uri: None,
            receive_hwm: None,
            topic_prefix_spec: TopicPrefixSpec::Prefix("".to_string()),
            pipeline_name: None,
            pipeline_stage_name: None,
            shutdown_authorization: None,
            max_width: DEFAULT_MAX_WIDTH,
            max_height: DEFAULT_MAX_HEIGHT,
            pass_through_mode: DEFAULT_PASS_THROUGH_MODE,
            blacklist_size: DEFAULT_BLACKLIST_SIZE,
            blacklist_ttl: DEFAULT_BLACKLIST_TTL,
            is_live: DEFAULT_IS_LIVE,
        }
    }
}

#[derive(Default)]
pub struct ZeromqSrc {
    settings: RwLock<Settings>,
    reader: OnceLock<SyncReader>,
}

#[glib::object_subclass]
impl ObjectSubclass for ZeromqSrc {
    const NAME: &'static str = "GstZeroMqSrc";
    type Type = super::ZeromqSrc;
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
        let mut settings = self.settings.write();
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
                settings.receive_hwm = Some(receive_hwm);
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
                settings.blacklist_ttl = blacklist_ttl;
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

impl GstObjectImpl for ZeromqSrc {}

impl ElementImpl for ZeromqSrc {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "ZeroMQ Savant Source",
                "Source/Video",
                "Creates video frames",
                "Ivan Kudriavtsev <ivan.a.kudryavtsev@gmail.com>, based on work of Sebastian Dröge <sebastian@centricular.com>",
            )
        });

        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: LazyLock<Vec<gst::PadTemplate>> = LazyLock::new(|| {
            // On the src pad, we can produce F32/F64 with any sample rate
            // and any number of channels
            let caps = gst_audio::AudioCapsBuilder::new_interleaved()
                .format_list([gst_audio::AUDIO_FORMAT_F32, gst_audio::AUDIO_FORMAT_F64])
                .build();
            // The src pad template must be named "src" for basesrc
            // and specific a pad that is always there
            let src_pad_template = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &caps,
            )
            .unwrap();

            vec![src_pad_template]
        });

        PAD_TEMPLATES.as_ref()
    }

    fn change_state(
        &self,
        transition: gst::StateChange,
    ) -> Result<gst::StateChangeSuccess, gst::StateChangeError> {
        // Configure live'ness once here just before starting the source
        if let gst::StateChange::ReadyToPaused = transition {
            self.obj().set_live(self.settings.read().is_live);
        }

        // Call the parent class' implementation of ::change_state()
        self.parent_change_state(transition)
    }
}

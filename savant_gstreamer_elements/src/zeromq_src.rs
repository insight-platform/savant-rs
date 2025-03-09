use anyhow::bail;
use gst::glib::ParamFlags;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst::{glib, BufferRef, ClockTime, ErrorMessage, FlowError};
use gst_base::prelude::BaseSrcExt;
use gst_base::subclass::base_src::{BaseSrcImpl, CreateSuccess};
use gst_base::subclass::prelude::PushSrcImpl;
use parking_lot::RwLock;
use pyo3::prelude::*;
use savant_core::message::{Message, MessageEnvelope};
use savant_core::transport::zeromq::{ReaderConfig, ReaderResult, SyncReader, TopicPrefixSpec};
use savant_core::utils::bytes_to_hex_string;
use savant_core_py::gst::{InvocationReason, REGISTERED_HANDLERS};
use savant_core_py::primitives::message::Message as PyMessage;
use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::{LazyLock, OnceLock};
// This module contains the private implementation details of our element

const DEFAULT_IS_LIVE: bool = false;
const DEFAULT_BLACKLIST_SIZE: u64 = 256;
const DEFAULT_BLACKLIST_TTL: u64 = 60;
const DEFAULT_MAX_WIDTH: u64 = 15360; // 16K resolution
const DEFAULT_MAX_HEIGHT: u64 = 8640; // 16K resolution
const DEFAULT_PASS_THROUGH_MODE: bool = false;
const DEFAULT_RECEIVE_HWM: i32 = 1000;
const DEFAULT_RECEIVE_TIMEOUT: i32 = 1;
const DEFAULT_FIX_IPC_PERMISSIONS: u32 = 0o777;
const DEFAULT_ROUTING_ID_CACHE_SIZE: usize = 512;
const DEFAULT_INVOKE_ON_MESSAGE: bool = false;
const DEFAULT_FILTER_FRAMES: bool = false;

const SAVANT_EOS_EVENT_NAME: &str = "savant-eos";
const SAVANT_EOS_EVENT_SOURCE_ID_PROPERTY: &str = "source-id";

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
    receive_hwm: i32,
    receive_timeout: i32,
    topic_prefix_spec: TopicPrefixSpec,
    pipeline_name: Option<String>,
    pipeline_stage_name: Option<String>,
    shutdown_authorization: Option<String>,
    max_width: u64,
    max_height: u64,
    pass_through_mode: bool,
    blacklist_size: u64,
    blacklist_ttl: u64,
    fix_ipc_permissions: Option<u32>,
    routing_cache_size: usize,
    is_live: bool,
    invoke_on_message: bool,
    filter_frames: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            socket_uri: None,
            receive_hwm: DEFAULT_RECEIVE_HWM,
            receive_timeout: DEFAULT_RECEIVE_TIMEOUT,
            topic_prefix_spec: TopicPrefixSpec::Prefix("".to_string()),
            pipeline_name: None,
            pipeline_stage_name: None,
            shutdown_authorization: None,
            max_width: DEFAULT_MAX_WIDTH,
            max_height: DEFAULT_MAX_HEIGHT,
            pass_through_mode: DEFAULT_PASS_THROUGH_MODE,
            blacklist_size: DEFAULT_BLACKLIST_SIZE,
            blacklist_ttl: DEFAULT_BLACKLIST_TTL,
            fix_ipc_permissions: None,
            routing_cache_size: DEFAULT_ROUTING_ID_CACHE_SIZE,
            is_live: DEFAULT_IS_LIVE,
            invoke_on_message: DEFAULT_INVOKE_ON_MESSAGE,
            filter_frames: DEFAULT_FILTER_FRAMES,
        }
    }
}

#[derive(Default)]
pub struct ZeromqSrc {
    settings: RwLock<Settings>,
    reader: OnceLock<SyncReader>,
}

impl ZeromqSrc {
    pub(crate) fn invoke_custom_py_function_on_message<'a>(
        &'_ self,
        message: &'a Message,
    ) -> anyhow::Result<Cow<'a, Message>> {
        if self.settings.read().invoke_on_message {
            let res = Python::with_gil(|py| {
                let element_name = self.obj().name().to_string();
                let handlers_bind = REGISTERED_HANDLERS.read();
                let handler = handlers_bind.get(&element_name);

                if let Some(handler) = handler {
                    let message = message.clone();
                    let py_message = PyMessage::new(message);
                    let res = handler.call1(
                        py,
                        (
                            element_name,
                            InvocationReason::IngressMessageTransformer,
                            py_message,
                        ),
                    );
                    match res {
                        Ok(res) => {
                            gst::trace!(CAT, imp = self, "Handler invoked successfully");
                            let message = res.extract::<PyMessage>(py);
                            match message {
                                Ok(message) => Ok(Cow::Owned(message.extract())),
                                Err(e) => {
                                    gst::error!(
                                        CAT,
                                        imp = self,
                                        "Handler invocation failed: {}",
                                        e
                                    );
                                    bail!("Handler invocation failed (cannot extract Message type): {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            gst::error!(CAT, imp = self, "Handler invocation failed: {}", e);
                            bail!("Handler invocation failed: {}", e);
                        }
                    }
                } else {
                    Ok(Cow::Borrowed(message))
                }
            })?;
            Ok(res)
        } else {
            Ok(Cow::Borrowed(message))
        }
    }

    fn handle_video_frame(
        &self,
        frame: &savant_core::primitives::frame::VideoFrameProxy,
        data: &Vec<Vec<u8>>,
    ) -> Option<Result<CreateSuccess, FlowError>> {
        todo!()
    }

    fn handle_video_frame_batch(
        &self,
        batch: &savant_core::primitives::rust::VideoFrameBatch,
        data: &Vec<Vec<u8>>,
    ) -> Option<Result<CreateSuccess, FlowError>> {
        todo!()
    }
}

impl ZeromqSrc {
    pub(crate) fn handle_unsupported_payload(&self, unsupported_message: &Message) {
        gst::warning!(
            CAT,
            imp = self,
            "Unsupported message payload {:?}, the message will be ignored.",
            unsupported_message
        );
    }
}

impl ZeromqSrc {
    pub(crate) fn handle_auxiliary_reader_states(&self, state: &ReaderResult) {
        match state {
            ReaderResult::Timeout => {
                gst::trace!(CAT, imp = self, "Timeout while waiting for message");
            }
            ReaderResult::PrefixMismatch { topic, routing_id } => {
                let routing_id_hex = routing_id
                    .as_ref()
                    .map(|r| bytes_to_hex_string(r))
                    .unwrap_or(String::new());
                let topic_str = String::from_utf8_lossy(topic);
                let prefix_spec = self.settings.read().topic_prefix_spec.clone();
                gst::trace!(
                    CAT,
                    imp = self,
                    "Prefix mismatch for routing_id{}/topic: {}, prefix spec is {:?}",
                    routing_id_hex,
                    topic_str,
                    prefix_spec
                );
            }
            ReaderResult::RoutingIdMismatch { topic, routing_id } => {
                let routing_id_hex = routing_id
                    .as_ref()
                    .map(|r| bytes_to_hex_string(&r))
                    .unwrap_or(String::new());
                let topic_str = String::from_utf8_lossy(topic);
                gst::trace!(
                    CAT,
                    imp = self,
                    "Routing ID {} mismatch for topic: {}",
                    routing_id_hex,
                    topic_str
                );
            }
            ReaderResult::TooShort(m) => {
                gst::trace!(CAT, imp = self, "Message is too short: {:?}", m);
            }
            ReaderResult::Blacklisted(topic) => {
                let topic_str = String::from_utf8_lossy(topic);
                gst::trace!(CAT, imp = self, "Blacklisted topic: {:?}", topic_str);
            }
            _ => unimplemented!("This state must not be reached in this method!"),
        }
    }
}

impl ZeromqSrc {
    pub(crate) fn handle_message(
        &self,
        message: &Message,
        data: &Vec<Vec<u8>>,
    ) -> Option<Result<CreateSuccess, FlowError>> {
        match message.payload() {
            MessageEnvelope::EndOfStream(eos) => self.handle_eos(eos),
            MessageEnvelope::VideoFrame(vf) => self.handle_video_frame(vf, data),
            MessageEnvelope::VideoFrameBatch(b) => self.handle_video_frame_batch(b, data),
            MessageEnvelope::Shutdown(shutdown) => self.handle_shutdown(shutdown),
            _ => unimplemented!("This state must not be reached in this method"),
        }
    }

    fn handle_eos(
        &self,
        eos: &savant_core::primitives::eos::EndOfStream,
    ) -> Option<Result<CreateSuccess, FlowError>> {
        gst::info!(
            CAT,
            imp = self,
            "Received EOS message for the source {}",
            eos.source_id
        );
        let savant_eos_event = build_savant_eos_event(&eos.source_id);
        let pads = self.obj().pads();
        for pad in pads {
            if pad.direction() == gst::PadDirection::Src {
                if !pad.push_event(savant_eos_event.clone()) {
                    gst::error!(
                        CAT,
                        imp = self,
                        "Failed to push EOS event to pad {}",
                        pad.name()
                    );
                    return Some(Err(FlowError::Error));
                }
            }
        }
        None
    }

    fn handle_shutdown(
        &self,
        shutdown: &savant_core::primitives::shutdown::Shutdown,
    ) -> Option<Result<CreateSuccess, FlowError>> {
        let settings_bind = self.settings.read();
        let shutdown_auth_opt = settings_bind.shutdown_authorization.as_ref();
        if shutdown_auth_opt.is_none() {
            gst::warning!(
                CAT,
                imp = self,
                "Received shutdown message but shutdown authorization is not set"
            );
            return None;
        }
        let shutdown_auth = shutdown_auth_opt.unwrap();
        if shutdown.get_auth() != shutdown_auth.as_str() {
            gst::warning!(
                CAT,
                imp = self,
                "Received shutdown message but authorization is incorrect"
            );
            return None;
        }
        gst::info!(CAT, imp = self, "Received shutdown message");
        for pad in self.obj().pads() {
            if pad.direction() == gst::PadDirection::Src {
                if !pad.push_event(gst::event::Eos::new()) {
                    gst::error!(
                        CAT,
                        imp = self,
                        "Failed to push EOS event to pad {}",
                        pad.name()
                    );
                }
            }
        }
        Some(Err(FlowError::Eos))
    }
}

fn build_savant_eos_event(source_id: &str) -> gst::Event {
    let eos_struct = gst::Structure::builder(SAVANT_EOS_EVENT_NAME)
        .field(SAVANT_EOS_EVENT_SOURCE_ID_PROPERTY, source_id)
        .build();
    gst::event::CustomDownstream::new(eos_struct)
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

impl BaseSrcImpl for ZeromqSrc {
    fn start(&self) -> Result<(), ErrorMessage> {
        let settings_bind = self.settings.read();

        let socket_uri = settings_bind.socket_uri.as_ref().ok_or_else(|| {
            gst::error!(CAT, imp = self, "No socket URI provided");
            gst::error_msg!(gst::ResourceError::Settings, ["No socket URI provided"])
        })?;

        let reader_settings_builder = || {
            ReaderConfig::new()
                .url(socket_uri)?
                .with_receive_timeout(settings_bind.receive_timeout)?
                .with_receive_hwm(settings_bind.receive_hwm)?
                .with_fix_ipc_permissions(settings_bind.fix_ipc_permissions)?
                .with_routing_cache_size(settings_bind.routing_cache_size)?
                .with_topic_prefix_spec(settings_bind.topic_prefix_spec.clone())?
                .with_source_blacklist_size(
                    NonZeroU64::new(settings_bind.blacklist_size)
                        .expect("Blacklist size must be non-zero"),
                )?
                .with_source_blacklist_ttl(
                    NonZeroU64::new(settings_bind.blacklist_ttl)
                        .expect("Blacklist TTL must be non-zero"),
                )?
                .build()
        };

        let reader_config = reader_settings_builder().map_err(|e| {
            gst::error!(
                CAT,
                imp = self,
                "Failed to create ZMQ reader settings: {}",
                e.to_string()
            );
            gst::error_msg!(
                gst::ResourceError::Settings,
                ["Failed to create ZMQ reader settings: {}", e.to_string()]
            )
        })?;

        self.reader.get_or_init(|| {
            SyncReader::new(&reader_config).unwrap_or_else(|e| {
                gst::error!(
                    CAT,
                    imp = self,
                    "Failed to create ZMQ reader for settings {:?}: {}",
                    &reader_config,
                    e.to_string()
                );
                panic!("Failed to create ZMQ reader: {}", e.to_string())
            })
        });

        Ok(())
    }

    fn stop(&self) -> Result<(), ErrorMessage> {
        let reader = self.reader.get().ok_or(gst::error_msg!(
            gst::ResourceError::Failed,
            ["Failed to receive reader object, not started"]
        ))?;
        reader.shutdown().map_err(|e| {
            gst::error!(
                CAT,
                imp = self,
                "Failed to shutdown ZMQ reader: {}",
                e.to_string()
            );
            gst::error_msg!(
                gst::ResourceError::Failed,
                ["Failed to shutdown ZMQ reader: {}", e.to_string()]
            )
        })
    }

    fn is_seekable(&self) -> bool {
        false
    }
}

impl PushSrcImpl for ZeromqSrc {
    fn create(&self, _buffer: Option<&mut BufferRef>) -> Result<CreateSuccess, FlowError> {
        gst::trace!(CAT, imp = self, "Creating new buffer");
        let reader = self.reader.get().ok_or_else(|| {
            gst::error!(
                CAT,
                imp = self,
                "Failed to receive zeromq reader object, not created."
            );
            FlowError::Error
        })?;
        // wait until gst element status is playing
        loop {
            let (change, cur, _) = self.obj().state(Some(ClockTime::from_mseconds(100)));
            if cur != gst::State::Playing {
                continue;
            }
            let read_result = reader.receive().map_err(|e| {
                gst::error!(
                    CAT,
                    imp = self,
                    "Failed to receive packet from ZMQ reader: {}",
                    e.to_string()
                );
                FlowError::Error
            })?;
            match &read_result {
                ReaderResult::Message {
                    message,
                    topic,
                    routing_id,
                    data,
                } => {
                    let message = self.invoke_custom_py_function_on_message(message.as_ref());
                    match message {
                        Ok(m) => match m.payload() {
                            MessageEnvelope::VideoFrameUpdate(_)
                            | MessageEnvelope::UserData(_)
                            | MessageEnvelope::Unknown(_) => {
                                self.handle_unsupported_payload(m.as_ref())
                            }
                            _ => {
                                let res = self.handle_message(m.as_ref(), data);
                                if let Some(res) = res {
                                    return res;
                                }
                            }
                        },
                        Err(e) => {
                            let routing_id_hex = routing_id
                                .as_ref()
                                .map(|r| bytes_to_hex_string(r))
                                .unwrap_or(String::new());
                            let topic_str = String::from_utf8_lossy(topic);
                            let prefix_spec = self.settings.read().topic_prefix_spec.clone();
                            gst::error!(
                                CAT,
                                imp = self,
                                "Failed to handle message for routing_id{}/topic {}, prefix spec is {:?}, error: {:?}",
                                routing_id_hex,
                                topic_str,
                                prefix_spec,
                                e
                            );
                            return Err(FlowError::Error);
                        }
                    }
                }
                _ => self.handle_auxiliary_reader_states(&read_result),
            }
        }
    }
}

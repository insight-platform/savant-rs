// This module contains the private implementation details of our element

use gst::prelude::*;
use gst::subclass::prelude::*;
use gst::{BufferRef, ClockTime, ErrorMessage, FlowError};
use gst_base::prelude::BaseSrcExt;
use gst_base::subclass::base_src::{BaseSrcImpl, CreateSuccess};
use gst_base::subclass::prelude::PushSrcImpl;
use parking_lot::Mutex;
use savant_core::message::{validate_seq_id, Message, MessageEnvelope};
use savant_core::transport::zeromq::{ReaderConfig, ReaderResult, SyncReader, TopicPrefixSpec};
use savant_core::utils::bytes_to_hex_string;
use savant_core::webserver::is_shutdown_set;
use std::num::NonZeroU64;
use std::sync::{LazyLock, OnceLock};

mod message_handlers;
mod object_impl;
mod py_functions;

use crate::OptionalGstFlowReturn;

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

const SKIP_PROCESSING: OptionalGstFlowReturn = None;
const ERROR_PROCESSING_RES_ERR: Result<CreateSuccess, FlowError> = Err(FlowError::Error);
const ERROR_PROCESSING_OPT: OptionalGstFlowReturn = Some(ERROR_PROCESSING_RES_ERR);
const EOS_PROCESSING_OPT: OptionalGstFlowReturn = Some(Err(FlowError::Eos));

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
    settings: Mutex<Settings>,
    reader: OnceLock<SyncReader>,
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
            self.obj().set_live(self.settings.lock().is_live);
        }

        // Call the parent class' implementation of ::change_state()
        self.parent_change_state(transition)
    }
}

impl BaseSrcImpl for ZeromqSrc {
    fn start(&self) -> Result<(), ErrorMessage> {
        let settings_bind = self.settings.lock();

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
                    e
                );
                panic!("Failed to create ZMQ reader: {}", e)
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
            let (_, cur, _) = self.obj().state(Some(ClockTime::from_mseconds(100)));
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
                            MessageEnvelope::VideoFrameUpdate(_) | MessageEnvelope::Unknown(_) => {
                                self.handle_unsupported_payload(m.as_ref())
                            }
                            _ => {
                                let res = self.handle_message(m.as_ref(), data);
                                if let Some(res) = res {
                                    return res;
                                }
                                gst::debug!(CAT, imp = self, "The message processor returned None. Skipping the message: {:?}", m);
                            }
                        },
                        Err(e) => {
                            let routing_id_hex = routing_id
                                .as_ref()
                                .map(|r| bytes_to_hex_string(r))
                                .unwrap_or(String::new());
                            let topic_str = String::from_utf8_lossy(topic);
                            let prefix_spec = self.settings.lock().topic_prefix_spec.clone();
                            gst::error!(
                                CAT,
                                imp = self,
                                "Failed to handle message for routing_id{}/topic {}, prefix spec is {:?}, error: {:?}",
                                routing_id_hex,
                                topic_str,
                                prefix_spec,
                                e
                            );
                            return ERROR_PROCESSING_RES_ERR;
                        }
                    }
                }
                _ => self.handle_auxiliary_reader_states(&read_result),
            }
        }
    }
}

impl ZeromqSrc {
    pub(crate) fn handle_message(
        &self,
        message: &Message,
        data: &[Vec<u8>],
    ) -> OptionalGstFlowReturn {
        if is_shutdown_set() {
            return self.handle_ws_shutdown();
        }
        validate_seq_id(message); // checks if the sequence id is valid and reports a warning if it is not
        match message.payload() {
            MessageEnvelope::UserData(_) => self.handle_user_data(message),
            MessageEnvelope::EndOfStream(eos) => self.handle_savant_stream_eos(eos),
            MessageEnvelope::VideoFrame(_) => self.handle_video_frame(message, data),
            MessageEnvelope::VideoFrameBatch(_) => self.handle_video_frame_batch(message, data),
            MessageEnvelope::Shutdown(shutdown) => self.handle_shutdown(shutdown),
            _ => panic!("This state must not be reached in this method"),
        }
    }
}

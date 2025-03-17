use base64::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer::{prelude::*, FlowError};
use gstreamer_base::subclass::base_src::CreateSuccess;
use savant_core::message::{save_message, Message};
use savant_core::primitives::eos::EndOfStream;
use savant_core::primitives::rust::{VideoFrameContent, VideoFrameProxy};
use savant_core::primitives::shutdown::Shutdown;
use savant_core::rust::{Pipeline, PropagatedContext};
use savant_core::transport::zeromq::ReaderResult;
use savant_core::utils::bytes_to_hex_string;
use savant_core::webserver::get_pipeline;
use savant_gstreamer::id_meta::SavantIdMetaKind;

use crate::utils::convert_ts;
use crate::zeromq_src::{CAT, EOS_PROCESSING_OPT, ERROR_PROCESSING_OPT, SKIP_PROCESSING};
use crate::{OptionalGstBufferReturn, OptionalGstFlowReturn};

use super::ZeromqSrc;
use crate::{
    SAVANT_EOS_EVENT_NAME, SAVANT_EOS_EVENT_SOURCE_ID_PROPERTY,
    SAVANT_USERDATA_EVENT_DATA_PROPERTY, SAVANT_USERDATA_EVENT_NAME,
};

impl ZeromqSrc {
    pub(crate) fn handle_user_data(&self, user_data_message: &Message) -> OptionalGstFlowReturn {
        let savant_userdata_event = build_savant_userdata_event(user_data_message);
        for pad in self.obj().pads() {
            if pad.direction() == gstreamer::PadDirection::Src
                && !pad.push_event(savant_userdata_event.clone())
            {
                gstreamer::error!(
                    CAT,
                    imp = self,
                    "Failed to push UserData event to pad {}",
                    pad.name()
                );
                return ERROR_PROCESSING_OPT;
            }
        }
        SKIP_PROCESSING
    }

    fn handle_video_frame_impl(
        &self,
        frame: &VideoFrameProxy,
        context: &PropagatedContext,
        data: &[Vec<u8>],
    ) -> OptionalGstBufferReturn {
        gstreamer::debug!(
            CAT,
            imp = self,
            "Received frame [{}]",
            get_frame_key_info_as_string(frame)
        );

        if self.is_greater_than_max_resolution(frame) {
            let settings_bind = self.settings.lock();
            let max_width = settings_bind.max_width;
            let max_height = settings_bind.max_height;

            gstreamer::warning!(
            CAT,
            imp = self,
            "Frame [{}] resolution is greater than max allowed resolution (W={}, H={}), skipping",
            get_frame_key_info_as_string(frame),
            max_width,
            max_height
        );
            return None;
        }

        let res = self.invoke_custom_ingress_py_function_on_frame(frame);
        if res.is_err() {
            gstreamer::error!(
                CAT,
                imp = self,
                "Error invoking custom ingress Python function. Error: {}",
                res.err().unwrap()
            );
            return Some(Err(FlowError::Error));
        }

        let res = res.unwrap();

        if !res {
            gstreamer::debug!(
                CAT,
                imp = self,
                "Custom ingress Python function returned False, skipping the frame [{}]",
                get_frame_key_info_as_string(frame)
            );
            return None;
        }

        gstreamer::debug!(
            CAT,
            imp = self,
            "Custom ingress Python function returned True, processing the frame [{}]",
            get_frame_key_info_as_string(frame)
        );
        let buf = self.build_frame_buffer(frame, data);
        if buf.is_none() {
            return None;
        }
        let buf = buf.unwrap();
        let frame_meta = self.register_frame_in_pipeline(frame, context);
        let buf = savant_gstreamer::GstBuffer::from(buf);
        if let Some(frame_meta) = frame_meta {
            buf.replace_id_meta(vec![frame_meta]).unwrap_or_else(|e| {
                panic!("Failed to replace ID meta for frame: {}", e);
            });
        }
        let buf = buf.extract().unwrap_or_else(|e| {
            panic!("Failed to extract GstBuffer from GstBuffer: {}", e);
        });
        Some(Ok(buf))
    }

    fn register_frame_in_pipeline(
        &self,
        frame: &VideoFrameProxy,
        context: &PropagatedContext,
    ) -> Option<SavantIdMetaKind> {
        let pipeline_info = self.pipeline_info.get_or_init(|| {
            let settings_bind = self.settings.lock();
            let pipeline_name = settings_bind.pipeline_name.as_ref();
            let stage_name = settings_bind.pipeline_stage_name.as_ref();
            if !pipeline_name.is_some() || !stage_name.is_some() {
                gstreamer::debug!(
                    CAT,
                    imp = self,
                    "Pipeline name or stage name is not set, skipping frame registration"
                );
                return None;
            }
            let pipeline_name = pipeline_name.unwrap().to_string();
            let stage_name = stage_name.unwrap().to_string();

            let res = get_pipeline(&pipeline_name).map(Pipeline::from);
            res.map(|p| (p, pipeline_name, stage_name))
        });
        if let Some((pipeline, pipeline_name, stage_name)) = pipeline_info {
            let frame_idx = pipeline
                .add_frame_with_telemetry(stage_name, frame.clone(), context.extract())
                .unwrap_or_else(|e| {
                    let message = format!(
                        "Failed to add frame [{}] to pipeline {}, stage {}: {}",
                        get_frame_key_info_as_string(frame),
                        pipeline_name,
                        stage_name,
                        e
                    );
                    gstreamer::error!(CAT, imp = self, "{}", &message);
                    panic!("{}", &message);
                });
            gstreamer::debug!(
                CAT,
                imp = self,
                "Frame [{}] registered in pipeline [{}]",
                frame_idx,
                pipeline_name
            );
            Some(SavantIdMetaKind::Frame(frame_idx))
        } else {
            None
        }
    }

    pub(crate) fn handle_video_frame(
        &self,
        message: &Message,
        data: &[Vec<u8>],
    ) -> OptionalGstFlowReturn {
        let frame = message
            .as_video_frame()
            .expect("Failed to get VideoFrame from Message");
        self.handle_video_frame_impl(&frame, message.get_span_context(), data)
            .map(|b_res| b_res.map(CreateSuccess::NewBuffer))
    }

    pub(crate) fn handle_video_frame_batch(
        &self,
        message: &Message,
        data: &[Vec<u8>],
    ) -> OptionalGstFlowReturn {
        let batch = message
            .as_video_frame_batch()
            .expect("Failed to get VideoFrameBatch from Message");

        let results = batch
            .frames()
            .iter()
            .flat_map(|(_, frame)| {
                self.handle_video_frame_impl(frame, message.get_span_context(), data)
            })
            .flatten()
            .collect::<Vec<_>>();

        if results.is_empty() {
            return SKIP_PROCESSING;
        }

        let mut buf_list = gstreamer::BufferList::new();
        let buf_list_mut = buf_list.make_mut();
        for buf in results {
            buf_list_mut.add(buf);
        }
        Some(Ok(CreateSuccess::NewBufferList(buf_list)))
    }

    pub(crate) fn handle_ws_shutdown(&self) -> OptionalGstFlowReturn {
        gstreamer::info!(CAT, imp = self, "Received shutdown signal from WebServer");
        self.send_eos_downstream_pads();
        EOS_PROCESSING_OPT
    }

    fn is_greater_than_max_resolution(&self, frame: &VideoFrameProxy) -> bool {
        let settings_bind = self.settings.lock();
        let max_width = settings_bind.max_width;
        let max_height = settings_bind.max_height;
        frame.get_width() > max_width as i64 || frame.get_height() > max_height as i64
    }

    fn build_frame_buffer(
        &self,
        frame: &VideoFrameProxy,
        external_content: &[Vec<u8>],
    ) -> Option<gstreamer::Buffer> {
        let content_spec = frame.get_content();
        match content_spec.as_ref() {
            VideoFrameContent::External(_) => {
                let external_content_opt = external_content.first();
                if external_content_opt.is_none() {
                    gstreamer::warning!(
                        CAT,
                        imp = self,
                        "No external content found for frame [{}] which has external content",
                        get_frame_key_info_as_string(frame)
                    );
                    return None;
                }
                let external_content = external_content_opt.unwrap();

                let mut buf = gstreamer::Buffer::with_size(external_content.len()).unwrap();
                {
                    let buf_mut = buf.make_mut();
                    let mut map = buf_mut.map_writable().unwrap();
                    let data = map.as_mut_slice();
                    data.copy_from_slice(external_content);
                }
                Some(buf)
            }
            VideoFrameContent::Internal(content) => {
                let mut buf = gstreamer::Buffer::with_size(content.len()).unwrap();
                {
                    let buf_mut = buf.make_mut();
                    let mut map = buf_mut.map_writable().unwrap();
                    let data = map.as_mut_slice();
                    data.copy_from_slice(content);
                }
                Some(buf)
            }
            VideoFrameContent::None => {
                gstreamer::debug!(
                    CAT,
                    imp = self,
                    "Frame [{}] has no content, skipping",
                    get_frame_key_info_as_string(frame)
                );
                None
            }
        }
    }

    pub(crate) fn handle_unsupported_payload(&self, m: &Message) {
        gstreamer::warning!(
            CAT,
            imp = self,
            "Unsupported message payload {:?}, the message will be ignored.",
            m
        );
    }

    pub(crate) fn handle_auxiliary_reader_states(&self, state: &ReaderResult) {
        match state {
            ReaderResult::Timeout => {
                gstreamer::debug!(CAT, imp = self, "Timeout while waiting for message");
            }
            ReaderResult::PrefixMismatch { topic, routing_id } => {
                let routing_id_hex = human_readable_routing_id(routing_id);
                let topic_str = String::from_utf8_lossy(topic);
                let prefix_spec = self.settings.lock().topic_prefix_spec.clone();
                gstreamer::debug!(
                    CAT,
                    imp = self,
                    "Prefix mismatch for routing_id{}/topic: {}, prefix spec is {:?}",
                    routing_id_hex,
                    topic_str,
                    prefix_spec
                );
            }
            ReaderResult::RoutingIdMismatch { topic, routing_id } => {
                let routing_id_hex = human_readable_routing_id(routing_id);
                let topic_str = String::from_utf8_lossy(topic);
                gstreamer::debug!(
                    CAT,
                    imp = self,
                    "Routing ID {} mismatch for topic: {}",
                    routing_id_hex,
                    topic_str
                );
            }
            ReaderResult::TooShort(m) => {
                gstreamer::debug!(CAT, imp = self, "Message is too short: {:?}", m);
            }
            ReaderResult::Blacklisted(topic) => {
                let topic_str = String::from_utf8_lossy(topic);
                gstreamer::debug!(CAT, imp = self, "Blacklisted topic: {:?}", topic_str);
            }
            _ => panic!("This state must not be reached in this method!"),
        }
    }

    pub(crate) fn handle_savant_stream_eos(&self, eos: &EndOfStream) -> OptionalGstFlowReturn {
        gstreamer::info!(
            CAT,
            imp = self,
            "Received EOS message for the source {}",
            eos.source_id
        );
        let savant_eos_event = build_savant_eos_event(&eos.source_id);
        for pad in self.obj().pads() {
            if pad.direction() == gstreamer::PadDirection::Src
                && !pad.push_event(savant_eos_event.clone())
            {
                gstreamer::error!(
                    CAT,
                    imp = self,
                    "Failed to push EOS event to pad {}",
                    pad.name()
                );
                return ERROR_PROCESSING_OPT;
            }
        }
        SKIP_PROCESSING
    }

    pub(crate) fn handle_shutdown(&self, shutdown: &Shutdown) -> OptionalGstFlowReturn {
        let settings_bind = self.settings.lock();
        let shutdown_auth_opt = settings_bind.shutdown_authorization.as_ref();
        if shutdown_auth_opt.is_none() {
            gstreamer::warning!(
                CAT,
                imp = self,
                "Received shutdown message but shutdown authorization is not set. The shutdown request will be ignored."
            );
            return SKIP_PROCESSING;
        }
        let shutdown_auth = shutdown_auth_opt.unwrap();
        if shutdown.get_auth() != shutdown_auth.as_str() {
            gstreamer::warning!(
                CAT,
                imp = self,
                "Received shutdown message but authorization does not match the expected value. The shutdown request will be ignored."
            );
            return SKIP_PROCESSING;
        }
        gstreamer::info!(
            CAT,
            imp = self,
            "Received shutdown message with correct authorization. The shutdown request will be processed."
        );
        self.send_eos_downstream_pads();
        EOS_PROCESSING_OPT
    }

    fn send_eos_downstream_pads(&self) {
        for pad in self.obj().pads() {
            if pad.direction() == gstreamer::PadDirection::Src
                && !pad.push_event(gstreamer::event::Eos::new())
            {
                gstreamer::error!(
                    CAT,
                    imp = self,
                    "Failed to push EOS event to pad {}",
                    pad.name()
                );
                panic!("Failed to push EOS event to pad {}", pad.name());
            }
        }
    }
}

fn human_readable_routing_id(routing_id: &Option<Vec<u8>>) -> String {
    routing_id
        .as_ref()
        .map(|r| bytes_to_hex_string(r))
        .unwrap_or(String::from("routing-id-not-set"))
}

fn get_frame_key_info_as_string(frame: &VideoFrameProxy) -> String {
    let frame_pts = convert_ts(frame.get_pts(), frame.get_time_base());
    let frame_dts = frame
        .get_dts()
        .map(|dts| convert_ts(dts, frame.get_time_base()));
    let frame_duration = frame
        .get_duration()
        .map(|duration| convert_ts(duration, frame.get_time_base()));
    let keyframe = frame.get_keyframe();
    let source_id = frame.get_source_id();
    let codec = frame.get_codec();
    let width = frame.get_width();
    let height = frame.get_height();
    let creation_timestamp_ns = frame.get_creation_timestamp_ns();
    let previous_keyframe = frame.get_previous_keyframe();
    format!(
        "Source: {}, PTS: {}, DTS: {:?}, Duration: {:?}, keyframe: {:?}, codec: {:?}, width: {}, height: {}, creation_timestamp_ns: {}, previous_keyframe: {:?}",
        source_id, frame_pts, frame_dts, frame_duration, keyframe, codec, width, height, creation_timestamp_ns, previous_keyframe
    )
}

fn build_savant_eos_event(source_id: &str) -> gstreamer::Event {
    let eos_struct = gstreamer::Structure::builder(SAVANT_EOS_EVENT_NAME)
        .field(SAVANT_EOS_EVENT_SOURCE_ID_PROPERTY, source_id)
        .build();
    gstreamer::event::CustomDownstream::new(eos_struct)
}

fn build_savant_userdata_event(data: &Message) -> gstreamer::Event {
    let serialized_data = save_message(data).expect("Failed to serialize UserData");
    let userdata_struct = gstreamer::Structure::builder(SAVANT_USERDATA_EVENT_NAME)
        .field(
            SAVANT_USERDATA_EVENT_DATA_PROPERTY,
            BASE64_STANDARD.encode(&serialized_data),
        )
        .build();
    gstreamer::event::CustomDownstream::new(userdata_struct)
}

pub mod label_filter;
pub mod label_filter_parser;

use crate::otlp::PropagatedContext;
use crate::primitives::eos::EndOfStream;
use crate::primitives::frame::VideoFrameProxy;
use crate::primitives::frame_batch::VideoFrameBatch;
use crate::primitives::frame_update::VideoFrameUpdate;
use crate::primitives::shutdown::Shutdown;
use crate::primitives::userdata::UserData;
use crate::primitives::WithAttributes;
use crate::protobuf::{deserialize, serialize};
use lru::LruCache;

pub struct SeqStore {
    counters: LruCache<(String, String), u64>,
}

pub const AUX_SEQ_ID_KEY: &str = "2c364c5fbe4de1c230e7df0b7d64ee1d";
pub const MAX_SEQ_STORE_SIZE: usize = 512;

impl Default for SeqStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SeqStore {
    pub fn new() -> Self {
        Self {
            counters: LruCache::new(std::num::NonZeroUsize::new(MAX_SEQ_STORE_SIZE).unwrap()),
        }
    }

    pub fn generate_message_seq_id(&mut self, system_id: &str, source: &str) -> u64 {
        let v = self
            .counters
            .get_or_insert_mut((system_id.to_string(), source.to_string()), || 0);
        *v += 1;
        *v
    }

    fn validate_seq_i_raw(&mut self, system_id: &str, source: &str, seq_id: u64) -> bool {
        let v = self
            .counters
            .get_or_insert_mut((system_id.to_string(), source.to_string()), || 0);
        if seq_id <= *v {
            log::trace!(target: "savant_rs::message::validate_seq_iq", 
                "SeqId reset for system={} source=[{}] expected seq_id = {}, received seq_id = {}", 
                system_id, source, *v + 1, seq_id);
            *v = seq_id;
            true
        } else if *v + 1 == seq_id {
            log::trace!(target: "savant_rs::message::validate_seq_iq", 
                "Successfully validated seq_id={seq_id} for system={} source=[{}]",
                system_id, source);
            *v += 1;
            true
        } else {
            let log_source = if source == AUX_SEQ_ID_KEY {
                "auxiliary (eos, shutdown, unknown or batch)"
            } else {
                source
            };
            log::warn!(target: "savant_rs::message::validate_seq_iq", 
                "Failed to validate message from system={} source=[{}] seq_id={}, expected seq_id={}. SeqId discrepancy is a symptom of message loss or stream termination without EOS", 
                system_id, log_source, seq_id, *v + 1);
            *v = seq_id;
            false
        }
    }

    pub fn reset_seq_id(&mut self, system_id: &str, source: &str) {
        self.counters
            .pop(&(system_id.to_string(), source.to_string()));
    }

    pub fn get_seq_id(&mut self, system_id: &str, source: &str) -> u64 {
        *(self
            .counters
            .get(&(system_id.to_string(), source.to_string()))
            .unwrap_or(&0))
    }

    pub fn validate_seq_id(&mut self, m: &Message) -> bool {
        let seq_id = m.meta.seq_id;
        let system_id = &m.meta.system_id;
        match &m.payload {
            MessageEnvelope::EndOfStream(eos) => {
                self.validate_seq_i_raw(system_id, &eos.source_id, seq_id);
                self.reset_seq_id(system_id, &eos.source_id);
                true
            }
            MessageEnvelope::VideoFrame(vf) => {
                self.validate_seq_i_raw(system_id, &vf.inner.read().source_id, seq_id)
            }
            MessageEnvelope::UserData(ud) => {
                self.validate_seq_i_raw(system_id, &ud.source_id, seq_id)
            }
            _ => self.validate_seq_i_raw(system_id, AUX_SEQ_ID_KEY, seq_id),
        }
    }
}

#[derive(Debug)]
pub enum MessageEnvelope {
    EndOfStream(EndOfStream),
    VideoFrame(VideoFrameProxy),
    VideoFrameBatch(VideoFrameBatch),
    VideoFrameUpdate(VideoFrameUpdate),
    UserData(UserData),
    Shutdown(Shutdown),
    Unknown(String),
}

impl Clone for MessageEnvelope {
    fn clone(&self) -> Self {
        match self {
            Self::EndOfStream(arg0) => Self::EndOfStream(arg0.clone()),
            Self::VideoFrame(arg0) => Self::VideoFrame(arg0.smart_copy()),
            Self::VideoFrameBatch(arg0) => Self::VideoFrameBatch(arg0.smart_copy()),
            Self::VideoFrameUpdate(arg0) => Self::VideoFrameUpdate(arg0.clone()),
            Self::UserData(arg0) => Self::UserData(arg0.clone()),
            Self::Shutdown(arg0) => Self::Shutdown(arg0.clone()),
            Self::Unknown(arg0) => Self::Unknown(arg0.clone()),
        }
    }
}

pub const VERSION_LEN: usize = 4;

#[derive(Debug, Clone)]
pub struct MessageMeta {
    pub protocol_version: String,
    pub routing_labels: Vec<String>,
    pub span_context: PropagatedContext,
    pub seq_id: u64,
    pub system_id: String,
}

impl Default for MessageMeta {
    fn default() -> Self {
        Self::new("")
    }
}

impl MessageMeta {
    pub fn new(system_id: &str) -> Self {
        Self {
            protocol_version: savant_protobuf::version().to_string(),
            routing_labels: Vec::default(),
            span_context: PropagatedContext::default(),
            seq_id: 0,
            system_id: system_id.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Message {
    pub(crate) meta: MessageMeta,
    pub(crate) payload: MessageEnvelope,
}

impl Message {
    #[cfg(test)]
    pub fn set_protocol_version(&mut self, version: String) {
        self.meta.protocol_version = version;
    }

    pub fn unknown(s: String) -> Self {
        Self {
            meta: MessageMeta::default(),
            payload: MessageEnvelope::Unknown(s),
        }
    }

    pub fn user_data(mut t: UserData) -> Self {
        // let seq_id = generate_message_seq_id(t.get_source_id());
        t.exclude_temporary_attributes();
        Self {
            meta: MessageMeta::default(),
            payload: MessageEnvelope::UserData(t),
        }
    }

    pub fn end_of_stream(eos: EndOfStream) -> Self {
        // clear_source_seq_id(&eos.source_id);
        // let seq_id = generate_message_seq_id(&eos.source_id);
        Self {
            meta: MessageMeta::default(),
            payload: MessageEnvelope::EndOfStream(eos),
        }
    }

    pub fn shutdown(shutdown: Shutdown) -> Self {
        Self {
            meta: MessageMeta::default(),
            payload: MessageEnvelope::Shutdown(shutdown),
        }
    }

    pub fn video_frame(frame: &VideoFrameProxy) -> Self {
        //let seq_id = generate_message_seq_id(frame.get_source_id().as_str());
        let frame_ref = frame.clone();
        // let frame_copy = frame.deep_copy();
        // frame_copy.exclude_temporary_attributes();
        // frame_copy.get_all_objects().iter().for_each(|o| {
        //     o.exclude_temporary_attributes();
        // });
        Self {
            meta: MessageMeta::default(),
            payload: MessageEnvelope::VideoFrame(frame_ref),
        }
    }

    pub fn video_frame_batch(batch: &VideoFrameBatch) -> Self {
        let batch_copy = batch.clone();
        Self {
            meta: MessageMeta::default(),
            payload: MessageEnvelope::VideoFrameBatch(batch_copy),
        }
    }

    pub fn video_frame_update(update: VideoFrameUpdate) -> Self {
        Self {
            meta: MessageMeta::default(),
            payload: MessageEnvelope::VideoFrameUpdate(update),
        }
    }

    pub fn payload(&self) -> &MessageEnvelope {
        &self.payload
    }

    pub fn payload_mut(&mut self) -> &mut MessageEnvelope {
        &mut self.payload
    }

    pub fn meta(&self) -> &MessageMeta {
        &self.meta
    }

    pub fn meta_mut(&mut self) -> &mut MessageMeta {
        &mut self.meta
    }

    pub fn get_labels(&self) -> Vec<String> {
        self.meta.routing_labels.clone()
    }
    pub fn set_labels(&mut self, labels: Vec<String>) {
        self.meta.routing_labels = labels;
    }

    pub fn get_system_id(&self) -> &String {
        &self.meta.system_id
    }
    pub fn set_system_id(&mut self, system_id: &str) {
        self.meta.system_id = system_id.to_string();
    }

    pub fn get_seq_id(&self) -> u64 {
        self.meta.seq_id
    }
    pub fn set_seq_id(&mut self, seq_id: u64) {
        self.meta.seq_id = seq_id;
    }

    pub fn set_span_context(&mut self, context: PropagatedContext) {
        self.meta.span_context = context;
    }
    pub fn get_span_context(&self) -> &PropagatedContext {
        &self.meta.span_context
    }
    pub fn is_unknown(&self) -> bool {
        matches!(self.payload, MessageEnvelope::Unknown(_))
    }
    pub fn is_shutdown(&self) -> bool {
        matches!(self.payload, MessageEnvelope::Shutdown(_))
    }
    pub fn is_end_of_stream(&self) -> bool {
        matches!(self.payload, MessageEnvelope::EndOfStream(_))
    }
    pub fn is_user_data(&self) -> bool {
        matches!(self.payload, MessageEnvelope::UserData(_))
    }
    pub fn is_video_frame(&self) -> bool {
        matches!(self.payload, MessageEnvelope::VideoFrame(_))
    }
    pub fn is_video_frame_update(&self) -> bool {
        matches!(self.payload, MessageEnvelope::VideoFrameUpdate(_))
    }
    pub fn is_video_frame_batch(&self) -> bool {
        matches!(self.payload, MessageEnvelope::VideoFrameBatch(_))
    }
    pub fn as_unknown(&self) -> Option<String> {
        match &self.payload {
            MessageEnvelope::Unknown(s) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn as_shutdown(&self) -> Option<&Shutdown> {
        match &self.payload {
            MessageEnvelope::Shutdown(shutdown) => Some(shutdown),
            _ => None,
        }
    }

    pub fn as_end_of_stream(&self) -> Option<&EndOfStream> {
        match &self.payload {
            MessageEnvelope::EndOfStream(eos) => Some(eos),
            _ => None,
        }
    }
    pub fn as_user_data(&self) -> Option<&UserData> {
        match &self.payload {
            MessageEnvelope::UserData(data) => Some(data),
            _ => None,
        }
    }
    pub fn as_video_frame(&self) -> Option<VideoFrameProxy> {
        match &self.payload {
            MessageEnvelope::VideoFrame(frame) => Some(frame.clone()),
            _ => None,
        }
    }
    pub fn as_video_frame_update(&self) -> Option<&VideoFrameUpdate> {
        match &self.payload {
            MessageEnvelope::VideoFrameUpdate(update) => Some(update),
            _ => None,
        }
    }
    pub fn as_video_frame_batch(&self) -> Option<&VideoFrameBatch> {
        match &self.payload {
            MessageEnvelope::VideoFrameBatch(batch) => Some(batch),
            _ => None,
        }
    }
}

pub fn load_message(bytes: &[u8]) -> Message {
    let m: Result<Message, _> = deserialize(bytes);

    if m.is_err() {
        return Message::unknown(format!("{:?}", m.err().unwrap()));
    }

    let m = m.unwrap();

    if m.meta.protocol_version != savant_protobuf::version() {
        return Message::unknown(format!(
            "Message protocol version mismatch: message version={:?}, program expects version={:?}.",
            m.meta.protocol_version,
            savant_protobuf::version()
        ));
    }

    m
}

pub fn save_message(m: &Message) -> anyhow::Result<Vec<u8>> {
    Ok(serialize(m)?)
}

#[cfg(test)]
mod tests {
    use crate::message::{load_message, save_message, Message, SeqStore};
    use crate::primitives::eos::EndOfStream;
    use crate::primitives::frame_batch::VideoFrameBatch;
    use crate::primitives::object::private::SealedWithFrame;
    use crate::primitives::shutdown::Shutdown;
    use crate::primitives::userdata::UserData;
    use crate::primitives::WithAttributes;
    use crate::test::gen_frame;
    use std::sync::Arc;

    #[test]
    fn test_save_load_eos() {
        let eos = EndOfStream::new("test");
        let m = Message::end_of_stream(eos);
        let res = save_message(&m).unwrap();
        let m = load_message(&res);
        assert!(m.is_end_of_stream());
    }

    #[test]
    fn test_save_load_shutdown() {
        let s = Shutdown::new("test");
        let m = Message::shutdown(s);
        let res = save_message(&m).unwrap();
        let m = load_message(&res);
        assert!(m.is_shutdown());
    }

    #[test]
    fn test_save_load_user_data() {
        let t = UserData::new("test");
        let m = Message::user_data(t);
        let res = save_message(&m).unwrap();
        let m = load_message(&res);
        assert!(m.is_user_data());
    }

    #[test]
    fn test_save_load_video_frame() {
        let m = Message::video_frame(&gen_frame());
        let res = save_message(&m).unwrap();
        let m = load_message(&res);
        assert!(m.is_video_frame());
        let frame = m.as_video_frame().unwrap();
        // ensure objects belong the frame
        let obj = frame.get_object(0).unwrap();
        assert!(Arc::ptr_eq(
            &obj.get_frame().unwrap().inner.0,
            &frame.inner.0
        ));
    }

    #[test]
    fn test_save_load_unknown() {
        let m = Message::unknown("x".to_string());
        let res = save_message(&m).unwrap();
        let m = load_message(&res);
        assert!(m.is_unknown());
    }

    #[test]
    fn test_save_load_batch() {
        let mut batch = VideoFrameBatch::new();
        batch.add(1, gen_frame());
        batch.add(2, gen_frame());
        batch.add(3, gen_frame());
        let m = Message::video_frame_batch(&batch);
        let res = save_message(&m).unwrap();
        let m = load_message(&res);
        assert!(m.is_video_frame_batch());

        let b = m.as_video_frame_batch().unwrap();
        assert!(b.get(1).is_some());
        assert!(b.get(2).is_some());
        assert!(b.get(3).is_some());
        let f = b.get(1).unwrap();
        let mut attrs = f.get_attributes();
        attrs.sort();

        assert_eq!(
            attrs,
            vec![
                ("system".into(), "test".into()),
                ("system".into(), "test2".into()),
                ("system2".into(), "test2".into()),
                ("test".into(), "test".into()),
            ]
        );

        let _ = f.access_objects_with_id(&[0]).pop().unwrap();
    }

    #[test]
    fn test_save_load_frame_with_temp_attributes() {
        let mut f = gen_frame();
        let attrs = f.get_attributes();
        assert_eq!(attrs.len(), 4);
        f.set_temporary_attribute("chronos", "temp", &None, false, vec![]);
        let attrs = f.get_attributes();
        assert_eq!(attrs.len(), 5);
        let m = Message::video_frame(&f);
        let res = save_message(&m).unwrap();
        let m = load_message(&res);
        assert!(m.is_video_frame());
        let f = m.as_video_frame().unwrap();
        let attrs = f.get_attributes();
        assert_eq!(attrs.len(), 4);
    }

    // #[test]
    // fn test_save_load_seq_ids() {
    //     let mut f = gen_frame();
    //     f.set_source_id("test_save_load_seq_ids");
    //     let ud = UserData::new(&f.get_source_id());
    //     let eos = EndOfStream::new(f.get_source_id());
    //     let mf = Message::video_frame(&f);
    //     assert_eq!(mf.meta.seq_id, 1);
    //     let mud = Message::user_data(ud);
    //     assert_eq!(mud.meta.seq_id, 2);
    //     let meos = Message::end_of_stream(eos);
    //     assert_eq!(meos.meta.seq_id, 1);

    //     let ud = UserData::new(&format!("{}-2", f.get_source_id()));
    //     let eos = EndOfStream::new(format!("{}-2", f.get_source_id()));
    //     let mud = Message::user_data(ud);
    //     assert_eq!(mud.meta.seq_id, 1);
    //     let meos = Message::end_of_stream(eos);
    //     assert_eq!(meos.meta.seq_id, 1);
    // }

    #[test]
    fn test_validate_sequence_ids() {
        let system_id = "test_validate_sequence_ids";
        let source_id = "test_source_id";
        let mut sender_seq_store = SeqStore::new();
        let mut recv_seq_store = SeqStore::new();
        let mut f = gen_frame();
        f.set_source_id(source_id);
        let ud = UserData::new(source_id);
        let eos = EndOfStream::new(source_id);

        let mut mf = Message::video_frame(&f);
        mf.meta.system_id = system_id.to_string();
        mf.meta.seq_id = sender_seq_store.generate_message_seq_id(system_id, source_id);
        let mut mud = Message::user_data(ud.clone());
        mud.meta.system_id = system_id.to_string();
        mud.meta.seq_id = sender_seq_store.generate_message_seq_id(system_id, source_id);
        let mut meos = Message::end_of_stream(eos.clone());
        meos.meta.system_id = system_id.to_string();
        meos.meta.seq_id = sender_seq_store.generate_message_seq_id(system_id, source_id);

        assert!(recv_seq_store.validate_seq_id(&mf));
        assert!(recv_seq_store.validate_seq_id(&mud));
        assert!(recv_seq_store.validate_seq_id(&meos));

        sender_seq_store.reset_seq_id(system_id, source_id);

        let mut mf = Message::video_frame(&f);
        mf.meta.system_id = system_id.to_string();
        mf.meta.seq_id = sender_seq_store.generate_message_seq_id(system_id, source_id);
        let mut mud = Message::user_data(ud);
        mud.meta.system_id = system_id.to_string();
        mud.meta.seq_id = sender_seq_store.generate_message_seq_id(system_id, source_id);
        let mut meos = Message::end_of_stream(eos);
        meos.meta.system_id = system_id.to_string();
        meos.meta.seq_id = sender_seq_store.generate_message_seq_id(system_id, source_id);

        assert!(recv_seq_store.validate_seq_id(&mf));
        assert!(recv_seq_store.validate_seq_id(&mud));
        assert!(recv_seq_store.validate_seq_id(&meos));
    }

    // #[test]
    // fn test_validate_sequence_ids_with_misses() {
    //     let mut f = gen_frame();
    //     f.set_source_id("test_validate_sequence_ids_with_misses");

    //     let ud = UserData::new(&f.get_source_id());
    //     let eos = EndOfStream::new(f.get_source_id());

    //     let mf = Message::video_frame(&f);
    //     let _ = Message::video_frame(&f);
    //     let mud = Message::user_data(ud.clone());
    //     let meos = Message::end_of_stream(eos.clone());

    //     assert!(validate_seq_id(&mf));
    //     assert!(!validate_seq_id(&mud));
    //     assert!(validate_seq_id(&meos));
    // }

    // #[test]
    // fn test_validate_sequence_id_reset_on_eos() {
    //     let mut f = gen_frame();
    //     f.set_source_id("test_validate_sequence_id_reset_on_eos");

    //     let eos = EndOfStream::new(f.get_source_id());

    //     let mf = Message::video_frame(&f);
    //     let meos = Message::end_of_stream(eos);

    //     assert!(validate_seq_id(&meos));
    //     assert!(validate_seq_id(&mf));
    // }
}

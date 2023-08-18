use crate::otlp::PropagatedContext;
use crate::primitives::eos::EndOfStream;
use crate::primitives::frame::{VideoFrame, VideoFrameProxy};
use crate::primitives::frame_batch::VideoFrameBatch;
use crate::primitives::frame_update::VideoFrameUpdate;
use crate::primitives::userdata::UserData;
use crate::primitives::{AttributeMethods, Attributive};
use crate::{trace, version_to_bytes_le};
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub enum MessageEnvelope {
    EndOfStream(EndOfStream),
    VideoFrame(Box<VideoFrame>),
    VideoFrameBatch(VideoFrameBatch),
    VideoFrameUpdate(VideoFrameUpdate),
    UserData(UserData),
    Unknown(String),
}

pub const VERSION_LEN: usize = 4;

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct MessageMeta {
    pub lib_version: [u8; VERSION_LEN],
    pub routing_labels: Vec<String>,
    pub span_context: PropagatedContext,
}

impl Default for MessageMeta {
    fn default() -> Self {
        Self::new()
    }
}

impl MessageMeta {
    pub fn new() -> Self {
        Self {
            lib_version: version_to_bytes_le(),
            routing_labels: Vec::default(),
            span_context: PropagatedContext::default(),
        }
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct Message {
    meta: MessageMeta,
    payload: MessageEnvelope,
}

impl Message {
    pub fn unknown(s: String) -> Self {
        Self {
            meta: MessageMeta::new(),
            payload: MessageEnvelope::Unknown(s),
        }
    }

    pub fn user_data(mut t: UserData) -> Self {
        t.exclude_temporary_attributes();

        Self {
            meta: MessageMeta::new(),
            payload: MessageEnvelope::UserData(t),
        }
    }

    pub fn end_of_stream(eos: EndOfStream) -> Self {
        Self {
            meta: MessageMeta::new(),
            payload: MessageEnvelope::EndOfStream(eos),
        }
    }
    pub fn video_frame(frame: &VideoFrameProxy) -> Self {
        let frame_copy = frame.deep_copy();

        frame_copy.exclude_temporary_attributes();
        frame_copy.get_all_objects().iter().for_each(|o| {
            o.exclude_temporary_attributes();
        });
        frame_copy.make_snapshot();

        let inner = trace!(frame_copy.inner.read()).clone();

        Self {
            meta: MessageMeta::new(),
            payload: MessageEnvelope::VideoFrame(inner),
        }
    }

    pub fn video_frame_batch(batch: &VideoFrameBatch) -> Self {
        let mut batch_copy = batch.deep_copy();
        batch_copy.prepare_before_save();
        Self {
            meta: MessageMeta::new(),
            payload: MessageEnvelope::VideoFrameBatch(batch_copy),
        }
    }

    pub fn video_frame_update(update: VideoFrameUpdate) -> Self {
        Self {
            meta: MessageMeta::new(),
            payload: MessageEnvelope::VideoFrameUpdate(update),
        }
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
    pub fn set_span_context(&mut self, context: PropagatedContext) {
        self.meta.span_context = context;
    }
    pub fn get_span_context(&self) -> &PropagatedContext {
        &self.meta.span_context
    }
    pub fn is_unknown(&self) -> bool {
        matches!(self.payload, MessageEnvelope::Unknown(_))
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
            MessageEnvelope::VideoFrame(frame) => Some(VideoFrameProxy::from_inner(*frame.clone())),
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
    let m: Result<Message, _> = rkyv::from_bytes(bytes);

    if m.is_err() {
        return Message::unknown(format!("{:?}", m.err().unwrap()));
    }

    let mut m = m.unwrap();

    if m.meta.lib_version != version_to_bytes_le() {
        return Message::unknown(format!(
            "Message CRC32 version mismatch: {:?} != {:?}. Expected version: {}",
            m.meta.lib_version,
            crate::version_crc32(),
            crate::version()
        ));
    }

    match &mut m.payload {
        MessageEnvelope::VideoFrame(f) => {
            f.restore();
        }
        MessageEnvelope::VideoFrameBatch(b) => {
            b.prepare_after_load();
        }
        _ => {}
    }

    m
}

pub fn save_message(m: &Message) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1024);
    buf.extend_from_slice(
        rkyv::to_bytes::<_, 1024>(m)
            .expect("Failed to serialize Message")
            .as_ref(),
    );
    buf
}

#[cfg(test)]
mod tests {
    use crate::message::{load_message, save_message, Message};
    use crate::primitives::attribute::AttributeMethods;
    use crate::primitives::eos::EndOfStream;
    use crate::primitives::frame_batch::VideoFrameBatch;
    use crate::primitives::userdata::UserData;
    use crate::primitives::Attribute;
    use crate::test::gen_frame;

    #[test]
    fn test_save_load_eos() {
        let eos = EndOfStream::new("test".to_string());
        let m = Message::end_of_stream(eos);
        let res = save_message(&m);
        let m = load_message(&res);
        assert!(m.is_end_of_stream());
    }

    #[test]
    fn test_save_load_user_data() {
        let t = UserData::new("test".to_string());
        let m = Message::user_data(t);
        let res = save_message(&m);
        let m = load_message(&res);
        assert!(m.is_user_data());
    }

    #[test]
    fn test_save_load_video_frame() {
        let m = Message::video_frame(&gen_frame());
        let res = save_message(&m);
        let m = load_message(&res);
        assert!(m.is_video_frame());
    }

    #[test]
    fn test_save_load_unknown() {
        let m = Message::unknown("x".to_string());
        let res = save_message(&m);
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
        let res = save_message(&m);
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

        let _ = f.access_objects_by_id(&vec![0]).pop().unwrap();
    }

    #[test]
    fn test_save_load_frame_with_temp_attributes() {
        let f = gen_frame();
        let tmp_attr =
            Attribute::temporary("chronos".to_string(), "temp".to_string(), vec![], None);
        let attrs = f.get_attributes();
        assert_eq!(attrs.len(), 4);
        f.set_attribute(tmp_attr);
        let attrs = f.get_attributes();
        assert_eq!(attrs.len(), 5);
        let m = Message::video_frame(&f);
        let res = save_message(&m);
        let m = load_message(&res);
        assert!(m.is_video_frame());
        let f = m.as_video_frame().unwrap();
        let attrs = f.get_attributes();
        assert_eq!(attrs.len(), 4);
    }
}

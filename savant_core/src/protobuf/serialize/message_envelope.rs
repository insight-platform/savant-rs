use crate::message::MessageEnvelope;
use crate::primitives::eos::EndOfStream;
use crate::primitives::frame::VideoFrame;
use crate::primitives::frame_batch::VideoFrameBatch;
use crate::primitives::frame_update::VideoFrameUpdate;
use crate::primitives::rust::{Shutdown, UserData};
use crate::protobuf::{generated, serialize};

impl From<&MessageEnvelope> for generated::message::Content {
    fn from(value: &MessageEnvelope) -> Self {
        match value {
            MessageEnvelope::EndOfStream(eos) => {
                generated::message::Content::EndOfStream(generated::EndOfStream {
                    source_id: eos.source_id.clone(),
                })
            }
            MessageEnvelope::VideoFrame(vf) => generated::message::Content::VideoFrame(vf.into()),
            MessageEnvelope::VideoFrameBatch(vfb) => {
                generated::message::Content::VideoFrameBatch(vfb.into())
            }

            MessageEnvelope::VideoFrameUpdate(vfu) => {
                generated::message::Content::VideoFrameUpdate(vfu.into())
            }
            MessageEnvelope::UserData(ud) => generated::message::Content::UserData(ud.into()),
            MessageEnvelope::Shutdown(s) => {
                generated::message::Content::Shutdown(generated::Shutdown {
                    auth: s.auth.clone(),
                })
            }
            MessageEnvelope::Unknown(m) => {
                generated::message::Content::Unknown(generated::Unknown { message: m.clone() })
            }
        }
    }
}

impl TryFrom<&generated::message::Content> for MessageEnvelope {
    type Error = serialize::Error;

    fn try_from(value: &generated::message::Content) -> Result<Self, Self::Error> {
        Ok(match value {
            generated::message::Content::EndOfStream(eos) => {
                MessageEnvelope::EndOfStream(EndOfStream {
                    source_id: eos.source_id.clone(),
                })
            }
            generated::message::Content::VideoFrame(vf) => {
                MessageEnvelope::VideoFrame(Box::new(VideoFrame::try_from(vf)?))
            }
            generated::message::Content::VideoFrameBatch(vfb) => {
                MessageEnvelope::VideoFrameBatch(VideoFrameBatch::try_from(vfb)?)
            }
            generated::message::Content::VideoFrameUpdate(vfu) => {
                MessageEnvelope::VideoFrameUpdate(VideoFrameUpdate::try_from(vfu)?)
            }
            generated::message::Content::UserData(ud) => {
                MessageEnvelope::UserData(UserData::try_from(ud)?)
            }
            generated::message::Content::Shutdown(s) => MessageEnvelope::Shutdown(Shutdown {
                auth: s.auth.clone(),
            }),
            generated::message::Content::Unknown(u) => MessageEnvelope::Unknown(u.message.clone()),
        })
    }
}

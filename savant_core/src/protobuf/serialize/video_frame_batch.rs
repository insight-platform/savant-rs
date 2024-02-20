use crate::primitives::frame::VideoFrameProxy;
use crate::primitives::frame_batch::VideoFrameBatch;
use crate::protobuf::{generated, serialize};

impl From<&VideoFrameBatch> for generated::VideoFrameBatch {
    fn from(batch: &VideoFrameBatch) -> Self {
        generated::VideoFrameBatch {
            batch: batch
                .frames()
                .iter()
                .map(|(id, f)| (*id, generated::VideoFrame::from(f)))
                .collect(),
        }
    }
}

impl TryFrom<&generated::VideoFrameBatch> for VideoFrameBatch {
    type Error = serialize::Error;

    fn try_from(b: &generated::VideoFrameBatch) -> Result<Self, Self::Error> {
        let mut batch = VideoFrameBatch::new();
        for (id, f) in b.batch.iter() {
            batch.add(*id, VideoFrameProxy::try_from(f)?);
        }
        Ok(batch)
    }
}

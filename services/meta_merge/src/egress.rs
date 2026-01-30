use savant_core::primitives::{eos::EndOfStream, frame::VideoFrame};

enum EgressMessage {
    VideoFrame(VideoFrame),
    EndOfStream(EndOfStream),
}

struct EgressItem {
    message: EgressMessage,
    data: Vec<Vec<u8>>,
}

struct EgressQueue {}

use crate::protocol::generated;

pub(crate) mod protocol;

impl TryFrom<generated::Message> for savant_core::message::Message {
    type Error = String;

    fn try_from(_: generated::Message) -> Result<Self, Self::Error> {
        todo!()
    }
}

pub fn serialize(_: savant_core::message::Message) -> Vec<u8> {
    todo!()
}

pub fn deserialize(_: &[u8]) -> savant_core::message::Message {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::protocol::generated::EndOfStream;

    #[test]
    fn test_items() {}
}

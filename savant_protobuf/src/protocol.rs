use savant_core::message::Message;

pub(crate) mod generated;
mod serialize;

impl From<Message> for generated::Message {
    fn from(m: Message) -> Self {
        generated::Message {
            lib_version: m.meta().lib_version.clone(),
            routing_labels: m.meta().routing_labels.clone(),
            propagated_context: m.meta().span_context.0.clone(),
            seq_id: m.meta().seq_id,
            content: Some(m.payload().into()),
        }
    }
}

use std::collections::VecDeque;

use crate::{configuration::ServiceConfiguration, egress_mapper::EgressMapper};
use log::warn;
use savant_core::{
    message::{label_filter::LabelFilterRule, label_filter_parser::LabelExpressionParser, Message},
    transport::zeromq::{NonBlockingWriter, WriteOperationResult, WriterResult},
};

struct SinkSpec {
    label_expression: Option<LabelFilterRule>,
    writer_name: String,
    writer: NonBlockingWriter,
    high_watermark: f64,
    op_results: VecDeque<WriteOperationResult>,
}

impl SinkSpec {
    pub fn can_send(&mut self) -> bool {
        let writer = &mut self.writer;
        if self.op_results.len() >= writer.get_max_inflight_messages() {
            warn!(
                "Sink {} has reached the limit of pending results (max: {})",
                self.writer_name,
                writer.get_max_inflight_messages()
            );
            return false;
        }

        let watermark = self.high_watermark;
        if writer.inflight_messages()
            >= (watermark * writer.get_max_inflight_messages() as f64) as usize
        {
            warn!(
                "High watermark reached for sink {}. Current inflight messages: {}, watermark: {}, max inflight messages: {}",
                self.writer_name,
                writer.inflight_messages(),
                watermark,
                writer.get_max_inflight_messages()
            );
        }

        if !writer.has_capacity() {
            warn!(
                "Sink {} has no capacity: check the downstream consumer. Current inflight messages: {}, watermark: {}, max inflight messages: {}. This message will be dropped for this sink.",
                self.writer_name,
                writer.inflight_messages(),
                watermark,
                writer.get_max_inflight_messages()
            );
            false
        } else {
            true
        }
    }

    pub fn send(&mut self, topic: &str, m: &Message, payload: &[&[u8]]) -> anyhow::Result<()> {
        let writer = &mut self.writer;

        if let Some(label_expression) = &self.label_expression {
            if !label_expression.matches(&m.get_labels()) {
                return Ok(());
            }
        }
        let res = writer.send_message(topic, m, payload);
        match res {
            Err(e) => {
                warn!("Failed to send message to sink {}: {}", self.writer_name, e);
            }
            Ok(res) => {
                self.op_results.push_back(res);
            }
        }
        Ok(())
    }

    pub fn handle_pending_responses(&mut self) -> anyhow::Result<()> {
        while !self.op_results.is_empty() {
            let res = self.op_results.front().unwrap();
            match res.try_get() {
                Ok(Some(res)) => {
                    self.op_results.pop_front();
                    let res = res?;
                    match res {
                        WriterResult::SendTimeout => {
                            warn!(
                                "Failed to send message to sink {}: {:?}",
                                self.writer_name, res
                            );
                        }
                        WriterResult::AckTimeout(_) => {
                            warn!(
                                "Failed to send message to sink {}: {:?}",
                                self.writer_name, res
                            );
                        }
                        _ => {}
                    }
                }
                _ => break,
            }
        }
        Ok(())
    }
}

pub struct Egress {
    sinks: Vec<SinkSpec>,
    source_mapper: EgressMapper,
}

impl Egress {
    pub fn new(config: &ServiceConfiguration) -> anyhow::Result<Self> {
        let mut sinks = Vec::new();
        for sink in &config.egress {
            let writer = NonBlockingWriter::try_from(&sink.socket)?;

            sinks.push(SinkSpec {
                label_expression: match &sink.matcher {
                    Some(matcher) => Some(LabelExpressionParser::parse(matcher.as_str())?),
                    None => None,
                },
                writer_name: sink.name.clone(),
                writer,
                high_watermark: sink.high_watermark.unwrap(),
                op_results: VecDeque::new(),
            });
        }
        Ok(Self {
            sinks,
            source_mapper: EgressMapper::from(config),
        })
    }

    pub fn process(
        &mut self,
        message_id: usize,
        topic: &str,
        m: &Message,
        payload: &[&[u8]],
    ) -> anyhow::Result<()> {
        for sink in &mut self.sinks {
            sink.handle_pending_responses()?;
            if !sink.can_send() {
                continue;
            }
            let sink_name = sink.writer_name.clone();
            let (new_topic, new_message) =
                self.source_mapper.map(&sink_name, message_id, topic, m)?;
            sink.send(&new_topic, &new_message, payload)?;
        }
        Ok(())
    }
}

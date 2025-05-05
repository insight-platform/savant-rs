use hashbrown::HashMap;
use lru::LruCache;
use savant_core::message::{Message, MessageEnvelope};
use std::{num::NonZeroUsize, time::SystemTime};

use crate::configuration::ServiceConfiguration;

pub struct EgressMapper {
    source_cache: LruCache<String, (String, SystemTime)>,
    topic_cache: LruCache<String, (String, SystemTime)>,
    source_mappers: HashMap<String, String>,
    topic_mappers: HashMap<String, String>,
}

impl EgressMapper {
    fn change_source_id(m: &mut Message, new_source_id: &str) {
        match m.payload_mut() {
            MessageEnvelope::EndOfStream(end_of_stream) => {
                end_of_stream.source_id = new_source_id.to_string();
            }
            MessageEnvelope::VideoFrame(video_frame) => {
                video_frame.set_source_id(new_source_id);
            }
            MessageEnvelope::UserData(user_data) => {
                user_data.source_id = new_source_id.to_string();
            }
            _ => {}
        }
    }

    pub fn map(
        &mut self,
        sink_name: &str,
        topic: &str,
        m: &Message,
    ) -> anyhow::Result<(String, Message)> {
        // only when not cached or video frame key frame
        let new_topic = self.map_topic(sink_name, topic, m)?;
        let new_m = self.map_source(sink_name, m)?;
        Ok((new_topic, new_m))
    }

    fn switch_allowed(m: &Message) -> bool {
        match m.payload() {
            MessageEnvelope::VideoFrame(video_frame_proxy) => {
                matches!(video_frame_proxy.get_keyframe(), Some(true))
            }
            _ => false,
        }
    }

    fn map_source(&mut self, sink_name: &str, m: &Message) -> anyhow::Result<Message> {
        let mut new_message = m.clone();

        let source_id_opt = match m.payload() {
            MessageEnvelope::EndOfStream(end_of_stream) => Some(end_of_stream.source_id.clone()),
            MessageEnvelope::VideoFrame(video_frame_proxy) => {
                Some(video_frame_proxy.get_source_id())
            }
            MessageEnvelope::UserData(user_data) => Some(user_data.source_id.clone()),
            _ => None,
        };

        if source_id_opt.is_none() {
            return Ok(new_message);
        }

        let source_id = source_id_opt.unwrap();

        let new_source_id_opt = self.get_mapped_source(
            sink_name,
            &source_id,
            &m.get_labels(),
            Self::switch_allowed(m),
        )?;

        if let Some(new_source_id) = new_source_id_opt {
            Self::change_source_id(&mut new_message, new_source_id.as_str());
        }
        Ok(new_message)
    }

    fn map_topic(&mut self, sink_name: &str, topic: &str, m: &Message) -> anyhow::Result<String> {
        let new_topic =
            self.get_mapped_topic(sink_name, topic, &m.get_labels(), Self::switch_allowed(m))?;
        Ok(new_topic)
    }

    fn get_mapped_source(
        &mut self,
        sink_name: &str,
        source_id: &str,
        labels: &[String],
        switch_allowed: bool,
    ) -> anyhow::Result<Option<String>> {
        let cached_source = self.source_cache.get(source_id);
        if let Some((cached_source, change_at)) = cached_source {
            if !switch_allowed || SystemTime::now() < *change_at {
                return Ok(Some(cached_source.clone()));
            }
        }

        // invoke handler and get mapped source name

        todo!()
    }

    fn get_mapped_topic(
        &mut self,
        sink_name: &str,
        topic: &str,
        labels: &[String],
        switch_allowed: bool,
    ) -> anyhow::Result<String> {
        let cached_topic = self.topic_cache.get(topic);
        if let Some((cached_topic, change_at)) = cached_topic {
            if !switch_allowed || SystemTime::now() < *change_at {
                return Ok(cached_topic.clone());
            }
        }
        todo!()
    }
}

impl From<&ServiceConfiguration> for EgressMapper {
    fn from(config: &ServiceConfiguration) -> Self {
        let mut source_handlers = HashMap::new();
        let mut topic_handlers = HashMap::new();

        for e in &config.egress {
            if let Some(handler) = &e.source_mapper {
                source_handlers.insert(e.name.clone(), handler.clone());
            }
            if let Some(handler) = &e.topic_mapper {
                topic_handlers.insert(e.name.clone(), handler.clone());
            }
        }

        Self {
            source_cache: LruCache::new(
                NonZeroUsize::new(config.common.name_cache.as_ref().unwrap().size).unwrap(),
            ),
            topic_cache: LruCache::new(
                NonZeroUsize::new(config.common.name_cache.as_ref().unwrap().size).unwrap(),
            ),
            source_mappers: source_handlers,
            topic_mappers: topic_handlers,
        }
    }
}

use hashbrown::HashMap;
use lru::LruCache;
use pyo3::Python;
use savant_core::message::{Message, MessageEnvelope};
use savant_core_py::REGISTERED_HANDLERS;
use std::{
    num::NonZeroUsize,
    time::{Duration, SystemTime},
};

use crate::configuration::ServiceConfiguration;

pub struct MapCache {
    cache: LruCache<String, (String, SystemTime)>,
    ttl: Duration,
}

impl MapCache {
    pub fn new(size: usize, ttl: Duration) -> Self {
        Self {
            cache: LruCache::new(NonZeroUsize::new(size).unwrap()),
            ttl,
        }
    }

    pub fn get_or_update(
        &mut self,
        key: &str,
        is_permited: bool,
        update: impl FnOnce() -> anyhow::Result<String>,
    ) -> anyhow::Result<String> {
        let cached_value = self.cache.get(key);
        if let Some((cached_value, change_at)) = cached_value {
            if !is_permited || SystemTime::now() < *change_at {
                return Ok(cached_value.clone());
            }
        }

        let new_value = update()?;
        let expires_at = SystemTime::now().checked_add(self.ttl).unwrap();
        self.cache
            .put(key.to_string(), (new_value.clone(), expires_at));
        Ok(new_value)
    }
}

pub struct EgressMapper {
    source_cache: MapCache,
    topic_cache: MapCache,
    source_mappers: HashMap<String, String>,
    topic_mappers: HashMap<String, String>,
}

impl EgressMapper {
    fn set_source(m: &mut Message, new_source_id: &str) {
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

        let new_source_id = self.get_mapped_source(
            sink_name,
            &source_id,
            &m.get_labels(),
            Self::switch_allowed(m),
        )?;
        Self::set_source(&mut new_message, &new_source_id);
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
    ) -> anyhow::Result<String> {
        self.source_cache
            .get_or_update(source_id, switch_allowed, || {
                let source_mapper = self.source_mappers.get(sink_name);
                if let Some(source_mapper) = source_mapper {
                    Python::with_gil(|py| {
                        let handlers_bind = REGISTERED_HANDLERS.read();
                        let handler = handlers_bind
                            .get(source_mapper.as_str())
                            .unwrap_or_else(|| panic!("Handler {} not found", source_mapper));
                        let res = handler.call1(py, (sink_name, source_id, labels.to_vec()))?;
                        let new_source_id = res.extract::<String>(py)?;
                        Ok(new_source_id)
                    })
                } else {
                    Ok(source_id.to_string())
                }
            })
    }

    fn get_mapped_topic(
        &mut self,
        sink_name: &str,
        topic: &str,
        labels: &[String],
        switch_allowed: bool,
    ) -> anyhow::Result<String> {
        self.topic_cache.get_or_update(topic, switch_allowed, || {
            let topic_mapper = self.topic_mappers.get(topic);
            if let Some(topic_mapper) = topic_mapper {
                Python::with_gil(|py| {
                    let handlers_bind = REGISTERED_HANDLERS.read();
                    let handler = handlers_bind
                        .get(topic_mapper.as_str())
                        .unwrap_or_else(|| panic!("Handler {} not found", topic_mapper));
                    let res = handler.call1(py, (sink_name, topic, labels.to_vec()))?;
                    let new_topic = res.extract::<String>(py)?;
                    Ok(new_topic)
                })
            } else {
                Ok(topic.to_string())
            }
        })
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

        let cache_size = config.common.name_cache.as_ref().unwrap().size;
        let cache_ttl = config.common.name_cache.as_ref().unwrap().ttl;

        Self {
            source_cache: MapCache::new(cache_size, cache_ttl),
            topic_cache: MapCache::new(cache_size, cache_ttl),
            source_mappers: source_handlers,
            topic_mappers: topic_handlers,
        }
    }
}

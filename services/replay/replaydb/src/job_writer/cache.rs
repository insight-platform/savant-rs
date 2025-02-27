use crate::job_writer::{JobWriter, SinkConfiguration};
use anyhow::Result;
use mini_moka::sync::{Cache, CacheBuilder};
use savant_core::transport::zeromq::NonBlockingWriter;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

pub struct JobWriterCache {
    cache: Cache<SinkConfiguration, Arc<Mutex<JobWriter>>>,
}

// safety: JobWriterCache is Send and Sync because the underlying cache is Send and Sync
unsafe impl Send for JobWriterCache {}

// safety: JobWriterCache is Send and Sync because the underlying cache is Send and Sync
unsafe impl Sync for JobWriterCache {}

impl JobWriterCache {
    pub fn new(max_capacity: NonZeroU64, ttl: Duration) -> Self {
        Self {
            cache: CacheBuilder::new(max_capacity.get())
                .time_to_live(ttl)
                .build(),
        }
    }

    pub fn get(&mut self, configuration: &SinkConfiguration) -> Result<Arc<Mutex<JobWriter>>> {
        if let Some(w) = self.cache.get(configuration) {
            Ok(w.clone())
        } else {
            let writer = Arc::new(Mutex::new(
                NonBlockingWriter::try_from(configuration)?.into(),
            ));
            self.cache.insert(configuration.clone(), writer.clone());
            Ok(writer)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::job_writer::cache::JobWriterCache;
    use crate::job_writer::SinkConfiguration;
    use anyhow::Result;
    use std::num::NonZeroU64;
    use std::sync::Arc;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test() -> Result<()> {
        let mut cache = JobWriterCache::new(
            NonZeroU64::new(1024).expect("Must be positive"),
            Duration::from_millis(10),
        );
        let conf = SinkConfiguration::new(
            "dealer+connect:ipc:///tmp/in",
            Duration::from_secs(1),
            3,
            Duration::from_secs(1),
            3,
            1000,
            1000,
            100,
        );
        let w1 = cache.get(&conf)?;
        let w2 = cache.get(&conf)?;
        assert!(Arc::ptr_eq(&w1, &w2));
        sleep(Duration::from_millis(20));
        let w3 = cache.get(&conf)?;
        assert!(!Arc::ptr_eq(&w1, &w3));
        Ok(())
    }
}

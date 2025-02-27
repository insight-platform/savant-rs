use crate::job::query::JobQuery;
use crate::job::{Job, RocksDbJob};
use crate::job_writer::cache::JobWriterCache;
use crate::store::rocksdb::RocksDbStore;
use crate::store::{Store, SyncRocksDbStore};
use anyhow::Result;
use savant_core::primitives::frame_update::VideoFrameUpdate;
use savant_core::utils::uuid_v7::incremental_uuid_v7;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

pub struct RocksDbJobFactory(JobFactory<RocksDbStore>);

impl RocksDbJobFactory {
    pub fn new(
        store: SyncRocksDbStore,
        writer_cache_max_capacity: NonZeroU64,
        writer_cache_ttl: Duration,
    ) -> Result<Self> {
        Ok(Self(JobFactory::new(
            store,
            writer_cache_max_capacity,
            writer_cache_ttl,
        )))
    }

    pub fn store(&self) -> SyncRocksDbStore {
        self.0.store.clone()
    }

    pub async fn create_job(&mut self, query: JobQuery) -> Result<RocksDbJob> {
        Ok(RocksDbJob(self.0.create_job(query).await?))
    }
}

pub(crate) struct JobFactory<S: Store> {
    store: Arc<Mutex<S>>,
    writer_cache: JobWriterCache,
}

impl<S> JobFactory<S>
where
    S: Store,
{
    pub fn new(
        store: Arc<Mutex<S>>,
        writer_cache_max_capacity: NonZeroU64,
        writer_cache_ttl: Duration,
    ) -> Self {
        Self {
            store,
            writer_cache: JobWriterCache::new(writer_cache_max_capacity, writer_cache_ttl),
        }
    }
    pub async fn create_job(&mut self, query: JobQuery) -> Result<Job<S>> {
        let writer = self.writer_cache.get(&query.sink)?;
        let job_id = incremental_uuid_v7();
        let anchor_uuid = uuid::Uuid::parse_str(&query.anchor_keyframe)?;
        let anchor_wait_duration = query
            .anchor_wait_duration
            .unwrap_or(Duration::from_millis(1));

        let mut update = VideoFrameUpdate::default();
        for attribute in query.attributes.iter() {
            update.add_frame_attribute(attribute.clone());
        }

        Job::new(
            self.store.clone(),
            writer,
            job_id,
            anchor_uuid,
            anchor_wait_duration,
            query.offset,
            query.stop_condition,
            query.configuration,
            Some(update),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::job::configuration::JobConfigurationBuilder;
    use crate::job::factory::RocksDbJobFactory;
    use crate::job::query::JobQuery;
    use crate::job::stop_condition::JobStopCondition;
    use crate::job_writer::SinkConfiguration;
    use crate::store::rocksdb::RocksDbStore;
    use crate::store::{gen_properly_filled_frame, JobOffset, Store};
    use anyhow::Result;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Mutex;

    #[tokio::test]
    async fn test_create_rocksdb_job() -> Result<()> {
        let dir = tempfile::TempDir::new()?;
        let path = dir.path();
        let store = Arc::new(Mutex::new(RocksDbStore::new(
            path,
            Duration::from_secs(60),
        )?));

        let mut factory =
            RocksDbJobFactory::new(store.clone(), 1024u64.try_into()?, Duration::from_secs(30))?;
        let f = gen_properly_filled_frame(true);
        let source_id = f.get_source_id();
        store
            .lock()
            .await
            .add_message(&f.to_message(), &[], &[])
            .await?;
        let f = gen_properly_filled_frame(true);
        store
            .lock()
            .await
            .add_message(&f.to_message(), &[], &[])
            .await?;

        let configuration = JobConfigurationBuilder::default()
            .min_duration(Duration::from_millis(7))
            .max_duration(Duration::from_millis(10))
            .stored_stream_id(source_id)
            .resulting_stream_id("resulting_source_id".to_string())
            .build()
            .unwrap();
        let stop_condition = JobStopCondition::frame_count(1);
        let offset = JobOffset::Blocks(1);
        let job_query = JobQuery::new(
            SinkConfiguration::test_dealer_connect_sink(),
            configuration,
            stop_condition,
            f.get_uuid(),
            None,
            offset,
            vec![],
        );
        let _job = factory.create_job(job_query).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_create_rocksdb_delayed_job() -> Result<()> {
        let dir = tempfile::TempDir::new()?;
        let path = dir.path();
        let store = Arc::new(Mutex::new(RocksDbStore::new(
            path,
            Duration::from_secs(60),
        )?));

        let mut factory =
            RocksDbJobFactory::new(store.clone(), 1024u64.try_into()?, Duration::from_secs(30))?;
        let f = gen_properly_filled_frame(true);
        let source_id = f.get_source_id();
        store
            .lock()
            .await
            .add_message(&f.to_message(), &[], &[])
            .await?;
        let f = gen_properly_filled_frame(true);

        let f_clone = f.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(500)).await;
            store
                .lock()
                .await
                .add_message(&f_clone.to_message(), &[], &[])
                .await
                .unwrap();
        });

        let configuration = JobConfigurationBuilder::default()
            .min_duration(Duration::from_millis(7))
            .max_duration(Duration::from_millis(10))
            .stored_stream_id(source_id)
            .resulting_stream_id("resulting_source_id".to_string())
            .build()
            .unwrap();
        let stop_condition = JobStopCondition::frame_count(1);
        let offset = JobOffset::Blocks(1);
        let job_query = JobQuery::new(
            SinkConfiguration::test_dealer_connect_sink(),
            configuration,
            stop_condition,
            f.get_uuid(),
            Some(Duration::from_secs(1)),
            offset,
            vec![],
        );
        let _job = factory.create_job(job_query).await?;
        Ok(())
    }
}

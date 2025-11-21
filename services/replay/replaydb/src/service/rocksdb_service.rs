use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::{bail, Result};
use hashbrown::HashMap;
use log::{info, warn};
use savant_core::transport::zeromq::{NonBlockingReader, NonBlockingWriter};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::job::configuration::JobConfiguration;
use crate::job::factory::RocksDbJobFactory;
use crate::job::query::JobQuery;
use crate::job::stop_condition::JobStopCondition;
use crate::job::SyncJobStopCondition;
use crate::service::configuration::ServiceConfiguration;
use crate::service::JobManager;
use crate::store::rocksdb::RocksDbStore;
use crate::store::{FrameData, Store, SyncRocksDbStore};
use crate::stream_processor::RocksDbStreamProcessor;

#[allow(clippy::type_complexity)]
pub struct RocksDbService {
    store: SyncRocksDbStore,
    stream_processor_job_handle: Option<JoinHandle<Result<()>>>,
    job_factory: RocksDbJobFactory,
    job_map: HashMap<
        Uuid,
        (
            JoinHandle<Result<()>>,
            JobConfiguration,
            Arc<SyncJobStopCondition>,
        ),
    >,
    job_eviction_ttl: Duration,
    stopped_jobs: HashMap<Uuid, (Option<String>, JobConfiguration, SystemTime)>,
    config: ServiceConfiguration,
}

impl TryFrom<&ServiceConfiguration> for SyncRocksDbStore {
    type Error = anyhow::Error;

    fn try_from(configuration: &ServiceConfiguration) -> Result<Self> {
        let store = RocksDbStore::new(&configuration.storage)?;
        let sync_store = Arc::new(Mutex::new(store));
        Ok(sync_store)
    }
}

impl TryFrom<&ServiceConfiguration> for RocksDbStreamProcessor {
    type Error = anyhow::Error;

    fn try_from(configuration: &ServiceConfiguration) -> Result<Self> {
        let store = SyncRocksDbStore::try_from(configuration)?;
        let input = NonBlockingReader::try_from(&configuration.in_stream)?;
        let output = if let Some(out_stream) = &configuration.out_stream {
            Some(NonBlockingWriter::try_from(out_stream)?)
        } else {
            None
        };
        Ok(RocksDbStreamProcessor::new(
            store,
            input,
            output,
            configuration.common.stats_period,
            configuration.common.pass_metadata_only,
        ))
    }
}

impl RocksDbService {
    pub fn new(config: &ServiceConfiguration) -> Result<Self> {
        let mut stream_processor = RocksDbStreamProcessor::try_from(config)?;
        let store = stream_processor.store();
        let local_store_ref = store.clone();
        let job_factory = RocksDbJobFactory::new(
            store,
            config.common.job_writer_cache_max_capacity.try_into()?,
            config.common.job_writer_cache_ttl,
        )?;

        let stream_processor_job_handle =
            Some(tokio::spawn(async move { stream_processor.run().await }));

        Ok(Self {
            store: local_store_ref,
            stream_processor_job_handle,
            job_factory,
            job_map: HashMap::new(),
            job_eviction_ttl: config.common.job_eviction_ttl,
            stopped_jobs: HashMap::new(),
            config: config.clone(),
        })
    }

    pub async fn find_keyframes(
        &mut self,
        source_id: &str,
        from: Option<u64>,
        to: Option<u64>,
        limit: usize,
    ) -> Result<Vec<Uuid>> {
        let mut store = self.store.lock().await;
        store.find_keyframes(source_id, from, to, limit).await
    }

    pub async fn get_keyframe_by_uuid(
        &mut self,
        source_id: &str,
        uuid: Uuid,
    ) -> Result<Option<FrameData>> {
        let mut store = self.store.lock().await;
        store.get_keyframe_by_uuid(source_id, uuid).await
    }
}

impl JobManager for RocksDbService {
    async fn add_job(&mut self, mut job_query: JobQuery) -> Result<Uuid> {
        // Apply default sink options from the configuration if needed
        let default_options = self.config.common.default_job_sink_options.clone();

        // Apply default sink options to the job query if they exist
        job_query = job_query.with_default_sink_options(default_options);

        let configuration = job_query.configuration.clone();
        let mut job = self.job_factory.create_job(job_query).await?;
        let job_id = job.get_id();
        let stop_condition = job.get_stop_condition_ref();
        let job_handle = tokio::spawn(async move { job.run_until_complete().await });
        self.job_map
            .insert(job_id, (job_handle, configuration, stop_condition));
        Ok(job_id)
    }

    async fn stop_job(&mut self, job_id: Uuid) -> Result<()> {
        if let Some((job_handle, conf, _)) = self.job_map.remove(&job_id) {
            job_handle.abort();
            self.stopped_jobs.insert(
                job_id,
                (
                    job_handle.await?.err().map(|e| e.to_string()),
                    conf,
                    SystemTime::now(),
                ),
            );
        } else {
            bail!("Job {} not found", job_id);
        }
        Ok(())
    }

    fn update_stop_condition(
        &mut self,
        job_id: Uuid,
        stop_condition: JobStopCondition,
    ) -> Result<()> {
        if let Some((_, _, stop_condition_ref)) = self.job_map.get_mut(&job_id) {
            let mut sc = stop_condition_ref.0.lock();
            *sc = stop_condition;
        }
        Ok(())
    }

    fn list_jobs(&self) -> Vec<(Uuid, JobConfiguration, JobStopCondition)> {
        self.job_map
            .iter()
            .map(|(uuid, (_, c, s))| (*uuid, c.clone(), s.0.lock().clone()))
            .collect()
    }

    fn list_stopped_jobs(&self) -> Vec<(Uuid, JobConfiguration, Option<String>)> {
        self.stopped_jobs
            .iter()
            .map(|(uuid, (res, conf, _))| (*uuid, conf.clone(), res.clone()))
            .collect()
    }

    async fn check_stream_processor_finished(&mut self) -> Result<bool> {
        if self.stream_processor_job_handle.is_none() {
            bail!("Stream processor job handle is none. No longer functional");
        }

        if self
            .stream_processor_job_handle
            .as_ref()
            .unwrap()
            .is_finished()
        {
            let sp = self.stream_processor_job_handle.take().unwrap();
            warn!("Stream processor job is finished. Stopping all jobs");
            sp.await??;
            for (uuid, (job_handle, _, _)) in self.job_map.drain() {
                info!("Stopping job: {}", uuid);
                job_handle.abort();
                info!("Job: {} stopped", uuid);
            }
            return Ok(true);
        }
        Ok(false)
    }

    async fn shutdown(&mut self) -> Result<()> {
        if let Some(sp) = self.stream_processor_job_handle.take() {
            info!("Stopping stream processor");
            sp.abort();
            let _ = sp.await;
            info!("Stream processor stopped");
        }
        for (uuid, (job_handle, _, _)) in self.job_map.drain() {
            info!("Stopping job: {}", uuid);
            job_handle.abort();
            let _ = job_handle.await;
            info!("Job: {} stopped", uuid);
        }
        Ok(())
    }

    async fn clean_stopped_jobs(&mut self) -> Result<()> {
        let mut to_remove = vec![];
        for (uuid, (job_handle, _, _)) in self.job_map.iter() {
            if job_handle.is_finished() {
                info!("Job: {} is finished and is marked for clean up.", uuid);
                to_remove.push(*uuid);
            }
        }
        for uuid in to_remove {
            let (job_handle, conf, _) = self.job_map.remove(&uuid).unwrap();
            let res = job_handle.await?;
            if let Err(e) = &res {
                warn!("Job: {} failed with error: {} was cleaned.", uuid, e);
            } else {
                info!("Job: {} finished successfully and cleaned.", uuid);
            }
            self.stopped_jobs.insert(
                uuid,
                (res.err().map(|e| e.to_string()), conf, SystemTime::now()),
            );
        }
        let to_remove = self
            .stopped_jobs
            .iter()
            .filter_map(|(uuid, (_, _, time))| {
                if time.elapsed().unwrap() > self.job_eviction_ttl {
                    info!(
                        "Job: {} was stopped for more than {:?}. Clearing the information about it.",
                        uuid, time
                    );
                    Some(*uuid)
                } else {
                    None
                }
            })
            .collect::<Vec<Uuid>>();

        for uuid in to_remove {
            self.stopped_jobs.remove(&uuid);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::env::set_var;

    use crate::service::configuration::ServiceConfiguration;
    use crate::service::rocksdb_service::RocksDbService;
    use crate::service::JobManager;

    #[tokio::test]
    async fn test_rocksdb_service() -> anyhow::Result<()> {
        let tmp_dir = tempfile::tempdir().unwrap();
        set_var("DB_PATH", tmp_dir.path().to_str().unwrap());
        set_var("SOCKET_PATH_IN", "in");
        set_var("SOCKET_PATH_OUT", "out");
        let config = ServiceConfiguration::new("assets/rocksdb.json")?;
        let mut service = RocksDbService::new(&config)?;
        service.shutdown().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_rockdb_service_opt_writer() -> anyhow::Result<()> {
        let tmp_dir = tempfile::tempdir().unwrap();
        set_var("DB_PATH", tmp_dir.path().to_str().unwrap());
        set_var("SOCKET_PATH_IN", "in");
        let config = ServiceConfiguration::new("assets/rocksdb_opt_out.json")?;
        let mut service = RocksDbService::new(&config)?;
        service.shutdown().await?;
        Ok(())
    }
}

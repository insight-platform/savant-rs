use std::future::Future;

use uuid::Uuid;

use crate::job::configuration::JobConfiguration;
use crate::job::query::JobQuery;
use crate::job::stop_condition::JobStopCondition;

pub mod configuration;
pub mod rocksdb_service;

pub trait JobManager {
    fn add_job(&mut self, job: JobQuery) -> impl Future<Output = anyhow::Result<Uuid>> + Send;
    fn stop_job(&mut self, job_id: Uuid) -> impl Future<Output = anyhow::Result<()>> + Send;
    fn update_stop_condition(
        &mut self,
        job_id: Uuid,
        stop_condition: JobStopCondition,
    ) -> anyhow::Result<()>;
    fn list_jobs(&self) -> Vec<(Uuid, JobConfiguration, JobStopCondition)>;
    fn list_stopped_jobs(&self) -> Vec<(Uuid, JobConfiguration, Option<String>)>;
    fn check_stream_processor_finished(
        &mut self,
    ) -> impl Future<Output = anyhow::Result<bool>> + Send;
    fn shutdown(&mut self) -> impl Future<Output = anyhow::Result<()>> + Send;
    fn clean_stopped_jobs(&mut self) -> impl Future<Output = anyhow::Result<()>> + Send;
}

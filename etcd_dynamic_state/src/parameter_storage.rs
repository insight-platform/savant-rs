use crate::etcd_api::{EtcdClient, KVOperator, Operation, WatchResult};
use async_trait::async_trait;
use glob::Pattern;
use hashbrown::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;

type ParameterDatabase = Arc<RwLock<HashMap<String, Vec<u8>>>>;

struct IdleKVOperator;

pub struct EtcdParameterStorage {
    client: Option<EtcdClient>,
    parameters: ParameterDatabase,
    prefix: String,
    general_update_key: String,
    per_stream_update_pattern: Pattern,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl EtcdParameterStorage {
    fn init(&mut self) -> Self {
        Self {
            client: self.client.take(),
            parameters: self.parameters.clone(),
            prefix: self.prefix.clone(),
            general_update_key: self.general_update_key.clone(),
            per_stream_update_pattern: self.per_stream_update_pattern.clone(),
            handle: None,
        }
    }

    pub fn new(
        client: EtcdClient,
        prefix: String,
        general_update_key: String,
        per_stream_update_pattern: Pattern,
    ) -> Self {
        Self {
            client: Some(client),
            parameters: Arc::new(RwLock::new(HashMap::new())),
            prefix,
            general_update_key,
            per_stream_update_pattern,
            handle: None,
        }
    }

    pub fn run(&mut self) -> anyhow::Result<()> {
        let mut etcd_worker = self.init();

        let handle = tokio::spawn(async move {
            let mut client = etcd_worker.client.take().unwrap();
            client
                .monitor(
                    Arc::new(tokio::sync::Mutex::new(Watcher {
                        parameters: etcd_worker.parameters.clone(),
                    })),
                    Arc::new(tokio::sync::Mutex::new(IdleKVOperator)),
                )
                .await
                .expect("Failed to monitor etcd");
        });
        self.handle = Some(handle);
        Ok(())
    }

    pub fn stop(&mut self) -> anyhow::Result<()> {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        let parameters = self.parameters.read();
        parameters.get(key).cloned()
    }
}

struct Watcher {
    parameters: ParameterDatabase,
}

#[async_trait]
impl WatchResult for Watcher {
    async fn notify(&mut self, res: Operation) -> anyhow::Result<()> {
        match res {
            Operation::Set {
                key,
                value,
                with_lease: _,
            } => {
                self.parameters.write().insert(key, value);
            }
            Operation::DelKey { key } => {
                dbg!(key);
            }
            Operation::DelPrefix { prefix } => {
                dbg!(prefix);
            }
            Operation::Nope => {}
        }

        Ok(())
    }
}

#[async_trait]
impl KVOperator for IdleKVOperator {
    async fn ops(&mut self) -> anyhow::Result<Vec<Operation>> {
        Ok(vec![])
    }
}

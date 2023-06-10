use crate::etcd_api::{EtcdClient, KVOperator, Operation, VarPathSpec, WatchResult};
use anyhow::bail;
use async_trait::async_trait;
use hashbrown::HashMap;
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;
use std::thread::sleep;
use tokio::runtime::Runtime;

type ParameterDatabase = Arc<RwLock<HashMap<String, (u32, Arc<Vec<u8>>)>>>;

const BLOCKING_WAIT_SLEEP_DELAY_MS: u64 = 10;

struct EtcdKVOperator {
    ops: Arc<Mutex<Vec<Operation>>>,
    parameters: ParameterDatabase,
}

pub struct EtcdParameterStorage {
    client: Option<EtcdClient>,
    parameters: ParameterDatabase,
    handle: Option<tokio::task::JoinHandle<anyhow::Result<()>>>,
    ops: Arc<Mutex<Vec<Operation>>>,
}

impl EtcdParameterStorage {
    fn init(&mut self) -> Self {
        Self {
            client: self.client.take(),
            parameters: self.parameters.clone(),
            handle: None,
            ops: self.ops.clone(),
        }
    }

    pub fn with_client(client: EtcdClient) -> Self {
        Self {
            client: Some(client),
            parameters: Arc::new(RwLock::new(HashMap::new())),
            handle: None,
            ops: Arc::new(Mutex::new(vec![])),
        }
    }

    pub fn run(&mut self, rt: &Runtime) -> anyhow::Result<()> {
        let mut etcd_worker = self.init();

        let handle = rt.spawn(async move {
            let mut client = etcd_worker.client.take().unwrap();
            let res = client
                .monitor(
                    Arc::new(tokio::sync::Mutex::new(Watcher {
                        parameters: etcd_worker.parameters.clone(),
                    })),
                    Arc::new(tokio::sync::Mutex::new(EtcdKVOperator {
                        ops: etcd_worker.ops.clone(),
                        parameters: etcd_worker.parameters.clone(),
                    })),
                )
                .await;

            match res {
                Ok(_) => {
                    log::info!("EtcdParameterStorage is successfully finished.");
                }
                Err(e) => {
                    let err_msg = format!("EtcdParameterStorage failed: {}", e);
                    log::error!("{}", &err_msg);
                    bail!(err_msg);
                }
            }
            Ok(())
        });
        self.handle = Some(handle);
        Ok(())
    }

    pub fn is_active(&self) -> bool {
        if let Some(h) = &self.handle {
            !h.is_finished()
        } else {
            false
        }
    }

    pub fn wait_for_key(&self, key: &str, mut timeout_ms: u64) -> anyhow::Result<bool> {
        if timeout_ms <= BLOCKING_WAIT_SLEEP_DELAY_MS {
            timeout_ms = BLOCKING_WAIT_SLEEP_DELAY_MS + 1;
        }

        while timeout_ms - BLOCKING_WAIT_SLEEP_DELAY_MS > 0 {
            if !self.is_active() {
                bail!("EtcdParameterStorage is not active");
            }
            if !self.is_key_present(key)? {
                sleep(std::time::Duration::from_millis(
                    BLOCKING_WAIT_SLEEP_DELAY_MS,
                ));
                timeout_ms -= BLOCKING_WAIT_SLEEP_DELAY_MS;
            } else {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn stop(&mut self, rt: Runtime) -> anyhow::Result<()> {
        if !self.is_active() {
            bail!("EtcdParameterStorage is not active");
        }

        if let Some(handle) = self.handle.take() {
            handle.abort();
            rt.block_on(async move {
                handle.await.unwrap_or_else(|e| {
                    if e.is_cancelled() {
                        log::info!("EtcdParameterStorage is successfully stopped.");
                        Ok(())
                    } else {
                        let error_msg = format!("EtcdParameterStorage failed to stop: {}", e);
                        log::error!("{}", &error_msg);
                        bail!(error_msg);
                    }
                })
            })?;
        }

        rt.shutdown_timeout(std::time::Duration::from_secs(5));

        Ok(())
    }

    pub fn get_data_checksum(&self, key: &str) -> anyhow::Result<Option<u32>> {
        if !self.is_active() {
            bail!("EtcdParameterStorage is not active");
        }

        let parameters = self.parameters.upgradable_read();
        let res = parameters.get(key);
        Ok(res.map(|(crc, _)| *crc))
    }

    pub fn order_data_update(&self, spec: VarPathSpec) -> anyhow::Result<()> {
        if !self.is_active() {
            bail!("EtcdParameterStorage is not active");
        }

        let op = Operation::Get { spec };
        self.ops.lock().push(op);
        Ok(())
    }

    pub fn get_data(&self, key: &str) -> anyhow::Result<Option<(u32, Arc<Vec<u8>>)>> {
        if !self.is_active() {
            bail!("EtcdParameterStorage is not active");
        }

        Ok(self.parameters.read().get(key).cloned())
    }

    pub fn set(&self, key: &str, value: Vec<u8>) -> anyhow::Result<()> {
        if !self.is_active() {
            bail!("EtcdParameterStorage is not active");
        }

        let op = Operation::Set {
            key: key.to_string(),
            value,
            with_lease: false,
        };

        self.ops.lock().push(op);
        Ok(())
    }

    pub fn is_key_present(&self, key: &str) -> anyhow::Result<bool> {
        if !self.is_active() {
            bail!("EtcdParameterStorage is not active");
        }

        let parameters = self.parameters.upgradable_read();
        Ok(parameters.get(key).is_some())
    }
}

struct Watcher {
    parameters: ParameterDatabase,
}

#[async_trait]
impl WatchResult for Watcher {
    async fn notify(&mut self, _client: &mut EtcdClient, res: Operation) -> anyhow::Result<()> {
        match res {
            Operation::Set {
                key,
                value,
                with_lease: _,
            } => {
                let crc = crc32fast::hash(&value);
                self.parameters.write().insert(key, (crc, Arc::new(value)));
            }
            Operation::DelKey { key } => {
                self.parameters.write().remove(&key);
            }
            Operation::DelPrefix { prefix } => {
                let mut parameters = self.parameters.write();
                let keys: Vec<String> = parameters
                    .keys()
                    .filter(|key| key.starts_with(&prefix))
                    .cloned()
                    .collect();

                for key in keys {
                    parameters.remove(&key);
                }
            }
            Operation::Get { spec: _ } => {
                bail!("Get should not be sent to watcher");
            }

            Operation::Nope => {
                bail!("Nope should not be sent to watcher");
            }
        }

        Ok(())
    }
}

#[async_trait]
impl KVOperator for EtcdKVOperator {
    async fn ops(&mut self, client: &mut EtcdClient) -> anyhow::Result<Vec<Operation>> {
        let ops: Vec<Operation> = self.ops.lock().drain(..).collect();

        let (get_ops, other_ops) = ops
            .into_iter()
            .partition(|op| matches!(op, Operation::Get { .. }));

        for o in get_ops {
            match o {
                Operation::Get { spec } => {
                    let res = client.fetch_vars(&vec![spec]).await?;
                    let mut parameters = self.parameters.write();
                    for (key, value) in res {
                        let crc = crc32fast::hash(&value);
                        parameters.insert(key, (crc, Arc::new(value)));
                    }
                }
                _ => bail!("Get should be the only operation in get_ops."),
            }
        }
        Ok(other_ops)
    }
}

#[cfg(test)]
mod tests {
    use crate::etcd_api::{EtcdClient, Operation, VarPathSpec};
    use std::thread::sleep;
    use tokio::runtime::Runtime;

    async fn init_client(hosts: Vec<String>) -> anyhow::Result<EtcdClient> {
        let mut client = EtcdClient::new(hosts, None, "parameters/node".into(), 5, 10).await?;

        client
            .kv_operations(vec![
                Operation::Set {
                    key: "parameters/node".into(),
                    value: "value".into(),
                    with_lease: false,
                },
                Operation::Set {
                    key: "parameters/node/stream1".into(),
                    value: "stream1".into(),
                    with_lease: false,
                },
            ])
            .await?;

        Ok(client)
    }

    #[test]
    fn test_monitor() -> anyhow::Result<()> {
        _ = env_logger::try_init();

        let runtime = Runtime::new().unwrap();

        let client = runtime
            .block_on(init_client(vec!["127.0.0.1:2379".into()]))
            .expect("Failed to init client");

        let mut parameter_storage = super::EtcdParameterStorage::with_client(client);
        parameter_storage
            .run(&runtime)
            .expect("Failed to run parameter storage");

        assert!(!parameter_storage.is_key_present("parameters/node").unwrap());

        parameter_storage
            .order_data_update(VarPathSpec::SingleVar("parameters/node".into()))
            .unwrap();
        assert!(parameter_storage
            .wait_for_key("parameters/node", 2000)
            .unwrap());

        let (crc, res) = parameter_storage
            .get_data("parameters/node")
            .unwrap()
            .expect("Failed to get value");

        assert_eq!(res.as_slice(), "value".as_bytes());
        assert!(parameter_storage.is_key_present("parameters/node").unwrap());
        assert_eq!(
            parameter_storage
                .get_data_checksum("parameters/node")
                .unwrap(),
            Some(crc)
        );

        parameter_storage
            .set("parameters/node", "value2".as_bytes().to_vec())
            .unwrap();

        sleep(std::time::Duration::from_secs(1));

        let (new_crc, res) = parameter_storage
            .get_data("parameters/node".into())
            .unwrap()
            .expect("Failed to get value");

        assert_eq!(res.as_slice(), "value2".as_bytes());
        assert_ne!(new_crc, crc);

        assert!(parameter_storage.is_active());

        parameter_storage
            .stop(runtime)
            .expect("Failed to stop parameter storage");

        assert!(!parameter_storage.is_active());

        Ok(())
    }

    #[test]
    fn test_wrong_ip() {
        _ = env_logger::try_init();

        let runtime = Runtime::new().unwrap();

        let client = runtime.block_on(init_client(vec!["127.0.0.1:12379".into()]));
        assert!(client.is_err());
    }
}

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use anyhow::Result;
use async_trait::async_trait;
use etcd_client::*;
use thiserror::Error;

use log::{info, warn};

const WATCH_WAIT_TTL: u64 = 100;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Key `{0}` doesn't exist!")]
    KeyDoesNotExist(String),
}

#[async_trait]
pub trait WatchResult {
    async fn notify(&mut self, client: &mut EtcdClient, res: Operation) -> Result<()>;
}

#[async_trait]
pub trait KVOperator {
    async fn ops(&mut self, client: &mut EtcdClient) -> Result<Vec<Operation>>;
}

pub struct EtcdClient {
    client: Client,
    watcher: (Watcher, WatchStream),
    lease_timeout: i64,
    lease_id: Option<i64>,
}

#[derive(Debug, Default)]
pub enum Operation {
    Set {
        key: String,
        value: Vec<u8>,
        with_lease: bool,
    },
    Get {
        spec: VarPathSpec,
    },
    DelKey {
        key: String,
    },
    DelPrefix {
        prefix: String,
    },
    #[default]
    Nope,
}

#[derive(Debug, Clone)]
pub enum VarPathSpec {
    SingleVar(String),
    Prefix(String),
}

impl VarPathSpec {
    pub fn new_var(p: &str, var: &str) -> VarPathSpec {
        VarPathSpec::SingleVar(
            Path::new(p)
                .join(Path::new(var))
                .to_str()
                .unwrap()
                .to_string(),
        )
    }

    pub fn new_prefix(prefix: &str, dir: &str) -> VarPathSpec {
        VarPathSpec::Prefix(
            Path::new(prefix)
                .join(Path::new(dir))
                .to_str()
                .unwrap()
                .into(),
        )
    }

    pub async fn get(&self, client: &mut Client) -> Result<(String, Vec<u8>)> {
        match self {
            VarPathSpec::SingleVar(key) => {
                let resp = client.get(key.as_bytes(), None).await?;
                match resp.kvs().first() {
                    Some(res) => {
                        let value = Vec::from(res.value());
                        info!("Etcd Get: Key={}, Value={:?}", res.key_str()?, value);
                        Ok((res.key_str()?.to_string(), value))
                    }
                    None => {
                        warn!("No value found for key: {:?}", key);
                        Err(ConfigError::KeyDoesNotExist(key.clone()).into())
                    }
                }
            }
            _ => panic!("get method is only defined for SingleVar"),
        }
    }

    pub async fn get_prefix(&self, client: &mut Client) -> Result<Vec<(String, Vec<u8>)>> {
        match self {
            VarPathSpec::Prefix(key) => {
                let resp = client
                    .get(key.as_bytes(), Some(GetOptions::new().with_prefix()))
                    .await?;
                let mut result = Vec::default();
                for kv in resp.kvs() {
                    let value = Vec::from(kv.value());
                    info!("Etcd Get Prefix: Key={}, Value={:?}", kv.key_str()?, value);
                    result.push((kv.key_str()?.to_string(), value));
                }
                Ok(result)
            }
            _ => panic!("get_prefix method is only defined for Prefix"),
        }
    }
}

impl EtcdClient {
    pub fn get_lease_id(&self) -> Option<i64> {
        self.lease_id
    }

    pub async fn new(
        uris: &[&str],
        credentials: &Option<(&str, &str)>,
        path: &str,
        lease_timeout: i64,
        connect_timeout: u64,
    ) -> Result<EtcdClient> {
        info!("Connecting to {:?} etcd server", &uris);
        let mut client = Client::connect(
            uris,
            Some({
                let mut opts = ConnectOptions::new();
                if let Some((user, password)) = credentials {
                    opts = opts.with_user(*user, *password);
                }
                opts.with_timeout(Duration::from_secs(connect_timeout))
            }),
        )
        .await?;

        info!("Watching for {} for configuration changes", &path);
        let (watcher, watch_stream) = client
            .watch(path, Some(WatchOptions::new().with_prefix()))
            .await?;

        let lease = client.lease_grant(lease_timeout, None).await?;

        Ok(EtcdClient {
            client,
            watcher: (watcher, watch_stream),
            lease_timeout,
            lease_id: Some(lease.id()),
        })
    }

    pub async fn fetch_vars(
        &mut self,
        var_spec: &Vec<VarPathSpec>,
    ) -> Result<Vec<(String, Vec<u8>)>> {
        let mut res = Vec::default();
        for v in var_spec {
            match v {
                VarPathSpec::SingleVar(_) => {
                    let value_pair = v.get(&mut self.client).await?;
                    res.push(value_pair);
                }
                VarPathSpec::Prefix(_) => {
                    let mut value_pairs = v.get_prefix(&mut self.client).await?;
                    res.append(&mut value_pairs);
                }
            }
        }
        Ok(res)
    }

    pub async fn kv_operations(&mut self, ops: Vec<Operation>) -> Result<()> {
        for op in ops {
            match op {
                Operation::Set {
                    key,
                    value,
                    with_lease,
                } => {
                    self.client
                        .put(
                            key,
                            value,
                            Some({
                                let mut opts = PutOptions::new();
                                if with_lease {
                                    opts = opts.with_lease(self.lease_id.unwrap());
                                }
                                opts
                            }),
                        )
                        .await?;
                }
                Operation::DelKey { key } => {
                    self.client.delete(key, None).await?;
                }
                Operation::DelPrefix { prefix } => {
                    self.client
                        .delete(prefix, Some(DeleteOptions::new().with_prefix()))
                        .await?;
                }
                Operation::Get { spec: _ } => {}
                Operation::Nope => (),
            }
        }
        Ok(())
    }

    pub async fn monitor(
        &mut self,
        watch_result: Arc<Mutex<dyn WatchResult + Send + Sync>>,
        kv_operator: Arc<Mutex<dyn KVOperator + Send + Sync>>,
    ) -> Result<()> {
        info!("Starting watching for changes on {:?}", self.watcher);

        if self.lease_id.is_none() {
            let lease = self.client.lease_grant(self.lease_timeout, None).await?;
            self.lease_id = Some(lease.id());
        }

        loop {
            self.client.lease_keep_alive(self.lease_id.unwrap()).await?;

            let res = tokio::time::timeout(
                Duration::from_millis(WATCH_WAIT_TTL),
                self.watcher.1.message(),
            )
            .await;

            if let Ok(res) = res {
                if let Some(resp) = res? {
                    if resp.canceled() {
                        return Ok(());
                    } else if resp.created() {
                        info!("Etcd watcher was successfully deployed.");
                    }

                    for event in resp.events() {
                        if EventType::Delete == event.event_type() {
                            if let Some(kv) = event.kv() {
                                watch_result
                                    .lock()
                                    .await
                                    .notify(
                                        self,
                                        Operation::DelKey {
                                            key: kv.key_str()?.into(),
                                        },
                                    )
                                    .await?;
                            }
                        }

                        if EventType::Put == event.event_type() {
                            if let Some(kv) = event.kv() {
                                watch_result
                                    .lock()
                                    .await
                                    .notify(
                                        self,
                                        Operation::Set {
                                            key: kv.key_str()?.to_string(),
                                            value: kv.value().to_vec(),
                                            with_lease: kv.lease() != 0,
                                        },
                                    )
                                    .await?;
                            }
                        }
                    }
                } else {
                    return Ok(());
                }
            }

            let ops = kv_operator.lock().await.ops(self).await?;
            self.kv_operations(ops).await?;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::etcd_api::{EtcdClient, KVOperator, Operation, VarPathSpec, WatchResult};
    use anyhow::Result;
    use async_trait::async_trait;
    use log::info;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Mutex;

    #[tokio::test]
    #[ignore]
    async fn test_monitor() -> Result<()> {
        _ = env_logger::try_init();
        let mut client = EtcdClient::new(&["127.0.0.1:2379"], &None, "local/node", 5, 10).await?;

        client
            .kv_operations(vec![
                Operation::Set {
                    key: "local/node".into(),
                    value: "value".into(),
                    with_lease: false,
                },
                Operation::Set {
                    key: "local/node/leased".into(),
                    value: "leased_value".into(),
                    with_lease: true,
                },
            ])
            .await?;

        let res = client
            .fetch_vars(&vec![VarPathSpec::SingleVar("local/node/leased".into())])
            .await?;

        assert_eq!(
            res,
            vec![("local/node/leased".into(), "leased_value".into())]
        );

        let res = client
            .fetch_vars(&vec![
                VarPathSpec::Prefix("local/node".into()),
                VarPathSpec::SingleVar("local/node/leased".into()),
            ])
            .await?;

        assert_eq!(
            res,
            vec![
                ("local/node".into(), "value".into()),
                ("local/node/leased".into(), "leased_value".into()),
                ("local/node/leased".into(), "leased_value".into())
            ]
        );

        #[derive(Default)]
        struct Watcher {
            counter: i32,
            watch_result: Operation,
        }

        #[async_trait]
        impl WatchResult for Watcher {
            async fn notify(&mut self, _client: &mut EtcdClient, res: Operation) -> Result<()> {
                info!("Operation: {:?}", &res);
                self.counter += 1;
                self.watch_result = res;
                Ok(())
            }
        }

        struct Operator {
            operation: Option<Operation>,
        }

        #[async_trait]
        impl KVOperator for Operator {
            async fn ops(&mut self, _client: &mut EtcdClient) -> Result<Vec<Operation>> {
                let op: Vec<_> = self.operation.take().into_iter().collect();
                Ok(op)
            }
        }

        // vec![VarPathSpec::SingleVar("local/node/leased".into())]
        let w = Arc::new(Mutex::new(Watcher::default()));
        let o = Arc::new(Mutex::new(Operator {
            operation: Some(Operation::Set {
                key: "local/node/leased".into(),
                value: "new_leased".into(),
                with_lease: true,
            }),
        }));

        let t = tokio::spawn(async move {
            match tokio::time::timeout(Duration::from_secs(5), client.monitor(w.clone(), o.clone()))
                .await
            {
                Ok(res) => {
                    panic!("Unexpected termination occurred: {:?}", res);
                }
                Err(_) => {
                    assert!(matches!(
                        w.lock().await.watch_result,
                        Operation::Set {
                            key: _,
                            value: _,
                            with_lease: true,
                        }
                    ));
                    assert_eq!(w.lock().await.counter, 3);
                }
            }
        });

        tokio::join!(t).0?;

        Ok(())
    }
}

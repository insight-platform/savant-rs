# savant-etcd

`savant-etcd` provides async etcd helpers for service discovery, dynamic configuration, and watched key-value state in Savant control-plane components. It wraps `etcd-client` with higher-level operations for prefix fetches, leased keys, watch handling, and in-memory state that fits Kubernetes-style deployments or other distributed systems that use etcd as a Consul alternative.

## What's inside

- **Low-level etcd API:** `etcd_api` defines `EtcdClient`, `Operation`, `VarPathSpec`, `WatchResult`, and `KVOperator`. `EtcdClient::new()` and `EtcdClient::new_with_tls()` establish a watched connection, `fetch_vars()` loads single keys or prefixes, `kv_operations()` applies batched set/delete actions, and `monitor()` drives the long-running watch loop.
- **Path and operation modeling:** `VarPathSpec::SingleVar` and `VarPathSpec::Prefix` describe whether you want a single key or a whole subtree. `Operation` models the update stream with `Set`, `Get`, `DelKey`, and `DelPrefix`, which makes it straightforward to build configuration sync or dynamic state refresh logic.
- **Watched local cache:** `parameter_storage::EtcdParameterStorage` maintains an in-memory database of watched values keyed by etcd path. It exposes `run()`, `is_active()`, `wait_for_key()`, `order_data_update()`, `get_data()`, `get_data_checksum()`, `set()`, and `is_key_present()` for services that want a lightweight cached view of etcd-backed configuration.
- **Error handling and lifecycle:** `ConfigError::KeyDoesNotExist` captures missing-key reads, and the monitor loop keeps leases alive while dispatching watch notifications into `WatchResult` and `KVOperator` implementations.

## Usage

```rust
use savant_etcd::etcd_api::{EtcdClient, Operation, VarPathSpec};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut client = EtcdClient::new(
        &["127.0.0.1:2379"],
        &None,
        "services/router",
        5,
        5,
    )
    .await?;

    client
        .kv_operations(vec![Operation::Set {
            key: "services/router/node-1".into(),
            value: br#"{"status":"ready"}"#.to_vec(),
            with_lease: true,
        }])
        .await?;

    let values = client
        .fetch_vars(&vec![VarPathSpec::Prefix("services/router".into())])
        .await?;

    assert!(!values.is_empty());
    Ok(())
}
```

## Install

```toml
[dependencies]
savant-etcd = "2"
```

## System requirements

- A reachable etcd v3 endpoint or cluster is required at runtime.
- TLS connections are supported through `EtcdClient::new_with_tls()`.
- Docker is only needed for this crate's integration tests.

## Documentation

- [docs.rs](https://docs.rs/savant-etcd)
- [Savant project site](https://insight-platform.github.io/savant-rs/)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).


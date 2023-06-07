use crate::etcd_api::EtcdClient;

pub struct EtcdParameterStorage {
    client: EtcdClient,
    prefix: String,
    general_update_key: String,
    per_stream_update_pattern: String,
}

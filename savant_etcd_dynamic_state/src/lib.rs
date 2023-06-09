use etcd_dynamic_state::etcd_api;
use etcd_dynamic_state::etcd_api::EtcdClient;
use etcd_dynamic_state::parameter_storage::EtcdParameterStorage as RustEtcdParameterStorage;
use parking_lot::Mutex;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use savant_rs::utils::byte_buffer::ByteBuffer;
use savant_rs::utils::python::no_gil;
use std::sync::Arc;
use tokio::runtime::Runtime;

#[pyfunction]
fn version() -> String {
    savant_rs::version()
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct VarPathSpec {
    inner: etcd_api::VarPathSpec,
}

impl From<VarPathSpec> for etcd_api::VarPathSpec {
    fn from(spec: VarPathSpec) -> Self {
        spec.inner
    }
}

impl From<etcd_api::VarPathSpec> for VarPathSpec {
    fn from(spec: etcd_api::VarPathSpec) -> Self {
        Self { inner: spec }
    }
}

#[pymethods]
impl VarPathSpec {
    #[staticmethod]
    fn single_var(key: String) -> Self {
        Self {
            inner: etcd_api::VarPathSpec::SingleVar(key),
        }
    }

    #[staticmethod]
    fn prefix(prefix: String) -> Self {
        Self {
            inner: etcd_api::VarPathSpec::Prefix(prefix),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct EtcdParameterStorage {
    inner: Arc<Mutex<RustEtcdParameterStorage>>,
    runtime: Option<Arc<Runtime>>,
}

fn default_hosts() -> Vec<String> {
    vec!["127.0.0.1:2379".to_string()]
}

fn default_path() -> String {
    "savant".to_string()
}

#[pymethods]
impl EtcdParameterStorage {
    #[new]
    #[pyo3(
        signature = (hosts = default_hosts(), credentials=None, path=default_path(), lease_timeout=60, connect_timeout=5)
    )]
    pub fn new(
        hosts: Vec<String>,
        credentials: Option<(String, String)>,
        path: String,
        lease_timeout: i64,
        connect_timeout: u64,
    ) -> PyResult<Self> {
        _ = env_logger::try_init();
        let runtime = Runtime::new().unwrap();

        let client = EtcdClient::new(hosts, credentials, path, lease_timeout, connect_timeout);

        let client = runtime
            .block_on(client)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to connect to etcd: {}", e)))?;

        let mut parameter_storage = RustEtcdParameterStorage::with_client(client);

        parameter_storage.run(&runtime).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to run etcd parameter storage: {}", e))
        })?;

        Ok(Self {
            inner: Arc::new(Mutex::new(parameter_storage)),
            runtime: Some(Arc::new(runtime)),
        })
    }

    fn stop(&mut self) -> PyResult<()> {
        no_gil(|| {
            let rt = self.runtime.take().ok_or_else(|| {
                PyRuntimeError::new_err("EtcdParameterStorage has already been stopped")
            })?;

            let rt = Arc::try_unwrap(rt).map_err(|survived_rt| {
                self.runtime = Some(survived_rt);
                PyRuntimeError::new_err(
                    "Failed to destroy EtcdParameterStorage: there are more than one references to the runtime."
                )
            })?;

            self.inner.lock().stop(rt).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to stop etcd parameter storage: {}", e))
            })?;

            Ok(())
        })
    }

    fn is_active(&self) -> bool {
        self.inner.lock().is_active()
    }

    fn wait_for_key(&self, key: String, timeout: i64) -> PyResult<bool> {
        no_gil(|| {
            self.inner
                .lock()
                .wait_for_key(&key, timeout as u64)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to wait for key: {}", e)))
        })
    }

    fn get_data_checksum(&self, key: String) -> PyResult<Option<u32>> {
        no_gil(|| {
            self.inner
                .lock()
                .get_data_checksum(&key)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get data checksum: {}", e)))
        })
    }

    fn order_data_update(&self, spec: VarPathSpec) -> PyResult<()> {
        no_gil(|| {
            self.inner
                .lock()
                .order_data_update(spec.into())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to order data update: {}", e)))
        })
    }

    fn get_data(&self, key: String) -> PyResult<Option<ByteBuffer>> {
        let data_opt = no_gil(|| {
            self.inner
                .lock()
                .get_data(&key)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get data: {}", e)))
        })?;

        match data_opt {
            Some((crc, data)) => Ok(Some(ByteBuffer::new(data, Some(crc)))),
            None => Ok(None),
        }
    }

    pub fn set_raw(&self, key: String, value: Vec<u8>) -> PyResult<()> {
        no_gil(|| {
            self.inner
                .lock()
                .set(&key, value)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to set data: {}", e)))
        })
    }

    pub fn set_byte_buffer(&self, key: String, value: ByteBuffer) -> PyResult<()> {
        no_gil(|| {
            self.inner
                .lock()
                .set(&key, value.bytes())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to set data: {}", e)))
        })
    }

    pub fn is_key_present(&self, key: String) -> PyResult<bool> {
        no_gil(|| {
            self.inner.lock().is_key_present(&key).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to check key presence: {}", e))
            })
        })
    }
}

#[pymodule]
fn savant_etcd_dynamic_state(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<VarPathSpec>()?;
    m.add_class::<EtcdParameterStorage>()?;
    Ok(())
}

use etcd_dynamic_state::etcd_api;
use etcd_dynamic_state::etcd_api::EtcdClient;
use etcd_dynamic_state::parameter_storage::EtcdParameterStorage as RustEtcdParameterStorage;
use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use savant_rs::utils::byte_buffer::ByteBuffer;
use savant_rs::utils::python::no_gil;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Returns the version of the library.
///
/// Returns
/// -------
/// str
///   The version of the library.
///
#[pyfunction]
fn version() -> String {
    savant_rs::version()
}

/// Allows setting the path specification for etcd key in the form of of a key or prefix.
///
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
    /// Creates a new VarPathSpec for a specific path.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///   The key specifying the path
    ///
    /// Returns
    /// -------
    /// VarPathSpec
    ///   The VarPathSpec object.
    ///
    #[staticmethod]
    fn single_var(key: String) -> Self {
        Self {
            inner: etcd_api::VarPathSpec::SingleVar(key),
        }
    }

    /// Creates a new VarPathSpec for a prefix.
    ///
    /// Parameters
    /// ----------
    /// prefix : str
    ///   The prefix specifying the path
    ///
    /// Returns
    /// -------
    /// VarPathSpec
    ///   The VarPathSpec object.
    ///
    #[staticmethod]
    fn prefix(prefix: String) -> Self {
        Self {
            inner: etcd_api::VarPathSpec::Prefix(prefix),
        }
    }
}

/// State Storage keeping the dynamic state backed by Etcd. The state is received from Etcd with watches. The object is fully asynchronous and GIL-free.
///
/// It is optimized for fetching the state from the local cache, rather then request for keys from Etcd and waiting for them to be received. Basically,
/// to use the state, you create the object specifying the prefix you are interested in, next you request the initiating of the state fetching the prefix (if necesary)
/// after that you just retrieve vars locally: all updates are automatically fetched and placed in local cache, if necessary removed, etc.
///
/// The object allows setting keys and their values but the method is not efficient, because the class is optimized for reading the state, not writing it. The setting may
/// introduce unexpected latency (up to 100 ms), so don't use it when you need to set the state rapidly.
///
/// Object creation:
///
/// Arguments
/// ----------
/// hosts: List[str]
///   The list of Etcd hosts to connect to.
///   Defaults to ``["127.0.0.1:2379"]``.
/// credentials: Optional[(str, str)]
///   The credentials to use for authentication.
///   Defaults to ``None``.
/// path: str
///   The path in Etcd used as the source of the state.
///   Defaults to ``"savant"``.
/// connect_timeout: int
///   The timeout for connecting to Etcd.
///   Defaults to ``5``.
///
/// Returns
/// -------
/// EtcdParameterStorage
///   The EtcdParameterStorage object.
///
/// Raises
/// ------
/// RuntimeError
///   If the connection to Etcd cannot be established.
///
///
/// .. code-block:: python
///
///   from savant_etcd_dynamic_state import EtcdParameterStorage, VarPathSpec
///   storage = EtcdParameterStorage(hosts=["127.0.0.1:2379"], credentials=None, watch_path="savant", connect_timeout=5)
///   storage.set_raw("savant/param1", b"abc")
///   storage.order_data_update(VarPathSpec.prefix("savant"))
///   res = storage.wait_for_key("savant/param1", 2000)
///   assert res == True
///   data = storage.get_data("savant/param1")
///   assert data.bytes.decode('utf-8') == "abc"
///   storage.shutdown()
///
///
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
        signature = (hosts = default_hosts(), credentials=None, watch_path=default_path(), connect_timeout=5)
    )]
    pub fn new(
        hosts: Vec<String>,
        credentials: Option<(String, String)>,
        watch_path: String,
        connect_timeout: u64,
    ) -> PyResult<Self> {
        _ = env_logger::try_init();
        let runtime = Runtime::new().unwrap();

        let client = EtcdClient::new(hosts, credentials, watch_path, 60, connect_timeout);

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

    /// Stops the EtcdParameterStorage. The method can be called only once per lifetime of the object.
    /// After the method is called, the object cannot be used anymore.
    ///
    /// The method is GIL-free.
    ///
    /// Returns
    /// -------
    /// None
    ///   The method does not return anything.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///   If the object has already been stopped.
    ///
    /// RuntimeError
    ///   If there are more than one references to the runtime.
    ///
    fn shutdown(&mut self) -> PyResult<()> {
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

    /// Checks if the storage is active.
    ///
    /// Returns
    /// -------
    /// bool
    ///   ``True`` if the storage is active, ``False`` otherwise.
    ///
    fn is_active(&self) -> bool {
        self.inner.lock().is_active()
    }

    /// Blocks the execution up for the specified amount of time waiting for the specified key to appear in the storage.
    ///
    /// The method is GIL-free.
    ///
    /// Arguments
    /// ----------
    /// key: str
    ///   The key to wait for.
    /// timeout: int
    ///   The timeout in milliseconds.
    ///
    /// Returns
    /// -------
    /// bool
    ///   ``True`` if the key has appeared, ``False`` otherwise.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the timeout is negative.
    ///
    /// RuntimeError
    ///   If the storage is not active.
    ///
    fn wait_for_key(&self, key: String, timeout: i64) -> PyResult<bool> {
        if timeout < 0 {
            return Err(PyValueError::new_err("Timeout cannot be negative"));
        }
        no_gil(|| {
            self.inner
                .lock()
                .wait_for_key(&key, timeout as u64)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to wait for key: {}", e)))
        })
    }

    /// Retrieves the checksum for the data kept under the specified key.
    /// The checksum is calculated using the CRC32 algorithm. The method returns ``None`` if the key does not exist.
    /// The method can be used to check if the data has changed.
    ///
    /// The method is GIL-free.
    ///
    /// Arguments
    /// ----------
    /// key: str
    ///   The key to get the checksum for.
    ///
    /// Returns
    /// -------
    /// int or None
    ///   The checksum for the data kept under the specified key or ``None`` if the key does not exist.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///   If the storage is not active.
    ///
    ///
    fn get_data_checksum(&self, key: String) -> PyResult<Option<u32>> {
        no_gil(|| {
            self.inner
                .lock()
                .get_data_checksum(&key)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get data checksum: {}", e)))
        })
    }

    /// Requests the storage to update the data kept under the specified key.
    ///
    /// The method is GIL-free.
    ///
    /// Arguments
    /// ----------
    /// key: :class:`VarPathSpec`
    ///   The key specification to update the data for.
    ///
    /// Returns
    /// -------
    /// None
    ///   The method does not return anything.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///   If the storage is not active.
    ///
    fn order_data_update(&self, spec: VarPathSpec) -> PyResult<()> {
        no_gil(|| {
            self.inner
                .lock()
                .order_data_update(spec.into())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to order data update: {}", e)))
        })
    }

    /// Retrieves the data kept under the specified key.
    ///
    /// The method is GIL-free.
    ///
    /// Arguments
    /// ----------
    /// key: str
    ///   The key to get the data for.
    ///
    /// Returns
    /// -------
    /// None or :class:`savant_rs.utils.ByteBuffer`
    ///   The data kept under the specified key or ``None`` if the key does not exist.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///   If the storage is not active.
    ///
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

    /// Sets the key and value from Python bytes.
    ///
    /// The method is GIL-free.
    ///
    /// Arguments
    /// ----------
    /// key: str
    ///   The key to set the data for.
    /// value: bytes
    ///   The data to set.
    ///
    /// Returns
    /// -------
    /// None
    ///   The method does not return anything.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///   If the storage is not active.
    ///
    pub fn set_raw(&self, key: String, value: Vec<u8>) -> PyResult<()> {
        no_gil(|| {
            self.inner
                .lock()
                .set(&key, value)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to set data: {}", e)))
        })
    }

    /// Sets the key and value from :class:`savant_rs.utils.ByteBuffer`.
    ///
    /// The method is GIL-free.
    ///
    /// Arguments
    /// ----------
    /// key: str
    ///   The key to set the data for.
    /// value: :class:`savant_rs.utils.ByteBuffer`
    ///   The data to set.
    ///
    /// Returns
    /// -------
    /// None
    ///   The method does not return anything.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///   If the storage is not active.
    ///
    pub fn set_byte_buffer(&self, key: String, value: ByteBuffer) -> PyResult<()> {
        no_gil(|| {
            self.inner
                .lock()
                .set(&key, value.bytes().to_vec())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to set data: {}", e)))
        })
    }

    /// Checks if the specified key is present in the storage.
    ///
    /// The method is GIL-free.
    ///
    /// Arguments
    /// ----------
    /// key: str
    ///   The key to check.
    ///
    /// Returns
    /// -------
    /// bool
    ///   ``True`` if the key is present, ``False`` otherwise.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///   If the storage is not active.
    ///
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

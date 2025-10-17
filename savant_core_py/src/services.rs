pub mod ffmpeg_source {
    use pyo3::{pyclass, pymethods};

    #[pyclass]
    #[derive(Debug, Clone)]
    pub struct StreamProperties {}

    #[pymethods]
    impl StreamProperties {
        #[new]
        pub fn new() -> Self {
            Self {}
        }
    }

    #[pyclass]
    #[derive(Debug, Clone)]
    pub struct StreamInfoVideoFile {
        #[pyo3(get, set)]
        path: String,
        #[pyo3(get, set)]
        looped: bool,
        #[pyo3(get, set)]
        sync: bool,
    }

    #[pymethods]
    impl StreamInfoVideoFile {
        #[new]
        #[pyo3(signature = (path, looped, sync))]
        pub fn new(path: String, looped: bool, sync: bool) -> Self {
            Self { path, looped, sync }
        }
    }

    #[pyclass]
    #[derive(Debug, Clone)]
    pub struct StreamInfoRTSP {
        #[pyo3(get, set)]
        url: String,
        #[pyo3(get, set)]
        tcp: bool,
    }

    #[pymethods]
    impl StreamInfoRTSP {
        #[new]
        #[pyo3(signature = (url, tcp))]
        pub fn new(url: String, tcp: bool) -> Self {
            Self { url, tcp }
        }
    }
}

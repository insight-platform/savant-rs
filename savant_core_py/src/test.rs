pub mod utils {
    use crate::primitives::frame::VideoFrame;
    use crate::primitives::object::VideoObject;
    use pyo3::pyfunction;

    #[pyfunction]
    pub fn gen_empty_frame() -> VideoFrame {
        VideoFrame(savant_core::test::gen_empty_frame())
    }

    #[pyfunction]
    pub fn gen_frame() -> VideoFrame {
        VideoFrame(savant_core::test::gen_frame())
    }

    pub fn gen_object(id: i64) -> VideoObject {
        VideoObject(savant_core::test::gen_object(id))
    }

    #[inline(always)]
    pub fn s(a: &str) -> String {
        a.to_string()
    }
}

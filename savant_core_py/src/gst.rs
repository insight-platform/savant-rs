mod attribute_meta;

use crate::gst::attribute_meta::SavantAttributeMeta;
use crate::primitives::attribute::Attribute;
use gst::BufferFlags;
use parking_lot::RwLock;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
#[derive(Clone)]
pub struct GstBuffer(Arc<RwLock<gst::Buffer>>);

impl GstBuffer {
    pub fn new(buffer: gst::Buffer) -> Self {
        Self(Arc::new(RwLock::new(buffer)))
    }

    pub fn extract(self) -> anyhow::Result<gst::Buffer> {
        let lock = Arc::try_unwrap(self.0).map_err(|_| {
            anyhow::anyhow!("Could not extract GstBuffer because multiple object references exist")
        })?;
        Ok(lock.into_inner())
    }
}

#[pymethods]
impl GstBuffer {
    #[getter]
    pub fn pts(&self) -> Option<u64> {
        let bind = self.0.read();
        bind.pts().map(|pts| pts.nseconds())
    }

    #[new]
    pub fn create_py() -> Self {
        Self(Arc::new(RwLock::new(gst::Buffer::new())))
    }

    #[setter]
    pub fn set_pts(&self, pts: u64) {
        let mut bind = self.0.write();
        bind.make_mut().set_pts(gst::ClockTime::from_nseconds(pts));
    }

    #[getter]
    pub fn dts(&self) -> Option<u64> {
        let bind = self.0.read();
        bind.dts().map(|dts| dts.nseconds())
    }

    #[setter]
    pub fn set_dts(&self, dts: u64) {
        let mut bind = self.0.write();
        bind.make_mut().set_dts(gst::ClockTime::from_nseconds(dts));
    }

    #[getter]
    pub fn dts_or_pts(&self) -> Option<u64> {
        let bind = self.0.read();
        bind.dts_or_pts().map(|dts_or_pts| dts_or_pts.nseconds())
    }

    #[getter]
    pub fn duration(&self) -> Option<u64> {
        let bind = self.0.read();
        bind.duration().map(|duration| duration.nseconds())
    }

    #[setter]
    pub fn set_duration(&self, duration: u64) {
        let mut bind = self.0.write();
        bind.make_mut()
            .set_duration(gst::ClockTime::from_nseconds(duration));
    }

    #[getter]
    pub fn is_writable(&self) -> bool {
        let bind = self.0.read();
        bind.is_writable()
    }

    #[getter]
    pub fn flags(&self) -> u32 {
        let bind = self.0.read();
        bind.flags().bits()
    }

    #[setter]
    pub fn set_flags(&self, flags: u32) {
        let mut bind = self.0.write();
        bind.make_mut()
            .set_flags(BufferFlags::from_bits_retain(flags));
    }

    pub fn unset_flags(&self, flags: u32) {
        let mut bind = self.0.write();
        bind.make_mut()
            .unset_flags(BufferFlags::from_bits_retain(flags));
    }

    #[getter]
    pub fn maxsize(&self) -> usize {
        let bind = self.0.read();
        bind.maxsize()
    }

    #[getter]
    pub fn n_memory(&self) -> usize {
        let bind = self.0.read();
        bind.n_memory()
    }

    #[getter]
    pub fn offset(&self) -> u64 {
        let bind = self.0.read();
        bind.offset()
    }

    #[setter]
    pub fn set_offset(&self, offset: u64) {
        let mut bind = self.0.write();
        bind.make_mut().set_offset(offset);
    }

    #[getter]
    pub fn offset_end(&self) -> u64 {
        let bind = self.0.read();
        bind.offset_end()
    }

    #[setter]
    pub fn set_offset_end(&self, offset_end: u64) {
        let mut bind = self.0.write();
        bind.make_mut().set_offset_end(offset_end);
    }

    #[getter]
    pub fn size(&self) -> usize {
        let bind = self.0.read();
        bind.size()
    }

    #[setter]
    pub fn set_size(&self, size: usize) {
        let mut bind = self.0.write();
        bind.make_mut().set_size(size);
    }

    pub fn copy(&self) -> Self {
        let bind = self.0.read();
        let new = Arc::new(RwLock::new(bind.copy()));
        Self(new)
    }

    pub fn copy_deep(&self) -> PyResult<Self> {
        let bind = self.0.read();
        let new_buf = bind
            .copy_deep()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self(Arc::new(RwLock::new(new_buf))))
    }

    pub fn append(&self, buffer: GstBuffer) -> PyResult<()> {
        let mut bind = self.0.write();
        bind.append(buffer.0.read().clone());
        Ok(())
    }

    #[getter]
    pub fn get_savant_meta(&self) -> Option<Vec<Attribute>> {
        let bind = self.0.read();
        let attribute_meta_opt = bind.meta::<SavantAttributeMeta>();
        if attribute_meta_opt.is_none() {
            return None;
        }
        let attribute_meta = attribute_meta_opt.unwrap();
        Some(
            attribute_meta
                .attributes()
                .iter()
                .map(|a| Attribute(a.clone()))
                .collect(),
        )
    }

    pub fn replace_savant_meta(&self, attributes: Vec<Attribute>) -> Option<Vec<Attribute>> {
        let mut bind = self.0.write();
        let old_attributes = self.get_savant_meta();

        let mut buffer = bind.get_mut()?;
        let core_attributes = attributes.iter().map(|a| a.0.clone()).collect::<Vec<_>>();
        SavantAttributeMeta::replace(&mut buffer, core_attributes);
        old_attributes
    }

    pub fn clear_savant_meta(&self) -> PyResult<Option<Vec<Attribute>>> {
        let mut bind = self.0.write();
        let old_attributes = self.get_savant_meta();
        if old_attributes.is_none() {
            return Ok(None);
        }

        let buffer_ref_mut = bind.get_mut().ok_or(PyRuntimeError::new_err(
            "Unable to get write access to the buffer.",
        ))?;

        let meta_ref_mut_opt = buffer_ref_mut.meta_mut::<SavantAttributeMeta>();
        if meta_ref_mut_opt.is_none() {
            return Ok(None);
        }

        let meta_ref_mut = meta_ref_mut_opt.unwrap();
        meta_ref_mut
            .remove()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(old_attributes)
    }
}

#[cfg(test)]
mod tests {
    use crate::gst::GstBuffer;
    use crate::primitives::attribute::Attribute;

    #[test]
    fn test_savant_meta() -> anyhow::Result<()> {
        gst::init().unwrap();
        let buf = GstBuffer::create_py();
        let old_attributes = vec![Attribute::new("test", "test", vec![], None, true, false)];
        let new_attributes = vec![Attribute::new("test2", "test2", vec![], None, true, false)];

        assert!(
            buf.get_savant_meta().is_none(),
            "Must return None, when no meta is present"
        );

        assert!(
            buf.clear_savant_meta()?.is_none(),
            "Must return None, when no meta is present"
        );

        assert!(
            buf.replace_savant_meta(old_attributes.clone()).is_none(),
            "Must return None, when no meta is present during replacement"
        );

        let old_attributes_retrieved = buf.get_savant_meta().unwrap();
        assert_eq!(
            old_attributes_retrieved, old_attributes,
            "Must return the old attributes when they are set"
        );

        let prev_attributes = buf.replace_savant_meta(new_attributes.clone());
        assert_eq!(
            prev_attributes.unwrap(),
            old_attributes,
            "Must return the old attributes when they are replaced with new attributes"
        );

        let new_attributes_retrieved = buf.get_savant_meta().unwrap();
        assert_eq!(
            new_attributes_retrieved, new_attributes,
            "Must return the new attributes"
        );

        let last_attrs = buf.clear_savant_meta()?.unwrap();
        assert_eq!(
            last_attrs, new_attributes,
            "Must return the new attributes when they are cleared"
        );

        unsafe {
            gst::deinit();
        }
        Ok(())
    }
}

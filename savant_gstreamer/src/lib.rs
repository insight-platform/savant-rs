pub mod id_meta;

use gst::BufferFlags;
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Clone)]
pub struct GstBuffer(Arc<RwLock<gst::Buffer>>);

impl From<gst::Buffer> for GstBuffer {
    fn from(buffer: gst::Buffer) -> Self {
        Self::new_from(buffer)
    }
}

impl Default for GstBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl GstBuffer {
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(gst::Buffer::new())))
    }

    fn new_from(buffer: gst::Buffer) -> Self {
        Self(Arc::new(RwLock::new(buffer)))
    }

    pub fn extract(self) -> anyhow::Result<gst::Buffer> {
        let lock = Arc::try_unwrap(self.0).map_err(|_| {
            anyhow::anyhow!("Could not extract GstBuffer because multiple object references exist")
        })?;
        Ok(lock.into_inner())
    }

    pub fn raw_pointer(&self) -> usize {
        let bind = self.0.read();
        bind.as_ptr() as usize
    }

    pub fn pts_ns(&self) -> Option<u64> {
        let bind = self.0.read();
        bind.pts().map(|pts| pts.nseconds())
    }

    pub fn set_pts_ns(&self, pts: u64) {
        let mut bind = self.0.write();
        bind.make_mut().set_pts(gst::ClockTime::from_nseconds(pts));
    }

    pub fn dts_ns(&self) -> Option<u64> {
        let bind = self.0.read();
        bind.dts().map(|dts| dts.nseconds())
    }

    pub fn set_dts_ns(&self, dts: u64) {
        let mut bind = self.0.write();
        bind.make_mut().set_dts(gst::ClockTime::from_nseconds(dts));
    }

    pub fn dts_or_pts_ns(&self) -> Option<u64> {
        let bind = self.0.read();
        bind.dts_or_pts().map(|dts_or_pts| dts_or_pts.nseconds())
    }

    pub fn duration_ns(&self) -> Option<u64> {
        let bind = self.0.read();
        bind.duration().map(|duration| duration.nseconds())
    }

    pub fn set_duration_ns(&self, duration: u64) {
        let mut bind = self.0.write();
        bind.make_mut()
            .set_duration(gst::ClockTime::from_nseconds(duration));
    }

    pub fn is_writable(&self) -> bool {
        let bind = self.0.read();
        bind.is_writable()
    }

    pub fn flags(&self) -> BufferFlags {
        let bind = self.0.read();
        bind.flags()
    }

    pub fn set_flags(&self, flags: BufferFlags) {
        let mut bind = self.0.write();
        bind.make_mut().set_flags(flags);
    }

    pub fn unset_flags(&self, flags: BufferFlags) {
        let mut bind = self.0.write();
        bind.make_mut().unset_flags(flags);
    }

    pub fn maxsize(&self) -> usize {
        let bind = self.0.read();
        bind.maxsize()
    }

    pub fn n_memory(&self) -> usize {
        let bind = self.0.read();
        bind.n_memory()
    }

    pub fn offset(&self) -> u64 {
        let bind = self.0.read();
        bind.offset()
    }

    pub fn set_offset(&self, offset: u64) {
        let mut bind = self.0.write();
        bind.make_mut().set_offset(offset);
    }

    pub fn offset_end(&self) -> u64 {
        let bind = self.0.read();
        bind.offset_end()
    }

    pub fn set_offset_end(&self, offset_end: u64) {
        let mut bind = self.0.write();
        bind.make_mut().set_offset_end(offset_end);
    }

    pub fn size(&self) -> usize {
        let bind = self.0.read();
        bind.size()
    }

    pub fn set_size(&self, size: usize) {
        let mut bind = self.0.write();
        bind.make_mut().set_size(size);
    }

    pub fn copy(&self) -> Self {
        let bind = self.0.read();
        let new = Arc::new(RwLock::new(bind.copy()));
        Self(new)
    }

    pub fn copy_deep(&self) -> anyhow::Result<Self> {
        let bind = self.0.read();
        let new_buf = bind.copy_deep()?;
        Ok(Self(Arc::new(RwLock::new(new_buf))))
    }

    pub fn append(&self, buffer: GstBuffer) {
        let mut bind = self.0.write();
        bind.append(buffer.0.read().clone());
    }

    pub fn get_id_meta(&self) -> Option<Vec<id_meta::SavantIdMetaKind>> {
        let bind = self.0.read();
        let meta = bind.meta::<id_meta::SavantIdMeta>()?;
        Some(meta.ids().to_vec())
    }

    pub fn replace_id_meta(
        &self,
        ids: Vec<id_meta::SavantIdMetaKind>,
    ) -> anyhow::Result<Option<Vec<id_meta::SavantIdMetaKind>>> {
        let old_ids = self.clear_id_meta()?;
        let mut bind = self.0.write();
        let buffer = bind
            .get_mut()
            .ok_or(anyhow::anyhow!("Unable to get write access to the buffer.",))?;
        id_meta::SavantIdMeta::replace(buffer, ids);
        Ok(old_ids)
    }

    pub fn clear_id_meta(&self) -> anyhow::Result<Option<Vec<id_meta::SavantIdMetaKind>>> {
        let old_ids = self.get_id_meta();
        if old_ids.is_none() {
            return Ok(None);
        }

        let mut bind = self.0.write();
        let buffer_ref_mut = bind
            .get_mut()
            .ok_or(anyhow::anyhow!("Unable to get write access to the buffer.",))?;

        let meta_ref_mut_opt = buffer_ref_mut.meta_mut::<id_meta::SavantIdMeta>();
        if meta_ref_mut_opt.is_none() {
            return Ok(None);
        }

        let meta_ref_mut = meta_ref_mut_opt.unwrap();
        meta_ref_mut.remove()?;

        Ok(old_ids)
    }

    pub fn memory(&self, idx: usize) -> Option<gst::Memory> {
        let bind = self.0.read();
        bind.memory(idx)
    }
}

#[cfg(test)]
mod tests {
    use crate::id_meta::SavantIdMetaKind::*;
    use crate::GstBuffer;
    #[test]
    fn test_savant_meta() -> anyhow::Result<()> {
        gst::init().unwrap();
        let buf = GstBuffer::new();
        let old_attributes = vec![Frame(0), Frame(1), Frame(2)];
        let new_attributes = vec![Frame(3), Frame(4), Frame(6)];

        assert!(
            buf.get_id_meta().is_none(),
            "Must return None, when no meta is present"
        );

        assert!(
            buf.clear_id_meta()?.is_none(),
            "Must return None, when no meta is present"
        );

        assert!(
            buf.replace_id_meta(old_attributes.clone())?.is_none(),
            "Must return None, when no meta is present during replacement"
        );

        let old_attributes_retrieved = buf.get_id_meta().unwrap();
        assert_eq!(
            old_attributes_retrieved, old_attributes,
            "Must return the old attributes when they are set"
        );

        let prev_attributes = buf.replace_id_meta(new_attributes.clone())?;
        assert_eq!(
            prev_attributes.unwrap(),
            old_attributes,
            "Must return the old attributes when they are replaced with new attributes"
        );

        let new_attributes_retrieved = buf.get_id_meta().unwrap();
        assert_eq!(
            new_attributes_retrieved, new_attributes,
            "Must return the new attributes"
        );

        let last_attrs = buf.clear_id_meta()?.unwrap();
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

use crate::primitives::message::video::object::VideoObject;
use crate::primitives::proxy::video_object_rbbox::VideoObjectRBBoxProxy;
use crate::primitives::proxy::{StrongInnerType, UpgradeableWeakInner, WeakInner};
use crate::primitives::{VideoObjectBBoxType, VideoObjectModification};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub struct VideoObjectTrackingDataProxy {
    object: WeakInner<VideoObject>,
}

impl VideoObjectTrackingDataProxy {
    pub fn new(object: StrongInnerType<VideoObject>) -> Self {
        Self {
            object: WeakInner::new(object),
        }
    }

    fn get_object(&self) -> StrongInnerType<VideoObject> {
        self.object.get_or_fail()
    }
}

#[pymethods]
impl VideoObjectTrackingDataProxy {
    fn is_defined(&self) -> bool {
        self.get_object().read().track_info.is_some()
    }

    #[getter]
    fn get_id(&self) -> PyResult<i64> {
        self.get_object()
            .read()
            .track_info
            .as_ref()
            .map(|x| x.id)
            .ok_or(pyo3::exceptions::PyAttributeError::new_err(
                "Track info is not defined",
            ))
    }

    #[setter]
    fn set_id(&self, id: i64) -> PyResult<()> {
        if !self.is_defined() {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "Track info is not defined",
            ));
        }
        let binding = self.get_object();
        let mut obj = binding.write();
        let new_track_info = obj.track_info.as_mut().map(|x| {
            let mut new_track_info = x.clone();
            new_track_info.id = id;
            new_track_info
        });
        obj.track_info = new_track_info;
        obj.add_modification(VideoObjectModification::TrackInfo);

        Ok(())
    }

    #[getter]
    fn get_bbox_ref(&self) -> PyResult<VideoObjectRBBoxProxy> {
        if self.is_defined() {
            Ok(VideoObjectRBBoxProxy::new(
                self.get_object(),
                VideoObjectBBoxType::TrackingInfo,
            ))
        } else {
            Err(pyo3::exceptions::PyAttributeError::new_err(
                "Track info is not defined",
            ))
        }
    }
}

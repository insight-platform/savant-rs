use crate::primitives::message::video::frame::{PyFrameTransformation, VideoFrame};
use crate::primitives::{Attribute, Object, PyVideoFrameContent};
use pyo3::{pyclass, pymethods};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[pyclass]
#[derive(Debug, Clone)]
pub struct ProxyVideoFrame {
    pub(crate) frame: Arc<Mutex<Box<VideoFrame>>>,
}

impl ProxyVideoFrame {
    pub(crate) fn new(object: VideoFrame) -> Self {
        ProxyVideoFrame {
            frame: Arc::new(Mutex::new(Box::new(object))),
        }
    }
}

#[pymethods]
impl ProxyVideoFrame {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(
        signature = (source_id, framerate, width, height, content, codec=None, keyframe=None, pts=0, dts=None, duration=None)
    )]
    pub fn new_py(
        source_id: String,
        framerate: String,
        width: i64,
        height: i64,
        content: PyVideoFrameContent,
        codec: Option<String>,
        keyframe: Option<bool>,
        pts: i64,
        dts: Option<i64>,
        duration: Option<i64>,
    ) -> Self {
        ProxyVideoFrame::new(VideoFrame {
            source_id,
            pts,
            framerate,
            width,
            height,
            dts,
            duration,
            codec,
            keyframe,
            transformations: vec![],
            content: content.inner,
            attributes: HashMap::default(),
            offline_objects: vec![],
            resident_objects: vec![],
        })
    }

    #[getter]
    pub fn get_source_id(&self) -> String {
        self.frame.lock().unwrap().source_id.clone()
    }

    #[setter]
    pub fn set_source_id(&mut self, source_id: String) {
        let mut frame = self.frame.lock().unwrap();
        frame.source_id = source_id;
    }

    #[getter]
    pub fn get_pts(&self) -> i64 {
        self.frame.lock().unwrap().pts
    }

    #[setter]
    pub fn set_pts(&mut self, pts: i64) {
        assert_eq!(pts >= 0, true, "pts must be greater than or equal to 0");
        let mut frame = self.frame.lock().unwrap();
        frame.pts = pts;
    }

    #[getter]
    pub fn get_framerate(&self) -> String {
        self.frame.lock().unwrap().framerate.clone()
    }

    #[setter]
    pub fn set_framerate(&mut self, framerate: String) {
        let mut frame = self.frame.lock().unwrap();
        frame.framerate = framerate;
    }

    #[getter]
    pub fn get_width(&self) -> i64 {
        self.frame.lock().unwrap().width
    }

    #[setter]
    pub fn set_width(&mut self, width: i64) {
        assert!(width > 0, "width must be greater than 0");
        let mut frame = self.frame.lock().unwrap();
        frame.width = width;
    }

    #[getter]
    pub fn get_height(&self) -> i64 {
        self.frame.lock().unwrap().height
    }

    #[setter]
    pub fn set_height(&mut self, height: i64) {
        assert!(height > 0, "height must be greater than 0");
        let mut frame = self.frame.lock().unwrap();
        frame.height = height;
    }

    #[getter]
    pub fn get_dts(&self) -> Option<i64> {
        let frame = self.frame.lock().unwrap();
        frame.dts
    }

    #[setter]
    pub fn set_dts(&mut self, dts: Option<i64>) {
        assert!(
            dts.is_none() || dts.unwrap() >= 0,
            "dts must be greater than or equal to 0"
        );
        let mut frame = self.frame.lock().unwrap();
        frame.dts = dts;
    }

    #[getter]
    pub fn get_duration(&self) -> Option<i64> {
        let frame = self.frame.lock().unwrap();
        frame.duration
    }

    #[setter]
    pub fn set_duration(&mut self, duration: Option<i64>) {
        assert!(
            duration.is_none() || duration.unwrap() >= 0,
            "duration must be greater than or equal to 0"
        );
        let mut frame = self.frame.lock().unwrap();
        frame.duration = duration;
    }

    #[getter]
    pub fn get_codec(&self) -> Option<String> {
        let frame = self.frame.lock().unwrap();
        frame.codec.clone()
    }

    #[setter]
    pub fn set_codec(&mut self, codec: Option<String>) {
        let mut frame = self.frame.lock().unwrap();
        frame.codec = codec;
    }

    pub fn clear_transformations(&mut self) {
        let mut frame = self.frame.lock().unwrap();
        frame.transformations.clear();
    }

    pub fn add_transformation(&mut self, transformation: PyFrameTransformation) {
        let mut frame = self.frame.lock().unwrap();
        frame.add_transformation(transformation.inner);
    }

    pub fn get_transformations(&self) -> Vec<PyFrameTransformation> {
        let frame = self.frame.lock().unwrap();
        frame
            .transformations
            .iter()
            .map(|t| PyFrameTransformation::new(t.clone()))
            .collect()
    }

    pub fn get_keyframe(&self) -> Option<bool> {
        let frame = self.frame.lock().unwrap();
        frame.keyframe
    }

    pub fn set_keyframe(&mut self, keyframe: Option<bool>) {
        let mut frame = self.frame.lock().unwrap();
        frame.keyframe = keyframe;
    }

    pub fn get_content(&self) -> PyVideoFrameContent {
        let frame = self.frame.lock().unwrap();
        PyVideoFrameContent::new(frame.content.clone())
    }

    pub fn set_content(&mut self, content: PyVideoFrameContent) {
        let mut frame = self.frame.lock().unwrap();
        frame.content = content.inner;
    }

    pub fn find_attributes(
        &self,
        creator: Option<String>,
        name: Option<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        let frame = self.frame.lock().unwrap();
        frame.find_attributes(creator, name, hint)
    }

    pub fn get_attribute(&self, creator: String, name: String) -> Option<Attribute> {
        let frame = self.frame.lock().unwrap();
        frame.get_attribute(creator, name)
    }

    pub fn delete_attribute(&self, creator: String, name: String) -> Option<Attribute> {
        let mut frame = self.frame.lock().unwrap();
        frame.delete_attribute(creator, name)
    }

    pub fn set_attribute(&mut self, attribute: Attribute) -> Option<Attribute> {
        let mut frame = self.frame.lock().unwrap();
        frame.set_attribute(attribute)
    }

    pub fn clear_attributes(&mut self) {
        let mut frame = self.frame.lock().unwrap();
        frame.clear_attributes();
    }

    pub fn get_object(&self, id: i64) -> Option<Object> {
        let frame = self.frame.lock().unwrap();
        frame.get_object(id)
    }

    pub fn access_objects(
        &self,
        negated: bool,
        creator: Option<String>,
        label: Option<String>,
    ) -> Vec<Object> {
        let frame = self.frame.lock().unwrap();
        frame.access_objects(negated, creator, label)
    }

    pub fn access_objects_by_id(&self, ids: Vec<i64>) -> Vec<Object> {
        let frame = self.frame.lock().unwrap();
        frame.access_objects_by_id(ids)
    }

    pub fn add_object(&mut self, object: Object) {
        let mut frame = self.frame.lock().unwrap();
        frame.add_object(object.object);
    }

    pub fn delete_objects_by_ids(&mut self, ids: Vec<i64>) {
        let mut frame = self.frame.lock().unwrap();
        frame.delete_objects_by_ids(ids);
    }

    pub fn delete_objects(
        &mut self,
        negated: bool,
        creator: Option<String>,
        label: Option<String>,
    ) {
        let mut frame = self.frame.lock().unwrap();
        frame.delete_objects(negated, creator, label);
    }

    pub fn clear_objects(&mut self) {
        let mut frame = self.frame.lock().unwrap();
        frame.clear_objects();
    }
}

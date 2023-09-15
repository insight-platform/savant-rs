use anyhow::bail;
use hashbrown::HashMap;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use rkyv::{with::Skip, Archive, Deserialize, Serialize};
use std::sync::Arc;

use crate::consts::BBOX_UNDEFINED;
use crate::json_api::ToSerdeJsonValue;
use crate::primitives::frame::{BelongingVideoFrame, VideoFrameProxy};
use crate::primitives::{Attribute, AttributeMethods, Attributive, OwnedRBBoxData, RBBox};
use crate::symbol_mapper::get_object_id;
use crate::trace;
use serde_json::Value;

#[derive(Clone, Debug, Copy)]
#[repr(C)]
pub enum VideoObjectBBoxType {
    Detection,
    TrackingInfo,
}

#[derive(Debug, Clone, Copy)]
pub enum VideoObjectBBoxTransformation {
    Scale(f32, f32),
    Shift(f32, f32),
}

#[derive(Debug, Clone)]
pub enum IdCollisionResolutionPolicy {
    GenerateNewId,
    Overwrite,
    Error,
}

#[derive(
    Archive,
    Deserialize,
    Serialize,
    Debug,
    Clone,
    derive_builder::Builder,
    serde::Serialize,
    serde::Deserialize,
)]
#[archive(check_bytes)]
pub struct VideoObject {
    pub id: i64,
    pub namespace: String,
    pub label: String,
    #[builder(default)]
    pub draw_label: Option<String>,
    pub detection_box: OwnedRBBoxData,
    #[builder(default)]
    pub attributes: HashMap<(String, String), Attribute>,
    #[builder(default)]
    pub confidence: Option<f32>,
    #[builder(default)]
    pub(crate) parent_id: Option<i64>,
    #[builder(default)]
    pub(crate) track_box: Option<OwnedRBBoxData>,
    #[builder(default)]
    pub track_id: Option<i64>,
    #[with(Skip)]
    #[builder(default)]
    pub namespace_id: Option<i64>,
    #[with(Skip)]
    #[builder(default)]
    pub label_id: Option<i64>,
    #[with(Skip)]
    #[builder(default)]
    #[serde(skip_deserializing, skip_serializing)]
    pub(crate) frame: Option<BelongingVideoFrame>,
}

const DEFAULT_ATTRIBUTES_COUNT: usize = 4;

impl Default for VideoObject {
    fn default() -> Self {
        Self {
            id: 0,
            namespace: "".to_string(),
            label: "".to_string(),
            draw_label: None,
            detection_box: BBOX_UNDEFINED.clone().try_into().unwrap(),
            attributes: HashMap::with_capacity(DEFAULT_ATTRIBUTES_COUNT),
            confidence: None,
            parent_id: None,
            track_id: None,
            track_box: None,
            namespace_id: None,
            label_id: None,
            frame: None,
        }
    }
}

impl ToSerdeJsonValue for VideoObject {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!({
            "id": self.id,
            "namespace": self.namespace,
            "label": self.label,
            "draw_label": self.draw_label,
            "bbox": self.detection_box.to_serde_json_value(),
            "attributes": self.attributes.values().map(|v| v.to_serde_json_value()).collect::<Vec<_>>(),
            "confidence": self.confidence,
            "parent": self.parent_id,
            "track_id": self.track_id,
            "track_box": self.track_box.as_ref().map(|x| x.to_serde_json_value()),
            "frame": self.get_parent_frame_source(),
            "pyobjects": "not_implemented",
        })
    }
}

impl VideoObject {
    pub fn get_parent_frame_source(&self) -> Option<String> {
        self.frame.as_ref().and_then(|f| {
            f.inner
                .upgrade()
                .map(|f| trace!(f.read_recursive()).source_id.clone())
        })
    }
}

/// Represents a video object. The object is a part of a video frame, it includes bounding
/// box, attributes, label, namespace label, etc. The objects are always accessible by reference. The only way to
/// copy the object by value is to call :py:meth:`VideoObject.detached_copy`.
///
/// :py:class:`VideoObject` is a part of :py:class:`VideoFrame` and may outlive it if there are references.
///
#[derive(Debug, Clone)]
pub struct VideoObjectProxy {
    pub inner: Arc<RwLock<VideoObject>>,
}

impl ToSerdeJsonValue for VideoObjectProxy {
    fn to_serde_json_value(&self) -> serde_json::Value {
        trace!(self.inner.read_recursive()).to_serde_json_value()
    }
}

impl Attributive for VideoObject {
    fn get_attributes_ref(&self) -> &HashMap<(String, String), Attribute> {
        &self.attributes
    }

    fn get_attributes_ref_mut(&mut self) -> &mut HashMap<(String, String), Attribute> {
        &mut self.attributes
    }

    fn take_attributes(&mut self) -> HashMap<(String, String), Attribute> {
        std::mem::take(&mut self.attributes)
    }

    fn place_attributes(&mut self, attributes: HashMap<(String, String), Attribute>) {
        self.attributes = attributes;
    }
}

impl AttributeMethods for VideoObjectProxy {
    fn exclude_temporary_attributes(&self) -> Vec<Attribute> {
        let mut inner = trace!(self.inner.write());
        inner.exclude_temporary_attributes()
    }

    fn restore_attributes(&self, attributes: Vec<Attribute>) {
        let mut inner = trace!(self.inner.write());
        inner.restore_attributes(attributes)
    }

    fn get_attributes(&self) -> Vec<(String, String)> {
        let inner = trace!(self.inner.read_recursive());
        inner.get_attributes()
    }

    fn get_attribute(&self, namespace: String, name: String) -> Option<Attribute> {
        let inner = trace!(self.inner.read_recursive());
        inner.get_attribute(namespace, name)
    }

    fn delete_attribute(&self, namespace: String, name: String) -> Option<Attribute> {
        let mut inner = trace!(self.inner.write());
        inner.delete_attribute(namespace, name)
    }

    fn set_attribute(&self, attribute: Attribute) -> Option<Attribute> {
        let mut inner = trace!(self.inner.write());
        inner.set_attribute(attribute)
    }

    fn clear_attributes(&self) {
        let mut inner = trace!(self.inner.write());
        inner.clear_attributes()
    }

    fn delete_attributes(&self, namespace: Option<String>, names: Vec<String>) {
        let mut inner = trace!(self.inner.write());
        inner.delete_attributes(namespace, names)
    }

    fn find_attributes(
        &self,
        namespace: Option<String>,
        names: Vec<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        let inner = trace!(self.inner.read_recursive());
        inner.find_attributes(namespace, names, hint)
    }
}

impl From<VideoObject> for VideoObjectProxy {
    fn from(value: VideoObject) -> Self {
        Self {
            inner: Arc::new(RwLock::new(value)),
        }
    }
}

impl From<Arc<RwLock<VideoObject>>> for VideoObjectProxy {
    fn from(value: Arc<RwLock<VideoObject>>) -> Self {
        Self { inner: value }
    }
}

impl VideoObjectProxy {
    pub fn get_parent_id(&self) -> Option<i64> {
        let inner = trace!(self.inner.read_recursive());
        inner.parent_id
    }

    pub fn get_inner(&self) -> Arc<RwLock<VideoObject>> {
        self.inner.clone()
    }

    pub fn get_inner_read(&self) -> RwLockReadGuard<VideoObject> {
        trace!(self.inner.read_recursive())
    }

    pub fn get_inner_write(&self) -> RwLockWriteGuard<VideoObject> {
        trace!(self.inner.write())
    }

    pub(crate) fn set_parent(&self, parent_opt: Option<i64>) {
        if let Some(parent) = parent_opt {
            if self.get_frame().is_none() {
                panic!("Cannot set parent to the object detached from a frame");
            }
            if self.get_id() == parent {
                panic!("Cannot set parent to itself");
            }
            let f = self.get_frame().unwrap();
            if !f.object_exists(parent) {
                panic!("Cannot set parent to the object which cannot be found in the frame");
            }
        }

        let mut inner = trace!(self.inner.write());
        inner.parent_id = parent_opt;
    }

    pub fn get_parent(&self) -> Option<VideoObjectProxy> {
        let frame = self.get_frame();
        let id = trace!(self.inner.read_recursive()).parent_id?;
        match frame {
            Some(f) => f.get_object(id),
            None => None,
        }
    }

    pub fn get_children(&self) -> Vec<VideoObjectProxy> {
        let frame = self.get_frame();
        let id = self.get_id();
        match frame {
            Some(f) => f.get_children(id),
            None => Vec::new(),
        }
    }

    pub(crate) fn attach_to_video_frame(&self, frame: VideoFrameProxy) {
        let mut inner = trace!(self.inner.write());
        inner.frame = Some(frame.into());
    }

    pub fn transform_geometry(&self, ops: &Vec<VideoObjectBBoxTransformation>) {
        for o in ops {
            match o {
                VideoObjectBBoxTransformation::Scale(kx, ky) => {
                    self.get_detection_box().scale(*kx, *ky);
                    if let Some(mut t) = self.get_track_box() {
                        t.scale(*kx, *ky);
                    }
                }
                VideoObjectBBoxTransformation::Shift(dx, dy) => {
                    self.get_detection_box().shift(*dx, *dy);
                    if let Some(mut t) = self.get_track_box() {
                        t.shift(*dx, *dy);
                    }
                }
            }
        }
    }
    pub fn get_track_id(&self) -> Option<i64> {
        let inner = trace!(self.inner.read_recursive());
        inner.track_id
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: i64,
        namespace: String,
        label: String,
        detection_box: RBBox,
        attributes: HashMap<(String, String), Attribute>,
        confidence: Option<f32>,
        track_id: Option<i64>,
        track_box: Option<RBBox>,
    ) -> Self {
        let (namespace_id, label_id) =
            get_object_id(&namespace, &label).map_or((None, None), |(c, o)| (Some(c), Some(o)));

        let object = VideoObject {
            id,
            namespace,
            label,
            detection_box: detection_box
                .try_into()
                .expect("Failed to convert RBBox to RBBoxData"),
            attributes,
            confidence,
            track_id,
            track_box: track_box
                .map(|b| b.try_into().expect("Failed to convert RBBox to RBBoxData")),
            namespace_id,
            label_id,
            ..Default::default()
        };
        Self {
            inner: Arc::new(RwLock::new(object)),
        }
    }
    pub fn get_detection_box(&self) -> RBBox {
        RBBox::borrowed_detection_box(self.inner.clone())
    }

    pub fn get_track_box(&self) -> Option<RBBox> {
        if self.get_track_id().is_some() {
            Some(RBBox::borrowed_track_box(self.inner.clone()))
        } else {
            None
        }
    }

    pub fn get_confidence(&self) -> Option<f32> {
        let inner = trace!(self.inner.read_recursive());
        inner.confidence
    }

    pub fn get_namespace(&self) -> String {
        trace!(self.inner.read_recursive()).namespace.clone()
    }

    pub fn get_namespace_id(&self) -> Option<i64> {
        let inner = trace!(self.inner.read_recursive());
        inner.namespace_id
    }

    pub fn get_label(&self) -> String {
        trace!(self.inner.read_recursive()).label.clone()
    }

    pub fn get_label_id(&self) -> Option<i64> {
        let inner = trace!(self.inner.read_recursive());
        inner.label_id
    }

    pub fn detached_copy(&self) -> Self {
        let inner = trace!(self.inner.read_recursive());
        let mut new_inner = inner.clone();
        new_inner.parent_id = None;
        new_inner.frame = None;
        Self {
            inner: Arc::new(RwLock::new(new_inner)),
        }
    }

    pub fn get_draw_label(&self) -> String {
        let inner = trace!(self.inner.read_recursive());
        inner.draw_label.as_ref().unwrap_or(&inner.label).clone()
    }

    pub fn get_frame(&self) -> Option<VideoFrameProxy> {
        let inner = trace!(self.inner.read_recursive());
        inner.frame.as_ref().map(|f| f.into())
    }

    pub fn get_id(&self) -> i64 {
        trace!(self.inner.read_recursive()).id
    }

    pub fn is_detached(&self) -> bool {
        let inner = trace!(self.inner.read_recursive());
        inner.frame.is_none()
    }

    pub fn is_spoiled(&self) -> bool {
        let inner = trace!(self.inner.read_recursive());
        match inner.frame {
            Some(ref f) => f.inner.upgrade().is_none(),
            None => false,
        }
    }
    pub fn set_detection_box(&self, bbox: RBBox) {
        let mut inner = trace!(self.inner.write());
        inner.detection_box = bbox
            .try_into()
            .expect("Failed to convert RBBox to RBBoxData");
    }

    pub fn set_track_info(&self, track_id: i64, bbox: RBBox) {
        let mut inner = trace!(self.inner.write());
        inner.track_box = Some(
            bbox.try_into()
                .expect("Failed to convert RBBox to RBBoxData"),
        );
        inner.track_id = Some(track_id);
    }

    pub fn set_track_box(&self, bbox: RBBox) {
        let mut inner = trace!(self.inner.write());
        inner.track_box = Some(
            bbox.try_into()
                .expect("Failed to convert RBBox to RBBoxData"),
        );
    }

    pub fn clear_track_info(&self) {
        let mut inner = trace!(self.inner.write());
        inner.track_box = None;
        inner.track_id = None;
    }

    pub fn set_draw_label(&self, draw_label: Option<String>) {
        let mut inner = trace!(self.inner.write());
        inner.draw_label = draw_label;
    }

    pub fn set_id(&self, id: i64) -> anyhow::Result<()> {
        if self.get_frame().is_some() {
            bail!("When object is attached to a frame, it is impossible to change its ID",);
        }

        let mut inner = trace!(self.inner.write());
        inner.id = id;
        Ok(())
    }

    pub fn set_namespace(&self, namespace: String) {
        let mut inner = trace!(self.inner.write());
        inner.namespace = namespace;
    }

    pub fn set_label(&self, label: String) {
        let mut inner = trace!(self.inner.write());
        inner.label = label;
    }

    pub fn set_confidence(&self, confidence: Option<f32>) {
        let mut inner = trace!(self.inner.write());
        inner.confidence = confidence;
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute::AttributeMethods;
    use crate::primitives::attribute_value::{AttributeValue, AttributeValueVariant};
    use crate::primitives::object::{
        IdCollisionResolutionPolicy, VideoObjectBBoxTransformation, VideoObjectBuilder,
        VideoObjectProxy,
    };
    use crate::primitives::{Attribute, RBBox};
    use crate::test::{gen_frame, s};

    fn get_object() -> VideoObjectProxy {
        VideoObjectProxy::from(
            VideoObjectBuilder::default()
                .id(1)
                .namespace("model".to_string())
                .label("label".to_string())
                .detection_box(
                    RBBox::new(0.0, 0.0, 1.0, 1.0, None)
                        .try_into()
                        .expect("Failed to convert RBBox to RBBoxData"),
                )
                .confidence(Some(0.5))
                .attributes(
                    vec![
                        Attribute::persistent(
                            "namespace".to_string(),
                            "name".to_string(),
                            vec![AttributeValue::new(
                                AttributeValueVariant::String("value".to_string()),
                                None,
                            )],
                            None,
                        ),
                        Attribute::persistent(
                            "namespace".to_string(),
                            "name2".to_string(),
                            vec![AttributeValue::new(
                                AttributeValueVariant::String("value2".to_string()),
                                None,
                            )],
                            None,
                        ),
                        Attribute::persistent(
                            "namespace2".to_string(),
                            "name".to_string(),
                            vec![AttributeValue::new(
                                AttributeValueVariant::String("value".to_string()),
                                None,
                            )],
                            None,
                        ),
                    ]
                    .into_iter()
                    .map(|a| ((a.get_namespace().into(), a.get_name().into()), a))
                    .collect(),
                )
                .parent_id(None)
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn test_delete_attributes() {
        let obj = get_object();
        obj.delete_attributes(None, vec![]);
        assert_eq!(obj.get_inner_read().attributes.len(), 0);

        let obj = get_object();
        obj.delete_attributes(Some(s("namespace")), vec![]);
        assert_eq!(obj.get_attributes().len(), 1);

        let obj = get_object();
        obj.delete_attributes(None, vec![s("name")]);
        assert_eq!(obj.get_inner_read().attributes.len(), 1);

        let t = get_object();
        t.delete_attributes(None, vec![s("name"), s("name2")]);
        assert_eq!(t.get_inner_read().attributes.len(), 0);
    }

    #[test]
    #[should_panic]
    fn self_parent_assignment_panic_trivial() {
        let obj = get_object();
        obj.set_parent(Some(obj.get_id()));
    }

    #[test]
    #[should_panic]
    fn self_parent_assignment_change_id() {
        let obj = get_object();
        let parent = obj.clone();
        _ = parent.set_id(2);
        obj.set_parent(Some(parent.get_id()));
    }

    #[test]
    #[should_panic(expected = "Frame is dropped, you cannot use attached objects anymore")]
    fn try_access_frame_object_after_frame_drop() {
        let f = gen_frame();
        let o = f.get_object(0).unwrap();
        drop(f);
        let _f = o.get_frame().unwrap();
    }

    #[test]
    #[should_panic(expected = "Only detached objects can be attached to a frame.")]
    fn reassign_object_from_dropped_frame_to_new_frame() {
        let f = gen_frame();
        let o = f.get_object(0).unwrap();
        drop(f);
        let f = gen_frame();
        f.delete_objects_by_ids(&[0]);
        f.add_object(&o, IdCollisionResolutionPolicy::Error)
            .unwrap();
    }

    #[test]
    fn reassign_clean_copy_from_dropped_to_new_frame() {
        let f = gen_frame();
        let o = f.get_object(0).unwrap();
        drop(f);
        let f = gen_frame();
        f.delete_objects_by_ids(&[0]);
        let copy = o.detached_copy();
        assert!(copy.is_detached(), "Clean copy is not attached");
        assert!(!copy.is_spoiled(), "Clean copy must be not spoiled");
        assert!(
            copy.get_parent().is_none(),
            "Clean copy must have no parent"
        );
        f.add_object(&o.detached_copy(), IdCollisionResolutionPolicy::Error)
            .unwrap();
    }

    #[test]
    fn test_transform_geometry() {
        let o = get_object();
        o.set_track_info(13, RBBox::new(0.0, 0.0, 10.0, 20.0, None));
        let ops = vec![VideoObjectBBoxTransformation::Shift(10.0, 20.0)];
        o.transform_geometry(&ops);
        let new_bb = o.get_detection_box();
        assert_eq!(new_bb.get_xc(), 10.0);
        assert_eq!(new_bb.get_yc(), 20.0);
        assert_eq!(new_bb.get_width(), 1.0);
        assert_eq!(new_bb.get_height(), 1.0);

        let new_track_bb = o.get_track_box().unwrap();
        assert_eq!(new_track_bb.get_xc(), 10.0);
        assert_eq!(new_track_bb.get_yc(), 20.0);
        assert_eq!(new_track_bb.get_width(), 10.0);
        assert_eq!(new_track_bb.get_height(), 20.0);

        let ops = vec![VideoObjectBBoxTransformation::Scale(2.0, 4.0)];
        o.transform_geometry(&ops);
        let new_bb = o.get_detection_box();
        assert_eq!(new_bb.get_xc(), 20.0);
        assert_eq!(new_bb.get_yc(), 80.0);
        assert_eq!(new_bb.get_width(), 2.0);
        assert_eq!(new_bb.get_height(), 4.0);

        let new_track_bb = o.get_track_box().unwrap();
        assert_eq!(new_track_bb.get_xc(), 20.0);
        assert_eq!(new_track_bb.get_yc(), 80.0);
        assert_eq!(new_track_bb.get_width(), 20.0);
        assert_eq!(new_track_bb.get_height(), 80.0);
    }
}

use anyhow::bail;
use parking_lot::{RwLockReadGuard, RwLockWriteGuard};
use rkyv::{with::Lock, with::Skip, Archive, Deserialize, Serialize};

use super::bbox::BBOX_UNDEFINED;
use crate::json_api::ToSerdeJsonValue;
use crate::primitives::frame::{BelongingVideoFrame, VideoFrameProxy};
use crate::primitives::{Attribute, AttributeMethods, Attributive, RBBox};
use crate::rwlock::SavantArcRwLock;
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
    derive_builder::Builder,
    serde::Serialize,
    serde::Deserialize,
)]
#[archive(check_bytes)]
pub struct VideoObject {
    pub(crate) id: i64,
    pub(crate) namespace: String,
    pub(crate) label: String,
    #[builder(default)]
    pub(crate) draw_label: Option<String>,
    pub(crate) detection_box: RBBox,
    #[builder(default)]
    pub(crate) attributes: Vec<Attribute>,
    #[builder(default)]
    pub(crate) confidence: Option<f32>,
    #[builder(default)]
    pub(crate) parent_id: Option<i64>,
    #[builder(default)]
    pub(crate) track_box: Option<RBBox>,
    #[builder(default)]
    pub(crate) track_id: Option<i64>,
    #[with(Skip)]
    #[builder(default)]
    pub(crate) namespace_id: Option<i64>,
    #[with(Skip)]
    #[builder(default)]
    pub(crate) label_id: Option<i64>,
    #[with(Skip)]
    #[builder(default)]
    #[serde(skip_deserializing, skip_serializing)]
    pub(crate) frame: Option<BelongingVideoFrame>,
}

impl Clone for VideoObject {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            namespace: self.namespace.clone(),
            label: self.label.clone(),
            draw_label: self.draw_label.clone(),
            detection_box: self.detection_box.copy(),
            attributes: self.attributes.clone(),
            confidence: self.confidence,
            parent_id: self.parent_id,
            track_id: self.track_id,
            track_box: self.track_box.as_ref().map(|tb| tb.copy()),
            namespace_id: self.namespace_id,
            label_id: self.label_id,
            frame: self.frame.clone(),
        }
    }
}

const DEFAULT_ATTRIBUTES_COUNT: usize = 4;

impl Default for VideoObject {
    fn default() -> Self {
        Self {
            id: 0,
            namespace: "".to_string(),
            label: "".to_string(),
            draw_label: None,
            detection_box: BBOX_UNDEFINED.clone(),
            attributes: Vec::with_capacity(DEFAULT_ATTRIBUTES_COUNT),
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
            "bbox": self.detection_box,
            "attributes": self.attributes.iter().filter_map(|v| if v.is_hidden { None } else { Some(v.to_serde_json_value()) }).collect::<Vec<_>>(),
            "confidence": self.confidence,
            "parent": self.parent_id,
            "track_id": self.track_id,
            "track_box": self.track_box,
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
#[derive(Debug, Clone, Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
pub struct VideoObjectProxy(#[with(Lock)] pub(crate) SavantArcRwLock<VideoObject>);

impl ToSerdeJsonValue for VideoObjectProxy {
    fn to_serde_json_value(&self) -> Value {
        trace!(self.0.read_recursive()).to_serde_json_value()
    }
}

impl Attributive for VideoObject {
    fn get_attributes_ref(&self) -> &Vec<Attribute> {
        &self.attributes
    }

    fn get_attributes_ref_mut(&mut self) -> &mut Vec<Attribute> {
        &mut self.attributes
    }
}

impl AttributeMethods for VideoObjectProxy {
    fn exclude_temporary_attributes(&self) -> Vec<Attribute> {
        let mut inner = trace!(self.0.write());
        inner.exclude_temporary_attributes()
    }

    fn restore_attributes(&self, attributes: Vec<Attribute>) {
        let mut inner = trace!(self.0.write());
        inner.restore_attributes(attributes)
    }

    fn get_attributes(&self) -> Vec<(String, String)> {
        let inner = trace!(self.0.read_recursive());
        inner.get_attributes()
    }

    fn get_attribute(&self, namespace: &str, name: &str) -> Option<Attribute> {
        let inner = trace!(self.0.read_recursive());
        inner.get_attribute(namespace, name)
    }

    fn delete_attribute(&self, namespace: &str, name: &str) -> Option<Attribute> {
        let mut inner = trace!(self.0.write());
        inner.delete_attribute(namespace, name)
    }

    fn set_attribute(&self, attribute: Attribute) -> Option<Attribute> {
        let mut inner = trace!(self.0.write());
        inner.set_attribute(attribute)
    }

    fn clear_attributes(&self) {
        let mut inner = trace!(self.0.write());
        inner.clear_attributes()
    }

    fn delete_attributes(&self, namespace: &Option<&str>, names: &[&str]) {
        let mut inner = trace!(self.0.write());
        inner.delete_attributes(namespace, names)
    }

    fn find_attributes(
        &self,
        namespace: &Option<&str>,
        names: &[&str],
        hint: &Option<&str>,
    ) -> Vec<(String, String)> {
        let inner = trace!(self.0.read_recursive());
        inner.find_attributes(namespace, names, hint)
    }
}

impl From<VideoObject> for VideoObjectProxy {
    fn from(value: VideoObject) -> Self {
        Self(SavantArcRwLock::new(value))
    }
}

impl From<SavantArcRwLock<VideoObject>> for VideoObjectProxy {
    fn from(value: SavantArcRwLock<VideoObject>) -> Self {
        Self(value)
    }
}

impl VideoObjectProxy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: i64,
        namespace: &str,
        label: &str,
        detection_box: RBBox,
        attributes: Vec<Attribute>,
        confidence: Option<f32>,
        track_id: Option<i64>,
        track_box: Option<RBBox>,
    ) -> Self {
        let (namespace_id, label_id) =
            get_object_id(namespace, label).map_or((None, None), |(c, o)| (Some(c), Some(o)));

        let object = VideoObject {
            id,
            namespace: namespace.to_string(),
            label: label.to_string(),
            detection_box: detection_box.clone(),
            attributes,
            confidence,
            track_id,
            track_box: track_box.clone(),
            namespace_id,
            label_id,
            ..Default::default()
        };
        Self(SavantArcRwLock::new(object))
    }
    pub fn inner_read_lock(&self) -> RwLockReadGuard<VideoObject> {
        trace!(self.0.read_recursive())
    }

    pub fn inner_write_lock(&self) -> RwLockWriteGuard<VideoObject> {
        trace!(self.0.write())
    }
    pub fn get_parent_id(&self) -> Option<i64> {
        let inner = trace!(self.inner_read_lock());
        inner.parent_id
    }
    pub fn get_inner(&self) -> SavantArcRwLock<VideoObject> {
        self.0.clone()
    }

    pub(crate) fn set_parent(&self, parent_opt: Option<i64>) -> anyhow::Result<()> {
        if let Some(parent) = parent_opt {
            if self.get_frame().is_none() {
                bail!("Cannot set parent to the object detached from a frame");
            }
            if self.get_id() == parent {
                bail!("Cannot set parent to itself");
            }
            let owning_frame =
                self.get_frame()
                    .ok_or(anyhow::anyhow!(
                "The object {:?} is not assigned to a frame, you cannot assign parent to it", self
            ))?;

            if !owning_frame.object_exists(parent) {
                bail!("Cannot set parent to the object which cannot be found in the frame");
            }

            // detect loops
            let mut id_chain = vec![self.get_id(), parent];
            loop {
                let parent_obj = owning_frame.get_object(*id_chain.last().unwrap()).unwrap();
                let his_parent_opt = parent_obj.get_parent_id();
                if let Some(his_parent) = his_parent_opt {
                    if id_chain.contains(&his_parent) {
                        bail!(
                            "A parent-Child Loop detected. Caused by setting a parent with ID={} to an object with ID={}, Loop goes through IDs: {:?}",
                            parent,
                            self.get_id(),
                            id_chain
                        );
                    }
                    id_chain.push(his_parent);
                } else {
                    break;
                }
            }
        }

        let mut inner = trace!(self.0.write());
        inner.parent_id = parent_opt;
        Ok(())
    }

    pub fn get_parent(&self) -> Option<VideoObjectProxy> {
        let frame = self.get_frame();
        let id = trace!(self.inner_read_lock()).parent_id?;
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
        let mut inner = trace!(self.0.write());
        inner.frame = Some(frame.into());
    }

    pub fn transform_geometry(&self, ops: &Vec<VideoObjectBBoxTransformation>) {
        for o in ops {
            match o {
                VideoObjectBBoxTransformation::Scale(kx, ky) => {
                    self.get_detection_box().scale(*kx, *ky);
                    if let Some(t) = self.get_track_box() {
                        t.scale(*kx, *ky);
                    }
                }
                VideoObjectBBoxTransformation::Shift(dx, dy) => {
                    self.get_detection_box().shift(*dx, *dy);
                    if let Some(t) = self.get_track_box() {
                        t.shift(*dx, *dy);
                    }
                }
            }
        }
    }
    pub fn get_track_id(&self) -> Option<i64> {
        let inner = trace!(self.inner_read_lock());
        inner.track_id
    }

    pub fn get_detection_box(&self) -> RBBox {
        let inner = trace!(self.inner_read_lock());
        inner.detection_box.clone()
    }

    pub fn get_track_box(&self) -> Option<RBBox> {
        let inner = trace!(self.inner_read_lock());
        inner.track_box.clone()
    }

    pub fn get_confidence(&self) -> Option<f32> {
        let inner = trace!(self.inner_read_lock());
        inner.confidence
    }

    pub fn get_namespace(&self) -> String {
        trace!(self.inner_read_lock()).namespace.clone()
    }

    pub fn get_namespace_id(&self) -> Option<i64> {
        let inner = trace!(self.inner_read_lock());
        inner.namespace_id
    }

    pub fn get_label(&self) -> String {
        trace!(self.inner_read_lock()).label.clone()
    }

    pub fn get_label_id(&self) -> Option<i64> {
        let inner = trace!(self.inner_read_lock());
        inner.label_id
    }

    pub fn detached_copy(&self) -> Self {
        let inner = trace!(self.inner_read_lock());
        let mut new_inner = inner.clone();
        new_inner.parent_id = None;
        new_inner.frame = None;
        Self(SavantArcRwLock::new(new_inner))
    }

    pub fn get_draw_label(&self) -> Option<String> {
        let inner = trace!(self.inner_read_lock());
        inner.draw_label.clone()
    }

    pub fn calculate_draw_label(&self) -> String {
        let inner = trace!(self.inner_read_lock());
        inner.draw_label.as_ref().unwrap_or(&inner.label).clone()
    }

    pub fn get_frame(&self) -> Option<VideoFrameProxy> {
        let inner = trace!(self.inner_read_lock());
        inner.frame.as_ref().map(|f| f.into())
    }

    pub fn get_id(&self) -> i64 {
        trace!(self.inner_read_lock()).id
    }

    pub fn is_detached(&self) -> bool {
        let inner = trace!(self.inner_read_lock());
        inner.frame.is_none()
    }

    pub fn is_spoiled(&self) -> bool {
        let inner = trace!(self.inner_read_lock());
        match inner.frame {
            Some(ref f) => f.inner.upgrade().is_none(),
            None => false,
        }
    }
    pub fn set_detection_box(&self, bbox: RBBox) {
        let mut inner = trace!(self.inner_write_lock());
        inner.detection_box = bbox;
    }

    pub fn set_track_info(&self, track_id: i64, bbox: RBBox) {
        let mut inner = trace!(self.inner_write_lock());
        inner.track_box = Some(bbox);
        inner.track_id = Some(track_id);
    }

    pub fn set_track_box(&self, bbox: RBBox) {
        let mut inner = trace!(self.inner_write_lock());
        inner.track_box = Some(bbox);
    }

    pub fn clear_track_info(&self) {
        let mut inner = trace!(self.inner_write_lock());
        inner.track_box = None;
        inner.track_id = None;
    }

    pub fn set_draw_label(&self, draw_label: Option<String>) {
        let mut inner = trace!(self.inner_write_lock());
        inner.draw_label = draw_label;
    }

    pub fn set_id(&self, id: i64) -> anyhow::Result<()> {
        if self.get_frame().is_some() {
            bail!("When object is attached to a frame, it is impossible to change its ID",);
        }

        let mut inner = trace!(self.inner_write_lock());
        inner.id = id;
        Ok(())
    }

    pub fn set_namespace(&self, namespace: &str) {
        let mut inner = trace!(self.inner_write_lock());
        inner.namespace = namespace.to_string();
    }

    pub fn set_label(&self, label: &str) {
        let mut inner = trace!(self.inner_write_lock());
        inner.label = label.to_string();
    }

    pub fn set_confidence(&self, confidence: Option<f32>) {
        let mut inner = trace!(self.inner_write_lock());
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
    use crate::test::{gen_empty_frame, gen_frame};

    fn get_object(id: i64) -> VideoObjectProxy {
        VideoObjectProxy::from(
            VideoObjectBuilder::default()
                .id(id)
                .namespace("model".to_string())
                .label("label".to_string())
                .detection_box(
                    RBBox::new(0.0, 0.0, 1.0, 1.0, None)
                        .try_into()
                        .expect("Failed to convert RBBox to RBBoxData"),
                )
                .confidence(Some(0.5))
                .attributes(vec![
                    Attribute::persistent(
                        "namespace".to_string(),
                        "name".to_string(),
                        vec![AttributeValue::new(
                            AttributeValueVariant::String("value".to_string()),
                            None,
                        )],
                        None,
                        false,
                    ),
                    Attribute::persistent(
                        "namespace".to_string(),
                        "name2".to_string(),
                        vec![AttributeValue::new(
                            AttributeValueVariant::String("value2".to_string()),
                            None,
                        )],
                        None,
                        false,
                    ),
                    Attribute::persistent(
                        "namespace2".to_string(),
                        "name".to_string(),
                        vec![AttributeValue::new(
                            AttributeValueVariant::String("value".to_string()),
                            None,
                        )],
                        None,
                        false,
                    ),
                ])
                .parent_id(None)
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn test_delete_attributes() {
        let obj = get_object(1);
        obj.delete_attributes(&None, &[]);
        assert_eq!(obj.inner_read_lock().attributes.len(), 0);

        let obj = get_object(1);
        obj.delete_attributes(&Some("namespace"), &[]);
        assert_eq!(obj.get_attributes().len(), 1);

        let obj = get_object(1);
        obj.delete_attributes(&None, &["name"]);
        assert_eq!(obj.inner_read_lock().attributes.len(), 1);

        let t = get_object(1);
        t.delete_attributes(&None, &["name", "name2"]);
        assert_eq!(t.inner_read_lock().attributes.len(), 0);
    }

    #[test]
    fn test_loop_2() {
        let f = gen_empty_frame();
        let o1 = get_object(1);
        f.add_object(&o1, IdCollisionResolutionPolicy::Error)
            .unwrap();
        let o2 = get_object(2);
        f.add_object(&o2, IdCollisionResolutionPolicy::Error)
            .unwrap();
        o1.set_parent(Some(o2.get_id())).unwrap();
        assert!(o2.set_parent(Some(o1.get_id())).is_err());
    }

    #[test]
    fn test_loop_3() {
        let f = gen_empty_frame();
        let o1 = get_object(1);
        f.add_object(&o1, IdCollisionResolutionPolicy::Error)
            .unwrap();
        let o2 = get_object(2);
        f.add_object(&o2, IdCollisionResolutionPolicy::Error)
            .unwrap();
        let o3 = get_object(3);
        f.add_object(&o3, IdCollisionResolutionPolicy::Error)
            .unwrap();
        o1.set_parent(Some(o2.get_id())).unwrap();
        o2.set_parent(Some(o3.get_id())).unwrap();
        assert!(o3.set_parent(Some(o1.get_id())).is_err());
    }

    #[test]
    fn self_parent_assignment_trivial() {
        let obj = get_object(1);
        assert!(obj.set_parent(Some(obj.get_id())).is_err());
    }

    #[test]
    fn self_parent_assignment_change_id() {
        let obj = get_object(1);
        let parent = obj.clone();
        _ = parent.set_id(2);
        assert!(obj.set_parent(Some(parent.get_id())).is_err());
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
        let o = get_object(1);
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

pub mod object_tree;

use anyhow::bail;
use serde_json::Value;
use std::fmt::Debug;

use crate::json_api::ToSerdeJsonValue;
use crate::primitives::frame::{BelongingVideoFrame, VideoFrameProxy};
use crate::primitives::object::private::{
    SealedObjectOperations, SealedWithFrame, SealedWithParent,
};
use crate::primitives::{Attribute, RBBox, WithAttributes};

use super::bbox::BBOX_UNDEFINED;

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

pub trait WithId {
    fn get_id(&self) -> i64;
    fn set_id(&mut self, id: i64);
}

impl WithId for VideoObject {
    fn get_id(&self) -> i64 {
        self.id
    }
    fn set_id(&mut self, id: i64) {
        self.id = id;
    }
}

#[derive(Debug, derive_builder::Builder, serde::Serialize, serde::Deserialize)]
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
    #[builder(default)]
    pub(crate) namespace_id: Option<i64>,
    #[builder(default)]
    pub(crate) label_id: Option<i64>,
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

impl VideoObject {
    pub fn set_id(&mut self, id: i64) -> anyhow::Result<()> {
        if self.get_frame().is_some() {
            bail!("When object is attached to a frame, it is impossible to change its ID",);
        }
        self.with_object_mut(|o| o.id = id);
        Ok(())
    }
}

impl ToSerdeJsonValue for VideoObject {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(self)
    }
}

/// Represents a video object. The object is a part of a video frame, it includes bounding
/// box, attributes, label, namespace label, etc. The objects are always accessible by reference. The only way to
/// copy the object by value is to call :py:meth:`VideoObject.detached_copy`.
///
/// :py:class:`VideoObject` is a part of :py:class:`VideoFrame` and may outlive it if there are references.
///
#[derive(Debug, Clone)]
pub struct BorrowedVideoObject(pub(crate) BelongingVideoFrame, pub(crate) i64);

impl ToSerdeJsonValue for BorrowedVideoObject {
    fn to_serde_json_value(&self) -> Value {
        self.with_object_ref(|o| o.to_serde_json_value())
    }
}

impl WithAttributes for VideoObject {
    fn with_attributes_ref<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Vec<Attribute>) -> R,
    {
        f(&self.attributes)
    }

    fn with_attributes_mut<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Vec<Attribute>) -> R,
    {
        f(&mut self.attributes)
    }
}

impl WithAttributes for BorrowedVideoObject {
    fn with_attributes_ref<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Vec<Attribute>) -> R,
    {
        let frame = <&BelongingVideoFrame as Into<VideoFrameProxy>>::into(&self.0);
        let frame = frame.inner.0.read_recursive();
        let object = frame
            .objects
            .get(&self.1)
            .unwrap_or_else(|| panic!("Object {} not found in the frame {}", self.1, frame.uuid));
        f(&object.attributes)
    }

    fn with_attributes_mut<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Vec<Attribute>) -> R,
    {
        let frame = <&BelongingVideoFrame as Into<VideoFrameProxy>>::into(&self.0);
        let mut frame = frame.inner.0.write();
        let uuid = frame.uuid;
        let object = frame
            .objects
            .get_mut(&self.1)
            .unwrap_or_else(|| panic!("Object {} not found in the frame {}", self.1, uuid));
        f(&mut object.attributes)
    }
}

pub(crate) mod private {
    use crate::primitives::frame::VideoFrameProxy;
    use crate::primitives::object::{BorrowedVideoObject, ObjectAccess, ObjectOperations};
    use anyhow::bail;

    pub trait SealedWithFrame: ObjectAccess
    where
        Self: Sized,
    {
        fn get_frame(&self) -> Option<VideoFrameProxy> {
            self.with_object_ref(|o| o.frame.as_ref().map(|f| f.into()))
        }
    }

    pub trait SealedObjectOperations
    where
        Self: Sized + ObjectAccess + SealedWithFrame,
    {
        fn attach_to_video_frame(&mut self, frame: VideoFrameProxy) {
            self.with_object_mut(|o| o.frame = Some(frame.into()));
        }
    }

    pub trait SealedWithParent: SealedWithFrame + ObjectOperations {
        fn get_parent(&self) -> Option<BorrowedVideoObject> {
            let frame = self.get_frame();
            let id = self.get_parent_id()?;
            match frame {
                Some(f) => f.get_object(id),
                None => None,
            }
        }
        fn set_parent(&mut self, parent_opt: Option<i64>) -> anyhow::Result<()> {
            if let Some(parent) = parent_opt {
                if self.get_frame().is_none() {
                    bail!("Cannot set parent to the object detached from a frame");
                }
                if self.get_id() == parent {
                    bail!("Cannot set parent to itself");
                }
                let owning_frame =
                    self.get_frame().ok_or(anyhow::anyhow!(
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
            self.with_object_mut(|o| o.parent_id = parent_opt);
            Ok(())
        }

        fn get_children(&self) -> Vec<BorrowedVideoObject> {
            let frame = self.get_frame();
            let id = self.get_id();
            match frame {
                Some(f) => f.get_children(id),
                None => Vec::new(),
            }
        }
    }
}

impl SealedObjectOperations for VideoObject {}
impl SealedObjectOperations for BorrowedVideoObject {}

impl SealedWithFrame for VideoObject {}
impl SealedWithFrame for BorrowedVideoObject {}

impl SealedWithParent for VideoObject {}
impl SealedWithParent for BorrowedVideoObject {}

pub trait ObjectAccess
where
    Self: Sized + Debug + Clone,
{
    fn with_object_ref<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&VideoObject) -> R;

    fn with_object_mut<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut VideoObject) -> R;
}

pub trait ObjectOperations: ObjectAccess
where
    Self: Sized + Debug + Clone,
{
    fn get_parent_id(&self) -> Option<i64> {
        self.with_object_ref(|o| o.parent_id)
    }

    fn get_id(&self) -> i64 {
        self.with_object_ref(|o| o.id)
    }

    fn transform_geometry(&mut self, ops: &Vec<VideoObjectBBoxTransformation>) {
        self.with_object_mut(|object| {
            for o in ops {
                match o {
                    VideoObjectBBoxTransformation::Scale(kx, ky) => {
                        object.get_detection_box().scale(*kx, *ky);
                        if let Some(t) = object.get_track_box() {
                            t.scale(*kx, *ky);
                        }
                    }
                    VideoObjectBBoxTransformation::Shift(dx, dy) => {
                        object.get_detection_box().shift(*dx, *dy);
                        if let Some(t) = object.get_track_box() {
                            t.shift(*dx, *dy);
                        }
                    }
                }
            }
        })
    }
    fn get_track_id(&self) -> Option<i64> {
        self.with_object_ref(|o| o.track_id)
    }

    fn set_track_id(&mut self, track_id: Option<i64>) {
        self.with_object_mut(|o| o.track_id = track_id);
    }

    fn get_detection_box(&self) -> RBBox {
        self.with_object_ref(|o| o.detection_box.clone())
    }

    fn get_track_box(&self) -> Option<RBBox> {
        self.with_object_ref(|o| o.track_box.clone())
    }

    fn get_confidence(&self) -> Option<f32> {
        self.with_object_ref(|o| o.confidence)
    }

    fn get_namespace(&self) -> String {
        self.with_object_ref(|o| o.namespace.clone())
    }

    fn get_namespace_id(&self) -> Option<i64> {
        self.with_object_ref(|o| o.namespace_id)
    }

    fn get_label(&self) -> String {
        self.with_object_ref(|o| o.label.clone())
    }

    fn get_label_id(&self) -> Option<i64> {
        self.with_object_ref(|o| o.label_id)
    }

    fn get_draw_label(&self) -> Option<String> {
        self.with_object_ref(|o| o.draw_label.clone())
    }

    fn calculate_draw_label(&self) -> String {
        self.with_object_ref(|o| {
            o.draw_label
                .as_ref()
                .unwrap_or(&o.label)
                .to_string()
                .clone()
        })
    }

    fn set_detection_box(&mut self, bbox: RBBox) {
        self.with_object_mut(|o| o.detection_box = bbox);
    }

    fn set_track_info(&mut self, track_id: i64, bbox: RBBox) {
        self.with_object_mut(|o| {
            o.track_box = Some(bbox);
            o.track_id = Some(track_id);
        });
    }

    fn set_track_box(&mut self, bbox: RBBox) {
        self.with_object_mut(|o| o.track_box = Some(bbox));
    }

    fn clear_track_info(&mut self) {
        self.with_object_mut(|o| {
            o.track_box = None;
            o.track_id = None;
        });
    }

    fn set_draw_label(&mut self, draw_label: Option<String>) {
        self.with_object_mut(|o| o.draw_label = draw_label);
    }

    fn set_namespace(&mut self, namespace: &str) {
        self.with_object_mut(|o| o.namespace = namespace.to_string());
    }

    fn set_label(&mut self, label: &str) {
        self.with_object_mut(|o| o.label = label.to_string());
    }

    fn set_confidence(&mut self, confidence: Option<f32>) {
        self.with_object_mut(|o| o.confidence = confidence);
    }

    fn detached_copy(&self) -> VideoObject {
        self.with_object_ref(|o| {
            let mut copy = o.clone();
            copy.parent_id = None;
            copy.frame = None;
            copy
        })
    }
}

impl ObjectOperations for VideoObject {}
impl ObjectOperations for BorrowedVideoObject {}

impl ObjectAccess for VideoObject {
    fn with_object_ref<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&VideoObject) -> R,
    {
        f(self)
    }

    fn with_object_mut<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut VideoObject) -> R,
    {
        f(self)
    }
}

impl ObjectAccess for BorrowedVideoObject {
    fn with_object_ref<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&VideoObject) -> R,
    {
        let frame = <&BelongingVideoFrame as Into<VideoFrameProxy>>::into(&self.0);
        let frame = frame.inner.read_recursive();
        let object = frame
            .objects
            .get(&self.1)
            .unwrap_or_else(|| panic!("Object {} not found in the frame {}", self.1, frame.uuid));
        f(object)
    }

    fn with_object_mut<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut VideoObject) -> R,
    {
        let frame = <&BelongingVideoFrame as Into<VideoFrameProxy>>::into(&self.0);
        let mut frame = frame.inner.0.write();
        let uuid = frame.uuid;
        let object = frame
            .objects
            .get_mut(&self.1)
            .unwrap_or_else(|| panic!("Object {} not found in the frame {}", self.1, uuid));
        f(object)
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute_value::AttributeValue;
    use crate::primitives::object::private::{SealedObjectOperations, SealedWithParent};
    use crate::primitives::object::{
        IdCollisionResolutionPolicy, ObjectOperations, VideoObject, VideoObjectBBoxTransformation,
        VideoObjectBuilder,
    };
    use crate::primitives::{Attribute, RBBox};
    use crate::test::{gen_empty_frame, gen_frame};

    fn generate_object(id: i64) -> VideoObject {
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
                    "namespace",
                    "name",
                    vec![AttributeValue::string("value", None)],
                    &None,
                    false,
                ),
                Attribute::persistent(
                    "namespace",
                    "name2",
                    vec![AttributeValue::string("value2", None)],
                    &None,
                    false,
                ),
                Attribute::persistent(
                    "namespace2",
                    "name",
                    vec![AttributeValue::string("value", None)],
                    &None,
                    false,
                ),
            ])
            .parent_id(None)
            .build()
            .unwrap()
    }

    #[test]
    fn test_id_change_prohibited_for_attached_objects() {
        let mut obj = generate_object(1);
        let f = gen_empty_frame();
        obj.attach_to_video_frame(f.clone());
        assert!(obj.set_id(2).is_err());
    }

    #[test]
    fn test_loop_2() {
        let f = gen_empty_frame();
        let o1 = generate_object(1);
        let mut o1 = f
            .add_object(o1, IdCollisionResolutionPolicy::Error)
            .unwrap();
        let o2 = generate_object(2);
        let mut o2 = f
            .add_object(o2, IdCollisionResolutionPolicy::Error)
            .unwrap();
        o1.set_parent(Some(o2.get_id())).unwrap();
        assert!(o2.set_parent(Some(o1.get_id())).is_err());
    }

    #[test]
    fn test_loop_3() {
        let f = gen_empty_frame();
        let o1 = generate_object(1);
        let mut o1 = f
            .add_object(o1, IdCollisionResolutionPolicy::Error)
            .unwrap();
        let o2 = generate_object(2);
        let mut o2 = f
            .add_object(o2, IdCollisionResolutionPolicy::Error)
            .unwrap();
        let o3 = generate_object(3);
        let mut o3 = f
            .add_object(o3, IdCollisionResolutionPolicy::Error)
            .unwrap();
        o1.set_parent(Some(o2.get_id())).unwrap();
        o2.set_parent(Some(o3.get_id())).unwrap();
        assert!(o3.set_parent(Some(o1.get_id())).is_err());
    }

    #[test]
    fn test_frameless_parent_assignment() {
        let mut obj = generate_object(1);
        assert!(obj.set_parent(Some(2)).is_err());
    }

    #[test]
    fn self_parent_assignment_trivial() {
        let mut obj = generate_object(1);
        assert!(obj.set_parent(Some(obj.get_id())).is_err());
    }

    #[test]
    fn self_parent_assignment_change_id() {
        let mut obj = generate_object(1);
        let mut parent = obj.clone();
        _ = parent.set_id(2);
        assert!(obj.set_parent(Some(parent.get_id())).is_err());
    }

    #[test]
    fn reassign_clean_copy_from_dropped_to_new_frame() {
        let f = gen_frame();
        let o = f.get_object(1).unwrap();
        let copy = o.detached_copy();
        drop(f);
        let f = gen_frame();
        f.delete_objects_with_ids(&[1]);
        assert!(
            copy.get_parent().is_none(),
            "Clean copy must have no parent"
        );
        f.add_object(copy, IdCollisionResolutionPolicy::Error)
            .unwrap();
    }

    #[test]
    fn test_transform_geometry() {
        let mut o = generate_object(1);
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

use crate::draw::DrawLabelKind;
use crate::json_api::ToSerdeJsonValue;
use crate::match_query::{and, IntExpression, MatchQuery, StringExpression};
use crate::message::Message;
use crate::primitives::frame_update::VideoFrameUpdate;
use crate::primitives::object::private::{
    SealedObjectOperations, SealedWithFrame, SealedWithParent,
};
use crate::primitives::object::{
    BorrowedVideoObject, IdCollisionResolutionPolicy, ObjectAccess, ObjectOperations, VideoObject,
    VideoObjectBBoxTransformation, VideoObjectBuilder,
};
use crate::primitives::{Attribute, RBBox, WithAttributes};
use crate::rwlock::{SavantArcRwLock, SavantRwLock};
use crate::trace;
use crate::utils::iter::fiter_map_with_control_flow;
use crate::utils::uuid_v7::incremental_uuid_v7;
use crate::version;
use anyhow::{anyhow, bail};
use derive_builder::Builder;
use hashbrown::{HashMap, HashSet};
use serde_json::Value;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::mem;
use std::sync::{Arc, Weak};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use super::object::object_tree::ObjectTree;

pub type VideoObjectTree = ObjectTree<VideoObject>;

#[derive(Debug, Hash)]
struct StreamCompatibilityInformation<'a> {
    pub source_id: &'a str,
    pub codec: &'a Option<String>,
    pub width: i64,
    pub height: i64,
}

impl<'a> StreamCompatibilityInformation<'a> {
    pub fn new(source_id: &'a str, codec: &'a Option<String>, width: i64, height: i64) -> Self {
        Self {
            source_id,
            codec,
            width,
            height,
        }
    }
}

#[derive(Debug, PartialEq, Clone, serde::Serialize)]
pub struct ExternalFrame {
    pub method: String,
    pub location: Option<String>,
}

impl ToSerdeJsonValue for ExternalFrame {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(self)
    }
}

impl ExternalFrame {
    pub fn new(method: &str, location: &Option<&str>) -> Self {
        Self {
            method: method.to_string(),
            location: location.map(String::from),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum VideoFrameContent {
    External(ExternalFrame),
    Internal(Vec<u8>),
    None,
}

impl ToSerdeJsonValue for VideoFrameContent {
    fn to_serde_json_value(&self) -> Value {
        match self {
            VideoFrameContent::External(data) => {
                serde_json::json!({"external": data.to_serde_json_value()})
            }
            VideoFrameContent::Internal(_) => {
                serde_json::json!({ "internal": Value::String("<blob-omitted>".to_string()) })
            }
            VideoFrameContent::None => Value::Null,
        }
    }
}

#[derive(Debug, PartialEq, Clone, serde::Serialize)]
pub enum VideoFrameTranscodingMethod {
    Copy,
    Encoded,
}

impl ToSerdeJsonValue for VideoFrameTranscodingMethod {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(self)
    }
}

#[derive(Debug, PartialEq, Clone, serde::Serialize)]
pub enum VideoFrameTransformation {
    InitialSize(u64, u64),
    Scale(u64, u64),
    Padding(u64, u64, u64, u64),
    ResultingSize(u64, u64),
}

impl ToSerdeJsonValue for VideoFrameTransformation {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(self)
    }
}

#[derive(Debug, Clone, Builder)]
pub struct VideoFrame {
    #[builder(setter(skip))]
    pub previous_frame_seq_id: Option<i64>,
    #[builder(setter(skip))]
    pub previous_keyframe: Option<u128>,
    pub source_id: String,
    pub uuid: u128,
    #[builder(setter(skip))]
    pub creation_timestamp_ns: u128,
    pub framerate: String,
    pub width: i64,
    pub height: i64,
    pub transcoding_method: VideoFrameTranscodingMethod,
    pub codec: Option<String>,
    pub keyframe: Option<bool>,
    #[builder(setter(skip))]
    pub time_base: (i32, i32),
    pub pts: i64,
    #[builder(setter(skip))]
    pub dts: Option<i64>,
    #[builder(setter(skip))]
    pub duration: Option<i64>,
    pub content: Arc<VideoFrameContent>,
    #[builder(setter(skip))]
    pub transformations: Vec<VideoFrameTransformation>,
    #[builder(setter(skip))]
    pub attributes: Vec<Attribute>,
    #[builder(setter(skip))]
    pub(crate) objects: HashMap<i64, VideoObject>,
    #[builder(setter(skip))]
    pub(crate) max_object_id: i64,
}

const DEFAULT_TRANSFORMATIONS_COUNT: usize = 4;
const DEFAULT_ATTRIBUTES_COUNT: usize = 8;
const DEFAULT_OBJECTS_COUNT: usize = 64;

impl Default for VideoFrame {
    fn default() -> Self {
        Self {
            previous_frame_seq_id: None,
            previous_keyframe: None,
            source_id: String::new(),
            uuid: incremental_uuid_v7().as_u128(),
            creation_timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            framerate: String::new(),
            width: 0,
            height: 0,
            transcoding_method: VideoFrameTranscodingMethod::Copy,
            codec: None,
            keyframe: None,
            time_base: (1, 1000000),
            pts: 0,
            dts: None,
            duration: None,
            content: Arc::new(VideoFrameContent::None),
            transformations: Vec::with_capacity(DEFAULT_TRANSFORMATIONS_COUNT),
            attributes: Vec::with_capacity(DEFAULT_ATTRIBUTES_COUNT),
            objects: HashMap::with_capacity(DEFAULT_OBJECTS_COUNT),
            max_object_id: 0,
        }
    }
}

impl ToSerdeJsonValue for VideoFrame {
    fn to_serde_json_value(&self) -> Value {
        let frame_uuid = Uuid::from_u128(self.uuid).to_string();
        let previous_keyframe = self
            .previous_keyframe
            .map(|v| Uuid::from_u128(v).to_string());

        let version = version();
        let mut objects = self.objects.values().collect::<Vec<_>>();
        objects.sort_by(|a, b| a.id.cmp(&b.id));
        let objects = objects
            .iter()
            .map(|o| o.to_serde_json_value())
            .collect::<Vec<_>>();
        serde_json::json!(
            {
                "previous_frame_seq_id": self.previous_frame_seq_id,
                "previous_keyframe": previous_keyframe,
                "version": version,
                "uuid": frame_uuid,
                "creation_timestamp_ns": if self.creation_timestamp_ns > 2^53 { 2^53 } else { self.creation_timestamp_ns },
                "type": "VideoFrame",
                "source_id": self.source_id,
                "framerate": self.framerate,
                "width": self.width,
                "height": self.height,
                "transcoding_method": self.transcoding_method.to_serde_json_value(),
                "codec": self.codec,
                "keyframe": self.keyframe,
                "time_base": self.time_base,
                "pts": self.pts,
                "dts": self.dts,
                "duration": self.duration,
                "content": self.content.to_serde_json_value(),
                "transformations": self.transformations.iter().map(|t| t.to_serde_json_value()).collect::<Vec<_>>(),
                "attributes": self.attributes.iter().filter_map(|v| if v.is_hidden { None } else { Some(v.to_serde_json_value()) }).collect::<Vec<_>>(),
                "objects": objects,
            }
        )
    }
}

impl WithAttributes for VideoFrame {
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

impl VideoFrame {
    pub fn stream_compatibility_hash(&self) -> u64 {
        let compatibility_info = StreamCompatibilityInformation::new(
            &self.source_id,
            &self.codec,
            self.width,
            self.height,
        );
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        compatibility_info.hash(&mut hasher);
        hasher.finish()
    }

    pub fn get_objects(&self) -> &HashMap<i64, VideoObject> {
        &self.objects
    }

    pub fn get_objects_mut(&mut self) -> &mut HashMap<i64, VideoObject> {
        &mut self.objects
    }

    pub fn smart_copy(&self) -> Self {
        let mut frame = self.clone();
        frame.objects.clear();
        for (id, o) in self.get_objects() {
            let mut copy = o.detached_copy();
            copy.parent_id = o.get_parent_id();
            frame.objects.insert(*id, copy);
        }
        frame
    }

    pub fn exclude_all_temporary_attributes(&mut self) {
        self.exclude_temporary_attributes();
        self.objects.values_mut().for_each(|o| {
            o.exclude_temporary_attributes();
        });
    }

    pub fn restore_all_temporary_attributes(
        &mut self,
        frame_attributes: Vec<Attribute>,
        object_attributes: HashMap<i64, Vec<Attribute>>,
    ) {
        self.restore_attributes(frame_attributes);
        for (id, attrs) in object_attributes {
            let o = self.objects.get_mut(&id).unwrap();
            o.restore_attributes(attrs);
        }
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct VideoFrameProxy {
    pub(crate) inner: SavantArcRwLock<Box<VideoFrame>>,
}

#[derive(Clone)]
pub struct BelongingVideoFrame {
    pub(crate) inner: Weak<SavantRwLock<Box<VideoFrame>>>,
}

impl Debug for BelongingVideoFrame {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.inner.upgrade() {
            Some(inner) => f
                .debug_struct("BelongingVideoFrame")
                .field("stream_id", &trace!(inner.read_recursive()).source_id)
                .finish(),
            None => f.debug_struct("Unset").finish(),
        }
    }
}

impl From<&VideoFrameProxy> for BelongingVideoFrame {
    fn from(value: &VideoFrameProxy) -> Self {
        Self {
            inner: Arc::downgrade(&value.inner.0),
        }
    }
}

impl From<VideoFrameProxy> for BelongingVideoFrame {
    fn from(value: VideoFrameProxy) -> Self {
        (&value).into()
    }
}

impl From<&BelongingVideoFrame> for VideoFrameProxy {
    fn from(value: &BelongingVideoFrame) -> Self {
        Self {
            inner: value
                .inner
                .upgrade()
                .expect("Frame is dropped, you cannot use attached objects anymore")
                .into(),
        }
    }
}

impl From<BelongingVideoFrame> for VideoFrameProxy {
    fn from(value: BelongingVideoFrame) -> Self {
        (&value).into()
    }
}

impl WithAttributes for VideoFrameProxy {
    fn with_attributes_ref<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Vec<Attribute>) -> R,
    {
        let bind = trace!(self.inner.read_recursive());
        f(&bind.attributes)
    }

    fn with_attributes_mut<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Vec<Attribute>) -> R,
    {
        let mut bind = trace!(self.inner.write());
        f(&mut bind.attributes)
    }
}

impl ToSerdeJsonValue for VideoFrameProxy {
    fn to_serde_json_value(&self) -> Value {
        let inner = trace!(self.inner.read_recursive()).clone();
        inner.to_serde_json_value()
    }
}

impl VideoFrameProxy {
    pub fn stream_compatibility_hash(&self) -> u64 {
        let inner = trace!(self.inner.read_recursive());
        inner.stream_compatibility_hash()
    }
    pub fn exclude_all_temporary_attributes(&self) {
        let mut inner = trace!(self.inner.write());
        inner.exclude_all_temporary_attributes()
    }

    pub(crate) fn get_object_count(&self) -> usize {
        let inner = trace!(self.inner.read_recursive());
        inner.objects.len()
    }

    pub fn memory_handle(&self) -> usize {
        self as *const Self as usize
    }

    pub fn transform_geometry(&self, ops: &Vec<VideoObjectBBoxTransformation>) {
        let objs = self.get_all_objects();
        for mut obj in objs {
            obj.transform_geometry(ops);
        }
    }

    pub fn smart_copy(&self) -> Self {
        let inner = trace!(self.inner.read());
        let inner_copy = inner.smart_copy();
        drop(inner);
        Self::from_inner(inner_copy)
    }

    pub fn prepare_after_load(&self) {
        let objects = self.get_all_objects();
        for mut o in objects {
            o.attach_to_video_frame(self.clone());
        }
    }

    pub(crate) fn get_inner(&self) -> SavantArcRwLock<Box<VideoFrame>> {
        self.inner.clone()
    }

    pub(crate) fn from_inner(inner: VideoFrame) -> Self {
        let res = VideoFrameProxy {
            inner: SavantArcRwLock::from(Arc::new(SavantRwLock::new(Box::new(inner)))),
        };
        res.fix_object_owned_frame();
        res
    }

    pub fn get_all_objects(&self) -> Vec<BorrowedVideoObject> {
        let inner = trace!(self.inner.read_recursive());
        inner
            .objects
            .values()
            .map(|o| BorrowedVideoObject(self.into(), o.get_id()))
            .collect()
    }

    pub fn has_objects(&self) -> bool {
        let inner = trace!(self.inner.read_recursive());
        !inner.objects.is_empty()
    }

    pub fn access_objects(&self, q: &MatchQuery) -> Vec<BorrowedVideoObject> {
        let inner = trace!(self.inner.read_recursive());
        let objects = inner.objects.values().cloned().collect::<Vec<_>>();
        drop(inner);
        fiter_map_with_control_flow(objects, |o| q.execute_with_new_context(o))
            .iter()
            .map(|o| BorrowedVideoObject(self.into(), o.get_id()))
            .collect()
    }

    pub fn get_json(&self) -> String {
        serde_json::to_string(&self.to_serde_json_value()).unwrap()
    }

    pub fn get_json_pretty(&self) -> String {
        serde_json::to_string_pretty(&self.to_serde_json_value()).unwrap()
    }

    pub fn access_objects_with_id(&self, ids: &[i64]) -> Vec<BorrowedVideoObject> {
        let inner = trace!(self.inner.read_recursive());
        let resident_objects = inner.objects.clone();
        drop(inner);

        ids.iter()
            .filter_map(|id| {
                if resident_objects.contains_key(id) {
                    Some(BorrowedVideoObject(self.into(), *id))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn delete_objects_with_ids(&self, ids: &[i64]) -> Vec<VideoObject> {
        let mut inner = trace!(self.inner.write());
        let objects = mem::take(&mut inner.objects);
        let (mut retained, removed): (HashMap<i64, VideoObject>, HashMap<i64, VideoObject>) =
            objects.into_iter().partition(|(id, _)| !ids.contains(id));

        retained.iter_mut().for_each(|(_, o)| {
            if let Some(parent_id) = o.parent_id {
                if removed.contains_key(&parent_id) {
                    o.parent_id = None;
                }
            }
        });
        inner.objects = retained;
        drop(inner);

        removed
            .into_values()
            .map(|mut o| {
                o.parent_id = None;
                o.frame = None;
                o
            })
            .collect()
    }

    pub fn object_exists(&self, id: i64) -> bool {
        let inner = trace!(self.inner.read_recursive());
        inner.objects.contains_key(&id)
    }

    pub fn delete_objects(&self, q: &MatchQuery) -> Vec<VideoObject> {
        let objs = self.access_objects(q);
        let ids = objs.iter().map(|o| o.get_id()).collect::<Vec<_>>();
        self.delete_objects_with_ids(&ids)
    }

    pub fn get_object(&self, id: i64) -> Option<BorrowedVideoObject> {
        let inner = trace!(self.inner.read_recursive());
        let obj = inner.objects.get(&id);
        obj.map(|_| BorrowedVideoObject(self.into(), id))
    }

    fn fix_object_owned_frame(&self) {
        self.get_all_objects()
            .into_iter()
            .for_each(|mut o| o.attach_to_video_frame(self.clone()));
    }

    pub fn set_draw_label(&self, q: &MatchQuery, label: DrawLabelKind) {
        let objects = self.access_objects(q);
        objects.into_iter().for_each(|mut o| match &label {
            DrawLabelKind::OwnLabel(l) => {
                o.set_draw_label(Some(l.clone()));
            }
            DrawLabelKind::ParentLabel(l) => {
                if let Some(mut p) = o.get_parent() {
                    p.set_draw_label(Some(l.clone()));
                }
            }
        });
    }

    pub fn set_parent(
        &self,
        q: &MatchQuery,
        parent: &BorrowedVideoObject,
    ) -> anyhow::Result<Vec<BorrowedVideoObject>> {
        let frame_opt = parent.get_frame();
        if let Some(frame) = frame_opt {
            if !Arc::ptr_eq(&frame.inner.0, &self.inner.0) {
                bail!(
                    "Parent object ID={} must be attached to the same frame.",
                    parent.get_id()
                );
            }
        } else {
            bail!("Parent object ID={} must be attached.", parent.get_id());
        }

        let mut objects = self.access_objects(q);
        objects
            .iter_mut()
            .try_for_each(|o| o.set_parent(Some(parent.get_id())))?;

        Ok(objects)
    }

    pub fn set_parent_by_id(&self, object_id: i64, parent_id: i64) -> anyhow::Result<()> {
        self.get_object(parent_id).ok_or(anyhow!(
            "Parent object with ID {} does not exist in the frame.",
            parent_id
        ))?;

        let mut object = self.get_object(object_id).ok_or(anyhow!(
            "Object with ID {} does not exist in the frame.",
            object_id
        ))?;

        object.set_parent(Some(parent_id))?;
        Ok(())
    }

    pub fn clear_parent(&self, q: &MatchQuery) -> Vec<BorrowedVideoObject> {
        let mut objects = self.access_objects(q);
        objects.iter_mut().for_each(|o| {
            o.set_parent(None).unwrap();
        });
        objects
    }

    pub fn get_children(&self, id: i64) -> Vec<BorrowedVideoObject> {
        self.access_objects(&MatchQuery::ParentId(IntExpression::EQ(id)))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_object(
        &self,
        namespace: &str,
        label: &str,
        parent_id: Option<i64>,
        detection_box: RBBox,
        confidence: Option<f32>,
        track_id: Option<i64>,
        track_box: Option<RBBox>,
        attributes: Vec<Attribute>,
    ) -> anyhow::Result<BorrowedVideoObject> {
        let id = self.get_max_object_id() + 1;
        if let Some(parent_id) = parent_id {
            if !self.object_exists(parent_id) {
                bail!(
                    "Parent object with ID {} does not exist in the frame.",
                    parent_id
                );
            }
        }
        let object = VideoObjectBuilder::default()
            .id(id)
            .detection_box(detection_box)
            .parent_id(parent_id)
            .attributes(attributes)
            .confidence(confidence)
            .namespace(namespace.to_string())
            .label(label.to_string())
            .track_id(track_id)
            .track_box(track_box)
            .build()
            .unwrap();
        self.add_object(object, IdCollisionResolutionPolicy::Error)
    }

    pub fn add_object(
        &self,
        mut object: VideoObject,
        policy: IdCollisionResolutionPolicy,
    ) -> anyhow::Result<BorrowedVideoObject> {
        let parent_id_opt = object.get_parent_id();
        if let Some(parent_id) = parent_id_opt {
            if !self.object_exists(parent_id) {
                bail!(
                    "Parent object with ID {} does not exist in the frame.",
                    parent_id
                );
            }
        }

        let object_id = object.get_id();
        let new_id = self.get_max_object_id() + 1;
        let mut inner = trace!(self.inner.write());
        object.attach_to_video_frame(self.clone());
        let assigned_object_id = if inner.objects.contains_key(&object_id) {
            match policy {
                IdCollisionResolutionPolicy::GenerateNewId => {
                    object.with_object_mut(|o| o.id = new_id);
                    inner.objects.insert(new_id, object);
                    new_id
                }
                IdCollisionResolutionPolicy::Overwrite => {
                    inner.objects.remove(&object_id).unwrap();
                    inner.objects.insert(object_id, object);
                    object_id
                }
                IdCollisionResolutionPolicy::Error => {
                    bail!("Object with ID {} already exists in the frame.", object_id);
                }
            }
        } else {
            inner.objects.insert(object_id, object);
            object_id
        };

        if assigned_object_id > inner.max_object_id {
            inner.max_object_id = assigned_object_id;
        }
        Ok(BorrowedVideoObject(self.into(), assigned_object_id))
    }

    pub fn get_max_object_id(&self) -> i64 {
        let inner = trace!(self.inner.read_recursive());
        inner.max_object_id
    }

    pub(crate) fn update_objects(&self, update: &VideoFrameUpdate) -> anyhow::Result<()> {
        use crate::primitives::frame_update::ObjectUpdatePolicy::*;
        let other_inner = update.objects.clone();

        let object_query = |o: &VideoObject| {
            and![
                MatchQuery::Label(StringExpression::EQ(o.label.clone())),
                MatchQuery::Namespace(StringExpression::EQ(o.namespace.clone()))
            ]
        };

        match &update.object_policy {
            AddForeignObjects => {
                for (mut obj, p) in other_inner {
                    let object_id = self.get_max_object_id() + 1;
                    obj.id = object_id;

                    self.add_object(obj, IdCollisionResolutionPolicy::GenerateNewId)?;
                    if let Some(p) = p {
                        self.set_parent_by_id(object_id, p)?;
                    }
                }
            }
            ErrorIfLabelsCollide => {
                for (mut obj, p) in other_inner {
                    let objs = self.access_objects(&object_query(&obj));
                    if !objs.is_empty() {
                        bail!(
                            "Objects with label '{}' and namespace '{}' already exists in the frame.",
                            obj.label,
                            obj.namespace
                        )
                    }

                    let object_id = self.get_max_object_id() + 1;
                    obj.id = object_id;

                    self.add_object(obj, IdCollisionResolutionPolicy::GenerateNewId)?;
                    if let Some(p) = p {
                        self.set_parent_by_id(object_id, p)?;
                    }
                }
            }
            ReplaceSameLabelObjects => {
                for (mut obj, p) in other_inner {
                    self.delete_objects(&object_query(&obj));

                    let object_id = self.get_max_object_id() + 1;
                    obj.id = object_id;

                    self.add_object(obj, IdCollisionResolutionPolicy::GenerateNewId)?;

                    if let Some(p) = p {
                        self.set_parent_by_id(object_id, p)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn update_frame_attributes(&self, update: &VideoFrameUpdate) -> anyhow::Result<()> {
        use crate::primitives::frame_update::AttributeUpdatePolicy::*;
        match &update.frame_attribute_policy {
            ReplaceWithForeign => {
                let mut inner = trace!(self.inner.write());
                let other_inner = update.get_frame_attributes().clone();
                other_inner.iter().for_each(|a| {
                    inner.set_attribute(a.clone());
                });
            }
            KeepOwn => {
                let mut inner = trace!(self.inner.write());
                let other_inner = update.get_frame_attributes();
                for attr in other_inner {
                    if inner.get_attribute(&attr.namespace, &attr.name).is_none() {
                        inner.set_attribute(attr.clone());
                    }
                }
            }
            Error => {
                let mut inner = trace!(self.inner.write());
                let other_inner = update.get_frame_attributes().clone();
                for attr in other_inner {
                    let key = (attr.namespace.clone(), attr.name.clone());
                    if inner.get_attribute(&attr.namespace, &attr.name).is_some() {
                        bail!(
                            "Attribute with name '{}' created by '{}' already exists in the frame.",
                            key.1,
                            key.0
                        );
                    }
                    inner.set_attribute(attr.clone());
                }
            }
        }

        Ok(())
    }

    pub(crate) fn update_object_attributes(&self, update: &VideoFrameUpdate) -> anyhow::Result<()> {
        use crate::primitives::frame_update::AttributeUpdatePolicy::*;
        let update_attrs = update.get_object_attributes().clone();
        for (id, attr) in update_attrs {
            let mut obj = self.get_object(id).ok_or(anyhow!(
                "Object with ID {} does not exist in the frame.",
                id
            ))?;
            match &update.object_attribute_policy {
                ReplaceWithForeign => {
                    obj.set_attribute(attr);
                }
                KeepOwn => {
                    if obj.get_attribute(&attr.namespace, &attr.name).is_none() {
                        obj.set_attribute(attr);
                    }
                }
                Error => {
                    if obj.get_attribute(&attr.namespace, &attr.name).is_some() {
                        bail!(
                            "Attribute with name '{}.{}' already exists in the object with ID {}.",
                            attr.namespace,
                            attr.name,
                            id
                        );
                    }
                }
            }
        }
        Ok(())
    }

    pub fn update(&self, update: &VideoFrameUpdate) -> anyhow::Result<()> {
        self.update_frame_attributes(update)?;
        self.update_object_attributes(update)?;
        self.update_objects(update)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source_id: &str,
        framerate: &str,
        width: i64,
        height: i64,
        content: VideoFrameContent,
        transcoding_method: VideoFrameTranscodingMethod,
        codec: &Option<&str>,
        keyframe: Option<bool>,
        time_base: (i64, i64),
        pts: i64,
        dts: Option<i64>,
        duration: Option<i64>,
    ) -> Self {
        VideoFrameProxy::from_inner(VideoFrame {
            source_id: source_id.to_string(),
            pts,
            framerate: framerate.to_string(),
            width,
            height,
            time_base: (time_base.0 as i32, time_base.1 as i32),
            dts,
            duration,
            transcoding_method,
            codec: codec.map(String::from),
            keyframe,
            content: Arc::new(content),
            ..Default::default()
        })
    }

    pub fn to_message(&self) -> Message {
        Message::video_frame(self)
    }

    pub fn get_source_id(&self) -> String {
        trace!(self.inner.read_recursive()).source_id.clone()
    }

    pub fn get_previous_frame_seq_id(&self) -> Option<i64> {
        trace!(self.inner.read_recursive()).previous_frame_seq_id
    }

    pub(crate) fn set_previous_frame_seq_id(&mut self, previous_frame_seq_id: Option<i64>) {
        let mut inner = trace!(self.inner.write());
        inner.previous_frame_seq_id = previous_frame_seq_id;
    }

    pub fn get_previous_keyframe(&self) -> Option<u128> {
        trace!(self.inner.read_recursive()).previous_keyframe
    }

    pub fn get_previous_keyframe_as_string(&self) -> Option<String> {
        self.get_previous_keyframe()
            .map(|ku| Uuid::from_u128(ku).to_string())
    }

    pub(crate) fn set_previous_keyframe(&mut self, previous_keyframe: Option<u128>) {
        let mut inner = trace!(self.inner.write());
        inner.previous_keyframe = previous_keyframe;
    }

    pub fn set_source_id(&mut self, source_id: &str) {
        let mut inner = trace!(self.inner.write());
        inner.source_id = source_id.to_string();
    }

    pub fn set_time_base(&mut self, time_base: (i32, i32)) {
        let mut inner = trace!(self.inner.write());
        inner.time_base = time_base;
    }
    pub fn get_time_base(&self) -> (i32, i32) {
        trace!(self.inner.read_recursive()).time_base
    }

    pub fn get_uuid(&self) -> Uuid {
        Uuid::from_u128(trace!(self.inner.read_recursive()).uuid)
    }

    pub fn get_uuid_u128(&self) -> u128 {
        trace!(self.inner.read_recursive()).uuid
    }

    pub fn get_uuid_as_string(&self) -> String {
        self.get_uuid().to_string()
    }

    pub fn get_creation_timestamp_ns(&self) -> u128 {
        trace!(self.inner.read_recursive()).creation_timestamp_ns
    }

    pub fn set_creation_timestamp_ns(&mut self, creation_timestamp_ns: u128) {
        let mut inner = trace!(self.inner.write());
        inner.creation_timestamp_ns = creation_timestamp_ns;
    }

    pub fn get_pts(&self) -> i64 {
        trace!(self.inner.read_recursive()).pts
    }
    pub fn set_pts(&mut self, pts: i64) {
        assert!(pts >= 0, "pts must be greater than or equal to 0");
        let mut inner = trace!(self.inner.write());
        inner.pts = pts;
    }

    pub fn get_framerate(&self) -> String {
        trace!(self.inner.read_recursive()).framerate.clone()
    }

    pub fn set_framerate(&mut self, framerate: &str) {
        let mut inner = trace!(self.inner.write());
        inner.framerate = framerate.to_string();
    }

    pub fn get_width(&self) -> i64 {
        trace!(self.inner.read_recursive()).width
    }

    pub fn set_width(&mut self, width: i64) {
        assert!(width > 0, "width must be greater than 0");
        let mut inner = trace!(self.inner.write());
        inner.width = width;
    }

    pub fn get_height(&self) -> i64 {
        trace!(self.inner.read_recursive()).height
    }

    pub fn set_height(&mut self, height: i64) {
        assert!(height > 0, "height must be greater than 0");
        let mut inner = trace!(self.inner.write());
        inner.height = height;
    }

    pub fn get_dts(&self) -> Option<i64> {
        let inner = trace!(self.inner.read_recursive());
        inner.dts
    }

    pub fn set_dts(&mut self, dts: Option<i64>) {
        assert!(
            dts.is_none() || dts.unwrap() >= 0,
            "dts must be greater than or equal to 0"
        );
        let mut inner = trace!(self.inner.write());
        inner.dts = dts;
    }

    pub fn get_duration(&self) -> Option<i64> {
        let inner = trace!(self.inner.read_recursive());
        inner.duration
    }

    pub fn set_duration(&mut self, duration: Option<i64>) {
        assert!(
            duration.is_none() || duration.unwrap() >= 0,
            "duration must be greater than or equal to 0"
        );
        let mut inner = trace!(self.inner.write());
        inner.duration = duration;
    }

    pub fn get_transcoding_method(&self) -> VideoFrameTranscodingMethod {
        let inner = trace!(self.inner.read_recursive());
        inner.transcoding_method.clone()
    }

    pub fn set_transcoding_method(&mut self, transcoding_method: VideoFrameTranscodingMethod) {
        let mut inner = trace!(self.inner.write());
        inner.transcoding_method = transcoding_method;
    }

    pub fn get_codec(&self) -> Option<String> {
        let inner = trace!(self.inner.read_recursive());
        inner.codec.clone()
    }

    pub fn set_codec(&mut self, codec: Option<String>) {
        let mut inner = trace!(self.inner.write());
        inner.codec = codec;
    }

    pub fn clear_transformations(&mut self) {
        let mut inner = trace!(self.inner.write());
        inner.transformations.clear();
    }

    pub fn add_transformation(&mut self, transformation: VideoFrameTransformation) {
        let mut inner = trace!(self.inner.write());
        inner.transformations.push(transformation);
    }

    pub fn get_transformations(&self) -> Vec<VideoFrameTransformation> {
        let inner = trace!(self.inner.read_recursive());
        inner.transformations.clone()
    }

    pub fn get_keyframe(&self) -> Option<bool> {
        let inner = trace!(self.inner.read_recursive());
        inner.keyframe
    }

    pub fn set_keyframe(&mut self, keyframe: Option<bool>) {
        let mut inner = trace!(self.inner.write());
        inner.keyframe = keyframe;
    }

    pub fn get_content(&self) -> Arc<VideoFrameContent> {
        let inner = trace!(self.inner.read_recursive());
        inner.content.clone()
    }

    pub fn set_content(&mut self, content: VideoFrameContent) {
        let mut inner = trace!(self.inner.write());
        inner.content = Arc::new(content);
    }

    pub fn clear_objects(&self) {
        let mut frame = trace!(self.inner.write());
        frame.objects.clear();
    }

    pub fn get_ancestory(&self, obj: &BorrowedVideoObject) -> Vec<i64> {
        let mut ids = vec![obj.get_id()];
        let mut current = obj.get_parent();
        while let Some(parent) = current {
            ids.push(parent.get_id());
            current = parent.get_parent();
        }
        ids
    }

    pub fn reduce_to_common_parents(&self, q: &MatchQuery) -> Vec<BorrowedVideoObject> {
        let mut reduced = true;
        let mut objects = self.access_objects(q);
        while reduced {
            let mut unique_ancestors = HashSet::new();
            let object_ids = objects.iter().map(|o| o.get_id()).collect::<HashSet<_>>();
            for obj in &objects {
                let ancestor_ids = self.get_ancestory(obj);
                // reverse order iteration
                for id in ancestor_ids.iter().rev() {
                    if object_ids.contains(id) {
                        unique_ancestors.insert(*id);
                        break;
                    }
                }
            }
            let mut new_objects = Vec::new();

            for obj in &objects {
                let id = obj.get_id();
                if unique_ancestors.contains(&id) {
                    new_objects.push(obj.clone());
                }
            }
            if new_objects.len() == objects.len() {
                reduced = false;
            }
            objects = new_objects;
        }
        objects
    }

    fn export_tree(&self, obj: &BorrowedVideoObject) -> VideoObjectTree {
        let mut tree = VideoObjectTree::new(obj.detached_copy());
        let current = obj.get_id();
        let children = self.get_children(current);
        for c in children {
            let child_tree = self.export_tree(&c);
            tree.add_child(child_tree);
        }
        tree
    }

    pub fn export_complete_object_trees(
        &self,
        q: &MatchQuery,
        delete_exported: bool,
    ) -> anyhow::Result<Vec<VideoObjectTree>> {
        let ancestors = self.reduce_to_common_parents(q);
        let mut trees = Vec::new();
        for o in &ancestors {
            trees.push(self.export_tree(o));
        }
        let trees = ancestors
            .iter()
            .map(|o| self.export_tree(o))
            .collect::<Vec<_>>();
        if delete_exported {
            let mut object_ids = Vec::new();
            for t in &trees {
                let ids = t.get_object_ids()?;
                object_ids.extend(ids);
            }
            self.delete_objects_with_ids(&object_ids);
        }
        Ok(trees)
    }

    pub fn import_object_trees(&self, trees: Vec<VideoObjectTree>) -> anyhow::Result<()> {
        for t in trees {
            t.walk_objects(
                &mut |object: &VideoObject,
                      _: Option<&VideoObject>,
                      owned_parent: Option<&BorrowedVideoObject>| {
                    let mut obj = self
                        .add_object(object.clone(), IdCollisionResolutionPolicy::GenerateNewId)
                        .unwrap();
                    if let Some(parent) = owned_parent {
                        obj.set_parent(Some(parent.get_id()))?;
                    }
                    Ok(obj)
                },
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use hashbrown::HashSet;

    use crate::draw::DrawLabelKind;
    use crate::match_query::{eq, one_of, MatchQuery as Q};
    use crate::primitives::object::private::{SealedWithFrame, SealedWithParent};
    use crate::primitives::object::VideoObject;
    use crate::primitives::object::{
        IdCollisionResolutionPolicy, ObjectOperations, VideoObjectBuilder,
    };
    use crate::primitives::{RBBox, WithAttributes};
    use crate::test::{gen_empty_frame, gen_frame, gen_object, s};
    use std::sync::Arc;

    fn gen_labeled_object(id: i64) -> VideoObject {
        let mut obj = gen_object(id);
        let label = format!("{:03}", id);
        obj.set_label(&label);
        obj
    }

    #[test]
    fn test_access_objects_by_id() {
        let t = gen_frame();
        let objects = t.access_objects_with_id(&vec![0, 1]);
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].get_id(), 0);
        assert_eq!(objects[1].get_id(), 1);
    }

    #[test]
    fn test_get_parent() {
        let frame = gen_frame();
        let o = frame.get_object(1).unwrap();
        let _ = o.get_parent().unwrap();
    }

    #[test]
    fn test_objects_by_id() {
        let t = gen_frame();
        let objects = t.access_objects_with_id(&vec![0, 1]);
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].get_id(), 0);
        assert_eq!(objects[1].get_id(), 1);
    }

    #[test]
    fn test_delete_objects_by_ids() {
        let f = gen_frame();
        f.delete_objects_with_ids(&[0, 1]);
        let objects = f.get_all_objects();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].get_id(), 2);
    }

    #[test]
    fn test_parent_cleared_when_delete_objects_by_ids() {
        let f = gen_frame();
        f.delete_objects_with_ids(&[0]);
        let o = f.get_object(1).unwrap();
        assert!(o.get_parent().is_none());
        let o = f.get_object(2).unwrap();
        assert!(o.get_parent().is_none());

        let f = gen_frame();
        f.delete_objects_with_ids(&[1]);
        let o = f.get_object(2).unwrap();
        assert!(o.get_parent().is_some());
    }

    #[test]
    fn test_parent_cleared_when_delete_objects_by_query() {
        let f = gen_frame();

        let o = f.get_object(0).unwrap();
        assert!(o.get_frame().is_some());

        let removed = f.delete_objects(&Q::Id(eq(0)));
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].get_id(), 0);
        assert!(removed[0].get_frame().is_none());

        let o = f.get_object(1).unwrap();
        assert!(o.get_parent().is_none());

        let o = f.get_object(2).unwrap();
        assert!(o.get_parent().is_none());
    }

    #[test]
    fn test_delete_all_objects() {
        let f = gen_frame();
        let objs = f.delete_objects(&Q::Idle);
        assert_eq!(objs.len(), 3);
        let objects = f.get_all_objects();
        assert!(objects.is_empty());
    }

    #[test]
    fn test_parent_not_added_to_frame() {
        let frame = gen_frame();
        let mut obj = frame.get_object(0).unwrap();
        assert!(obj.set_parent(Some(155)).is_err());
    }

    #[test]
    fn test_no_children() {
        let frame = gen_frame();
        let obj = frame.get_object(2).unwrap();
        assert!(frame.get_children(obj.get_id()).is_empty());
    }

    #[test]
    fn test_two_children() {
        let frame = gen_frame();
        let obj = frame.get_object(0).unwrap();
        assert_eq!(frame.get_children(obj.get_id()).len(), 2);
    }

    #[test]
    fn set_parent_draw_label() {
        let frame = gen_frame();
        frame.set_draw_label(&Q::Idle, DrawLabelKind::ParentLabel(s("draw")));
        let parent_object = frame.get_object(0).unwrap();
        assert_eq!(parent_object.calculate_draw_label(), s("draw"));

        let child_object = frame.get_object(1).unwrap();
        assert_ne!(child_object.calculate_draw_label(), s("draw"));
    }

    #[test]
    fn set_own_draw_label() {
        let frame = gen_frame();
        frame.set_draw_label(&Q::Idle, DrawLabelKind::OwnLabel(s("draw")));
        let parent_object = frame.get_object(0).unwrap();
        assert_eq!(parent_object.calculate_draw_label(), s("draw"));

        let child_object = frame.get_object(1).unwrap();
        assert_eq!(child_object.calculate_draw_label(), s("draw"));

        let child_object = frame.get_object(2).unwrap();
        assert_eq!(child_object.calculate_draw_label(), s("draw"));
    }

    #[test]
    fn test_set_clear_parent_ops() -> anyhow::Result<()> {
        let frame = gen_frame();
        let parent = frame.get_object(0).unwrap();
        frame.clear_parent(&Q::Id(one_of(&[1, 2])));
        let obj = frame.get_object(1).unwrap();
        assert!(obj.get_parent().is_none());
        let obj = frame.get_object(2).unwrap();
        assert!(obj.get_parent().is_none());

        frame.set_parent(&Q::Id(one_of(&[1, 2])), &parent)?;
        let obj = frame.get_object(1).unwrap();
        assert!(obj.get_parent().is_some());

        let obj = frame.get_object(2).unwrap();
        assert!(obj.get_parent().is_some());
        Ok(())
    }

    #[test]
    fn retrieve_children() {
        let frame = gen_frame();
        let parent = frame.get_object(0).unwrap();
        let children = frame.get_children(parent.get_id());
        assert_eq!(children.len(), 2);
    }

    #[test]
    #[should_panic]
    fn attach_object_with_detached_parent() {
        let p = VideoObjectBuilder::default()
            .id(11)
            .namespace(s("random"))
            .label(s("something"))
            .detection_box(RBBox::new(1.0, 2.0, 10.0, 20.0, None).try_into().unwrap())
            .build()
            .unwrap();

        let o = VideoObjectBuilder::default()
            .id(23)
            .namespace(s("random"))
            .label(s("something"))
            .detection_box(RBBox::new(1.0, 2.0, 10.0, 20.0, None).try_into().unwrap())
            .parent_id(Some(p.get_id()))
            .build()
            .unwrap();

        let f = gen_frame();
        f.add_object(o, IdCollisionResolutionPolicy::Error).unwrap();
    }

    #[test]
    fn set_wrong_parent_as_parent() {
        let f1 = gen_frame();
        let f2 = gen_frame();
        let f1o = f1.get_object(0).unwrap();
        assert!(f2.set_parent(&Q::Id(eq(1)), &f1o).is_err());
    }

    #[test]
    fn normally_transfer_parent() -> anyhow::Result<()> {
        let f1 = gen_frame();
        let f2 = gen_frame();
        let mut object = f1.delete_objects_with_ids(&[0]).pop().unwrap();
        assert!(object.get_frame().is_none());
        _ = object.set_id(33);
        f2.add_object(object, IdCollisionResolutionPolicy::Error)
            .unwrap();

        let borrowed_object = f2.get_object(33).unwrap();
        f2.set_parent(&Q::Id(eq(1)), &borrowed_object)?;
        Ok(())
    }

    #[test]
    fn deleted_objects_clean() {
        let frame = gen_frame();
        let removed = frame.delete_objects_with_ids(&[1]).pop().unwrap();
        assert!(removed.get_parent().is_none());
    }

    #[test]
    fn deep_copy() {
        let mut f = gen_frame();
        let new_f = f.smart_copy();

        // check that objects are copied
        let mut o = f.get_object(0).unwrap();
        let new_o = new_f.get_object(0).unwrap();
        let label = "new label";
        o.set_label(label);
        assert_ne!(new_o.get_label(), label);

        // check that attributes are copied
        f.clear_attributes();
        assert!(f.get_attributes().is_empty());
        assert!(!new_f.get_attributes().is_empty());

        // check that the objects are attached to the new frame
        let o = new_f.get_object(0).unwrap();
        assert!(o.get_frame().is_some());
        assert!(Arc::ptr_eq(&o.get_frame().unwrap().inner.0, &new_f.inner.0));
    }

    #[test]
    fn test_create_object() -> anyhow::Result<()> {
        let frame = gen_empty_frame();
        let obj = frame.create_object(
            "some-namespace",
            "some-label",
            None,
            RBBox::new(0.0, 0.0, 0.0, 0.0, None).try_into().unwrap(),
            None,
            None,
            None,
            vec![],
        )?;
        assert_eq!(obj.get_id(), 1);
        Ok(())
    }

    #[test]
    fn add_objects_test_policy_error() {
        let frame = gen_empty_frame();

        let object = gen_object(0);
        frame
            .add_object(object, IdCollisionResolutionPolicy::Error)
            .unwrap();

        let object = gen_object(0);
        assert!(frame
            .add_object(object, IdCollisionResolutionPolicy::Error)
            .is_err());
    }

    #[test]
    fn add_objects_test_policy_generate_new_id() {
        let frame = gen_empty_frame();

        let object = gen_object(0);
        frame
            .add_object(object, IdCollisionResolutionPolicy::GenerateNewId)
            .unwrap();

        let object = gen_object(0);
        frame
            .add_object(object, IdCollisionResolutionPolicy::GenerateNewId)
            .unwrap();
        assert_eq!(frame.get_max_object_id(), 1);
        let objs = frame.get_all_objects();
        assert_eq!(objs.len(), 2);
    }

    #[test]
    fn add_objects_test_policy_overwrite() {
        let frame = gen_empty_frame();

        let object = gen_object(0);
        frame
            .add_object(object, IdCollisionResolutionPolicy::Overwrite)
            .unwrap();

        let object = gen_object(0);
        assert!(frame
            .add_object(object, IdCollisionResolutionPolicy::Overwrite)
            .is_ok());

        assert_eq!(frame.get_max_object_id(), 0);
        let objs = frame.get_all_objects();
        assert_eq!(objs.len(), 1);
    }

    #[test]
    fn test_get_common_parents() -> anyhow::Result<()> {
        //         0
        //        / \
        //       1   3
        //      / \    \
        //     2   6    5
        //    /
        //   4
        let frame = gen_empty_frame();
        let obj0 = gen_object(0);
        let obj0_ref = frame.add_object(obj0, IdCollisionResolutionPolicy::Error)?;

        let obj1 = gen_object(1);
        let obj1_ref = frame.add_object(obj1, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(1)), &obj0_ref)?;

        let obj2 = gen_object(2);
        let obj2_ref = frame.add_object(obj2, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(2)), &obj1_ref)?;

        let obj3 = gen_object(3);
        let obj3_ref = frame.add_object(obj3, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(3)), &obj0_ref)?;

        let obj4 = gen_object(4);
        let _obj4_ref = frame.add_object(obj4, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(4)), &obj2_ref)?;

        let obj5 = gen_object(5);
        let _obj5_ref = frame.add_object(obj5, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(5)), &obj3_ref)?;

        let obj6 = gen_object(6);
        let _obj6_ref = frame.add_object(obj6, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(6)), &obj1_ref)?;

        let common_parents = frame.reduce_to_common_parents(&Q::Id(eq(3)));
        assert_eq!(common_parents.len(), 1);
        assert_eq!(common_parents[0].get_id(), 3);

        let common_parents = frame.reduce_to_common_parents(&Q::Id(eq(0)));
        assert_eq!(common_parents.len(), 1);
        assert_eq!(common_parents[0].get_id(), 0);

        let common_parents = frame.reduce_to_common_parents(&Q::Id(one_of(&[0, 1, 3])));
        assert_eq!(common_parents.len(), 1);
        assert_eq!(common_parents[0].get_id(), 0);

        let common_parents = frame.reduce_to_common_parents(&Q::Id(one_of(&[0, 4])));
        assert_eq!(common_parents.len(), 1);
        assert_eq!(common_parents[0].get_id(), 0);

        let common_parents = frame.reduce_to_common_parents(&Q::Id(one_of(&[1, 4])));
        assert_eq!(common_parents.len(), 1);
        assert_eq!(common_parents[0].get_id(), 1);

        let common_parents = frame.reduce_to_common_parents(&Q::Id(one_of(&[1, 2, 3])));
        assert_eq!(common_parents.len(), 2);
        let ids = common_parents
            .iter()
            .map(|o| o.get_id())
            .collect::<Vec<_>>();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));

        let common_parents = frame.reduce_to_common_parents(&Q::Id(one_of(&[2, 4, 6, 5])));
        assert_eq!(common_parents.len(), 3);
        let ids = common_parents
            .iter()
            .map(|o| o.get_id())
            .collect::<Vec<_>>();
        assert!(ids.contains(&2));
        assert!(ids.contains(&6));
        assert!(ids.contains(&5));

        Ok(())
    }

    #[test]
    fn test_export_tree() -> anyhow::Result<()> {
        //         0
        //        / \
        //       1   3
        //      / \    \
        //     2   6    5
        //    /
        //   4
        let frame = gen_empty_frame();
        let obj0 = gen_object(0);
        let obj0_ref = frame.add_object(obj0, IdCollisionResolutionPolicy::Error)?;

        let obj1 = gen_object(1);
        let obj1_ref = frame.add_object(obj1, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(1)), &obj0_ref)?;

        let obj2 = gen_object(2);
        let obj2_ref = frame.add_object(obj2, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(2)), &obj1_ref)?;

        let obj3 = gen_object(3);
        let obj3_ref = frame.add_object(obj3, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(3)), &obj0_ref)?;

        let obj4 = gen_object(4);
        let _obj4_ref = frame.add_object(obj4, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(4)), &obj2_ref)?;

        let obj5 = gen_object(5);
        let _obj5_ref = frame.add_object(obj5, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(5)), &obj3_ref)?;

        let obj6 = gen_object(6);
        let _obj6_ref = frame.add_object(obj6, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(6)), &obj1_ref)?;

        let tree = frame.export_tree(&obj0_ref);
        let mut ids = HashSet::new();
        tree.walk_objects(
            &mut |o: &VideoObject, _: Option<&VideoObject>, _: Option<&()>| {
                ids.insert(o.get_id());
                Ok(())
            },
        )?;

        assert_eq!(ids.len(), 7);
        let expected_ids = vec![0, 1, 2, 3, 4, 5, 6]
            .into_iter()
            .collect::<HashSet<_>>();
        assert_eq!(ids, expected_ids);

        let tree = frame.export_tree(&obj3_ref);
        let mut ids = HashSet::new();
        tree.walk_objects(
            &mut |o: &VideoObject, _: Option<&VideoObject>, _: Option<&()>| {
                ids.insert(o.get_id());
                Ok(())
            },
        )?;
        assert_eq!(ids.len(), 2);
        let expected_ids = vec![3, 5].into_iter().collect::<HashSet<_>>();
        assert_eq!(ids, expected_ids);

        Ok(())
    }

    #[test]
    fn test_export_import_complete_object_trees() -> anyhow::Result<()> {
        //         0
        //        / \
        //       1   3
        //      / \    \
        //     2   6    5
        //    /
        //   4

        let gen_object = |id: i64| gen_labeled_object(id);
        let frame = gen_empty_frame();
        let obj0 = gen_object(0);
        let obj0_ref = frame.add_object(obj0, IdCollisionResolutionPolicy::Error)?;

        let obj1 = gen_object(1);
        let obj1_ref = frame.add_object(obj1, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(1)), &obj0_ref)?;

        let obj2 = gen_object(2);
        let obj2_ref = frame.add_object(obj2, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(2)), &obj1_ref)?;

        let obj3 = gen_object(3);
        let obj3_ref = frame.add_object(obj3, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(3)), &obj0_ref)?;

        let obj4 = gen_object(4);
        let _obj4_ref = frame.add_object(obj4, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(4)), &obj2_ref)?;

        let obj5 = gen_object(5);
        let _obj5_ref = frame.add_object(obj5, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(5)), &obj3_ref)?;

        let obj6 = gen_object(6);
        let _obj6_ref = frame.add_object(obj6, IdCollisionResolutionPolicy::Error)?;
        frame.set_parent(&Q::Id(eq(6)), &obj1_ref)?;

        let trees = frame.export_complete_object_trees(&Q::Id(one_of(&[1, 3, 2])), true)?;
        assert_eq!(trees.len(), 2);
        let objects = frame.get_all_objects();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].get_id(), 0);

        let mut object_tree_ids = trees
            .iter()
            .map(|t| {
                let mut ids = t.get_object_ids().unwrap();
                ids.sort();
                ids
            })
            .collect::<Vec<_>>();
        object_tree_ids.sort_by(|a, b| a.len().cmp(&b.len()));
        assert_eq!(object_tree_ids[0], vec![3, 5]);
        assert_eq!(object_tree_ids[1], vec![1, 2, 4, 6]);

        frame.import_object_trees(trees)?;
        let objects = frame.get_all_objects();
        assert_eq!(objects.len(), 7);
        for o in objects {
            println!(
                "Label: {}, ID: {}, Parent: {:?}",
                o.get_label(),
                o.get_id(),
                o.get_parent().map(|p| p.get_id()).unwrap_or(-1)
            );
        }

        println!("---Exporting again, without removal---");
        let trees = frame.export_complete_object_trees(&Q::Id(one_of(&[1, 3, 2])), false)?;
        assert_eq!(trees.len(), 2);
        let objects = frame.get_all_objects();
        assert_eq!(objects.len(), 7);

        frame.import_object_trees(trees)?;
        let objects = frame.get_all_objects();
        assert_eq!(objects.len(), 13);
        for o in objects {
            println!(
                "Label: {}, ID: {}, Parent: {:?}",
                o.get_label(),
                o.get_id(),
                o.get_parent().map(|p| p.get_id()).unwrap_or(-1)
            );
        }

        Ok(())
    }
}

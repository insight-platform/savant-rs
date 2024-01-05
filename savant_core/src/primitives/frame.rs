use crate::draw::DrawLabelKind;
use crate::json_api::ToSerdeJsonValue;
use crate::match_query::{and, IntExpression, MatchQuery, StringExpression};
use crate::message::Message;
use crate::primitives::attribute::AttributeMethods;
use crate::primitives::frame_update::VideoFrameUpdate;
use crate::primitives::object::{
    IdCollisionResolutionPolicy, VideoObject, VideoObjectBBoxTransformation, VideoObjectBBoxType,
    VideoObjectProxy,
};
use crate::primitives::{Attribute, Attributive};
use crate::rwlock::{SavantArcRwLock, SavantRwLock};
use crate::trace;
use crate::version;
use anyhow::{anyhow, bail};
use derive_builder::Builder;
use hashbrown::HashMap;
use rkyv::{with::Lock, with::Skip, Archive, Deserialize, Serialize};
use savant_utils::iter::fiter_map_with_control_flow;
use serde_json::Value;
use std::fmt::{Debug, Formatter};
use std::mem;
use std::sync::{Arc, Weak};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct ExternalFrame {
    pub method: String,
    pub location: Option<String>,
}

impl ToSerdeJsonValue for ExternalFrame {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!({
            "method": self.method,
            "location": self.location,
        })
    }
}

impl ExternalFrame {
    pub fn new(method: String, location: Option<String>) -> Self {
        Self { method, location }
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
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

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub enum VideoFrameTranscodingMethod {
    Copy,
    Encoded,
}

impl ToSerdeJsonValue for VideoFrameTranscodingMethod {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(format!("{:?}", self))
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub enum VideoFrameTransformation {
    InitialSize(u64, u64),
    Scale(u64, u64),
    Padding(u64, u64, u64, u64),
    ResultingSize(u64, u64),
}

impl ToSerdeJsonValue for VideoFrameTransformation {
    fn to_serde_json_value(&self) -> Value {
        match self {
            VideoFrameTransformation::InitialSize(width, height) => {
                serde_json::json!({"initial_size": [width, height]})
            }
            VideoFrameTransformation::Scale(width, height) => {
                serde_json::json!({"scale": [width, height]})
            }
            VideoFrameTransformation::Padding(left, top, right, bottom) => {
                serde_json::json!({"padding": [left, top, right, bottom]})
            }
            VideoFrameTransformation::ResultingSize(width, height) => {
                serde_json::json!({"resulting_size": [width, height]})
            }
        }
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, Clone, Builder)]
#[archive(check_bytes)]
pub struct VideoFrame {
    #[builder(setter(skip))]
    pub previous_frame_seq_id: Option<i64>,
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
    pub(crate) objects: HashMap<i64, VideoObjectProxy>,
    #[with(Skip)]
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
            source_id: String::new(),
            uuid: Uuid::new_v4().as_u128(),
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
        let version = version();
        serde_json::json!(
            {
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
                "objects": self.objects.values().map(|o| o.to_serde_json_value()).collect::<Vec<_>>(),
            }
        )
    }
}

impl Attributive for VideoFrame {
    fn get_attributes_ref(&self) -> &Vec<Attribute> {
        &self.attributes
    }

    fn get_attributes_ref_mut(&mut self) -> &mut Vec<Attribute> {
        &mut self.attributes
    }

    fn take_attributes(&mut self) -> Vec<Attribute> {
        mem::take(&mut self.attributes)
    }

    fn place_attributes(&mut self, mut attributes: Vec<Attribute>) {
        self.attributes.append(&mut attributes);
    }
}

impl VideoFrame {
    pub fn get_objects(&self) -> &HashMap<i64, VideoObjectProxy> {
        &self.objects
    }

    pub fn smart_copy(&self) -> Self {
        let mut frame = self.clone();
        frame.objects.clear();
        for (id, o) in self.get_objects() {
            let copy = o.detached_copy();
            copy.0.write().parent_id = o.get_parent_id();
            frame.objects.insert(*id, copy);
        }
        frame
    }
    pub fn exclude_all_temporary_attributes(&mut self) {
        self.exclude_temporary_attributes();
        self.objects.values().for_each(|o| {
            let mut object_bind = trace!(o.0.write());
            object_bind.exclude_temporary_attributes();
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
            let mut object_bind = trace!(o.0.write());
            object_bind.restore_attributes(attrs);
        }
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
#[repr(C)]
pub struct VideoFrameProxy {
    #[with(Lock)]
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

impl From<VideoFrameProxy> for BelongingVideoFrame {
    fn from(value: VideoFrameProxy) -> Self {
        Self {
            inner: Arc::downgrade(&value.inner.0),
        }
    }
}

impl From<BelongingVideoFrame> for VideoFrameProxy {
    fn from(value: BelongingVideoFrame) -> Self {
        Self {
            inner: value
                .inner
                .upgrade()
                .expect("Frame is dropped, you cannot use attached objects anymore")
                .into(),
        }
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

impl AttributeMethods for VideoFrameProxy {
    fn exclude_temporary_attributes(&self) -> Vec<Attribute> {
        let mut inner = trace!(self.inner.write());
        inner.exclude_temporary_attributes()
    }

    fn restore_attributes(&self, attributes: Vec<Attribute>) {
        let mut inner = trace!(self.inner.write());
        inner.restore_attributes(attributes);
    }

    fn get_attributes(&self) -> Vec<(String, String)> {
        let inner = trace!(self.inner.read_recursive());
        inner.get_attributes()
    }

    fn get_attribute(&self, namespace: &str, name: &str) -> Option<Attribute> {
        let inner = trace!(self.inner.read_recursive());
        inner.get_attribute(namespace, name)
    }

    fn delete_attribute(&self, namespace: &str, name: &str) -> Option<Attribute> {
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

    fn delete_attributes(&self, namespace: &Option<&str>, names: &[&str]) {
        let mut inner = trace!(self.inner.write());
        inner.delete_attributes(namespace, names)
    }

    fn find_attributes(
        &self,
        namespace: &Option<&str>,
        names: &[&str],
        hint: &Option<&str>,
    ) -> Vec<(String, String)> {
        let inner = trace!(self.inner.read_recursive());
        inner.find_attributes(namespace, names, hint)
    }
}

impl ToSerdeJsonValue for VideoFrameProxy {
    fn to_serde_json_value(&self) -> Value {
        let inner = trace!(self.inner.read_recursive()).clone();
        inner.to_serde_json_value()
    }
}

impl VideoFrameProxy {
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
        for obj in objs {
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
        for o in objects {
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

    pub fn get_all_objects(&self) -> Vec<VideoObjectProxy> {
        let inner = trace!(self.inner.read_recursive());
        inner.objects.values().cloned().collect()
    }

    pub fn access_objects(&self, q: &MatchQuery) -> Vec<VideoObjectProxy> {
        let inner = trace!(self.inner.read_recursive());
        let objects = inner.objects.values().cloned().collect::<Vec<_>>();
        drop(inner);
        fiter_map_with_control_flow(objects, |o| q.execute_with_new_context(o))
    }

    pub fn get_json(&self) -> String {
        serde_json::to_string(&self.to_serde_json_value()).unwrap()
    }

    pub fn get_json_pretty(&self) -> String {
        serde_json::to_string_pretty(&self.to_serde_json_value()).unwrap()
    }

    pub fn access_objects_by_id(&self, ids: &[i64]) -> Vec<VideoObjectProxy> {
        let inner = trace!(self.inner.read_recursive());
        let resident_objects = inner.objects.clone();
        drop(inner);

        ids.iter()
            .flat_map(|id| {
                let o = resident_objects.get(id).cloned();
                o
            })
            .collect()
    }

    pub fn delete_objects_by_ids(&self, ids: &[i64]) -> Vec<VideoObjectProxy> {
        self.clear_parent(&MatchQuery::ParentId(IntExpression::OneOf(ids.to_vec())));
        let mut inner = trace!(self.inner.write());
        let objects = mem::take(&mut inner.objects);
        let (retained, removed) = objects.into_iter().partition(|(id, _)| !ids.contains(id));
        inner.objects = retained;
        drop(inner);

        removed.into_values().map(|o| o.detached_copy()).collect()
    }

    pub fn object_exists(&self, id: i64) -> bool {
        let inner = trace!(self.inner.read_recursive());
        inner.objects.contains_key(&id)
    }

    pub fn delete_objects(&self, q: &MatchQuery) -> Vec<VideoObjectProxy> {
        let objs = self.access_objects(q);
        let ids = objs.iter().map(|o| o.get_id()).collect::<Vec<_>>();
        self.delete_objects_by_ids(&ids)
    }

    pub fn get_object(&self, id: i64) -> Option<VideoObjectProxy> {
        let inner = trace!(self.inner.read_recursive());
        inner.objects.get(&id).cloned()
    }

    fn fix_object_owned_frame(&self) {
        self.get_all_objects()
            .iter()
            .for_each(|o| o.attach_to_video_frame(self.clone()));
    }

    pub fn set_draw_label(&self, q: &MatchQuery, label: DrawLabelKind) {
        let objects = self.access_objects(q);
        objects.iter().for_each(|o| match &label {
            DrawLabelKind::OwnLabel(l) => {
                o.set_draw_label(Some(l.clone()));
            }
            DrawLabelKind::ParentLabel(l) => {
                if let Some(p) = o.get_parent().as_ref() {
                    p.set_draw_label(Some(l.clone()));
                }
            }
        });
    }

    pub fn set_parent(
        &self,
        q: &MatchQuery,
        parent: &VideoObjectProxy,
    ) -> anyhow::Result<Vec<VideoObjectProxy>> {
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

        let objects = self.access_objects(q);
        objects
            .iter()
            .try_for_each(|o| o.set_parent(Some(parent.get_id())))?;

        Ok(objects)
    }

    pub fn set_parent_by_id(&self, object_id: i64, parent_id: i64) -> anyhow::Result<()> {
        self.get_object(parent_id).ok_or(anyhow!(
            "Parent object with ID {} does not exist in the frame.",
            parent_id
        ))?;

        let object = self.get_object(object_id).ok_or(anyhow!(
            "Object with ID {} does not exist in the frame.",
            object_id
        ))?;

        object.set_parent(Some(parent_id))?;
        Ok(())
    }

    pub fn clear_parent(&self, q: &MatchQuery) -> Vec<VideoObjectProxy> {
        let objects = self.access_objects(q);
        objects.iter().for_each(|o| {
            o.set_parent(None).unwrap();
        });
        objects
    }

    pub fn get_children(&self, id: i64) -> Vec<VideoObjectProxy> {
        self.access_objects(&MatchQuery::ParentId(IntExpression::EQ(id)))
    }

    pub fn add_object(
        &self,
        object: &VideoObjectProxy,
        policy: IdCollisionResolutionPolicy,
    ) -> anyhow::Result<()> {
        let parent_id_opt = object.get_parent_id();
        if let Some(parent_id) = parent_id_opt {
            if !self.object_exists(parent_id) {
                bail!(
                    "Parent object with ID {} does not exist in the frame.",
                    parent_id
                );
            }
        }

        if !object.is_detached() {
            bail!("Only detached objects can be attached to a frame.");
        }

        let object_id = object.get_id();
        let new_id = self.get_max_object_id() + 1;
        let mut inner = trace!(self.inner.write());
        if inner.objects.contains_key(&object_id) {
            match policy {
                IdCollisionResolutionPolicy::GenerateNewId => {
                    object.set_id(new_id)?;
                    inner.objects.insert(new_id, object.clone());
                }
                IdCollisionResolutionPolicy::Overwrite => {
                    let old = inner.objects.remove(&object_id).unwrap();
                    let mut guard = trace!(old.0.write());
                    guard.frame = None;
                    guard.parent_id = None;
                    inner.objects.insert(object_id, object.clone());
                }
                IdCollisionResolutionPolicy::Error => {
                    bail!("Object with ID {} already exists in the frame.", object_id);
                }
            }
        } else {
            inner.objects.insert(object_id, object.clone());
        }

        object.attach_to_video_frame(self.clone());
        let object_id = object.get_id();
        if object_id > inner.max_object_id {
            inner.max_object_id = object_id;
        }
        Ok(())
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

                    self.add_object(
                        &VideoObjectProxy::from(obj),
                        IdCollisionResolutionPolicy::GenerateNewId,
                    )?;
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

                    self.add_object(
                        &VideoObjectProxy::from(obj),
                        IdCollisionResolutionPolicy::GenerateNewId,
                    )?;
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

                    self.add_object(
                        &VideoObjectProxy::from(obj),
                        IdCollisionResolutionPolicy::GenerateNewId,
                    )?;

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
            let obj = self.get_object(id).ok_or(anyhow!(
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
        source_id: String,
        framerate: String,
        width: i64,
        height: i64,
        content: VideoFrameContent,
        transcoding_method: VideoFrameTranscodingMethod,
        codec: Option<String>,
        keyframe: Option<bool>,
        time_base: (i64, i64),
        pts: i64,
        dts: Option<i64>,
        duration: Option<i64>,
    ) -> Self {
        VideoFrameProxy::from_inner(VideoFrame {
            source_id,
            pts,
            framerate,
            width,
            height,
            time_base: (time_base.0 as i32, time_base.1 as i32),
            dts,
            duration,
            transcoding_method,
            codec,
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

    pub fn set_source_id(&mut self, source_id: String) {
        let mut inner = trace!(self.inner.write());
        inner.source_id = source_id;
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

    pub fn set_framerate(&mut self, framerate: String) {
        let mut inner = trace!(self.inner.write());
        inner.framerate = framerate;
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

    pub fn check_frame_fit(
        objs: &Vec<VideoObjectProxy>,
        max_width: f32,
        max_height: f32,
        bbox_type: VideoObjectBBoxType,
    ) -> Result<(), i64> {
        for obj in objs {
            let bb_opt = match bbox_type {
                VideoObjectBBoxType::Detection => Some(obj.get_detection_box()),
                VideoObjectBBoxType::TrackingInfo => obj.get_track_box(),
            };

            bb_opt
                .map(|bb| {
                    let vertices = bb.get_vertices();
                    for (x, y) in vertices {
                        if x < 0.0 || x > max_width || y < 0.0 || y > max_height {
                            return Err(obj.get_id());
                        }
                    }
                    Ok(())
                })
                .unwrap_or(Ok(()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::draw::DrawLabelKind;
    use crate::match_query::{eq, one_of, MatchQuery};
    use crate::primitives::frame::VideoFrameProxy;
    use crate::primitives::object::{
        IdCollisionResolutionPolicy, VideoObjectBBoxType, VideoObjectBuilder, VideoObjectProxy,
    };
    use crate::primitives::{AttributeMethods, RBBox};
    use crate::test::{gen_empty_frame, gen_frame, gen_object, s};
    use std::sync::Arc;

    #[test]
    fn test_access_objects_by_id() {
        let t = gen_frame();
        let objects = t.access_objects_by_id(&vec![0, 1]);
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].get_id(), 0);
        assert_eq!(objects[1].get_id(), 1);
    }

    #[test]
    fn test_objects_by_id() {
        let t = gen_frame();
        let objects = t.access_objects_by_id(&vec![0, 1]);
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].get_id(), 0);
        assert_eq!(objects[1].get_id(), 1);
    }

    #[test]
    fn test_delete_objects_by_ids() {
        let f = gen_frame();
        f.delete_objects_by_ids(&[0, 1]);
        let objects = f.get_all_objects();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].get_id(), 2);
    }

    #[test]
    fn test_parent_cleared_when_delete_objects_by_ids() {
        let f = gen_frame();
        f.delete_objects_by_ids(&[0]);
        let o = f.get_object(1).unwrap();
        assert!(o.get_parent().is_none());
        let o = f.get_object(2).unwrap();
        assert!(o.get_parent().is_none());

        let f = gen_frame();
        f.delete_objects_by_ids(&[1]);
        let o = f.get_object(2).unwrap();
        assert!(o.get_parent().is_some());
    }

    #[test]
    fn test_parent_cleared_when_delete_objects_by_query() {
        let f = gen_frame();

        let o = f.get_object(0).unwrap();
        assert!(o.get_frame().is_some());

        let removed = f.delete_objects(&MatchQuery::Id(eq(0)));
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
        let objs = f.delete_objects(&MatchQuery::Idle);
        assert_eq!(objs.len(), 3);
        let objects = f.get_all_objects();
        assert!(objects.is_empty());
    }

    #[test]
    fn test_parent_not_added_to_frame() {
        let parent = VideoObjectProxy::from(
            VideoObjectBuilder::default()
                .parent_id(None)
                .namespace(s("some-model"))
                .label(s("some-label"))
                .id(155)
                .detection_box(RBBox::new(0.0, 0.0, 0.0, 0.0, None).try_into().unwrap())
                .build()
                .unwrap(),
        );
        let frame = gen_frame();
        let obj = frame.get_object(0).unwrap();
        assert!(obj.set_parent(Some(parent.get_id())).is_err());
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
        frame.set_draw_label(&MatchQuery::Idle, DrawLabelKind::ParentLabel(s("draw")));
        let parent_object = frame.get_object(0).unwrap();
        assert_eq!(parent_object.calculate_draw_label(), s("draw"));

        let child_object = frame.get_object(1).unwrap();
        assert_ne!(child_object.calculate_draw_label(), s("draw"));
    }

    #[test]
    fn set_own_draw_label() {
        let frame = gen_frame();
        frame.set_draw_label(&MatchQuery::Idle, DrawLabelKind::OwnLabel(s("draw")));
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
        frame.clear_parent(&MatchQuery::Id(one_of(&[1, 2])));
        let obj = frame.get_object(1).unwrap();
        assert!(obj.get_parent().is_none());
        let obj = frame.get_object(2).unwrap();
        assert!(obj.get_parent().is_none());

        frame.set_parent(&MatchQuery::Id(one_of(&[1, 2])), &parent)?;
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
        let p = VideoObjectProxy::from(
            VideoObjectBuilder::default()
                .id(11)
                .namespace(s("random"))
                .label(s("something"))
                .detection_box(RBBox::new(1.0, 2.0, 10.0, 20.0, None).try_into().unwrap())
                .build()
                .unwrap(),
        );

        let o = VideoObjectProxy::from(
            VideoObjectBuilder::default()
                .id(23)
                .namespace(s("random"))
                .label(s("something"))
                .detection_box(RBBox::new(1.0, 2.0, 10.0, 20.0, None).try_into().unwrap())
                .parent_id(Some(p.get_id()))
                .build()
                .unwrap(),
        );

        let f = gen_frame();
        f.add_object(&o, IdCollisionResolutionPolicy::Error)
            .unwrap();
    }

    #[test]
    fn set_detached_parent_as_parent() {
        let f = gen_frame();
        let o = VideoObjectProxy::from(
            VideoObjectBuilder::default()
                .id(11)
                .namespace(s("random"))
                .label(s("something"))
                .detection_box(RBBox::new(1.0, 2.0, 10.0, 20.0, None).try_into().unwrap())
                .build()
                .unwrap(),
        );
        assert!(f.set_parent(&MatchQuery::Id(eq(0)), &o).is_err());
    }

    #[test]
    fn set_wrong_parent_as_parent() {
        let f1 = gen_frame();
        let f2 = gen_frame();
        let f1o = f1.get_object(0).unwrap();
        assert!(f2.set_parent(&MatchQuery::Id(eq(1)), &f1o).is_err());
    }

    #[test]
    fn normally_transfer_parent() -> anyhow::Result<()> {
        let f1 = gen_frame();
        let f2 = gen_frame();
        let mut o = f1.delete_objects_by_ids(&[0]).pop().unwrap();
        assert!(o.get_frame().is_none());
        _ = o.set_id(33);
        f2.add_object(&o, IdCollisionResolutionPolicy::Error)
            .unwrap();
        o = f2.get_object(33).unwrap();
        f2.set_parent(&MatchQuery::Id(eq(1)), &o)?;
        Ok(())
    }

    #[test]
    fn ensure_object_spoiled_when_frame_is_dropped() {
        let frame = gen_frame();
        let object = frame.get_object(0).unwrap();
        assert!(
            !object.is_spoiled(),
            "Object is expected to be in a normal state."
        );
        drop(frame);
        assert!(object.is_spoiled(), "Object is expected to be spoiled");
    }

    #[test]
    #[should_panic(expected = "Only detached objects can be attached to a frame.")]
    fn ensure_spoiled_object_cannot_be_added() {
        let frame = gen_frame();
        frame
            .add_object(&gen_object(111), IdCollisionResolutionPolicy::Error)
            .unwrap();
        let old_object = frame.get_object(111).unwrap();
        drop(frame);
        let frame = gen_frame();
        assert!(old_object.is_spoiled(), "Object is expected to be spoiled");
        frame
            .add_object(&old_object, IdCollisionResolutionPolicy::Error)
            .unwrap();
    }

    #[test]
    fn deleted_objects_clean() {
        let frame = gen_frame();
        let removed = frame.delete_objects_by_ids(&[0]).pop().unwrap();
        assert!(removed.is_detached());
        assert!(removed.get_parent().is_none());
    }

    #[test]
    fn deep_copy() {
        let f = gen_frame();
        let new_f = f.smart_copy();

        // check that objects are copied
        let o = f.get_object(0).unwrap();
        let new_o = new_f.get_object(0).unwrap();
        let label = s("new label");
        o.set_label(label.clone());
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
    fn add_objects_test_policy_error() {
        let frame = gen_empty_frame();

        let object = gen_object(0);
        frame
            .add_object(&object, IdCollisionResolutionPolicy::Error)
            .unwrap();

        let object = gen_object(0);
        assert!(frame
            .add_object(&object, IdCollisionResolutionPolicy::Error)
            .is_err());
    }

    #[test]
    fn add_objects_test_policy_generate_new_id() {
        let frame = gen_empty_frame();

        let object = gen_object(0);
        frame
            .add_object(&object, IdCollisionResolutionPolicy::GenerateNewId)
            .unwrap();

        let object = gen_object(0);
        frame
            .add_object(&object, IdCollisionResolutionPolicy::GenerateNewId)
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
            .add_object(&object, IdCollisionResolutionPolicy::Overwrite)
            .unwrap();

        let object = gen_object(0);
        assert!(frame
            .add_object(&object, IdCollisionResolutionPolicy::Overwrite)
            .is_ok());

        assert_eq!(frame.get_max_object_id(), 0);
        let objs = frame.get_all_objects();
        assert_eq!(objs.len(), 1);
    }

    #[test]
    fn check_frame_fit() {
        let objects = vec![gen_object(0), gen_object(1), gen_object(2), gen_object(3)];
        let res = VideoFrameProxy::check_frame_fit(
            &objects,
            100.0,
            100.0,
            VideoObjectBBoxType::Detection,
        );
        assert!(matches!(res, Err(0)));

        let res = VideoFrameProxy::check_frame_fit(
            &objects,
            300.0,
            300.0,
            VideoObjectBBoxType::TrackingInfo,
        );
        assert!(res.is_ok());
    }
}

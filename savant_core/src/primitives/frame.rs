use crate::draw::DrawLabelKind;
use crate::match_query::{and, IntExpression, MatchQuery, StringExpression};
use crate::message::Message;
use crate::primitives::attribute::AttributeMethods;
use crate::primitives::frame_update::VideoFrameUpdate;
use crate::primitives::object::{
    IdCollisionResolutionPolicy, VideoObject, VideoObjectBBoxTransformation, VideoObjectProxy,
};
use crate::primitives::{Attribute, Attributive};
use crate::to_json_value::ToSerdeJsonValue;
use anyhow::{anyhow, bail};
use derive_builder::Builder;
use parking_lot::RwLock;
use rkyv::{with::Skip, Archive, Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::mem;
use std::sync::{Arc, Weak};

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
                serde_json::json!({ "internal": Value::Null })
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
    pub source_id: String,
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
    pub content: VideoFrameContent,
    #[builder(setter(skip))]
    pub transformations: Vec<VideoFrameTransformation>,
    #[builder(setter(skip))]
    pub attributes: HashMap<(String, String), Attribute>,
    #[builder(setter(skip))]
    pub offline_objects: HashMap<i64, VideoObject>,
    #[with(Skip)]
    #[builder(setter(skip))]
    pub(crate) resident_objects: HashMap<i64, Arc<RwLock<VideoObject>>>,
    #[with(Skip)]
    #[builder(setter(skip))]
    pub(crate) max_object_id: i64,
}

impl Default for VideoFrame {
    fn default() -> Self {
        Self {
            source_id: String::new(),
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
            content: VideoFrameContent::None,
            transformations: Vec::new(),
            attributes: HashMap::new(),
            offline_objects: HashMap::new(),
            resident_objects: HashMap::new(),
            max_object_id: 0,
        }
    }
}

impl ToSerdeJsonValue for VideoFrame {
    fn to_serde_json_value(&self) -> Value {
        use crate::version;
        serde_json::json!(
            {
                "version": version(),
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
                "attributes": self.attributes.values().map(|v| v.to_serde_json_value()).collect::<Vec<_>>(),
                "objects": self.resident_objects.values().map(|o| o.read_recursive().to_serde_json_value()).collect::<Vec<_>>(),
            }
        )
    }
}

impl Attributive for Box<VideoFrame> {
    fn get_attributes_ref(&self) -> &HashMap<(String, String), Attribute> {
        &self.attributes
    }

    fn get_attributes_ref_mut(&mut self) -> &mut HashMap<(String, String), Attribute> {
        &mut self.attributes
    }

    fn take_attributes(&mut self) -> HashMap<(String, String), Attribute> {
        mem::take(&mut self.attributes)
    }

    fn place_attributes(&mut self, attributes: HashMap<(String, String), Attribute>) {
        self.attributes = attributes;
    }
}

impl VideoFrame {
    pub(crate) fn preserve(&mut self) {
        self.offline_objects = self
            .resident_objects
            .iter()
            .map(|(id, o)| (*id, o.read_recursive().clone()))
            .collect();
    }

    pub(crate) fn restore(&mut self) {
        self.resident_objects = mem::take(&mut self.offline_objects)
            .into_iter()
            .map(|(id, o)| (id, Arc::new(RwLock::new(o))))
            .collect();
    }

    pub fn deep_copy(&self) -> Self {
        let mut frame = self.clone();
        frame.preserve();
        frame.restore();
        frame
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct VideoFrameProxy {
    pub(crate) inner: Arc<RwLock<Box<VideoFrame>>>,
    pub(crate) is_parallelized: bool,
}

#[derive(Clone)]
pub struct BelongingVideoFrame {
    pub(crate) inner: Weak<RwLock<Box<VideoFrame>>>,
}

impl Debug for BelongingVideoFrame {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.inner.upgrade() {
            Some(inner) => f
                .debug_struct("BelongingVideoFrame")
                .field("stream_id", &inner.read_recursive().source_id)
                .finish(),
            None => f.debug_struct("Unset").finish(),
        }
    }
}

impl From<VideoFrameProxy> for BelongingVideoFrame {
    fn from(value: VideoFrameProxy) -> Self {
        Self {
            inner: Arc::downgrade(&value.inner),
        }
    }
}

impl From<BelongingVideoFrame> for VideoFrameProxy {
    fn from(value: BelongingVideoFrame) -> Self {
        Self {
            inner: value
                .inner
                .upgrade()
                .expect("Frame is dropped, you cannot use attached objects anymore"),
            is_parallelized: false,
        }
    }
}

impl From<&BelongingVideoFrame> for VideoFrameProxy {
    fn from(value: &BelongingVideoFrame) -> Self {
        Self {
            inner: value
                .inner
                .upgrade()
                .expect("Frame is dropped, you cannot use attached objects anymore"),
            is_parallelized: false,
        }
    }
}

impl AttributeMethods for VideoFrameProxy {
    fn exclude_temporary_attributes(&self) -> Vec<Attribute> {
        let mut inner = self.inner.write();
        inner.exclude_temporary_attributes()
    }

    fn restore_attributes(&self, attributes: Vec<Attribute>) {
        let mut inner = self.inner.write();
        inner.restore_attributes(attributes);
    }

    fn get_attributes(&self) -> Vec<(String, String)> {
        let inner = self.inner.read_recursive();
        inner.get_attributes()
    }

    fn get_attribute(&self, namespace: String, name: String) -> Option<Attribute> {
        let inner = self.inner.read_recursive();
        inner.get_attribute(namespace, name)
    }

    fn delete_attribute(&self, namespace: String, name: String) -> Option<Attribute> {
        let mut inner = self.inner.write();
        inner.delete_attribute(namespace, name)
    }

    fn set_attribute(&self, attribute: Attribute) -> Option<Attribute> {
        let mut inner = self.inner.write();
        inner.set_attribute(attribute)
    }

    fn clear_attributes(&self) {
        let mut inner = self.inner.write();
        inner.clear_attributes()
    }

    fn delete_attributes(&self, namespace: Option<String>, names: Vec<String>) {
        let mut inner = self.inner.write();
        inner.delete_attributes(namespace, names)
    }

    fn find_attributes(
        &self,
        namespace: Option<String>,
        names: Vec<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        let inner = self.inner.read_recursive();
        inner.find_attributes(namespace, names, hint)
    }
}

impl ToSerdeJsonValue for VideoFrameProxy {
    fn to_serde_json_value(&self) -> Value {
        let inner = self.inner.read_recursive().clone();
        inner.to_serde_json_value()
    }
}

impl VideoFrameProxy {
    pub fn transform_geometry(&self, ops: &Vec<VideoObjectBBoxTransformation>) {
        let objs = self.access_objects(&MatchQuery::Idle);
        for obj in objs {
            obj.transform_geometry(ops);
        }
    }

    pub fn deep_copy(&self) -> Self {
        let inner = self.inner.read_recursive();
        let inner_copy = inner.deep_copy();
        drop(inner);
        Self::from_inner(inner_copy)
    }

    pub fn get_inner(&self) -> Arc<RwLock<Box<VideoFrame>>> {
        self.inner.clone()
    }

    pub(crate) fn from_inner(inner: VideoFrame) -> Self {
        VideoFrameProxy {
            inner: Arc::new(RwLock::new(Box::new(inner))),
            is_parallelized: false,
        }
    }

    pub fn access_objects(&self, q: &MatchQuery) -> Vec<VideoObjectProxy> {
        let inner = self.inner.read_recursive();
        let resident_objects = inner.resident_objects.clone();
        drop(inner);

        // if self.is_parallelized {
        //     resident_objects
        //         .par_iter()
        //         .filter_map(|(_, o)| {
        //             let obj = VideoObjectProxy::from_arced_inner_object(o.clone());
        //             if q.execute_with_new_context(&obj) {
        //                 Some(obj)
        //             } else {
        //                 None
        //             }
        //         })
        //         .collect()
        // } else {
        resident_objects
            .iter()
            .filter_map(|(_, o)| {
                let obj = VideoObjectProxy::from(o.clone());
                if q.execute_with_new_context(&obj) {
                    Some(obj)
                } else {
                    None
                }
            })
            .collect()
        // }
    }

    pub fn get_json(&self) -> String {
        serde_json::to_string(&self.to_serde_json_value()).unwrap()
    }

    pub fn get_json_pretty(&self) -> String {
        serde_json::to_string_pretty(&self.to_serde_json_value()).unwrap()
    }

    pub fn access_objects_by_id(&self, ids: &[i64]) -> Vec<VideoObjectProxy> {
        let inner = self.inner.read_recursive();
        let resident_objects = inner.resident_objects.clone();
        drop(inner);

        ids.iter()
            .flat_map(|id| {
                let o = resident_objects
                    .get(id)
                    .map(|o| VideoObjectProxy::from(o.clone()));
                o
            })
            .collect()
    }

    pub fn delete_objects_by_ids(&self, ids: &[i64]) -> Vec<VideoObjectProxy> {
        self.clear_parent(&MatchQuery::ParentId(IntExpression::OneOf(ids.to_vec())));
        let mut inner = self.inner.write();
        let objects = mem::take(&mut inner.resident_objects);
        let (retained, removed) = objects.into_iter().partition(|(id, _)| !ids.contains(id));
        inner.resident_objects = retained;
        drop(inner);

        removed
            .into_values()
            .map(|o| {
                let o = VideoObjectProxy::from(o);
                o.detached_copy()
            })
            .collect()
    }

    pub fn object_exists(&self, id: i64) -> bool {
        let inner = self.inner.read_recursive();
        inner.resident_objects.contains_key(&id)
    }

    pub fn delete_objects(&self, q: &MatchQuery) -> Vec<VideoObjectProxy> {
        let objs = self.access_objects(q);
        let ids = objs.iter().map(|o| o.get_id()).collect::<Vec<_>>();
        self.delete_objects_by_ids(&ids)
    }

    pub fn get_object(&self, id: i64) -> Option<VideoObjectProxy> {
        let inner = self.inner.read_recursive();
        inner
            .resident_objects
            .get(&id)
            .map(|o| VideoObjectProxy::from(o.clone()))
    }

    pub fn make_snapshot(&self) {
        let mut inner = self.inner.write();
        inner.preserve();
    }

    fn fix_object_owned_frame(&self) {
        self.access_objects(&MatchQuery::Idle)
            .iter()
            .for_each(|o| o.attach_to_video_frame(self.clone()));
    }

    pub fn restore_from_snapshot(&self) {
        {
            let inner = self.inner.write();
            let resident_objects = inner.resident_objects.clone();
            drop(inner);

            resident_objects.iter().for_each(|(_, o)| {
                let mut o = o.write();
                o.frame = None
            });

            let mut inner = self.inner.write();
            inner.restore();
        }
        self.fix_object_owned_frame();
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

    pub fn set_parent(&self, q: &MatchQuery, parent: &VideoObjectProxy) -> Vec<VideoObjectProxy> {
        let frame = parent.get_frame();
        assert!(
            frame.is_some() && Arc::ptr_eq(&frame.unwrap().inner, &self.inner),
            "Parent object must be attached to the same frame"
        );
        let objects = self.access_objects(q);
        objects.iter().for_each(|o| {
            o.set_parent(Some(parent.get_id()));
        });

        objects
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

        object.set_parent(Some(parent_id));
        Ok(())
    }

    pub fn clear_parent(&self, q: &MatchQuery) -> Vec<VideoObjectProxy> {
        let objects = self.access_objects(q);
        objects.iter().for_each(|o| {
            o.set_parent(None);
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
        let mut inner = self.inner.write();
        if inner.resident_objects.contains_key(&object_id) {
            match policy {
                IdCollisionResolutionPolicy::GenerateNewId => {
                    object.set_id(new_id)?;
                    inner.resident_objects.insert(new_id, object.inner.clone());
                }
                IdCollisionResolutionPolicy::Overwrite => {
                    let old = inner.resident_objects.remove(&object_id).unwrap();
                    old.write().frame = None;
                    old.write().parent_id = None;
                    inner
                        .resident_objects
                        .insert(object_id, object.inner.clone());
                }
                IdCollisionResolutionPolicy::Error => {
                    bail!("Object with ID {} already exists in the frame.", object_id);
                }
            }
        } else {
            inner
                .resident_objects
                .insert(object_id, object.inner.clone());
        }

        object.attach_to_video_frame(self.clone());
        let object_id = object.get_id();
        if object_id > inner.max_object_id {
            inner.max_object_id = object_id;
        }
        Ok(())
    }

    pub fn get_max_object_id(&self) -> i64 {
        let inner = self.inner.read_recursive();
        inner.max_object_id
    }

    pub fn update_objects(&self, update: &VideoFrameUpdate) -> anyhow::Result<()> {
        use crate::primitives::frame_update::ObjectUpdatePolicy::*;
        let other_inner = update.objects.clone();

        let object_query = |o: &VideoObject| {
            and![
                MatchQuery::Label(StringExpression::EQ(o.label.clone())),
                MatchQuery::Namespace(StringExpression::EQ(o.namespace.clone()))
            ]
        };

        match &update.object_collision_resolution_policy {
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

    pub fn update_attributes(&self, update: &VideoFrameUpdate) -> anyhow::Result<()> {
        use crate::primitives::frame_update::AttributeUpdatePolicy::*;
        match &update.attribute_collision_resolution_policy {
            ReplaceWithForeign => {
                let mut inner = self.inner.write();
                let other_inner = update.get_attributes().clone();
                inner.attributes.extend(
                    other_inner
                        .into_iter()
                        .map(|a| ((a.namespace.clone(), a.name.clone()), a)),
                );
            }
            KeepOwn => {
                let mut inner = self.inner.write();
                let other_inner = update.get_attributes().clone();
                for attr in other_inner {
                    let key = (attr.namespace.clone(), attr.name.clone());
                    inner.attributes.entry(key).or_insert(attr);
                }
            }
            Error => {
                let mut inner = self.inner.write();
                let other_inner = update.get_attributes().clone();
                for attr in other_inner {
                    let key = (attr.namespace.clone(), attr.name.clone());
                    if inner.attributes.contains_key(&key) {
                        bail!(
                            "Attribute with name '{}' created by '{}' already exists in the frame.",
                            key.1,
                            key.0
                        );
                    }
                    inner.attributes.insert(key, attr);
                }
            }
        }

        Ok(())
    }

    pub fn update(&self, update: &VideoFrameUpdate) -> anyhow::Result<()> {
        self.update_objects(update)?;
        self.update_attributes(update)?;
        Ok(())
    }
    pub fn set_parallelized(&mut self, is_parallelized: bool) {
        self.is_parallelized = is_parallelized;
    }

    pub fn get_parallelized(&self) -> bool {
        self.is_parallelized
    }

    pub fn memory_handle(&self) -> usize {
        self as *const Self as usize
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
            content,
            ..Default::default()
        })
    }

    pub fn to_message(&self) -> Message {
        Message::video_frame(self)
    }

    pub fn get_source_id(&self) -> String {
        self.inner.read_recursive().source_id.clone()
    }

    pub fn set_source_id(&mut self, source_id: String) {
        let mut inner = self.inner.write();
        inner.source_id = source_id;
    }

    pub fn set_time_base(&mut self, time_base: (i32, i32)) {
        let mut inner = self.inner.write();
        inner.time_base = time_base;
    }
    pub fn get_time_base(&self) -> (i32, i32) {
        self.inner.read_recursive().time_base
    }

    pub fn get_pts(&self) -> i64 {
        self.inner.read_recursive().pts
    }
    pub fn set_pts(&mut self, pts: i64) {
        assert!(pts >= 0, "pts must be greater than or equal to 0");
        let mut inner = self.inner.write();
        inner.pts = pts;
    }

    pub fn get_framerate(&self) -> String {
        self.inner.read_recursive().framerate.clone()
    }

    pub fn set_framerate(&mut self, framerate: String) {
        let mut inner = self.inner.write();
        inner.framerate = framerate;
    }

    pub fn get_width(&self) -> i64 {
        self.inner.read_recursive().width
    }

    pub fn set_width(&mut self, width: i64) {
        assert!(width > 0, "width must be greater than 0");
        let mut inner = self.inner.write();
        inner.width = width;
    }

    pub fn get_height(&self) -> i64 {
        self.inner.read_recursive().height
    }

    pub fn set_height(&mut self, height: i64) {
        assert!(height > 0, "height must be greater than 0");
        let mut inner = self.inner.write();
        inner.height = height;
    }

    pub fn get_dts(&self) -> Option<i64> {
        let inner = self.inner.read_recursive();
        inner.dts
    }

    pub fn set_dts(&mut self, dts: Option<i64>) {
        assert!(
            dts.is_none() || dts.unwrap() >= 0,
            "dts must be greater than or equal to 0"
        );
        let mut inner = self.inner.write();
        inner.dts = dts;
    }

    pub fn get_duration(&self) -> Option<i64> {
        let inner = self.inner.read_recursive();
        inner.duration
    }

    pub fn set_duration(&mut self, duration: Option<i64>) {
        assert!(
            duration.is_none() || duration.unwrap() >= 0,
            "duration must be greater than or equal to 0"
        );
        let mut inner = self.inner.write();
        inner.duration = duration;
    }

    pub fn get_transcoding_method(&self) -> VideoFrameTranscodingMethod {
        let inner = self.inner.read_recursive();
        inner.transcoding_method.clone()
    }

    pub fn set_transcoding_method(&mut self, transcoding_method: VideoFrameTranscodingMethod) {
        let mut inner = self.inner.write();
        inner.transcoding_method = transcoding_method;
    }

    pub fn get_codec(&self) -> Option<String> {
        let inner = self.inner.read_recursive();
        inner.codec.clone()
    }

    pub fn set_codec(&mut self, codec: Option<String>) {
        let mut inner = self.inner.write();
        inner.codec = codec;
    }

    pub fn clear_transformations(&mut self) {
        let mut inner = self.inner.write();
        inner.transformations.clear();
    }

    pub fn add_transformation(&mut self, transformation: VideoFrameTransformation) {
        let mut inner = self.inner.write();
        inner.transformations.push(transformation);
    }

    pub fn get_transformations(&self) -> Vec<VideoFrameTransformation> {
        let inner = self.inner.read_recursive();
        inner.transformations.clone()
    }

    pub fn get_keyframe(&self) -> Option<bool> {
        let inner = self.inner.read_recursive();
        inner.keyframe
    }

    pub fn set_keyframe(&mut self, keyframe: Option<bool>) {
        let mut inner = self.inner.write();
        inner.keyframe = keyframe;
    }

    pub fn get_content(&self) -> VideoFrameContent {
        let inner = self.inner.read_recursive();
        inner.content.clone()
    }

    pub fn set_content(&mut self, content: VideoFrameContent) {
        let mut inner = self.inner.write();
        inner.content = content;
    }

    pub fn clear_objects(&self) {
        let mut frame = self.inner.write();
        frame.resident_objects.clear();
    }
}

#[cfg(test)]
mod tests {
    use crate::draw::DrawLabelKind;
    use crate::match_query::{eq, one_of, MatchQuery};
    use crate::primitives::attribute_value::AttributeValueVariant;
    use crate::primitives::object::{
        IdCollisionResolutionPolicy, VideoObjectBuilder, VideoObjectProxy,
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
    fn test_get_attribute() {
        let t = gen_frame();
        let attribute = t.get_attribute("system".to_string(), "test".to_string());
        assert!(attribute.is_some());
        let v = attribute.as_ref().unwrap().get_values().get(0).unwrap();
        assert!(
            matches!(v.get_value(), AttributeValueVariant::String(s) if s == &String::from("1"))
        );
    }

    #[test]
    fn test_find_attributes() {
        let t = gen_frame();
        let mut attributes = t.find_attributes(Some("system".to_string()), vec![], None);
        attributes.sort();
        assert_eq!(attributes.len(), 2);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));
        assert_eq!(attributes[1], ("system".to_string(), "test2".to_string()));

        let attributes =
            t.find_attributes(Some("system".to_string()), vec!["test".to_string()], None);
        assert_eq!(attributes.len(), 1);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));

        let attributes = t.find_attributes(
            Some("system".to_string()),
            vec!["test".to_string()],
            Some("test".to_string()),
        );
        assert_eq!(attributes.len(), 1);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));

        let mut attributes = t.find_attributes(None, vec![], Some("test".to_string()));
        attributes.sort();
        assert_eq!(attributes.len(), 2);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));
        assert_eq!(attributes[1], ("system".to_string(), "test2".to_string()));
    }

    #[test]
    fn test_delete_objects_by_ids() {
        let f = gen_frame();
        f.delete_objects_by_ids(&[0, 1]);
        let objects = f.access_objects(&MatchQuery::Idle);
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
        let objects = f.access_objects(&MatchQuery::Idle);
        assert!(objects.is_empty());
    }

    #[test]
    fn test_snapshot_simple() {
        let f = gen_frame();
        f.make_snapshot();
        let o = f.access_objects_by_id(&vec![0]).pop().unwrap();
        o.set_namespace(s("modified"));
        f.restore_from_snapshot();
        let o = f.access_objects_by_id(&vec![0]).pop().unwrap();
        assert_eq!(o.get_namespace(), s("test"));
    }

    #[test]
    #[should_panic]
    fn test_panic_snapshot_no_parent_added_to_frame() {
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
        obj.set_parent(Some(parent.get_id()));
        frame.make_snapshot();
    }

    #[test]
    fn test_snapshot_with_parent_added_to_frame() {
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
        frame
            .add_object(&parent, IdCollisionResolutionPolicy::Error)
            .unwrap();
        let obj = frame.get_object(0).unwrap();
        obj.set_parent(Some(parent.get_id()));
        frame.make_snapshot();
        frame.restore_from_snapshot();
        let obj = frame.get_object(0).unwrap();
        assert_eq!(obj.get_parent().unwrap().inner.read_recursive().id, 155);
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
        assert_eq!(parent_object.get_draw_label(), s("draw"));

        let child_object = frame.get_object(1).unwrap();
        assert_ne!(child_object.get_draw_label(), s("draw"));
    }

    #[test]
    fn set_own_draw_label() {
        let frame = gen_frame();
        frame.set_draw_label(&MatchQuery::Idle, DrawLabelKind::OwnLabel(s("draw")));
        let parent_object = frame.get_object(0).unwrap();
        assert_eq!(parent_object.get_draw_label(), s("draw"));

        let child_object = frame.get_object(1).unwrap();
        assert_eq!(child_object.get_draw_label(), s("draw"));

        let child_object = frame.get_object(2).unwrap();
        assert_eq!(child_object.get_draw_label(), s("draw"));
    }

    #[test]
    fn test_set_clear_parent_ops() {
        let frame = gen_frame();
        let parent = frame.get_object(0).unwrap();
        frame.clear_parent(&MatchQuery::Id(one_of(&[1, 2])));
        let obj = frame.get_object(1).unwrap();
        assert!(obj.get_parent().is_none());
        let obj = frame.get_object(2).unwrap();
        assert!(obj.get_parent().is_none());

        frame.set_parent(&MatchQuery::Id(one_of(&[1, 2])), &parent);
        let obj = frame.get_object(1).unwrap();
        assert!(obj.get_parent().is_some());

        let obj = frame.get_object(2).unwrap();
        assert!(obj.get_parent().is_some());
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
    #[should_panic]
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
        f.set_parent(&MatchQuery::Id(eq(0)), &o);
    }

    #[test]
    #[should_panic]
    fn set_wrong_parent_as_parent() {
        let f1 = gen_frame();
        let f2 = gen_frame();
        let f1o = f1.get_object(0).unwrap();
        f2.set_parent(&MatchQuery::Id(eq(1)), &f1o);
    }

    #[test]
    fn normally_transfer_parent() {
        let f1 = gen_frame();
        let f2 = gen_frame();
        let mut o = f1.delete_objects_by_ids(&[0]).pop().unwrap();
        assert!(o.get_frame().is_none());
        _ = o.set_id(33);
        f2.add_object(&o, IdCollisionResolutionPolicy::Error)
            .unwrap();
        o = f2.get_object(33).unwrap();
        f2.set_parent(&MatchQuery::Id(eq(1)), &o);
    }

    #[test]
    fn frame_is_properly_set_after_snapshotting() {
        let frame = gen_frame();
        frame.make_snapshot();
        frame.restore_from_snapshot();
        let o = frame.get_object(0).unwrap();
        let saved_frame = o.get_frame();
        assert!(saved_frame.is_some());
        assert!(Arc::ptr_eq(&frame.inner, &saved_frame.unwrap().inner));
    }

    #[test]
    fn ensure_owned_objects_detached_after_snapshot() {
        let frame = gen_frame();
        frame
            .add_object(&gen_object(111), IdCollisionResolutionPolicy::Error)
            .unwrap();
        frame.make_snapshot();
        let object = frame.get_object(111).unwrap();
        assert!(!object.is_detached(), "Object is expected to be attached");

        frame.restore_from_snapshot();
        assert!(object.is_detached(), "Object is expected to be detached");

        let o = frame.get_object(0).unwrap();
        assert!(!o.is_detached(), "Object is expected to be attached");
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
        let new_f = f.deep_copy();

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
        let objs = frame.access_objects(&MatchQuery::Idle);
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
        let objs = frame.access_objects(&MatchQuery::Idle);
        assert_eq!(objs.len(), 1);
    }
}

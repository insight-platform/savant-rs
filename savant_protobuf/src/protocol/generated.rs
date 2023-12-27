#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BytesAttributeValueVariant {
    #[prost(int64, repeated, tag = "1")]
    pub dims: ::prost::alloc::vec::Vec<i64>,
    #[prost(bytes = "vec", tag = "2")]
    pub data: ::prost::alloc::vec::Vec<u8>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StringAttributeValueVariant {
    #[prost(string, tag = "1")]
    pub data: ::prost::alloc::string::String,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StringVectorAttributeValueVariant {
    #[prost(string, repeated, tag = "1")]
    pub data: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct IntegerAttributeValueVariant {
    #[prost(int64, tag = "1")]
    pub data: i64,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct IntegerVectorAttributeValueVariant {
    #[prost(int64, repeated, tag = "1")]
    pub data: ::prost::alloc::vec::Vec<i64>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FloatAttributeValueVariant {
    #[prost(double, tag = "1")]
    pub data: f64,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FloatVectorAttributeValueVariant {
    #[prost(double, repeated, tag = "1")]
    pub data: ::prost::alloc::vec::Vec<f64>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BooleanAttributeValueVariant {
    #[prost(bool, tag = "1")]
    pub data: bool,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BooleanVectorAttributeValueVariant {
    #[prost(bool, repeated, tag = "1")]
    pub data: ::prost::alloc::vec::Vec<bool>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BoundingBox {
    #[prost(float, tag = "1")]
    pub xc: f32,
    #[prost(float, tag = "2")]
    pub yc: f32,
    #[prost(float, tag = "3")]
    pub width: f32,
    #[prost(float, tag = "4")]
    pub height: f32,
    #[prost(float, optional, tag = "5")]
    pub angle: ::core::option::Option<f32>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BoundingBoxAttributeValueVariant {
    #[prost(message, optional, tag = "1")]
    pub data: ::core::option::Option<BoundingBox>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BoundingBoxVectorAttributeValueVariant {
    #[prost(message, repeated, tag = "1")]
    pub data: ::prost::alloc::vec::Vec<BoundingBox>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Point {
    #[prost(float, tag = "1")]
    pub x: f32,
    #[prost(float, tag = "2")]
    pub y: f32,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PointAttributeValueVariant {
    #[prost(message, optional, tag = "1")]
    pub data: ::core::option::Option<Point>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PointVectorAttributeValueVariant {
    #[prost(message, repeated, tag = "1")]
    pub data: ::prost::alloc::vec::Vec<Point>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PolygonalAreaTag {
    #[prost(string, optional, tag = "1")]
    pub tag: ::core::option::Option<::prost::alloc::string::String>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PolygonalAreaTags {
    #[prost(message, repeated, tag = "1")]
    pub tags: ::prost::alloc::vec::Vec<PolygonalAreaTag>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PolygonalArea {
    #[prost(message, repeated, tag = "1")]
    pub points: ::prost::alloc::vec::Vec<Point>,
    #[prost(message, optional, tag = "2")]
    pub tags: ::core::option::Option<PolygonalAreaTags>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PolygonAttributeValueVariant {
    #[prost(message, optional, tag = "1")]
    pub data: ::core::option::Option<PolygonalArea>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PolygonVectorAttributeValueVariant {
    #[prost(message, repeated, tag = "1")]
    pub data: ::prost::alloc::vec::Vec<PolygonalArea>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct IntersectionEdge {
    #[prost(uint64, tag = "1")]
    pub id: u64,
    #[prost(string, optional, tag = "2")]
    pub tag: ::core::option::Option<::prost::alloc::string::String>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Intersection {
    #[prost(enumeration = "IntersectionKind", tag = "1")]
    pub kind: i32,
    #[prost(message, repeated, tag = "2")]
    pub edges: ::prost::alloc::vec::Vec<IntersectionEdge>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct IntersectionAttributeValueVariant {
    #[prost(message, optional, tag = "1")]
    pub data: ::core::option::Option<Intersection>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NoneAttributeValueVariant {}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AttributeValue {
    #[prost(float, optional, tag = "1")]
    pub confidence: ::core::option::Option<f32>,
    #[prost(
        oneof = "attribute_value::Value",
        tags = "2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18"
    )]
    pub value: ::core::option::Option<attribute_value::Value>,
}
/// Nested message and enum types in `AttributeValue`.
pub mod attribute_value {
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Value {
        #[prost(message, tag = "2")]
        Bytes(super::BytesAttributeValueVariant),
        #[prost(message, tag = "3")]
        String(super::StringAttributeValueVariant),
        #[prost(message, tag = "4")]
        StringVector(super::StringVectorAttributeValueVariant),
        #[prost(message, tag = "5")]
        Integer(super::IntegerAttributeValueVariant),
        #[prost(message, tag = "6")]
        IntegerVector(super::IntegerVectorAttributeValueVariant),
        #[prost(message, tag = "7")]
        Float(super::FloatAttributeValueVariant),
        #[prost(message, tag = "8")]
        FloatVector(super::FloatVectorAttributeValueVariant),
        #[prost(message, tag = "9")]
        Boolean(super::BooleanAttributeValueVariant),
        #[prost(message, tag = "10")]
        BooleanVector(super::BooleanVectorAttributeValueVariant),
        #[prost(message, tag = "11")]
        BoundingBox(super::BoundingBoxAttributeValueVariant),
        #[prost(message, tag = "12")]
        BoundingBoxVector(super::BoundingBoxVectorAttributeValueVariant),
        #[prost(message, tag = "13")]
        Point(super::PointAttributeValueVariant),
        #[prost(message, tag = "14")]
        PointVector(super::PointVectorAttributeValueVariant),
        #[prost(message, tag = "15")]
        Polygon(super::PolygonAttributeValueVariant),
        #[prost(message, tag = "16")]
        PolygonVector(super::PolygonVectorAttributeValueVariant),
        #[prost(message, tag = "17")]
        Intersection(super::IntersectionAttributeValueVariant),
        #[prost(message, tag = "18")]
        None(super::NoneAttributeValueVariant),
    }
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Attribute {
    #[prost(string, tag = "1")]
    pub namespace: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub name: ::prost::alloc::string::String,
    #[prost(message, repeated, tag = "3")]
    pub values: ::prost::alloc::vec::Vec<AttributeValue>,
    #[prost(string, optional, tag = "4")]
    pub hint: ::core::option::Option<::prost::alloc::string::String>,
    #[prost(bool, tag = "5")]
    pub is_persistent: bool,
    #[prost(bool, tag = "6")]
    pub is_hidden: bool,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UserData {
    #[prost(string, tag = "1")]
    pub source_id: ::prost::alloc::string::String,
    #[prost(message, repeated, tag = "2")]
    pub attributes: ::prost::alloc::vec::Vec<Attribute>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ObjectAttribute {
    #[prost(int64, tag = "1")]
    pub object_id: i64,
    #[prost(message, optional, tag = "2")]
    pub attribute: ::core::option::Option<Attribute>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct VideoObject {
    #[prost(int64, tag = "1")]
    pub id: i64,
    #[prost(int64, optional, tag = "2")]
    pub parent_id: ::core::option::Option<i64>,
    #[prost(string, tag = "3")]
    pub namespace: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub label: ::prost::alloc::string::String,
    #[prost(string, optional, tag = "5")]
    pub draw_label: ::core::option::Option<::prost::alloc::string::String>,
    #[prost(message, optional, tag = "6")]
    pub detection_box: ::core::option::Option<BoundingBox>,
    #[prost(message, repeated, tag = "7")]
    pub attributes: ::prost::alloc::vec::Vec<Attribute>,
    #[prost(float, optional, tag = "8")]
    pub confidence: ::core::option::Option<f32>,
    #[prost(message, optional, tag = "9")]
    pub track_box: ::core::option::Option<BoundingBox>,
    #[prost(int64, optional, tag = "10")]
    pub track_id: ::core::option::Option<i64>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct VideoObjectWithForeignParent {
    #[prost(message, optional, tag = "1")]
    pub object: ::core::option::Option<VideoObject>,
    #[prost(int64, optional, tag = "2")]
    pub parent_id: ::core::option::Option<i64>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct VideoFrameUpdate {
    #[prost(message, repeated, tag = "1")]
    pub frame_attributes: ::prost::alloc::vec::Vec<Attribute>,
    #[prost(message, repeated, tag = "2")]
    pub object_attributes: ::prost::alloc::vec::Vec<ObjectAttribute>,
    #[prost(message, repeated, tag = "3")]
    pub objects: ::prost::alloc::vec::Vec<VideoObjectWithForeignParent>,
    #[prost(enumeration = "AttributeUpdatePolicy", tag = "4")]
    pub frame_attribute_policy: i32,
    #[prost(enumeration = "AttributeUpdatePolicy", tag = "5")]
    pub object_attribute_policy: i32,
    #[prost(enumeration = "ObjectUpdatePolicy", tag = "6")]
    pub object_policy: i32,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EndOfStream {
    #[prost(string, tag = "1")]
    pub source_id: ::prost::alloc::string::String,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Unknown {
    #[prost(string, tag = "1")]
    pub message: ::prost::alloc::string::String,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Shutdown {
    #[prost(string, tag = "1")]
    pub auth: ::prost::alloc::string::String,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExternalFrame {
    #[prost(string, tag = "1")]
    pub method: ::prost::alloc::string::String,
    #[prost(string, optional, tag = "2")]
    pub location: ::core::option::Option<::prost::alloc::string::String>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NoneFrame {}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InitialSize {
    #[prost(uint64, tag = "1")]
    pub width: u64,
    #[prost(uint64, tag = "2")]
    pub height: u64,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Scale {
    #[prost(uint64, tag = "1")]
    pub width: u64,
    #[prost(uint64, tag = "2")]
    pub height: u64,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Padding {
    #[prost(uint64, tag = "1")]
    pub padding_left: u64,
    #[prost(uint64, tag = "2")]
    pub padding_top: u64,
    #[prost(uint64, tag = "3")]
    pub padding_right: u64,
    #[prost(uint64, tag = "4")]
    pub padding_bottom: u64,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ResultingSize {
    #[prost(uint64, tag = "1")]
    pub width: u64,
    #[prost(uint64, tag = "2")]
    pub height: u64,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct VideoFrameTransformation {
    #[prost(
        oneof = "video_frame_transformation::Transformation",
        tags = "20, 21, 22, 23"
    )]
    pub transformation: ::core::option::Option<
        video_frame_transformation::Transformation,
    >,
}
/// Nested message and enum types in `VideoFrameTransformation`.
pub mod video_frame_transformation {
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Transformation {
        #[prost(message, tag = "20")]
        InitialSize(super::InitialSize),
        #[prost(message, tag = "21")]
        Scale(super::Scale),
        #[prost(message, tag = "22")]
        Padding(super::Padding),
        #[prost(message, tag = "23")]
        ResultingSize(super::ResultingSize),
    }
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct VideoFrame {
    #[prost(int64, optional, tag = "1")]
    pub previous_frame_seq_id: ::core::option::Option<i64>,
    #[prost(string, tag = "2")]
    pub source_id: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub uuid: ::prost::alloc::string::String,
    #[prost(uint64, tag = "4")]
    pub creation_timestamp_ns_high: u64,
    #[prost(uint64, tag = "5")]
    pub creation_timestamp_ns_low: u64,
    #[prost(string, tag = "6")]
    pub framerate: ::prost::alloc::string::String,
    #[prost(int64, tag = "7")]
    pub width: i64,
    #[prost(int64, tag = "8")]
    pub height: i64,
    #[prost(enumeration = "VideoFrameTranscodingMethod", tag = "9")]
    pub transcoding_method: i32,
    #[prost(string, optional, tag = "10")]
    pub codec: ::core::option::Option<::prost::alloc::string::String>,
    #[prost(bool, optional, tag = "11")]
    pub keyframe: ::core::option::Option<bool>,
    #[prost(int32, tag = "12")]
    pub time_base_numerator: i32,
    #[prost(int32, tag = "13")]
    pub time_base_denominator: i32,
    #[prost(int64, tag = "14")]
    pub pts: i64,
    #[prost(int64, optional, tag = "15")]
    pub dts: ::core::option::Option<i64>,
    #[prost(int64, optional, tag = "16")]
    pub duration: ::core::option::Option<i64>,
    #[prost(message, repeated, tag = "23")]
    pub transformations: ::prost::alloc::vec::Vec<VideoFrameTransformation>,
    #[prost(message, repeated, tag = "24")]
    pub attributes: ::prost::alloc::vec::Vec<Attribute>,
    #[prost(message, repeated, tag = "25")]
    pub objects: ::prost::alloc::vec::Vec<VideoObject>,
    #[prost(oneof = "video_frame::Content", tags = "17, 18, 19")]
    pub content: ::core::option::Option<video_frame::Content>,
}
/// Nested message and enum types in `VideoFrame`.
pub mod video_frame {
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Content {
        #[prost(bytes, tag = "17")]
        Internal(::prost::alloc::vec::Vec<u8>),
        #[prost(message, tag = "18")]
        External(super::ExternalFrame),
        #[prost(message, tag = "19")]
        None(super::NoneFrame),
    }
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct VideoFrameBatch {
    #[prost(map = "int64, message", tag = "1")]
    pub batch: ::std::collections::HashMap<i64, VideoFrame>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Message {
    #[prost(string, tag = "1")]
    pub lib_version: ::prost::alloc::string::String,
    #[prost(string, repeated, tag = "2")]
    pub routing_labels: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(map = "string, string", tag = "3")]
    pub propagated_context: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
    #[prost(uint64, tag = "4")]
    pub seq_id: u64,
    #[prost(oneof = "message::Content", tags = "5, 6, 7, 8, 9, 10, 11")]
    pub content: ::core::option::Option<message::Content>,
}
/// Nested message and enum types in `Message`.
pub mod message {
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Content {
        #[prost(message, tag = "5")]
        VideoFrame(super::VideoFrame),
        #[prost(message, tag = "6")]
        VideoFrameBatch(super::VideoFrameBatch),
        #[prost(message, tag = "7")]
        VideoFrameUpdate(super::VideoFrameUpdate),
        #[prost(message, tag = "8")]
        UserData(super::UserData),
        #[prost(message, tag = "9")]
        EndOfStream(super::EndOfStream),
        #[prost(message, tag = "10")]
        Unknown(super::Unknown),
        #[prost(message, tag = "11")]
        Shutdown(super::Shutdown),
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum IntersectionKind {
    Enter = 0,
    Inside = 1,
    Leave = 2,
    Cross = 3,
    Outside = 4,
}
impl IntersectionKind {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            IntersectionKind::Enter => "ENTER",
            IntersectionKind::Inside => "INSIDE",
            IntersectionKind::Leave => "LEAVE",
            IntersectionKind::Cross => "CROSS",
            IntersectionKind::Outside => "OUTSIDE",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "ENTER" => Some(Self::Enter),
            "INSIDE" => Some(Self::Inside),
            "LEAVE" => Some(Self::Leave),
            "CROSS" => Some(Self::Cross),
            "OUTSIDE" => Some(Self::Outside),
            _ => None,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum AttributeUpdatePolicy {
    ReplaceWithForeign = 0,
    KeepOwn = 1,
    Error = 2,
}
impl AttributeUpdatePolicy {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            AttributeUpdatePolicy::ReplaceWithForeign => "REPLACE_WITH_FOREIGN",
            AttributeUpdatePolicy::KeepOwn => "KEEP_OWN",
            AttributeUpdatePolicy::Error => "ERROR",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "REPLACE_WITH_FOREIGN" => Some(Self::ReplaceWithForeign),
            "KEEP_OWN" => Some(Self::KeepOwn),
            "ERROR" => Some(Self::Error),
            _ => None,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ObjectUpdatePolicy {
    AddForeignObjects = 0,
    ErrorIfLabelsCollide = 1,
    ReplaceSameLabelObjects = 2,
}
impl ObjectUpdatePolicy {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            ObjectUpdatePolicy::AddForeignObjects => "ADD_FOREIGN_OBJECTS",
            ObjectUpdatePolicy::ErrorIfLabelsCollide => "ERROR_IF_LABELS_COLLIDE",
            ObjectUpdatePolicy::ReplaceSameLabelObjects => "REPLACE_SAME_LABEL_OBJECTS",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "ADD_FOREIGN_OBJECTS" => Some(Self::AddForeignObjects),
            "ERROR_IF_LABELS_COLLIDE" => Some(Self::ErrorIfLabelsCollide),
            "REPLACE_SAME_LABEL_OBJECTS" => Some(Self::ReplaceSameLabelObjects),
            _ => None,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum VideoFrameTranscodingMethod {
    Copy = 0,
    Encoded = 1,
}
impl VideoFrameTranscodingMethod {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            VideoFrameTranscodingMethod::Copy => "COPY",
            VideoFrameTranscodingMethod::Encoded => "ENCODED",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "COPY" => Some(Self::Copy),
            "ENCODED" => Some(Self::Encoded),
            _ => None,
        }
    }
}

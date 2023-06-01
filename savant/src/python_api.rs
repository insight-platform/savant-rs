pub mod primitives {
    pub mod geometry {
        #[doc(inline)]
        pub use crate::primitives::Intersection;
        #[doc(inline)]
        pub use crate::primitives::IntersectionKind;
        #[doc(inline)]
        pub use crate::primitives::Point;
        #[doc(inline)]
        pub use crate::primitives::PolygonalArea;
        #[doc(inline)]
        pub use crate::primitives::PythonBBox;
        #[doc(inline)]
        pub use crate::primitives::RBBox;
        #[doc(inline)]
        pub use crate::primitives::Segment;
    }

    pub mod metadata {
        #[doc(inline)]
        pub use crate::primitives::message::video::frame::PyFrameTransformation;
        #[doc(inline)]
        pub use crate::primitives::message::video::frame::PyVideoFrameContent;
        #[doc(inline)]
        pub use crate::primitives::Attribute;
        #[doc(inline)]
        pub use crate::primitives::Object;
        #[doc(inline)]
        pub use crate::primitives::ObjectModification;
        #[doc(inline)]
        pub use crate::primitives::ObjectTrack;
        #[doc(inline)]
        pub use crate::primitives::ObjectVectorView;
        #[doc(inline)]
        pub use crate::primitives::Value;
        #[doc(inline)]
        pub use crate::primitives::VideoFrame;
        #[doc(inline)]
        pub use crate::primitives::VideoFrameBatch;
        #[doc(inline)]
        pub use crate::primitives::VideoTranscodingMethod;
    }

    pub mod messaging {
        #[doc(inline)]
        pub use crate::primitives::EndOfStream;
        #[doc(inline)]
        pub use crate::primitives::Message;
        #[doc(inline)]
        pub use crate::primitives::VideoFrame;
    }

    pub mod draw_spec {
        #[doc(inline)]
        pub use crate::primitives::BoundingBoxDraw;
        #[doc(inline)]
        pub use crate::primitives::ColorDraw;
        #[doc(inline)]
        pub use crate::primitives::DotDraw;
        #[doc(inline)]
        pub use crate::primitives::LabelDraw;
        #[doc(inline)]
        pub use crate::primitives::LabelPosition;
        #[doc(inline)]
        pub use crate::primitives::LabelPositionKind;
        #[doc(inline)]
        pub use crate::primitives::ObjectDraw;
        #[doc(inline)]
        pub use crate::primitives::PaddingDraw;
        #[doc(inline)]
        pub use crate::primitives::PySetDrawLabelKind;
    }
}

pub mod utils {
    pub mod testing {
        #[doc(inline)]
        pub use crate::test::utils::gen_frame;
        #[doc(inline)]
        pub use crate::utils::round_2_digits;
    }

    pub mod serdes {
        #[doc(inline)]
        pub use crate::primitives::message::loader::load_message_gil as load_message;
        #[doc(inline)]
        pub use crate::primitives::message::saver::save_message_gil as save_message;
    }

    pub mod numpy {
        #[doc(inline)]
        pub use crate::utils::np::np_nalgebra::NalgebraDMatrix;
        #[doc(inline)]
        pub use crate::utils::np::np_ndarray::NDarray;

        pub mod bbox_vector_ops {
            #[doc(inline)]
            pub use crate::utils::bbox::bboxes_to_ndarray_gil;
            #[doc(inline)]
            pub use crate::utils::bbox::ndarray_to_bboxes_gil;
            #[doc(inline)]
            pub use crate::utils::bbox::ndarray_to_rotated_bboxes_gil;
            #[doc(inline)]
            pub use crate::utils::bbox::rotated_bboxes_to_ndarray_gil;
        }

        pub mod ndarray {
            #[doc(inline)]
            pub use crate::utils::np::np_ndarray::ndarray_to_np_gil;
            #[doc(inline)]
            pub use crate::utils::np::np_ndarray::np_to_ndarray_gil;
        }

        pub mod nalgebra {
            #[doc(inline)]
            pub use crate::utils::np::np_nalgebra::matrix_to_np_gil;
            #[doc(inline)]
            pub use crate::utils::np::np_nalgebra::np_to_matrix_gil;
        }
    }

    pub mod udf_api {
        #[doc(inline)]
        pub use crate::utils::pluggable_udf_api::register_plugin_function_gil;
    }
}

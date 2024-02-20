use crate::primitives::frame::VideoFrameTransformation;
use crate::protobuf::generated;

impl From<&VideoFrameTransformation> for generated::VideoFrameTransformation {
    fn from(value: &VideoFrameTransformation) -> Self {
        match value {
            VideoFrameTransformation::InitialSize(w, h) => generated::VideoFrameTransformation {
                transformation: Some(
                    generated::video_frame_transformation::Transformation::InitialSize(
                        generated::InitialSize {
                            width: *w,
                            height: *h,
                        },
                    ),
                ),
            },
            VideoFrameTransformation::Scale(w, h) => generated::VideoFrameTransformation {
                transformation: Some(
                    generated::video_frame_transformation::Transformation::Scale(
                        generated::Scale {
                            width: *w,
                            height: *h,
                        },
                    ),
                ),
            },
            VideoFrameTransformation::Padding(l, t, r, b) => generated::VideoFrameTransformation {
                transformation: Some(
                    generated::video_frame_transformation::Transformation::Padding(
                        generated::Padding {
                            padding_left: *l,
                            padding_top: *t,
                            padding_right: *r,
                            padding_bottom: *b,
                        },
                    ),
                ),
            },
            VideoFrameTransformation::ResultingSize(w, h) => generated::VideoFrameTransformation {
                transformation: Some(
                    generated::video_frame_transformation::Transformation::ResultingSize(
                        generated::ResultingSize {
                            width: *w,
                            height: *h,
                        },
                    ),
                ),
            },
        }
    }
}

impl From<&generated::VideoFrameTransformation> for VideoFrameTransformation {
    fn from(value: &generated::VideoFrameTransformation) -> Self {
        match &value.transformation {
            Some(generated::video_frame_transformation::Transformation::InitialSize(is)) => {
                VideoFrameTransformation::InitialSize(is.width, is.height)
            }
            Some(generated::video_frame_transformation::Transformation::Scale(s)) => {
                VideoFrameTransformation::Scale(s.width, s.height)
            }
            Some(generated::video_frame_transformation::Transformation::Padding(p)) => {
                VideoFrameTransformation::Padding(
                    p.padding_left,
                    p.padding_top,
                    p.padding_right,
                    p.padding_bottom,
                )
            }
            Some(generated::video_frame_transformation::Transformation::ResultingSize(rs)) => {
                VideoFrameTransformation::ResultingSize(rs.width, rs.height)
            }
            None => unreachable!("Transformation is not set"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::frame::VideoFrameTransformation;
    use crate::protobuf::generated;

    #[test]
    fn test_video_frame_transformation_initial_size() {
        assert_eq!(
            VideoFrameTransformation::InitialSize(1, 2),
            VideoFrameTransformation::from(&generated::VideoFrameTransformation {
                transformation: Some(
                    generated::video_frame_transformation::Transformation::InitialSize(
                        generated::InitialSize {
                            width: 1,
                            height: 2
                        }
                    )
                )
            })
        );
        assert_eq!(
            generated::VideoFrameTransformation {
                transformation: Some(
                    generated::video_frame_transformation::Transformation::InitialSize(
                        generated::InitialSize {
                            width: 1,
                            height: 2
                        }
                    )
                )
            },
            generated::VideoFrameTransformation::from(&VideoFrameTransformation::InitialSize(1, 2))
        );
    }

    #[test]
    fn test_video_frame_transformation_scale() {
        assert_eq!(
            VideoFrameTransformation::Scale(1, 2),
            VideoFrameTransformation::from(&generated::VideoFrameTransformation {
                transformation: Some(
                    generated::video_frame_transformation::Transformation::Scale(
                        generated::Scale {
                            width: 1,
                            height: 2
                        }
                    )
                )
            })
        );
        assert_eq!(
            generated::VideoFrameTransformation {
                transformation: Some(
                    generated::video_frame_transformation::Transformation::Scale(
                        generated::Scale {
                            width: 1,
                            height: 2
                        }
                    )
                )
            },
            generated::VideoFrameTransformation::from(&VideoFrameTransformation::Scale(1, 2))
        );
    }

    #[test]
    fn test_video_frame_transformation_padding() {
        assert_eq!(
            VideoFrameTransformation::Padding(1, 2, 3, 4),
            VideoFrameTransformation::from(&generated::VideoFrameTransformation {
                transformation: Some(
                    generated::video_frame_transformation::Transformation::Padding(
                        generated::Padding {
                            padding_left: 1,
                            padding_top: 2,
                            padding_right: 3,
                            padding_bottom: 4,
                        }
                    )
                )
            })
        );
        assert_eq!(
            generated::VideoFrameTransformation {
                transformation: Some(
                    generated::video_frame_transformation::Transformation::Padding(
                        generated::Padding {
                            padding_left: 1,
                            padding_top: 2,
                            padding_right: 3,
                            padding_bottom: 4,
                        }
                    )
                )
            },
            generated::VideoFrameTransformation::from(&VideoFrameTransformation::Padding(
                1, 2, 3, 4
            ))
        );
    }

    #[test]
    fn test_video_frame_transformation_resulting_size() {
        assert_eq!(
            VideoFrameTransformation::ResultingSize(1, 2),
            VideoFrameTransformation::from(&generated::VideoFrameTransformation {
                transformation: Some(
                    generated::video_frame_transformation::Transformation::ResultingSize(
                        generated::ResultingSize {
                            width: 1,
                            height: 2
                        }
                    )
                )
            })
        );
        assert_eq!(
            generated::VideoFrameTransformation {
                transformation: Some(
                    generated::video_frame_transformation::Transformation::ResultingSize(
                        generated::ResultingSize {
                            width: 1,
                            height: 2
                        }
                    )
                )
            },
            generated::VideoFrameTransformation::from(&VideoFrameTransformation::ResultingSize(
                1, 2
            ))
        );
    }
}

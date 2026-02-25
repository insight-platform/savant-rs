use crate::primitives::frame::VideoFrameTransformation;
use savant_protobuf::generated;

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
            VideoFrameTransformation::LetterBox(ow, oh, pl, pt, pr, pb) => {
                generated::VideoFrameTransformation {
                    transformation: Some(
                        generated::video_frame_transformation::Transformation::LetterBox(
                            generated::LetterBox {
                                outer_width: *ow,
                                outer_height: *oh,
                                padding_left: *pl,
                                padding_top: *pt,
                                padding_right: *pr,
                                padding_bottom: *pb,
                            },
                        ),
                    ),
                }
            }
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
            VideoFrameTransformation::Crop(l, t, r, b) => generated::VideoFrameTransformation {
                transformation: Some(generated::video_frame_transformation::Transformation::Crop(
                    generated::Crop {
                        left: *l,
                        top: *t,
                        right: *r,
                        bottom: *b,
                    },
                )),
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
            Some(generated::video_frame_transformation::Transformation::LetterBox(lb)) => {
                VideoFrameTransformation::LetterBox(
                    lb.outer_width,
                    lb.outer_height,
                    lb.padding_left,
                    lb.padding_top,
                    lb.padding_right,
                    lb.padding_bottom,
                )
            }
            Some(generated::video_frame_transformation::Transformation::Padding(p)) => {
                VideoFrameTransformation::Padding(
                    p.padding_left,
                    p.padding_top,
                    p.padding_right,
                    p.padding_bottom,
                )
            }
            Some(generated::video_frame_transformation::Transformation::Crop(c)) => {
                VideoFrameTransformation::Crop(c.left, c.top, c.right, c.bottom)
            }
            None => unreachable!("Transformation is not set"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::frame::VideoFrameTransformation;
    use savant_protobuf::generated;

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
    fn test_video_frame_transformation_letter_box() {
        assert_eq!(
            VideoFrameTransformation::LetterBox(100, 80, 5, 5, 5, 5),
            VideoFrameTransformation::from(&generated::VideoFrameTransformation {
                transformation: Some(
                    generated::video_frame_transformation::Transformation::LetterBox(
                        generated::LetterBox {
                            outer_width: 100,
                            outer_height: 80,
                            padding_left: 5,
                            padding_top: 5,
                            padding_right: 5,
                            padding_bottom: 5,
                        }
                    )
                )
            })
        );
        assert_eq!(
            generated::VideoFrameTransformation {
                transformation: Some(
                    generated::video_frame_transformation::Transformation::LetterBox(
                        generated::LetterBox {
                            outer_width: 100,
                            outer_height: 80,
                            padding_left: 5,
                            padding_top: 5,
                            padding_right: 5,
                            padding_bottom: 5,
                        }
                    )
                )
            },
            generated::VideoFrameTransformation::from(&VideoFrameTransformation::LetterBox(
                100, 80, 5, 5, 5, 5
            ))
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
    fn test_video_frame_transformation_crop() {
        assert_eq!(
            VideoFrameTransformation::Crop(10, 20, 30, 40),
            VideoFrameTransformation::from(&generated::VideoFrameTransformation {
                transformation: Some(generated::video_frame_transformation::Transformation::Crop(
                    generated::Crop {
                        left: 10,
                        top: 20,
                        right: 30,
                        bottom: 40,
                    }
                ))
            })
        );
        assert_eq!(
            generated::VideoFrameTransformation {
                transformation: Some(generated::video_frame_transformation::Transformation::Crop(
                    generated::Crop {
                        left: 10,
                        top: 20,
                        right: 30,
                        bottom: 40,
                    }
                ))
            },
            generated::VideoFrameTransformation::from(&VideoFrameTransformation::Crop(
                10, 20, 30, 40
            ))
        );
    }
}

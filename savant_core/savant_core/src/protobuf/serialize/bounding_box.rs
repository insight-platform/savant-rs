use crate::primitives::{RBBox, RBBoxData, BBOX_ELEMENT_UNDEFINED};
use savant_protobuf::generated;

impl From<&RBBox> for generated::BoundingBox {
    fn from(value: &RBBox) -> Self {
        generated::BoundingBox {
            xc: value.get_xc(),
            yc: value.get_yc(),
            width: value.get_width(),
            height: value.get_height(),
            angle: value.get_angle(),
        }
    }
}

impl From<&generated::BoundingBox> for RBBox {
    fn from(value: &generated::BoundingBox) -> Self {
        RBBox::new(value.xc, value.yc, value.width, value.height, value.angle)
    }
}

impl From<&generated::BoundingBox> for RBBoxData {
    fn from(value: &generated::BoundingBox) -> Self {
        RBBoxData {
            xc: value.xc.into(),
            yc: value.yc.into(),
            width: value.width.into(),
            height: value.height.into(),
            angle: value.angle.unwrap_or(BBOX_ELEMENT_UNDEFINED).into(),
            has_modifications: false.into(),
        }
    }
}

impl From<&RBBoxData> for generated::BoundingBox {
    fn from(value: &RBBoxData) -> Self {
        generated::BoundingBox {
            xc: value.xc.get(),
            yc: value.yc.get(),
            width: value.width.get(),
            height: value.height.get(),
            angle: {
                let angle = value.angle.get();
                if angle == BBOX_ELEMENT_UNDEFINED {
                    None
                } else {
                    Some(angle)
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::{RBBox, RBBoxData};
    use savant_protobuf::generated;

    #[test]
    fn test_bounding_box() {
        assert_eq!(
            RBBox::new(1.0, 2.0, 3.0, 4.0, Some(5.0)),
            RBBox::from(&generated::BoundingBox {
                xc: 1.0,
                yc: 2.0,
                width: 3.0,
                height: 4.0,
                angle: Some(5.0),
            })
        );
        assert_eq!(
            generated::BoundingBox {
                xc: 1.0,
                yc: 2.0,
                width: 3.0,
                height: 4.0,
                angle: Some(5.0),
            },
            generated::BoundingBox::from(&RBBox::new(1.0, 2.0, 3.0, 4.0, Some(5.0)))
        );
    }

    #[test]
    fn test_owned_bounding_box() {
        assert_eq!(
            RBBoxData::new(1.0, 2.0, 3.0, 4.0, Some(5.0)),
            RBBoxData::from(&generated::BoundingBox {
                xc: 1.0,
                yc: 2.0,
                width: 3.0,
                height: 4.0,
                angle: Some(5.0),
            })
        );
        assert_eq!(
            generated::BoundingBox {
                xc: 1.0,
                yc: 2.0,
                width: 3.0,
                height: 4.0,
                angle: Some(5.0),
            },
            generated::BoundingBox::from(&RBBoxData::new(1.0, 2.0, 3.0, 4.0, Some(5.0)))
        );
    }
}

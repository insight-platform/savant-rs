use crate::primitives::{OwnedRBBoxData, RBBox};
use crate::protobuf::generated;

impl From<RBBox> for generated::BoundingBox {
    fn from(value: RBBox) -> Self {
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

impl From<&OwnedRBBoxData> for generated::BoundingBox {
    fn from(value: &OwnedRBBoxData) -> Self {
        generated::BoundingBox {
            xc: value.xc,
            yc: value.yc,
            width: value.width,
            height: value.height,
            angle: value.angle,
        }
    }
}

impl From<&generated::BoundingBox> for OwnedRBBoxData {
    fn from(value: &generated::BoundingBox) -> Self {
        OwnedRBBoxData {
            xc: value.xc,
            yc: value.yc,
            width: value.width,
            height: value.height,
            angle: value.angle,
            has_modifications: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::{OwnedRBBoxData, RBBox};
    use crate::protobuf::generated;

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
            generated::BoundingBox::from(RBBox::new(1.0, 2.0, 3.0, 4.0, Some(5.0)))
        );
    }

    #[test]
    fn test_owned_bounding_box() {
        assert_eq!(
            OwnedRBBoxData {
                xc: 1.0,
                yc: 2.0,
                width: 3.0,
                height: 4.0,
                angle: Some(5.0),
                has_modifications: false,
            },
            OwnedRBBoxData::from(&generated::BoundingBox {
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
            generated::BoundingBox::from(&OwnedRBBoxData {
                xc: 1.0,
                yc: 2.0,
                width: 3.0,
                height: 4.0,
                angle: Some(5.0),
                has_modifications: false,
            })
        );
    }
}

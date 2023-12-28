use crate::primitives::any_object::AnyObject;
use crate::primitives::attribute_value::{AttributeValue, AttributeValueVariant};
use crate::primitives::{Attribute, IntersectionKind};
use crate::protobuf::{generated, serialize};
use std::sync::Arc;

impl From<&AttributeValueVariant> for generated::attribute_value::Value {
    fn from(value: &AttributeValueVariant) -> Self {
        match value {
            AttributeValueVariant::Bytes(dims, data) => {
                generated::attribute_value::Value::Bytes(generated::BytesAttributeValueVariant {
                    dims: dims.clone(),
                    data: data.clone(),
                })
            }
            AttributeValueVariant::String(s) => {
                generated::attribute_value::Value::String(generated::StringAttributeValueVariant {
                    data: s.clone(),
                })
            }
            AttributeValueVariant::StringVector(sv) => {
                generated::attribute_value::Value::StringVector(
                    generated::StringVectorAttributeValueVariant { data: sv.clone() },
                )
            }
            AttributeValueVariant::Integer(i) => generated::attribute_value::Value::Integer(
                generated::IntegerAttributeValueVariant { data: *i },
            ),
            AttributeValueVariant::IntegerVector(iv) => {
                generated::attribute_value::Value::IntegerVector(
                    generated::IntegerVectorAttributeValueVariant { data: iv.clone() },
                )
            }
            AttributeValueVariant::Float(f) => {
                generated::attribute_value::Value::Float(generated::FloatAttributeValueVariant {
                    data: *f,
                })
            }
            AttributeValueVariant::FloatVector(fv) => {
                generated::attribute_value::Value::FloatVector(
                    generated::FloatVectorAttributeValueVariant { data: fv.clone() },
                )
            }
            AttributeValueVariant::Boolean(b) => generated::attribute_value::Value::Boolean(
                generated::BooleanAttributeValueVariant { data: *b },
            ),
            AttributeValueVariant::BooleanVector(bv) => {
                generated::attribute_value::Value::BooleanVector(
                    generated::BooleanVectorAttributeValueVariant { data: bv.clone() },
                )
            }
            AttributeValueVariant::BBox(bb) => generated::attribute_value::Value::BoundingBox(
                generated::BoundingBoxAttributeValueVariant {
                    data: Some(generated::BoundingBox::from(bb)),
                },
            ),
            AttributeValueVariant::BBoxVector(bbv) => {
                generated::attribute_value::Value::BoundingBoxVector(
                    generated::BoundingBoxVectorAttributeValueVariant {
                        data: bbv.iter().map(generated::BoundingBox::from).collect(),
                    },
                )
            }
            AttributeValueVariant::Point(p) => {
                generated::attribute_value::Value::Point(generated::PointAttributeValueVariant {
                    data: Some(generated::Point { x: p.x, y: p.y }),
                })
            }
            AttributeValueVariant::PointVector(pv) => {
                generated::attribute_value::Value::PointVector(
                    generated::PointVectorAttributeValueVariant {
                        data: pv
                            .iter()
                            .map(|p| generated::Point { x: p.x, y: p.y })
                            .collect(),
                    },
                )
            }
            AttributeValueVariant::Polygon(poly) => generated::attribute_value::Value::Polygon(
                generated::PolygonAttributeValueVariant {
                    data: Some(poly.into()),
                },
            ),
            AttributeValueVariant::PolygonVector(pv) => {
                generated::attribute_value::Value::PolygonVector(
                    generated::PolygonVectorAttributeValueVariant {
                        data: pv.iter().map(|poly| poly.into()).collect(),
                    },
                )
            }
            AttributeValueVariant::Intersection(is) => {
                generated::attribute_value::Value::Intersection(
                    generated::IntersectionAttributeValueVariant {
                        data: Some(generated::Intersection {
                            kind: generated::IntersectionKind::from(&is.kind) as i32,
                            edges: is
                                .edges
                                .iter()
                                .map(|e| generated::IntersectionEdge {
                                    id: e.0 as u64,
                                    tag: e.1.clone(),
                                })
                                .collect(),
                        }),
                    },
                )
            }
            AttributeValueVariant::TemporaryValue(_) => {
                generated::attribute_value::Value::Temporary(generated::TemporaryValueVariant {})
            }
            AttributeValueVariant::None => {
                generated::attribute_value::Value::None(generated::NoneAttributeValueVariant {})
            }
        }
    }
}

impl TryFrom<&generated::attribute_value::Value> for AttributeValueVariant {
    type Error = serialize::Error;

    fn try_from(value: &generated::attribute_value::Value) -> Result<Self, Self::Error> {
        Ok(match value {
            generated::attribute_value::Value::Bytes(b) => {
                AttributeValueVariant::Bytes(b.dims.clone(), b.data.clone())
            }
            generated::attribute_value::Value::String(s) => {
                AttributeValueVariant::String(s.data.clone())
            }
            generated::attribute_value::Value::StringVector(sv) => {
                AttributeValueVariant::StringVector(sv.data.clone())
            }
            generated::attribute_value::Value::Integer(i) => AttributeValueVariant::Integer(i.data),
            generated::attribute_value::Value::IntegerVector(iv) => {
                AttributeValueVariant::IntegerVector(iv.data.clone())
            }
            generated::attribute_value::Value::Float(f) => AttributeValueVariant::Float(f.data),
            generated::attribute_value::Value::FloatVector(fv) => {
                AttributeValueVariant::FloatVector(fv.data.clone())
            }
            generated::attribute_value::Value::Boolean(b) => AttributeValueVariant::Boolean(b.data),
            generated::attribute_value::Value::BooleanVector(bv) => {
                AttributeValueVariant::BooleanVector(bv.data.clone())
            }
            generated::attribute_value::Value::BoundingBox(bb) => {
                AttributeValueVariant::BBox(bb.data.as_ref().unwrap().into())
            }
            generated::attribute_value::Value::BoundingBoxVector(bbv) => {
                AttributeValueVariant::BBoxVector(bbv.data.iter().map(|bb| bb.into()).collect())
            }
            generated::attribute_value::Value::Point(p) => {
                AttributeValueVariant::Point(crate::primitives::Point::new(
                    p.data.as_ref().unwrap().x,
                    p.data.as_ref().unwrap().y,
                ))
            }
            generated::attribute_value::Value::PointVector(pv) => {
                AttributeValueVariant::PointVector(
                    pv.data
                        .iter()
                        .map(|p| crate::primitives::Point::new(p.x, p.y))
                        .collect(),
                )
            }
            generated::attribute_value::Value::Polygon(poly) => {
                AttributeValueVariant::Polygon(poly.data.as_ref().unwrap().into())
            }
            generated::attribute_value::Value::PolygonVector(pv) => {
                AttributeValueVariant::PolygonVector(
                    pv.data.iter().map(|poly| poly.into()).collect(),
                )
            }
            generated::attribute_value::Value::Intersection(i) => {
                AttributeValueVariant::Intersection(crate::primitives::Intersection {
                    kind: IntersectionKind::from(&i.data.as_ref().unwrap().kind.try_into()?),
                    edges: i
                        .data
                        .as_ref()
                        .unwrap()
                        .edges
                        .iter()
                        .map(|e| (e.id as usize, e.tag.clone()))
                        .collect(),
                })
            }
            generated::attribute_value::Value::None(_) => AttributeValueVariant::None,
            generated::attribute_value::Value::Temporary(_) => {
                AttributeValueVariant::TemporaryValue(AnyObject::new(Box::new(())))
            }
        })
    }
}

impl From<&AttributeValue> for generated::AttributeValue {
    fn from(value: &AttributeValue) -> Self {
        generated::AttributeValue {
            confidence: value.confidence,
            value: Some(generated::attribute_value::Value::from(&value.value)),
        }
    }
}

impl TryFrom<&generated::AttributeValue> for AttributeValue {
    type Error = serialize::Error;
    fn try_from(value: &generated::AttributeValue) -> Result<Self, Self::Error> {
        Ok(AttributeValue {
            confidence: value.confidence,
            value: AttributeValueVariant::try_from(value.value.as_ref().unwrap())?,
        })
    }
}

impl From<&Attribute> for generated::Attribute {
    fn from(a: &Attribute) -> Self {
        generated::Attribute {
            namespace: a.namespace.clone(),
            name: a.name.clone(),
            values: a.values.iter().map(|v| v.into()).collect(),
            hint: a.hint.clone(),
            is_persistent: a.is_persistent,
            is_hidden: a.is_hidden,
        }
    }
}

impl TryFrom<&generated::Attribute> for Attribute {
    type Error = serialize::Error;
    fn try_from(value: &generated::Attribute) -> Result<Self, Self::Error> {
        Ok(Attribute {
            namespace: value.namespace.clone(),
            name: value.name.clone(),
            values: Arc::new(
                value
                    .values
                    .iter()
                    .map(|v| v.try_into())
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            hint: value.hint.clone(),
            is_persistent: value.is_persistent,
            is_hidden: value.is_hidden,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute_value::{AttributeValue, AttributeValueVariant};
    use crate::primitives::{Attribute, IntersectionKind};
    use crate::protobuf::generated;
    use std::sync::Arc;

    #[test]
    fn test_attribute_value_variant_bytes() {
        let dims = vec![1, 2, 3];
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let av = AttributeValueVariant::Bytes(dims.clone(), data.clone());
        assert_eq!(
            av,
            AttributeValueVariant::try_from(&generated::attribute_value::Value::Bytes(
                generated::BytesAttributeValueVariant {
                    dims: dims.clone(),
                    data: data.clone(),
                }
            ))
            .unwrap()
        );
        assert_eq!(
            generated::attribute_value::Value::Bytes(generated::BytesAttributeValueVariant {
                dims: dims.clone(),
                data: data.clone(),
            }),
            generated::attribute_value::Value::from(&av)
        );
    }

    #[test]
    fn test_attribute_value_variant_string() {
        let s = "string".to_string();
        let av = AttributeValueVariant::String(s.clone());
        assert_eq!(
            av,
            AttributeValueVariant::try_from(&generated::attribute_value::Value::String(
                generated::StringAttributeValueVariant { data: s.clone() }
            ))
            .unwrap()
        );
        assert_eq!(
            generated::attribute_value::Value::String(generated::StringAttributeValueVariant {
                data: s.clone()
            }),
            generated::attribute_value::Value::from(&av)
        );
    }

    #[test]
    fn test_attribute_value_variant_string_vector() {
        let sv = vec!["string".to_string(), "vector".to_string()];
        let av = AttributeValueVariant::StringVector(sv.clone());
        assert_eq!(
            av,
            AttributeValueVariant::try_from(&generated::attribute_value::Value::StringVector(
                generated::StringVectorAttributeValueVariant { data: sv.clone() }
            ))
            .unwrap()
        );
        assert_eq!(
            generated::attribute_value::Value::StringVector(
                generated::StringVectorAttributeValueVariant { data: sv.clone() }
            ),
            generated::attribute_value::Value::from(&av)
        );
    }

    #[test]
    fn test_attribute_value_variant_integer() {
        let i = 42;
        let av = AttributeValueVariant::Integer(i);
        assert_eq!(
            av,
            AttributeValueVariant::try_from(&generated::attribute_value::Value::Integer(
                generated::IntegerAttributeValueVariant { data: i }
            ))
            .unwrap()
        );
        assert_eq!(
            generated::attribute_value::Value::Integer(generated::IntegerAttributeValueVariant {
                data: i
            }),
            generated::attribute_value::Value::from(&av)
        );
    }

    #[test]
    fn test_attribute_value_variant_integer_vector() {
        let iv = vec![1, 2, 3];
        let av = AttributeValueVariant::IntegerVector(iv.clone());
        assert_eq!(
            av,
            AttributeValueVariant::try_from(&generated::attribute_value::Value::IntegerVector(
                generated::IntegerVectorAttributeValueVariant { data: iv.clone() }
            ))
            .unwrap()
        );
        assert_eq!(
            generated::attribute_value::Value::IntegerVector(
                generated::IntegerVectorAttributeValueVariant { data: iv.clone() }
            ),
            generated::attribute_value::Value::from(&av)
        );
    }

    #[test]
    fn test_attribute_value_variant_float() {
        let f = 42.0;
        let av = AttributeValueVariant::Float(f);
        assert_eq!(
            av,
            AttributeValueVariant::try_from(&generated::attribute_value::Value::Float(
                generated::FloatAttributeValueVariant { data: f }
            ))
            .unwrap()
        );
        assert_eq!(
            generated::attribute_value::Value::Float(generated::FloatAttributeValueVariant {
                data: f
            }),
            generated::attribute_value::Value::from(&av)
        );
    }

    #[test]
    fn test_attribute_value_variant_float_vector() {
        let fv = vec![1.0, 2.0, 3.0];
        let av = AttributeValueVariant::FloatVector(fv.clone());
        assert_eq!(
            av,
            AttributeValueVariant::try_from(&generated::attribute_value::Value::FloatVector(
                generated::FloatVectorAttributeValueVariant { data: fv.clone() }
            ))
            .unwrap()
        );
        assert_eq!(
            generated::attribute_value::Value::FloatVector(
                generated::FloatVectorAttributeValueVariant { data: fv.clone() }
            ),
            generated::attribute_value::Value::from(&av)
        );
    }

    #[test]
    fn test_attribute_value_variant_boolean() {
        let b = true;
        let av = AttributeValueVariant::Boolean(b);
        assert_eq!(
            av,
            AttributeValueVariant::try_from(&generated::attribute_value::Value::Boolean(
                generated::BooleanAttributeValueVariant { data: b }
            ))
            .unwrap()
        );
        assert_eq!(
            generated::attribute_value::Value::Boolean(generated::BooleanAttributeValueVariant {
                data: b
            }),
            generated::attribute_value::Value::from(&av)
        );
    }

    #[test]
    fn test_attribute_value_variant_boolean_vector() {
        let bv = vec![true, false];
        let av = AttributeValueVariant::BooleanVector(bv.clone());
        assert_eq!(
            av,
            AttributeValueVariant::try_from(&generated::attribute_value::Value::BooleanVector(
                generated::BooleanVectorAttributeValueVariant { data: bv.clone() }
            ))
            .unwrap()
        );
        assert_eq!(
            generated::attribute_value::Value::BooleanVector(
                generated::BooleanVectorAttributeValueVariant { data: bv.clone() }
            ),
            generated::attribute_value::Value::from(&av)
        );
    }

    #[test]
    fn test_attribute_value_variant_bbox() {
        let bb = crate::primitives::RBBox::new(1.0, 2.0, 3.0, 4.0, Some(5.0))
            .to_owned()
            .unwrap();
        let av = AttributeValueVariant::BBox(bb.clone());
        assert_eq!(
            av,
            AttributeValueVariant::try_from(&generated::attribute_value::Value::BoundingBox(
                generated::BoundingBoxAttributeValueVariant {
                    data: Some(generated::BoundingBox::from(&bb))
                }
            ))
            .unwrap()
        );
        assert_eq!(
            generated::attribute_value::Value::BoundingBox(
                generated::BoundingBoxAttributeValueVariant {
                    data: Some(generated::BoundingBox::from(&bb))
                }
            ),
            generated::attribute_value::Value::from(&av)
        );
    }

    #[test]
    fn test_attribute_value_variant_bbox_vector() {
        let bbv = vec![
            crate::primitives::RBBox::new(1.0, 2.0, 3.0, 4.0, Some(5.0))
                .to_owned()
                .unwrap(),
            crate::primitives::RBBox::new(6.0, 7.0, 8.0, 9.0, Some(10.0))
                .to_owned()
                .unwrap(),
        ];
        let av = AttributeValueVariant::BBoxVector(bbv.clone());
        assert_eq!(
            av,
            AttributeValueVariant::try_from(&generated::attribute_value::Value::BoundingBoxVector(
                generated::BoundingBoxVectorAttributeValueVariant {
                    data: bbv.iter().map(|bb| bb.into()).collect()
                }
            ))
            .unwrap()
        );
        assert_eq!(
            generated::attribute_value::Value::BoundingBoxVector(
                generated::BoundingBoxVectorAttributeValueVariant {
                    data: bbv.iter().map(|bb| bb.into()).collect()
                }
            ),
            generated::attribute_value::Value::from(&av)
        );
    }

    #[test]
    fn test_attribute_value_variant_point() {
        let p = crate::primitives::Point::new(1.0, 2.0);
        let av = AttributeValueVariant::Point(p.clone());
        assert_eq!(
            av,
            AttributeValueVariant::try_from(&generated::attribute_value::Value::Point(
                generated::PointAttributeValueVariant {
                    data: Some(generated::Point { x: p.x, y: p.y })
                }
            ))
            .unwrap()
        );
        assert_eq!(
            generated::attribute_value::Value::Point(generated::PointAttributeValueVariant {
                data: Some(generated::Point { x: p.x, y: p.y })
            }),
            generated::attribute_value::Value::from(&av)
        );
    }

    #[test]
    fn test_attribute_value_variant_point_vector() {
        let pv = vec![
            crate::primitives::Point::new(1.0, 2.0),
            crate::primitives::Point::new(3.0, 4.0),
        ];
        let av = AttributeValueVariant::PointVector(pv.clone());
        assert_eq!(
            av,
            AttributeValueVariant::try_from(&generated::attribute_value::Value::PointVector(
                generated::PointVectorAttributeValueVariant {
                    data: pv
                        .iter()
                        .map(|p| generated::Point { x: p.x, y: p.y })
                        .collect()
                }
            ))
            .unwrap()
        );
        assert_eq!(
            generated::attribute_value::Value::PointVector(
                generated::PointVectorAttributeValueVariant {
                    data: pv
                        .iter()
                        .map(|p| generated::Point { x: p.x, y: p.y })
                        .collect()
                }
            ),
            generated::attribute_value::Value::from(&av)
        );
    }

    #[test]
    fn test_attribute_value_variant_polygon() {
        let poly = crate::primitives::PolygonalArea::new(
            vec![
                crate::primitives::Point::new(1.0, 2.0),
                crate::primitives::Point::new(3.0, 4.0),
            ],
            Some(vec![Some("tag".to_string()), None]),
        );
        let av = AttributeValueVariant::Polygon(poly.clone());
        assert_eq!(
            av,
            AttributeValueVariant::try_from(&generated::attribute_value::Value::Polygon(
                generated::PolygonAttributeValueVariant {
                    data: Some(generated::PolygonalArea::from(&poly))
                }
            ))
            .unwrap()
        );
        assert_eq!(
            generated::attribute_value::Value::Polygon(generated::PolygonAttributeValueVariant {
                data: Some(generated::PolygonalArea::from(&poly))
            }),
            generated::attribute_value::Value::from(&av)
        );
    }

    #[test]
    fn test_attribute_value_variant_polygon_vector() {
        let pv = vec![
            crate::primitives::PolygonalArea::new(
                vec![
                    crate::primitives::Point::new(1.0, 2.0),
                    crate::primitives::Point::new(3.0, 4.0),
                ],
                Some(vec![Some("tag".to_string()), None]),
            ),
            crate::primitives::PolygonalArea::new(
                vec![
                    crate::primitives::Point::new(5.0, 6.0),
                    crate::primitives::Point::new(7.0, 8.0),
                ],
                Some(vec![Some("tag".to_string()), None]),
            ),
        ];
        let av = AttributeValueVariant::PolygonVector(pv.clone());
        assert_eq!(
            av,
            AttributeValueVariant::try_from(&generated::attribute_value::Value::PolygonVector(
                generated::PolygonVectorAttributeValueVariant {
                    data: pv.iter().map(|poly| poly.into()).collect()
                }
            ))
            .unwrap()
        );
        assert_eq!(
            generated::attribute_value::Value::PolygonVector(
                generated::PolygonVectorAttributeValueVariant {
                    data: pv.iter().map(|poly| poly.into()).collect()
                }
            ),
            generated::attribute_value::Value::from(&av)
        );
    }

    #[test]
    fn test_attribute_value_variant_intersection() {
        let is = crate::primitives::Intersection {
            kind: IntersectionKind::Cross,
            edges: vec![(1, Some("tag".to_string())), (2, None)],
        };
        let av = AttributeValueVariant::Intersection(is.clone());
        assert_eq!(
            av,
            AttributeValueVariant::try_from(&generated::attribute_value::Value::Intersection(
                generated::IntersectionAttributeValueVariant {
                    data: Some(generated::Intersection {
                        kind: generated::IntersectionKind::from(&is.kind) as i32,
                        edges: is
                            .edges
                            .iter()
                            .map(|e| generated::IntersectionEdge {
                                id: e.0 as u64,
                                tag: e.1.clone(),
                            })
                            .collect(),
                    })
                }
            ))
            .unwrap()
        );
        assert_eq!(
            generated::attribute_value::Value::Intersection(
                generated::IntersectionAttributeValueVariant {
                    data: Some(generated::Intersection {
                        kind: generated::IntersectionKind::from(&is.kind) as i32,
                        edges: is
                            .edges
                            .iter()
                            .map(|e| generated::IntersectionEdge {
                                id: e.0 as u64,
                                tag: e.1.clone(),
                            })
                            .collect(),
                    })
                }
            ),
            generated::attribute_value::Value::from(&av)
        );
    }

    #[test]
    fn test_attribute_value_variant_none() {
        let av = AttributeValueVariant::None;
        assert_eq!(
            av,
            AttributeValueVariant::try_from(&generated::attribute_value::Value::None(
                generated::NoneAttributeValueVariant {}
            ))
            .unwrap()
        );
        assert_eq!(
            generated::attribute_value::Value::None(generated::NoneAttributeValueVariant {}),
            generated::attribute_value::Value::from(&av)
        );
    }

    #[test]
    fn test_attribute_value() {
        let av = AttributeValue {
            confidence: Some(0.5),
            value: AttributeValueVariant::String("string".to_string()),
        };
        assert_eq!(
            av,
            AttributeValue::try_from(&generated::AttributeValue {
                confidence: Some(0.5),
                value: Some(generated::attribute_value::Value::String(
                    generated::StringAttributeValueVariant {
                        data: "string".to_string()
                    }
                ))
            })
            .unwrap()
        );
        assert_eq!(
            generated::AttributeValue {
                confidence: Some(0.5),
                value: Some(generated::attribute_value::Value::String(
                    generated::StringAttributeValueVariant {
                        data: "string".to_string()
                    }
                ))
            },
            generated::AttributeValue::from(&av)
        );
    }

    #[test]
    fn test_attribute() {
        let a = Attribute {
            namespace: "namespace".to_string(),
            name: "name".to_string(),
            values: Arc::new(vec![AttributeValue {
                confidence: Some(0.5),
                value: AttributeValueVariant::String("string".to_string()),
            }]),
            hint: Some("hint".to_string()),
            is_persistent: true,
            is_hidden: false,
        };
        assert_eq!(
            a,
            Attribute::try_from(&generated::Attribute {
                namespace: "namespace".to_string(),
                name: "name".to_string(),
                values: vec![generated::AttributeValue {
                    confidence: Some(0.5),
                    value: Some(generated::attribute_value::Value::String(
                        generated::StringAttributeValueVariant {
                            data: "string".to_string()
                        }
                    ))
                }],
                hint: Some("hint".to_string()),
                is_persistent: true,
                is_hidden: false,
            })
            .unwrap()
        );
        assert_eq!(
            generated::Attribute {
                namespace: "namespace".to_string(),
                name: "name".to_string(),
                values: vec![generated::AttributeValue {
                    confidence: Some(0.5),
                    value: Some(generated::attribute_value::Value::String(
                        generated::StringAttributeValueVariant {
                            data: "string".to_string()
                        }
                    ))
                }],
                hint: Some("hint".to_string()),
                is_persistent: true,
                is_hidden: false,
            },
            generated::Attribute::from(&a)
        );
    }
}

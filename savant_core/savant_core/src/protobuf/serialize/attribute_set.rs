use crate::primitives::attribute_set::AttributeSet;
use crate::primitives::Attribute;
use crate::protobuf::serialize;
use savant_protobuf::generated;

impl From<&AttributeSet> for generated::AttributeSet {
    fn from(ud: &AttributeSet) -> Self {
        let attributes = ud
            .attributes
            .iter()
            .map(generated::Attribute::from)
            .collect();

        generated::AttributeSet { attributes }
    }
}

impl TryFrom<&generated::AttributeSet> for AttributeSet {
    type Error = serialize::Error;

    fn try_from(value: &generated::AttributeSet) -> Result<Self, Self::Error> {
        let attributes = value
            .attributes
            .iter()
            .filter(|a| a.is_persistent)
            .map(Attribute::try_from)
            .collect::<Result<_, _>>()?;

        Ok(AttributeSet { attributes })
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute_set::AttributeSet;
    use crate::primitives::attribute_value::AttributeValue;
    use crate::primitives::Attribute;
    use savant_protobuf::generated;

    #[test]
    fn test_attribute_set() {
        assert_eq!(
            AttributeSet {
                attributes: vec![
                    (Attribute::new(
                        "namespace",
                        "name",
                        vec![AttributeValue::string("value", Some(1.0))],
                        &Some("hint"),
                        true,
                        true
                    ))
                ]
            },
            AttributeSet::try_from(&generated::AttributeSet {
                attributes: vec![generated::Attribute {
                    namespace: "namespace".to_string(),
                    name: "name".to_string(),
                    hint: Some("hint".to_string()),
                    is_persistent: true,
                    values: vec![generated::AttributeValue {
                        confidence: Some(1.0),
                        value: Some(generated::attribute_value::Value::String(
                            generated::StringAttributeValueVariant {
                                data: "value".to_string()
                            }
                        ))
                    }],
                    is_hidden: true,
                }]
                .into_iter()
                .collect(),
            })
            .unwrap()
        );
        assert_eq!(
            generated::AttributeSet {
                attributes: vec![generated::Attribute {
                    namespace: "namespace".to_string(),
                    name: "name".to_string(),
                    hint: Some("hint".to_string()),
                    is_persistent: true,
                    values: vec![generated::AttributeValue {
                        confidence: Some(1.0),
                        value: Some(generated::attribute_value::Value::String(
                            generated::StringAttributeValueVariant {
                                data: "value".to_string()
                            }
                        ))
                    }],
                    is_hidden: true,
                }]
                .into_iter()
                .collect(),
            },
            generated::AttributeSet::from(&AttributeSet {
                attributes: vec![
                    (Attribute::new(
                        "namespace",
                        "name",
                        vec![AttributeValue::string("value", Some(1.0))],
                        &Some("hint"),
                        true,
                        true
                    ))
                ]
            })
        );
    }
}

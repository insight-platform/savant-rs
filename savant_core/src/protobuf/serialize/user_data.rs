use crate::primitives::rust::UserData;
use crate::primitives::Attribute;
use crate::protobuf::serialize;
use savant_protobuf::generated;

impl From<&UserData> for generated::UserData {
    fn from(ud: &UserData) -> Self {
        let attributes = ud
            .attributes
            .iter()
            .map(generated::Attribute::from)
            .collect();

        generated::UserData {
            source_id: ud.source_id.clone(),
            attributes,
        }
    }
}

impl TryFrom<&generated::UserData> for UserData {
    type Error = serialize::Error;

    fn try_from(value: &generated::UserData) -> Result<Self, Self::Error> {
        let attributes = value
            .attributes
            .iter()
            .filter(|a| a.is_persistent)
            .map(Attribute::try_from)
            .collect::<Result<_, _>>()?;

        Ok(UserData {
            source_id: value.source_id.clone(),
            attributes,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute_value::AttributeValue;
    use crate::primitives::userdata::UserData;
    use crate::primitives::Attribute;
    use savant_protobuf::generated;

    #[test]
    fn test_user_data() {
        assert_eq!(
            UserData {
                source_id: "source_id".to_string(),
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
            UserData::try_from(&generated::UserData {
                source_id: "source_id".to_string(),
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
            generated::UserData {
                source_id: "source_id".to_string(),
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
            generated::UserData::from(&UserData {
                source_id: "source_id".to_string(),
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

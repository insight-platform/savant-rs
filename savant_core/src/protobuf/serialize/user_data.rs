use crate::primitives::rust::UserData;
use crate::primitives::Attribute;
use crate::protobuf::{generated, serialize};
use hashbrown::HashMap;

impl From<&UserData> for generated::UserData {
    fn from(ud: &UserData) -> Self {
        generated::UserData {
            source_id: ud.source_id.clone(),
            attributes: ud.attributes.values().map(|a| a.into()).collect(),
        }
    }
}

impl TryFrom<&generated::UserData> for UserData {
    type Error = serialize::Error;

    fn try_from(value: &generated::UserData) -> Result<Self, Self::Error> {
        Ok(UserData {
            source_id: value.source_id.clone(),
            attributes: value
                .attributes
                .iter()
                .map(|a| Attribute::try_from(a).map(|a| ((a.namespace.clone(), a.name.clone()), a)))
                .collect::<Result<HashMap<(String, String), Attribute>, _>>()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute_value::{AttributeValue, AttributeValueVariant};
    use crate::primitives::Attribute;

    #[test]
    fn test_user_data() {
        use crate::primitives::userdata::UserData;
        use crate::protobuf::generated;

        assert_eq!(
            UserData {
                source_id: "source_id".to_string(),
                attributes: vec![(
                    ("namespace".to_string(), "name".to_string()),
                    Attribute::new(
                        "namespace".to_string(),
                        "name".to_string(),
                        vec![AttributeValue::new(
                            AttributeValueVariant::String("value".to_string()),
                            Some(1.0)
                        )],
                        Some("hint".to_string()),
                        true,
                        true
                    )
                )]
                .into_iter()
                .collect(),
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
                attributes: vec![(
                    ("namespace".to_string(), "name".to_string()),
                    Attribute::new(
                        "namespace".to_string(),
                        "name".to_string(),
                        vec![AttributeValue::new(
                            AttributeValueVariant::String("value".to_string()),
                            Some(1.0)
                        )],
                        Some("hint".to_string()),
                        true,
                        true
                    )
                )]
                .into_iter()
                .collect(),
            })
        );
    }
}

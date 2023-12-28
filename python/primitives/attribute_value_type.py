from savant_rs.primitives import AttributeValueType

# not AttributeValueType is hashable
#
d = {
    AttributeValueType.Bytes: "Hello",
    AttributeValueType.Integer: 1,
}

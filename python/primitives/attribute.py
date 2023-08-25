from savant_rs.primitives import AttributeValueType, AttributeValue, Attribute

from savant_rs.logging import LogLevel, set_log_level
set_log_level(LogLevel.Trace)

attr = Attribute(namespace="some", name="attr", hint="x", values=[
    AttributeValue.bytes(dims=[8, 3, 8, 8], blob=bytes(3 * 8 * 8), confidence=None),
    AttributeValue.bytes_from_list(dims=[4, 1], blob=[0, 1, 2, 3], confidence=None),
    AttributeValue.integer(1, confidence=0.5),
    AttributeValue.float(1.0, confidence=0.5),
])
print(attr.json)

attr2 = Attribute.from_json(attr.json)
print(attr2.json)

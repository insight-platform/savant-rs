from savant_rs.primitives import AttributeValueType

from savant_rs.logging import LogLevel, set_log_level
set_log_level(LogLevel.Trace)

# not AttributeValueType is hashable
#
d = {
    AttributeValueType.Bytes: "Hello",
    AttributeValueType.Integer: 1,
}

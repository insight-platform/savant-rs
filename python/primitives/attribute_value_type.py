from savant_rs.logging import LogLevel, set_log_level
from savant_rs.primitives import AttributeValueType

set_log_level(LogLevel.Trace)

# not AttributeValueType is hashable
#
d = {
    AttributeValueType.Bytes: "Hello",
    AttributeValueType.Integer: 1,
}

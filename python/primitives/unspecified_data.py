from savant_rs.primitives import UnspecifiedData, Attribute, AttributeValue
from savant_rs.utils.serialization import save_message_to_bytes, load_message_from_bytes, Message

from savant_rs.logging import LogLevel, set_log_level
set_log_level(LogLevel.Trace)

t = UnspecifiedData("abc")
t.set_attribute(Attribute(namespace="some", name="attr", hint="x", values=[AttributeValue.float(1.0, confidence=0.5)]))
print("Before")
print(t.json_pretty())

m = Message.unspecified(t)
s = save_message_to_bytes(m)
new_m = load_message_from_bytes(s)
assert new_m.is_unspecified()

t = new_m.as_unspecified()
assert t.source_id == "abc"

print("After")
print(t.json_pretty())

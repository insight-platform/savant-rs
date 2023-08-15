from savant_rs.primitives import UserData, Attribute, AttributeValue
from savant_rs.utils.serialization import save_message_to_bytes, load_message_from_bytes, Message

from savant_rs.logging import LogLevel, set_log_level
set_log_level(LogLevel.Trace)

t = UserData("abc")
t.set_attribute(Attribute(namespace="some", name="attr", hint="x", values=[AttributeValue.float(1.0, confidence=0.5)]))
print("Before")
print(t.json_pretty())

m = Message.user_data(t)
s = save_message_to_bytes(m)
new_m = load_message_from_bytes(s)
assert new_m.is_user_data()

t = new_m.as_user_data()
assert t.source_id == "abc"

print("After")
print(t.json_pretty())

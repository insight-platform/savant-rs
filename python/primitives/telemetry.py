from savant_rs.primitives import Telemetry, Attribute, AttributeValue
from savant_rs.utils.serialization import save_message_to_bytes, load_message_from_bytes, Message


t = Telemetry("abc")
t.set_attribute(Attribute(creator="some", name="attr", hint="x", values=[AttributeValue.float(1.0, confidence=0.5)]))
print("Before")
print(t.json_pretty())

m = Message.telemetry(t)
s = save_message_to_bytes(m)
new_m = load_message_from_bytes(s)
assert new_m.is_telemetry()

t = new_m.as_telemetry()
print("After")
print(t.json_pretty())

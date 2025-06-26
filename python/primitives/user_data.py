from savant_rs.primitives import AttributeValue, UserData
from savant_rs.utils.serialization import (Message, load_message_from_bytes,
                                           save_message_to_bytes)

t = UserData("abc")
t.set_persistent_attribute(
    namespace="some",
    name="attr",
    hint="x",
    is_hidden=False,
    values=[AttributeValue.float(1.0, confidence=0.5)],
)

pb = t.to_protobuf()
restored = UserData.from_protobuf(pb)
assert t.json == restored.json

print("Before")
print(t.json_pretty)

m = Message.user_data(t)
s = save_message_to_bytes(m)
new_m = load_message_from_bytes(s)
assert new_m.is_user_data()

t = new_m.as_user_data()
assert t.source_id == "abc"

print("After")
print(t.json_pretty)

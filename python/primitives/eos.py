from savant_rs import version
from savant_rs.primitives import EndOfStream
from savant_rs.utils.serialization import save_message_to_bytes, load_message_from_bytes, Message

print("Savant version:", version())

e = EndOfStream("abc")

m = Message.end_of_stream(e)
s = save_message_to_bytes(m)
new_m = load_message_from_bytes(s)
assert new_m.is_end_of_stream()

e = new_m.as_end_of_stream()
assert e.source_id == "abc"

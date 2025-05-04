from savant_rs import version
from savant_rs.primitives import EndOfStream
from savant_rs.utils.serialization import load_message_from_bytes, save_message_to_bytes

print("Savant version:", version())

e = EndOfStream("abc")

m = e.to_message()
s = save_message_to_bytes(m)
new_m = load_message_from_bytes(s)
assert new_m.is_end_of_stream()

e = new_m.as_end_of_stream()
assert e.source_id == "abc"

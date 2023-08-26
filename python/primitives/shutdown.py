from savant_rs.logging import LogLevel, set_log_level
from savant_rs.primitives import Shutdown
from savant_rs.utils.serialization import save_message_to_bytes, load_message_from_bytes, Message

set_log_level(LogLevel.Trace)

e = Shutdown("abc")

m = Message.shutdown(e)
s = save_message_to_bytes(m)
new_m = load_message_from_bytes(s)
assert new_m.is_shutdown()

e = new_m.as_shutdown()
assert e.auth == "abc"

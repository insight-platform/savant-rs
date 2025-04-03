from time import sleep
from savant_rs.utils import relative_time_uuid_v7, incremental_uuid_v7
import uuid

now_uuid = incremental_uuid_v7()
parsed_now_uuid = uuid.UUID(now_uuid)
print("now_uuid: ", now_uuid)
print("parsed_now_uuid: ", parsed_now_uuid)

future_uuid = relative_time_uuid_v7(now_uuid, 1000)
parsed_future_uuid = uuid.UUID(future_uuid)
print("future_uuid: ", future_uuid)
print("parsed_future_uuid: ", parsed_future_uuid)

int_uuid = parsed_now_uuid.int
int_future_uuid = parsed_future_uuid.int

assert int_future_uuid > int_uuid

sleep(1)

int_new_now_uuid = uuid.UUID(incremental_uuid_v7()).int

assert int_new_now_uuid > int_future_uuid


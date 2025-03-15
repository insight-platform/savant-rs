import time

import savant_rs.utils as utils

for i in range(10):
    print(utils.incremental_uuid_v7())

time.sleep(1 / 1000)

for i in range(10):
    print(utils.incremental_uuid_v7())

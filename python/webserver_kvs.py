import threading

import savant_rs.webserver as ws
from savant_rs.logging import LogLevel, set_log_level
from savant_rs.primitives import Attribute, AttributeValue
import savant_rs.webserver.kvs as kvs

set_log_level(LogLevel.Info)

import requests
from time import sleep, time

attr = Attribute(namespace="some", name="attr", hint="x", values=[
    AttributeValue.bytes(dims=[8, 3, 8, 8], blob=bytes(3 * 8 * 8), confidence=None),
    AttributeValue.bytes_from_list(dims=[4, 1], blob=[0, 1, 2, 3], confidence=None),
    AttributeValue.integer(1, confidence=0.5),
    AttributeValue.float(1.0, confidence=0.5),
    AttributeValue.floats([1.0, 2.0, 3.0])
])


def abi():
    global attr

    kvs.set_attributes([attr], 1000)

    attributes = kvs.search_attributes("*", "*")
    assert len(attributes) == 1

    attributes = kvs.search_attributes(None, "*")
    assert len(attributes) == 1

    attribute = kvs.get_attribute("some", "attr")
    assert attribute.name == attr.name and attribute.namespace == attr.namespace

    nonexistent_attribute = kvs.get_attribute("some", "other")
    assert nonexistent_attribute is None

    removed_attribute = kvs.del_attribute("some", "attr")

    kvs.set_attributes([removed_attribute], 500)

    sleep(0.55)

    auto_removed_attribute = kvs.get_attribute("some", "attr")
    assert auto_removed_attribute is None

    kvs.del_attributes("*", "*")


def api(base_url: str):
    global attr
    binary_attributes = kvs.serialize_attributes([attr])

    response = requests.post(f'{base_url}/kvs/set', data=binary_attributes)
    assert response.status_code == 200

    response = requests.post(f'{base_url}/kvs/set-with-ttl/1000', data=binary_attributes)
    assert response.status_code == 200

    response = requests.post(f'{base_url}/kvs/delete/*/*')
    assert response.status_code == 200

    response = requests.post(f'{base_url}/kvs/set', data=binary_attributes)
    assert response.status_code == 200

    response = requests.post(f'{base_url}/kvs/delete-single/some/attr')
    assert response.status_code == 200
    removed_attributes = kvs.deserialize_attributes(response.content)
    assert len(removed_attributes) == 1

    response = requests.post(f'{base_url}/kvs/delete-single/some/attr')
    assert response.status_code == 200
    removed_attributes = kvs.deserialize_attributes(response.content)
    assert len(removed_attributes) == 0

    response = requests.post(f'{base_url}/kvs/set', data=binary_attributes)
    assert response.status_code == 200

    response = requests.get(f'{base_url}/kvs/search/*/*')
    assert response.status_code == 200
    attributes = kvs.deserialize_attributes(response.content)
    assert len(attributes) == 1

    response = requests.get(f'{base_url}/kvs/search-keys/*/*')
    assert response.status_code == 200
    attributes = response.json()
    assert attributes == [["some", "attr"]]

    response = requests.get(f'{base_url}/kvs/get/some/attr')
    assert response.status_code == 200
    attributes = kvs.deserialize_attributes(response.content)
    assert len(attributes) == 1


if __name__ == "__main__":
    abi()
    port = 8080
    ws.init_webserver(port)
    sleep(0.1)
    api(f'http://localhost:{port}')


    def abi_receiver():
        subscription = kvs.KvsSubscription("events", 100)
        counter = 0
        while True:
            counter += 1
            event = subscription.recv()
            if event is None:
                break
            # if counter % 1000 == 0 or counter % 1001 == 0:
            #     print(event)
        print(f'Done: {counter}')


    subscription_thread = threading.Thread(target=abi_receiver)
    subscription_thread.start()

    for _ in range(10_000):
        sleep(0.0001)  # to avoid message drop due to queue overflow
        kvs.set_attributes([attr])
        kvs.del_attribute("some", "attr")

    ws.stop_webserver()
    subscription_thread.join()

# use
# ./websocat -U --ping-interval 1 --ping-timeout 2 ws://localhost:8080/kvs/events/meta
# to see the event metadata

# use
# ./websocat -U --ping-interval 1 --ping-timeout 2 ws://localhost:8080/kvs/events/full
# to see the full event

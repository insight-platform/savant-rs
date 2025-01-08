from savant_rs.metrics import *
from savant_rs.webserver import *
import requests
from time import sleep

set_extra_labels({"extra_label": "extra_value"})

counter = CounterFamily.get_or_create_counter_family(
    name="counter",
    description="counter description",
    label_names=["label1", "label2"],
    unit="unit",
)

gauge = GaugeFamily.get_or_create_gauge_family(
    name="gauge",
    description="gauge description",
    label_names=["label1", "label2"],
    unit=None,
)

assert counter.set(1, ["value1", "value2"]) == 1  # new value returns the same value
assert gauge.set(1.0, ["value1", "value2"]) == 1.0  # new value returns the same value
assert counter.inc(1, ["value1", "value2"]) == 1  # updated value returns the previous value
assert gauge.set(2.0, ["value1", "value2"]) == 1.0  # updated value returns the previous value
assert counter.get(["value1", "value2"]) == 2
assert gauge.get(["value1", "value2"]) == 2.0

set_shutdown_token("shutdown")
init_webserver(8080)

sleep(0.1)

response = requests.get("http://localhost:8080/metrics")
assert response.status_code == 200
print(response.text)
assert "counter_unit_total" in response.text
assert "gauge" in response.text

# set Ctrl+C handler
import signal
import os


def handler(signum, _frame):
    print("Signal handler called with signal", signum)
    stop_webserver()
    print("Webserver stopped")


signal.signal(signal.SIGINT, handler)

response = requests.post("http://localhost:8080/shutdown/shutdown/signal")
assert response.status_code == 200

assert is_shutdown_set()

assert counter.delete(["value1", "value2"]) == 2
assert gauge.delete(["value1", "value2"]) == 2.0

delete_metric_family("counter")
delete_metric_family("gauge")

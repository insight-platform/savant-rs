from random import choice
from timeit import default_timer as timer

from savant_rs.logging import LogLevel, set_log_level
set_log_level(LogLevel.Trace)

from savant_rs.utils.symbol_mapper import RegistrationPolicy, \
    build_model_object_key, \
    is_model_registered, \
    is_object_registered, \
    parse_compound_key, \
    validate_base_key, \
    clear_symbol_maps, \
    dump_registry, \
    get_model_id, \
    get_object_id, \
    get_object_ids, \
    get_model_name, \
    get_object_label, \
    get_object_labels, \
    register_model_objects

models = ["model1", "model2", "model3", "model4", "model5", "model6", "model7", "model8", "model9", "model10"]

clear_symbol_maps()

key = build_model_object_key("model1", "object_1_model1")
assert key == "model1.object_1_model1"

num = 0
for m in models:
    d = dict([(id, f"object_{id}_{m}") for id in range(1000)])
    t = timer()
    register_model_objects(m, d, RegistrationPolicy.ErrorIfNonUnique)
    num += timer() - t

print(f"Time to register: {num}")

model = get_model_name(1)
assert model == "model2"

assert is_model_registered("model2")
assert is_object_registered("model2", "object_1_model2")

m, l = parse_compound_key("model2.object_1_model2")
key = validate_base_key("model2")

label = get_object_label(1, 0)
assert label == "object_0_model2"


def simple_count():
    total_time = timer()
    random_model = choice(models)
    random_object = f"object_{choice(range(1000))}_{random_model}"
    num = 1000
    for _ in range(num):
        get_object_id(model_name=random_model, object_label=random_object)
    return num, timer() - total_time


num, total = simple_count()
print(f"Time to get in individually (total: {total})")


def batched_count():
    total_time = timer()
    random_model = choice(models)
    random_objects = [f"object_{choice(range(1000))}_{random_model}" for _ in range(100)]
    num = 10
    for _ in range(num):
        get_object_ids(model_name=random_model, object_labels=random_objects)
    return num, timer() - total_time

num, single_total_time = batched_count()
num, total = batched_count()

print(f"Time to get in bulk(total: {total})")

ids = get_object_ids(model_name="model1", object_labels=["object_1_model1", "object_2_model1", "object_X_model1"])
assert ids == [('object_1_model1', 1), ('object_2_model1', 2), ('object_X_model1', 1000)]

labels = get_object_labels(model_id=0, object_ids=[1, 2, 100000])
assert labels == [(1, 'object_1_model1'), (2, 'object_2_model1'), (100000, None)]

# pprint(dump_registry())

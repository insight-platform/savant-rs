from savant_rs.utils import RegistrationPolicy, \
    build_model_object_key, \
    is_model_registered, \
    is_object_registered, \
    parse_compound_key, \
    validate_base_key, \
    clear_symbol_maps, \
    get_model_id, \
    get_object_id, \
    get_object_ids, \
    get_model_name, \
    get_object_label, \
    get_object_labels, \
    register_model_objects

from random import choice

from timeit import default_timer as timer
models = ["model1", "model2", "model3", "model4", "model5", "model6", "model7", "model8", "model9", "model10"]

clear_symbol_maps()

key = build_model_object_key("model1", "object_1_model1")
assert key == "model1.object_1_model1"

res = 0
for m in models:
    d = dict([ (id, f"object_{id}_{m}") for id in range(1000)])
    t = timer()
    register_model_objects(m, d, RegistrationPolicy.ErrorIfNonUnique)
    res += timer() - t

print(f"Time to register: {res}")

model = get_model_name(1)
assert model == "model2"

assert is_model_registered("model2")
assert is_object_registered("model2", "object_1_model2")

m, l = parse_compound_key("model2.object_1_model2")
key = validate_base_key("model2")

label = get_object_label(1, 0)
assert label == "object_0_model2"

res = 0
for _ in range(10_000):
    random_model = choice(models)
    random_object = f"object_{choice(range(1000))}_{random_model}"
    t = timer()
    m = get_model_id(model_name=random_model)
    m, o = get_object_id(model_name=random_model, object_label=random_object)
    res += timer() - t

print(f"Time to get: {res}")

ids = get_object_ids(model_name="model1", object_labels=["object_1_model1", "object_2_model1"])
assert ids == [('object_1_model1', 1), ('object_2_model1', 2)]

labels = get_object_labels(model_id=0, object_ids=[1, 2])
assert labels == [(1, 'object_1_model1'), (2, 'object_2_model1')]



"""Thread-safe singleton metaclass."""
import threading
from random import random
from threading import Lock
from typing import Any, Dict, List, Tuple, Optional
from numba import jit


class SingletonMeta(type):
    """Thread-safe singleton metaclass."""

    def __new__(cls, name, bases, attrs):
        # Assume the target class is created
        # (i.e. this method to be called) in the main thread.
        _cls = super().__new__(cls, name, bases, attrs)
        _cls.__shared_instance_lock__ = Lock()
        return _cls

    def __call__(cls, *args, **kwargs):
        with cls.__shared_instance_lock__:
            try:
                return cls.__shared_instance__
            except AttributeError:
                cls.__shared_instance__ = super(SingletonMeta, cls).__call__(
                    *args, **kwargs
                )
                return cls.__shared_instance__


class ModelObjectRegistry(metaclass=SingletonMeta):
    """Model.Object registry.

    Stores model+object with relevant model_uid+class_id to manage model
    dependencies (operate-on).
    """

    REGISTRY_KEY_SEPARATOR: str = '.'

    def __init__(self):
        # {'model_name.object_label': (model_uid, object_class_id)}
        self._object_registry: Dict[str, Tuple[int, int]] = {}
        # {'model_name': model_uid}
        self._model_registry: Dict[str, int] = {}
        self._uid2name: Dict[int, str] = {}
        # model_uid sequential generation
        self._model_uid = 0

    def __str__(self):
        return f'{self._model_registry}, {self._object_registry}'

    @property
    def new_model_uid(self):
        """Generate new model uid."""
        self._model_uid += 1
        return self._model_uid

    @staticmethod
    def model_object_key(model_name: str, object_label: str) -> str:
        """Returns unique key for specified model object type, used in the
        registry and in NvDsObjectMeta.obj_label value."""

        if not model_name:  # frame object
            return object_label
        return f'{model_name}{ModelObjectRegistry.REGISTRY_KEY_SEPARATOR}{object_label}'

    @staticmethod
    def parse_model_object_key(model_object_key: str) -> Tuple[str, str]:
        """Parses model object type key, returns model name and object
        label."""

        model_name, object_label = '', model_object_key
        if ModelObjectRegistry.REGISTRY_KEY_SEPARATOR in model_object_key:
            model_name, object_label = model_object_key.split(
                ModelObjectRegistry.REGISTRY_KEY_SEPARATOR
            )
        return model_name, object_label

    def register_model(
        self,
        model_element_name: str,
        model_output_object_labels: Optional[Dict[int, str]] = None,
    ) -> int:
        """Register a model element."""
        model_uid = self.new_model_uid
        self._model_registry[model_element_name] = model_uid
        self._uid2name[model_uid] = model_element_name
        # register model objects
        if model_output_object_labels:
            for class_id, label in model_output_object_labels.items():
                self._object_registry[
                    self.model_object_key(model_element_name, label)
                ] = (model_uid, class_id)
        return model_uid

    def get_name(self, uid: int) -> str:
        """Get model name from uid."""
        return self._uid2name[uid]

    def get_model_uid(self, model_name: str) -> int:
        """Get model uid from name."""
        return self._model_registry[model_name]

    def is_model_object_key_registered(self, model_object_key: str) -> bool:
        """Check if model object key is registered."""
        return model_object_key in self._object_registry

    def get_model_object_ids(
        self,
        model_object_key: Optional[str] = None,
        model_name: Optional[str] = None,
        object_label: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Returns tuple(model_uid, class_id) for specified key or specified
        model object type. Adds new record if there is no model info in the
        registry.

        :param model_object_key:
        :param model_name:
        :param object_label:
        :return: (model_uid, class_id)
        """
        assert model_object_key is not None or (
            model_name is not None and object_label is not None
        )

        if model_object_key is None:
            model_object_key = ModelObjectRegistry.model_object_key(
                model_name, object_label
            )

        if model_object_key not in self._object_registry:
            model_name, object_label = self.parse_model_object_key(model_object_key)
            model_uid, class_id = None, 0
            # try to find model_uid for model first
            if model_name:
                model_class_ids = [
                    v
                    for k, v in self._object_registry.items()
                    if k.startswith(
                        f'{model_name}{ModelObjectRegistry.REGISTRY_KEY_SEPARATOR}'
                    )
                ]
                if model_class_ids:
                    model_uid = model_class_ids[0][0]
                    class_id = max(class_id for _, class_id in model_class_ids) + 1

            if not model_uid:
                try:
                    model_uid = self.get_model_uid(model_name)
                except KeyError:
                    model_uid = self.register_model(model_name)

            self._object_registry[model_object_key] = model_uid, class_id

        return self._object_registry[model_object_key]


if __name__ == '__main__':
    from timeit import default_timer as timer
    from random import choice
    registry = ModelObjectRegistry()
    models = ["model1", "model2", "model3", "model4", "model5", "model6", "model7", "model8", "model9", "model10"]


    @jit(nopython=True, nogil=True)
    def plus_one() -> int:
        return 1


    cntr = 0
    exit_flag = False


    def threaded_count():
        global cntr, exit_flag
        while not exit_flag:
            cntr += plus_one()

    t = timer()
    cnt = 0
    while cnt < 10_000_000:
        cnt += plus_one()

    mil_count = timer() - t
    print(f"Time to count to million: {mil_count}")

    res = 0
    for m in models:
        d = dict([(id, f"object_{id}_{m}") for id in range(1000)])
        t = timer()
        registry.register_model(m, d)
        res += timer() - t

    print(f"Time to register: {res}")

    def simple_count():
        res = 0
        total_time = timer()
        for _ in range(10_000_000):
            random_model = choice(models)
            random_object = f"object_{choice(range(1000))}_{random_model}"
            t = timer()
            m = registry.get_model_uid(random_model)
            m, o = registry.get_model_object_ids(model_name=random_model, object_label=random_object)
            res += timer() - t
        return res, timer() - total_time


    cntr = 0
    exit_flag = False
    t = threading.Thread(target=threaded_count)
    t.start()
    res, total_time = simple_count()
    exit_flag = True
    t.join()
    print(f"Time to get in individually ({total_time}): {res}, cnt: {cntr}, time spent in counter {cntr / 10_000_000 * mil_count}")

Configuration
=============

The Meta Merge service is configured using a JSON configuration file that defines ingress sources, egress destination, Python handler initialization, merge callbacks, and queue settings.

Configuration File Structure
-----------------------------

The configuration file has three main sections:

.. code-block:: json

   {
       "ingress": [ /* Ingress configurations */ ],
       "egress": { /* Egress configuration */ },
       "common": { /* Common service settings */ }
   }

Ingress Configuration
---------------------

The ``ingress`` section defines input sources for the Meta Merge service. Each ingress configuration specifies a ZeroMQ socket for receiving messages, an optional Python handler, and an EOS (end-of-stream) policy.

.. code-block:: json

   {
       "ingress": [
           {
               "name": "ingress1",
               "socket": {
                   "url": "router+bind:tcp://127.0.0.1:6667",
                   "options": {
                       "receive_timeout": {"secs": 1, "nanos": 0},
                       "receive_hwm": 1000,
                       "topic_prefix_spec": "none",
                       "source_cache_size": 1000,
                       "fix_ipc_permissions": 511,
                       "inflight_ops": 100
                   }
               },
               "eos_policy": "allow"
           },
           {
               "name": "ingress2",
               "socket": {
                   "url": "router+bind:tcp://127.0.0.1:6668"
               },
               "eos_policy": "deny"
           }
       ]
   }

Ingress Parameters:

* **name** (string): Unique identifier for the ingress source
* **socket** (object): ZeroMQ socket configuration
   * **url** (string): ZeroMQ socket URL (supports router, rep, sub patterns)
   * **options** (object, optional): Socket-specific options
      * **receive_timeout** (duration): Timeout for receive operations
      * **receive_hwm** (integer): High water mark for incoming messages
      * **topic_prefix_spec** (string): Topic prefix specification
      * **source_cache_size** (integer): Size of source identifier cache
      * **fix_ipc_permissions** (integer): IPC socket permission mask
      * **inflight_ops** (integer): Maximum concurrent operations
* **handler** (string, optional): Name of Python handler function for processing incoming messages
* **eos_policy** (string, optional): Controls whether EOS messages from this ingress are forwarded. Values: ``"allow"`` or ``"deny"``

.. important::

   The ``eos_policy`` with value ``"allow"`` must be set on **at most one** ingress. If multiple ingress streams have ``eos_policy`` set to ``"allow"``, the service will reject the configuration because it could lead to multiple EOS messages being delivered to downstream services. Set it to ``"deny"`` on all other ingress streams that carry copies of the same source.

Egress Configuration
--------------------

The ``egress`` section defines the single output destination for merged frames. Unlike the Router service, Meta Merge has a single egress.

.. code-block:: json

   {
       "egress": {
           "socket": {
               "url": "dealer+bind:tcp://127.0.0.1:3333"
           }
       }
   }

Egress Parameters:

* **socket** (object): ZeroMQ socket configuration
   * **url** (string): ZeroMQ socket URL for outgoing messages

Common Configuration
--------------------

The ``common`` section contains service-wide settings including Python handler initialization, callbacks configuration, queue settings, and idle behavior.

.. code-block:: json

   {
       "common": {
           "init": {
               "python_root": "${PYTHON_MODULE_ROOT:-/opt/python}",
               "module_name": "module",
               "function_name": "init",
               "args": [
                   {
                       "params": {
                           "home_dir": "${HOME}",
                           "user_name": "${USER}"
                       }
                   }
               ]
           },
           "callbacks": {
               "on_merge": "merge_handler",
               "on_head_expire": "head_expired_handler",
               "on_head_ready": "head_ready_handler",
               "on_late_arrival": "late_arrival_handler",
               "on_unsupported_message": "unsupported_message_handler",
               "on_send": "send_handler"
           },
           "idle_sleep": {"secs": 0, "nanos": 1000},
           "queue": {
               "max_duration": {"secs": 5, "nanos": 0}
           }
       }
   }

Init Parameters
^^^^^^^^^^^^^^^^

* **init** (object): Python handler initialization
   * **python_root** (string): Path to Python modules directory
   * **module_name** (string): Python module name to import
   * **function_name** (string): Initialization function name
   * **args** (object, optional): Arguments passed to initialization function

Callbacks Parameters
^^^^^^^^^^^^^^^^^^^^^

* **callbacks** (object): Python callback handler names
   * **on_merge** (string, required): Handler called when a frame copy arrives for merging
   * **on_head_expire** (string, required): Handler called when the head frame expires
   * **on_head_ready** (string, required): Handler called when the head frame is marked ready
   * **on_late_arrival** (string, required): Handler called when a frame arrives after the queue head
   * **on_unsupported_message** (string, optional): Handler called for non-video, non-EOS messages
   * **on_send** (string, optional): Handler called before sending a message, can override the topic

Queue Parameters
^^^^^^^^^^^^^^^^^

* **queue** (object, optional): Merge queue configuration
   * **max_duration** (duration): Maximum time a frame can wait in the queue before being expired. Default: 5 seconds

Other Parameters
^^^^^^^^^^^^^^^^^

* **idle_sleep** (duration, optional): Sleep duration when no messages are available. Default: 1ms

Environment Variable Substitution
----------------------------------

Configuration values support environment variable substitution using the format ``${VAR_NAME:-default_value}``:

.. code-block:: json

   {
       "common": {
           "init": {
               "python_root": "${PYTHON_MODULE_ROOT:-/opt/python}",
               "args": [
                   {
                       "params": {
                           "home_dir": "${HOME}",
                           "user_name": "${USER}"
                       }
                   }
               ]
           }
       }
   }

Python Handler Development
--------------------------

Python handlers are callable objects registered during initialization. Below is a complete example of a Python handler module for Meta Merge:

.. code-block:: python

   from typing import Any, Optional
   from savant_rs import register_handler, version
   from savant_rs.logging import log, LogLevel
   from savant_rs.primitives import VideoFrame
   from savant_rs.utils.serialization import Message


   class EgressItem:
       """Type hint interface for the EgressItem passed to callbacks.

       Attributes:
           video_frame: The video frame being merged.
           state: A dictionary for accumulating merge state across callbacks.
           data: List of binary data payloads.
           labels: List of string labels.
       """
       @property
       def video_frame(self) -> VideoFrame: ...
       @video_frame.setter
       def video_frame(self, video_frame: VideoFrame): ...
       @property
       def state(self) -> dict[str, Any]: ...
       @state.setter
       def state(self, state: dict[str, Any]): ...
       @property
       def data(self) -> list[bytes]: ...
       @data.setter
       def data(self, data: list[bytes]): ...
       @property
       def labels(self) -> list[str]: ...
       @labels.setter
       def labels(self, labels: list[str]): ...


   class MergeHandler:
       def __call__(
           self,
           ingress_name: str,
           topic: str,
           current_state: EgressItem,
           incoming_state: Optional[EgressItem],
       ) -> bool:
           """Called when a frame copy arrives for merging.

           :param ingress_name: Name of the ingress that received the message.
           :param topic: ZMQ topic of the message.
           :param current_state: Current state of the egress item in the queue.
           :param incoming_state: Incoming copy of the frame, or None for the
               first arrival (which automatically becomes current_state).
           :return: True if the merge is complete and the frame should be sent.
           """
           if incoming_state is not None:
               # Merge attributes from the incoming frame
               pass
           return False


   class HeadExpiredHandler:
       def __call__(self, state: EgressItem) -> Optional[Message]:
           """Called when the head of the queue has expired (exceeded max_duration).

           :param state: The expired egress item.
           :return: A Message to send downstream, or None to drop the frame.
           """
           return None


   class HeadReadyHandler:
       def __call__(self, state: EgressItem) -> Optional[Message]:
           """Called when the head of the queue is marked as ready.

           :param state: The ready egress item.
           :return: A Message to send downstream, or None to drop the frame.
           """
           return None


   class LateArrivalHandler:
       def __call__(self, state: EgressItem):
           """Called when a frame arrives after the queue head has advanced.

           :param state: The late-arriving egress item.
           """
           pass


   class UnsupportedMessageHandler:
       def __call__(
           self,
           ingress_name: str,
           topic: str,
           message: Message,
           data: list[bytes],
       ):
           """Called for messages that are neither video frames nor EOS.

           :param ingress_name: Name of the ingress.
           :param topic: ZMQ topic of the message.
           :param message: The unsupported message object.
           :param data: Data payloads.
           """
           pass


   class SendHandler:
       def __call__(
           self,
           message: Message,
           message_state: Optional[dict[Any, Any]],
           data: list[bytes],
           labels: list[str],
       ) -> Optional[str]:
           """Called before sending a message to the egress.

           :param message: The message to send.
           :param message_state: The state dictionary accumulated during merging.
           :param data: Data payloads.
           :param labels: Labels.
           :return: Optional topic string to override the default (source ID),
               or None to use the default.
           """
           return None


   def init(params: Any):
       """Initialization function called once at service startup.

       Register all callback handlers here.
       """
       register_handler("merge_handler", MergeHandler())
       register_handler("head_expired_handler", HeadExpiredHandler())
       register_handler("head_ready_handler", HeadReadyHandler())
       register_handler("late_arrival_handler", LateArrivalHandler())
       register_handler("unsupported_message_handler", UnsupportedMessageHandler())
       register_handler("send_handler", SendHandler())
       return True

The EgressItem Object
^^^^^^^^^^^^^^^^^^^^^^

The ``EgressItem`` object passed to callbacks provides the following properties:

* **video_frame** (``VideoFrame``): The video frame being merged. Supports getting and setting.
* **state** (``dict``): A dictionary for accumulating merge state across callbacks. Persist counters, flags, or any other state between ``on_merge`` calls here.
* **data** (``list[bytes]``): Binary data payloads associated with the frame.
* **labels** (``list[str]``): String labels associated with the frame.

All properties support both getting and setting, so handlers can modify the frame, state, data, and labels in-place.

Complete Configuration Example
-------------------------------

Below is a complete configuration example with two ingress streams and all callback handlers:

.. code-block:: json

   {
       "ingress": [
           {
               "name": "detection_pipeline",
               "socket": {
                   "url": "router+bind:tcp://0.0.0.0:6667",
                   "options": {
                       "receive_timeout": {"secs": 1, "nanos": 0},
                       "receive_hwm": 1000,
                       "topic_prefix_spec": "none",
                       "source_cache_size": 1000,
                       "inflight_ops": 100
                   }
               },
               "eos_policy": "allow"
           },
           {
               "name": "classification_pipeline",
               "socket": {
                   "url": "router+bind:tcp://0.0.0.0:6668",
                   "options": {
                       "receive_timeout": {"secs": 1, "nanos": 0},
                       "receive_hwm": 1000,
                       "topic_prefix_spec": "none",
                       "source_cache_size": 1000,
                       "inflight_ops": 100
                   }
               },
               "eos_policy": "deny"
           }
       ],
       "egress": {
           "socket": {
               "url": "dealer+bind:tcp://0.0.0.0:3333"
           }
       },
       "common": {
           "init": {
               "python_root": "${PYTHON_MODULE_ROOT:-/opt/python}",
               "module_name": "module",
               "function_name": "init",
               "args": null
           },
           "callbacks": {
               "on_merge": "merge_handler",
               "on_head_expire": "head_expired_handler",
               "on_head_ready": "head_ready_handler",
               "on_late_arrival": "late_arrival_handler",
               "on_unsupported_message": "unsupported_message_handler",
               "on_send": "send_handler"
           },
           "idle_sleep": {"secs": 0, "nanos": 1000000},
           "queue": {
               "max_duration": {"secs": 5, "nanos": 0}
           }
       }
   }

This configuration:

1. Accepts frames from two pipelines: ``detection_pipeline`` on port 6667 and ``classification_pipeline`` on port 6668.
2. Only the ``detection_pipeline`` ingress forwards EOS messages (``eos_policy: "allow"``), while the ``classification_pipeline`` suppresses them (``eos_policy: "deny"``).
3. Merged frames are sent to the egress on port 3333.
4. Python handlers are loaded from the ``/opt/python`` directory (configurable via the ``PYTHON_MODULE_ROOT`` environment variable).
5. Frames that are not marked ready within 5 seconds are expired.
6. The service sleeps for 1ms when idle (no incoming messages).

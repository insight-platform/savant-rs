Configuration
=============

The Router service is configured using a JSON configuration file that defines ingress sources, egress destinations, routing rules, and Python handler specifications.

Configuration File Structure
-----------------------------

The configuration file has three main sections:

.. code-block:: json

   {
       "ingres": [ /* Ingress configurations */ ],
       "egress": [ /* Egress configurations */ ],
       "common": { /* Common service settings */ }
   }

Ingress Configuration
---------------------

The ``ingres`` section defines input sources for the Router service. Each ingress configuration specifies a ZeroMQ socket for receiving messages and an optional Python handler for processing.

.. code-block:: json

   {
       "ingres": [
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
               "handler": "ingress_handler"
           }
       ]
   }

Ingress Parameters:

* **name** (string): Unique identifier for the ingress source
* **socket** (object): ZeroMQ socket configuration
   * **url** (string): ZeroMQ socket URL (supports router, rep, sub patterns)
   * **options** (object): Socket-specific options
      * **receive_timeout** (duration): Timeout for receive operations
      * **receive_hwm** (integer): High water mark for incoming messages
      * **topic_prefix_spec** (string): Topic prefix specification
      * **source_cache_size** (integer): Size of source identifier cache
      * **fix_ipc_permissions** (integer): IPC socket permission mask
      * **inflight_ops** (integer): Maximum concurrent operations
* **handler** (string, optional): Name of Python handler function for processing incoming messages

Egress Configuration
--------------------

The ``egress`` section defines output destinations for routed messages. Each egress can have matching conditions, source/topic mappers, and backpressure control.

.. code-block:: json

   {
       "egress": [
           {
               "name": "egress1",
               "socket": {
                   "url": "dealer+bind:tcp://127.0.0.1:3333"
               },
               "high_watermark": 0.9,
               "matcher": "[label1] & [label2]",
               "source_mapper": "egress_source_handler",
               "topic_mapper": "egress_topic_handler"
           },
           {
               "name": "egress2",
               "socket": {
                   "url": "dealer+bind:tcp://127.0.0.1:3334"
               },
               "high_watermark": 0.9,
               "matcher": "[label1] & ([label3] | [label2])"
           }
       ]
   }

Egress Parameters:

* **name** (string): Unique identifier for the egress destination
* **socket** (object): ZeroMQ socket configuration
   * **url** (string): ZeroMQ socket URL for outgoing messages
* **high_watermark** (float, optional): Backpressure threshold (0.0-1.0), default: 0.9
* **matcher** (string, optional): Boolean expression for message label matching
* **source_mapper** (string, optional): Name of Python handler for source ID transformation
* **topic_mapper** (string, optional): Name of Python handler for topic transformation

Matching Expressions
^^^^^^^^^^^^^^^^^^^^^

The ``matcher`` field supports boolean logic with message labels:

* **[label_name]**: Check if label exists
* **&**: Logical AND
* **|**: Logical OR
* **()**: Grouping for precedence

Examples:

* ``"[vehicle]"``: Messages with "vehicle" label
* ``"[person] & [alert]"``: Messages with both "person" and "alert" labels
* ``"[vehicle] | [person]"``: Messages with either "vehicle" or "person" labels
* ``"[urgent] & ([person] | [vehicle])"``: Complex conditions with grouping

Common Configuration
--------------------

The ``common`` section contains service-wide settings including Python handler initialization and performance tuning.

.. code-block:: json

   {
       "common": {
           "init": {
               "python_root": "${PYTHON_MODULE_ROOT:-/opt/python}",
               "module_name": "handlers",
               "function_name": "init",
               "args": {
                   "config_param": "value"
               }
           },
           "name_cache": {
               "ttl": {"secs": 10, "nanos": 0},
               "size": 1000
           },
           "idle_sleep": {"secs": 0, "nanos": 1000000}
       }
   }

Common Parameters:

* **init** (object): Python handler initialization
   * **python_root** (string): Path to Python modules directory
   * **module_name** (string): Python module name to import
   * **function_name** (string): Initialization function name
   * **args** (object, optional): Arguments passed to initialization function
* **name_cache** (object, optional): Label caching configuration
   * **ttl** (duration): Time-to-live for cache entries, default: 1 second
   * **size** (integer): Maximum cache size, default: 1000
* **idle_sleep** (duration, optional): Sleep duration when no messages, default: 1ms

Environment Variable Substitution
----------------------------------

Configuration values support environment variable substitution using the format ``${VAR_NAME:-default_value}``:

.. code-block:: json

   {
       "common": {
           "init": {
               "python_root": "${PYTHON_MODULE_ROOT:-/opt/python}",
               "args": {
                   "home_dir": "${HOME}",
                   "user_name": "${USER}"
               }
           }
       }
   }

Python Handler Development
--------------------------

Python handlers are functions that process messages at different stages of routing. Create a Python module with handler classes:

.. code-block:: python

   from savant_rs import register_handler
   from savant_rs.utils.serialization import Message

   class IngressHandler:
       def __call__(self, message_id: int, ingress_name: str, 
                    topic: str, message: Message):
           # Process incoming message
           if topic == "detection":
               message.labels = ["processed", "vehicle"]
           return message

   class EgressSourceHandler:
       def __call__(self, message_id: int, egress_name: str, 
                    source: str, labels: list[str]):
           # Transform source ID based on routing logic
           return f"transformed_{source}"

   class EgressTopicHandler:
       def __call__(self, message_id: int, egress_name: str, 
                    topic: str, labels: list[str]):
           # Transform topic for specific egress
           return f"processed_{topic}"

   def init(params):
       """Called once during service initialization"""
       register_handler("ingress_handler", IngressHandler())
       register_handler("egress_source_handler", EgressSourceHandler())
       register_handler("egress_topic_handler", EgressTopicHandler())
       return True


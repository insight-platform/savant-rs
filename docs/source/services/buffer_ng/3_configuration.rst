Configuration
=============

The Buffer NG service uses JSON configuration files to define its behavior. The configuration is divided into several sections: ingress, egress, and common settings.

Configuration Structure
------------------------

The configuration file has the following structure:

.. code-block:: json

   {
     "ingress": {
       "socket": {
         "url": "tcp://source:5555",
         "options": { ... }
       }
     },
     "egress": {
       "socket": {
         "url": "tcp://sink:5556",
         "options": { ... }
       }
     },
     "common": {
       "message_handler_init": { ... },
       "telemetry": { ... },
       "buffer": { ... },
       "idle_sleep": { ... }
     }
   }

Ingress Configuration
---------------------

The ingress section defines how the service receives messages from upstream sources.

Socket Configuration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   "ingress": {
     "socket": {
       "url": "router+bind:tcp://127.0.0.1:6666",
       "options": {
         "receive_timeout": {
           "secs": 1,
           "nanos": 0
         },
         "receive_hwm": 1000,
         "topic_prefix_spec": "none",
         "source_cache_size": 1000,
         "fix_ipc_permissions": 511,
         "inflight_ops": 100
       }
     }
   }

**Parameters:**

- ``url``: ZeroMQ socket URL for receiving messages. Example patterns:
  - ``router+bind:tcp://host:port`` - Router socket binding to TCP.
  - ``dealer+connect:tcp://host:port`` - Dealer socket connecting to TCP
  - ``sub+bind:tcp://host:port`` - Subscriber socket binding to TCP.
  - ``sub+connect:ipc:///path/to/socket`` - Subscriber socket connecting to IPC.

- ``receive_timeout``: Timeout for receiving messages (in seconds and nanoseconds).
- ``receive_hwm``: High water mark for receiving messages (queue size limit).
- ``topic_prefix_spec``: Topic prefix specification ("none" for no prefix).
- ``source_cache_size``: Size of the source cache for connection management.
- ``fix_ipc_permissions``: IPC socket permissions (Unix-style octal).
- ``inflight_ops``: Number of in-flight operations for connection management.

Egress Configuration
--------------------

The egress section defines how the service sends messages to downstream destinations.

Socket Configuration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   "egress": {
     "socket": {
       "url": "dealer+bind:tcp://127.0.0.1:6667",
       "options": {
         "send_timeout": {
           "secs": 1,
           "nanos": 0
         },
         "send_retries": 3,
         "receive_timeout": {
           "secs": 1,
           "nanos": 0
         },
         "receive_retries": 3,
         "send_hwm": 1000,
         "receive_hwm": 1000,
         "inflight_ops": 100
       }
     }
   }

**Parameters:**

- ``url``: ZeroMQ socket URL for sending messages. Example patterns:
  - ``dealer+bind:tcp://host:port`` - Dealer socket binding to TCP.
  - ``push+connect:tcp://host:port`` - Push socket connecting to TCP.
  - ``pub+bind:tcp://host:port`` - Publisher socket binding to TCP.
  - ``pub+connect:ipc:///path/to/socket`` - Push socket binding to IPC.

- ``send_timeout``: Timeout for sending messages (in seconds and nanoseconds). 
- ``send_retries``: Number of retries for failed send operations.
- ``receive_timeout``: Timeout for receiving acknowledgments (in seconds and nanoseconds).
- ``receive_retries``: Number of retries for failed receive operations.
- ``send_hwm``: High water mark for sending messages (queue size limit).
- ``receive_hwm``: High water mark for receiving acknowledgments (queue size limit).
- ``inflight_ops``: Number of in-flight operations for connection management.

Common Configuration
--------------------

The common section contains settings that apply to the entire service.

Buffer Configuration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   "buffer": {
     "path": "/tmp/buffer",
     "max_length": 1000000,
     "full_threshold_percentage": 90,
     "reset_on_start": true
   }

**Parameters:**

- ``path``: Path to the RocksDB database directory for storing buffered messages
- ``max_length``: Maximum number of messages that can be stored in the buffer
- ``full_threshold_percentage``: Percentage threshold (0-100) at which the buffer is considered "full" for monitoring purposes
- ``reset_on_start``: Whether to clear the buffer when the service starts (true) or preserve existing data (false)

Message Handler Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   "message_handler_init": {
     "python_root": "/opt/python",
     "module_name": "module",
     "function_name": "init",
     "args": [
       {
         "params": {
           "home_dir": "/home/user",
           "user_name": "user"
         }
       }
     ],
     "invocation_context": "AfterReceive"
   }

**Parameters:**

- ``python_root``: Root directory for Python modules
- ``module_name``: Name of the Python module to import
- ``function_name``: Name of the function to call for initialization
- ``args``: Arguments to pass to the initialization function (optional)
- ``invocation_context``: When to invoke the handler ("AfterReceive" or "BeforeSend")

Telemetry Configuration
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   "telemetry": {
     "port": 8080,
     "stats_log_interval": {
       "secs": 60,
       "nanos": 0
     },
     "metrics_extra_labels": null
   }

**Parameters:**

- ``port``: Port number for the web-based telemetry interface
- ``stats_log_interval``: Interval for logging statistics (in seconds and nanoseconds)
- ``metrics_extra_labels``: Additional labels to include in metrics (optional)

Idle Sleep Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   "idle_sleep": {
     "secs": 0,
     "nanos": 1000
   }

**Parameters:**

- ``secs``: Seconds component of the idle sleep duration
- ``nanos``: Nanoseconds component of the idle sleep duration

Environment Variable Substitution
---------------------------------

The configuration file supports environment variable substitution using the `${VARIABLE_NAME:-default_value}` syntax:

.. code-block:: json

   {
     "ingress": {
       "socket": {
         "url": "${ZMQ_SRC_ENDPOINT}"
       }
     },
     "egress": {
       "socket": {
         "url": "${ZMQ_SINK_ENDPOINT}"
       }
     },
     "common": {
       "buffer": {
         "path": "${BUFFER_PATH:-/tmp/buffer}",
         "max_length": ${BUFFER_LEN:-1000000},
         "full_threshold_percentage": ${BUFFER_THRESHOLD_PERCENTAGE:-90},
         "reset_on_start": ${BUFFER_RESET_ON_RESTART:-true}
       },
       "telemetry": {
         "stats_log_interval": {
           "secs": ${STATS_LOG_INTERVAL:-60},
           "nanos": 0
         },
         "metrics_extra_labels": ${METRICS_EXTRA_LABELS:-null}
       }
     }
   }

Python Handler Development
--------------------------

Buffer NG supports Python handlers for custom message processing. The handler should implement the following interface:

.. code-block:: python

   def init(params: Any) -> Callable:
       """
       Initialize the message handler.
       
       :param params: Configuration parameters passed from the service
       :return: Message handler function or None
       """
       return MessageHandler()
       # or
       return None # to ignore python handler

   class MessageHandler:
       def __call__(self, topic: str, message: Message) -> (str, Message):
           """
           Process a message.
           
           :param topic: ZMQ topic of the message
           :param message: Message object to process
           :return: Tuple of (topic, message) or None to drop the message
           """
           # Custom processing logic here
           return topic, message
           # or
           return None # to drop the message

The handler can be invoked at two points:

- **AfterReceive**: Called after receiving a message from ingress, before storing in buffer
- **BeforeSend**: Called after retrieving a message from buffer, before sending to egress

Configuration Examples
----------------------

Basic Configuration
^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "ingress": {
       "socket": {
         "url": "router+bind:tcp://127.0.0.1:6666"
       }
     },
     "egress": {
       "socket": {
         "url": "dealer+bind:tcp://127.0.0.1:6667"
       }
     },
     "common": {
       "buffer": {
         "path": "/tmp/buffer",
         "max_length": 1000000,
         "full_threshold_percentage": 90,
         "reset_on_start": true
       },
       "telemetry": {
         "port": 8080
       }
     }
   }

Full Configuration with Python Handlers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "ingress": {
       "socket": {
         "url": "router+bind:tcp://127.0.0.1:6666",
         "options": {
           "receive_timeout": {
             "secs": 1,
             "nanos": 0
           },
           "receive_hwm": 1000,
           "topic_prefix_spec": "none",
           "source_cache_size": 1000,
           "fix_ipc_permissions": 511,
           "inflight_ops": 100
         }
       }
     },
     "egress": {
       "socket": {
         "url": "dealer+bind:tcp://127.0.0.1:6667",
         "options": {
           "send_timeout": {
             "secs": 1,
             "nanos": 0
           },
           "send_retries": 3,
           "receive_timeout": {
             "secs": 1,
             "nanos": 0
           },
           "receive_retries": 3,
           "send_hwm": 1000,
           "receive_hwm": 1000,
           "inflight_ops": 100
         }
       }
     },
     "common": {
       "idle_sleep": {
         "secs": 0,
         "nanos": 1000
       },
       "message_handler_init": {
         "python_root": "/opt/python",
         "module_name": "module",
         "function_name": "init",
         "args": [
           {
             "params": {
               "home_dir": "/home/user",
               "user_name": "user"
             }
           }
         ],
         "invocation_context": "AfterReceive"
       },
       "telemetry": {
         "port": 8080,
         "stats_log_interval": {
           "secs": 60,
           "nanos": 0
         },
         "metrics_extra_labels": null
       },
       "buffer": {
         "path": "/tmp/buffer",
         "max_length": 1000000,
         "full_threshold_percentage": 90,
         "reset_on_start": true
       }
     }
   }

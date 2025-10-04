use savant_core::metrics::{
    get_or_create_counter_family, get_or_create_gauge_family, SharedCounterFamily,
    SharedGaugeFamily,
};

// def __init__(self, aggregator: StatsAggregator):
//     super().__init__(
//         thread_name='StatsLogger',
//         logger_name=f'{LOGGER_NAME}.{self.__class__.__name__}',
//         daemon=True,
//     )
//     self._stats_aggregator = aggregator
//     self._metrics: Dict[str, Union[CounterFamily, GaugeFamily]] = dict()
//     self.counter(
//         'received_messages',
//         'Number of messages received by the adapter',
//     )
//     self.counter(
//         'pushed_messages',
//         'Number of messages pushed by the adapter',
//     )
//     self.counter(
//         'dropped_messages',
//         'Number of messages dropped by the adapter',
//     )
//     self.counter(
//         'sent_messages',
//         'Number of messages sent to the sink ZeroMQ socket',
//     )
//     self.gauge(
//         'buffer_size',
//         'Number of messages in the buffer',
//     )
//     self.gauge(
//         'payload_size',
//         'Size of messages in the buffer',
//     )
//     self.gauge(
//         'last_received_message',
//         'Number of messages received from the adapter',
//     )
//     self.gauge(
//         'last_pushed_message',
//         'Number of messages pushed from the adapter',
//     )
//     self.gauge(
//         'last_dropped_message',
//         'Number of messages dropped by the adapter',
//     )
//     self.gauge(
//         'last_sent_message',
//         'Number of messages sent to the sink ZeroMQ socket',
//     )

pub struct IngressMetrics {
    pub ingress_python_none_messages: SharedCounterFamily,
    pub received_messages: SharedCounterFamily,
    pub last_received_message: SharedGaugeFamily,
    pub pushed_messages: SharedCounterFamily,
    pub last_pushed_message: SharedGaugeFamily,
    pub dropped_messages: SharedCounterFamily,
    pub last_dropped_message: SharedGaugeFamily,
    pub buffer_size: SharedGaugeFamily,
    pub payload_size: SharedGaugeFamily,
}

impl IngressMetrics {
    pub fn new() -> Self {
        Self {
            ingress_python_none_messages: get_or_create_counter_family(
                "ingress_python_none_messages",
                Some("Number of messages dropped by the adapter"),
                &[],
                None,
            ),
            received_messages: get_or_create_counter_family(
                "received_messages",
                Some("Number of messages received by the adapter"),
                &[],
                None,
            ),
            last_received_message: get_or_create_gauge_family(
                "last_received_message",
                Some("Number of messages received by the adapter"),
                &[],
                None,
            ),
            pushed_messages: get_or_create_counter_family(
                "pushed_messages",
                Some("Number of messages pushed by the adapter"),
                &[],
                None,
            ),
            last_pushed_message: get_or_create_gauge_family(
                "last_pushed_message",
                Some("Number of messages pushed by the adapter"),
                &[],
                None,
            ),
            dropped_messages: get_or_create_counter_family(
                "dropped_messages",
                Some("Number of messages dropped by the adapter"),
                &[],
                None,
            ),
            last_dropped_message: get_or_create_gauge_family(
                "last_dropped_message",
                Some("Number of messages dropped by the adapter"),
                &[],
                None,
            ),
            buffer_size: get_or_create_gauge_family(
                "buffer_size",
                Some("Number of messages in the buffer"),
                &[],
                None,
            ),
            payload_size: get_or_create_gauge_family(
                "payload_size",
                Some("Size of messages in the buffer"),
                &[],
                None,
            ),
        }
    }
}

pub struct EgressMetrics {
    pub egress_python_none_messages: SharedCounterFamily,
    pub popped_messages: SharedCounterFamily,
    pub last_popped_message: SharedGaugeFamily,
    pub sent_messages: SharedCounterFamily,
    pub last_sent_message: SharedGaugeFamily,
    pub undelivered_messages: SharedCounterFamily,
    pub last_undelivered_message: SharedGaugeFamily,
}

impl EgressMetrics {
    pub fn new() -> Self {
        Self {
            egress_python_none_messages: get_or_create_counter_family(
                "egress_python_none_messages",
                Some("Number of messages dropped by the adapter"),
                &[],
                None,
            ),
            popped_messages: get_or_create_counter_family(
                "popped_messages",
                Some("Number of messages popped from the buffer"),
                &[],
                None,
            ),
            last_popped_message: get_or_create_gauge_family(
                "last_popped_message",
                Some("Number of messages popped from the buffer"),
                &[],
                None,
            ),
            sent_messages: get_or_create_counter_family(
                "sent_messages",
                Some("Number of messages sent to the sink ZeroMQ socket"),
                &["reason"],
                None,
            ),
            last_sent_message: get_or_create_gauge_family(
                "last_sent_message",
                Some("Number of messages sent to the sink ZeroMQ socket"),
                &["reason"],
                None,
            ),
            undelivered_messages: get_or_create_counter_family(
                "undelivered_messages",
                Some("Number of messages undelivered to the sink ZeroMQ socket"),
                &["reason"],
                None,
            ),
            last_undelivered_message: get_or_create_gauge_family(
                "last_undelivered_message",
                Some("Number of messages undelivered to the sink ZeroMQ socket"),
                &["reason"],
                None,
            ),
        }
    }
}

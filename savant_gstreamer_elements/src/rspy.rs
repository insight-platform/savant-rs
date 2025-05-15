use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer::{glib, Buffer, Event, StateChange};
use gstreamer_audio::glib::ParamFlags;
use parking_lot::{Mutex, RwLock};
use pyo3::prelude::*;
use savant_core_py::gst::{FlowResult, InvocationReason};
use savant_core_py::REGISTERED_HANDLERS;
use std::sync::{Arc, LazyLock};

static CAT: LazyLock<gstreamer::DebugCategory> = LazyLock::new(|| {
    gstreamer::DebugCategory::new(
        "rspy",
        gstreamer::DebugColorFlags::empty(),
        Some("PyFunc Element"),
    )
});

#[derive(Debug, Clone, Default)]
pub struct RsPySettings {
    pub pipeline_name: Option<String>,
    pub pipeline_stage: Option<String>,
}

#[derive(Default)]
pub struct InteropParameters {
    buffer: Option<Buffer>,
    source_event: Option<Event>,
    sink_event: Option<Event>,
}

// Struct containing all the element data
pub struct RsPy {
    src_pad: gstreamer::Pad,
    sink_pad: gstreamer::Pad,
    settings: RwLock<RsPySettings>,
    interop: Arc<Mutex<InteropParameters>>,
}

impl RsPy {
    // Called whenever a new buffer is passed to our sink pad. Here buffers should be processed and
    // whenever some output buffer is available have to push it out of the source pad.
    // Here we just pass through all buffers directly
    //
    // See the documentation of gstreamer::Buffer and gstreamer::BufferRef to see what can be done with
    // buffers.
    fn sink_chain(
        &self,
        pad: &gstreamer::Pad,
        buffer: Buffer,
    ) -> Result<gstreamer::FlowSuccess, gstreamer::FlowError> {
        gstreamer::log!(CAT, obj = pad, "Handling buffer {:?}", buffer);
        {
            let mut bind = self.interop.lock();
            bind.buffer = Some(buffer.clone());
        }

        Python::with_gil(|py| {
            let element_name = self.obj().name().to_string();
            let handlers_bind = REGISTERED_HANDLERS.read();
            let handler = handlers_bind
                .get(&element_name)
                .unwrap_or_else(|| panic!("Handler {} not found", element_name));

            let res = handler.call1(py, (element_name, InvocationReason::Buffer)); // , shared_buf.clone()
            match res {
                Err(e) => {
                    log::error!("Error calling buffer processing function: {:?}", e);
                    Err(gstreamer::FlowError::Error)
                }
                Ok(r) => {
                    let flow_res = r.extract::<FlowResult>(py);
                    match flow_res {
                        Ok(res) => match res {
                            FlowResult::CustomSuccess2 => {
                                Ok(gstreamer::FlowSuccess::CustomSuccess2)
                            }
                            FlowResult::CustomSuccess1 => {
                                Ok(gstreamer::FlowSuccess::CustomSuccess1)
                            }
                            FlowResult::CustomSuccess => Ok(gstreamer::FlowSuccess::CustomSuccess),
                            FlowResult::Ok => Ok(gstreamer::FlowSuccess::Ok),
                            FlowResult::NotLinked => Err(gstreamer::FlowError::NotLinked),
                            FlowResult::Flushing => Err(gstreamer::FlowError::Flushing),
                            FlowResult::Eos => Err(gstreamer::FlowError::Eos),
                            FlowResult::NotNegotiated => Err(gstreamer::FlowError::NotNegotiated),
                            FlowResult::Error => Err(gstreamer::FlowError::Error),
                            FlowResult::NotSupported => Err(gstreamer::FlowError::NotSupported),
                            FlowResult::CustomError => Err(gstreamer::FlowError::CustomError),
                            FlowResult::CustomError1 => Err(gstreamer::FlowError::CustomError1),
                            FlowResult::CustomError2 => Err(gstreamer::FlowError::CustomError2),
                        },
                        Err(e) => {
                            log::error!("Error extracting FlowResult: {:?}", e);
                            Err(gstreamer::FlowError::Error)
                        }
                    }
                }
            }
        })?;
        self.src_pad.push(buffer)
    }

    fn invoke_for_event(&self, reason: InvocationReason) -> bool {
        Python::with_gil(|py| {
            let element_name = self.obj().name().to_string();
            let handlers_bind = REGISTERED_HANDLERS.read();
            let handler = handlers_bind
                .get(&element_name)
                .unwrap_or_else(|| panic!("Handler {} not found", element_name));
            let res = handler.call1(py, (element_name, reason));
            match res {
                Ok(obj) => {
                    let bool_res = obj.extract::<bool>(py);
                    bool_res.unwrap_or_else(|e| {
                        log::error!("Error extracting bool result: {:?}", e);
                        false
                    })
                }
                Err(e) => {
                    log::error!("Error calling event processing function: {:?}", e);
                    false
                }
            }
        })
    }

    // Called whenever an event arrives at the sink pad. It has to be handled accordingly and in
    // most cases has to be either passed to Pad::event_default() on this pad for default handling,
    // or Pad::push_event() on all pads with the opposite direction for direct forwarding.
    // Here we just pass through all events directly to the source pad.
    //
    // See the documentation of gstreamer::Event and gstreamer::EventRef to see what can be done with
    // events, and especially the gstreamer::EventView type for inspecting events.
    fn sink_event(&self, pad: &gstreamer::Pad, event: Event) -> bool {
        gstreamer::log!(CAT, obj = pad, "Handling event {:?}", event);
        {
            let mut bind = self.interop.lock();
            bind.sink_event = Some(event.clone());
        }
        self.invoke_for_event(InvocationReason::SinkEvent) && self.src_pad.push_event(event)
    }

    // Called whenever an event arrives to the source pad. It has to be handled accordingly and in
    // most cases has to be either passed to Pad::event_default() on the same pad for default
    // handling, or Pad::push_event() on all pads with the opposite direction for direct
    // forwarding.
    // Here we just pass through all events directly to the sink pad.
    //
    // See the documentation of gstreamer::Event and gstreamer::EventRef to see what can be done with
    // events, and especially the gstreamer::EventView type for inspecting events.
    fn src_event(&self, pad: &gstreamer::Pad, event: Event) -> bool {
        gstreamer::log!(CAT, obj = pad, "Handling event {:?}", event);
        {
            let mut bind = self.interop.lock();
            bind.source_event = Some(event.clone());
        }
        self.invoke_for_event(InvocationReason::SourceEvent) && self.sink_pad.push_event(event)
    }

    // Called whenever a query is sent to the sink pad. It has to be answered if the element can
    // handle it, potentially by forwarding the query first to the peer pads of the pads with the
    // opposite direction, or false has to be returned. Default handling can be achieved with
    // Pad::query_default() on this pad and forwarding with Pad::peer_query() on the pads with the
    // opposite direction.
    // Here we just forward all queries directly to the source pad's peers.
    //
    // See the documentation of gstreamer::Query and gstreamer::QueryRef to see what can be done with
    // queries, and especially the gstreamer::QueryView type for inspecting and modifying queries.
    fn sink_query(&self, pad: &gstreamer::Pad, query: &mut gstreamer::QueryRef) -> bool {
        gstreamer::log!(CAT, obj = pad, "Handling query {:?}", query);
        self.src_pad.peer_query(query)
    }

    // Called whenever a query is sent to the source pad. It has to be answered if the element can
    // handle it, potentially by forwarding the query first to the peer pads of the pads with the
    // opposite direction, or false has to be returned. Default handling can be achieved with
    // Pad::query_default() on this pad and forwarding with Pad::peer_query() on the pads with the
    // opposite direction.
    // Here we just forward all queries directly to the sink pad's peers.
    //
    // See the documentation of gstreamer::Query and gstreamer::QueryRef to see what can be done with
    // queries, and especially the gstreamer::QueryView type for inspecting and modifying queries.
    fn src_query(&self, pad: &gstreamer::Pad, query: &mut gstreamer::QueryRef) -> bool {
        gstreamer::log!(CAT, obj = pad, "Handling query {:?}", query);
        self.sink_pad.peer_query(query)
    }
}

// This trait registers our type with the GObject object system and
// provides the entry points for creating a new instance and setting
// up the class data
#[glib::object_subclass]
impl ObjectSubclass for RsPy {
    const NAME: &'static str = "GstRsPy";
    type Type = super::RsPy;
    type ParentType = gstreamer::Element;

    // Called when a new instance is to be created. We need to return an instance
    // of our struct here and also get the class struct passed in case it's needed
    fn with_class(klass: &Self::Class) -> Self {
        // Create our two pads from the templates that were registered with
        // the class and set all the functions on them.
        //
        // Each function is wrapped in catch_panic_pad_function(), which will
        // - Catch panics from the pad functions and instead of aborting the process
        //   it will simply convert them into an error message and poison the element
        //   instance
        // - Extract our RsPy struct from the object instance and pass it to us
        //
        // Details about what each function is good for is next to each function definition
        let templ = klass.pad_template("sink").unwrap();
        let sinkpad = gstreamer::Pad::builder_from_template(&templ)
            .chain_function(|pad, parent, buffer| {
                RsPy::catch_panic_pad_function(
                    parent,
                    || Err(gstreamer::FlowError::Error),
                    |rspy| rspy.sink_chain(pad, buffer),
                )
            })
            .event_function(|pad, parent, event| {
                RsPy::catch_panic_pad_function(parent, || false, |rspy| rspy.sink_event(pad, event))
            })
            .query_function(|pad, parent, query| {
                RsPy::catch_panic_pad_function(parent, || false, |rspy| rspy.sink_query(pad, query))
            })
            .build();

        let templ = klass.pad_template("src").unwrap();
        let srcpad = gstreamer::Pad::builder_from_template(&templ)
            .event_function(|pad, parent, event| {
                RsPy::catch_panic_pad_function(parent, || false, |rspy| rspy.src_event(pad, event))
            })
            .query_function(|pad, parent, query| {
                RsPy::catch_panic_pad_function(parent, || false, |rspy| rspy.src_query(pad, query))
            })
            .build();

        Self {
            src_pad: srcpad,
            sink_pad: sinkpad,
            settings: RwLock::new(RsPySettings::default()),
            interop: Arc::new(Mutex::new(InteropParameters::default())),
        }
    }
}

// Implementation of glib::Object virtual methods
impl ObjectImpl for RsPy {
    fn properties() -> &'static [glib::ParamSpec] {
        // Metadata for the properties
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            vec![
                glib::ParamSpecString::builder("savant-pipeline-name")
                    .nick("PipelineName")
                    .blurb("The pipeline to work with")
                    .flags(ParamFlags::WRITABLE)
                    .build(),
                glib::ParamSpecString::builder("savant-pipeline-stage")
                    .nick("PipelineStage")
                    .blurb("The stage to work with")
                    .flags(ParamFlags::WRITABLE)
                    .build(),
                glib::ParamSpecBoxed::builder::<Buffer>("current-buffer")
                    .nick("Buffer")
                    .blurb("The buffer to process")
                    .flags(ParamFlags::READABLE)
                    .build(),
                glib::ParamSpecBoxed::builder::<Event>("sink-event")
                    .nick("SinkEvent")
                    .blurb("The sink event to process")
                    .flags(ParamFlags::READABLE)
                    .build(),
                glib::ParamSpecBoxed::builder::<Event>("source-event")
                    .nick("SourceEvent")
                    .blurb("The source event to process")
                    .flags(ParamFlags::READABLE)
                    .build(),
            ]
        });

        PROPERTIES.as_ref()
    }

    // Called whenever a value of a property is changed. It can be called
    // at any time from any thread.
    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut settings = self.settings.write();
        match pspec.name() {
            "savant-pipeline-name" => {
                let pipeline_name = value.get().expect("type checked upstream");
                gstreamer::info!(
                    CAT,
                    imp = self,
                    "Changing pipeline name to {}",
                    pipeline_name
                );
                settings.pipeline_name = Some(pipeline_name);
            }
            "savant-pipeline-stage" => {
                let pipeline_stage = value.get().expect("type checked upstream");
                gstreamer::info!(
                    CAT,
                    imp = self,
                    "Changing pipeline stage to {}",
                    pipeline_stage
                );
                settings.pipeline_stage = Some(pipeline_stage);
            }
            _ => panic!(
                "Set operation for property {} is not supported.",
                pspec.name()
            ),
        }
    }

    // Called whenever a value of a property is read. It can be called
    // at any time from any thread.
    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        match pspec.name() {
            "current-buffer" => {
                let buf = self.interop.lock().buffer.take();
                buf.to_value()
            }
            "sink-event" => {
                let sink_event = self.interop.lock().sink_event.take();
                sink_event.to_value()
            }
            "source-event" => {
                let source_event = self.interop.lock().source_event.take();
                source_event.to_value()
            }
            _ => panic!(
                "Get operation for property {} is not supported.",
                pspec.name()
            ),
        }
    }

    // Called right after construction of a new instance
    fn constructed(&self) {
        // Call the parent class' ::constructed() implementation first
        self.parent_constructed();

        // Here we actually add the pads we created in RsPy::new() to the
        // element so that GStreamer is aware of their existence.
        let obj = self.obj();
        obj.add_pad(&self.sink_pad).unwrap();
        obj.add_pad(&self.src_pad).unwrap();
    }
}

impl GstObjectImpl for RsPy {}

// Implementation of gstreamer::Element virtual methods
impl ElementImpl for RsPy {
    // Set the element specific metadata. This information is what
    // is visible from gst-inspect-1.0 and can also be programmatically
    // retrieved from the gstreamer::Registry after initial registration
    // without having to load the plugin in memory.
    fn metadata() -> Option<&'static gstreamer::subclass::ElementMetadata> {
        static ELEMENT_METADATA: LazyLock<gstreamer::subclass::ElementMetadata> = LazyLock::new(
            || {
                gstreamer::subclass::ElementMetadata::new(
                "RsPy",
                "Custom PyFunc Invocation Module",
                "A module invoking a custom Python function for buffers and events",
                "Ivan Kudriavtsev <ivan.a.kudryavtsev@gmail.com>, based on work of Sebastian Dröge <sebastian@centricular.com>",
            )
            },
        );

        Some(&*ELEMENT_METADATA)
    }

    // Create and add pad templates for our sink and source pad. These
    // are later used for actually creating the pads and beforehand
    // already provide information to GStreamer about all possible
    // pads that could exist for this type.
    //
    // Actual instances can create pads based on those pad templates
    fn pad_templates() -> &'static [gstreamer::PadTemplate] {
        static PAD_TEMPLATES: LazyLock<Vec<gstreamer::PadTemplate>> = LazyLock::new(|| {
            // Our element can accept any possible caps on both pads
            let caps = gstreamer::Caps::new_any();
            let src_pad_template = gstreamer::PadTemplate::new(
                "src",
                gstreamer::PadDirection::Src,
                gstreamer::PadPresence::Always,
                &caps,
            )
            .unwrap();

            let sink_pad_template = gstreamer::PadTemplate::new(
                "sink",
                gstreamer::PadDirection::Sink,
                gstreamer::PadPresence::Always,
                &caps,
            )
            .unwrap();

            vec![src_pad_template, sink_pad_template]
        });

        PAD_TEMPLATES.as_ref()
    }

    // Called whenever the state of the element should be changed. This allows for
    // starting up the element, allocating/deallocating resources or shutting down
    // the element again.
    fn change_state(
        &self,
        transition: StateChange,
    ) -> Result<gstreamer::StateChangeSuccess, gstreamer::StateChangeError> {
        let res = Python::with_gil(|py| {
            let element_name = self.obj().name().to_string();
            let handlers_bind = REGISTERED_HANDLERS.read();
            let handler = handlers_bind
                .get(&element_name)
                .unwrap_or_else(|| panic!("Handler {} not found", element_name));

            let current_state = transition.current() as i32;
            let next_state = transition.next() as i32;

            handler.call1(
                py,
                (
                    element_name,
                    InvocationReason::StateChange,
                    current_state,
                    next_state,
                ),
            )
        });

        if let Err(e) = res {
            log::error!("Error calling state change function: {:?}", e);
            return Err(gstreamer::StateChangeError);
        }

        let bool_res = Python::with_gil(|py| {
            res.unwrap().extract::<bool>(py).unwrap_or_else(|e| {
                log::error!("Error extracting bool result: {:?}", e);
                false
            })
        });

        if bool_res {
            self.parent_change_state(transition)
        } else {
            Err(gstreamer::StateChangeError)
        }
    }
}

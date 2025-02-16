// Copyright (C) 2018 Sebastian Dröge <sebastian@centricular.com>
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::py_handler::PyHandler;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst::{glib, Buffer};
use parking_lot::RwLock;
use pyo3::prelude::*;
use savant_core_py::gst::FlowResult;
use std::sync::{LazyLock, OnceLock};
use std::time::Instant;
// This module contains the private implementation details of our element

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "rspy",
        gst::DebugColorFlags::empty(),
        Some("PyFunc Element"),
    )
});

#[derive(Debug, Clone, Default)]
pub struct RsPySettings {
    module: Option<String>,
    class: Option<String>,
    parameters: Option<String>,
}

// Struct containing all the element data
pub struct RsPy {
    src_pad: gst::Pad,
    sink_pad: gst::Pad,
    instance: OnceLock<PyHandler>,
    settings: RwLock<RsPySettings>,
}

impl RsPy {
    fn load_handler(&self) -> PyHandler {
        Python::with_gil(|py| {
            let settings = self.settings.read();
            let module_name = settings.module.as_ref().expect("Module name must be set");
            let class_name = settings.class.as_ref().expect("Class name must be set");
            let default_parameters = "{}".to_string();
            let parameters = settings.parameters.as_ref().unwrap_or(&default_parameters);

            PyHandler::new(py, module_name, class_name, parameters)
                .expect("Error initializing Python handler")
        })
    }

    // Called whenever a new buffer is passed to our sink pad. Here buffers should be processed and
    // whenever some output buffer is available have to push it out of the source pad.
    // Here we just pass through all buffers directly
    //
    // See the documentation of gst::Buffer and gst::BufferRef to see what can be done with
    // buffers.
    fn sink_chain(
        &self,
        pad: &gst::Pad,
        buffer: Buffer,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        gst::log!(CAT, obj = pad, "Handling buffer {:?}", buffer);
        let now = Instant::now();
        let shared_buf = savant_core_py::gst::GstBuffer::from_gst_buffer(buffer);

        Python::with_gil(|py| {
            let handler = self.instance.get_or_init(|| self.load_handler());
            let element_name = self.obj().name().to_string();
            let res = handler.call(py, (element_name, shared_buf.clone()));
            match res {
                Err(e) => {
                    log::error!("Error calling buffer processing function: {:?}", e);
                    Err(gst::FlowError::Error)
                }
                Ok(r) => {
                    let flow_res = r.extract::<FlowResult>(py);
                    match flow_res {
                        Ok(res) => match res {
                            FlowResult::CustomSuccess2 => Ok(gst::FlowSuccess::CustomSuccess2),
                            FlowResult::CustomSuccess1 => Ok(gst::FlowSuccess::CustomSuccess1),
                            FlowResult::CustomSuccess => Ok(gst::FlowSuccess::CustomSuccess),
                            FlowResult::Ok => Ok(gst::FlowSuccess::Ok),
                            FlowResult::NotLinked => Err(gst::FlowError::NotLinked),
                            FlowResult::Flushing => Err(gst::FlowError::Flushing),
                            FlowResult::Eos => Err(gst::FlowError::Eos),
                            FlowResult::NotNegotiated => Err(gst::FlowError::NotNegotiated),
                            FlowResult::Error => Err(gst::FlowError::Error),
                            FlowResult::NotSupported => Err(gst::FlowError::NotSupported),
                            FlowResult::CustomError => Err(gst::FlowError::CustomError),
                            FlowResult::CustomError1 => Err(gst::FlowError::CustomError1),
                            FlowResult::CustomError2 => Err(gst::FlowError::CustomError2),
                        },
                        Err(e) => {
                            log::error!("Error extracting FlowResult: {:?}", e);
                            Err(gst::FlowError::Error)
                        }
                    }
                }
            }
        })?;

        let elapsed = now.elapsed();
        log::debug!("Elapsed time: {:?}", elapsed);
        match shared_buf.extract() {
            Ok(buf) => self.src_pad.push(buf),
            Err(e) => {
                log::error!(
                    "Error acquiring Gstreamer Buffer object ownership back to Rust: {:?}",
                    e
                );
                Err(gst::FlowError::Error)
            }
        }
    }

    // Called whenever an event arrives at the sink pad. It has to be handled accordingly and in
    // most cases has to be either passed to Pad::event_default() on this pad for default handling,
    // or Pad::push_event() on all pads with the opposite direction for direct forwarding.
    // Here we just pass through all events directly to the source pad.
    //
    // See the documentation of gst::Event and gst::EventRef to see what can be done with
    // events, and especially the gst::EventView type for inspecting events.
    fn sink_event(&self, pad: &gst::Pad, event: gst::Event) -> bool {
        gst::log!(CAT, obj = pad, "Handling event {:?}", event);
        self.src_pad.push_event(event)
    }

    // Called whenever an event arrives to the source pad. It has to be handled accordingly and in
    // most cases has to be either passed to Pad::event_default() on the same pad for default
    // handling, or Pad::push_event() on all pads with the opposite direction for direct
    // forwarding.
    // Here we just pass through all events directly to the sink pad.
    //
    // See the documentation of gst::Event and gst::EventRef to see what can be done with
    // events, and especially the gst::EventView type for inspecting events.
    fn src_event(&self, pad: &gst::Pad, event: gst::Event) -> bool {
        gst::log!(CAT, obj = pad, "Handling event {:?}", event);
        self.sink_pad.push_event(event)
    }

    // Called whenever a query is sent to the sink pad. It has to be answered if the element can
    // handle it, potentially by forwarding the query first to the peer pads of the pads with the
    // opposite direction, or false has to be returned. Default handling can be achieved with
    // Pad::query_default() on this pad and forwarding with Pad::peer_query() on the pads with the
    // opposite direction.
    // Here we just forward all queries directly to the source pad's peers.
    //
    // See the documentation of gst::Query and gst::QueryRef to see what can be done with
    // queries, and especially the gst::QueryView type for inspecting and modifying queries.
    fn sink_query(&self, pad: &gst::Pad, query: &mut gst::QueryRef) -> bool {
        gst::log!(CAT, obj = pad, "Handling query {:?}", query);
        self.src_pad.peer_query(query)
    }

    // Called whenever a query is sent to the source pad. It has to be answered if the element can
    // handle it, potentially by forwarding the query first to the peer pads of the pads with the
    // opposite direction, or false has to be returned. Default handling can be achieved with
    // Pad::query_default() on this pad and forwarding with Pad::peer_query() on the pads with the
    // opposite direction.
    // Here we just forward all queries directly to the sink pad's peers.
    //
    // See the documentation of gst::Query and gst::QueryRef to see what can be done with
    // queries, and especially the gst::QueryView type for inspecting and modifying queries.
    fn src_query(&self, pad: &gst::Pad, query: &mut gst::QueryRef) -> bool {
        gst::log!(CAT, obj = pad, "Handling query {:?}", query);
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
    type ParentType = gst::Element;

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
        let sinkpad = gst::Pad::builder_from_template(&templ)
            .chain_function(|pad, parent, buffer| {
                RsPy::catch_panic_pad_function(
                    parent,
                    || Err(gst::FlowError::Error),
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
        let srcpad = gst::Pad::builder_from_template(&templ)
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
            instance: OnceLock::new(),
            settings: RwLock::new(RsPySettings::default()),
        }
    }
}

// Implementation of glib::Object virtual methods
impl ObjectImpl for RsPy {
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

    fn properties() -> &'static [glib::ParamSpec] {
        // Metadata for the properties
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            vec![
                glib::ParamSpecString::builder("module")
                    .nick("PythonModule")
                    .blurb("Python module to load")
                    .build(),
                glib::ParamSpecString::builder("class")
                    .nick("Class")
                    .blurb("The Class representing the object")
                    .build(),
                glib::ParamSpecString::builder("parameters")
                    .nick("Parameters")
                    .blurb("The **kwargs for the class constructor")
                    .default_value("{}")
                    .build(),
            ]
        });

        PROPERTIES.as_ref()
    }

    // Called whenever a value of a property is changed. It can be called
    // at any time from any thread.
    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        match pspec.name() {
            "module" => {
                let mut settings = self.settings.write();
                let module = value.get().expect("type checked upstream");
                gst::info!(
                    CAT,
                    imp = self,
                    "Changing module name from {:?} to {}",
                    settings.module,
                    module
                );
                settings.module = Some(module);
            }
            "class" => {
                let mut settings = self.settings.write();
                let class = value.get().expect("type checked upstream");
                gst::info!(
                    CAT,
                    imp = self,
                    "Changing init function from {:?} to {}",
                    settings.class,
                    class
                );
                settings.class = Some(class);
            }
            "parameters" => {
                let mut settings = self.settings.write();
                let parameters = value.get().expect("type checked upstream");
                gst::info!(
                    CAT,
                    imp = self,
                    "Changing parameters from {:?} to {}",
                    settings.parameters,
                    parameters
                );
                settings.parameters = Some(parameters);
            }
            _ => unimplemented!(),
        }
    }

    // Called whenever a value of a property is read. It can be called
    // at any time from any thread.
    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        match pspec.name() {
            "module" => {
                let settings = self.settings.read();
                settings.module.to_value()
            }
            "class" => {
                let settings = self.settings.read();
                settings.class.to_value()
            }
            "parameters" => {
                let settings = self.settings.read();
                settings.parameters.to_value()
            }
            _ => unimplemented!(),
        }
    }
}

impl GstObjectImpl for RsPy {}

// Implementation of gst::Element virtual methods
impl ElementImpl for RsPy {
    // Set the element specific metadata. This information is what
    // is visible from gst-inspect-1.0 and can also be programmatically
    // retrieved from the gst::Registry after initial registration
    // without having to load the plugin in memory.
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "RsPy",
                "Module invoking custom Python functions for buffers and events",
                "Does nothing with the data",
                "Ivan Kudriavtsev <ivan.a.kudryavtsev@gmail.com>, based on work of Sebastian Dröge <sebastian@centricular.com>",
            )
        });

        Some(&*ELEMENT_METADATA)
    }

    // Create and add pad templates for our sink and source pad. These
    // are later used for actually creating the pads and beforehand
    // already provide information to GStreamer about all possible
    // pads that could exist for this type.
    //
    // Actual instances can create pads based on those pad templates
    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: LazyLock<Vec<gst::PadTemplate>> = LazyLock::new(|| {
            // Our element can accept any possible caps on both pads
            let caps = gst::Caps::new_any();
            let src_pad_template = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &caps,
            )
            .unwrap();

            let sink_pad_template = gst::PadTemplate::new(
                "sink",
                gst::PadDirection::Sink,
                gst::PadPresence::Always,
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
        transition: gst::StateChange,
    ) -> Result<gst::StateChangeSuccess, gst::StateChangeError> {
        gst::trace!(CAT, imp = self, "Changing state {:?}", transition);

        // Call the parent class' implementation of ::change_state()
        self.parent_change_state(transition)
    }
}

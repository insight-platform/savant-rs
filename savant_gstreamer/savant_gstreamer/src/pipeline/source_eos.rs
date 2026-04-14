use gstreamer as gst;

const SOURCE_EOS_EVENT_NAME: &str = "savant.pipeline.source_eos";
const SOURCE_ID_FIELD: &str = "source_id";

/// Build a per-source logical EOS event (custom downstream).
pub fn build_source_eos_event(source_id: &str) -> gst::Event {
    let structure = gst::Structure::builder(SOURCE_EOS_EVENT_NAME)
        .field(SOURCE_ID_FIELD, source_id)
        .build();
    gst::event::CustomDownstream::new(structure)
}

/// Parse a per-source logical EOS event and return `source_id` when matched.
pub fn parse_source_eos_event(event: &gst::Event) -> Option<String> {
    if event.type_() != gst::EventType::CustomDownstream {
        return None;
    }
    let structure = event.structure()?;
    if structure.name() != SOURCE_EOS_EVENT_NAME {
        return None;
    }
    structure.get::<String>(SOURCE_ID_FIELD).ok()
}

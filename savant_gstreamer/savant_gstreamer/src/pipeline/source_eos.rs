use gstreamer as gst;

const SOURCE_EOS_EVENT_NAME: &str = "savant.pipeline.source_eos";
const SOURCE_ID_FIELD: &str = "source_id";

/// Build a per-source logical EOS event (serialized custom downstream).
///
/// The event is **serialized with the buffer flow**, so it is ordered
/// relative to in-flight buffers. `GstVideoDecoder`-based elements
/// (nvv4l2decoder, avdec_*, vaapi*, ...) queue serialized custom
/// downstream events against the currently-parsed frame and release
/// them when that frame is pushed downstream. If no more frames follow
/// the event — which is the normal situation for source-EOS — the
/// event is discarded on the decoder's EOS drain.
///
/// The pipeline runner compensates for this decoder-generic behaviour
/// with a *rescue probe* pair installed around every processing
/// element: the sink-side probe records custom-downstream events as
/// they enter, the src-side probe removes them as they exit, and any
/// events still "in flight" when a real `GstEvent::Eos` reaches the
/// src-side probe are re-injected downstream before the EOS passes.
/// See [`crate::pipeline::runner`] for the implementation.
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

#[cfg(test)]
mod tests {
    use super::*;
    use gstreamer as gst;

    #[test]
    fn builds_serialized_custom_downstream_event() {
        gst::init().unwrap();
        let event = build_source_eos_event("cam-1");
        assert_eq!(event.type_(), gst::EventType::CustomDownstream);
        assert!(event.is_serialized());
    }

    #[test]
    fn parses_event() {
        gst::init().unwrap();
        let event = build_source_eos_event("cam-1");
        assert_eq!(parse_source_eos_event(&event), Some("cam-1".to_string()));
    }

    #[test]
    fn ignores_unrelated_events() {
        gst::init().unwrap();
        let structure = gst::Structure::builder("other.event").build();
        let event = gst::event::CustomDownstream::new(structure);
        assert_eq!(parse_source_eos_event(&event), None);
    }

    #[test]
    fn seqnum_is_stable_across_clones() {
        gst::init().unwrap();
        let event = build_source_eos_event("cam-1");
        let sn1 = event.seqnum();
        let sn2 = event.clone().seqnum();
        assert_eq!(sn1, sn2);
    }
}

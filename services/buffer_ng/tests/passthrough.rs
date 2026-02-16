mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use serial_test::serial;

use common::*;

/// Pure passthrough: no Python handler on either side.
/// Sends N frames + EOS through the buffer and verifies all arrive at the
/// destination unchanged.
///
/// Topology: source (req) -> (rep) buffer_ng (dealer) -> (router) dest
#[test]
#[serial]
fn passthrough_no_handler() -> Result<()> {
    init_python();

    let shutdown = Arc::new(AtomicBool::new(false));
    let num_frames = 5u32;

    let ingress_ipc = ipc_addr("pt_no_handler", "ingress");
    let egress_ipc = ipc_addr("pt_no_handler", "egress");
    let buffer_dir = tempfile::tempdir()?;
    let buffer_path_str = buffer_dir.path().to_str().unwrap().to_string();

    let conf = make_service_conf(&ingress_ipc, &egress_ipc, &buffer_path_str);

    let mut dest_reader = start_dest_reader(&egress_ipc)?;
    let service_thread = start_service(conf, None, None, shutdown.clone());
    let mut source_writer = start_source_writer(&ingress_ipc)?;

    let mut frames_at_dest = 0usize;
    let mut eos_at_dest = false;

    for step in frame_eos_steps(num_frames) {
        let deadline = Instant::now() + Duration::from_secs(5);
        step.send(&source_writer)?;

        if step.is_eos {
            let ok = wait_for_dest(&dest_reader, deadline, |m| m.is_end_of_stream());
            assert!(ok, "EOS was not delivered within deadline");
            eos_at_dest = true;
        } else {
            let ok = wait_for_dest(&dest_reader, deadline, |m| m.is_video_frame());
            assert!(ok, "frame was not delivered within deadline");
            frames_at_dest += 1;
        }
    }

    assert_eq!(frames_at_dest, num_frames as usize);
    assert!(eos_at_dest, "EOS should have been propagated to dest");

    shutdown.store(true, Ordering::SeqCst);
    service_thread.join().expect("service thread panicked")?;
    source_writer.shutdown()?;
    dest_reader.shutdown()?;

    Ok(())
}

/// AfterReceive handler: the stamping handler runs on every message
/// **before** it enters the buffer.  We verify that each video frame
/// arriving at the destination carries the `(ingress, stamped)` attribute.
///
/// Topology: source (req) -> (rep) [handler] buffer_ng (dealer) -> (router) dest
#[test]
#[serial]
fn passthrough_after_receive_handler() -> Result<()> {
    init_python();

    let shutdown = Arc::new(AtomicBool::new(false));
    let num_frames = 5u32;

    let ingress_ipc = ipc_addr("pt_after_recv", "ingress");
    let egress_ipc = ipc_addr("pt_after_recv", "egress");
    let buffer_dir = tempfile::tempdir()?;
    let buffer_path_str = buffer_dir.path().to_str().unwrap().to_string();

    let conf = make_service_conf(&ingress_ipc, &egress_ipc, &buffer_path_str);
    let ingress_handler = Some(make_stamping_handler("ingress", "stamped"));

    let mut dest_reader = start_dest_reader(&egress_ipc)?;
    let service_thread = start_service(conf, ingress_handler, None, shutdown.clone());
    let mut source_writer = start_source_writer(&ingress_ipc)?;

    let mut frames_at_dest = 0usize;
    let mut eos_at_dest = false;

    for step in frame_eos_steps(num_frames) {
        let deadline = Instant::now() + Duration::from_secs(5);
        step.send(&source_writer)?;

        if step.is_eos {
            let ok = wait_for_dest(&dest_reader, deadline, |m| m.is_end_of_stream());
            assert!(ok, "EOS was not delivered within deadline");
            eos_at_dest = true;
        } else {
            let ok = wait_for_dest(&dest_reader, deadline, |m| {
                if !m.is_video_frame() {
                    return false;
                }
                let f = m.as_video_frame().unwrap();
                assert_has_stamp(&f, "ingress", "stamped");
                true
            });
            assert!(ok, "frame was not delivered within deadline");
            frames_at_dest += 1;
        }
    }

    assert_eq!(frames_at_dest, num_frames as usize);
    assert!(eos_at_dest, "EOS should have been propagated to dest");

    shutdown.store(true, Ordering::SeqCst);
    service_thread.join().expect("service thread panicked")?;
    source_writer.shutdown()?;
    dest_reader.shutdown()?;

    Ok(())
}

/// BeforeSend handler: the stamping handler runs on every message **after**
/// it is popped from the buffer, right before egress.  We verify that each
/// video frame arriving at the destination carries the `(egress, stamped)`
/// attribute.
///
/// Topology: source (req) -> (rep) buffer_ng [handler] (dealer) -> (router) dest
#[test]
#[serial]
fn passthrough_before_send_handler() -> Result<()> {
    init_python();

    let shutdown = Arc::new(AtomicBool::new(false));
    let num_frames = 5u32;

    let ingress_ipc = ipc_addr("pt_before_send", "ingress");
    let egress_ipc = ipc_addr("pt_before_send", "egress");
    let buffer_dir = tempfile::tempdir()?;
    let buffer_path_str = buffer_dir.path().to_str().unwrap().to_string();

    let conf = make_service_conf(&ingress_ipc, &egress_ipc, &buffer_path_str);
    let egress_handler = Some(make_stamping_handler("egress", "stamped"));

    let mut dest_reader = start_dest_reader(&egress_ipc)?;
    let service_thread = start_service(conf, None, egress_handler, shutdown.clone());
    let mut source_writer = start_source_writer(&ingress_ipc)?;

    let mut frames_at_dest = 0usize;
    let mut eos_at_dest = false;

    for step in frame_eos_steps(num_frames) {
        let deadline = Instant::now() + Duration::from_secs(5);
        step.send(&source_writer)?;

        if step.is_eos {
            let ok = wait_for_dest(&dest_reader, deadline, |m| m.is_end_of_stream());
            assert!(ok, "EOS was not delivered within deadline");
            eos_at_dest = true;
        } else {
            let ok = wait_for_dest(&dest_reader, deadline, |m| {
                if !m.is_video_frame() {
                    return false;
                }
                let f = m.as_video_frame().unwrap();
                assert_has_stamp(&f, "egress", "stamped");
                true
            });
            assert!(ok, "frame was not delivered within deadline");
            frames_at_dest += 1;
        }
    }

    assert_eq!(frames_at_dest, num_frames as usize);
    assert!(eos_at_dest, "EOS should have been propagated to dest");

    shutdown.store(true, Ordering::SeqCst);
    service_thread.join().expect("service thread panicked")?;
    source_writer.shutdown()?;
    dest_reader.shutdown()?;

    Ok(())
}

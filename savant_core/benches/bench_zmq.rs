#![feature(test)]

extern crate test;

use savant_core::message::Message;
use savant_core::test::gen_frame;
use savant_core::transport::zeromq::reader::ReaderResult;
use savant_core::transport::zeromq::{
    NoopResponder, Reader, ReaderConfig, Writer, WriterConfig, WriterResult, ZmqSocketProvider,
};
use std::thread;
use test::Bencher;

#[bench]
fn bench_zmq_dealer_router(b: &mut Bencher) -> anyhow::Result<()> {
    let path = "/tmp/test/dealer-router";
    std::fs::remove_dir_all(path).unwrap_or_default();

    let reader = Reader::<NoopResponder, ZmqSocketProvider>::new(
        &ReaderConfig::new()
            .url(&format!("router+bind:ipc://{}", path))?
            .with_fix_ipc_permissions(Some(0o777))?
            .build()?,
    )?;

    let mut writer = Writer::<NoopResponder, ZmqSocketProvider>::new(
        &WriterConfig::new()
            .url(&format!("dealer+connect:ipc://{}", path))?
            .build()?,
    )?;

    let reader_thread = thread::spawn(move || {
        let mut reader = reader;
        loop {
            let res = reader.receive();
            if res.is_err() {
                break;
            }
            if let Ok(res) = res {
                match res {
                    ReaderResult::EndOfStream { .. } => {
                        break;
                    }
                    ReaderResult::Message {
                        message: _,
                        topic,
                        routing_id,
                        data,
                    } if topic == b"test" && routing_id.is_some() && data.is_empty() => {}
                    _ => {
                        panic!("Unexpected result: {:?}", res);
                    }
                }
            }
        }
    });

    let m = Message::video_frame(&gen_frame());

    b.iter(|| {
        let res = writer.send_message(b"test", &m, &[])?;
        assert!(matches!(res, WriterResult::Success(_)));
        Ok::<(), anyhow::Error>(())
    });
    writer.send_eos(b"test")?;
    reader_thread.join().unwrap();
    Ok(())
}

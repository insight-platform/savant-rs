#![feature(test)]

extern crate test;

use savant_core::message::Message;
use savant_core::test::gen_frame;
use savant_core::transport::zeromq::reader::ReaderResult;
use savant_core::transport::zeromq::{
    NonBlockingReader, NonBlockingWriter, NoopResponder, Reader, ReaderConfig, Writer,
    WriterConfig, WriterResult, ZmqSocketProvider,
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
                    } if topic == b"test" && routing_id.is_some() && data.len() == 1 => {}
                    _ => {
                        panic!("Unexpected result: {:?}", res);
                    }
                }
            }
        }
    });

    let m = Message::video_frame(&gen_frame());
    b.iter(|| {
        for _ in 0..100 {
            let res = writer.send_message("test", &m, &[&[0; 128 * 1024]])?;
            assert!(matches!(res, WriterResult::Success { .. }));
        }
        Ok::<(), anyhow::Error>(())
    });

    writer.send_eos("test")?;
    reader_thread.join().unwrap();
    Ok(())
}

#[bench]
fn bench_nonblocking_zmq_dealer_router(b: &mut Bencher) -> anyhow::Result<()> {
    let path = "/tmp/test/dealer-router";
    std::fs::remove_dir_all(path).unwrap_or_default();

    let mut reader = NonBlockingReader::new(
        &ReaderConfig::new()
            .url(&format!("router+bind:ipc://{}", path))?
            .with_fix_ipc_permissions(Some(0o777))?
            .with_receive_timeout(1000)?
            .build()?,
        100,
    )?;
    reader.start()?;

    let mut writer = NonBlockingWriter::new(
        &WriterConfig::new()
            .url(&format!("dealer+connect:ipc://{}", path))?
            .build()?,
        100,
    )?;
    writer.start()?;

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
                    } if topic == b"test" && routing_id.is_some() && data.len() == 1 => {}
                    _ => {
                        panic!("Unexpected result: {:?}", res);
                    }
                }
            }
        }
        let _ = reader.shutdown();
    });

    let m = Message::video_frame(&gen_frame());

    b.iter(|| {
        let mut results = Vec::new();
        for _ in 0..100 {
            results.push(writer.send_message("test", &m, &[&[0; 128 * 1024]])?);
        }
        results.into_iter().for_each(|r| {
            assert!(matches!(r.get(), Ok(WriterResult::Success { .. })));
        });
        Ok::<(), anyhow::Error>(())
    });

    writer.send_eos("test")?;
    thread::sleep(std::time::Duration::from_millis(1000));
    writer.shutdown()?;
    reader_thread.join().unwrap();
    Ok(())
}

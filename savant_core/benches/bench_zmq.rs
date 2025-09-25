use criterion::{criterion_group, criterion_main, Criterion};
use savant_core::message::Message;
use savant_core::test::gen_frame;
use savant_core::transport::zeromq::reader::ReaderResult;
use savant_core::transport::zeromq::{
    NonBlockingReader, NonBlockingWriter, NoopResponder, Reader, ReaderConfig, Writer,
    WriterConfig, WriterResult, ZmqSocketProvider,
};
use std::hint::black_box;
use std::thread;

fn zmq_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("zmq");

    group.bench_function("dealer_router", |b| {
        let path = "/tmp/test/dealer-router";
        std::fs::remove_dir_all(path).unwrap_or_default();

        let reader = Reader::<NoopResponder, ZmqSocketProvider>::new(
            &ReaderConfig::new()
                .url(&format!("router+bind:ipc://{}", path))
                .expect("Failed to set URL")
                .with_fix_ipc_permissions(Some(0o777))
                .expect("Failed to set permissions")
                .build()
                .expect("Failed to build reader config"),
        )
        .expect("Failed to create reader");

        let mut writer = Writer::<NoopResponder, ZmqSocketProvider>::new(
            &WriterConfig::new()
                .url(&format!("dealer+connect:ipc://{}", path))
                .expect("Failed to set URL")
                .build()
                .expect("Failed to build writer config"),
        )
        .expect("Failed to create writer");

        let reader_thread = thread::spawn(move || loop {
            let res = reader.receive();
            if res.is_err() {
                break;
            }
            if let Ok(res) = res {
                match res {
                    ReaderResult::Message { message, .. } if message.is_end_of_stream() => {
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
        });

        let m = Message::video_frame(&gen_frame());
        b.iter(|| {
            for _ in 0..100 {
                let res = black_box(
                    writer
                        .send_message("test", &m, &[&[0; 128 * 1024]])
                        .expect("Failed to send message"),
                );
                assert!(matches!(res, WriterResult::Success { .. }));
            }
        });

        writer.send_eos("test").expect("Failed to send EOS");
        reader_thread.join().unwrap();
    });

    group.bench_function("nonblocking_dealer_router", |b| {
        let path = "/tmp/test/dealer-router-nonblocking";
        std::fs::remove_dir_all(path).unwrap_or_default();

        let mut reader = NonBlockingReader::new(
            &ReaderConfig::new()
                .url(&format!("router+bind:ipc://{}", path))
                .expect("Failed to set URL")
                .with_fix_ipc_permissions(Some(0o777))
                .expect("Failed to set permissions")
                .with_receive_timeout(1000)
                .expect("Failed to set timeout")
                .build()
                .expect("Failed to build reader config"),
            100,
        )
        .expect("Failed to create reader");
        reader.start().expect("Failed to start reader");

        let mut writer = NonBlockingWriter::new(
            &WriterConfig::new()
                .url(&format!("dealer+connect:ipc://{}", path))
                .expect("Failed to set URL")
                .build()
                .expect("Failed to build writer config"),
            100,
        )
        .expect("Failed to create writer");
        writer.start().expect("Failed to start writer");

        let reader_thread = thread::spawn(move || {
            let mut reader = reader;
            loop {
                let res = reader.receive();
                if res.is_err() {
                    break;
                }
                if let Ok(res) = res {
                    match res {
                        ReaderResult::Message { message, .. } if message.is_end_of_stream() => {
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
                results.push(
                    writer
                        .send_message("test", &m, &[&[0; 128 * 1024]])
                        .expect("Failed to send message"),
                );
            }
            results.into_iter().for_each(|r| {
                assert!(matches!(r.get(), Ok(WriterResult::Success { .. })));
            });
        });

        writer.send_eos("test").expect("Failed to send EOS");
        thread::sleep(std::time::Duration::from_millis(1000));
        writer.shutdown().expect("Failed to shutdown writer");
        reader_thread.join().unwrap();
    });

    group.finish();
}

criterion_group!(benches, zmq_benchmarks);
criterion_main!(benches);

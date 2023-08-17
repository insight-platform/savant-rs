pub fn enable_dl_detection() {
    // only for #[cfg]
    use parking_lot::deadlock;
    use std::thread;
    use std::time::Duration;

    // Create a background thread which checks for deadlocks every 10s
    thread::spawn(move || loop {
        thread::sleep(Duration::from_secs(5));
        log::trace!(target: "parking_lot::deadlock_detector", "Checking for deadlocks");
        let deadlocks = deadlock::check_deadlock();
        if deadlocks.is_empty() {
            continue;
        }

        log::error!(target: "parking_lot::deadlock_detector", "{} deadlocks detected", deadlocks.len());

        for (i, threads) in deadlocks.iter().enumerate() {
            log::error!(target: "parking_lot::deadlock_detector", "Deadlock #{}", i);
            for t in threads {
                log::error!(target: "parking_lot::deadlock_detector", "Thread Id {:#?}", t.thread_id());
                log::error!(target: "parking_lot::deadlock_detector", "{:#?}", t.backtrace());
            }
        }
    });
}

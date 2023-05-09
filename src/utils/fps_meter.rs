// use pyo3::prelude::*;
//
// #[pyclass]
// #[derive(Debug)]
// pub struct FpsMeter {
//     period_frames: Option<u64>,
//     period_seconds: Option<f64>,
//     frame_counter: u64,
//     last_frame_counter: u64,
//     start_time: u64,
//     last_start_time: u64,
// }
//
// #[pymethods]
// impl FpsMeter {
//     #[new]
//     pub fn new(period_frames: Option<i64>, period_seconds: Option<f64>) -> Self {
//         match (period_frames, period_seconds) {
//             (Some(_), Some(_)) => {
//                 panic!("Cannot specify both period_frames and period_seconds")
//             }
//             (None, None) => panic!("Must specify either period_frames or period_seconds"),
//             (Some(period_frames), None) => {
//                 if period_frames <= 0 {
//                     panic!("period_frames must be positive")
//                 }
//             }
//             (None, Some(period_seconds)) => {
//                 if period_seconds <= 0.0 {
//                     panic!("period_seconds must be positive")
//                 }
//             }
//         }
//         Self {
//             period_frames: u64::try_from(period_frames.unwrap_or(-1)).ok(),
//             period_seconds,
//             frame_counter: 0,
//             last_frame_counter: 0,
//             start_time: 0,
//             last_start_time: 0,
//         }
//     }
//
//     #[getter(period_frames)]
//     pub fn get_period_frames(&self) -> Option<u64> {
//         self.period_frames
//     }
//
//     #[setter(period_frames)]
//     pub fn set_period_frames(&mut self, period_frames: i64) {
//         assert!(matches!(self.period_seconds, None));
//         self.period_frames = u64::try_from(period_frames).ok();
//     }
//
//     #[getter(period_seconds)]
//     pub fn get_period_seconds(&self) -> Option<f64> {
//         self.period_seconds
//     }
//
//     #[setter(period_seconds)]
//     pub fn set_period_seconds(&mut self, period_seconds: f64) {
//         assert!(matches!(self.period_frames, None));
//         self.period_seconds = Some(period_seconds);
//     }
//
//     #[getter(frame_counter)]
//     pub fn get_frame_counter(&self) -> u64 {
//         self.last_frame_counter
//     }
//
//     #[getter(exec_seconds)]
//     pub fn get_exec_seconds(&self) -> u64 {
//         let now = std::time::SystemTime::now()
//             .duration_since(std::time::UNIX_EPOCH)
//             .unwrap()
//             .as_secs();
//         now - self.last_start_time
//     }
//
//     #[getter(fps)]
//     pub fn get_fps(&self) -> f64 {
//         self.frame_counter as f64 / self.exec_seconds() as f64
//     }
//
//     pub fn start(&mut self) {
//         self.last_start_time = 0;
//         self.start_time = std::time::SystemTime::now()
//             .duration_since(std::time::UNIX_EPOCH)
//             .unwrap()
//             .as_secs();
//
//         self.last_frame_counter = 0;
//         self.frame_counter = 0;
//     }
//
//     #[getter(message)]
//     pub fn get_message(&self) -> String {
//         format!(
//             "Processed {} {}, {:.2}",
//             self.frame_counter,
//             if self.frame_counter == 1 {
//                 "frame".into()
//             } else {
//                 "frames".into()
//             },
//             self.get_fps()
//         )
//     }
//
//     pub fn reset_counter(&mut self) {
//         self.frame_counter = 0;
//         self.start_time = std::time::SystemTime::now()
//             .duration_since(std::time::UNIX_EPOCH)
//             .unwrap()
//             .as_secs();
//     }
//
//     // pub fn is_period_passed(&self) -> bool {
//     //     if let Some(period_frames) = self.period_frames {
//     //         self.frame_counter >= period_frames
//     //     } else {
//     //         let now = std::time::SystemTime::now()
//     //             .duration_since(std::time::UNIX_EPOCH)
//     //             .unwrap()
//     //             .as_secs();
//     //         now - self.start_time >= self.period_seconds.unwrap()
//     //     }
//     // }
// }

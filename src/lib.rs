#![warn(missing_docs)]
#![allow(rustdoc::bare_urls)]
#![doc = include_str!("../README.md")]
//! # Simple video player example
//! ```
#![doc = include_str!("../examples/main.rs")]
//! ```

#[cfg(feature = "audio")]
mod audio;

#[cfg(feature = "audio")]
pub use audio::*;

#[cfg(feature = "hls")]
mod hls;
#[cfg(feature = "default-overlay")]
mod overlay;
#[cfg(feature = "default-overlay")]
pub use overlay::*;
mod player;
pub use player::*;
mod stream;
#[cfg(feature = "subtitles")]
mod subtitle;

/// Simple audio device handle
pub trait AudioDevice: Send {
    // no impl needed yet
}

/// A fallback device which just drains the audio stream
pub(crate) struct NoAudioDevice {
    #[allow(unused)]
    handle: std::thread::JoinHandle<()>,
}

impl NoAudioDevice {
    pub fn new(rx: std::sync::mpsc::Receiver<stream::AudioSamples>) -> Self {
        let h = std::thread::Builder::new()
            .name("empty-audio-device".to_owned())
            .spawn(move || {
                loop {
                    match rx.recv() {
                        Ok(_) => {
                            // noop
                        }
                        Err(_) => {
                            break;
                        }
                    }
                }
            })
            .unwrap();
        Self { handle: h }
    }
}

impl AudioDevice for NoAudioDevice {}

pub(crate) fn format_time(secs: f32) -> String {
    const MIN: f32 = 60.0;
    const HR: f32 = MIN * 60.0;

    if secs >= HR {
        format!(
            "{:0>2.0}h {:0>2.0}m {:0>2.0}s",
            (secs / HR).floor(),
            ((secs % HR) / MIN).floor(),
            (secs % MIN).floor()
        )
    } else if secs >= MIN {
        format!(
            "{:0>2.0}m {:0>2.0}s",
            (secs / MIN).floor(),
            (secs % MIN).floor()
        )
    } else {
        format!("{:0>2.2}s", secs)
    }
}

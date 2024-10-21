#![warn(missing_docs)]
#![allow(rustdoc::bare_urls)]
#![doc = include_str!("../README.md")]
//! # Simple video player example
//! ```
#![doc = include_str!("../examples/main.rs")]
//! ```

mod subtitle;
mod audio;
mod player;
mod ffmpeg;

pub use audio::*;
pub use player::*;

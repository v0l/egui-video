#![warn(missing_docs)]
#![allow(rustdoc::bare_urls)]
#![doc = include_str!("../README.md")]
//! # Simple video player example
//! ```
#![doc = include_str!("../examples/main.rs")]
//! ```

mod audio;
pub use audio::*;
#[cfg(feature = "hls")]
mod hls;
#[cfg(feature = "default-overlay")]
mod overlay;
mod player;
pub use player::*;
mod stream;
#[cfg(feature = "subtitles")]
mod subtitle;
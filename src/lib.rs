#![warn(missing_docs)]
#![allow(rustdoc::bare_urls)]
#![doc = include_str!("../README.md")]
//! # Simple video player example
//! ```
#![doc = include_str!("../examples/main.rs")]
//! ```

mod audio;
pub mod ffmpeg;
mod overlay;
mod player;
mod subtitle;

use crate::overlay::DefaultOverlay;
pub use audio::*;
use egui::{Response, Ui, Widget};
pub use ffmpeg_rs_raw::ffmpeg_sys_the_third;
pub use player::*;

pub struct Player {
    inner: CustomPlayer<DefaultOverlay>,
}

impl Player {
    pub fn new(ctx: &egui::Context, input_path: &String) -> Self {
        Self {
            inner: CustomPlayer::new(DefaultOverlay {}, ctx, input_path),
        }
    }
}

impl PlayerControls for Player {
    fn elapsed(&self) -> f32 {
        self.inner.elapsed()
    }

    fn duration(&self) -> f32 {
        self.inner.duration()
    }

    fn framerate(&self) -> f32 {
        self.inner.framerate()
    }

    fn size(&self) -> (u16, u16) {
        self.inner.size()
    }

    fn state(&self) -> PlayerState {
        self.inner.state()
    }

    fn pause(&mut self) {
        self.inner.pause()
    }

    fn start(&mut self) {
        self.inner.start()
    }

    fn stop(&mut self) {
        self.inner.stop()
    }

    fn seek(&mut self, seek_frac: f32) {
        self.inner.seek(seek_frac)
    }

    fn set_volume(&mut self, volume: u8) {
        self.inner.set_volume(volume)
    }

    fn volume(&self) -> u8 {
        self.inner.volume()
    }

    fn volume_f32(&self) -> f32 {
        self.inner.volume_f32()
    }

    fn set_volume_f32(&mut self, volume: f32) {
        self.inner.set_volume_f32(volume)
    }

    fn looping(&self) -> bool {
        self.inner.looping()
    }

    fn set_looping(&mut self, looping: bool) {
        self.inner.set_looping(looping)
    }

    fn debug(&self) -> bool {
        self.inner.debug()
    }

    fn set_debug(&mut self, debug: bool) {
        self.inner.set_debug(debug)
    }
}

impl Widget for &mut Player {
    fn ui(self, ui: &mut Ui) -> Response {
        self.inner.ui(ui)
    }
}

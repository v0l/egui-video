#![warn(missing_docs)]
#![allow(rustdoc::bare_urls)]
#![doc = include_str!("../README.md")]
//! # Simple video player example
//! ```
#![doc = include_str!("../examples/main.rs")]
//! ```

mod audio;
#[cfg(feature = "hls")]
mod hls;
pub mod media_player;
mod overlay;
mod player;
#[cfg(feature = "subtitles")]
mod subtitle;

use crate::overlay::DefaultOverlay;
pub use audio::*;
use egui::{Response, Ui, Widget};
pub use ffmpeg_rs_raw;
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

    pub fn render(&mut self, ui: &mut Ui) -> Response {
        self.inner.render(ui)
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

    fn volume(&self) -> f32 {
        self.inner.volume()
    }

    fn set_volume(&mut self, volume: f32) {
        self.inner.set_volume(volume)
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

    fn playback_speed(&self) -> f32 {
        self.inner.playback_speed()
    }

    fn set_playback_speed(&mut self, speed: f32) {
        self.inner.set_playback_speed(speed)
    }

    fn maintain_aspect(&self) -> bool {
        self.inner.maintain_aspect()
    }

    fn set_maintain_aspect(&mut self, maintain_aspect: bool) {
        self.inner.set_maintain_aspect(maintain_aspect)
    }

    fn fullscreen(&self) -> bool {
        self.inner.fullscreen()
    }

    fn set_fullscreen(&mut self, fullscreen: bool) {
        self.inner.set_fullscreen(fullscreen)
    }

    fn key_binds(&self) -> bool {
        self.inner.key_binds()
    }

    fn set_key_binds(&mut self, key_binds: bool) {
        self.inner.set_key_binds(key_binds)
    }
}

impl Widget for &mut Player {
    fn ui(self, ui: &mut Ui) -> Response {
        self.inner.ui(ui)
    }
}

use crate::stream::{AudioSamples, DecoderInfo, MediaDecoder, SubtitlePacket, VideoFrame};
#[cfg(feature = "subtitles")]
use crate::subtitle::Subtitle;
use crate::{AudioDevice, BungeeWrapper};
use anyhow::Result;
use cpal::traits::DeviceTrait;
use cpal::{SampleFormat, Stream};
use egui::load::SizedTexture;
use egui::{
    Align2, Color32, ColorImage, Event, FontId, Image, Key, Rect, Response, Sense, Stroke,
    StrokeKind, TextureHandle, TextureOptions, Ui, Vec2, Widget, pos2, vec2,
};
use log::{error, info, trace};
use std::collections::VecDeque;
use std::ops::Add;
use std::sync::atomic::{AtomicBool, AtomicI16, AtomicI64, AtomicU8, AtomicU16, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

#[cfg(not(feature = "subtitles"))]
struct Subtitle;

/// Generic overlay for player controls
pub trait PlayerOverlay: Send {
    fn show(&self, ui: &mut Ui, frame_response: &Response, p: &PlaybackInfo) -> PlaybackUpdate;
}

/// Current state object from the player
#[derive(Debug, Clone)]
pub struct PlaybackInfo {
    pub state: PlayerState,
    pub duration: f32,
    pub elapsed: f32,
    pub volume: f32,
    pub muted: bool,
    pub playback_speed: f32,
    pub looping: bool,
    pub fullscreen: bool,
    pub debug: bool,
}

/// Requested change in player state
#[derive(Debug, Clone, Default)]
pub struct PlaybackUpdate {
    /// Set the player state
    pub set_state: Option<PlayerState>,
    /// Set the playback volume
    pub set_volume: Option<f32>,
    /// Seek to playback position as a percentage of the duration of the media ??
    pub set_seek: Option<f32>,
    /// Set the audio to muted
    pub set_muted: Option<bool>,
    /// Set the playback speed
    pub set_playback_speed: Option<f32>,
    /// Set playback to loop
    pub set_looping: Option<bool>,
    /// Set player fullscreen mode
    pub set_fullscreen: Option<bool>,
    /// Set debug information display
    pub set_debug: Option<bool>,
}

/// The [`Player`] processes and controls streams of video/audio.
/// This is what you use to show a video file.
/// Initialize once, and use the [`Player::ui`] or [`Player::ui_at()`] functions to show the playback.
pub struct Player<T> {
    overlay: T,
    state: Arc<AtomicU8>,
    volume: Arc<AtomicU16>,
    playback_speed: Arc<AtomicI16>,
    debug: bool,

    avg_fps: f32,
    avg_fps_start: Instant,
    last_frame_counter: u64,

    /// The video frame to display
    frame: TextureHandle,
    /// Start presentation time for the current frame
    frame_pts: f64,
    /// End presentation timestamp for the current frame
    frame_pts_end: f64,
    /// Clock time when the frame began
    frame_instant: Instant,
    /// The current PTS in milliseconds
    current_pts: Arc<AtomicI64>,

    /// How many frames have been rendered so far
    frame_counter: u64,
    /// Maintain video aspect ratio
    maintain_aspect: bool,
    /// If player should fullscreen
    fullscreen: bool,
    /// If key presses should be handled
    key_binds: bool,

    /// Stream info
    stream_info: Option<DecoderInfo>,

    ctx: egui::Context,
    input_path: String,
    audio: Option<(AudioDevice, Stream)>,
    subtitle: Option<Subtitle>,

    /// Media stream decoder thread
    media_player: MediaDecoder,
    rx_metadata: Receiver<DecoderInfo>,
    rx_video: Receiver<VideoFrame>,
    rx_subtitle: Receiver<SubtitlePacket>,

    /// An error which prevented playback
    error: Option<String>,

    /// Message to show on scree for a short time (usually from keyboard input)
    osd: Option<String>,
    osd_end: Instant,
}

/// The possible states of a [`Player`].
#[derive(PartialEq, Clone, Copy, Debug)]
#[repr(u8)]
#[non_exhaustive]
pub enum PlayerState {
    /// No playback.
    Stopped,
    /// Stream is seeking. Inner bool represents whether the seek is currently in progress.
    Seeking,
    /// Playback is paused.
    Paused,
    /// Playback is ongoing.
    Playing,
}

impl From<u8> for PlayerState {
    fn from(value: u8) -> Self {
        match value {
            0 => PlayerState::Stopped,
            1 => PlayerState::Seeking,
            2 => PlayerState::Paused,
            3 => PlayerState::Playing,
            _ => PlayerState::Stopped,
        }
    }
}

impl<T> Player<T>
where
    T: PlayerOverlay + 'static,
{
    /// Store the next image
    fn load_frame(&mut self, frame: VideoFrame) {
        trace!(
            "Loading video frame idx={}, pts={}, dur={}",
            self.frame_counter, frame.pts, frame.duration
        );
        self.frame.set(frame.data, TextureOptions::default());
        self.frame_pts = frame.pts;
        self.frame_pts_end = frame.pts + frame.duration;
        self.frame_counter += 1;
        self.frame_instant = Instant::now();
    }

    /// The current presentation time
    fn current_pts(&self) -> f64 {
        let now = Instant::now();
        let diff = now.duration_since(self.frame_instant).as_secs_f64();
        self.frame_pts + diff
    }

    fn request_repaint_for_next_frame(&self) {
        let now = Instant::now();
        let frame_duration = self.frame_pts_end - self.frame_pts;
        let next_frame = self
            .frame_instant
            .add(Duration::from_secs_f64(frame_duration));
        if now > next_frame {
            trace!("request repaint now!");
            self.ctx.request_repaint();
        } else {
            let tt_nf = next_frame - now;
            trace!("request repaint for {}ms", tt_nf.as_millis());
            self.ctx.request_repaint_after(tt_nf);
        }
    }

    /// Check if the current frame should be flipped
    fn check_load_frame(&mut self) -> bool {
        if self.state.load(Ordering::Relaxed) == PlayerState::Paused as u8 {
            // force frame to start now, while paused
            self.frame_instant = Instant::now();
            return false;
        }

        let now = Instant::now();
        now >= self.frame_end_instant()
    }

    /// Duration of the current frame
    fn frame_duration(&self) -> Duration {
        Duration::from_secs_f64(self.frame_pts_end - self.frame_pts)
    }

    /// Instant when the current frame ends
    fn frame_end_instant(&self) -> Instant {
        self.frame_instant.add(self.frame_duration())
    }

    /// Handle key input
    fn handle_keys(&mut self, ui: &mut Ui, state: &PlaybackInfo) -> PlaybackUpdate {
        const SEEK_STEP: f32 = 5.0;
        const VOLUME_STEP: f32 = 0.1;
        const SPEED_STEP: f32 = 0.1;

        if !self.key_binds {
            return PlaybackUpdate::default();
        }

        let mut p_ret = PlaybackUpdate::default();
        ui.input(|inputs| {
            for e in &inputs.events {
                match e {
                    Event::Key { key, pressed, .. } if *pressed => match key {
                        Key::Space => {
                            if state.state == PlayerState::Playing {
                                p_ret.set_state.replace(PlayerState::Paused);
                            } else {
                                p_ret.set_state.replace(PlayerState::Playing);
                            }
                        }
                        Key::OpenBracket => {
                            p_ret
                                .set_playback_speed
                                .replace(state.playback_speed - SPEED_STEP);
                        }
                        Key::CloseBracket => {
                            p_ret
                                .set_playback_speed
                                .replace(state.playback_speed + SPEED_STEP);
                        }
                        Key::ArrowRight => {
                            p_ret.set_seek.replace(state.elapsed + SEEK_STEP);
                        }
                        Key::ArrowLeft => {
                            p_ret.set_seek.replace(state.elapsed - SEEK_STEP);
                        }
                        Key::ArrowUp => {
                            p_ret.set_volume.replace(state.volume + VOLUME_STEP);
                        }
                        Key::ArrowDown => {
                            p_ret.set_volume.replace(state.volume - VOLUME_STEP);
                        }
                        Key::F => {
                            p_ret.set_fullscreen.replace(!state.fullscreen);
                        }
                        Key::F1 => {
                            p_ret.set_debug.replace(!state.debug);
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        });
        p_ret
    }

    fn process_state(&mut self) {
        if let Ok(md) = self.rx_metadata.try_recv() {
            self.stream_info.replace(md);
        }

        if self.state.load(Ordering::Relaxed) == PlayerState::Stopped as u8 {
            // nothing to do, playback is stopped
            return;
        }

        //self.media_player.set_target_size(size);
        let current_pts = self.frame_pts + self.frame_instant.elapsed().as_secs_f64();
        self.current_pts
            .store((current_pts / 1000.0) as _, Ordering::Relaxed);

        // check if we should load the next video frame
        if !self.check_load_frame() {
            self.request_repaint_for_next_frame();
            return;
        }

        // reset avg fps every 1s
        let n_frames = self.frame_counter - self.last_frame_counter;
        if n_frames >= 30 {
            self.avg_fps = n_frames as f32 / (Instant::now() - self.avg_fps_start).as_secs_f32();
            self.avg_fps_start = Instant::now();
            self.last_frame_counter = self.frame_counter;
        }

        if let Ok(msg) = self.rx_video.recv() {
            self.load_frame(msg);
            // break on video frame
            // once we load the next frame this loop will not call again until
            // this frame is over (pts + duration)
            self.request_repaint_for_next_frame();
            return;
        }

        // if no frames were found just request repaint again
        self.request_repaint_for_next_frame();
    }

    fn generate_frame_image(&self, size: Vec2) -> Image {
        Image::new(SizedTexture::new(self.frame.id(), size)).sense(Sense::click())
    }

    fn render_frame(&self, ui: &mut Ui) -> Response {
        self.render_frame_at(ui, ui.available_rect_before_wrap())
    }

    /// Exact size of the video frame inside a given [Rect]
    fn video_frame_size(&self, rect: Rect) -> Vec2 {
        if self.maintain_aspect {
            let v_index = self.media_player.video_stream_index.load(Ordering::Relaxed);
            let bv = self
                .stream_info
                .as_ref()
                .and_then(|s| s.streams.iter().find(|v| v.index == v_index as _));
            let video_size = bv
                .map(|v| vec2(v.width as f32, v.height as f32))
                .unwrap_or(rect.size());
            let ratio = video_size.x / video_size.y;
            let rect_ratio = rect.width() / rect.height();
            if ratio > rect_ratio {
                let h = rect.width() / ratio;
                vec2(rect.width().floor(), h.floor())
            } else if ratio < rect_ratio {
                let w = rect.height() * ratio;
                vec2(w.floor(), rect.height().floor())
            } else {
                rect.size()
            }
        } else {
            rect.size()
        }
    }

    fn render_frame_at(&self, ui: &mut Ui, rect: Rect) -> Response {
        let video_size = self.video_frame_size(rect);
        ui.painter()
            .rect(rect, 0.0, Color32::BLACK, Stroke::NONE, StrokeKind::Middle);
        ui.put(rect, self.generate_frame_image(video_size))
    }

    fn render_subtitles(&mut self, ui: &mut Ui) {
        #[cfg(feature = "subtitles")]
        if let Some(s) = self.subtitle.as_ref() {
            let sub_end = s.pts + s.duration;
            if sub_end < self.current_pts() {
                self.subtitle.take();
            } else {
                ui.add(s);
            }
        }
    }

    // fn render_debug(&mut self, ui: &mut Ui, frame_response: &Response) {
    //     let painter = ui.painter();
    //
    //     const PADDING: f32 = 5.0;
    //     let vec_padding = vec2(PADDING, PADDING);
    //     let job = self.debug_inner(frame_response.rect);
    //     let galley = painter.layout_job(job);
    //     let mut bg_pos = galley
    //         .rect
    //         .translate(frame_response.rect.min.to_vec2() + vec_padding);
    //     bg_pos.max += vec_padding * 2.0;
    //     painter.rect_filled(bg_pos, PADDING, Color32::from_black_alpha(150));
    //     painter.galley(bg_pos.min + vec_padding, galley, Color32::PLACEHOLDER);
    // }

    fn show_osd(&mut self, msg: &str) {
        self.osd = Some(msg.to_string());
        self.osd_end = Instant::now() + Duration::from_secs(2);
    }

    // fn debug_inner(&mut self, frame_response: Rect) -> LayoutJob {
    //     let v_pts = self.elapsed();
    //     let a_pts = self
    //         .audio
    //         .as_ref()
    //         .map(|a| a.buffer.lock().map(|b| b.audio_pos).unwrap_or(0.0))
    //         .unwrap_or(0.0);
    //     let font = TextFormat::simple(FontId::monospace(11.), Color32::WHITE);
    //
    //     let mut layout = LayoutJob::default();
    //     let buffer = self.pts_to_sec(self.media_player.buffer_size());
    //     layout.append(
    //         &format!(
    //             "sync: v:{:.3}s, a:{:.3}s, a-sync:{:.3}s, buffer: {:.3}s",
    //             v_pts,
    //             a_pts,
    //             a_pts - v_pts,
    //             buffer
    //         ),
    //         0.0,
    //         font.clone(),
    //     );
    //
    //     let video_size = self.video_frame_size(frame_response);
    //     layout.append(
    //         &format!(
    //             "\nplayback: {:.2} fps ({:.2}x), volume={:.0}%, resolution={}x{}",
    //             self.avg_fps,
    //             self.avg_fps / self.framerate(),
    //             100.0 * self.volume(),
    //             video_size.x,
    //             video_size.y
    //         ),
    //         0.0,
    //         font.clone(),
    //     );
    //
    //     if let Some(info) = self.info.as_ref() {
    //         let bitrate_str = if info.bitrate > 1_000_000 {
    //             format!("{:.1}M", info.bitrate as f32 / 1_000_000.0)
    //         } else if info.bitrate > 1_000 {
    //             format!("{:.1}k", info.bitrate as f32 / 1_000.0)
    //         } else {
    //             info.bitrate.to_string()
    //         };
    //
    //         layout.append(
    //             &format!(
    //                 "\nduration: {}, bitrate: {}",
    //                 format_time(info.duration),
    //                 bitrate_str
    //             ),
    //             0.0,
    //             font.clone(),
    //         );
    //
    //         fn print_chan(
    //             layout: &mut LayoutJob,
    //             font: TextFormat,
    //             chan: Option<&StreamInfo>,
    //             decoders: &Vec<DecoderInfo>,
    //         ) {
    //             if let Some(c) = chan {
    //                 layout.append(&format!("\n{}", c), 0.0, font.clone());
    //                 if let Some(decoder) = decoders.iter().find(|d| d.index == c.index) {
    //                     layout.append(&format!("\n\tdecoder={}", decoder.codec), 0.0, font.clone());
    //                 }
    //             }
    //         }
    //         print_chan(&mut layout, font.clone(), info.best_video(), &self.decoders);
    //         print_chan(&mut layout, font.clone(), info.best_audio(), &self.decoders);
    //         print_chan(
    //             &mut layout,
    //             font.clone(),
    //             info.best_subtitle(),
    //             &self.decoders,
    //         );
    //     }
    //
    //     layout
    // }

    /// Create a new [`Player`].
    pub fn new(overlay: T, ctx: &egui::Context, input_path: &str) -> Result<Self> {
        // volume arc
        let vol = Arc::new(AtomicU16::new(u16::MAX));
        let state = Arc::new(AtomicU8::new(1));
        let speed = Arc::new(AtomicI16::new(100));
        let mute = Arc::new(AtomicBool::new(true));
        let pts = Arc::new(AtomicI64::new(0));

        let (media_player, streams) =
            MediaDecoder::new(input_path).expect("Failed to create media playback");
        let audio = AudioDevice::open_default_audio_stream(
            vol.clone(),
            mute.clone(),
            state.clone(),
            speed.clone(),
            pts.clone(),
            streams.audio,
        )?;

        let init_size = ctx.available_rect();
        Ok(Self {
            state,
            playback_speed: speed,
            overlay,
            key_binds: false,
            input_path: input_path.to_string(),
            volume: vol,
            frame: ctx.load_texture(
                "video_frame",
                ColorImage::filled(
                    [init_size.width() as usize, init_size.height() as usize],
                    Color32::BLACK,
                ),
                Default::default(),
            ),
            frame_instant: Instant::now(),
            frame_pts: 0.0,
            frame_pts_end: 0.0,
            current_pts: pts,
            ctx: ctx.clone(),
            audio: Some(audio),
            subtitle: None,
            media_player,
            rx_metadata: streams.metadata,
            rx_video: streams.video,
            debug: false,
            avg_fps: 0.0,
            avg_fps_start: Instant::now(),
            frame_counter: 0,
            last_frame_counter: 0,
            error: None,
            osd: None,
            maintain_aspect: true,
            fullscreen: false,
            osd_end: Instant::now(),
            stream_info: None,
            rx_subtitle: streams.subtitle,
        })
    }
}

impl<T> Widget for &mut Player<T>
where
    T: PlayerOverlay + 'static,
{
    fn ui(self, ui: &mut Ui) -> Response {
        self.render(ui)
    }
}

impl<T> Player<T>
where
    T: PlayerOverlay + 'static,
{
    /// Render player in available space
    pub fn render(&mut self, ui: &mut Ui) -> Response {
        let size = ui.available_size();

        let state = PlaybackInfo {
            state: self.state.load(Ordering::Relaxed).into(),
            duration: 0.0,
            elapsed: 0.0,
            volume: 0.0,
            muted: false,
            playback_speed: 1.0,
            looping: false,
            fullscreen: false,
            debug: false,
        };
        let upd = self.handle_keys(ui, &state);
        self.process_update(upd);
        self.process_state();
        let frame_response = self.render_frame(ui);
        self.render_subtitles(ui);
        self.render_overlay(ui, &frame_response, &state);
        if let Some(error) = &self.error {
            ui.painter().text(
                pos2(size.x / 2.0, size.y / 2.0),
                Align2::CENTER_BOTTOM,
                error,
                FontId::proportional(30.),
                Color32::DARK_RED,
            );
        }
        if self.osd_end < Instant::now() {
            self.osd.take();
        }
        if let Some(osd) = &self.osd {
            ui.painter().text(
                pos2(size.x - 10.0, 50.0),
                Align2::RIGHT_TOP,
                osd,
                FontId::proportional(20.),
                Color32::WHITE,
            );
        }
        // if self.debug {
        //     self.render_debug(ui, &frame_response);
        // }
        frame_response
    }

    fn process_update(&mut self, updates: PlaybackUpdate) {
        // TODO: change internal state
    }

    fn render_overlay(&mut self, ui: &mut Ui, frame: &Response, state: &PlaybackInfo) {
        let upd = self.overlay.show(ui, frame, state);
        self.process_update(upd);
    }
}

use crate::audio::{AudioBuffer, PlayerAudioStream};
use crate::media_player::{AudioSamplesData, DecoderInfo, MediaPlayer, MediaPlayerData};
#[cfg(feature = "subtitles")]
use crate::subtitle::Subtitle;
use crate::AudioDevice;
use cpal::traits::DeviceTrait;
use cpal::SampleFormat;
use egui::load::SizedTexture;
use egui::text::LayoutJob;
use egui::{
    pos2, vec2, Align2, Color32, ColorImage, Event, FontId, Image, Key, Rect, Response, Sense,
    Stroke, TextFormat, TextureHandle, TextureOptions, Ui, Vec2, ViewportCommand, ViewportId,
    Widget,
};
use ffmpeg_rs_raw::ffmpeg_sys_the_third::AVMediaType;
use ffmpeg_rs_raw::{format_time, DemuxerInfo, StreamInfo};
use log::{debug, info, trace};
use std::sync::atomic::{AtomicI16, AtomicU16, AtomicU8, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(not(feature = "subtitles"))]
struct Subtitle;

#[derive(Debug)]
/// IPC for player
enum PlayerMessage {
    /// Set player state
    SetState(PlayerState),
    /// Seek to position in seconds
    Seek(f32),
    /// Set player volume
    SetVolume(f32),
    /// Set player looping
    SetLooping(bool),
    /// Set debug overlay
    SetDebug(bool),
    /// Select playing stream
    SelectStream(AVMediaType, usize),
    /// Set playback speed
    SetPlaybackSpeed(f32),
    /// Set video aspect to be the same as source content
    SetMaintainAspect(bool),
    /// Change fullscreen state
    SetFullscreen(bool),
    /// Change key_binds state
    SetKeyBinds(bool),
}

/// The [`CustomPlayer`] processes and controls streams of video/audio.
/// This is what you use to show a video file.
/// Initialize once, and use the [`CustomPlayer::ui`] or [`CustomPlayer::ui_at()`] functions to show the playback.
pub struct CustomPlayer<T> {
    overlay: T,

    state: Arc<AtomicU8>,
    looping: bool,
    volume: Arc<AtomicU16>,
    playback_speed: Arc<AtomicI16>,
    debug: bool,

    avg_fps: f32,
    avg_fps_start: Instant,
    last_frame_counter: u64,

    /// The video frame to display
    frame: TextureHandle,
    /// Video frame PTS
    frame_pts: i64,
    /// Duration to show [frame]
    frame_duration: i64,
    /// Realtime of when [frame] was shown
    frame_start: Instant,
    /// How many frames have been rendered so far
    frame_counter: u64,
    /// Maintain video aspect ratio
    maintain_aspect: bool,
    /// If player should fullscreen
    fullscreen: bool,
    /// If key resses should be handled
    key_binds: bool,

    /// Stream info
    info: Option<DemuxerInfo>,
    decoders: Vec<DecoderInfo>,

    ctx: egui::Context,
    input_path: String,
    audio: Option<PlayerAudioStream>,
    subtitle: Option<Subtitle>,

    /// ffmpeg media player
    media_player: MediaPlayer,

    /// An error which prevented playback
    error: Option<String>,

    /// Message to show on scree for a short time (usually from keyboard input)
    osd: Option<String>,
    osd_end: Instant,
}

/// The possible states of a [`CustomPlayer`].
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

/// A media player controller
pub trait PlayerControls {
    /// Elapsed time in seconds
    fn elapsed(&self) -> f32;
    /// Duration in seconds
    fn duration(&self) -> f32;
    /// Framerate in frames per second
    fn framerate(&self) -> f32;
    /// Video size in screen logical pixels
    fn size(&self) -> (u16, u16);
    /// Player state (play / paused etc.)
    fn state(&self) -> PlayerState;
    /// Pause playback
    fn pause(&mut self);
    /// Start playback (or resume)
    fn start(&mut self);
    /// Stop playback completely
    fn stop(&mut self);
    /// Seek to a new location in the video as seconds
    fn seek(&mut self, pos: f32);
    /// Get the volume (0-1)
    fn volume(&self) -> f32;
    /// Set the volume (0-1)
    fn set_volume(&mut self, volume: f32);
    /// If the playback will loop after it ends
    fn looping(&self) -> bool;
    /// Set playback looping
    fn set_looping(&mut self, looping: bool);
    /// Is the debug overlay showing
    fn debug(&self) -> bool;
    /// Set debug overlay visibility
    fn set_debug(&mut self, debug: bool);
    /// Get the playback speed multiplier (1x = 1.0)
    fn playback_speed(&self) -> f32;
    /// Set the playback speed (1x = 1.0)
    fn set_playback_speed(&mut self, speed: f32);
    /// Is the aspect ratio maintained during scaling
    fn maintain_aspect(&self) -> bool;
    /// Set if the aspect ratio should be maintained when scaling
    fn set_maintain_aspect(&mut self, maintain_aspect: bool);
    /// If player is in fullscreen mode
    fn fullscreen(&self) -> bool;
    /// Set player is fullscreen or not
    fn set_fullscreen(&mut self, fullscreen: bool);
    /// If keyboard inputs are accepted
    fn key_binds(&self) -> bool;
    /// Set if keyboard inputs are handled
    fn set_key_binds(&mut self, key_binds: bool);
}

/// Wrapper to store player info and pass to overlay impl
pub struct PlayerOverlayState {
    elapsed: f32,
    duration: f32,
    framerate: f32,
    playback_speed: f32,
    size: (u16, u16),
    volume: f32,
    looping: bool,
    debug: bool,
    state: PlayerState,
    maintain_aspect: bool,
    fullscreen: bool,
    key_binds: bool,
    inbox: Sender<PlayerMessage>,
}

impl PlayerControls for PlayerOverlayState {
    fn elapsed(&self) -> f32 {
        self.elapsed
    }

    fn duration(&self) -> f32 {
        self.duration
    }

    fn framerate(&self) -> f32 {
        self.framerate
    }

    fn size(&self) -> (u16, u16) {
        self.size
    }

    fn state(&self) -> PlayerState {
        self.state
    }

    fn pause(&mut self) {
        self.inbox
            .send(PlayerMessage::SetState(PlayerState::Paused))
            .unwrap();
    }

    fn start(&mut self) {
        self.inbox
            .send(PlayerMessage::SetState(PlayerState::Playing))
            .unwrap();
    }

    fn stop(&mut self) {
        self.inbox
            .send(PlayerMessage::SetState(PlayerState::Stopped))
            .unwrap();
    }

    fn seek(&mut self, seek: f32) {
        self.inbox.send(PlayerMessage::Seek(seek)).unwrap();
    }

    fn volume(&self) -> f32 {
        self.volume as f32 / u16::MAX as f32
    }

    fn set_volume(&mut self, volume: f32) {
        self.inbox.send(PlayerMessage::SetVolume(volume)).unwrap();
    }

    fn looping(&self) -> bool {
        self.looping
    }

    fn set_looping(&mut self, looping: bool) {
        self.inbox.send(PlayerMessage::SetLooping(looping)).unwrap();
    }

    fn debug(&self) -> bool {
        self.debug
    }

    fn set_debug(&mut self, debug: bool) {
        self.inbox.send(PlayerMessage::SetDebug(debug)).unwrap();
    }

    fn playback_speed(&self) -> f32 {
        self.playback_speed
    }

    fn set_playback_speed(&mut self, speed: f32) {
        self.inbox
            .send(PlayerMessage::SetPlaybackSpeed(speed))
            .unwrap()
    }

    fn maintain_aspect(&self) -> bool {
        self.maintain_aspect
    }

    fn set_maintain_aspect(&mut self, maintain_aspect: bool) {
        self.inbox
            .send(PlayerMessage::SetMaintainAspect(maintain_aspect))
            .unwrap()
    }

    fn fullscreen(&self) -> bool {
        self.fullscreen
    }

    fn set_fullscreen(&mut self, fullscreen: bool) {
        self.inbox
            .send(PlayerMessage::SetFullscreen(fullscreen))
            .unwrap()
    }

    fn key_binds(&self) -> bool {
        self.key_binds
    }

    fn set_key_binds(&mut self, key_binds: bool) {
        self.inbox
            .send(PlayerMessage::SetKeyBinds(key_binds))
            .unwrap()
    }
}

/// Overlay controls for the video player
pub trait PlayerOverlay {
    fn show(&self, ui: &mut Ui, frame: &Response, state: &mut PlayerOverlayState);
}

impl<T> Drop for CustomPlayer<T> {
    fn drop(&mut self) {
        self.state
            .store(PlayerState::Stopped as u8, Ordering::Relaxed);
    }
}

impl<T> CustomPlayer<T> {
    /// Store the next image
    fn load_frame(&mut self, image: ColorImage, pts: i64, duration: i64) {
        self.frame.set(image, TextureOptions::default());
        self.frame_pts = pts;
        self.frame_duration = (duration as f32 / self.playback_speed()) as i64;
        self.frame_counter += 1;

        let now = Instant::now();
        let frame_end = self.frame_start + self.pts_to_duration(self.frame_duration);
        let frame_delay = if now > frame_end {
            now - frame_end
        } else {
            Duration::ZERO
        };
        self.frame_start = Instant::now() - frame_delay;
    }

    fn request_repaint_for_next_frame(&self) {
        let now = Instant::now();
        let next_frame = self.next_frame_timestamp();
        if now > next_frame {
            trace!("request repaint now!");
            self.ctx.request_repaint();
        } else {
            let ttnf = next_frame - now;
            trace!("request repaint for {}ms", ttnf.as_millis());
            self.ctx.request_repaint_after(ttnf);
        }
    }

    /// Timestamp when the next frame should be shown
    fn next_frame_timestamp(&self) -> Instant {
        self.frame_start + self.pts_to_duration(self.frame_duration)
    }

    /// Check if the current frame should be flipped
    fn check_load_frame(&mut self) -> bool {
        if self.state.load(Ordering::Relaxed) == PlayerState::Paused as u8 {
            // force frame to start now, while paused
            self.frame_start = Instant::now();
            return false;
        }
        let now = Instant::now();
        now >= self.next_frame_timestamp()
    }

    /// Toggle playback
    fn toggle_play_pause(&mut self) {
        match self.state() {
            PlayerState::Playing => self.pause(),
            PlayerState::Paused | PlayerState::Stopped => self.start(),
            _ => {}
        }
    }

    /// Handle key input
    fn handle_keys(&mut self, ui: &mut Ui) {
        const SEEK_STEP: f32 = 5.0;
        const VOLUME_STEP: f32 = 0.1;
        const SPEED_STEP: f32 = 0.1;

        if !self.key_binds {
            return;
        }

        ui.input(|inputs| {
            for e in &inputs.events {
                match e {
                    Event::Key { key, pressed, .. } if *pressed => match key {
                        Key::Space => {
                            self.toggle_play_pause();
                        }
                        Key::OpenBracket => {
                            self.set_playback_speed(self.playback_speed() - SPEED_STEP);
                        }
                        Key::CloseBracket => {
                            self.set_playback_speed(self.playback_speed() + SPEED_STEP);
                        }
                        Key::ArrowRight => {
                            self.seek(self.elapsed() + SEEK_STEP);
                        }
                        Key::ArrowLeft => {
                            self.seek(self.elapsed() - SEEK_STEP);
                        }
                        Key::ArrowUp => {
                            self.set_volume(self.volume() + VOLUME_STEP);
                        }
                        Key::ArrowDown => {
                            self.set_volume(self.volume() - VOLUME_STEP);
                        }
                        Key::F => {
                            self.set_fullscreen(!self.fullscreen());
                            self.ctx.send_viewport_cmd_to(
                                ViewportId::ROOT,
                                ViewportCommand::Fullscreen(self.fullscreen),
                            );
                        }
                        Key::F1 => {
                            self.set_debug(!self.debug());
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        });
    }

    fn process_state(&mut self, size: Vec2) {
        if self.state.load(Ordering::Relaxed) == PlayerState::Stopped as u8 {
            // nothing to do, playback is stopped
            return;
        }

        self.media_player.set_target_size(size);

        // check if we should load the next video frame
        if !self.check_load_frame() {
            self.request_repaint_for_next_frame();
            return;
        }

        // reset avg fps every 1s
        let n_frames = self.frame_counter - self.last_frame_counter;
        if n_frames >= self.framerate() as u64 {
            self.avg_fps = n_frames as f32 / (Instant::now() - self.avg_fps_start).as_secs_f32();
            self.avg_fps_start = Instant::now();
            self.last_frame_counter = self.frame_counter;
        }

        while let Some(msg) = self.media_player.next() {
            match msg {
                MediaPlayerData::MediaInfo(i) => {
                    self.info = Some(i);
                }
                MediaPlayerData::VideoFrame(pts, duration, f) => {
                    self.load_frame(f, pts, duration);

                    let v_pts = self.pts_to_sec(self.frame_pts);
                    if let Some(a) = self.audio.as_mut() {
                        if let Ok(mut q) = a.buffer.lock() {
                            q.video_pos = v_pts as f32;
                        }
                    }
                    // break on video frame
                    // once we load the next frame this loop will not call again until
                    // this frame is over (pts + duration)
                    self.request_repaint_for_next_frame();
                    break;
                }
                MediaPlayerData::AudioSamples(data) => {
                    if let Some(a) = self.audio.as_mut() {
                        if let Ok(mut q) = a.buffer.lock() {
                            q.add_samples(data.samples);
                        }
                    }
                }
                MediaPlayerData::Subtitles(pts, duration, text, codec) => {
                    #[cfg(feature = "subtitles")]
                    {
                        self.subtitle = Some(Subtitle::new(text, pts, duration, codec));
                    }
                }
                MediaPlayerData::Error(e) => {
                    self.error = Some(e);
                    self.stop();
                }
                MediaPlayerData::DecoderInfo(i) => {
                    self.decoders.push(i);
                }
            }
        }

        if !self.media_player.is_running() && self.media_player.buffer_len() == 0 {
            trace!("stopping, media_player is not running");
            self.stop();
            if self.looping {
                self.start();
            }
        }

        // if no frames were found just request repaint again
        self.request_repaint_for_next_frame();
    }

    fn process_player_message(&mut self, m: PlayerMessage) {
        match m {
            PlayerMessage::SetState(s) => match s {
                PlayerState::Stopped => self.stop(),
                PlayerState::Paused => self.pause(),
                PlayerState::Playing => self.start(),
                _ => {}
            },
            PlayerMessage::Seek(v) => {
                self.seek(v);
            }
            PlayerMessage::SetVolume(v) => {
                self.set_volume(v);
            }
            PlayerMessage::SelectStream(_, _) => {}
            PlayerMessage::SetLooping(l) => self.looping = l,
            PlayerMessage::SetDebug(d) => self.debug = d,
            PlayerMessage::SetPlaybackSpeed(s) => {
                self.set_playback_speed(s);
            }
            PlayerMessage::SetMaintainAspect(a) => self.set_maintain_aspect(a),
            PlayerMessage::SetFullscreen(f) => self.set_fullscreen(f),
            PlayerMessage::SetKeyBinds(f) => self.set_key_binds(f),
        }
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
            let bv = self.info.as_ref().and_then(|i| i.best_video());
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
            .rect(rect, 0.0, Color32::BLACK, Stroke::default());
        ui.put(rect, self.generate_frame_image(video_size))
    }

    fn render_subtitles(&mut self, ui: &mut Ui) {
        #[cfg(feature = "subtitles")]
        if let Some(s) = self.subtitle.as_ref() {
            let sub_end = s.pts + s.duration;
            if sub_end < self.frame_pts {
                self.subtitle.take();
            } else {
                ui.add(s);
            }
        }
    }

    fn render_debug(&mut self, ui: &mut Ui, frame_response: &Response) {
        let painter = ui.painter();

        const PADDING: f32 = 5.0;
        let vec_padding = vec2(PADDING, PADDING);
        let job = self.debug_inner(frame_response.rect);
        let galley = painter.layout_job(job);
        let mut bg_pos = galley
            .rect
            .translate(frame_response.rect.min.to_vec2() + vec_padding);
        bg_pos.max += vec_padding * 2.0;
        painter.rect_filled(bg_pos, PADDING, Color32::from_black_alpha(150));
        painter.galley(bg_pos.min + vec_padding, galley, Color32::PLACEHOLDER);
    }

    fn pts_to_sec(&self, pts: i64) -> f64 {
        let tbn = self.media_player.tbn();
        (pts as f64 / tbn.den as f64) * tbn.num as f64
    }

    fn pts_to_duration(&self, pts: i64) -> Duration {
        Duration::from_secs_f64(self.pts_to_sec(pts))
    }

    fn show_osd(&mut self, msg: &str) {
        self.osd = Some(msg.to_string());
        self.osd_end = Instant::now() + Duration::from_secs(2);
    }

    fn debug_inner(&mut self, frame_response: Rect) -> LayoutJob {
        let v_pts = self.elapsed();
        let a_pts = self
            .audio
            .as_ref()
            .map(|a| a.buffer.lock().map(|b| b.audio_pos).unwrap_or(0.0))
            .unwrap_or(0.0);
        let font = TextFormat::simple(FontId::monospace(11.), Color32::WHITE);

        let mut layout = LayoutJob::default();
        let buffer = self.pts_to_sec(self.media_player.buffer_size());
        layout.append(
            &format!(
                "sync: v:{:.3}s, a:{:.3}s, a-sync:{:.3}s, buffer: {:.3}s",
                v_pts,
                a_pts,
                a_pts - v_pts,
                buffer
            ),
            0.0,
            font.clone(),
        );

        let video_size = self.video_frame_size(frame_response);
        layout.append(
            &format!(
                "\nplayback: {:.2} fps ({:.2}x), volume={:.0}%, resolution={}x{}",
                self.avg_fps,
                self.avg_fps / self.framerate(),
                100.0 * self.volume(),
                video_size.x,
                video_size.y
            ),
            0.0,
            font.clone(),
        );

        if let Some(info) = self.info.as_ref() {
            let bitrate_str = if info.bitrate > 1_000_000 {
                format!("{:.1}M", info.bitrate as f32 / 1_000_000.0)
            } else if info.bitrate > 1_000 {
                format!("{:.1}k", info.bitrate as f32 / 1_000.0)
            } else {
                info.bitrate.to_string()
            };

            layout.append(
                &format!(
                    "\nduration: {}, bitrate: {}",
                    format_time(info.duration),
                    bitrate_str
                ),
                0.0,
                font.clone(),
            );

            fn print_chan(
                layout: &mut LayoutJob,
                font: TextFormat,
                chan: Option<&StreamInfo>,
                decoders: &Vec<DecoderInfo>,
            ) {
                if let Some(c) = chan {
                    layout.append(&format!("\n{}", c), 0.0, font.clone());
                    if let Some(decoder) = decoders.iter().find(|d| d.index == c.index) {
                        layout.append(&format!("\n\tdecoder={}", decoder.codec), 0.0, font.clone());
                    }
                }
            }
            print_chan(&mut layout, font.clone(), info.best_video(), &self.decoders);
            print_chan(&mut layout, font.clone(), info.best_audio(), &self.decoders);
            print_chan(
                &mut layout,
                font.clone(),
                info.best_subtitle(),
                &self.decoders,
            );
        }

        layout
    }

    fn open_default_audio_stream(
        volume: Arc<AtomicU16>,
        state: Arc<AtomicU8>,
        rx: Receiver<AudioSamplesData>,
    ) -> Option<PlayerAudioStream> {
        if let Ok(a) = AudioDevice::new() {
            if let Ok(cfg) = a.0.default_output_config() {
                info!(
                    "Default audio device config: {}Hz, {}ch, {:?}",
                    cfg.sample_rate().0,
                    cfg.channels(),
                    cfg.sample_format()
                );
                let buffer = Arc::new(Mutex::new(AudioBuffer::new(
                    cfg.sample_rate().0,
                    cfg.channels() as u8,
                )));
                let b_clone = buffer.clone();
                let i_state = state.clone();
                if let Ok(stream) = a.0.build_output_stream_raw(
                    &cfg.config(),
                    SampleFormat::F32,
                    move |data: &mut cpal::Data, info: &cpal::OutputCallbackInfo| {
                        let dst: &mut [f32] = data.as_slice_mut().unwrap();
                        dst.fill(0.0);
                        loop {
                            let state = i_state.load(Ordering::Relaxed);
                            if state == PlayerState::Stopped as u8 {
                                break;
                            }
                            if let Ok(mut buf) = b_clone.lock() {
                                // take samples from channel
                                while let Ok(m) = rx.try_recv() {
                                    buf.add_samples(m.samples);
                                }
                                // do nothing, leave silence in dst
                                if state == PlayerState::Paused as u8 {
                                    break;
                                }
                                // data didn't start yet just leave silence
                                if buf.video_pos == 0.0 {
                                    break;
                                }
                                // not enough samples, block thread
                                if buf.samples.len() < dst.len() {
                                    drop(buf);
                                    std::thread::sleep(Duration::from_millis(5));
                                    debug!("Audio: buffer underrun, playback is too slow!");
                                    continue;
                                }
                                // lazy audio sync
                                let sync = buf.video_pos - buf.audio_pos;
                                if sync > 0.1 {
                                    //let drop_samples = buf.audio_delay_samples() as usize;
                                    //buf.take_samples(drop_samples);
                                    //debug!("Audio: dropping {drop_samples} samples, playback is too fast!");
                                }
                                let v = volume.load(Ordering::Relaxed) as f32 / u16::MAX as f32;
                                let w_len = dst.len().min(buf.samples.len());
                                let mut i = 0;
                                for s in buf.take_samples(w_len) {
                                    dst[i] = s * v;
                                    i += 1;
                                }
                                break;
                            }
                        }
                    },
                    move |e| {
                        panic!("{}", e);
                    },
                    None,
                ) {
                    return Some(PlayerAudioStream {
                        device: a,
                        config: cfg,
                        stream,
                        player_state: state.clone(),
                        buffer,
                    });
                }
            }
        }

        None
    }

    /// Create a new [`CustomPlayer`].
    pub fn new(overlay: T, ctx: &egui::Context, input_path: &String) -> Self {
        // volume arc
        let vol = Arc::new(AtomicU16::new(u16::MAX));
        let state = Arc::new(AtomicU8::new(PlayerState::Stopped as u8));
        let playback_speed = Arc::new(AtomicI16::new(100));

        // Open audio device
        let (tx, rx) = mpsc::channel();
        let audio = Self::open_default_audio_stream(vol.clone(), state.clone(), rx);

        let mut media_player = MediaPlayer::new(input_path);
        if let Some(a) = &audio {
            media_player = media_player.with_audio_chan(tx);
            media_player.set_target_sample_rate(a.config.sample_rate().0);
        }

        let init_size = ctx.available_rect();
        Self {
            state,
            playback_speed,
            overlay,
            key_binds: false,
            input_path: input_path.clone(),
            looping: true,
            volume: vol,
            frame: ctx.load_texture(
                "video_frame",
                ColorImage::new(
                    [init_size.width() as usize, init_size.height() as usize],
                    Color32::BLACK,
                ),
                Default::default(),
            ),
            frame_start: Instant::now(),
            frame_pts: 0,
            frame_duration: 0,
            info: None,
            decoders: vec![],
            ctx: ctx.clone(),
            audio,
            subtitle: None,
            media_player,
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
        }
    }
}

impl<T> PlayerControls for CustomPlayer<T> {
    /// The elapsed duration of the stream in seconds
    fn elapsed(&self) -> f32 {
        self.pts_to_sec(self.frame_pts) as f32
    }

    fn duration(&self) -> f32 {
        self.info.as_ref().map_or(0.0, |info| info.duration)
    }

    fn framerate(&self) -> f32 {
        self.info
            .as_ref()
            .and_then(|i| i.best_video())
            .map_or(0.0, |i| i.fps)
    }

    fn size(&self) -> (u16, u16) {
        self.info
            .as_ref()
            .and_then(|i| i.best_video())
            .map_or((0, 0), |i| (i.width as u16, i.height as u16))
    }

    fn state(&self) -> PlayerState {
        match self.state.load(Ordering::Relaxed) {
            0 => PlayerState::Stopped,
            1 => PlayerState::Seeking,
            2 => PlayerState::Paused,
            3 => PlayerState::Playing,
            _ => panic!("Unexpected state"),
        }
    }

    /// Pause the stream.
    fn pause(&mut self) {
        self.state
            .store(PlayerState::Paused as u8, Ordering::Relaxed);
        self.show_osd("Pause");
    }

    /// Start the stream.
    fn start(&mut self) {
        self.state
            .store(PlayerState::Playing as u8, Ordering::Relaxed);
        self.show_osd("Play");
        if self.media_player.start() {
            self.frame_counter = 0;
            self.frame_start = Instant::now();
            self.frame_duration = 0;
            self.frame_pts = 0;
            self.last_frame_counter = 0;
            self.info = None;
            self.error = None;
            self.avg_fps = 0.0;
            self.avg_fps_start = Instant::now();
            self.subtitle = None;
            if let Some(q) = self.audio.as_ref() {
                if let Ok(mut q) = q.buffer.lock() {
                    q.audio_pos = 0.0;
                    q.video_pos = 0.0;
                }
            }
        }
    }

    /// Stop the stream.
    fn stop(&mut self) {
        self.state
            .store(PlayerState::Stopped as u8, Ordering::Relaxed);
        self.show_osd("Stop");
    }

    /// Seek to a location in the stream.
    fn seek(&mut self, pos: f32) {
        self.media_player.seek_to(pos);
    }

    fn volume(&self) -> f32 {
        self.volume.load(Ordering::Relaxed) as f32 / u16::MAX as f32
    }

    fn set_volume(&mut self, volume: f32) {
        let new_volume = (u16::MAX as f32 * volume) as u16;
        self.volume.store(new_volume, Ordering::Relaxed);
        self.show_osd(&format!("Volume: {:.0}%", 100.0 * volume));
    }

    fn looping(&self) -> bool {
        self.looping
    }

    fn set_looping(&mut self, looping: bool) {
        self.looping = looping;
        self.show_osd(if looping { "Loop: yes" } else { "Loop: no" });
    }

    fn debug(&self) -> bool {
        self.debug
    }

    fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
        self.show_osd(if debug { "Debug: yes" } else { "Debug: no" });
    }

    fn playback_speed(&self) -> f32 {
        self.playback_speed.load(Ordering::Relaxed) as f32 / 100.0
    }

    fn set_playback_speed(&mut self, speed: f32) {
        let new_speed = (100.0 * speed.max(0.01)).round() as i16;
        self.playback_speed.store(new_speed, Ordering::Relaxed);
        self.show_osd(&format!("Speed: {:.2}x", speed));
    }

    fn maintain_aspect(&self) -> bool {
        self.maintain_aspect
    }

    fn set_maintain_aspect(&mut self, maintain_aspect: bool) {
        self.maintain_aspect = maintain_aspect;
    }

    fn fullscreen(&self) -> bool {
        self.fullscreen
    }

    fn set_fullscreen(&mut self, fullscreen: bool) {
        self.fullscreen = fullscreen;
    }

    fn key_binds(&self) -> bool {
        self.key_binds
    }

    fn set_key_binds(&mut self, key_binds: bool) {
        self.key_binds = key_binds;
    }
}

impl<T> Widget for &mut CustomPlayer<T>
where
    T: PlayerOverlay,
{
    fn ui(self, ui: &mut Ui) -> Response {
        self.render(ui)
    }
}

impl<T> CustomPlayer<T>
where
    T: PlayerOverlay,
{
    /// Render player in available space
    pub fn render(&mut self, ui: &mut Ui) -> Response {
        let size = ui.available_size();

        self.handle_keys(ui);
        self.process_state(size);
        let frame_response = self.render_frame(ui);
        self.render_subtitles(ui);
        self.render_overlay(ui, &frame_response);
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
        if self.debug {
            self.render_debug(ui, &frame_response);
        }
        frame_response
    }

    fn render_overlay(&mut self, ui: &mut Ui, frame: &Response) {
        let (tx, rx) = mpsc::channel();
        let mut state = PlayerOverlayState {
            elapsed: self.elapsed(),
            duration: self.duration(),
            framerate: self.framerate(),
            playback_speed: self.playback_speed(),
            size: self.size(),
            volume: self.volume(),
            looping: self.looping,
            debug: self.debug,
            state: self.state(),
            maintain_aspect: self.maintain_aspect,
            fullscreen: self.fullscreen,
            key_binds: self.key_binds,
            inbox: tx,
        };
        self.overlay.show(ui, frame, &mut state);

        // drain inbox
        while let Ok(m) = rx.try_recv() {
            self.process_player_message(m);
        }
    }
}

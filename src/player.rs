use crate::media_player::{DecoderMessage, MediaPlayer};
use crate::subtitle::Subtitle;
use crate::AudioDevice;
use cpal::traits::DeviceTrait;
use cpal::{SampleFormat, Stream, SupportedStreamConfig};
use egui::load::SizedTexture;
use egui::text::LayoutJob;
use egui::{
    pos2, vec2, Align2, Color32, ColorImage, FontId, Image, Rect, Response, Sense, TextFormat,
    TextureHandle, TextureOptions, Ui, Vec2, Widget,
};
use egui_inbox::{UiInbox, UiInboxSender};
use ffmpeg_rs_raw::ffmpeg_sys_the_third::AVMediaType;
use ffmpeg_rs_raw::DemuxerInfo;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[derive(Debug)]
/// IPC for player
enum PlayerMessage {
    /// Set player state
    SetState(PlayerState),
    /// Seek to position in seconds
    Seek(f32),
    /// Set player volume
    SetVolume(u8),
    /// Set player looping
    SetLooping(bool),
    /// Set debug overlay
    SetDebug(bool),
    /// Select playing stream
    SelectStream(AVMediaType, usize),
    /// Set playback speed
    SetPlaybackSpeed(f32),
}

/// The [`CustomPlayer`] processes and controls streams of video/audio.
/// This is what you use to show a video file.
/// Initialize once, and use the [`CustomPlayer::ui`] or [`CustomPlayer::ui_at()`] functions to show the playback.
pub struct CustomPlayer<T> {
    exiting: Arc<AtomicBool>,
    overlay: T,
    player_state: PlayerState,
    looping: bool,
    volume: Arc<AtomicU8>,
    playback_speed: Arc<AtomicU8>,
    debug: bool,

    avg_fps: f32,
    avg_fps_start: Instant,
    last_frame_counter: u64,
    pts_audio: i64,

    /// The video frame to display
    frame: TextureHandle,
    /// Video frame PTS
    frame_pts: i64,
    /// Duration to show [frame]
    frame_duration: i64,
    /// Realtime of when [frame] was shown
    frame_start: Instant,
    /// When playback was started
    play_start: Instant,
    /// How many frames have been rendered so far
    frame_counter: u64,

    frame_timer: Option<JoinHandle<()>>,

    /// Stream info
    info: Option<DemuxerInfo>,

    ctx: egui::Context,
    input_path: String,
    audio: Option<PlayerAudioStream>,
    subtitle: Option<Subtitle>,

    /// ffmpeg media player
    media_player: MediaPlayer,

    /// An error which prevented playback
    error: Option<String>,
}

struct PlayerAudioStream {
    pub device: AudioDevice,
    pub config: SupportedStreamConfig,
    pub stream: Stream,
    pub buffer: Arc<Mutex<AudioBuffer>>,
}

struct AudioBuffer {
    channels: u8,
    sample_rate: u32,
    paused: bool,
    // buffered samples
    samples: VecDeque<f32>,
    /// Video position in seconds
    video_pos: f32,
    /// Audio position in seconds
    audio_pos: f32,
}

impl AudioBuffer {
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        Self {
            channels,
            sample_rate,
            paused: false,
            samples: VecDeque::new(),
            video_pos: 0.0,
            audio_pos: 0.0,
        }
    }

    pub fn add_samples(&mut self, samples: Vec<f32>) {
        self.samples.extend(samples);
    }

    pub fn take_samples(&mut self, n: usize) -> Vec<f32> {
        assert_eq!(0, n % self.channels as usize, "Must be a multiple of 2");
        self.audio_pos += (n as f32 / self.sample_rate as f32 / self.channels as f32);
        self.samples.drain(..n).collect()
    }

    pub fn set_video_pos(&mut self, pos: f32) {
        self.video_pos = pos;
    }

    pub fn audio_delay_secs(&self) -> f32 {
        self.video_pos - self.audio_pos
    }

    pub fn audio_delay_samples(&self) -> isize {
        let samples_f32 = self.sample_rate as f32 * self.audio_delay_secs();
        samples_f32 as isize * self.channels as isize
    }

    pub fn pause(&mut self, paused: bool) {
        self.paused = paused;
    }
}

/// The possible states of a [`CustomPlayer`].
#[repr(u8)]
#[derive(PartialEq, Clone, Copy, Debug)]
pub enum PlayerState {
    /// No playback.
    Stopped,
    /// Streams have reached the end of the file.
    EndOfFile,
    /// Stream is seeking. Inner bool represents whether the seek is currently in progress.
    Seeking(bool),
    /// Playback is paused.
    Paused,
    /// Playback is ongoing.
    Playing,
    /// Playback is scheduled to restart.
    Restarting,
}

pub trait PlayerControls {
    fn elapsed(&self) -> f32;
    fn duration(&self) -> f32;
    fn framerate(&self) -> f32;
    fn size(&self) -> (u16, u16);
    fn state(&self) -> PlayerState;
    fn pause(&mut self);
    fn start(&mut self);
    fn stop(&mut self);
    fn seek(&mut self, seek_frac: f32);
    fn set_volume(&mut self, volume: u8);
    fn volume(&self) -> u8;
    fn volume_f32(&self) -> f32;
    fn set_volume_f32(&mut self, volume: f32);
    fn looping(&self) -> bool;
    fn set_looping(&mut self, looping: bool);
    fn debug(&self) -> bool;
    fn set_debug(&mut self, debug: bool);
    fn playback_speed(&self) -> f32;
    fn set_playback_speed(&mut self, speed: f32);
}

/// Wrapper to store player info and pass to overlay impl
pub struct PlayerOverlayState {
    elapsed: f32,
    duration: f32,
    framerate: f32,
    playback_speed: f32,
    size: (u16, u16),
    volume: u8,
    looping: bool,
    debug: bool,
    state: PlayerState,
    inbox: UiInboxSender<PlayerMessage>,
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

    fn set_volume(&mut self, volume: u8) {
        self.inbox.send(PlayerMessage::SetVolume(volume)).unwrap();
    }

    fn volume(&self) -> u8 {
        self.volume
    }

    fn volume_f32(&self) -> f32 {
        self.volume as f32 / u8::MAX as f32
    }

    fn set_volume_f32(&mut self, volume: f32) {
        self.inbox
            .send(PlayerMessage::SetVolume((u8::MAX as f32 * volume) as u8))
            .unwrap();
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
}

pub trait PlayerOverlay {
    fn show(&self, ui: &mut Ui, frame: &Response, state: &mut PlayerOverlayState);
}

impl<T> Drop for CustomPlayer<T> {
    fn drop(&mut self) {
        self.exiting.store(true, Ordering::Relaxed);
        if let Some(j) = self.frame_timer.take() {
            j.join().unwrap();
        }
    }
}

impl<T> CustomPlayer<T> {
    /// Store the next image
    fn load_frame(&mut self, image: ColorImage, pts: i64, duration: i64) {
        self.frame.set(image, TextureOptions::default());
        self.frame_pts = pts;
        self.frame_duration = (duration as f32 * self.playback_speed()) as i64;
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

    /// Check if the current frame should be flipped
    fn check_load_frame(&mut self) -> bool {
        if self.player_state != PlayerState::Playing {
            // force frame to start now, while paused
            self.frame_start = Instant::now();
            return false;
        }
        let now = Instant::now();
        now >= (self.frame_start + self.pts_to_duration(self.frame_duration))
    }

    fn next_frame_max_pts(&self) -> i64 {
        self.frame_pts + self.frame_duration + self.frame_duration
    }

    fn process_state(&mut self, size: Vec2) {
        self.media_player.set_target_size(size);

        // check if we should load the next video frame
        if !self.check_load_frame() {
            return;
        }

        // reset avg fps timer every 0.5s
        if self.frame_counter != 0
            && self.frame_counter % (self.framerate() / 2.0).max(1.0) as u64 == 0
        {
            let n_frames = self.frame_counter - self.last_frame_counter;
            self.avg_fps = n_frames as f32 / (Instant::now() - self.avg_fps_start).as_secs_f32();
            self.avg_fps_start = Instant::now();
            self.last_frame_counter = self.frame_counter;
        }

        while let Some(msg) = self.media_player.next() {
            match msg {
                DecoderMessage::MediaInfo(i) => {
                    self.info = Some(i);
                    let fps = self.framerate();
                    if fps > 0.0 {
                        let cx = self.ctx.clone();
                        let closing = self.exiting.clone();
                        self.frame_timer = Some(std::thread::spawn(move || loop {
                            if closing.load(Ordering::Relaxed) {
                                break;
                            }
                            cx.request_repaint();
                            std::thread::sleep(Duration::from_secs_f32(0.5 / fps));
                        }));
                    }
                }
                DecoderMessage::VideoFrame(pts, duration, f) => {
                    if self.frame_counter == 0 {
                        self.play_start = Instant::now();
                    }

                    self.load_frame(f, pts, duration);

                    let v_pts = self.pts_to_sec(self.frame_pts);
                    if let Some(a) = self.audio.as_mut() {
                        if let Ok(mut q) = a.buffer.lock() {
                            q.set_video_pos(v_pts as f32);
                        }
                    }
                    // break on video frame
                    // once we load the next frame this loop will not call again until
                    // this frame is over (pts + duration)
                    break;
                }
                DecoderMessage::AudioSamples(pts, duration, s) => {
                    if let Some(a) = self.audio.as_mut() {
                        if let Ok(mut q) = a.buffer.lock() {
                            q.add_samples(s);
                        }
                    }
                    self.pts_audio = pts;
                }
                DecoderMessage::Subtitles(pts, duration, text) => {
                    self.subtitle = Some(Subtitle::from_text(&text));
                }
                DecoderMessage::Error(e) => {
                    self.error = Some(e);
                    self.stop();
                }
            }
        }
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
        }
    }

    fn generate_frame_image(&self, size: Vec2) -> Image {
        Image::new(SizedTexture::new(self.frame.id(), size)).sense(Sense::click())
    }

    fn render_frame(&self, ui: &mut Ui) -> Response {
        self.render_frame_at(ui, ui.available_rect_before_wrap())
    }

    fn render_frame_at(&self, ui: &mut Ui, rect: Rect) -> Response {
        ui.put(rect, self.generate_frame_image(rect.size()))
    }

    fn render_subtitles(&mut self, ui: &mut Ui, frame_response: &Response) {
        if let Some(s) = self.subtitle.as_ref() {
            ui.painter().text(
                frame_response.rect.min
                    + vec2(
                        frame_response.rect.width() / 2.0,
                        frame_response.rect.height() - 40.,
                    ),
                Align2::CENTER_BOTTOM,
                &s.text,
                FontId::proportional(16.),
                Color32::WHITE,
            );
        }
    }

    fn render_debug(&mut self, ui: &mut Ui, frame_response: &Response) {
        let painter = ui.painter();

        const PADDING: f32 = 5.0;
        let vec_padding = vec2(PADDING, PADDING);
        let job = self.debug_inner();
        let galley = painter.layout_job(job);
        let mut bg_pos = galley
            .rect
            .translate(frame_response.rect.min.to_vec2() + vec_padding);
        bg_pos.max += vec_padding * 2.0;
        painter.rect_filled(bg_pos, PADDING, Color32::from_black_alpha(150));
        painter.galley(bg_pos.min + vec_padding, galley, Color32::PLACEHOLDER);
    }

    fn pts_to_sec(&self, pts: i64) -> f64 {
        (pts as f64 / self.media_player.tbn.den as f64) * self.media_player.tbn.num as f64
    }

    fn pts_to_duration(&self, pts: i64) -> Duration {
        Duration::from_secs_f64(self.pts_to_sec(pts))
    }

    fn debug_inner(&mut self) -> LayoutJob {
        let v_pts = self.elapsed();
        let a_pts = if let Some(a) = self.audio.as_ref() {
            a.buffer.lock().map_or(0.0, |a| a.audio_pos)
        } else {
            0.0
        };
        let font = TextFormat::simple(FontId::monospace(11.), Color32::WHITE);

        let mut layout = LayoutJob::default();
        layout.append(
            &format!(
                "sync: v:{:.3}s, a:{:.3}s, a-sync:{:.3}s",
                v_pts,
                a_pts,
                a_pts - v_pts
            ),
            0.0,
            font.clone(),
        );

        layout.append(
            &format!(
                "\nplayback: {:.2} fps ({:.2}x)",
                self.avg_fps,
                self.avg_fps / self.framerate()
            ),
            0.0,
            font.clone(),
        );

        let buffer = self.pts_to_sec(self.media_player.buffer_size());
        layout.append(&format!("\nbuffer: {:.3}s", buffer), 0.0, font.clone());

        let bv = self.info.as_ref().and_then(|i| i.best_video());
        if let Some(bv) = bv {
            layout.append(&format!("\n{}", bv), 0.0, font.clone());
        }

        let ba = self.info.as_ref().and_then(|i| i.best_audio());
        if let Some(ba) = ba {
            layout.append(&format!("\n{}", ba), 0.0, font.clone());
        }

        let bs = self.info.as_ref().and_then(|i| i.best_subtitle());
        if let Some(bs) = bs {
            layout.append(&format!("\n{}", bs), 0.0, font.clone());
        }

        layout
    }

    fn open_default_audio_stream(
        volume: Arc<AtomicU8>,
        exit: Arc<AtomicBool>,
        rx: Receiver<DecoderMessage>,
    ) -> Option<PlayerAudioStream> {
        if let Ok(a) = AudioDevice::new() {
            if let Ok(cfg) = a.0.default_output_config() {
                let buffer = Arc::new(Mutex::new(AudioBuffer::new(
                    cfg.sample_rate().0,
                    cfg.channels() as u8,
                )));
                let b_clone = buffer.clone();
                if let Ok(stream) = a.0.build_output_stream_raw(
                    &cfg.config(),
                    SampleFormat::F32,
                    move |data: &mut cpal::Data, info: &cpal::OutputCallbackInfo| {
                        let mut dst: &mut [f32] = data.as_slice_mut().unwrap();
                        dst.fill(0.0);
                        loop {
                            if exit.load(Ordering::Relaxed) {
                                break;
                            }
                            if let Ok(mut buf) = b_clone.lock() {
                                // take samples from channel
                                while let Ok(m) = rx.try_recv() {
                                    match m {
                                        DecoderMessage::AudioSamples(pts, duration, samples) => {
                                            buf.add_samples(samples)
                                        }
                                        _ => panic!("Unexpected message in audio channel"),
                                    }
                                }
                                // do nothing, leave silence in dst
                                if buf.paused {
                                    return;
                                }
                                // data didn't start yet just leave silence
                                if buf.video_pos == 0.0 {
                                    break;
                                }
                                // not enough samples, block thread
                                if buf.samples.len() < dst.len() {
                                    drop(buf);
                                    std::thread::sleep(Duration::from_millis(5));
                                    continue;
                                }
                                // lazy audio sync
                                let sync = buf.audio_delay_secs();
                                if sync > 0.01 {
                                    let drop_samples = buf.audio_delay_samples() as usize;
                                    buf.take_samples(drop_samples);
                                }
                                let v = volume.load(Ordering::Relaxed) as f32 / u8::MAX as f32;
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
                        buffer,
                    });
                }
            }
        }

        None
    }

    /// Create a new [`CustomPlayer`].
    pub fn new(overlay: T, ctx: &egui::Context, input_path: &String) -> Self {
        /// volume arc
        let vol = Arc::new(AtomicU8::new(200));
        /// exit signal (on drop)
        let exit = Arc::new(AtomicBool::new(false));
        let playback_speed = Arc::new(AtomicU8::new(127));

        /// Open audio device
        let (tx, rx) = mpsc::channel();
        let audio = Self::open_default_audio_stream(vol.clone(), exit.clone(), rx);

        let mut media_player = MediaPlayer::new(input_path).with_audio_chan(tx);
        if let Some(a) = &audio {
            media_player.set_target_sample_rate(a.config.sample_rate().0);
        }

        let init_size = ctx.available_rect();
        Self {
            exiting: exit,
            playback_speed,
            overlay,
            input_path: input_path.clone(),
            looping: false,
            volume: vol,
            player_state: PlayerState::Stopped,
            pts_audio: 0,
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
            frame_timer: None,
            info: None,
            ctx: ctx.clone(),
            audio,
            subtitle: None,
            media_player,
            debug: false,
            avg_fps: 0.0,
            avg_fps_start: Instant::now(),
            frame_counter: 0,
            last_frame_counter: 0,
            play_start: Instant::now(),
            error: None,
        }
    }
}

impl<T> PlayerControls for CustomPlayer<T> {
    /// The elapsed duration of the stream in seconds
    fn elapsed(&self) -> f32 {
        self.frame_counter as f32 / self.framerate()
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
        self.player_state
    }

    /// Pause the stream.
    fn pause(&mut self) {
        self.player_state = PlayerState::Paused;
        self.audio
            .as_ref()
            .map(|b| b.buffer.lock().map(|mut c| c.paused = true));
    }

    /// Start the stream.
    fn start(&mut self) {
        self.media_player.start();
        self.player_state = PlayerState::Playing;
        self.play_start = Instant::now();
        self.audio
            .as_ref()
            .map(|b| b.buffer.lock().map(|mut c| c.paused = false));
    }

    /// Stop the stream.
    fn stop(&mut self) {
        self.player_state = PlayerState::Stopped;
        self.audio
            .as_ref()
            .map(|b| b.buffer.lock().map(|mut c| c.paused = true));
    }

    /// Seek to a location in the stream.
    fn seek(&mut self, seek_frac: f32) {
        // TODO: set target PTS
    }

    fn set_volume(&mut self, volume: u8) {
        self.volume.store(volume, Ordering::Relaxed);
    }

    fn volume(&self) -> u8 {
        self.volume.load(Ordering::Relaxed)
    }

    fn volume_f32(&self) -> f32 {
        self.volume() as f32
    }

    fn set_volume_f32(&mut self, volume: f32) {
        let new_volume = (u8::MAX as f32 * volume) as u8;
        self.set_volume(new_volume);
    }

    fn looping(&self) -> bool {
        self.looping
    }

    fn set_looping(&mut self, looping: bool) {
        self.looping = looping;
    }

    fn debug(&self) -> bool {
        self.debug
    }

    fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    fn playback_speed(&self) -> f32 {
        127.0 / self.playback_speed.load(Ordering::Relaxed) as f32
    }

    fn set_playback_speed(&mut self, speed: f32) {
        self.playback_speed
            .store((u8::MAX as f32 * speed) as u8, Ordering::Relaxed);
    }
}

impl<T> Widget for &mut CustomPlayer<T>
where
    T: PlayerOverlay,
{
    fn ui(self, ui: &mut Ui) -> Response {
        let size = ui.available_size();

        self.process_state(size);
        let frame_response = self.render_frame(ui);
        self.render_subtitles(ui, &frame_response);
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
        if self.debug {
            self.render_debug(ui, &frame_response);
        }
        frame_response
    }
}

impl<T> CustomPlayer<T>
where
    T: PlayerOverlay,
{
    fn render_overlay(&mut self, ui: &mut Ui, frame: &Response) {
        let inbox = UiInbox::new();
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
            inbox: inbox.sender(),
        };
        self.overlay.show(ui, frame, &mut state);

        // drain inbox
        let r = inbox.read(ui);
        for m in r {
            self.process_player_message(m);
        }
    }
}

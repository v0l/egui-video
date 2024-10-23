use crate::AudioDevice;
use cpal::traits::DeviceTrait;
use cpal::{Stream, SupportedStreamConfig};
use egui::load::SizedTexture;
use egui::{
    vec2, Align2, Color32, ColorImage, FontId, Image, Rect, Response, Sense, TextFormat,
    TextureHandle, Ui, Vec2, Widget,
};

use crate::ffmpeg::MediaPlayer;
use crate::subtitle::Subtitle;
use egui::text::LayoutJob;
use egui_inbox::{UiInbox, UiInboxSender};
use ffmpeg_rs_raw::ffmpeg_sys_the_third::AVMediaType;
use ffmpeg_rs_raw::DemuxerInfo;
use ringbuf::consumer::Consumer;
use ringbuf::producer::Producer;
use ringbuf::storage::Heap;
use ringbuf::traits::Split;
use ringbuf::{CachingProd, HeapRb, SharedRb};
use std::sync::atomic::{AtomicI64, AtomicU8, Ordering};
use std::sync::Arc;

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
}

#[derive(Debug)]
/// Messages received from the decoder
pub enum DecoderMessage {
    MediaInfo(DemuxerInfo),
    /// Video frame from the decoder
    VideoFrame(i64, i64, ColorImage),
    /// Audio samples from the decoder
    AudioSamples(i64, i64, Vec<f32>),
    Subtitles(i64, i64, String),
}

impl DecoderMessage {
    pub fn pts(&self) -> i64 {
        match self {
            DecoderMessage::AudioSamples(pts, _, _) => *pts,
            DecoderMessage::VideoFrame(pts, _, _) => *pts,
            _ => 0,
        }
    }
}

impl PartialEq<Self> for DecoderMessage {
    fn eq(&self, other: &Self) -> bool {
        other.cmp(self).is_eq()
    }
}

impl Eq for DecoderMessage {}

impl Ord for DecoderMessage {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let pts_a = match self {
            DecoderMessage::AudioSamples(pts, _, _) => *pts,
            DecoderMessage::VideoFrame(pts, _, _) => *pts,
            _ => 0,
        };
        let pts_b = match other {
            DecoderMessage::AudioSamples(pts, _, _) => *pts,
            DecoderMessage::VideoFrame(pts, _, _) => *pts,
            _ => 0,
        };
        pts_b.cmp(&pts_a)
    }
}

impl PartialOrd for DecoderMessage {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The [`CustomPlayer`] processes and controls streams of video/audio.
/// This is what you use to show a video file.
/// Initialize once, and use the [`CustomPlayer::ui`] or [`CustomPlayer::ui_at()`] functions to show the playback.
pub struct CustomPlayer<T>
where
    T: PlayerOverlay,
{
    overlay: T,
    player_state: PlayerState,
    texture_handle: TextureHandle,
    looping: bool,
    volume: Arc<AtomicU8>,
    debug: bool,

    /// Next PTS
    pts_video: Arc<AtomicI64>,
    pts_audio: Arc<AtomicI64>,
    /// Stream info
    info: Option<DemuxerInfo>,

    ctx: egui::Context,
    input_path: String,
    audio: Option<PlayerAudioStream>,
    subtitle: Option<Subtitle>,

    /// ffmpeg media player
    media_player: MediaPlayer,
}

struct PlayerAudioStream {
    pub device: AudioDevice,
    pub config: SupportedStreamConfig,
    pub stream: Stream,
    pub tx: CachingProd<Arc<SharedRb<Heap<f32>>>>,
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
}

/// Wrapper to store player info and pass to overlay impl
pub struct PlayerOverlayState {
    elapsed: f32,
    duration: f32,
    framerate: f32,
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
}

pub trait PlayerOverlay {
    fn show(&self, ui: &mut Ui, frame: &Response, state: &mut PlayerOverlayState);
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

    fn process_state(&mut self, size: Vec2) {
        while let Some(msg) = self.media_player.next(size) {
            match msg {
                DecoderMessage::MediaInfo(i) => {
                    println!("{}", &i);
                    self.info = Some(i);
                }
                DecoderMessage::VideoFrame(pts, duration, f) => {
                    self.texture_handle.set(f, Default::default());
                    self.pts_video.store(pts, Ordering::Relaxed);
                }
                DecoderMessage::AudioSamples(pts, duration, s) => {
                    if let Some(a) = self.audio.as_mut() {
                        a.tx.push_slice(&s);
                    }
                    self.pts_audio.store(pts, Ordering::Relaxed);
                }
                DecoderMessage::Subtitles(pts, duration, text) => {
                    self.subtitle = Some(Subtitle::from_text(&text))
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
        }
    }

    fn generate_frame_image(&self) -> Image {
        Image::new(SizedTexture::from_handle(&self.texture_handle)).sense(Sense::click())
    }

    fn render_frame(&self, ui: &mut Ui) -> Response {
        ui.add(self.generate_frame_image())
    }

    fn render_frame_at(&self, ui: &mut Ui, rect: Rect) -> Response {
        ui.put(rect, self.generate_frame_image())
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
        painter.rect_filled(
            bg_pos,
            PADDING,
            Color32::from_rgba_unmultiplied(0, 0, 0, 150),
        );
        painter.galley(bg_pos.min + vec_padding, galley, Color32::PLACEHOLDER);
    }

    fn pts_to_sec(&self, pts: i64) -> f32 {
        pts as f32 * (self.media_player.tbn.num as f32 / self.media_player.tbn.den as f32)
    }

    fn debug_inner(&mut self) -> LayoutJob {
        let v_pts = self.pts_to_sec(self.pts_video.load(Ordering::Relaxed));
        let a_pts = self.pts_to_sec(self.pts_audio.load(Ordering::Relaxed));
        let font = TextFormat::simple(FontId::monospace(11.), Color32::WHITE);

        let mut layout = LayoutJob::default();
        layout.append(
            &format!(
                "sync: v:{:.3}s, a:{:.3}s, a-sync:{:.3}s",
                v_pts,
                a_pts,
                v_pts - a_pts
            ),
            0.0,
            font.clone(),
        );

        let max_pts = self.pts_to_sec(self.media_player.pts_max());
        let buf_len = if max_pts != 0.0 { max_pts - v_pts } else { 0.0 };
        layout.append(&format!("\nbuffer: {:.3}s", buf_len), 0.0, font.clone());

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

    fn open_default_audio_stream(volume: Arc<AtomicU8>) -> Option<PlayerAudioStream> {
        if let Ok(a) = AudioDevice::new() {
            if let Ok(cfg) = a.0.default_output_config() {
                let audio_sample_buffer = HeapRb::<f32>::new(16384);
                let (tx, mut rx) = audio_sample_buffer.split();
                if let Ok(stream) = a.0.build_output_stream_raw(
                    &cfg.config(),
                    cfg.sample_format(),
                    move |data: &mut cpal::Data, info: &cpal::OutputCallbackInfo| {
                        let dst: &mut [f32] = data.as_slice_mut().unwrap();
                        dst.fill(0.0);
                        rx.pop_slice(dst);

                        // mul volume
                        let v = volume.load(Ordering::Relaxed) as f32 / u8::MAX as f32;
                        for s in dst {
                            *s *= v;
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
                        tx,
                    });
                }
            }
        }

        None
    }

    /// Create a new [`CustomPlayer`].
    pub fn new(overlay: T, ctx: &egui::Context, input_path: &String) -> Self {
        let texture_handle = ctx.load_texture(
            "video_frame",
            ColorImage::new([1, 1], Color32::BLACK),
            Default::default(),
        );

        /// volume arc
        let vol = Arc::new(AtomicU8::new(255));

        /// Open audio device
        let audio = Self::open_default_audio_stream(vol.clone());

        Self {
            overlay,
            input_path: input_path.clone(),
            texture_handle,
            looping: false,
            volume: vol,
            player_state: PlayerState::Stopped,
            pts_video: Arc::new(AtomicI64::new(0)),
            pts_audio: Arc::new(AtomicI64::new(0)),
            info: None,
            ctx: ctx.clone(),
            audio,
            subtitle: None,
            media_player: MediaPlayer::new(input_path, ctx),
            debug: false,
        }
    }
}

impl<T> PlayerControls for CustomPlayer<T>
where
    T: PlayerOverlay,
{
    /// The elapsed duration of the stream in seconds
    fn elapsed(&self) -> f32 {
        // timebase is always 90k
        self.pts_video.load(Ordering::Relaxed) as f32 * (1.0 / 90_000.0)
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
        self.media_player.set_paused(true);
    }

    /// Start the stream.
    fn start(&mut self) {
        self.media_player.start();
        self.player_state = PlayerState::Playing;
        self.media_player.set_paused(false);
    }

    /// Stop the stream.
    fn stop(&mut self) {
        self.player_state = PlayerState::Stopped;
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
        if self.debug {
            self.render_debug(ui, &frame_response);
        }
        frame_response
    }
}

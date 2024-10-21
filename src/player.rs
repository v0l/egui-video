use crate::AudioDevice;
use cpal::traits::DeviceTrait;
use cpal::{Stream, SupportedStreamConfig};
use egui::load::SizedTexture;
use egui::{vec2, Align2, Color32, ColorImage, FontId, Image, Rect, Response, Rounding, Sense, Shadow, Spinner, TextureHandle, TextureOptions, Ui, Vec2};

use crate::ffmpeg::{DemuxerInfo, MediaPlayer};
use ffmpeg_sys_the_third::{AVMediaType, AVRational};
use ringbuf::consumer::Consumer;
use ringbuf::producer::Producer;
use ringbuf::storage::Heap;
use ringbuf::traits::Split;
use ringbuf::{CachingProd, HeapRb, SharedRb};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;

/// IPC for player
enum PlayerMessage {
    /// Select playing stream
    SelectStream(AVMediaType, usize),
}

#[derive(Debug)]
/// Messages received from the decoder
pub enum DecoderMessage {
    MediaInfo(DemuxerInfo),
    /// Video frame from the decoder
    VideoFrame(ColorImage),
    /// Audio samples from the decoder
    AudioSamples(Vec<f32>),
}

/// Configurable aspects of a [`Player`].
pub struct PlayerOptions {
    /// Should the stream loop if it finishes?
    pub looping: AtomicBool,
    /// The volume of the audio stream.
    pub audio_volume: Arc<AtomicU8>,
    /// The maximum volume of the audio stream.
    pub max_audio_volume: Arc<AtomicU8>,
    /// The texture options for the displayed video frame.
    pub texture_options: TextureOptions,
}

impl Default for PlayerOptions {
    fn default() -> Self {
        Self {
            looping: AtomicBool::new(true),
            max_audio_volume: Arc::new(AtomicU8::new(u8::MAX)),
            audio_volume: Arc::new(AtomicU8::new(127)),
            texture_options: TextureOptions::default(),
        }
    }
}

/// The [`Player`] processes and controls streams of video/audio.
/// This is what you use to show a video file.
/// Initialize once, and use the [`Player::ui`] or [`Player::ui_at()`] functions to show the playback.
pub struct Player {
    player_state: PlayerState,
    texture_handle: TextureHandle,
    options: PlayerOptions,

    /// Current PTS
    pts: i64,
    /// Timebase of the video stream
    timebase: AVRational,

    ctx: egui::Context,
    input_path: String,
    audio: Option<PlayerAudioStream>,

    /// ffmpeg media player
    media_player: MediaPlayer,
}

struct PlayerAudioStream {
    pub device: AudioDevice,
    pub config: SupportedStreamConfig,
    pub stream: Stream,
    pub tx: CachingProd<Arc<SharedRb<Heap<f32>>>>,
}

/// The possible states of a [`Player`].
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


impl Player {
    /// The elapsed duration of the stream in seconds
    pub fn elapsed(&self) -> f32 {
        self.pts as f32 * (self.timebase.num as f32 / self.timebase.den as f32)
    }

    pub fn framerate(&self) -> f32 {
        0.0
    }

    pub fn size(&self) -> (u16, u16) {
        (0, 0)
    }

    pub fn state(&self) -> PlayerState {
        self.player_state
    }

    /// Pause the stream.
    pub fn pause(&mut self) {
        self.player_state = PlayerState::Paused;
    }

    /// Resume the stream from a paused state.
    pub fn resume(&mut self) {
        self.player_state = PlayerState::Playing;
    }

    /// Stop the stream.
    pub fn stop(&mut self) {
        self.player_state = PlayerState::Stopped;
    }

    /// Seek to a location in the stream.
    pub fn seek(&mut self, seek_frac: f32) {
        // TODO: set target PTS
    }

    /// Start the stream.
    pub fn start(&mut self) {
        self.media_player.start();
    }

    pub fn set_volume(&mut self, volume: u8) {
        let new_volume = volume.min(self.max_volume());
        self.options.audio_volume.store(new_volume, Ordering::Relaxed);
    }

    pub fn volume(&self) -> u8 {
        self.options.audio_volume.load(Ordering::Relaxed)
    }

    pub fn volume_f32(&self) -> f32 {
        self.volume() as f32 / self.max_volume() as f32
    }

    pub fn max_volume(&self) -> u8 {
        self.options.max_audio_volume.load(Ordering::Relaxed)
    }

    pub fn set_max_volume(&mut self, volume: u8) {
        self.options.max_audio_volume.store(volume, Ordering::Relaxed);
    }

    pub fn set_volume_f32(&mut self, volume: f32) {
        let new_volume = (u8::MAX as f32 * volume) as u8;
        self.set_volume(new_volume);
    }

    pub fn ui(&mut self, ui: &mut Ui, size: Vec2) -> Response {
        self.process_state(size);
        let frame_response = self.render_frame(ui);
        self.render_controls(ui, &frame_response);
        self.render_subtitles(ui, &frame_response);
        frame_response
    }

    pub fn ui_at(&mut self, ui: &mut Ui, rect: Rect) -> Response {
        self.process_state(rect.size());
        let frame_response = self.render_frame_at(ui, rect);
        self.render_controls(ui, &frame_response);
        self.render_subtitles(ui, &frame_response);
        frame_response
    }

    fn process_state(&mut self, size: Vec2) {
        for msg in self.media_player.next(&self.ctx, size) {
            match msg {
                DecoderMessage::VideoFrame(f) => {
                    self.texture_handle.set(f, self.options.texture_options);
                }
                DecoderMessage::AudioSamples(s) => {
                    if let Some(a) = self.audio.as_mut() {
                        a.tx.push_slice(&s);
                    }
                }
                _ => {}
            }
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
        // TODO(v0l): reimplement this
    }

    fn render_controls(&mut self, ui: &mut Ui, frame_response: &Response) {
        let hovered = ui.rect_contains_pointer(frame_response.rect);
        let currently_seeking = matches!(self.player_state, PlayerState::Seeking(_));
        let is_stopped = matches!(self.player_state, PlayerState::Stopped);
        let is_paused = matches!(self.player_state, PlayerState::Paused);
        let animation_time = 0.2;
        let seekbar_anim_frac = ui.ctx().animate_bool_with_time(
            frame_response.id.with("seekbar_anim"),
            hovered || currently_seeking || is_paused || is_stopped,
            animation_time,
        );

        if seekbar_anim_frac <= 0. {
            return;
        }

        let seekbar_width_offset = 20.;
        let fullseekbar_width = frame_response.rect.width() - seekbar_width_offset;

        let seekbar_width = fullseekbar_width * self.elapsed();

        let seekbar_offset = 20.;
        let seekbar_pos =
            frame_response.rect.left_bottom() + vec2(seekbar_width_offset / 2., -seekbar_offset);
        let seekbar_height = 3.;
        let mut fullseekbar_rect =
            Rect::from_min_size(seekbar_pos, vec2(fullseekbar_width, seekbar_height));

        let mut seekbar_rect =
            Rect::from_min_size(seekbar_pos, vec2(seekbar_width, seekbar_height));
        let seekbar_interact_rect = fullseekbar_rect.expand(10.);

        let seekbar_response = ui.interact(
            seekbar_interact_rect,
            frame_response.id.with("seekbar"),
            Sense::click_and_drag(),
        );

        let seekbar_hovered = seekbar_response.hovered();
        let seekbar_hover_anim_frac = ui.ctx().animate_bool_with_time(
            frame_response.id.with("seekbar_hover_anim"),
            seekbar_hovered || currently_seeking,
            animation_time,
        );

        if seekbar_hover_anim_frac > 0. {
            let new_top = fullseekbar_rect.top() - (3. * seekbar_hover_anim_frac);
            fullseekbar_rect.set_top(new_top);
            seekbar_rect.set_top(new_top);
        }

        let seek_indicator_anim = ui.ctx().animate_bool_with_time(
            frame_response.id.with("seek_indicator_anim"),
            currently_seeking,
            animation_time,
        );

        if currently_seeking {
            let seek_indicator_shadow = Shadow {
                offset: vec2(10.0, 20.0),
                blur: 15.0,
                spread: 0.0,
                color: Color32::from_black_alpha(96).linear_multiply(seek_indicator_anim),
            };
            let spinner_size = 20. * seek_indicator_anim;
            ui.painter()
                .add(seek_indicator_shadow.as_shape(frame_response.rect, Rounding::ZERO));
            ui.put(
                Rect::from_center_size(frame_response.rect.center(), Vec2::splat(spinner_size)),
                Spinner::new().size(spinner_size),
            );
        }

        if seekbar_hovered || currently_seeking {
            if let Some(hover_pos) = seekbar_response.hover_pos() {
                if seekbar_response.clicked() || seekbar_response.dragged() {
                    let seek_frac = ((hover_pos - frame_response.rect.left_top()).x
                        - seekbar_width_offset / 2.)
                        .max(0.)
                        .min(fullseekbar_width)
                        / fullseekbar_width;
                    seekbar_rect.set_right(
                        hover_pos
                            .x
                            .min(fullseekbar_rect.right())
                            .max(fullseekbar_rect.left()),
                    );
                    if is_stopped {
                        self.start()
                    }
                    self.seek(seek_frac);
                }
            }
        }
        let text_color = Color32::WHITE.linear_multiply(seekbar_anim_frac);

        let pause_icon = if is_paused {
            "▶"
        } else if is_stopped {
            "◼"
        } else if currently_seeking {
            "↔"
        } else {
            "⏸"
        };
        let audio_volume_frac = self.volume_f32();
        let sound_icon = if audio_volume_frac > 0.7 {
            "🔊"
        } else if audio_volume_frac > 0.4 {
            "🔉"
        } else if audio_volume_frac > 0. {
            "🔈"
        } else {
            "🔇"
        };

        let mut icon_font_id = FontId::default();
        icon_font_id.size = 16.;

        let subtitle_icon = "💬";
        let stream_icon = "🔁";
        let icon_margin = 5.;
        let text_y_offset = -7.;
        let sound_icon_offset = vec2(-5., text_y_offset);
        let sound_icon_pos = fullseekbar_rect.right_top() + sound_icon_offset;

        let stream_index_icon_offset = vec2(-30., text_y_offset + 1.);
        let stream_icon_pos = fullseekbar_rect.right_top() + stream_index_icon_offset;

        let contraster_alpha: u8 = 100;
        let pause_icon_offset = vec2(3., text_y_offset);
        let pause_icon_pos = fullseekbar_rect.left_top() + pause_icon_offset;

        let duration_text_offset = vec2(25., text_y_offset);
        let duration_text_pos = fullseekbar_rect.left_top() + duration_text_offset;
        let mut duration_text_font_id = FontId::default();
        duration_text_font_id.size = 14.;

        let shadow = Shadow {
            offset: vec2(10.0, 20.0),
            blur: 15.0,
            spread: 0.0,
            color: Color32::from_black_alpha(25).linear_multiply(seekbar_anim_frac),
        };

        let mut shadow_rect = frame_response.rect;
        shadow_rect.set_top(shadow_rect.bottom() - seekbar_offset - 10.);

        let fullseekbar_color = Color32::GRAY.linear_multiply(seekbar_anim_frac);
        let seekbar_color = Color32::WHITE.linear_multiply(seekbar_anim_frac);

        ui.painter()
            .add(shadow.as_shape(shadow_rect, Rounding::ZERO));

        ui.painter().rect_filled(
            fullseekbar_rect,
            Rounding::ZERO,
            fullseekbar_color.linear_multiply(0.5),
        );
        ui.painter()
            .rect_filled(seekbar_rect, Rounding::ZERO, seekbar_color);
        ui.painter().text(
            pause_icon_pos,
            Align2::LEFT_BOTTOM,
            pause_icon,
            icon_font_id.clone(),
            text_color,
        );

        ui.painter().text(
            duration_text_pos,
            Align2::LEFT_BOTTOM,
            format!("{}/{}", self.elapsed(), 0),
            duration_text_font_id,
            text_color,
        );

        if seekbar_hover_anim_frac > 0. {
            ui.painter().circle_filled(
                seekbar_rect.right_center(),
                7. * seekbar_hover_anim_frac,
                seekbar_color,
            );
        }

        if frame_response.clicked() {
            let mut reset_stream = false;
            let mut start_stream = false;

            match self.player_state {
                PlayerState::Stopped => start_stream = true,
                PlayerState::EndOfFile => reset_stream = true,
                PlayerState::Paused => self.resume(),
                PlayerState::Playing => self.pause(),
                _ => (),
            }

            if reset_stream {
                self.resume();
            } else if start_stream {
                self.start();
            }
        }

        let is_subtitle_cyclable = false;
        let is_audio_cyclable = false;

        if is_audio_cyclable || is_subtitle_cyclable {
            let stream_icon_rect = ui.painter().text(
                stream_icon_pos,
                Align2::RIGHT_BOTTOM,
                stream_icon,
                icon_font_id.clone(),
                text_color,
            );
            let stream_icon_hovered = ui.rect_contains_pointer(stream_icon_rect);
            let mut stream_info_hovered = false;
            let mut cursor = stream_icon_rect.right_top() + vec2(0., 5.);
            let cursor_offset = vec2(3., 15.);
            let stream_anim_id = frame_response.id.with("stream_anim");
            let mut stream_anim_frac: f32 = ui
                .ctx()
                .memory_mut(|m| *m.data.get_temp_mut_or_default(stream_anim_id));

            let mut draw_row = |stream_type: AVMediaType| {
                let text = match stream_type {
                    AVMediaType::AVMEDIA_TYPE_AUDIO => format!(
                        "{} {}/{}",
                        sound_icon, 1, 1
                    ),
                    AVMediaType::AVMEDIA_TYPE_SUBTITLE => format!(
                        "{} {}/{}",
                        subtitle_icon, 1, 1
                    ),
                    _ => unreachable!(),
                };

                let text_position = cursor - cursor_offset;
                let text_galley =
                    ui.painter()
                        .layout_no_wrap(text.clone(), icon_font_id.clone(), text_color);

                let background_rect =
                    Rect::from_min_max(text_position - text_galley.size(), text_position)
                        .expand(5.);

                let background_color =
                    Color32::from_black_alpha(contraster_alpha).linear_multiply(stream_anim_frac);

                ui.painter()
                    .rect_filled(background_rect, Rounding::same(5.), background_color);

                if ui.rect_contains_pointer(background_rect.expand(5.)) {
                    stream_info_hovered = true;
                }

                if ui
                    .interact(
                        background_rect,
                        frame_response.id.with(&text),
                        Sense::click(),
                    )
                    .clicked()
                {
                    // TODO: cycle stream
                };

                let text_rect = ui.painter().text(
                    text_position,
                    Align2::RIGHT_BOTTOM,
                    text,
                    icon_font_id.clone(),
                    text_color.linear_multiply(stream_anim_frac),
                );

                cursor.y = text_rect.top();
            };

            if stream_anim_frac > 0. {
                if is_audio_cyclable {
                    draw_row(AVMediaType::AVMEDIA_TYPE_AUDIO);
                }
                if is_subtitle_cyclable {
                    draw_row(AVMediaType::AVMEDIA_TYPE_SUBTITLE);
                }
            }

            stream_anim_frac = ui.ctx().animate_bool_with_time(
                stream_anim_id,
                stream_icon_hovered || (stream_info_hovered && stream_anim_frac > 0.),
                animation_time,
            );

            ui.ctx()
                .memory_mut(|m| m.data.insert_temp(stream_anim_id, stream_anim_frac));
        }

        let sound_icon_rect = ui.painter().text(
            sound_icon_pos,
            Align2::RIGHT_BOTTOM,
            sound_icon,
            icon_font_id.clone(),
            text_color,
        );
        if ui
            .interact(
                sound_icon_rect,
                frame_response.id.with("sound_icon_sense"),
                Sense::click(),
            )
            .clicked()
        {
            if self.volume() != 0 {
                self.set_volume(0);
            } else {
                self.set_volume(50)
            }
        }

        let sound_slider_outer_height = 75.;

        let mut sound_slider_rect = sound_icon_rect;
        sound_slider_rect.set_bottom(sound_icon_rect.top() - icon_margin);
        sound_slider_rect.set_top(sound_slider_rect.top() - sound_slider_outer_height);

        let sound_slider_interact_rect = sound_slider_rect.expand(icon_margin);
        let sound_hovered = ui.rect_contains_pointer(sound_icon_rect);
        let sound_slider_hovered = ui.rect_contains_pointer(sound_slider_interact_rect);
        let sound_anim_id = frame_response.id.with("sound_anim");
        let mut sound_anim_frac: f32 = ui
            .ctx()
            .memory_mut(|m| *m.data.get_temp_mut_or_default(sound_anim_id));
        sound_anim_frac = ui.ctx().animate_bool_with_time(
            sound_anim_id,
            sound_hovered || (sound_slider_hovered && sound_anim_frac > 0.),
            0.2,
        );
        ui.ctx()
            .memory_mut(|m| m.data.insert_temp(sound_anim_id, sound_anim_frac));
        let sound_slider_bg_color =
            Color32::from_black_alpha(contraster_alpha).linear_multiply(sound_anim_frac);
        let sound_bar_color =
            Color32::from_white_alpha(contraster_alpha).linear_multiply(sound_anim_frac);
        let mut sound_bar_rect = sound_slider_rect;
        sound_bar_rect
            .set_top(sound_bar_rect.bottom() - audio_volume_frac * sound_bar_rect.height());

        ui.painter()
            .rect_filled(sound_slider_rect, Rounding::same(5.), sound_slider_bg_color);

        ui.painter()
            .rect_filled(sound_bar_rect, Rounding::same(5.), sound_bar_color);
        let sound_slider_resp = ui.interact(
            sound_slider_rect,
            frame_response.id.with("sound_slider_sense"),
            Sense::click_and_drag(),
        );
        if sound_anim_frac > 0. && sound_slider_resp.clicked() || sound_slider_resp.dragged() {
            if let Some(hover_pos) = ui.ctx().input(|i| i.pointer.hover_pos()) {
                let sound_frac = 1.
                    - ((hover_pos - sound_slider_rect.left_top()).y
                    / sound_slider_rect.height())
                    .max(0.)
                    .min(1.);
                self.set_volume_f32(sound_frac);
            }
        }
    }

    fn open_default_audio_stream(volume: Arc<AtomicU8>) -> Option<PlayerAudioStream> {
        if let Ok(a) = AudioDevice::new() {
            if let Ok(cfg) = a.0.default_output_config() {
                let audio_sample_buffer = HeapRb::<f32>::new(1024 * 1024);
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
                    }, None) {
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

    /// Create a new [`Player`].
    pub fn new(ctx: &egui::Context, input_path: &String) -> Self {
        let options = PlayerOptions::default();
        let texture_handle =
            ctx.load_texture("vidstream", ColorImage::new([1, 1], Color32::BLACK), options.texture_options);

        /// Open audio device
        let audio = Self::open_default_audio_stream(options.audio_volume.clone());
        

        Self {
            input_path: input_path.clone(),
            texture_handle,
            player_state: PlayerState::Stopped,
            options,
            pts: 0,
            timebase: AVRational { num: 0, den: 1 },
            ctx: ctx.clone(),
            audio,
            media_player: MediaPlayer::new(input_path),
        }
    }
}

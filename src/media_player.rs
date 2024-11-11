use crate::ffmpeg_sys_the_third::{
    av_frame_free, av_frame_unref, av_get_media_type_string, av_packet_free, av_rescale_q,
    av_sample_fmt_is_planar, AVCodecID, AVFrame, AVMediaType, AVPacket, AVPixelFormat, AVRational,
    AVSampleFormat, AVStream, AV_NOPTS_VALUE,
};
#[cfg(feature = "hls")]
use crate::hls::DemuxerIsh;
#[cfg(feature = "hls")]
use crate::hls::HlsStream;

use anyhow::{Error, Result};
use egui::{Color32, ColorImage, Vec2};
use ffmpeg_rs_raw::{get_frame_from_hw, rstr, Decoder, Demuxer, DemuxerInfo, Resample, Scaler};
use log::{error, trace, warn};
use std::collections::BinaryHeap;
use std::ffi::CStr;
use std::fmt::{Display, Formatter};
use std::mem::transmute;
use std::sync::atomic::{
    AtomicBool, AtomicI32, AtomicI64, AtomicU16, AtomicU32, AtomicU8, Ordering,
};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;
use std::{ptr, slice};

/// Constructs a [ColorImage] from a raw [AvFrame]
pub unsafe fn video_frame_to_image(mut frame: *mut AVFrame) -> ColorImage {
    let size = [(*frame).width as usize, (*frame).height as usize];
    let pixels = size[0] * size[1];

    let pixels: Vec<Color32> = match transmute((*frame).format) {
        AVPixelFormat::AV_PIX_FMT_RGB24 => {
            // we must copy the data because Color32 is RGBA
            let ret = map_frame_to_pixels(frame);
            av_frame_free(&mut frame);
            ret
        }
        AVPixelFormat::AV_PIX_FMT_RGBA => {
            if (*frame).width * 4 == (*frame).linesize[0] {
                // zero-copy transmute buffer directly to Vec<Color32>
                // !! this is probably not a good idea !!
                let buf = (*frame).data[0];
                (*frame).buf.as_mut_ptr().add(0).write(ptr::null_mut());
                (*frame).data.as_mut_ptr().add(0).write(ptr::null_mut());
                let ret = Vec::from_raw_parts(buf as *mut Color32, pixels, pixels);
                av_frame_unref(frame);
                ret
            } else {
                // rows are not packed, we must copy
                let ret = map_frame_to_pixels(frame);
                av_frame_free(&mut frame);
                ret
            }
        }
        _ => panic!("Pixel format not supported!"),
    };
    ColorImage { size, pixels }
}

unsafe fn map_frame_to_pixels(frame: *mut AVFrame) -> Vec<Color32> {
    let stride = (*frame).linesize[0] as usize;
    let lines = (*frame).height as usize;
    let data = slice::from_raw_parts_mut((*frame).data[0], stride * lines);
    let bytes = match transmute((*frame).format) {
        AVPixelFormat::AV_PIX_FMT_RGB24 => 3,
        AVPixelFormat::AV_PIX_FMT_RGBA => 4,
        _ => panic!("Pixel format not supported!"),
    };
    (0..lines)
        .map(|r| {
            let offset = r * stride;
            data[offset..offset + stride]
                .chunks_exact(bytes)
                .take((*frame).width as usize)
                .map(|c| match bytes {
                    3 => Color32::from_rgb(c[0], c[1], c[2]),
                    4 => Color32::from_rgba_premultiplied(c[0], c[1], c[2], c[3]),
                    _ => panic!("not possible"),
                })
        })
        .flatten()
        .collect()
}

#[derive(Debug, Clone)]
pub struct DecoderInfo {
    /// Stream index
    pub index: usize,
    /// Which codec is being used for decoding
    pub codec: String,
}

#[derive(Debug)]
/// Messages received from the [MediaPlayer]
pub enum MediaPlayerData {
    /// Stream information
    MediaInfo(DemuxerInfo),
    /// Basic decoder info
    DecoderInfo(DecoderInfo),
    /// A fatal error occurred during media playback
    Error(String),
    /// Video frame from the decoder
    VideoFrame(i64, i64, ColorImage),
    /// Audio samples from the decoder
    AudioSamples(AudioSamplesData),
    /// Subtitle frames from the decoder
    Subtitles(i64, i64, String, AVCodecID),
}

#[derive(Debug)]
pub struct AudioSamplesData {
    pub pts: i64,
    pub duration: i64,
    pub samples: Vec<f32>,
}

impl MediaPlayerData {
    pub fn pts(&self) -> i64 {
        match self {
            MediaPlayerData::AudioSamples(a) => a.pts,
            MediaPlayerData::VideoFrame(pts, _, _) => *pts,
            MediaPlayerData::Subtitles(pts, _, _, _) => *pts,
            _ => 0,
        }
    }
}

impl PartialEq<Self> for MediaPlayerData {
    fn eq(&self, other: &Self) -> bool {
        other.cmp(self).is_eq()
    }
}

impl Eq for MediaPlayerData {}

impl Ord for MediaPlayerData {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let pts_a = self.pts();
        let pts_b = other.pts();
        pts_b.cmp(&pts_a)
    }
}

impl PartialOrd for MediaPlayerData {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone)]
struct MediaPlayerState {
    pub width: Arc<AtomicU16>,
    pub height: Arc<AtomicU16>,
    pub sample_rate: Arc<AtomicU32>,
    pub channels: Arc<AtomicU8>,
    pub pts_max: Arc<AtomicI64>,
    pub pts_min: Arc<AtomicI64>,
    pub running: Arc<AtomicBool>,
    pub buffer_size: Arc<AtomicI32>,
    pub media_queue: Arc<Mutex<BinaryHeap<MediaPlayerData>>>,
    pub audio_chan: Option<Sender<AudioSamplesData>>,
    pub tbn_num: Arc<AtomicI32>,
    pub tbn_den: Arc<AtomicI32>,
    pub seek_to: Arc<Mutex<Option<f32>>>,
}

impl Default for MediaPlayerState {
    fn default() -> Self {
        const DEFAULT_TBN: i32 = 90_000;

        Self {
            running: Arc::new(AtomicBool::new(true)),
            buffer_size: Arc::new(AtomicI32::new(DEFAULT_TBN)),
            tbn_num: Arc::new(AtomicI32::new(1)),
            tbn_den: Arc::new(AtomicI32::new(DEFAULT_TBN)),

            // defaults
            width: Arc::new(Default::default()),
            height: Arc::new(Default::default()),
            sample_rate: Arc::new(Default::default()),
            channels: Arc::new(Default::default()),
            pts_max: Arc::new(Default::default()),
            pts_min: Arc::new(Default::default()),
            media_queue: Arc::new(Mutex::new(Default::default())),
            audio_chan: None,
            seek_to: Arc::new(Mutex::new(None)),
        }
    }
}

impl MediaPlayerState {
    pub fn tbn(&self) -> AVRational {
        AVRational {
            num: self.tbn_num.load(Ordering::Relaxed),
            den: self.tbn_den.load(Ordering::Relaxed),
        }
    }
}

/// Provides raw buffered video/audio/subtitles for playback
pub struct MediaPlayer {
    input: String,
    thread: Option<JoinHandle<()>>,
    state: MediaPlayerState,
}

impl Display for MediaPlayer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.input)
    }
}

impl Drop for MediaPlayer {
    fn drop(&mut self) {
        self.state.running.store(false, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().expect("join failed");
        }
    }
}

impl MediaPlayer {
    pub fn new(input: &str) -> Self {
        Self {
            input: input.to_string(),
            thread: None,
            state: MediaPlayerState::default(),
        }
    }

    pub fn tbn(&self) -> AVRational {
        self.state.tbn()
    }

    pub fn with_audio_chan(mut self, chan: Sender<AudioSamplesData>) -> Self {
        self.state.audio_chan = Some(chan);
        self
    }

    /// Get the size of the buffer as AV_TIMEBASE units
    pub fn buffer_size(&self) -> i64 {
        let max = self.state.pts_max.load(Ordering::Relaxed);
        let min = self.state.pts_min.load(Ordering::Relaxed);
        max - min
    }

    /// Get the number of messages in the queue
    pub fn buffer_len(&self) -> usize {
        self.state.media_queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Set the video target size in pixels
    pub fn set_target_size(&self, size: Vec2) {
        self.state.width.store(size.x as u16, Ordering::Relaxed);
        self.state.height.store(size.y as u16, Ordering::Relaxed);
    }

    /// Set the audio sample rate, all samples are stereo f32
    pub fn set_target_sample_rate(&self, sample_rate: u32) {
        self.state.sample_rate.store(sample_rate, Ordering::Relaxed);
    }

    /// Seek to position time in seconds
    pub fn seek_to(&self, time: f32) {
        if let Ok(mut s) = self.state.seek_to.lock() {
            s.replace(time);
        }
    }

    fn store_min_pts(&self, r: Option<&MediaPlayerData>) {
        if let Some(m) = r.as_ref() {
            let pts = m.pts();
            if pts != 0 {
                self.state.pts_min.store(pts, Ordering::Relaxed);
            }
        }
    }

    /// Pop the next message from the player in PTS order
    ///
    /// You must read from this endpoint at the correct rate to match the video rate.
    /// Internally the buffer will build up to [target_buffer_size]
    pub fn next(&mut self) -> Option<MediaPlayerData> {
        if let Ok(mut q) = self.state.media_queue.lock() {
            let r = q.pop();
            self.store_min_pts(r.as_ref());
            r
        } else {
            None
        }
    }

    /// Pop the next message if [f_check] is true
    pub fn next_if<F>(&mut self, mut f_check: F) -> Option<MediaPlayerData>
    where
        F: FnMut(&MediaPlayerData) -> bool,
    {
        if let Ok(mut q) = self.state.media_queue.lock() {
            if let Some(r) = q.peek() {
                if f_check(r) {
                    let r = q.pop();
                    self.store_min_pts(r.as_ref());
                    return r;
                }
            }
        }
        None
    }

    pub fn stop(&mut self) {
        self.state.running.store(false, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().expect("join failed");
        }
    }

    pub fn is_running(&self) -> bool {
        if let Some(t) = self.thread.as_ref() {
            !t.is_finished()
        } else {
            false
        }
    }

    pub fn start(&mut self) -> bool {
        if let Some(t) = &self.thread {
            if t.is_finished() {
                self.thread.take();
            } else {
                return false;
            }
        }
        let state = self.state.clone();
        let input = self.input.clone();
        let q_err = self.state.media_queue.clone();
        self.thread = Some(std::thread::spawn(move || {
            match MediaPlayerThread::new(&input, state) {
                Ok(media_thread) => {
                    if let Err(e) = unsafe { media_thread.run() } {
                        error!("{}", e);
                        if let Ok(mut q) = q_err.lock() {
                            q.push(MediaPlayerData::Error(e.to_string()));
                        }
                    }
                }
                Err(e) => {
                    error!("{}", e);
                    if let Ok(mut q) = q_err.lock() {
                        q.push(MediaPlayerData::Error(e.to_string()));
                    }
                }
            }
        }));
        true
    }
}

struct MediaPlayerThread {
    state: MediaPlayerState,
    #[cfg(feature = "hls")]
    demuxer: Box<dyn DemuxerIsh>,
    #[cfg(not(feature = "hls"))]
    demuxer: Demuxer,
    decoder: Decoder,
    scale: Scaler,
    resample: Resample,
    media_info: Option<DemuxerInfo>,
}

impl Display for MediaPlayerThread {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(info) = &self.media_info {
            if let Some(v) = info.best_video() {
                writeln!(f, "{}", v)?;
                if let Some(decoder) = self.decoder.get_decoder(v.index as i32) {
                    writeln!(f, "\t{}", decoder)?;
                }
            }
            if let Some(v) = info.best_audio() {
                writeln!(f, "{}", v)?;
                if let Some(decoder) = self.decoder.get_decoder(v.index as i32) {
                    writeln!(f, "\t{}", decoder)?;
                }
            }
            if let Some(v) = info.best_subtitle() {
                writeln!(f, "{}", v)?;
                if let Some(decoder) = self.decoder.get_decoder(v.index as i32) {
                    writeln!(f, "\t{}", decoder)?;
                }
            }
        }
        Ok(())
    }
}

impl MediaPlayerThread {
    pub fn new(input: &str, state: MediaPlayerState) -> Result<MediaPlayerThread> {
        let sr = state.sample_rate.load(Ordering::Relaxed);
        let mut decoder = Decoder::new();
        decoder.enable_hw_decoder_any();

        Ok(Self {
            state,
            demuxer: Self::open_demuxer(input)?,
            decoder,
            scale: Scaler::new(),
            resample: Resample::new(AVSampleFormat::AV_SAMPLE_FMT_FLT, sr, 2),
            media_info: None,
        })
    }

    #[cfg(feature = "hls")]
    fn open_demuxer(input: &str) -> Result<Box<dyn DemuxerIsh>> {
        if input.contains(".m3u8") {
            Ok(Box::new(HlsStream::new(input)))
        } else {
            Ok(Box::new(Demuxer::new(input)?))
        }
    }

    #[cfg(not(feature = "hls"))]
    fn open_demuxer(input: &str) -> Result<Demuxer> {
        Ok(Demuxer::new(input)?)
    }

    unsafe fn run(mut self) -> Result<(), Error> {
        self.state.pts_min.store(0, Ordering::Relaxed);
        self.state.pts_max.store(0, Ordering::Relaxed);
        self.state.running.store(true, Ordering::Relaxed);

        loop {
            if !self.state.running.load(Ordering::Relaxed) {
                break;
            }

            // wait until probe result
            if self.media_info.as_ref().is_none() {
                match self.demuxer.probe_input() {
                    Ok(p) => {
                        self.media_info = Some(p.clone());
                        let mut q = self
                            .state
                            .media_queue
                            .lock()
                            .map_err(|e| Error::msg(e.to_string()))?;
                        for c in &p.streams {
                            let ctx = self.decoder.setup_decoder(c, None)?;
                            q.push(MediaPlayerData::DecoderInfo(DecoderInfo {
                                index: c.index,
                                codec: ctx.codec_name(),
                            }));
                        }
                        q.push(MediaPlayerData::MediaInfo(p));
                    }
                    Err(e) => {
                        let mut q = self
                            .state
                            .media_queue
                            .lock()
                            .map_err(|e| Error::msg(e.to_string()))?;
                        q.push(MediaPlayerData::Error(e.to_string()));
                        break;
                    }
                }
            }

            // unwrap
            let probed = self.media_info.as_ref().unwrap();

            // check if buffer is full
            let buf_size = self.state.pts_max.load(Ordering::Relaxed)
                - self.state.pts_min.load(Ordering::Relaxed);
            if buf_size > self.state.buffer_size.load(Ordering::Relaxed) as i64 {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }

            // handle seeking
            if let Some(s) = self
                .state
                .seek_to
                .lock()
                .map_err(|e| Error::msg(e.to_string()))?
                .take()
            {
                trace!("seeking to={}", s);
            }

            // read frames
            let (mut pkt, stream) = self.demuxer.get_packet()?;
            if !stream.is_null() && !pkt.is_null() && !probed.is_best_stream(stream) {
                av_packet_free(&mut pkt);
                continue;
            }

            let media_type = (*(*stream).codecpar).codec_type;
            if media_type == AVMediaType::AVMEDIA_TYPE_VIDEO
                || media_type == AVMediaType::AVMEDIA_TYPE_AUDIO
            {
                match self.decoder.decode_pkt(pkt) {
                    Ok(frames) => {
                        for frame in frames {
                            let mut frame = self.process_frame(frame, stream)?;
                            av_frame_free(&mut frame);
                        }
                    }
                    Err(e) => warn!("failed to decode packet: {}", e),
                }
            } else if media_type == AVMediaType::AVMEDIA_TYPE_SUBTITLE {
                let subs = CStr::from_ptr((*pkt).data as _).to_str()?;
                let (pts, duration) = self.get_pkt_time(pkt, stream);
                self.send_decoder_msg(MediaPlayerData::Subtitles(
                    pts,
                    duration,
                    subs.to_string(),
                    (*(*stream).codecpar).codec_id,
                ))?;
            }

            if pkt.is_null() {
                break;
            }
            av_packet_free(&mut pkt);
        }
        self.state.running.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Push message to queue
    fn send_decoder_msg(&self, msg: MediaPlayerData) -> Result<()> {
        if self.state.audio_chan.is_some() {
            if let MediaPlayerData::AudioSamples(s) = msg {
                if let Some(audio) = self.state.audio_chan.as_ref() {
                    audio.send(s)?;
                }
                return Ok(());
            }
        }

        let mut q = self
            .state
            .media_queue
            .lock()
            .map_err(|e| Error::msg(e.to_string()))?;
        q.push(msg);
        Ok(())
    }

    /// Get the pts & duration of a frame in local timescale
    unsafe fn get_frame_time(&self, frame: *mut AVFrame, stream: *mut AVStream) -> (i64, i64) {
        let tbn = self.state.tbn();
        let stream_start = if (*stream).start_time == AV_NOPTS_VALUE {
            0
        } else {
            (*stream).start_time
        };
        let pts = av_rescale_q((*frame).pts - stream_start, (*stream).time_base, tbn);
        let duration = av_rescale_q((*frame).duration, (*stream).time_base, tbn);
        (pts, duration)
    }

    /// Get the pts & duration of a packet in local timescale
    unsafe fn get_pkt_time(&self, pkt: *mut AVPacket, stream: *mut AVStream) -> (i64, i64) {
        let tbn = self.state.tbn();
        let stream_start = if (*stream).start_time == AV_NOPTS_VALUE {
            0
        } else {
            (*stream).start_time
        };
        let pts = av_rescale_q((*pkt).pts - stream_start, (*stream).time_base, tbn);
        let duration = av_rescale_q((*pkt).duration, (*stream).time_base, tbn);
        (pts, duration)
    }

    /// Send decoded frames to player
    unsafe fn process_frame(
        &mut self,
        frame: *mut AVFrame,
        stream: *mut AVStream,
    ) -> Result<*mut AVFrame, Error> {
        let frame = get_frame_from_hw(frame)?;
        let (pts, duration) = self.get_frame_time(frame, stream);
        let media_type = (*(*stream).codecpar).codec_type;

        if media_type == AVMediaType::AVMEDIA_TYPE_VIDEO {
            let width = self.state.width.load(Ordering::Relaxed);
            let height = self.state.height.load(Ordering::Relaxed);
            let frame =
                self.scale
                    .process_frame(frame, width, height, AVPixelFormat::AV_PIX_FMT_RGBA)?;
            let image = video_frame_to_image(frame);
            self.send_decoder_msg(MediaPlayerData::VideoFrame(pts, duration, image))?;
            let _ = self
                .state
                .pts_max
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
                    if pts > v {
                        Some(pts)
                    } else {
                        None
                    }
                });
        } else if media_type == AVMediaType::AVMEDIA_TYPE_AUDIO {
            let mut frame = self.resample.process_frame(frame)?;
            let is_planar = av_sample_fmt_is_planar(transmute((*frame).format)) == 1;
            let plane_mul = if is_planar {
                1
            } else {
                (*frame).ch_layout.nb_channels
            };
            let size = ((*frame).nb_samples * plane_mul) as usize;

            let samples = slice::from_raw_parts((*frame).data[0] as *const f32, size);
            let data = AudioSamplesData {
                pts,
                duration,
                samples: samples.to_vec(),
            };
            let msg = MediaPlayerData::AudioSamples(data);
            self.send_decoder_msg(msg)?;
            av_frame_free(&mut frame);
        } else {
            warn!(
                "unsupported frame type: {}",
                rstr!(av_get_media_type_string(media_type))
            );
        }
        Ok(frame)
    }
}

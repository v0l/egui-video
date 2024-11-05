use crate::ffmpeg_sys_the_third::{
    av_frame_free, av_frame_unref, av_packet_free, av_rescale_q, av_sample_fmt_is_planar, AVFrame,
    AVMediaType, AVPixelFormat, AVRational, AVSampleFormat, AVStream,
};
#[cfg(feature = "hls")]
use crate::hls::DemuxerIsh;
#[cfg(feature = "hls")]
use crate::hls::HlsStream;

use crate::ffmpeg_sys_the_third::AVHWDeviceType::AV_HWDEVICE_TYPE_MEDIACODEC;
use egui::{Color32, ColorImage, Vec2};
use ffmpeg_rs_raw::ffmpeg_sys_the_third::AVHWDeviceType::{
    AV_HWDEVICE_TYPE_CUDA, AV_HWDEVICE_TYPE_VDPAU,
};
use ffmpeg_rs_raw::{Decoder, Demuxer, DemuxerInfo, Resample, Scaler};
use std::collections::BinaryHeap;
use std::mem::transmute;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU8, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, RwLock};
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
    let mut data = slice::from_raw_parts_mut((*frame).data[0], stride * lines);
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

#[derive(Debug)]
/// Messages received from the decoder
pub enum DecoderMessage {
    MediaInfo(DemuxerInfo),
    /// A fatal error occurred during media playback
    Error(String),
    /// Video frame from the decoder
    VideoFrame(i64, i64, ColorImage),
    /// Audio samples from the decoder
    AudioSamples(i64, i64, Vec<f32>),
    /// Subtitle frames from the decoder
    Subtitles(i64, i64, String),
}

impl DecoderMessage {
    pub fn pts(&self) -> i64 {
        match self {
            DecoderMessage::AudioSamples(pts, _, _) => *pts,
            DecoderMessage::VideoFrame(pts, _, _) => *pts,
            DecoderMessage::Subtitles(pts, _, _) => *pts,
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
        let pts_a = self.pts();
        let pts_b = other.pts();
        pts_b.cmp(&pts_a)
    }
}

impl PartialOrd for DecoderMessage {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub struct MediaPlayer {
    input: String,
    thread: Option<JoinHandle<()>>,

    target_size: Arc<RwLock<[u16; 2]>>,
    target_sample_rate: Arc<AtomicU32>,
    target_channels: Arc<AtomicU8>,
    pts_max: Arc<AtomicI64>,
    pts_min: Arc<AtomicI64>,
    running: Arc<AtomicBool>,

    /// Number of timebase units of frames to store in [media_queue] (using [pts_max]/[pts_in])
    target_buffer_size: Arc<AtomicU32>,

    /// Queue of frames/subtitles/audio samples
    media_queue: Arc<Mutex<BinaryHeap<DecoderMessage>>>,

    /// Audio channel to bypass [media_queue]
    /// Audio needs more buffering than video frames
    audio_chan: Option<std::sync::mpsc::Sender<DecoderMessage>>,

    pub tbn: AVRational,
}

impl Drop for MediaPlayer {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().expect("join failed");
        }
    }
}

impl MediaPlayer {
    pub fn new(input: &str) -> Self {
        /// 90k is a very common timebase den because its divided well into various common
        /// fps' and audio sample rates
        const DEFAULT_TBN: i32 = 90_000;

        Self {
            input: input.to_string(),
            thread: None,
            target_size: Arc::new(RwLock::new([0, 0])),
            target_sample_rate: Arc::new(Default::default()),
            target_channels: Arc::new(AtomicU8::new(2)),
            pts_max: Arc::new(AtomicI64::new(0)),
            pts_min: Arc::new(AtomicI64::new(0)),
            running: Arc::new(AtomicBool::new(true)),
            media_queue: Arc::new(Mutex::new(BinaryHeap::new())),
            audio_chan: None,
            tbn: AVRational {
                num: 1,
                den: DEFAULT_TBN,
            },
            target_buffer_size: Arc::new(AtomicU32::new(DEFAULT_TBN as u32 * 2)),
        }
    }

    pub fn with_audio_chan(mut self, chan: Sender<DecoderMessage>) -> Self {
        self.audio_chan = Some(chan);
        self
    }

    /// Get the size of the buffer as AV_TIMEBASE units
    pub fn buffer_size(&self) -> i64 {
        let max = self.pts_max.load(Ordering::Relaxed);
        let min = self.pts_min.load(Ordering::Relaxed);
        max - min
    }

    /// Set the video target size in pixels
    pub fn set_target_size(&mut self, size: Vec2) {
        if let Ok(mut s) = self.target_size.write() {
            s[0] = size.x as u16;
            s[1] = size.y as u16;
        }
    }

    /// Set the audio sample rate, all samples are stereo f32
    pub fn set_target_sample_rate(&mut self, sample_rate: u32) {
        self.target_sample_rate
            .store(sample_rate, Ordering::Relaxed);
    }

    /// Pop the next message from the player in PTS order
    ///
    /// You must read from this endpoint at the correct rate to match the video rate.
    /// Internally the buffer will build up to [target_buffer_size]
    pub fn next(&mut self) -> Option<DecoderMessage> {
        if let Ok(mut q) = self.media_queue.lock() {
            let r = q.pop();
            if let Some(DecoderMessage::VideoFrame(pts, _, _)) = r.as_ref() {
                self.pts_min.store(*pts, Ordering::Relaxed);
            }
            r
        } else {
            None
        }
    }

    /// Pop the next message if [f_check] is true
    pub fn next_if<F>(&mut self, mut f_check: F) -> Option<DecoderMessage>
    where
        F: FnMut(&DecoderMessage) -> bool,
    {
        if let Ok(mut q) = self.media_queue.lock() {
            if let Some(r) = q.peek() {
                if f_check(r) {
                    let r = q.pop();
                    if let Some(DecoderMessage::VideoFrame(pts, _, _)) = r.as_ref() {
                        self.pts_min.store(*pts, Ordering::Relaxed);
                    }
                    return r;
                }
            }
        }
        None
    }

    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().expect("join failed");
        }
    }

    #[cfg(feature = "hls")]
    fn open_demuxer(input: &str) -> Box<dyn DemuxerIsh> {
        if input.contains(".m3u8") {
            Box::new(HlsStream::new(input))
        } else {
            Box::new(Demuxer::new(input))
        }
    }

    #[cfg(not(feature = "hls"))]
    fn open_demuxer(input: &str) -> Demuxer {
        Demuxer::new(input)
    }

    pub fn start(&mut self) {
        if let Some(t) = &self.thread {
            if t.is_finished() {
                self.thread.take();
            } else {
                return;
            }
        }
        let input = self.input.clone();
        let tx = self.media_queue.clone();
        let target_size = self.target_size.clone();
        let max_pts = self.pts_max.clone();
        let min_pts = self.pts_min.clone();
        let choke_point = self.target_buffer_size.clone();
        let tbn = self.tbn;
        let running = self.running.clone();
        let audio_sample_rate = self.target_sample_rate.clone();
        let audio_tx = self.audio_chan.clone();

        self.thread = Some(std::thread::spawn(move || {
            running.store(true, Ordering::Relaxed);
            let mut probed: Option<DemuxerInfo> = None;
            let mut demux = Self::open_demuxer(&input);
            let mut decode = Decoder::new();
            let mut scale: Option<Scaler> = None;
            let mut resample: Option<Resample> = None;

            decode.enable_hw_decoder(AV_HWDEVICE_TYPE_VDPAU);
            decode.enable_hw_decoder(AV_HWDEVICE_TYPE_CUDA);
            decode.enable_hw_decoder(AV_HWDEVICE_TYPE_MEDIACODEC);
            unsafe {
                loop {
                    if !running.load(Ordering::Relaxed) {
                        break;
                    }
                    // wait until probe result
                    if probed.as_ref().is_none() {
                        match demux.probe_input() {
                            Ok(p) => {
                                probed = Some(p.clone());
                                for c in &p.channels {
                                    decode
                                        .setup_decoder(c, None)
                                        .expect("Failed to setup decoder");
                                }
                                eprintln!("{}", decode);
                                if let Ok(mut q) = tx.lock() {
                                    q.push(DecoderMessage::MediaInfo(p));
                                }
                            }
                            Err(e) => {
                                if let Ok(mut q) = tx.lock() {
                                    q.push(DecoderMessage::Error(e.to_string()));
                                }
                                break;
                            }
                        }
                    }

                    if scale.is_none() {
                        scale = Some(Scaler::new(AVPixelFormat::AV_PIX_FMT_RGBA))
                    }
                    if resample.is_none() {
                        let sr = audio_sample_rate.load(Ordering::Relaxed);
                        resample = Some(Resample::new(AVSampleFormat::AV_SAMPLE_FMT_FLT, sr, 2))
                    }

                    // unwrap
                    let probed = probed.as_ref().unwrap();
                    let scale = scale.as_mut().unwrap();
                    let resample = resample.as_mut().unwrap();

                    let buf_size =
                        max_pts.load(Ordering::Relaxed) - min_pts.load(Ordering::Relaxed);
                    if buf_size > choke_point.load(Ordering::Relaxed) as i64 {
                        std::thread::sleep(Duration::from_millis(10));
                        continue;
                    }

                    // read frames
                    let (mut pkt, stream) = demux.get_packet().expect("failed to get packet");
                    if pkt.is_null() || !probed.is_best_stream(stream) {
                        av_packet_free(&mut pkt);
                        continue;
                    }

                    if let Ok(frames) = decode.decode_pkt(pkt, stream) {
                        let size = target_size.read().expect("failed to read size");
                        for (mut frame, stream) in frames {
                            Self::process_frame(
                                &tx,
                                &tbn,
                                frame,
                                stream,
                                scale,
                                resample,
                                size[0],
                                size[1],
                                audio_tx.clone(),
                            );

                            let f_pts = av_rescale_q(
                                (*frame).pts - (*stream).start_time,
                                (*stream).time_base,
                                tbn,
                            );
                            let _ = max_pts.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
                                if f_pts > v {
                                    Some(f_pts)
                                } else {
                                    None
                                }
                            });

                            av_frame_free(&mut frame);
                        }
                    }

                    if pkt.is_null() {
                        break;
                    }
                    av_packet_free(&mut pkt);
                }
            }
        }));
    }

    /// Send decoded frames to player
    unsafe fn process_frame(
        tx: &Arc<Mutex<BinaryHeap<DecoderMessage>>>,
        tbn: &AVRational,
        frame: *mut AVFrame,
        stream: *mut AVStream,
        scale: &mut Scaler,
        resample: &mut Resample,
        width: u16,
        height: u16,
        audio_chan: Option<Sender<DecoderMessage>>,
    ) {
        unsafe {
            let pts = av_rescale_q(
                (*frame).pts - (*stream).start_time,
                (*stream).time_base,
                *tbn,
            );
            let duration = av_rescale_q((*frame).duration, (*stream).time_base, *tbn);

            if (*(*stream).codecpar).codec_type == AVMediaType::AVMEDIA_TYPE_VIDEO {
                match scale.process_frame(frame, width, height) {
                    Ok(frame) => {
                        let image = video_frame_to_image(frame);
                        if let Ok(mut q) = tx.lock() {
                            q.push(DecoderMessage::VideoFrame(pts, duration, image));
                        }
                    }
                    Err(e) => panic!("{}", e),
                }
            } else if (*(*stream).codecpar).codec_type == AVMediaType::AVMEDIA_TYPE_AUDIO {
                match resample.process_frame(frame) {
                    Ok(mut frame) => {
                        let is_planar = av_sample_fmt_is_planar(transmute((*frame).format)) == 1;
                        let plane_mul = if is_planar {
                            1
                        } else {
                            (*frame).ch_layout.nb_channels
                        };
                        let size = ((*frame).nb_samples * plane_mul) as usize;

                        let samples = slice::from_raw_parts((*frame).data[0] as *const f32, size);
                        let msg = DecoderMessage::AudioSamples(pts, duration, samples.to_vec());
                        if let Some(atx) = audio_chan {
                            atx.send(msg).expect("Failed to write audio to channel");
                        } else {
                            if let Ok(mut q) = tx.lock() {
                                q.push(msg);
                            }
                        }
                        av_frame_free(&mut frame);
                    }
                    Err(e) => panic!("{}", e),
                }
            } else if (*(*stream).codecpar).codec_type == AVMediaType::AVMEDIA_TYPE_SUBTITLE {
                let str: &[u8] =
                    slice::from_raw_parts((*frame).data[0], (*frame).linesize[0] as usize);
                let str = String::from_utf8_lossy(str).to_string();
                if let Ok(mut q) = tx.lock() {
                    q.push(DecoderMessage::Subtitles(pts, duration, str));
                }
            }
        }
    }
}

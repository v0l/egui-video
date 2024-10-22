use crate::ffmpeg::decode::Decoder;
use crate::ffmpeg::demux::Demuxer;
use crate::DecoderMessage;
use egui::{Color32, ColorImage, Vec2};
use ffmpeg_sys_the_third::{
    av_frame_free, av_make_error_string, av_packet_free, av_q2d, av_rescale_q,
    av_sample_fmt_is_planar, AVFrame, AVMediaType, AVPixelFormat, AVRational, AVSampleFormat,
    AVStream,
};
use std::collections::BinaryHeap;
use std::ffi::CStr;
use std::mem::transmute;
use std::slice;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU16, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub mod decode;
pub mod demux;
pub mod resample;
pub mod scale;

use crate::ffmpeg::resample::Resample;
use crate::ffmpeg::scale::Scaler;
pub use demux::DemuxerInfo;

#[macro_export]
macro_rules! return_ffmpeg_error {
    ($x:expr) => {
        if $x < 0 {
            return Err(Error::msg(get_ffmpeg_error_msg($x)));
        }
    };
}

pub fn get_ffmpeg_error_msg(ret: libc::c_int) -> String {
    unsafe {
        const BUF_SIZE: usize = 512;
        let mut buf: [libc::c_char; BUF_SIZE] = [0; BUF_SIZE];
        av_make_error_string(buf.as_mut_ptr(), BUF_SIZE, ret);
        String::from(CStr::from_ptr(buf.as_ptr()).to_str().unwrap())
    }
}

pub unsafe fn video_frame_to_image(frame: *const AVFrame) -> ColorImage {
    let size = [(*frame).width as usize, (*frame).height as usize];
    let data = (*frame).data[0];
    let stride = (*frame).linesize[0] as usize;
    let pixel_size_bytes = 3;
    let byte_width = pixel_size_bytes * size[0];
    let mut pixels = vec![];
    for line in 0..size[1] {
        let begin = line * stride;
        let end = begin + byte_width;
        let data_line = slice::from_raw_parts(data.add(begin) as *const u8, end - begin);
        pixels.extend(
            data_line
                .chunks_exact(pixel_size_bytes)
                .map(|p| Color32::from_rgb(p[0], p[1], p[2])),
        )
    }
    ColorImage { size, pixels }
}

pub struct MediaPlayer {
    input: String,
    thread: Option<JoinHandle<()>>,

    ctx: egui::Context,
    target_height: Arc<AtomicU16>,
    target_width: Arc<AtomicU16>,
    pts_max: Arc<AtomicI64>,
    running: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,

    media_queue: Arc<Mutex<BinaryHeap<DecoderMessage>>>,
    pub tbn: AVRational,
}

impl Drop for MediaPlayer {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

impl MediaPlayer {
    pub fn new(input: &str, ctx: &egui::Context) -> Self {
        Self {
            input: input.to_string(),
            thread: None,
            ctx: ctx.clone(),
            target_width: Arc::new(AtomicU16::new(0)),
            target_height: Arc::new(AtomicU16::new(0)),
            pts_max: Arc::new(AtomicI64::new(0)),
            running: Arc::new(AtomicBool::new(true)),
            paused: Arc::new(AtomicBool::new(false)),
            media_queue: Arc::new(Mutex::new(BinaryHeap::new())),
            tbn: AVRational {
                num: 1,
                den: 90_000,
            },
        }
    }

    pub fn pts_max(&self) -> i64 {
        self.pts_max.load(Ordering::Relaxed)
    }

    pub fn next(&mut self, size: Vec2) -> Option<DecoderMessage> {
        // update target size
        self.target_height.store(size.y as u16, Ordering::Relaxed);
        self.target_width.store(size.x as u16, Ordering::Relaxed);

        if let Ok(mut q) = self.media_queue.lock() {
            if let [head, .., tail] = q.as_slice() {
                let size = tail.pts() - head.pts();
                if size < (self.tbn.den / 2) as i64 {
                    return None;
                }
            }
            q.pop()
        } else {
            None
        }
    }

    pub fn set_paused(&mut self, p: bool) {
        self.paused.store(p, Ordering::Relaxed);
    }

    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
        self.thread = None;
    }

    /// Get the number of seconds offset from realtime playback
    unsafe fn pts_sync(start: Instant, pts: i64, tbn: AVRational) -> f64 {
        let duration = (Instant::now() - start).as_secs_f64();
        let pts_duration = pts as f64 * av_q2d(tbn) as f64;
        pts_duration - duration
    }

    pub fn start(&mut self) {
        if self.thread.is_some() {
            return;
        }
        let input = self.input.clone();
        let tx = self.media_queue.clone();
        let width = self.target_width.clone();
        let height = self.target_height.clone();
        let ctx = self.ctx.clone();
        let max_pts = self.pts_max.clone();
        let tbn = self.tbn;
        let running = self.running.clone();
        let paused = self.paused.clone();
        self.thread = Some(std::thread::spawn(move || {
            running.store(true, Ordering::Relaxed);
            let mut probed: Option<DemuxerInfo> = None;
            let mut demux = Demuxer::new(&input);
            let mut decode = Decoder::new();
            let mut scale: Option<Scaler> = None;
            let mut resample: Option<Resample> = None;
            let mut start = Instant::now();
            let mut last_pts = 0;

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
                                if let Ok(mut q) = tx.lock() {
                                    q.push(DecoderMessage::MediaInfo(p));
                                    ctx.request_repaint();
                                }
                            }
                            Err(e) => {
                                panic!("{}", e);
                            }
                        }
                    }

                    if scale.is_none() {
                        scale = Some(Scaler::new(AVPixelFormat::AV_PIX_FMT_RGB24))
                    }
                    if resample.is_none() {
                        resample = Some(Resample::new(AVSampleFormat::AV_SAMPLE_FMT_FLT, 44_100, 2))
                    }

                    let buf_size = Self::pts_sync(start, max_pts.load(Ordering::Relaxed), tbn);
                    if buf_size > 2.0 || paused.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_millis(5));
                        //move start timestamp when pausing
                        start = Instant::now()
                            - Duration::from_secs_f64(last_pts as f64 * av_q2d(tbn) as f64);
                        continue;
                    }

                    // read frames
                    let (mut pkt, stream) = demux.get_packet().expect("failed to get packet");
                    if !probed.as_ref().unwrap().is_best_stream(stream) {
                        av_packet_free(&mut pkt);
                        continue;
                    }

                    if let Ok(mut frames) = decode.decode_pkt(pkt, stream) {
                        let width = width.load(Ordering::Relaxed);
                        let height = height.load(Ordering::Relaxed);
                        frames.sort_by(|a, b| (*a.0).pts.cmp(&(*b.0).pts));
                        for (mut frame, stream) in frames {
                            Self::process_frame(
                                &tx,
                                &tbn,
                                frame,
                                stream,
                                scale.as_mut().unwrap(),
                                resample.as_mut().unwrap(),
                                width,
                                height,
                            );

                            // sleep until ready to display frame
                            if (*(*stream).codecpar).codec_type == AVMediaType::AVMEDIA_TYPE_VIDEO {
                                let f_pts = av_rescale_q(
                                    (*frame).pts - (*stream).start_time,
                                    (*stream).time_base,
                                    tbn,
                                );
                                let diff = Self::pts_sync(start, f_pts, tbn);
                                if diff > 0.0 {
                                    std::thread::sleep(Duration::from_secs_f64(diff));
                                }

                                last_pts = max_pts
                                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
                                        if v < f_pts {
                                            Some(f_pts)
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or(last_pts);
                            }

                            av_frame_free(&mut frame);
                        }
                    }

                    ctx.request_repaint();
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
    ) {
        unsafe {
            if (*(*stream).codecpar).codec_type == AVMediaType::AVMEDIA_TYPE_VIDEO {
                match scale.process_frame(frame, width, height) {
                    Ok(mut frame) => {
                        let pts = av_rescale_q(
                            (*frame).pts - (*stream).start_time,
                            (*stream).time_base,
                            *tbn,
                        );
                        let duration = av_rescale_q((*frame).duration, (*stream).time_base, *tbn);
                        let image = video_frame_to_image(frame);
                        if let Ok(mut q) = tx.lock() {
                            q.push(DecoderMessage::VideoFrame(pts, duration, image));
                        }
                        av_frame_free(&mut frame);
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
                        let pts = av_rescale_q(
                            (*frame).pts - (*stream).start_time,
                            (*stream).time_base,
                            *tbn,
                        );
                        let duration = av_rescale_q((*frame).duration, (*stream).time_base, *tbn);

                        let samples = slice::from_raw_parts((*frame).data[0] as *const f32, size);
                        if let Ok(mut q) = tx.lock() {
                            q.push(DecoderMessage::AudioSamples(
                                pts,
                                duration,
                                samples.to_vec(),
                            ));
                        }
                        av_frame_free(&mut frame);
                    }
                    Err(e) => panic!("{}", e),
                }
            } else if (*(*stream).codecpar).codec_type == AVMediaType::AVMEDIA_TYPE_SUBTITLE {
                let pts = av_rescale_q(
                    (*frame).pts - (*stream).start_time,
                    (*stream).time_base,
                    *tbn,
                );
                let duration = av_rescale_q((*frame).duration, (*stream).time_base, *tbn);
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

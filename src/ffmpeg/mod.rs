use crate::ffmpeg::decode::Decoder;
use crate::ffmpeg::demux::Demuxer;
use crate::DecoderMessage;
use egui::{Color32, ColorImage, Vec2};
use egui_inbox::{UiInbox, UiInboxSender};
use ffmpeg_sys_the_third::{av_frame_free, av_make_error_string, av_packet_free, av_sample_fmt_is_planar, AVFrame, AVMediaType, AVPacket, AVPixelFormat, AVSampleFormat, AVStream};
use std::ffi::CStr;
use std::mem::transmute;
use std::slice;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

mod demux;
mod decode;
mod scale;
mod resample;

use crate::ffmpeg::resample::Resample;
use crate::ffmpeg::scale::Scaler;
pub use demux::DemuxerInfo;

pub enum MediaPayload {
    Flush,
    MediaInfo(DemuxerInfo),
    AvPacket(*mut AVPacket, *mut AVStream),
    AvFrame(*mut AVFrame, *mut AVStream),
}

unsafe impl Send for MediaPayload {}

impl Drop for MediaPayload {
    fn drop(&mut self) {
        match self {
            MediaPayload::AvPacket(pkt, _) => unsafe {
                av_packet_free(pkt);
            }
            MediaPayload::AvFrame(frm, _) => unsafe {
                av_frame_free(frm);
            }
            _ => {}
        }
    }
}

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

unsafe fn video_frame_to_image(frame: *const AVFrame) -> ColorImage {
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
    inbox: UiInbox<DecoderMessage>,

    target_height: Arc<AtomicU16>,
    target_width: Arc<AtomicU16>,
}

impl MediaPlayer {
    pub fn new(input: &str) -> Self {
        Self {
            input: input.to_string(),
            thread: None,
            inbox: UiInbox::new(),
            target_width: Arc::new(AtomicU16::new(0)),
            target_height: Arc::new(AtomicU16::new(0)),
        }
    }

    pub fn next(&mut self, ctx: &egui::Context, size: Vec2) -> impl Iterator<Item=DecoderMessage> {
        self.target_height.store(size.y as u16, Ordering::Relaxed);
        self.target_width.store(size.x as u16, Ordering::Relaxed);
        self.inbox.read(ctx)
    }

    pub fn start(&mut self) {
        if self.thread.is_some() {
            return;
        }
        let input = self.input.clone();
        let tx = self.inbox.sender();
        let width = self.target_width.clone();
        let height = self.target_height.clone();
        self.thread = Some(std::thread::spawn(move || {
            let mut probed: Option<DemuxerInfo> = None;
            let mut demux = Demuxer::new(&input);
            let mut decode = Decoder::new();
            let mut scale: Option<Scaler> = None;
            let mut resample: Option<Resample> = None;

            unsafe {
                loop {
                    // wait until probe result
                    if probed.as_ref().is_none() {
                        match demux.probe_input() {
                            Ok(p) => {
                                probed = Some(p.clone());
                                tx.send(DecoderMessage::MediaInfo(p)).unwrap();
                            }
                            Err(e) => {
                                panic!("{}", e);
                            }
                        }
                    }

                    if scale.is_none() {
                        scale = Some(Scaler::new(
                            AVPixelFormat::AV_PIX_FMT_RGB24,
                        ))
                    }
                    if resample.is_none() {
                        resample = Some(Resample::new(
                            AVSampleFormat::AV_SAMPLE_FMT_FLT,
                            44_100,
                            2,
                        ))
                    }

                    // read frames
                    let (mut pkt, stream) = demux.get_packet().expect("failed to get packet");
                    if !probed.as_ref().unwrap().is_best_stream(stream) {
                        av_packet_free(&mut pkt);
                        continue;
                    }
                    if let Ok(frames) = decode.process(pkt, stream) {
                        let width = width.load(Ordering::Relaxed);
                        let height = height.load(Ordering::Relaxed);
                        for frame in frames {
                            Self::process_frame(
                                &tx,
                                frame,
                                scale.as_mut().unwrap(),
                                resample.as_mut().unwrap(),
                                width, height,
                            );
                        }
                    }
                    av_packet_free(&mut pkt);
                }
            }
        }));
    }

    /// Send decoded frames to player
    unsafe fn process_frame(
        tx: &UiInboxSender<DecoderMessage>,
        frame: MediaPayload,
        scale: &mut Scaler,
        resample: &mut Resample,
        width: u16,
        height: u16,
    ) {
        if let MediaPayload::AvFrame(frm, stream) = frame { unsafe {
            if (*(*stream).codecpar).codec_type == AVMediaType::AVMEDIA_TYPE_VIDEO {
                match scale.process_frame(frm, width, height) {
                    Ok(frame) => {
                        let image = video_frame_to_image(frame);
                        tx.send(DecoderMessage::VideoFrame(image)).unwrap();
                    }
                    Err(e) => panic!("{}", e)
                }
            } else if (*(*stream).codecpar).codec_type == AVMediaType::AVMEDIA_TYPE_AUDIO {
                match resample.process_frame(frm) {
                    Ok(frame) => {
                        let is_planar = av_sample_fmt_is_planar(transmute((*frame).format)) == 1;
                        let plane_mul = if is_planar { 1 } else { (*frame).ch_layout.nb_channels };
                        let size = ((*frame).nb_samples * plane_mul) as usize;

                        let samples = slice::from_raw_parts((*frame).data[0] as *const f32, size);
                        tx.send(DecoderMessage::AudioSamples(samples.to_vec())).unwrap();
                    }
                    Err(e) => panic!("{}", e)
                }
            }
        } }
    }
}
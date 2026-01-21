use crate::stream::{
    AudioSamples, DecoderInfo, MediaDecoder, MediaStreams, StreamInfo, SubtitlePacket, VideoFrame,
};
use anyhow::{Context, Result, anyhow};
use egui::{Color32, ColorImage, Vec2};
use ffmpeg_rs_raw::ffmpeg_sys_the_third::{
    AV_NOPTS_VALUE, AVMediaType, AVPixelFormat, AVSampleFormat, av_frame_move_ref,
    av_get_pix_fmt_name, av_get_sample_fmt_name, av_q2d, avcodec_get_name,
};
use ffmpeg_rs_raw::{AvFrameRef, Decoder, Demuxer, Resample, Scaler, get_frame_from_hw, rstr};
use log::{debug, error, info};
use std::mem::transmute;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

fn video_frame_to_image(frame: &AvFrameRef) -> ColorImage {
    let pixels: Vec<Color32> = match unsafe { transmute(frame.format) } {
        AVPixelFormat::AV_PIX_FMT_RGB24 | AVPixelFormat::AV_PIX_FMT_RGBA => {
            map_frame_to_pixels(frame)
        }
        _ => panic!("Pixel format not supported!"),
    };
    ColorImage {
        source_size: Vec2::new(frame.width as _, frame.height as _),
        size: [frame.width as _, frame.height as _],
        pixels,
    }
}

fn map_frame_to_pixels(frame: &AvFrameRef) -> Vec<Color32> {
    let stride = frame.linesize[0] as usize;
    let lines = frame.height as usize;
    let data = unsafe { std::slice::from_raw_parts_mut((*frame).data[0], stride * lines) };
    let bytes = match unsafe { transmute((*frame).format) } {
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

impl MediaDecoder {
    /// Creates a new media player stream and returns the receiver channel
    pub fn new(input: &str) -> Result<(Self, MediaStreams)> {
        let (tx_m, rx_m) = sync_channel(1);
        let (tx_v, rx_v) = sync_channel(10);
        let (tx_a, rx_a) = sync_channel(10_000);
        let (tx_s, rx_s) = sync_channel(10);
        let stream_v = Arc::new(AtomicUsize::new(0));
        let stream_a = Arc::new(AtomicUsize::new(0));
        let thread = Self::decoder_thread(
            input,
            stream_v.clone(),
            stream_a.clone(),
            tx_m,
            tx_v,
            tx_a,
            tx_s,
        )?;
        Ok((
            Self {
                thread,
                looping: Default::default(),
                video_stream_index: stream_v,
                audio_stream_index: stream_a,
            },
            MediaStreams {
                metadata: rx_m,
                video: rx_v,
                audio: rx_a,
                subtitle: rx_s,
            },
        ))
    }

    fn decoder_thread(
        input: &str,
        selected_video: Arc<AtomicUsize>,
        selected_audio: Arc<AtomicUsize>,
        tx_m: SyncSender<DecoderInfo>,
        tx_v: SyncSender<VideoFrame>,
        tx_a: SyncSender<AudioSamples>,
        tx_s: SyncSender<SubtitlePacket>,
    ) -> Result<JoinHandle<()>> {
        let input = input.to_string();

        let thread = std::thread::Builder::new()
            .name("media-decoder".to_string())
            .spawn(move || {
                let mut demuxer = Demuxer::new(&input).unwrap();
                let mut decoder = Decoder::new();
                decoder.enable_hw_decoder_any();

                let probe = unsafe { demuxer.probe_input() };
                let probe = match probe {
                    Ok(r) => r,
                    Err(e) => {
                        error!("Failed to probe media {}", e);
                        return;
                    }
                };

                for stream in probe.streams.iter() {
                    info!(
                        "Setting up decoder for stream #{}: {} {}x{}",
                        stream.index, stream.codec, stream.width, stream.height
                    );
                    if let Err(e) = decoder.setup_decoder(stream, None) {
                        error!("Failed to setup decoder {}", e);
                        return;
                    }
                }

                let inf = DecoderInfo {
                    bitrate: probe.bitrate as _,
                    streams: probe
                        .streams
                        .iter()
                        .map(|s| StreamInfo {
                            index: s.index as _,
                            codec: unsafe {
                                let n = avcodec_get_name(transmute(s.codec as i32));
                                rstr!(n).to_string()
                            },
                            format: unsafe {
                                if s.width != 0 {
                                    let n = av_get_pix_fmt_name(transmute(s.format as i32));
                                    rstr!(n).to_string()
                                } else {
                                    let n = av_get_sample_fmt_name(transmute(s.format as i32));
                                    rstr!(n).to_string()
                                }
                            },
                            channels: s.channels,
                            sample_rate: s.sample_rate as _,
                            width: s.width as _,
                            height: s.height as _,
                        })
                        .collect(),
                };

                // pick the best video/audio/subtitle stream
                let pick_video = probe
                    .streams
                    .iter()
                    .filter(|s| s.width != 0)
                    .max_by_key(|s| s.width * s.height)
                    .map(|s| s.index)
                    .unwrap_or(
                        probe
                            .streams
                            .iter()
                            .find(|s| s.width != 0)
                            .map(|s| s.index)
                            .unwrap_or(0),
                    );
                let pick_audio = probe
                    .streams
                    .iter()
                    .filter(|s| s.sample_rate != 0)
                    .max_by_key(|s| s.bitrate)
                    .map(|s| s.index)
                    .unwrap_or(
                        probe
                            .streams
                            .iter()
                            .find(|s| s.sample_rate != 0)
                            .map(|s| s.index)
                            .unwrap_or(0),
                    );
                selected_video.store(pick_video, Ordering::Relaxed);
                selected_audio.store(pick_audio, Ordering::Relaxed);

                if let Err(e) = tx_m.send(inf) {
                    error!("Sender closed, shutting down: {}", e);
                    return;
                }

                let mut scaler = Scaler::new();
                let mut resampler = Resample::new(AVSampleFormat::AV_SAMPLE_FMT_S32, 44_100, 2);
                loop {
                    // push some data into the payload now
                    let pkt = unsafe { demuxer.get_packet() };
                    let Ok((pkt, _)) = pkt else {
                        error!("Failed to get packet!");
                        break;
                    };

                    let v_index = selected_video.load(Ordering::Relaxed);
                    let a_index = selected_audio.load(Ordering::Relaxed);
                    if let Some(pkt) = pkt.as_ref()
                        && !(pkt.stream_index == v_index as _ || pkt.stream_index == a_index as _)
                    {
                        // skip packet, not playing
                        continue;
                    }
                    let Ok(frames) = decoder.decode_pkt(pkt.as_ref()) else {
                        error!("Failed to decode video packet!");
                        break;
                    };
                    for (frame, stream_index) in frames {
                        let stream = unsafe { demuxer.get_stream(stream_index as _).unwrap() };
                        let frame = match get_frame_from_hw(frame) {
                            Ok(f) => f,
                            Err(e) => {
                                error!("Failed to get frame from hw buffer {}", e);
                                return;
                            }
                        };
                        let q = unsafe { av_q2d((*stream).time_base) };
                        let res = match unsafe { (*(*stream).codecpar).codec_type } {
                            AVMediaType::AVMEDIA_TYPE_VIDEO => {
                                // convert to RBGA
                                let new_frame = match scaler.process_frame(
                                    &frame,
                                    frame.width as _,
                                    frame.height as _,
                                    AVPixelFormat::AV_PIX_FMT_RGBA,
                                ) {
                                    Ok(f) => f,
                                    Err(e) => {
                                        error!("Failed to process video frame! {}", e);
                                        return;
                                    }
                                };
                                tx_v.send(VideoFrame {
                                    data: video_frame_to_image(&new_frame),
                                    stream_index,
                                    pts: if frame.pts != AV_NOPTS_VALUE {
                                        frame.pts as f64 * q
                                    } else {
                                        0.0
                                    },
                                    duration: if frame.duration != AV_NOPTS_VALUE {
                                        frame.duration as f64 * q
                                    } else {
                                        0.0
                                    },
                                })
                                .map_err(|e| anyhow!("Failed to send video frame {}", e))
                            }
                            AVMediaType::AVMEDIA_TYPE_AUDIO => {
                                let Ok(frame) = resampler.process_frame(&frame) else {
                                    error!("Failed to process audio frame!");
                                    continue;
                                };
                                tx_a.send(AudioSamples {
                                    data: unsafe {
                                        // TODO: check alignment
                                        std::slice::from_raw_parts(
                                            frame.data[0] as *mut _,
                                            (frame.nb_samples * frame.ch_layout.nb_channels)
                                                as usize,
                                        )
                                    }
                                    .to_vec(),
                                    samples: frame.nb_samples as usize,
                                    stream_index,
                                    pts: if frame.pts != AV_NOPTS_VALUE {
                                        frame.pts as f64 * q
                                    } else {
                                        0.0
                                    },
                                    duration: if frame.duration != AV_NOPTS_VALUE {
                                        frame.duration as f64 * q
                                    } else {
                                        0.0
                                    },
                                })
                                .map_err(|e| anyhow!("Failed to send audio frame {}", e))
                            }
                            AVMediaType::AVMEDIA_TYPE_SUBTITLE => tx_s
                                .send(SubtitlePacket {
                                    data: vec![],
                                    stream_index,
                                })
                                .map_err(|e| anyhow!("Failed to send subtitle frame {}", e)),
                            _ => continue,
                        };
                        if let Err(e) = res {
                            error!("Sender closed, shutting down: {}", e);
                            return;
                        }
                    }
                    if pkt.is_none() {
                        info!("EOF");
                        break;
                    }
                }
            })?;
        Ok(thread)
    }
}

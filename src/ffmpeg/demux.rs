use std::ptr;
use std::time::Instant;

use anyhow::Error;
use ffmpeg_sys_the_third::*;

use crate::ffmpeg::get_ffmpeg_error_msg;
use crate::return_ffmpeg_error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, PartialEq)]
pub struct DemuxerInfo {
    pub channels: Vec<StreamInfoChannel>,
}

unsafe impl Send for DemuxerInfo {}
unsafe impl Sync for DemuxerInfo {}

impl DemuxerInfo {
    pub fn best_stream(&self, t: StreamChannelType) -> Option<&StreamInfoChannel> {
        self.channels.iter().filter(|a| a.channel_type == t).reduce(|acc, channel| {
            if channel.best_metric() > acc.best_metric() {
                channel
            } else {
                acc
            }
        })
    }

    pub fn best_video(&self) -> Option<&StreamInfoChannel> {
        self.best_stream(StreamChannelType::Video)
    }

    pub fn best_audio(&self) -> Option<&StreamInfoChannel> {
        self.best_stream(StreamChannelType::Audio)
    }

    pub fn best_subtitle(&self) -> Option<&StreamInfoChannel> {
        self.best_stream(StreamChannelType::Subtitle)
    }

    pub unsafe fn is_best_stream(&self, stream: *mut AVStream) -> bool {
        match (*(*stream).codecpar).codec_type {
            AVMediaType::AVMEDIA_TYPE_VIDEO =>
                (*stream).index == self.best_video().map_or(usize::MAX, |r| r.index) as libc::c_int,
            AVMediaType::AVMEDIA_TYPE_AUDIO =>
                (*stream).index == self.best_audio().map_or(usize::MAX, |r| r.index) as libc::c_int,
            AVMediaType::AVMEDIA_TYPE_SUBTITLE =>
                (*stream).index == self.best_video().map_or(usize::MAX, |r| r.index) as libc::c_int,
            _ => false,
        }
    }
}

impl Display for DemuxerInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Demuxer Info:")?;
        for c in &self.channels {
            write!(f, "\n{}", c)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum StreamChannelType {
    Video,
    Audio,
    Subtitle,
}

impl Display for StreamChannelType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                StreamChannelType::Video => "video",
                StreamChannelType::Audio => "audio",
                StreamChannelType::Subtitle => "subtitle",
            }
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StreamInfoChannel {
    pub index: usize,
    pub channel_type: StreamChannelType,
    pub width: usize,
    pub height: usize,
    pub fps: f32,
    pub sample_rate: usize,
    pub format: usize,
    pub bitrate: usize,
    pub duration: f32,
}

impl StreamInfoChannel {
    pub fn best_metric(&self) -> f32 {
        match self.channel_type {
            StreamChannelType::Video => self.bitrate as f32,
            StreamChannelType::Audio => self.bitrate as f32,
            StreamChannelType::Subtitle => self.bitrate as f32,
        }
    }
}

impl Display for StreamInfoChannel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} #{}: size={}x{},fps={}",
            self.channel_type, self.index, self.width, self.height, self.fps
        )
    }
}

pub(crate) struct Demuxer {
    ctx: *mut AVFormatContext,
    input: String,
    started: Instant,
}

unsafe impl Send for Demuxer {}

unsafe impl Sync for Demuxer {}

impl Demuxer {
    pub fn new(input: &str) -> Self {
        unsafe {
            let ps = avformat_alloc_context();

            Self {
                ctx: ps,
                input: input.to_string(),
                started: Instant::now(),
            }
        }
    }

    pub unsafe fn probe_input(&mut self) -> Result<DemuxerInfo, Error> {
        let ret = avformat_open_input(
            &mut self.ctx,
            format!("{}\0", self.input).as_ptr() as *const libc::c_char,
            ptr::null_mut(),
            ptr::null_mut(),
        );
        return_ffmpeg_error!(ret);

        if avformat_find_stream_info(self.ctx, ptr::null_mut()) < 0 {
            return Err(Error::msg("Could not find stream info"));
        }
        av_dump_format(self.ctx, 0, ptr::null_mut(), 0);

        let mut channel_infos = vec![];

        for n in 0..(*self.ctx).nb_streams as usize {
            let stream = *(*self.ctx).streams.add(n);
            match (*(*stream).codecpar).codec_type {
                AVMediaType::AVMEDIA_TYPE_VIDEO => {
                    channel_infos.push(StreamInfoChannel {
                        index: (*stream).index as usize,
                        channel_type: StreamChannelType::Video,
                        width: (*(*stream).codecpar).width as usize,
                        height: (*(*stream).codecpar).height as usize,
                        fps: av_q2d((*stream).avg_frame_rate) as f32,
                        format: (*(*stream).codecpar).format as usize,
                        sample_rate: 0,
                        bitrate: (*(*stream).codecpar).bit_rate as usize,
                        duration: av_q2d((*stream).time_base) as f32 * (*stream).duration as f32,
                    });
                }
                AVMediaType::AVMEDIA_TYPE_UNKNOWN => {}
                AVMediaType::AVMEDIA_TYPE_AUDIO => {
                    channel_infos.push(StreamInfoChannel {
                        index: (*stream).index as usize,
                        channel_type: StreamChannelType::Audio,
                        width: (*(*stream).codecpar).width as usize,
                        height: (*(*stream).codecpar).height as usize,
                        fps: 0.0,
                        format: (*(*stream).codecpar).format as usize,
                        sample_rate: (*(*stream).codecpar).sample_rate as usize,
                        bitrate: (*(*stream).codecpar).bit_rate as usize,
                        duration: av_q2d((*stream).time_base) as f32 * (*stream).duration as f32,
                    });
                }
                AVMediaType::AVMEDIA_TYPE_DATA => {}
                AVMediaType::AVMEDIA_TYPE_SUBTITLE => {}
                AVMediaType::AVMEDIA_TYPE_ATTACHMENT => {}
                AVMediaType::AVMEDIA_TYPE_NB => {}
                _ => {}
            }
        }

        let info = DemuxerInfo {
            channels: channel_infos
        };
        Ok(info)
    }

    pub unsafe fn get_packet(&mut self) -> Result<(*mut AVPacket, *mut AVStream), Error> {
        let pkt: *mut AVPacket = av_packet_alloc();
        let ret = av_read_frame(self.ctx, pkt);
        if ret == AVERROR_EOF {
            return Ok((ptr::null_mut(), ptr::null_mut()));
        }
        if ret < 0 {
            let msg = get_ffmpeg_error_msg(ret);
            return Err(Error::msg(msg));
        }
        let stream = *(*self.ctx).streams.add((*pkt).stream_index as usize);
        let pkg = (pkt, stream);
        Ok(pkg)
    }
}

impl Drop for Demuxer {
    fn drop(&mut self) {
        unsafe {
            avformat_free_context(self.ctx);
            self.ctx = ptr::null_mut();
        }
    }
}

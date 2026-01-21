use anyhow::bail;
use egui::ColorImage;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicIsize};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::JoinHandle;

#[cfg(feature = "ffmpeg")]
mod ffmpeg;

#[derive(Clone, Debug)]
pub struct DecoderInfo {
    pub bitrate: u64,
    pub duration: f32,
    pub streams: Vec<StreamInfo>,
}

#[derive(Clone, Debug)]
pub enum StreamType {
    Video,
    Audio,
    Subtitle,
}

#[derive(Clone, Debug)]
pub struct StreamInfo {
    pub r#type: StreamType,
    pub index: i32,
    pub codec: String,
    pub format: String,
    pub channels: u8,
    pub sample_rate: u32,
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub language: Option<String>,
}

impl Display for StreamInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.r#type {
            StreamType::Video => {
                write!(
                    f,
                    "V #{}: {} {}x{}@{}",
                    self.index, self.codec, self.width, self.height, self.format,
                )
            }
            StreamType::Audio => {
                write!(
                    f,
                    "A #{}: {} {}ch {}@{} {}",
                    self.index,
                    self.codec,
                    self.channels,
                    self.format,
                    self.sample_rate,
                    self.language.as_ref().map(|s| s.as_str()).unwrap_or("unk")
                )
            }
            StreamType::Subtitle => {
                write!(
                    f,
                    "S #{}: {} {}",
                    self.index,
                    self.codec,
                    self.language.as_ref().map(|s| s.as_str()).unwrap_or("unk")
                )
            }
        }
    }
}

#[derive(Clone)]
pub struct VideoFrame {
    /// Frame as an egui image
    pub data: ColorImage,
    /// The stream index this frame belongs to
    pub stream_index: i32,
    /// Presentation timestamp
    pub pts: f64,
    /// Duration this frame should be shown
    pub duration: f64,
}

#[derive(Clone)]
pub struct AudioSamples {
    /// Raw audio samples
    pub data: Vec<i32>,
    /// The stream index this frame belongs to
    pub stream_index: i32,
    /// Presentation timestamp
    pub pts: f64,
    /// Duration this frame should be shown
    pub duration: f64,
    /// Number of samples in [data]
    pub samples: usize,
}

#[derive(Clone)]
pub struct SubtitlePacket {
    pub data: Vec<u8>,
    pub stream_index: i32,
}

/// Container holding the channels for each media type
pub struct MediaStreams {
    pub metadata: Receiver<DecoderInfo>,
    pub video: Receiver<VideoFrame>,
    pub audio: Receiver<AudioSamples>,
    pub subtitle: Receiver<SubtitlePacket>,
}

/// Media stream producer, creates a stream of decoded data from a path or url.
/// To shut down the media stream you must drop the receiver channel(s)
pub struct MediaDecoder {
    /// Thread which decodes the media stream
    thread: JoinHandle<()>,
    /// If the stream should loop
    pub looping: Arc<AtomicBool>,
    /// The index of the primary video stream being decoded
    pub video_stream_index: Arc<AtomicIsize>,
    /// The index of the primary audio stream being decoded
    pub audio_stream_index: Arc<AtomicIsize>,
    /// The index of the primary subtitle stream being decoded
    pub subtitle_stream_index: Arc<AtomicIsize>,
}

impl MediaDecoder {
    /// Creates a new media player stream and returns the receiver channel
    pub fn new(input: &str) -> anyhow::Result<(Self, MediaStreams)> {
        let (tx_m, rx_m) = sync_channel(1);
        let (tx_v, rx_v) = sync_channel(10);
        let (tx_a, rx_a) = sync_channel(1_000);
        let (tx_s, rx_s) = sync_channel(10);
        let stream_v = Arc::new(AtomicIsize::new(-1));
        let stream_a = Arc::new(AtomicIsize::new(-1));
        let stream_s = Arc::new(AtomicIsize::new(-1));
        let thread = Self::spawn_decoder(
            input,
            stream_v.clone(),
            stream_a.clone(),
            stream_s.clone(),
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
                subtitle_stream_index: stream_s,
            },
            MediaStreams {
                metadata: rx_m,
                video: rx_v,
                audio: rx_a,
                subtitle: rx_s,
            },
        ))
    }

    #[allow(unused_variables)]
    fn spawn_decoder(
        input: &str,
        selected_video: Arc<AtomicIsize>,
        selected_audio: Arc<AtomicIsize>,
        selected_subtitle: Arc<AtomicIsize>,
        tx_m: SyncSender<DecoderInfo>,
        tx_v: SyncSender<VideoFrame>,
        tx_a: SyncSender<AudioSamples>,
        tx_s: SyncSender<SubtitlePacket>,
    ) -> anyhow::Result<JoinHandle<()>> {
        #[cfg(feature = "ffmpeg")]
        return ffmpeg::ffmpeg_decoder_thread(
            input,
            selected_video,
            selected_audio,
            selected_subtitle,
            tx_m,
            tx_v,
            tx_a,
            tx_s,
        );
        bail!("No decoder impl available!")
    }
}

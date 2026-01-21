use egui::ColorImage;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::thread::JoinHandle;

#[cfg(feature = "ffmpeg")]
mod ffmpeg;

#[derive(Clone, Debug)]
pub struct DecoderInfo {
    pub bitrate: u64,
    pub streams: Vec<StreamInfo>,
}

#[derive(Clone, Debug)]
pub struct StreamInfo {
    pub index: i32,
    pub codec: String,
    pub format: String,
    pub channels: u8,
    pub sample_rate: u32,
    pub width: u32,
    pub height: u32,
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
    pub video_stream_index: Arc<AtomicUsize>,
    /// The index of the primary audio stream being decoded
    pub audio_stream_index: Arc<AtomicUsize>,
}

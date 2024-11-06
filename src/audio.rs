use cpal::traits::HostTrait;
use cpal::{Stream, SupportedStreamConfig};
use std::collections::VecDeque;
use std::sync::atomic::AtomicU8;
use std::sync::{Arc, Mutex};

/// The playback device. Needs to be initialized (and kept alive!) for use by a [`Player`].
pub struct AudioDevice(pub(crate) cpal::Device);

impl AudioDevice {
    pub fn from_device(device: cpal::Device) -> Self {
        Self(device)
    }

    /// Create a new [`AudioDevice`] from an existing [`cpal::Host`]. An [`AudioDevice`] is required for using audio.
    pub fn from_subsystem(audio_sys: &cpal::Host) -> Result<AudioDevice, String> {
        if let Some(dev) = audio_sys.default_output_device() {
            Ok(AudioDevice(dev))
        } else {
            Err("No default audio device".to_owned())
        }
    }

    /// Create a new [`AudioDevice`]. Creates an [`cpal::Host`]. An [`AudioDevice`] is required for using audio.
    pub fn new() -> Result<AudioDevice, String> {
        let host = cpal::default_host();
        Self::from_subsystem(&host)
    }
}

pub(crate) struct PlayerAudioStream {
    pub device: AudioDevice,
    pub config: SupportedStreamConfig,
    pub stream: Stream,
    pub player_state: Arc<AtomicU8>,
    pub buffer: Arc<Mutex<AudioBuffer>>,
}

pub(crate) struct AudioBuffer {
    channels: u8,
    sample_rate: u32,
    // buffered samples
    pub samples: VecDeque<f32>,
    /// Video position in seconds
    pub video_pos: f32,
    /// Audio position in seconds
    pub audio_pos: f32,
}

impl AudioBuffer {
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        Self {
            channels,
            sample_rate,
            samples: VecDeque::new(),
            video_pos: 0.0,
            audio_pos: 0.0,
        }
    }

    pub fn add_samples(&mut self, samples: Vec<f32>) {
        self.samples.extend(samples);
    }

    pub fn take_samples(&mut self, n: usize) -> Vec<f32> {
        assert_eq!(0, n % self.channels as usize, "Must be a multiple of 2");
        let will_drain = self.samples.drain(..n.min(self.samples.len()));
        self.audio_pos += will_drain.len() as f32 / self.sample_rate as f32 / self.channels as f32;
        will_drain.collect()
    }

    pub fn audio_delay_samples(&self) -> isize {
        let samples_f32 = self.sample_rate as f32 * (self.video_pos - self.audio_pos);
        samples_f32 as isize * self.channels as isize
    }
}

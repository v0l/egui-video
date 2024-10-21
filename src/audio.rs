use cpal::traits::HostTrait;

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

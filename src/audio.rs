use crate::PlayerState;
use crate::stream::AudioSamples;
use anyhow::Result;
use anyhow::bail;
use bungee_sys::BungeeStream;
use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{SampleFormat, Stream};
use log::{info, trace};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicI16, AtomicI64, AtomicU8, AtomicU16, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, mpsc};

/// The playback device. Needs to be initialized (and kept alive!) for use by a [`Player`].
pub struct AudioDevice(pub(crate) cpal::Device);

pub struct AudioDeviceHandle {
    device: AudioDevice,
    stream: Stream,
}

impl crate::AudioDevice for AudioDeviceHandle {}

impl AudioDevice {
    pub fn from_device(device: cpal::Device) -> Self {
        Self(device)
    }

    /// Create a new [`AudioDevice`] from an existing [`cpal::Host`]. An [`AudioDevice`] is required for using audio.
    pub fn from_subsystem(audio_sys: &cpal::Host) -> Result<AudioDevice> {
        if let Some(dev) = audio_sys.default_output_device() {
            Ok(AudioDevice(dev))
        } else {
            bail!("No default audio device available");
        }
    }

    /// Create a new [`AudioDevice`]. Creates an [`cpal::Host`]. An [`AudioDevice`] is required for using audio.
    pub fn new() -> Result<AudioDevice> {
        let host = cpal::default_host();
        Self::from_subsystem(&host)
    }

    pub fn open_default_audio_stream(
        _volume: Arc<AtomicU16>,
        _mute: Arc<AtomicBool>,
        state: Arc<AtomicU8>,
        _speed: Arc<AtomicI16>,
        position: Arc<AtomicI64>,
        rx: Receiver<AudioSamples>,
    ) -> Result<AudioDeviceHandle> {
        let device = AudioDevice::new()?;
        let cfg = device.0.default_output_config()?;
        info!(
            "Default audio device config: {}Hz, {}ch, {:?}",
            cfg.sample_rate(),
            cfg.channels(),
            cfg.sample_format()
        );

        let channels = cfg.channels() as u8;
        let mut simple_queue = VecDeque::new();
        let _bungee_stream = BungeeWrapper::new(channels, cfg.sample_rate() as _)?;
        let stream = device.0.build_output_stream_raw(
            &cfg.config(),
            SampleFormat::I32,
            move |data: &mut cpal::Data, _info: &cpal::OutputCallbackInfo| {
                if data.len() == 0 {
                    return;
                }
                let dst: &mut [i32] = data.as_slice_mut().unwrap();
                dst.fill(0);
                let state = state.load(Ordering::Relaxed);
                if state == PlayerState::Stopped as u8 || state == PlayerState::Paused as u8 {
                    return;
                }
                let current_pts = position.load(Ordering::Relaxed) as f64 / 1000.0;
                let _frame_pts = current_pts;

                // fill queue until dst is satisfied
                while simple_queue.len() < dst.len() {
                    // take samples from channel
                    match rx.try_recv() {
                        Ok(m) => {
                            simple_queue.extend(m.data);
                        }
                        Err(mpsc::TryRecvError::Empty) => {
                            trace!("Audio underrun!");
                            continue;
                        }
                        Err(_) => {
                            break;
                        }
                    }
                }
                if simple_queue.len() >= dst.len() {
                    let drain_samples = simple_queue.drain(..dst.len()).collect::<Vec<_>>();
                    dst.copy_from_slice(&drain_samples);
                }
            },
            move |e| {
                panic!("{}", e);
            },
            None,
        )?;

        Ok(AudioDeviceHandle { device, stream })
    }
}

pub struct BungeeWrapper {
    ctx: *mut BungeeStream,
}

unsafe impl Send for BungeeWrapper {}

impl BungeeWrapper {
    pub fn new(channels: u8, sample_rate: u32) -> Result<BungeeWrapper> {
        let ctx = {
            let samples = bungee_sys::SampleRates {
                input: sample_rate as _,
                output: sample_rate as _,
            };
            let stretcher = bungee_sys::stretcher::create(samples, channels as _, 2);
            if stretcher.is_null() {
                bail!("Failed to create stretcher");
            }
            let ctx = bungee_sys::stream::create(stretcher, channels as _, 8192);
            if ctx.is_null() {
                bail!("Failed to create stretcher stream");
            }
            ctx
        };
        Ok(Self { ctx })
    }

    pub fn process(
        &mut self,
        frame: AudioSamples,
        out_data: &mut [f32],
        out_samples: usize,
    ) -> Result<()> {
        let ret = bungee_sys::stream::process(
            self.ctx,
            frame.data.as_ptr() as *const _,
            out_data.as_mut_ptr() as *mut _,
            frame.samples as _,
            out_samples as _,
            1.0,
        );
        if ret != 0 {
            bail!("Failed to process audio");
        }
        Ok(())
    }
}

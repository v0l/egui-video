use crate::ffmpeg_sys_the_third::{AVPacket, AVStream};
use anyhow::Error;
use ffmpeg_rs_raw::{Demuxer, DemuxerInfo};
use m3u8_rs::{MediaSegment, Playlist, VariantStream};
use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::time::Duration;
use url::Url;

pub struct HlsStream {
    url: String,
    playlist: Option<Playlist>,
    current_variant: Option<VariantStream>,
    demuxer_map: HashMap<String, Demuxer>,
}

impl HlsStream {
    pub fn new(url: &str) -> Self {
        Self {
            url: url.to_string(),
            playlist: None,
            current_variant: None,
            demuxer_map: HashMap::new(),
        }
    }

    pub fn load(&mut self) -> Result<(), Error> {
        let rsp = ureq::get(&self.url).call()?;

        let mut bytes = Vec::new();
        rsp.into_reader().read_to_end(&mut bytes)?;

        let parsed = m3u8_rs::parse_playlist(&bytes);
        match parsed {
            Ok((_, playlist)) => {
                self.playlist = Some(playlist);
                Ok(())
            }
            Err(e) => {
                anyhow::bail!("{}", e);
            }
        }
    }

    pub fn variants(&self) -> Vec<VariantStream> {
        if let Some(Playlist::MasterPlaylist(ref pl)) = self.playlist {
            pl.variants
                .iter()
                .map(|v| {
                    let mut vc = v.clone();
                    let u: Url = self.url.parse().unwrap();
                    vc.uri = u.join(&vc.uri).unwrap().to_string();
                    vc
                })
                .collect()
        } else {
            vec![VariantStream::default()]
        }
    }

    pub fn set_variant(&mut self, var: VariantStream) {
        self.current_variant = Some(var);
    }

    /// Pick a variant automatically
    pub fn auto_variant(&self) -> Option<VariantStream> {
        self.variants().into_iter().next()
    }

    pub fn current_variant(&self) -> Option<VariantStream> {
        if let Some(variant) = &self.current_variant {
            Some(variant.clone())
        } else {
            self.auto_variant()
        }
    }

    fn variant_demuxer(&mut self, var: &VariantStream) -> Result<&mut Demuxer, Error> {
        if !self.demuxer_map.contains_key(&var.uri) {
            let demux = Demuxer::new_custom_io(
                VariantReader::new(var.clone()),
                Some("video.ts".to_string()),
            );
            self.demuxer_map.insert(var.uri.clone(), demux);
        }
        Ok(self
            .demuxer_map
            .get_mut(&var.uri)
            .expect("demuxer not found"))
    }

    fn current_demuxer(&mut self) -> Result<&mut Demuxer, Error> {
        let v = if let Some(v) = self.current_variant() {
            v
        } else {
            anyhow::bail!("no variants available");
        };
        self.variant_demuxer(&v)
    }
}

struct VariantReader {
    variant: VariantStream,
    last_segment: Option<MediaSegment>,
    buffer: Vec<u8>,
}

impl VariantReader {
    fn new(variant: VariantStream) -> Self {
        Self {
            variant,
            last_segment: None,
            buffer: Vec::new(),
        }
    }

    pub fn read_next_segment(&mut self) -> Result<Option<Box<dyn Read>>, Error> {
        let req = ureq::get(&self.variant.uri).call()?;

        let mut bytes = Vec::new();
        req.into_reader().read_to_end(&mut bytes)?;
        let parsed = m3u8_rs::parse_playlist(&bytes);
        let playlist = match parsed {
            Ok((_, playlist)) => match playlist {
                Playlist::MasterPlaylist(_) => {
                    anyhow::bail!("Unexpected MasterPlaylist response");
                }
                Playlist::MediaPlaylist(mp) => mp,
            },
            Err(e) => {
                anyhow::bail!("{}", e);
            }
        };

        if let Some(next_seg) = playlist.segments.last() {
            if let Some(ref last) = self.last_segment {
                if last.eq(next_seg) {
                    return Ok(None);
                }
            }

            self.last_segment = Some(next_seg.clone());
            let u: Url = self.variant.uri.parse()?;

            let req = ureq::get(u.join(&next_seg.uri)?.as_ref()).call()?;
            Ok(Some(req.into_reader()))
        } else {
            Ok(None)
        }
    }
}

impl Read for VariantReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        while self.buffer.len() < buf.len() {
            if let Some(mut s) = self
                .read_next_segment()
                .map_err(|e| std::io::Error::other(e))?
            {
                let mut buf = Vec::new();
                let len = s.read_to_end(&mut buf)?;
                self.buffer.extend(buf[..len].iter().as_slice());
            } else {
                std::thread::sleep(Duration::from_millis(1000));
            }
        }
        let cpy = buf.len().min(self.buffer.len());
        let mut z = 0;
        for x in self.buffer.drain(..cpy) {
            buf[z] = x;
            z += 1;
        }
        eprintln!("write: {}", cpy);
        Ok(cpy)
    }
}

pub trait DemuxerIsh {
    unsafe fn probe_input(&mut self) -> Result<DemuxerInfo, Error>;
    unsafe fn get_packet(&mut self) -> Result<(*mut AVPacket, *mut AVStream), Error>;
}

impl DemuxerIsh for HlsStream {
    unsafe fn probe_input(&mut self) -> Result<DemuxerInfo, Error> {
        if self.playlist.is_none() {
            self.load()?;
        }

        let demux = self.current_demuxer()?;
        demux.probe_input()
    }

    unsafe fn get_packet(&mut self) -> Result<(*mut AVPacket, *mut AVStream), Error> {
        let demux = self.current_demuxer()?;
        demux.get_packet()
    }
}

impl DemuxerIsh for Demuxer {
    unsafe fn probe_input(&mut self) -> Result<DemuxerInfo, Error> {
        self.probe_input()
    }

    unsafe fn get_packet(&mut self) -> Result<(*mut AVPacket, *mut AVStream), Error> {
        self.get_packet()
    }
}

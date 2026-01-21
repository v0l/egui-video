use anyhow::Result;
use ffmpeg_rs_raw::{AvPacketRef, Demuxer, DemuxerInfo};
use itertools::Itertools;
use log::info;
use m3u8_rs::{MediaPlaylist, MediaPlaylistType, MediaSegment, Playlist, VariantStream};
use std::collections::HashMap;
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

    pub fn load(&mut self) -> Result<()> {
        let bytes = ureq::get(&self.url).call()?.body_mut().read_to_vec()?;

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

    /// Return variants from master playlist
    pub fn variants(&self) -> Vec<VariantStream> {
        if let Some(Playlist::MasterPlaylist(ref pl)) = self.playlist {
            pl.variants
                .iter()
                .map(|v| {
                    // patch url to be fully defined (if relative to master)
                    if v.uri.starts_with("http") {
                        v.clone()
                    } else {
                        let mut vc = v.clone();
                        let u: Url = self.url.parse().unwrap();
                        vc.uri = u.join(&vc.uri).unwrap().to_string();
                        vc
                    }
                })
                .collect()
        } else {
            // TODO: map to single variant
            vec![VariantStream::default()]
        }
    }

    pub fn set_variant(&mut self, var: VariantStream) {
        self.current_variant = Some(var);
    }

    /// Pick a variant automatically
    pub fn auto_variant(&self) -> Option<VariantStream> {
        self.variants()
            .into_iter()
            .sorted_by(|a, b| a.bandwidth.cmp(&b.bandwidth).reverse())
            .next()
    }

    pub fn current_variant(&self) -> Option<VariantStream> {
        if let Some(variant) = &self.current_variant {
            Some(variant.clone())
        } else {
            self.auto_variant()
        }
    }

    fn variant_demuxer(&mut self, var: &VariantStream) -> Result<&mut Demuxer> {
        if !self.demuxer_map.contains_key(&var.uri) {
            let demux =
                Demuxer::new_custom_io(VariantReader::new(var.clone()), Some(var.uri.clone()))?;
            self.demuxer_map.insert(var.uri.clone(), demux);
        }
        Ok(self
            .demuxer_map
            .get_mut(&var.uri)
            .expect("demuxer not found"))
    }

    fn current_demuxer(&mut self) -> Result<&mut Demuxer> {
        let v = if let Some(v) = self.current_variant() {
            v
        } else {
            anyhow::bail!("no variants available");
        };
        self.variant_demuxer(&v)
    }
}

struct VariantReader {
    /// The type of stream (Live/VOD)
    kind: MediaPlaylistType,
    /// The current variant stream
    variant: VariantStream,
    /// List of already loaded segments
    prev: HashMap<String, MediaSegment>,
    /// Internal buffer of stream data
    buffer: Vec<u8>,
}

impl VariantReader {
    fn new(variant: VariantStream) -> Self {
        Self {
            kind: Default::default(),
            variant,
            prev: HashMap::new(),
            buffer: Vec::new(),
        }
    }

    fn load_playlist(&self) -> Result<MediaPlaylist> {
        let bytes = ureq::get(&self.variant.uri)
            .call()?
            .body_mut()
            .read_to_vec()?;
        let parsed = m3u8_rs::parse_playlist(&bytes);
        match parsed {
            Ok((_, playlist)) => match playlist {
                Playlist::MasterPlaylist(_) => {
                    anyhow::bail!("Unexpected MasterPlaylist response");
                }
                Playlist::MediaPlaylist(mp) => Ok(mp),
            },
            Err(e) => {
                anyhow::bail!("{}", e);
            }
        }
    }

    /// Return the next segment which should be loaded
    fn get_next_segment<'a>(&self, playlist: &'a MediaPlaylist) -> Option<&'a MediaSegment> {
        for seg in &playlist.segments {
            if !self.prev.contains_key(&seg.uri) {
                return Some(seg);
            }
        }
        None
    }

    pub fn read_next_segment(&mut self) -> Result<Option<Box<dyn Read>>> {
        let playlist = self.load_playlist()?;
        if let Some(pk) = &playlist.playlist_type {
            self.kind = pk.clone();
        }

        if let Some(next_seg) = self.get_next_segment(&playlist) {
            let u: Url = self.variant.uri.parse()?;

            let u = u.join(&next_seg.uri)?;
            info!("Loading segment: {}", &u);
            let req = ureq::get(u.as_ref()).call()?;

            self.prev.insert(next_seg.uri.clone(), next_seg.clone());
            Ok(Some(Box::new(req.into_body().into_reader())))
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
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        let cpy = buf.len().min(self.buffer.len());
        let mut z = 0;
        for x in self.buffer.drain(..cpy) {
            buf[z] = x;
            z += 1;
        }
        Ok(cpy)
    }
}

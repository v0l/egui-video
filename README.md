# egui-video, a video playing library for [`egui`](https://github.com/emilk/egui)
[![crates.io](https://img.shields.io/crates/v/egui-video)](https://crates.io/crates/egui-video)
[![docs](https://docs.rs/egui-video/badge.svg)](https://docs.rs/egui-video/latest/egui_video/)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/n00kii/egui-video/blob/main/README.md)

https://github.com/n00kii/egui-video/assets/57325298/c618ff0a-9ad2-4cf0-b14a-dda65dc54b23

Plays videos in egui from any source FFMPEG supports.

## Dependencies:
 - requires ffmpeg 6 or 7. follow the build instructions [here](https://github.com/zmwangx/rust-ffmpeg/wiki/Notes-on-building)

## Usage:
```rust
// called once (creating a player)
let mut player = egui_video::Player::new(ctx, my_media_path)?;
player.play();
// called every frame (showing the player)
player.ui(ui, player.size);
```
## Contributions
are welcome :)
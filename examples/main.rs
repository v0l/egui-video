use eframe::NativeOptions;
use egui::{CentralPanel, DragValue, Grid, Slider, TextEdit, Window};
use egui_video::{AudioDevice, Player};
fn main() {
    let _ = eframe::run_native(
        "app",
        NativeOptions::default(),
        Box::new(|_| Ok(Box::new(App::default()))),
    );
}
struct App {
    audio_device: AudioDevice,
    player: Option<Player>,

    media_path: String,
    stream_size_scale: f32,
    seek_frac: f32,
}

impl Default for App {
    fn default() -> Self {
        Self {
            audio_device: AudioDevice::new().unwrap(),
            media_path: "https://test-streams.mux.dev/tos_ismc/main.m3u8".to_string(),
            stream_size_scale: 1.,
            seek_frac: 0.,
            player: None,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.add_enabled_ui(!self.media_path.is_empty(), |ui| {
                    if ui.button("load").clicked() {
                        let mut p = Player::new(ctx, &self.media_path.replace("\"", ""));
                        p.start();
                        self.player = Some(p);
                    }
                });
                ui.add_enabled_ui(!self.media_path.is_empty(), |ui| {
                    if ui.button("clear").clicked() {
                        self.player = None;
                    }
                });

                let tedit_resp = ui.add_sized(
                    [ui.available_width(), ui.available_height()],
                    TextEdit::singleline(&mut self.media_path)
                        .hint_text("click to set path"),
                );
            });
            ui.separator();
            if let Some(player) = self.player.as_mut() {
                Window::new("info").show(ctx, |ui| {
                    Grid::new("info_grid").show(ui, |ui| {
                        ui.label("frame rate");
                        ui.label(player.framerate().to_string());
                        ui.end_row();

                        ui.label("size");
                        ui.label(format!("{}x{}", player.size().0, player.size().1));
                        ui.end_row();

                        ui.label("elapsed / duration");
                        ui.label(format!("{}/{}", player.elapsed(), 0));
                        ui.end_row();

                        ui.label("state");
                        ui.label(format!("{:?}", player.state()));
                        ui.end_row();

                        ui.label("has audio?");
                        ui.label(true.to_string());
                        ui.end_row();

                        ui.label("has subtitles?");
                        ui.label(false.to_string());
                        ui.end_row();
                    });
                });
                Window::new("controls").show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("seek to:").clicked() {
                            player.seek(self.seek_frac);
                        }
                        ui.add(
                            DragValue::new(&mut self.seek_frac)
                                .speed(0.05)
                                .range(0.0..=1.0),
                        );
                        //ui.checkbox(&mut player.options.looping, "loop");
                    });
                    ui.horizontal(|ui| {
                        ui.label("size scale");
                        ui.add(Slider::new(&mut self.stream_size_scale, 0.0..=2.));
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("play").clicked() {
                            player.start()
                        }
                        if ui.button("unpause").clicked() {
                            player.resume();
                        }
                        if ui.button("pause").clicked() {
                            player.pause();
                        }
                        if ui.button("stop").clicked() {
                            player.stop();
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("volume");
                        let mut volume = player.volume();
                        if ui
                            .add(Slider::new(
                                &mut volume,
                                0..=u8::MAX,
                            ))
                            .changed()
                        {
                            player.set_volume(volume);
                        };
                    });
                });

                player.ui(ui, ui.available_size());
            }
        });
    }
}

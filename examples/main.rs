use eframe::NativeOptions;
use egui::{vec2, CentralPanel, TextEdit, Widget};
use egui_video::{AudioDevice, Player, PlayerControls};

fn main() {
    env_logger::init();
    let mut opt = NativeOptions::default();
    opt.viewport.inner_size = Some(vec2(1280., 740.));

    let _ = eframe::run_native("app", opt, Box::new(|_| Ok(Box::new(App::default()))));
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
                        p.set_debug(true);
                        self.player = Some(p);
                    }
                });
                ui.add_enabled_ui(!self.media_path.is_empty(), |ui| {
                    if ui.button("clear").clicked() {
                        self.player = None;
                    }
                });

                ui.add_sized(
                    [ui.available_width(), ui.available_height()],
                    TextEdit::singleline(&mut self.media_path).hint_text("click to set path"),
                );
            });
            ui.separator();
            if let Some(player) = self.player.as_mut() {
                player.ui(ui);
            }
        });
    }
}

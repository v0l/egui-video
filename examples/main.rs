use eframe::NativeOptions;
use egui::{CentralPanel, Response, TextEdit, Ui, ViewportBuilder, Widget};
use egui_video::{AudioDevice, PlaybackInfo, PlaybackUpdate, Player, PlayerOverlay};

struct EmptyOverlay;

impl PlayerOverlay for EmptyOverlay {
    fn show(&self, ui: &mut Ui, frame_response: &Response, p: &PlaybackInfo) -> PlaybackUpdate {
        Default::default()
    }
}

fn main() {
    env_logger::init();
    let mut opt = NativeOptions::default();
    opt.viewport = ViewportBuilder::default().with_inner_size([1270.0, 740.0]);

    let _ = eframe::run_native("app", opt, Box::new(|cc| Ok(Box::new(App::default()))));
}

struct App {
    audio_device: AudioDevice,
    player: Option<Player<EmptyOverlay>>,

    media_path: String,
}

impl Default for App {
    fn default() -> Self {
        Self {
            audio_device: AudioDevice::new().unwrap(),
            media_path:
                "https://api-core.zap.stream/537a365c-f1ec-44ac-af10-22d14a7319fb/hls/live.m3u8"
                    .to_string(),
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
                        if let Ok(p) =
                            Player::new(EmptyOverlay, ctx, &self.media_path.replace("\"", ""))
                        {
                            self.player = Some(p);
                        }
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

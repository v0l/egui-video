use eframe::NativeOptions;
use egui::{CentralPanel, TextEdit, ViewportBuilder, Widget};
use egui_video::{DefaultOverlay, Player};

fn main() {
    env_logger::init();
    let mut opt = NativeOptions::default();
    opt.viewport = ViewportBuilder::default().with_inner_size([1270.0, 740.0]);

    let _ = eframe::run_native("app", opt, Box::new(|cc| Ok(Box::new(App::default()))));
}

struct App {
    player: Option<Player<DefaultOverlay>>,

    media_path: String,
}

impl Default for App {
    fn default() -> Self {
        Self {
            media_path:
                "/home/kieran/Downloads/www.UIndex.org    -    Fallout S02E04 The the Snow 2160p AMZN WEB-DL DD 5 1 Atmos H 265-playWEB.mkv"
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
                        if let Ok(mut p) =
                            Player::new(DefaultOverlay, ctx, &self.media_path.replace("\"", ""))
                        {
                            p.enable_keybinds(true);
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

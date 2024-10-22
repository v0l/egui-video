use egui::{Align2, Color32, Margin, Pos2};

mod ass;

#[derive(Debug)]
pub struct Subtitle {
    pub text: String,
    pub fade: FadeEffect,
    pub alignment: Align2,
    pub primary_fill: Color32,
    pub position: Option<Pos2>,
    pub font_size: f32,
    pub margin: Margin,
    pub remaining_duration_ms: i64,
}

// todo, among others
// struct Transition<'a> {
//     offset_start_ms: i64,
//     offset_end_ms: i64,
//     accel: f64,
//     field: SubtitleField<'a>,
// }

enum SubtitleField<'a> {
    Fade(FadeEffect),
    Alignment(Align2),
    PrimaryFill(Color32),
    Position(Pos2),
    Undefined(&'a str),
}

#[derive(Debug, Default)]
pub struct FadeEffect {
    _fade_in_ms: i64,
    _fade_out_ms: i64,
}

impl Default for Subtitle {
    fn default() -> Self {
        Self {
            text: String::new(),
            fade: FadeEffect {
                _fade_in_ms: 0,
                _fade_out_ms: 0,
            },
            remaining_duration_ms: 0,
            font_size: 30.,
            margin: Margin::same(85.),
            alignment: Align2::CENTER_CENTER,
            primary_fill: Color32::WHITE,
            position: None,
        }
    }
}

impl Subtitle {
    pub(crate) fn from_text(text: &str) -> Self {
        Subtitle::default().with_text(text)
    }
    pub(crate) fn with_text(mut self, text: &str) -> Self {
        self.text = String::from(text);
        self
    }
    pub(crate) fn with_duration_ms(mut self, duration_ms: i64) -> Self {
        self.remaining_duration_ms = duration_ms;
        self
    }
}

impl FadeEffect {
    fn _is_zero(&self) -> bool {
        self._fade_in_ms == 0 && self._fade_out_ms == 0
    }
}

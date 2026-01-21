use crate::ffmpeg_sys_the_third::AVCodecID;
use crate::subtitle::ass::parse_ass_subtitle;
use crate::subtitle::srt::parse_srt_subtitle;
use egui::text::LayoutJob;
use egui::{vec2, Align2, Color32, FontId, Margin, Pos2, Response, Stroke, TextFormat, Ui, Widget};

mod ass;
mod srt;

#[derive(Debug)]
pub struct Subtitle {
    text: String,
    fade: FadeEffect,
    alignment: Align2,
    primary_fill: Color32,
    position: Option<Pos2>,
    font_size: f32,
    margin: Margin,
    bold: bool,
    italic: bool,
    underline: bool,
    strikethrough: bool,
    pub(crate) pts: f64,
    pub(crate) duration: f64,
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
            font_size: 36.,
            margin: Margin::ZERO,
            bold: false,
            italic: false,
            underline: false,
            strikethrough: false,
            pts: 0,
            alignment: Align2::CENTER_CENTER,
            primary_fill: Color32::WHITE,
            position: None,
            duration: 0,
        }
    }
}

impl Subtitle {
    pub(crate) fn new(text: String, pts: i64, duration: i64, codec: AVCodecID) -> Self {
        if let Some(mut sub) = match codec {
            AVCodecID::AV_CODEC_ID_ASS => parse_ass_subtitle(&text).ok(),
            AVCodecID::AV_CODEC_ID_SUBRIP => parse_srt_subtitle(&text).ok(),
            _ => None,
        } {
            sub.pts = pts;
            sub.duration = duration;
            return sub;
        }

        Subtitle {
            text,
            pts,
            duration,
            ..Default::default()
        }
    }
}

impl FadeEffect {
    fn _is_zero(&self) -> bool {
        self._fade_in_ms == 0 && self._fade_out_ms == 0
    }
}

impl Widget for &Subtitle {
    fn ui(self, ui: &mut Ui) -> Response {
        let rect = ui.available_rect_before_wrap();

        let mut job = LayoutJob::default();
        job.halign = self.alignment.y();

        let format = TextFormat {
            font_id: FontId::proportional(self.font_size),
            color: self.primary_fill,
            valign: self.alignment.x(),
            italics: self.italic,
            strikethrough: if self.strikethrough {
                Stroke::new((self.font_size * 0.05).min(1.0), self.primary_fill)
            } else {
                Stroke::NONE
            },
            underline: if self.underline {
                Stroke::new((self.font_size * 0.05).min(1.0), self.primary_fill)
            } else {
                Stroke::NONE
            },
            ..Default::default()
        };
        job.append(&self.text, 0.0, format);
        let painter = ui.painter();
        let galley = painter.layout_job(job);

        let pos = rect.min
            + vec2(
                rect.width() / 2.0,
                rect.height() - 30.0 - galley.rect.height(),
            );
        painter.galley(pos, galley.clone(), Color32::TRANSPARENT);

        // TODO(v0l): stroke text

        ui.response()
    }
}

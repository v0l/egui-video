use crate::stream::StreamType;
use crate::{PlaybackInfo, PlaybackUpdate, PlayerOverlay, PlayerState, format_time};
use egui::{
    Align2, Color32, CornerRadius, FontId, Rect, Response, Sense, Shadow, Spinner, Ui, Vec2, vec2,
};

/// Basic player overlay impl
pub struct DefaultOverlay;

impl PlayerOverlay for DefaultOverlay {
    fn show(&self, ui: &mut Ui, frame_response: &Response, p: &PlaybackInfo) -> PlaybackUpdate {
        let hovered = ui.rect_contains_pointer(frame_response.rect);
        let currently_seeking = matches!(p.state, PlayerState::Seeking);
        let is_stopped = matches!(p.state, PlayerState::Stopped);
        let is_paused = matches!(p.state, PlayerState::Paused);
        let animation_time = 0.2;
        let seekbar_anim_frac = ui.ctx().animate_bool_with_time(
            frame_response.id.with("seekbar_anim"),
            hovered || currently_seeking || is_paused || is_stopped,
            animation_time,
        );

        if seekbar_anim_frac <= 0. {
            return PlaybackUpdate::default();
        }

        let seekbar_width_offset = 20.;
        let fullseekbar_width = frame_response.rect.width() - seekbar_width_offset;

        let seekbar_width = fullseekbar_width
            * if p.duration != 0.0 {
                (p.elapsed / p.duration).max(1.0)
            } else {
                0.0
            };

        let seekbar_offset = 20.;
        let seekbar_pos =
            frame_response.rect.left_bottom() + vec2(seekbar_width_offset / 2., -seekbar_offset);
        let seekbar_height = 3.;
        let mut fullseekbar_rect =
            Rect::from_min_size(seekbar_pos, vec2(fullseekbar_width, seekbar_height));

        let mut seekbar_rect =
            Rect::from_min_size(seekbar_pos, vec2(seekbar_width, seekbar_height));
        let seekbar_interact_rect = fullseekbar_rect.expand(10.);

        let seekbar_response = ui.interact(
            seekbar_interact_rect,
            frame_response.id.with("seekbar"),
            Sense::click_and_drag(),
        );

        let seekbar_hovered = seekbar_response.hovered();
        let seekbar_hover_anim_frac = ui.ctx().animate_bool_with_time(
            frame_response.id.with("seekbar_hover_anim"),
            seekbar_hovered || currently_seeking,
            animation_time,
        );

        if seekbar_hover_anim_frac > 0. {
            let new_top = fullseekbar_rect.top() - (3. * seekbar_hover_anim_frac);
            fullseekbar_rect.set_top(new_top);
            seekbar_rect.set_top(new_top);
        }

        let seek_indicator_anim = ui.ctx().animate_bool_with_time(
            frame_response.id.with("seek_indicator_anim"),
            currently_seeking,
            animation_time,
        );

        if currently_seeking {
            let seek_indicator_shadow = Shadow {
                offset: [10, 20],
                blur: 15,
                spread: 0,
                color: Color32::from_black_alpha(96).linear_multiply(seek_indicator_anim),
            };
            let spinner_size = 20. * seek_indicator_anim;
            ui.painter()
                .add(seek_indicator_shadow.as_shape(frame_response.rect, CornerRadius::ZERO));
            ui.put(
                Rect::from_center_size(frame_response.rect.center(), Vec2::splat(spinner_size)),
                Spinner::new().size(spinner_size),
            );
        }

        let mut p_ret = PlaybackUpdate::default();
        if seekbar_hovered || currently_seeking {
            if let Some(hover_pos) = seekbar_response.hover_pos() {
                if seekbar_response.clicked() || seekbar_response.dragged() {
                    let seek_frac = ((hover_pos - frame_response.rect.left_top()).x
                        - seekbar_width_offset / 2.)
                        .max(0.)
                        .min(fullseekbar_width)
                        / fullseekbar_width;
                    seekbar_rect.set_right(
                        hover_pos
                            .x
                            .min(fullseekbar_rect.right())
                            .max(fullseekbar_rect.left()),
                    );
                    if is_stopped {
                        p_ret.set_state.replace(PlayerState::Playing);
                    }
                    p_ret.set_seek.replace(seek_frac);
                }
            }
        }
        let text_color = Color32::WHITE.linear_multiply(seekbar_anim_frac);

        let pause_icon = if is_paused {
            "▶"
        } else if is_stopped {
            "◼"
        } else if currently_seeking {
            "↔"
        } else {
            "⏸"
        };
        let sound_icon = if p.volume > 0.7 {
            "🔊"
        } else if p.volume > 0.4 {
            "🔉"
        } else if p.volume > 0. {
            "🔈"
        } else {
            "🔇"
        };

        let mut icon_font_id = FontId::default();
        icon_font_id.size = 16.;

        let subtitle_icon = "💬";
        let stream_icon = "🔁";
        let icon_margin = 5.;
        let text_y_offset = -7.;
        let sound_icon_offset = vec2(-5., text_y_offset);
        let sound_icon_pos = fullseekbar_rect.right_top() + sound_icon_offset;

        let stream_index_icon_offset = vec2(-30., text_y_offset + 1.);
        let stream_icon_pos = fullseekbar_rect.right_top() + stream_index_icon_offset;

        let contraster_alpha: u8 = 100;
        let pause_icon_offset = vec2(3., text_y_offset);
        let pause_icon_pos = fullseekbar_rect.left_top() + pause_icon_offset;

        let duration_text_offset = vec2(25., text_y_offset);
        let duration_text_pos = fullseekbar_rect.left_top() + duration_text_offset;
        let mut duration_text_font_id = FontId::default();
        duration_text_font_id.size = 14.;

        let shadow = Shadow {
            offset: [10, 20],
            blur: 15,
            spread: 0,
            color: Color32::from_black_alpha(25).linear_multiply(seekbar_anim_frac),
        };

        let mut shadow_rect = frame_response.rect;
        shadow_rect.set_top(shadow_rect.bottom() - seekbar_offset - 10.);

        let fullseekbar_color = Color32::GRAY.linear_multiply(seekbar_anim_frac);
        let seekbar_color = Color32::WHITE.linear_multiply(seekbar_anim_frac);

        ui.painter()
            .add(shadow.as_shape(shadow_rect, CornerRadius::ZERO));

        ui.painter().rect_filled(
            fullseekbar_rect,
            CornerRadius::ZERO,
            fullseekbar_color.linear_multiply(0.5),
        );
        ui.painter()
            .rect_filled(seekbar_rect, CornerRadius::ZERO, seekbar_color);
        ui.painter().text(
            pause_icon_pos,
            Align2::LEFT_BOTTOM,
            pause_icon,
            icon_font_id.clone(),
            text_color,
        );

        if p.elapsed.is_finite() {
            ui.painter().text(
                duration_text_pos,
                Align2::LEFT_BOTTOM,
                if p.duration > 0.0 {
                    format!("{} / {}", format_time(p.elapsed), format_time(p.duration))
                } else {
                    format_time(p.elapsed)
                },
                duration_text_font_id,
                text_color,
            );
        }

        if seekbar_hover_anim_frac > 0. {
            ui.painter().circle_filled(
                seekbar_rect.right_center(),
                7. * seekbar_hover_anim_frac,
                seekbar_color,
            );
        }

        if frame_response.clicked() {
            match p.state {
                PlayerState::Stopped | PlayerState::Paused => {
                    p_ret.set_state.replace(PlayerState::Playing);
                }
                PlayerState::Playing | PlayerState::Seeking => {
                    p_ret.set_state.replace(PlayerState::Paused);
                }
                _ => {}
            }
        }

        let is_subtitle_cyclable = false;
        let is_audio_cyclable = false;

        if is_audio_cyclable || is_subtitle_cyclable {
            let stream_icon_rect = ui.painter().text(
                stream_icon_pos,
                Align2::RIGHT_BOTTOM,
                stream_icon,
                icon_font_id.clone(),
                text_color,
            );
            let stream_icon_hovered = ui.rect_contains_pointer(stream_icon_rect);
            let mut stream_info_hovered = false;
            let mut cursor = stream_icon_rect.right_top() + vec2(0., 5.);
            let cursor_offset = vec2(3., 15.);
            let stream_anim_id = frame_response.id.with("stream_anim");
            let mut stream_anim_frac: f32 = ui
                .ctx()
                .memory_mut(|m| *m.data.get_temp_mut_or_default(stream_anim_id));

            let mut draw_row = |stream_type: StreamType| {
                let text = match stream_type {
                    StreamType::Audio => format!("{} {}/{}", sound_icon, 1, 1),
                    StreamType::Subtitle => format!("{} {}/{}", subtitle_icon, 1, 1),
                    _ => unreachable!(),
                };

                let text_position = cursor - cursor_offset;
                let text_galley =
                    ui.painter()
                        .layout_no_wrap(text.clone(), icon_font_id.clone(), text_color);

                let background_rect =
                    Rect::from_min_max(text_position - text_galley.size(), text_position)
                        .expand(5.);

                let background_color =
                    Color32::from_black_alpha(contraster_alpha).linear_multiply(stream_anim_frac);

                ui.painter()
                    .rect_filled(background_rect, CornerRadius::same(5), background_color);

                if ui.rect_contains_pointer(background_rect.expand(5.)) {
                    stream_info_hovered = true;
                }

                if ui
                    .interact(
                        background_rect,
                        frame_response.id.with(&text),
                        Sense::click(),
                    )
                    .clicked()
                {
                    // TODO: cycle stream
                };

                let text_rect = ui.painter().text(
                    text_position,
                    Align2::RIGHT_BOTTOM,
                    text,
                    icon_font_id.clone(),
                    text_color.linear_multiply(stream_anim_frac),
                );

                cursor.y = text_rect.top();
            };

            if stream_anim_frac > 0. {
                if is_audio_cyclable {
                    draw_row(StreamType::Audio);
                }
                if is_subtitle_cyclable {
                    draw_row(StreamType::Subtitle);
                }
            }

            stream_anim_frac = ui.ctx().animate_bool_with_time(
                stream_anim_id,
                stream_icon_hovered || (stream_info_hovered && stream_anim_frac > 0.),
                animation_time,
            );

            ui.ctx()
                .memory_mut(|m| m.data.insert_temp(stream_anim_id, stream_anim_frac));
        }

        let sound_icon_rect = ui.painter().text(
            sound_icon_pos,
            Align2::RIGHT_BOTTOM,
            sound_icon,
            icon_font_id.clone(),
            text_color,
        );
        if ui
            .interact(
                sound_icon_rect,
                frame_response.id.with("sound_icon_sense"),
                Sense::click(),
            )
            .clicked()
        {
            if p.muted {
                p_ret.set_muted.replace(false);
            } else {
                p_ret.set_muted.replace(true);
            }
        }

        let sound_slider_outer_height = 75.;

        let mut sound_slider_rect = sound_icon_rect;
        sound_slider_rect.set_bottom(sound_icon_rect.top() - icon_margin);
        sound_slider_rect.set_top(sound_slider_rect.top() - sound_slider_outer_height);

        let sound_slider_interact_rect = sound_slider_rect.expand(icon_margin);
        let sound_hovered = ui.rect_contains_pointer(sound_icon_rect);
        let sound_slider_hovered = ui.rect_contains_pointer(sound_slider_interact_rect);
        let sound_anim_id = frame_response.id.with("sound_anim");
        let mut sound_anim_frac: f32 = ui
            .ctx()
            .memory_mut(|m| *m.data.get_temp_mut_or_default(sound_anim_id));
        sound_anim_frac = ui.ctx().animate_bool_with_time(
            sound_anim_id,
            sound_hovered || (sound_slider_hovered && sound_anim_frac > 0.),
            0.2,
        );
        ui.ctx()
            .memory_mut(|m| m.data.insert_temp(sound_anim_id, sound_anim_frac));
        let sound_slider_bg_color =
            Color32::from_black_alpha(contraster_alpha).linear_multiply(sound_anim_frac);
        let sound_bar_color =
            Color32::from_white_alpha(contraster_alpha).linear_multiply(sound_anim_frac);
        let mut sound_bar_rect = sound_slider_rect;
        sound_bar_rect.set_top(sound_bar_rect.bottom() - (sound_bar_rect.height() * p.volume));

        ui.painter().rect_filled(
            sound_slider_rect,
            CornerRadius::same(5),
            sound_slider_bg_color,
        );

        ui.painter()
            .rect_filled(sound_bar_rect, CornerRadius::same(5), sound_bar_color);
        let sound_slider_resp = ui.interact(
            sound_slider_rect,
            frame_response.id.with("sound_slider_sense"),
            Sense::click_and_drag(),
        );
        if sound_anim_frac > 0. && sound_slider_resp.clicked() || sound_slider_resp.dragged() {
            if let Some(hover_pos) = ui.ctx().input(|i| i.pointer.hover_pos()) {
                let sound_frac = 1.
                    - ((hover_pos - sound_slider_rect.left_top()).y / sound_slider_rect.height())
                        .max(0.)
                        .min(1.);
                p_ret.set_volume.replace(sound_frac);
            }
        }
        p_ret
    }
}

use nom::branch::alt;
use nom::bytes::complete::{tag, take_until};
use nom::combinator::{all_consuming, map, rest};
use nom::error::Error;
use nom::sequence::delimited;
use nom::IResult;
use nom::Parser;
use std::default::Default;

pub enum TagKind {
    Unknown,
    Bold,
    Italic,
    Underline,
    Strikethrough,
}

pub struct Tagged<'a> {
    pub kind: TagKind,
    pub content: &'a str,
}

fn i(input: &str) -> IResult<&str, Tagged> {
    delimited(
        tag("<i>"),
        map(take_until("</i>"), |t| Tagged {
            kind: TagKind::Italic,
            content: t,
        }),
        tag("</i>"),
    )
    .parse(input)
}

fn b(input: &str) -> IResult<&str, Tagged> {
    delimited(
        tag("<b>"),
        map(take_until("</b>"), |t| Tagged {
            kind: TagKind::Bold,
            content: t,
        }),
        tag("</b>"),
    )
    .parse(input)
}

fn u(input: &str) -> IResult<&str, Tagged> {
    delimited(
        tag("<u>"),
        map(take_until("</u>"), |t| Tagged {
            kind: TagKind::Underline,
            content: t,
        }),
        tag("</u>"),
    )
    .parse(input)
}

fn s(input: &str) -> IResult<&str, Tagged> {
    delimited(
        tag("<s>"),
        map(take_until("</s>"), |t| Tagged {
            kind: TagKind::Strikethrough,
            content: t,
        }),
        tag("</s>"),
    )
    .parse(input)
}

fn any(input: &str) -> IResult<&str, Tagged<'_>, Error<&str>> {
    all_consuming(alt((
        i,
        b,
        u,
        s,
        map(rest, |r| Tagged {
            kind: TagKind::Unknown,
            content: r,
        }),
    )))
    .parse(input)
}

pub(crate) fn parse_srt_subtitle(input: &str) -> Result<super::Subtitle, anyhow::Error> {
    any(input)
        .map(|(_, srt)| {
            Ok(super::Subtitle {
                text: srt.content.to_owned(),
                bold: matches!(srt.kind, TagKind::Bold),
                italic: matches!(srt.kind, TagKind::Italic),
                underline: matches!(srt.kind, TagKind::Underline),
                strikethrough: matches!(srt.kind, TagKind::Strikethrough),
                ..super::Subtitle::default()
            })
        })
        .map_err(|e| anyhow::Error::msg(e.to_string()))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_unknown() {
        let input = "Some shit text here, idk what im doing\nsave me";
        let i = parse_srt_subtitle(input).unwrap();
        assert_eq!(i.text, input);
        assert_eq!(i.italic, false);
        assert_eq!(i.underline, false);
        assert_eq!(i.strikethrough, false);
        assert_eq!(i.bold, false);
    }

    #[test]
    fn parse_italic() {
        let input = "<i>Some text goes here.</i>";
        let i = parse_srt_subtitle(input).unwrap();
        assert_eq!(i.text, "Some text goes here.");
        assert_eq!(i.italic, true);
        assert_eq!(i.underline, false);
        assert_eq!(i.strikethrough, false);
        assert_eq!(i.bold, false);
    }

    #[test]
    fn parse_bold() {
        let input = "<b>123 Some text goes here.</b>";
        let i = parse_srt_subtitle(input).unwrap();
        assert_eq!(i.text, "123 Some text goes here.");
        assert_eq!(i.italic, false);
        assert_eq!(i.underline, false);
        assert_eq!(i.strikethrough, false);
        assert_eq!(i.bold, true);
    }

    #[test]
    fn parse_underline() {
        let input = "<u>nom is really confusing</u>";
        let i = parse_srt_subtitle(input).unwrap();
        assert_eq!(i.text, "nom is really confusing");
        assert_eq!(i.italic, false);
        assert_eq!(i.underline, true);
        assert_eq!(i.strikethrough, false);
        assert_eq!(i.bold, false);
    }

    #[test]
    fn parse_strikethrough() {
        let input = "<s></s>";
        let i = parse_srt_subtitle(input).unwrap();
        assert_eq!(i.text, "");
        assert_eq!(i.italic, false);
        assert_eq!(i.underline, false);
        assert_eq!(i.strikethrough, true);
        assert_eq!(i.bold, false);
    }

    #[test]
    fn parse_extra_chars() {
        let input = "<s></s> Some other text here";
        assert!(parse_srt_subtitle(input).is_err())
    }
}

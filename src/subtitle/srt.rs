use crate::subtitle::Subtitle;
use log::info;
use nom::character::complete::{alpha1, char};
use nom::combinator::opt;
use nom::error::Error;
use nom::sequence::{delimited, preceded};
use nom::IResult;

pub enum TagKind {
    Unknown,
    Bold,
    Italic,
    Underline,
}

pub struct Tagged<'a> {
    pub kind: TagKind,
    pub content: &'a str,
}

fn tagger(input: &str) -> IResult<&str, &str> {
    delimited(char('<'), alpha1, char('>'))(input)
}

fn tagged(input: &str) -> IResult<&str, &str, Error<&str>> {
    preceded(opt(tagger), alpha1)(input)
}

pub(crate) fn parse_srt(input: &str) -> Result<Subtitle, anyhow::Error> {
    if let Ok((text, style)) = tagged(input) {
        info!("{}", text);
    }
    anyhow::bail!("failed to parse subtitle")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_srt() {
        let italic = "<i>Some text goes here.</i>";
        parse_srt(italic).unwrap();
    }
}

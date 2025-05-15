use super::label_filter::LabelFilterRule;
use anyhow::{anyhow, Result};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::complete::{char, multispace0},
    combinator::map,
    multi::many0,
    sequence::{delimited, pair, preceded},
    IResult,
};

/// A parser for human-friendly tag expressions
///
/// Syntax examples:
/// - "tag1"                   - Set("tag1")
/// - [tag1]                   - Set("tag1")
/// - !"tag1"                  - Not(Set("tag1"))
/// - ![tag1]                  - Not(Set("tag1"))
/// - "tag1" & "tag2"          - And([Set("tag1"), Set("tag2")])
/// - [tag1] & [tag2]          - And([Set("tag1"), Set("tag2")])
/// - "tag1" | "tag2"          - Or([Set("tag1"), Set("tag2")])
/// - [tag1] | [tag2]          - Or([Set("tag1"), Set("tag2")])
/// - "tag1" & !("tag2" | "tag3") - And([Set("tag1"), Not(Or([Set("tag2"), Set("tag3")]))])
/// - [tag1] & !([tag2] | [tag3]) - And([Set("tag1"), Not(Or([Set("tag2"), Set("tag3")]))])
pub struct LabelExpressionParser;

impl LabelExpressionParser {
    /// Parse a human-friendly tag expression into a LabelFilterRule
    pub fn parse(input: &str) -> Result<LabelFilterRule> {
        match expression(input) {
            Ok(("", rule)) => Ok(rule),
            Ok((remaining, _)) => Err(anyhow!("Unparsed input remains: {}", remaining)),
            Err(e) => Err(anyhow!("Failed to parse expression: {}", e)),
        }
    }
}

// Parser for a double-quoted string
fn double_quoted_string(input: &str) -> IResult<&str, String> {
    delimited(
        char('"'),
        map(take_while1(|c: char| c != '"'), |s: &str| s.to_string()),
        char('"'),
    )(input)
}

// Parser for a bracket-quoted string
fn bracket_quoted_string(input: &str) -> IResult<&str, String> {
    delimited(
        char('['),
        map(take_while1(|c: char| c != ']'), |s: &str| s.to_string()),
        char(']'),
    )(input)
}

// Parser for a tag string (either double-quoted or bracket-quoted)
fn quoted_string(input: &str) -> IResult<&str, String> {
    alt((double_quoted_string, bracket_quoted_string))(input)
}

// Parser for a primary expression (tag, or parenthesized expression)
fn primary(input: &str) -> IResult<&str, LabelFilterRule> {
    alt((
        // Set("tag")
        map(quoted_string, |s| LabelFilterRule::Set(s)),
        // Not(expr)
        map(preceded(pair(char('!'), multispace0), primary), |rule| {
            LabelFilterRule::Not(Box::new(rule))
        }),
        // (expr)
        delimited(
            pair(char('('), multispace0),
            expression,
            pair(multispace0, char(')')),
        ),
    ))(input)
}

// Parser for AND expressions
fn and_expr(input: &str) -> IResult<&str, LabelFilterRule> {
    let (input, first) = primary(input)?;

    // Parse any remaining terms separated by &
    let (input, rest) = many0(preceded(
        delimited(multispace0, alt((tag("&"), tag("AND"))), multispace0),
        primary,
    ))(input)?;

    if rest.is_empty() {
        Ok((input, first))
    } else {
        let mut rules = vec![first];
        rules.extend(rest);
        Ok((input, LabelFilterRule::And(rules)))
    }
}

// Parser for OR expressions
fn or_expr(input: &str) -> IResult<&str, LabelFilterRule> {
    let (input, first) = and_expr(input)?;

    // Parse any remaining terms separated by |
    let (input, rest) = many0(preceded(
        delimited(multispace0, alt((tag("|"), tag("OR"))), multispace0),
        and_expr,
    ))(input)?;

    if rest.is_empty() {
        Ok((input, first))
    } else {
        let mut rules = vec![first];
        rules.extend(rest);
        Ok((input, LabelFilterRule::Or(rules)))
    }
}

// Main expression parser
fn expression(input: &str) -> IResult<&str, LabelFilterRule> {
    let (input, _) = multispace0(input)?;
    let (input, result) = or_expr(input)?;
    let (input, _) = multispace0(input)?;
    Ok((input, result))
}

#[cfg(test)]
mod tests {
    use super::super::label_filter::LabelFilterRule::*;
    use super::*;

    #[test]
    fn test_parse_simple_tag() {
        let result = LabelExpressionParser::parse(r#""tag1""#).unwrap();
        assert_eq!(result, Set("tag1".to_string()));

        let result = LabelExpressionParser::parse("[tag1]").unwrap();
        assert_eq!(result, Set("tag1".to_string()));
    }

    #[test]
    fn test_parse_not() {
        let result = LabelExpressionParser::parse(r#"!"tag1""#).unwrap();
        assert_eq!(result, Not(Box::new(Set("tag1".to_string()))));

        let result = LabelExpressionParser::parse("![tag1]").unwrap();
        assert_eq!(result, Not(Box::new(Set("tag1".to_string()))));
    }

    #[test]
    fn test_parse_and() {
        let result = LabelExpressionParser::parse(r#""tag1" & "tag2""#).unwrap();
        assert_eq!(
            result,
            And(vec![Set("tag1".to_string()), Set("tag2".to_string())])
        );

        let result = LabelExpressionParser::parse("[tag1] & [tag2]").unwrap();
        assert_eq!(
            result,
            And(vec![Set("tag1".to_string()), Set("tag2".to_string())])
        );

        // Mix of quote styles
        let result = LabelExpressionParser::parse(r#""tag1" & [tag2]"#).unwrap();
        assert_eq!(
            result,
            And(vec![Set("tag1".to_string()), Set("tag2".to_string())])
        );
    }

    #[test]
    fn test_parse_or() {
        let result = LabelExpressionParser::parse(r#""tag1" | "tag2""#).unwrap();
        assert_eq!(
            result,
            Or(vec![Set("tag1".to_string()), Set("tag2".to_string())])
        );

        let result = LabelExpressionParser::parse("[tag1] | [tag2]").unwrap();
        assert_eq!(
            result,
            Or(vec![Set("tag1".to_string()), Set("tag2".to_string())])
        );
    }

    #[test]
    fn test_parse_complex() {
        let result = LabelExpressionParser::parse(r#""tag1" & !("tag2" | "tag3")"#).unwrap();
        assert_eq!(
            result,
            And(vec![
                Set("tag1".to_string()),
                Not(Box::new(Or(vec![
                    Set("tag2".to_string()),
                    Set("tag3".to_string())
                ])))
            ])
        );

        let result = LabelExpressionParser::parse("[tag1] & !([tag2] | [tag3])").unwrap();
        assert_eq!(
            result,
            And(vec![
                Set("tag1".to_string()),
                Not(Box::new(Or(vec![
                    Set("tag2".to_string()),
                    Set("tag3".to_string())
                ])))
            ])
        );
    }

    #[test]
    fn test_parse_whitespace() {
        let result = LabelExpressionParser::parse(r#"  "tag1"  &  "tag2"  "#).unwrap();
        assert_eq!(
            result,
            And(vec![Set("tag1".to_string()), Set("tag2".to_string())])
        );

        let result = LabelExpressionParser::parse("  [tag1]  &  [tag2]  ").unwrap();
        assert_eq!(
            result,
            And(vec![Set("tag1".to_string()), Set("tag2".to_string())])
        );
    }

    #[test]
    fn test_parse_word_operators() {
        let result = LabelExpressionParser::parse(r#""tag1" AND "tag2" OR "tag3""#).unwrap();
        assert_eq!(
            result,
            Or(vec![
                And(vec![Set("tag1".to_string()), Set("tag2".to_string())]),
                Set("tag3".to_string())
            ])
        );

        let result = LabelExpressionParser::parse("[tag1] AND [tag2] OR [tag3]").unwrap();
        assert_eq!(
            result,
            Or(vec![
                And(vec![Set("tag1".to_string()), Set("tag2".to_string())]),
                Set("tag3".to_string())
            ])
        );
    }

    #[test]
    fn test_mixed_quote_styles() {
        let result = LabelExpressionParser::parse(r#"[tag1] & "tag2" | [tag3]"#).unwrap();
        assert_eq!(
            result,
            Or(vec![
                And(vec![Set("tag1".to_string()), Set("tag2".to_string())]),
                Set("tag3".to_string())
            ])
        );
    }
}

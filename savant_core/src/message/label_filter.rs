use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LabelFilterRule {
    #[serde(rename = "set")]
    Set(String),
    #[serde(rename = "unset")]
    Unset(String),
    #[serde(rename = "and")]
    And(Vec<LabelFilterRule>),
    #[serde(rename = "or")]
    Or(Vec<LabelFilterRule>),
    #[serde(rename = "not")]
    Not(Box<LabelFilterRule>),
}

impl LabelFilterRule {
    pub fn matches(&self, value: &[String]) -> bool {
        match self {
            LabelFilterRule::Set(expected) => value.iter().any(|v| v == expected),
            LabelFilterRule::Unset(expected) => value.iter().all(|v| v != expected),
            LabelFilterRule::And(rules) => rules.iter().all(|r| r.matches(value)),
            LabelFilterRule::Or(rules) => rules.iter().any(|r| r.matches(value)),
            LabelFilterRule::Not(rule) => !rule.matches(value),
        }
    }

    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    pub fn to_yaml(&self) -> String {
        serde_yaml::to_string(&serde_json::to_value(self).unwrap()).unwrap()
    }

    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn from_yaml(yaml: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_value(serde_yaml::from_str(yaml)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::LabelFilterRule::*;
    use crate::message::label_filter::LabelFilterRule;

    #[test]
    fn test_matching_rule() {
        let rule = Set("test".to_string());
        assert_eq!(rule, Set("test".to_string()));
        assert_ne!(rule, Set("test2".to_string()));
    }

    #[test]
    fn test_matching_rule_matches() {
        let rule = Or(vec![Set("test".to_string()), Set("test2".to_string())]);
        assert!(rule.matches(&["test".to_string()]));
        assert!(rule.matches(&["test2".to_string()]));

        let rule = And(vec![Set("test".to_string()), Set("test2".to_string())]);

        assert!(!rule.matches(&["test".to_string()]));
        assert!(!rule.matches(&["test2".to_string()]));
        assert!(rule.matches(&["test".to_string(), "test2".to_string()]));

        let rule = Not(Box::new(Set("test".to_string())));
        assert!(!rule.matches(&["test".to_string()]));
        assert!(rule.matches(&["test2".to_string()]));

        let rule = And(vec![Set("test".to_string()), Unset("test2".to_string())]);
        assert!(rule.matches(&["test".to_string(), "test3".to_string()]));
        assert!(!rule.matches(&["test".to_string(), "test2".to_string()]));

        let rule = And(vec![
            Set("test".to_string()),
            Not(Box::new(Set("test2".to_string()))),
        ]);
        assert!(rule.matches(&["test".to_string(), "test3".to_string()]));
        assert!(!rule.matches(&["test".to_string(), "test2".to_string()]));
    }

    #[test]
    fn json_save_load() {
        let rule = Or(vec![
            Set("test".to_string()),
            Not(Box::new(Set("test2".to_string()))),
        ]);
        let json = rule.to_json_pretty();
        println!("{}", json);
        let yaml = rule.to_yaml();
        println!("{}", yaml);
        let rule2 = LabelFilterRule::from_json(&json).unwrap();
        assert_eq!(rule, rule2);
    }
}

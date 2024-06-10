use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchingRule {
    #[serde(rename = "eq")]
    Eq(String),
    #[serde(rename = "ne")]
    Ne(String),
    #[serde(rename = "and")]
    And(Vec<MatchingRule>),
    #[serde(rename = "or")]
    Or(Vec<MatchingRule>),
    #[serde(rename = "not")]
    Not(Box<MatchingRule>),
}

impl MatchingRule {
    pub fn matches(&self, value: &[String]) -> bool {
        match self {
            MatchingRule::Eq(expected) => value.iter().any(|v| v == expected),
            MatchingRule::Ne(expected) => value.iter().all(|v| v != expected),
            MatchingRule::And(rules) => rules.iter().all(|r| r.matches(value)),
            MatchingRule::Or(rules) => rules.iter().any(|r| r.matches(value)),
            MatchingRule::Not(rule) => !rule.matches(value),
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
    use super::MatchingRule::*;
    use crate::message::routing_label_matching::MatchingRule;

    #[test]
    fn test_matching_rule() {
        let rule = Eq("test".to_string());
        assert_eq!(rule, Eq("test".to_string()));
        assert_ne!(rule, Eq("test2".to_string()));
    }

    #[test]
    fn test_matching_rule_matches() {
        let rule = Or(vec![Eq("test".to_string()), Eq("test2".to_string())]);
        assert!(rule.matches(&["test".to_string()]));
        assert!(rule.matches(&["test2".to_string()]));

        let rule = And(vec![Eq("test".to_string()), Eq("test2".to_string())]);

        assert!(!rule.matches(&["test".to_string()]));
        assert!(!rule.matches(&["test2".to_string()]));
        assert!(rule.matches(&["test".to_string(), "test2".to_string()]));

        let rule = Not(Box::new(Eq("test".to_string())));
        assert!(!rule.matches(&["test".to_string()]));
        assert!(rule.matches(&["test2".to_string()]));

        let rule = And(vec![Eq("test".to_string()), Ne("test2".to_string())]);
        assert!(rule.matches(&["test".to_string(), "test3".to_string()]));
        assert!(!rule.matches(&["test".to_string(), "test2".to_string()]));

        let rule = And(vec![
            Eq("test".to_string()),
            Not(Box::new(Eq("test2".to_string()))),
        ]);
        assert!(rule.matches(&["test".to_string(), "test3".to_string()]));
        assert!(!rule.matches(&["test".to_string(), "test2".to_string()]));
    }

    #[test]
    fn json_save_load() {
        let rule = Or(vec![Eq("test".to_string()), Eq("test2".to_string())]);
        let json = rule.to_json_pretty();
        println!("{}", json);
        let yaml = rule.to_yaml();
        println!("{}", yaml);
        let rule2 = MatchingRule::from_json(&json).unwrap();
        assert_eq!(rule, rule2);
    }
}

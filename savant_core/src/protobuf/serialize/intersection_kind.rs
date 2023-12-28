use crate::primitives::IntersectionKind;
use crate::protobuf::generated;

impl From<&IntersectionKind> for generated::IntersectionKind {
    fn from(kind: &IntersectionKind) -> Self {
        match kind {
            IntersectionKind::Inside => generated::IntersectionKind::Inside,
            IntersectionKind::Outside => generated::IntersectionKind::Outside,
            IntersectionKind::Enter => generated::IntersectionKind::Enter,
            IntersectionKind::Leave => generated::IntersectionKind::Leave,
            IntersectionKind::Cross => generated::IntersectionKind::Cross,
        }
    }
}

impl From<&generated::IntersectionKind> for IntersectionKind {
    fn from(kind: &generated::IntersectionKind) -> Self {
        match kind {
            generated::IntersectionKind::Inside => IntersectionKind::Inside,
            generated::IntersectionKind::Outside => IntersectionKind::Outside,
            generated::IntersectionKind::Enter => IntersectionKind::Enter,
            generated::IntersectionKind::Leave => IntersectionKind::Leave,
            generated::IntersectionKind::Cross => IntersectionKind::Cross,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::IntersectionKind;
    use crate::protobuf::generated;

    #[test]
    fn test_intersection_kind() {
        assert_eq!(
            generated::IntersectionKind::Inside,
            generated::IntersectionKind::from(&IntersectionKind::Inside)
        );
        assert_eq!(
            generated::IntersectionKind::Outside,
            generated::IntersectionKind::from(&IntersectionKind::Outside)
        );
        assert_eq!(
            generated::IntersectionKind::Enter,
            generated::IntersectionKind::from(&IntersectionKind::Enter)
        );
        assert_eq!(
            generated::IntersectionKind::Leave,
            generated::IntersectionKind::from(&IntersectionKind::Leave)
        );
        assert_eq!(
            generated::IntersectionKind::Cross,
            generated::IntersectionKind::from(&IntersectionKind::Cross)
        );
        assert_eq!(
            IntersectionKind::Inside,
            IntersectionKind::from(&generated::IntersectionKind::Inside)
        );
        assert_eq!(
            IntersectionKind::Outside,
            IntersectionKind::from(&generated::IntersectionKind::Outside)
        );
        assert_eq!(
            IntersectionKind::Enter,
            IntersectionKind::from(&generated::IntersectionKind::Enter)
        );
        assert_eq!(
            IntersectionKind::Leave,
            IntersectionKind::from(&generated::IntersectionKind::Leave)
        );
        assert_eq!(
            IntersectionKind::Cross,
            IntersectionKind::from(&generated::IntersectionKind::Cross)
        );
    }
}

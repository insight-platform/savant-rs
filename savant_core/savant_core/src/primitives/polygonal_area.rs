use crate::primitives::point::Point;
use crate::primitives::{Intersection, IntersectionKind, Segment};
use anyhow::bail;
use geo::line_intersection::line_intersection;
use geo::{Contains, EuclideanDistance, Line, LineIntersection, LineString};

#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PolygonalArea {
    pub(crate) vertices: Vec<Point>,
    pub(crate) tags: Option<Vec<Option<String>>>,
    #[serde(skip_deserializing, skip_serializing)]
    polygon: Option<geo::Polygon>,
}

impl PolygonalArea {
    pub fn get_vertices(&self) -> &[Point] {
        &self.vertices
    }

    pub fn get_tags(&self) -> Option<&[Option<String>]> {
        self.tags.as_deref()
    }

    pub fn contains_many_points(&mut self, points: &[Point]) -> Vec<bool> {
        self.build_polygon();
        points.iter().map(|p| self.contains(p)).collect::<Vec<_>>()
    }

    pub fn crossed_by_segments(&mut self, segments: &[Segment]) -> Vec<Intersection> {
        self.build_polygon();
        segments
            .iter()
            .map(|s| self.crossed_by_segment(s))
            .collect::<Vec<_>>()
    }

    pub fn get_polygon(&mut self) -> geo::Polygon {
        self.build_polygon();
        self.polygon.as_ref().unwrap().clone()
    }

    pub fn is_self_intersecting(&mut self) -> bool {
        use geo::algorithm::line_intersection::LineIntersection::*;
        self.build_polygon();
        let poly = self.polygon.as_ref().unwrap();
        let exterior = poly.exterior();
        exterior.lines().any(|l| {
            exterior.lines().filter(|l2| &l != l2).any(|l2| {
                let res = line_intersection(l, l2);
                match res {
                    Some(li) => match li {
                        SinglePoint {
                            intersection: _,
                            is_proper,
                        } => is_proper,
                        _ => true,
                    },
                    _ => false,
                }
            })
        })
    }

    pub fn crossed_by_segment(&mut self, seg: &Segment) -> Intersection {
        self.build_polygon();
        let seg = Line::from([
            (seg.begin.x as f64, seg.begin.y as f64),
            (seg.end.x as f64, seg.end.y as f64),
        ]);
        let poly = self.polygon.as_ref().unwrap();

        let mut intersections = poly
            .exterior()
            .lines()
            .enumerate()
            .flat_map(|(indx, l)| match line_intersection(l, seg) {
                None => None,
                Some(intersection) => match intersection {
                    LineIntersection::SinglePoint {
                        intersection,
                        is_proper: _,
                    } => Some((indx, seg.start.euclidean_distance(&intersection))),
                    LineIntersection::Collinear { intersection } => {
                        Some((indx, seg.start.euclidean_distance(&intersection.start)))
                    }
                },
            })
            .collect::<Vec<_>>();
        intersections.sort_by(|(_, ld), (_, rd)| ld.partial_cmp(rd).unwrap());
        let intersections = intersections
            .into_iter()
            .map(|(e, _)| e)
            .collect::<Vec<_>>();

        let contains_start = poly.contains(&seg.start) || poly.exterior().contains(&seg.start);
        let contains_end = poly.contains(&seg.end) || poly.exterior().contains(&seg.end);

        Intersection::new(
            match (contains_start, contains_end, intersections.is_empty()) {
                (false, false, false) => IntersectionKind::Cross,
                (false, false, true) => IntersectionKind::Outside,
                (true, true, _) => IntersectionKind::Inside,
                (true, false, _) => IntersectionKind::Leave,
                (false, true, _) => IntersectionKind::Enter,
            },
            intersections
                .iter()
                .map(|i| (*i, self.get_tag(*i).unwrap()))
                .collect(),
        )
    }

    pub fn contains(&mut self, p: &Point) -> bool {
        self.build_polygon();
        self.polygon
            .as_ref()
            .unwrap()
            .contains(&geo::Point::from((p.x as f64, p.y as f64)))
    }

    pub fn build_polygon(&mut self) {
        let p = self
            .polygon
            .take()
            .unwrap_or_else(|| Self::gen_polygon(&self.vertices));
        self.polygon.replace(p);
    }
}

// class methods
impl PolygonalArea {
    pub fn new(vertices: Vec<Point>, tags: Option<Vec<Option<String>>>) -> anyhow::Result<Self> {
        if let Some(t) = &tags {
            if vertices.len() != t.len() {
                bail!("Vertices and tags must have the same length");
            }
        }

        let polygon = Some(Self::gen_polygon(&vertices));
        Ok(Self {
            polygon,
            tags,
            vertices,
        })
    }

    fn gen_polygon(vertices: &[Point]) -> geo::Polygon {
        geo::Polygon::new(
            LineString::from(
                vertices
                    .iter()
                    .map(|p| geo::Point::from((p.x as f64, p.y as f64)))
                    .collect::<Vec<geo::Point>>(),
            ),
            vec![],
        )
    }

    pub fn points_positions(polys: &mut [Self], points: &[Point]) -> Vec<Vec<bool>> {
        polys
            .iter_mut()
            .map(|p| {
                p.build_polygon();
                points.iter().map(|pt| p.contains(pt)).collect()
            })
            .collect::<Vec<_>>()
    }

    pub fn segments_intersections(
        polys: &mut [Self],
        segments: &[Segment],
    ) -> Vec<Vec<Intersection>> {
        let segments = &segments;
        polys
            .iter_mut()
            .map(|p| {
                p.build_polygon();
                segments
                    .iter()
                    .map(|seg| p.crossed_by_segment(seg))
                    .collect()
            })
            .collect::<Vec<_>>()
    }

    pub fn get_tag(&self, edge: usize) -> anyhow::Result<Option<String>> {
        let tags = self.tags.as_ref();
        match tags {
            None => Ok(None),
            Some(tags) => {
                if tags.len() <= edge {
                    bail!(format!("Index {edge} out of range!"));
                } else {
                    Ok(tags.get(edge).unwrap().clone())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PolygonalArea;
    use crate::primitives::point::Point;
    use crate::primitives::{Intersection, IntersectionKind, Segment};

    const UPPER: &str = "upper";
    const RIGHT: &str = "right";
    const LOWER: &str = "lower";
    const LEFT: &str = "left";

    fn get_square_area(xc: f32, yc: f32, l: f32) -> Vec<Point> {
        let l2 = l / 2.0;

        vec![
            Point::new(xc - l2, yc + l2),
            Point::new(xc + l2, yc + l2),
            Point::new(xc + l2, yc - l2),
            Point::new(xc - l2, yc - l2),
        ]
    }

    #[test]
    fn contains() -> anyhow::Result<()> {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(0.99, 0.0);
        let p3 = Point::new(1.0, 0.0);

        let mut area1 = PolygonalArea::new(get_square_area(0.0, 0.0, 2.0), None)?;
        let area2 = PolygonalArea::new(get_square_area(-1.0, 0.0, 2.0), None)?;

        assert!(area1.contains(&p1));
        assert!(area1.contains(&p2));
        assert!(!area1.contains(&p3));

        assert_eq!(
            area1.contains_many_points(&[p1.clone(), p2.clone(), p3.clone()]),
            vec![true, true, false]
        );

        assert_eq!(
            PolygonalArea::points_positions(&mut [area1, area2], &[p1, p2, p3]),
            vec![vec![true, true, false], vec![false, false, false]]
        );

        Ok(())
    }

    #[test]
    fn segment_intersects() -> anyhow::Result<()> {
        let mut area = PolygonalArea::new(
            get_square_area(0.0, 0.0, 2.0),
            Some(vec![Some(UPPER.into()), None, Some(LOWER.into()), None]),
        )?;

        let seg1 = Segment::new(Point::new(0.0, 2.0), Point::new(0.0, 0.0));
        let res = area.crossed_by_segment(&seg1);
        assert_eq!(
            res,
            Intersection::new(IntersectionKind::Enter, vec![(0, Some(UPPER.into()))])
        );

        let seg2 = Segment::new(Point::new(0.0, 0.0), Point::new(0.0, -2.0));
        let res = area.crossed_by_segment(&seg2);
        assert_eq!(
            res,
            Intersection::new(IntersectionKind::Leave, vec![(2, Some(LOWER.into()))])
        );

        let seg3 = Segment::new(Point::new(0.0, 0.0), Point::new(0.0, -0.5));
        let res = area.crossed_by_segment(&seg3);
        assert_eq!(res, Intersection::new(IntersectionKind::Inside, vec![]));

        let seg4 = Segment::new(Point::new(-1.0, 2.0), Point::new(1.0, 2.0));
        let res = area.crossed_by_segment(&seg4);
        assert_eq!(res, Intersection::new(IntersectionKind::Outside, vec![]));

        let seg5 = Segment::new(Point::new(-2.0, 0.0), Point::new(2.0, 0.0));
        let res = area.crossed_by_segment(&seg5);
        assert_eq!(
            res,
            Intersection::new(IntersectionKind::Cross, vec![(3, None), (1, None)])
        );

        let seg6 = Segment::new(Point::new(0.0, 2.0), Point::new(0.0, -2.0));
        let res = area.crossed_by_segment(&seg6);
        assert_eq!(
            res,
            Intersection::new(
                IntersectionKind::Cross,
                vec![(0, Some(UPPER.into())), (2, Some(LOWER.into()))]
            )
        );

        let seg7 = Segment::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0));
        let res = area.crossed_by_segment(&seg7);
        assert_eq!(
            res,
            Intersection::new(
                IntersectionKind::Inside,
                vec![(0, Some(UPPER.into())), (1, None)]
            )
        );

        let seg8 = Segment::new(Point::new(2.0, 2.0), Point::new(1.0, 1.0));
        let res = area.crossed_by_segment(&seg8);
        assert_eq!(
            res,
            Intersection::new(
                IntersectionKind::Enter,
                vec![(0, Some(UPPER.into())), (1, None)]
            )
        );

        let seg9 = Segment::new(Point::new(-1.0, -1.0), Point::new(1.0, 1.0));
        let res = area.crossed_by_segment(&seg9);
        assert_eq!(
            res,
            Intersection::new(
                IntersectionKind::Inside,
                vec![
                    (2, Some(LOWER.into())),
                    (3, None),
                    (0, Some(UPPER.into())),
                    (1, None),
                ]
            )
        );

        let seg9 = Segment::new(Point::new(0.0, 1.0), Point::new(1.0, 0.0));
        let res = area.crossed_by_segment(&seg9);
        assert_eq!(
            res,
            Intersection::new(
                IntersectionKind::Inside,
                vec![(0, Some(UPPER.into())), (1, None),]
            )
        );

        let seg10 = Segment::new(Point::new(-2.0, 1.0), Point::new(2.0, 1.0));
        let res = area.crossed_by_segment(&seg10);
        assert_eq!(
            res,
            Intersection::new(
                IntersectionKind::Cross,
                vec![(0, Some(UPPER.into())), (3, None), (1, None)]
            )
        );

        let seg11 = Segment::new(Point::new(2.0, 1.0), Point::new(-2.0, 1.0));
        let res = area.crossed_by_segment(&seg11);
        assert_eq!(
            res,
            Intersection::new(
                IntersectionKind::Cross,
                vec![(1, None), (0, Some(UPPER.into())), (3, None),]
            )
        );

        Ok(())
    }

    #[test]
    fn multi_seg_crossing() -> anyhow::Result<()> {
        let area1 = PolygonalArea::new(
            get_square_area(0.0, 0.0, 2.0),
            Some(vec![
                Some(format!("{UPPER}_1")),
                Some(format!("{RIGHT}_1")),
                Some(format!("{LOWER}_1")),
                Some(format!("{LEFT}_1")),
            ]),
        )?;

        let area2 = PolygonalArea::new(
            get_square_area(1.0, 1.0, 2.0),
            Some(vec![
                Some(format!("{UPPER}_2")),
                Some(format!("{RIGHT}_2")),
                Some(format!("{LOWER}_2")),
                Some(format!("{LEFT}_2")),
            ]),
        )?;

        let seg1 = Segment::new(Point::new(-2.0, 0.5), Point::new(3.0, 0.5));
        let seg2 = Segment::new(Point::new(-0.5, 2.0), Point::new(-0.5, -2.0));
        let intersections =
            PolygonalArea::segments_intersections(&mut [area1, area2], &[seg1, seg2]);
        assert_eq!(
            intersections,
            vec![
                vec![
                    Intersection::new(
                        IntersectionKind::Cross,
                        vec![
                            (3, Some(format!("{LEFT}_1"))),
                            (1, Some(format!("{RIGHT}_1"))),
                        ]
                    ),
                    Intersection::new(
                        IntersectionKind::Cross,
                        vec![
                            (0, Some(format!("{UPPER}_1"))),
                            (2, Some(format!("{LOWER}_1")))
                        ]
                    )
                ],
                vec![
                    Intersection::new(
                        IntersectionKind::Cross,
                        vec![
                            (3, Some(format!("{LEFT}_2"))),
                            (1, Some(format!("{RIGHT}_2"))),
                        ]
                    ),
                    Intersection::new(IntersectionKind::Outside, vec![])
                ]
            ]
        );

        Ok(())
    }

    #[test]
    fn test_self_intersecting() -> anyhow::Result<()> {
        let mut area = PolygonalArea::new(
            vec![
                Point::new(0.0, 0.0),
                Point::new(1.0, 0.0),
                Point::new(1.0, 1.0),
                Point::new(0.0, 1.0),
            ],
            None,
        )?;
        assert!(!area.is_self_intersecting());

        let mut area = PolygonalArea::new(
            vec![
                Point::new(0.0, 0.0),
                Point::new(1.0, 1.0),
                Point::new(1.0, 0.0),
                Point::new(0.0, 1.0),
            ],
            None,
        )?;
        assert!(area.is_self_intersecting());

        let mut area = PolygonalArea::new(
            vec![
                Point::new(0.0, 0.0),
                Point::new(1.0, 0.0),
                Point::new(0.5, 0.0),
                Point::new(1.0, 1.0),
                Point::new(0.0, 1.0),
            ],
            None,
        )?;
        assert!(area.is_self_intersecting());

        Ok(())
    }
}

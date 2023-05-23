use crate::primitives::point::Point;
use crate::primitives::to_json_value::ToSerdeJsonValue;
use crate::primitives::{Intersection, IntersectionKind, Segment};
use crate::utils::python::no_gil;
use geo::line_intersection::line_intersection;
use geo::{Contains, Intersects, Line, LineString};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rkyv::{with::Skip, Archive, Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, Default, Clone, PartialEq)]
#[archive(check_bytes)]
pub struct PolygonalArea {
    pub vertices: Arc<Vec<Point>>,
    pub tags: Arc<Option<Vec<Option<String>>>>,
    #[with(Skip)]
    polygon: Option<geo::Polygon>,
}

impl ToSerdeJsonValue for PolygonalArea {
    fn to_serde_json_value(&self) -> Value {
        let mut vertices = Vec::new();
        for v in self.vertices.iter() {
            vertices.push(v.to_serde_json_value());
        }
        serde_json::json!({
            "vertices": vertices,
            "tags": self.tags.as_ref().as_ref().unwrap_or(&Vec::new()),
        })
    }
}

#[pymethods]
impl PolygonalArea {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[new]
    pub fn new(vertices: Vec<Point>, tags: Option<Vec<Option<String>>>) -> Self {
        if let Some(t) = &tags {
            assert_eq!(vertices.len(), t.len());
        }

        let polygon = Some(Self::gen_polygon(&vertices));
        Self {
            polygon,
            tags: Arc::new(tags),
            vertices: Arc::new(vertices),
        }
    }

    pub fn get_tag(&self, edge: usize) -> PyResult<Option<String>> {
        let tags = self.tags.as_ref().as_ref();
        match tags {
            None => Ok(None),
            Some(tags) => {
                if tags.len() <= edge {
                    Err(PyValueError::new_err(format!("Index {edge} out of range!")))
                } else {
                    Ok(tags.get(edge).unwrap().clone())
                }
            }
        }
    }

    #[pyo3(name = "crossed_by_segment")]
    pub fn crossed_by_segment_py(&mut self, seg: &Segment) -> Intersection {
        no_gil(|| {
            self.build_polygon();
            self.crossed_by_segment(seg)
        })
    }

    pub fn crossed_by_segments(&mut self, segs: Vec<Segment>) -> Vec<Intersection> {
        no_gil(|| {
            self.build_polygon();
            segs.iter().map(|s| self.crossed_by_segment(s)).collect()
        })
    }

    #[pyo3(name = "is_self_intersecting")]
    pub fn is_self_intersecting_py(&mut self) -> bool {
        no_gil(|| {
            self.build_polygon();
            self.is_self_intersecting()
        })
    }

    #[pyo3(name = "contains")]
    pub fn contains_py(&mut self, p: &Point) -> bool {
        no_gil(|| {
            self.build_polygon();
            self.contains(p)
        })
    }

    pub fn contains_many_points(&mut self, points: Vec<Point>) -> Vec<bool> {
        no_gil(|| {
            self.build_polygon();
            points.iter().map(|p| self.contains(p)).collect()
        })
    }

    #[staticmethod]
    pub fn points_positions(polys: Vec<Self>, points: Vec<Point>) -> Vec<Vec<bool>> {
        no_gil(|| {
            let pts = &points;
            polys
                .into_par_iter()
                .map(|mut p| {
                    p.build_polygon();
                    pts.iter().map(|pt| p.contains(pt)).collect()
                })
                .collect()
        })
    }

    #[staticmethod]
    pub fn segments_intersections(
        polys: Vec<Self>,
        segments: Vec<Segment>,
    ) -> Vec<Vec<Intersection>> {
        no_gil(|| {
            let segments = &segments;
            polys
                .into_par_iter()
                .map(|mut p| {
                    p.build_polygon();
                    segments
                        .iter()
                        .map(|seg| p.crossed_by_segment_py(seg))
                        .collect()
                })
                .collect()
        })
    }
}

impl PolygonalArea {
    pub fn is_self_intersecting(&self) -> bool {
        use geo::algorithm::line_intersection::LineIntersection::*;
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
        let seg = Line::from([(seg.begin.x, seg.begin.y), (seg.end.x, seg.end.y)]);
        let poly = self.polygon.as_ref().unwrap();

        let intersections = poly
            .exterior()
            .lines()
            .enumerate()
            .flat_map(|(indx, l)| if l.intersects(&seg) { Some(indx) } else { None })
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

    pub fn contains(&self, p: &Point) -> bool {
        self.polygon
            .as_ref()
            .unwrap()
            .contains(&geo::Point::from((p.x, p.y)))
    }

    fn gen_polygon(vertices: &[Point]) -> geo::Polygon {
        geo::Polygon::new(
            LineString::from(
                vertices
                    .iter()
                    .map(|p| geo::Point::from((p.x, p.y)))
                    .collect::<Vec<geo::Point>>(),
            ),
            vec![],
        )
    }

    pub fn build_polygon(&mut self) {
        let p = self
            .polygon
            .take()
            .unwrap_or_else(|| Self::gen_polygon(&self.vertices));
        self.polygon.replace(p);
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

    fn get_square_area(xc: f64, yc: f64, l: f64) -> Vec<Point> {
        let l2 = l / 2.0;

        vec![
            Point::new(xc - l2, yc + l2),
            Point::new(xc + l2, yc + l2),
            Point::new(xc + l2, yc - l2),
            Point::new(xc - l2, yc - l2),
        ]
    }

    #[test]
    fn contains() {
        pyo3::prepare_freethreaded_python();

        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(0.99, 0.0);
        let p3 = Point::new(1.0, 0.0);

        let mut area1 = PolygonalArea::new(get_square_area(0.0, 0.0, 2.0), None);
        let area2 = PolygonalArea::new(get_square_area(-1.0, 0.0, 2.0), None);

        assert!(area1.contains_py(&p1));
        assert!(area1.contains_py(&p2));
        assert!(!area1.contains_py(&p3));

        assert_eq!(
            area1.contains_many_points(vec![p1.clone(), p2.clone(), p3.clone()]),
            vec![true, true, false]
        );

        assert_eq!(
            PolygonalArea::points_positions(vec![area1, area2], vec![p1, p2, p3],),
            vec![vec![true, true, false], vec![false, false, false]]
        )
    }

    #[test]
    fn archive() {
        let area = PolygonalArea::new(
            get_square_area(0.0, 0.0, 2.0),
            Some(vec![Some("1".into()), None, None, None]),
        );
        let bytes = rkyv::to_bytes::<_, 256>(&area).unwrap();
        let area2 = rkyv::from_bytes::<PolygonalArea>(&bytes[..]);
        assert!(area2.is_ok());
        assert_eq!(area2.as_ref().unwrap().vertices, area.vertices);
        assert_eq!(
            area2.as_ref().unwrap().tags.as_ref().as_ref().unwrap(),
            &vec![Some("1".into()), None, None, None]
        );

        let area_err = rkyv::from_bytes::<PolygonalArea>(vec![].as_slice());
        assert!(area_err.is_err());

        let area_err = rkyv::from_bytes::<PolygonalArea>(vec![1, 2, 3].as_slice());
        assert!(area_err.is_err());
    }

    #[test]
    fn contains_after_archive() {
        pyo3::prepare_freethreaded_python();

        let area = PolygonalArea::new(get_square_area(0.0, 0.0, 2.0), None);

        let bytes = rkyv::to_bytes::<_, 256>(&area).unwrap();
        let area = rkyv::from_bytes::<PolygonalArea>(&bytes[..]).unwrap();
        let p1 = Point::new(0.0, 0.0);
        assert!(area.clone().contains_py(&p1));

        assert_eq!(
            area.clone().contains_many_points(vec![p1.clone()]),
            vec![true]
        );

        assert_eq!(
            PolygonalArea::points_positions(vec![area], vec![p1],),
            vec![vec![true]]
        )
    }

    #[test]
    fn segment_intersects() {
        pyo3::prepare_freethreaded_python();

        let mut area = PolygonalArea::new(
            get_square_area(0.0, 0.0, 2.0),
            Some(vec![Some(UPPER.into()), None, Some(LOWER.into()), None]),
        );

        let seg1 = Segment::new(Point::new(0.0, 2.0), Point::new(0.0, 0.0));
        let res = area.crossed_by_segment_py(&seg1);
        assert_eq!(
            res,
            Intersection::new(IntersectionKind::Enter, vec![(0, Some(UPPER.into()))])
        );

        let seg2 = Segment::new(Point::new(0.0, 0.0), Point::new(0.0, -2.0));
        let res = area.crossed_by_segment_py(&seg2);
        assert_eq!(
            res,
            Intersection::new(IntersectionKind::Leave, vec![(2, Some(LOWER.into()))])
        );

        let seg3 = Segment::new(Point::new(0.0, 0.0), Point::new(0.0, -0.5));
        let res = area.crossed_by_segment_py(&seg3);
        assert_eq!(res, Intersection::new(IntersectionKind::Inside, vec![]));

        let seg4 = Segment::new(Point::new(-1.0, 2.0), Point::new(1.0, 2.0));
        let res = area.crossed_by_segment_py(&seg4);
        assert_eq!(res, Intersection::new(IntersectionKind::Outside, vec![]));

        let seg5 = Segment::new(Point::new(-2.0, 0.0), Point::new(2.0, 0.0));
        let res = area.crossed_by_segment_py(&seg5);
        assert_eq!(
            res,
            Intersection::new(IntersectionKind::Cross, vec![(1, None), (3, None)])
        );

        let seg6 = Segment::new(Point::new(0.0, 2.0), Point::new(0.0, -2.0));
        let res = area.crossed_by_segment_py(&seg6);
        assert_eq!(
            res,
            Intersection::new(
                IntersectionKind::Cross,
                vec![(0, Some(UPPER.into())), (2, Some(LOWER.into()))]
            )
        );

        let seg7 = Segment::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0));
        let res = area.crossed_by_segment_py(&seg7);
        assert_eq!(
            res,
            Intersection::new(
                IntersectionKind::Inside,
                vec![(0, Some(UPPER.into())), (1, None)]
            )
        );

        let seg8 = Segment::new(Point::new(2.0, 2.0), Point::new(1.0, 1.0));
        let res = area.crossed_by_segment_py(&seg8);
        assert_eq!(
            res,
            Intersection::new(
                IntersectionKind::Enter,
                vec![(0, Some(UPPER.into())), (1, None)]
            )
        );

        let seg9 = Segment::new(Point::new(-1.0, -1.0), Point::new(1.0, 1.0));
        let res = area.crossed_by_segment_py(&seg9);
        assert_eq!(
            res,
            Intersection::new(
                IntersectionKind::Inside,
                vec![
                    (0, Some(UPPER.into())),
                    (1, None),
                    (2, Some(LOWER.into())),
                    (3, None)
                ]
            )
        );

        let seg9 = Segment::new(Point::new(0.0, 1.0), Point::new(1.0, 0.0));
        let res = area.crossed_by_segment_py(&seg9);
        assert_eq!(
            res,
            Intersection::new(
                IntersectionKind::Inside,
                vec![(0, Some(UPPER.into())), (1, None),]
            )
        );
    }

    #[test]
    fn multi_seg_crossing() {
        let area1 = PolygonalArea::new(
            get_square_area(0.0, 0.0, 2.0),
            Some(vec![
                Some(format!("{UPPER}_1")),
                Some(format!("{RIGHT}_1")),
                Some(format!("{LOWER}_1")),
                Some(format!("{LEFT}_1")),
            ]),
        );

        let area2 = PolygonalArea::new(
            get_square_area(1.0, 1.0, 2.0),
            Some(vec![
                Some(format!("{UPPER}_2")),
                Some(format!("{RIGHT}_2")),
                Some(format!("{LOWER}_2")),
                Some(format!("{LEFT}_2")),
            ]),
        );

        let seg1 = Segment::new(Point::new(-2.0, 0.5), Point::new(3.0, 0.5));
        let seg2 = Segment::new(Point::new(-0.5, 2.0), Point::new(-0.5, -2.0));
        let intersections =
            PolygonalArea::segments_intersections(vec![area1, area2], vec![seg1, seg2]);
        assert_eq!(
            intersections,
            vec![
                vec![
                    Intersection::new(
                        IntersectionKind::Cross,
                        vec![
                            (1, Some(format!("{RIGHT}_1"))),
                            (3, Some(format!("{LEFT}_1")))
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
                            (1, Some(format!("{RIGHT}_2"))),
                            (3, Some(format!("{LEFT}_2")))
                        ]
                    ),
                    Intersection::new(IntersectionKind::Outside, vec![])
                ]
            ]
        );
    }

    #[test]
    fn test_self_intersecting() {
        let area = PolygonalArea::new(
            vec![
                Point::new(0.0, 0.0),
                Point::new(1.0, 0.0),
                Point::new(1.0, 1.0),
                Point::new(0.0, 1.0),
            ],
            None,
        );
        assert!(!area.is_self_intersecting());

        let area = PolygonalArea::new(
            vec![
                Point::new(0.0, 0.0),
                Point::new(1.0, 1.0),
                Point::new(1.0, 0.0),
                Point::new(0.0, 1.0),
            ],
            None,
        );
        assert!(area.is_self_intersecting());

        let area = PolygonalArea::new(
            vec![
                Point::new(0.0, 0.0),
                Point::new(1.0, 0.0),
                Point::new(0.5, 0.0),
                Point::new(1.0, 1.0),
                Point::new(0.0, 1.0),
            ],
            None,
        );
        assert!(area.is_self_intersecting());
    }
}

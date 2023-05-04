use crate::primitives::point::Point;
use crate::primitives::{Intersection, IntersectionKind, LineSegment};
use geo::{Contains, Intersects, Line, LineString};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rkyv::{with::Skip, Archive, Deserialize, Serialize};
use std::sync::Arc;

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, Default, Clone)]
#[archive(check_bytes)]
pub struct PolygonalArea {
    pub vertices: Arc<Vec<Point>>,
    pub tags: Arc<Option<Vec<Option<String>>>>,
    #[with(Skip)]
    polygon: Option<geo::Polygon>,
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

    pub fn intersects(&mut self, seg: &LineSegment) -> Intersection {
        self.build_polygon();
        self.intersects_int(seg)
    }

    pub fn contains(&mut self, p: &Point) -> bool {
        self.build_polygon();
        self.contains_int(p)
    }

    pub fn contains_many_points_py(&mut self, points: Vec<Point>) -> Vec<bool> {
        Python::with_gil(|py| {
            self.build_polygon();
            py.allow_threads(|| points.iter().map(|p| self.contains_int(p)).collect())
        })
    }

    #[staticmethod]
    pub fn contains_many_points_polys_py(points: Vec<Point>, polys: Vec<Self>) -> Vec<Vec<bool>> {
        let pts = &points;
        polys
            .into_par_iter()
            .map(|mut p| {
                p.build_polygon();
                pts.iter().map(|pt| p.contains_int(pt)).collect()
            })
            .collect()
    }
}

impl PolygonalArea {
    pub fn intersects_int(&mut self, seg: &LineSegment) -> Intersection {
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

    pub fn contains_int(&self, p: &Point) -> bool {
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
    use crate::primitives::{Intersection, IntersectionKind, LineSegment};

    fn get_area1(xc: f64, yc: f64, l: f64) -> PolygonalArea {
        let l2 = l / 2.0;
        PolygonalArea::new(
            vec![
                Point::new(xc - l2, yc + l2),
                Point::new(xc + l2, yc + l2),
                Point::new(xc + l2, yc - l2),
                Point::new(xc - l2, yc - l2),
            ],
            None,
        )
    }

    #[test]
    fn contains() {
        pyo3::prepare_freethreaded_python();

        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(0.99, 0.0);
        let p3 = Point::new(1.0, 0.0);

        let mut area1 = get_area1(0.0, 0.0, 2.0);
        let area2 = get_area1(-1.0, 0.0, 2.0);

        assert!(area1.contains(&p1));
        assert!(area1.contains(&p2));
        assert!(!area1.contains(&p3));

        assert_eq!(
            area1.contains_many_points_py(vec![p1.clone(), p2.clone(), p3.clone()]),
            vec![true, true, false]
        );

        assert_eq!(
            PolygonalArea::contains_many_points_polys_py(vec![p1, p2, p3], vec![area1, area2]),
            vec![vec![true, true, false], vec![false, false, false]]
        )
    }

    #[test]
    fn archive() {
        let area = get_area1(0.0, 0.0, 2.0);
        let bytes = rkyv::to_bytes::<_, 256>(&area).unwrap();
        let area2 = rkyv::from_bytes::<PolygonalArea>(&bytes[..]);
        assert!(area2.is_ok());
        assert_eq!(area2.unwrap().vertices, area.vertices);

        let area_err = rkyv::from_bytes::<PolygonalArea>(vec![].as_slice());
        assert!(area_err.is_err());

        let area_err = rkyv::from_bytes::<PolygonalArea>(vec![1, 2, 3].as_slice());
        assert!(area_err.is_err());
    }

    #[test]
    fn contains_after_archive() {
        pyo3::prepare_freethreaded_python();

        let area = PolygonalArea::new(
            vec![
                Point::new(-1.0, 1.0),
                Point::new(1.0, 1.0),
                Point::new(1.0, -1.0),
                Point::new(-1.0, -1.0),
            ],
            None,
        );

        let bytes = rkyv::to_bytes::<_, 256>(&area).unwrap();
        let area = rkyv::from_bytes::<PolygonalArea>(&bytes[..]).unwrap();
        let p1 = Point::new(0.0, 0.0);
        assert!(area.clone().contains(&p1));

        assert_eq!(
            area.clone().contains_many_points_py(vec![p1.clone()]),
            vec![true]
        );

        assert_eq!(
            PolygonalArea::contains_many_points_polys_py(vec![p1], vec![area]),
            vec![vec![true]]
        )
    }

    #[test]
    fn segment_intersects() {
        pyo3::prepare_freethreaded_python();

        let upper = "upper";
        let lower = "lower";

        let mut area = PolygonalArea::new(
            vec![
                Point::new(-1.0, 1.0),
                Point::new(1.0, 1.0),
                Point::new(1.0, -1.0),
                Point::new(-1.0, -1.0),
            ],
            Some(vec![Some(upper.into()), None, Some(lower.into()), None]),
        );

        let seg1 = LineSegment::new(Point::new(0.0, 2.0), Point::new(0.0, 0.0));
        let res = area.intersects(&seg1);
        assert_eq!(
            res,
            Intersection::new(IntersectionKind::Enter, vec![(0, Some(upper.into()))])
        );

        let seg2 = LineSegment::new(Point::new(0.0, 0.0), Point::new(0.0, -2.0));
        let res = area.intersects(&seg2);
        assert_eq!(
            res,
            Intersection::new(IntersectionKind::Leave, vec![(2, Some(lower.into()))])
        );

        let seg3 = LineSegment::new(Point::new(0.0, 0.0), Point::new(0.0, -0.5));
        let res = area.intersects(&seg3);
        assert_eq!(res, Intersection::new(IntersectionKind::Inside, vec![]));

        let seg4 = LineSegment::new(Point::new(-1.0, 2.0), Point::new(1.0, 2.0));
        let res = area.intersects(&seg4);
        assert_eq!(res, Intersection::new(IntersectionKind::Outside, vec![]));

        let seg5 = LineSegment::new(Point::new(-2.0, 0.0), Point::new(2.0, 0.0));
        let res = area.intersects(&seg5);
        assert_eq!(
            res,
            Intersection::new(IntersectionKind::Cross, vec![(1, None), (3, None)])
        );

        let seg6 = LineSegment::new(Point::new(0.0, 2.0), Point::new(0.0, -2.0));
        let res = area.intersects(&seg6);
        assert_eq!(
            res,
            Intersection::new(
                IntersectionKind::Cross,
                vec![(0, Some(upper.into())), (2, Some(lower.into()))]
            )
        );

        let seg7 = LineSegment::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0));
        let res = area.intersects(&seg7);
        assert_eq!(
            res,
            Intersection::new(
                IntersectionKind::Inside,
                vec![(0, Some(upper.into())), (1, None)]
            )
        );

        let seg8 = LineSegment::new(Point::new(2.0, 2.0), Point::new(1.0, 1.0));
        let res = area.intersects(&seg8);
        assert_eq!(
            res,
            Intersection::new(
                IntersectionKind::Enter,
                vec![(0, Some(upper.into())), (1, None)]
            )
        );

        let seg9 = LineSegment::new(Point::new(-1.0, -1.0), Point::new(1.0, 1.0));
        let res = area.intersects(&seg9);
        assert_eq!(
            res,
            Intersection::new(
                IntersectionKind::Inside,
                vec![(0, Some(upper.into())), (1, None)]
            )
        );
    }
}

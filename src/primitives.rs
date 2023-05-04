use geo::{Contains, LineString};
use pyo3::prelude::*;
use rayon::prelude::*;
use rkyv::{with::Skip, Archive, Deserialize, Serialize};

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct Point {
    x: f64,
    y: f64,
}

#[pymethods]
impl Point {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[new]
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, Clone, Default)]
#[archive(check_bytes)]
pub struct PolygonalArea {
    pub vertices: Vec<Point>,
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
    pub fn new(vertices: Vec<Point>) -> Self {
        let polygon = Some(Self::gen_polygon(&vertices));
        Self { polygon, vertices }
    }

    pub fn contains(&mut self, p: &Point) -> bool {
        self.gen_poly();
        self.contains_int(p)
    }

    pub fn contains_many_points(&mut self, points: Vec<Point>) -> Vec<bool> {
        Python::with_gil(|py| {
            self.gen_poly();
            py.allow_threads(|| points.iter().map(|p| self.contains_int(p)).collect())
        })
    }

    #[staticmethod]
    pub fn contains_many_points_polys(points: Vec<Point>, polys: Vec<Self>) -> Vec<Vec<bool>> {
        let pts = &points;
        polys
            .into_par_iter()
            .map(|mut p| {
                p.gen_poly();
                pts.iter().map(|pt| p.contains_int(pt)).collect()
            })
            .collect()
    }
}

impl PolygonalArea {
    fn contains_int(&self, p: &Point) -> bool {
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

    fn gen_poly(&mut self) {
        let p = self
            .polygon
            .take()
            .unwrap_or_else(|| Self::gen_polygon(&self.vertices));
        self.polygon.replace(p);
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::{Point, PolygonalArea};

    fn get_area1(xc: f64, yc: f64, l: f64) -> PolygonalArea {
        let l2 = l / 2.0;
        PolygonalArea::new(vec![
            Point::new(xc - l2, yc + l2),
            Point::new(xc + l2, yc + l2),
            Point::new(xc + l2, yc - l2),
            Point::new(xc - l2, yc - l2),
        ])
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
            area1.contains_many_points(vec![p1.clone(), p2.clone(), p3.clone()]),
            vec![true, true, false]
        );

        assert_eq!(
            PolygonalArea::contains_many_points_polys(vec![p1, p2, p3], vec![area1, area2]),
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

        let area = PolygonalArea::new(vec![
            Point::new(-1.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(1.0, -1.0),
            Point::new(-1.0, -1.0),
        ]);

        let bytes = rkyv::to_bytes::<_, 256>(&area).unwrap();
        let area = rkyv::from_bytes::<PolygonalArea>(&bytes[..]).unwrap();
        let p1 = Point::new(0.0, 0.0);
        assert!(area.clone().contains(&p1));

        assert_eq!(
            area.clone().contains_many_points(vec![p1.clone()]),
            vec![true]
        );

        assert_eq!(
            PolygonalArea::contains_many_points_polys(vec![p1], vec![area]),
            vec![vec![true]]
        )
    }
}

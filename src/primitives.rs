use geo::{Contains, LineString};
use pyo3::prelude::*;
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
        self.polygon
            .as_ref()
            .unwrap()
            .contains(&geo::Point::from((p.x, p.y)))
    }

    pub fn contains_many(&mut self, points: Vec<Point>) -> Vec<bool> {
        self.gen_poly();
        points.iter().map(|p| self.contains(p)).collect()
    }
}

impl PolygonalArea {
    fn gen_polygon(vertices: &Vec<Point>) -> geo::Polygon {
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

    #[test]
    fn contains() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(0.99, 0.0);
        let p3 = Point::new(1.0, 0.0);
        let mut area = PolygonalArea::new(vec![
            Point::new(-1.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(1.0, -1.0),
            Point::new(-1.0, -1.0),
        ]);

        assert!(area.contains(&p1));
        assert!(area.contains(&p2));
        assert!(!area.contains(&p3));

        assert_eq!(
            area.contains_many(vec![p1, p2, p3]),
            vec![true, true, false]
        );
    }

    #[test]
    fn archive() {
        let area = PolygonalArea::new(vec![
            Point::new(-1.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(1.0, -1.0),
            Point::new(-1.0, -1.0),
        ]);
        let bytes = rkyv::to_bytes::<_, 256>(&area).unwrap();
        let area2 = rkyv::from_bytes::<PolygonalArea>(&bytes[..]);
        assert!(area2.is_ok());
        assert_eq!(area2.unwrap().vertices, area.vertices);

        let area_err = rkyv::from_bytes::<PolygonalArea>(vec![].as_slice());
        assert!(area_err.is_err());

        let area_err = rkyv::from_bytes::<PolygonalArea>(vec![1, 2, 3].as_slice());
        assert!(area_err.is_err());
    }
}

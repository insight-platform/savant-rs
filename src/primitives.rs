use geo::{Contains, LineString};
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct Point {
    point: geo::Point,
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
        Self {
            point: geo::Point::from((x, y)),
        }
    }
}

#[pyclass]
#[derive(Debug)]
pub struct PolygonalArea {
    pub vertices: geo::Polygon,
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
        let exterior = vertices
            .iter()
            .map(|p| geo::Point::from((p.point.x(), p.point.y())))
            .collect::<Vec<geo::Point>>();

        Self {
            vertices: geo::Polygon::new(LineString::from(exterior), vec![]),
        }
    }

    pub fn contains(&self, p: &Point) -> bool {
        self.vertices.contains(&p.point)
    }

    pub fn contains_many(&self, points: Vec<Point>) -> Vec<bool> {
        points.iter().map(|p| self.contains(p)).collect()
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
        let area = PolygonalArea::new(vec![
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
}

use crate::detach;
use crate::primitives::point::Point;
use crate::primitives::{Intersection, Segment};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use savant_core::primitives::rust;
use std::mem;

#[pyclass]
#[derive(Debug, PartialEq, Clone)]
pub struct PolygonalArea(pub(crate) rust::PolygonalArea);

impl PolygonalArea {
    pub fn get_polygon(&mut self) -> geo::Polygon {
        self.0.get_polygon()
    }
}

#[pymethods]
impl PolygonalArea {
    pub fn contains_many_points(&mut self, points: Vec<Point>) -> Vec<bool> {
        let points = unsafe { mem::transmute::<Vec<Point>, Vec<rust::Point>>(points) };
        self.0.contains_many_points(&points)
    }

    pub fn crossed_by_segments(&mut self, segments: Vec<Segment>) -> Vec<Intersection> {
        let segments = unsafe { mem::transmute::<Vec<Segment>, Vec<rust::Segment>>(segments) };
        let intersections = self.0.crossed_by_segments(&segments);
        unsafe { mem::transmute::<Vec<rust::Intersection>, Vec<Intersection>>(intersections) }
    }

    pub fn is_self_intersecting(&mut self) -> bool {
        self.0.is_self_intersecting()
    }

    pub fn crossed_by_segment(&mut self, segment: &Segment) -> Intersection {
        let segment = &segment.0;
        let intersection = self.0.crossed_by_segment(segment);
        Intersection(intersection)
    }

    pub fn contains(&mut self, p: &Point) -> bool {
        self.0.contains(&p.0)
    }

    pub fn build_polygon(&mut self) {
        self.0.build_polygon();
    }

    pub fn get_tag(&self, edge: usize) -> PyResult<Option<String>> {
        self.0
            .get_tag(edge)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    #[pyo3(name = "points_positions")]
    #[pyo3(signature = (polys, points, no_gil=false))]
    fn points_positions_gil(polys: Vec<Self>, points: Vec<Point>, no_gil: bool) -> Vec<Vec<bool>> {
        let mut polys = unsafe { mem::transmute::<Vec<Self>, Vec<rust::PolygonalArea>>(polys) };
        let points = unsafe { mem::transmute::<Vec<Point>, Vec<rust::Point>>(points) };
        detach!(no_gil, || {
            rust::PolygonalArea::points_positions(&mut polys, &points)
        })
    }

    #[staticmethod]
    #[pyo3(name = "segments_intersections")]
    #[pyo3(signature = (polys, segments, no_gil=false))]
    fn segments_intersections_gil(
        polys: Vec<Self>,
        segments: Vec<Segment>,
        no_gil: bool,
    ) -> Vec<Vec<Intersection>> {
        let mut polys = unsafe { mem::transmute::<Vec<Self>, Vec<rust::PolygonalArea>>(polys) };
        let segments = unsafe { mem::transmute::<Vec<Segment>, Vec<rust::Segment>>(segments) };
        let intersections = detach!(no_gil, || {
            rust::PolygonalArea::segments_intersections(&mut polys, &segments)
        });
        unsafe {
            mem::transmute::<Vec<Vec<rust::Intersection>>, Vec<Vec<Intersection>>>(intersections)
        }
    }

    #[new]
    #[pyo3(signature = (vertices, tags=None))]
    pub fn new(vertices: Vec<Point>, tags: Option<Vec<Option<String>>>) -> Self {
        let vertices = unsafe { mem::transmute::<Vec<Point>, Vec<rust::Point>>(vertices) };
        Self(rust::PolygonalArea::new(vertices, tags))
    }
}

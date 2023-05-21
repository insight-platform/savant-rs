use crate::primitives::to_json_value::ToSerdeJsonValue;
use crate::primitives::{Point, PolygonalArea};
use crate::utils::python::no_gil;
use crate::utils::round_2_digits;
use pyo3::{pyclass, pymethods, Py, PyAny};
use rkyv::{Archive, Deserialize, Serialize};

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct RBBox {
    #[pyo3(get, set)]
    pub xc: f64,
    #[pyo3(get, set)]
    pub yc: f64,
    #[pyo3(get, set)]
    pub width: f64,
    #[pyo3(get, set)]
    pub height: f64,
    #[pyo3(get, set)]
    pub angle: Option<f64>,
}

impl Default for RBBox {
    fn default() -> Self {
        Self {
            xc: 0.0,
            yc: 0.0,
            width: 0.0,
            height: 0.0,
            angle: None,
        }
    }
}

impl ToSerdeJsonValue for RBBox {
    fn to_serde_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "xc": self.xc,
            "yc": self.yc,
            "width": self.width,
            "height": self.height,
            "angle": self.angle,
        })
    }
}

#[pymethods]
impl RBBox {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[new]
    pub fn new(xc: f64, yc: f64, width: f64, height: f64, angle: Option<f64>) -> Self {
        Self {
            xc,
            yc,
            width,
            height,
            angle,
        }
    }

    #[pyo3(name = "scale")]
    pub fn scale_py(&mut self, scale_x: f64, scale_y: f64) {
        no_gil(|| {
            self.scale(scale_x, scale_y);
        })
    }

    #[getter]
    #[pyo3(name = "vertices")]
    pub fn vertices_py(&self) -> Vec<(f64, f64)> {
        no_gil(|| self.vertices())
    }

    #[getter]
    #[pyo3(name = "vertices_rounded")]
    pub fn vertices_rounded_py(&self) -> Vec<(f64, f64)> {
        no_gil(|| self.vertices_rounded())
    }

    #[getter]
    #[pyo3(name = "vertices_int")]
    pub fn vertices_int_py(&self) -> Vec<(i64, i64)> {
        no_gil(|| self.vertices_int())
    }

    #[pyo3(name = "as_polygonal_area")]
    pub fn as_polygonal_area_py(&self) -> PolygonalArea {
        no_gil(|| self.as_polygonal_area())
    }

    #[getter]
    #[pyo3(name = "wrapping_box")]
    pub fn wrapping_box_py(&self) -> BBox {
        no_gil(|| self.wrapping_bbox())
    }

    #[pyo3(name = "as_graphical_wrapping_box")]
    pub fn as_graphical_wrapping_box_py(
        &self,
        padding: f64,
        border_width: f64,
        max_x: f64,
        max_y: f64,
    ) -> BBox {
        no_gil(|| self.graphical_wrapping_bbox(padding, border_width, max_x, max_y))
    }
}

impl RBBox {
    pub fn scale(&mut self, scale_x: f64, scale_y: f64) {
        match self.angle {
            None => {
                self.xc *= scale_x;
                self.yc *= scale_y;
                self.width *= scale_x;
                self.height *= scale_y;
            }
            Some(angle) => {
                let scale_x2 = scale_x * scale_x;
                let scale_y2 = scale_y * scale_y;
                let cotan = (angle * std::f64::consts::PI / 180.0).tan().powi(-1);
                let cotan_2 = cotan * cotan;
                let scale_angle =
                    (scale_x * angle.signum() / (scale_x2 + scale_y2 * cotan_2).sqrt()).acos();
                let nscale_height = ((scale_x2 + scale_y2 * cotan_2) / (1.0 + cotan_2)).sqrt();
                let ayh = 1.0 / ((90.0 - angle) / 180.0 * std::f64::consts::PI).tan();
                let nscale_width = ((scale_x2 + scale_y2 * ayh * ayh) / (1.0 + ayh * ayh)).sqrt();

                self.angle = Some(90.0 - (scale_angle * 180.0 / std::f64::consts::PI));
                self.xc *= scale_x;
                self.yc *= scale_y;
                self.width *= nscale_width;
                self.height *= nscale_height;
            }
        }
    }

    pub fn vertices(&self) -> Vec<(f64, f64)> {
        let angle = self.angle.unwrap_or(0.0);
        let angle = angle * std::f64::consts::PI / 180.0;
        let cos = angle.cos();
        let sin = angle.sin();
        let x = self.xc;
        let y = self.yc;
        let w = self.width / 2.0;
        let h = self.height / 2.0;
        vec![
            (x + w * cos - h * sin, y + w * sin + h * cos),
            (x + w * cos + h * sin, y + w * sin - h * cos),
            (x - w * cos + h * sin, y - w * sin - h * cos),
            (x - w * cos - h * sin, y - w * sin + h * cos),
        ]
    }

    pub fn vertices_rounded(&self) -> Vec<(f64, f64)> {
        self.vertices()
            .into_iter()
            .map(|(x, y)| (round_2_digits(x), round_2_digits(y)))
            .collect::<Vec<_>>()
    }

    pub fn vertices_int(&self) -> Vec<(i64, i64)> {
        self.vertices()
            .into_iter()
            .map(|(x, y)| (x as i64, y as i64))
            .collect::<Vec<_>>()
    }

    pub fn as_polygonal_area(&self) -> PolygonalArea {
        PolygonalArea::new(
            self.vertices()
                .into_iter()
                .map(|(x, y)| Point::new(x, y))
                .collect::<Vec<_>>(),
            None,
        )
    }

    pub fn wrapping_bbox(&self) -> BBox {
        if self.angle.is_none() {
            BBox::new(self.xc, self.yc, self.width, self.height)
        } else {
            let mut vertices = self.vertices();
            let (initial_vtx_x, initial_vtx_y) = vertices.pop().unwrap();
            let (mut min_x, mut min_y, mut max_x, mut max_y) =
                (initial_vtx_x, initial_vtx_y, initial_vtx_x, initial_vtx_y);
            for v in vertices {
                let (vtx_x, vtx_y) = v;
                if vtx_x < min_x {
                    min_x = vtx_x;
                }
                if vtx_x > max_x {
                    max_x = vtx_x;
                }
                if vtx_y < min_y {
                    min_y = vtx_y;
                }
                if vtx_y > max_y {
                    max_y = vtx_y;
                }
            }
            BBox::new(
                (min_x + max_x) / 2.0,
                (min_y + max_y) / 2.0,
                max_x - min_x,
                max_y - min_y,
            )
        }
    }

    pub fn graphical_wrapping_bbox(
        &self,
        padding: f64,
        border_width: f64,
        max_x: f64,
        max_y: f64,
    ) -> BBox {
        assert!(padding >= 0.0 && border_width >= 0.0 && max_x >= 0.0 && max_y >= 0.0);
        let bbox = self.wrapping_bbox();
        let left = 0.0f64.max(bbox.get_left() - padding - border_width).floor();
        let top = 0.0f64.max(bbox.get_top() - padding - border_width).floor();
        let right = max_x.min(bbox.get_right() + padding + border_width).ceil();
        let bottom = max_y.min(bbox.get_bottom() + padding + border_width).ceil();

        let mut width = 1.0f64.max(right - left);
        if width as i64 % 2 != 0 {
            width += 1.0;
        }

        let mut height = 1.0f64.max(bottom - top);
        if height as i64 % 2 != 0 {
            height += 1.0;
        }

        BBox::new(left + width / 2.0, top + height / 2.0, width, height)
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct BBox {
    rbbox: RBBox,
}

#[pymethods]
impl BBox {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self.rbbox)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[new]
    pub fn new(xc: f64, yc: f64, width: f64, height: f64) -> Self {
        Self {
            rbbox: RBBox::new(xc, yc, width, height, None),
        }
    }

    #[staticmethod]
    pub fn ltrb(left: f64, top: f64, right: f64, bottom: f64) -> Self {
        let width = right - left;
        let height = bottom - top;

        let xc = (left + right) / 2.0;
        let yc = (top + bottom) / 2.0;

        Self {
            rbbox: RBBox::new(xc, yc, width, height, None),
        }
    }

    #[staticmethod]
    pub fn ltwh(left: f64, top: f64, width: f64, height: f64) -> Self {
        let xc = left + width / 2.0;
        let yc = top + height / 2.0;

        Self {
            rbbox: RBBox::new(xc, yc, width, height, None),
        }
    }

    #[getter]
    pub fn get_xc(&self) -> f64 {
        self.rbbox.xc
    }

    #[setter]
    pub fn set_xc(&mut self, xc: f64) {
        self.rbbox.xc = xc;
    }

    #[getter]
    pub fn get_yc(&self) -> f64 {
        self.rbbox.yc
    }

    #[setter]
    pub fn set_yc(&mut self, yc: f64) {
        self.rbbox.yc = yc;
    }

    #[getter]
    pub fn get_width(&self) -> f64 {
        self.rbbox.width
    }

    #[setter]
    pub fn set_width(&mut self, width: f64) {
        self.rbbox.width = width;
    }

    #[getter]
    pub fn get_height(&self) -> f64 {
        self.rbbox.height
    }

    #[setter]
    pub fn set_height(&mut self, height: f64) {
        self.rbbox.height = height;
    }

    #[getter]
    pub fn get_top(&self) -> f64 {
        self.rbbox.yc - self.rbbox.height / 2.0
    }

    #[setter]
    pub fn set_top(&mut self, top: f64) {
        self.rbbox.yc = top + self.rbbox.height / 2.0;
    }

    #[getter]
    pub fn get_left(&self) -> f64 {
        self.rbbox.xc - self.rbbox.width / 2.0
    }

    #[setter]
    pub fn set_left(&mut self, left: f64) {
        self.rbbox.xc = left + self.rbbox.width / 2.0;
    }

    #[getter]
    pub fn get_right(&self) -> f64 {
        self.rbbox.xc + self.rbbox.width / 2.0
    }

    #[getter]
    pub fn get_bottom(&self) -> f64 {
        self.rbbox.yc + self.rbbox.height / 2.0
    }

    #[getter]
    pub fn vertices(&self) -> Vec<(f64, f64)> {
        no_gil(|| self.rbbox.vertices())
    }

    #[getter]
    pub fn vertices_rounded(&self) -> Vec<(f64, f64)> {
        no_gil(|| self.rbbox.vertices_rounded())
    }

    #[getter]
    pub fn vertices_int(&self) -> Vec<(i64, i64)> {
        no_gil(|| self.rbbox.vertices_int())
    }

    #[getter]
    pub fn wrapping_box(&self) -> BBox {
        no_gil(|| self.rbbox.wrapping_bbox())
    }

    pub fn as_graphical_wrapping_box(
        &self,
        padding: f64,
        border_width: f64,
        max_x: f64,
        max_y: f64,
    ) -> BBox {
        no_gil(|| {
            self.rbbox
                .graphical_wrapping_bbox(padding, border_width, max_x, max_y)
        })
    }

    pub fn as_ltrb(&self) -> (f64, f64, f64, f64) {
        let top = self.get_top();
        let left = self.get_left();
        let bottom = self.get_bottom();
        let right = self.get_right();
        (left, top, right, bottom)
    }

    pub fn as_ltrb_int(&self) -> (i64, i64, i64, i64) {
        let top = self.get_top().floor();
        let left = self.get_left().floor();
        let bottom = self.get_bottom().ceil();
        let right = self.get_right().ceil();
        (left as i64, top as i64, right as i64, bottom as i64)
    }

    pub fn as_ltwh(&self) -> (f64, f64, f64, f64) {
        let top = self.get_top();
        let left = self.get_left();
        let width = self.get_width();
        let height = self.get_height();
        (left, top, width, height)
    }

    pub fn as_ltwh_int(&self) -> (i64, i64, i64, i64) {
        let top = self.get_top().floor();
        let left = self.get_left().floor();
        let width = self.get_width().ceil();
        let height = self.get_height().ceil();
        (left as i64, top as i64, width as i64, height as i64)
    }

    pub fn as_xcycwh(&self) -> (f64, f64, f64, f64) {
        let xc = self.get_xc();
        let yc = self.get_yc();
        let width = self.get_width();
        let height = self.get_height();
        (xc, yc, width, height)
    }

    pub fn as_xcycwh_int(&self) -> (i64, i64, i64, i64) {
        let xc = self.get_xc();
        let yc = self.get_yc();
        let width = self.get_width();
        let height = self.get_height();
        (xc as i64, yc as i64, width as i64, height as i64)
    }

    pub fn as_rbbox(&self) -> RBBox {
        self.rbbox.clone()
    }

    pub fn scale(&mut self, scale_x: f64, scale_y: f64) {
        no_gil(|| {
            self.rbbox.scale(scale_x, scale_y);
        })
    }

    pub fn as_polygonal_area(&self) -> PolygonalArea {
        no_gil(|| self.rbbox.as_polygonal_area())
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::RBBox;
    use crate::utils::round_2_digits;

    #[test]
    fn test_scale_no_angle() {
        let mut bbox = RBBox::new(0.0, 0.0, 100.0, 100.0, None);
        bbox.scale(2.0, 2.0);
        assert_eq!(bbox.xc, 0.0);
        assert_eq!(bbox.yc, 0.0);
        assert_eq!(bbox.width, 200.0);
        assert_eq!(bbox.height, 200.0);
        assert_eq!(bbox.angle, None);
    }

    #[test]
    fn test_scale_with_angle() {
        let mut bbox = RBBox::new(0.0, 0.0, 100.0, 100.0, Some(45.0));
        bbox.scale(2.0, 3.0);
        //dbg!(&bbox);
        assert_eq!(bbox.xc, 0.0);
        assert_eq!(bbox.yc, 0.0);
        assert_eq!(round_2_digits(bbox.width), 254.95);
        assert_eq!(round_2_digits(bbox.height), 254.95);
        assert_eq!(bbox.angle.map(round_2_digits), Some(33.69));
    }

    #[test]
    fn test_vertices() {
        let bbox = RBBox::new(0.0, 0.0, 100.0, 100.0, Some(45.0));
        let vertices = bbox.vertices_rounded();
        assert_eq!(vertices.len(), 4);
        assert_eq!(vertices[0], (0.0, 70.71));
        assert_eq!(vertices[1], (70.71, 0.0));
        assert_eq!(vertices[2], (-0.0, -70.71));
        assert_eq!(vertices[3], (-70.71, 0.0));
    }

    #[test]
    fn test_wrapping_bbox() {
        let bbox = RBBox::new(0.0, 0.0, 100.0, 100.0, Some(45.0));
        let wrapped = bbox.wrapping_bbox();
        assert_eq!(wrapped.rbbox.xc, 0.0);
        assert_eq!(wrapped.rbbox.yc, 0.0);
        assert_eq!(round_2_digits(wrapped.rbbox.width), 141.42);
        assert_eq!(round_2_digits(wrapped.rbbox.height), 141.42);
        assert_eq!(wrapped.rbbox.angle, None);

        let bbox = RBBox::new(0.0, 0.0, 50.0, 100.0, None);
        let wrapped = bbox.wrapping_bbox();
        assert_eq!(wrapped.rbbox.xc, 0.0);
        assert_eq!(wrapped.rbbox.yc, 0.0);
        assert_eq!(round_2_digits(wrapped.rbbox.width), 50.0);
        assert_eq!(round_2_digits(wrapped.rbbox.height), 100.0);
        assert_eq!(wrapped.rbbox.angle, None);

        let bbox = RBBox::new(0.0, 0.0, 50.0, 100.0, Some(90.0));
        let wrapped = bbox.wrapping_bbox();
        assert_eq!(wrapped.rbbox.xc, 0.0);
        assert_eq!(wrapped.rbbox.yc, 0.0);
        assert_eq!(round_2_digits(wrapped.rbbox.width), 100.0);
        assert_eq!(round_2_digits(wrapped.rbbox.height), 50.0);
        assert_eq!(wrapped.rbbox.angle, None);
    }

    #[test]
    fn test_graphical_wrapping_box() {
        let bbox = RBBox::new(50.0, 50.0, 100.0, 100.0, None);
        let wrapped = bbox.graphical_wrapping_bbox(0.0, 0.0, 200.0, 200.0);
        assert_eq!(wrapped.rbbox.xc, 50.0);
        assert_eq!(wrapped.rbbox.yc, 50.0);
        assert_eq!(wrapped.rbbox.width, 100.0);
        assert_eq!(wrapped.rbbox.height, 100.0);

        let bbox = RBBox::new(100.0, 100.0, 100.0, 100.0, Some(45.0));
        let wrapped = bbox.graphical_wrapping_bbox(0.0, 0.0, 500.0, 500.0);
        assert_eq!(wrapped.rbbox.xc, 100.0);
        assert_eq!(wrapped.rbbox.yc, 100.0);
        assert_eq!(round_2_digits(wrapped.rbbox.width), 142.0);
        assert_eq!(round_2_digits(wrapped.rbbox.height), 142.0);

        let bbox = RBBox::new(100.0, 100.0, 100.0, 100.0, Some(45.0));
        let wrapped = bbox.graphical_wrapping_bbox(10.0, 0.0, 500.0, 500.0);
        assert_eq!(wrapped.rbbox.xc, 100.0);
        assert_eq!(wrapped.rbbox.yc, 100.0);
        assert_eq!(round_2_digits(wrapped.rbbox.width), 162.0);
        assert_eq!(round_2_digits(wrapped.rbbox.height), 162.0);

        let bbox = RBBox::new(100.0, 100.0, 100.0, 100.0, Some(30.0));
        let wrapped = bbox.graphical_wrapping_bbox(0.0, 0.0, 500.0, 500.0);
        assert_eq!(wrapped.rbbox.xc, 100.0);
        assert_eq!(wrapped.rbbox.yc, 100.0);
        assert_eq!(round_2_digits(wrapped.rbbox.width), 138.0);
        assert_eq!(round_2_digits(wrapped.rbbox.height), 138.0);
    }
}

use crate::primitives::to_json_value::ToSerdeJsonValue;
use pyo3::{pyclass, pymethods, Py, PyAny, Python};
use rkyv::{Archive, Deserialize, Serialize};

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct BBox {
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

impl ToSerdeJsonValue for BBox {
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
impl BBox {
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
        Python::with_gil(|py| {
            py.allow_threads(|| {
                self.scale(scale_x, scale_y);
            })
        });
    }
}

impl BBox {
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

                //dbg!([scale_x, scale_y, scale_angle, nscale_height, nscale_width]);
                self.angle = Some(90.0 - (scale_angle * 180.0 / std::f64::consts::PI));
                self.xc *= scale_x;
                self.yc *= scale_y;
                self.width *= nscale_width;
                self.height *= nscale_height;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::BBox;

    #[test]
    fn test_scale_no_angle() {
        let mut bbox = BBox::new(0.0, 0.0, 100.0, 100.0, None);
        bbox.scale(2.0, 2.0);
        assert_eq!(bbox.xc, 0.0);
        assert_eq!(bbox.yc, 0.0);
        assert_eq!(bbox.width, 200.0);
        assert_eq!(bbox.height, 200.0);
        assert_eq!(bbox.angle, None);
    }

    #[test]
    fn test_scale_with_angle() {
        let mut bbox = BBox::new(0.0, 0.0, 100.0, 100.0, Some(45.0));
        bbox.scale(2.0, 3.0);
        //dbg!(&bbox);
        assert_eq!(bbox.xc, 0.0);
        assert_eq!(bbox.yc, 0.0);
        assert_eq!(bbox.width, 254.9509756796392);
        assert_eq!(bbox.height, 254.9509756796392);
        assert_eq!(bbox.angle, Some(33.69006752597978));
    }
}

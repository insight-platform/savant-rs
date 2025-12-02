use crate::primitives::{BBoxMetricType, RBBox};
use geo::{Area, BooleanOps, MultiPolygon};
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use std::collections::HashMap;
use std::sync::Arc;

fn sequential_solely_owned_areas(bboxes: &[&RBBox]) -> Vec<f64> {
    let mut areas = Vec::with_capacity(bboxes.len());
    for (i, bbox) in bboxes.iter().enumerate() {
        let others = bboxes
            .iter()
            .enumerate()
            .filter(|(j, _)| i != *j)
            .filter(|(_, b)| matches!(bbox.calculate_intersection(b), Ok(x) if x > 0.0))
            .map(|(_, b)| *b)
            .collect::<Vec<_>>();
        let union = calculate_union_area(&others);
        let bbox_area = MultiPolygon::new(vec![bbox
            .get_as_polygonal_area()
            .expect("Always safe because bbox is valid")
            .get_polygon()]);
        let area = bbox_area.difference(&union).unsigned_area();
        areas.push(area);
    }
    areas
}

fn rayon_solely_owned_areas(bboxes: &[&RBBox]) -> Vec<f64> {
    bboxes
        .par_iter()
        .map(|bbox| {
            let others = bboxes
                .iter()
                .filter(|b| {
                    !Arc::ptr_eq(&b.0, &bbox.0)
                        && matches!(bbox.calculate_intersection(b), Ok(x) if x > 0.0)
                })
                .copied()
                .collect::<Vec<_>>();
            let union = calculate_union_area(&others);
            let bbox_area = MultiPolygon::new(vec![bbox
                .get_as_polygonal_area()
                .expect("Always safe because bbox is valid")
                .get_polygon()]);
            bbox_area.difference(&union).unsigned_area()
        })
        .collect()
}

pub fn solely_owned_areas(bboxes: &[&RBBox], parallel: bool) -> Vec<f64> {
    if parallel {
        rayon_solely_owned_areas(bboxes)
    } else {
        sequential_solely_owned_areas(bboxes)
    }
}

pub fn calculate_union_area(bboxes: &[&RBBox]) -> MultiPolygon<f64> {
    if bboxes.is_empty() {
        return MultiPolygon::new(Vec::new());
    }

    let mut union = MultiPolygon::new(vec![bboxes[0]
        .get_as_polygonal_area()
        .expect("Always safe because bbox is valid")
        .get_polygon()]);

    for bbox in &bboxes[1..] {
        let mp = MultiPolygon::new(vec![bbox
            .get_as_polygonal_area()
            .expect("Always safe because bbox is valid")
            .get_polygon()]);
        union = union.union(&mp);
    }

    union
}

pub fn associate_bboxes(
    candidates: &[&RBBox],
    owners: &[&RBBox],
    metric: BBoxMetricType,
    threshold: f32,
) -> HashMap<usize, Vec<(usize, f32)>> {
    let mut associations = HashMap::new();
    for (ci, c) in candidates.iter().enumerate() {
        associations.insert(ci, Vec::new());
        for (co, o) in owners.iter().enumerate() {
            let mv = match metric {
                BBoxMetricType::IoU => c.iou(o),
                BBoxMetricType::IoSelf => c.ios(o),
                BBoxMetricType::IoOther => c.ioo(o),
            };
            if let Ok(mv) = mv {
                if mv > threshold {
                    associations.entry(ci).and_modify(|v| v.push((co, mv)));
                }
            }
        }
    }
    for (_, v) in associations.iter_mut() {
        v.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    }

    associations
}

#[cfg(test)]
mod tests {
    use crate::primitives::{BBoxMetricType, RBBox};
    use geo::Area;

    #[test]
    fn test_calculate_union_area() {
        let bb1 = RBBox::new(0.0, 0.0, 2.0, 2.0, Some(0.0));
        let bb2 = RBBox::new(2.0, 0.0, 4.0, 2.0, Some(0.0));

        let empty_union = super::calculate_union_area(&[]);
        assert_eq!(empty_union.unsigned_area(), 0.0);

        let union = super::calculate_union_area(&[&bb1]);
        assert_eq!(union.unsigned_area(), 4.0);

        let union = super::calculate_union_area(&[&bb1, &bb2]);
        assert_eq!(union.unsigned_area(), 10.0);

        let self_union = super::calculate_union_area(&[&bb1, &bb1]);
        assert_eq!(self_union.unsigned_area(), 4.0);
    }

    #[test]
    fn test_solely_owned_areas() {
        let bb1 = RBBox::new(0.0, 0.0, 2.0, 2.0, Some(0.0));
        let bb2 = RBBox::new(2.0, 0.0, 4.0, 2.0, Some(0.0));

        let areas = super::sequential_solely_owned_areas(&[&bb1, &bb2]);
        assert_eq!(areas, vec![2.0, 6.0]);

        let areas = super::rayon_solely_owned_areas(&[&bb1, &bb2]);
        assert_eq!(areas, vec![2.0, 6.0]);

        let areas = super::solely_owned_areas(&[&bb1, &bb2], true);
        assert_eq!(areas, vec![2.0, 6.0]);

        let areas = super::solely_owned_areas(&[&bb1, &bb2], false);
        assert_eq!(areas, vec![2.0, 6.0]);
    }

    #[test]
    fn test_complex_owned_areas() {
        let red = RBBox::ltrb(0.0, 2.0, 2.0, 4.0);
        let green = RBBox::ltrb(1.0, 3.0, 5.0, 5.0);
        let yellow = RBBox::ltrb(1.0, 1.0, 3.0, 6.0);
        let purple = RBBox::ltrb(4.0, 0.0, 7.0, 2.0);

        for flavor in [true, false] {
            let areas = super::solely_owned_areas(&[&red, &green, &yellow, &purple], flavor);

            let red_area = areas[0];
            let green_area = areas[1];
            let yellow_area = areas[2];
            let purple_area = areas[3];

            assert_eq!(red_area, 2.0);
            assert_eq!(green_area, 4.0);
            assert_eq!(yellow_area, 5.0);
            assert_eq!(purple_area, 6.0);
        }
    }

    #[test]
    fn test_associate_boxes() {
        let lp1 = RBBox::ltrb(0.0, 1.0, 2.0, 2.0);
        let lp2 = RBBox::ltrb(5.0, 2.0, 8.0, 3.0);
        let lp3 = RBBox::ltrb(100.0, 0.0, 6.0, 3.0);
        let owner1 = RBBox::ltrb(1.0, 0.0, 6.0, 3.0);
        let owner2 = RBBox::ltrb(6.0, 1.0, 9.0, 4.0);
        let associations_iou = super::associate_bboxes(
            &[&lp1, &lp2, &lp3],
            &[&owner1, &owner2],
            BBoxMetricType::IoU,
            0.01,
        );
        let lp1_associations = associations_iou.get(&0).unwrap();
        let lp2_associations = associations_iou.get(&1).unwrap();
        let lp3_associations = associations_iou.get(&2).unwrap();
        assert!(matches!(lp1_associations.as_slice(), [(0, _)]));
        assert!(matches!(lp2_associations.as_slice(), [(1, _), (0, _)]));
        assert!(lp3_associations.is_empty());
    }
}

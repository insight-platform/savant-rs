use crate::primitives::RBBox;
use crate::test::utils::gen_object;

#[test]
fn test_owned_getters() {
    let rbbox = RBBox::new(1.0, 2.0, 3.0, 4.0, Some(30.0));
    assert_eq!(rbbox.get_xc(), 1.0);
    assert_eq!(rbbox.get_yc(), 2.0);
    assert_eq!(rbbox.get_width(), 3.0);
    assert_eq!(rbbox.get_height(), 4.0);
    assert_eq!(rbbox.get_angle(), Some(30.0));
    assert_eq!(rbbox.get_area(), 12.0);
}

#[test]
fn test_borrowed_detection_getters() {
    let o = gen_object(1);
    let detection_box = o.get_detection_box();
    assert_eq!(detection_box.get_xc(), 1.0);
    assert_eq!(detection_box.get_yc(), 2.0);
    assert_eq!(detection_box.get_width(), 10.0);
    assert_eq!(detection_box.get_height(), 20.0);
    assert_eq!(detection_box.get_angle(), None);
    assert_eq!(detection_box.get_area(), 200.0);
}

#[test]
fn test_borrowed_track_getters() {
    let o = gen_object(1);
    let track_box = o.get_track_box().unwrap();
    assert_eq!(track_box.get_xc(), 100.0);
    assert_eq!(track_box.get_yc(), 200.0);
    assert_eq!(track_box.get_width(), 10.0);
    assert_eq!(track_box.get_height(), 20.0);
    assert_eq!(track_box.get_angle(), None);
    assert_eq!(track_box.get_area(), 200.0);
}

#[test]
fn test_ensure_detection_box_shared() {
    let o = gen_object(1);
    let mut detection_box1 = o.get_detection_box();
    let detection_box2 = o.get_detection_box();
    detection_box1.set_xc(33.0);
    assert_eq!(detection_box2.get_xc(), 33.0);
}

#[test]
fn test_ensure_track_box_shared() {
    let o = gen_object(1);
    let mut track_box1 = o.get_track_box().unwrap();
    let track_box2 = o.get_track_box().unwrap();
    track_box1.set_xc(33.0);
    assert_eq!(track_box2.get_xc(), 33.0);
}

#[test]
fn test_owned_setters() {
    let mut rbbox = RBBox::new(1.0, 2.0, 3.0, 4.0, Some(30.0));
    rbbox.set_xc(10.0);
    rbbox.set_yc(20.0);
    rbbox.set_width(30.0);
    rbbox.set_height(40.0);
    rbbox.set_angle(Some(300.0));
    assert_eq!(rbbox.get_xc(), 10.0);
    assert_eq!(rbbox.get_yc(), 20.0);
    assert_eq!(rbbox.get_width(), 30.0);
    assert_eq!(rbbox.get_height(), 40.0);
    assert_eq!(rbbox.get_angle(), Some(300.0));
}

#[test]
fn test_borrowed_detection_setters() {
    let o = gen_object(1);
    let mut detection_box = o.get_detection_box();
    detection_box.set_xc(10.0);
    detection_box.set_yc(20.0);
    detection_box.set_width(30.0);
    detection_box.set_height(40.0);
    detection_box.set_angle(Some(300.0));
    assert_eq!(detection_box.get_xc(), 10.0);
    assert_eq!(detection_box.get_yc(), 20.0);
    assert_eq!(detection_box.get_width(), 30.0);
    assert_eq!(detection_box.get_height(), 40.0);
    assert_eq!(detection_box.get_angle(), Some(300.0));
}

#[test]
fn test_borrowed_track_setters() {
    let o = gen_object(1);
    let mut track_box = o.get_track_box().unwrap();
    track_box.set_xc(10.0);
    track_box.set_yc(20.0);
    track_box.set_width(30.0);
    track_box.set_height(40.0);
    track_box.set_angle(Some(300.0));
    assert_eq!(track_box.get_xc(), 10.0);
    assert_eq!(track_box.get_yc(), 20.0);
    assert_eq!(track_box.get_width(), 30.0);
    assert_eq!(track_box.get_height(), 40.0);
    assert_eq!(track_box.get_angle(), Some(300.0));
}

#[test]
fn compare_owned_boxes() {
    let b1 = RBBox::new(1.0, 2.0, 3.0, 4.0, Some(30.0));
    assert!(b1.geometric_eq(&b1));
    assert!(b1.almost_eq(&b1, 1e-6));
    let b2 = RBBox::new(1.0, 2.0, 3.0, 4.0, Some(30.0));
    assert!(b1.geometric_eq(&b2));
    assert!(b1.almost_eq(&b2, 1e-6));
}

#[test]
fn compare_borrowed_detection_boxes() {
    let o1 = gen_object(1);
    let o2 = gen_object(1);
    let b1 = o1.get_detection_box();
    let b2 = o2.get_detection_box();
    assert!(b1.geometric_eq(&b2));
    assert!(b1.almost_eq(&b2, 1e-6));
}

#[test]
fn compare_borrowed_track_boxes() {
    let o1 = gen_object(1);
    let o2 = gen_object(1);
    let b1 = o1.get_track_box().unwrap();
    let b2 = o2.get_track_box().unwrap();
    assert!(b1.geometric_eq(&b2));
    assert!(b1.almost_eq(&b2, 1e-6));
}

#[test]
fn compare_mixed_detection_track_boxes() {
    let o1 = gen_object(1);
    let o2 = gen_object(1);
    let mut b1 = o1.get_detection_box();
    b1.set_xc(100.0);
    b1.set_yc(200.0);
    b1.set_width(10.0);
    b1.set_height(20.0);
    b1.set_angle(Some(300.0));
    let mut b2 = o2.get_track_box().unwrap();
    b2.set_xc(100.0);
    b2.set_yc(200.0);
    b2.set_width(10.0);
    b2.set_height(20.0);
    b2.set_angle(Some(300.0));
    assert!(b1.geometric_eq(&b2));
    assert!(b1.almost_eq(&b2, 1e-6));
}

#[test]
fn compare_mixed_owned_detection_boxes() {
    let b1 = RBBox::new(100.0, 200.0, 10.0, 20.0, Some(300.0));
    let o2 = gen_object(1);
    let mut b2 = o2.get_detection_box();
    b2.set_xc(100.0);
    b2.set_yc(200.0);
    b2.set_width(10.0);
    b2.set_height(20.0);
    b2.set_angle(Some(300.0));
    assert!(b1.geometric_eq(&b2));
    assert!(b1.almost_eq(&b2, 1e-6));
}

#[test]
fn test_bbox_width_to_height_ratio() {
    let b1 = RBBox::new(100.0, 200.0, 10.0, 20.0, Some(300.0));
    assert_eq!(b1.get_width_to_height_ratio(), 0.5);
}

#[test]
fn test_modification_owned_box() {
    let mut b = RBBox::new(100.0, 200.0, 10.0, 20.0, Some(300.0));
    assert_eq!(b.is_modified(), false);
    b.set_modification_status(true);
    assert_eq!(b.is_modified(), true);
}

#[test]
fn test_modification_borrowed_detection_box() {
    let o = gen_object(1);
    let mut b = o.get_detection_box();
    assert_eq!(b.is_modified(), false);
    b.set_modification_status(true);
    assert_eq!(b.is_modified(), true);
}

#[test]
fn test_modification_borrowed_track_box() {
    let o = gen_object(1);
    let mut b = o.get_track_box().unwrap();
    assert_eq!(b.is_modified(), false);
    b.set_modification_status(true);
    assert_eq!(b.is_modified(), true);
}

#[test]
fn test_setters_modify_owned_box() {
    let mut b = RBBox::new(100.0, 200.0, 10.0, 20.0, Some(300.0));
    assert_eq!(b.is_modified(), false);

    b.set_xc(100.0);
    assert_eq!(b.is_modified(), true);

    b.set_modification_status(false);
    b.set_yc(200.0);
    assert_eq!(b.is_modified(), true);

    b.set_modification_status(false);
    b.set_width(10.0);
    assert_eq!(b.is_modified(), true);

    b.set_modification_status(false);
    b.set_height(20.0);
    assert_eq!(b.is_modified(), true);

    b.set_modification_status(false);
    b.set_angle(Some(300.0));
    assert_eq!(b.is_modified(), true);
}

#[test]
fn test_setters_borrowed_detection_box() {
    let o = gen_object(1);
    let mut b = o.get_detection_box();
    assert_eq!(b.is_modified(), false);

    b.set_xc(100.0);
    assert_eq!(b.is_modified(), true);

    b.set_modification_status(false);
    b.set_yc(200.0);
    assert_eq!(b.is_modified(), true);

    b.set_modification_status(false);
    b.set_width(10.0);
    assert_eq!(b.is_modified(), true);

    b.set_modification_status(false);
    b.set_height(20.0);
    assert_eq!(b.is_modified(), true);

    b.set_modification_status(false);
    b.set_angle(Some(300.0));
    assert_eq!(b.is_modified(), true);
}

#[test]
fn test_setters_borrowed_track_box() {
    let o = gen_object(1);
    let mut b = o.get_track_box().unwrap();
    assert_eq!(b.is_modified(), false);

    b.set_xc(100.0);
    assert_eq!(b.is_modified(), true);

    b.set_modification_status(false);
    b.set_yc(200.0);
    assert_eq!(b.is_modified(), true);

    b.set_modification_status(false);
    b.set_width(10.0);
    assert_eq!(b.is_modified(), true);

    b.set_modification_status(false);
    b.set_height(20.0);
    assert_eq!(b.is_modified(), true);

    b.set_modification_status(false);
    b.set_angle(Some(300.0));
    assert_eq!(b.is_modified(), true);
}

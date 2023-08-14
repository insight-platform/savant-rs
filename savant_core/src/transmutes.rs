#[cfg(test)]
mod tests {
    struct X(i64);

    #[test]
    fn test_transmute_i64_to_x() {
        let int = -1;
        let x = unsafe { std::mem::transmute::<i64, X>(int) };
        assert_eq!(x.0, -1);
    }

    #[test]
    fn test_transmute_x_to_i64() {
        let x = X(-1);
        let int = unsafe { std::mem::transmute::<X, i64>(x) };
        assert_eq!(int, -1);
    }

    #[test]
    fn test_transmute_vec_i64_to_x() {
        let int = vec![-1, 0, 1];
        let x = unsafe { std::mem::transmute::<Vec<i64>, Vec<X>>(int) };
        assert_eq!(x[0].0, -1);
        assert_eq!(x[1].0, 0);
        assert_eq!(x[2].0, 1);
    }

    #[test]
    fn test_transmute_vec_x_to_i64() {
        let x = vec![X(-1), X(0), X(1)];
        let int = unsafe { std::mem::transmute::<Vec<X>, Vec<i64>>(x) };
        assert_eq!(int[0], -1);
        assert_eq!(int[1], 0);
        assert_eq!(int[2], 1);
    }

    #[test]
    fn test_transmute_ref_i64_to_x() {
        let int = -1;
        let x = unsafe { std::mem::transmute::<&i64, &X>(&int) };
        assert_eq!(x.0, -1);
    }

    #[test]
    fn test_transmute_opt_i64_to_opt_x() {
        let int = Some(-1);
        let x = unsafe { std::mem::transmute::<Option<i64>, Option<X>>(int) };
        assert_eq!(x.unwrap().0, -1);
    }
}
